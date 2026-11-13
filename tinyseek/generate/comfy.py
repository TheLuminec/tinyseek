"""Generate a synthetic image set through a ComfyUI server (e.g. Qwen Image 2.1 Turbo).

    python -m tinyseek.generate.comfy --config configs/generate_comfy.yaml prompt=backpack num_images=50

The server runs the model; this module only fills a workflow graph per image, queues it, and
stores the result in the same layout as ``tinyseek.generate.sample``, so labeling, camera
simulation and training work unchanged. Resumable: existing images are skipped. Each image's
text, field choices and seed are logged to ``metadata.jsonl``, and the workflow is copied
next to the images.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from tinyseek.config import deep_merge, load_config, save_run_config
from tinyseek.generate.sample import image_seed

UNFILLED = re.compile(r"\{[^}]*\}|\[[A-Z][A-Z /]+\]")


def scene_fields(cfg: dict) -> dict[str, list[str]]:
    """The shared field lists with the object's own (subject, placement, ...) merged over them."""
    return deep_merge(cfg["fields"], cfg["objects"][cfg["prompt"]])


def scene_prompt(cfg: dict, index: int) -> tuple[str, dict[str, str]]:
    """Deterministically fill the template for image ``index``; returns (text, chosen fields)."""
    rng = np.random.default_rng([cfg["seed"], index])
    chosen = {name: options[rng.integers(len(options))] for name, options in sorted(scene_fields(cfg).items())}
    text = " ".join(cfg["template"].format(**chosen).split())
    if UNFILLED.search(text):
        raise ValueError(f"unfilled placeholder in prompt: {text!r}")
    return text, chosen


def patch_workflow(graph: dict, inputs: dict[str, list[str]], values: dict) -> dict:
    """Copy ``graph`` with each value written to its ``[node id, input name]`` slot."""
    out = copy.deepcopy(graph)
    for key, (node, name) in inputs.items():
        if name not in out[node]["inputs"]:
            raise KeyError(f"workflow node {node!r} has no input {name!r}")
        out[node]["inputs"][name] = values[key]
    return out


def per_image_nodes(graph: dict, inputs: dict[str, list[str]]) -> set[str]:
    """Nodes that differ per image: those receiving a per-image value and everything downstream."""
    varying = {node for node, _ in inputs.values()}
    grew = True
    while grew:
        grew = False
        for node_id, node in graph.items():
            links = [v[0] for v in node["inputs"].values() if isinstance(v, list)]
            if node_id not in varying and varying.intersection(links):
                varying.add(node_id)
                grew = True
    return varying


def batch_workflow(graph: dict, inputs: dict[str, list[str]], values: list[dict]) -> dict:
    """One graph that makes ``len(values)`` images: the model loaders are shared and the
    per-image nodes are repeated as ``<id>#<k>``. Run as a single job, the server loads the
    text encoder once for all the texts and the image model once for all the samples,
    where one job per image reloads both every time (they do not fit in VRAM together).
    """
    varying = per_image_nodes(graph, inputs)
    out = {node_id: copy.deepcopy(node) for node_id, node in graph.items() if node_id not in varying}
    for k, image_values in enumerate(values):
        patched = patch_workflow(graph, inputs, image_values)
        for node_id in varying:
            node = patched[node_id]
            for name, value in node["inputs"].items():
                if isinstance(value, list) and value[0] in varying:
                    node["inputs"][name] = [f"{value[0]}#{k}", *value[1:]]
            out[f"{node_id}#{k}"] = node
    return out


class ComfyClient:
    def __init__(self, server: str, timeout_s: float = 900, poll_s: float = 1.0) -> None:
        self.server = server.rstrip("/")
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.client_id = uuid.uuid4().hex

    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(self.server + path, timeout=60) as response:
            return response.read()

    def submit(self, graph: dict) -> str:
        body = json.dumps({"prompt": graph, "client_id": self.client_id}).encode()
        request = urllib.request.Request(self.server + "/prompt", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())["prompt_id"]
        except urllib.error.HTTPError as err:  # the server explains which node input it rejected
            raise RuntimeError(f"server rejected the workflow: {err.read().decode(errors='replace')[:2000]}") from err

    def wait(self, prompt_id: str) -> dict:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            entry = json.loads(self._get(f"/history/{prompt_id}")).get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"generation failed: {json.dumps(status.get('messages'))[:2000]}")
                if status.get("completed", True):
                    return entry
            time.sleep(self.poll_s)
        raise TimeoutError(f"no result for {prompt_id} after {self.timeout_s} s")

    def image(self, entry: dict, node: str) -> Image.Image:
        ref = entry["outputs"][node]["images"][0]
        query = urllib.parse.urlencode({k: ref.get(k, "") for k in ("filename", "subfolder", "type")})
        return Image.open(io.BytesIO(self._get("/view?" + query))).convert("RGB")

    def generate(self, graph: dict, nodes: list[str]) -> list[Image.Image]:
        entry = self.wait(self.submit(graph))
        return [self.image(entry, node) for node in nodes]


def stored_size(image: Image.Image, save_width: int | None) -> Image.Image:
    if not save_width or image.width <= save_width:
        return image
    return image.resize((save_width, round(image.height * save_width / image.width)), Image.LANCZOS)


def generate(cfg: dict) -> Path:
    out_dir = Path(cfg["data_root"]) / "synthetic" / cfg["prompt"] / cfg["synth_set"]
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(cfg["num_images"]) if not (out_dir / f"{i:06d}.jpg").exists()]
    if not todo:
        print(f"{out_dir}: all {cfg['num_images']} images exist")
        return out_dir
    graph = json.loads(Path(cfg["workflow"]).read_text())
    save_run_config(cfg, out_dir)
    (out_dir / "workflow.json").write_text(json.dumps(graph, indent=1))
    client = ComfyClient(os.environ.get("TINYSEEK_COMFY_URL", cfg["server"]), cfg.get("timeout_s", 900))

    per_job = max(1, cfg.get("images_per_job", 1))
    with (out_dir / "metadata.jsonl").open("a") as meta:
        for start in range(0, len(todo), per_job):
            idx = todo[start : start + per_job]
            scenes = [scene_prompt(cfg, i) for i in idx]
            values = [{"text": text, "seed": image_seed(cfg["seed"], i), "width": cfg["width"],
                       "height": cfg["height"]} for i, (text, _) in zip(idx, scenes)]
            started = time.monotonic()
            images = client.generate(batch_workflow(graph, cfg["workflow_inputs"], values),
                                     [f"{cfg['output_node']}#{k}" for k in range(len(idx))])
            seconds = (time.monotonic() - started) / len(idx)
            for i, (text, chosen), value, image in zip(idx, scenes, values, images):
                stored_size(image, cfg.get("save_width")).save(out_dir / f"{i:06d}.jpg", quality=95)
                meta.write(json.dumps({"index": i, "file_name": f"{i:06d}.jpg", "prompt": text, "seed": value["seed"],
                                       "fields": chosen, "seconds": round(seconds, 1), "job_images": len(idx)}) + "\n")
            meta.flush()
            print(f"{cfg['prompt']} {start + len(idx)}/{len(todo)} images {idx[0]}-{idx[-1]} "
                  f"{seconds:.0f}s each", flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/generate_comfy.yaml")
    parser.add_argument("--print-prompts", type=int, metavar="N", help="print the first N prompts and exit")
    parser.add_argument("overrides", nargs="*", help="e.g. prompt=backpack num_images=100")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.print_prompts:
        for i in range(args.print_prompts):
            print(f"[{i}] {scene_prompt(cfg, i)[0]}\n")
        return
    print(generate(cfg))


if __name__ == "__main__":
    main()
