import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image

from tinyseek.config import load_config
from tinyseek.generate.comfy import (
    ComfyClient,
    batch_workflow,
    generate,
    patch_workflow,
    per_image_nodes,
    scene_prompt,
)


def comfy_cfg(**extra) -> dict:
    return dict(load_config("configs/generate_comfy.yaml"), **extra)


def workflow(cfg: dict) -> dict:
    return json.loads(Path(cfg["workflow"]).read_text())


@pytest.mark.parametrize("prompt", ["backpack", "cup"])
def test_scene_prompts_are_filled_deterministic_and_varied(prompt):
    cfg = comfy_cfg(prompt=prompt)
    texts = [scene_prompt(cfg, i)[0] for i in range(40)]
    assert texts == [scene_prompt(cfg, i)[0] for i in range(40)]
    assert len(set(texts)) > 30
    for text in texts:
        assert "{" not in text and "[" not in text
    assert scene_prompt(dict(cfg, seed=1), 0)[0] != texts[0] or scene_prompt(dict(cfg, seed=1), 1)[0] != texts[1]


def test_unfilled_placeholder_is_rejected():
    cfg = comfy_cfg(template="The [BACKPACK DESCRIPTION] is {placement}.")
    with pytest.raises(ValueError, match="unfilled"):
        scene_prompt(cfg, 0)


def test_patch_workflow_sets_inputs_without_touching_the_template():
    cfg = comfy_cfg()
    graph = workflow(cfg)
    values = {"text": "a scene", "seed": 7, "width": 640, "height": 480}
    patched = patch_workflow(graph, cfg["workflow_inputs"], values)
    assert patched["text"]["inputs"]["prompt"] == "a scene"
    assert patched["noise"]["inputs"]["noise_seed"] == 7
    assert (patched["latent"]["inputs"]["width"], patched["latent"]["inputs"]["height"]) == (640, 480)
    assert graph["text"]["inputs"]["prompt"] == "" and graph["noise"]["inputs"]["noise_seed"] == 0
    with pytest.raises(KeyError):
        patch_workflow(graph, {"text": ["text", "no_such_input"]}, values)


def test_batch_workflow_shares_loaders_and_repeats_per_image_nodes():
    cfg = comfy_cfg()
    graph = workflow(cfg)
    assert per_image_nodes(graph, cfg["workflow_inputs"]) == {"text", "latent", "noise", "guider", "sample",
                                                             "decode", "image"}
    values = [{"text": f"scene {k}", "seed": 10 + k, "width": 640, "height": 480} for k in range(3)]
    batch = batch_workflow(graph, cfg["workflow_inputs"], values)
    shared = {"unet", "clip", "vae", "model", "sampler", "sigmas"}
    assert set(batch) == shared | {f"{n}#{k}" for n in ("text", "latent", "noise", "guider", "sample", "decode",
                                                        "image") for k in range(3)}
    for k in range(3):
        assert batch[f"text#{k}"]["inputs"]["prompt"] == f"scene {k}"
        assert batch[f"noise#{k}"]["inputs"]["noise_seed"] == 10 + k
        assert batch[f"text#{k}"]["inputs"]["clip"] == ["clip", 0]          # shared loader
        assert batch[f"guider#{k}"]["inputs"]["model"] == ["model", 0]
        assert batch[f"guider#{k}"]["inputs"]["positive"] == [f"text#{k}", 0]
        assert batch[f"guider#{k}"]["inputs"]["negative"] == [f"text#{k}", 1]
        assert batch[f"sample#{k}"]["inputs"]["latent_image"] == [f"latent#{k}", 0]
        assert batch[f"image#{k}"]["inputs"]["images"] == [f"decode#{k}", 0]
    for node in batch.values():  # every link points at a node in the graph
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in batch


class FakeComfy(BaseHTTPRequestHandler):
    graphs: ClassVar[list[dict]] = []
    queue_polls: ClassVar[list[dict]] = []   # answers to /queue, in order; then an empty queue

    def _send(self, body: bytes, kind: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        graph = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["prompt"]
        FakeComfy.graphs.append(graph)
        self._send(json.dumps({"prompt_id": f"p{len(FakeComfy.graphs)}"}).encode())

    def do_GET(self):
        if self.path.startswith("/history/"):
            pid = self.path.rsplit("/", 1)[1]
            graph = FakeComfy.graphs[int(pid[1:]) - 1]
            outputs = {node: {"images": [{"filename": f"{pid}_{node}.png", "subfolder": "", "type": "temp"}]}
                       for node in graph if node.startswith("image#")}
            entry = {"status": {"status_str": "success", "completed": True}, "outputs": outputs}
            self._send(json.dumps({pid: entry}).encode())
        elif self.path == "/queue":
            empty = {"queue_running": [], "queue_pending": []}
            self._send(json.dumps(FakeComfy.queue_polls.pop(0) if FakeComfy.queue_polls else empty).encode())
        else:
            buf = io.BytesIO()
            Image.new("RGB", (1184, 896), (90, 80, 70)).save(buf, "PNG")
            self._send(buf.getvalue(), "image/png")

    def log_message(self, *args):
        pass


@pytest.mark.parametrize(("per_job", "jobs"), [(1, [1, 1, 1, 1, 1]), (4, [4, 1])])
def test_generate_stores_a_resumable_set(tmp_path, per_job, jobs):
    FakeComfy.graphs = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = comfy_cfg(data_root=str(tmp_path), num_images=5, synth_set="t", images_per_job=per_job,
                        server=f"http://127.0.0.1:{server.server_port}")
        out = generate(cfg)
        assert sorted(p.name for p in out.glob("*.jpg")) == [f"{i:06d}.jpg" for i in range(5)]
        assert Image.open(out / "000000.jpg").size == (800, 605)
        rows = [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines()]
        assert [r["index"] for r in rows] == [0, 1, 2, 3, 4]
        assert len({r["seed"] for r in rows}) == 5
        sent = [(g[f"text#{k}"]["inputs"]["prompt"], g[f"noise#{k}"]["inputs"]["noise_seed"])
                for g in FakeComfy.graphs for k in range(sum(n.startswith("text#") for n in g))]
        assert sent == [(r["prompt"], r["seed"]) for r in rows]   # same images whatever the job size
        assert sent == [(scene_prompt(cfg, i)[0], rows[i]["seed"]) for i in range(5)]
        assert [sum(n.startswith("image#") for n in g) for g in FakeComfy.graphs] == jobs
        assert set(rows[0]["fields"]) == {"subject", "placement", "location", "framing", "clutter", "lighting",
                                         "viewpoint"}
        assert (out / "workflow.json").exists() and (out / "config.yaml").exists()

        generate(dict(cfg, num_images=6))  # resumes: only the missing image is requested
        assert sum(n.startswith("image#") for n in FakeComfy.graphs[-1]) == 1
        assert len(FakeComfy.graphs) == len(jobs) + 1
    finally:
        server.shutdown()


def test_wait_idle_returns_only_after_the_queue_stays_empty(capsys):
    busy = {"queue_running": [["job"]], "queue_pending": [["a"], ["b"]]}
    empty = {"queue_running": [], "queue_pending": []}
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = ComfyClient(f"http://127.0.0.1:{server.server_port}")
        FakeComfy.queue_polls = [busy]
        assert client.queued() == 3
        FakeComfy.queue_polls = [busy, empty, busy]   # a job arriving after one empty poll resets the count
        client.wait_idle(poll_s=0.01)
        assert FakeComfy.queue_polls == []
        out = capsys.readouterr().out
        assert out.count("3 job(s) queued, waiting") == 2 and out.count("idle, confirming") == 2
    finally:
        server.shutdown()
        FakeComfy.queue_polls = []
