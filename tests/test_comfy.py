import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image

from tinyseek.config import load_config
from tinyseek.generate.comfy import generate, patch_workflow, scene_prompt


def comfy_cfg(**extra) -> dict:
    return dict(load_config("configs/generate_comfy.yaml"), **extra)


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
    graph = json.loads(Path(cfg["workflow"]).read_text())
    values = {"text": "a scene", "seed": 7, "width": 640, "height": 480}
    patched = patch_workflow(graph, cfg["workflow_inputs"], values)
    assert patched["text"]["inputs"]["prompt"] == "a scene"
    assert patched["noise"]["inputs"]["noise_seed"] == 7
    assert (patched["latent"]["inputs"]["width"], patched["latent"]["inputs"]["height"]) == (640, 480)
    assert graph["text"]["inputs"]["prompt"] == "" and graph["noise"]["inputs"]["noise_seed"] == 0
    with pytest.raises(KeyError):
        patch_workflow(graph, {"text": ["text", "no_such_input"]}, values)


class FakeComfy(BaseHTTPRequestHandler):
    graphs: ClassVar[list[dict]] = []

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
            images = [{"filename": f"{pid}.png", "subfolder": "", "type": "temp"}]
            entry = {"status": {"status_str": "success", "completed": True}, "outputs": {"image": {"images": images}}}
            self._send(json.dumps({pid: entry}).encode())
        else:
            buf = io.BytesIO()
            Image.new("RGB", (1184, 896), (90, 80, 70)).save(buf, "PNG")
            self._send(buf.getvalue(), "image/png")

    def log_message(self, *args):
        pass


def test_generate_stores_a_resumable_set(tmp_path):
    FakeComfy.graphs = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = comfy_cfg(data_root=str(tmp_path), num_images=3, synth_set="t",
                        server=f"http://127.0.0.1:{server.server_port}")
        out = generate(cfg)
        assert sorted(p.name for p in out.glob("*.jpg")) == ["000000.jpg", "000001.jpg", "000002.jpg"]
        assert Image.open(out / "000000.jpg").size == (800, 605)
        rows = [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines()]
        assert [r["index"] for r in rows] == [0, 1, 2]
        for row, graph in zip(rows, FakeComfy.graphs):
            assert graph["text"]["inputs"]["prompt"] == row["prompt"]
            assert graph["noise"]["inputs"]["noise_seed"] == row["seed"]
            assert set(row["fields"]) == {"subject", "placement", "location", "framing", "clutter", "lighting"}
        assert len({r["seed"] for r in rows}) == 3
        assert (out / "workflow.json").exists() and (out / "config.yaml").exists()

        generate(dict(cfg, num_images=4))  # resumes: only the missing image is requested
        assert len(FakeComfy.graphs) == 4
    finally:
        server.shutdown()
