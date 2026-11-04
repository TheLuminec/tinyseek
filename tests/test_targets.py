import numpy as np
import torch

from tinyseek.train.loss import centernet_loss
from tinyseek.train.model import TinyCenterNet
from tinyseek.train.targets import decode, encode_targets


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def test_encode_decode_roundtrip():
    boxes = np.array([[10.0, 20.0, 50.0, 90.0], [70.0, 64.0, 120.0, 100.0]], dtype=np.float32)
    t = encode_targets(boxes, np.zeros(2, np.int64), num_classes=1, input_size=128, stride=8)
    assert t["heatmap"].shape == (1, 16, 16)
    assert t["mask"].sum() == 2
    assert t["heatmap"].max() == 1.0

    # Build "perfect" head outputs from the targets and check decoding recovers the boxes.
    heat = torch.full((1, 1, 16, 16), logit(1e-3))
    size = torch.zeros(1, 2, 256)
    offset = torch.zeros(1, 2, 256)
    for k in range(2):
        idx = int(t["index"][k])
        heat.view(-1)[idx] = logit(0.9)
        size[0, :, idx] = torch.from_numpy(t["size"][k])
        offset[0, :, idx] = torch.from_numpy(t["offset"][k])
    out, scores, classes = decode(heat, size.view(1, 2, 16, 16), offset.view(1, 2, 16, 16), stride=8, top_k=2)
    got = sorted(out[0].numpy().tolist())
    want = sorted(boxes.tolist())
    np.testing.assert_allclose(got, want, atol=1e-3)
    assert torch.allclose(scores, torch.tensor([[0.9, 0.9]]), atol=1e-6)
    assert (classes == 0).all()


def test_encode_empty():
    t = encode_targets(np.zeros((0, 4), np.float32), np.zeros(0, np.int64), 1, 128, 8)
    assert t["heatmap"].sum() == 0 and t["mask"].sum() == 0


def test_model_shapes_and_loss_backward():
    model = TinyCenterNet(pretrained=False)
    out = model(torch.randn(2, 3, 128, 128))
    assert out["heatmap"].shape == (2, 1, 16, 16)
    assert out["size"].shape == out["offset"].shape == (2, 2, 16, 16)

    boxes = np.array([[10.0, 20.0, 50.0, 90.0]], dtype=np.float32)
    t = encode_targets(boxes, np.zeros(1, np.int64), 1, 128, 8)
    batch = {k: torch.from_numpy(np.stack([v, v])) for k, v in t.items()}
    loss, parts = centernet_loss(out, batch)
    loss.backward()
    assert np.isfinite(loss.item()) and set(parts) == {"heatmap", "size", "offset"}
