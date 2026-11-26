"""Fast tests for tinyseek.train.qat: graph<->model mapping, BatchNorm-identity write-back, dev split."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tinyseek.data.dataset import Record
from tinyseek.train import qat as Q
from tinyseek.train.model import ExportWrapper, TinyCenterNet


def small_model() -> TinyCenterNet:
    torch.manual_seed(0)
    model = TinyCenterNet(pretrained=False, head_channels=8)
    for m in model.modules():  # non-trivial BatchNorm statistics, so folding matters
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.2, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
    return model.eval()


def fold(model: nn.Module, pairs: dict) -> dict:
    """Conv+BatchNorm folding as torch.onnx does it in eval mode."""
    mods = dict(model.named_modules())
    out = {}
    for name, bn_name in pairs.items():
        conv = mods[name]
        w = conv.weight.detach().clone()
        b = conv.bias.detach().clone() if conv.bias is not None else torch.zeros(w.shape[0])
        if bn_name is not None:
            bn = mods[bn_name]
            k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w = w * k.view(-1, 1, 1, 1)
            b = (b - bn.running_mean) * k + bn.bias
        out[name] = (w.detach(), b.detach())
    return out


@pytest.mark.parametrize("op_name, path", [
    ("/model/backbone/conv_stem/Conv", "backbone.conv_stem"),
    ("/model/backbone/blocks.0/blocks.0.0/conv_dw/Conv", "backbone.blocks.0.0.conv_dw"),
    ("/model/backbone/blocks.5/blocks.5.2/conv_pwl/Conv", "backbone.blocks.5.2.conv_pwl"),
    ("/model/lateral.2/lateral.2.0/Conv", "lateral.2.0"),
    ("/model/smooth/smooth.1/smooth.1.0/Conv", "smooth.1.0"),
    ("/model/heatmap/heatmap.0/heatmap.0.1/heatmap.0.1.0/Conv", "heatmap.0.1.0"),
    ("/model/heatmap/heatmap.1/Conv", "heatmap.1"),
])
def test_module_path(op_name, path):
    assert Q.module_path(op_name) == path


def test_module_path_rejects_foreign_ops():
    with pytest.raises(ValueError):
        Q.module_path("/Concat")


def test_module_paths_exist_in_model():
    model = small_model()
    names = dict(model.named_modules())
    for op in ["/model/backbone/blocks.3/blocks.3.1/conv_dw/Conv", "/model/offset/offset.1/Conv",
               "/model/size/size.0/size.0.0/size.0.0.0/Conv"]:
        assert isinstance(names[Q.module_path(op)], nn.Conv2d)


def test_conv_bn_pairs():
    model = small_model()
    pairs = Q.conv_bn_pairs(model, 32)
    convs = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    assert sorted(pairs) == sorted(convs)
    assert pairs["backbone.conv_stem"] == "backbone.bn1"
    assert pairs["lateral.0.0"] == "lateral.0.1"
    assert pairs["heatmap.1"] is None and pairs["size.1"] is None
    assert model.training is False


def test_write_back_reproduces_float_function():
    model = small_model()
    pairs = Q.conv_bn_pairs(model, 32)
    fused = fold(model, pairs)
    fresh = TinyCenterNet(pretrained=False, head_channels=8).eval()
    Q.write_back(fresh, fused, pairs)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        ref, got = ExportWrapper(model)(x), ExportWrapper(fresh)(x)
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4)
    # BatchNorm is identity, so folding the written-back model returns the same fused weights.
    refold = fold(fresh, pairs)
    for name, (w, b) in fused.items():
        assert torch.allclose(refold[name][0], w, atol=1e-6)
        assert torch.allclose(refold[name][1], b, atol=1e-6)


def test_write_back_requires_every_conv():
    model = small_model()
    pairs = Q.conv_bn_pairs(model, 32)
    fused = fold(model, pairs)
    fused.pop("backbone.conv_stem")
    with pytest.raises(KeyError):
        Q.write_back(model, fused, pairs)


def test_write_back_rejects_shape_mismatch():
    model = small_model()
    pairs = Q.conv_bn_pairs(model, 32)
    fused = fold(model, pairs)
    w, b = fused["heatmap.1"]
    fused["heatmap.1"] = (w[:, :4], b)
    with pytest.raises(ValueError):
        Q.write_back(model, fused, pairs)


def records(n: int) -> list[Record]:
    return [Record(Path(f"img{i}.jpg"), i, np.zeros((0, 4), np.float32)) for i in range(n)]


def test_split_dev_is_seeded_disjoint_and_complete():
    recs = records(50)
    train, dev = Q.split_dev(recs, 10, seed=3)
    assert len(dev) == 10 and len(train) == 40
    assert {r.image_id for r in train}.isdisjoint(r.image_id for r in dev)
    assert sorted(r.image_id for r in train + dev) == list(range(50))
    assert [r.image_id for r in Q.split_dev(recs, 10, seed=3)[1]] == [r.image_id for r in dev]
    assert [r.image_id for r in Q.split_dev(recs, 10, seed=4)[1]] != [r.image_id for r in dev]
    assert [r.image_id for r in train] == sorted(r.image_id for r in train)
    with pytest.raises(ValueError):
        Q.split_dev(recs, 50, seed=0)


def test_calibration_records_match_export_draw():
    recs = records(30)
    got = Q.calibration_records({"seed": 2}, recs, 8)
    idx = np.random.default_rng(2).choice(30, 8, replace=False)  # tinyseek.train.export's draw
    assert [r.image_id for r in got] == [int(i) for i in idx]
    assert len(Q.calibration_records({"seed": 2}, recs, 100)) == 30


def test_split_output_channels():
    out = torch.arange(2 * 5 * 3 * 3, dtype=torch.float32).view(2, 5, 3, 3)
    d = Q.split_output(out, 1)
    assert d["heatmap"].shape == (2, 1, 3, 3) and d["size"].shape == (2, 2, 3, 3)
    assert torch.equal(d["offset"], out[:, 3:5])


def fake_graph() -> SimpleNamespace:
    def var(value, param=True):
        return SimpleNamespace(value=value, is_parameter=param)

    x = var(torch.zeros(1), param=False)
    ops = {
        "a": SimpleNamespace(name="/model/backbone/conv_stem/Conv", type="Conv",
                             inputs=[x, var(torch.ones(2, 3, 1, 1)), var(torch.zeros(2))]),
        "b": SimpleNamespace(name="/model/heatmap/heatmap.1/Conv", type="Conv",
                             inputs=[x, var(torch.ones(1, 2, 1, 1)), var(torch.full((1,), 0.5))]),
        "c": SimpleNamespace(name="/model/backbone/bn1/act/Clip", type="Clip",
                             inputs=[x, var(torch.tensor(0.0)), var(torch.tensor(6.0))]),
    }
    return SimpleNamespace(operations=ops)


def test_trainable_tensors_are_conv_parameters_only():
    g = fake_graph()
    params = Q.trainable_tensors(g)
    assert len(params) == 4 and all(p.dim() >= 1 for p in params)
    heads = Q.trainable_tensors(g, heads_only=True)
    assert len(heads) == 2 and heads[1].item() == 0.5


def test_resolve_settings_coerces_yaml_strings():
    q = Q.resolve_settings({"lr": "1e-4", "rounds": 3, "final_eval": "false", "batch_size": 16.0})
    assert q["lr"] == pytest.approx(1e-4) and isinstance(q["lr"], float)
    assert q["rounds"] == 3 and q["final_eval"] is False and q["batch_size"] == 16
    assert q["calib_images"] == Q.QAT_DEFAULTS["calib_images"]
    with pytest.raises(ValueError):
        Q.resolve_settings({"learning_rate": 1e-4})


def test_shipped_config_uses_known_settings():
    from tinyseek.config import load_config

    q = Q.resolve_settings(load_config("configs/qat.yaml")["qat"])
    assert q["calib_images"] == 512  # tinyseek.train.export's default


def test_exponent_moves():
    before = {"a": [-7, -8, -5], "b": [-5, -9, -4], "gone": [0]}
    after = {"a": [-7, -8, -5], "b": [-5, -8, -4], "new": [1]}
    assert Q.exponent_moves(before, after) == {"b": [[-5, -9, -4], [-5, -8, -4]]}


def test_exponent_jitter_shifts_by_one_and_restores():
    cfgs = [SimpleNamespace(scale=torch.tensor([2.0 ** -e])) for e in range(3, 9)]
    before = [c.scale.clone() for c in cfgs]
    rng = np.random.default_rng(0)
    seen_shift = False
    for _ in range(20):
        with Q.ExponentJitter(cfgs, 0.5, rng) as shifted:
            n = 0
            for c, b in zip(cfgs, before):
                ratio = float(c.scale / b)
                assert ratio in (0.5, 1.0, 2.0)
                n += ratio != 1.0
            assert n == len(shifted)
            seen_shift |= n > 0
        assert all(torch.equal(c.scale, b) for c, b in zip(cfgs, before))
    assert seen_shift
    with Q.ExponentJitter(cfgs, 0.0, rng) as shifted:
        assert shifted == []


def test_fused_from_graph_copies():
    g = fake_graph()
    fused = Q.fused_from_graph(g)
    assert sorted(fused) == ["backbone.conv_stem", "heatmap.1"]
    fused["heatmap.1"][1].fill_(9.0)
    assert g.operations["b"].inputs[2].value.item() == 0.5
