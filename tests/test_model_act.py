import pytest
import torch
from torch import nn

from tinyseek.train.model import build_model


@pytest.mark.parametrize(("act", "want"), [("relu6", nn.ReLU6), ("relu", nn.ReLU)])
def test_backbone_activation_option(act, want):
    model = build_model({"backbone_act": act, "pretrained": False}).eval()
    kinds = {type(m) for m in model.backbone.modules() if isinstance(m, (nn.ReLU, nn.ReLU6))}
    assert kinds == {want}
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert out["heatmap"].shape == (1, 1, 8, 8)


def test_unknown_backbone_activation_is_refused():
    with pytest.raises(ValueError):
        build_model({"backbone_act": "gelu", "pretrained": False})
