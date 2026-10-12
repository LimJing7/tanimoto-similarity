import torch

from tani_predictor.config import ModelConfig
from tani_predictor.model import TaniRegressor


def tiny_model() -> TaniRegressor:
    cfg = ModelConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64)
    torch.manual_seed(0)
    return TaniRegressor(cfg).eval()


def test_forward_shapes_and_range() -> None:
    model = tiny_model()
    ids = torch.randint(4, 142, (3, 10))
    out = model(ids, torch.ones_like(ids), torch.rand(3))
    assert out["logits"].shape == (3,)
    assert bool(((out["logits"] >= 0) & (out["logits"] <= 1)).all())
    assert out["loss"].ndim == 0


def test_padding_does_not_change_prediction() -> None:
    model = tiny_model()
    ids = torch.randint(4, 142, (1, 7))
    padded = torch.cat([ids, torch.zeros(1, 5, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, 7), torch.zeros(1, 5)], dim=1).long()
    with torch.no_grad():
        a = model(ids, torch.ones_like(ids))["logits"]
        b = model(padded, mask)["logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_backbone_uses_sdpa_attention() -> None:
    assert tiny_model().backbone.config._attn_implementation == "sdpa"


def test_cudnn_sdpa_backend_is_disabled() -> None:
    torch.backends.cuda.enable_cudnn_sdp(True)
    tiny_model()
    assert not torch.backends.cuda.cudnn_sdp_enabled()  # type: ignore[no-untyped-call]
