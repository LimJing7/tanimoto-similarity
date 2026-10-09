from pathlib import Path

import pytest
from pydantic import ValidationError

from tani_predictor.config import Config, ModelConfig, TrainConfig


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"hidden_size": 64, "hidden_sizes": 1})


def test_invalid_value_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(hidden_size=0)


def test_from_yaml_with_overrides(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  hidden_size: 64\ntrain:\n  learning_rate: 0.001\n")
    cfg = Config.from_yaml(path, ["train.learning_rate=5e-4", "data.max_specs=7"])
    assert cfg.model.hidden_size == 64
    assert cfg.train.learning_rate == 5e-4
    assert cfg.data.max_specs == 7


def test_shipped_configs_validate() -> None:
    root = Path(__file__).resolve().parent.parent / "configs"
    for name in ("base.yaml", "tiny.yaml"):
        Config.from_yaml(root / name)


def test_save_steps_must_align_with_eval_steps() -> None:
    TrainConfig(eval_steps=500, save_steps=1000)
    with pytest.raises(ValidationError, match="multiple of eval_steps"):
        TrainConfig(eval_steps=1000, save_steps=1500)
