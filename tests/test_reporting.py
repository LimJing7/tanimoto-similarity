import dataclasses
import os
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import Dataset

from tani_predictor.config import ModelConfig, TrainConfig
from tani_predictor.data import Collator, Item
from tani_predictor.guard import NonFiniteGuardTrainer
from tani_predictor.model import TaniRegressor
from tani_predictor.train import configure_reporting, make_training_args


class ToyData(Dataset[Item]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> Item:
        return {"input_ids": [1, 4, 5 + index, 6, 2], "labels": 0.5}


def test_report_to_accepts_tensorboard_and_rejects_unknown() -> None:
    assert TrainConfig(report_to="tensorboard").report_to == "tensorboard"
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"report_to": "mlflow"})


def test_configure_reporting_pins_a_stable_log_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TENSORBOARD_LOGGING_DIR", raising=False)
    configure_reporting(TrainConfig(output_dir=tmp_path, report_to="none"))
    assert "TENSORBOARD_LOGGING_DIR" not in os.environ
    configure_reporting(TrainConfig(output_dir=tmp_path, report_to="tensorboard"))
    assert os.environ["TENSORBOARD_LOGGING_DIR"] == str(tmp_path / "tensorboard")


def test_configure_reporting_respects_an_existing_setting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TENSORBOARD_LOGGING_DIR", "/somewhere/else")
    configure_reporting(TrainConfig(output_dir=tmp_path, report_to="tensorboard"))
    assert os.environ["TENSORBOARD_LOGGING_DIR"] == "/somewhere/else"


def test_make_training_args_maps_reporting_and_eval(tmp_path: Path) -> None:
    args = make_training_args(TrainConfig(output_dir=tmp_path, report_to="tensorboard"), has_eval=True)
    assert args.report_to == ["tensorboard"]
    assert args.eval_strategy == "steps" and args.load_best_model_at_end
    no_eval = make_training_args(TrainConfig(output_dir=tmp_path), has_eval=False)
    assert no_eval.eval_strategy == "no" and not no_eval.load_best_model_at_end


def test_training_writes_tensorboard_scalars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_dir = tmp_path / "tensorboard"
    monkeypatch.setenv("TENSORBOARD_LOGGING_DIR", str(log_dir))
    cfg = ModelConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    train_cfg = TrainConfig(
        output_dir=tmp_path, report_to="tensorboard", max_steps=3, per_device_batch_size=2,
        logging_steps=1, num_workers=0, bf16=False, eval_steps=1000, save_steps=1000,
    )
    args = dataclasses.replace(make_training_args(train_cfg, has_eval=False), use_cpu=True)
    trainer = NonFiniteGuardTrainer(
        model=TaniRegressor(cfg), args=args, train_dataset=ToyData(), data_collator=Collator(0)
    )
    trainer.train()
    del trainer

    events = EventAccumulator(str(log_dir))
    events.Reload()
    tags = set(events.Tags()["scalars"])
    assert {"train/loss", "train/grad_norm", "train/learning_rate"} <= tags
    assert [p.step for p in events.Scalars("train/loss")] == [1, 2, 3]
    assert torch.isfinite(torch.tensor(events.Scalars("train/loss")[-1].value))
