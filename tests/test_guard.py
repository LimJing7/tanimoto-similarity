from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments

from tani_predictor.config import ModelConfig
from tani_predictor.data import Collator, Item
from tani_predictor.guard import NonFiniteGuardTrainer
from tani_predictor.model import TaniRegressor


class ToyData(Dataset[Item]):
    """Eight examples; example 3 has a NaN label, so its batch yields a NaN loss and gradient."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> Item:
        torch.manual_seed(index)
        ids = [1, 4, *torch.randint(6, 100, (5,)).tolist(), 2]
        return {"input_ids": ids, "labels": float("nan") if index == 3 else 0.5}


def test_nan_step_is_skipped_and_weights_stay_finite(tmp_path: Path) -> None:
    cfg = ModelConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    model = TaniRegressor(cfg)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        max_steps=4,
        use_cpu=True,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
        seed=0,
        data_seed=0,
    )
    # sequential sampling makes the NaN example (index 3) land in the second batch
    trainer = NonFiniteGuardTrainer(model=model, args=args, train_dataset=ToyData(), data_collator=Collator(0))
    trainer._get_train_sampler = lambda *a, **k: torch.utils.data.SequentialSampler(ToyData())  # type: ignore[method-assign]
    trainer.train()

    assert trainer.nonfinite_steps == 1
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    dumps = list(tmp_path.glob("nonfinite_step*.pt"))
    assert len(dumps) == 1
    dump = torch.load(dumps[0])
    assert "weights" in dump and len(dump["microbatches"]) == 1
