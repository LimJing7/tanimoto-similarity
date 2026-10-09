"""Train the regressor with the HF Trainer.

Run: ``python -m tani_predictor.train --config configs/base.yaml [--set train.learning_rate=1e-4 ...]``
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import EvalPrediction, Trainer, TrainerCallback, TrainingArguments

from tani_predictor.config import Config
from tani_predictor.data import CandidateDataset, Collator
from tani_predictor.evaluate import MODEL_FILE, compute_metrics
from tani_predictor.model import TaniRegressor, count_parameters
from tani_predictor.tokenizer import TaniTokenizer


class ResampleCallback(TrainerCallback):
    """Draw a fresh stratified candidate sample at the start of every epoch after the first."""

    def __init__(self, dataset: CandidateDataset) -> None:
        self.dataset = dataset
        self.epoch = 0

    def on_epoch_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        if self.epoch > 0:
            self.dataset.resample(self.epoch)
        self.epoch += 1


def make_dataset(cfg: Config, tokenizer: TaniTokenizer, split: str, evaluation: bool) -> CandidateDataset:
    return CandidateDataset(
        cfg.data.processed_dir,
        split,
        tokenizer,
        cfg.data.max_len,
        cfg.data.eval_candidates_per_spec if evaluation else cfg.data.candidates_per_spec,
        cfg.data.num_target_bins,
        cfg.data.seed,
        cfg.data.max_specs,
        fixed=evaluation,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="section.key=value")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config, args.overrides)
    tokenizer = TaniTokenizer(cfg.data.vocab_path)
    if len(tokenizer) != cfg.model.vocab_size:
        raise SystemExit(f"vocab has {len(tokenizer)} tokens but model.vocab_size={cfg.model.vocab_size}")

    train_ds = make_dataset(cfg, tokenizer, "train", evaluation=False)
    eval_ds = make_dataset(cfg, tokenizer, "val", evaluation=True)
    has_eval = len(eval_ds) > 0
    print(f"train: {len(train_ds):,} examples from {len(train_ds.spec_ids):,} specs; "
          f"val: {len(eval_ds):,} examples" + ("" if has_eval else " (no val specs: evaluation disabled)"))

    model = TaniRegressor(cfg.model)
    print(f"parameters: {count_parameters(model):,}")

    def metrics_fn(pred: EvalPrediction) -> dict[str, float]:
        scores: NDArray[np.float32] = np.asarray(pred.predictions, dtype=np.float32).reshape(-1)
        return compute_metrics(scores, eval_ds.targets, eval_ds.spec_index, eval_ds.tanimoto)

    t = cfg.train
    training_args = TrainingArguments(
        output_dir=str(t.output_dir),
        per_device_train_batch_size=t.per_device_batch_size,
        per_device_eval_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        num_train_epochs=t.num_epochs,
        max_steps=t.max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=t.warmup_ratio,
        bf16=t.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=t.num_workers,
        dataloader_persistent_workers=False,  # workers must be re-created each epoch to see the resampled data
        logging_steps=t.logging_steps,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=t.eval_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_mean_spec_spearman" if has_eval else None,
        greater_is_better=True,
        seed=t.seed,
        report_to=t.report_to,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if has_eval else None,
        data_collator=Collator(tokenizer.pad_id),
        compute_metrics=metrics_fn if has_eval else None,
        callbacks=[ResampleCallback(train_ds)],
    )
    trainer.train()

    if trainer.is_world_process_zero():
        t.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.accelerator.unwrap_model(model).state_dict(), t.output_dir / MODEL_FILE)
        (t.output_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
        (t.output_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
        print(f"saved model to {t.output_dir / MODEL_FILE}")


if __name__ == "__main__":
    main()
