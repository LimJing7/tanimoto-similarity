"""Validated configuration and manifest models (pydantic v2)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

PACKAGE_DIR: Path = Path(__file__).resolve().parent
DEFAULT_VOCAB_PATH: Path = PACKAGE_DIR / "vocab" / "tani-vocab.json"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Frozen):
    vocab_size: int = Field(142, gt=0)
    hidden_size: int = Field(768, gt=0)
    num_hidden_layers: int = Field(6, gt=0)
    num_attention_heads: int = Field(12, gt=0)
    intermediate_size: int = Field(3072, gt=0)
    max_position_embeddings: int = Field(8192, gt=0)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    gradient_checkpointing: bool = False


class DataConfig(_Frozen):
    processed_dir: Path = Path("data/processed")
    vocab_path: Path = DEFAULT_VOCAB_PATH
    max_len: int = Field(8192, gt=0)
    candidates_per_spec: int = Field(128, gt=0)
    eval_candidates_per_spec: int = Field(256, gt=0)
    num_target_bins: int = Field(10, gt=0)
    max_specs: int | None = Field(None, gt=0)
    seed: int = 0


class TrainConfig(_Frozen):
    output_dir: Path = Path("runs/base")
    learning_rate: float = Field(3e-4, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    num_epochs: float = Field(1.0, gt=0)
    max_steps: int = -1
    per_device_batch_size: int = Field(8, gt=0)
    gradient_accumulation_steps: int = Field(4, gt=0)
    bf16: bool = True
    num_workers: int = Field(4, ge=0)
    logging_steps: int = Field(50, gt=0)
    eval_steps: int = Field(1000, gt=0)
    save_steps: int = Field(1000, gt=0)
    seed: int = 0
    report_to: Literal["none", "wandb"] = "none"


class Config(_Frozen):
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()

    @classmethod
    def from_yaml(cls, path: Path, overrides: Sequence[str] = ()) -> Self:
        """Load a YAML file and apply ``section.key=value`` overrides before validating."""
        raw: Any = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a mapping at the top level")
        for item in overrides:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"override must look like section.key=value, got {item!r}")
            node: dict[str, Any] = raw
            *parents, leaf = key.split(".")
            for part in parents:
                child = node.setdefault(part, {})
                if not isinstance(child, dict):
                    raise ValueError(f"cannot override {key!r}: {part!r} is not a section")
                node = child
            node[leaf] = yaml.safe_load(value)
        return cls.model_validate(raw)


class Manifest(_Frozen):
    """Provenance record written next to the processed dataset."""

    results_root: str
    experiments: list[str]
    n_tsv_files: int
    n_specs: int
    n_rows: int
    n_rows_before_dedupe: int
    specs_per_split: dict[str, int]
    rows_per_split: dict[str, int]
    gt_smiles_non_identical: int
    gt_smiles_formula_mismatches: int
    n_smiles_stereo_stripped: int
    max_tanimoto_diff_on_merge: float
    quantile_fit_values: int
    seed: int
    created: str
