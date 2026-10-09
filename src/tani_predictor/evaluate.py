"""Metrics and checkpoint evaluation.

Run: ``python -m tani_predictor.evaluate --config configs/base.yaml --checkpoint runs/base --split val``
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load_file
from scipy import stats
from torch.utils.data import DataLoader

from tani_predictor.config import Config
from tani_predictor.data import CandidateDataset, Collator
from tani_predictor.model import TaniRegressor
from tani_predictor.tokenizer import TaniTokenizer

MODEL_FILE = "model.pt"


def find_weights(run_dir: Path) -> Path:
    """``model.pt`` if present, else the weights in the newest Trainer ``checkpoint-N`` (interrupted runs)."""
    final = run_dir / MODEL_FILE
    if final.exists():
        return final
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]), reverse=True)
    for ckpt in checkpoints:
        for name in ("model.safetensors", "pytorch_model.bin"):
            if (ckpt / name).exists():
                return ckpt / name
    raise FileNotFoundError(f"no {MODEL_FILE} or checkpoint-*/ weights found in {run_dir}")


def load_weights(model: TaniRegressor, path: Path) -> None:
    state = load_file(path) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu")
    model.load_state_dict(state)


def compute_metrics(
    pred: NDArray[np.float32],
    target: NDArray[np.float32],
    spec_index: NDArray[np.int64],
    tanimoto: NDArray[np.float32],
) -> dict[str, float]:
    """Global regression metrics plus per-spec ranking metrics.

    ``top1_tanimoto`` is the raw Tanimoto of each spec's highest-scored candidate;
    ``best_tanimoto`` is the best available among the evaluated candidates, ``mean_tanimoto``
    the average candidate (what random ranking would give), and ``top1_is_best`` the fraction of
    specs whose top-scored candidate is the best one in the pool.
    """
    metrics: dict[str, float] = {"mse": float(np.mean((pred - target) ** 2))}
    if len(pred) > 1 and np.ptp(pred) > 0 and np.ptp(target) > 0:
        metrics["pearson"] = float(stats.pearsonr(pred, target).statistic)
        metrics["spearman"] = float(stats.spearmanr(pred, target).statistic)
    else:
        metrics["pearson"] = metrics["spearman"] = 0.0

    spec_spearman: list[float] = []
    top1: list[float] = []
    best: list[float] = []
    mean: list[float] = []
    is_best: list[float] = []
    for s in np.unique(spec_index):
        sel = spec_index == s
        p, t, tani = pred[sel], target[sel], tanimoto[sel]
        top1.append(float(tani[int(np.argmax(p))]))
        best.append(float(tani.max()))
        is_best.append(float(tani[int(np.argmax(p))] >= tani.max()))
        mean.append(float(tani.mean()))
        if len(p) > 2 and np.ptp(p) > 0 and np.ptp(t) > 0:
            spec_spearman.append(float(stats.spearmanr(p, t).statistic))
    metrics["mean_spec_spearman"] = float(np.mean(spec_spearman)) if spec_spearman else 0.0
    metrics["top1_tanimoto"] = float(np.mean(top1)) if top1 else 0.0
    metrics["best_tanimoto"] = float(np.mean(best)) if best else 0.0
    metrics["mean_tanimoto"] = float(np.mean(mean)) if mean else 0.0
    metrics["top1_is_best"] = float(np.mean(is_best)) if is_best else 0.0
    return metrics


@torch.no_grad()
def predict(
    model: TaniRegressor, dataset: CandidateDataset, batch_size: int, device: torch.device
) -> NDArray[np.float32]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=Collator(dataset.tokenizer.pad_id))
    out: list[NDArray[np.float32]] = []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))["logits"]
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


ALL_CANDIDATES = 10**9  # larger than any pool, so every candidate of every spec is scored


def evaluate_split(
    model: TaniRegressor,
    cfg: Config,
    split: str,
    device: torch.device,
    batch_size: int,
    candidates_per_spec: int | None = None,
) -> dict[str, float]:
    tokenizer = TaniTokenizer(cfg.data.vocab_path)
    dataset = CandidateDataset(
        cfg.data.processed_dir,
        split,
        tokenizer,
        cfg.data.max_len,
        candidates_per_spec or cfg.data.eval_candidates_per_spec,
        cfg.data.num_target_bins,
        cfg.data.seed,
        cfg.data.max_specs,
        fixed=True,
    )
    if len(dataset) == 0:
        raise SystemExit(f"split {split!r} has no specs in {cfg.data.processed_dir}")
    pred = predict(model, dataset, batch_size, device)
    return compute_metrics(pred, dataset.targets, dataset.spec_index, dataset.tanimoto)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="Run directory (model.pt or checkpoint-*/)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--all-candidates",
        action="store_true",
        help="Score every candidate of every spec (instead of eval_candidates_per_spec) for faithful ranking metrics.",
    )
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="section.key=value")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaniRegressor(cfg.model)
    weights = find_weights(args.checkpoint)
    print(f"loading {weights}")
    load_weights(model, weights)
    model.to(device)
    k = ALL_CANDIDATES if args.all_candidates else None
    metrics = evaluate_split(model, cfg, args.split, device, args.batch_size, k)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
