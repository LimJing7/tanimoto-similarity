"""Locate a NaN gradient from a ``nonfinite_step<N>.pt`` dump written by ``NonFiniteGuardTrainer``.

Replays the dumped micro-batches with the dumped (still finite) weights in bf16 and fp32, narrows the
failure down to rows and row pairs, and runs autograd anomaly detection to name the offending backward op.
This is how the cuDNN SDPA backward NaN was found. Needs a GPU.

Run from the repo root: ``python -m tests.diagnostics.nan_dump_analysis runs/base/nonfinite_step1915.pt``
"""

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from tani_predictor.config import Config
from tani_predictor.model import TaniRegressor

Batch = dict[str, Tensor]


def grad_step(model: TaniRegressor, batch: Batch, bf16: bool) -> tuple[float, float]:
    """One forward/backward; returns (loss, unclipped grad norm)."""
    model.zero_grad(set_to_none=True)
    ids, mask, labels = (batch[k].cuda() for k in ("input_ids", "attention_mask", "labels"))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        out = model(ids, mask, labels)
    out["loss"].backward()
    return out["loss"].item(), torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()


def trimmed(batch: Batch, rows: list[int]) -> Batch:
    """The given rows as their own batch, padded only to their longest sequence."""
    longest = int(batch["attention_mask"][rows].sum(dim=1).max())
    return {k: (v[rows, :longest] if v.ndim == 2 else v[rows]) for k, v in batch.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path, help="nonfinite_step<N>.pt (the first dump of a run contains the weights)")
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = ap.parse_args()

    dump: dict[str, Any] = torch.load(args.dump)
    cfg = Config.from_yaml(args.config)
    model = TaniRegressor(cfg.model).cuda()
    model.load_state_dict(dump["weights"])
    model.train()
    # The guard's dump is only useful if the NaN is reproduced with the same backend settings as training.
    torch.backends.cuda.enable_cudnn_sdp(True)

    weights_ok = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    print(f"step {dump['global_step']}, grad_norm {dump['grad_norm']}, {len(dump['microbatches'])} micro-batches")
    print(f"weights finite: {weights_ok}, max |w| = {max(p.abs().max().item() for p in model.parameters()):.2f}")

    print("\n== micro-batches: bf16 vs fp32 grad norm")
    bad: list[Batch] = []
    for i, mb in enumerate(dump["microbatches"]):
        (loss_b, gn_b), (loss_f, gn_f) = grad_step(model, mb, True), grad_step(model, mb, False)
        padded_len = mb["input_ids"].shape[1]
        print(f"micro-batch {i}: padded_len {padded_len} | bf16 loss {loss_b:.4f} grad {gn_b}"
              f" | fp32 loss {loss_f:.4f} grad {gn_f}")
        if not math.isfinite(gn_b):
            bad.append(mb)
    if not bad:
        print("no micro-batch reproduces a non-finite bf16 gradient")
        return

    mb = bad[0]
    n_rows = mb["input_ids"].shape[0]
    lengths = mb["attention_mask"].sum(dim=1).tolist()
    print(f"\n== first bad micro-batch, rows alone (no padding); lengths {lengths}")
    for r in range(n_rows):
        gn = grad_step(model, trimmed(mb, [r]), True)[1]
        print(f"row {r}: len {int(lengths[r]):5d} bf16 grad {gn:.4f}" + ("   <-- NaN" if not math.isfinite(gn) else ""))

    longest = max(range(n_rows), key=lambda r: lengths[r])
    print(f"\n== longest row ({longest}) paired with each other row (padded batch)")
    for r in range(n_rows):
        if r != longest:
            gn = grad_step(model, trimmed(mb, sorted([r, longest])), True)[1]
            print(f"rows {sorted([r, longest])}: bf16 grad {gn}")

    print("\n== autograd anomaly detection on the first bad micro-batch (bf16)")
    try:
        with torch.autograd.set_detect_anomaly(True):
            grad_step(model, mb, True)
        print("no anomaly raised")
    except RuntimeError as err:
        print(str(err)[:300])


if __name__ == "__main__":
    main()
