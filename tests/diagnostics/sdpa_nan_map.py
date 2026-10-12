"""Inspect a captured failing SDPA call: input values, mask structure, and where the cuDNN ``dq`` NaN lands.

Uses ``sdpa_failing_layer.pt`` from ``capture_sdpa_tensors``. Reports whether q/k/v/grad_output contain NaN, inf or
exact zeros, whether any query row is fully masked, and which (sample, head, row) elements of the cuDNN ``dq`` are
NaN relative to the padding and to rows whose incoming gradient is exactly zero.

Run from the repo root: ``python -m tests.diagnostics.sdpa_nan_map runs/nan_repro/sdpa_failing_layer.pt``
"""

import argparse
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path, help="sdpa_failing_layer.pt")
    args = ap.parse_args()

    rec: dict[str, Any] = torch.load(args.capture)
    q, k, v, mask, g = (rec[n] for n in ("q", "k", "v", "attn_mask", "grad_output"))
    dq = rec["standalone_cudnn"]["grads"]["dq"]
    dq_ok = rec["standalone_default_backends"]["grads"]["dq"]
    n_samples, _, length, _ = q.shape
    print(f"q/k {tuple(q.shape)} {q.dtype}, v {v.dtype}, mask {mask.dtype} {tuple(mask.shape)}, grad_output {g.dtype}")

    print("\n== 1. inputs: any inf/nan, extreme values, exact zeros")
    for name, t in (("q", q), ("k", k), ("v", v), ("grad_output", g)):
        t = t.float()
        print(
            f"{name:12s} nan {int(torch.isnan(t).sum())} inf {int(torch.isinf(t).sum())} "
            f"max|x| {t.abs().max():.3g} frac exactly 0: {(t == 0).float().mean():.1%}"
        )

    print("\n== 2. mask (True = attend)")
    lengths: list[int] = mask[:, 0, -1].sum(-1).tolist()  # the last query row sees every real key
    print("per-sample real lengths:", lengths, "| padded length", length)
    print("query rows that can attend to NO key (all -inf):", int((~mask.any(-1)).sum()), "of", n_samples * length)

    print("\n== 3. where is the cuDNN dq NaN?")
    nan = ~torch.isfinite(dq)
    rows = nan.any(-1)  # (sample, head, row)
    print("total NaN elements:", int(nan.sum()), "of", dq.numel())
    for b in range(n_samples):
        if not rows[b].any():
            continue
        idx = rows[b].any(0).nonzero().flatten()
        heads = sorted(set(rows[b].any(1).nonzero().flatten().tolist()))
        print(
            f"sample {b} (len {lengths[b]}): {int(rows[b].sum())} NaN (head,row) pairs; heads {heads}; "
            f"rows {int(idx.min())}..{int(idx.max())}; padded rows among them: {int((idx >= lengths[b]).sum())}; "
            f"whole rows NaN: {bool(nan[b][rows[b]].all())}"
        )
    zero_rows = g.abs().amax(-1) == 0
    both = int((rows & zero_rows).sum())
    print("NaN rows that also have an all-zero incoming gradient:", both, "of", int(rows.sum()))
    print(
        "reference (non-cuDNN) dq: max|dq| overall", float(dq_ok.abs().max()),
        "| max|dq| at the cuDNN-NaN positions", float(dq_ok[nan].abs().max()),
    )


if __name__ == "__main__":
    main()
