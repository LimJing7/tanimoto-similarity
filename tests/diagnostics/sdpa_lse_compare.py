"""Compare cuDNN's forward log-sum-exp with a float32 reference on a captured failing SDPA call.

Calls the raw ``aten._scaled_dot_product_cudnn_attention`` op to get its saved log-sum-exp (the softmax normalizer
that the backward kernel uses) and compares it with ``logsumexp`` of the masked scores computed directly. For the
rows whose ``dq`` is NaN it prints the number of visible keys, the max score and both log-sum-exp values, to show
whether any inf/-inf is produced in the forward pass. Needs a GPU.

Run from the repo root: ``python -m tests.diagnostics.sdpa_lse_compare runs/nan_repro/sdpa_failing_layer.pt``
"""

import argparse
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path, help="sdpa_failing_layer.pt")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--head", type=int, default=6, help="head to print row-by-row")
    ap.add_argument("--rows", type=int, default=22, help="number of leading query rows to print")
    args = ap.parse_args()

    rec: dict[str, Any] = torch.load(args.capture)
    q, k, v, mask = (rec[n].cuda() for n in ("q", "k", "v", "attn_mask"))
    nan_rows = (~torch.isfinite(rec["standalone_cudnn"]["grads"]["dq"])).any(-1).cuda()  # (sample, head, row)
    qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
    scale = float(rec["scale"])
    bias = torch.zeros(mask.shape, dtype=torch.bfloat16, device="cuda").masked_fill(~mask, float("-inf"))

    res = torch.ops.aten._scaled_dot_product_cudnn_attention(qb, kb, vb, bias, True, 0.0, False, False, scale=scale)
    scores = (torch.einsum("bhid,bhjd->bhij", qb.float(), kb.float()) * scale).masked_fill(~mask, float("-inf"))
    lse_ref = torch.logsumexp(scores, -1)  # (sample, head, row)
    lse = res[1].reshape(lse_ref.shape).float()
    print(f"cuDNN op returned {len(res)} items; lse {tuple(res[1].shape)} {res[1].dtype}")
    bad_cudnn, bad_ref = int((~torch.isfinite(lse)).sum()), int((~torch.isfinite(lse_ref)).sum())
    print("cuDNN lse non-finite:", bad_cudnn, "| ref lse non-finite:", bad_ref)
    diff = (lse - lse_ref).abs()
    print(f"|lse_cudnn - lse_ref|: max over healthy rows {float(diff[~nan_rows].max()):.3g}; "
          f"max over the dq-NaN rows {float(diff[nan_rows].max()):.3g}")

    s, h = args.sample, args.head
    print(f"\nsample {s}, head {h}: row | keys visible | max score | lse_ref | lse_cudnn | dq NaN")
    for i in range(args.rows):
        print(
            f"  row {i:3d}  keys {int(mask[s, 0, i].sum()):4d}  max score {float(scores[s, h, i].max()):8.2f}  "
            f"lse_ref {float(lse_ref[s, h, i]):8.3f}  lse_cudnn {float(lse[s, h, i]):8.3f}  "
            f"dq NaN: {bool(nan_rows[s, h, i])}"
        )
    probs = torch.exp(scores - lse.unsqueeze(-1))
    print("\nP = exp(score - lse_cudnn): non-finite", int((~torch.isfinite(probs)).sum()),
          "| rows where sum(P) != 1 (|err| > 1e-2):", int(((probs.sum(-1) - 1).abs() > 1e-2).sum()))


if __name__ == "__main__":
    main()
