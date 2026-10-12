"""Test the hypothesis that cuDNN's SDPA backward gives NaN when a row's log-sum-exp is below -ln(FLT_MAX).

A. Cross-tabulates every (sample, head, row) of a captured failing call: lse below -88.72 vs NaN in cuDNN ``dq``,
   split by whether the incoming gradient row is exactly zero.
B. Intervention: adds a constant ``c`` to every allowed logit (softmax and gradients are mathematically unchanged,
   only the log-sum-exp shifts) and counts NaN ``dq`` elements for the cuDNN and efficient backends. Needs a GPU.

Run from the repo root: ``python -m tests.diagnostics.sdpa_lse_threshold runs/nan_repro/sdpa_failing_layer.pt``
"""

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def count_nan_dq(q: Any, k: Any, v: Any, bias: Any, grad: Any, backend: SDPBackend) -> int:
    qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
    with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(backend):
        out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=bias, dropout_p=0.0, is_causal=False, scale=0.125)
    out.backward(grad.to(out.dtype))  # type: ignore[no-untyped-call]
    return int((~torch.isfinite(qq.grad)).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path, help="sdpa_failing_layer.pt")
    ap.add_argument("--shifts", type=float, nargs="*", default=[0.0, 5.0, 10.0, 15.0, 20.0, 40.0])
    args = ap.parse_args()

    rec: dict[str, Any] = torch.load(args.capture)
    q, k, v, mask, g = (rec[n].cuda() for n in ("q", "k", "v", "attn_mask", "grad_output"))
    nan_rows = (~torch.isfinite(rec["standalone_cudnn"]["grads"]["dq"].cuda())).any(-1)
    qb, kb = q.bfloat16(), k.bfloat16()
    scores = (torch.einsum("bhid,bhjd->bhij", qb.float(), kb.float()) * 0.125).masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, -1)
    threshold = -math.log(torch.finfo(torch.float32).max)
    print(f"-ln(FLT_MAX) = {threshold:.3f}   (1/exp(lse) overflows fp32 when lse is below this)")

    print("\n== A. cross-tab over all (sample, head, row): lse below threshold vs dq NaN")
    below = lse < threshold
    zero_grad = g.float().abs().amax(-1) == 0
    for name, sel in (("zero incoming grad", zero_grad), ("non-zero incoming grad", ~zero_grad)):
        print(
            f"{name:24s} rows: {int(sel.sum()):6d} | lse < thr: {int((below & sel).sum()):4d} | "
            f"dq NaN: {int((nan_rows & sel).sum()):4d} | both: {int((below & nan_rows & sel).sum()):4d}"
        )
    print(f"largest lse among NaN rows: {float(lse[nan_rows].max()):.3f} | "
          f"smallest lse among finite rows: {float(lse[~nan_rows].min()):.3f}")

    print("\n== B. intervention: add c to every allowed logit (no change to softmax or gradients)")
    print("       c   min lse   NaN dq elements (cuDNN)   (efficient backend)")
    base = torch.zeros(mask.shape, dtype=torch.bfloat16, device="cuda").masked_fill(~mask, float("-inf"))
    for c in args.shifts:
        bias = base + torch.where(mask, c, 0.0).bfloat16()
        cudnn = count_nan_dq(q, k, v, bias, g, SDPBackend.CUDNN_ATTENTION)
        eff = count_nan_dq(q, k, v, bias, g, SDPBackend.EFFICIENT_ATTENTION)
        print(f"{c:8.1f}  {float(lse.min()) + c:8.2f}  {cudnn:>22d}   {eff:>18d}")


if __name__ == "__main__":
    main()
