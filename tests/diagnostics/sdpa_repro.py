"""Standalone repro (torch only) of the cuDNN SDPA bf16 NaN gradient, from ``sdpa_failing_layer.pt``.

Replays one captured ``scaled_dot_product_attention`` call (real q/k/v, boolean padding mask, incoming
gradient) under each SDPA backend and reports which outputs and gradients are non-finite. Needs a GPU.

Run: ``python tests/diagnostics/sdpa_repro.py runs/nan_repro/sdpa_failing_layer.pt``
"""

import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main(path: str) -> None:
    rec: dict[str, Any] = torch.load(path)
    print("versions:", rec["versions"], "| captured from layer", rec["layer"])
    mask = rec["attn_mask"]
    print(f"q {tuple(rec['q'].shape)} {rec['dtypes']['q']}, k {rec['dtypes']['k']}, v {rec['dtypes']['v']}, "
          f"mask {mask.dtype!s} {tuple(mask.shape)}, dropout_p={rec['dropout_p']}, is_causal={rec['is_causal']}, "
          f"scale={rec['scale']}")
    backends = [
        ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
    ]
    for name, backend in backends:
        q, k, v = (rec[n].cuda().requires_grad_(True) for n in ("q", "k", "v"))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask.cuda(), dropout_p=rec["dropout_p"], is_causal=rec["is_causal"],
                    scale=rec["scale"],
                )
            out.backward(rec["grad_output"].cuda().to(out.dtype))  # type: ignore[no-untyped-call]
        except RuntimeError as err:
            print(f"{name:20s} unavailable: {str(err)[:80]}")
            continue
        tensors = (("out", out), ("dq", q.grad), ("dk", k.grad), ("dv", v.grad))
        finite = {n: bool(torch.isfinite(t).all()) for n, t in tensors}
        print(f"{name:20s} finite: {finite}")


if __name__ == "__main__":
    main(sys.argv[1])
