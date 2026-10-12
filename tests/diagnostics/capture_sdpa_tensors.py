"""Capture every tensor involved in a micro-batch whose bf16 backward gives NaN gradients.

For the first micro-batch of a ``nonfinite_step<N>.pt`` dump with a non-finite bf16 gradient, this records the
model inputs and, for every layer's ``scaled_dot_product_attention`` call, its query/key/value, attention mask,
arguments and the gradient flowing into its output. Each layer's call is then replayed standalone (cuDNN backend
vs the default backends) to show which layer produces NaN on its own. NaN gradients flow down the network, so
only layers whose recorded inputs and incoming gradient are all finite can be judged; the others are reported as
contaminated. Everything is saved to one file, and the culprit call alone to a slim file that ``sdpa_repro``
replays without the model. Needs a GPU.

Run from the repo root: ``python -m tests.diagnostics.capture_sdpa_tensors runs/base/nonfinite_step1915.pt``
"""

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from tani_predictor.config import Config
from tani_predictor.model import TaniRegressor
from tests.diagnostics.nan_dump_analysis import Batch, grad_step

Record = dict[str, Any]


def run_sdpa(rec: Record, cudnn: bool) -> dict[str, Any]:
    """Replay one recorded SDPA call standalone; returns finiteness of output and input gradients."""
    torch.backends.cuda.enable_cudnn_sdp(cudnn)
    q, k, v = (rec[n].cuda().requires_grad_(True) for n in ("q", "k", "v"))
    mask = None if rec["attn_mask"] is None else rec["attn_mask"].cuda()
    # The real call happens under bf16 autocast (q/k arrive as fp32, v as bf16 and are cast inside the op).
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=rec["dropout_p"], is_causal=rec["is_causal"], scale=rec["scale"]
        )
    out.backward(rec["grad_output"].cuda().to(out.dtype))  # type: ignore[no-untyped-call]
    grads = {n: t.grad for n, t in (("dq", q), ("dk", k), ("dv", v)) if t.grad is not None}
    return {
        "out_finite": bool(torch.isfinite(out).all()),
        **{f"{n}_finite": bool(torch.isfinite(g).all()) for n, g in grads.items()},
        "grads": {n: g.detach().cpu() for n, g in grads.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--out", type=Path, default=Path("runs/nan_repro/sdpa_capture.pt"))
    ap.add_argument("--slim-out", type=Path, default=Path("runs/nan_repro/sdpa_failing_layer.pt"))
    args = ap.parse_args()

    dump: dict[str, Any] = torch.load(args.dump)
    model = TaniRegressor(Config.from_yaml(args.config).model).cuda()
    model.load_state_dict(dump["weights"])
    model.train()
    torch.backends.cuda.enable_cudnn_sdp(True)

    bad: tuple[int, Batch] | None = None
    for i, mb in enumerate(dump["microbatches"]):
        if not math.isfinite(grad_step(model, mb, True)[1]):
            bad = (i, mb)
            break
    if bad is None:
        raise SystemExit("no micro-batch in this dump reproduces a non-finite bf16 gradient")
    index, mb = bad

    records: list[Record] = []
    original = F.scaled_dot_product_attention

    def recorder(q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None, **kw: Any) -> Tensor:
        out: Tensor = original(q, k, v, attn_mask=attn_mask, **kw)
        rec: Record = {
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "v": v.detach().cpu(),
            "attn_mask": None if attn_mask is None else attn_mask.detach().cpu(),
            "dtypes": {"q": str(q.dtype), "k": str(k.dtype), "v": str(v.dtype)},
            "dropout_p": kw.get("dropout_p", 0.0),
            "is_causal": kw.get("is_causal", False),
            "scale": kw.get("scale"),
        }
        out.register_hook(lambda g: rec.__setitem__("grad_output", g.detach().cpu()))  # type: ignore[no-untyped-call]
        records.append(rec)
        return out

    F.scaled_dot_product_attention = recorder  # type: ignore[assignment]
    try:
        grad_step(model, mb, True)
    finally:
        F.scaled_dot_product_attention = original

    print(f"micro-batch {index}: {len(records)} SDPA calls captured")
    print("layer  q shape               mask             q/k/v+grad finite  cuDNN standalone   other backends")
    culprits: list[int] = []
    for layer, rec in enumerate(records):
        clean = all(bool(torch.isfinite(rec[n]).all()) for n in ("q", "k", "v", "grad_output"))
        rec["inputs_and_grad_output_finite"] = clean
        cudnn, other = run_sdpa(rec, True), run_sdpa(rec, False)
        rec["standalone_cudnn"], rec["standalone_default_backends"] = cudnn, other

        def verdict(res: dict[str, Any], judged: bool = clean) -> str:
            if not judged:
                return "n/a (contaminated)"
            bad = [n for n in ("out", "dq", "dk", "dv") if not res.get(f"{n}_finite", True)]
            return "NaN in " + ",".join(bad) if bad else "finite"

        if clean and verdict(cudnn) != "finite" and verdict(other) == "finite":
            culprits.append(layer)
        mask = rec["attn_mask"]
        mask_desc = "None" if mask is None else f"{str(mask.dtype).removeprefix('torch.')} {tuple(mask.shape)}"
        print(f"{layer:5d}  {tuple(rec['q'].shape)!s:21s} {mask_desc:16s} {clean!s:18s} {verdict(cudnn):18s} "
              f"{verdict(other)}")
    print(f"culprit layer(s) (clean inputs, cuDNN NaN, other backends finite): {culprits}")
    torch.backends.cuda.enable_cudnn_sdp(True)

    versions = {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),  # type: ignore[no-untyped-call]
        "gpu": torch.cuda.get_device_name(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_dump": str(args.dump),
            "global_step": dump["global_step"],
            "microbatch_index": index,
            "microbatch": mb,
            "layers": records,
            "versions": versions,
        },
        args.out,
    )
    print(f"saved {args.out} ({args.out.stat().st_size / 2**20:.0f} MiB)")
    if culprits:
        torch.save(
            {**records[culprits[-1]], "layer": culprits[-1], "microbatch_index": index, "versions": versions},
            args.slim_out,
        )
        print(f"saved culprit layer {culprits[-1]} alone to {args.slim_out} "
              f"({args.slim_out.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
