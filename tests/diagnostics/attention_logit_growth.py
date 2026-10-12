"""Why are some attention logits so negative, and do they grow during training?

Part 1 looks at one captured failing SDPA call (``sdpa_failing_layer.pt``): the q/k norms and the q_i.k_i cosine at the
first positions, the score block among the first keys, and how later rows score against the first tokens.
Part 2 runs the dumped failing micro-batch through the model with several weight sets (random init, an optional
healthy checkpoint, the dump's weights) and reports per layer the lowest row log-sum-exp, the share of rows below
-ln(FLT_MAX), the largest |score| and the median q/k norms. Needs a GPU.

Run from the repo root:
``python -m tests.diagnostics.attention_logit_growth runs/nan_repro/sdpa_failing_layer.pt \\
    /tmp/repro_run/nonfinite_step1915.pt --checkpoint runs/base/checkpoint-1000/model.safetensors``
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor

from tani_predictor.config import Config
from tani_predictor.model import TaniRegressor

THRESHOLD = -math.log(torch.finfo(torch.float32).max)  # -88.72: 1/exp(lse) overflows fp32 below this


def structure_of_failing_rows(rec: dict[str, Any], mb: dict[str, Tensor], inverse: dict[int, str], sample: int) -> None:
    nan_rows = (~torch.isfinite(rec["standalone_cudnn"]["grads"]["dq"])).any(-1)  # (sample, head, row)
    head = int(nan_rows.sum(dim=(0, 2)).argmax())
    q, k = rec["q"][sample, head].float(), rec["k"][sample, head].float()
    mask = rec["attn_mask"][sample, 0]
    length = int(mb["attention_mask"][sample].sum())
    scores = (q @ k.T) * float(rec["scale"])
    print(f"== 1. structure of the failing rows (captured layer {rec['layer']}, sample {sample}, head {head})")
    print("first tokens:", [inverse[int(i)] for i in mb["input_ids"][sample, :22]])
    for name, t in (("q", q), ("k", k)):
        norms = t.norm(dim=-1)
        print(f"|{name}| positions 0..5: {[round(float(x), 1) for x in norms[:6]]} | median over positions: "
              f"{float(norms.median()):.1f}")
    cos = (q * k).sum(-1) / (q.norm(dim=-1) * k.norm(dim=-1))
    print(f"cos(q_i, k_i) i=0..5: {[round(float(x), 2) for x in cos[:6]]} | "
          f"self scores {[round(float(scores[i, i]), 1) for i in range(6)]}")
    print("score block, rows 0..5 x keys 0..5 (None = masked):")
    for i in range(6):
        print("   ", [round(float(scores[i, j]), 1) if mask[i, j] else None for j in range(6)])
    late = torch.arange(200, length)
    late_lse = torch.logsumexp(scores[late].masked_fill(~mask[late], float("-inf")), -1)
    print(f"later rows 200..{length - 1}: mean score vs key 0 {float(scores[late, 0].mean()):.1f}, vs key 1 "
          f"{float(scores[late, 1].mean()):.1f}, median over keys {float(scores[late][:, :length].median()):.1f}, "
          f"mean row lse {float(late_lse.mean()):.1f}")


def logit_stats(model: TaniRegressor, mb: dict[str, Tensor]) -> list[dict[str, float]]:
    """Per layer: lowest valid-row lse, share of rows below the threshold, max |score|, median q/k norms."""
    attn_mask = mb["attention_mask"].cuda()
    valid = attn_mask.bool()
    stats: list[dict[str, float]] = []
    original = F.scaled_dot_product_attention

    def recorder(q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None, **kw: Any) -> Tensor:
        assert attn_mask is not None
        out: Tensor = original(q, k, v, attn_mask=attn_mask, **kw)
        scores = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * float(kw.get("scale", 0.125))
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        lse = torch.logsumexp(scores, -1)  # (batch, head, row)
        rows = valid[:, None, :].expand_as(lse)
        stats.append({
            "min_lse": float(lse[rows].min()),
            "below": float((lse[rows] < THRESHOLD).float().mean()),
            "max_abs": float(scores[torch.isfinite(scores)].abs().max()),
            "q": float(q.float().norm(dim=-1)[rows].median()),
            "k": float(k.float().norm(dim=-1)[rows].median()),
        })
        return out

    F.scaled_dot_product_attention = recorder  # type: ignore[assignment]
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(mb["input_ids"].cuda(), attn_mask)
    finally:
        F.scaled_dot_product_attention = original
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path, help="sdpa_failing_layer.pt")
    ap.add_argument("dump", type=Path, help="nonfinite_step<N>.pt (must contain the weights)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="healthy model.safetensors to compare against")
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--microbatch", type=int, default=None, help="micro-batch index in the dump (default: capture)")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    rec: dict[str, Any] = torch.load(args.capture)
    dump: dict[str, Any] = torch.load(args.dump)
    index = args.microbatch if args.microbatch is not None else rec.get("microbatch_index")
    if index is None:
        raise SystemExit("this capture does not record its micro-batch; pass --microbatch N")
    mb: dict[str, Tensor] = dump["microbatches"][index]
    vocab: dict[str, int] = json.loads(cfg.data.vocab_path.read_text())
    structure_of_failing_rows(rec, mb, {i: t for t, i in vocab.items()}, args.sample)

    print("\n== 2. do the logits grow with training? same micro-batch, different weights (valid rows only)")
    weight_sets: dict[str, dict[str, Tensor] | None] = {"random init": None}
    if args.checkpoint is not None:
        weight_sets[f"checkpoint {args.checkpoint.parent.name}"] = load_file(args.checkpoint)
    weight_sets[f"dump (step {dump['global_step']})"] = dump["weights"]
    for name, weights in weight_sets.items():
        torch.manual_seed(0)
        model = TaniRegressor(cfg.model).cuda().eval()
        if weights is not None:
            model.load_state_dict(weights)
        print(f"{name}:")
        print("  layer  min lse   rows lse<-88.7   max|score|   median|q|  median|k|")
        for layer, s in enumerate(logit_stats(model, mb)):
            print(f"  {layer:5d}  {s['min_lse']:8.1f}  {s['below']:13.4%}  {s['max_abs']:10.1f}  "
                  f"{s['q']:9.1f}  {s['k']:9.1f}")


if __name__ == "__main__":
    main()
