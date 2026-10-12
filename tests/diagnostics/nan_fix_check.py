"""Check that disabling the cuDNN SDPA backend fixes a NaN gradient from a ``nonfinite_step<N>.pt`` dump.

Runs every dumped micro-batch with the cuDNN backend enabled and disabled, comparing gradient norms and the
time of a forward/backward on the last micro-batch. Needs a GPU.

Run from the repo root: ``python -m tests.diagnostics.nan_fix_check runs/base/nonfinite_step1915.pt``
"""

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch

from tani_predictor.config import Config
from tani_predictor.model import TaniRegressor
from tests.diagnostics.nan_dump_analysis import Batch, grad_step


def mean_time(model: TaniRegressor, batch: Batch, repeats: int = 10) -> float:
    grad_step(model, batch, True)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        grad_step(model, batch, True)
    torch.cuda.synchronize()
    return (time.time() - start) / repeats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = ap.parse_args()

    dump: dict[str, Any] = torch.load(args.dump)
    model = TaniRegressor(Config.from_yaml(args.config).model).cuda()
    model.load_state_dict(dump["weights"])
    model.train()

    for label, enabled in (("cuDNN SDPA enabled", True), ("cuDNN SDPA disabled", False)):
        torch.backends.cuda.enable_cudnn_sdp(enabled)
        norms = [grad_step(model, mb, True)[1] for mb in dump["microbatches"]]
        shown = [round(n, 3) if math.isfinite(n) else n for n in norms]
        ms = mean_time(model, dump["microbatches"][-1]) * 1000
        print(f"{label:22s} grad norms {shown} | {ms:.0f} ms per forward+backward")


if __name__ == "__main__":
    main()
