"""A Trainer that skips optimizer steps with non-finite gradients instead of poisoning the weights."""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import Trainer


class NonFiniteGradientError(RuntimeError):
    pass


class NonFiniteGuardTrainer(Trainer):
    """Skips a step whose gradient norm is NaN/inf (the weights stay untouched) and records what caused it.

    The HF Trainer clips gradients with a NaN norm by multiplying them by NaN and then steps the
    optimizer, which turns every weight into NaN for good. Here the gradients are zeroed instead.
    On the first few events the micro-batches of the step (and, once, the still-finite weights) are
    saved to ``<output_dir>/nonfinite_step<N>.pt`` so the failure can be replayed. Training aborts after
    ``max_consecutive`` skipped steps in a row, since the model is then clearly diverging.
    """

    max_dumps: int = 3
    max_consecutive: int = 20

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.nonfinite_steps = 0
        self._consecutive = 0
        self._window: list[dict[str, Tensor]] = []

    def training_step(self, model: nn.Module, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> Tensor:
        self._window.append({k: v.detach().cpu() for k, v in inputs.items() if isinstance(v, Tensor)})
        loss: Tensor = super().training_step(model, inputs, *args, **kwargs)
        return loss

    def _clip_grad_norm(self, model: nn.Module) -> Tensor:
        norm: Tensor = super()._clip_grad_norm(model)  # type: ignore[no-untyped-call]
        window, self._window = self._window, []
        if bool(torch.isfinite(norm)):
            self._consecutive = 0
            return norm

        self.nonfinite_steps += 1
        self._consecutive += 1
        step = self.state.global_step + 1
        print(f"WARNING: non-finite gradient norm at step {step}; skipping the update ({self.nonfinite_steps} so far)")
        if self.is_world_process_zero() and self.nonfinite_steps <= self.max_dumps:
            dump: dict[str, Any] = {"global_step": step, "grad_norm": float(norm), "microbatches": window}
            if self.nonfinite_steps == 1:
                dump["weights"] = self.accelerator.unwrap_model(model).state_dict()
            path = Path(self.args.output_dir or ".") / f"nonfinite_step{step}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dump, path)
            print(f"  saved debug dump to {path}")
        model.zero_grad()
        if self._consecutive >= self.max_consecutive:
            raise NonFiniteGradientError(f"{self._consecutive} consecutive non-finite gradient steps; stopping")
        return norm
