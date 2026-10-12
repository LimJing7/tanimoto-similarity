"""Llama backbone with a scalar regression head reading the final ``<eos>`` token."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import LlamaConfig, LlamaModel

from tani_predictor.config import ModelConfig


def build_llama_config(cfg: ModelConfig) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        max_position_embeddings=cfg.max_position_embeddings,
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
        tie_word_embeddings=False,
        use_cache=False,
    )


class TaniRegressor(nn.Module):
    """Predicts a value in [0, 1] (scaled Tanimoto) from right-padded token ids."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # PyTorch's cuDNN SDPA backward returns NaN dq (torch 2.14, H200, bf16, explicit padding mask) for query
        # rows whose softmax log-sum-exp is below -ln(FLT_MAX) = -88.72 and whose incoming gradient is exactly 0
        # (here: every row except <eos>). The loss stays finite, the grad-norm is NaN, then the weights are NaN.
        # Flash/mem-efficient/math give correct gradients, so the cuDNN backend is switched off globally.
        # TODO: this global switch is a stopgap (about 35% slower on the longest batches); see README "TODO".
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.cfg = cfg
        self.backbone = LlamaModel(build_llama_config(cfg))
        self.head = nn.Linear(cfg.hidden_size, 1)
        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor | None = None
    ) -> dict[str, Tensor]:
        hidden: Tensor = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1  # right padding: the final real token is <eos>
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        pred = torch.sigmoid(self.head(pooled).squeeze(-1).float())
        out: dict[str, Tensor] = {"logits": pred}
        if labels is not None:
            out["loss"] = F.mse_loss(pred, labels.float())
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
