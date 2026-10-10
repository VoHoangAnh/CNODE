"""Baselines of Table I.

* RNN / LSTM / GRU / Transformer: discrete sequence models. Following Sec. III-B,
  they assume a regular sampling interval of one year, i.e. they see the visit
  *order* but not the true visit times. Trained with teacher-forced next-visit
  prediction; at test time the targets are forecast autoregressively.
* Neural ODE: latent ODE without alignment/contrastive learning -- built with
  ``CNODE(use_alignment=False, lambda_c=0)`` in ``cnode.models.build_model``.
* Persistence (last observation carried forward): a sanity floor, not in the paper.

LLMTime (zero-shot LLM forecasting) is not re-implemented here; see the README.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from .losses import masked_mse


class SequenceForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, kind: str = "gru",
                 n_layers: int = 1, n_heads: int = 4, dropout: float = 0.1, max_len: int = 32):
        super().__init__()
        self.kind = kind
        if kind in ("rnn", "gru", "lstm"):
            cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
            self.backbone = cls(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
            self.inp = nn.Identity()
        elif kind == "transformer":
            self.inp = nn.Linear(input_dim, hidden_dim)
            layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, 2 * hidden_dim, dropout,
                                               batch_first=True)
            self.backbone = nn.TransformerEncoder(layer, max(n_layers, 2))
            pe = torch.zeros(max_len, hidden_dim)
            pos = torch.arange(max_len).float()[:, None]
            div = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(1e4) / hidden_dim))
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            self.register_buffer("pe", pe)   # position = visit index (1-year grid)
        else:
            raise ValueError(f"Unknown sequence model '{kind}'")
        self.out = nn.Linear(hidden_dim, input_dim)

    def _step_all(self, x: torch.Tensor) -> torch.Tensor:
        """Return next-visit predictions for every prefix: out[:, k] predicts x[:, k+1]."""
        if self.kind == "transformer":
            K = x.shape[1]
            h = self.inp(x) + self.pe[:K]
            causal = torch.triu(torch.ones(K, K, dtype=torch.bool, device=x.device), 1)
            h = self.backbone(h, mask=causal)
        else:
            h, _ = self.backbone(x)
        return x + self.out(h)    # residual: predict the change from the current visit

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, valid = batch["X"], batch["valid"].float()
        nxt = self._step_all(x)
        pred = torch.cat([x[:, :1], nxt[:, :-1]], 1)   # align: pred[:, k] estimates x[:, k]
        mask = valid.clone()
        mask[:, 0] = 0.0                                # the first visit is never predicted
        loss = masked_mse(x, pred, mask)
        return {"loss": loss, "mse": loss.detach(), "pred": pred}

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Autoregressive forecast: observed visits are kept, targets are rolled out."""
        x = batch["X"].clone()
        ctx, valid = batch["context"], batch["valid"]
        K = x.shape[1]
        for k in range(1, K):
            need = valid[:, k] & ~ctx[:, k]
            if need.any():
                nxt = self._step_all(x[:, :k])[:, -1]
                x[need, k] = nxt[need]
        return x


class Persistence(nn.Module):
    """Forecast = last observed visit (no parameters)."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))   # lets the shared trainer run unchanged

    def forward(self, batch):
        loss = self.dummy * 0.0
        return {"loss": loss, "mse": loss.detach(), "pred": self.predict(batch)}

    @torch.no_grad()
    def predict(self, batch):
        x = batch["X"].clone()
        last = batch["n_context"] - 1
        b = torch.arange(x.shape[0])
        x[~batch["context"]] = x[b, last].unsqueeze(1).expand_as(x)[~batch["context"]]
        return x
