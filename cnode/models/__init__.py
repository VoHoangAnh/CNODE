"""Model factory."""

from __future__ import annotations

from .baselines import Persistence, SequenceForecaster
from .cnode import CNODE, CNODEConfig

MODEL_NAMES = ["cnode", "neural_ode", "gru", "rnn", "lstm", "transformer", "persistence"]


def build_model(name: str, input_dim: int, cond_dim: int, cfg: dict):
    """Instantiate a model by name. ``cfg`` is the ``model`` section of the YAML config."""
    name = name.lower()
    if name in ("cnode", "neural_ode"):
        kw = {k: v for k, v in cfg.get("cnode", {}).items()}
        if name == "neural_ode":
            kw.update(use_alignment=False, lambda_c=0.0)
        return CNODE(CNODEConfig(input_dim=input_dim, cond_dim=cond_dim, **kw))
    if name in ("gru", "rnn", "lstm", "transformer"):
        return SequenceForecaster(input_dim, kind=name, **cfg.get("sequence", {}))
    if name == "persistence":
        return Persistence()
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


__all__ = ["CNODE", "CNODEConfig", "SequenceForecaster", "Persistence", "build_model", "MODEL_NAMES"]
