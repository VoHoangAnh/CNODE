"""Dataset, fold-wise normalisation and batching for longitudinal forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

META_COLUMNS = ["age", "sex"]
_VISIT_META = {"subject", "image_id", "visit", "acq_date", "visit_idx", "time"}


@dataclass
class Subject:
    """One subject's longitudinal record."""

    sid: str
    times: np.ndarray   # (K,) years since first scan, strictly increasing
    X: np.ndarray       # (K, D) shape features
    C: np.ndarray       # (Dc,) subcortical volumes at first visit
    M: np.ndarray       # (Dm,) metadata (age, sex) at first visit


def load_subjects(data_dir: str | Path) -> tuple[List[Subject], List[str], List[str]]:
    """Load ``visits.csv``/``subjects.csv`` produced by ``cnode.data.ppmi``."""
    data_dir = Path(data_dir)
    visits = pd.read_csv(data_dir / "visits.csv", dtype={"subject": str})
    subjects = pd.read_csv(data_dir / "subjects.csv", dtype={"subject": str}).set_index("subject")

    feat_cols = [c for c in visits.columns if c not in _VISIT_META]
    vol_cols = [c for c in subjects.columns if c.startswith("vol_")]

    out: List[Subject] = []
    for sid, g in visits.sort_values(["subject", "time"]).groupby("subject", sort=True):
        if sid not in subjects.index:
            continue
        row = subjects.loc[sid]
        out.append(Subject(
            sid=sid,
            times=g["time"].to_numpy(np.float32),
            X=g[feat_cols].to_numpy(np.float32),
            C=row[vol_cols].to_numpy(np.float32),
            M=row[META_COLUMNS].to_numpy(np.float32),
        ))
    return out, feat_cols, vol_cols


class Normalizer:
    """Z-score statistics fitted on the training fold only (no test leakage)."""

    def fit(self, subjects: Sequence[Subject]) -> "Normalizer":
        X = np.concatenate([s.X for s in subjects], 0)
        C = np.stack([s.C for s in subjects])
        M = np.stack([s.M for s in subjects])
        eps = 1e-6
        self.x_mu, self.x_sd = X.mean(0), X.std(0) + eps
        self.c_mu, self.c_sd = C.mean(0), C.std(0) + eps
        # Only continuous metadata (age) is standardised; sex stays binary.
        self.m_mu = np.array([M[:, 0].mean(), 0.0], np.float32)
        self.m_sd = np.array([M[:, 0].std() + eps, 1.0], np.float32)
        return self

    def transform(self, subjects: Sequence[Subject]) -> List[Subject]:
        return [Subject(s.sid, s.times.copy(),
                        ((s.X - self.x_mu) / self.x_sd).astype(np.float32),
                        ((s.C - self.c_mu) / self.c_sd).astype(np.float32),
                        ((s.M - self.m_mu) / self.m_sd).astype(np.float32))
                for s in subjects]


class LongitudinalDataset(Dataset):
    """Each item is one subject. The last ``n_forecast`` visits are forecast targets."""

    def __init__(self, subjects: Sequence[Subject], n_forecast: int = 1):
        if any(len(s.times) <= n_forecast for s in subjects):
            raise ValueError("Every subject needs at least n_forecast + 1 visits.")
        self.subjects = list(subjects)
        self.n_forecast = n_forecast

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict:
        s = self.subjects[idx]
        return {"subject": s, "n_context": len(s.times) - self.n_forecast}


def collate(items: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad a list of subjects to a dense batch.

    Returned masks:
        valid   -- visit exists
        context -- visit is observed by the encoder (first n_context visits)
        target  -- visit is a forecast target (held out from the encoder)
    """
    B = len(items)
    K = max(len(it["subject"].times) for it in items)
    D = items[0]["subject"].X.shape[1]
    times = torch.zeros(B, K)
    X = torch.zeros(B, K, D)
    valid = torch.zeros(B, K, dtype=torch.bool)
    context = torch.zeros(B, K, dtype=torch.bool)
    for b, it in enumerate(items):
        s, n = it["subject"], len(it["subject"].times)
        times[b, :n] = torch.from_numpy(s.times)
        X[b, :n] = torch.from_numpy(s.X)
        valid[b, :n] = True
        context[b, : it["n_context"]] = True
    return {
        "times": times,
        "X": X,
        "valid": valid,
        "context": context,
        "target": valid & ~context,
        "C": torch.from_numpy(np.stack([it["subject"].C for it in items])),
        "M": torch.from_numpy(np.stack([it["subject"].M for it in items])),
        "n_context": torch.tensor([it["n_context"] for it in items]),
    }
