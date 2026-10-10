"""Training, early stopping and 5-fold subject-level cross-validation (Table I protocol)."""

from __future__ import annotations

import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from .data.dataset import LongitudinalDataset, Normalizer, Subject, collate, load_subjects
from .metrics import forecast_metrics
from .models import build_model

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    """Forecast metrics on the target visits of every subject in ``loader``."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = _to(batch, device)
        pred = model.predict(batch)
        m = batch["target"]
        ys.append(batch["X"][m].cpu().numpy())
        ps.append(pred[m].cpu().numpy())
    return forecast_metrics(np.concatenate(ys), np.concatenate(ps))


def fit(model, train_set, val_set, tcfg: Dict, device) -> Dict:
    """Train with Adam; keep the weights with the best validation forecast MSE."""
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    g = torch.Generator().manual_seed(tcfg.get("seed", 0))
    train_loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True,
                              collate_fn=collate, generator=g)
    val_loader = DataLoader(val_set, batch_size=256, collate_fn=collate)

    best = {"val_mse": float("inf"), "epoch": -1, "state": copy.deepcopy(model.state_dict())}
    patience, bad = tcfg.get("patience", 30), 0
    history = []
    for epoch in range(tcfg["epochs"]):
        model.train()
        logs = []
        for batch in train_loader:
            batch = _to(batch, device)
            out = model(batch)
            opt.zero_grad()
            out["loss"].backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
            opt.step()
            logs.append({k: float(v.detach()) for k, v in out.items() if k != "pred"})
        val = evaluate(model, val_loader, device)
        rec = {"epoch": epoch, **pd.DataFrame(logs).mean().to_dict(), "val_mse": val["mse"]}
        history.append(rec)
        if val["mse"] < best["val_mse"] - 1e-6:
            best = {"val_mse": val["mse"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if epoch % tcfg.get("log_every", 25) == 0:
            logger.info("epoch %3d | loss %.4f | val mse %.4f", epoch, rec["loss"], val["mse"])
    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "best_val_mse": best["val_mse"], "history": history}


def _visit_counts(subjects: Sequence[Subject]) -> np.ndarray:
    return np.array([len(s.times) for s in subjects])


def _safe_stratify(labels: np.ndarray, n_splits: int):
    """Stratify by visit count when every class is large enough, else fall back to None."""
    _, counts = np.unique(labels, return_counts=True)
    return labels if counts.min() >= n_splits else None


def run_cv(data_dir: str, out_dir: str, model_names: List[str], cfg: Dict) -> pd.DataFrame:
    """Subject-level K-fold CV; the same folds are used for every model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg, ecfg = cfg["train"], cfg["eval"]
    device = torch.device(tcfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    subjects, feat_names, vol_names = load_subjects(data_dir)
    logger.info("Loaded %d subjects, %d shape features, %d volume features",
                len(subjects), len(feat_names), len(vol_names))
    counts = _visit_counts(subjects)
    n_splits = ecfg.get("n_folds", 5)
    strat = _safe_stratify(counts, n_splits)
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=ecfg.get("seed", 0))
    folds = list(splitter.split(np.zeros(len(subjects)), strat if strat is not None else np.zeros(len(subjects))))

    rows = []
    for fold, (tr_idx, te_idx) in enumerate(folds):
        tr_counts = counts[tr_idx]
        inner_tr, inner_val = train_test_split(
            tr_idx, test_size=ecfg.get("val_fraction", 0.1), random_state=ecfg.get("seed", 0) + fold,
            stratify=_safe_stratify(tr_counts, 2))
        norm = Normalizer().fit([subjects[i] for i in inner_tr])
        nf = ecfg.get("n_forecast", 1)
        ds = {k: LongitudinalDataset(norm.transform([subjects[i] for i in idx]), nf)
              for k, idx in (("train", inner_tr), ("val", inner_val), ("test", te_idx))}
        cond_dim = len(vol_names) + 2

        for name in model_names:
            set_seed(tcfg.get("seed", 0) + fold)
            model = build_model(name, len(feat_names), cond_dim, cfg["model"])
            t0 = time.time()
            info = fit(model, ds["train"], ds["val"], {**tcfg, "seed": tcfg.get("seed", 0) + fold}, device)
            test = evaluate(model, DataLoader(ds["test"], batch_size=256, collate_fn=collate), device)
            rows.append({"model": name, "fold": fold, **test, "best_epoch": info["best_epoch"],
                         "train_seconds": time.time() - t0})
            logger.info("fold %d | %-11s | MSE %.4f RMSE %.4f R2 %.4f", fold, name,
                        test["mse"], test["rmse"], test["r2"])
            if ecfg.get("save_checkpoints", True):
                ck = out_dir / "checkpoints"
                ck.mkdir(exist_ok=True)
                torch.save({"model": name, "state_dict": model.state_dict(), "config": cfg,
                            "normalizer": norm.__dict__, "feature_names": feat_names,
                            "volume_names": vol_names}, ck / f"{name}_fold{fold}.pt")
            pd.DataFrame(info["history"]).to_csv(out_dir / f"history_{name}_fold{fold}.csv", index=False)

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(out_dir / "results_per_fold.csv", index=False)
    summary = summarize(per_fold)
    summary.to_csv(out_dir / "results_summary.csv", index=False)
    (out_dir / "results_table.md").write_text(to_markdown(summary))
    with open(out_dir / "config_used.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    return summary


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    agg = per_fold.groupby("model", sort=False)[["mse", "rmse", "r2"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def to_markdown(summary: pd.DataFrame) -> str:
    """Render a Table-I-style markdown table (mean +- std over folds)."""
    lines = ["| Model | MSE ↓ | RMSE ↓ | R² ↑ |", "|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['model']} | {r['mse_mean']:.4f} ± {r['mse_std']:.4f} | "
                     f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f} | {r['r2_mean']:.4f} ± {r['r2_std']:.4f} |")
    return "\n".join(lines) + "\n"
