"""Evaluation metrics (Sec. III-A.2): MSE, RMSE and R^2 on forecast visits.

All metrics are computed on z-scored shape features (statistics from the
training fold), over the held-out target visits only.

R^2 is computed per feature across target visits and averaged
(``sklearn.metrics.r2_score(..., multioutput="uniform_average")``).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import r2_score


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """``y_true``/``y_pred``: (n_target_visits, D)."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average"))}
