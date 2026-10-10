"""Synthetic PPMI-like cohort for smoke tests and for running the code without PPMI access.

The generator mimics the cohort statistics reported in Sec. III-A of the paper
(161 PD subjects; 50 with three visits, 111 with two; inter-visit intervals in
[0.61, 2.27] years) and produces files in exactly the same format as
``cnode.data.ppmi.build_cohort``. Ground truth follows the CNODE assumptions:
each subject has a latent disease stage ``s(t) = tau_i + gamma_i * t`` whose
onset ``tau_i`` and speed ``gamma_i`` depend on age and subcortical volumes, and
regional thickness decays non-linearly with the stage.

Results on synthetic data say nothing about performance on real PPMI data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .freesurfer import DEFAULT_SUBCORTICAL, region_feature_names
from .ppmi import cohort_summary


def make_synthetic_cohort(out_dir: str | Path, n_three: int = 50, n_two: int = 111,
                          n_vol: int = len(DEFAULT_SUBCORTICAL), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    feat_names = region_feature_names()
    D = len(feat_names)
    n = n_three + n_two

    # Subject-level conditions.
    age = rng.normal(62.0, 9.0, n).clip(35, 85)
    sex = rng.integers(0, 2, n).astype(float)
    vol_base = rng.uniform(0.4, 6.0, n_vol)               # per-mille of eTIV
    vol = vol_base * (1 + 0.08 * rng.standard_normal((n, n_vol)))

    # Hidden alignment parameters driven by the conditions (what CNODE should learn).
    vol_z = (vol - vol.mean(0)) / vol.std(0)
    tau = 0.04 * (age - 62) - 0.4 * vol_z[:, :8].mean(1) + 0.3 * rng.standard_normal(n)
    gamma = np.exp(0.25 * (age - 62) / 9 - 0.2 * vol_z[:, 4:6].mean(1) + 0.15 * rng.standard_normal(n))

    # Region-specific atrophy profiles.
    base = rng.uniform(2.0, 3.2, D)
    amp = rng.uniform(0.05, 0.35, D)
    center = rng.uniform(-0.5, 1.5, D)
    width = rng.uniform(0.4, 1.2, D)

    rows_v, rows_s = [], []
    for i in range(n):
        k = 3 if i < n_three else 2
        gaps = rng.uniform(0.61, 2.27, k - 1)
        times = np.concatenate([[0.0], np.cumsum(gaps)])
        subj_offset = 0.05 * rng.standard_normal(D)
        sid = f"S{i:04d}"
        for j, t in enumerate(times):
            stage = tau[i] + gamma[i] * t
            x = base + subj_offset - amp / (1 + np.exp(-(stage - center) / width))
            x = x + 0.02 * rng.standard_normal(D)
            rows_v.append({"subject": sid, "image_id": f"I{i:04d}{j}", "visit": f"V{j:02d}",
                           "acq_date": "", "visit_idx": j, "time": float(t),
                           **dict(zip(feat_names, x))})
        rows_s.append({"subject": sid, "age": age[i], "sex": sex[i],
                       **{f"vol_{s}": vol[i, m] for m, s in enumerate(DEFAULT_SUBCORTICAL[:n_vol])}})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visits = pd.DataFrame(rows_v)
    visits.to_csv(out_dir / "visits.csv", index=False)
    pd.DataFrame(rows_s).to_csv(out_dir / "subjects.csv", index=False)
    summary = cohort_summary(visits)
    summary["synthetic"] = True
    with open(out_dir / "cohort_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
