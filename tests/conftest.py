"""Shared fixtures: fake FreeSurfer subjects and a tiny synthetic cohort."""

from pathlib import Path

import numpy as np
import pytest

from cnode.data.freesurfer import DEFAULT_SUBCORTICAL, DK_REGIONS
from cnode.data.synthetic import make_synthetic_cohort


def write_fake_fs_subject(root: Path, name: str, rng: np.random.Generator, thickness_scale=1.0):
    stats = root / name / "stats"
    stats.mkdir(parents=True)
    for hemi in ("lh", "rh"):
        lines = ["# Table of FreeSurfer cortical parcellation anatomical statistics",
                 "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd"]
        for r in DK_REGIONS:
            t = rng.uniform(2.0, 3.0) * thickness_scale
            lines.append(f"{r} 1000 700 2000 {t:.3f} 0.5 0.1 0.02 10 1.0")
        (stats / f"{hemi}.aparc.stats").write_text("\n".join(lines) + "\n")
    lines = ["# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1500000.0, mm^3",
             "# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange"]
    names = [("Left-Thalamus-Proper" if s == "Left-Thalamus" else s) for s in DEFAULT_SUBCORTICAL]
    for i, s in enumerate(names):
        lines.append(f"{i+1} {i+10} 5000 {rng.uniform(500, 8000):.1f} {s} 80 10 40 110 70")
    (stats / "aseg.stats").write_text("\n".join(lines) + "\n")


@pytest.fixture
def synthetic_dir(tmp_path):
    make_synthetic_cohort(tmp_path / "synth", n_three=12, n_two=18, seed=1)
    return tmp_path / "synth"


@pytest.fixture
def tiny_cfg():
    return {
        "model": {"cnode": {"latent_dim": 4, "hidden_dim": 16, "enc_hidden_dim": 16,
                            "align_hidden_dim": 8, "ode_method": "rk4"},
                  "sequence": {"hidden_dim": 16}},
        "train": {"epochs": 3, "batch_size": 8, "lr": 1e-3, "patience": 5, "seed": 0,
                  "device": "cpu", "grad_clip": 5.0},
        "eval": {"n_folds": 2, "val_fraction": 0.2, "n_forecast": 1, "seed": 0,
                 "save_checkpoints": False},
    }
