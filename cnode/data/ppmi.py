"""Build the longitudinal CNODE cohort from PPMI downloads + FreeSurfer outputs.

Pipeline (see README, "PPMI preprocessing"):

1. Read the LONI IDA image-collection manifest (one row per T1 scan).
2. Keep participants whose enrolment cohort is Parkinson's disease.
3. Deduplicate scans: one scan per acquisition session; scans closer than
   ``min_gap_days`` are treated as the same session (repeat/rescan).
4. Locate each scan's FreeSurfer directory, skip failed reconstructions.
5. Extract shape features ``X_{i,k}`` (68 DK regional thicknesses, or
   vertex-wise thickness) for every scan and condition features from the
   *first* scan only: subcortical volumes ``C_i`` and metadata ``M_i`` = (age, sex).
6. Quality-control thickness values, keep subjects with >= ``min_visits`` scans.
7. Express visit times in years since the subject's first scan (``t_{i,k} - t_{i,1}``).
8. Write two tidy CSVs: ``visits.csv`` (long format) and ``subjects.csv``.

Column names in PPMI exports change between data releases, so every column
used here is configurable through ``PPMIConfig`` (and the YAML config).
Normalisation (z-scoring) is intentionally *not* done here: it must be fitted
inside each cross-validation training fold to avoid information leakage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .freesurfer import (
    DEFAULT_SUBCORTICAL,
    extract_region_thickness,
    extract_subcortical_volumes,
    extract_vertex_thickness,
    region_feature_names,
)

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.25


@dataclass
class PPMIConfig:
    """All paths and column names needed to build the cohort."""

    # --- inputs -----------------------------------------------------------
    manifest_csv: str                       # LONI image-collection CSV
    freesurfer_dir: str                     # $SUBJECTS_DIR containing recon-all outputs
    participant_status_csv: Optional[str] = None   # PPMI "Participant_Status.csv"
    demographics_csv: Optional[str] = None         # PPMI "Demographics.csv"

    # --- manifest columns (LONI collection export) -------------------------
    col_subject: str = "Subject"
    col_image_id: str = "Image Data ID"
    col_acq_date: str = "Acq Date"
    col_visit: str = "Visit"
    col_group: str = "Group"        # e.g. "PD", "Control", "Prodromal"
    col_age: str = "Age"            # age at scan (used if no demographics file)
    col_sex: str = "Sex"            # "M"/"F"

    # --- study-data columns -----------------------------------------------
    col_patno: str = "PATNO"
    col_cohort: str = "COHORT"
    pd_cohort_values: List = field(default_factory=lambda: [1, "1", "Parkinson's Disease", "PD"])
    col_birthdate: str = "BIRTHDT"   # "MM/YYYY" in PPMI
    col_demo_sex: str = "SEX"        # 0 = female, 1 = male in PPMI coding
    manifest_pd_values: List[str] = field(default_factory=lambda: ["PD"])

    # --- FreeSurfer directory naming --------------------------------------
    # Python format string; available keys: subject, image_id.
    # Cross-sectional stream: "{subject}_{image_id}"
    # Longitudinal stream:   "{subject}_{image_id}.long.{subject}_base"
    fs_dir_pattern: str = "{subject}_{image_id}"

    # --- features ----------------------------------------------------------
    feature_type: str = "region"            # "region" (68 DK) or "vertex"
    vertex_target: str = "fsaverage5"
    subcortical_structures: List[str] = field(default_factory=lambda: list(DEFAULT_SUBCORTICAL))
    normalize_volumes_by_etiv: bool = True

    # --- cohort rules -----------------------------------------------------
    min_visits: int = 2
    max_visits: Optional[int] = None        # keep the first N scans if set
    min_gap_days: int = 90                  # scans closer than this are one session
    thickness_range: tuple = (0.5, 5.0)     # plausible regional thickness (mm)


# ---------------------------------------------------------------------------
# Manifest / clinical tables
# ---------------------------------------------------------------------------
def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse LONI dates (typically ``M/D/YYYY``) robustly."""
    parsed = pd.to_datetime(series, format="%m/%d/%Y", errors="coerce")
    fallback = pd.to_datetime(series[parsed.isna()], errors="coerce")
    parsed.loc[parsed.isna()] = fallback
    return parsed


def load_manifest(cfg: PPMIConfig) -> pd.DataFrame:
    """Load the scan manifest and standardise its columns."""
    df = pd.read_csv(cfg.manifest_csv)
    required = [cfg.col_subject, cfg.col_image_id, cfg.col_acq_date]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Manifest is missing columns {missing}; available: {list(df.columns)}")

    out = pd.DataFrame({
        "subject": df[cfg.col_subject].astype(str).str.strip(),
        "image_id": df[cfg.col_image_id].astype(str).str.strip(),
        "acq_date": _parse_dates(df[cfg.col_acq_date].astype(str)),
    })
    for src, dst in [(cfg.col_visit, "visit"), (cfg.col_group, "group"),
                     (cfg.col_age, "age_at_scan"), (cfg.col_sex, "sex_str")]:
        out[dst] = df[src] if src in df.columns else np.nan

    n_bad = out["acq_date"].isna().sum()
    if n_bad:
        logger.warning("Dropping %d scans with unparseable acquisition dates", n_bad)
    return out.dropna(subset=["acq_date"]).reset_index(drop=True)


def filter_pd_cohort(scans: pd.DataFrame, cfg: PPMIConfig) -> pd.DataFrame:
    """Keep only participants enrolled in the PD cohort."""
    if cfg.participant_status_csv:
        status = pd.read_csv(cfg.participant_status_csv)
        status[cfg.col_patno] = status[cfg.col_patno].astype(str).str.strip()
        allowed = {str(v) for v in cfg.pd_cohort_values}
        pd_ids = set(status.loc[status[cfg.col_cohort].astype(str).isin(allowed), cfg.col_patno])
        if not pd_ids:
            raise ValueError("No PD participants found; check `col_cohort` / `pd_cohort_values`.")
        keep = scans["subject"].isin(pd_ids)
    else:
        logger.info("No participant-status file; filtering on manifest column '%s'", cfg.col_group)
        keep = scans["group"].astype(str).isin(cfg.manifest_pd_values)
    logger.info("PD filter: %d -> %d scans", len(scans), int(keep.sum()))
    return scans.loc[keep].reset_index(drop=True)


def deduplicate_sessions(scans: pd.DataFrame, min_gap_days: int) -> pd.DataFrame:
    """Keep one scan per session.

    PPMI often contains several T1 series per session (repeats, different
    sequences). Scans within ``min_gap_days`` of the previously kept scan are
    considered the same session; the earliest (then lowest image id) is kept.
    """
    scans = scans.sort_values(["subject", "acq_date", "image_id"]).reset_index(drop=True)
    keep_rows = []
    for _, g in scans.groupby("subject", sort=False):
        last = None
        for idx, row in g.iterrows():
            if last is None or (row["acq_date"] - last).days >= min_gap_days:
                keep_rows.append(idx)
                last = row["acq_date"]
    out = scans.loc[keep_rows].reset_index(drop=True)
    logger.info("Session de-duplication: %d -> %d scans", len(scans), len(out))
    return out


def attach_demographics(scans: pd.DataFrame, cfg: PPMIConfig) -> pd.DataFrame:
    """Compute age at scan and sex (0 = female, 1 = male)."""
    scans = scans.copy()
    if cfg.demographics_csv:
        demo = pd.read_csv(cfg.demographics_csv)
        demo[cfg.col_patno] = demo[cfg.col_patno].astype(str).str.strip()
        demo = demo.drop_duplicates(cfg.col_patno, keep="last").set_index(cfg.col_patno)
        # PPMI birth dates are de-identified to month/year ("MM/YYYY"); use mid-month.
        birth = pd.to_datetime(demo[cfg.col_birthdate].astype(str), format="%m/%Y", errors="coerce")
        birth = birth + pd.Timedelta(days=14)
        scans["birth"] = scans["subject"].map(birth)
        scans["age"] = (scans["acq_date"] - scans["birth"]).dt.days / DAYS_PER_YEAR
        scans["sex"] = scans["subject"].map(demo[cfg.col_demo_sex]).astype(float)
        scans = scans.drop(columns="birth")
    else:
        scans["age"] = pd.to_numeric(scans["age_at_scan"], errors="coerce")
        scans["sex"] = scans["sex_str"].astype(str).str.upper().str[0].map({"M": 1.0, "F": 0.0})
    n_missing = scans[["age", "sex"]].isna().any(axis=1).sum()
    if n_missing:
        logger.warning("%d scans lack age/sex and will be dropped", n_missing)
    return scans.dropna(subset=["age", "sex"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _fs_subject_dir(row: pd.Series, cfg: PPMIConfig) -> Path:
    name = cfg.fs_dir_pattern.format(subject=row["subject"], image_id=row["image_id"])
    return Path(cfg.freesurfer_dir) / name


def extract_features(scans: pd.DataFrame, cfg: PPMIConfig):
    """Extract shape and volume features; drop scans whose recon failed or fail QC."""
    shape, vols, ok = [], [], []
    lo, hi = cfg.thickness_range
    for _, row in scans.iterrows():
        sdir = _fs_subject_dir(row, cfg)
        try:
            if cfg.feature_type == "region":
                x = extract_region_thickness(sdir)
            elif cfg.feature_type == "vertex":
                x = extract_vertex_thickness(sdir, cfg.vertex_target)
            else:
                raise ValueError(f"Unknown feature_type '{cfg.feature_type}'")
            c = extract_subcortical_volumes(sdir, cfg.subcortical_structures,
                                            cfg.normalize_volumes_by_etiv)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.warning("Skipping %s: %s", sdir.name, exc)
            ok.append(False)
            continue

        # Regional QC: implausible thickness indicates a failed surface reconstruction.
        # Vertex-wise maps legitimately contain zeros (medial wall), so only NaNs are checked.
        bad = (not np.all(np.isfinite(x))) or (
            cfg.feature_type == "region" and (x.min() < lo or x.max() > hi)
        )
        if bad or not np.all(np.isfinite(c)):
            logger.warning("QC failure for %s; scan dropped", sdir.name)
            ok.append(False)
            continue
        shape.append(x)
        vols.append(c)
        ok.append(True)

    kept = scans.loc[ok].reset_index(drop=True)
    return kept, np.stack(shape) if shape else np.empty((0, 0)), np.stack(vols) if vols else np.empty((0, 0))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_cohort(cfg: PPMIConfig, out_dir: str | Path) -> Dict[str, float]:
    """Run the full preprocessing pipeline and write ``visits.csv``/``subjects.csv``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scans = load_manifest(cfg)
    scans = filter_pd_cohort(scans, cfg)
    scans = deduplicate_sessions(scans, cfg.min_gap_days)
    scans = attach_demographics(scans, cfg)
    scans, X, C = extract_features(scans, cfg)
    if len(scans) == 0:
        raise RuntimeError("No usable scans after feature extraction; check paths/patterns.")

    if cfg.feature_type == "region":
        feat_names = region_feature_names()
    else:
        feat_names = [f"v{j:06d}" for j in range(X.shape[1])]
    vol_names = [f"vol_{s}" for s in cfg.subcortical_structures]

    scans = scans.sort_values(["subject", "acq_date"]).reset_index()
    X, C = X[scans["index"].to_numpy()], C[scans["index"].to_numpy()]
    scans = scans.drop(columns="index")

    # Visit index and time since first scan (years).
    scans["visit_idx"] = scans.groupby("subject").cumcount()
    first_date = scans.groupby("subject")["acq_date"].transform("min")
    scans["time"] = (scans["acq_date"] - first_date).dt.days / DAYS_PER_YEAR

    if cfg.max_visits is not None:
        mask = scans["visit_idx"] < cfg.max_visits
        scans, X, C = scans[mask].reset_index(drop=True), X[mask.to_numpy()], C[mask.to_numpy()]

    counts = scans.groupby("subject")["visit_idx"].transform("count")
    mask = (counts >= cfg.min_visits).to_numpy()
    scans, X, C = scans[mask].reset_index(drop=True), X[mask], C[mask]

    # Long-format visit table.
    visits = pd.concat([
        scans[["subject", "image_id", "visit", "acq_date", "visit_idx", "time"]].reset_index(drop=True),
        pd.DataFrame(X, columns=feat_names),
    ], axis=1)
    visits["acq_date"] = visits["acq_date"].dt.strftime("%Y-%m-%d")

    # Subject-level conditions from the FIRST visit (Sec. II-A).
    first = (scans["visit_idx"] == 0).to_numpy()
    subjects = pd.concat([
        scans.loc[first, ["subject", "age", "sex"]].reset_index(drop=True),
        pd.DataFrame(C[first], columns=vol_names),
    ], axis=1)

    visits.to_csv(out_dir / "visits.csv", index=False)
    subjects.to_csv(out_dir / "subjects.csv", index=False)

    summary = cohort_summary(visits)
    summary.update({"feature_type": cfg.feature_type, "n_shape_features": len(feat_names),
                    "n_volume_features": len(vol_names)})
    with open(out_dir / "cohort_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Cohort summary: %s", summary)
    return summary


def cohort_summary(visits: pd.DataFrame) -> Dict[str, float]:
    """Statistics comparable to Sec. III-A of the paper."""
    per_subj = visits.groupby("subject")
    n_visits = per_subj.size()
    intervals = per_subj["time"].apply(lambda t: np.diff(np.sort(t.to_numpy()))).explode().astype(float)
    return {
        "n_subjects": int(len(n_visits)),
        "n_scans": int(len(visits)),
        "visits_per_subject": {int(k): int(v) for k, v in n_visits.value_counts().sort_index().items()},
        "interval_mean_years": float(intervals.mean()) if len(intervals) else float("nan"),
        "interval_min_years": float(intervals.min()) if len(intervals) else float("nan"),
        "interval_max_years": float(intervals.max()) if len(intervals) else float("nan"),
    }
