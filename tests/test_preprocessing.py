import json

import numpy as np
import pandas as pd
import pytest

from cnode.data.freesurfer import (extract_region_thickness, extract_subcortical_volumes,
                                   region_feature_names)
from cnode.data.ppmi import PPMIConfig, build_cohort, deduplicate_sessions

from .conftest import write_fake_fs_subject


def test_region_and_volume_parsers(tmp_path):
    write_fake_fs_subject(tmp_path, "s1", np.random.default_rng(0))
    x = extract_region_thickness(tmp_path / "s1")
    assert x.shape == (68,) and len(region_feature_names()) == 68
    v = extract_subcortical_volumes(tmp_path / "s1")
    assert v.shape == (14,) and np.all(v > 0)
    # Volumes are normalised by eTIV (1.5e6 mm^3) and scaled by 1e3.
    raw = extract_subcortical_volumes(tmp_path / "s1", normalize_by_etiv=False)
    np.testing.assert_allclose(v, raw / 1.5e6 * 1e3, rtol=1e-5)


def test_session_deduplication():
    scans = pd.DataFrame({
        "subject": ["a"] * 4,
        "image_id": ["I1", "I2", "I3", "I4"],
        "acq_date": pd.to_datetime(["2015-01-01", "2015-01-20", "2016-02-01", "2017-03-01"]),
    })
    out = deduplicate_sessions(scans, min_gap_days=90)
    assert out["image_id"].tolist() == ["I1", "I3", "I4"]


def test_build_cohort_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    fs = tmp_path / "fs"
    rows = []
    plan = {  # subject -> (cohort, list of (image, date))
        "1001": (1, [("I1", "1/5/2014"), ("I2", "1/20/2014"), ("I3", "2/1/2015"), ("I4", "3/1/2016")]),
        "1002": (1, [("I5", "6/1/2013"), ("I6", "7/15/2014")]),
        "1003": (1, [("I7", "6/1/2013")]),                      # single visit -> dropped
        "2001": (2, [("I8", "6/1/2013"), ("I9", "6/1/2014")]),  # healthy control -> dropped
        "1004": (1, [("I10", "6/1/2013"), ("I11", "6/1/2014")]),  # failed recon -> dropped
    }
    for subj, (_, scans) in plan.items():
        for img, date in scans:
            rows.append({"Subject": subj, "Image Data ID": img, "Acq Date": date,
                         "Visit": "V", "Group": "PD", "Age": 60, "Sex": "M"})
            if not (subj == "1004" and img == "I11"):
                bad = subj == "1004"
                write_fake_fs_subject(fs, f"{subj}_{img}", rng, thickness_scale=10 if bad else 1)
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    pd.DataFrame({"PATNO": list(plan), "COHORT": [c for c, _ in plan.values()]}).to_csv(
        tmp_path / "status.csv", index=False)
    pd.DataFrame({"PATNO": list(plan), "BIRTHDT": ["05/1950"] * 5, "SEX": [1, 0, 1, 1, 0]}).to_csv(
        tmp_path / "demo.csv", index=False)

    cfg = PPMIConfig(manifest_csv=str(tmp_path / "manifest.csv"), freesurfer_dir=str(fs),
                     participant_status_csv=str(tmp_path / "status.csv"),
                     demographics_csv=str(tmp_path / "demo.csv"))
    summary = build_cohort(cfg, tmp_path / "out")
    visits = pd.read_csv(tmp_path / "out" / "visits.csv", dtype={"subject": str})
    subjects = pd.read_csv(tmp_path / "out" / "subjects.csv", dtype={"subject": str})

    assert summary["n_subjects"] == 2
    assert summary["visits_per_subject"] == {2: 1, 3: 1}
    assert sorted(subjects["subject"]) == ["1001", "1002"]
    # The rescan 15 days after baseline (I2) is merged into the baseline session.
    assert "I2" not in set(visits["image_id"])
    # Time is measured in years from the first scan.
    t = visits[visits.subject == "1001"]["time"].to_numpy()
    assert t[0] == 0 and t[1] == pytest.approx(392 / 365.25, abs=1e-6)
    # Age at baseline from the de-identified MM/YYYY birth date (+14 days).
    age = subjects.set_index("subject").loc["1001", "age"]
    assert age == pytest.approx(63.6, abs=0.1)
    assert json.loads((tmp_path / "out" / "cohort_summary.json").read_text())["n_shape_features"] == 68
