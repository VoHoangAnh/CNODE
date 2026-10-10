"""Parsers for FreeSurfer outputs used by CNODE.

Two kinds of shape features are supported:

* ``region`` (default) -- the average cortical thickness of the 34 Desikan-Killiany
  regions per hemisphere (68 values), read from ``stats/{lh,rh}.aparc.stats``.
  This is the feature set described in Sec. II-A / III-A of the paper
  ("average thickness ... within each of the 34 predefined bilateral cortical
  [regions] per hemisphere"; "features were extracted from 68 ... regions").
* ``vertex`` -- vertex-wise thickness resampled to a common template
  (e.g. ``fsaverage5``) with ``mri_surf2surf``; read from
  ``surf/{lh,rh}.thickness.<target>.mgh``. This requires ``nibabel``.

Condition features (``C_i``) are subcortical structure volumes read from
``stats/aseg.stats`` and normalised by the estimated total intracranial volume
(eTIV) to remove head-size effects.

Only the Python standard library is required for the region-level parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

# The 34 Desikan-Killiany cortical labels in the order FreeSurfer writes them
# (the deprecated "corpuscallosum" label is excluded by FreeSurfer >= 5.3).
DK_REGIONS: List[str] = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
]

# Subcortical structures commonly implicated in PD (basal ganglia, thalamus,
# limbic structures). Names follow FreeSurfer's aseg.stats "StructName" column.
DEFAULT_SUBCORTICAL: List[str] = [
    "Left-Thalamus", "Right-Thalamus",
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    "Left-Pallidum", "Right-Pallidum",
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Left-Accumbens-area", "Right-Accumbens-area",
]

# FreeSurfer 7 renamed "Left-Thalamus-Proper" to "Left-Thalamus"; accept both.
_ALIASES: Dict[str, str] = {
    "Left-Thalamus-Proper": "Left-Thalamus",
    "Right-Thalamus-Proper": "Right-Thalamus",
}


def region_feature_names() -> List[str]:
    """Return the 68 feature names ``<hemi>_<region>_thickness``."""
    return [f"{h}_{r}_thickness" for h in ("lh", "rh") for r in DK_REGIONS]


def _read_table(path: Path) -> tuple[List[str], List[List[str]], List[str]]:
    """Read a FreeSurfer ``*.stats`` file.

    Returns the column headers, data rows and raw ``# Measure`` comment lines.
    """
    headers: List[str] = []
    rows: List[List[str]] = []
    measures: List[str] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("# ColHeaders"):
                headers = line.split()[2:]
            elif line.startswith("# Measure"):
                measures.append(line)
            elif not line.startswith("#"):
                rows.append(line.split())
    if not headers:
        raise ValueError(f"No '# ColHeaders' line found in {path}")
    return headers, rows, measures


def read_aparc_thickness(stats_file: Path | str) -> Dict[str, float]:
    """Read ``ThickAvg`` per Desikan-Killiany region from ``?h.aparc.stats``."""
    headers, rows, _ = _read_table(Path(stats_file))
    name_idx = headers.index("StructName")
    thick_idx = headers.index("ThickAvg")
    return {r[name_idx]: float(r[thick_idx]) for r in rows}


def read_aseg_volumes(stats_file: Path | str) -> tuple[Dict[str, float], Optional[float]]:
    """Read structure volumes (mm^3) and eTIV from ``aseg.stats``."""
    path = Path(stats_file)
    headers, rows, measures = _read_table(path)
    name_idx = headers.index("StructName")
    vol_idx = headers.index("Volume_mm3")
    volumes = {_ALIASES.get(r[name_idx], r[name_idx]): float(r[vol_idx]) for r in rows}

    etiv: Optional[float] = None
    for m in measures:
        # e.g. "# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated ..., 1543210.5, mm^3"
        if "EstimatedTotalIntraCranialVol" in m:
            parts = [p.strip() for p in m.split(",")]
            etiv = float(parts[-2])
    return volumes, etiv


def extract_region_thickness(subject_dir: Path | str) -> np.ndarray:
    """Return the 68-dim regional thickness vector for one FreeSurfer subject."""
    subject_dir = Path(subject_dir)
    values: List[float] = []
    for hemi in ("lh", "rh"):
        stats = read_aparc_thickness(subject_dir / "stats" / f"{hemi}.aparc.stats")
        missing = [r for r in DK_REGIONS if r not in stats]
        if missing:
            raise KeyError(f"{subject_dir}: missing DK regions in {hemi}: {missing}")
        values.extend(stats[r] for r in DK_REGIONS)
    return np.asarray(values, dtype=np.float32)


def extract_vertex_thickness(subject_dir: Path | str, target: str = "fsaverage5") -> np.ndarray:
    """Return vertex-wise thickness resampled to ``target`` (both hemispheres).

    The files are expected to be produced beforehand, e.g.::

        mri_surf2surf --s <subj> --hemi lh --sval-xyz ... (see scripts/run_freesurfer.sh)

    which writes ``surf/lh.thickness.<target>.mgh``.
    """
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Vertex-wise features require `pip install nibabel`.") from exc

    subject_dir = Path(subject_dir)
    parts = []
    for hemi in ("lh", "rh"):
        f = subject_dir / "surf" / f"{hemi}.thickness.{target}.mgh"
        parts.append(np.asarray(nib.load(str(f)).get_fdata()).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def extract_subcortical_volumes(
    subject_dir: Path | str,
    structures: Iterable[str] = DEFAULT_SUBCORTICAL,
    normalize_by_etiv: bool = True,
) -> np.ndarray:
    """Return subcortical volumes (optionally as a fraction of eTIV x 1e3)."""
    volumes, etiv = read_aseg_volumes(Path(subject_dir) / "stats" / "aseg.stats")
    out = []
    for s in structures:
        if s not in volumes:
            raise KeyError(f"{subject_dir}: structure '{s}' not found in aseg.stats")
        out.append(volumes[s])
    arr = np.asarray(out, dtype=np.float32)
    if normalize_by_etiv:
        if etiv is None or etiv <= 0:
            raise ValueError(f"{subject_dir}: eTIV not found; cannot normalise volumes")
        # Scale by 1e3 so values are O(1) (per-mille of intracranial volume).
        arr = arr / etiv * 1e3
    return arr
