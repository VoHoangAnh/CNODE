# CNODE — Conditional Neural ODE for Longitudinal Parkinson's Disease Progression Forecasting

Unofficial PyTorch re-implementation of

> X. Wang, Y. Zhao, K. Han, X. Luo, S. van Rooij, J. Stevens, L. He, L. Zhan, Y. Sun, W. Wang, C. Yang.
> **Conditional Neural ODE for Longitudinal Parkinson's Disease Progression Forecasting.**
> IEEE BIBM 2025, pp. 4131–4134. doi:[10.1109/BIBM66473.2025.11356813](https://doi.org/10.1109/BIBM66473.2025.11356813)

This repository is not affiliated with the authors. It reproduces the method from the
paper's description; where the paper is under-specified, the choices made are listed in
[Implementation notes](#implementation-notes-and-deviations).

## Method in one paragraph

Each PD subject *i* has a few irregularly spaced T1 MRI visits. Shape features
`X_{i,k}` (cortical thickness) are modelled as a continuous latent trajectory. Two small
regressors map the baseline condition features — subcortical volumes `C_i` and metadata
`M_i` (age, sex) — to an onset shift `τ_i` and progression speed `γ_i`, placing every visit
on a shared disease timeline `t̃ = τ_i + γ_i (t − t_{i,1})`. A metadata-weighted InfoNCE loss
pulls visits with similar morphology to nearby aligned times. A latent ODE (GRU recognition
encoder → `z_{i,0}` → `ODESolve` on the aligned timeline → MLP decoder) forecasts features at
arbitrary future times.

| Paper | Code |
|---|---|
| Eq. 1 `τ_i, γ_i = f(M_i, C_i)` | `cnode/models/cnode.py::AlignmentHead` |
| Eq. 2 aligned time | `CNODE.align_times` |
| Eq. 3–4 weighted InfoNCE | `cnode/models/losses.py::metadata_weighted_infonce` |
| Eq. 5–6 encoder | `RecognitionEncoder` |
| Eq. 7 ODE solve | `CNODE.integrate` (+ `_AlignedODEFunc`) |
| Eq. 8–9 decoder | `CNODE.decoder` |
| Eq. 10–12 objective | `CNODE.forward` |
| Table I protocol | `cnode/train.py::run_cv` |
| Fig. 2 λ-ratio sweep | `cnode sweep` |

## Installation

```bash
git clone <this-repo> && cd cnode-pd
pip install -e ".[dev]"          # add ".[vertex]" for vertex-wise features (nibabel)
pytest                            # 8 tests, ~10 s on CPU
```

Python ≥ 3.9, PyTorch ≥ 2.1, `torchdiffeq`.

## Quick start (no PPMI access needed)

```bash
cnode synth --out data/synthetic                       # PPMI-shaped synthetic cohort
cnode cv --config configs/cnode.yaml --data data/synthetic --out runs/synthetic
cnode sweep --config configs/cnode.yaml --data data/synthetic --out runs/sweep   # Fig. 2
```

`runs/<name>/` contains `results_per_fold.csv`, `results_summary.csv`, a Table-I-style
`results_table.md`, per-epoch training histories and per-fold checkpoints (model weights +
the fold's normalisation statistics).

## PPMI preprocessing

PPMI data cannot be redistributed. Apply for access at <https://www.ppmi-info.org/access-data-specimens/download-data>
and download through LONI IDA.

### 1. Download

* **Imaging:** build an image collection of 3D T1-weighted scans (MPRAGE / SPGR) for the PD
  cohort and download it together with its **collection CSV** (one row per scan, with
  subject ID, image ID, acquisition date, visit, group, age, sex).
* **Study data:** `Participant_Status` (enrolment cohort) and `Demographics` (birth date, sex).

Column names vary between PPMI releases — open the CSVs and edit the `col_*` keys in
`configs/ppmi.yaml`. Nothing else in the code assumes specific column names.

### 2. FreeSurfer

```bash
# DICOM -> NIfTI if needed:  dcm2niix -z y -f %i_%j -o nifti/ dicom/<series>
# scans.csv header: subject,image_id,nifti
SUBJECTS_DIR=/data/fs NJOBS=8 bash scripts/run_freesurfer.sh scans.csv
# VERTEX=1 additionally resamples thickness to fsaverage5 for vertex-wise features
```

The script runs cross-sectional `recon-all`, then the FreeSurfer **longitudinal stream**
(unbiased within-subject template + `-long` re-processing). The paper does not state which
stream was used; the longitudinal one is recommended because it reduces within-subject
measurement noise, which is of the same order as one year of PD-related atrophy. Do visual
QC of the surfaces before continuing.

### 3. Build the cohort

```bash
cnode preprocess --config configs/ppmi.yaml --out data/ppmi_cohort
```

What `cnode/data/ppmi.py` does, in order:

1. **PD filter** — keeps participants whose `COHORT` is Parkinson's disease (healthy controls,
   SWEDD and prodromal participants are excluded). Without a status file it falls back to the
   collection CSV's `Group` column.
2. **Session de-duplication** — PPMI often has several T1 series per session (repeats,
   alternative sequences). Scans within `min_gap_days` (90) of the previous kept scan are
   merged, keeping the earliest.
3. **Demographics** — age at scan is computed from the de-identified `MM/YYYY` birth date
   (mid-month assumed); sex is coded 0 = female, 1 = male.
4. **Features** — for each scan:
   * shape features `X_{i,k}`: `ThickAvg` of the 34 Desikan–Killiany regions per hemisphere
     from `stats/?h.aparc.stats` → **68 features** (or vertex-wise thickness with
     `feature_type: vertex`);
   * condition features `C_i`: 14 subcortical volumes (thalamus, caudate, putamen, pallidum,
     hippocampus, amygdala, accumbens; both hemispheres) from `stats/aseg.stats`, divided by
     eTIV to remove head-size effects. `Thalamus-Proper` (FreeSurfer < 7) is mapped to `Thalamus`.
5. **QC** — scans with a missing/failed reconstruction or implausible regional thickness
   (outside 0.5–5 mm) are dropped.
6. **Cohort rules** — keep subjects with ≥ 2 usable scans; optionally truncate to the first
   `max_visits` (3, as in the paper's cohort).
7. **Time axis** — visit times in years since the subject's first scan.
8. **Conditions from the first visit** — `C_i` and `M_i` are taken from the baseline scan
   only (Sec. II-A), so they are available at forecast time.

Outputs: `visits.csv` (long format: subject, image, date, visit index, time, 68 features),
`subjects.csv` (age, sex, volumes at baseline) and `cohort_summary.json` (subject count,
visits per subject, mean/min/max interval — compare with Sec. III-A: 161 subjects, 50 with
three visits, 111 with two, intervals 0.61–2.27 y, mean 1.11 y).

**Normalisation is deliberately not part of preprocessing.** Z-scoring of features,
volumes and age is fitted on each cross-validation training fold only
(`cnode/data/dataset.py::Normalizer`) to avoid test-set leakage.

## Evaluation protocol

* 5-fold cross-validation split **by subject**, stratified by number of visits; the same
  folds are used for every model.
* In each training fold, 10 % of subjects are held out for early stopping.
* Task: given all but the last visit (plus `C_i`, `M_i`), forecast the last visit.
* Metrics on z-scored features of the forecast visits: MSE, RMSE = √MSE, and R² averaged
  over features.
* Baselines: RNN, LSTM, GRU, Transformer (fed a regular 1-year grid, i.e. visit order only,
  as in Sec. III-B), latent Neural ODE (CNODE without alignment and contrastive loss), and
  last-observation-carried-forward as a sanity floor. LLMTime is not re-implemented; use the
  official code at <https://github.com/ngruver/llmtime>.

## Results on the synthetic cohort

The table below is produced by `cnode cv` on `cnode synth` data (5 folds, CPU, ~4 min). It
**only verifies that the pipeline runs end to end** and says nothing about PPMI performance.

| Model | MSE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|
| CNODE | 0.5179 ± 0.0175 | 0.7195 ± 0.0121 | 0.3355 ± 0.0413 |
| Neural ODE | 0.5494 ± 0.0195 | 0.7411 ± 0.0133 | 0.2958 ± 0.0390 |
| GRU | 0.3156 ± 0.0495 | 0.5605 ± 0.0433 | 0.5889 ± 0.0671 |
| RNN | 0.3341 ± 0.0263 | 0.5777 ± 0.0227 | 0.5623 ± 0.0360 |
| LSTM | 0.3058 ± 0.0468 | 0.5518 ± 0.0413 | 0.6021 ± 0.0653 |
| Transformer | 0.2591 ± 0.0141 | 0.5089 ± 0.0138 | 0.6638 ± 0.0168 |
| Persistence | 0.8399 ± 0.0782 | 0.9157 ± 0.0431 | −0.1868 ± 0.1955 |

The alignment consistently improves over the plain latent Neural ODE, but on this synthetic
data the discrete baselines are better. They predict the *change* from the last visit, so
subject-specific offsets carry over for free, whereas the latent ODE must re-encode them
through a bottleneck from ~115 training subjects and overfits. The paper's Table I numbers
(e.g. CNODE MSE 0.0258) are on its own PPMI feature scaling, which is not reported, and are
not comparable to the numbers above.

## Implementation notes and deviations

The paper leaves several details open. Choices made here:

1. **Features.** The paper mentions both "vertex-wise medial thickness" and regional average
   thickness over the 34 DK regions per hemisphere ("68 regions"). The default is the 68
   regional values; vertex-wise thickness is available via `feature_type: vertex`.
2. **Time similarity in Eq. 4.** "Cosine similarity between aligned times" is degenerate for
   scalars (always ±1). `s(t̃_i, t̃_j) = −(t̃_i − t̃_j)²` is used instead.
3. **Positive pairs.** Pairs from the same subject are excluded as positives (they would
   collapse `γ_i → 0`) but remain negatives. `w_ij = (1 + cos([M_i,C_i],[M_j,C_j]))/2` keeps
   weights non-negative. The loss is averaged per positive and per anchor.
4. **Symbol clash.** The paper uses `τ` for both the onset shift and the InfoNCE temperature;
   the code calls the latter `contrast_temperature`.
5. **Time-dependent vector field.** `g(z, t̃)` takes the aligned time as input; with an
   autonomous field, the onset shift `τ_i` would have no effect on the dynamics.
6. **Batched integration.** The ODE is solved in elapsed time `s` with `dz/ds = γ_i·g(z, τ_i + γ_i s)`,
   which is exactly Eq. 7 by the chain rule and lets a batch share one solver grid
   (verified in `tests/test_models.py`).
7. **Encoder.** "Enc_ϕ" is not specified; a GRU over observed visits with aligned elapsed time
   as an extra input is used.
8. **Loss weights.** Fig. 2 reports that λ_KL : λ_C = 1 : 1 works best but not the absolute
   scale; both are set to 0.1. `γ_i` uses softplus and the alignment is initialised at
   `τ = 0, γ = 1`.
9. **Hyperparameters** not in the paper (latent size, hidden sizes, δ, temperature, learning
   rate, epochs) are marked in `configs/cnode.yaml`.

## Repository layout

```
cnode/
  data/freesurfer.py   FreeSurfer stats parsers (aparc thickness, aseg volumes, eTIV)
  data/ppmi.py         PPMI cohort construction
  data/dataset.py      dataset, fold-wise normalisation, batching
  data/synthetic.py    PPMI-shaped synthetic cohort
  models/cnode.py      CNODE (Eqs. 1-12)
  models/losses.py     MSE, KL, metadata-weighted InfoNCE
  models/baselines.py  RNN/LSTM/GRU/Transformer, persistence
  train.py             training, early stopping, cross-validation
  metrics.py           MSE / RMSE / R²
  cli.py               `cnode` command
configs/               cnode.yaml (training), ppmi.yaml (preprocessing)
scripts/run_freesurfer.sh
tests/
```

## Citation

If you use this code, cite the original paper:

```bibtex
@inproceedings{wang2025cnode,
  title     = {Conditional Neural ODE for Longitudinal Parkinson's Disease Progression Forecasting},
  author    = {Wang, Xiaoda and Zhao, Yuji and Han, Kaiqiao and Luo, Xiao and van Rooij, Sanne and
               Stevens, Jennifer and He, Lifang and Zhan, Liang and Sun, Yizhou and Wang, Wei and Yang, Carl},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages     = {4131--4134},
  year      = {2025},
  doi       = {10.1109/BIBM66473.2025.11356813}
}
```

Data used in preparing this work should acknowledge PPMI according to the PPMI Data Use
Agreement and publication policy.

## License

MIT (code only). PPMI data are subject to the PPMI Data Use Agreement.
