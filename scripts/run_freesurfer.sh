#!/usr/bin/env bash
# FreeSurfer processing for the PPMI T1 scans used by CNODE.
#
# Input: a CSV with header "subject,image_id,nifti" (one row per T1 scan), e.g.
#   3001,I123456,/data/ppmi/nifti/3001_I123456.nii.gz
# DICOMs downloaded from LONI can be converted first with:
#   dcm2niix -z y -f %i_%j -o /data/ppmi/nifti /data/ppmi/dicom/<series>
#
# Steps
#   1. cross-sectional recon-all for every scan            -> <subject>_<image_id>
#   2. unbiased within-subject template (subjects >= 2 scans) -> <subject>_base
#   3. longitudinal re-processing of each scan            -> <subject>_<image_id>.long.<subject>_base
#   4. (optional, VERTEX=1) resample thickness to fsaverage5 for vertex-wise features
#
# The longitudinal stream reduces within-subject measurement noise, which matters
# when forecasting small atrophy changes over ~1 year intervals.
#
# Usage: SUBJECTS_DIR=/data/fs NJOBS=8 VERTEX=0 bash scripts/run_freesurfer.sh scans.csv

set -euo pipefail

CSV=${1:?"usage: $0 scans.csv"}
: "${SUBJECTS_DIR:?SUBJECTS_DIR must be set}"
: "${FREESURFER_HOME:?FreeSurfer must be sourced (FREESURFER_HOME unset)}"
NJOBS=${NJOBS:-4}
VERTEX=${VERTEX:-0}
TARGET=${TARGET:-fsaverage5}
export SUBJECTS_DIR

cross() {  # $1 subject  $2 image_id  $3 nifti
  local sid="$1_$2"
  if [[ -f "$SUBJECTS_DIR/$sid/scripts/recon-all.done" ]]; then return 0; fi
  recon-all -s "$sid" -i "$3" -all -qcache || echo "FAILED cross $sid" >&2
}
export -f cross

# 1) cross-sectional
tail -n +2 "$CSV" | tr ',' ' ' | xargs -P "$NJOBS" -n 3 bash -c 'cross "$@"' _

# 2) + 3) longitudinal, per subject
long_subject() {  # $1 subject, remaining: image ids
  local subj="$1"; shift
  local tps=()
  for img in "$@"; do
    [[ -f "$SUBJECTS_DIR/${subj}_${img}/scripts/recon-all.done" ]] && tps+=(-tp "${subj}_${img}")
  done
  (( ${#tps[@]} / 2 >= 2 )) || { echo "skip $subj: <2 usable scans" >&2; return 0; }
  [[ -f "$SUBJECTS_DIR/${subj}_base/scripts/recon-all.done" ]] || \
    recon-all -base "${subj}_base" "${tps[@]}" -all || { echo "FAILED base $subj" >&2; return 0; }
  for img in "$@"; do
    local tp="${subj}_${img}"
    [[ -d "$SUBJECTS_DIR/$tp" ]] || continue
    [[ -f "$SUBJECTS_DIR/${tp}.long.${subj}_base/scripts/recon-all.done" ]] || \
      recon-all -long "$tp" "${subj}_base" -all || echo "FAILED long $tp" >&2
  done
}
export -f long_subject

tail -n +2 "$CSV" | awk -F, '{a[$1]=a[$1]" "$2} END {for (s in a) print s a[s]}' | \
  xargs -P "$NJOBS" -L 1 bash -c 'long_subject "$@"' _

# 4) vertex-wise thickness on a common template
if [[ "$VERTEX" == "1" ]]; then
  for d in "$SUBJECTS_DIR"/*.long.*; do
    s=$(basename "$d")
    for h in lh rh; do
      out="$d/surf/$h.thickness.$TARGET.mgh"
      [[ -f "$out" ]] || mri_surf2surf --s "$s" --hemi "$h" --sval "$d/surf/$h.thickness" \
        --trgsubject "$TARGET" --tval "$out" --fwhm-trg 10 --cortex
    done
  done
fi
echo "done. Check FreeSurfer QC (e.g. visual inspection / Euler number) before building the cohort."
