#!/usr/bin/env bash

# Shell script to batch-generate HTML overlays for each NNIT mask in a directory,
# delete any .nii.gz files that fail (e.g. all-zero/None volumes),
# and produce a single-organ mask viewer per file.

# --- Configuration --------------------------------
BODY="ct.nii.gz"
SEG_DIR="segmentations"
OUT_DIR="html_output6"

THRESH=0.5
DOWNSAMPLE=2
MAX_TRI=50000
OP_BODY=0.4
OP_HEART=0.2

mkdir -p "${OUT_DIR}"

# --- Batch loop ------------------------------------
for seg in "${SEG_DIR}"/*.nii.gz; do
  fname=$(basename "${seg}" .nii.gz)
  out="${OUT_DIR}/${fname}.html"

  echo "=== Processing ${seg} → ${out} ==="
  python3 main6.py \
    "${BODY}" "${seg}" "${out}" \
    --threshold "${THRESH}" \
    --downsample "${DOWNSAMPLE}" \
    --max_triangles "${MAX_TRI}" \
    --opacity_body "${OP_BODY}" \
    --opacity_heart "${OP_HEART}"

  if [ $? -eq 0 ]; then
    echo "✔ Success: ${fname}"
  else
    echo "✖ Failed: ${fname} — deleting ${seg}"
    rm -f "${seg}"
  fi
  echo

done
