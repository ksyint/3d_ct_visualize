#!/usr/bin/env bash

# Example usage script for nifti_to_interactive_viewer.py
# Adjust the paths and parameters as needed.

BODY_NII="ct.nii.gz"      # e.g., ct1.nii.gz
MASK_NII="segmentations/colon.nii.gz"      # e.g., heart.nii.gz
OUTPUT_DIR="html_output3"
THRESH=0.5
DOWNSAMPLE=2
MAX_TRI=20000
OP_BODY=0.5
OP_MASK=0.8
FLAT="--flatshading"

mkdir -p "$OUTPUT_DIR"

python3 main3.py \
  "$BODY_NII" "$MASK_NII" "$OUTPUT_DIR" \
  --threshold $THRESH \
  --downsample $DOWNSAMPLE \
  --max_triangles $MAX_TRI \
  --opacity_body $OP_BODY \
  --opacity_mask $OP_MASK \
  $FLAT

