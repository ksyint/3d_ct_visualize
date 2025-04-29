#!/usr/bin/env bash
# run_nifti_viewer_all.sh
# Loops through all .nii and .nii.gz masks in a directory and runs the updated viewer script.
# Usage:
#   ./run_nifti_viewer_all.sh <body_nifti> <segmentations_dir> <output_base_dir> [options]


# Required arguments
BODY_NII="ct.nii.gz"
SEG_DIR="segmentations"
OUTPUT_BASE="html_output5"
shift 3

# Default parameters (override via command-line args if needed)
THRESH=0.5
DOWNSAMPLE=1
MAX_TRIANGLES=50000
OP_BODY=0.5
OP_MASK=0.8
FLAT_FLAG="--flatshading"

# Create base output directory
mkdir -p "$OUTPUT_BASE"

for MASK_PATH in "$SEG_DIR"/*.nii "$SEG_DIR"/*.nii.gz; do
  [ -e "$MASK_PATH" ] || continue

  # Derive mask name without extension
  MASK_NAME=$(basename "$MASK_PATH")
  MASK_NAME=${MASK_NAME%.nii.gz}
  MASK_NAME=${MASK_NAME%.nii}

  # Prepare per-mask output directory
  OUT_DIR="$OUTPUT_BASE/$MASK_NAME"
  mkdir -p "$OUT_DIR"

  python3 main5.py \
    "$BODY_NII" "$MASK_PATH" "$OUT_DIR" \
    --threshold $THRESH \
    --downsample $DOWNSAMPLE \
    --max_triangles $MAX_TRIANGLES \
    --opacity_body $OP_BODY \
    --opacity_mask $OP_MASK \
    $FLAT_FLAG "$@"

  if [ $? -eq 0 ]; then
    echo "Success: Outputs saved to $OUT_DIR"
  else
    echo "Error occurred for mask: $MASK_NAME"
  fi
done

echo "\nAll masks processed."
