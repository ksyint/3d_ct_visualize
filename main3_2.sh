#!/usr/bin/env bash
# run_nifti_viewer_all3.sh
# Processes all .nii/.nii.gz masks in the segmentations directory using main3.py.
# Usage:
#   ./run_nifti_viewer_all3.sh <body_nii> <segmentations_dir> <output_base_dir> [options]


# Required arguments\ nBODY_NII="$1"
SEG_DIR="segmentations"
OUTPUT_BASE="html_output3"


# Default parameters (override via command-line args if desired)
THRESH=0.5
DOWNSAMPLE=2
MAX_TRI=20000
OP_BODY=0.5
OP_MASK=0.8
FLAT="--flatshading"

# Ensure base output directory exists
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

  echo "Running main3.py on mask: $MASK_NAME"
  python3 main3.py \
    "ct.nii.gz" "$MASK_PATH" "$OUT_DIR" \
    --threshold $THRESH \
    --downsample $DOWNSAMPLE \
    --max_triangles $MAX_TRI \
    --opacity_body $OP_BODY \
    --opacity_mask $OP_MASK \
    $FLAT "$@"

  if [ $? -eq 0 ]; then
    echo "Done: outputs saved to $OUT_DIR"
  else
    echo "Error occurred for: $MASK_NAME"
  fi
  echo "------------------------------------"
done

echo "All masks processed."
