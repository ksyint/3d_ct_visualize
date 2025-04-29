#!/usr/bin/env bash
# run_nifti_viewer_all.sh
#
# Executes `main4.py` for all .nii.gz masks in the segmentations directory.
# Usage:
#   ./run_nifti_viewer_all.sh ct.nii.gz segmentations html_output4 [--threshold 0.5] [...] [--flatshading]



BODY_NIFTI="ct.nii.gz"
SEG_DIR="segmentations"
OUTPUT_BASE="html_output4"
shift 3

# Create output base folder if needed
mkdir -p "$OUTPUT_BASE"

echo "Processing all masks in: $SEG_DIR"
for MASK_PATH in "$SEG_DIR"/*.nii "$SEG_DIR"/*.nii.gz; do
  # Skip if no files found
  [ -e "$MASK_PATH" ] || continue

  MASK_NAME=$(basename "$MASK_PATH" .nii.gz)
  MASK_NAME=${MASK_NAME%.nii}
  OUTPUT_DIR="$OUTPUT_BASE/${MASK_NAME}"
  mkdir -p "$OUTPUT_DIR"

  echo "Running viewer on mask: $MASK_NAME"
  python3 main4.py \
    "$BODY_NIFTI" "$MASK_PATH" "$OUTPUT_DIR" "$@"

  if [ $? -eq 0 ]; then
    echo "Done: Outputs saved to $OUTPUT_DIR"
  else
    echo "Error occurred for mask: $MASK_NAME"
  fi
  echo "------------------------------------"
done

echo "All masks processed."
