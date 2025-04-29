#!/usr/bin/env bash
# run_nifti_viewer.sh
#
# Wrapper script to execute the `nifti_to_interactive_viewer.py` script with 3D volume, extent, and length metrics.
# Usage:
#   ./run_nifti_viewer.sh <body.nii.gz> <mask.nii.gz> <output_dir> [--threshold 0.5] [--downsample 1] \
#                        [--roi xmin,xmax,ymin,ymax,zmin,zmax] [--max_triangles 50000] \
#                        [--opacity_body 0.5] [--opacity_mask 0.8] [--flatshading]


BODY_NIFTI="ct.nii.gz"
MASK_NIFTI="segmentations/colon.nii.gz"
OUTPUT_DIR="html_output4"
shift 3

# Invoke the Python viewer script with all passed options
echo "Running viewer on mask: $(basename "$MASK_NIFTI")"
python3 main4.py \
  "$BODY_NIFTI" "$MASK_NIFTI" "$OUTPUT_DIR" "$@"

# Notify user
if [ $? -eq 0 ]; then
  echo "Done: Outputs saved to $OUTPUT_DIR"
else
  echo "Error occurred during execution."
fi