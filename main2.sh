
# 이미지내 장기 여러개 멀티마스크 띄운거 한개 만들기

set -euo pipefail

BODY="ct.nii.gz"

masks=(
  "segmentations/kidney_left.nii.gz"
  "segmentations/kidney_right.nii.gz"
  "segmentations/liver.nii.gz"
)

OUT_DIR="html_output2"
mkdir -p "$OUT_DIR"

labels=()
for m in "${masks[@]}"; do
  b=$(basename "$m")
  labels+=( "${b%.nii.gz}" )
done

joined=$(IFS=***; echo "${labels[*]}")
OUT_HTML="$OUT_DIR/${joined}.html"

python main2.py \
  "$BODY" \
  "$OUT_HTML" \
  "${masks[@]}" \
  --threshold 0.5 \
  --downsample 2 \
  --max_triangles 50000 \
  --opacity_body 0.4 \
  --opacity_masks 0.9 \
  --flatshading

echo "✔ Generated HTML: $OUT_HTML"
echo "✔ Generated labels: ${OUT_DIR}/${joined}.txt"
