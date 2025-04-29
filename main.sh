# 기존 segmentations 내 모든 마스크 nii 를 html로 만들고 nii 중에 none값인거 모두 거르기, 장기하나 싱글 마스크 띄운거 여러개 
BODY="/home/sunsykim/sooyong/totalsegment/chest_ct/ct_scans/coronacases_org_001.nii"

SEG_DIR="/home/sunsykim/sooyong/totalsegment/chest_ct/output1"

OUT_DIR="chest_ct_ct_scans_coronacases_org_001"

mkdir -p "$OUT_DIR"

THRESH=0.5
DOWNSAMPLE=2
MAX_TRI=50000
OP_BODY=0.4
OP_HEART=0.9

for seg in "$SEG_DIR"/*.nii.gz; do
  fname=$(basename "$seg" .nii.gz)
  out="$OUT_DIR/${fname}.html"

  echo "=== Processing $seg → $out ==="
  if python main.py \
      "$BODY" "$seg" "$out" \
      --threshold "$THRESH" \
      --downsample "$DOWNSAMPLE" \
      --max_triangles "$MAX_TRI" \
      --opacity_body "$OP_BODY" \
      --opacity_heart "$OP_HEART"
  then
    echo "✔ Success: $fname"
  else
    echo "✖ Error on $fname — deleting input file"
    rm -f "$seg"
  fi
done
