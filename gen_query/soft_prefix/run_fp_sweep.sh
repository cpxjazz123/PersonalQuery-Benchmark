#!/usr/bin/env bash
# 第一人称 (first-person) prompt sweep, 串行 3 run:
#   1. baseline α=0 L16 (D_off)
#   2. B L16 α=0.3
#   3. B L26 α=0.5 + maskCJK

set -e
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
LOG_DIR=/home/wlia0047/hj82_scratch2/wenyu/user_style_steering/sweep_logs
mkdir -p "$LOG_DIR"

RUNS=(
  "0.0 16 fp_L16_a0"
  "0.3 16 fp_L16_a0.30"
  "0.5 26 fp_L26_a0.50_maskcjk"
)

for run in "${RUNS[@]}"; do
  read -r alpha layer suffix <<< "$run"
  echo ""
  echo "==================== fp sweep: layer=$layer alpha=$alpha suffix=$suffix ===================="
  log="$LOG_DIR/${suffix}.log"
  INJECT_GEN_MAX_RECORDS=8 INJECT_GEN_BATCH=4 \
  INJECT_MODE=direct_mean INJECT_LAYERS=$layer INJECT_ALPHA=$alpha \
  INJECT_MASK_CJK=1 \
  INJECT_OUT_SUFFIX=$suffix \
  /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
    /home/wlia0047/ar57/wenyu/PersoanlQuery/gen_query/soft_prefix/generate_with_user_inject.py \
    > "$log" 2>&1
  echo "  done. tail:"
  tail -6 "$log" | sed 's/^/    /'
done

echo ""
echo "==================== all 3 fp sweep runs done ===================="
ls -la /home/wlia0047/ar57/wenyu/PersoanlQuery/result/query_records_with_query_inject_fp_*.json
