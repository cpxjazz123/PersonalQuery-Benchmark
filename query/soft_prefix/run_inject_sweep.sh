#!/usr/bin/env bash
# α × layer 6-point sweep + baseline D_off, 串行执行 (GPU 单进程).
# 每个 run 输出独立 file (INJECT_OUT_SUFFIX 隔离), log 走 scratch2/sweep_logs/.

set -e
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
SWEEP_LOG_DIR=/home/wlia0047/hj82_scratch2/wenyu/user_style_steering/sweep_logs
mkdir -p "$SWEEP_LOG_DIR"

# (mode, layer, alpha, suffix)
RUNS=(
  "none 16 0.0 baseline_a0_L16"
  "direct_mean 16 0.05 b_L16_a0.05"
  "direct_mean 16 0.10 b_L16_a0.10"
  "direct_mean 16 0.20 b_L16_a0.20"
  "direct_mean 20 0.05 b_L20_a0.05"
  "direct_mean 20 0.10 b_L20_a0.10"
  "direct_mean 20 0.20 b_L20_a0.20"
)

for run in "${RUNS[@]}"; do
  read -r mode layer alpha suffix <<< "$run"
  echo ""
  echo "==================== sweep run: mode=$mode layer=$layer alpha=$alpha suffix=$suffix ===================="
  log="$SWEEP_LOG_DIR/${suffix}.log"
  if [ "$mode" = "none" ]; then
    # baseline D_off: 不注入, alpha=0
    INJECT_GEN_MAX_RECORDS=8 INJECT_GEN_BATCH=4 \
    INJECT_MODE=direct_mean INJECT_LAYERS=$layer INJECT_ALPHA=0.0 \
    INJECT_MASK_CJK=1 \
    INJECT_OUT_SUFFIX=$suffix \
    /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
      /home/wlia0047/ar57/wenyu/PersoanlQuery/query/soft_prefix/generate_with_user_inject.py \
      > "$log" 2>&1
  else
    INJECT_GEN_MAX_RECORDS=8 INJECT_GEN_BATCH=4 \
    INJECT_MODE=$mode INJECT_LAYERS=$layer INJECT_ALPHA=$alpha \
    INJECT_MASK_CJK=1 \
    INJECT_OUT_SUFFIX=$suffix \
    /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
      /home/wlia0047/ar57/wenyu/PersoanlQuery/query/soft_prefix/generate_with_user_inject.py \
      > "$log" 2>&1
  fi
  echo "  done. tail:"
  tail -5 "$log" | sed 's/^/    /'
done

echo ""
echo "==================== all 7 sweep runs done ===================="
ls -la /home/wlia0047/ar57/wenyu/PersoanlQuery/result/query_records_with_query_inject_*.json | tail -10
