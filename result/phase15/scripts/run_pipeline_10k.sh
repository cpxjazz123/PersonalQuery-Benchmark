#!/usr/bin/env bash
# Orchestrator for Stages 2-7 of 10K user pipeline.
# Run from /home/wlia0047/ar57/wenyu/PersoanlQuery
#
# Prereqs:
#   - Stage 0 done: /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage1_filtered_users_reviews_10000u.json
#   - Stage 1 done: /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/sentences_for_rewrite_10k.jsonl
#   - Stage 5 done: result/query_records_10k.json (9982 records)
#   - Stub for VADES: result/query_by_expression_style_no_depth_check_10.json
#
# Stages 2,3: GPU heavy (Qwen forward via vLLM/transformers)
# Stage 4: VADES training (GPU)
# Stage 6: strict_nohallu generation (GPU, requires Stage 4 user distributions OR alpha=0)
# Stage 7: pick_best_cand (GPU, requires Stage 3 residuals + Stage 6 output)

set -e

REPO=/home/wlia0047/ar57/wenyu/PersoanlQuery
SCRATCH=/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades
PY=/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python
LOGDIR=$SCRATCH/logs
mkdir -p $LOGDIR

cd $REPO

echo "=== Stage 2: Qwen neutral rewrites (~3 hours) ==="
export VADES_NEUTRAL_MAX_USERS=10000
export VADES_NEUTRAL_SENTS_PER_USER=15
export VADES_NEUTRAL_REWRITE_BATCH=16
nohup $PY -u gaussian/generate_neutral_rewrites.py > $LOGDIR/stage2_rewrites_10k.log 2>&1 &
PID_STAGE2=$!
echo "  PID=$PID_STAGE2"

# Wait for Stage 2 to finish
wait $PID_STAGE2
echo "=== Stage 2 done ==="

echo "=== Stage 3: Qwen residual extraction (~3 hours) ==="
export VADES_HIDDEN_BATCH=16
export VADES_HIDDEN_MAX_LENGTH=128
export VADES_HIDDEN_LAYERS=16,20,24,26
nohup $PY -u gaussian/extract_residual_hidden.py > $LOGDIR/stage3_residual_10k.log 2>&1 &
PID_STAGE3=$!
echo "  PID=$PID_STAGE3"
wait $PID_STAGE3
echo "=== Stage 3 done ==="

echo "=== Stage 4: VADES Gaussian training (~30 min) ==="
export VADES_REVIEW_SOURCE=$SCRATCH/stage1_filtered_users_reviews_10000u.json
export VADES_RESIDUAL_HIDDEN_NPZ=$SCRATCH/residual_hidden_10k.npz
export VADES_RESIDUAL_HIDDEN_LAYER=26
export VADES_COVARIANCE_MODE=diagonal_residual_llm
export VADES_MAX_USERS=10000
export VADES_INPUT_DIR=$SCRATCH
export VADES_SENTENCE_CACHE=$SCRATCH/sentences_for_rewrite_10k.jsonl
export VADES_OUTPUT_TAG=vades_10k
nohup $PY -u gaussian/gaussian_vades.py train > $LOGDIR/stage4_vades_10k.log 2>&1 &
PID_STAGE4=$!
echo "  PID=$PID_STAGE4"
wait $PID_STAGE4
echo "=== Stage 4 done ==="

echo "=== Stage 6: strict_nohallu gen 10K (~1-2 hours) ==="
export STRICT_K=4
export STRICT_ALPHA=0.0
export STRICT_LAYERS=16
export STRICT_BATCH=8
export STRICT_MAX_NEW=128
export STRICT_OUT_SUFFIX=strict_10k
export STRICT_MAX_RECORDS=0  # no limit
nohup $PY -u gen_query/generate_strict_nohallu.py > $LOGDIR/stage6_strict_nohallu_10k.log 2>&1 &
PID_STAGE6=$!
echo "  PID=$PID_STAGE6"
wait $PID_STAGE6
echo "=== Stage 6 done ==="

echo "=== Stage 7: pick_best_cand 10K (~30 min) ==="
export PB_LAYERS=26
export PB_PCA_D=32
export PB_TAU=0.5
export PB_QWEN_BATCH=16
export PB_MAX_INPUT_LENGTH=256
nohup $PY -u select_query/pick_best_cand_10k.py > $LOGDIR/stage7_pick_best_10k.log 2>&1 &
PID_STAGE7=$!
echo "  PID=$PID_STAGE7"
wait $PID_STAGE7
echo "=== Stage 7 done ==="

echo "=== ALL DONE ==="
ls -la $SCRATCH/best_cands_10k.json