#!/bin/bash
set -euo pipefail

LLAMA_FACTORY_DIR="/home/wlia0047/ar57/wenyu/MultiAgent/Agent_Foundation_Models/LLaMA-Factory"
CONFIG_FILE="examples/train_lora/qwen2_5_0_5b_memory_policy_sft.yaml"

cd "$LLAMA_FACTORY_DIR"
PYTHONPATH="$LLAMA_FACTORY_DIR/src:${PYTHONPATH:-}" python3 -m llamafactory.cli train "$CONFIG_FILE"
