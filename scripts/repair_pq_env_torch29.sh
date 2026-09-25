#!/usr/bin/env bash
set -euo pipefail
PY=/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python
PIP=/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/pip
LOG=/home/wlia0047/hj82_scratch2/wenyu/pq_torch29_repair.log
exec > >(tee -a "$LOG") 2>&1
echo "=== repair start $(date -Is) ==="
"$PY" -c "import torch; print('before torch', torch.__version__)" || true
"$PIP" uninstall -y torch torchvision torchaudio triton xformers 2>/dev/null || true
rm -rf /home/wlia0047/ar57_scratch/wenyu/pq_env/lib/python3.10/site-packages/torch-2.6.0.dist-info
rm -rf /home/wlia0047/ar57_scratch/wenyu/pq_env/lib/python3.10/site-packages/torchvision-0.19.0.dist-info
rm -rf /home/wlia0047/ar57_scratch/wenyu/pq_env/lib/python3.10/site-packages/torchaudio-2.11.0.dist-info
"$PIP" install --no-cache-dir \
  'torch==2.9.0' 'torchvision==0.24.0' 'torchaudio==2.9.0' \
  'transformers>=4.56,<5' 'tokenizers>=0.21.1'
"$PY" -c "import torch; print('after torch', torch.__version__)"
"$PY" -c "import torch._inductor.graph; print('inductor ok')"
"$PY" -c "import vllm; print('vllm', vllm.__version__)"
echo "=== repair done $(date -Is) ==="
