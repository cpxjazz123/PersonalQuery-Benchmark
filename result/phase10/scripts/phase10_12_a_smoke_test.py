#!/usr/bin/env python3
"""Smoke test: 生成 12 条候选验证句法控制 prompt 设计合理性.
(每种 cond × 1 candidate, 1 个 pair)
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

import phase10_12_a_syntactic_diversity as mod

# 1 个测试 pair
attrs_5 = {"Brand": "TL Care", "Color": "Ecru", "Material": "Cotton"}

print("=" * 70)
print("Smoke test: 12 种句法控制 × 1 candidate")
print("=" * 70)

from llm_client import create_qwen_local_client
client = create_qwen_local_client()

for cond_idx, cond in enumerate(mod.SYNTAX_CONDITIONS):
    prompt = mod.build_prompt(attrs_5, attrs_5, cond)
    result = client.call(prompt, max_tokens=120, temperature=0.9)
    print(f"\n[C{cond_idx}] cond={cond}")
    print(f"  Q: {result.strip()[:200]}")