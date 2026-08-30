#!/usr/bin/env python3
"""Phase 6.C.1-eval-top8-scaleup — 同 eval_top8 逻辑, 但对比 Phase 6.A init vs PIP scaleup 500steps.

scale-up 配置 (2026-08-30):
- MAX_STEPS: 200 → 500
- LAMBDA_COS: 0.5 → 1.0
- LAMBDA_MARGIN: 0.5 → 1.0
- 其他不变

复用 syntax_subspace_pca48_pip_indirect_eval_top8 全部逻辑,只覆盖 CKPT_PATHS。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")

# 复用 train 配置常量(eval 端相同)
import syntax_subspace_pca48_pip_indirect_eval_top8 as base

# 覆盖 CKPT_PATHS: 对比 6.A vs scaleup 500-steps (而非 6.A vs 200-step)
base.CKPT_PATHS = [
    ("Phase 6.A (init)", SCRATCH / "pca48_ckpt" / "projector_final.pt"),
    ("PIP scaleup 500steps", SCRATCH / "pca48_pip_indirect_ckpt" / "pip_indirect_scaleup_500steps.pt"),
]

# 复用其它常量(同 base)
N_PAIRS = base.N_PAIRS
N_PCS = base.N_PCS
ALPHAS = base.ALPHAS
N_CANDIDATES = base.N_CANDIDATES
TEMPERATURE = base.TEMPERATURE
TOP_P = base.TOP_P
MAX_NEW_TOKENS = base.MAX_NEW_TOKENS
SEED = base.SEED


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [pip_eval_top8_scaleup] {msg}", flush=True)


def main():
    log("=== Phase 6.C.1-eval-top8-scaleup — 6.A vs PIP scaleup 500steps ===")
    log(f"  config: N_PAIRS={N_PAIRS}, N_PCS={N_PCS}, ALPHAS={ALPHAS}, N_CANDIDATES={N_CANDIDATES}")
    log(f"  ckpts: {base.CKPT_PATHS}")

    # 复用 base.main 走全部 eval 流程
    # 但要 patch 输出路径
    base.log = log  # 重定向 log

    # Monkey-patch 输出 JSON 路径
    import syntax_subspace_pca48_pip_indirect_eval_top8 as _b
    _orig_main = _b.main

    def patched_main():
        # 直接调 base.main 但 override 输出路径
        # base.main 末尾会写 phase6c1_pip_eval_top8.json
        # 我们用 monkey-patch 替换 Path 用法:
        real_main_call = base.main
        # base.main 内部 hardcoded: out_path = LOG_DIR / "phase6c1_pip_eval_top8.json"
        # 改 LOG_DIR... 但这是 module-level 常量
        # 更简单: 自己写一个 wrapper
        pass

    # 干脆直接调 base.main() 跑完后, 复制 JSON 内容覆盖原文件
    base.main()

    # base.main 写到了 phase6c1_pip_eval_top8.json, 移动到 scaleup 专用名
    import shutil
    src = _b.LOG_DIR / "phase6c1_pip_eval_top8.json"
    dst = _b.LOG_DIR / "phase6c1_pip_eval_top8_scaleup_500steps.json"
    shutil.move(str(src), str(dst))
    log(f"  moved → {dst}")


if __name__ == "__main__":
    main()
