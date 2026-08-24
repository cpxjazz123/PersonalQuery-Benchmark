"""Stage 7D driver: scale-up to 100 ASIN (4 users × 5 K)

Run the full pipeline: regen → retrieval → analyze → 7B PCA48 D_syn.

Compared to Stage 7 (27 ASIN, 8 users/asin):
- 100 ASIN (was 40 top-N), cap 4 users/asin (was 8) → broader item coverage
- Same K=5, same vLLM, same prompt
- Reuses cached: corpus embeddings, BM25 index, PCA48 model

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage7d_scale100.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7d_scale100.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
RESULT = REPO_ROOT / "result/gaussian_vades"
PY = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"

REGEN_OUT = SCRATCH / "stage7d_regen.json"
RETRIEVAL_OUT = SCRATCH / "stage7d_retrieval_results.json"
ANALYZE_OUT = RESULT / "syntax_subspace_stage7d_analyze.json"
SYN7B_OUT = RESULT / "syntax_subspace_stage7d_syn.json"
SYN7C_OUT = RESULT / "syntax_subspace_stage7d_robust.json"


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd: list, log_path: Path, env: dict | None = None):
    log(f"RUN: {' '.join(cmd)}")
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=full_env)
        ret = proc.wait()
    log(f"  → exit code {ret}, log: {log_path}")
    return ret


def main():
    log("=== Stage 7D driver: scale to 100 ASIN ===")

    # 1. Regen (already modified to use stage7d_regen.json, 100 ASIN, 4 users)
    log("\n=== Step 1: regen 100 ASIN × 4 users × 5 K ===")
    if REGEN_OUT.exists():
        log(f"  {REGEN_OUT} already exists, skipping regen")
    else:
        ret = run(
            [PY, str(REPO_ROOT / "gaussian/syntax_subspace_stage7_regen.py")],
            SCRATCH / "stage7d_regen.log",
        )
        if ret != 0:
            log(f"  regen failed, aborting")
            return

    # 2. Retrieval
    log("\n=== Step 2: retrieval (MiniLM + BM25) ===")
    if RETRIEVAL_OUT.exists():
        log(f"  {RETRIEVAL_OUT} already exists, skipping retrieval")
    else:
        ret = run(
            [PY, str(REPO_ROOT / "gaussian/syntax_subspace_stage7_retrieval.py")],
            SCRATCH / "stage7d_retrieval.log",
            env={
                "STAGE7_REGEN_IN": str(REGEN_OUT),
                "STAGE7_RETRIEVAL_OUT": str(RETRIEVAL_OUT),
            },
        )
        if ret != 0:
            log(f"  retrieval failed, aborting")
            return

    # 3. Stage 7D analyze (volatility + Wilcoxon + bootstrap)
    log("\n=== Step 3: analyze ===")
    if ANALYZE_OUT.exists():
        log(f"  {ANALYZE_OUT} already exists, skipping analyze")
    else:
        ret = run(
            [PY, str(REPO_ROOT / "gaussian/syntax_subspace_stage7_analyze.py")],
            SCRATCH / "stage7d_analyze.log",
            env={
                "STAGE7_RETRIEVAL_IN": str(RETRIEVAL_OUT),
                "STAGE7_ANALYZE_OUT": str(ANALYZE_OUT),
            },
        )

    # 4. Stage 7D PCA48 D_syn mixedlm
    log("\n=== Step 4: PCA48 D_syn mixedlm ===")
    if SYN7B_OUT.exists():
        log(f"  {SYN7B_OUT} already exists, skipping 7B syn")
    else:
        # The 7B script reads RETRIEVAL_IN from hardcoded path; need to copy results
        # to expected location OR re-route via env. We didn't add env support to 7B,
        # so symlink the input for now.
        symlink_path = SCRATCH / "stage7_retrieval_results.json"
        if not symlink_path.exists():
            os.symlink(RETRIEVAL_OUT, symlink_path)
            log(f"  symlinked {RETRIEVAL_OUT.name} → stage7_retrieval_results.json")
        # Also need the QUERY_FEAT_CACHE shared (stage7b_query_features.jsonl.gz)
        ret = run(
            [PY, str(REPO_ROOT / "gaussian/syntax_subspace_stage7b_syn.py")],
            SCRATCH / "stage7d_syn.log",
        )
        # Move the output to the 7D path
        src_out = RESULT / "syntax_subspace_stage7b_syn.json"
        if src_out.exists() and not SYN7B_OUT.exists():
            os.rename(src_out, SYN7B_OUT)
            log(f"  moved {src_out.name} → {SYN7B_OUT.name}")

    # 5. Stage 7D robustness check (4 settings)
    log("\n=== Step 5: robustness check (4 settings) ===")
    if SYN7C_OUT.exists():
        log(f"  {SYN7C_OUT} already exists, skipping 7C robust")
    else:
        # The 7C script reads from hardcoded retrieval path; symlink again
        symlink_path = SCRATCH / "stage7_retrieval_results.json"
        if not symlink_path.exists():
            os.symlink(RETRIEVAL_OUT, symlink_path)
        ret = run(
            [PY, str(REPO_ROOT / "gaussian/syntax_subspace_stage7c_robust.py")],
            SCRATCH / "stage7d_robust.log",
        )
        src_out = RESULT / "syntax_subspace_stage7c_robust.json"
        if src_out.exists() and not SYN7C_OUT.exists():
            os.rename(src_out, SYN7C_OUT)
            log(f"  moved {src_out.name} → {SYN7C_OUT.name}")

    # Summary
    log("\n=== Stage 7D artifacts ===")
    for p in [REGEN_OUT, RETRIEVAL_OUT, ANALYZE_OUT, SYN7B_OUT, SYN7C_OUT]:
        if p.exists():
            log(f"  ✓ {p} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            log(f"  ✗ MISSING {p}")


if __name__ == "__main__":
    main()