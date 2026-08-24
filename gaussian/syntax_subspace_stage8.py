"""Stage 8 driver: orchestrate retrieval + analyze + 7B PCA48 + 7C robustness.

Runs:
  1. MiniLM + BM25 retrieval on stage8_regen.json
  2. analyze.py → stage8_analyze.json (Wilcoxon + bootstrap + mixedlm)
  3. 7B PCA48 D_syn mixedlm → stage8_syn.json (rename from default 7b output)
  4. 7C 4-setting robustness → stage8_robust.json (rename)

Reuses existing scripts via env vars + symlinks:
  - gaussian/syntax_subspace_stage7_retrieval.py
  - gaussian/syntax_subspace_stage7_analyze.py
  - gaussian/syntax_subspace_stage7b_syn.py
  - gaussian/syntax_subspace_stage7c_robust.py

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
RESULT = REPO_ROOT / "result/gaussian_vades"
PY = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"

REGEN_OUT = SCRATCH / "stage8_regen.json"
RETRIEVAL_OUT = SCRATCH / "stage8_retrieval_results.json"
ANALYZE_OUT = RESULT / "syntax_subspace_stage8_analyze.json"
SYN8B_OUT = RESULT / "syntax_subspace_stage8_syn.json"
SYN8C_OUT = RESULT / "syntax_subspace_stage8_robust.json"


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
    log("=== Stage 8 driver: 100 ASIN × 10 users × K=3 ===")

    # 1. Retrieval
    log("\n=== Step 1: retrieval (MiniLM + BM25) ===")
    if RETRIEVAL_OUT.exists():
        log(f"  {RETRIEVAL_OUT} already exists, skipping")
    else:
        ret = run(
            [PY, str(REPO_ROOT / "gaussian/syntax_subspace_stage7_retrieval.py")],
            SCRATCH / "stage8_retrieval.log",
            env={
                "STAGE7_REGEN_IN": str(REGEN_OUT),
                "STAGE7_RETRIEVAL_OUT": str(RETRIEVAL_OUT),
            },
        )
        if ret != 0:
            log(f"  retrieval failed, aborting")
            return

    # 2. Analyze
    log("\n=== Step 2: analyze ===")
    if ANALYZE_OUT.exists():
        log(f"  {ANALYZE_OUT} already exists, skipping")
    else:
        ret = run(
            [PY, str(REPO_ROOT / "gaussian/syntax_subspace_stage7_analyze.py")],
            SCRATCH / "stage8_analyze.log",
            env={
                "STAGE7_RETRIEVAL_IN": str(RETRIEVAL_OUT),
                "STAGE7_ANALYZE_OUT": str(ANALYZE_OUT),
            },
        )

    # 3. PCA48 D_syn mixedlm (uses symlink for retrieval path)
    log("\n=== Step 3: PCA48 D_syn mixedlm ===")
    if SYN8B_OUT.exists():
        log(f"  {SYN8B_OUT} already exists, skipping")
    else:
        sym_path = SCRATCH / "stage7_retrieval_results.json"
        if not sym_path.exists():
            os.symlink(RETRIEVAL_OUT, sym_path)
            log(f"  symlinked {RETRIEVAL_OUT.name} → stage7_retrieval_results.json")
        ret = run(
            [PY, str(REPO_ROOT / "gaussian/syntax_subspace_stage7b_syn.py")],
            SCRATCH / "stage8_syn.log",
        )
        src_out = RESULT / "syntax_subspace_stage7b_syn.json"
        if src_out.exists() and not SYN8B_OUT.exists():
            os.rename(src_out, SYN8B_OUT)
            log(f"  moved {src_out.name} → {SYN8B_OUT.name}")

    # 4. 7C robustness
    log("\n=== Step 4: robustness check ===")
    if SYN8C_OUT.exists():
        log(f"  {SYN8C_OUT} already exists, skipping")
    else:
        # Need stage7b_syn.json symlink (was moved in step 3)
        sym_7b = RESULT / "syntax_subspace_stage7b_syn.json"
        if not sym_7b.exists():
            os.symlink(SYN8B_OUT, sym_7b)
            log(f"  symlinked {SYN8B_OUT.name} → syntax_subspace_stage7b_syn.json")
        # Ensure stage7 retrieval/regen symlinks (may have been removed by previous run)
        for src_name, dst_name in [
            ("stage7_retrieval_results.json", "stage7_retrieval_results.json"),
            ("stage7_regen.json", "stage7_regen.json"),
        ]:
            src = SCRATCH / ("stage8" + src_name[len("stage7"):])
            dst = SCRATCH / dst_name
            if not dst.exists() and src.exists():
                os.symlink(src, dst)
                log(f"  symlinked {src.name} → {dst_name}")
        ret = run(
            [PY, str(REPO_ROOT / "gaussian/syntax_subspace_stage7c_robust.py")],
            SCRATCH / "stage8_robust.log",
        )
        src_out = RESULT / "syntax_subspace_stage7c_robust.json"
        if src_out.exists() and not SYN8C_OUT.exists():
            os.rename(src_out, SYN8C_OUT)
            log(f"  moved {src_out.name} → {SYN8C_OUT.name}")

    # Summary
    log("\n=== Stage 8 artifacts ===")
    for p in [REGEN_OUT, RETRIEVAL_OUT, ANALYZE_OUT, SYN8B_OUT, SYN8C_OUT]:
        if p.exists():
            log(f"  ✓ {p} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            log(f"  ✗ MISSING {p}")


if __name__ == "__main__":
    main()