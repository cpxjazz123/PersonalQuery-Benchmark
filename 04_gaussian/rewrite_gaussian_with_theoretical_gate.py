"""重写 Stage 04 user_gaussian_stats: 写入两套理论 gate。

依据: 2026-09-15 用户决定 — 不依赖 val 句子的经验 percentile,
直接从 (μ, Σ) 高斯假设 + 卡方分布反推理论 gate_T。

Stage 08 (selection) 用 **高侧 inclusion** (q=0.95, χ²=26.30) — D² ≤ gate_T 视为"核心内"
Stage 10 (typo rejection) 用 **低侧 rejection** (q=0.05, χ²=7.96) — D² ≤ gate_T 视为"太内/拒入"

本脚本读已有 user_gaussian_stats.json (full + rank1),
对所有用户写入 d2_q{XX}_theoretical 字段 (q in [0.05, 0.50, 0.75, 0.95])。
cohort_gates[* ] 同时写 gate_T_high_theoretical (q=0.95, Stage 08 用)
和 gate_T_low_theoretical (q=0.05, Stage 10 用),不覆盖原 gate_T。

不重训 encoder, 不重算 Σ — 只加 theoretical 字段。
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from scipy.stats import chi2

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = REPO_ROOT / "result/04_gaussian"

THEORETICAL_QS = (0.05, 0.50, 0.75, 0.95)
HIGH_Q = 0.95
LOW_Q = 0.05


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rewrite(stats_path: Path, label: str, z_dim: int) -> None:
    log(f"=== rewrite {label}: {stats_path.name} ===")
    if not stats_path.exists():
        log(f"  missing, skip")
        return
    bak = stats_path.with_suffix(stats_path.suffix + ".pre_theoretical_gate")
    if not bak.exists():
        shutil.copy2(stats_path, bak)
        log(f"  backup → {bak.name}")
    d = json.loads(stats_path.read_text())
    cfg = d.get("config", {})
    cfg["theoretical_gate_quantiles"] = list(THEORETICAL_QS)
    cfg["theoretical_gate_values"] = {
        f"q{int(q*100):02d}": float(chi2.ppf(q, df=z_dim))
        for q in THEORETICAL_QS
    }
    cfg["gate_source_note"] = (
        f"2026-09-15: theoretical gate_T = chi2({z_dim}, q) for q in {THEORETICAL_QS} "
        "(Stage 08: high q=0.95 inclusion; Stage 10: low q=0.05 rejection; "
        "replaces empirical d2_q{XX} from val sentences)")
    high_val = float(chi2.ppf(HIGH_Q, df=z_dim))
    low_val = float(chi2.ppf(LOW_Q, df=z_dim))
    n_users = 0
    n_pairs = 0
    for uid, u in d.get("users", {}).items():
        if not isinstance(u, dict):
            continue
        for q in THEORETICAL_QS:
            u[f"d2_q{int(q*100):02d}_theoretical"] = float(chi2.ppf(q, df=z_dim))
        n_users += 1
    for asin, cohort in d.get("cohort_gates", {}).items():
        for uid, gate in cohort.items():
            if isinstance(gate, dict):
                gate["gate_T_high_theoretical"] = high_val  # Stage 08
                gate["gate_T_low_theoretical"] = low_val   # Stage 10
                n_pairs += 1
    d["config"] = cfg
    stats_path.write_text(json.dumps(d))
    log(f"  wrote {n_users} users × {len(THEORETICAL_QS)} quantiles + {n_pairs} cohort pairs")
    log(f"  cohort gate_T_high_theoretical = {high_val:.4f} (Stage 08)")
    log(f"  cohort gate_T_low_theoretical  = {low_val:.4f} (Stage 10)")
    log(f"  → {stats_path}")


def main() -> None:
    t0 = time.time()
    log("=== rewrite_gaussian_with_theoretical_gate ===")
    rewrite(OUT_DIR / "user_gaussian_stats.json", "full Σ schema", z_dim=16)
    rewrite(OUT_DIR / "user_gaussian_stats_rank1.json", "rank1+residual schema", z_dim=16)
    log(f"=== DONE ({time.time()-t0:.1f}s) ===")


if __name__ == "__main__":
    main()
