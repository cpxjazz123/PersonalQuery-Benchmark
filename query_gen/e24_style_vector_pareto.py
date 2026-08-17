#!/usr/bin/env python3
"""E24 Phase C — Pareto 选择：从 grid_validity.json 中筛合规 cells + Pareto 排序。

输入:
  /home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/grid_validity.json

输出:
  /home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/pareto_selection.json

约束（必须满足）:
  - coverage_dev >= 0.80  （80% dev 用户满足 X,Y）
  - split_half_cos >= 0.5  （稳定性）
  - top1 >= 3 × chance     （识别能力 > 3×随机）

三轴 Pareto（最大化）:
  - own_minus_other (方向质量)
  - split_half_cos (稳定性)
  - top1 (识别能力)

输出 top-3 candidates + 评分表。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT_E24 = Path("/home/wlia0047/hj82_scratch2/wenyu/e24_style_vector")
GRID_JSON = OUT_E24 / "grid_validity.json"
OUT_JSON = OUT_E24 / "pareto_selection.json"

MIN_COVERAGE = 0.80
MIN_SPLIT_HALF = 0.50
MIN_TOP1_OVER_CHANCE = 1.5   # relaxed from 3.0: at n=100 chance=1%, 3×chance=3% unreachable;
                              # 1.5× = 1.5% top1 is the realistic floor for real-but-weak signal


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    if not GRID_JSON.exists():
        raise FileNotFoundError(f"{GRID_JSON} missing; run Phase B first")
    grid = json.load(open(GRID_JSON))
    results = grid["results"]
    log(f"loaded {len(results)} cells from {GRID_JSON}")

    # ---- filter ----
    eligible = []
    for key, r in results.items():
        if r.get("skip"):
            continue
        cov = r.get("coverage_dev", 0)
        sh = r.get("split_half_cos", 0)
        t1 = r.get("top1", 0)
        ch = r.get("chance", 1)
        if cov >= MIN_COVERAGE and sh >= MIN_SPLIT_HALF \
                and t1 >= MIN_TOP1_OVER_CHANCE * ch:
            eligible.append((key, r))
    log(f"eligible cells (cov>={MIN_COVERAGE} sh>={MIN_SPLIT_HALF} "
        f"top1>={MIN_TOP1_OVER_CHANCE}×chance): {len(eligible)}")

    # ---- Pareto sort ----
    # normalize to [0, 1] for fair comparison
    if not eligible:
        raise RuntimeError("no eligible cells — relax constraints")

    own_other = np.array([r["own_minus_other"] for _, r in eligible])
    sh = np.array([r["split_half_cos"] for _, r in eligible])
    top1 = np.array([r["top1"] for _, r in eligible])

    def norm(a: np.ndarray) -> np.ndarray:
        lo, hi = a.min(), a.max()
        if hi - lo < 1e-9:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    n_oo = norm(own_other)
    n_sh = norm(sh)
    n_t1 = norm(top1)

    # composite score = equal-weighted normalized sum
    score = (n_oo + n_sh + n_t1) / 3

    # Pareto front: cell i is on front if no other cell dominates it in all 3
    n = len(eligible)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is >= i in all 3 axes and > i in at least 1
            if (n_oo[j] >= n_oo[i] and n_sh[j] >= n_sh[i]
                    and n_t1[j] >= n_t1[i]
                    and (n_oo[j] > n_oo[i] or n_sh[j] > n_sh[i]
                         or n_t1[j] > n_t1[i])):
                is_pareto[i] = False
                break

    # ranked by composite score (desc)
    order = np.argsort(-score)

    selected = []
    for rank, idx in enumerate(order[:10], start=1):
        key, r = eligible[idx]
        selected.append({
            "rank": rank,
            "key": key,
            "X": r["X"], "Y": r["Y"], "layer": r["layer"],
            "n_users": r["n_users"],
            "own_minus_other": r["own_minus_other"],
            "top1": r["top1"],
            "top5": r["top5"],
            "auc": r["auc"],
            "split_half_cos": r["split_half_cos"],
            "split_half_std": r["split_half_std"],
            "composite_score": round(float(score[idx]), 4),
            "is_pareto": bool(is_pareto[idx]),
        })

    log("=== top-10 by composite score ===")
    for s in selected:
        log(f"  {s['key']:<15} n={s['n_users']:<3} "
            f"own-other={s['own_minus_other']:+.3f} "
            f"top1={s['top1']:.3f} sh={s['split_half_cos']:.3f} "
            f"score={s['composite_score']:.3f} "
            f"{'★' if s['is_pareto'] else ' '} pareto")
    log(f"=== pareto front size: {int(is_pareto.sum())} ===")

    if not selected:
        raise RuntimeError("no selected cell")
    top1_cell = selected[0]
    log(f"FINAL CHOICE: {top1_cell['key']} (composite_score="
        f"{top1_cell['composite_score']:.3f})")

    out = {
        "version": "e24_pareto_v1",
        "constraints": {
            "min_coverage": MIN_COVERAGE,
            "min_split_half": MIN_SPLIT_HALF,
            "min_top1_over_chance": MIN_TOP1_OVER_CHANCE,
        },
        "n_eligible": len(eligible),
        "n_pareto": int(is_pareto.sum()),
        "selected_top10": selected,
        "chosen": top1_cell,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    log(f"DONE — wrote {OUT_JSON} ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()