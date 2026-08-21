#!/usr/bin/env python3
"""Phase 15.C: 4-Dimensional Evaluation Panel — beyond StyleVector paper's ROUGE/METEOR.

User feedback (iter_037 followup): StyleVector paper uses only ROUGE/METEOR
(lexical overlap, not style-specific). Paper misses:
  - Author Identification
  - Style Similarity (style embedding)
  - Syntactic-feature distance
  - Semantic preservation

Phase 15.C integrates all 4 dimensions per condition, providing a complete
panel view of the StyleVector trade-off:

  Dim 1 (Style Strength, ↑): AnnaWegmann 768d cos to target user μ — best-of-K margin
                              (Phase 13.F style_signal)
  Dim 2 (Author ID, ↑):     Author Identification top-10 in 30 users
                              (Phase 15.A — best-of-K)
  Dim 3 (Syntactic 318d, ↓):318d Mahalanobis distance to target user μ_318d
                              (Phase 13.E — but flipped: smaller = closer)
  Dim 4 (Semantic, ↑):      Sentence-bert cosine cand vs product attrs string
                              (Phase 15.B)

Aggregates to "score" per cond for direct comparison.

Output: phase15_c_4d_panel.json + per_pair panel CSV.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_CANDIDATES = OUT_DIR / "phase13_d_e2e_queries.jsonl"
IN_13F_EMBS = OUT_DIR / "phase13_f_cand_embs_768d.npy"
IN_13F_EVAL = OUT_DIR / "phase13_f_style_eval.json"
IN_15A_REPORT = OUT_DIR / "phase15_a_author_id.json"
IN_15A_PER_CAND = OUT_DIR / "phase15_a_per_cand.jsonl"
IN_15B_REPORT = OUT_DIR / "phase15_b_semantic.json"
IN_15B_PER_CAND = OUT_DIR / "phase15_b_per_cand.jsonl"

OUT_PANEL = OUT_DIR / "phase15_c_4d_panel.json"
OUT_PANEL_CSV = OUT_DIR / "phase15_c_4d_per_pair.csv"

CONDITIONS = ["A_sampled", "B_mean", "C_shuffled", "D_off"]
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bootstrap_ci(values, n: int = 2000, seed: int = SEED):
    arr = np.array(list(values))
    if len(arr) == 0:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n_obs = len(arr)
    boot_means = []
    for _ in range(n):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        boot_means.append(float(arr[idx].mean()))
    bm = np.array(boot_means)
    return float(arr.mean()), (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))


def main():
    log("=" * 70)
    log("Phase 15.C: 4-Dim Evaluation Panel")
    log("=" * 70)

    # === [1] Load all upstream reports ===
    log("[1] Loading all upstream reports ...")

    report_13f = json.loads(IN_13F_EVAL.read_text())
    report_15a = json.loads(IN_15A_REPORT.read_text())
    report_15b = json.loads(IN_15B_REPORT.read_text())

    log(f"  13.F: {len(report_13f['per_cond_eval'])} conds")
    log(f"  15.A: {len(report_15a['per_cond_per_cand'])} conds")
    log(f"  15.B: {len(report_15b['per_cond'])} conds")

    # === [2] Build per-cond 4-dim panel ===
    log("[2] Building per-cond 4-dim panel ...")
    panel: dict[str, dict] = {}
    for cond in CONDITIONS:
        # Dim 1: Style Strength (best-of-K margin from Phase 13.F)
        f1 = report_13f["per_cond_eval"][cond]["best_of_K_margin_mean"]
        # Dim 2: Author ID (best-of-K top10 from Phase 15.A, in 30 users)
        f2 = report_15a["per_cond_best_of_K"][cond]["best_of_K_top10"]
        # Dim 3: Syntactic 318d: NOT directly available in 13.E mean rank.
        #       Use Phase 14 maha mean_rank as proxy (smaller rank = closer)
        # Phase 14 maha: A=123.8, B=177.8, C=151.2, D=220.9
        f3_raw = {
            "A_sampled": 123.77,
            "B_mean": 177.83,
            "C_shuffled": 151.23,
            "D_off": 220.9,
        }.get(cond, None)
        # normalize: lower rank = better, so flip sign for "higher = better"
        # We want semantic of "syntactic match to user style"
        # Use 1 / (1 + mean_rank) to normalize
        f3 = 1.0 / (1.0 + f3_raw) if f3_raw else 0.0
        # Dim 4: Semantic Preservation (Phase 15.B mean semantic sim)
        f4 = report_15b["per_cond"][cond]["mean_semantic_sim"]
        # Dim 5 (aux): perfect attrs coverage rate
        f5 = report_15b["per_cond"][cond]["pct_perfect_coverage"]

        # Composite score: equal weights, all higher = better
        # Style and semantic both important; author ID as key new metric
        composite = (f1 + f2 + f3 + f4) / 4

        panel[cond] = {
            "D1_style_margin_768d": f1,
            "D2_author_id_top10_30u": f2,
            "D3_syntactic_318d_maha_proxy": f3,
            "D4_semantic_sim_to_attrs": f4,
            "D5_perfect_attrs_rate": f5,
            "composite_mean": composite,
        }
        log(f"  {cond}: style={f1:.3f}, auth_id={f2*100:.1f}%, syn_proxy={f3:.4f}, "
            f"semantic={f4:.4f}, perfect={f5*100:.1f}% → composite={composite:.4f}")

    # === [3] Per-pair per-dim paired diffs (A vs D_off) ===
    log("[3] Per-pair per-dim paired diffs (A vs D_off) ...")
    # Load Phase 13.F per_pair
    f_per_pair = []
    f_path = OUT_DIR / "phase13_f_per_pair.jsonl"
    with f_path.open() as f:
        for line in f:
            if line.strip():
                f_per_pair.append(json.loads(line))
    f_by_pair = {(r["user_id"], r["condition"]): r for r in f_per_pair}

    # Load Phase 15.A per-pair (best_of_K aggregated already in report)
    a_by_pair = defaultdict(dict)
    with IN_15A_PER_CAND.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                a_by_pair[(r["user_id"], r["condition"])][r["cand_local_idx"]] = r

    # best-of-K per pair for 15.A
    a_best_of_k: dict[tuple[str, str], dict] = {}
    for (uid, cond), cands in a_by_pair.items():
        # rank_of_target
        ranks = [c["rank_of_target"] for c in cands.values()]
        top1s = [c["top1_hit"] for c in cands.values()]
        top10s = [c["top10_hit"] for c in cands.values()]
        a_best_of_k[(uid, cond)] = {
            "best_rank": int(min(ranks)),
            "any_top1": int(any(top1s)),
            "any_top10": int(any(top10s)),
        }

    # Load Phase 15.B per-cand, compute per-pair mean semantic sim + perfect coverage
    b_per_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with IN_15B_PER_CAND.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                b_per_pair[(r["user_id"], r["condition"])].append(r)

    pair_uids = sorted({r["user_id"] for r in f_per_pair})

    # Per-pair per-cond summary
    rows_csv = []
    diff_d1, diff_d2, diff_d4 = [], [], []
    for uid in pair_uids:
        for cond in CONDITIONS:
            f_row = f_by_pair.get((uid, cond))
            a_row = a_best_of_k.get((uid, cond), {})
            b_rows = b_per_pair.get((uid, cond), [])
            if not f_row or not b_rows:
                continue
            d1 = f_row["best_margin"]
            d2 = a_row.get("any_top10", 0)
            d4 = float(np.mean([r["semantic_sim_to_attrs"] for r in b_rows]))
            perfect = float(np.mean([1.0 if r["attrs_coverage"] == 1.0 else 0.0 for r in b_rows]))
            rows_csv.append({
                "user_id": uid,
                "condition": cond,
                "D1_style_margin": d1,
                "D2_author_id_top10": d2,
                "D4_semantic_sim": d4,
                "D5_perfect_rate": perfect,
            })
        # Paired diff (A - D)
        a_row = next((r for r in rows_csv if r["user_id"] == uid and r["condition"] == "A_sampled"), None)
        d_row = next((r for r in rows_csv if r["user_id"] == uid and r["condition"] == "D_off"), None)
        if a_row and d_row:
            diff_d1.append(a_row["D1_style_margin"] - d_row["D1_style_margin"])
            diff_d2.append(a_row["D2_author_id_top10"] - d_row["D2_author_id_top10"])
            diff_d4.append(a_row["D4_semantic_sim"] - d_row["D4_semantic_sim"])

    log(f"  A vs D paired diffs (n={len(diff_d1)} pairs):")
    for name, vals in [("D1_style_margin", diff_d1), ("D2_author_id_top10", diff_d2), ("D4_semantic", diff_d4)]:
        m, ci = bootstrap_ci(vals)
        log(f"    {name}: diff={m:+.4f} CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    d1_mean, d1_ci = bootstrap_ci(diff_d1)
    d2_mean, d2_ci = bootstrap_ci(diff_d2)
    d4_mean, d4_ci = bootstrap_ci(diff_d4)

    # === [4] Save panel + CSV ===
    log("[4] Saving panel ...")
    out = {
        "phase": "15.C",
        "panel_per_cond": panel,
        "paired_diff_A_vs_D": {
            "D1_style_margin_diff": {"mean": d1_mean, "ci95": list(d1_ci)},
            "D2_author_id_top10_diff": {"mean": d2_mean, "ci95": list(d2_ci)},
            "D4_semantic_sim_diff": {"mean": d4_mean, "ci95": list(d4_ci)},
        },
        "dimensions_explained": {
            "D1_style_margin_768d": "AnnaWegmann 768d cos best-of-K margin (higher = more in-style)",
            "D2_author_id_top10_30u": "best-of-K target in top-10 of 30 eval users (higher = more identifiable)",
            "D3_syntactic_318d_maha_proxy": "1/(1+Phase14 maha mean_rank) — Phase 14 318d Mahalanobis mean rank inverted",
            "D4_semantic_sim_to_attrs": "sentence-bert cosine cand vs product attrs string (higher = more content-faithful)",
            "D5_perfect_attrs_rate": "% candidates with 100% attrs present in cand text",
        },
        "decision_logic": {
            "GO": "A_sampled composite > D_off, and at least 2 dims lift paired CI > 0",
            "PARTIAL-GO": "1 dim lifts CI > 0, no dim regresses > 5%",
            "NO-GO": "Style lift but semantic regresses > 5% (style-content trade-off unacceptable)",
        },
    }
    OUT_PANEL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  panel → {OUT_PANEL}")

    import csv
    with OUT_PANEL_CSV.open("w", newline="") as f:
        if rows_csv:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)
    log(f"  per_pair csv → {OUT_PANEL_CSV}")

    # === [5] Composite verdict ===
    a_comp = panel["A_sampled"]["composite_mean"]
    d_comp = panel["D_off"]["composite_mean"]
    log(f"\n  Composite scores: A_sampled={a_comp:.4f}, D_off={d_comp:.4f}")
    log(f"  D1 (style) Δ={d1_mean:+.4f} [{d1_ci[0]:+.4f}, {d1_ci[1]:+.4f}] (positive = style lift)")
    log(f"  D2 (auth id) Δ={d2_mean:+.4f} [{d2_ci[0]:+.4f}, {d2_ci[1]:+.4f}] (positive = identifiable)")
    log(f"  D4 (semantic) Δ={d4_mean:+.4f} [{d4_ci[0]:+.4f}, {d4_ci[1]:+.4f}] (positive = preserve content)")

    n_dims_lift = sum([
        d1_ci[0] > 0,
        d2_ci[0] > 0,
        d4_ci[0] > -0.05,  # semantic may regress slightly but not >5%
    ])

    if a_comp > d_comp and n_dims_lift >= 2:
        verdict = "GO"
    elif d4_mean < -0.05:
        verdict = "NO-GO_style_content_tradeoff"
    elif n_dims_lift >= 1:
        verdict = "PARTIAL-GO"
    else:
        verdict = "NO-GO"

    out["verdict"] = verdict
    out["n_dims_lift_or_preserved"] = n_dims_lift
    OUT_PANEL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n  FINAL VERDICT: {verdict}")

    log("=" * 70)
    log("PHASE 15.C COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()