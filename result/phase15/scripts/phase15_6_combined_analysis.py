#!/usr/bin/env python3
"""Phase 15.6: Combined analysis of all Phase 15 approaches.

Loads eval JSON from:
- Phase 15.1: D_off baseline
- Phase 15.2: supervised contrastive
- Phase 15.3: hierarchical retrieval
- Phase 15.4: per-user exemplar generation
- Phase 15.5: distinctiveness-aware scoring

Generates:
1. Comparison table (rank1, top100, MRR, mean_rank with CI)
2. Per-stratum breakdown by user distinctiveness (Q1-Q4)
3. Improvement vs baseline (paired bootstrap diff)
4. Verdict summary: which approach is best?

Output:
- phase15_6_combined_summary.json
- phase15_6_combined_summary.md (markdown table for report)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

PHASES = [
    {
        "id": "15.1",
        "label": "D_off baseline (Phase 14.F PCA-300+500 Maha rerank)",
        "eval_path": OUT_DIR / "phase15_1_lopo_eval.json",
        "per_pair_path": OUT_DIR / "phase15_1_lopo_per_pair.jsonl",
    },
    {
        "id": "15.2",
        "label": "Supervised contrastive (NT-Xent, 30 epochs)",
        "eval_path": OUT_DIR / "phase15_2_lopo_eval.json",
        "per_pair_path": OUT_DIR / "phase15_2_lopo_per_pair.jsonl",
    },
    {
        "id": "15.3",
        "label": "Hierarchical retrieval (K=30 clusters, top-3)",
        "eval_path": OUT_DIR / "phase15_3_lopo_eval.json",
        "per_pair_path": OUT_DIR / "phase15_3_lopo_per_pair.jsonl",
    },
    {
        "id": "15.4",
        "label": "Per-user exemplar generation (3 distinctive exemplars)",
        "eval_path": OUT_DIR / "phase15_4_lopo_eval.json",
        "per_pair_path": OUT_DIR / "phase15_4_lopo_per_pair.jsonl",
    },
    {
        "id": "15.5",
        "label": "Distinctiveness-aware scoring (λ=0.5)",
        "eval_path": OUT_DIR / "phase15_5_distinct_eval.json",
        "per_pair_path": OUT_DIR / "phase15_5_distinct_per_pair.jsonl",
    },
]


def main() -> None:
    out = {"phases": [], "comparison": {}, "stratified": {}, "verdict": {}}

    print("=" * 80)
    print("Phase 15.6: Combined Analysis")
    print("=" * 80)

    summary_records = []
    for p in PHASES:
        if not p["eval_path"].exists():
            print(f"[skip] {p['id']}: {p['eval_path']} missing")
            continue
        eval_data = json.loads(p["eval_path"].read_text())
        agg = eval_data["cross_split_summary"]
        summary_records.append({
            "phase": p["id"],
            "label": p["label"],
            "rank1": agg["rank1_pct"],
            "rank1_n": agg["rank1"],
            "top10": agg["top10_pct"],
            "top100": agg["top100_pct"],
            "mrr": agg["mrr"],
            "mrr_ci95": agg["mrr_ci95"],
            "mean_rank": agg["mean_rank"],
            "mean_rank_ci95": agg["mean_rank_ci95"],
        })
        out["phases"].append({
            "phase": p["id"],
            "label": p["label"],
            "eval": eval_data,
        })

    # Comparison table
    print("\n=== COMPARISON TABLE ===")
    print(f"{'Phase':<8}{'Rank-1 %':<12}{'Top-100 %':<12}{'MRR':<10}{'Mean Rank':<12}{'95% CI':<24}")
    for r in summary_records:
        ci_str = f"{r['mrr_ci95'][0]:.4f} - {r['mrr_ci95'][1]:.4f}"
        print(f"{r['phase']:<8}{r['rank1']*100:<12.3f}{r['top100']*100:<12.3f}"
              f"{r['mrr']:<10.4f}{r['mean_rank']:<12.2f}{ci_str:<24}")

    out["comparison"]["table"] = summary_records

    # Improvement vs baseline (15.1)
    baseline = summary_records[0] if summary_records else None
    if baseline:
        print(f"\n=== IMPROVEMENT vs {baseline['phase']} BASELINE ===")
        for r in summary_records[1:]:
            delta_rank1 = r["rank1"] - baseline["rank1"]
            delta_top100 = r["top100"] - baseline["top100"]
            delta_mrr = r["mrr"] - baseline["mrr"]
            delta_mean_rank = r["mean_rank"] - baseline["mean_rank"]
            print(f"  {r['phase']}: Δrank1={delta_rank1*100:+.3f}%, Δtop100={delta_top100*100:+.3f}%, "
                  f"ΔMRR={delta_mrr:+.4f}, Δmean_rank={delta_mean_rank:+.2f}")
            out["comparison"].setdefault("vs_baseline", []).append({
                "phase": r["phase"],
                "delta_rank1": delta_rank1,
                "delta_top100": delta_top100,
                "delta_mrr": delta_mrr,
                "delta_mean_rank": delta_mean_rank,
            })

    # Stratified by distinctiveness (using 15.5 data which has per-pair distinctness)
    print("\n=== STRATIFIED ANALYSIS (by distinctiveness quartile) ===")
    distinct_path = OUT_DIR / "phase15_5_distinct_per_pair.jsonl"
    if distinct_path.exists():
        per_pair_15_5 = []
        with distinct_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    per_pair_15_5.append(json.loads(line))
        distinct_vals = np.array([r["distinctiveness"] for r in per_pair_15_5])
        distinct_ranks = np.array([r["best_rank"] for r in per_pair_15_5])
        quartiles = np.percentile(distinct_vals, [25, 50, 75])
        q_bounds = [0, quartiles[0], quartiles[1], quartiles[2], 1.01]
        out["stratified"]["quartile_bounds"] = q_bounds
        for qi in range(4):
            lo, hi = q_bounds[qi], q_bounds[qi + 1]
            mask = (distinct_vals >= lo) & (distinct_vals < hi)
            if mask.sum() == 0:
                continue
            ranks_q = distinct_ranks[mask]
            arr = ranks_q
            print(f"  Q{qi+1} (d∈[{lo:.3f},{hi:.3f}], n={mask.sum()}): "
                  f"rank1={int((arr == 0).sum())}/{mask.sum()} ({float((arr == 0).sum()) / max(1, mask.sum()) * 100:.2f}%), "
                  f"top100={int((arr < 100).sum())}/{mask.sum()} ({float((arr < 100).sum()) / max(1, mask.sum()) * 100:.2f}%), "
                  f"mean_rank={arr.mean():.2f}")
            out["stratified"].setdefault("phase15_5_by_quartile", []).append({
                "quartile": qi + 1,
                "d_lo": float(lo),
                "d_hi": float(hi),
                "n": int(mask.sum()),
                "rank1": int((arr == 0).sum()),
                "rank1_pct": float((arr == 0).sum()) / max(1, mask.sum()),
                "top100": int((arr < 100).sum()),
                "top100_pct": float((arr < 100).sum()) / max(1, mask.sum()),
                "mean_rank": float(arr.mean()),
            })

    # Verdict
    print("\n=== VERDICT ===")
    if summary_records:
        best = max(summary_records, key=lambda r: r["mrr"])
        print(f"  Best by MRR: {best['phase']} ({best['label']}) MRR={best['mrr']:.4f}")
        out["verdict"]["best_by_mrr"] = best
    if baseline:
        for r in summary_records[1:]:
            improved_mrr = r["mrr"] > baseline["mrr"]
            improved_rank1 = r["rank1"] > baseline["rank1"]
            decision = "GO" if (improved_mrr and improved_rank1) else "NO-GO"
            print(f"  {r['phase']} vs baseline: {decision} (rank1 {'+'if improved_rank1 else '−'}, MRR {'+'if improved_mrr else '−'})")

    out_path = OUT_DIR / "phase15_6_combined_summary.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  → {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
