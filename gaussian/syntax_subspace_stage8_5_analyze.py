"""Stage 8.5: Analyze retrieval + MixedLM.

Compares three variants (selected / random / farthest) under BM25 vs MiniLM.

1. Per-variant summary: MRR, Hit@10, mean_rank
2. Paired tests: selected vs random, selected vs farthest
3. MixedLM: RR ~ C(retriever) + C(variant) + interactions + (1|asin)

Output: stage8_5_analyze.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8_5_analyze.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_analyze.log 2>&1 &
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, ttest_rel


# === Paths ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
PER_QUERY_IN = SCRATCH / "stage8_5_retrieval_per_query.json"
SELECTION_IN = SCRATCH / "stage8_5_selection.json"
ANALYZE_OUT = SCRATCH / "stage8_5_analyze.json"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    log("=== Stage 8.5: Retrieval Analysis ===")

    # === 1. Load ===
    log("\n=== 1. Loading per-query results ===")
    pq = json.load(open(PER_QUERY_IN))
    queries = pq["queries"]
    log(f"  {len(queries)} queries")

    sel = json.load(open(SELECTION_IN))
    sel_distances = {}  # (entry_idx, variant) → distance
    for i, e in enumerate(sel["entries"]):
        sel_distances[(i, "selected")] = e["selected_distance"]
        sel_distances[(i, "random")] = e["random_distance"]
        sel_distances[(i, "farthest")] = e["farthest_distance"]

    # === 2. Per-variant summary ===
    log("\n=== 2. Per-variant summary ===")
    by_variant = collections.defaultdict(lambda: {
        "bm25_RR": [], "minilm_RR": [], "bm25_hit10": [], "minilm_hit10": [],
        "bm25_rank": [], "minilm_rank": [], "entry_idx": [],
    })
    for q in queries:
        v = q["variant"]
        by_variant[v]["bm25_RR"].append(q["bm25_RR"])
        by_variant[v]["minilm_RR"].append(q["minilm_RR"])
        by_variant[v]["bm25_hit10"].append(q["bm25_hit10"])
        by_variant[v]["minilm_hit10"].append(q["minilm_hit10"])
        if q["bm25_rank"]:
            by_variant[v]["bm25_rank"].append(q["bm25_rank"])
        if q["minilm_rank"]:
            by_variant[v]["minilm_rank"].append(q["minilm_rank"])
        by_variant[v]["entry_idx"].append(q["entry_idx"])

    summary = {}
    for v in ("selected", "random", "farthest"):
        d = by_variant[v]
        summary[v] = {
            "n": len(d["bm25_RR"]),
            "bm25_MRR": float(np.mean(d["bm25_RR"])),
            "bm25_MRR_std": float(np.std(d["bm25_RR"])),
            "bm25_Hit@10": float(np.mean(d["bm25_hit10"])),
            "minilm_MRR": float(np.mean(d["minilm_RR"])),
            "minilm_MRR_std": float(np.std(d["minilm_RR"])),
            "minilm_Hit@10": float(np.mean(d["minilm_hit10"])),
            "bm25_mean_rank": float(np.mean(d["bm25_rank"])) if d["bm25_rank"] else None,
            "minilm_mean_rank": float(np.mean(d["minilm_rank"])) if d["minilm_rank"] else None,
        }
    for v in ("selected", "random", "farthest"):
        s = summary[v]
        bm25_rank_str = f"{s['bm25_mean_rank']:.1f}" if s["bm25_mean_rank"] else "n/a"
        minilm_rank_str = f"{s['minilm_mean_rank']:.1f}" if s["minilm_mean_rank"] else "n/a"
        log(f"  {v} (n={s['n']}):")
        log(f"    BM25:   MRR={s['bm25_MRR']:.4f}, Hit@10={s['bm25_Hit@10']*100:.1f}%, mean_rank={bm25_rank_str}")
        log(f"    MiniLM: MRR={s['minilm_MRR']:.4f}, Hit@10={s['minilm_Hit@10']*100:.1f}%, mean_rank={minilm_rank_str}")

    # === 3. Paired tests: selected vs random, selected vs farthest ===
    log("\n=== 3. Paired tests (per entry_idx) ===")

    # Build paired RR vectors: (entry_idx) → (selected_RR, random_RR, farthest_RR) per retriever
    by_entry = collections.defaultdict(lambda: {"selected": {}, "random": {}, "farthest": {}})
    for q in queries:
        ei = q["entry_idx"]
        v = q["variant"]
        by_entry[ei][v] = {
            "bm25_RR": q["bm25_RR"],
            "minilm_RR": q["minilm_RR"],
            "bm25_rank": q["bm25_rank"],
            "minilm_rank": q["minilm_rank"],
        }

    paired_tests = {}
    for retriever in ("bm25", "minilm"):
        metric = f"{retriever}_RR"
        sel_arr = []
        rnd_arr = []
        far_arr = []
        for ei, d in by_entry.items():
            if all(v in d for v in ("selected", "random", "farthest")):
                sel_arr.append(d["selected"][metric])
                rnd_arr.append(d["random"][metric])
                far_arr.append(d["farthest"][metric])
        sel_arr = np.array(sel_arr)
        rnd_arr = np.array(rnd_arr)
        far_arr = np.array(far_arr)

        w_sr, p_sr = wilcoxon(sel_arr, rnd_arr, alternative="greater")
        w_sf, p_sf = wilcoxon(sel_arr, far_arr, alternative="greater")
        w_rf, p_rf = wilcoxon(rnd_arr, far_arr, alternative="greater")

        log(f"  {retriever}:")
        log(f"    selected vs random: mean diff={(sel_arr - rnd_arr).mean():+.4f}, W={w_sr:.1f}, p={p_sr:.4g}")
        log(f"    selected vs farthest: mean diff={(sel_arr - far_arr).mean():+.4f}, W={w_sf:.1f}, p={p_sf:.4g}")
        log(f"    random vs farthest: mean diff={(rnd_arr - far_arr).mean():+.4f}, W={w_rf:.1f}, p={p_rf:.4g}")

        paired_tests[retriever] = {
            "n": len(sel_arr),
            "selected_vs_random": {
                "mean_diff": float((sel_arr - rnd_arr).mean()),
                "W": float(w_sr),
                "p_value": float(p_sr),
                "alternative": "greater",
            },
            "selected_vs_farthest": {
                "mean_diff": float((sel_arr - far_arr).mean()),
                "W": float(w_sf),
                "p_value": float(p_sf),
                "alternative": "greater",
            },
            "random_vs_farthest": {
                "mean_diff": float((rnd_arr - far_arr).mean()),
                "W": float(w_rf),
                "p_value": float(p_rf),
                "alternative": "greater",
            },
        }

    # === 4. Sub-group analysis: per-user-source (per_user vs fallback) ===
    log("\n=== 4. Per-user-source analysis ===")
    by_source_variant = collections.defaultdict(lambda: collections.defaultdict(lambda: {"bm25_RR": [], "minilm_RR": []}))
    for q in queries:
        s = q["user_source"]
        v = q["variant"]
        by_source_variant[s][v]["bm25_RR"].append(q["bm25_RR"])
        by_source_variant[s][v]["minilm_RR"].append(q["minilm_RR"])

    source_results = {}
    for s in by_source_variant:
        source_results[s] = {}
        for v in ("selected", "random", "farthest"):
            d = by_source_variant[s][v]
            if d["bm25_RR"]:
                source_results[s][v] = {
                    "n": len(d["bm25_RR"]),
                    "bm25_MRR": float(np.mean(d["bm25_RR"])),
                    "minilm_MRR": float(np.mean(d["minilm_RR"])),
                }
        # Log
        log(f"  {s}:")
        for v in ("selected", "random", "farthest"):
            if v in source_results[s]:
                log(f"    {v}: n={source_results[s][v]['n']}, "
                    f"BM25 MRR={source_results[s][v]['bm25_MRR']:.4f}, "
                    f"MiniLM MRR={source_results[s][v]['minilm_MRR']:.4f}")

    # === 5. Selected vs Random per-(asin, user) improvement ===
    log("\n=== 5. Per-(asin,user) improvement breakdown ===")
    improvement_by_asin = {}
    for ei, d in by_entry.items():
        asin = queries[[q["entry_idx"] for q in queries].index(ei)]["asin"]
        if asin not in improvement_by_asin:
            improvement_by_asin[asin] = {"sel_minus_rnd_bm25": [], "sel_minus_rnd_minilm": []}
        if all(v in d for v in ("selected", "random")):
            improvement_by_asin[asin]["sel_minus_rnd_bm25"].append(
                d["selected"]["bm25_RR"] - d["random"]["bm25_RR"]
            )
            improvement_by_asin[asin]["sel_minus_rnd_minilm"].append(
                d["selected"]["minilm_RR"] - d["random"]["minilm_RR"]
            )

    n_asin_positive_bm25 = sum(1 for v in improvement_by_asin.values() if np.mean(v["sel_minus_rnd_bm25"]) > 0)
    n_asin_positive_minilm = sum(1 for v in improvement_by_asin.values() if np.mean(v["sel_minus_rnd_minilm"]) > 0)
    log(f"  ASINs with selected > random (avg): BM25={n_asin_positive_bm25}/{len(improvement_by_asin)}, "
        f"MiniLM={n_asin_positive_minilm}/{len(improvement_by_asin)}")

    # === 6. Save ===
    log("\n=== 6. Saving ===")
    output = {
        "config": {"description": "Stage 8.5 retrieval analysis"},
        "summary_by_variant": summary,
        "paired_tests": paired_tests,
        "summary_by_user_source": source_results,
        "asin_level_positive_count": {
            "bm25": n_asin_positive_bm25,
            "minilm": n_asin_positive_minilm,
            "total": len(improvement_by_asin),
        },
    }
    with open(ANALYZE_OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {ANALYZE_OUT}")


if __name__ == "__main__":
    main()