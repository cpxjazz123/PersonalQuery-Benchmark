#!/usr/bin/env python3
"""E10 — Objective style consistency check (R2.3, replaces §3.3 mock).

For each (user_id, asin) retained query:
1. Extract 20-dim syntactic feature vector (from 12_complexity features dict).
2. Compare with target user's historical feature distribution.
   - User center = mean of user's training features
   - User 95% range = mean ± 1.96*std per dim, then a query is "in range"
     if every dim falls within that band
3. Compute in-range ratio per query.
4. Negative control: for each retained query, sample a random non-user query
   (different user_id) and compute its Mahalanobis-like distance to the same
   user center; compare with retained-query distance.
5. Paired Wilcoxon test: retained < random?

Outputs:
- per-query records with distance + in_range flag
- per-category in-range ratio + Wilcoxon p-value
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

COMPLEX_DIR = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E10_style_consistency")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
FEATURE_KEYS = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]
assert len(FEATURE_KEYS) == 20, f"need 20 features, got {len(FEATURE_KEYS)}"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_records(category: str) -> List[Dict]:
    """Each record: user_id, asin, query_text, features (dict of 20 keys)."""
    p = COMPLEX_DIR / category / "strict5550_query_gmm_features.jsonl"
    out = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            feat = r.get("features", {})
            vec = [float(feat.get(k, 0.0)) for k in FEATURE_KEYS]
            out.append({
                "user_id": r["user_id"],
                "asin": r["asin"],
                "query_text": r["query_text"],
                "feature_vec": vec,
            })
    return out


def build_user_centers(records: List[Dict]) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """Per user: (mean_vec, std_vec, n_train).  std uses sample std (ddof=1) so
    that single-record users yield std=0 (handled in in_user_range)."""
    bucket: Dict[str, List[np.ndarray]] = defaultdict(list)
    for r in records:
        bucket[r["user_id"]].append(np.asarray(r["feature_vec"], dtype=np.float64))
    centers = {}
    for uid, vecs in bucket.items():
        m = np.asarray(vecs, dtype=np.float64)
        if len(vecs) >= 2:
            s = m.std(axis=0, ddof=1)
        else:
            s = np.zeros(m.shape[1], dtype=np.float64)
        centers[uid] = (m.mean(axis=0), s, len(vecs))
    return centers


def in_user_range(
    vec: np.ndarray, mean: np.ndarray, std: np.ndarray, n_train: int
) -> Tuple[bool, float]:
    """Returns (in_95_range, mean_normalized_distance).

    For dims where the user has zero variance (n_train<=1 or all identical values),
    we compare absolute difference to a tolerance of 1.0 instead of using
    the std normalization (avoids divide-by-zero blow-up).
    """
    # zero-variance dims: must match exactly within tolerance
    zero_mask = std < 1e-8
    if zero_mask.any():
        abs_diff_zero = np.abs(vec - mean)[zero_mask]
        z_zero = abs_diff_zero / 1.0  # tolerance = 1 unit
    else:
        z_zero = np.array([], dtype=np.float64)
    # nonzero-variance dims: z-score
    if (~zero_mask).any():
        z_nz = np.abs((vec - mean) / np.maximum(std, 1e-9))[~zero_mask]
    else:
        z_nz = np.array([], dtype=np.float64)
    z = np.concatenate([z_zero, z_nz])
    in_range = bool(np.all(z <= 1.96))
    mean_z = float(np.mean(z)) if z.size else 0.0
    return in_range, mean_z


def sample_negative(
    retained_user: str,
    all_records: List[Dict],
    rng: np.random.Generator,
) -> Dict:
    """Sample a random non-user query."""
    while True:
        idx = int(rng.integers(0, len(all_records)))
        if all_records[idx]["user_id"] != retained_user:
            return all_records[idx]


def process_category(category: str, seed: int = 42) -> Dict:
    log(f"\n=== {category} ===")
    records = load_records(category)
    log(f"  loaded {len(records)} records")
    centers = build_user_centers(records)
    log(f"  users={len(centers)}")

    rng = np.random.default_rng(seed)
    per_query = []
    retained_z = []
    negative_z = []
    for r in records:
        uid = r["user_id"]
        if uid not in centers:
            continue
        mean, std, n_train = centers[uid]
        vec = np.asarray(r["feature_vec"], dtype=np.float64)
        in_range, z = in_user_range(vec, mean, std, n_train)
        retained_z.append(z)
        # Sample one negative
        neg = sample_negative(uid, records, rng)
        neg_vec = np.asarray(neg["feature_vec"], dtype=np.float64)
        _, neg_z = in_user_range(neg_vec, mean, std, n_train)
        negative_z.append(neg_z)
        per_query.append({
            "user_id": uid,
            "asin": r["asin"],
            "mean_z_retained": z,
            "in_user_95pct_range": in_range,
            "mean_z_negative": neg_z,
            "negative_user_id": neg["user_id"],
        })
    log(f"  scored {len(per_query)} queries with negative controls")

    in_range_count = sum(1 for x in per_query if x["in_user_95pct_range"])
    in_range_ratio = in_range_count / len(per_query) if per_query else 0.0
    log(f"  in-range ratio = {in_range_count}/{len(per_query)} = {in_range_ratio:.4f}")

    # Paired Wilcoxon (retained z < negative z)
    from scipy.stats import wilcoxon
    if len(retained_z) >= 10:
        diff = np.asarray(negative_z) - np.asarray(retained_z)
        diff = diff[diff != 0]
        if len(diff) >= 10:
            stat, pval = wilcoxon(diff, alternative="greater")
        else:
            stat, pval = 0.0, 1.0
    else:
        stat, pval = 0.0, 1.0
    log(f"  Wilcoxon stat={stat:.2f} p={pval:.4e}")

    return {
        "category": category,
        "n_queries": len(per_query),
        "n_users": len(centers),
        "n_in_user_95pct_range": in_range_count,
        "in_range_ratio": in_range_ratio,
        "mean_z_retained": float(np.mean(retained_z)),
        "mean_z_negative": float(np.mean(negative_z)),
        "mean_diff": float(np.mean(np.asarray(negative_z) - np.asarray(retained_z))),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(pval),
        "per_query": per_query,
    }


def write_summary_md(sums: List[Dict]) -> str:
    rows = ["# E10 — Objective style consistency (20-dim syntactic features, R2.3)\n",
            "**Method** — per user compute mean/std of 20-dim syntactic features; for each retained query compute |z|=(vec-mean)/std across dims; *in user 95% range* = all dims |z|≤1.96; negative control = random non-user query, same user center; paired Wilcoxon (neg > retained) one-sided.\n",
            "| Category | n_queries | n_users | in_range_ratio | mean_z_retained | mean_z_negative | mean_diff | Wilcoxon p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in sums:
        rows.append(
            f"| {s['category']} | {s['n_queries']} | {s['n_users']} | "
            f"{s['in_range_ratio']:.4f} | {s['mean_z_retained']:.4f} | "
            f"{s['mean_z_negative']:.4f} | {s['mean_diff']:.4f} | "
            f"{s['wilcoxon_p']:.4e} |"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "summary_full.md"
    with open(p, "w") as f:
        f.write("\n".join(rows))
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sums = []
    for cat in args.categories:
        try:
            s = process_category(cat, seed=args.seed)
            # Drop per_query from saved JSON if too large; keep aggregate only
            save = {k: v for k, v in s.items() if k != "per_query"}
            with open(OUT_DIR / f"{cat}_style.json", "w") as f:
                json.dump(save, f, indent=2, default=str)
            with open(OUT_DIR / f"{cat}_per_query.jsonl", "w") as f:
                for q in s["per_query"]:
                    f.write(json.dumps(q, default=str) + "\n")
            sums.append(s)
        except Exception as e:
            log(f"ERROR {cat}: {e}")
            import traceback
            traceback.print_exc()
    if sums:
        p = write_summary_md(sums)
        log(f"wrote {p}")
    log("=== done ===")


if __name__ == "__main__":
    main()