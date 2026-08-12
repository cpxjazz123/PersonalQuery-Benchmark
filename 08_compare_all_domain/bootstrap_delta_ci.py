#!/usr/bin/env python3
"""[Reviewer-pilot] Bootstrap 95% CI for Table 1 Δ values (revised).

iter #77 originally claimed the per-query Δ data was missing — but it is
actually nested inside each raw_*_results[i].all_query_records list (one
record per matched pair_id). The Stage 7 eval pipeline retains it; my
iter #77 bootstrap script just looked at the wrong path.

This iter #78 script extracts per-query Δ from raw_*.all_query_records
and computes proper bootstrap 95% CIs for every (domain × retriever ×
metric) cell.

Input:  result/personal_query/09_noisy_retrieval/<cat>/
          syntax_depth_correct_vs_noisy_results.json

Output: result/personal_query/08_compare_all_domain/bootstrap_delta_ci.json
Prints table + confidence interval summary to stdout.

Bootstrap: B=10_000, percentile 95% CI on the per-cell mean Δ.

Per loop.md §1: no fallback / no default; missing data → raise or report
n=0 with deg_warning.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")
RESULT_ROOT = REPO_ROOT / "result" / "personal_query"
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
METRICS = ["P@1", "N@1", "MR@1", "H@1",
           "P@3", "N@3", "MR@3", "H@3",
           "P@5", "N@5", "MR@5", "H@5",
           "P@10", "N@10", "MR@10", "H@10"]
BOOTSTRAP_B = 10_000
RNG_SEED = 42
MIN_USABLE_N = 5  # < 5 → under-powered, only point estimate reported


def _percentile_ci(samples: List[float], alpha: float = 0.05) -> tuple[float, float]:
    s = sorted(samples)
    lo = max(0, int(round(len(s) * (alpha / 2))) - 1)
    hi = min(len(s) - 1, int(round(len(s) * (1 - alpha / 2))) - 1)
    return s[lo], s[hi]


def _bootstrap_mean_ci(values: List[float], b: int, rng: random.Random) -> Dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "ci_width": float("nan"),
                "deg_warning": "no_per_query_records"}
    if n == 1:
        v = values[0]
        return {"n": 1, "mean": v, "ci_low": v, "ci_high": v, "ci_width": 0.0,
                "deg_warning": "n=1 → degenerate CI"}
    boot_means = []
    for _ in range(b):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.mean(sample))
    lo, hi = _percentile_ci(boot_means)
    warn = f"n={n} → under-powered; recommend n>={MIN_USABLE_N}*3" if n < MIN_USABLE_N else None
    return {"n": n, "mean": float(statistics.mean(values)),
            "ci_low": float(lo), "ci_high": float(hi),
            "ci_width": float(hi - lo), "deg_warning": warn}


def _extract_per_query_deltas(category: str) -> Dict[str, Dict[str, List[float]]]:
    """Per-query Δ values indexed by retriever -> metric -> list[float].

    Pulls from raw_correct_results[i].all_query_records AND raw_noisy_results[i].all_query_records.
    """
    p = RESULT_ROOT / "09_noisy_retrieval" / category / "syntax_depth_correct_vs_noisy_results.json"
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    correct_raw = d.get("raw_correct_results", [])
    noisy_raw = d.get("raw_noisy_results", [])
    by_ret_metric: Dict[str, Dict[str, List[float]]] = {}
    for correct_entry, noisy_entry in zip(correct_raw, noisy_raw):
        if not (isinstance(correct_entry, dict) and isinstance(noisy_entry, dict)):
            continue
        retriever = correct_entry.get("retriever") or noisy_entry.get("retriever")
        if not retriever:
            continue
        c_records = correct_entry.get("all_query_records", [])
        n_records = noisy_entry.get("all_query_records", [])
        if len(c_records) != len(n_records):
            raise ValueError(
                f"{category}/{retriever}: correct {len(c_records)} vs noisy {len(n_records)} per-query records mismatch"
            )
        per_metric_deltas: Dict[str, List[float]] = {m: [] for m in METRICS}
        for c_rec, n_rec in zip(c_records, n_records):
            cm = c_rec.get("metrics", {})
            nm = n_rec.get("metrics", {})
            if not isinstance(cm, dict) or not isinstance(nm, dict):
                raise TypeError(f"{category}/{retriever}: per-query metrics missing on records")
            for m in METRICS:
                if m not in cm or m not in nm:
                    raise KeyError(f"{category}/{retriever}: metric {m} missing in per-query records")
                per_metric_deltas[m].append(float(nm[m] - cm[m]))
        by_ret_metric[retriever] = per_metric_deltas
    return by_ret_metric


def main():
    rng = random.Random(RNG_SEED)
    all_rows: List[Dict] = []
    summary: List[Dict] = []
    for cat in CATEGORIES:
        try:
            data = _extract_per_query_deltas(cat)
        except (FileNotFoundError, ValueError, TypeError, KeyError) as e:
            print(f"skip {cat}: {e}")
            continue
        for retriever in sorted(data):
            per_metric = data[retriever]
            row: Dict = {"category": cat, "retriever": retriever, "per_metric": {}}
            usable_metrics: List[str] = []
            for m in METRICS:
                values = per_metric.get(m, [])
                ci = _bootstrap_mean_ci(values, BOOTSTRAP_B, rng)
                row["per_metric"][m] = ci
                if ci["n"] >= MIN_USABLE_N and ci.get("deg_warning") is None:
                    usable_metrics.append(m)
            row["usable_metrics"] = usable_metrics
            all_rows.append(row)
            summary.append({
                "category": cat,
                "retriever": retriever,
                "n_queries": next(iter(per_metric.values())) and len(next(iter(per_metric.values()))) or 0,
                "n_usable_metrics": len(usable_metrics),
            })
    out_path = RESULT_ROOT / "08_compare_all_domain" / "bootstrap_delta_ci.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows, "summary": summary}, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {out_path}\n")

    print("=== Coverage summary (per category × retriever) ===")
    print(f"{'Category':<26} {'Retriever':<10} {'n':>4}  {'usable_metrics':>16}  (≥{MIN_USABLE_N} queries)")
    for s in summary:
        print(f"{s['category']:<26} {s['retriever']:<10} {s['n_queries']:>4}  {s['n_usable_metrics']:>16}")

    print("\n=== Bootstrap 95% CI for Table 1 Δ values (n≥5) ===")
    print(f"{'Domain':<26} {'Retriever':<10} {'metric':<8} {'Δ (point)':>10} {'CI low':>8} {'CI high':>8} {'n':>5} {'pass':>6}")
    printed = 0
    for row in all_rows:
        for m in METRICS:
            ci = row["per_metric"][m]
            if ci["n"] >= MIN_USABLE_N and ci.get("deg_warning") is None:
                # Simple "Δ significantly negative" if CI_high < 0
                sig_neg = "yes" if ci["ci_high"] < 0 else ("no" if ci["ci_low"] > 0 else "ns")
                print(f"{row['category']:<26} {row['retriever']:<10} {m:<8} "
                      f"{ci['mean']:>10.3f} {ci['ci_low']:>8.3f} {ci['ci_high']:>8.3f} "
                      f"{ci['n']:>5} {sig_neg:>6}")
                printed += 1
    if printed == 0:
        print("  (no usable (domain × retriever × metric) cell has n>=5)")

    print("\n=== Heads-up on under-powered cells (1<n<5) — point estimate only ===")
    for row in all_rows:
        for m in METRICS:
            ci = row["per_metric"][m]
            if 0 < ci["n"] < MIN_USABLE_N:
                print(f"  {row['category']} / {row['retriever']} / {m}: Δ={ci['mean']:+.3f} (n={ci['n']}, no CI)")


if __name__ == "__main__":
    main()
