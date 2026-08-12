#!/usr/bin/env python3
"""E1 (R1.6/R3.2/R3.4) — Multi-metric evaluation + Wilcoxon + Spearman, full quantity.

Reads 09_noisy_retrieval per-query records (correct vs noisy), computes:
- Per-retriever per-domain metrics (H@10, N@10, P@10=R@10, MR@10) on correct vs noisy
- Paired Wilcoxon per (retriever, domain, metric)
- Spearman rho (ΔH@10 across retrievers vs aggregate H@10) per domain

Coverage: 3 domains × retrievers that have per-query records in 09_noisy_retrieval.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon, spearmanr


DEFAULT_RESULT = "/fs04/ar57/wenyu/PersoanlQuery/result/personal_query"
DEFAULT_OUT = "/home/wlia0047/hj82_scratch2/wenyu/RAG"
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
METRICS = ["H@10", "N@10", "MR@10", "P@10"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_09(category: str, result_dir: str) -> Dict:
    p = os.path.join(result_dir, "09_noisy_retrieval", category,
                     "syntax_depth_correct_vs_noisy_results.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def extract_per_query(noise_data: Dict) -> Dict[str, Dict[str, List[float]]]:
    """对每个 retriever, 提取 per-query H@10 (correct) 和 H@10 (noisy) 列表."""
    out: Dict[str, Dict[str, List[float]]] = {}
    correct_map = {r.get("retriever"): r for r in noise_data.get("raw_correct_results", [])}
    noisy_map = {r.get("retriever"): r for r in noise_data.get("raw_noisy_results", [])}
    common = set(correct_map) & set(noisy_map)
    for r in common:
        correct_records = correct_map[r].get("all_query_records", [])
        noisy_records = noisy_map[r].get("all_query_records", [])
        if len(correct_records) != len(noisy_records):
            log(f"  WARN {r}: correct/noisy records mismatch ({len(correct_records)} vs {len(noisy_records)})")
            continue
        correct_metrics = {m: [] for m in METRICS}
        noisy_metrics = {m: [] for m in METRICS}
        for cr, nr in zip(correct_records, noisy_records):
            cm = cr.get("metrics", {})
            nm = nr.get("metrics", {})
            for m in METRICS:
                if m in cm:
                    correct_metrics[m].append(float(cm[m]))
                if m in nm:
                    noisy_metrics[m].append(float(nm[m]))
        out[r] = {**{f"{m}_correct": correct_metrics[m] for m in METRICS},
                  **{f"{m}_noisy": noisy_metrics[m] for m in METRICS}}
    return out


def compute_per_retriever_metrics(per_query: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """对每个 retriever 算 correct/noisy 各 metric 的 mean."""
    out: Dict[str, Dict[str, float]] = {}
    for r, d in per_query.items():
        out[r] = {}
        for k, vs in d.items():
            out[r][k] = float(np.mean(vs)) if vs else 0.0
    return out


def compute_wilcoxon_per_retriever(per_query: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """对每个 retriever + 每个 metric 做配对 Wilcoxon (correct vs noisy)."""
    out: Dict[str, Dict[str, float]] = {}
    for r, d in per_query.items():
        out[r] = {}
        for m in METRICS:
            c = np.asarray(d.get(f"{m}_correct", []), dtype=np.float64)
            n = np.asarray(d.get(f"{m}_noisy", []), dtype=np.float64)
            if len(c) < 2 or len(c) != len(n):
                out[r][m] = {"stat": float("nan"), "p": float("nan"), "n": int(len(c))}
                continue
            try:
                stat, p = wilcoxon(c, n)
                out[r][m] = {"stat": float(stat), "p": float(p), "n": int(len(c))}
            except ValueError as e:
                out[r][m] = {"stat": float("nan"), "p": float("nan"), "n": int(len(c)), "err": str(e)}
    return out


def compute_spearman_delta_vs_h10(metrics: Dict[str, Dict[str, float]]) -> Dict:
    """跨 retriever 算 Spearman(ΔH@10, H@10_correct)."""
    retrievers = list(metrics.keys())
    if len(retrievers) < 3:
        return {"rho": 0.0, "p": 1.0, "n": len(retrievers)}
    delta = []
    h10 = []
    for r in retrievers:
        m = metrics[r]
        dh = m.get("H@10_correct", 0.0) - m.get("H@10_noisy", 0.0)
        delta.append(dh)
        h10.append(m.get("H@10_correct", 0.0))
    if np.std(delta) < 1e-6 or np.std(h10) < 1e-6:
        return {"rho": 0.0, "p": 1.0, "n": len(retrievers)}
    rho, p = spearmanr(delta, h10)
    return {"rho": float(rho), "p": float(p), "n": len(retrievers)}


def run_e1(categories: List[str], result_dir: str, out_dir: str) -> Dict:
    e1_dir = os.path.join(out_dir, "E1_multimetric")
    os.makedirs(e1_dir, exist_ok=True)
    all_data: Dict[str, Dict] = {}
    summary_rows: List[List[str]] = []

    for cat in categories:
        log(f"  [E1] {cat}: loading 09_noisy_retrieval...")
        nd = load_09(cat, result_dir)
        if not nd:
            log(f"  [E1] {cat}: skip (no data)")
            continue
        per_query = extract_per_query(nd)
        log(f"  [E1] {cat}: per-retriever per-query data for {sorted(per_query.keys())}")
        metrics = compute_per_retriever_metrics(per_query)
        wil = compute_wilcoxon_per_retriever(per_query)
        sp = compute_spearman_delta_vs_h10(metrics)
        all_data[cat] = {
            "retrievers_covered": sorted(per_query.keys()),
            "n_retrievers": len(per_query),
            "per_retriever_metrics": metrics,
            "wilcoxon_per_retriever": wil,
            "spearman_delta_h10_vs_h10_correct": sp,
        }
        with open(os.path.join(e1_dir, f"{cat}_multimetric.json"), "w") as f:
            json.dump(all_data[cat], f, indent=2, default=str)
        for r, w in wil.items():
            for m in METRICS:
                d = w[m]
                if "p" in d and not np.isnan(d["p"]):
                    summary_rows.append([cat, r, m, str(d["n"]), f"{d['stat']:.4e}", f"{d['p']:.4e}"])

    # write markdown summary
    md = ["# E1 — Multi-metric Evaluation + Wilcoxon + Spearman (full quantity)\n",
          "Inputs: 09_noisy_retrieval per-query records (correct vs noisy)\n",
          "Method: paired Wilcoxon per (retriever, metric); Spearman across retrievers\n",
          "\n## Per-domain retriever coverage\n",
          "| Category | Retrievers | n |\n",
          "|---|---|---|"]
    for cat, d in all_data.items():
        md.append(f"| {cat} | {', '.join(d['retrievers_covered'])} | {d['n_retrievers']} |")
    md += ["\n## Paired Wilcoxon (correct vs noisy) per (retriever, metric)\n",
           "| Category | Retriever | Metric | n_pairs | stat | p |\n",
           "|---|---|---|---|---|---|"]
    md += [f"| {' | '.join(r)} |" for r in summary_rows]
    md += ["\n## Spearman rho (ΔH@10 vs H@10_correct, across retrievers)\n",
           "| Category | rho | p | n_retrievers |\n",
           "|---|---|---|---|"]
    for cat, d in all_data.items():
        sp = d["spearman_delta_h10_vs_h10_correct"]
        md.append(f"| {cat} | {sp['rho']:.4f} | {sp['p']:.4e} | {sp['n']} |")
    summary_path = os.path.join(e1_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(md))
    log(f"  [E1] wrote {summary_path}")
    return all_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default=DEFAULT_RESULT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = ap.parse_args()
    log("=== E1 multi-metric + Wilcoxon + Spearman (full quantity) ===")
    run_e1(args.categories, args.result_dir, args.out_dir)
    log("=== done ===")


if __name__ == "__main__":
    main()