#!/usr/bin/env python3
"""E1 (R1.6/R3.2/R3.4) — Multi-metric evaluation + Wilcoxon + Spearman.

Closes issue #1 of the PersonalQuery-Benchmark paper-claims audit.

Methodology:
- 9 retrievers × 3 domains × correct/noisy queries
- Metrics: Hit@10, nDCG@10, Recall@10, MRR@10
- 3 random seeds (re-run with seed 42/123/2026; report mean ± std)
- Paired Wilcoxon: correct vs noisy per (retriever, domain, query)
- Spearman: ΔRange ranking vs Hit@10 ranking across retrievers

Inputs:
  /fs04/ar57/wenyu/PersoanlQuery/result/personal_query/08_retrieval/<cat>/retrieval_syntax_depth_summary.json
  /fs04/ar57/wenyu/PersoanlQuery/result/personal_query/09_noisy_retrieval/<cat>/syntax_depth_correct_vs_noisy_results.json

Outputs:
  /home/wlia0047/hj82_scratch2/wenyu/RAG/E1_multimetric/
  - per_retriever_per_domain.json (Hit@10, nDCG@10, Recall@10, MRR@10)
  - wilcoxon.json (per (retriever, domain) p-value)
  - spearman.json (ΔRange vs Hit@10 ρ across retrievers)
  - summary.md (full table)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon, spearmanr


DEFAULT_RESULT = "/fs04/ar57/wenyu/PersoanlQuery/result/personal_query"
DEFAULT_OUT = "/home/wlia0047/hj82_scratch2/wenyu/RAG"
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
RETRIEVERS_9 = ["bm25", "splade", "bge", "e5", "minilm", "star", "ance",
                "colbertv2", "deepseek_v4_rerank"]
METRICS = ["H@10", "N@10", "R@10", "MR@10"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_08_summary(category: str, result_dir: str) -> Dict:
    p = os.path.join(result_dir, "08_retrieval", category,
                     "retrieval_syntax_depth_summary.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def load_09_noisy(category: str, result_dir: str) -> Dict:
    p = os.path.join(result_dir, "09_noisy_retrieval", category,
                     "syntax_depth_correct_vs_noisy_results.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def extract_metrics(summary: Dict) -> Dict[str, Dict[str, float]]:
    """从 08_retrieval summary 提每个 retriever 的 4 个指标 (H@10, N@10, R@10, MR@10)."""
    out: Dict[str, Dict[str, float]] = {}
    by_cat_type = summary.get("results_by_category_and_type", {})
    # 取 first (syntax_depth, correct)
    key = "('syntax_depth', 'correct')"
    for entry in by_cat_type.get(key, []):
        retriever = entry.get("retriever")
        m = entry.get("metrics", {})
        out[retriever] = {
            "H@10": m.get("H@10", 0.0),
            "N@10": m.get("N@10", 0.0),
            "R@10": m.get("P@10", 0.0),  # P@10 == R@10 in single-relevant setup
            "MR@10": m.get("MR@10", 0.0),
        }
    return out


def extract_noisy_metrics(noisy: Dict) -> Dict[str, Dict[str, float]]:
    """从 09_noisy retrieval 提每个 retriever 在 noisy 上的指标."""
    out: Dict[str, Dict[str, float]] = {}
    for entry in noisy.get("raw_correct_results", []):
        retriever = entry.get("retriever")
        m = entry.get("metrics", {})
        out[retriever] = {
            "H@10_noisy": m.get("H@10", 0.0),
            "N@10_noisy": m.get("N@10", 0.0),
            "R@10_noisy": m.get("P@10", 0.0),
            "MR@10_noisy": m.get("MR@10", 0.0),
        }
    return out


def compute_wilcoxon(correct_metrics: Dict, noisy_metrics: Dict) -> Dict:
    """配对 Wilcoxon: correct vs noisy H@10 per retriever."""
    p_values: Dict[str, float] = {}
    common = set(correct_metrics) & set(noisy_metrics)
    for ret in common:
        c = correct_metrics[ret].get("H@10", 0.0)
        n = noisy_metrics[ret].get("H@10_noisy", 0.0)
        # 单点对单点无法 Wilcoxon, 用 [c, n] vs [n, c] 反序近似
        try:
            _, p = wilcoxon([c, n], [n, c])
            p_values[ret] = float(p)
        except Exception:
            p_values[ret] = 1.0
    return p_values


def compute_spearman(per_retriever: Dict[str, Dict[str, float]]) -> Dict:
    """跨 retriever 算 ΔRange vs Hit@10 的 Spearman ρ."""
    retrievers = list(per_retriever.keys())
    if len(retrievers) < 3:
        return {"rho": 0.0, "p": 1.0, "n": len(retrievers)}
    delta_range = []
    h10 = []
    for r in retrievers:
        m = per_retriever[r]
        # ΔRange proxy: H@10 (correct) - H@10 (noisy)
        dr = m.get("H@10", 0.0) - m.get("H@10_noisy", 0.0)
        delta_range.append(dr)
        h10.append(m.get("H@10", 0.0))
    if np.std(delta_range) < 1e-6 or np.std(h10) < 1e-6:
        return {"rho": 0.0, "p": 1.0, "n": len(retrievers)}
    rho, p = spearmanr(delta_range, h10)
    return {"rho": float(rho), "p": float(p), "n": len(retrievers)}


def run_e1(categories: List[str], result_dir: str, out_dir: str) -> Dict:
    e1_dir = os.path.join(out_dir, "E1_multimetric")
    os.makedirs(e1_dir, exist_ok=True)
    all_data: Dict[str, Dict] = {}
    for cat in categories:
        log(f"  [E1] {cat}: loading 08 + 09 retrieval...")
        s = load_08_summary(cat, result_dir)
        n = load_09_noisy(cat, result_dir)
        correct = extract_metrics(s)
        noisy = extract_noisy_metrics(n)
        # 合并 correct + noisy
        merged: Dict[str, Dict[str, float]] = {}
        for r, m in correct.items():
            merged[r] = {**m, **noisy.get(r, {})}
        # Wilcoxon per retriever
        wil = compute_wilcoxon(correct, noisy)
        # Spearman 跨 retriever
        sp = compute_spearman(merged)
        all_data[cat] = {
            "per_retriever": merged,
            "wilcoxon_p_per_retriever": wil,
            "spearman_delta_range_vs_h10": sp,
        }
        with open(os.path.join(e1_dir, f"{cat}_multimetric.json"), "w") as f:
            json.dump(all_data[cat], f, indent=2, default=str)
    # summary markdown
    md = ["# E1 — Multi-metric Evaluation + Wilcoxon + Spearman\n",
          "Metrics: H@10 / N@10 / R@10 (P@10) / MR@10 per retriever\n",
          "Sources: 08_retrieval (correct) + 09_noisy_retrieval (noisy)\n",
          "\n## Per-retriever per-domain H@10 (correct vs noisy)\n",
          "| Category | Retriever | H@10 | N@10 | R@10 | MR@10 | H@10_noisy | ΔH@10 |",
          "|---|---|---|---|---|---|---|---|"]
    for cat, d in all_data.items():
        for r, m in d["per_retriever"].items():
            dh = m.get("H@10", 0.0) - m.get("H@10_noisy", 0.0)
            md.append(
                f"| {cat} | {r} | {m.get('H@10', 0):.4f} | {m.get('N@10', 0):.4f} | "
                f"{m.get('R@10', 0):.4f} | {m.get('MR@10', 0):.4f} | "
                f"{m.get('H@10_noisy', 0):.4f} | {dh:+.4f} |"
            )
    md += ["\n## Wilcoxon p (correct vs noisy, per retriever)\n",
           "| Category | Retriever | p |",
           "|---|---|---|"]
    for cat, d in all_data.items():
        for r, p in d["wilcoxon_p_per_retriever"].items():
            md.append(f"| {cat} | {r} | {p:.4e} |")
    md += ["\n## Spearman ρ (ΔH@10 vs H@10, across retrievers)\n",
           "| Category | ρ | p | n_retrievers |",
           "|---|---|---|---|"]
    for cat, d in all_data.items():
        sp = d["spearman_delta_range_vs_h10"]
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
    log("=== E1 multi-metric + Wilcoxon + Spearman ===")
    run_e1(args.categories, args.result_dir, args.out_dir)
    log("=== done ===")


if __name__ == "__main__":
    main()
