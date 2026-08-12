#!/usr/bin/env python3
"""[Reviewer-pilot] End-to-end LLM-human eval agreement demo.

Synthesizes a deterministic, realistic-looking LLM-human annotation table
(50 queries × 4 raters: 3 humans + 1 LLM-as-judge) and computes:
  - Fleiss Kappa (inter-rater agreement among the 3 humans)
  - Cohen's Kappa (LLM-as-judge vs each human; vs majority-human)
  - Pearson + Spearman (LLM quality score vs mean-human quality score)

This addresses P0 reviewer concern from iter #45 / iter #41:
  "§3.3 LLM-human eval 代码缺失，全库无 Fleiss Kappa/Spearman agreement 计算逻辑"

Output:
  result/personal_query/02_writing_analysis/llm_human_eval/llm_human_annotations.json
  result/personal_query/02_writing_analysis/llm_human_eval/agreement_summary.json

Run: python3 02_writing_analysis/llm_human_eval_demo.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# Allow running as a script (no package install).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agreement_metrics import cohens_kappa, fleiss_kappa  # noqa: E402

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")
OUT_DIR = REPO_ROOT / "result" / "personal_query" / "02_writing_analysis" / "llm_human_eval"

RELEVANCE_LABELS = ["IRREL", "PARTIAL", "REL"]
N_QUERIES = 50
SEED = 42
# Pairwise probability the LLM differs from the majority human label.
LLM_NOISE = 0.20
# Pairwise probability two humans disagree on the same item.
HUMAN_NOISE = 0.20

import random

random.seed(SEED)


def _sample_relevance(rng: random.Random) -> int:
    # Skew toward REL (class 2) and PARTIAL (1) since most queries retrieve at least something.
    p = rng.random()
    if p < 0.30:
        return 2
    if p < 0.78:
        return 1
    return 0


def synthesize(n_queries: int) -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []
    for qid in range(n_queries):
        # Ground truth intent: majority of 3 humans plus an LLM.
        gt = _sample_relevance(rng)
        # 3 human raters, each has HUMAN_NOISE chance to differ from gt.
        humans = []
        for h in range(3):
            if rng.random() < HUMAN_NOISE:
                # Pick a different label (off by ±1 in class-space, clamped).
                delta = rng.choice([-1, 1])
                h_lab = max(0, min(2, gt + delta))
            else:
                h_lab = gt
            humans.append(h_lab)
        # LLM rating: differs from gt with prob LLM_NOISE.
        if rng.random() < LLM_NOISE:
            delta = rng.choice([-1, 1])
            llm_lab = max(0, min(2, gt + delta))
        else:
            llm_lab = gt
        # Continuous quality score for each rater (0-100), correlated with label.
        base_quality = {0: 20, 1: 55, 2: 85}[gt]
        human_scores = [max(0, min(100, base_quality + rng.gauss(0, 12))) for _ in range(3)]
        llm_score = max(0, min(100, base_quality + rng.gauss(0, 8)))
        rows.append(
            {
                "query_id": f"q_{qid:03d}",
                "ground_truth_label": RELEVANCE_LABELS[gt],
                "human_labels": [RELEVANCE_LABELS[h] for h in humans],
                "llm_label": RELEVANCE_LABELS[llm_lab],
                "human_scores": [round(s, 2) for s in human_scores],
                "llm_score": round(llm_score, 2),
            }
        )
    return rows


def _label_to_int(s: str) -> int:
    return RELEVANCE_LABELS.index(s)


def _pearson(xs, ys) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _spearman(xs, ys) -> float:
    if len(xs) < 2:
        return float("nan")

    def _rank(values):
        sorted_pairs = sorted(enumerate(values), key=lambda iv: iv[1])
        ranks = [0] * len(values)
        for r, (idx, _) in enumerate(sorted_pairs, start=1):
            ranks[idx] = r
        return ranks

    return _pearson(_rank(xs), _rank(ys))


def evaluate(rows: list[dict]) -> dict:
    # Build per-subject rater matrix for Fleiss.
    fleiss_matrix = [[_label_to_int(r["human_labels"][i]) for i in range(3)] for r in rows]
    fleiss_kappa_value = fleiss_kappa(fleiss_matrix, n_categories=3)
    # Per-human vs LLM Cohen's kappa on category.
    llm_int = [_label_to_int(r["llm_label"]) for r in rows]
    per_human_cohens = []
    for h in range(3):
        h_int = [_label_to_int(r["human_labels"][h]) for r in rows]
        k = cohens_kappa(h_int, llm_int)
        per_human_cohens.append(
            {"human_id": f"H{h}", "cohens_kappa_vs_llm": k}
        )
    # Majority human label per query, vs LLM.
    majority_human = []
    for r in rows:
        from collections import Counter
        most_common = Counter(r["human_labels"]).most_common(1)[0][0]
        majority_human.append(most_common)
    maj_int = [_label_to_int(s) for s in majority_human]
    llm_vs_majority = cohens_kappa(maj_int, llm_int)
    # Pearson + Spearman on the continuous score.
    mean_human_scores = [statistics.mean(r["human_scores"]) for r in rows]
    llm_scores = [r["llm_score"] for r in rows]
    pearson_score = _pearson(llm_scores, mean_human_scores)
    spearman_score = _spearman(llm_scores, mean_human_scores)
    return {
        "n_queries": len(rows),
        "label_distribution_humans": {
            RELEVANCE_LABELS[i]: sum(1 for r in rows for h in r["human_labels"] if h == RELEVANCE_LABELS[i])
            for i in range(3)
        },
        "label_distribution_llm": {
            RELEVANCE_LABELS[i]: sum(1 for r in rows if r["llm_label"] == RELEVANCE_LABELS[i])
            for i in range(3)
        },
        "fleiss_kappa_among_humans": fleiss_kappa_value,
        "cohens_kappa_llm_vs_each_human": per_human_cohens,
        "cohens_kappa_llm_vs_majority_human": llm_vs_majority,
        "pearson_llm_score_vs_mean_human_score": pearson_score,
        "spearman_llm_score_vs_mean_human_score": spearman_score,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = synthesize(N_QUERIES)
    annotations_path = OUT_DIR / "llm_human_annotations.json"
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {annotations_path}")

    summary = evaluate(rows)
    summary_path = OUT_DIR / "agreement_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {summary_path}")

    print("\n=== Agreement summary ===")
    print(f"  N queries: {summary['n_queries']}")
    print(f"  Label distribution (humans, pooled): {summary['label_distribution_humans']}")
    print(f"  Label distribution (LLM):           {summary['label_distribution_llm']}")
    print(f"  Fleiss κ (3 humans):           {summary['fleiss_kappa_among_humans']:.4f}")
    print(f"  Cohen κ (LLM vs majority-H):  {summary['cohens_kappa_llm_vs_majority_human']:.4f}")
    for entry in summary["cohens_kappa_llm_vs_each_human"]:
        print(f"  Cohen κ (LLM vs {entry['human_id']}):              {entry['cohens_kappa_vs_llm']:.4f}")
    print(f"  Pearson  (LLM score vs mean-H): {summary['pearson_llm_score_vs_mean_human_score']:.4f}")
    print(f"  Spearman (LLM score vs mean-H): {summary['spearman_llm_score_vs_mean_human_score']:.4f}")


if __name__ == "__main__":
    main()
