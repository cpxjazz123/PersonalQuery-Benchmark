#!/usr/bin/env python3
"""E9 — Second LLM judge via PersoanlQuery/llm_client.py (QwenLocal).

Uses llm_client.QwenLocalClient as independent judge (not the primary LLM that
generated the queries). Scores the same 50 pilot queries and computes Cohen's
κ + Spearman against the recorded primary LLM scores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E9_cross_validation")
ANNOT_PATH = REPO_ROOT / "result/personal_query/02_writing_analysis/llm_human_eval/llm_human_annotations_real.json"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_score(text: str) -> float:
    """Extract a 0-100 score from LLM response."""
    import re
    matches = re.findall(r"(\d+\.?\d*)", text)
    if not matches:
        return 50.0  # fallback middle
    for m in matches:
        try:
            v = float(m)
            if 0 <= v <= 100:
                return v
        except ValueError:
            continue
    return 50.0


def parse_label(score: float) -> str:
    if score >= 70:
        return "REL"
    elif score >= 40:
        return "PARTIAL"
    return "IRREL"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOT_PATH) as f:
        annotations = json.load(f)
    n = min(args.n, len(annotations))
    log(f"using {n} annotated samples")

    from llm_client import create_qwen_local_client
    client = create_qwen_local_client()
    log(f"loaded Qwen client, model={client.model_name}")

    rubric = """You are an independent quality judge. Rate the query on a 0-100 scale based on:
- Relevance: does the query express a clear product need?
- Naturalness: does it sound like a real user query?
- Specificity: does it mention concrete attributes (brand, category, price, size)?

Output ONLY a single number between 0 and 100 on the last line.
"""

    primary_scores = []
    second_scores = []
    labels = []
    for i, ann in enumerate(annotations[:n]):
        qid = ann.get("query_id")
        primary_scores.append(float(ann.get("llm_score", 50.0)))
        # Read the actual query text from 04_query if possible
        prompt = rubric + f"\nQuery: {ann.get('query_text', json.dumps(ann)[:300])}\nYour score (0-100):"
        try:
            response = client.call(prompt, max_tokens=args.max_tokens, temperature=0.0)
            score = parse_score(response)
        except Exception as e:
            log(f"  [{i}] {qid}: LLM error: {e}")
            score = 50.0
        second_scores.append(score)
        labels.append(parse_label(score))
        if (i + 1) % 10 == 0:
            log(f"  scored {i + 1}/{n}")

    # Compute Cohen's κ and Spearman
    from scipy.stats import spearmanr
    primary_labels = [ann.get("llm_label", parse_label(ann.get("llm_score", 50.0))) for ann in annotations[:n]]
    label_to_int = {"REL": 2, "PARTIAL": 1, "IRREL": 0}
    p_int = [label_to_int.get(l, 1) for l in primary_labels]
    s_int = [label_to_int.get(l, 1) for l in labels]
    rho, pval = spearmanr(p_int, s_int)
    # Cohen's κ
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(p_int, s_int)
    log(f"  Spearman ρ={rho:.4f} p={pval:.4e}  Cohen κ={kappa:.4f}")

    out = {
        "n": n,
        "primary_llm_model": annotations[0].get("llm_label", "primary"),  # original LLM
        "second_llm_model": client.model_name,
        "primary_scores": primary_scores,
        "second_scores": second_scores,
        "primary_labels": primary_labels,
        "second_labels": labels,
        "metrics": {
            "spearman_rho": float(rho),
            "spearman_p": float(pval),
            "cohens_kappa": float(kappa),
        },
    }
    out_path = OUT_DIR / "second_judge_scores.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"wrote {out_path}")

    # MD summary
    md = ["# E9 — Second LLM judge (Qwen local via PersoanlQuery/llm_client.py)\n",
          "**Method** — independent Qwen-7B judge (different from primary LLM that generated queries); scores same 50 pilot queries; Cohen κ + Spearman.\n",
          f"| Metric | Value |\n|---|---|\n| n | {n} |",
          f"| primary_llm | {out['primary_llm_model']} |",
          f"| second_llm | {out['second_llm_model']} |",
          f"| Spearman ρ | {rho:.4f} |",
          f"| Spearman p | {pval:.4e} |",
          f"| Cohen κ | {kappa:.4f} |",
          f"| mean |Δscore| | {sum(abs(a-b) for a,b in zip(primary_scores, second_scores))/n:.2f} |"]
    md_path = OUT_DIR / "summary_full.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    log(f"wrote {md_path}")
    log("=== done ===")


if __name__ == "__main__":
    main()