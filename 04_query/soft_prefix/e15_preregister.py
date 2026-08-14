#!/usr/bin/env python3
"""E15-A1/A2 — Pre-registration and vector contract (hardcoded).

Fixes BEFORE any held-out results are seen (per issue rules):
- model/checkpoint, candidate layers, four syntax axes, primary metric,
  primary contrast, SESOI, alpha, seeds, sample sizes, exclusion rules,
  multiple-comparison method.
- vector contract: same dimension/order/normalizer for train & inference;
  correct/shuffled/zero/global must not be overwritten downstream.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/e15")

PREREGISTER = {
    "model": "Qwen2.5-1.5B-Instruct (989aa7980e4cf806f80c7fef2b1adb7bc71aa306)",
    "checkpoint_commit": "f51af0a4",
    "axes": [
        {"name": "opener", "metric": "opener_class_match", "direction": "+ = more adverbial/personal opener"},
        {"name": "coordination", "metric": "coordination_count", "direction": "+ = more coordinating clauses"},
        {"name": "subordination", "metric": "advcl_count", "direction": "+ = more subordinate clauses"},
        {"name": "modifier_density", "metric": "modifier_density", "direction": "+ = denser modifiers"},
    ],
    "primary_metric": "standardized effect of axis feature (positive vs negative intervention, paired)",
    "primary_contrast": "alpha=+1 vs alpha=-1 on the same item (fixed prompt/decoding)",
    "sesoi": 0.3,
    "alpha_level": 0.05,
    "one_sided": True,
    "multiple_comparison": "bonferroni_across_axes (4 axes)",
    "seed": 42,
    "min_pairs_per_axis_train": 120,
    "min_pairs_per_axis_dev": 30,
    "min_pairs_per_axis_test": 40,
    "exclusion_rules": [
        "parser failure on either member of a pair",
        "attribute mismatch between pair members",
        "generation length < 5 tokens",
        "any fallback/degenerated output",
    ],
    "candidate_layers": "last token hidden state of all 28 layers (dev-selected)",
    "axis_extraction_methods": ["mean_difference", "logistic", "pca"],
    "intervention_alpha_values": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "selectivity_condition": "target axis effect significant AND off-target axes effect size < target",
    "stability_requirement": "same sign on >= 80% of held-out items per axis",
}

VECTOR_CONTRACT = {
    "vector_dim": 34,
    "components": {
        "feature_means": "20-dim clause-feature means (user reviews, sentence-level)",
        "opener_hist": "10-dim opener-class histogram (user review first words)",
        "style_channels": "4-dim style intensity (advmod/coordination/advcl/amod)",
    },
    "normalization": "per-dim train-only standardization; NO whole-vector L2 (E15-E1)",
    "train_inference_same": True,
    "controls": {
        "correct": "user's own vector",
        "shuffled": "seeded permutation of user vectors (no self-map)",
        "zero": "all-zero vector",
        "global": "train-set mean vector",
    },
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, obj in [("e15_preregister.json", PREREGISTER), ("vector_contract.json", VECTOR_CONTRACT)]:
        path = OUT_DIR / name
        path.write_text(json.dumps(obj, indent=2))
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}: hash={h}")


if __name__ == "__main__":
    main()
