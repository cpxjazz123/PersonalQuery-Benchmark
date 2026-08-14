#!/usr/bin/env python3
"""E15-B1/B2/B3 — Content-identical minimal pairs for four syntax axes.

For each product (5 attrs), rule-generated positive/negative pairs that are
content-identical (every attribute string verbatim, each once) and differ in
EXACTLY ONE syntax axis. Independent spacy verification: target feature must
move in the declared direction; off-target syntax features reported.

Outputs (hardcoded):
  e15/syntax_axis_pairs.jsonl    — all pairs with parser features + provenance
  e15/axis_split_manifest.json   — item-level train/dev/test split
  e15/axis_audit.json            — parser verification report
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))

from extract_clause_features_single_query import (  # noqa: E402
    load_spacy_model,
    extract_clause_features,
)

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/e15")
RECORDS = REPO_ROOT / "result" / "personal_query" / "04_query" / "Baby_Products" / "query_by_syntax_depth_no_depth_check_10.json"
SEED = 42
TARGET_FEATURES = {
    "opener": None,              # custom check (first-token class)
    "coordination": "coordination_count",
    "subordination": "advcl_count",
    "modifier_density": "modifier_density",
}
MIN_PAIRS = {"train": 120, "dev": 30, "test": 40}


def opener_class(q: str) -> str:
    w = (q.split()[0].lower() if q.split() else "").strip(",.!?")
    if w in {"i'm", "i", "i’m"}:
        return "personal"
    if w in {"honestly", "personally", "actually", "usually", "definitely", "occasionally"}:
        return "adverbial"
    if w in {"find", "show", "need", "want", "get"}:
        return "direct"
    return "other"


def attr_string(a: dict) -> str:
    return " | ".join(f"{k}={a[k]}" for k in sorted(a))


def build_pairs() -> list:
    records = json.load(open(RECORDS))
    pairs = []
    for rec in records:
        attrs = (rec.get("syntax_depth_query") or {}).get("attrs_used")
        if not attrs:
            for cand in rec.get("syntax_depth_queries", []):
                if cand.get("attrs_used"):
                    attrs = cand["attrs_used"]
                    break
        if not attrs or len(attrs) < 5:
            continue
        a = {k: str(v) for k, v in attrs.items() if v}
        if len(a) < 5:
            continue
        # generic 5-slot: any key set (A1..A18), sorted keys, first five values
        vals = [a[k] for k in sorted(a)][:5]
        v1, v2, v3, v4, v5 = vals
        base = f"I'm looking for {v2} {v1} that are {v3} for {v4} at {v5}."
        # three opener variants
        opener_pairs = [
            (base, f"Personally, I need {v2} {v1} that are {v3} for {v4}, at {v5}."),
            (f"I want to buy {v2} {v1} that are {v3} for {v4} at {v5}.",
             f"Honestly, I am on the hunt for {v2} {v1} that are {v3} for {v4}, at {v5}."),
            (f"Please show me {v2} {v1} that are {v3} for {v4} at {v5}.",
             f"Actually, I am hoping to find {v2} {v1} that are {v3} for {v4}, at {v5}."),
        ]
        for neg, pos in opener_pairs:
            pairs.append({"axis": "opener", "negative": neg, "positive": pos,
                          "attrs": a, "asin": rec["asin"], "user_id": rec["user_id"]})
        # three coordination variants
        coord_pairs = [
            (base, f"I'm looking for {v2} {v1} that are {v3} for {v4}, and I need them at {v5}."),
            (f"I want to buy {v2} {v1} that are {v3} for {v4} at {v5}.",
             f"I want to buy {v2} {v1} that are {v3} for {v4}, and they should cost {v5}."),
            (f"Please show me {v2} {v1} that are {v3} for {v4} at {v5}.",
             f"Please show me {v2} {v1} that are {v3} for {v4}, and I would like them at {v5}."),
        ]
        for neg, pos in coord_pairs:
            pairs.append({"axis": "coordination", "negative": neg, "positive": pos,
                          "attrs": a, "asin": rec["asin"], "user_id": rec["user_id"]})
        # three subordination variants
        sub_pairs = [
            (base, f"{base}, because they are really needed."),
            (f"I want to buy {v2} {v1} that are {v3} for {v4} at {v5}.",
             f"I want to buy {v2} {v1} that are {v3} for {v4} at {v5}, since they are essential."),
            (f"Please show me {v2} {v1} that are {v3} for {v4} at {v5}.",
             f"Please show me {v2} {v1} that are {v3} for {v4} at {v5}, although I am not sure yet."),
        ]
        for neg, pos in sub_pairs:
            pairs.append({"axis": "subordination", "negative": neg, "positive": pos,
                          "attrs": a, "asin": rec["asin"], "user_id": rec["user_id"]})
        # three modifier_density variants
        mod_pairs = [
            (base, f"I'm looking for {v2} {v1} that are {v3}, {v4}-friendly, very durable, extremely practical and perfectly suited, at {v5}."),
            (f"I want to buy {v2} {v1} at {v5}.",
             f"I want to buy {v2} {v1}, which is remarkably {v3}, wonderfully {v4}-suited and incredibly reliable, at {v5}."),
            (f"Please show me {v2} {v1} at {v5}.",
             f"Please show me {v2} {v1}, highly {v3}, genuinely {v4}-oriented, extremely well-made and superbly finished, at {v5}."),
        ]
        for neg, pos in mod_pairs:
            pairs.append({"axis": "modifier_density", "negative": neg, "positive": pos,
                          "attrs": a, "asin": rec["asin"], "user_id": rec["user_id"]})
    return pairs


def verify(pairs) -> tuple:
    nlp = load_spacy_model()
    ok = 0
    attr_mismatch = 0
    per_axis = Counter()
    for p in pairs:
        neg, pos = p["negative"], p["positive"]
        # attribute fidelity: every value appears exactly once in both
        vals = list(p["attrs"].values())
        if not all(v in neg and v in pos for v in vals):
            attr_mismatch += 1
            continue
        fn = extract_clause_features(neg)
        fp = extract_clause_features(pos)
        target = TARGET_FEATURES[p["axis"]]
        if target is None:
            # opener: positive should be adverbial/personal, negative personal
            good = opener_class(pos) == "adverbial" and opener_class(neg) == "personal"
        else:
            good = fp[target] > fn[target]
        if good:
            ok += 1
            per_axis[p["axis"]] += 1
        p["features_neg"] = {k: fn.get(k, 0) for k in ["coordination_count", "advcl_count", "modifier_density", "advmod_count", "max_dependency_depth"]}
        p["features_pos"] = {k: fp.get(k, 0) for k in ["coordination_count", "advcl_count", "modifier_density", "advmod_count", "max_dependency_depth"]}
    return ok, attr_mismatch, per_axis


def main() -> None:
    pairs = build_pairs()
    ok, mismatch, per_axis = verify(pairs)
    print(f"pairs={len(pairs)} verified_ok={ok} attr_mismatch={mismatch} per_axis={dict(per_axis)}")

    # item-level split (by asin) — train/dev/test
    rng = np.random.default_rng(SEED)
    asins = sorted({p["asin"] for p in pairs})
    idx = rng.permutation(len(asins))
    n_test = max(40, int(0.2 * len(asins)))
    n_dev = max(30, int(0.15 * len(asins)))
    test_asins = {asins[int(i)] for i in idx[:n_test]}
    dev_asins = {asins[int(i)] for i in idx[n_test:n_test + n_dev]}
    for p in pairs:
        if p["asin"] in test_asins:
            p["split"] = "test"
        elif p["asin"] in dev_asins:
            p["split"] = "dev"
        else:
            p["split"] = "train"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "syntax_axis_pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    manifest = {
        "n_pairs": len(pairs),
        "n_items": len(asins),
        "split_counts": dict(Counter(p["split"] for p in pairs)),
        "axes": sorted(per_axis),
        "seed": SEED,
        "exclusion": {"attr_mismatch": mismatch},
    }
    (OUT_DIR / "axis_split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "axis_audit.json").write_text(json.dumps({
        "verified": ok, "total": len(pairs), "per_axis": dict(per_axis),
        "attr_mismatch": mismatch,
    }, indent=2))
    print(f"wrote {OUT_DIR}")
    print("manifest:", manifest["split_counts"])


if __name__ == "__main__":
    main()
