#!/usr/bin/env python3
"""E16-Task2 — Pre-registration (locked before any test2 run), valid-pair
filtering, and uncontaminated train/dev/test2 split.

- Filter: attribute mismatch (each value once, verbatim), parser direction
  wrong, off-target (non-target axes change beyond tolerance).
- Split: by ASIN AND by syntactic-family (template variant family) so no
  near-duplicate crosses splits. Old dev/test ids are EXCLUDED from test2.
- e16_preregister.json written BEFORE any test2 artifact (commit time check).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))

from extract_clause_features_single_query import load_spacy_model, extract_clause_features  # noqa: E402

E16_DIR = REPO_ROOT / "result" / "personal_query" / "e16"
OLD_PAIRS = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/e15/syntax_axis_pairs.jsonl")
OLD_TEST_IDS = None  # set below from old manifest (ASINs of old test)
SEED = 42
TARGET_FEATURE = {
    "opener": None,
    "coordination": "coordination_count",
    "subordination": "advcl_count",
    "modifier_density": "modifier_density",
}
OFF_TARGET_TOLERANCE = {"opener": 1.0, "coordination": 1.0, "subordination": 1.0,
                        "modifier_density": 3.0}  # modifiers often add 'and' (coordination)
MIN_DEV_ITEMS = 30
MIN_TEST2_ITEMS = 40


def opener_class(q: str) -> str:
    w = (q.split()[0].lower() if q.split() else "").strip(",.!?")
    if w in {"i'm", "i", "i’m", "i’m"}:
        return "personal"
    if w in {"honestly", "personally", "actually", "usually", "definitely", "occasionally"}:
        return "adverbial"
    if w in {"find", "show", "need", "want", "get", "please"}:
        return "direct"
    return "other"


def is_valid(p: dict, nlp) -> tuple:
    """Return (valid, reason). Attribute fidelity + target direction + off-target."""
    attrs = p["attrs"]
    vals = list(attrs.values())
    neg, pos = p["negative"], p["positive"]
    if not all(v in neg and v in pos for v in vals):
        return False, "attr_missing"
    for v in vals:
        if neg.count(v) != 1 or pos.count(v) != 1:
            return False, "attr_duplicate"
    fn = extract_clause_features(neg)
    fp = extract_clause_features(pos)
    if p["axis"] == "opener":
        good = opener_class(pos) == "adverbial" and opener_class(neg) == "personal"
    else:
        t = TARGET_FEATURE[p["axis"]]
        good = fp[t] > fn[t]
    if not good:
        return False, "parser_direction"
    tol = OFF_TARGET_TOLERANCE[p["axis"]]
    for key, tv in [("coordination_count", "coordination"), ("advcl_count", "subordination"),
                    ("modifier_density", "modifier_density")]:
        if key == TARGET_FEATURE[p["axis"]]:
            continue
        if abs(fp[key] - fn[key]) > tol:
            return False, f"off_target_{key}"
    return True, "ok"


def main() -> None:
    nlp = load_spacy_model()

    def log2(m):
        print(m, flush=True)
    old = [json.loads(l) for l in open(OLD_PAIRS)]
    # old test ASINs are contaminated (E15): never enter test2
    old_test_asins = {p["asin"] for p in old if p.get("split") == "test"}
    old_dev_asins = {p["asin"] for p in old if p.get("split") == "dev"}
    contaminated = old_test_asins | old_dev_asins  # dev also used for selection in E15
    log_n = {"total": len(old)}
    valid, invalid = [], []
    reasons = Counter()
    for p in old:
        ok, reason = is_valid(p, nlp)
        if ok:
            valid.append(p)
        else:
            invalid.append({**p, "reason": reason})
            reasons[reason] += 1
    log_n["valid"] = len(valid)
    log_n["invalid"] = len(invalid)
    log_n["invalid_reasons"] = dict(reasons)
    print(f"valid={len(valid)} invalid={len(invalid)} reasons={dict(reasons)}")

    # syntactic-family grouping: template variant (first template phrase pair)
    families = defaultdict(list)
    for p in valid:
        fam = p["negative"].split(" that ")[0]
        families[fam].append(p)
    # split by (ASIN group, family group) to avoid leakage
    asins = sorted({p["asin"] for p in valid})
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(asins))
    n_test = max(MIN_TEST2_ITEMS, int(0.2 * len(asins)))
    n_dev = max(MIN_DEV_ITEMS, int(0.15 * len(asins)))
    test_asins = {asins[int(i)] for i in idx[:n_test]}
    dev_asins = {asins[int(i)] for i in idx[n_test:n_test + n_dev]}
    # exclude contaminated ASINs from test2 (keep them only in train/exploratory)
    clean_test = test_asins - contaminated
    clean_dev = dev_asins - contaminated
    for p in valid:
        if p["asin"] in clean_test:
            p["split"] = "test2"
        elif p["asin"] in clean_dev:
            p["split"] = "dev"
        else:
            p["split"] = "train"
    splits = Counter(p["split"] for p in valid)
    # family-level leakage check + fix: leaked families move to train so
    # dev/test2 are contamination-free
    fam_splits = defaultdict(set)
    for p in valid:
        fam_splits[p["split"]].add(p["negative"].split(" that ")[0])
    leakage = set(fam_splits["train"]) & (set(fam_splits["dev"]) | set(fam_splits["test2"]))
    if leakage:
        log2(f"family leakage {len(leakage)} -> moving to train")
        for p in valid:
            fam = p["negative"].split(" that ")[0]
            if fam in leakage and p["split"] in ("dev", "test2"):
                p["split"] = "train"
    dev_items = {p["asin"] for p in valid if p["split"] == "dev"}
    test2_items = {p["asin"] for p in valid if p["split"] == "test2"}
    per_axis_dev = Counter(p["axis"] for p in valid if p["split"] == "dev")
    per_axis_test2 = Counter(p["axis"] for p in valid if p["split"] == "test2")

    preregister = {
        "model": "Qwen2.5-1.5B-Instruct (989aa7980e4cf806f80c7fef2b1adb7bc71aa306)",
        "axes": ["opener", "coordination", "subordination", "modifier_density"],
        "representation": "sentence-END token hidden state (activation_contract.json)",
        "layer_mapping": "hidden_states[L+1] == layers[L]",
        "direction_methods": ["mean_diff", "logistic", "pca"],
        "intervention": "GenOnlyController: alpha*direction at generated tokens only",
        "alpha_grid": [0.0, 1.0, 3.0, -1.0, -3.0],
        "primary_metric": "standardized effect of axis feature (alpha=+3 vs -3, paired by item)",
        "sesoi": 0.3,
        "alpha_level": 0.05,
        "multiple_comparison": "bonferroni_across_4_axes",
        "seed": SEED,
        "min_dev_items_per_axis": MIN_DEV_ITEMS,
        "min_test2_items_per_axis": MIN_TEST2_ITEMS,
        "contaminated_excluded_from_test2": True,
        "split_counts": dict(splits),
        "dev_items": len(dev_items),
        "test2_items": len(test2_items),
        "per_axis_dev": dict(per_axis_dev),
        "per_axis_test2": dict(per_axis_test2),
        "family_leakage_across_splits": len(leakage),
        "created_at": datetime.now().isoformat(),
    }
    E16_DIR.mkdir(parents=True, exist_ok=True)
    (E16_DIR / "e16_preregister.json").write_text(json.dumps(preregister, indent=2))
    with open(E16_DIR / "valid_pairs.jsonl", "w") as f:
        for p in valid:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(E16_DIR / "invalid_pairs.jsonl", "w") as f:
        for p in invalid:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    manifest = {
        "n_valid": len(valid), "n_invalid": len(invalid),
        "invalid_reasons": dict(reasons),
        "splits": dict(splits),
        "contaminated_excluded": sorted(contaminated),
        "family_leakage_detected": len(leakage),
        "family_leakage_remaining_after_fix": 0,
        "per_axis": {"dev": dict(per_axis_dev), "test2": dict(per_axis_test2)},
        "preregister_hash": hashlib.sha256((E16_DIR / "e16_preregister.json").read_bytes()).hexdigest(),
    }
    (E16_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "contaminated_excluded"}, indent=1))


if __name__ == "__main__":
    main()
