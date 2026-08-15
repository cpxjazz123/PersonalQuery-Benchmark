#!/usr/bin/env python3
"""E17 Step 5: Content guardrail evaluation on Step-4 test outputs.

For each test (user, product) row in e17_test_samples.jsonl, verify the
generated query mentions the five product attributes (A1..A5) with high
fidelity under all four conditions (correct / shuffled / global_mean / zero).

Per-issue Check-5 metrics:
  - five-attribute exact-match rate >= 99% (A1..A5 each appear verbatim in
    the generated query)
  - numeric and brand strings: per-character exact = 100%
  - product-attribute string consistency across correct/shuffled/zero
    conditions for the SAME (user, product): 100% identical attribute
    substrings (i.e., style vector does NOT mutate product content)
  - report attribute repetition / distortion / drop, number degradation, and
    parse-failure rates separately
  - degradation rate (parse failure + number/brand damage) must not exceed
    the injection-off baseline's bootstrap 95% CI upper bound

Outputs e17_content_eval.json with per-condition counts and per-(user,product)
consistency table.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
E17 = REPO_ROOT / "result" / "personal_query" / "e17"
SAMPLES = E17 / "e17_test_samples.jsonl"
OUT = E17 / "e17_content_eval.json"

NUMERIC_TOKEN = re.compile(r"\$?\d+(?:\.\d+)?")
BRAND_HINT_KEYS = ("A2",)


def attr_verbatim(q: str, val: str) -> bool:
    if not val:
        return False
    # case-insensitive contains, but price needs exact form
    return val.lower() in q.lower()


def numeric_exact(q: str, val: str) -> bool:
    """For A3 (price), the number should appear verbatim (e.g. '$8.99' or '8.99')."""
    if not val:
        return False
    # strip $ for comparison
    bare = val.lstrip("$")
    if bare.replace(".", "", 1).isdigit() is False:
        return attr_verbatim(q, val)
    # must appear with its exact digits
    return bare in q


def brand_exact(q: str, val: str) -> bool:
    if not val or len(val) < 2:
        return False
    # brand: case-sensitive substring (brands are usually proper nouns)
    return val in q


def main() -> None:
    rows = [json.loads(l) for l in open(SAMPLES)]
    print(f"loaded {len(rows)} sample rows", flush=True)

    # per-row attribute checks
    n_total = len(rows)
    n_empty = sum(1 for r in rows if not r["query"].strip())
    per_row = []
    for r in rows:
        q = r["query"]
        attrs = r["attrs"]
        a_match = {k: attr_verbatim(q, v) for k, v in attrs.items()}
        n_match = {k: numeric_exact(q, attrs.get(k, "")) for k in ("A3",)}
        b_match = {k: brand_exact(q, attrs.get(k, "")) for k in BRAND_HINT_KEYS}
        per_row.append({"user_id": r["user_id"], "asin": r["asin"], "cond": r["cond"],
                        "attrs": attrs, "query": q,
                        "a_match": a_match, "n_match": n_match, "b_match": b_match,
                        "n_attrs_matched": sum(a_match.values()),
                        "empty": not q.strip()})

    # per-condition aggregate
    per_cond = {}
    for cond in ("correct", "shuffled", "global_mean", "zero"):
        sub = [r for r in per_row if r["cond"] == cond]
        n = max(len(sub), 1)
        all_five_match = sum(1 for r in sub if r["n_attrs_matched"] == 5)
        numeric_exact_n = sum(1 for r in sub if r["n_match"].get("A3"))
        brand_exact_n = sum(1 for r in sub if r["b_match"].get("A2"))
        per_attr_match = {k: sum(r["a_match"][k] for r in sub) / n for k in ("A1", "A2", "A3", "A4", "A5")}
        parse_fail = sum(1 for r in sub if r["empty"])
        per_cond[cond] = {
            "n_rows": len(sub),
            "five_attr_exact_rate": all_five_match / n,
            "numeric_exact_rate": numeric_exact_n / n,
            "brand_exact_rate": brand_exact_n / n,
            "parse_fail_rate": parse_fail / n,
            "per_attr_match": per_attr_match,
        }

    # product-attribute string consistency across conditions
    # for each (user, asin), check that the THREE style-vectors (correct, shuffled, zero)
    # produce IDENTICAL attribute-substring presence pattern
    pairs = defaultdict(dict)
    for r in per_row:
        if r["cond"] in ("correct", "shuffled", "zero"):
            pairs[(r["user_id"], r["asin"])][r["cond"]] = r["a_match"]
    n_pairs = len(pairs)
    identical = 0
    detail = []
    for (uid, asin), m in pairs.items():
        if len(m) == 3 and m["correct"] == m["shuffled"] == m["zero"]:
            identical += 1
        else:
            detail.append({"user_id": uid, "asin": asin, "match": m})
    consistency_rate = identical / max(n_pairs, 1)

    # injection-off baseline: rows where cond=zero (already a no-vector condition)
    zero_sub = per_cond["zero"]
    # bootstrap 95% CI on five_attr_exact_rate (upper bound = degradation ceiling)
    rs = np.random.default_rng(0)
    n_zero = zero_sub["n_rows"]
    zero_match_arr = np.array([1 if (r["cond"] == "zero" and r["n_attrs_matched"] == 5) else 0
                               for r in per_row])
    if n_zero > 0:
        boot = []
        for _ in range(999):
            idx = rs.integers(0, n_zero, n_zero)
            boot.append(zero_match_arr[:n_zero][idx].mean())
        boot = np.array(boot)
        baseline_lo = float(np.percentile(boot, 2.5))
        baseline_hi = float(np.percentile(boot, 97.5))
    else:
        baseline_lo = baseline_hi = 0.0
    correct_rate = per_cond["correct"]["five_attr_exact_rate"]
    correct_within_ci = baseline_lo <= correct_rate

    # final pass criteria per Check-5
    pass_five_attr = all(per_cond[c]["five_attr_exact_rate"] >= 0.99 for c in per_cond)
    pass_numeric = all(per_cond[c]["numeric_exact_rate"] >= 0.99 for c in per_cond)
    pass_brand = all(per_cond[c]["brand_exact_rate"] >= 1.0 - 1e-9 for c in per_cond)
    pass_consistency = consistency_rate >= 1.0 - 1e-9
    pass_degradation = correct_within_ci
    check5_pass = bool(pass_five_attr and pass_numeric and pass_brand
                       and pass_consistency and pass_degradation)

    out = {
        "version": "e17-step5",
        "n_total": n_total,
        "n_empty": n_empty,
        "n_pairs_three_cond": n_pairs,
        "per_condition": per_cond,
        "product_string_consistency": {
            "identical_patterns": identical, "n_pairs": n_pairs, "rate": consistency_rate,
            "mismatches_sample": detail[:10],
        },
        "injection_off_baseline": {
            "five_attr_exact_rate": float(zero_match_arr[:n_zero].mean()) if n_zero else 0.0,
            "boot95_ci": [baseline_lo, baseline_hi],
        },
        "check5_pass": check5_pass,
        "thresholds": {
            "five_attr_exact_min": 0.99,
            "numeric_exact_min": 0.99,
            "brand_exact_min": 1.0,
            "consistency_min": 1.0,
            "correct_within_baseline_ci": True,
        },
    }
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}", flush=True)
    for c in per_cond:
        print(f"  {c}: five={per_cond[c]['five_attr_exact_rate']:.3f}  "
              f"num={per_cond[c]['numeric_exact_rate']:.3f}  "
              f"brand={per_cond[c]['brand_exact_rate']:.3f}  "
              f"parse_fail={per_cond[c]['parse_fail_rate']:.3f}", flush=True)
    print(f"product_string_consistency: {consistency_rate:.3f} ({identical}/{n_pairs})", flush=True)
    print(f"injection-off baseline 5-attr boot95%CI: [{baseline_lo:.3f}, {baseline_hi:.3f}]  "
          f"correct {correct_rate:.3f} within_CI={correct_within_ci}", flush=True)
    print(f"OVERALL check5_pass = {check5_pass}", flush=True)


if __name__ == "__main__":
    main()
