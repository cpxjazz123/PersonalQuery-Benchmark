#!/usr/bin/env python3
"""E11/E12 — evaluate soft-prefix generated queries (template pipeline).

Content integrity (E12):
- five-attribute exact (each attr exactly once) on the FINAL query
- per-attr exact-once rates
- numeric exact rate (price A3 value appears verbatim)
- brand/model exact rate (A2)
- missing / duplicate / mutation classification
- raw template valid rate, retry rate, fallback rate (from generation json)

Degeneration:
- repeated-character / repeated-token / repeated 3-gram ratios
- consecutive-digit/punctuation anomaly rate

Style consistency (vs user VADES center, VADES LATENT space via shared encoder):
- VADES distance of the final query to the target user's user_mu
- in-95%-range z-score, syntactic feature stats

Statistics:
- paired Wilcoxon aligned by (user_id, asin) (issue #12: NOT by user_id)
- user-level bootstrap 95% CI on the mean distance difference
- held-out users are the primary population; train users reported as
  diagnostics only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query"))
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))

from user_style_vectors import (  # noqa: E402
    load_vades_profiles,
    load_feature_scaler,
    FEATURE_KEYS,
)
from content_validation import content_valid  # noqa: E402
from template_placeholder import (  # noqa: E402
    PLACEHOLDER_BY_ATTR,
    parse_template,
)
from extract_clause_features_single_query import (  # noqa: E402
    load_spacy_model,
    extract_clause_features,
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_feature_vec(query: str, nlp) -> Optional[np.ndarray]:
    try:
        feats = extract_clause_features(query)
    except Exception:
        return None
    vec = np.asarray([float(feats.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)
    if not np.isfinite(vec).all():
        return None
    return vec


def in_user_range_z(
    latent_mu: np.ndarray, mu: np.ndarray, logvar: np.ndarray
) -> Tuple[bool, float]:
    std = np.exp(0.5 * logvar)
    zero_mask = std < 1e-8
    if zero_mask.any():
        z_zero = np.abs(latent_mu - mu)[zero_mask] / 1.0
    else:
        z_zero = np.array([], dtype=np.float32)
    if (~zero_mask).any():
        z_nz = np.abs((latent_mu - mu) / np.maximum(std, 1e-9))[~zero_mask]
    else:
        z_nz = np.array([], dtype=np.float32)
    z = np.concatenate([z_zero, z_nz])
    return bool(np.all(z <= 1.96)), float(np.mean(z)) if z.size else 0.0


def attr_occurrences(query: str, attrs_used: Dict[str, str]) -> Dict[str, int]:
    """Count raw (case-insensitive) occurrences of each attr value in query."""
    counts = {}
    for key, value in attrs_used.items():
        if not value:
            counts[key] = 0
            continue
        counts[key] = len(re.findall(re.escape(str(value)), query, re.IGNORECASE))
    return counts


def degradation_stats(query: str) -> dict:
    toks = re.findall(r"\S+", query or "")
    n_tok = len(toks)
    # repeated character runs (>=3 identical chars)
    char_runs = len(re.findall(r"(.)\1{2,}", query or ""))
    # consecutive digits/punct anomalies like 8.9.9.9.9999
    digit_anom = len(re.findall(r"\d[.,]\d[.,]\d", query or ""))
    # repeated 3-gram
    grams = [query[i:i + 3] for i in range(max(0, len(query) - 2))]
    dup_3gram = len(grams) - len(set(grams))
    return {
        "char_runs": char_runs,
        "digit_anomalies": digit_anom,
        "dup_3gram_ratio": dup_3gram / max(len(grams), 1),
    }


def content_classification(final: str, attrs: Dict[str, str], counts: Dict[str, int]) -> Dict[str, object]:
    missing = [k for k in attrs if counts.get(k, 0) == 0]
    duplicated = [k for k in attrs if counts.get(k, 0) > 1]
    # numeric corruption: price (A3) must appear verbatim
    price = str(attrs.get("A3", ""))
    numeric_ok = bool(price) and price in final
    # brand (A2) verbatim
    brand = str(attrs.get("A2", ""))
    brand_ok = bool(brand) and brand in final
    return {
        "five_attr_exact": content_valid(final, attrs),
        "missing": missing,
        "duplicated": duplicated,
        "numeric_exact": bool(numeric_ok),
        "brand_exact": bool(brand_ok),
        "per_attr_exact_once": {k: (counts.get(k, 0) == 1) for k in attrs},
    }


def bootstrap_pair_ci(diffs: List[float], seed: int, n_boot: int = 2000) -> dict:
    if len(diffs) < 3:
        return {"ci_low": None, "ci_high": None, "mean": float(np.mean(diffs)) if diffs else None}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot.append(float(np.mean(sample)))
    boot = np.asarray(boot)
    return {
        "mean": float(np.mean(diffs)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="train checkpoint (split manifest)")
    ap.add_argument("--category", default="Baby_Products")
    ap.add_argument("--gen_files", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--main_label", default="B5", help="label used as the main condition in paired tests")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = load_vades_profiles(args.category)
    scaler = load_feature_scaler(args.category)
    from vades_latent import VadesLatentEncoder
    latent_encoder = VadesLatentEncoder(args.category)
    with open(Path(args.checkpoint_dir) / "split_manifest.json") as f:
        split = json.load(f)
    test_users = set(split["test_users"])
    log(f"profiles={len(profiles)} test_users={len(test_users)}")

    records_p = REPO_ROOT / "result" / "personal_query" / "04_query" / args.category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(records_p) as f:
        records = json.load(f)
    attrs_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    for rec in records:
        attrs = (rec.get("syntax_depth_query") or {}).get("attrs_used")
        if not attrs:
            for cand in rec.get("syntax_depth_queries", []):
                if cand.get("attrs_used"):
                    attrs = cand["attrs_used"]
                    break
        if attrs:
            attrs_by_key[(rec["user_id"], rec["asin"])] = attrs

    nlp = load_spacy_model()

    runs: Dict[str, str] = {}
    for item in args.gen_files:
        label, path = item.split("=", 1)
        runs[label] = path

    per_label: Dict[str, List[dict]] = {}
    for label, path in runs.items():
        with open(path) as f:
            payload = json.load(f)
        results = payload["results"]
        samples = []
        for r in results:
            uid, asin = r["user_id"], r["asin"]
            final = r.get("final_query") or ""
            raw = r.get("raw_template") or ""
            attrs = attrs_by_key.get((uid, asin))
            base = {
                "user_id": uid,
                "asin": asin,
                "split": "test" if uid in test_users else "train",
                "final_query": final,
                "raw_template": raw,
                "retry_count": r.get("retry_count", 0),
                "fallback": bool(r.get("fallback", False)),
                "failure_reason": r.get("failure_reason"),
            }
            if attrs:
                required = {PLACEHOLDER_BY_ATTR[k] for k in attrs if k in PLACEHOLDER_BY_ATTR}
            else:
                required = None
            parsed = parse_template(raw, required=required) if raw else {"valid": False}
            base["raw_template_valid"] = bool(parsed.get("valid", False))
            if attrs:
                counts = attr_occurrences(final, attrs)
                base["content"] = content_classification(final, attrs, counts)
                base["degradation"] = degradation_stats(final)
            else:
                base["content"] = None
                base["degradation"] = None
            if profiles.get(uid) is not None:
                vec = extract_feature_vec(final, nlp)
                if vec is not None:
                    # E13-A: distance in the VADES LATENT space (shared encoder),
                    # not the removed standardized-raw-features vs user_mu path.
                    cand_mu = latent_encoder.encode_feature_vec(vec)
                    mu = profiles[uid]["user_mu"].astype(np.float32)
                    lv = profiles[uid]["user_logvar"].astype(np.float32)
                    dist = float(np.linalg.norm(cand_mu - mu))
                    in_range, mean_z = in_user_range_z(cand_mu, mu, lv)
                    base["vades_dist"] = dist
                    base["in_user_95pct_range"] = in_range
                    base["mean_z"] = mean_z
            samples.append(base)
        per_label[label] = samples
        n_valid = sum(1 for s in samples if s["content"] and s["content"]["five_attr_exact"])
        log(f"{label}: n={len(samples)} five_attr_exact={n_valid} "
            f"raw_template_valid={sum(1 for s in samples if s['raw_template_valid'])} "
            f"fallback={sum(1 for s in samples if s['fallback'])}")

    # Aggregate per label
    summary = {"category": args.category, "labels": {}}
    for label, samples in per_label.items():
        n = len(samples)
        cts = [s["content"] for s in samples if s["content"] is not None]
        degs = [s["degradation"] for s in samples if s["degradation"] is not None]
        agg = {
            "n": n,
            "five_attr_exact_final": float(np.mean([c["five_attr_exact"] for c in cts])) if cts else None,
            "numeric_exact": float(np.mean([c["numeric_exact"] for c in cts])) if cts else None,
            "brand_exact": float(np.mean([c["brand_exact"] for c in cts])) if cts else None,
            "missing_rate": float(np.mean([len(c["missing"]) / 5 for c in cts])) if cts else None,
            "duplicate_rate": float(np.mean([len(c["duplicated"]) / 5 for c in cts])) if cts else None,
            "raw_template_valid_rate": float(np.mean([s["raw_template_valid"] for s in samples])),
            "retry_rate": float(np.mean([s["retry_count"] > 0 for s in samples])),
            "fallback_rate": float(np.mean([s["fallback"] for s in samples])),
            "digit_anomaly_rate": float(np.mean([d["digit_anomalies"] > 0 for d in degs])) if degs else None,
            "dup_3gram_ratio": float(np.mean([d["dup_3gram_ratio"] for d in degs])) if degs else None,
            "vades_dist": float(np.mean([s["vades_dist"] for s in samples if "vades_dist" in s])) if any("vades_dist" in s for s in samples) else None,
            "in_range_ratio": float(np.mean([s.get("in_user_95pct_range", False) for s in samples])) if any("in_user_95pct_range" in s for s in samples) else None,
            "mean_z": float(np.mean([s["mean_z"] for s in samples if "mean_z" in s])) if any("mean_z" in s for s in samples) else None,
        }
        test_samples = [s for s in samples if s["split"] == "test"]
        if test_samples:
            agg["heldout_n"] = len(test_samples)
            agg["heldout_vades_dist"] = float(np.mean([s["vades_dist"] for s in test_samples if "vades_dist" in s])) if any("vades_dist" in s for s in test_samples) else None
            agg["heldout_five_attr_exact"] = float(np.mean([s["content"]["five_attr_exact"] for s in test_samples if s["content"]])) if test_samples else None
        summary["labels"][label] = agg

    # Paired tests aligned by (user_id, asin)
    main_label = args.main_label
    pair_results = {}
    for control_label in ["B3", "B1", "zero", "global_mean"]:
        if main_label not in per_label or control_label not in per_label:
            continue
        main_by_key = {(s["user_id"], s["asin"]): s for s in per_label[main_label]}
        ctrl_by_key = {(s["user_id"], s["asin"]): s for s in per_label[control_label]}
        common = sorted(set(main_by_key) & set(ctrl_by_key))
        diffs = []
        for key in common:
            m, c = main_by_key[key], ctrl_by_key[key]
            if "vades_dist" in m and "vades_dist" in c:
                diffs.append(float(c["vades_dist"] - m["vades_dist"]))
        if len(diffs) >= 8:
            w_stat, w_p = stats.wilcoxon(np.asarray(diffs), alternative="greater")
        else:
            w_stat, w_p = 0.0, 1.0
        ci = bootstrap_pair_ci(diffs, seed=args.seed)
        pair_results[f"{main_label}_vs_{control_label}"] = {
            "n_paired": len(common),
            "mean_diff": float(np.mean(diffs)) if diffs else None,
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "bootstrap_ci": ci,
        }
        log(f"pair {main_label} vs {control_label}: n={len(common)} "
            f"mean_diff={np.mean(diffs) if diffs else None:.4f} wilcoxon_p={w_p:.4e}")
    summary["paired_tests"] = pair_results

    with open(out_dir / "per_sample.jsonl", "w") as f:
        for label, samples in per_label.items():
            for s in samples:
                f.write(json.dumps({"label": label, **s}, ensure_ascii=False) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        f"# E11/E12 soft-prefix evaluation ({args.category})",
        "",
        "## Content integrity & template pipeline",
        "| Label | n | 5-attr exact | numeric exact | brand exact | missing | duplicate | raw tpl valid | retry | fallback | digit anomaly |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, agg in summary["labels"].items():
        md_lines.append(
            f"| {label} | {agg['n']} | {agg['five_attr_exact_final']:.4f} | {agg['numeric_exact']:.4f} | "
            f"{agg['brand_exact']:.4f} | {agg['missing_rate']:.4f} | {agg['duplicate_rate']:.4f} | "
            f"{agg['raw_template_valid_rate']:.4f} | {agg['retry_rate']:.4f} | {agg['fallback_rate']:.4f} | "
            f"{agg['digit_anomaly_rate']:.4f} |"
        )
    md_lines.append("")
    md_lines.append("## Style consistency (VADES distance, lower = closer to user style)")
    md_lines.append("| Label | n | VADES dist | in-95% range | mean_z | held-out n | held-out dist | held-out 5-attr |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, agg in summary["labels"].items():
        md_lines.append(
            f"| {label} | {agg['n']} | {agg['vades_dist']:.4f} | {agg['in_range_ratio']:.4f} | {agg['mean_z']:.4f} | "
            f"{agg.get('heldout_n','-')} | {agg.get('heldout_vades_dist', float('nan')):.4f} | "
            f"{agg.get('heldout_five_attr_exact', float('nan')):.4f} |"
        )
    md_lines.append("")
    md_lines.append("## Paired style tests (main vs control, aligned by (user_id, asin))")
    md_lines.append("| Pair | n | mean diff | Wilcoxon p (one-sided) | bootstrap 95% CI |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for pair, res in pair_results.items():
        ci = res["bootstrap_ci"]
        ci_str = f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]" if ci["ci_low"] is not None else "n/a"
        md_lines.append(f"| {pair} | {res['n_paired']} | {res['mean_diff']:.4f} | {res['wilcoxon_p']:.4e} | {ci_str} |")
    with open(out_dir / "summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    log(f"wrote {out_dir / 'per_sample.jsonl'}, {out_dir / 'summary.json'}, {out_dir / 'summary.md'}")
    log("=== done ===")


if __name__ == "__main__":
    main()
