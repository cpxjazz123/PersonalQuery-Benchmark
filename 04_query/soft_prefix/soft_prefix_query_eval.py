#!/usr/bin/env python3
"""E11 — evaluate soft-prefix generated queries.

Content correctness:
- five-attribute coverage (each of A1..A5 appears exactly once)
- missing-attribute (hallucination) rate

Style consistency (vs user VADES center, standardized clause-feature space):
- VADES distance of the generated query to the target user's user_mu
- in-95%-range z-score (E10-style), syntactic feature distribution stats

Statistics:
- paired Wilcoxon: correct vector < shuffled / zero controls (one-sided)
- user-level bootstrap 95% CI on the mean distance difference
- overall + warm/cold user breakdown

Inputs: one or more generation JSONs (output of soft_prefix_query_generate.py)
labeled by experiment id (e.g. B5, B3, B1).
"""
from __future__ import annotations

import argparse
import json
import sys
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
from extract_clause_features_single_query import (  # noqa: E402
    load_spacy_model,
    extract_clause_features,
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_feature_vec(query: str, nlp) -> Optional[np.ndarray]:
    try:
        doc = nlp(query)
        feats = extract_clause_features(query)
    except Exception:
        return None
    vec = np.asarray([float(feats.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)
    if not np.isfinite(vec).all():
        return None
    return vec


def standardize(vec: np.ndarray, scaler: dict) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    return (vec - mean) / np.maximum(scale, 1e-9)


def in_user_range_z(
    std_vec: np.ndarray, mu: np.ndarray, logvar: np.ndarray
) -> Tuple[bool, float]:
    """z-score of the query latent under the user's diagonal Gaussian;
    dims with near-zero variance use absolute distance tolerance 1.0."""
    std = np.exp(0.5 * logvar)
    zero_mask = std < 1e-8
    if zero_mask.any():
        z_zero = np.abs(std_vec - mu)[zero_mask] / 1.0
    else:
        z_zero = np.array([], dtype=np.float32)
    if (~zero_mask).any():
        z_nz = np.abs((std_vec - mu) / np.maximum(std, 1e-9))[~zero_mask]
    else:
        z_nz = np.array([], dtype=np.float32)
    z = np.concatenate([z_zero, z_nz])
    return bool(np.all(z <= 1.96)), float(np.mean(z)) if z.size else 0.0


def content_stats(query: str, attrs_used: Dict[str, str]) -> dict:
    ok = content_valid(query, attrs_used)
    n_covered = 0
    for key, value in attrs_used.items():
        if value and value.lower() in query.lower():
            n_covered += 1
    return {
        "five_attrs_exact": ok,
        "n_attrs_present": n_covered,
        "missing_attrs": len(attrs_used) - n_covered,
    }


def bootstrap_user_ci(diffs_by_user: Dict[str, List[float]], seed: int, n_boot: int = 2000) -> dict:
    """User-level bootstrap CI of the mean difference (each user contributes
    the mean of their sample differences; resample users with replacement)."""
    keys = list(diffs_by_user.keys())
    user_means = {u: float(np.mean(v)) for u, v in diffs_by_user.items()}
    if len(keys) < 3:
        return {"n_users": len(keys), "ci_low": None, "ci_high": None, "mean": float(np.mean(list(user_means.values()))) if keys else None}
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(keys, size=len(keys), replace=True)
        boot_means.append(float(np.mean([user_means[u] for u in sample])))
    boot_means = np.asarray(boot_means)
    return {
        "n_users": len(keys),
        "mean": float(np.mean(list(user_means.values()))),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="train checkpoint (split manifest)")
    ap.add_argument("--category", default="Baby_Products")
    ap.add_argument("--gen_files", nargs="+", required=True, help="label=path pairs, e.g. B5=/out/b5.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--main_label", default="B5", help="label used as the main condition in paired tests")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = load_vades_profiles(args.category)
    scaler = load_feature_scaler(args.category)
    with open(Path(args.checkpoint_dir) / "split_manifest.json") as f:
        split = json.load(f)
    test_users = set(split["test_users"])
    log(f"profiles={len(profiles)} test_users={len(test_users)}")

    records_p = REPO_ROOT / "result" / "personal_query" / "04_query" / args.category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(records_p) as f:
        records = json.load(f)
    attrs_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    for rec in records:
        attrs = rec.get("syntax_depth_query", {}).get("attrs_used")
        if not attrs:
            for cand in rec.get("syntax_depth_queries", []):
                if cand.get("attrs_used"):
                    attrs = cand["attrs_used"]
                    break
        if attrs:
            attrs_by_key[(rec["user_id"], rec["asin"])] = attrs

    nlp = load_spacy_model()

    runs: Dict[str, dict] = {}
    for item in args.gen_files:
        label, path = item.split("=", 1)
        runs[label] = path

    per_label = {}
    for label, path in runs.items():
        with open(path) as f:
            payload = json.load(f)
        results = payload["results"]
        samples = []
        for r in results:
            uid, asin = r["user_id"], r["asin"]
            query = r.get("generated_query") or ""
            attrs = attrs_by_key.get((uid, asin))
            profile = profiles.get(uid)
            base = {
                "user_id": uid,
                "asin": asin,
                "split": "test" if uid in test_users else "train",
                "query": query,
            }
            if attrs:
                base.update(content_stats(query, attrs))
            if profile is not None:
                vec = extract_feature_vec(query, nlp)
                if vec is not None:
                    std_vec = standardize(vec, scaler)
                    mu = profile["user_mu"].astype(np.float32)
                    lv = profile["user_logvar"].astype(np.float32)
                    dist = float(np.linalg.norm(std_vec - mu))
                    in_range, mean_z = in_user_range_z(std_vec, mu, lv)
                    base.update({"vades_dist": dist, "in_user_95pct_range": in_range, "mean_z": mean_z})
            samples.append(base)
        per_label[label] = samples
        log(f"{label}: {len(samples)} samples, "
            f"with_vades_dist={sum(1 for s in samples if 'vades_dist' in s)}, "
            f"five_attrs_exact={sum(1 for s in samples if s.get('five_attrs_exact'))}")

    # Aggregate per label
    summary = {"category": args.category, "labels": {}}
    for label, samples in per_label.items():
        n = len(samples)
        agg = {
            "n": n,
            "five_attrs_exact": float(np.mean([s.get("five_attrs_exact", False) for s in samples])),
            "mean_missing_attrs": float(np.mean([s.get("missing_attrs", 5) for s in samples])),
            "vades_dist": float(np.mean([s["vades_dist"] for s in samples if "vades_dist" in s])) if any("vades_dist" in s for s in samples) else None,
            "in_range_ratio": float(np.mean([s.get("in_user_95pct_range", False) for s in samples])) if any("in_user_95pct_range" in s for s in samples) else None,
            "mean_z": float(np.mean([s["mean_z"] for s in samples if "mean_z" in s])) if any("mean_z" in s for s in samples) else None,
        }
        test_samples = [s for s in samples if s["split"] == "test"]
        if test_samples:
            agg["cold_test_n"] = len(test_samples)
            agg["cold_test_vades_dist"] = float(np.mean([s["vades_dist"] for s in test_samples if "vades_dist" in s])) if any("vades_dist" in s for s in test_samples) else None
            agg["cold_test_five_attrs_exact"] = float(np.mean([s.get("five_attrs_exact", False) for s in test_samples]))
        summary["labels"][label] = agg

    # Paired comparisons: main label (B5) vs controls (B3 shuffled / zero / B1)
    main_label = args.main_label
    control_pairs = [(main_label, "B3"), (main_label, "B1"), (main_label, "zero"), (main_label, "global_mean")]
    pair_results = {}
    for main_label, control_label in control_pairs:
        if main_label not in per_label or control_label not in per_label:
            continue
        main_samples = {s["user_id"]: s for s in per_label[main_label]}
        control_samples = {s["user_id"]: s for s in per_label[control_label]}
        common = sorted(set(main_samples) & set(control_samples))
        diffs = []
        diffs_by_user: Dict[str, List[float]] = {}
        for uid in common:
            m = main_samples[uid]
            c = control_samples[uid]
            if "vades_dist" in m and "vades_dist" in c:
                diff = float(c["vades_dist"] - m["vades_dist"])
                diffs.append(diff)
                diffs_by_user.setdefault(uid, []).append(diff)
        if len(diffs) >= 8:
            w_stat, w_p = stats.wilcoxon(np.asarray(diffs), alternative="greater")
        else:
            w_stat, w_p = 0.0, 1.0
        ci = bootstrap_user_ci(diffs_by_user, seed=args.seed)
        pair_results[f"{main_label}_vs_{control_label}"] = {
            "n_paired_users": len(common),
            "mean_diff": float(np.mean(diffs)) if diffs else None,
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "bootstrap_user_ci": ci,
        }
        log(f"pair {main_label} vs {control_label}: n={len(common)} "
            f"mean_diff={np.mean(diffs) if diffs else None:.4f} wilcoxon_p={w_p:.4e}")

    summary["paired_tests"] = pair_results

    per_sample_path = out_dir / "per_sample.jsonl"
    with open(per_sample_path, "w") as f:
        for label, samples in per_label.items():
            for s in samples:
                f.write(json.dumps({"label": label, **s}, ensure_ascii=False) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        f"# E11 soft-prefix evaluation ({args.category})",
        "",
        "| Label | n | 5-attr exact | missing-attr mean | VADES dist | in-95% range | mean_z | cold-test dist | cold-test 5-attr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, agg in summary["labels"].items():
        md_lines.append(
            f"| {label} | {agg['n']} | {agg['five_attrs_exact']:.4f} | {agg['mean_missing_attrs']:.4f} | "
            f"{agg['vades_dist']:.4f} | {agg['in_range_ratio']:.4f} | {agg['mean_z']:.4f} | "
            f"{agg.get('cold_test_vades_dist', float('nan')):.4f} | {agg.get('cold_test_five_attrs_exact', float('nan')):.4f} |"
        )
    md_lines.append("")
    md_lines.append("## Paired style tests (main vs control, VADES distance)")
    md_lines.append("| Pair | n users | mean diff | Wilcoxon p (one-sided) | bootstrap 95% CI |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for pair, res in pair_results.items():
        ci = res["bootstrap_user_ci"]
        ci_str = f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]" if ci["ci_low"] is not None else "n/a"
        md_lines.append(f"| {pair} | {res['n_paired_users']} | {res['mean_diff']:.4f} | {res['wilcoxon_p']:.4e} | {ci_str} |")
    with open(out_dir / "summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    log(f"wrote {per_sample_path}, {out_dir / 'summary.json'}, {out_dir / 'summary.md'}")
    log("=== done ===")


if __name__ == "__main__":
    main()
