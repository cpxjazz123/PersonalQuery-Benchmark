#!/usr/bin/env python3
"""E14-G — Held-out style evaluation for copy-aware generations.

For each generated query: extract 20-dim syntactic features (spacy), project
into the VADES latent space (shared encoder), and measure the distance to the
user's own style center (user_mu). Paired (user_id, asin) tests compare the
correct-vector condition against shuffled/zero controls, restricted to
held-out users (split == test), with user-level clustered bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "query"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from user_style_vectors import load_vades_profiles, FEATURE_KEYS  # noqa: E402
from vades_latent import VadesLatentEncoder  # noqa: E402
from extract_clause_features_single_query import (  # noqa: E402
    load_spacy_model,
    extract_clause_features,
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_latent(query: str, nlp, encoder) -> np.ndarray:
    feats = {k: float(extract_clause_features(query).get(k, 0.0)) for k in FEATURE_KEYS}
    vec = np.asarray([feats[k] for k in FEATURE_KEYS], dtype=np.float32)
    return encoder.encode_feature_vec(vec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--category", default="Baby_Products")
    ap.add_argument("--gen_files", nargs="+", required=True, help="label=path")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint_dir)
    split = json.load(open(ckpt / "split_manifest.json"))
    test_users = set(split["test_users"])
    log(f"test users: {len(test_users)}")

    profiles = load_vades_profiles(args.category)
    encoder = VadesLatentEncoder(args.category)
    nlp = load_spacy_model()

    runs: Dict[str, str] = {}
    for item in args.gen_files:
        label, path = item.split("=", 1)
        runs[label] = path

    per_label: Dict[str, Dict[Tuple[str, str], dict]] = {}
    for label, path in runs.items():
        payload = json.load(open(path))
        by_key = {}
        for r in payload["results"]:
            q = r.get("generated_query") or ""
            uid = r["user_id"]
            if profiles.get(uid) is None or not q:
                continue
            latent = extract_latent(q, nlp, encoder)
            mu = profiles[uid]["user_mu"].astype(np.float32)
            by_key[(uid, r["asin"])] = {
                "dist": float(np.linalg.norm(latent - mu)),
                "test": uid in test_users,
            }
        per_label[label] = by_key
        n_test = sum(1 for v in by_key.values() if v["test"])
        log(f"{label}: n={len(by_key)} held-out={n_test}")

    out = {"category": args.category, "test_users": len(test_users), "labels": {}}
    for label, by_key in per_label.items():
        all_d = [v["dist"] for v in by_key.values()]
        test_d = [v["dist"] for v in by_key.values() if v["test"]]
        out["labels"][label] = {
            "n": len(all_d),
            "mean_dist_all": float(np.mean(all_d)),
            "n_heldout": len(test_d),
            "mean_dist_heldout": float(np.mean(test_d)) if test_d else None,
        }
        log(f"  {label}: held-out mean dist = {np.mean(test_d):.4f}" if test_d else f"  {label}: no held-out")

    # paired held-out tests: main vs controls (aligned by (user_id, asin))
    main_label = args.gen_files[0].split("=")[0]
    main = per_label[main_label]
    for control_label in [l for l in per_label if l != main_label]:
        ctrl = per_label[control_label]
        common = sorted(set(main) & set(ctrl))
        held = [k for k in common if main[k]["test"]]
        diffs = [ctrl[k]["dist"] - main[k]["dist"] for k in held]
        if len(diffs) >= 8:
            w, p = stats.wilcoxon(np.asarray(diffs), alternative="greater")
        else:
            w, p = 0.0, 1.0
        # user-level clustered bootstrap
        user_diffs = defaultdict(list)
        for k in held:
            user_diffs[k[0]].append(ctrl[k]["dist"] - main[k]["dist"])
        users = sorted(user_diffs)
        rng = np.random.default_rng(args.seed)
        boot = []
        for _ in range(2000):
            sample = rng.choice(users, size=len(users), replace=True)
            boot.append(float(np.mean([np.mean(user_diffs[u]) for u in sample])))
        boot = np.asarray(boot)
        res = {
            "n_paired_heldout": len(held),
            "n_users_heldout": len(users),
            "mean_diff": float(np.mean(diffs)),
            "wilcoxon_p": float(p),
            "bootstrap_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        }
        out.setdefault("paired_tests", {})[f"{main_label}_vs_{control_label}"] = res
        log(f"  held-out {main_label} vs {control_label}: n={len(held)} users={len(users)} "
            f"mean_diff={np.mean(diffs):.4f} p={p:.4f} CI=[{np.percentile(boot,2.5):.4f},{np.percentile(boot,97.5):.4f}]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "heldout_style.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {out_dir / 'heldout_style.json'}")
    log("=== done ===")


if __name__ == "__main__":
    main()
