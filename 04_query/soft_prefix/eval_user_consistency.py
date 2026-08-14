#!/usr/bin/env python3
"""E14 — User-consistency & distinguishability evaluation (hardcoded config).

For the multi-product copy-aware generations:
- SAME user across products: syntactic distance should be SMALL (style stable)
- DIFFERENT users: syntactic distance should be LARGE (style distinguishable)
Distinguishability ratio = mean(inter-user) / mean(intra-user).
All parameters hardcoded (CLAUDE.md rule 9).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))

from extract_clause_features_single_query import (  # noqa: E402
    load_spacy_model,
    extract_clause_features,
)

GEN_FILE = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/e14p_sv/gen_multi/Baby_Products_copyaware_vec=vades.json")
OUT_FILE = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/e14p_sv/user_consistency.json")
SEED = 0
FEATURES = ["max_dependency_depth", "mean_dependency_depth", "relcl_count", "advcl_count",
            "acl_count", "clause_nesting_depth", "modifier_density", "coordination_count",
            "amod_count", "advmod_count"]
MAX_OTHER_SAMPLES = 3


def feature_vec(query: str, nlp) -> np.ndarray:
    f = extract_clause_features(query)
    return np.asarray([float(f.get(k, 0.0)) for k in FEATURES], dtype=np.float32)


def main() -> None:
    data = json.load(open(GEN_FILE))
    results = data["results"]
    print(f"total generations: {len(results)}")
    nlp = load_spacy_model()
    by_user: dict = defaultdict(list)
    for r in results:
        q = r.get("generated_query") or ""
        if q:
            by_user[r["user_id"]].append(feature_vec(q, nlp))
    users = [u for u in by_user if len(by_user[u]) >= 2]
    print(f"users with >=2 products: {len(users)} / {len(by_user)}")

    same = []
    for uid in users:
        vecs = by_user[uid]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                same.append(float(np.linalg.norm(vecs[i] - vecs[j])))

    rng = np.random.default_rng(SEED)
    diff = []
    for uid in users:
        others = rng.choice([u for u in by_user if u != uid],
                            size=min(MAX_OTHER_SAMPLES, len(by_user) - 1), replace=False)
        for o in others:
            diff.append(float(np.linalg.norm(by_user[uid][0] - by_user[o][0])))

    ratio = float(np.mean(diff)) / float(np.mean(same)) if same and diff else None
    summary = {
        "n_total": len(results),
        "n_users_multi_product": len(users),
        "intra_user_dist_mean": float(np.mean(same)) if same else None,
        "intra_user_dist_std": float(np.std(same)) if same else None,
        "inter_user_dist_mean": float(np.mean(diff)) if diff else None,
        "inter_user_dist_std": float(np.std(diff)) if diff else None,
        "distinguishability_ratio": ratio,
    }
    print(json.dumps(summary, indent=2))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT_FILE, "w"), indent=2)
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
