"""E14 — Statistic-driven style skeletons.

Design (user requirement): style is derived ONLY from the user's SYNTACTIC
STATISTICS (per-user means of the 20 clause features over their review
sentences) — no review sentences are ever injected into prompts or used as
targets. The per-user statistic vector deterministically selects a query-style
skeleton (opener, clause type, coordination, modifier level) from a template
library. Style is therefore interpretable (which statistic drove which
skeleton), reproducible (pure function of statistics + user id), and fully
decoupled from product content (skeletons contain only <A*> placeholders).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))

from genre_bridge import SKELETONS  # noqa: E402  (template library)

FEATURES20 = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]

# Deterministic per-user opener variants (user-identity factor, content-free).
ADV_OPENERS = [
    "Occasionally", "Usually", "Honestly", "Personally", "Actually", "Definitely",
]


def load_user_feature_means(category: str) -> Dict[str, np.ndarray]:
    """Per-user mean of the 20 clause features over their review sentences."""
    p = (
        REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
        / category / "vades_lite_sentence_user_distribution_train10_holdout10_sentences.jsonl"
    )
    bucket: Dict[str, List[List[float]]] = defaultdict(list)
    for line in open(p):
        row = json.loads(line)
        bucket[row["user_id"]].append([float(row["features"][k]) for k in FEATURES20])
    return {uid: np.mean(v, axis=0).astype(np.float32) for uid, v in bucket.items()}


def stat_to_skeleton(feat: np.ndarray, user_id: str, seed: int = 0) -> str:
    """Deterministic statistic -> skeleton mapping.

    Thresholds follow the Baby_Products user-level quantiles (q75) so that
    roughly a quarter of users fall into each structure class.
    """
    d = {k: float(feat[i]) for i, k in enumerate(FEATURES20)}
    coordination = d["coordination_count"]
    advcl = d["advcl_count"]
    relcl = d["relcl_count"]
    modifier = d["modifier_density"] + 0.05 * d["amod_count"]
    depth = d["max_dependency_depth"]

    if relcl >= 0.20:
        idx = 2          # which-relative
    elif advcl >= 0.47 and coordination >= 1.5:
        idx = 6          # because-clause
    elif advcl >= 0.47:
        idx = 8          # when-clause
    elif coordination >= 1.5:
        idx = 1          # and-conjoined
    elif modifier >= 0.15:
        idx = 0          # that-relative with dense modifiers
    elif depth >= 5.8:
        idx = 5          # gerund
    else:
        idx = 10         # simple declarative

    h = hashlib.sha256(f"{user_id}:{seed}".encode()).hexdigest()
    opener = ADV_OPENERS[int(h[:8], 16) % len(ADV_OPENERS)]
    return SKELETONS[idx]["template"].format(opener=f"{opener}, ")


def stat_to_skeleton_for_user(user_feat_means: Dict[str, np.ndarray], user_id: str, seed: int = 0) -> Optional[str]:
    if user_id not in user_feat_means:
        return None
    return stat_to_skeleton(user_feat_means[user_id], user_id, seed=seed)


def skeleton_stats(skeleton: str) -> Dict[str, object]:
    """Which statistics would produce this skeleton (interpretability)."""
    return {
        "opener": skeleton.split(",", 1)[0] if "," in skeleton else "",
        "has_because": "because" in skeleton,
        "has_when": "when I need" in skeleton,
        "has_which": "which" in skeleton,
        "has_and": " and " in skeleton,
        "n_placeholders": sum(skeleton.count(f"<A{i}>") for i in range(1, 6)),
    }
