"""E11 — User style latent vector injection: VADES -> soft prefix.

Shared helpers: user style vector loading/aggregation, soft-prefix projector,
content validation, and user/time split manifests.

Main route: VADES 20-dim `user_mu` per user (single-Gaussian diagonal VADES).
E5 mean is kept as a control route only (B4).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
VADES_TAG = "vades_lite_sentence_user_distribution_train10_holdout10"
FEATURE_KEYS = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]
assert len(FEATURE_KEYS) == 20

VADES_LATENT_DIM = 20


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector; zero vector stays zero (no NaN)."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        return vec / norm
    return np.zeros_like(vec)


def load_vades_profiles(category: str, tag: str = VADES_TAG) -> Dict[str, Dict[str, np.ndarray]]:
    """Load user_mu/user_logvar from VADES user profiles jsonl."""
    p = (
        REPO_ROOT
        / "result" / "personal_query" / "12_complexity_analysis_clause_features"
        / category / f"{tag}_user_profiles.jsonl"
    )
    profiles: Dict[str, Dict[str, np.ndarray]] = {}
    with open(p) as f:
        for line in f:
            row = json.loads(line)
            if "user_mu" not in row:
                continue
            profiles[row["user_id"]] = {
                "user_mu": np.asarray(row["user_mu"], dtype=np.float32),
                "user_logvar": np.asarray(row["user_logvar"], dtype=np.float32),
            }
    return profiles


def load_feature_scaler(category: str, tag: str = VADES_TAG) -> Dict:
    """Load the StandardScaler used to standardize clause features in VADES."""
    p = (
        REPO_ROOT
        / "result" / "personal_query" / "12_complexity_analysis_clause_features"
        / category / f"{tag}_feature_scaler.json"
    )
    with open(p) as f:
        return json.load(f)


def load_candidate_feature_rows(category: str) -> List[dict]:
    """Load per-candidate 20-dim clause features (10 candidates per record).

    Returns [] when the file has not been built yet (caller may build it).
    """
    p = (
        REPO_ROOT
        / "result" / "personal_query" / "12_complexity_analysis_clause_features"
        / category / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
    )
    if not p.exists():
        return []
    rows: List[dict] = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def global_mean_vector(profiles: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    """Global mean of L2-normalized user_mu vectors (itself re-normalized)."""
    mus = np.stack([p["user_mu"] for p in profiles.values()])
    return l2_normalize(mus.mean(axis=0))


class UserVectorProvider:
    """Deterministic per-user style vector provider.

    Modes:
      none        (B1)  no prefix at all (no user conditioning)
      zero        (B3)  zero vector (conditioning ablated)
      global_mean (B2)  global mean of user_mu (conditioning ablated)
      shuffled    (B3)  another user's vector via fixed seeded permutation
      vades       (B5)  L2-normalized VADES user_mu (main route)
      e5_mean     (B4)  L2-normalized mean of E5 review embeddings (control)

    Missing users (no VADES profile) get the zero vector with an explicit
    ``has_vector=False`` flag so cold-start conditioning is recorded, not
    silently randomized. All vectors are L2-normalized after aggregation.
    """

    MODES = {"none", "zero", "global_mean", "shuffled", "vades", "e5_mean"}

    def __init__(
        self,
        mode: str,
        profiles: Dict[str, Dict[str, np.ndarray]],
        user_ids: List[str],
        seed: int = 42,
        e5_vecs: Optional[Dict[str, np.ndarray]] = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}, must be one of {sorted(self.MODES)}")
        self.mode = mode
        self.profiles = profiles
        self.user_ids = list(user_ids)
        self.seed = seed
        self.e5_vecs = e5_vecs or {}
        self.permutation: Dict[str, str] = {}
        if mode == "shuffled":
            rng = np.random.default_rng(seed)
            available = sorted(set(self.profiles.keys()))
            perm = rng.permutation(len(available))
            self.permutation = {
                src: available[int(perm[i])] for i, src in enumerate(available)
            }
        self._global_mean = global_mean_vector(profiles)

    @property
    def vector_dim(self) -> int:
        if self.mode == "e5_mean":
            return 1024
        return VADES_LATENT_DIM

    def get(self, user_id: str) -> Tuple[Optional[np.ndarray], bool]:
        """Return (vector, has_vector). vector is None only for mode 'none'."""
        if self.mode == "none":
            return None, False
        if self.mode == "zero":
            return np.zeros(self.vector_dim, dtype=np.float32), False
        if self.mode == "global_mean":
            return self._global_mean.astype(np.float32), True
        if self.mode == "shuffled":
            src = self.permutation.get(user_id)
            if src is None or src not in self.profiles:
                return np.zeros(self.vector_dim, dtype=np.float32), False
            return l2_normalize(self.profiles[src]["user_mu"]).astype(np.float32), True
        if self.mode == "vades":
            p = self.profiles.get(user_id)
            if p is None:
                return np.zeros(self.vector_dim, dtype=np.float32), False
            return l2_normalize(p["user_mu"]).astype(np.float32), True
        if self.mode == "e5_mean":
            vec = self.e5_vecs.get(user_id)
            if vec is None:
                return np.zeros(self.vector_dim, dtype=np.float32), False
            return l2_normalize(vec).astype(np.float32), True
        raise AssertionError(f"unhandled mode {self.mode}")

    def manifest(self) -> dict:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "vector_dim": self.vector_dim,
            "permutation": self.permutation if self.mode == "shuffled" else {},
            "n_profiles": len(self.profiles),
        }
