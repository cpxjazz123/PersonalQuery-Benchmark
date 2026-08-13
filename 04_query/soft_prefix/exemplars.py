"""E14 — Exemplar-primed style generation: user review sentences as style
exemplars.

Core idea (design proposal adopted): abstract 20-dim vector conditioning was
shown (#11/#12/#13) to change templates but not align to the user's style
center. LLMs mimic concrete exemplars much better than abstract vectors, so
the style condition becomes the user's OWN review sentences: deterministic,
user-specific, and obviously style-bearing.

Selection: encode every review sentence of the user into the VADES latent
space (shared encoder from vades_latent.py), rank by latent distance to the
user's own user_mu, and take the k nearest sentences as style exemplars.
Target queries never enter the exemplar pool (different source; checked by
construction).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))

from user_style_vectors import (  # noqa: E402
    load_vades_profiles,
    FEATURE_KEYS,
)
from vades_latent import VadesLatentEncoder  # noqa: E402

CLAUSE_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
TAG = "vades_lite_sentence_user_distribution_train10_holdout10"


class ExemplarProvider:
    """Per-user style exemplars (review sentences nearest the user's VADES
    center in latent space). Cached per category."""

    def __init__(self, category: str, k: int = 3, seed: int = 42):
        self.category = category
        self.k = k
        self.rng = np.random.default_rng(seed)
        sentences_p = CLAUSE_DIR / category / f"{TAG}_sentences.jsonl"
        if not sentences_p.exists():
            raise FileNotFoundError(f"sentence file missing: {sentences_p}")
        self.sentences_by_user: Dict[str, List[dict]] = {}
        for line in open(sentences_p):
            row = json.loads(line)
            self.sentences_by_user.setdefault(row["user_id"], []).append(row)
        self.profiles = load_vades_profiles(category)
        self.latent_encoder = VadesLatentEncoder(category)
        self._cache: Dict[Tuple[str, int], List[str]] = {}

    def exemplars_for(self, user_id: str, k: Optional[int] = None) -> List[str]:
        """Top-k review sentences nearest the user's VADES center (latent)."""
        k = k or self.k
        key = (user_id, k)
        if key in self._cache:
            return self._cache[key]
        profile = self.profiles.get(user_id)
        sentences = self.sentences_by_user.get(user_id, [])
        if profile is None or not sentences:
            self._cache[key] = []
            return []
        mu = profile["user_mu"].astype(np.float32)
        scored: List[Tuple[float, str]] = []
        for s in sentences:
            feats = np.asarray([float(s["features"][fk]) for fk in FEATURE_KEYS], dtype=np.float32)
            s_mu = self.latent_encoder.encode_feature_vec(feats)
            scored.append((float(np.linalg.norm(s_mu - mu)), s["sentence_text"]))
        scored.sort(key=lambda t: t[0])
        out = [text for _, text in scored[:k]]
        self._cache[key] = out
        return out

    def manifest(self) -> dict:
        return {
            "category": self.category,
            "k": self.k,
            "n_users_with_sentences": len(self.sentences_by_user),
            "n_users_with_profiles": len(self.profiles),
            "selection": "VADES latent distance to user_mu (shared encoder)",
        }
