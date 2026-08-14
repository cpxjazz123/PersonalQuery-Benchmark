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

    def __init__(self, category: str, k: int = 3, seed: int = 42, distinctive: bool = True):
        self.category = category
        self.k = k
        self.distinctive = distinctive
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
        self._global_dist: Dict[str, float] = {}
        if distinctive:
            from genre_bridge import build_global_skeleton_distribution
            all_sentences = [r["sentence_text"] for ss in self.sentences_by_user.values() for r in ss]
            self._global_dist = build_global_skeleton_distribution(all_sentences)
        self._cache: Dict[Tuple[str, int], List[str]] = {}

    def exemplars_for(self, user_id: str, k: Optional[int] = None) -> List[str]:
        """Style exemplars for a user.

        distinctive=True: the user's sentences whose genre-bridged skeleton is
        rarest globally (maximizes cross-user distinguishability).
        distinctive=False: sentences nearest the user's VADES center (latent).
        """
        k = k or self.k
        key = (user_id, k)
        if key in self._cache:
            return self._cache[key]
        sentences = [r["sentence_text"] for r in self.sentences_by_user.get(user_id, [])]
        if not sentences:
            self._cache[key] = []
            return []
        if self.distinctive:
            from genre_bridge import select_distinctive_sentences
            out = select_distinctive_sentences(sentences, self._global_dist, k=k)
        else:
            profile = self.profiles.get(user_id)
            if profile is None:
                self._cache[key] = []
                return []
            mu = profile["user_mu"].astype(np.float32)
            scored = []
            for r in self.sentences_by_user[user_id]:
                feats = np.asarray([float(r["features"][fk]) for fk in FEATURE_KEYS], dtype=np.float32)
                s_mu = self.latent_encoder.encode_feature_vec(feats)
                scored.append((float(np.linalg.norm(s_mu - mu)), r["sentence_text"]))
            scored.sort(key=lambda t: t[0])
            out = [t for _, t in scored[:k]]
        self._cache[key] = out
        return out

    def user_skeleton(self, user_id: str) -> Optional[str]:
        """The user's own bridged query skeleton (most distinctive exemplar).
        Product-agnostic: contains only the five placeholders."""
        exs = self.exemplars_for(user_id, k=1)
        if not exs:
            return None
        from genre_bridge import bridge_review_to_query_skeleton
        return bridge_review_to_query_skeleton(exs[0])

    def manifest(self) -> dict:
        return {
            "category": self.category,
            "k": self.k,
            "distinctive": self.distinctive,
            "n_users_with_sentences": len(self.sentences_by_user),
            "n_users_with_profiles": len(self.profiles),
            "selection": ("genre-bridged distinctive skeleton" if self.distinctive
                          else "VADES latent distance to user_mu (shared encoder)"),
        }
