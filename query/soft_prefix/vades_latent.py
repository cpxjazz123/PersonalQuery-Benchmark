"""E13-A — Shared VADES latent-space encoding (representation alignment).

Bug being fixed (#12/#13): the previous y+/y- selection and the style
evaluation compared *standardized raw 20-dim clause features* directly against
the VADES ``user_mu``. Same dimensionality does NOT imply the same latent
space: ``user_mu`` lives in the VADES sentence-encoder latent space while raw
features live in the input feature space. This module is the ONE shared path
``encode_query_to_vades_latent(query) -> candidate_mu`` used by BOTH data
selection and evaluation (I-A2), backed by the frozen VADES sentence encoder +
the StandardScaler saved at VADES training time (I-A1 artifact contract).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from train_vades_lite_sentence_latent_threshold import (  # noqa: E402
    SentenceEncoder,
    SentenceEncoderStudentT,
)

from user_style_vectors import (  # noqa: E402
    load_feature_scaler,
    FEATURE_KEYS,
)

CLAUSE_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"


class VadesLatentEncoder:
    """Frozen VADES sentence encoder + scaler for one category.

    Artifact contract (V-A1): fail-fast unless encoder.pt, user table, feature
    scaler, feature order and latent dim all agree with the expected schema.
    """

    def __init__(self, category: str, tag: str = "vades_lite_sentence_user_distribution_train10_holdout10"):
        self.category = category
        self.tag = tag
        base = CLAUSE_DIR / category

        encoder_path = base / "vades_encoder.pt"
        user_table_path = base / "vades_user_table.pt"
        scaler_path = base / f"{tag}_feature_scaler.json"
        missing = [p for p in (encoder_path, user_table_path, scaler_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"VADES artifact contract failed for {category}: missing {missing}"
            )

        enc_ckpt = torch.load(encoder_path, map_location="cpu")
        self.input_dim = int(enc_ckpt["input_dim"])
        self.hidden_dim = int(enc_ckpt["hidden_dim"])
        self.latent_dim = int(enc_ckpt["latent_dim"])
        self.covariance_mode = enc_ckpt["covariance_mode"]
        self.encoder_dist = enc_ckpt["encoder_dist"]
        if self.latent_dim != 20:
            raise ValueError(f"latent_dim must be 20, got {self.latent_dim}")
        if self.input_dim != len(FEATURE_KEYS):
            raise ValueError(f"encoder input_dim {self.input_dim} != {len(FEATURE_KEYS)} features")

        if self.encoder_dist == "student_t":
            encoder = SentenceEncoderStudentT(self.input_dim, self.hidden_dim, self.latent_dim)
        else:
            encoder = SentenceEncoder(self.input_dim, self.hidden_dim, self.latent_dim)
        sd = enc_ckpt["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in sd):
            # torch.compile during VADES training prefixes state keys.
            sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        encoder.load_state_dict(sd)
        encoder.eval()
        self.encoder = encoder

        scaler = load_feature_scaler(category, tag)
        if list(scaler["feature_names"]) != FEATURE_KEYS:
            raise ValueError(f"feature order mismatch for {category}")
        self.feature_names = list(scaler["feature_names"])
        self.mean = np.asarray(scaler["mean"], dtype=np.float32)
        self.scale = np.asarray(scaler["scale"], dtype=np.float32)

        self.artifact_hash = hashlib.sha256(
            json.dumps(
                {
                    "encoder": hashlib.sha256(Path(encoder_path).read_bytes()).hexdigest(),
                    "scaler": hashlib.sha256(Path(scaler_path).read_bytes()).hexdigest(),
                    "feature_names": self.feature_names,
                    "latent_dim": self.latent_dim,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]

    @torch.no_grad()
    def encode_feature_vec(self, features: np.ndarray) -> np.ndarray:
        """features: [20] or [N,20] raw clause features -> latent mu [20] or [N,20]."""
        feats = np.asarray(features, dtype=np.float32)
        single = feats.ndim == 1
        if single:
            feats = feats[None, :]
        scaled = (feats - self.mean) / np.maximum(self.scale, 1e-9)
        x = torch.tensor(scaled, dtype=torch.float32)
        mu = self.encoder(x)[0]  # encoder returns (mu, logvar, ...)
        out = mu.numpy().astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("encoder produced non-finite latent")
        return out[0] if single else out

    def encode_query_to_vades_latent(self, query: str, nlp) -> np.ndarray:
        """Shared entry point: query text -> candidate_mu (20-dim latent)."""
        from extract_clause_features_single_query import extract_clause_features
        feats = {k: float(extract_clause_features(query).get(k, 0.0)) for k in FEATURE_KEYS}
        vec = np.asarray([feats[k] for k in FEATURE_KEYS], dtype=np.float32)
        return self.encode_feature_vec(vec)

    def manifest(self) -> dict:
        return {
            "category": self.category,
            "tag": self.tag,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "covariance_mode": self.covariance_mode,
            "encoder_dist": self.encoder_dist,
            "artifact_hash": self.artifact_hash,
            "feature_names": self.feature_names,
        }
