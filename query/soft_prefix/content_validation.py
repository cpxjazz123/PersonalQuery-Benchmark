"""Content validation and y+/y- selection for E11 SFT data construction.

Rules (issue #11):
- prompt must contain the five product attributes explicitly;
- supervision target must NOT be the fixed first candidate (queries[0]);
- y+ = content-valid candidate closest to the user's VADES center;
- y- = content-valid candidate farthest from the user's VADES center
  (kept for the optional preference stage);
- if the VADES-retained query is content-valid it is preferred as y+
  (teacher distillation of the old rerank flow);
- records without any content-valid candidate are skipped and recorded.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query"))
from common.attribute_helpers import (  # noqa: E402
    validate_query_uses_exactly_five_attrs,
)

from user_style_vectors import FEATURE_KEYS  # noqa: E402
from template_placeholder import (  # noqa: E402
    replace_attrs_with_placeholders,
)


def build_attr_prompt(attrs_used: Dict[str, str]) -> str:
    """Explicit five-attribute prompt shared by training and generation."""
    lines = [
        "Product attributes:",
    ]
    for key in sorted(attrs_used):
        lines.append(f"- {key}: {attrs_used[key]}")
    lines.append(
        "Write one natural shopping query that uses every attribute exactly once."
    )
    return "\n".join(lines)


def templateize_query(query: str, attrs_used: Dict[str, str]):
    """E12: convert a content-valid query into a placeholder template.

    Returns (template, None) or (None, reason). Values that cannot be
    losslessly templated are skipped at data-build time (recorded), never
    trained on in raw form.
    """
    return replace_attrs_with_placeholders(query, attrs_used)


def content_valid(query: str, attrs_used: Dict[str, str]) -> bool:
    ok, _ = validate_query_uses_exactly_five_attrs(query, attrs_used)
    return ok


def _feature_vec(candidate: dict) -> np.ndarray:
    feat = candidate.get("features", {})
    return np.asarray(
        [float(feat.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32
    )


class StyleDatasetBuilder:
    """Builds (attrs, y+, y-) per record using VADES distances.

    Distances are computed in the same standardized space the VADES sentence
    encoder was trained in (scaler saved by the VADES training script).
    """

    def __init__(
        self,
        category: str,
        vades_profiles: Dict[str, Dict[str, np.ndarray]],
        scaler: dict,
        seed: int = 42,
        latent_encoder=None,
    ):
        self.category = category
        self.profiles = vades_profiles
        self.scaler = scaler
        self.feature_names = list(scaler["feature_names"])
        self.mean = np.asarray(scaler["mean"], dtype=np.float32)
        self.scale = np.asarray(scaler["scale"], dtype=np.float32)
        if len(self.feature_names) != len(FEATURE_KEYS) or self.feature_names != FEATURE_KEYS:
            raise ValueError("scaler feature_names mismatch FEATURE_KEYS")
        self.rng = np.random.default_rng(seed)
        if latent_encoder is None:
            # E13-A: default to the shared VADES latent encoder; the raw
            # cross-space path must NOT be used for distance.
            from vades_latent import VadesLatentEncoder
            latent_encoder = VadesLatentEncoder(category)
        self.latent_encoder = latent_encoder

    def distance_to_user(self, candidate: dict, user_id: str) -> Optional[float]:
        """VADES LATENT-space euclidean distance between the candidate query
        and the user's user_mu (E13-A). Both live in the sentence-encoder
        latent space, so the distance is meaningful (unlike the removed
        standardized-raw-features vs user_mu path)."""
        profile = self.profiles.get(user_id)
        if profile is None:
            return None
        mu = profile["user_mu"].astype(np.float32)
        cand_mu = self.latent_encoder.encode_feature_vec(_feature_vec(candidate))
        return float(np.linalg.norm(cand_mu - mu))

    def build_record(
        self,
        record: dict,
        candidates: List[dict],
        retained_query: Optional[str] = None,
        template_targets: bool = True,
    ) -> Optional[dict]:
        """Return data row or None (record skipped, logged via reason).

        With ``template_targets=True`` (E12) every supervision target is the
        placeholder template of a content-valid candidate; targets that
        cannot be losslessly templated are skipped and the reason recorded.
        """
        user_id = record["user_id"]
        asin = record["asin"]
        valid = []
        for cand in candidates:
            attrs_used = cand.get("attrs_used")
            query = cand.get("query", "")
            if not attrs_used or not query:
                continue
            if not content_valid(query, attrs_used):
                continue
            valid.append(cand)
        if not valid:
            return None
        # Canonical five attributes: first content-valid candidate's attrs.
        attrs_used = valid[0]["attrs_used"]
        # Re-validate every candidate against the canonical attribute set so
        # all distances are scored under the same prompt condition.
        canonical_valid = [
            c for c in candidates if content_valid(c.get("query", ""), attrs_used)
        ]
        if not canonical_valid:
            return None

        scored: List[Tuple[float, dict]] = []
        for cand in canonical_valid:
            dist = self.distance_to_user(cand, user_id)
            if dist is None:
                continue
            scored.append((dist, cand))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0])

        y_minus_candidate = scored[-1][1]
        y_plus_candidate = scored[0][1]
        y_plus_query = y_plus_candidate["query"]
        y_plus_index = canonical_valid.index(y_plus_candidate)
        # Prefer the VADES-retained query as the teacher target when it is
        # content-valid under the canonical attributes.
        if retained_query and content_valid(retained_query, attrs_used):
            y_plus_query = retained_query
            for cand in canonical_valid:
                if cand.get("query") == retained_query:
                    y_plus_index = canonical_valid.index(cand)
                    break
        y_plus_template = y_plus_query
        if template_targets:
            y_plus_template, tpl_err = templateize_query(y_plus_query, attrs_used)
            if y_plus_template is None:
                return None  # recorded as skipped below (caller adds reason)
        return {
            "user_id": user_id,
            "asin": asin,
            "attrs_used": attrs_used,
            "y_plus_query": y_plus_template if template_targets else y_plus_query,
            "y_plus_raw_query": y_plus_query if template_targets else None,
            "y_plus_index": int(y_plus_index),
            "y_minus_query": y_minus_candidate["query"],
            "y_minus_index": int(canonical_valid.index(y_minus_candidate)),
            "y_plus_vades_dist": float(scored[0][0]),
            "y_minus_vades_dist": float(scored[-1][0]),
            "n_content_valid_candidates": len(canonical_valid),
            "is_template_target": bool(template_targets),
        }

    def build_all(
        self,
        records: List[dict],
        candidate_rows: List[dict],
        retained_by_key: Dict[Tuple[str, str], Optional[str]],
        supervision: str = "primary",
        template_targets: bool = True,
        topk_temperature: float = 1.0,
    ) -> Tuple[List[dict], List[dict]]:
        """Build data rows for all records; returns (rows, skipped).

        E13-B supervision strategies (I-B3):
          primary        (default) each (user_id, asin) has exactly ONE SFT
                         target: the content-valid candidate nearest the user
                         in VADES latent space (or the retained teacher).
          all_candidates every content-valid candidate is an equal-weight SFT
                         target (the old #11/E12 behavior).
          topk_weighted  every content-valid candidate is a target but each is
                         weighted by softmax(-latent_distance / T) so the
                         user-near candidates dominate.

        With ``template_targets=True`` (E12) all targets are placeholder
        templates; non-templatable targets are skipped with a recorded reason.
        """
        by_key: Dict[Tuple[str, str], List[dict]] = {}
        for cand in candidate_rows:
            by_key.setdefault((cand["user_id"], cand["asin"]), []).append(cand)
        rows: List[dict] = []
        skipped: List[dict] = []
        for rec in records:
            key = (rec["user_id"], rec["asin"])
            cands = by_key.get(key, [])
            if not cands:
                skipped.append({**rec, "reason": "no_candidate_features"})
                continue
            retained = retained_by_key.get(key)
            primary = self.build_record(rec, cands, retained_query=retained, template_targets=template_targets)
            if primary is None:
                skipped.append({**rec, "reason": "no_templatable_y_plus"})
                continue
            primary = dict(primary)
            primary["is_primary"] = True
            primary["weight"] = 1.0
            rows.append(primary)
            if supervision in ("all_candidates", "topk_weighted"):
                attrs = primary["attrs_used"]
                dists: List[Tuple[float, dict]] = []
                extras = []
                for cand in cands:
                    query = cand.get("query", "")
                    if not content_valid(query, attrs):
                        continue
                    dist = self.distance_to_user(cand, rec["user_id"])
                    if dist is None:
                        continue
                    target = query
                    if template_targets:
                        target, tpl_err = templateize_query(query, attrs)
                        if target is None:
                            skipped.append({
                                "user_id": rec["user_id"], "asin": rec["asin"],
                                "candidate_query": query[:80],
                                "reason": f"not_templatable: {tpl_err}",
                            })
                            continue
                    extras.append({
                        "user_id": rec["user_id"],
                        "asin": rec["asin"],
                        "attrs_used": attrs,
                        "y_plus_query": target,
                        "y_plus_raw_query": query if template_targets else None,
                        "y_plus_index": int(cands.index(cand)),
                        "y_minus_query": primary["y_minus_query"],
                        "y_minus_index": primary["y_minus_index"],
                        "y_plus_vades_dist": float(dist),
                        "y_minus_vades_dist": primary["y_minus_vades_dist"],
                        "n_content_valid_candidates": primary["n_content_valid_candidates"],
                        "is_primary": False,
                        "is_template_target": bool(template_targets),
                        "dist": float(dist),
                    })
                if supervision == "topk_weighted":
                    d = np.asarray([e["dist"] for e in extras], dtype=np.float64)
                    w = np.exp(-d / max(topk_temperature, 1e-9))
                    w = w / w.sum()
                    for e, wi in zip(extras, w):
                        e["weight"] = float(wi)
                else:
                    for e in extras:
                        e["weight"] = 1.0
                rows.extend(extras)
        return rows, skipped
