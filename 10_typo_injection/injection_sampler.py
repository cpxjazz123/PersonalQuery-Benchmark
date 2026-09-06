"""Injection Sampler — Bernoulli sampling + single typo application + Gaussian gate.

For each (uid, source query):
  1. Tokenize query with spaCy → get multi-level structural context per token
  2. For each token, per-context Bernoulli(p_u(error | sig)) decides whether it
     gets a typo (single-shot, hierarchy fallback full → d4 → d3 → clause →
     depth → sibling → coarse → p_u_err → DEFAULT)
  3. If at least one token fires, pick the one with highest per-context rate
  4. Apply mechanism (sampled from user's mechanism_totals; restricted to
     INJECTABLE_MECHANISMS — semantic_substitution is excluded)
  5. Verify **full Gaussian constraint** in 32d supervised syntax space:
       D²(z_after, μ_u) = (z_after - μ_u)ᵀ Σ_u⁻¹ (z_after - μ_u) ≤ D²_threshold
  6. Verify **semantic similarity constraint**: cosine(MiniLM(orig), MiniLM(typo)) ≥ 0.9
     (preserves user intent — typos should NOT shift semantics)

Encoding pipelines (matches `08_select_query/syntax_select_mahalanobis_gate.py`):
  Syntax 32d:  text → spaCy doc → extract_struct_rules (21737 binary count)
                  → normalize (x/(1+x)) → frozen _SupEncoder → z (32d)
  Semantic:    text → sentence-transformers/all-MiniLM-L6-v2 → e (384d)
                  → cosine similarity
"""
from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy
import torch

from error_location import (
    _context_signature, _ensure_spacy, _word_shape,
    _rich_signatures, _default_rate_hierarchical,
)
from typo_classifier import reverse_mechanism

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
PCFG_PIPELINE = REPO_ROOT / "03_spacy_encode/syntax_pcfg_pipeline.py"
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
ENCODER_PT = CACHE_DIR / "adaptive_encoder.pt"
ENCODER_DEVICE = "cpu"  # 32d encode is trivial; CPU is fine

# Semantic similarity gate (MiniLM bi-encoder, 384d, cosine ≥ 0.9)
SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.9
SEMANTIC_CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/hf_cache")
_semantic_encoder = None

# Mechanism set used for sampling (semantic_substitution excluded — would require
# rephrasing semantics, which would corrupt user style; only surface-level
# character transformations are allowed)
#
# Taxonomy:
#   Surface-form errors (capitalization/punctuation, NOT traditional typo):
#     - case_error       (Apple ↔ apple)
#     - apostrophe_error (don't ↔ dont)
#   Typo / character-level errors (传统 typo):
#     - keyboard_adjacent (teh → the)
#     - letter_swap       (prodcut → product)
#     - letter_repetition (reallly → really)
#
# Total category: writing-error mechanisms (surface-form + typo)
INJECTABLE_MECHANISMS = (
    "keyboard_adjacent",
    "letter_swap",
    "letter_repetition",
    "case_error",
    "apostrophe_error",
)

# Two-category breakdown for paper reporting
TYPO_MECHANISMS = ("keyboard_adjacent", "letter_swap", "letter_repetition")
SURFACE_FORM_MECHANISMS = ("case_error", "apostrophe_error")

DEFAULT_P_ERROR = 0.02  # fallback when context unseen


@dataclass
class InjectionMeta:
    position: int            # 0-indexed position in the query
    original_token: str
    typo_token: str
    mechanism: str
    mechanism_category: str   # "typo" or "surface_form" or "none"
    confidence: float        # w_i (per-context error rate)
    d_mahalanobis_before: float  # full Mahalanobis D²(z, μ_u) using Σ_u⁻¹
    d_mahalanobis_after: float   # same, on injected query
    gaussian_pass: bool      # True iff D²_after ≤ user threshold
    semantic_sim: float = 1.0  # cosine(MiniLM(orig), MiniLM(typo))
    semantic_pass: bool = True # True iff semantic_sim ≥ SEMANTIC_THRESHOLD
    min_d_competitor: float = -1.0  # min d² to any competitor (for audit)
    n_competitors: int = 0         # number of competitors checked
    exclusive_pass: bool = True    # True iff z_after outside ALL competitors' Q_95 cores


# Lazy-loaded encoder + rule_to_id
_encoder = None
_rule_to_id: Dict[str, int] = {}


def _load_pcfg_module():
    spec = importlib.util.spec_from_file_location(
        "syntax_pcfg_pipeline", str(PCFG_PIPELINE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_encoder():
    """Load _SupEncoder from adaptive_encoder.pt (must eval)."""
    global _encoder
    if _encoder is None:
        ckpt = torch.load(ENCODER_PT, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        _pcfg = _load_pcfg_module()
        model = _pcfg._SupEncoder(
            cfg["vocab_size"], cfg["z_dim"], tuple(cfg["hidden"]),
            cfg["n_users"], cfg["dropout"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _encoder = model
    return _encoder


def _load_rule_to_id() -> Dict[str, int]:
    """Load vocab.json (list of 21737 rule strings) → {rule_str: id}."""
    global _rule_to_id
    if not _rule_to_id:
        with open(CACHE_DIR / "vocab.json") as f:
            vocab_list = json.load(f)
        _rule_to_id = {r: i for i, r in enumerate(vocab_list)}
    return _rule_to_id


def _encode_queries_32d(texts: List[str], nlp, encoder, rule_to_id: Dict[str, int],
                        vocab_size: int) -> np.ndarray:
    """texts → 32d supervised z (same pipeline as select_query)."""
    _pcfg = _load_pcfg_module()
    extract_struct_rules = _pcfg.extract_struct_rules

    n = len(texts)
    counts = np.zeros((n, vocab_size), dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        for r in extract_struct_rules(doc):
            j = rule_to_id.get(r)
            if j is not None:
                counts[i, j] = 1.0
    counts = counts / (1.0 + counts)
    with torch.no_grad():
        z, _ = encoder(torch.tensor(counts, dtype=torch.float32, device=ENCODER_DEVICE))
    return z.cpu().numpy().astype(np.float32)


def _spacy_token_contexts(query: str) -> List[Tuple]:
    """Token-level contexts with multi-level structural signatures.

    Returns list of tuples:
      (alpha_idx, word, pos, dep, parent_pos, length, shape, sigs_dict)
    where sigs_dict maps level name → sig_string (full/d4/d3/clause/depth/sibling/coarse).
    """
    nlp = _ensure_spacy()
    doc = nlp(query)
    out = []
    for tok in doc:
        if not tok.is_alpha:
            continue
        sigs = _rich_signatures(tok, doc)
        out.append((tok.i, tok.text, tok.pos_, tok.dep_, tok.head.pos_,
                    len(tok.text), _word_shape(tok.text), sigs))
    return out


def _split_query_tokens(query: str) -> List[str]:
    """Return list of whitespace-separated tokens preserving original word order."""
    return query.split(" ")


def _mahalanobis_d2(z: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> float:
    """Full Mahalanobis squared distance D²(z, μ) = (z-μ)ᵀ Σ⁻¹ (z-μ).

    Uses per-user Σ_u⁻¹ (32×32). Negative values (numerical) are clamped to 0.
    """
    diff = z - mu
    d2 = float(diff @ sigma_inv @ diff)
    return max(d2, 0.0)


def _load_semantic_encoder():
    """Lazy-load MiniLM bi-encoder for semantic similarity."""
    global _semantic_encoder
    if _semantic_encoder is None:
        os.environ.setdefault("HF_HOME", str(SEMANTIC_CACHE_DIR))
        os.environ.setdefault("HF_HUB_CACHE", str(SEMANTIC_CACHE_DIR / "hub"))
        from sentence_transformers import SentenceTransformer
        _semantic_encoder = SentenceTransformer(SEMANTIC_MODEL_ID, device="cpu")
    return _semantic_encoder


def _semantic_cosine(texts_a: List[str], texts_b: List[str]) -> np.ndarray:
    """Batch cosine similarity between paired MiniLM embeddings.

    Returns array of cosine sims, shape (n,).
    """
    enc = _load_semantic_encoder()
    ea = enc.encode(texts_a, normalize_embeddings=True,
                    batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    eb = enc.encode(texts_b, normalize_embeddings=True,
                    batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    return (ea * eb).sum(axis=1)


def _apply_typo_to_query(query: str, position: int, typo: str) -> Optional[str]:
    """Replace the position-th whitespace token in query with typo.

    Returns None if position is out of range or token is non-alpha.
    """
    tokens = _split_query_tokens(query)
    if position >= len(tokens):
        return None
    if not any(c.isalpha() for c in tokens[position]):
        return None
    tokens[position] = typo
    return " ".join(tokens)


def _sample_mechanism(model: Dict, rng: random.Random) -> str:
    """Sample a mechanism proportional to user's empirical counts.

    Sampling is restricted to INJECTABLE_MECHANISMS — semantic_substitution is
    deliberately excluded because it would require changing semantics, which is
    not a writing error and would corrupt user style (BART history confirmed).
    """
    totals = model.get("mechanism_totals", {})
    weights = {m: totals.get(m, 0) for m in INJECTABLE_MECHANISMS}
    total_w = sum(weights.values())
    if total_w == 0:
        return rng.choice(INJECTABLE_MECHANISMS)
    items = list(weights.items())
    r = rng.random() * total_w
    cum = 0.0
    for mech, w in items:
        cum += w
        if r <= cum:
            return mech
    return items[-1][0]


def _categorize(mech: str) -> str:
    if mech in TYPO_MECHANISMS:
        return "typo"
    if mech in SURFACE_FORM_MECHANISMS:
        return "surface_form"
    return "none"


def _default_rate(model: Dict, sigs) -> float:
    """Per-context error rate with hierarchical fallback.

    sigs may be either:
      - dict[level → sig_string] (preferred; full hierarchy)
      - str (legacy single sig; treated as `coarse` level)
    """
    if isinstance(sigs, str):
        sigs = {"coarse": sigs}
    return _default_rate_hierarchical(model, sigs)


import random


def sample_injection(
    query: str,
    uid: str,
    user_model: Dict,
    mu_u: np.ndarray,
    sigma_inv: np.ndarray,
    d2_threshold: float,
    competitors: Optional[Dict[str, Dict]] = None,
    seed: int = 42,
    nlp=None,
    encoder=None,
    rule_to_id: Optional[Dict[str, int]] = None,
    vocab_size: int = 21737,
) -> Tuple[Optional[str], Optional[InjectionMeta]]:
    """Apply at most one typo injection to `query` respecting user's error profile.

    Gates (all must pass to return injected):
      1. Bernoulli(per-context structural rate) → at least one token fires
      2. Apply mechanism, encode typo query in 32d supervised space
      3. d²(z_after, μ_u) ≤ user Q_95        (target inside own core)
      4. d²(z_after, μ_comp) > comp Q_95 ∀comp (typo outside ALL competitors' cores;
                                                non-shared zone, matches 08 logic)
      5. cosine(MiniLM(orig), MiniLM(typo)) ≥ 0.9 (semantic preservation)

    Returns (injected_query, meta). If any gate fails, returns (None, meta) with
    the corresponding pass flag set to False. Caller can introspect meta to see
    which gate failed.
    """
    rng = random.Random(seed)

    if nlp is None:
        nlp = _ensure_spacy()
    if encoder is None:
        encoder = _load_encoder()
    if rule_to_id is None:
        rule_to_id = _load_rule_to_id()

    # Encode original query (32d)
    feat_orig = _encode_queries_32d([query], nlp, encoder, rule_to_id, vocab_size)[0]
    d2_before = _mahalanobis_d2(feat_orig, mu_u, sigma_inv)

    # Per-token contexts
    contexts = _spacy_token_contexts(query)
    if not contexts:
        meta = InjectionMeta(
            position=-1, original_token="", typo_token="",
            mechanism="none", mechanism_category="none",
            confidence=0.0, d_mahalanobis_before=d2_before,
            d_mahalanobis_after=d2_before, gaussian_pass=False,
        )
        return None, meta

    # Per-context Bernoulli — each token uses its own P_u(error | c_i)
    # at the finest hierarchy level the user has data for (full → d4 → d3 →
    # clause → depth → sibling → coarse → p_u_err → DEFAULT). Tokens in
    # error-prone structural positions (e.g. ADVCL-headed subordinate clauses,
    # d3=DET|det|NOUN) get higher individual rates.
    candidates = []
    for ctx in contexts:
        alpha_idx, word, pos, dep, parent_pos, length, shape, sigs = ctx
        p_ctx = _default_rate(user_model, sigs)
        if rng.random() < p_ctx:
            candidates.append((alpha_idx, word, p_ctx, sigs))

    if not candidates:
        # No per-context Bernoulli hit
        meta = InjectionMeta(
            position=-1, original_token="", typo_token="",
            mechanism="none", mechanism_category="none",
            confidence=0.0, d_mahalanobis_before=d2_before,
            d_mahalanobis_after=d2_before, gaussian_pass=False,
        )
        return None, meta

    # Pick highest-weighted Bernoulli-pass candidate (single shot)
    ws_pos, word, w_i, sigs = max(candidates, key=lambda c: c[2])

    mech = _sample_mechanism(user_model, rng)
    typo = reverse_mechanism(word, mech, rng_seed=seed + ws_pos)

    # If reverse_mechanism couldn't transform (e.g. token too short / no alpha),
    # fall back to no injection
    if typo == word or typo is None:
        meta = InjectionMeta(
            position=ws_pos, original_token=word, typo_token="",
            mechanism=mech, mechanism_category=_categorize(mech),
            confidence=float(w_i), d_mahalanobis_before=d2_before,
            d_mahalanobis_after=d2_before, gaussian_pass=False,
        )
        return None, meta

    injected = _apply_typo_to_query(query, ws_pos, typo)
    if injected is None:
        meta = InjectionMeta(
            position=ws_pos, original_token=word, typo_token=typo,
            mechanism=mech, mechanism_category=_categorize(mech),
            confidence=float(w_i), d_mahalanobis_before=d2_before,
            d_mahalanobis_after=d2_before, gaussian_pass=False,
        )
        return None, meta

    # Gaussian constraint: full Mahalanobis D² vs user threshold
    feat_inj = _encode_queries_32d([injected], nlp, encoder, rule_to_id, vocab_size)[0]
    d2_after = _mahalanobis_d2(feat_inj, mu_u, sigma_inv)
    gaussian_pass = bool(d2_after <= d2_threshold)

    # Semantic similarity constraint: MiniLM cosine ≥ 0.9
    sem_sim = float(_semantic_cosine([query], [injected])[0])
    semantic_pass = bool(sem_sim >= SEMANTIC_THRESHOLD)

    # Exclusive cohort gate: typo query must be OUTSIDE every competitor's Q_95 core
    min_d_comp = -1.0
    n_comps = 0
    exclusive_pass = True
    if competitors:
        diff_comp = feat_inj  # already encoded
        min_d_comp = float("inf")
        for cuid, comp in competitors.items():
            if cuid == uid:
                continue
            c_mu = np.asarray(comp["mu"], dtype=np.float32)
            c_inv = np.asarray(comp["sigma_inv"], dtype=np.float32)
            c_T = comp["gate_T"]
            d2_c = _mahalanobis_d2(diff_comp, c_mu, c_inv)
            n_comps += 1
            if d2_c < min_d_comp:
                min_d_comp = d2_c
            if d2_c <= c_T:
                exclusive_pass = False
                break

    meta = InjectionMeta(
        position=ws_pos,
        original_token=word,
        typo_token=typo,
        mechanism=mech,
        mechanism_category=_categorize(mech),
        confidence=float(w_i),
        d_mahalanobis_before=d2_before,
        d_mahalanobis_after=d2_after,
        gaussian_pass=gaussian_pass,
        semantic_sim=sem_sim,
        semantic_pass=semantic_pass,
        min_d_competitor=min_d_comp if min_d_comp != float("inf") else -1.0,
        n_competitors=n_comps,
        exclusive_pass=exclusive_pass,
    )
    if not gaussian_pass or not semantic_pass or not exclusive_pass:
        return None, meta
    return injected, meta