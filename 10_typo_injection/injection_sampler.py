"""Injection Sampler — char-level typo injection with user-history reuse.

For each (uid, source query):
  1. Tokenize query with spaCy → get multi-level structural context per token
  2. Rank tokens by P_u(char_level_error | sig) at finest hierarchy level
     (full → d4 → d3 → clause → depth → sibling → coarse)
  3. For top-ranked token, look up transformation_history:
       (orig_token_lower, full_sig) → (orig_token_lower, d3_sig) →
       (orig_token_lower, coarse_sig) → (orig_token_lower, any sig)
     Pick the most common char-level typo form this user has done.
     If no user history → fall back to char_level_mechanism_totals + generic
     reverse_mechanism.
  4. Validate the typo candidate:
       - typo != orig
       - edit distance ≤ MAX_EDIT_DISTANCE (2)
       - |len(typo) - len(orig)| ≤ MAX_LEN_DELTA (1)
       - len(orig) ≥ MIN_TOKEN_LEN (3)
       - typo is NOT a meaningfully-different English word
         (valid_words.is_meaningful_change returns False)
       - typo passes 32d syntax Gaussian gate (D²_after ≤ user Q_95)
       - typo passes MiniLM cosine ≥ 0.9 semantic preservation gate
       - typo stays OUTSIDE every cohort competitor's Q_95 core
  5. Inject (single typo per query) and return.

Output emits ONLY char-level typos. Surface-form mechanisms (case_error /
apostrophe_error) are NOT in the result list — they're tracked separately
in cohort_summary. If a user has only surface-form history and no char-level
pattern, the query is marked n_surface_form_only_skip.

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
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy
import torch

from error_location import (
    _char_level_rate_hierarchical,
    _ensure_spacy, _lookup_transformation_history, _word_shape,
    _rich_signatures, _default_rate_hierarchical,
)
from typo_classifier import (
    CHAR_LEVEL_MECHANISMS, SURFACE_FORM_MECHANISMS,
    reverse_mechanism,
)
from valid_words import is_meaningful_change

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

# Minimal-edit + semantic-preservation constraints
MIN_TOKEN_LEN = 3
MAX_EDIT_DISTANCE = 2
MAX_LEN_DELTA = 1

DEFAULT_P_ERROR = 0.02  # fallback when context unseen


@dataclass
class InjectionMeta:
    position: int            # 0-indexed position in the query
    original_token: str
    typo_token: str
    mechanism: str           # one of CHAR_LEVEL_MECHANISMS
    confidence: float        # char_level rate at the selected context
    context_sig: str         # which signature was used (e.g. "full:NOUN|dobj|VERB|...")
    transformation_source: str  # "user_historical_full" / "user_historical_d3" /
                                # "user_historical_coarse" / "user_historical_any" /
                                # "generic_fallback"
    edit_distance: int       # Levenshtein(orig, typo)
    len_delta: int           # len(typo) - len(orig)
    d_mahalanobis_before: float
    d_mahalanobis_after: float
    gaussian_pass: bool
    semantic_sim: float = 1.0
    semantic_pass: bool = True
    min_d_competitor: float = -1.0
    n_competitors: int = 0
    exclusive_pass: bool = True


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
    """Token-level contexts with multi-level structural signatures."""
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


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein distance (small string inputs, naive DP is fine)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = cur
    return prev[-1]


def _mahalanobis_d2(z: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> float:
    """Full Mahalanobis squared distance D²(z, μ) = (z-μ)ᵀ Σ⁻¹ (z-μ)."""
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
    """Batch cosine similarity between paired MiniLM embeddings."""
    enc = _load_semantic_encoder()
    ea = enc.encode(texts_a, normalize_embeddings=True,
                    batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    eb = enc.encode(texts_b, normalize_embeddings=True,
                    batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    return (ea * eb).sum(axis=1)


def _apply_typo_to_query(query: str, position: int, typo: str) -> Optional[str]:
    """Replace the position-th whitespace token in query with typo."""
    tokens = _split_query_tokens(query)
    if position >= len(tokens):
        return None
    if not any(c.isalpha() for c in tokens[position]):
        return None
    tokens[position] = typo
    return " ".join(tokens)


def _validate_minimality(orig: str, typo: str) -> Tuple[bool, int, int]:
    """Validate minimal-edit constraint. Returns (ok, edit_distance, len_delta).

    ok iff:
      - typo != orig
      - len(orig) ≥ MIN_TOKEN_LEN
      - edit_distance(lower(orig), lower(typo)) ≤ MAX_EDIT_DISTANCE
      - |len(typo) - len(orig)| ≤ MAX_LEN_DELTA
      - typo is NOT a meaningfully-different English word
    """
    if typo == orig or not typo:
        return False, 0, 0
    if len(orig) < MIN_TOKEN_LEN:
        return False, 0, len(typo) - len(orig)
    ed = _levenshtein(orig.lower(), typo.lower())
    ld = len(typo) - len(orig)
    if ed > MAX_EDIT_DISTANCE:
        return False, ed, ld
    if abs(ld) > MAX_LEN_DELTA:
        return False, ed, ld
    if is_meaningful_change(orig, typo):
        return False, ed, ld
    return True, ed, ld


def _sample_user_historical_typo(
    user_model: Dict, orig_token: str, sigs: Dict[str, str], rng: random.Random,
) -> Optional[Tuple[str, str, str]]:
    """Try to find a user-historical char-level typo for (orig_token, sig).

    Returns (typo_token, mechanism, source_tag) if found, else None.
    Source tag ∈ {user_historical_full, user_historical_d3, user_historical_coarse,
                  user_historical_any} so we know which fallback level matched.
    """
    orig_lower = orig_token.lower()
    history = _lookup_transformation_history(user_model, orig_lower, sigs)
    if not history:
        return None
    for level_tag, level in (("full", "full"), ("d3", "d3"), ("coarse", "coarse"),
                              ("any", None)):
        if level is not None:
            sig = sigs.get(level)
            if sig is None:
                continue
            key = (orig_lower, sig)
            from error_location import _default_rate_hierarchical as _dr
            # Need to look up directly — use the function above for level-specific
            forms = [f for (o, s), forms in user_model.get("transformation_history", {}).items()
                     for f in forms if o == orig_lower and s == sig]
        else:
            forms = [f for (o, _s), forms in user_model.get("transformation_history", {}).items()
                     for f in forms if o == orig_lower]
        if not forms:
            continue
        # Pick the most common char-level typo (forms already sorted by count desc)
        char_forms = [f for f in forms if f[1] in CHAR_LEVEL_MECHANISMS]
        if not char_forms:
            continue
        typo, mech, _cnt = char_forms[0]
        return typo, mech, f"user_historical_{level_tag}"
    return None


def _sample_generic_typo(
    user_model: Dict, orig_token: str, rng: random.Random,
) -> Optional[Tuple[str, str]]:
    """Generic fallback: sample a char-level mechanism from user's global char-level
    distribution and apply reverse_mechanism."""
    totals = user_model.get("char_level_mechanism_totals", {})
    weights = {m: totals.get(m, 0) for m in CHAR_LEVEL_MECHANISMS}
    total_w = sum(weights.values())
    if total_w == 0:
        return None
    items = list(weights.items())
    r = rng.random() * total_w
    cum = 0.0
    mech = items[-1][0]
    for m, w in items:
        cum += w
        if r <= cum:
            mech = m
            break
    typo = reverse_mechanism(orig_token, mech, rng_seed=rng.randint(0, 10**9))
    if typo == orig_token or not typo:
        return None
    return typo, mech


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
    """Apply at most one char-level typo injection to `query`.

    Position selection: rank alpha tokens by P_u(char_level_error | sig).
    Transformation selection: reuse user-historical char-level typo at this
    (orig_token, sig) if available; otherwise generic char-level mechanism.
    Gates (all must pass):
      1. Minimal edit (edit distance ≤ 2, len delta ≤ 1, len ≥ 3, not different word)
      2. d²(z_after, μ_u) ≤ user Q_95
      3. d²(z_after, μ_comp) > comp Q_95 ∀ comp (non-shared zone)
      4. cosine(MiniLM(orig), MiniLM(typo)) ≥ 0.9

    Returns (injected_query, meta). If any gate fails, returns (None, meta) with
    the corresponding pass flag set to False.
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
            mechanism="none", confidence=0.0, context_sig="",
            transformation_source="none", edit_distance=0, len_delta=0,
            d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
            gaussian_pass=False,
        )
        return None, meta

    # Reject early if user has zero char-level history
    if user_model.get("n_char_level_edits", 0) == 0:
        meta = InjectionMeta(
            position=-1, original_token="", typo_token="",
            mechanism="none", confidence=0.0, context_sig="",
            transformation_source="surface_form_only", edit_distance=0, len_delta=0,
            d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
            gaussian_pass=False,
        )
        return None, meta

    # Rank tokens by char_level P(error | sig) — pick the highest
    # If user has zero char-level error rate at any sig, P returns 0.0;
    # but the global p_u_char_level_err is used as fallback (already 0).
    scored = []
    for ctx in contexts:
        alpha_idx, word, pos, dep, parent_pos, length, shape, sigs = ctx
        rate = _char_level_rate_hierarchical(user_model, sigs)
        scored.append((alpha_idx, word, rate, sigs))
    if not scored:
        meta = InjectionMeta(
            position=-1, original_token="", typo_token="",
            mechanism="none", confidence=0.0, context_sig="",
            transformation_source="none", edit_distance=0, len_delta=0,
            d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
            gaussian_pass=False,
        )
        return None, meta

    # Pick top-scored token; tie-break by query order (left-to-right)
    scored.sort(key=lambda x: (-x[2], x[0]))
    ws_pos, word, w_i, sigs = scored[0]

    # Look up user-historical char-level typo for (orig_word, sigs)
    picked = _sample_user_historical_typo(user_model, word, sigs, rng)
    if picked is not None:
        typo, mech, source = picked
    else:
        # Generic fallback
        gen = _sample_generic_typo(user_model, word, rng)
        if gen is None:
            meta = InjectionMeta(
                position=ws_pos, original_token=word, typo_token="",
                mechanism="none", confidence=float(w_i),
                context_sig=sigs.get("full", ""),
                transformation_source="generic_no_mechanism",
                edit_distance=0, len_delta=0,
                d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
                gaussian_pass=False,
            )
            return None, meta
        typo, mech = gen
        source = "generic_fallback"

    # Validate minimal-edit constraint
    ok, edit_dist, len_delta = _validate_minimality(word, typo)
    if not ok:
        meta = InjectionMeta(
            position=ws_pos, original_token=word, typo_token=typo,
            mechanism=mech, confidence=float(w_i),
            context_sig=sigs.get("full", ""),
            transformation_source=source,
            edit_distance=edit_dist, len_delta=len_delta,
            d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
            gaussian_pass=False,
        )
        return None, meta

    injected = _apply_typo_to_query(query, ws_pos, typo)
    if injected is None:
        meta = InjectionMeta(
            position=ws_pos, original_token=word, typo_token=typo,
            mechanism=mech, confidence=float(w_i),
            context_sig=sigs.get("full", ""),
            transformation_source=source,
            edit_distance=edit_dist, len_delta=len_delta,
            d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
            gaussian_pass=False,
        )
        return None, meta

    # Gaussian constraint
    feat_inj = _encode_queries_32d([injected], nlp, encoder, rule_to_id, vocab_size)[0]
    d2_after = _mahalanobis_d2(feat_inj, mu_u, sigma_inv)
    gaussian_pass = bool(d2_after <= d2_threshold)

    # Semantic similarity constraint
    sem_sim = float(_semantic_cosine([query], [injected])[0])
    semantic_pass = bool(sem_sim >= SEMANTIC_THRESHOLD)

    # Exclusive cohort gate
    min_d_comp = -1.0
    n_comps = 0
    exclusive_pass = True
    if competitors:
        min_d_comp = float("inf")
        for cuid, comp in competitors.items():
            if cuid == uid:
                continue
            c_mu = np.asarray(comp["mu"], dtype=np.float32)
            c_inv = np.asarray(comp["sigma_inv"], dtype=np.float32)
            c_T = comp["gate_T"]
            d2_c = _mahalanobis_d2(feat_inj, c_mu, c_inv)
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
        confidence=float(w_i),
        context_sig=sigs.get("full", ""),
        transformation_source=source,
        edit_distance=edit_dist,
        len_delta=len_delta,
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
