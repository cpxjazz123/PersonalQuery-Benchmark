"""Error Location Model — Structural context → per-user P_u(error | c_i).

Builds (from SErCL user_word_edits) a per-user error-probability table indexed by
**multi-level structural signatures** ranging from coarse (POS|DEP) to fine
(D3 rule + clause type + depth + sibling). Each token in the user's history
writes to ALL signature levels simultaneously, so every level has its own
Laplace-smoothed rate derived from the same observations.

Signatures (in lookup priority order):
  full    = pos|dep|parent_pos|gp_pos|clause_marker|depth_bucket  (richest)
  d4      = D4|gp_pos|parent_pos|dep|child_pos                     (grandparent rule)
  d3      = D3|parent_pos|dep|child_pos                            (parent rule)
  clause  = clause_marker|pos                                      (clause type)
  depth   = depth_bucket|pos|dep                                   (syntactic depth)
  sibling = pos|sibling_left_pos|sibling_right_pos                 (local subtree)
  coarse  = pos|dep                                                (fallback)

Sampler does hierarchical lookup: try `full` first, fall back to `d4`, `d3`,
`clause`, `depth`, `sibling`, `coarse`, then user p_u_err, then DEFAULT 0.02.

Also computes per-user p_u_err = (n_edits) / (n_tokens) — the global rate.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import spacy

from typo_classifier import classify_mechanism

# spaCy model — used only for token-level POS/DEP context extraction
SPACY_MODEL = "en_core_web_sm"

# Length buckets
_LEN_BUCKETS = [(1, 3), (4, 6), (7, 10), (11, 1000)]

# Hierarchy lookup order (finest → coarsest)
SIG_LEVELS = ("full", "d4", "d3", "clause", "depth", "sibling", "coarse")

# Clause-opening dependencies in spaCy: advcl/relcl/ccomp/acl start a clause;
# prep opens a prepositional phrase (treated as clause-light for typo modeling).
_CLAUSE_DEPS = ("advcl", "relcl", "ccomp", "acl")


def _clause_marker(tok, doc) -> str:
    """Walk up the head chain; return first clause-opening dep or ROOT."""
    cur = tok
    visited = set()
    while cur.head != cur and cur.i not in visited:
        visited.add(cur.i)
        cur = cur.head
        if cur.dep_ in _CLAUSE_DEPS:
            return cur.dep_.upper()
        if cur.dep_ == "prep":
            return "PREP"
    return "ROOT"


def _depth_bucket(tok, doc) -> str:
    """Bucket depth in dependency tree: d0 (root), d1, d2, d3+."""
    depth = 0
    cur = tok
    visited = set()
    while cur.head != cur and cur.i not in visited and depth < 10:
        visited.add(cur.i)
        cur = cur.head
        depth += 1
    if depth <= 0:
        return "d0"
    if depth == 1:
        return "d1"
    if depth == 2:
        return "d2"
    return "d3+"


def _grandparent_pos(tok, doc) -> str:
    """POS of grandparent (head's head), or ROOT if no grandparent."""
    h = tok.head
    if h is None or h == tok:
        return "ROOT"
    gp = h.head
    if gp == h or gp is None:
        return "ROOT"
    return gp.pos_


def _sibling_pos(tok, direction: str) -> str:
    """POS of nearest sibling on given side, or NONE."""
    head = tok.head
    if head == tok or head is None:
        return "NONE"
    if direction == "left":
        cands = [c for c in head.children if c.i < tok.i]
        return cands[-1].pos_ if cands else "NONE"
    cands = [c for c in head.children if c.i > tok.i]
    return cands[0].pos_ if cands else "NONE"


def _rich_signatures(tok, doc) -> Dict[str, str]:
    """Multi-level structural signatures for one token, all levels written.

    Returns dict[level_name → sig_string] for levels in SIG_LEVELS.
    """
    pos = tok.pos_
    dep = tok.dep_
    h = tok.head
    parent_pos = h.pos_ if h and h != tok else "ROOT"
    gp_pos = _grandparent_pos(tok, doc)
    clause = _clause_marker(tok, doc)
    depth_bkt = _depth_bucket(tok, doc)
    sib_l = _sibling_pos(tok, "left")
    sib_r = _sibling_pos(tok, "right")
    length = len(tok.text)
    shape = _word_shape(tok.text)
    return {
        "full":    f"{pos}|{dep}|{parent_pos}|{gp_pos}|{clause}|{depth_bkt}",
        "d4":      f"D4|{gp_pos}|{parent_pos}|{dep}|{pos}",
        "d3":      f"D3|{parent_pos}|{dep}|{pos}",
        "clause":  f"{clause}|{pos}",
        "depth":   f"{depth_bkt}|{pos}|{dep}",
        "sibling": f"{pos}|{sib_l}|{sib_r}",
        "coarse":  f"{pos}|{dep}",
    }


def _default_rate_hierarchical(model: Dict, sigs: Dict[str, str]) -> float:
    """Hierarchical lookup: try finest sig first, fall back to coarser.

    Returns P_u(error | sig) at the finest level the user has data for.
    If no level seen, falls back to user's global p_u_err, then DEFAULT_P_ERROR.
    """
    rates = model.get("error_rate_by_context", {})
    for level in SIG_LEVELS:
        sig = sigs.get(level)
        if sig is None:
            continue
        r = rates.get(sig)
        if r is not None:
            return r
    p_user = model.get("p_u_err", 0.0)
    return p_user if p_user > 0 else 0.02


def _len_bucket(n: int) -> str:
    for lo, hi in _LEN_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "11+"


def _word_shape(w: str) -> str:
    """Coarse shape: lower/upper/title/digit/mixed."""
    if not w:
        return "empty"
    if w.isdigit():
        return "digit"
    if w.islower():
        return "lower"
    if w.isupper():
        return "upper"
    if w[0].isupper() and w[1:].islower():
        return "title"
    if any(c.isdigit() for c in w):
        return "mixed"
    return "other"


def _context_signature(pos: str, dep: str, parent_pos: str, length: int, word: str) -> str:
    return f"{pos}|{dep}|{parent_pos}|{_len_bucket(length)}|{_word_shape(word)}"


def _ensure_spacy():
    """Lazy-load spaCy, return nlp pipeline."""
    if not hasattr(_ensure_spacy, "_nlp"):
        _ensure_spacy._nlp = spacy.load(SPACY_MODEL, disable=["ner", "lemmatizer"])
    return _ensure_spacy._nlp


def extract_token_contexts(sentence: str) -> List[Tuple[int, str, str, str, int, str]]:
    """Return list of (idx, word, pos, dep, parent_pos, len_bucket, shape) for tokens.

    Args:
        sentence: input sentence

    Returns:
        List of tuples (token_idx, word, pos, dep, parent_pos, len_bucket, shape)
        where token_idx is the position in the original sentence (0-indexed, splits on
        whitespace AFTER spaCy tokenization — we use spaCy's own positions).
    """
    nlp = _ensure_spacy()
    doc = nlp(sentence)
    out = []
    for tok in doc:
        if not tok.is_alpha:
            continue
        out.append((tok.i, tok.text, tok.pos_, tok.dep_, tok.head.pos_, len(tok.text), _word_shape(tok.text)))
    return out


def _build_user_history(user_word_edits: List[dict]) -> Dict[str, Dict[str, int]]:
    """From a user's word_edits list, build per-context mechanism histogram at all
    signature levels (full/d4/d3/clause/depth/sibling/coarse).

    Returns: dict[context_sig][mechanism] = count
        where context_sig is any of the multi-level signatures, mechanism is one
        of the typo_classifier outputs. Each observed token writes its counts
        to ALL signature levels simultaneously so every level gets an
        independent Laplace-smoothed rate.
    """
    nlp = _ensure_spacy()

    history: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in user_word_edits:
        orig_sent = entry.get("orig_sent", "")
        edits = entry.get("edits", [])
        if not orig_sent or not edits:
            continue

        doc = nlp(orig_sent)

        edit_pairs = []
        for e in edits:
            inc = e.get("incorrect_word", "")
            cor = e.get("corrected_word", "")
            if inc:
                edit_pairs.append((inc, cor))

        for tok in doc:
            if not tok.is_alpha:
                continue
            word_l = tok.text.lower()
            sigs = _rich_signatures(tok, doc)

            matched_mech = None
            for inc, cor in edit_pairs:
                inc_stripped = re.sub(r"[^a-zA-Z']", "", inc).lower()
                if inc_stripped == word_l:
                    mech = classify_mechanism(tok.text, cor)
                    if mech != "semantic_substitution":
                        matched_mech = mech
                    break

            # Write to every signature level
            for level in SIG_LEVELS:
                sig = sigs[level]
                if matched_mech:
                    history[sig][matched_mech] += 1
                history[sig]["__total__"] += 1
    return history


def build_user_error_model(uid: str, user_word_edits: List[dict]) -> Dict:
    """Build per-user error probability model.

    Returns dict with:
      p_u_err: global error rate (float, n_edits / n_tokens)
      n_tokens: total tokens analyzed
      n_edits: total errors (excluding semantic_substitution which is not injectable)
      context_probs: {context_sig: {"mechanisms": {mech: count}, "total": int}}
      error_rate_by_context: {context_sig: p(error|context)}
      mechanism_totals: {mech: total_count} aggregated across all contexts
    """
    history = _build_user_history(user_word_edits)

    n_tokens = sum(h.get("__total__", 0) for h in history.values())
    n_edits = sum(sum(v for k, v in h.items()
                      if k not in ("__total__", "__user_historical__"))
                  for h in history.values())
    p_u_err = n_edits / n_tokens if n_tokens else 0.0

    context_probs = {}
    error_rate_by_context = {}
    mechanism_totals: Dict[str, int] = defaultdict(int)
    alpha = 0.1
    for sig, hist in history.items():
        total = hist.get("__total__", 0)
        mech_only = {k: v for k, v in hist.items()
                     if k not in ("__total__", "__user_historical__")}
        n_err_ctx = sum(mech_only.values())
        rate = (n_err_ctx + alpha) / (total + alpha * 2)
        context_probs[sig] = {"mechanisms": mech_only, "total": total}
        error_rate_by_context[sig] = rate
        for mech, cnt in mech_only.items():
            mechanism_totals[mech] += cnt

    return {
        "uid": uid,
        "p_u_err": p_u_err,
        "n_tokens": n_tokens,
        "n_edits": n_edits,
        "context_probs": context_probs,
        "error_rate_by_context": error_rate_by_context,
        "mechanism_totals": dict(mechanism_totals),
    }


def load_sercl_profile(path: Path) -> Dict[str, dict]:
    """Load SErCL profile JSON. Returns {uid: {p_u_err, n_tokens, n_edits,
    context_probs, error_rate_by_context}}."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    profiles = raw.get("user_profiles", {})
    word_edits = raw.get("user_word_edits", {})
    out = {}
    for uid, we_list in word_edits.items():
        if not we_list:
            continue
        out[uid] = build_user_error_model(uid, we_list)
    return out


if __name__ == "__main__":
    import sys
    profile_path = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/09_sercl_user_profile/user_sercl_profile.json")
    profiles = load_sercl_profile(profile_path)
    n_users = len(profiles)
    avg_p_err = sum(p["p_u_err"] for p in profiles.values()) / max(1, n_users)
    print(f"loaded {n_users} user error models; mean p_u_err = {avg_p_err:.4f}")
    # sample
    uid0 = list(profiles.keys())[0]
    p0 = profiles[uid0]
    print(f"  sample uid={uid0}: p_u_err={p0['p_u_err']:.4f}, n_tokens={p0['n_tokens']}, n_edits={p0['n_edits']}")
    print(f"  n_contexts={len(p0['error_rate_by_context'])}")
    sample_ctx = list(p0['error_rate_by_context'].keys())[:3]
    for ctx in sample_ctx:
        print(f"    {ctx}: P(err)={p0['error_rate_by_context'][ctx]:.4f}")