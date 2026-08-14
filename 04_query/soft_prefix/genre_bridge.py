"""E14-4 — Genre-bridged style exemplars.

Problem found in the E14 prototype: injecting the user's raw review sentences
as style exemplars fails because of a GENRE MISMATCH — reviews are narrative
("Occasionally his butt gets quite red...") while queries are short shopping
expressions ("I'm searching for X at 8.99"). The model cannot transfer
narrative syntax to the query genre.

Bridge: keep the review's SYNTACTIC FINGERPRINT (opener, connectors, clause
structure, modifier density) but render it in the QUERY GENRE (first-person
shopping expression with the five attribute placeholders). The bridged
skeleton is deterministic per user and genre-aligned, so the LLM can actually
mimic it.

Pipeline (train and generate share this module):
  review sentence
    -> extract_syntactic_fingerprint(sentence)      (spacy)
    -> bridge_fingerprint_to_query_skeleton(fp)     (deterministic rules)
    -> prompt exemplar: "Example: <skeleton with placeholders>"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))

from extract_clause_features_single_query import load_spacy_model  # noqa: E402

CONNECTORS = {"and", "but", "which", "that", "because", "while", "since", "so", "when", "if", "where", "although"}
# Pure adverbs only: a leading adverb is a natural, visible query opener
# ("Occasionally, I need..."). Prepositions/conjunctions ("For some reason",
# "When...") are dropped to keep the bridged sentence natural.
OPENERS = {"occasionally", "frequently", "usually", "sometimes", "always", "never",
           "however", "though", "also", "once", "eventually", "definitely",
           "honestly", "personally", "finally", "overall", "really", "truly",
           "actually", "simply", "easily", "quickly", "carefully", "gently"}

SKELETONS: List[Dict] = [
    # 0: adverb opener + that-relative
    {"clause_min": 2, "template": "{opener}I need <A2> <A1> that are <A4> for <A5>, at <A3>."},
    # 1: and-conjoined
    {"clause_min": 2, "template": "{opener}I want <A2> <A1> and I need the <A4> kind for <A5>, priced at <A3>."},
    # 2: which-relative
    {"clause_min": 2, "template": "{opener}I am after <A2> <A1> which is <A4>, works for <A5>, and costs <A3>."},
    # 3: to-infinitive
    {"clause_min": 1, "template": "{opener}I am looking for <A4> <A1> by <A2> to use for <A5> at <A3>."},
    # 4: simple + with-phrase
    {"clause_min": 1, "template": "{opener}I need <A2> <A1> with <A4> style for <A5>, around <A3>."},
    # 5: gerund style
    {"clause_min": 1, "template": "{opener}I am searching for <A2> <A1> that fits <A5> and comes in <A4> for <A3>."},
    # 6: because/reason clause
    {"clause_min": 2, "template": "{opener}I need <A2> <A1> for <A5> because it should be <A4>, and my budget is <A3>."},
    # 7: although-concessive
    {"clause_min": 2, "template": "{opener}I am buying <A2> <A1> although I mainly need it <A4> for <A5>, under <A3>."},
    # 8: when-temporal
    {"clause_min": 2, "template": "{opener}I search for <A2> <A1> when I need the <A4> option for <A5>, at around <A3>."},
    # 9: so-result
    {"clause_min": 2, "template": "{opener}I got <A2> <A1> so it could be <A4> enough for <A5>, close to <A3>."},
    # 10: simple declarative
    {"clause_min": 1, "template": "{opener}I need <A2> <A1> in <A4>, meant for <A5>, at <A3>."},
    # 11: since-reason
    {"clause_min": 2, "template": "{opener}I want <A2> <A1> since <A4> works best for <A5> and <A3> is fine."},
]

_NLP_CACHE = {}


def _nlp():
    if "_nlp" not in _NLP_CACHE:
        _NLP_CACHE["_nlp"] = load_spacy_model()
    return _NLP_CACHE["_nlp"]


def extract_syntactic_fingerprint(sentence: str) -> Dict[str, object]:
    """Deterministic syntactic fingerprint of a review sentence.

    Fields: opener (leading adverb/preposition or first content word),
    connectors (clause-level conjunctions), clause_count, modifier_density,
    length.
    """
    nlp = _nlp()
    doc = nlp(sentence or "")
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if not tokens:
        return {"opener": "I'm looking for", "connectors": [], "clause_count": 1, "modifier_density": 0.0, "length": 0}
    first = tokens[0].text.lower().strip("'\"")
    opener = first if first in OPENERS else ""
    connectors = sorted({t.text.lower() for t in tokens if t.text.lower() in CONNECTORS})
    amod = sum(1 for t in tokens if t.dep_ == "amod")
    advmod = sum(1 for t in tokens if t.dep_ == "advmod")
    n_clauses = 1 + sum(1 for t in tokens if t.dep_ in ("cc", "mark", "advcl", "ccomp", "xcomp", "relcl"))
    return {
        "opener": opener,
        "connectors": connectors,
        "clause_count": max(1, n_clauses),
        "modifier_density": round((amod + advmod) / max(len(tokens), 1), 3),
        "length": len(tokens),
    }


def bridge_fingerprint_to_query_skeleton(fp: Dict[str, object]) -> str:
    """Map a fingerprint to a query-genre skeleton with exactly the five
    placeholders <A1>..<A5> (deterministic; genre-aligned)."""
    n_clauses = int(fp["clause_count"])
    connectors = list(fp["connectors"])
    has_and = "and" in connectors
    has_which = any(c in connectors for c in ("which", "that"))
    has_to = any(c in connectors for c in ("to",))
    has_because = "because" in connectors
    has_although = "although" in connectors
    has_when = "when" in connectors
    has_so = "so" in connectors
    has_since = "since" in connectors
    density = float(fp["modifier_density"])
    opener = str(fp["opener"])
    opener_fmt = f"{opener.title()}, " if opener else ""

    if has_because:
        idx = 6
    elif has_although:
        idx = 7
    elif has_when:
        idx = 8
    elif has_so:
        idx = 9
    elif has_since:
        idx = 11
    elif has_and:
        idx = 1
    elif has_which:
        idx = 2
    elif has_to:
        idx = 3
    elif density >= 0.25:
        idx = 0
    elif n_clauses >= 2:
        idx = 5
    else:
        idx = 10 if hash(opener) % 2 == 0 else 4
    return SKELETONS[idx]["template"].format(opener=opener_fmt)


def bridge_review_to_query_skeleton(sentence: str) -> Optional[str]:
    """Full bridge: review sentence -> fingerprint -> query-genre skeleton."""
    fp = extract_syntactic_fingerprint(sentence)
    return bridge_fingerprint_to_query_skeleton(fp)


def build_global_skeleton_distribution(sentences: List[str]) -> Dict[str, float]:
    """Empirical distribution of bridged skeletons over all sentences
    (deterministic; used to pick DISTINCTIVE per-user exemplars)."""
    counts: Dict[str, int] = {}
    for s in sentences:
        skel = bridge_review_to_query_skeleton(s)
        if skel:
            counts[skel] = counts.get(skel, 0) + 1
    total = sum(counts.values()) or 1
    return {k: c / total for k, c in counts.items()}


def select_distinctive_sentences(
    sentences: List[str], global_dist: Dict[str, float], k: int = 3
) -> List[str]:
    """Pick the user's sentences whose bridged skeleton is rarest globally
    (style signal: the rarer the structure, the more user-distinctive)."""
    import math
    scored = []
    for s in sentences:
        skel = bridge_review_to_query_skeleton(s)
        if not skel:
            continue
        p = global_dist.get(skel, 1e-6)
        scored.append((-math.log(max(p, 1e-6)), s, skel))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [s for _, s, _ in scored[:k]]


def verify_skeleton(skeleton: str) -> bool:
    """Each placeholder exactly once, none glued."""
    for ph in ["<A1>", "<A2>", "<A3>", "<A4>", "<A5>"]:
        if skeleton.count(ph) != 1:
            return False
    if re.search(r"[A-Za-z0-9]<A[1-9][0-9]?>|<A[1-9][0-9]?>[A-Za-z0-9]", skeleton):
        return False
    return True
