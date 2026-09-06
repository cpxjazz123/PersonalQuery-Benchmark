"""Typo Mechanism Classifier — Levenshtein alignment → error type classification.

Classifies (incorrect_word, corrected_word) pairs from SErCL user_word_edits into
mechanisms used to INJECT realistic typos into personalized queries.

Char-level mechanisms (true typos; the only kind emitted by Stage 10 result):
  keyboard_adjacent    : single-letter substitution, both letters on QWERTY neighbors
  letter_swap          : two adjacent letters transposed (product → prodcut)
  letter_repetition    : correct word has one extra repeated letter (really → reallly)
  letter_deletion      : correct word has one extra letter (te → the)
  letter_insertion     : correct word has one missing letter (the → te)

Surface-form mechanisms (NOT emitted as typos; tracked in summary only):
  case_error           : only case differs (Apple ↔ apple)
  apostrophe_error     : don't ↔ dont, it's ↔ its (apostrophe insertion/deletion)

Not injectable:
  semantic_substitution: doesn't fit any above (e.g. it's → that)

Note: SErCL R:simple edits provide direct (incorrect→correct) mappings; we keep them
verbatim for user_historical mechanism (highest priority, most realistic).
"""
from __future__ import annotations

import re
from typing import Tuple

# QWERTY adjacency — letters within 1 key distance (row-aware)
_QWERTY_ROWS = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
]
_QWERTY_ADJ: dict[str, set[str]] = {c: set() for c in "abcdefghijklmnopqrstuvwxyz"}
for _row in _QWERTY_ROWS:
    for _i, _c in enumerate(_row):
        _QWERTY_ADJ[_c].add(_c)  # self
        if _i + 1 < len(_row):
            _QWERTY_ADJ[_c].add(_row[_i + 1])
        if _i - 1 >= 0:
            _QWERTY_ADJ[_c].add(_row[_i - 1])
# Diagonal adjacencies (q↔a, w↔as, e↔sd, r↔df, t←→fg, y←→gh, u←→hj, ...)
for _r, _next_r in zip(_QWERTY_ROWS, _QWERTY_ROWS[1:]):
    for _i, _c in enumerate(_r):
        if _i < len(_next_r):
            _QWERTY_ADJ[_c].add(_next_r[_i])
        if _i + 1 < len(_next_r):
            _QWERTY_ADJ[_c].add(_next_r[_i + 1])
        if _i - 1 >= 0 and _i - 1 < len(_next_r):
            _QWERTY_ADJ[_c].add(_next_r[_i - 1])

# Apostrophe-sensitive tokens
_APOSTROPHE_PAIRS = {
    ("dont", "don't"), ("don't", "dont"),
    ("doesnt", "doesn't"), ("doesn't", "doesnt"),
    ("didnt", "didn't"), ("didn't", "didnt"),
    ("isnt", "isn't"), ("isn't", "isnt"),
    ("wasnt", "wasn't"), ("wasn't", "wasnt"),
    ("hasnt", "hasn't"), ("hasn't", "hasnt"),
    ("havent", "haven't"), ("haven't", "havent"),
    ("hadnt", "hadn't"), ("hadn't", "hadnt"),
    ("wont", "won't"), ("won't", "wont"),
    ("wouldnt", "wouldn't"), ("wouldn't", "wouldnt"),
    ("shouldnt", "shouldn't"), ("shouldn't", "shouldnt"),
    ("couldnt", "couldn't"), ("couldn't", "couldnt"),
    ("cant", "can't"), ("can't", "cant"),
    ("ill", "i'll"), ("i'll", "ill"),
    ("ive", "i've"), ("i've", "ive"),
    ("im", "i'm"), ("i'm", "im"),
    ("id", "i'd"), ("i'd", "id"),
    ("its", "it's"), ("it's", "its"),
    ("youre", "you're"), ("you're", "youre"),
    ("theyre", "they're"), ("they're", "theyre"),
    ("were", "we're"), ("we're", "were"),
}

# Mechanism taxonomy (used by Stage 10 for both position-rank and output schema)
CHAR_LEVEL_MECHANISMS = (
    "keyboard_adjacent",
    "letter_swap",
    "letter_repetition",
    "letter_insertion",
    "letter_deletion",
)
SURFACE_FORM_MECHANISMS = (
    "case_error",
    "apostrophe_error",
)
# All mechanisms except semantic_substitution
ALL_WRITING_MECHANISMS = CHAR_LEVEL_MECHANISMS + SURFACE_FORM_MECHANISMS


def _normalize_token(w: str) -> str:
    """Strip surrounding punctuation for comparison."""
    return re.sub(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$", "", w)


def _strip_apostrophe(w: str) -> str:
    return w.replace("'", "")


def classify_mechanism(incorrect: str, corrected: str) -> str:
    """Classify typo mechanism from (incorrect, corrected) pair.

    Returns one of: keyboard_adjacent, letter_swap, letter_repetition,
    case_error, apostrophe_error, user_historical, semantic_substitution.

    Args:
        incorrect: word as user typed it (typo)
        corrected: corrected version
    """
    if not incorrect or not corrected:
        return "semantic_substitution"

    inc_l = incorrect.lower()
    cor_l = corrected.lower()

    # Case-only difference
    if inc_l == cor_l:
        return "case_error"

    # Apostrophe variants
    pair = (inc_l, cor_l)
    if pair in _APOSTROPHE_PAIRS:
        return "apostrophe_error"
    if inc_l.replace("'", "") == cor_l.replace("'", "") and inc_l != cor_l:
        return "apostrophe_error"

    # Same length: check letter substitution or swap
    if len(inc_l) == len(cor_l):
        diffs = [(i, inc_l[i], cor_l[i]) for i in range(len(inc_l)) if inc_l[i] != cor_l[i]]
        if len(diffs) == 1:
            i, a, b = diffs[0]
            if (a in _QWERTY_ADJ) and (b in _QWERTY_ADJ):
                if b in _QWERTY_ADJ[a] and a in _QWERTY_ADJ[b]:
                    return "keyboard_adjacent"
            return "semantic_substitution"
        if len(diffs) == 2:
            # Adjacent swap (diffs at i and i+1)
            i0, a0, b0 = diffs[0]
            i1, a1, b1 = diffs[1]
            if i1 == i0 + 1 and a0 == b1 and a1 == b0:
                return "letter_swap"
        # >2 diffs of same length: probably semantic
        return "semantic_substitution"

    # Length difference of 1: insertion/deletion. Direction matters:
    #   - shorter correct → incorrect has INSERTION (one extra letter)
    #   - longer correct → incorrect has DELETION (one letter missing)
    # Within INSERTION: if the extra letter matches an adjacent letter in the
    # corrected word, call it letter_repetition (e.g. really→reallly: extra
    # 'l' next to existing 'l'). Otherwise call it letter_insertion.
    if abs(len(inc_l) - len(cor_l)) == 1:
        if len(inc_l) > len(cor_l):
            longer, shorter = inc_l, cor_l
            direction = "insertion"
        else:
            longer, shorter = cor_l, inc_l
            direction = "deletion"
        # Find the index where they diverge
        div_i = -1
        for i in range(len(shorter)):
            if shorter[i] != longer[i]:
                div_i = i
                break
        if div_i == -1:
            div_i = len(shorter)
        # Check whether shorter matches longer with one char removed at div_i
        if longer[:div_i] + longer[div_i+1:] != shorter:
            return "semantic_substitution"
        if direction == "deletion":
            return "letter_deletion"
        # insertion: is the inserted letter at div_i the same as its neighbor
        # in the longer (corrected) word?
        inserted = longer[div_i]
        left = longer[div_i - 1] if div_i > 0 else ""
        right = longer[div_i + 1] if div_i + 1 < len(longer) else ""
        if inserted == left or inserted == right:
            return "letter_repetition"
        return "letter_insertion"

    return "semantic_substitution"


def reverse_mechanism(correct_word: str, mechanism: str, rng_seed: int | None = None) -> str:
    """Generate a typo version of `correct_word` by applying the given mechanism.

    Inverse of classify_mechanism: takes a clean word + desired mechanism,
    returns a corrupted version suitable for injection.

    Args:
        correct_word: clean word (typo-free)
        mechanism: one of keyboard_adjacent/letter_swap/letter_repetition/
                   case_error/apostrophe_error/semantic_substitution
        rng_seed: unused (deterministic for now; can be replaced with rng)
    """
    import random
    rng = random.Random(rng_seed) if rng_seed is not None else None

    w = correct_word
    if not w:
        return w

    if mechanism == "semantic_substitution":
        # No defined injection pattern — return unchanged (caller will reject)
        return w

    if mechanism == "case_error":
        if w[0].isupper():
            return w[0].lower() + w[1:]
        return w[0].upper() + w[1:]

    wl = w.lower()

    if mechanism == "letter_repetition":
        # Repeat a letter at a position where it matches an adjacent letter
        # in the corrected word (e.g. really[3]='l' is adjacent to itself).
        candidates = []
        for i in range(1, len(w) - 1):
            if w[i].isalpha() and (w[i] == w[i-1] or w[i] == w[i+1]):
                candidates.append(i)
        if not candidates:
            # Fallback: any alpha letter at non-edge position
            candidates = [i for i in range(1, len(w) - 1) if w[i].isalpha()]
        if not candidates:
            return w + w[-1]
        i = rng.choice(candidates) if rng else candidates[0]
        return w[:i+1] + w[i] + w[i+1:]

    if mechanism == "letter_insertion":
        # Insert an extra letter at a position where it doesn't match neighbors
        # (e.g. apple[2]+l → applle but l doesn't match neighbors, prefer repetition).
        # Fallback: pick any alpha position not adjacent to same letter.
        candidates = [i for i in range(len(w)) if w[i].isalpha()]
        # Avoid positions adjacent to same letter (those are better for repetition)
        non_repeat = [i for i in candidates
                      if (i == 0 or w[i] != w[i-1]) and (i + 1 >= len(w) or w[i] != w[i+1])]
        pool = non_repeat if non_repeat else candidates
        if not pool:
            return w + w[-1] if w else w
        i = rng.choice(pool) if rng else pool[0]
        return w[:i+1] + w[i] + w[i+1:]

    if mechanism == "letter_deletion":
        # Remove one letter (e.g. the → te)
        candidates = [i for i in range(len(w)) if w[i].isalpha()]
        if not candidates:
            return w
        i = rng.choice(candidates) if rng else candidates[0]
        return w[:i] + w[i+1:]

    if mechanism == "letter_swap":
        candidates = [i for i in range(len(w) - 1) if w[i].isalpha() and w[i+1].isalpha()]
        if not candidates:
            return w
        i = rng.choice(candidates) if rng else candidates[0]
        return w[:i] + w[i+1] + w[i] + w[i+2:]

    if mechanism == "keyboard_adjacent":
        candidates = [i for i in range(len(w)) if w[i].isalpha()]
        if not candidates:
            return w
        i = rng.choice(candidates) if rng else candidates[0]
        c = wl[i]
        if c not in _QWERTY_ADJ or not _QWERTY_ADJ[c]:
            return w
        neighbors = [n for n in _QWERTY_ADJ[c] if n != c]
        if not neighbors:
            return w
        new_c = rng.choice(neighbors) if rng else neighbors[0]
        new_char = new_c.upper() if w[i].isupper() else new_c
        return w[:i] + new_char + w[i+1:]

    if mechanism == "apostrophe_error":
        # Add or remove apostrophe
        if "'" in wl:
            return wl.replace("'", "")
        # Try to insert ' between letters (e.g., dont → don't)
        for i in range(1, len(wl)):
            cand = wl[:i] + "'" + wl[i:]
            if (cand, wl) in _APOSTROPHE_PAIRS or (wl, cand) in _APOSTROPHE_PAIRS:
                return cand
        return w + "'"

    return w