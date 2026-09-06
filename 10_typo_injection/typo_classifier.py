"""Typo Mechanism Classifier — Levenshtein alignment → error type classification.

Classifies (incorrect_word, corrected_word) pairs from SErCL user_word_edits into
mechanisms used to INJECT realistic typos into personalized queries.

Mechanisms (priority high → low):
  user_historical      : user's own typo pair (from user_word_edits R: simple-sub op)
  keyboard_adjacent    : single-letter substitution, both letters on QWERTY neighbors
  letter_swap          : two adjacent letters transposed (product → prodcut)
  letter_repetition    : correct word has one extra repeated letter (really → reallly)
  case_error           : only case differs (Apple ↔ apple)
  apostrophe_error     : don't ↔ dont, it's ↔ its (apostrophe insertion/deletion)
  semantic_substitution: doesn't fit any above (NOT injectable, e.g. it's → that)

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

    # Length difference of 1: insertion/deletion (letter_repetition)
    if abs(len(inc_l) - len(cor_l)) == 1:
        longer, shorter = (inc_l, cor_l) if len(inc_l) > len(cor_l) else (cor_l, inc_l)
        # Find first index where they differ
        for i in range(len(shorter)):
            if shorter[i] != longer[i]:
                # Check if shorter is longer with one letter removed at position i
                if longer[:i] + longer[i+1:] == shorter:
                    return "letter_repetition"
                return "semantic_substitution"
        # All chars match except one extra at end
        return "letter_repetition"

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
        # Repeat a letter (pick a non-edge letter with alpha)
        candidates = [i for i in range(1, len(w) - 1) if w[i].isalpha()]
        if not candidates:
            return w + w[-1]
        i = rng.choice(candidates) if rng else candidates[0]
        return w[:i+1] + w[i] + w[i+1:]

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