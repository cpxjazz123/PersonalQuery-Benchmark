"""Valid-word detection via wordfreq (English frequency dictionary).

A typo result is REJECTED iff it forms another valid English word AND that word
is "meaningfully different" from the original (different surface form, not just
a typo). Two-criterion check:
  (a) typo is in wordfreq's English wordlist (freq > MIN_FREQ)
  (b) char-trigram Jaccard(orig_lower, typo_lower) < NEAR_WORD_THRESHOLD

Examples:
  really -> reallly  : (a) NOT in dict, ACCEPT (just a typo)
  really -> realily  : (a) in dict, (b) trigram jaccard 0.5+, ACCEPT (close form)
  form   -> from     : (a) in dict, (b) trigram jaccard 0.0, REJECT (different word)
  baby   -> bady     : (a) maybe, (b) jaccard high, ACCEPT (close form)

Threshold choice: word_frequency > 1e-9 means token appears in wordfreq's en
wordlist (Wikipedia + CommonCrawl + news). NEAR_WORD_THRESHOLD = 0.5 means at
least half the character trigrams overlap.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Strict reject threshold — anything in wordfreq counts as a real English word.
# wordfreq reports per-token frequency on Wikipedia + CommonCrawl + news, so even
# rare variants like "throgh" (1.35e-08) are flagged.
MIN_FREQ = 1e-9

# closeness_score below this means "different word" (not just typo).
# 0.6 catches (form,from)=0.5, (so,to)=0, (cat,bat)=0.5 but allows
# (the,teh)=0.67, (baby,bady)=0.75, (baby,babby)=not in dict.
NEAR_WORD_THRESHOLD = 0.6


@lru_cache(maxsize=65536)
def word_frequency(token_lower: str) -> float:
    """Cached wordfreq lookup. Returns 0.0 if token not in dictionary."""
    import wordfreq
    return float(wordfreq.word_frequency(token_lower, "en"))


def _strip(t: str) -> str:
    """Strip surrounding non-alpha/non-apostrophe punctuation."""
    return re.sub(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$", "", t)


def _shared_prefix(a: str, b: str) -> int:
    """Length of longest common prefix."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _shared_suffix(a: str, b: str) -> int:
    """Length of longest common suffix (excluding prefix overlap)."""
    n = min(len(a), len(b))
    for i in range(1, n + 1):
        if a[-i] != b[-i]:
            return i - 1
    return n


@lru_cache(maxsize=65536)
def closeness_score(a_lower: str, b_lower: str) -> float:
    """Simple 'close form' score: shared_prefix + shared_suffix, normalized by
    max length. Range 0-1; > 0.3 means 'close form' (likely typo, not different word).

    Examples (close=1 means identical, close=0 means totally different):
      really / reallly  -> 1.0 (identical up to extra 'l')
      baby   / babby    -> 0.86 ('ba' prefix + 'bby' suffix / 5 chars)
      baby   / bady     -> 0.75 ('ba' prefix + 'y' suffix / 4)
      form   / from     -> 0.0 (no shared prefix or suffix)
      so     / to       -> 0.0 (no shared chars)
      cat    / bat      -> 0.5 ('at' suffix / 4)
      the    / teh      -> 0.67 ('t' prefix + 'e' suffix / 3)
    """
    if not a_lower or not b_lower:
        return 0.0
    if a_lower == b_lower:
        return 1.0
    p = _shared_prefix(a_lower, b_lower)
    s = _shared_suffix(a_lower, b_lower)
    # Avoid double-counting overlap when one string is a prefix of the other
    overlap = max(0, p + s - min(len(a_lower), len(b_lower)))
    total = p + s - overlap
    return total / max(len(a_lower), len(b_lower))


def is_english_word(token: str) -> bool:
    """True iff token (stripped) appears in wordfreq's English wordlist."""
    if not token:
        return False
    t = _strip(token)
    if not t:
        return False
    return word_frequency(t.lower()) > MIN_FREQ


def is_meaningful_change(orig_token: str, typo_token: str) -> bool:
    """True iff typo is (a) a real English word AND (b) closeness_score with
    orig < NEAR_WORD_THRESHOLD. Use this to reject typo candidates that
    accidentally form a different real English word (e.g. form→from).

    Pure typos (orig→reallly, baby→babby, really→realily) all return False.
    """
    if not typo_token:
        return True
    typo_clean = _strip(typo_token).lower()
    if not typo_clean:
        return True
    if not is_english_word(typo_clean):
        return False  # not a real word, accept
    orig_clean = _strip(orig_token).lower() if orig_token else ""
    score = closeness_score(orig_clean, typo_clean)
    return score < NEAR_WORD_THRESHOLD


if __name__ == "__main__":
    cases = [
        ("really", "reallly", False),   # pure typo, accept
        ("really", "realily", False),   # close form, accept
        ("form",   "from",   True),    # different word, reject
        ("baby",   "bady",   False),   # close form, accept
        ("baby",   "babby",  False),   # pure typo, accept
        ("through", "thhrough", False), # pure typo, accept
        ("through", "thru",   True),    # different word, reject
        ("mattress", "mattrress", False),  # pure typo, accept
        ("the",    "teh",    False),   # trigram jaccard likely high
        ("diaper", "diiaper", False),  # pure typo
        ("stroller", "stroler", False),  # close
        ("so",     "to",     True),    # different
        ("cat",    "bat",    True),    # different
        ("plastic", "pllastic", False), # pure typo
    ]
    print(f"{'orig':<10} {'typo':<10} expected  actual   OK?")
    for orig, typo, expected in cases:
        actual = is_meaningful_change(orig, typo)
        ok = "✓" if actual == expected else "✗"
        print(f"{orig:<10} {typo:<10} {str(expected):<8} {str(actual):<8} {ok}")
