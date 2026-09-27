"""Build and clean Stage 11 product document cache in scratch.

Split out of `syntax_subspace_retrieval_unified.py` on 2026-09-22.
Two responsibilities:
  1. Assemble each record of `data/meta_Baby_Products_2023.jsonl` into a structured
     document (product / brand / category / dimensions / weight / description / features).
  2. Apply 4 cleaning tiers to the assembled doc string:
       Tier 1: marketing noise stripping (Brand Story / From the Manufacturer /
               See more / Add to Cart / Worry-free / Customer Service template noise)
       Tier 2: content dedup (dimensions/weight repeated in description,
               product title repeated in description/features,
               features[0] == full product name)
       Tier 3: format normalization (features renumber, ALL CAPS lines to title case,
               URL / hashtag cleanup)
       Tier 5: encoding repair (mojibake, mixed non-English chars)

Cleaning rules design goals:
  - No fallback (Rule 7): rule miss -> keep original text, do not substitute.
  - Do not modify source data: META_FILE is read-only.
  - Cache signature includes META_FILE mtime+size + n_asins, invalidates on source change.
  - Output overwrites `$PQ_SCRATCH/stage11_corpus_cache/<category>/asin_to_doc.json`;
    downstream BM25/SPLADE/MiniLM/MPNet/BGE/GTE/ColBERTv2 (7 retrievers) auto-consume the cleaned version.

Consumers:
  - 11_syntactic_evaluation/syntax_subspace_retrieval_unified.py (main + dense_retrieve)
  - 12_typo_evaluation / 13_syntactic_rerank / 14_typo_rerank all consume the same json.

Run (no CLI args, per Rule 4):
  cd "$REPO_ROOT"
  $PQ_PYTHON 11_syntactic_evaluation/build_asin_to_doc.py
"""
from __future__ import annotations

import collections
import hashlib
import html
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterable

# ===========================================================================
# PATHS (hardcoded, per Rule 4)
# ===========================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
# User directive 2026-09-23: data directory migrated from REPO_ROOT/data to hj82 same-name data directory.
DATA_DIR = Path(os.environ.get("PQ_DATA_DIR", str(REPO_ROOT / "data")))
META_FILE = DATA_DIR / "meta_Baby_Products_2023.jsonl"
SCRATCH = Path(os.environ.get("PQ_SCRATCH", str(REPO_ROOT / "scratch")))
PQ_PYTHON = os.environ.get("PQ_PYTHON", sys.executable)
ASIN_TO_DOC_CACHE = SCRATCH / "stage11_corpus_cache" / "baby" / "asin_to_doc.json"
# All 3 category corpus caches are written to hj82_scratch2 to avoid stacking cache under result/.
CATEGORY_INPUTS = [
    # (category_key, subdir)
    ("Baby",                "baby"),
    ("Musical_Instruments", "musical"),
    ("Video_Games",         "video_games"),
]

# ===========================================================================
# LOG
# ===========================================================================
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ===========================================================================
# Basic character cleaning (equivalent to original _clean_doc_text)
# ===========================================================================
_ALLOWED_PUNCTUATION = set("-_/.,:%&+()[]'\"#")


def _clean_doc_text(value) -> str:
    """Basic character cleaning: HTML escape, emoji, non-alphanumeric symbols -> space.

    Preserves # / & / + and other common product-doc symbols (features use '#N:' numbering,
    category uses '/' separator, dimensions use 'x' separator).
    """
    if isinstance(value, list):
        value = " ".join(str(x) for x in value if x is not None)
    elif not isinstance(value, str):
        return ""
    value = html.unescape(value)
    value = re.sub(r"<[^>]*>", " ", value)
    value = value.replace("\r", " ").replace("\n", " ")
    cleaned = []
    for char in value:
        category = unicodedata.category(char)
        if category.startswith("C") or category.startswith("So"):
            cleaned.append(" ")
        elif char.isalnum() or char.isspace() or char in _ALLOWED_PUNCTUATION:
            cleaned.append(char)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split()).strip()


# ===========================================================================
# Tier 5: encoding repair (lightweight ftfy replacement, no third-party libs)
# ===========================================================================
# Common mojibake byte residues (UTF-8/Latin-1/CP1252 cross-decoding errors)
_MOJIBAKE_MAP = {
    "\u00c2\u00a0": " ",       # NBSP via UTF-8 of Latin-1 0xA0
    "\u00c2\u2019": "'",       # curly quote
    "\u00c2\u201c": '"',
    "\u00c2\u201d": '"',
    "\u00c2\u00ae": "(R)",     # ®
    "\u00c2\u2122": "(TM)",    # ™
    "\u00c2\u00b0": "deg",     # °
    "\u00c2\u00bd": "1/2",
    "\u00c2\u00bc": "1/4",
    "\u00c2\u00be": "3/4",
    "\u00e2\u0080\u0099": "'",  # ’
    "\u00e2\u0080\u009c": '"',  # “
    "\u00e2\u0080\u009d": '"',  # ”
    "\u00e2\u0080\u0093": "-",  # –
    "\u00e2\u0080\u0094": "-",  # —
    "\u00e2\u0080\u00a6": "...",# …
    "\u00c3\u00a9": "e",       # é
    "\u00c3\u00a8": "e",       # è
    "\u00c3\u00a0": "a",       # à
    "\u00c3\u00ae": "i",       # î
    "\u00c3\u00b1": "n",       # ñ → n
    "\u00ef\u00bf\u00bd": "",  # U+FFFD replacement
}
_MOJIBAKE_PATTERN = re.compile("|".join(re.escape(k) for k in _MOJIBAKE_MAP.keys()))

# Isolated mojibake chars (Latin-1/CP1252 bytes erroneously decoded as UTF-8)
_ISOLATED_MOJIBAKE_RE = re.compile(r"[\u00c2\u00e2](?!\S)")
# Isolated â inside words (residual mojibake, not source data); preserved as ''

# Non-English chars (CJK + Latin diacritics + Cyrillic + Greek + Arabic + Hebrew)
_NON_EN_RE = re.compile(
    r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af"
    r"\u0400-\u04ff\u0370-\u03ff\u0590-\u05ff"
    r"\u0600-\u06ff]"
)


def _fix_encoding(text: str) -> str:
    """Repair mojibake + strip mixed non-English characters.

    Mojibake is a deterministic character replacement; mixed CJK and similar characters
    are typically data collection errors, stripped wholesale.
    """
    if not text:
        return text
    fixed = _MOJIBAKE_PATTERN.sub(lambda m: _MOJIBAKE_MAP[m.group(0)], text)
    fixed = _ISOLATED_MOJIBAKE_RE.sub("", fixed)
    fixed = _NON_EN_RE.sub(" ", fixed)
    return re.sub(r"\s+", " ", fixed).strip()


# ===========================================================================
# Tier 1: marketing noise stripping
# ===========================================================================
# These patterns describe clauses/fragments that carry no information value in product descriptions.
# Match form: each pattern is a regex matching clauses in description / features.
# Design principle: only match "full sentences" or "standalone phrases" to avoid deleting real product info.
_MARKETING_SENTENCE_PATTERNS = [
    # UI / template header
    r"^\s*Product Description\s*[:.\-]?\s*$",
    r"^\s*Product Details\s*[:.\-]?\s*$",
    r"^\s*From the Manufacturer\s*[:.\-]?\s*$",
    r"^\s*Brand Story\s*(?:By\s+[\w\s&\-'.]+)?\s*[:.\-]?\s*$",
    r"^\s*See more\s*\.?\s*$",
    r"^\s*Read more\s*\.?\s*$",
    # CTA / sales pitch
    r"^\s*Click(?:\s+[\"'])?Add to Cart(?:\s+[\"'])?\s+now\.?\s*$",
    r"^\s*Add to Cart(?:\s+now)?\.?\s*$",
    r"^\s*Order\s+Now(?:\s+Without\s+Risk)?\.?\s*$",
    r"^\s*Buy it now\.?\s*$",
    r"^\s*Don'?t hesitate(?:\s+any\s+more)?(?:,)?\s*buy it now\.?\s*$",
    r"^\s*Make these?\s+[\w\s]+part of your\s+[\w\s]+(?:daily\s+life|essentials?)\.?\s*$",
    # after-sales promises
    r"^\s*24[\- ]?hour\s+friendly\s+customer\s+service\.?\s*$",
    r"^\s*24[\- ]?h\s+(?:friendly\s+)?customer\s+service\.?\s*$",
    r"^\s*We offer\s+24[\- ]?h(?:our)?\s+(?:friendly\s+)?(?:customer\s+)?service\.?\s*$",
    r"^\s*Worry[\- ]?free\s+after[\- ]?sales\.?\s*$",
    r"^\s*We (?:will )?(?:try|are committed) to (?:provide|solve|help).*?(?:customer service|issue|problem).*?\.?\s*$",
    r"^\s*For any reason you are not 100%\s+satisfied.*?\.?\s*$",
    r"^\s*If you (?:have any|are not).*?(?:question|issue|problem).*?please.*?\.?\s*$",
    r"^\s*Please do not hesitate to contact us.*?\.?\s*$",
    r"^\s*(?:Contact|Reach out to) us (?:at once|right away|now).*?\.?\s*$",
    r"^\s*We (?:offer|provide) 24h? service.*?\.?\s*$",
    # generic marketing phrases
    r"^\s*It(?:'s| is) (?:really |so )?(?:a )?(?:great|good|nice|perfect|amazing) (?:gift|choice|option)(?:\s+for\s+(?:the\s+)?[\w\s]+)?\.?\s*$",
    r"^\s*The (?:perfect|ideal|best) (?:gift|choice|option) for\s+.*?\.?\s*$",
    r"^\s*Makes? (?:a )?(?:great|perfect|ideal) (?:gift|present)\.?\s*$",
    # all-caps header line (marketing slogan): starts with all-caps word
    r"^\s*([A-Z][A-Z&\-]{2,}\s+){3,}[A-Z][A-Z&\-:]*\s*\.?\s*$",
    # substring-level (only delete on full match)
]
_MARKETING_SUBSTRING_PATTERNS = [
    (r"\bBrand Story\s+By\s+\S+(?:\s+\S+){0,3}\.?\s*", " "),
    (r"\bProduct Description\s*[:.\-]?\s*", " "),
    (r"\bFrom the Manufacturer\s*[:.\-]?\s*", " "),
    (r"\bSee more\.?\s*", " "),
    (r"\bRead more\.?\s*", " "),
    (r"Click(?:\s+[\"'])?Add to Cart(?:\s+[\"'])?\s+now\.?", " "),
    (r"Order Now Without Risk\.?", " "),
    (r"\bAdd to Cart\s+now\.?", " "),
    (r"\b24[\- ]?hour friendly customer service\.?", " "),
    (r"\bWorry[\- ]?free after[\- ]?sales\.?", " "),
]


def _strip_marketing_noise(text: str) -> str:
    """Tier 1: delete pure marketing-noise clauses/fragments."""
    if not text:
        return text
    # Step 1: clause-level whole-sentence deletion (sentence-level)
    # Sentence split: uses . ! ? + newline / capital-letter start as boundaries
    sentences = re.split(r"(?<=[.!?])\s+|(?<=\.)\s*\n", text)
    kept = []
    for s in sentences:
        s_strip = s.strip()
        if not s_strip:
            continue
        is_marketing = False
        for pat in _MARKETING_SENTENCE_PATTERNS:
            if re.match(pat, s_strip, re.IGNORECASE if pat.startswith(r"^\s*[A-Z]") else 0):
                is_marketing = True
                break
        if not is_marketing:
            kept.append(s)
    text = " ".join(kept)
    # Step 2: substring-level deletion (some phrases are not at sentence start)
    for pat, repl in _MARKETING_SUBSTRING_PATTERNS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    # Step 3: URL cleanup
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bwww\.\S+", " ", text)
    # Trailing whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===========================================================================
# Tier 2: content dedup
# ===========================================================================
def _extract_numeric_tokens(text: str) -> list[str]:
    """Extract numeric / unit tokens from text (used for dimensions/weight dup detection)."""
    return re.findall(r"\d+(?:\.\d+)?", text)


def _dedup_repeated_content(desc: str, dims: str, wt: str, product_title: str) -> str:
    """Tier 2: from description, delete clauses whose numbers repeat dimensions/weight,
    and the first sentence if it equals the full product title.
    """
    if not desc:
        return desc
    sentences = re.split(r"(?<=[.!?])\s+", desc)
    kept = []
    dim_nums = _extract_numeric_tokens(dims) if dims else []
    wt_nums = _extract_numeric_tokens(wt) if wt else []
    for s in sentences:
        s_strip = s.strip()
        if not s_strip:
            continue
        # Check: does this sentence mostly repeat dimensions numbers?
        nums_in_s = _extract_numeric_tokens(s_strip)
        if dim_nums and len(nums_in_s) >= 2:
            overlap = sum(1 for n in nums_in_s if n in dim_nums)
            # Most numbers (>=2 and >=80% coverage) overlap with dimensions -> restating size
            if overlap >= max(2, int(len(nums_in_s) * 0.8)):
                continue
        # Check: does this sentence mostly repeat weight numbers?
        if wt_nums and len(nums_in_s) == 1:
            if nums_in_s[0] in wt_nums and len(s_strip) < 60:
                continue
        # Check: does this sentence equal the full product title (duplicate)?
        if product_title and len(product_title) >= 20:
            pt_lower = product_title.lower()
            s_lower = s_strip.lower()
            # Sentence contains product_title and length <= 1.5x of title -> restating
            if pt_lower in s_lower and len(s_strip) <= int(len(product_title) * 1.5):
                continue
        kept.append(s)
    desc = " ".join(kept)
    return re.sub(r"\s+", " ", desc).strip()


def _dedup_features_title(features: list[str], product_title: str) -> list[str]:
    """Tier 2: if features[0] equals or is highly similar to product title, delete it."""
    if not features or not product_title or len(product_title) < 15:
        return features
    pt_lower = product_title.lower()
    first = features[0].strip()
    first_lower = first.lower()
    # features[0] contains product_title and length <= 1.5x -> restating
    if pt_lower in first_lower and len(first) <= int(len(product_title) * 1.5):
        return features[1:]
    # features[0] exactly equals product_title
    if first_lower == pt_lower:
        return features[1:]
    return features


# ===========================================================================
# Tier 3: format normalization
# ===========================================================================
# Known product abbreviations / trademarks / material acronyms; preserve uppercase when title-casing
_KEEP_UPPER = {
    "BPA", "CPSIA", "FDA", "LFGB", "PVC", "PEVA", "PE", "PU", "PP",
    "USA", "UK", "EU", "USD", "TOG", "NICU", "PVC", "TPU", "TPE",
    "DIY", "OEM", "ODM", "SGS", "CE", "ROHS", "ROHS",
    "USB", "AC", "DC", "LED", "LCD", "HD", "GPS",
    "XL", "XXL", "XS", "SM", "MD", "LG",
    "DIY", "BPA", "SIDS", "HALO", "ECR", "ECR4KIDS",
    "DCDC", "MICROFIBER", "MICROFIBRE", "POLYESTER",
    "NYLON", "COTTON", "SILICONE", "ALUMINUM", "ALUMINIUM",
    "STAINLESS", "STEEL", "WOOD", "METAL", "PLASTIC",
    "OZ", "ML", "LB", "LBS", "KG", "G", "CM", "MM", "IN", "INCH", "INCHES",
    "FT", "FEET", "YARD", "YDS",
}


def _normalize_caps_line(line: str) -> str:
    """Tier 3.1: detect if first N words are all uppercase (marketing-title style); if so -> title case.

    Threshold: of first 5 alpha words, uppercase >= 4 and length > 30 chars.
    """
    if len(line) < 30:
        return line
    words = line.split()
    if len(words) < 3:
        return line
    # Take first 5 words containing letters
    head_words = []
    for w in words:
        if any(c.isalpha() for c in w):
            head_words.append(w)
        if len(head_words) >= 5:
            break
    if len(head_words) < 3:
        return line
    # Check if most head_words are uppercase
    upper_cnt = sum(1 for w in head_words if w.isupper() or sum(1 for c in w if c.isupper()) >= len([c for c in w if c.isalpha()]) - 1)
    if upper_cnt < 3:
        return line
    # Convert to title case: preserve acronyms uppercase
    out = []
    for w in words:
        stripped = w.strip(".,:;()[]'\"")
        if stripped.upper() in _KEEP_UPPER:
            out.append(w.upper())
        elif any(c.isdigit() for c in w):
            out.append(w)
        elif w.isupper() and len(w) <= 4:
            out.append(w)
        else:
            out.append(w.capitalize() if w.isupper() else w)
    return " ".join(out)


def _renumber_features(features: list[str]) -> list[str]:
    """Tier 3.2: renumber features (in current order 1..N); existing '#N:' prefix is replaced."""
    cleaned = [f.strip() for f in features if f and f.strip()]
    return [f"#{i}: {feat}" for i, feat in enumerate(cleaned, 1)]


# ===========================================================================
# Field-level cleaning pipeline
# ===========================================================================
def _clean_description(desc: str, dims: str, weight: str, product_title: str) -> str:
    """Apply all 4 cleaning tiers to the description field."""
    if not desc:
        return desc
    desc = _strip_marketing_noise(desc)
    # Tier 5
    desc = _fix_encoding(desc)
    # Tier 2: dimensions/weight number duplicates + product title duplicates -> clause-level delete
    desc = _dedup_repeated_content(desc, dims, weight, product_title)
    # Tier 3.1: ALL CAPS line normalization
    # description has already been normalized to single line (\n -> space in build_meta_corpus)
    desc = _normalize_caps_line(desc)
    return desc


def _clean_features(features: list[str], product_title: str) -> list[str]:
    """Clean features list: Tier 1 (marketing) + Tier 2 (title dup) + Tier 3 (renumber)."""
    cleaned = []
    for f in features:
        if not f:
            continue
        c = _strip_marketing_noise(f)
        c = _fix_encoding(c)
        if c:
            cleaned.append(c)
    # Tier 2: does features[0] duplicate the product title?
    cleaned = _dedup_features_title(cleaned, product_title)
    # Tier 3.2: renumber
    return _renumber_features(cleaned)


def _clean_simple_field(value: str) -> str:
    """Apply Tier 5 + Tier 1 substring cleanup to simple fields like brand / category / dimensions / weight."""
    if not value:
        return value
    value = _strip_marketing_noise(value)
    value = _fix_encoding(value)
    return re.sub(r"\s+", " ", value).strip()


# ===========================================================================
# Cache signature (compatible with syntax_subspace_retrieval_unified.py)
# ===========================================================================
def _asin_content_sha1(asin_to_doc: dict[str, str]) -> str:
    """Compute sha1 over asin_to_doc sorted contents (40 hex chars)."""
    content_h = hashlib.sha1()
    keys = sorted(asin_to_doc.keys())
    log(f"  hashing {len(keys)} ASIN documents")
    for index, k in enumerate(keys, start=1):
        content_h.update(k.encode())
        content_h.update(b"\x00")
        content_h.update(asin_to_doc[k].encode("utf-8", errors="ignore"))
        content_h.update(b"\x00")
        if index % 100_000 == 0 or index == len(keys):
            log(f"  hash progress: {index}/{len(keys)} ASINs")
    return content_h.hexdigest()


def _corpus_signature(asin_to_doc: dict[str, str] | None = None,
                     meta_file: Path | None = None) -> str:
    """Fast corpus fingerprint.

    Formula: `sha1(meta_mtime+size + "|" + content_sha1)[:16]`

    - meta_mtime+size: META_FILE state change invalidates
    - content_sha1: cleaned content change invalidates
    - 16 hex truncation: compact fingerprint
    - 2026-09-23: fixed self-reference bug - old version read .sig file content as
      fingerprint input, but .sig content itself is the previous fingerprint (incl.
      meta mtime+size), causing the cache check vs build-end algorithm inputs to
      never match, creating an infinite mismatch loop. New version requires .sig
      line 1 to be content_sha1 (40 hex); cache check reads line 1 directly.
    """
    mf = Path(meta_file) if meta_file is not None else META_FILE
    st = mf.stat()
    if asin_to_doc is not None:
        content_sha1 = _asin_content_sha1(asin_to_doc)
    elif mf == META_FILE:
        sig_path = ASIN_TO_DOC_CACHE.with_suffix(ASIN_TO_DOC_CACHE.suffix + ".sig")
        if sig_path.exists():
            # .sig line 1 = content_sha1 (40 hex), remaining lines are backward-compat comment lines
            content_sha1 = sig_path.read_text(encoding="utf-8").splitlines()[0].strip()
        else:
            content_sha1 = "0" * 40
    else:
        content_sha1 = "0" * 40
    h = hashlib.sha1()
    h.update(f"{mf.name}|mtime={int(st.st_mtime)}|size={st.st_size}".encode())
    h.update(b"|content_sha1=")
    h.update(content_sha1.encode())
    return h.hexdigest()[:16]


def _sig_path_for(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".sig")


# ===========================================================================
# Main entry: build_meta_corpus
# ===========================================================================
def build_meta_corpus(force: bool = False) -> dict[str, str]:
    """Build and clean structured product documents, cached to ASIN_TO_DOC_CACHE.

    Args:
        force: Force rebuild (ignore sig hit). Default False -> on sig hit, read cache directly.

    Returns:
        dict[asin, doc_text].

    Raises:
        FileNotFoundError: META_FILE missing (Rule 7: no silent fallback)
        ValueError: META_FILE record missing required fields (Rule 7: no silent fallback)
    """
    sig_path = _sig_path_for(ASIN_TO_DOC_CACHE)
    if not force and ASIN_TO_DOC_CACHE.exists() and sig_path.exists():
        current_sig = _corpus_signature(meta_file=META_FILE)
        # .sig file format: line 1 = content_sha1 (40 hex), line 2 = fingerprint (16 hex).
        # Cache check compares the fingerprint (line 2), aligned with _corpus_signature output.
        cached_sig = sig_path.read_text(encoding="utf-8").splitlines()
        cached_fingerprint = cached_sig[1].strip() if len(cached_sig) >= 2 else cached_sig[0].strip()
        if cached_fingerprint == current_sig:
            t0 = time.time()
            asin_to_doc = json.load(open(ASIN_TO_DOC_CACHE, encoding="utf-8"))
            log(f"  ✓ asin_to_doc cache hit ({len(asin_to_doc)} ASINs, "
                f"sig={current_sig}, {time.time() - t0:.2f}s)")
            return asin_to_doc
        log(f"  ⚠ asin_to_doc sig mismatch (cached={cached_fingerprint}, "
            f"current={current_sig}), rebuilding...")
    elif not force and ASIN_TO_DOC_CACHE.exists() and not sig_path.exists():
        log(f"  ⚠ asin_to_doc sig missing, rebuilding...")
    else:
        log(f"  ⚠ asin_to_doc cache missing or force=True, building...")

    if not META_FILE.exists():
        raise FileNotFoundError(
            f"META_FILE not found: {META_FILE}. "
            f"Required for Stage 11/12/13/14 retrieval (Rule 7: no fallback)."
        )

    asin_to_doc: dict[str, str] = {}
    n_records = 0
    n_skip_no_title = 0
    n_cleaned_features = 0
    n_cleaned_desc = 0
    n_dropped_title_dup = 0
    log(f"  loading metadata from {META_FILE}")
    t0 = time.time()
    with open(META_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_records += 1
            if n_records % 50_000 == 0:
                log(f"  metadata progress: {n_records:,} records, "
                    f"{len(asin_to_doc):,} ASIN docs ({time.time() - t0:.1f}s)")
            asin = _clean_doc_text(r.get("parent_asin"))
            if not asin:
                raise ValueError(
                    f"Record #{n_records} missing required field 'parent_asin'"
                )
            details = r.get("details")
            if not isinstance(details, dict):
                raise ValueError(
                    f"Required details field is not an object for ASIN {asin}"
                )
            raw_title = r.get("title")
            # Missing title is expected (14 of 217k records, ~0.006%): skip record
            if not raw_title:
                n_skip_no_title += 1
                continue
            categories = r.get("categories")
            if not isinstance(categories, list):
                raise ValueError(
                    f"Required categories field is not a list for ASIN {asin}"
                )
            raw_features = r.get("features")
            if not isinstance(raw_features, list):
                raise ValueError(
                    f"Required features field is not a list for ASIN {asin}"
                )

            # Field-level cleaning
            title = _clean_doc_text(raw_title)
            brand = _clean_simple_field(_clean_doc_text(details.get("Brand")))
            category = " / ".join(
                _clean_simple_field(_clean_doc_text(x))
                for x in categories if x is not None
            )
            dimensions = _clean_simple_field(_clean_doc_text(details.get("Product Dimensions")))
            weight = _clean_simple_field(_clean_doc_text(details.get("Item Weight")))
            description = _clean_doc_text(r.get("description"))
            if description:
                new_desc = _clean_description(description, dimensions, weight, title)
                if new_desc != description:
                    n_cleaned_desc += 1
                description = new_desc
            raw_feature_values = [_clean_doc_text(x) for x in raw_features]
            raw_feature_values = [x for x in raw_feature_values if x]
            features = _clean_features(raw_feature_values, title)
            if len(features) < len(raw_feature_values):
                # at least one feature was cleaned away
                if features and features[0] != f"#1: {raw_feature_values[0]}":
                    n_dropped_title_dup += 1

            # Assemble
            lines = [f"product: {title}"]
            if brand:
                lines.append(f"brand: {brand}")
            if category:
                lines.append(f"category: {category}")
            if dimensions:
                lines.append(f"dimensions: {dimensions}")
            if weight:
                lines.append(f"weight: {weight}")
            if description:
                lines.append(f"description: {description}")
            if features:
                lines.append("features:")
                lines.extend(features)
            asin_to_doc[asin] = "\n".join(lines)
    log(f"  loaded {len(asin_to_doc)} ASIN docs from {n_records} records "
        f"({time.time() - t0:.1f}s, skipped_no_title={n_skip_no_title})")
    log(f"  cleaning stats: description_cleaned={n_cleaned_desc}, "
        f"features_title_dedup={n_dropped_title_dup}")
    log(f"  writing {len(asin_to_doc):,} ASIN documents to {ASIN_TO_DOC_CACHE}")

    ASIN_TO_DOC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(ASIN_TO_DOC_CACHE, "w", encoding="utf-8") as f:
        json.dump(asin_to_doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    # 2026-09-23: .sig line 1 writes content_sha1 (40 hex), line 2 writes fingerprint (16 hex).
    # Next cache check reads .sig line 1 -> inline-compute fingerprint -> guaranteed cache hit.
    content_sha1 = _asin_content_sha1(asin_to_doc)
    current_sig = _corpus_signature(asin_to_doc=asin_to_doc, meta_file=META_FILE)
    sig_path = _sig_path_for(ASIN_TO_DOC_CACHE)
    sig_path.write_text(f"{content_sha1}\n{current_sig}\n")
    log(f"  cached → {ASIN_TO_DOC_CACHE} "
        f"({ASIN_TO_DOC_CACHE.stat().st_size / 1e6:.1f} MB, "
        f"{time.time() - t0:.1f}s, sig={current_sig})")
    return asin_to_doc


# ===========================================================================
# CLI entry
# ===========================================================================
def main_task_body(force: bool = False) -> None:
    """Standalone run entry: generate and clean asin_to_doc.json."""
    log("=== build_asin_to_doc.py ===")
    log(f"  META_FILE = {META_FILE}")
    log(f"  OUTPUT    = {ASIN_TO_DOC_CACHE}")
    asin_to_doc = build_meta_corpus(force=force)
    log(f"  generated {len(asin_to_doc)} ASIN documents")
    log(f"  output → {ASIN_TO_DOC_CACHE}")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """User directive 2026-09-23: run 3 categories serially.

    For each category, bind META_FILE and ASIN_TO_DOC_CACHE to the category-specific path,
    then call main_task_body() to build the corpus cache at
    `$PQ_SCRATCH/stage11_corpus_cache/<subdir>/`.
    """
    global SENT_CACHE, UID_TO_SENTS, ASIN_USERS_PATH, ATTRIBUTES_PATH, META_FILE, OUT_DIR, OUT_PATH, ASIN_TO_DOC_CACHE  # noqa
    # backup current (Baby) defaults
    saved = {
        k: v for k, v in globals().items()
        if k in {"SENT_CACHE", "UID_TO_SENTS", "ASIN_USERS_PATH", "ATTRIBUTES_PATH",
                 "META_FILE", "OUT_DIR", "OUT_PATH", "ASIN_TO_DOC_CACHE"}
        and isinstance(v, Path)
    }
    base_out = REPO_ROOT / "result" / Path(__file__).parent.name
    for category, subdir in CATEGORY_INPUTS:
        log(f"\n========== [{category}] (subdir={subdir}) ==========")
        # Reset all known category-dependent paths to point at the per-category subdir.
        if "SENT_CACHE" in saved:
            SENT_CACHE = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "UID_TO_SENTS" in saved:
            UID_TO_SENTS = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "ASIN_USERS_PATH" in saved:
            ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract" / f"asin_to_users_{subdir}.pkl"
        if "ATTRIBUTES_PATH" in saved:
            ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction" / f"product_attributes_{subdir}.pkl"
        if "META_FILE" in saved:
            META_FILE = DATA_DIR / {
                "baby": "meta_Baby_Products_2023.jsonl",
                "musical": "meta_Musical_Instruments.jsonl",
                "video_games": "meta_Video_Games.jsonl",
            }[subdir]
        if "OUT_DIR" in saved:
            OUT_DIR = base_out / subdir
        if "OUT_PATH" in saved:
            OUT_PATH = base_out / subdir / saved["OUT_PATH"].name
        if "ASIN_TO_DOC_CACHE" in saved:
            ASIN_TO_DOC_CACHE = (SCRATCH / "stage11_corpus_cache"
                                 / subdir / saved["ASIN_TO_DOC_CACHE"].name)
        OUT_DIR.mkdir(parents=True, exist_ok=True) if "OUT_DIR" in saved else None
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True) if "OUT_PATH" in saved else None
        ASIN_TO_DOC_CACHE.parent.mkdir(parents=True, exist_ok=True) if "ASIN_TO_DOC_CACHE" in saved else None
        try:
            main_task_body()
        except Exception as e:
            log(f"[{category}] FAILED: {e!r}")
            raise
    # Restore Baby defaults (for import compatibility with downstream).
    for k, v in saved.items():
        globals()[k] = v


if __name__ == "__main__":
    main()