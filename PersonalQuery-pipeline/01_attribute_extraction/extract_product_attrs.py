#!/usr/bin/env python3
"""Product attribute extraction from meta_<category>_2023.jsonl.

Reads product metadata for one or more Amazon categories, extracts structured
attributes per ASIN (Brand, Main Category, Color, Material, Size, ..., no
numeric values), and writes one pickle file per category:

  - result/01_attribute_extraction/product_attributes_baby.pkl         (Baby)
  - result/01_attribute_extraction/product_attributes_musical.pkl      (Musical_Instruments)
  - result/01_attribute_extraction/product_attributes_video_games.pkl  (Video_Games)

Each output dict maps asin -> {field_name: string_value}. Three categories
get separate files; not merged (user directive 2026-09-23).

2026-09-23: switch to pickle output (downstream is Python-only; dropping json
saves ~30MB).

Also exports select_top_attrs() utility used by gaussian/build_user.py
to pick top-N attrs per ASIN with numeric value exclusion.

All parameters hard-coded (Rule 3); no CLI arguments.
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import re
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# === Inputs ===
# User directive 2026-09-23: data dir migrated wholesale from
# ar57 (PersoanlQuery/data) to hj82; cross-LUSTRE same-disk move preserving
# all existing product paths. Downstream scripts update DATA_DIR uniformly.
DATA_DIR = Path(os.environ.get("PQ_DATA_DIR", str(REPO_ROOT / "data")))

# User directive 2026-09-23: each of three categories gets its own meta and
# its own output file (not merged). All three share the same extract /
# filter / select logic; output filenames are distinguished by a
# semantic category suffix (not a version number — Rule 17 compliant).
# User directive 2026-09-23: switch to .pkl (downstream Python-only).
META_INPUTS = [
    # (category_key, input_path, output_path)
    # User directive 2026-09-23: keep Baby / Musical_Instruments / Video_Games.
    ("Baby",               DATA_DIR / "meta_Baby_Products_2023.jsonl",
                           REPO_ROOT / "result" / "01_attribute_extraction" / "product_attributes_baby.pkl"),
    ("Musical_Instruments", DATA_DIR / "meta_Musical_Instruments.jsonl",
                           REPO_ROOT / "result" / "01_attribute_extraction" / "product_attributes_musical.pkl"),
    ("Video_Games",         DATA_DIR / "meta_Video_Games.jsonl",
                           REPO_ROOT / "result" / "01_attribute_extraction" / "product_attributes_video_games.pkl"),
]

# User directive 2026-09-04: multiprocessing speedup (13 cores, reserve 3
# for OS/IO).
N_WORKERS = min(10, max(1, (os.cpu_count() or 4) - 3))
# chunk size: each worker takes N lines at once to reduce IPC overhead
CHUNK_SIZE = 5_000

# === Step 1 — extract_attrs parameters ===
MAX_STR_LEN = 200
TOP_LEVEL_NUMERIC_FIELDS = ("average_rating", "rating_number", "price")

# User directive 2026-09-04: filter attrs whose value contains digits at the
# source (single-attr dimension).
EXCLUDE_NUMERIC_VALUES = True

# User directive 2026-09-04: after numeric filtering, skip the whole ASIN
# when its surviving attr count drops below MIN_ATTRS_PER_ASIN.
MIN_ATTRS_PER_ASIN = 5

# User directive 2026-09-04: phrase attr filter — a value whose word count
# exceeds MAX_VALUE_WORDS is treated as a phrase and skipped (>3-word
# values are usually full Feature sentences that are hard for an LLM to
# inject naturally into a query).
MAX_VALUE_WORDS = 3

# User directive 2026-09-04: yes/no single-value attr filter — an attr whose
# value is yes/no/true/false is binary metadata (e.g. 'Is Discontinued By
# Manufacturer: Yes') that carries no information for natural-query
# generation, so skip it.
_YES_NO_VALUES = {"yes", "no", "true", "false"}


def _has_ascii_alpha(s: str) -> bool:
    """Whether value/key contains at least one ASCII letter."""
    return any(c.isascii() and c.isalpha() for c in str(s))


def _is_non_cjk(s: str) -> bool:
    """Whether value/key contains no CJK (Chinese/Japanese/Korean) chars.
    Accented Latin (ü/é/ñ) is allowed."""
    return not any(
        0x4E00 <= ord(c) <= 0x9FFF  # CJK Unified Ideographs
        or 0x3040 <= ord(c) <= 0x309F  # Hiragana
        or 0x30A0 <= ord(c) <= 0x30FF  # Katakana
        or 0xAC00 <= ord(c) <= 0xD7AF  # Hangul
        for c in str(s)
    )

# === select_top_attrs parameters (shared with downstream cohort construction) ===
MAX_ATTRS = 5
MAX_ATTR_VALUE_LEN = 100
EXCLUDE_NUMERIC_ATTRS = True
_NUMERIC_KEYWORDS = {"price", "average rating", "rating number", "item weight",
                     "item model number", "date first available",
                     "package dimensions", "product dimensions",
                     "minimum weight recommendation",
                     "maximum weight recommendation",
                     "batteries required", "is discontinued by manufacturer"}
# User directive 2026-08-29: filter non-semantic attribute types as well
# (treated the same as numeric attrs). These keys describe product
# metadata (routing/category/origin/counts/date/rank) rather than the
# product's own semantic features (brand/color/material/size, etc.).
# They are useless when an LLM uses them to generate a query — e.g.
# "Main Category: Baby Products" adds nothing to a query.
_NON_SEMANTIC_KEYWORDS = {
    # routing/category (not a feature)
    "main category", "department",
    # origin (metadata)
    "country", "country of origin", "country/region of origin",
    # counts (not a feature)
    "number of items", "number of pieces", "unit count",
    # date / rank (metadata)
    "date", "date listed", "best sellers rank",
}
ATTR_PRIORITY = [
    "Brand", "Main Category", "Item model number", "Manufacturer",
    "Color", "Material", "Material Type", "Fabric Type", "Frame Material",
    "Item Weight", "Product Dimensions", "Size", "Style", "Pattern", "Theme",
    "Age Range (Description)", "Special Feature", "Target gender",
    "Batteries required", "Number Of Items", "Is Discontinued By Manufacturer",
    "Date First Available", "Country/Region of origin", "Country of Origin",
    "Price", "Average Rating", "Rating Number",
]


def log(msg: str) -> None:
    print(f"[extract_product_attrs] {msg}", flush=True)


def has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)


def extract_attrs(d: dict) -> dict:
    """Extract structured attributes from a single product meta dict.

    Supports two meta formats:
    - New (2023): features (list) + details (dict)
    - Old (2018): top-level fields only

    Returns dict {field_name: string_value}. No numeric value filtering
    here — that is select_top_attrs()'s job (called downstream).
    """
    out: dict = {}

    # New 2023 format: details dict
    details = d.get("details")
    if isinstance(details, dict):
        for k, v in details.items():
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s or len(s) > MAX_STR_LEN:
                continue
            out[k] = s

    # New 2023 format: features list
    features = d.get("features")
    if isinstance(features, list):
        for i, v in enumerate(features):
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s or len(s) > MAX_STR_LEN:
                continue
            # features are anonymous; name them by index
            out[f"Feature {i+1}"] = s

    # Old 2018 format: top-level fields (also used as fallback)
    for field in ("brand", "main_category"):
        v = d.get(field)
        if isinstance(v, str) and v.strip():
            name = "Brand" if field == "brand" else "Main Category"
            if name not in out:
                out[name] = v.strip()

    for field in TOP_LEVEL_NUMERIC_FIELDS:
        v = d.get(field)
        if v is None or v == "":
            continue
        out[field.replace("_", " ").title().replace(" ", " ")] = v

    # User directive 2026-09-04: filter (k, v) whose value contains digits
    # at the source (single-attr dimension)
    if EXCLUDE_NUMERIC_VALUES:
        out = {k: v for k, v in out.items() if not has_digit(str(v))}

    # User directive 2026-09-04: value de-duplication (case-insensitive) —
    # if the same value already appeared, drop the new (k, v) and keep the
    # first-seen key
    seen_values: set[str] = set()
    deduped: dict = {}
    for k, v in out.items():
        v_key = str(v).strip().lower()
        if v_key in seen_values:
            continue
        seen_values.add(v_key)
        deduped[k] = v
    out = deduped

    # User directive 2026-09-04: phrase filter — value word count above
    # MAX_VALUE_WORDS is treated as a phrase and skipped
    out = {
        k: v
        for k, v in out.items()
        if len(re.findall(r"\S+", str(v))) <= MAX_VALUE_WORDS
    }

    # User directive 2026-09-04: yes/no binary value filter
    out = {
        k: v
        for k, v in out.items()
        if str(v).strip().lower() not in _YES_NO_VALUES
    }

    # User directive 2026-09-04: non-English key/value filter — both key
    # and value must contain at least one ASCII letter (CJK / pure digits
    # / pure symbols are all skipped)
    out = {
        k: v
        for k, v in out.items()
        if _has_ascii_alpha(k) and _has_ascii_alpha(v) and _is_non_cjk(v)
    }

    # User directive 2026-09-04: take first segment of multi-value — when a
    # value contains a comma, keep only the first segment (e.g.
    # 'Lightweight, Breathable' → 'Lightweight'), then re-run yes/no /
    # phrase / English filtering on the first segment
    cleaned: dict = {}
    for k, v in out.items():
        if "," in str(v):
            first = str(v).split(",", 1)[0].strip()
            if not first:
                continue
            # re-run strict filtering
            if (
                _has_ascii_alpha(k)
                and _has_ascii_alpha(first)
                and _is_non_cjk(first)
                and first.lower() not in _YES_NO_VALUES
                and len(re.findall(r"\S+", first)) <= MAX_VALUE_WORDS
            ):
                cleaned[k] = first
        else:
            cleaned[k] = v
    out = cleaned

    return out


def select_top_attrs(asin_attrs: dict, max_n: int = MAX_ATTRS) -> dict:
    """Select top-N attributes for an ASIN, excluding numeric values.

    Used by both Stage 1 vLLM prompt construction and downstream cohort
    selection. Returns dict {field_name: string_value}, with priority
    order ATTR_PRIORITY first, then any remaining fields.

    Numeric exclusion: any value with digits, or any key in _NUMERIC_KEYWORDS,
    is dropped. Implements user directive 2026-08-28: "filter numeric attr values".

    Args:
        asin_attrs: dict of field_name -> string_value from extract_attrs()
        max_n: maximum number of attrs to return (Stage 1 = 5, Stage 4 cohort = 4)
    """
    def _skip(k: str, s: str) -> bool:
        if not s or len(s) > MAX_ATTR_VALUE_LEN:
            return True
        k_low = k.lower()
        # User directive 2026-08-29: non-semantic attribute type filter (treated
        # the same as numeric attrs)
        if any(nk in k_low for nk in _NON_SEMANTIC_KEYWORDS):
            return True
        if EXCLUDE_NUMERIC_ATTRS:
            if any(nk in k_low for nk in _NUMERIC_KEYWORDS):
                return True
            if has_digit(s):
                return True
        return False

    out: dict = {}
    used: set[str] = set()
    for k in ATTR_PRIORITY:
        v = asin_attrs.get(k)
        if not v:
            continue
        s = str(v).strip()
        if _skip(k, s):
            continue
        out[k] = s
        used.add(k)
        if len(out) >= max_n:
            return out
    for k, v in asin_attrs.items():
        if k in used:
            continue
        if v is None:
            continue
        s = str(v).strip()
        if _skip(k, s):
            continue
        out[k] = s
        if len(out) >= max_n:
            break
    return out


def _process_one(raw_line: str):
    """Worker entry: 1 metadata line → (asin, attrs) or None.

    Must be module-level + pure function so multiprocessing.Pool can pickle
    it. Returning None signals skip (no asin or attrs < MIN); the main
    process ignores None during reduce.
    """
    line = raw_line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    asin = d.get("parent_asin") or d.get("asin")
    if not asin:
        return None
    attrs = extract_attrs(d)
    if len(attrs) < MIN_ATTRS_PER_ASIN:
        return None
    return (asin, attrs)


def _process_chunk(chunk: list[str]):
    """Worker entry (chunked): N metadata lines → list of (asin, attrs).

    Reduces IPC round-trips versus _process_one; the main process feeds
    chunks via imap_unordered.
    """
    out = []
    for line in chunk:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        asin = d.get("parent_asin") or d.get("asin")
        if not asin:
            continue
        attrs = extract_attrs(d)
        if len(attrs) < MIN_ATTRS_PER_ASIN:
            continue
        out.append((asin, attrs))
    return out


def step1_extract_attrs(category: str, meta_path: Path,
                        out_path: Path) -> dict[str, dict]:
    """Extract one category's meta → attrs dict and write it to out_path.

    User directive 2026-09-23: expanded from a single Baby run to three
    categories (Baby / Pet_Supplies / Grocery_and_Gourmet_Food); each
    category gets its own output file.
    """
    log(f"=== Step 1: extract_attrs [{category}] ===")
    log(f"  input={meta_path}")
    log(f"  output={out_path}")
    log(f"  workers={N_WORKERS}, chunk_size={CHUNK_SIZE}")

    product_attrs: dict[str, dict] = {}
    n_total = 0
    n_with_attrs = 0
    n_with_details = 0
    n_top_level_only = 0
    field_counter: dict[str, int] = {}

    # Main process reads all lines into memory at once (1.5 GB) to avoid
    # workers contending for the same file descriptor after fork (POSIX
    # behaviour).
    if not meta_path.exists():
        raise FileNotFoundError(f"meta file not found: {meta_path}")
    _open = gzip.open if str(meta_path).endswith('.gz') else open
    mode = 'rt' if _open is gzip.open else 'r'
    with _open(meta_path, mode) as f:
        all_lines = f.readlines()
    n_total = len(all_lines)
    log(f"  read {n_total} lines from metadata; dispatching to {N_WORKERS} workers")

    # chunked dispatch
    chunks = [
        all_lines[i : i + CHUNK_SIZE]
        for i in range(0, len(all_lines), CHUNK_SIZE)
    ]
    with Pool(processes=N_WORKERS) as pool:
        for sub in pool.imap_unordered(_process_chunk, chunks):
            for asin, attrs in sub:
                if attrs:
                    product_attrs[asin] = attrs
                    n_with_attrs += 1
                    has_details = any(
                        k not in ("Brand", "Main Category", "Average Rating",
                                  "Rating Number", "Price")
                        for k in attrs
                    )
                    if has_details:
                        n_with_details += 1
                    else:
                        n_top_level_only += 1
                    for k in attrs:
                        field_counter[k] = field_counter.get(k, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # User directive 2026-09-23: switch to pickle.dump (downstream
    # Python-only; dropping json saves ~30MB).
    with open(out_path, 'wb') as f:
        pickle.dump(product_attrs, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  [{category}] total metadata products={n_total}, with attrs={n_with_attrs}")
    log(f"    details-based={n_with_details}, top-level only={n_top_level_only}")
    log(f"    distinct attr fields={len(field_counter)}")
    log(f"    skipped ASINs (filtered attrs <{MIN_ATTRS_PER_ASIN})={n_total - n_with_attrs}")
    log(f"  wrote → {out_path}")
    return product_attrs


def main() -> None:
    log("=== extract_product_attrs.py — Step 1 (multi-category) ===")
    for category, meta_path, out_path in META_INPUTS:
        step1_extract_attrs(category, meta_path, out_path)
    log("=== DONE ===")


if __name__ == "__main__":
    main()
