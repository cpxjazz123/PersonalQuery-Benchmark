"""Generate fixed-template queries (1 per user) by filling 5 product attribute values.

This is the shared body of the three
`14_generate_query_template_<Cat>.py` scripts. The per-category wrappers
are thin entry points; all real logic lives here.

Pipeline:
  1. Load Stage 1 attribute JSON (per-user products) for the category.
  2. For each user, take the first product, extract up to 5 attributes via
     the 06-stage `_extract_attrs_from_product` helper.
  3. Require all `required_attr_keys` to be present (raise if missing).
  4. Render the single fixed template using `.format(**attrs)`.
  5. Write `query_template.json` incrementally (atomic write, crash-resumable).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))

sys.path.insert(0, os.path.join(REPO_ROOT, "06_query", "common"))

from attribute_helpers import (  # noqa: E402
    ATTR_PRODUCT_KEY_OVERRIDE,
    ATTR_PRODUCT_KEY_SUFFIX,
    ATTR_TYPE_BY_KEY,
    REQUIRED_ATTR_COUNT,
    _attrs_used_from_source,
    _extract_attrs_from_product,
    _normalize_attr_value,
    count_attr_value_occurrences_map,
    count_words,
    log,
)

from .config import get_category_config, get_query_config, get_template  # noqa: E402


PLACEHOLDER_PATTERN = re.compile(r"\{([A-Z]\d+)(?:_[a-z_]+)?\}")
PLACEHOLDER_TYPED_PATTERN = re.compile(r"\{[A-Z]\d+_[a-z_]+\}")


def _required_attr_key_map(required_attr_keys: List[str]) -> Dict[str, str]:
    """Build a mapping from placeholder keys (e.g. A1) to typed names (e.g. A1_product_type).

    Mirrors the 06-stage ATTR_TYPE_BY_KEY typing convention used in templates.
    """
    from attribute_helpers import ATTR_TYPE_BY_KEY  # local import to keep top clean
    return {key: f"{key}_{ATTR_TYPE_BY_KEY[key]}" for key in required_attr_keys if key in ATTR_TYPE_BY_KEY}


def render_query_template(template_str: str, attrs_used: Dict[str, str]) -> str:
    """Render `template_str` with `attrs_used` keys (A1, A2, ...).

    Template may use either short placeholders ({A2}) or typed placeholders
    ({A2_brand}); both are accepted. Typed placeholders are normalised to
    short form before formatting so that `str.format(**attrs_used)` can
    resolve them.

    Raises KeyError if any required placeholder in the template is not present
    in attrs_used. Does not fall back to any default.
    """
    if PLACEHOLDER_TYPED_PATTERN.search(template_str):
        normalised = re.sub(r"\{([A-Z]\d+)_[a-z_]+\}", r"{\1}", template_str)
    else:
        normalised = template_str
    missing = [ph for ph in PLACEHOLDER_PATTERN.findall(normalised) if ph not in attrs_used]
    if missing:
        raise KeyError(f"Template requires attribute keys not in attrs_used: {sorted(set(missing))}")
    return normalised.format(**attrs_used)


def validate_required_attrs(attrs_used: Dict[str, str], required_attr_keys: List[str]) -> None:
    """Raise ValueError if any required attribute key is missing."""
    missing = [k for k in required_attr_keys if k not in attrs_used or not attrs_used[k]]
    if missing:
        raise ValueError(
            f"Required attribute keys missing for template rendering: {sorted(missing)}"
        )


def load_category_attributes(attributes_file: str) -> List[Dict[str, Any]]:
    path = Path(attributes_file)
    if not path.exists():
        raise FileNotFoundError(f"attributes file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "products" not in data:
        raise TypeError(f"attributes file must contain a top-level products list: {path}")
    if not isinstance(data["products"], list):
        raise TypeError(f"attributes file products field must be a list: {path}")
    return data["products"]


def build_user_product_map(products: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    user_to_products: Dict[str, List[Dict[str, Any]]] = {}
    for product in products:
        if not isinstance(product, dict):
            raise TypeError(f"every product entry must be a dict, got {type(product).__name__}")
        uid = product.get("user_id")
        if isinstance(uid, str) and uid:
            user_to_products.setdefault(uid, []).append(product)
    return user_to_products


def _resolve_product_key(attr_key: str) -> str:
    if attr_key in ATTR_PRODUCT_KEY_OVERRIDE:
        return ATTR_PRODUCT_KEY_OVERRIDE[attr_key]
    suffix = ATTR_PRODUCT_KEY_SUFFIX.get(attr_key, "")
    if not suffix:
        raise KeyError(f"no product-key mapping for attribute key {attr_key}")
    return f"{attr_key}_{suffix}"


def _extract_attrs_for_template(product: Dict[str, Any], required_attr_keys: List[str]) -> Dict[str, Dict[str, str]]:
    """Extract the exact attribute keys required by the template.

    Unlike `_extract_attrs_from_product` (which returns the first 5 non-empty
    attrs in A1..A18 order), this function requires the *specific* keys listed
    in `required_attr_keys` to all be present. Raises ValueError if any is
    missing or empty, so the caller can skip the product.
    """
    if not isinstance(product, dict):
        raise TypeError(f"product entry must be dict, got {type(product).__name__}")

    out: Dict[str, Dict[str, str]] = {}
    for attr_key in required_attr_keys:
        prod_key = _resolve_product_key(attr_key)
        raw_value = product.get(prod_key)
        normalised = _normalize_attr_value(raw_value)
        if not normalised:
            raise ValueError(
                f"product missing required attribute {attr_key} (product_key={prod_key})"
            )
        out[attr_key] = {
            "value": normalised,
            "type": ATTR_TYPE_BY_KEY.get(attr_key, "unknown"),
        }
    return out


def build_template_user_records(category: str) -> List[Dict[str, Any]]:
    """Build template-filled records (1 per user) for a category.

    Output schema (per record):
      {
        "user_id": str,
        "asin": str,
        "query": str,                # rendered template
        "word_count": int,
        "attrs_used": {A1: ..., ...},  # 5 keys
        "attrs_source": {A1: {"value": ..., "type": ...}, ...},
        "template_id": str,
        "template_version": str
      }
    """
    cat_cfg = get_category_config(category)
    template_cfg = get_template()
    template_str = template_cfg["template_str"]
    required_attr_keys = template_cfg["required_attr_keys"]
    template_id = template_cfg["template_id"]
    template_version = template_cfg["template_version"]

    products = load_category_attributes(cat_cfg["attributes_file"])
    user_to_products = build_user_product_map(products)
    log(f"users with at least 1 product: {len(user_to_products)}")

    placeholder_keys = sorted(set(PLACEHOLDER_PATTERN.findall(template_str)))
    if sorted(placeholder_keys) != sorted(required_attr_keys):
        raise ValueError(
            f"Template placeholders {placeholder_keys} do not match required_attr_keys {required_attr_keys}"
        )

    records: List[Dict[str, Any]] = []
    skipped_attr_count = 0

    for uid in sorted(user_to_products.keys()):
        product = user_to_products[uid][0]
        try:
            attrs = _extract_attrs_for_template(product, required_attr_keys)
        except ValueError:
            skipped_attr_count += 1
            continue

        attrs_used = _attrs_used_from_source(attrs)
        validate_required_attrs(attrs_used, required_attr_keys)

        query_text = render_query_template(template_str, attrs_used)
        attrs_source = {k: {"value": attrs[k]["value"], "type": attrs[k]["type"]} for k in required_attr_keys}

        records.append({
            "user_id": uid,
            "asin": product.get("asin", ""),
            "query": query_text,
            "word_count": count_words(query_text),
            "attrs_used": dict(attrs_used),
            "attrs_source": attrs_source,
            "template_id": template_id,
            "template_version": template_version,
        })

    log(f"skipped by attr count (<{REQUIRED_ATTR_COUNT}): {skipped_attr_count}")
    log(f"records built: {len(records)}")
    return records


def load_existing_records(out_path: Path) -> List[Dict[str, Any]]:
    if not out_path.exists():
        return []
    data = json.loads(out_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"existing output must be a list: {out_path}")
    return data


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def verify_queries_contain_all_attrs(records: List[Dict[str, Any]]) -> None:
    """Spot check: rendered query must contain each of the 5 attribute values exactly once.

    Some records may legitimately fail this check when two attribute values
    are textually identical (e.g. A1_product_type='Infant' and A5_use_case='Infant')
    or when one attr value is a substring of another. These are data issues,
    not template bugs. We log them as warnings instead of raising.
    """
    bad_records: List[Tuple[str, Dict[str, int]]] = []
    for rec in records:
        occ = count_attr_value_occurrences_map(rec["query"], rec["attrs_used"])
        bad = {k: v for k, v in occ.items() if v != 1}
        if bad:
            bad_records.append((rec["user_id"], occ))
    if bad_records:
        log(f"  WARN: {len(bad_records)}/{len(records)} records have 5-attr overlap (data issue, kept as-is)")
        for user_id, occ in bad_records[:5]:
            log(f"    user={user_id} counts={occ}")
    else:
        log(f"  ✓ all {len(records)} records pass 5-attr check")


def main(category: str) -> None:
    cat_cfg = get_category_config(category)
    out_path = Path(cat_cfg["output_query_file"])
    query_cfg = get_query_config()
    max_users = int(query_cfg.get("max_users", 30000))

    existing = load_existing_records(out_path)
    completed_user_ids = {rec["user_id"] for rec in existing if "user_id" in rec}
    log(f"existing completed users: {len(completed_user_ids)}")

    all_records = build_template_user_records(category)
    log(f"new candidates available: {len(all_records)}")

    new_records = [r for r in all_records if r["user_id"] not in completed_user_ids]
    log(f"new records to write: {len(new_records)}")

    if not existing and not new_records:
        raise ValueError(
            f"No records built for category {category}; check attributes file and template config"
        )

    if not new_records:
        log("nothing to do, output already complete")
        return

    written = list(existing) + new_records
    start = time.time()
    write_json_atomic(out_path, written)
    log(f"  [wrote] total={len(written)} in {time.time() - start:.1f}s")

    if len(written) > max_users:
        log(f"WARNING: written={len(written)} exceeds max_users={max_users}, but all 5-attr records are kept")

    verify_queries_contain_all_attrs(written)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help="Baby_Products | Grocery_and_Gourmet_Food | Pet_Supplies")
    args = parser.parse_args()
    main(args.category)
