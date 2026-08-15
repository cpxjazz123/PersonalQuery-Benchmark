"""Attribute extraction, normalization, and 5-attribute validation helpers.

Shared by 06_query stage-6 scripts that build expression_style queries. These
helpers were originally part of `06_generate_by_persona_placeholder_*.py`
(which has since been removed) and were inlined into the depth-checked
expression_style scripts; this module re-homes them so that the
no-depth-check variants can import them instead of dynamic-loading the
removed placeholder scripts.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation


REQUIRED_ATTR_COUNT = 5
VALID_ATTR_KEYS = {f'A{i}' for i in range(1, 19)}

PLACEHOLDER_ATTR_VALUES = {
    'none',
    'n/a',
    'na',
    'null',
    'unknown',
    'see description',
}

ATTR_TYPE_BY_KEY = {
    'A1': 'product_type', 'A2': 'brand', 'A3': 'price', 'A4': 'appearance',
    'A5': 'use_case', 'A6': 'detailed', 'A7': 'material', 'A8': 'safety',
    'A9': 'durability', 'A10': 'ease_of_use', 'A11': 'temperature_resistance',
    'A12': 'surface', 'A13': 'reusability', 'A14': 'size', 'A15': 'weight',
    'A16': 'compatibility', 'A17': 'flavor', 'A18': 'quality',
}

ATTR_PRODUCT_KEY_OVERRIDE = {
    'A1': 'A1_product_type',
}

ATTR_PRODUCT_KEY_SUFFIX = {
    'A2': 'brand', 'A3': 'price', 'A4': 'appearance', 'A5': 'use_case',
    'A6': 'detailed', 'A7': 'material', 'A8': 'safety', 'A9': 'durability',
    'A10': 'ease_of_use', 'A11': 'temperature_resistance', 'A12': 'surface',
    'A13': 'reusability', 'A14': 'size', 'A15': 'weight', 'A16': 'compatibility',
    'A17': 'flavor', 'A18': 'quality',
}

SKIP_ATTR_KEYS = {'A14'}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def count_words(text: str) -> int:
    return len(text.split())


def _format_dict_key(key: str) -> str:
    return re.sub(r'\s+', ' ', key.replace('_', ' ')).strip()


def _is_placeholder_attr_text(raw_value: str) -> bool:
    normalized = re.sub(r'\s+', ' ', raw_value).strip().casefold()
    return normalized in PLACEHOLDER_ATTR_VALUES


def _canonicalize_attr_text(raw_value: str) -> str:
    raw_value = re.sub(r'\s+', ' ', raw_value).strip()
    if '(' in raw_value:
        prefix = raw_value.split('(', 1)[0].strip()
        if prefix:
            raw_value = prefix
    return raw_value.strip()


def _normalize_scalar_attr(value) -> str | None:
    if value is None or value == '':
        return None
    if isinstance(value, (int, float)):
        raw_value = str(value).strip()
    elif isinstance(value, str):
        raw_value = value.strip()
    else:
        return None
    if not raw_value:
        return None
    if ';' in raw_value:
        raw_value = raw_value.split(';')[0].strip()
    elif ',' in raw_value:
        raw_value = raw_value.split(',')[0].strip()
    raw_value = _canonicalize_attr_text(raw_value)
    if not raw_value or _is_placeholder_attr_text(raw_value):
        return None
    return raw_value if raw_value else None


def _normalize_attr_value(value) -> str | None:
    if isinstance(value, dict):
        parts = []
        for child_key, child_value in value.items():
            child_text = _normalize_attr_value(child_value)
            if child_text:
                parts.append(f"{_format_dict_key(str(child_key))}: {child_text}")
        return '; '.join(parts) if parts else None
    if isinstance(value, list):
        for item in value:
            item_text = _normalize_attr_value(item)
            if item_text:
                return item_text
        return None
    return _normalize_scalar_attr(value)


def _normalize_attr_identity(raw_value: str) -> str:
    normalized = _canonicalize_attr_text(raw_value).casefold()
    try:
        numeric_value = Decimal(normalized)
    except InvalidOperation:
        numeric_value = None
    if numeric_value is not None:
        return f"NUM::{format(numeric_value.normalize(), 'f')}"
    return f"TEXT::{normalized}"


def _iter_attr_value_candidates(value):
    if isinstance(value, dict):
        for child_value in value.values():
            yield from _iter_attr_value_candidates(child_value)
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_attr_value_candidates(item)
        return
    normalized = _normalize_scalar_attr(value)
    if normalized:
        yield normalized


def _resolve_attr_product_key(key: str) -> str:
    if key in ATTR_PRODUCT_KEY_OVERRIDE:
        return ATTR_PRODUCT_KEY_OVERRIDE[key]
    suffix = ATTR_PRODUCT_KEY_SUFFIX.get(key, '')
    if not suffix:
        return ''
    return f'{key}_{suffix}'


def _extract_attrs_from_product(prod: dict) -> dict:
    """Pick up to REQUIRED_ATTR_COUNT non-empty attributes from a product entry,
    preferring the canonical A1..A18 ordering. Skips A14 (size).
    """
    attr_keys = [f'A{i}' for i in range(1, 19)]
    attrs: dict = {}
    used_value_identities: set = set()
    for key in attr_keys:
        if len(attrs) >= REQUIRED_ATTR_COUNT:
            break
        if key in SKIP_ATTR_KEYS:
            continue
        prod_key = _resolve_attr_product_key(key)
        if not prod_key:
            continue
        raw_prod_value = prod.get(prod_key)
        selected_value = None
        for candidate in _iter_attr_value_candidates(raw_prod_value):
            identity = _normalize_attr_identity(candidate)
            if identity in used_value_identities:
                continue
            selected_value = candidate
            used_value_identities.add(identity)
            break
        if selected_value:
            attrs[key] = {'value': selected_value, 'type': ATTR_TYPE_BY_KEY.get(key, 'unknown')}
            if len(attrs) >= REQUIRED_ATTR_COUNT:
                break
    return attrs


def _format_attrs_for_prompt(attrs: dict) -> str:
    """Format selected attributes for inclusion in the LLM prompt, skipping
    placeholder values.
    """
    lines = []
    for key in attrs.keys():
        info = attrs.get(key)
        if isinstance(info, str):
            value = info if info and not _is_placeholder_attr_text(info) else ''
            attr_type = 'unknown'
        else:
            value = (info.get('value', '') or '') if info else ''
            attr_type = info.get('type', 'unknown') if info else 'unknown'
        if value and not _is_placeholder_attr_text(value):
            lines.append(f"- {key} ({attr_type}): {value}")
    return '\n'.join(lines)


def _attrs_used_from_source(attrs: dict) -> dict:
    """Return the source product attributes used for query generation."""
    if not attrs:
        raise ValueError("source attrs must not be empty")
    if len(attrs) < REQUIRED_ATTR_COUNT:
        raise ValueError(
            f"source attrs must contain at least {REQUIRED_ATTR_COUNT} attributes, got {len(attrs)}"
        )
    attrs_used: dict = {}
    for key, info in attrs.items():
        if not isinstance(info, dict):
            raise TypeError(f"source attr {key} must be a dict")
        if 'value' not in info:
            raise KeyError(f"source attr {key} is missing value")
        value = info['value']
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"source attr {key}.value must be a non-empty string")
        attrs_used[key] = value.strip()
    return attrs_used


def _normalize_variant_token(token: str) -> str:
    token = token.casefold()
    if len(token) > 4 and token.endswith('ies'):
        token = token[:-3] + 'y'
    elif len(token) > 3 and token.endswith('es') and not token.endswith(('aes', 'ees', 'oes')):
        token = token[:-2]
    elif len(token) > 3 and token.endswith('s') and not token.endswith('ss'):
        token = token[:-1]

    for suffix in ('ingly', 'edly', 'ing', 'ed', 'en', 'ly', 'ness', 'ment', 'er', 'est', 'al', 'ic', 'ish'):
        if len(token) - len(suffix) >= 3 and token.endswith(suffix):
            token = token[:-len(suffix)]
            break
    return token


def _build_attr_value_pattern(attr_value: str) -> str | None:
    if not isinstance(attr_value, str) or not attr_value.strip():
        return None
    attr_value = _canonicalize_attr_text(attr_value)
    if not attr_value:
        return None
    try:
        numeric_value = Decimal(attr_value)
    except InvalidOperation:
        numeric_value = None
    if numeric_value is not None:
        normalized = format(numeric_value.normalize(), 'f')
        if '.' in normalized:
            integer_part, fractional_part = normalized.split('.', 1)
            return rf"(?<![A-Za-z0-9])\$?{re.escape(integer_part)}\.{re.escape(fractional_part)}(?:0+)?(?![A-Za-z0-9])"
        return rf"(?<![A-Za-z0-9])\$?{re.escape(normalized)}(?:\.0+)?(?![A-Za-z0-9])"
    if re.fullmatch(r"[A-Za-z0-9' ]+", attr_value):
        return rf"\b{re.escape(attr_value)}\b"
    return re.escape(attr_value)


def _find_variant_token_spans(query: str, attr_value: str) -> list[tuple[int, int]]:
    attr_value = _canonicalize_attr_text(attr_value)
    attr_tokens = [match.group(0) for match in re.finditer(r"[A-Za-z0-9']+", attr_value)]
    if not attr_tokens:
        return []
    query_tokens = list(re.finditer(r"[A-Za-z0-9']+", query))
    if len(query_tokens) < len(attr_tokens):
        return []

    normalized_attr_tokens = [_normalize_variant_token(token) for token in attr_tokens]
    spans = []
    window = len(attr_tokens)
    for start_index in range(len(query_tokens) - window + 1):
        token_window = query_tokens[start_index:start_index + window]
        normalized_query_tokens = [_normalize_variant_token(token.group(0)) for token in token_window]
        if normalized_query_tokens == normalized_attr_tokens:
            spans.append((token_window[0].start(), token_window[-1].end()))
    return spans


def _attr_value_variant_signature(attr_value: str) -> tuple[str, ...]:
    attr_value = _canonicalize_attr_text(attr_value)
    return tuple(
        _normalize_variant_token(match.group(0))
        for match in re.finditer(r"[A-Za-z0-9']+", attr_value)
    )


def count_attr_value_occurrences(query: str, attr_value: str) -> int:
    counts = count_attr_value_occurrences_map(query, {"attr": attr_value})
    return counts["attr"]


def count_attr_value_occurrences_map(query: str, attrs_used: dict) -> dict[str, int]:
    """Count non-overlapping occurrences of each attribute value in the query.

    Long attribute values are matched first so that a shorter value cannot
    claim a span already consumed by a longer one. Variants with the same
    signature (casefold + simple suffix-strip) are deduplicated.
    """
    signature_counts: dict[tuple[str, ...], int] = {}
    for value in attrs_used.values():
        if not isinstance(value, str):
            continue
        signature = _attr_value_variant_signature(value)
        if signature:
            signature_counts[signature] = signature_counts.get(signature, 0) + 1

    matches_by_key: dict = {}
    counts: dict = {}
    for key, value in attrs_used.items():
        pattern = _build_attr_value_pattern(value)
        if pattern is None:
            matches = []
        else:
            matches = [match.span() for match in re.finditer(pattern, query, re.IGNORECASE)]
        if isinstance(value, str) and signature_counts.get(_attr_value_variant_signature(value), 0) == 1:
            matches.extend(_find_variant_token_spans(query, value))
        if not matches:
            matches_by_key[key] = []
            counts[key] = 0
            continue
        matches_by_key[key] = sorted(set(matches))
        counts[key] = 0

    occupied_spans: list[tuple[int, int]] = []
    ordered_keys = sorted(
        attrs_used,
        key=lambda key: (-len(str(attrs_used[key]).strip()), key),
    )
    for key in ordered_keys:
        for span in matches_by_key[key]:
            if any(not (span[1] <= used[0] or span[0] >= used[1]) for used in occupied_spans):
                continue
            occupied_spans.append(span)
            counts[key] += 1
    return counts


def validate_query_uses_exactly_five_attrs(query: str, attrs_used: dict) -> tuple[bool, str]:
    """Check that the query uses exactly the five provided attribute values,
    each exactly once. Returns (ok, error_message).
    """
    invalid_keys = sorted(key for key in attrs_used if key not in VALID_ATTR_KEYS)
    if invalid_keys:
        return False, f"attrs_used 包含非法属性键: {', '.join(invalid_keys)}"
    if len(attrs_used) != REQUIRED_ATTR_COUNT:
        return False, f"attrs_used 数量不等于 {REQUIRED_ATTR_COUNT}，实际 {len(attrs_used)}"

    occurrence_counts = count_attr_value_occurrences_map(query, attrs_used)
    bad_attrs = []
    for key, value in attrs_used.items():
        count = occurrence_counts[key]
        if count != 1:
            bad_attrs.append(f"{key}={value!r} 出现 {count} 次")

    if bad_attrs:
        return False, "; ".join(bad_attrs)
    return True, ""
