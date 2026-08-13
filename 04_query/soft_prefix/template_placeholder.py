"""E12 — Attribute placeholder template module.

Deterministic attribute-value protection: the LLM only ever generates a
query *template* containing the five placeholders <A1>..<A5>; the final query
is produced by verbatim replacement of the placeholder with the original
attribute value. Numbers, brands, CamelCase and punctuation-sensitive
attribute values therefore can never be mutated by the LLM (issue #12).

Conversion of training targets uses the variant-tolerant span matching of
common/attribute_helpers.py (plural/case inflection), and replacement is
guarded so that placeholder-like tokens glued to alphanumerics are never
accepted. Train/generate/eval all share this module.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query"))

from common.attribute_helpers import (  # noqa: E402
    _build_attr_value_pattern,
    _find_variant_token_spans,
    _attr_value_variant_signature,
)

PLACEHOLDER_BY_ATTR: Dict[str, str] = {f"A{i}": f"<A{i}>" for i in range(1, 19)}
ATTR_BY_PLACEHOLDER: Dict[str, str] = {v: k for k, v in PLACEHOLDER_BY_ATTR.items()}
REQUIRED_PLACEHOLDERS = {"<A1>", "<A2>", "<A3>", "<A4>", "<A5>"}
# Guards: a placeholder must not be glued to an alphanumeric on either side
# (would survive naive replace() and corrupt the final value).
_GUARDED_PH = re.compile(r"(?<![A-Za-z0-9])<A[1-9][0-9]?>(?![A-Za-z0-9])")
_ANY_PH = re.compile(r"<A[1-9][0-9]?>")
_FORBIDDEN = {"<A0>"}


def build_attr_mapping_prompt(attrs_used: Dict[str, str]) -> str:
    """Prompt with an explicit placeholder->value mapping.

    The model sees the real values (so it can write meaningfully about them)
    but must emit the *placeholders* in its query template.
    """
    lines = ["Product attributes:", ""]
    for key in sorted(attrs_used, key=lambda k: int(k[1:])):
        ph = PLACEHOLDER_BY_ATTR.get(key, f"<{key}>")
        lines.append(f"{ph} = {attrs_used[key]}")
    lines.append("")
    lines.append(
        "Write one natural shopping query TEMPLATE that uses every placeholder "
        "<A1>, <A2>, <A3>, <A4>, <A5> exactly once."
    )
    return "\n".join(lines)


def _attr_spans(query: str, attrs_used: Dict[str, str]) -> Dict[str, List[Tuple[int, int]]]:
    """Variant-tolerant spans per attr key (same logic as
    count_attr_value_occurrences_map: exact pattern + variant token spans,
    dedup by variant signature)."""
    signature_counts: Dict[tuple, int] = {}
    for value in attrs_used.values():
        if not isinstance(value, str):
            continue
        sig = _attr_value_variant_signature(value)
        if sig:
            signature_counts[sig] = signature_counts.get(sig, 0) + 1

    matches_by_key: Dict[str, List[Tuple[int, int]]] = {}
    for key, value in attrs_used.items():
        if not isinstance(value, str):
            matches_by_key[key] = []
            continue
        pattern = _build_attr_value_pattern(value)
        matches: List[Tuple[int, int]] = []
        if pattern is not None:
            matches = [m.span() for m in re.finditer(pattern, query, re.IGNORECASE)]
        if signature_counts.get(_attr_value_variant_signature(value), 0) == 1:
            matches.extend(_find_variant_token_spans(query, value))
        matches_by_key[key] = sorted(set(matches))
    return matches_by_key


def replace_attrs_with_placeholders(
    query: str, attrs_used: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    """Replace the five attribute values in ``query`` by placeholders.

    Uses variant-tolerant spans (plural/case inflection handled), then
    consumes spans longest-first so an embedded value (e.g. "Baby" inside
    "Baby Trend") is never shadowed. Returns (template, None) on success or
    (None, reason) when the query cannot be losslessly templated.
    """
    attrs = {k: v for k, v in attrs_used.items() if k in PLACEHOLDER_BY_ATTR}
    if len(attrs) != 5:
        return None, f"need exactly 5 attrs in A1..A18, got {sorted(attrs)}"
    query = query.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    spans = _attr_spans(query, attrs)

    # Resolve overlapping spans: longest value wins; ties by attr key.
    ordered = sorted(
        ((key, value, span) for key, value in attrs.items() for span in spans.get(key, [])),
        key=lambda kv: (-(kv[2][1] - kv[2][0]), kv[1]),
    )
    occupied: List[Tuple[int, int]] = []
    replacements: List[Tuple[Tuple[int, int], str]] = []
    per_key_spans: Dict[str, int] = {}
    for key, value, span in ordered:
        if any(not (span[1] <= o[0] or span[0] >= o[1]) for o in occupied):
            continue
        occupied.append(span)
        replacements.append((span, PLACEHOLDER_BY_ATTR[key]))
        per_key_spans[key] = per_key_spans.get(key, 0) + 1

    bad = [f"{k}:{per_key_spans.get(k, 0)}x" for k in attrs if per_key_spans.get(k, 0) != 1]
    if bad:
        return None, f"attr value not exactly once in query: {bad}"

    template = ""
    pos = 0
    for span, ph in sorted(replacements, key=lambda r: r[0][0]):
        template += query[pos:span[0]] + ph
        pos = span[1]
    template += query[pos:]

    parsed = parse_template(template, required=_required_placeholders(attrs))
    if not parsed["valid"]:
        return None, f"converted template invalid: missing={parsed['missing']} dup={parsed['duplicated']}"
    return template, None


def _required_placeholders(attrs_used: Dict[str, str]) -> set:
    return {PLACEHOLDER_BY_ATTR[k] for k in attrs_used if k in PLACEHOLDER_BY_ATTR}


def parse_template(template: str, required: Optional[set] = None) -> Dict[str, object]:
    """Validate a raw generated template.

    A placeholder counts only when not glued to an alphanumeric character.
    ``required`` defaults to the five canonical placeholders <A1>..<A5>.
    """
    counts: Dict[str, int] = {}
    for ph in _GUARDED_PH.findall(template or ""):
        counts[ph] = counts.get(ph, 0) + 1
    required = required if required is not None else REQUIRED_PLACEHOLDERS
    missing = sorted(required - set(counts))
    duplicated = sorted(ph for ph, c in counts.items() if c > 1)
    extra = sorted(set(_ANY_PH.findall(template or "")) - required)
    for token in _FORBIDDEN:
        if token in (template or ""):
            extra.append(token)
    if (re.search(r"[A-Za-z0-9]<A[1-9][0-9]?>", template or "")
            or re.search(r"<A[1-9][0-9]?>[A-Za-z0-9]", template or "")):
        extra.append("glued")
    valid = not missing and not duplicated and not extra and sum(counts.values()) == len(required)
    return {
        "valid": bool(valid),
        "missing": missing,
        "duplicated": duplicated,
        "extra": sorted(set(extra)),
        "placeholders": sorted(counts),
        "raw": template,
    }


def replace_placeholders_with_attrs(
    template: str, attrs_used: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    """Verbatim placeholder -> original value replacement.

    No inflection, case normalization or number formatting. Returns
    (final_query, None) or (None, reason).
    """
    required = _required_placeholders(attrs_used)
    parsed = parse_template(template, required=required)
    if not parsed["valid"]:
        return None, f"invalid template: missing={parsed['missing']} dup={parsed['duplicated']} extra={parsed['extra']}"
    final = template
    for key, ph in PLACEHOLDER_BY_ATTR.items():
        if ph not in required:
            continue
        value = attrs_used.get(key)
        if value is None or not str(value):
            return None, f"no attribute value for {key}"
        # parse_template already rejected glued/duplicated/extra forms, so
        # plain replace() is verbatim and safe.
        final = final.replace(ph, str(value))
    if _GUARDED_PH.search(final):
        return None, "placeholder left unreplaced"
    return final, None
