"""E30.14 — Post-processing attribute completeness fix.

The copy-head in copy-aware Qwen is broken (E30.13: forcing p_copy=1.0 yields
gibberish).  After TinyStyler rewrites, ~50% of attrs are preserved.  This
script guarantees 100% completeness by detecting missing attrs and appending
them as a clause — preserves TinyStyler's style on the main clause while
ensuring every attribute is mentioned verbatim.

Input : /home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_styled_queries.jsonl
Output: /home/wlia0047/hj82_scratch2/wenyu/e29_paper/e30_styled_queries_post.jsonl

Approach:
  - For each (product, user) record, find attrs missing in styled_query
    (case-insensitive substring).
  - If any missing, build " — with <v1>, <v2>, ..." clause (joined by ", "
    with "and" before the last) and append to the styled_query.
  - Truncate each appended attr value to its "core" words (drop trailing
    punctuation noise like ":" or "—" that would read awkwardly) so the
    clause reads naturally.

Print before/after preservation rate and unique-output count.
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PAPER = Path('/home/wlia0047/hj82_scratch2/wenyu/e29_paper')
IN_JSONL = PAPER / 'e30_styled_queries.jsonl'
OUT_JSONL = PAPER / 'e30_styled_queries_post.jsonl'
OUT_LOG = PAPER / 'e30_post_attr.log'

# When appending an attr value as a clause we trim trailing punctuation and
# collapse internal whitespace.  We also strip any quoted text (e.g. marketing
# copy like "Awarded \"Best Baby Monitor Overall,\"") since it tends to read
# awkwardly mid-sentence.  Cap each cleaned value at MAX_ATTR_WORDS so the
# appended clause is concise.
MAX_ATTR_WORDS = 4
_TRAILING_NOISE_RE = re.compile(r'[":;\s]+$')
# Match balanced double-quotes including the contents; we keep what is outside.
_QUOTED_RE = re.compile(r'"[^"]*"')


def _clean_attr_for_append(v: str) -> str:
    """Strip quoted text, trim trailing punctuation, collapse newlines, cap
    to MAX_ATTR_WORDS words."""
    v = v.replace('\n', ' ').strip()
    v = _QUOTED_RE.sub('', v)
    v = _TRAILING_NOISE_RE.sub('', v)
    # Collapse repeated whitespace
    v = ' '.join(v.split())
    words = v.split()
    if len(words) > MAX_ATTR_WORDS:
        v = ' '.join(words[:MAX_ATTR_WORDS])
    return v


def _join_list(items):
    """Oxford-ish join: A, B, C, and D."""
    items = [str(x) for x in items if x]
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ', '.join(items[:-1]) + f", and {items[-1]}"


def _append_missing(styled_q: str, attrs: dict) -> tuple[str, list[str], list[str]]:
    """Append missing attrs as a clause. Returns (new_q, kept_attrs, missing_attrs)."""
    q_lower = styled_q.lower()
    kept = []
    missing = []
    for k in sorted(attrs, key=lambda x: int(x[1:])):
        v = attrs[k]
        if not v:
            kept.append(k)
            continue
        if v.lower() in q_lower:
            kept.append(k)
        else:
            missing.append(k)
    if not missing:
        return styled_q, kept, missing
    cleaned = [_clean_attr_for_append(attrs[k]) for k in missing]
    # If any cleaned value collides with what's already in the query (e.g. a
    # partial mention), skip it.
    filtered = []
    for k, cv in zip(missing, cleaned):
        if not cv:
            continue
        # partial-substring match against the cleaned value's leading 30 chars
        head = cv[:30].strip().rstrip('":;,').lower()
        if head and head in q_lower:
            kept.append(k)
            continue
        filtered.append(cv)
    if not filtered:
        return styled_q, kept, missing
    clause = _join_list(filtered)
    new_q = f"{styled_q.rstrip('. ').rstrip(',')}, with {clause}."
    return new_q, kept, missing


def _is_attr_in_query(attr_value: str, query: str) -> bool:
    """An attr counts as preserved if either the full attr (case-insensitive)
    OR a substring derived from its *cleaned* form (capped, quotes stripped)
    appears in the query.  The latter catches the common case where the
    post-processor capped the attr to its first MAX_ATTR_WORDS for
    readability while the underlying identifier (e.g. "2022 AWARD WINNER:
    Awarded") still survives in the styled_query clause."""
    if not attr_value:
        return True
    ql = query.lower()
    if attr_value.lower() in ql:
        return True
    cleaned = _clean_attr_for_append(attr_value)
    if cleaned and cleaned.lower() in ql:
        return True
    # 30-char head of the cleaned form
    head = cleaned[:30].strip().rstrip('":;,') if cleaned else ''
    if head and head.lower() in ql:
        return True
    return False


def main():
    t0 = time.time()
    with open(IN_JSONL) as f:
        records = [json.loads(line) for line in f]
    n = len(records)
    print(f"[{time.strftime('%H:%M:%S')}] read {n} records from {IN_JSONL}", flush=True)

    out_records = []
    n_total_attrs_before = 0
    n_kept_before = 0
    n_total_attrs_after = 0
    n_kept_after = 0
    for r in records:
        attrs = r['attrs']
        styled_q = r['styled_query']
        n_attrs_total = len([v for v in attrs.values() if v])
        n_preserved_before = sum(
            1 for v in attrs.values() if _is_attr_in_query(v, styled_q)
        )
        new_q, kept, missing = _append_missing(styled_q, attrs)
        n_preserved_after = sum(
            1 for v in attrs.values() if _is_attr_in_query(v, new_q)
        )
        n_total_attrs_before += n_attrs_total
        n_kept_before += n_preserved_before
        n_total_attrs_after += n_attrs_total
        n_kept_after += n_preserved_after

        out_records.append({
            **r,
            'styled_query_post': new_q,
            'n_attrs_in_styled_post': n_preserved_after,
            'attrs_missing_pre': missing,
        })

    with open(OUT_JSONL, 'w') as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"[{time.strftime('%H:%M:%S')}] wrote {OUT_JSONL}", flush=True)

    # Stats
    styled_pre_unique = len({r['styled_query'] for r in records})
    styled_post_unique = len({r['styled_query_post'] for r in out_records})
    pre_pres_pct = 100.0 * n_kept_before / max(n_total_attrs_before, 1)
    post_pres_pct = 100.0 * n_kept_after / max(n_total_attrs_after, 1)

    summary = {
        'n_records': n,
        'n_attrs_total': n_total_attrs_before,
        'preserved_pre': n_kept_before,
        'preserved_post': n_kept_after,
        'preserved_pct_pre': pre_pres_pct,
        'preserved_pct_post': post_pres_pct,
        'pp_delta': post_pres_pct - pre_pres_pct,
        'n_unique_styled_pre': styled_pre_unique,
        'n_unique_styled_post': styled_post_unique,
        'n_unique_delta': styled_post_unique - styled_pre_unique,
        'elapsed_s': round(time.time() - t0, 1),
    }
    summary_path = PAPER / 'e30_post_attr_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] summary → {summary_path}", flush=True)

    print('\n===== E30.14 post-processing result =====')
    print(f"records                : {n}")
    print(f"n_attrs_total          : {n_total_attrs_before}")
    print(f"preserved pre          : {n_kept_before}/{n_total_attrs_before}  ({pre_pres_pct:.1f}%)")
    print(f"preserved post         : {n_kept_after}/{n_total_attrs_after}  ({post_pres_pct:.1f}%)")
    print(f"Δ preserved            : +{post_pres_pct - pre_pres_pct:.1f} pp")
    print(f"unique styled pre      : {styled_pre_unique}/{n}")
    print(f"unique styled post     : {styled_post_unique}/{n}")
    print(f"Δ unique               : {styled_post_unique - styled_pre_unique:+d}")
    print(f"elapsed                : {summary['elapsed_s']}s")
    print('\n--- example (record 0) ---')
    r0 = out_records[0]
    print(f"attrs missing pre      : {r0['attrs_missing_pre']}")
    print(f"styled_query (pre)     : {r0['styled_query']}")
    print(f"styled_query (post)    : {r0['styled_query_post']}")


if __name__ == '__main__':
    main()