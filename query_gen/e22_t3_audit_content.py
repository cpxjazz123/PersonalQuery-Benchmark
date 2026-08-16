#!/usr/bin/env python3
"""E22 T3 — content evaluator audit (offline, no GPU).

Loads the saved one_prod result JSON and audits, per (user query, attribute):
    expected raw value | generated raw value | normalized value | failure reason

Normalization is PRESENTATION-ONLY (never changes attribute meaning):
  NFKC, strip zero-width chars, lowercase, collapse whitespace, strip edge
  quotes/punctuation, straighten curly quotes.

Failure taxonomy:
  OK           raw substring match
  OK_NORM      passes after presentation normalization only
  PARTIAL      >half of the value's words present in order but not contiguous
  INFLECTED    lemma-level match but surface form differs (model rewording)
  MISSING      no match at all
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import spacy

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
RESULT = REPO_ROOT / "result" / "e22_t3" / "e22_t3_one_prod.json"

ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060]")
CURLY = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"',
                       "\u201d": '"'})
EDGE_PUNCT = " \t\r\n.,;:!?\"'()[]{}<>-"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = ZERO_WIDTH.sub("", s)
    s = s.translate(CURLY)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(EDGE_PUNCT)


def words(s: str) -> list[str]:
    return [w for w in re.split(r"[^0-9a-z$%&']+", norm(s)) if w]


def audit_value(value: str, q: str, nlp) -> dict:
    v_raw = str(value)
    v_n = norm(v_raw)
    q_n = norm(q)
    res = {"expected_raw": v_raw, "expected_norm": v_n,
           "generated_raw": None, "generated_norm": None,
           "reason": None, "match": False}
    # 1) raw
    if v_raw.lower() in q.lower():
        res.update(reason="OK", match=True)
        return res
    # 2) presentation-normalized
    if v_n and v_n in q_n:
        res.update(reason="OK_NORM", match=True)
        return res
    # 3) partial: ordered subsequence of words
    vw = words(v_raw)
    qw = words(q)
    if len(vw) >= 2:
        matched = [w for w in vw if w in set(qw)]
        in_order = 0
        j = 0
        for w in matched:
            while j < len(qw) and qw[j] != w:
                j += 1
            if j < len(qw):
                in_order += 1
                j += 1
        frac = in_order / len(vw)
        if frac > 0.5:
            res.update(reason=f"PARTIAL words={in_order}/{len(vw)}",
                       generated_raw=q)
            return res
    # 4) inflected: lemma match
    try:
        vl = {t.lemma_.lower() for t in nlp(v_raw)}
        ql = {t.lemma_.lower() for t in nlp(q)}
        hit = vl & ql
        if hit and len(vl) > 0 and len(hit) / len(vl) >= 0.6:
            res.update(reason=f"INFLECTED lemmas={sorted(hit)[:4]}",
                       generated_raw=q)
            return res
    except Exception:
        pass
    res.update(reason="MISSING", generated_raw=q)
    return res


def main() -> None:
    d = json.load(open(RESULT))
    attrs: dict[str, str] = d["attrs"]
    queries: dict[str, str] = d["generated_queries"]
    nlp = spacy.load("en_core_web_sm")
    rows = []
    per_attr: dict[str, dict] = {k: {"OK": 0, "OK_NORM": 0, "PARTIAL": 0,
                                     "INFLECTED": 0, "MISSING": 0}
                                 for k in attrs}
    for u, q in queries.items():
        for k, v in attrs.items():
            r = audit_value(v, q, nlp)
            r["user"] = u[:10]
            r["attr"] = k
            per_attr[k][r["reason"].split(" ")[0]] += 1
            rows.append(r)
    hdr = f"{'user':>10} {'attr':<26} {'expected_raw':<34} reason"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['user']:>10} {r['attr']:<26} {r['expected_raw']:<34} "
              f"{r['reason']}")
    print()
    print("per-attribute summary:")
    for k in attrs:
        s = per_attr[k]
        total = sum(s.values())
        print(f"  {k:<26} OK={s['OK']:>2} OK_NORM={s['OK_NORM']:>2} "
              f"PARTIAL={s['PARTIAL']:>2} INFLECTED={s['INFLECTED']:>2} "
              f"MISSING={s['MISSING']:>2} (n={total})")


if __name__ == "__main__":
    main()
