"""Build 20-dim clause features for the 10 candidate queries per record.

Writes result/personal_query/12_complexity_analysis_clause_features/<cat>/
query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl, the
format expected by the VADES pipeline and the E11 data builder.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from extract_clause_features_single_query import (  # noqa: E402
    load_spacy_model,
    extract_clause_features_from_doc,
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_candidate_features_file(category: str, records: List[dict]) -> List[dict]:
    out_p = (
        REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
        / category / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    nlp = load_spacy_model()
    rows: List[dict] = []
    for rec in records:
        uid, asin = rec["user_id"], rec["asin"]
        candidates = rec.get("syntax_depth_queries") or rec.get("expression_style_queries") or []
        for cand_idx, cand in enumerate(candidates, start=1):
            query = cand.get("query", "")
            if not query:
                continue
            doc = nlp(query)
            extracted = extract_clause_features_from_doc(doc, query)
            rows.append(
                {
                    "user_id": uid,
                    "asin": asin,
                    "candidate_index": cand_idx,
                    "query": query,
                    "word_count": int(cand.get("word_count", len(query.split()))),
                    "target_depth": cand.get("target_depth"),
                    "user_avg_depth": cand.get("user_avg_depth"),
                    "attrs_used": cand.get("attrs_used"),
                    "features": extracted,
                }
            )
        if len(rows) % 200 == 0:
            log(f"  {category}: {len(rows)} candidate rows...")
    with open(out_p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"wrote {out_p} ({len(rows)} rows)")
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--base", default=str(REPO_ROOT))
    args = ap.parse_args()
    records_p = Path(args.base) / "result" / "personal_query" / "query" / args.category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(records_p) as f:
        records = json.load(f)
    log(f"building candidate features for {args.category} ({len(records)} records)...")
    build_candidate_features_file(args.category, records)
    log("=== done ===")


if __name__ == "__main__":
    main()
