#!/usr/bin/env python3
"""Build a minimal RAW_CANDIDATE_QUERY_FILE for VADES train (Stage 4 stub).

VADES diagonal_residual_llm mode requires a JSON file at:
  /fs04/ar57/wenyu/PersoanlQuery/result/query_by_expression_style_no_depth_check_10.json
Format: [{user_id, asin, expression_style_queries: [{query, word_count}, ...]}, ...]

For 10K pipeline, VADES only needs candidate_rows to exist for inference path.
The actual ranking happens via pick_best_cand_10k.py using Stage 3 residuals,
not via VADES regeneration. So this stub gives VADES 1 placeholder candidate
per user to satisfy the file requirement.

Reads: /home/wlia0047/ar57/wenyu/PersoanlQuery/result/query_records_10k.json
Writes: /home/wlia0047/ar57/wenyu/PersoanlQuery/result/query_by_expression_style_no_depth_check_10.json
"""
from __future__ import annotations
import json
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
RECORDS_IN = REPO_ROOT / "result/query_records_10k.json"
OUT = REPO_ROOT / "result/query_by_expression_style_no_depth_check_10.json"


def main() -> None:
    records = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    print(f"[stub] loaded {len(records)} records")
    out = []
    for r in records:
        attrs = r.get("attrs_used", {})
        # Build a placeholder query from attrs values
        attr_values = [str(v) for v in attrs.values() if v]
        placeholder = " ".join(attr_values[:5]) if attr_values else "placeholder query"
        out.append({
            "user_id": r["user_id"],
            "asin": r["asin"],
            "expression_style_queries": [
                {"query": placeholder, "word_count": len(placeholder.split())}
            ],
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"[stub] wrote {len(out)} entries → {OUT}")


if __name__ == "__main__":
    main()