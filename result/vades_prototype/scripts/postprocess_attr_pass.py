#!/usr/bin/env python3
"""一次性 post-process: 修复 phase10_candidates.jsonl 的 attr_pass 判定 (check 太严).

对每条 candidate:
  1. 用 raw query + 修正后的 check_attr_pass (容忍复数/格式) 判定
  2. 如果 attr_pass_raw=False 但可被 append 兜底, 更新 candidate_query 为兜底版本
  3. 重新计算 attr_pass
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
IN_FILE = OUT_DIR / "phase10_candidates.jsonl"
OUT_FILE = OUT_DIR / "phase10_candidates_post.jsonl"
META_FILE = OUT_DIR / "phase10_candidates_post_meta.json"


def check_attr_pass(query: str, attrs: dict) -> bool:
    q_lower = query.lower()
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", q_lower)
    for k, v in attrs.items():
        if not v:
            return False
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        all_tokens = v_clean.split()
        tokens = []
        for t in all_tokens:
            if len(t) >= 3:
                tokens.append(t)
            elif re.match(r"^\d+(\.\d+)?$", t):
                tokens.append(t)
            elif k == "Brand" and re.match(r"^[a-zA-Z]+$", t) and len(t) >= 2:
                tokens.append(t)
        if not tokens:
            return False
        found = False
        for t in tokens:
            t_lower = t.lower()
            if t_lower in q_clean:
                found = True
                break
            if t_lower.endswith("s") and len(t_lower) > 3 and t_lower[:-1] in q_clean:
                found = True
                break
            if (t_lower + "s") in q_clean:
                found = True
                break
        if not found:
            return False
    return True


def append_missing_attrs(query: str, attrs: dict) -> str:
    if check_attr_pass(query, attrs):
        return query
    missing = []
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", query.lower())
    for k, v in attrs.items():
        if not v:
            continue
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        all_tokens = v_clean.split()
        tokens = []
        for t in all_tokens:
            if len(t) >= 3:
                tokens.append(t)
            elif re.match(r"^\d+(\.\d+)?$", t):
                tokens.append(t)
        found = False
        for t in tokens:
            t_lower = t.lower()
            if t_lower in q_clean:
                found = True
                break
            if t_lower.endswith("s") and len(t_lower) > 3 and t_lower[:-1] in q_clean:
                found = True
                break
            if (t_lower + "s") in q_clean:
                found = True
                break
        if not found:
            words = str(v).split()[:4]
            cleaned = " ".join(w.rstrip(":;,.") for w in words).strip()
            if cleaned:
                missing.append(f"{k} {cleaned}")
    if not missing:
        return query
    if len(missing) == 1:
        clause = f"with {missing[0]}"
    elif len(missing) == 2:
        clause = f"with {missing[0]} and {missing[1]}"
    else:
        clause = "with " + ", ".join(missing[:-1]) + f", and {missing[-1]}"
    return query.rstrip(" .") + f", {clause}."


def main():
    n_total = 0
    n_pass_after = 0
    n_changed = 0
    results = []
    with IN_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            n_total += 1
            raw = r.get("candidate_query_raw") or r.get("candidate_query")
            # 旧 candidate_query 已经是 (旧 check + 旧 append 兜底) 的结果, 现在重做:
            old_query = r["candidate_query"]
            new_query = append_missing_attrs(raw, r["attrs"])
            new_pass = check_attr_pass(new_query, r["attrs"])
            if new_query != old_query:
                n_changed += 1
            r["candidate_query"] = new_query
            r["attr_pass"] = new_pass
            r["needed_append"] = not check_attr_pass(raw, r["attrs"])
            if new_pass:
                n_pass_after += 1
            results.append(r)

    OUT_FILE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    meta = {
        "n_total": n_total,
        "n_attr_pass": n_pass_after,
        "attr_pass_rate": n_pass_after / max(n_total, 1),
        "n_changed_query": n_changed,
        "in_file": str(IN_FILE),
        "out_file": str(OUT_FILE),
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[post-attr] total={n_total}, attr_pass_after={n_pass_after} ({meta['attr_pass_rate']:.4%}), "
          f"changed={n_changed}", flush=True)
    print(f"[post-attr] 已写入 {OUT_FILE}")


if __name__ == "__main__":
    main()
