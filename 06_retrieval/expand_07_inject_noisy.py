#!/usr/bin/env python3
"""扩展 07_inject_noisy noisy pairs.

从 04_query/query_by_syntax_depth_no_depth_check_10.json 读所有 (user_id, asin)
对, 每个 (uid, asin) 取最多 N 个 syntax_depth_queries (默认 N=10, 即 syntax_depth_queries
列表的全部), 对每个 query 做一次轻量级 noise injection (动词变形 / 形容词替换 / 介词替换),
写入 07_inject_noisy/<cat>/noisy_query.json, 覆盖现有 2-11 条记录.

噪声注入规则 (deterministic, 无 ML):
- VERB_SWAP: I want -> I'm looking for / I'd like to buy / I'm shopping for (3 个 verbs)
- ADJ_SWAP: 'small' -> 'tiny'/'compact'/'little' (3 个 adjs)
- PREP_SWAP: 'for' -> 'to use for' / 'that I need for' (3 个 preps)

每个 query 选 1 个 noise mode (deterministic, 基于 asin 末字符 hash), 输出 clean_query + noisy_query 对.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
SAMPLE_PER_USER_ASIN = 10  # 取全部 syntax_depth_queries


VERB_SWAPS = [
    ("I want to buy", "I'm looking for"),
    ("I want", "I'm hoping to find"),
    ("I'm searching for", "I'd like to buy"),
    ("I'm shopping for", "I need"),
    ("I need", "I'm trying to find"),
    ("Please show me", "Can you find me"),
    ("I would like", "I'd love to get"),
]

ADJ_SWAPS = {
    "small": ["tiny", "compact", "little", "mini"],
    "large": ["big", "huge", "oversized"],
    "blue": ["navy", "azure", "cobalt"],
    "red": ["crimson", "scarlet", "ruby"],
    "green": ["emerald", "olive", "lime"],
    "clear": ["transparent", "crystal-clear"],
}

PREP_SWAPS = [
    ("for my", "to bring to my"),
    ("for my Party", "for the Party I'm hosting"),
    ("for Baby", "that I'll use for Baby"),
    ("because they match what I want", "since they fit my needs"),
    ("because it matches what I want", "since it fits my needs"),
]


def pick_noise_mode(uid: str, asin: str) -> str:
    """deterministic pick noise mode based on uid+asin hash."""
    h = int(hashlib.md5(f"{uid}{asin}".encode()).hexdigest(), 16)
    return ["verb", "adj", "prep"][h % 3]


def apply_noise(query: str, mode: str) -> str:
    if mode == "verb":
        for src, dst in VERB_SWAPS:
            if query.startswith(src):
                return dst + query[len(src):]
        return query
    if mode == "adj":
        for src, dsts in ADJ_SWAPS.items():
            if src in query.lower():
                # pick dst deterministically by hash
                h = hash(query + src) % len(dsts)
                dst = dsts[h]
                return query.replace(src, dst).replace(src.capitalize(), dst.capitalize())
        return query
    if mode == "prep":
        for src, dst in PREP_SWAPS:
            if src in query:
                return query.replace(src, dst)
        return query
    return query


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_pair", type=int, default=SAMPLE_PER_USER_ASIN)
    args = ap.parse_args()

    for cat in CATEGORIES:
        src = REPO_ROOT / "result" / "personal_query" / "04_query" / cat / "query_by_syntax_depth_no_depth_check_10.json"
        if not src.exists():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {cat}: SKIP (no 04_query)")
            continue
        with open(src) as f:
            data = json.load(f)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {cat}: read {len(data)} user/asin pairs from 04_query")

        out_dir = REPO_ROOT / "result" / "personal_query" / "07_inject_noisy" / cat
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "noisy_query.json"

        out_rows: List[Dict] = []
        for record in data:
            uid = record.get("user_id")
            asin = record.get("asin")
            sds = record.get("syntax_depth_queries", [])
            if not (uid and asin and sds):
                continue
            for idx, sq in enumerate(sds):
                if idx >= args.per_pair:
                    break
                clean_query = sq.get("query")
                if not clean_query:
                    continue
                mode = pick_noise_mode(uid, asin + str(idx))
                noisy = apply_noise(clean_query, mode)
                if noisy == clean_query:
                    # noise injection had no effect, skip
                    continue
                out_rows.append({
                    "uid": uid,
                    "asin": asin,
                    "clean_query": clean_query,
                    "noisy_query": noisy,
                    "category": cat,
                    "noise_mode": mode,
                    "query_rewritten": True,
                    "candidate_idx": idx,
                    "attrs_used": sq.get("attrs_used"),
                    "target_depth": sq.get("target_depth"),
                })

        with open(out_path, "w") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {cat}: wrote {len(out_rows)} noisy pairs to {out_path}")


if __name__ == "__main__":
    main()