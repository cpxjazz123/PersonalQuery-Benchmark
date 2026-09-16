#!/usr/bin/env python3
"""重算 result/07_gen_query/pool_queries.json 的 pass 标记。

原因:Stage 7 之前有 3 条 content check (全覆盖 / 无重复 / extras ≤ 3),
用户 2026-09-13 决定去掉 extras 阈值,只保留全覆盖 + 无重复两条。

不重跑 vLLM,直接读现有 pool + product_attributes.json,
按 Stage 7 generate_candidates.check_content() 的 tokenize + 同口径(取 attrs 前 5)
重算每条 query 的 pass,落 result/07_gen_query/pool_queries_recomputed.json。

注:Stage 8 selection 当前不消费 candidate-level pass,
新字段只是把"按新过滤口径能通过"的 query 标记出来方便对照。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
POOL_PATH = REPO_ROOT / "result/07_gen_query/pool_queries.json"
ATTRS_PATH = REPO_ROOT / "result/01_attribute_extraction/product_attributes.json"
OUT_PATH = REPO_ROOT / "result/07_gen_query/pool_queries_recomputed.json"


def tokenize(text: str):
    return re.findall(r"\b\w+(?:['-]\w+)*\b", text)


def check_pass(query: str, attrs: dict) -> dict:
    """对齐 Stage 7 check_content + 新过滤(去掉 extras 阈值)。"""
    tokens = tokenize(query.lower())
    normalized = " ".join(tokens)
    covered = {}
    missing = []
    for key, value in attrs.items():
        value_tokens = tokenize(str(value).lower())
        if not value_tokens:
            raise ValueError(f"attribute {key} has no tokenized value")
        value_text = " ".join(value_tokens)
        covered[key] = value_text in normalized
        if not covered[key]:
            missing.append(key)
    repeated = len(tokens) != len(set(tokens))
    known_tokens = {
        token for value in attrs.values()
        for token in tokenize(str(value).lower())
    }
    extras = [t for t in tokens if t not in known_tokens]
    # 新过滤:全覆盖 + 无重复(去掉 extras ≤ 3 阈值)
    content_pass = (len(missing) == 0 and not repeated)
    return {
        "covered": covered,
        "missing": missing,
        "repeated": repeated,
        "extras": extras,
        "extras_n": len(extras),
        "pass": content_pass,
    }


def main():
    pool_doc = json.load(open(POOL_PATH))
    pool = pool_doc["pool"]
    attrs_all = json.load(open(ATTRS_PATH))

    out_pool: dict[str, list[str]] = {}
    out_pass: dict[str, list[dict]] = {}
    n_pass_new = 0
    n_pass_old_est = 0
    n_missing_attrs = 0
    n_full_coverage = 0
    n_no_repeat = 0
    n_extras_over_3 = 0
    n_old_fail_new_pass = 0   # 旧 fail 但新 pass:全覆盖 ∧ 无重复 ∧ extras > 3

    # 旧 pool 没有 per-cand pass 字段,逐条重算 + 模拟旧/新两种逻辑下的 pass
    for asin, queries in pool.items():
        attrs = attrs_all.get(asin)
        if not attrs:
            n_missing_attrs += 1
            out_pool[asin] = list(queries)
            out_pass[asin] = [{"pass": False, "reason": "no_attrs"} for _ in queries]
            continue
        attrs5 = dict(list(attrs.items())[:5])
        new_qs = []
        new_ps = []
        for q in queries:
            r = check_pass(q, attrs5)
            new_qs.append(q)
            new_ps.append({
                "pass": r["pass"],
                "missing": r["missing"],
                "repeated": r["repeated"],
                "extras_n": r["extras_n"],
            })
            # 单条件统计
            if not r["missing"]:
                n_full_coverage += 1
            if not r["repeated"]:
                n_no_repeat += 1
            if r["extras_n"] > 3:
                n_extras_over_3 += 1
            # 新 pass
            if r["pass"]:
                n_pass_new += 1
            # 旧 pass(估计):全覆盖 ∧ 无重复 ∧ extras ≤ 3
            old_pass = (not r["missing"]) and (not r["repeated"]) and (r["extras_n"] <= 3)
            if old_pass:
                n_pass_old_est += 1
            # 旧 fail 但新 pass(本应被砍但去掉阈值后恢复)
            if r["pass"] and not old_pass:
                n_old_fail_new_pass += 1
        out_pool[asin] = new_qs
        out_pass[asin] = new_ps

    n_total = sum(len(v) for v in pool.values())
    summary = {
        "n_asins": len(pool),
        "n_total_queries": n_total,
        "new_pass_count": n_pass_new,
        "new_pass_rate": n_pass_new / max(1, n_total),
        "old_pass_count_estimated": n_pass_old_est,   # 全覆盖 ∧ 无重复 ∧ extras ≤ 3
        "old_pass_rate_estimated": n_pass_old_est / max(1, n_total),
        "full_coverage_count": n_full_coverage,
        "no_repeat_count": n_no_repeat,
        "extras_gt_3_count": n_extras_over_3,
        "missing_attrs_asins": n_missing_attrs,
        "recovered_by_removing_extras_threshold": n_old_fail_new_pass,
    }

    out_doc = {
        "config": {
            **pool_doc.get("config", {}),
            "recomputed_by": "recompute_pool_pass.py",
            "recompute_date": "2026-09-13",
            "new_filter": "full_coverage AND no_repeat (extras threshold removed)",
            "old_filter": "full_coverage AND no_repeat AND extras<=3",
        },
        "summary": summary,
        "pool": out_pool,         # 文本不动(下游 Stage 8 兼容)
        "pass_per_query": out_pass,  # 新增:每 ASIN 每 query 的 pass/missing/repeated/extras_n
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out_doc, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Stage 7 pool pass recompute (drop extras ≤ 3 threshold)")
    print("=" * 60)
    print(f"  ASINs: {summary['n_asins']}  total queries: {summary['n_total_queries']}")
    print(f"  full coverage (5 attrs): {n_full_coverage} "
          f"({100*n_full_coverage/summary['n_total_queries']:.1f}%)")
    print(f"  no repeat token: {n_no_repeat} "
          f"({100*n_no_repeat/summary['n_total_queries']:.1f}%)")
    print(f"  extras > 3 (raw, 不区分其他 fail): {n_extras_over_3} "
          f"({100*n_extras_over_3/summary['n_total_queries']:.1f}%)")
    print(f"  === new pass (全覆盖 ∧ 无重复): {n_pass_new} "
          f"({summary['new_pass_rate']*100:.1f}%)")
    print(f"  === old pass estimate (再加 extras ≤ 3): {n_pass_old_est} "
          f"({summary['old_pass_rate_estimated']*100:.1f}%)")
    print(f"  → 去掉 extras 阈值后额外恢复 {n_old_fail_new_pass} 条 "
          f"({100*n_old_fail_new_pass/summary['n_total_queries']:.1f}%) query 可通过 filter")
    print(f"  → 写入 {OUT_PATH}")


if __name__ == "__main__":
    main()
