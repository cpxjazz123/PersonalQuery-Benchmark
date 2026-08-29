#!/usr/bin/env python3
"""User cohort construction for Syntax Subspace pipeline.

Step 2: scan_reviews (single pass over Baby_Products_2023.jsonl.gz)
    Returns:
      - asin_users[asin]            -> {uid set} (commenters per ASIN)
      - user_per_asin_count[asin]   -> Counter[uid -> count] (multi-review per ASIN)
      - user_total[uid]             -> total reviews
      - first_asin_per_user[uid]    -> first asin (供 query_records primary)

Step 3: build_query_records
    top-K most-active users × primary asin × top-5 non-numeric attrs
    -> result/query_records_10k.json
    (供 copy_aware_generate / copy_aware_train 使用;
     Stage 1 vLLM pool 不再依赖此文件, 仅 legacy copy_aware 链路消费)

Step 4: build_stage8_5_asins
    Top-N ASINs × ALL commenters × top-4 non-numeric attrs
    -> scratch2/stage8_5_asins.json
    (供 Syntax Subspace pipeline 5 stages 使用)

无用户级过滤 (用户指令 2026-08-29):
  - MIN_TOTAL_WORDS / mean_wc ≥ 20 全部撤销
  - 每个有至少 1 条评论的用户都进入 cohort
  - 用户质量过滤推迟到 Stage 3 Gaussian fit + reliability filter

属性字段选择复用 attribute_extraction/extract_product_attrs.select_top_attrs:
  - Step 3: max_n = 5 (Step 3 默认)
  - Step 4: max_n = 4 (Stage 4 prompt 上限)

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

# 让 build_user_cohort.py 能 import 兄弟目录 attribute_extraction/extract_product_attrs
sys.path.insert(0, str(REPO_ROOT))
from attribute_extraction.extract_product_attrs import select_top_attrs  # noqa: E402

# === Inputs ===
DATA = Path("/fs04/ar57/wenyu/PersoanlQuery/data")
REVIEWS_GZ = DATA / "Baby_Products_2023.jsonl.gz"

# === Outputs ===
QUERY_RECORDS_JSON = REPO_ROOT / "result" / "query_records_10k.json"
STAGE8_5_ASINS_JSON = SCRATCH / "stage8_5_asins.json"

# === Step 3 — build_query_records 参数 ===
TOP_K_USERS = 10_000
MAX_RECORDS = 10_000

# === Step 4 — build_stage8_5_asins 参数 ===
# 用户指令 2026-08-28: 删除 cohort user per ASIN 的所有硬阈值
# (MIN_REVIEWS_PER_USER=20, MIN_USERS_PER_ASIN=10, MAX_USERS_PER_ASIN=10)。
# 用户指令 2026-08-29: 撤销所有用户级过滤 (MIN_TOTAL_WORDS=1000, mean_wc≥20)。
# 用户质量 (Gaussian reliability) 推迟到 Stage 3 + reliability filter。
TOP_N_ASINS = 10_000  # 去掉 1409 硬上限, 取全部候选
MAX_ATTRS_FOR_LLM = 4


def log(msg: str) -> None:
    print(f"[build_user_cohort] {msg}", flush=True)


# ============================================================
# Step 2 — scan_reviews (single pass)
# ============================================================
def step2_scan_reviews() -> tuple[dict[str, set[str]], dict[str, Counter[str]],
                                  Counter[str], dict[str, str]]:
    """单次扫描 reviews, 返回
      - asin_users: asin -> {uid set}
      - user_per_asin_count: asin -> Counter[uid -> count]
      - user_total: Counter[uid -> total reviews]
      - first_asin_per_user: uid -> first asin seen (供 query_records primary)
    """
    log("=== Step 2: scan_reviews (single pass) ===")
    asin_users: dict[str, set[str]] = defaultdict(set)
    user_per_asin_count: dict[str, Counter[str]] = defaultdict(Counter)
    user_total: Counter[str] = Counter()
    first_asin_per_user: dict[str, str] = {}

    n = 0
    with gzip.open(REVIEWS_GZ, "rt") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            uid = r.get("reviewerID") or r.get("user_id")
            asin = r.get("asin")
            if not uid or not asin:
                continue
            user_total[uid] += 1
            asin_users[asin].add(uid)
            user_per_asin_count[asin][uid] += 1
            if uid not in first_asin_per_user:
                first_asin_per_user[uid] = asin
            n += 1
            if n % 1_000_000 == 0:
                log(f"  {n/1e6:.1f}M records")

    log(f"  done: {n} records, {len(asin_users)} products, "
        f"{len(user_total)} users (no user-level filter)")
    return asin_users, user_per_asin_count, user_total, first_asin_per_user


# ============================================================
# Step 3 — build_query_records
# ============================================================
def step3_build_query_records(
    product_attrs: dict[str, dict],
    user_total: Counter[str],
    first_asin_per_user: dict[str, str],
) -> None:
    log("=== Step 3: build_query_records ===")
    top_users = [u for u, _ in user_total.most_common(TOP_K_USERS)]
    log(f"  top-{TOP_K_USERS} active users selected")

    records: list[dict] = []
    n_with_attrs = 0
    n_no_attrs = 0
    for uid in top_users:
        primary_asin = first_asin_per_user.get(uid)
        if not primary_asin:
            continue
        asin_attrs = product_attrs.get(primary_asin)
        if not asin_attrs:
            n_no_attrs += 1
            continue
        # Step 3: max_n 默认 = 5 (select_top_attrs 默认参数)
        attrs = select_top_attrs(asin_attrs)
        if not attrs:
            n_no_attrs += 1
            continue
        n_with_attrs += 1
        records.append({
            "user_id": uid,
            "asin": primary_asin,
            "attrs_used": attrs,
            "n_attrs": len(attrs),
            "n_product_attrs": len(asin_attrs),
        })
        if len(records) >= MAX_RECORDS:
            break

    QUERY_RECORDS_JSON.parent.mkdir(parents=True, exist_ok=True)
    QUERY_RECORDS_JSON.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"  wrote {len(records)} records, with_attrs={n_with_attrs}, "
        f"skipped={n_no_attrs}")
    log(f"  wrote → {QUERY_RECORDS_JSON}")
    if records:
        sample = records[0]
        log(f"  sample: user={sample['user_id'][:8]}... asin={sample['asin']} "
            f"({sample['n_attrs']}/{sample['n_product_attrs']} attrs)")


# ============================================================
# Step 4 — build_stage8_5_asins
# ============================================================
def step4_build_stage8_5_asins(
    product_attrs: dict[str, dict],
    asin_users: dict[str, set[str]],
    user_per_asin_count: dict[str, Counter[str]],
    user_total: Counter[str],
) -> None:
    log("=== Step 4: build_stage8_5_asins ===")
    # 用户指令 2026-08-28: 删除 MIN_REVIEWS_PER_USER 阈值, 所有评论过该 ASIN 的 user 进入 cohort。
    # 用户指令 2026-08-29: 撤掉 mean_wc ≥ 20 过滤, 所有有评论的用户都进入 cohort。
    # 用户质量 (Gaussian reliability) 推迟到 Stage 3 + reliability filter。
    eligible_users_set = set(user_total.keys())
    log(f"  all commenters (no user filter): {len(eligible_users_set)}")

    eligible_count: dict[str, int] = {}
    for asin, uset in asin_users.items():
        # 用户指令 2026-08-28: 删除 MIN_STRONG_SIGNAL_USERS_PER_ASIN 下限。
        # 每 ASIN 配全部有评论的用户, 即使只有 1 个用户也允许进入 cohort。
        c = len([u for u in uset if u in eligible_users_set])
        eligible_count[asin] = c
    log(f"  ASINs with ≥1 commenters: {len(eligible_count)}")

    ranked = sorted(eligible_count.items(), key=lambda kv: kv[1], reverse=True)
    ranked = ranked[:TOP_N_ASINS]
    counts = [c for _, c in ranked]
    if counts:
        log(f"  top {TOP_N_ASINS} ASINs: min={min(counts)}, "
            f"median={sorted(counts)[len(counts)//2]}, max={max(counts)}")

    asins_out: list[dict] = []
    skipped_no_attrs = 0
    for asin, _ in ranked:
        adoc = product_attrs.get(asin) or {}
        # Step 4: max_n = MAX_ATTRS_FOR_LLM = 4 (Stage 4 prompt 上限)
        attrs_used = select_top_attrs(adoc, max_n=MAX_ATTRS_FOR_LLM)
        if len(attrs_used) < 1:
            skipped_no_attrs += 1
            continue
        # 用户指令 2026-08-28: 删除 MAX_USERS_PER_ASIN 上限, 保留所有评论过该 ASIN 的用户
        top_users = sorted(
            [u for u in user_per_asin_count[asin]
             if u in eligible_users_set],
            key=lambda u: user_per_asin_count[asin][u],
            reverse=True,
        )
        asins_out.append({
            "asin": asin,
            "n_users_eligible": eligible_count[asin],
            "users_sampled": top_users,
            "attrs_used": attrs_used,
            "filter": {
                "note": "no user filter (all commenters enter cohort)",
            },
        })

    log(f"  final ASINs: {len(asins_out)}, skipped (no attrs): "
        f"{skipped_no_attrs}")

    config = {
        "description": (f"top {TOP_N_ASINS} ASINs × all commenters "
                        f"(no user filter, "
                        f"max {MAX_ATTRS_FOR_LLM} non-numeric attrs/ASIN, "
                        f"no per-ASIN user cap, no min review count)"),
        "TOP_N_ASINS": TOP_N_ASINS,
        "MAX_ATTRS_FOR_LLM": MAX_ATTRS_FOR_LLM,
    }
    out = {
        "config": config,
        "n_asins": len(asins_out),
        "asins": asins_out,
    }
    STAGE8_5_ASINS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STAGE8_5_ASINS_JSON.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"  wrote → {STAGE8_5_ASINS_JSON}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    log("=== build_user_cohort.py — Steps 2-4 ===")
    log("(Step 1 由 attribute_extraction/extract_product_attrs.py 单独跑)")

    # 必须先跑 Step 1 生成 product_attributes.json (依赖此文件读 attrs)
    product_attrs_path = REPO_ROOT / "result" / "product_attributes.json"
    if not product_attrs_path.exists():
        log(f"ERROR: {product_attrs_path} 不存在")
        log("请先跑: python attribute_extraction/extract_product_attrs.py")
        raise SystemExit(1)
    product_attrs = json.loads(product_attrs_path.read_text(encoding="utf-8"))
    log(f"  loaded {len(product_attrs)} ASIN attrs from {product_attrs_path.name}")

    # Step 2
    asin_users, user_per_asin_count, user_total, first_asin_per_user = \
        step2_scan_reviews()

    # Step 3 (用 Step 2 的 in-memory 数据)
    step3_build_query_records(product_attrs, user_total, first_asin_per_user)

    # Step 4
    step4_build_stage8_5_asins(product_attrs, asin_users,
                               user_per_asin_count, user_total)

    log("=== ALL DONE ===")


if __name__ == "__main__":
    main()
