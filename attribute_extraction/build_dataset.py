#!/usr/bin/env python3
"""End-to-end dataset construction for Syntax Subspace pipeline + copy_aware route.

Step 1: extract_attrs
    meta_Baby_Products_2023.jsonl.gz -> result/product_attributes.json
    (结构化属性: details 中的短字符串 + top-level 兜底 Brand/Main Category/...)

Step 2: scan_reviews
    Baby_Products_2023.jsonl.gz 单次扫描, 同时累计:
      - user_total[uid] = 评论总数
      - asin_users[asin] = 评论过该 ASIN 的 user set
      - user_per_asin_count[asin][uid] = (uid, asin) 关联计数
      - all_pairs[(uid, asin)] = 全部 (user, asin) 关联, 用于 build_query_records

Step 3: build_query_records
    top-10K most active users × primary (user, asin) pair + product_attributes
    -> result/query_records_10k.json
    (供 copy_aware_generate / copy_aware_train 使用, 属性选择策略:
     ATTR_PRIORITY + MAX_ATTRS=5 + exclude_numeric=True)

Step 4: build_stage8_5_asins
    heavy users (≥MIN_REVIEWS_PER_USER) × ASINs with ≥MIN_USERS_PER_ASIN
    × top-10 users/ASIN × attrs_used (PREFERRED_ATTRS) filter
    -> scratch2/stage8_5_asins_1409_u20.json
    (供 Syntax Subspace pipeline 5 stages 使用)

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

# === Inputs ===
DATA = Path("/fs04/ar57/wenyu/PersoanlQuery/data")
REVIEWS_GZ = DATA / "Baby_Products_2023.jsonl.gz"
META_GZ = DATA / "meta_Baby_Products_2023.jsonl.gz"

# === Outputs ===
PRODUCT_ATTRS_JSON = REPO_ROOT / "result" / "product_attributes.json"
QUERY_RECORDS_JSON = REPO_ROOT / "result" / "query_records_10k.json"
STAGE8_5_ASINS_JSON = SCRATCH / "stage8_5_asins.json"

# === Step 1 — extract_attrs 参数 ===
MAX_STR_LEN = 200
TOP_LEVEL_NUMERIC_FIELDS = ("average_rating", "rating_number", "price")

# === Step 3 — build_query_records 参数 ===
TOP_K_USERS = 10_000
MAX_RECORDS = 10_000
ATTR_PRIORITY = [
    "Brand", "Main Category", "Item model number", "Manufacturer",
    "Color", "Material", "Material Type", "Fabric Type", "Frame Material",
    "Item Weight", "Product Dimensions", "Size", "Style", "Pattern", "Theme",
    "Age Range (Description)", "Special Feature", "Target gender",
    "Batteries required", "Number Of Items", "Is Discontinued By Manufacturer",
    "Date First Available", "Country/Region of origin", "Country of Origin",
    "Price", "Average Rating", "Rating Number",
]
MAX_ATTRS = 5
MAX_ATTR_VALUE_LEN = 100
EXCLUDE_NUMERIC_ATTRS = True
_NUMERIC_KEYWORDS = {"price", "average rating", "rating number", "item weight",
                     "item model number", "date first available",
                     "package dimensions", "product dimensions",
                     "minimum weight recommendation",
                     "maximum weight recommendation",
                     "batteries required", "is discontinued by manufacturer"}

# === Step 4 — build_stage8_5_asins 参数 ===
# 用户指令 2026-08-28: 删除 cohort user per ASIN 的所有硬阈值 (MIN_REVIEWS_PER_USER=20,
# MIN_USERS_PER_ASIN=10, MAX_USERS_PER_ASIN=10)。
# 每 ASIN 配所有评论过该 ASIN 的 heavy+strong-signal 用户 (mean_wc≥MIN_MEAN_WC)。
# 用户质量过滤 (Gaussian fit reliability) 推迟到 Stage 3 之后,
# 由 selection 阶段按 LBF/n_samples 过滤, 而不是 cohort 阶段。
# Step 4 复用 Step 3 的 select_top_attrs(max_n=4) — 包含数值过滤 + 4 上限
TOP_N_ASINS = 10_000   # 去掉 1409 硬上限, 取全部候选
MAX_ATTRS_FOR_LLM = 4
# 用户指令 2026-08-27: "强信号用户" = mean_wc ≥ 20 词/review
# 实证 (feat_density_diag3.py): 1-2 词 review 96.7% all-zero nz,
# mean_wc < 20 用户 Gaussian 与 global pool LBF 中位数 ≤ 0,
# Mahalanobis 信号无法独立于 random selection。
# 强信号用户筛选保留 (mean_wc 过滤仍是有意义的最低信号条件);
# cohort 不再 cap 每 ASIN 上限, 也不再要求最低 user 数 / 最低 review 数。
MIN_MEAN_WC = 20

# 用户指令 2026-08-29: 二级过滤 = 总字数 MIN_TOTAL_WORDS = 1000
# 实证 (旧 4324-user cohort 分布): median 1736 words, p10=703, p25=1017
# < 500 words: σ² Ledoit-Wolf 收缩必要, μ 估计不可靠 (n_samples < PCA_dim=48)
# 500-1000 words: 渐近稳定, μ 误差 < 5%
# 1000-2000 words: σ² 收敛, μ 误差 < 2% (推荐下限)
# > 2000 words: 接近真实分布, SOTA
# 用户决策: 1000 (≈ 50 reviews × 20 wc 或 30 reviews × 33 wc)
MIN_TOTAL_WORDS = 1000


def log(msg: str) -> None:
    print(f"[build_dataset] {msg}", flush=True)


# ============================================================
# Step 1 — extract_attrs
# ============================================================
def extract_attrs(d: dict) -> dict:
    details = d.get("details")
    if not isinstance(details, dict):
        details = {}
    out: dict = {}

    # details 中的所有短字符串字段
    for k, v in details.items():
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s or len(s) > MAX_STR_LEN:
            continue
        out[k] = s

    # 用户指令 2026-08-27: 删除 Brand fallback (Rule 7 禁止降级)
    # 原代码: details 无 Brand → store → title[0], 三层降级
    # 现: 仅从 details 取; Brand 缺失则 out 无 Brand 键,
    # 下游 select_top_attrs 按 ATTR_PRIORITY 跳过, 不影响其他字段

    # top-level 结构化字段
    main_cat = d.get("main_category")
    if isinstance(main_cat, str) and main_cat.strip():
        out["Main Category"] = main_cat.strip()

    for field in TOP_LEVEL_NUMERIC_FIELDS:
        v = d.get(field)
        if v is None or v == "":
            continue
        out[field.replace("_", " ").title().replace(" ", " ")] = v

    return out


def step1_extract_attrs() -> dict[str, dict]:
    log("=== Step 1: extract_attrs ===")
    product_attrs: dict[str, dict] = {}
    n_total = 0
    n_with_attrs = 0
    n_with_details = 0
    n_top_level_only = 0
    field_counter: dict[str, int] = {}

    with gzip.open(META_GZ, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_total += 1
            asin = d.get("parent_asin")
            if not asin:
                continue
            attrs = extract_attrs(d)
            if attrs:
                product_attrs[asin] = attrs
                n_with_attrs += 1
                has_details = any(
                    k not in ("Brand", "Main Category", "Average Rating",
                              "Rating Number", "Price")
                    for k in attrs
                )
                if has_details:
                    n_with_details += 1
                else:
                    n_top_level_only += 1
                for k in attrs:
                    field_counter[k] = field_counter.get(k, 0) + 1

    PRODUCT_ATTRS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PRODUCT_ATTRS_JSON.write_text(
        json.dumps(product_attrs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"  total metadata products={n_total}, with attrs={n_with_attrs}")
    log(f"    details-based={n_with_details}, top-level only={n_top_level_only}")
    log(f"    distinct attr fields={len(field_counter)}")
    log(f"  wrote → {PRODUCT_ATTRS_JSON}")
    return product_attrs


# ============================================================
# Step 2 — scan_reviews
# ============================================================
def step2_scan_reviews(
    product_attrs: dict[str, dict],
) -> tuple[dict[str, set[str]], dict[str, Counter[str]], Counter[str],
           Counter[str], dict[str, str]]:
    """单次扫描 reviews: 返回
      - asin_users: asin -> {uid set}
      - user_per_asin_count: asin -> Counter[uid -> count]
      - user_total: Counter[uid -> total reviews]
      - user_words_total: Counter[uid -> total words (sum of len(text.split()))]
      - first_asin_per_user: uid -> first asin seen (供 query_records primary)
    """
    log("=== Step 2: scan_reviews (single pass) ===")
    asin_users: dict[str, set[str]] = defaultdict(set)
    user_per_asin_count: dict[str, Counter[str]] = defaultdict(Counter)
    user_total: Counter[str] = Counter()
    user_words_total: Counter[str] = Counter()
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
            text = r.get("text") or ""
            if not uid or not asin:
                continue
            user_total[uid] += 1
            # 用户指令 2026-08-27: 累计 words 总数供 mean_wc 计算 (强信号用户过滤)
            # split() 后 join 长度即为 word count, 不重复 tokenize
            if text:
                user_words_total[uid] += len(text.split())
            asin_users[asin].add(uid)
            user_per_asin_count[asin][uid] += 1
            if uid not in first_asin_per_user:
                first_asin_per_user[uid] = asin
            n += 1
            if n % 1_000_000 == 0:
                log(f"  {n/1e6:.1f}M records")

    log(f"  done: {n} records, {len(asin_users)} products, "
        f"{len(user_total)} users, words total: {sum(user_words_total.values())}")
    return asin_users, user_per_asin_count, user_total, user_words_total, first_asin_per_user


# ============================================================
# Step 3 — build_query_records
# ============================================================
def has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)


def select_top_attrs(asin_attrs: dict, max_n: int = MAX_ATTRS) -> dict:
    """与原 build_query_records.py::select_top_attrs 一致的字段选择逻辑.
    用户指令: 去掉含数字的 value, 上限 max_n (Step 3 默认 5, Step 4 用 4).
    """
    def _skip(k: str, s: str) -> bool:
        if not s or len(s) > MAX_ATTR_VALUE_LEN:
            return True
        if EXCLUDE_NUMERIC_ATTRS:
            k_low = k.lower()
            if any(nk in k_low for nk in _NUMERIC_KEYWORDS):
                return True
            if has_digit(s):
                return True
        return False

    out: dict = {}
    used: set[str] = set()
    for k in ATTR_PRIORITY:
        v = asin_attrs.get(k)
        if not v:
            continue
        s = str(v).strip()
        if _skip(k, s):
            continue
        out[k] = s
        used.add(k)
        if len(out) >= max_n:
            return out
    for k, v in asin_attrs.items():
        if k in used:
            continue
        if v is None:
            continue
        s = str(v).strip()
        if _skip(k, s):
            continue
        out[k] = s
        if len(out) >= max_n:
            break
    return out


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
    user_words_total: Counter[str],
) -> None:
    log("=== Step 4: build_stage8_5_asins ===")
    # 用户指令 2026-08-28: 删除 MIN_REVIEWS_PER_USER 阈值, 所有评论过该 ASIN 的 user 进入 cohort。
    # 用户质量 (Gaussian reliability) 推迟到 Stage 3 + reliability filter。
    eligible_users_set = set(user_total.keys())

    # 用户指令 2026-08-27: 强信号用户 = mean_wc ≥ MIN_MEAN_WC (20 词/review)
    # mean_wc = user_words_total[u] / user_total[u]
    # 短 review (1-2 词) 的 spaCy 182d features 96.7% 全 0,
    # 强信号用户在 Mahalanobis 空间才能与 global pool 区分, 否则 random 等价
    strong_users: dict[str, float] = {}
    for u in eligible_users_set:
        wc = user_words_total.get(u, 0)
        nt = user_total[u]
        if nt == 0:
            continue
        # 用户指令 2026-08-29: 二级过滤 = 总字数 MIN_TOTAL_WORDS = 1000
        # 单 review ≥ 20 words (强信号) AND 总 ≥ 1000 words (拟合质量)
        if wc < MIN_TOTAL_WORDS:
            continue
        mean_wc = wc / nt
        if mean_wc >= MIN_MEAN_WC:
            strong_users[u] = mean_wc
    log(f"  strong-signal users (mean_wc ≥ {MIN_MEAN_WC} AND "
        f"total ≥ {MIN_TOTAL_WORDS} words): {len(strong_users)} "
        f"(of {len(eligible_users_set)} all commenters)")

    eligible_count: dict[str, int] = {}
    for asin, uset in asin_users.items():
        # 用户指令 2026-08-28: 删除 MIN_STRONG_SIGNAL_USERS_PER_ASIN 下限。
        # 每 ASIN 配全部 strong-signal 用户, 即使只有 1 个用户也允许进入 cohort。
        c = len([u for u in uset if u in strong_users])
        eligible_count[asin] = c
    log(f"  ASINs with ≥1 strong-signal users (mean_wc≥{MIN_MEAN_WC}): {len(eligible_count)}")

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
        # 复用 Step 3 的 select_top_attrs: 去掉含数字的 value, 上限 4 个
        attrs_used = select_top_attrs(adoc, max_n=MAX_ATTRS_FOR_LLM)
        if len(attrs_used) < 1:
            skipped_no_attrs += 1
            continue
        # 用户指令 2026-08-28: 删除 MAX_USERS_PER_ASIN 上限, 保留所有 strong-signal 用户
        top_users = sorted(
            [u for u in user_per_asin_count[asin]
             if u in strong_users],
            key=lambda u: user_per_asin_count[asin][u],
            reverse=True,
        )
        asins_out.append({
            "asin": asin,
            "n_users_eligible": eligible_count[asin],
            "users_sampled": top_users,
            "attrs_used": attrs_used,
            "filter": {
                "MIN_MEAN_WC": MIN_MEAN_WC,
            },
        })

    log(f"  final ASINs: {len(asins_out)}, skipped (no attrs): "
        f"{skipped_no_attrs}")

    config = {
        "description": (f"top {TOP_N_ASINS} ASINs × all strong-signal users "
                        f"(mean_wc ≥ {MIN_MEAN_WC}, "
                        f"max {MAX_ATTRS_FOR_LLM} non-numeric attrs/ASIN, "
                        f"no per-ASIN user cap, no min review count)"),
        "TOP_N_ASINS": TOP_N_ASINS,
        "MAX_ATTRS_FOR_LLM": MAX_ATTRS_FOR_LLM,
        "MIN_MEAN_WC": MIN_MEAN_WC,
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
    log("=== build_dataset.py — 4-step end-to-end ===")

    # Step 1
    product_attrs = step1_extract_attrs()

    # Step 2
    asin_users, user_per_asin_count, user_total, user_words_total, first_asin_per_user = \
        step2_scan_reviews(product_attrs)

    # Step 3 (用 Step 2 的 in-memory 数据, 不再依赖 stage1 中间文件)
    step3_build_query_records(product_attrs, user_total, first_asin_per_user)

    # Step 4
    step4_build_stage8_5_asins(product_attrs, asin_users,
                                user_per_asin_count, user_total,
                                user_words_total)

    log("=== ALL DONE ===")


if __name__ == "__main__":
    main()