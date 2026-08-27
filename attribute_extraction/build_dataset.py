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
STAGE8_5_ASINS_JSON = SCRATCH / "stage8_5_asins_1409_u20.json"

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
# 用户指令: 1) 去掉有数字的属性(value 含数字), 2) 传给 LLM 上限 4 个
# Step 4 复用 Step 3 的 select_top_attrs(max_n=4) — 包含数值过滤 + 4 上限
MIN_REVIEWS_PER_USER = 20
MIN_USERS_PER_ASIN = 10
MAX_USERS_PER_ASIN = 10
TOP_N_ASINS = 1409
MAX_ATTRS_FOR_LLM = 4


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

    # Brand fallback: details 无 Brand 时用 store / title
    if "Brand" not in out:
        store = d.get("store")
        if isinstance(store, str) and store.strip():
            out["Brand"] = store.strip()
        else:
            title = d.get("title")
            if isinstance(title, str):
                tok = title.split()
                if tok:
                    cand = tok[0].strip()
                    if cand:
                        out["Brand"] = cand

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
           dict[str, dict[str, int]]]:
    """单次扫描 reviews: 返回
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
        f"{len(user_total)} users")
    return asin_users, user_per_asin_count, user_total, first_asin_per_user


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
) -> None:
    log("=== Step 4: build_stage8_5_asins ===")
    heavy_users = {u for u, c in user_total.items()
                   if c >= MIN_REVIEWS_PER_USER}
    log(f"  heavy users (≥{MIN_REVIEWS_PER_USER} total reviews): {len(heavy_users)}")

    eligible_count: dict[str, int] = {}
    for asin, uset in asin_users.items():
        c = len([u for u in uset if u in heavy_users])
        if c >= MIN_USERS_PER_ASIN:
            eligible_count[asin] = c
    log(f"  ASINs with ≥{MIN_USERS_PER_ASIN} heavy reviewers: "
        f"{len(eligible_count)}")

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
        top_users = [u for u, _ in user_per_asin_count[asin]
                     .most_common(MAX_USERS_PER_ASIN)]
        asins_out.append({
            "asin": asin,
            "n_users_eligible": eligible_count[asin],
            "users_sampled": top_users,
            "attrs_used": attrs_used,
        })

    log(f"  final ASINs: {len(asins_out)}, skipped (no attrs): "
        f"{skipped_no_attrs}")

    config = {
        "description": (f"top {TOP_N_ASINS} ASINs × top-{MAX_USERS_PER_ASIN} "
                        f"users (each ≥{MIN_REVIEWS_PER_USER} reviews, "
                        f"max {MAX_ATTRS_FOR_LLM} non-numeric attrs/ASIN)"),
        "MIN_REVIEWS_PER_USER": MIN_REVIEWS_PER_USER,
        "MIN_USERS_PER_ASIN": MIN_USERS_PER_ASIN,
        "MAX_USERS_PER_ASIN": MAX_USERS_PER_ASIN,
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
    log("=== build_dataset.py — 4-step end-to-end ===")

    # Step 1
    product_attrs = step1_extract_attrs()

    # Step 2
    asin_users, user_per_asin_count, user_total, first_asin_per_user = \
        step2_scan_reviews(product_attrs)

    # Step 3 (用 Step 2 的 in-memory 数据, 不再依赖 stage1 中间文件)
    step3_build_query_records(product_attrs, user_total, first_asin_per_user)

    # Step 4
    step4_build_stage8_5_asins(product_attrs, asin_users,
                                user_per_asin_count, user_total)

    log("=== ALL DONE ===")


if __name__ == "__main__":
    main()