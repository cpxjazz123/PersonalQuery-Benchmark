#!/usr/bin/env python3
"""User pipeline: cohort construction + per-user review metadata.

合并 gaussian/ 的两个职责到单一脚本(用户指令 2026-08-29):
  - Phase 1 (Steps 2 + 4): cohort construction
      - step2_scan_reviews: 单次扫描 reviews → 3-tuple
      - step4_build_stage8_5_asins: 全部 ASIN × all commenters × top-5 non-numeric attrs
        (MAX_ATTRS_FOR_LLM=5, 对齐 Stage 1 N_INPUT=5)
  - Phase 2 (Stage 3): per-user review metadata
      - 之前: PCA48 + Welford online + shrinkage → user_gaussians.json (1.5GB)
      - 现在: 仅统计 n_reviews + source (~1.5GB after no-cohort)
      - 删除原因: Stage 4 strict alignment 用 _syntax_subspace_prepare() 重新计算
        whitened user-means, 不依赖 Phase 2 输出的 mu/sigma_diag/n_words/n_sentences
        (下游 Stage 4 只读 source + n_reviews)
      - 输出位置: scratch2/stage8_5_user_gaussians.json (Rule 10: >100MB 不放 result/)

Idempotent: 每 phase 检查输出文件 + signature, 命中 cache 直接跳过。

参数全部硬编码 (Rule 3), 不接受 CLI 参数.

Running order:
  python attribute_extraction/extract_product_attrs.py   # Step 1
  python gaussian/build_user.py                          # Phase 1 + Phase 2
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

# 让 build_user.py 能 import 兄弟目录 attribute_extraction/extract_product_attrs
sys.path.insert(0, str(REPO_ROOT))
from attribute_extraction.extract_product_attrs import select_top_attrs  # noqa: E402

# 让 build_user.py 能 import 兄弟目录 common/syntax_subspace_utils
sys.path.insert(0, str(REPO_ROOT / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, GAUSSIANS_OUT, MIN_REVIEWS_FOR_PER_USER, REVIEW_GZ, log,
)

# === Inputs ===
DATA = Path("/fs04/ar57/wenyu/PersoanlQuery/data")
REVIEWS_GZ = DATA / "Baby_Products_2023.jsonl.gz"

# === Outputs ===
STAGE8_5_ASINS_JSON = SCRATCH / "stage8_5_asins.json"

# === Phase 1 — cohort construction 参数 ===
TOP_N_ASINS = 10_000
# 用户指令 2026-08-29: 对齐 Stage 1 N_INPUT=5(原 MAX_ATTRS_FOR_LLM=4 是遗留,
# Stage 1 已直接读 product_attributes.json 取 top-5, stage8_5_asins.json 的 attrs_used
# 仅供 Stage 4 select 链路追踪)。
MAX_ATTRS_FOR_LLM = 5


# ============================================================================
# Phase 1 — cohort construction (Steps 2-4)
# ============================================================================
def phase1_cohort_construction() -> bool:
    """Steps 2 + 4: scan reviews + build_stage8_5_asins.

    Returns True if Phase 1 ran end-to-end, False if skipped (cache hit).
    """
    # --- Cache detection ---
    sig_payload = json.dumps({
        "TOP_N_ASINS": TOP_N_ASINS,
        "MAX_ATTRS_FOR_LLM": MAX_ATTRS_FOR_LLM,
        "REVIEWS_GZ": str(REVIEWS_GZ),
        "REVIEWS_GZ_mtime": REVIEWS_GZ.stat().st_mtime if REVIEWS_GZ.exists() else 0,
        "PRODUCT_ATTRS_JSON": "result/product_attributes.json",
        "PRODUCT_ATTRS_JSON_mtime": (REPO_ROOT / "result/product_attributes.json").stat().st_mtime
            if (REPO_ROOT / "result/product_attributes.json").exists() else 0,
    }, sort_keys=True)
    sig_hash = hashlib.sha1(sig_payload.encode("utf-8")).hexdigest()[:12]
    log(f"[build_user] Phase 1 signature: {sig_hash}")

    if STAGE8_5_ASINS_JSON.exists():
        try:
            cached = json.load(open(STAGE8_5_ASINS_JSON))
            cached_sig = cached.get("config", {}).get("cache_signature")
            if cached_sig == sig_hash:
                cached_n = cached.get("n_asins", 0)
                log(f"[build_user] Phase 1 CACHE HIT: {cached_n} ASINs from {STAGE8_5_ASINS_JSON}")
                log("  (upstream files + config unchanged, skip Phase 1)")
                return False
            else:
                log(f"  Phase 1 cache signature mismatch (cached={cached_sig}, current={sig_hash}), recomputing...")
        except Exception as e:
            log(f"  Phase 1 cache load failed ({e!r}), recomputing...")
    else:
        log(f"  no Phase 1 cache at {STAGE8_5_ASINS_JSON}, computing from scratch...")

    log("[build_user] === Phase 1: cohort construction ===")

    # 必须先跑 Step 1 生成 product_attributes.json
    product_attrs_path = REPO_ROOT / "result" / "product_attributes.json"
    if not product_attrs_path.exists():
        log(f"ERROR: {product_attrs_path} 不存在")
        log("请先跑: python attribute_extraction/extract_product_attrs.py")
        raise SystemExit(1)
    product_attrs = json.loads(product_attrs_path.read_text(encoding="utf-8"))
    log(f"  loaded {len(product_attrs)} ASIN attrs from {product_attrs_path.name}")

    # Step 2 (returns 3-tuple; first_asin_per_user dropped since Step 3 deleted)
    asin_users, user_per_asin_count, user_total = _step2_scan_reviews()

    # Step 4
    _step4_build_stage8_5_asins(product_attrs, asin_users, user_per_asin_count, user_total)

    # Tag cache signature into stage8_5_asins.json for idempotent re-runs
    if STAGE8_5_ASINS_JSON.exists():
        cached = json.load(open(STAGE8_5_ASINS_JSON))
        cached.setdefault("config", {})["cache_signature"] = sig_hash
        cached.setdefault("config", {})["phase"] = "1_cohort_construction"
        STAGE8_5_ASINS_JSON.write_text(
            json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(f"[build_user] Phase 1 wrote → {STAGE8_5_ASINS_JSON} (signature={sig_hash})")

    return True


def _step2_scan_reviews() -> tuple[dict[str, set[str]], dict[str, Counter[str]],
                                  Counter[str]]:
    """单次扫描 reviews, 返回 3-tuple:
      - asin_users: asin -> {uid set}
      - user_per_asin_count: asin -> Counter[uid -> count]
      - user_total: Counter[uid -> total reviews]

    用户指令 2026-08-29: 去掉 first_asin_per_user(原为 Step 3 用,Step 3 已删除,
    Rule 7 禁止保留无人消费的字段)。
    """
    log("  [Step 2] scan_reviews (single pass)")
    asin_users: dict[str, set[str]] = defaultdict(set)
    user_per_asin_count: dict[str, Counter[str]] = defaultdict(Counter)
    user_total: Counter[str] = Counter()

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
            n += 1
            if n % 1_000_000 == 0:
                log(f"    {n/1e6:.1f}M records")

    log(f"  done: {n} records, {len(asin_users)} products, "
        f"{len(user_total)} users (no user-level filter)")
    return asin_users, user_per_asin_count, user_total


def _step4_build_stage8_5_asins(
    product_attrs: dict[str, dict],
    asin_users: dict[str, set[str]],
    user_per_asin_count: dict[str, Counter[str]],
    user_total: Counter[str],
) -> None:
    log("  [Step 4] build_stage8_5_asins")
    eligible_users_set = set(user_total.keys())
    log(f"    all commenters (no user filter): {len(eligible_users_set)}")

    eligible_count: dict[str, int] = {}
    for asin, uset in asin_users.items():
        c = len([u for u in uset if u in eligible_users_set])
        eligible_count[asin] = c
    log(f"    ASINs with ≥1 commenters: {len(eligible_count)}")

    ranked = sorted(eligible_count.items(), key=lambda kv: kv[1], reverse=True)
    ranked = ranked[:TOP_N_ASINS]
    counts = [c for _, c in ranked]
    if counts:
        log(f"    top {TOP_N_ASINS} ASINs: min={min(counts)}, "
            f"median={sorted(counts)[len(counts)//2]}, max={max(counts)}")

    asins_out: list[dict] = []
    skipped_no_attrs = 0
    for asin, _ in ranked:
        adoc = product_attrs.get(asin) or {}
        attrs_used = select_top_attrs(adoc, max_n=MAX_ATTRS_FOR_LLM)
        if len(attrs_used) < 1:
            skipped_no_attrs += 1
            continue
        top_users = sorted(
            [u for u in user_per_asin_count[asin] if u in eligible_users_set],
            key=lambda u: user_per_asin_count[asin][u],
            reverse=True,
        )
        asins_out.append({
            "asin": asin,
            "n_users_eligible": eligible_count[asin],
            "users_sampled": top_users,
            "attrs_used": attrs_used,
            "filter": {"note": "no user filter (all commenters enter cohort)"},
        })

    log(f"    final ASINs: {len(asins_out)}, skipped (no attrs): {skipped_no_attrs}")

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
    log(f"    wrote → {STAGE8_5_ASINS_JSON}")


# ============================================================================
# Phase 2 — Stage 3 user review metadata
# ============================================================================
def phase2_user_gaussians() -> None:
    """Stage 3: per-user review metadata (n_reviews + source).

    User directive 2026-08-29: 删除 PCA48 + Welford + shrinkage 计算。
    Stage 4 strict alignment 用 _syntax_subspace_prepare() 重新计算 whitened
    user-means, 不依赖 Phase 2 输出的 mu/sigma_diag/n_words/n_sentences。
    下游 Stage 4 只读 source + n_reviews。
    """
    # --- Cache detection ---
    sig_payload = json.dumps({
        "MIN_REVIEWS_FOR_PER_USER": MIN_REVIEWS_FOR_PER_USER,
        "ASINS_IN": str(ASINS_IN),
        "ASINS_IN_mtime": ASINS_IN.stat().st_mtime if ASINS_IN.exists() else 0,
        "REVIEW_GZ": str(REVIEW_GZ),
        "REVIEW_GZ_mtime": REVIEW_GZ.stat().st_mtime if REVIEW_GZ.exists() else 0,
    }, sort_keys=True)
    sig_hash = hashlib.sha1(sig_payload.encode("utf-8")).hexdigest()[:12]
    log(f"[build_user] Phase 2 signature: {sig_hash}")

    if GAUSSIANS_OUT.exists():
        try:
            cached = json.load(open(GAUSSIANS_OUT))
            cached_sig = cached.get("config", {}).get("cache_signature")
            if cached_sig == sig_hash:
                cached_users = cached.get("users", {})
                log(f"[build_user] Phase 2 CACHE HIT: {len(cached_users)} users from {GAUSSIANS_OUT}")
                log("  (upstream files + config unchanged, skip Phase 2)")
                return
            else:
                log(f"  Phase 2 cache signature mismatch (cached={cached_sig}, current={sig_hash}), recomputing...")
        except Exception as e:
            log(f"  Phase 2 cache load failed ({e!r}), recomputing...")
    else:
        log(f"  no Phase 2 cache at {GAUSSIANS_OUT}, computing from scratch...")

    log("[build_user] === Phase 2: per-user review metadata ===")

    log("\n=== 1. Loading target users ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    target_users = set()
    for a in asin_data:
        for uid in a["users_sampled"]:
            target_users.add(uid)
    log(f"  target users: {len(target_users)}")

    log("\n=== 2. Scanning review corpus (count only) ===")
    user_n_reviews: Counter[str] = Counter()
    n_records = 0
    for line in gzip.open(REVIEW_GZ, "rt", encoding="utf-8"):
        r = json.loads(line)
        uid = r.get("reviewerID") or r.get("user_id")
        if uid in target_users and r.get("text"):
            user_n_reviews[uid] += 1
        n_records += 1
        if n_records % 2_000_000 == 0:
            log(f"    {n_records/1e6:.1f}M records")
    log(f"  total records: {n_records}")
    log(f"  users with reviews: {len(user_n_reviews)}")

    if user_n_reviews:
        counts = list(user_n_reviews.values())
        log(f"  review count: min={min(counts)}, "
            f"mean={sum(counts)/len(counts):.1f}, max={max(counts)}")
    n_high = sum(1 for c in user_n_reviews.values() if c >= MIN_REVIEWS_FOR_PER_USER)
    log(f"  users with >={MIN_REVIEWS_FOR_PER_USER} reviews: {n_high}")
    n_zero = len(target_users) - len(user_n_reviews)
    log(f"  users with 0 reviews: {n_zero}")

    log("\n=== 3. Saving ===")
    user_metadata: dict[str, dict] = {}
    for uid in target_users:
        n = user_n_reviews.get(uid, 0)
        user_metadata[uid] = {
            "n_reviews": n,
            "source": "per_user" if n >= MIN_REVIEWS_FOR_PER_USER else "no_reviews",
        }

    GAUSSIANS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GAUSSIANS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 8.5: per-user review metadata (n_reviews + source). "
                                "Phase 2 no longer computes per-user Gaussian — Stage 4 strict alignment "
                                "computes whitened user-means from _syntax_subspace_prepare."),
                "MIN_REVIEWS_FOR_PER_USER": MIN_REVIEWS_FOR_PER_USER,
                "cache_signature": sig_hash,
            },
            "users": user_metadata,
            "n_users_with_reviews": len(user_n_reviews),
            "n_users_skipped": n_zero,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {GAUSSIANS_OUT}")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    log("=== build_user.py — Phase 1 (cohort) + Phase 2 (Stage 3 Gaussian) ===")

    # Phase 1 — cohort construction (Steps 2-4)
    ran_p1 = phase1_cohort_construction()

    # Phase 2 — per-user Gaussian fitting (Stage 3)
    phase2_user_gaussians()

    log("=== build_user.py — ALL DONE ===")


if __name__ == "__main__":
    main()