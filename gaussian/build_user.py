#!/usr/bin/env python3
"""User pipeline: cohort construction + per-user Gaussian fitting.

合并 gaussian/ 的两个职责到单一脚本(用户指令 2026-08-29):
  - Phase 1 (Steps 2 + 4): cohort construction
      - step2_scan_reviews: 单次扫描 reviews → 3-tuple
      - step4_build_stage8_5_asins: 全部 ASIN × all commenters × top-5 non-numeric attrs
        (MAX_ATTRS_FOR_LLM=5, 对齐 Stage 1 N_INPUT=5)
  - Phase 2 (Stage 3): per-user Gaussian fitting
      - PCA48 + Welford online + shrinkage
      - writes user_gaussians.json (1.5GB after no-cohort, Rule 10 → scratch2/)
      - 下游 Stage 4 strict alignment / Stage 5 multi-retriever 均依赖
        mu + sigma_diag 做 Mahalanobis 距离(Phase 35 series)

Idempotent: 每 phase 检查输出文件 + signature, 命中 cache 直接跳过。

参数全部硬编码 (Rule 3), 不接受 CLI 参数.

Running order:
  python attribute_extraction/extract_product_attrs.py   # Step 1
  python gaussian/build_user.py                          # Phase 1 + Phase 2
"""
from __future__ import annotations

import collections
import gc
import gzip
import hashlib
import json
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

# 让 build_user.py 能 import 兄弟目录 attribute_extraction/extract_product_attrs
sys.path.insert(0, str(REPO_ROOT))
from attribute_extraction.extract_product_attrs import select_top_attrs  # noqa: E402

# 让 build_user.py 能 import 兄弟目录 common/syntax_subspace_utils
sys.path.insert(0, str(REPO_ROOT / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_OUT, LAMBDA,
    MAX_R_FLOOR, MIN_INLIER_FRAC, MIN_N_QUALITY_USERS_PER_ASIN,
    MIN_REVIEWS_FOR_PER_USER, MIN_SIGMA_MEAN,
    PCA_DIM, R_95, REVIEW_GZ, VAR_EPS, log, feat_key,
)

# === Inputs ===
DATA = Path("/fs04/ar57/wenyu/PersoanlQuery/data")
REVIEWS_GZ = DATA / "Baby_Products_2023.jsonl.gz"

# === Outputs ===
STAGE8_5_ASINS_JSON = SCRATCH / "stage8_5_asins.json"
# 用户指令 2026-08-29: Section 6 Welford streaming (~13 min) 输出缓存。
# 当 MIN_REVIEWS_FOR_PER_USER / MIN_SIGMA_MEAN / LAMBDA / PCA_DIM 不变,只重跑
# Section 7 即可。下次 Phase 2 跳过 streaming 节省 ~13 min。
WELFORD_CHECKPOINT = SCRATCH / "welford_checkpoint.npz"
WELFORD_SIG_JSON = SCRATCH / "welford_sig.json"

# === Phase 1 — cohort construction 参数 ===
# 用户指令 2026-08-29: 拆掉 TOP_N_ASINS=10_000 上限, 改为全集 296K ASINs,
# 但加 MIN_USERS_PER_ASIN=2 下限(至少 2 个用户评论, 后续 Phase 2 Gaussian 质量过滤后再收紧)。
# "Gaussian 质量" 的判定在 Phase 2 末尾 (基于 mean(sigma_diag) ≥ MIN_SIGMA_MEAN, 不是评论数),
# 在 step_post_filter_cohort_by_gaussian_quality() 里重新过滤 ASIN。
TOP_N_ASINS = None  # None = include all ASINs (Baby Products 296K)
MIN_USERS_PER_ASIN = 2  # 评论者门槛 (Phase 1 粗筛, Gaussian 质量门槛在 Phase 2 后重判)
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
        "MIN_USERS_PER_ASIN": MIN_USERS_PER_ASIN,
        "MAX_ATTRS_FOR_LLM": MAX_ATTRS_FOR_LLM,
        "MIN_ATTRS_REQUIRED": MAX_ATTRS_FOR_LLM,  # 用户指令 2026-08-29: 严格 <5 attrs 过滤
        "REVIEWS_GZ": str(REVIEWS_GZ),
        "REVIEWS_GZ_mtime": REVIEWS_GZ.stat().st_mtime if REVIEWS_GZ.exists() else 0,
        "PRODUCT_ATTRS_JSON": "result/product_attributes.json",
        "PRODUCT_ATTRS_JSON_mtime": (REPO_ROOT / "result/product_attributes.json").stat().st_mtime
            if (REPO_ROOT / "result/product_attributes.json").exists() else 0,
        # 用户指令 2026-08-29: 非语义属性过滤 (hash of the list to capture change)
        "NON_SEMANTIC_KEYWORDS": sorted([
            "main category", "department",
            "country", "country of origin", "country/region of origin",
            "number of items", "number of pieces", "unit count",
            "date", "date listed", "best sellers rank",
        ]),
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

    # 用户指令 2026-08-29: Phase 1 cohort 仍按 raw 评论者数 (≥MIN_USERS_PER_ASIN=2) 粗筛,
    # Gaussian 质量门槛 (mean sigma_diag ≥ MIN_SIGMA_MEAN) 在 Phase 2 末尾
    # step_post_filter_cohort_by_gaussian_quality() 重新判定。
    eligible_count: dict[str, int] = {}
    for asin, uset in asin_users.items():
        c = len([u for u in uset if u in eligible_users_set])
        eligible_count[asin] = c
    log(f"    ASINs with ≥1 commenters: {len(eligible_count)}")

    ranked = sorted(eligible_count.items(), key=lambda kv: kv[1], reverse=True)
    # Apply TOP_N_ASINS upper limit (None = no upper limit)
    if TOP_N_ASINS is not None:
        ranked = ranked[:TOP_N_ASINS]
    # Apply MIN_USERS_PER_ASIN lower limit (用户指令 2026-08-29)
    ranked = [(a, c) for a, c in ranked if c >= MIN_USERS_PER_ASIN]
    counts = [c for _, c in ranked]
    label = f"all (TOP_N_ASINS={TOP_N_ASINS})" if TOP_N_ASINS is None else f"top {TOP_N_ASINS}"
    if counts:
        log(f"    {label} ASINs (≥{MIN_USERS_PER_ASIN} users): n={len(ranked)}, "
            f"min={min(counts)}, median={sorted(counts)[len(counts)//2]}, max={max(counts)}")

    asins_out: list[dict] = []
    skipped_no_attrs = 0
    for asin, _ in ranked:
        adoc = product_attrs.get(asin) or {}
        attrs_used = select_top_attrs(adoc, max_n=MAX_ATTRS_FOR_LLM)
        # 用户指令 2026-08-29: 严格 attrs filter (<5 直接过滤, 之前 <1 太宽松)
        if len(attrs_used) < MAX_ATTRS_FOR_LLM:
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
            "filter": {
                "note": (f"no user filter (all commenters enter cohort), "
                         f"TOP_N_ASINS={TOP_N_ASINS}, MIN_USERS_PER_ASIN={MIN_USERS_PER_ASIN}")
            },
        })

    log(f"    final ASINs: {len(asins_out)}, skipped (no attrs): {skipped_no_attrs}")

    config = {
        "description": (f"{label} ASINs (≥{MIN_USERS_PER_ASIN} users) × all commenters "
                        f"(no user filter, "
                        f"max {MAX_ATTRS_FOR_LLM} non-numeric attrs/ASIN, "
                        f"no per-ASIN user cap, no min review count)"),
        "TOP_N_ASINS": TOP_N_ASINS,
        "MIN_USERS_PER_ASIN": MIN_USERS_PER_ASIN,
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
# Phase 2 — Stage 3 user Gaussians
# ============================================================================
def phase2_user_gaussians() -> None:
    """Stage 3: per-user Gaussian fitting (PCA48 + Welford online + shrinkage).

    用户指令 2026-08-29: 重构为 per-user 持久化 cache, keyed by user_id。
    关键改动:
      - sig 不依赖 cohort (ASINS_IN/REVIEW_GZ), 只依赖 Gaussian 计算参数 +
        FEAT_CACHE。
      - 现有 user_gaussians.json 视为 cached_users, 加载 scaler/PCA/fnames。
      - new_users = target_users - cached_users, 仅对这些新用户跑 Welford + inlier。
      - merge: user_gaussians = cached ∪ new_passed_gate。
      - Section 9 用合并后的 user_gaussians 重新过滤 cohort (MIN_N_QUALITY_USERS_PER_ASIN)。

    收益: cohort 调整 (e.g. ≥2 quality users) 不强制重算 1.25M users' Gaussian,
    只算 ~40K 新增 users (~30 sec)。
    """
    # --- 1. Sig (cohort-independent) ---
    # 用户指令 2026-08-29: 不依赖 FEAT_CACHE_mtime — cached users 的 Gaussian 由
    # 其 user_id 锁定 (reviews 不可变), 新的 FEAT_CACHE entries 仅来自新用户, 不影响
    # cached Gaussians 的有效性。Phase 1 cohort (ASINS_IN/REVIEW_GZ) 也不入 sig — per-user
    # Gaussian 本身与 cohort 无关, 只有哪些 users 在 cohort 里影响 target_users。
    sig_payload = json.dumps({
        "PCA_DIM": PCA_DIM,
        "LAMBDA": LAMBDA,
        "VAR_EPS": VAR_EPS,
        "MIN_REVIEWS_FOR_PER_USER": MIN_REVIEWS_FOR_PER_USER,
        "MIN_SIGMA_MEAN": MIN_SIGMA_MEAN,
        "MAX_R_FLOOR": MAX_R_FLOOR,
        "R_95": R_95,
        "MIN_INLIER_FRAC": MIN_INLIER_FRAC,
        "FEAT_CACHE": str(FEAT_CACHE),
    }, sort_keys=True)
    sig_hash = hashlib.sha1(sig_payload.encode("utf-8")).hexdigest()[:12]
    log(f"[build_user] Phase 2 signature: {sig_hash} (cohort-independent)")

    # --- 2. Load existing user_gaussians.json as per-user cache ---
    # 用户指令 2026-08-29: cache 有效性按算法参数 (PCA_DIM/LAMBDA/VAR_EPS/3-class thresholds/
    # FEAT_CACHE path) 判定, 不依赖 mtime/cohort deps。Cached 用户 Gaussian 由 user_id
    # 锁定, 与 cohort 或 FEAT_CACHE_mtime 无关。当算法参数全一致时, 即使 sig 字符串格式
    # 不同 (旧 sig 包含 mtime + MIN_N_QUALITY_USERS_PER_ASIN) 也视为 cache hit。
    cached_users: dict = {}
    cached_scaler = cached_pca = None  # sklearn-like objects reconstructed below
    cached_scaler_mean = cached_scaler_scale = None
    cached_pca_components = cached_pca_ev = cached_pca_evr = cached_pca_mean = None
    cached_all_fnames = cached_fnames_f3 = None
    ALG_PARAM_KEYS = (
        "PCA_DIM", "LAMBDA", "VAR_EPS", "MIN_REVIEWS_FOR_PER_USER",
        "MIN_SIGMA_MEAN", "MAX_R_FLOOR", "R_95", "MIN_INLIER_FRAC",
    )
    if GAUSSIANS_OUT.exists():
        try:
            cached_doc = json.load(open(GAUSSIANS_OUT))
            cached_cfg = cached_doc.get("config", {})
            cached_sig = cached_cfg.get("cache_signature")
            # 优先按 sig 匹配 (相同算法 + 相同 sig payload 格式); 否则 fallback 到算法参数逐项匹配
            alg_match = all(cached_cfg.get(k) == globals()[k] for k in ALG_PARAM_KEYS)
            cache_hit = (cached_sig == sig_hash) or alg_match
            if cache_hit:
                cached_users = cached_doc.get("users", {})
                cached_scaler_mean = cached_doc["scaler_mean"]
                cached_scaler_scale = cached_doc["scaler_scale"]
                cached_pca_components = cached_doc["pca_components"]
                cached_pca_ev = cached_doc["pca_explained_variance"]
                cached_pca_evr = cached_doc["pca_explained_variance_ratio"]
                cached_pca_mean = cached_doc["pca_mean"]
                cached_all_fnames = cached_doc.get("feature_names_ordered", [])
                cached_fnames_f3 = cached_doc.get("fnames_f3", [])
                hit_reason = ("sig exact match" if cached_sig == sig_hash
                              else "alg params match (sig format changed)")
                log(f"[build_user] Gaussian cache HIT ({hit_reason}): "
                    f"{len(cached_users)} users, sig={cached_sig}->{sig_hash}")
            else:
                log(f"  Gaussian cache algorithm mismatch, will rebuild full Welford")
        except Exception as e:
            log(f"  Gaussian cache load failed ({e!r})")
    else:
        log(f"  no Gaussian cache at {GAUSSIANS_OUT}, building from scratch")

    log("[build_user] === Phase 2: per-user Gaussian (Stage 3) ===")

    # --- 3. Load cohort target users ---
    log("\n=== 1. Loading target users from cohort ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    target_users = set()
    user_to_asins = collections.defaultdict(set)
    for a in asin_data:
        for uid in a["users_sampled"]:
            target_users.add(uid)
            user_to_asins[uid].add(a["asin"])
    log(f"  target users (cohort): {len(target_users)}")

    # --- 4. Identify new users (not in cache) ---
    cached_user_ids = set(cached_users.keys())
    new_users_set = target_users - cached_user_ids
    log(f"  cached users: {len(cached_user_ids)}, new users: {len(new_users_set)}")

    # 用户指令 2026-08-29: capture original cohort target users count for n_users_skipped reporting
    # (target_users 在 else 分支被 reassigned 为 new_users_set, 原值需要保留)。
    n_users_target_total = len(target_users)

    if not new_users_set and cached_scaler_mean is not None:
        # Full cache hit — no new computation needed
        log("  full cache hit: all target users already in cache, skipping Welford/inlier")
        user_gaussians = dict(cached_users)
        ss = pca = None  # markers for "use cached"
        all_fnames = cached_all_fnames
        fnames_sub = cached_fnames_f3
    else:
        # Compute Gaussian for new users only (restricting target_users)
        log(f"  restricting Welford to {len(new_users_set)} new users")
        target_users = new_users_set  # restrict scope
        (user_gaussians_new, ss, pca, all_fnames,
         fnames_sub) = _welford_for_user_subset(
            target_users, sig_hash,
            cached_scaler_mean, cached_scaler_scale,
            cached_pca_components, cached_pca_ev, cached_pca_evr, cached_pca_mean,
            cached_all_fnames, cached_fnames_f3,
        )
        # Merge: cached (already passed gate) + new (passed gate this run)
        user_gaussians = dict(cached_users)
        user_gaussians.update(user_gaussians_new)
        log(f"  merged: {len(user_gaussians)} total users "
            f"({len(cached_users)} cached + {len(user_gaussians_new)} new passed gate)")

    log("\n=== 8. Saving ===")
    GAUSSIANS_OUT.parent.mkdir(parents=True, exist_ok=True)

    # 用户指令 2026-08-29: 当 ss is None 时 (full cache hit), 用 cached scaler/PCA/fnames
    if ss is not None:
        scaler_mean_out = ss.mean_.tolist()
        scaler_scale_out = ss.scale_.tolist()
        pca_components_out = pca.components_.tolist()
        pca_ev_out = pca.explained_variance_.tolist()
        pca_evr_out = pca.explained_variance_ratio_.tolist()
        pca_mean_out = pca.mean_.tolist()
        fnames_f3_out = fnames_sub
        all_fnames_out = all_fnames
    else:
        scaler_mean_out = cached_scaler_mean
        scaler_scale_out = cached_scaler_scale
        pca_components_out = cached_pca_components
        pca_ev_out = cached_pca_ev
        pca_evr_out = cached_pca_evr
        pca_mean_out = cached_pca_mean
        fnames_f3_out = cached_fnames_f3
        all_fnames_out = cached_all_fnames

    EXCL_PREFIXES = ("open_", "close_", "posbg_", "postg_", "depbg_")
    EXCL_EXACT = ("opener", "stype", "has_passive", "is_interrog", "has_cond",
                  "acl", "advcl", "ccomp", "xcomp", "relcl")

    with open(GAUSSIANS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 8.5: per-user Gaussian + StandardScaler + PCA48 from "
                                "Baby_Products review corpus, F3_CoreStruct spaCy features "
                                "(Welford online accumulation). 用户指令 2026-08-29: 改为 "
                                "per-user 持久化 cache (keyed by user_id), cohort 调整不强制 "
                                "重算 cached users 的 Gaussian。"),
                "PCA_DIM": PCA_DIM,
                "LAMBDA": LAMBDA,
                "VAR_EPS": VAR_EPS,
                "MIN_REVIEWS_FOR_PER_USER": MIN_REVIEWS_FOR_PER_USER,
                "MIN_SIGMA_MEAN": MIN_SIGMA_MEAN,
                "MAX_R_FLOOR": MAX_R_FLOOR,
                "R_95": R_95,
                "MIN_INLIER_FRAC": MIN_INLIER_FRAC,
                "F3_EXCLUDE_PREFIXES": list(EXCL_PREFIXES),
                "F3_EXCLUDE_EXACT": list(EXCL_EXACT),
                "cache_signature": sig_hash,
            },
            "scaler_mean": scaler_mean_out,
            "scaler_scale": scaler_scale_out,
            "pca_components": pca_components_out,
            "pca_explained_variance": pca_ev_out,
            "pca_explained_variance_ratio": pca_evr_out,
            "pca_mean": pca_mean_out,
            "feature_names_ordered": all_fnames_out,
            "fnames_f3": fnames_f3_out,
            "users": user_gaussians,
            "n_users_with_gaussian": len(user_gaussians),
            "n_users_skipped": max(0, n_users_target_total - len(user_gaussians)),
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {GAUSSIANS_OUT}")

    # 用户指令 2026-08-29: 用合并后的 user_gaussians 重新过滤 cohort
    step_post_filter_cohort_by_gaussian_quality(user_gaussians)


def step_post_filter_cohort_by_gaussian_quality(user_gaussians: dict) -> None:
    """用户指令 2026-08-29: Phase 1 按 raw 评论者数粗筛, 此步骤用真实 Gaussian 质量
    (3-class gate: Q1 σ_mean ≥ MIN_SIGMA_MEAN AND Q2 r_floor ≤ MAX_R_FLOOR AND
     Q3 inlier_frac ≥ MIN_INLIER_FRAC) 重判 cohort。要求每个 ASIN 有
    ≥MIN_N_QUALITY_USERS_PER_ASIN 个 Gaussian 质量达标的用户。

    流程: 读 stage8_5_asins.json → 查 user_gaussians keys → 计数 → 过滤 → 重写文件。
    """
    log("\n=== 9. Re-filtering cohort by 3-class Gaussian quality ===")
    quality_uids = set(user_gaussians.keys())  # 已被 3-class gate 过滤
    log(f"  users with quality Gaussian (3-class gate): {len(quality_uids)}")

    with open(STAGE8_5_ASINS_JSON, "r", encoding="utf-8") as f:
        asins_doc = json.load(f)

    old_asins = asins_doc.get("asins", [])
    log(f"  cohort before filter: {len(old_asins)} ASINs")

    new_asins = []
    n_kept = 0
    n_dropped_no_quality = 0
    n_dropped_lt_min_quality = 0
    for entry in old_asins:
        users_sampled = entry.get("users_sampled", [])
        n_quality = sum(1 for u in users_sampled if u in quality_uids)
        entry["n_users_quality"] = n_quality
        if n_quality == 0:
            n_dropped_no_quality += 1
            continue
        if n_quality < MIN_N_QUALITY_USERS_PER_ASIN:
            n_dropped_lt_min_quality += 1
            continue
        new_asins.append(entry)
        n_kept += 1

    log(f"  dropped (no quality users): {n_dropped_no_quality}")
    log(f"  dropped (<{MIN_N_QUALITY_USERS_PER_ASIN} quality users): {n_dropped_lt_min_quality}")
    log(f"  final cohort: {n_kept} ASINs (after Gaussian quality filter)")

    asins_doc["asins"] = new_asins
    asins_doc["n_asins"] = len(new_asins)
    asins_doc["config"]["description"] = (
        f"{len(new_asins)} ASINs with ≥{MIN_N_QUALITY_USERS_PER_ASIN} 3-class-quality users "
        f"(Q1 σ_mean ≥ {MIN_SIGMA_MEAN}, Q2 r_floor ≤ {MAX_R_FLOOR}, "
        f"Q3 inlier_frac ≥ {MIN_INLIER_FRAC})"
    )
    asins_doc["config"]["MIN_N_QUALITY_USERS_PER_ASIN"] = MIN_N_QUALITY_USERS_PER_ASIN
    asins_doc["config"]["MIN_SIGMA_MEAN"] = MIN_SIGMA_MEAN
    asins_doc["config"]["MAX_R_FLOOR"] = MAX_R_FLOOR
    asins_doc["config"]["MIN_INLIER_FRAC"] = MIN_INLIER_FRAC

    with open(STAGE8_5_ASINS_JSON, "w", encoding="utf-8") as f:
        json.dump(asins_doc, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {STAGE8_5_ASINS_JSON}")


# ============================================================================
# 用户指令 2026-08-29: Welford + inlier + 3-class gate for user subset
# ============================================================================
def _welford_for_user_subset(target_users, sig_hash,
                              cached_scaler_mean, cached_scaler_scale,
                              cached_pca_components, cached_pca_ev,
                              cached_pca_evr, cached_pca_mean,
                              cached_all_fnames, cached_fnames_f3):
    """Welford + inlier + 3-class gate for a subset of users.

    用户指令 2026-08-29: 重构 cache 后, 新用户 Gaussian 计算走这个函数。
    当 cached_scaler_mean 不为 None, 重用 user_gaussians.json 里的 scaler/PCA arrays
    (与 cached entries 在同一特征空间, 可直接对比)。当为 None 时 fit fresh, 并返回
    ss/pca 让 caller 写回 JSON。

    Returns: (user_gaussians_new, ss_or_None, pca_or_None, all_fnames, fnames_sub, counts)
    """
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.decomposition import PCA as _PCA

    log(f"\n=== Welford for user subset ({len(target_users)} users) ===")

    # --- 1. Load reviews for target_users only (avoid full Baby_Products scan cost) ---
    target_users_set = set(target_users)
    user_review_texts: dict = collections.defaultdict(list)
    n_records = 0
    for line in gzip.open(REVIEW_GZ, "rt", encoding="utf-8"):
        r = json.loads(line)
        uid = r.get("reviewerID") or r.get("user_id")
        if uid in target_users_set and r.get("text"):
            user_review_texts[uid].append(r["text"])
        n_records += 1
        if n_records % 2_000_000 == 0:
            log(f"    {n_records/1e6:.1f}M records, {len(user_review_texts)} users found")
    log(f"  target users with reviews: {len(user_review_texts)} / {len(target_users)}")

    # --- 2. Load FEAT_CACHE fnames (lazy sample 1000 entries to avoid 10GB set overhead) ---
    seen_keys: set = set()
    all_fnames_set: set = set()
    FNAMES_SAMPLE_N = 1000
    n_fnames_samples = 0
    if FEAT_CACHE.exists():
        try:
            with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    seen_keys.add(rec["k"])
                    if n_fnames_samples < FNAMES_SAMPLE_N:
                        all_fnames_set.update(rec["v"].keys())
                        n_fnames_samples += 1
        except (EOFError, gzip.BadGzipFile, zlib.error) as e:
            log(f"  WARN cache gzip stream truncated ({type(e).__name__}: {e!r})")
    all_fnames = sorted(all_fnames_set)
    log(f"  FEAT_CACHE fnames: {len(all_fnames)} (sampled from {n_fnames_samples} entries)")

    # --- 3. F3_CoreStruct subset (drop n-gram prefixes + semantic tags) ---
    EXCL_PREFIXES = ("open_", "close_", "posbg_", "postg_", "depbg_")
    EXCL_EXACT = ("opener", "stype", "has_passive", "is_interrog", "has_cond",
                  "acl", "advcl", "ccomp", "xcomp", "relcl")
    fnames_sub = [n for n in all_fnames
                  if not any(n.startswith(p) for p in EXCL_PREFIXES)
                  and n not in EXCL_EXACT]
    col_idx = [all_fnames.index(n) for n in fnames_sub]
    log(f"  F3_CoreStruct: {len(fnames_sub)} / {len(all_fnames)} features")

    # --- 4. Build sha1_to_uid_idx for target users (only) ---
    target_user_list = sorted(target_users)
    uid_to_idx = {u: i for i, u in enumerate(target_user_list)}
    n_users = len(target_user_list)
    user_n_reviews: dict = {}
    user_n_words: dict = {}
    sha1_to_uid_idx: dict = collections.defaultdict(list)
    for uid, texts in user_review_texts.items():
        user_n_reviews[uid] = len(texts)
        user_n_words[uid] = sum(len(t.split()) for t in texts)
        uid_idx = uid_to_idx.get(uid)
        if uid_idx is None:
            continue
        for t in texts:
            t = t.replace("\n", " ").strip()
            if not t:
                continue
            sha1_to_uid_idx[feat_key(t)].append(uid_idx)
    target_shas = set(sha1_to_uid_idx.keys())
    log(f"  sha1_to_uid_idx: {len(sha1_to_uid_idx)} unique keys, {n_users} users")
    del user_review_texts
    gc.collect()

    # --- 5. Reconstruct or fit scaler/PCA ---
    ss = None
    pca = None
    if cached_scaler_mean is not None:
        # Reuse cached sklearn objects (manual attribute injection; .transform() works).
        log("  reusing cached scaler/PCA from user_gaussians.json (sig matched)")
        ss = _SS()
        ss.mean_ = np.asarray(cached_scaler_mean, dtype=np.float64)
        ss.scale_ = np.asarray(cached_scaler_scale, dtype=np.float64)
        ss.var_ = ss.scale_ ** 2
        ss.n_features_in_ = len(ss.mean_)

        pca = _PCA(n_components=PCA_DIM)
        pca.components_ = np.asarray(cached_pca_components, dtype=np.float64)
        pca.explained_variance_ = np.asarray(cached_pca_ev, dtype=np.float64)
        pca.explained_variance_ratio_ = np.asarray(cached_pca_evr, dtype=np.float64)
        pca.mean_ = np.asarray(cached_pca_mean, dtype=np.float64)
        pca.n_components_ = PCA_DIM
        pca.n_features_in_ = len(pca.mean_)
    else:
        # Fit fresh scaler/PCA on FEAT_CACHE target-user entries (5K cap for speed).
        log("  no cached scaler/PCA, fitting fresh on target users' FEAT_CACHE")
        fit_records: list = []
        N_FIT_MAX = 5000
        with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec["k"] in target_shas:
                    fit_records.append(rec)
                    if len(fit_records) >= N_FIT_MAX:
                        break
        X_fit_full = np.array(
            [[r["v"].get(n, 0.0) for n in all_fnames] for r in fit_records],
            dtype=np.float64,
        )
        X_fit_sub = X_fit_full[:, col_idx]
        ss = _SS()
        ss.fit(X_fit_sub)
        X_fit_sub_scaled = ss.transform(X_fit_sub)
        pca = _PCA(n_components=PCA_DIM, random_state=42)
        pca.fit(X_fit_sub_scaled)
        log(f"  StandardScaler fitted on {len(fit_records)} sentences "
            f"(F3 {len(fnames_sub)}d → PCA{PCA_DIM}, "
            f"EV={pca.explained_variance_ratio_.sum():.4f})")
        del X_fit_full, X_fit_sub, X_fit_sub_scaled, fit_records
        gc.collect()

    # --- 6. Welford streaming on FEAT_CACHE for target users only ---
    counts = np.zeros(n_users, dtype=np.int64)
    mean_acc = np.zeros((n_users, PCA_DIM), dtype=np.float64)
    M2_acc = np.zeros((n_users, PCA_DIM), dtype=np.float64)
    n_seen = 0
    log("  streaming FEAT_CACHE for Welford accumulation...")
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = rec["k"]
            v = rec["v"]
            uid_idxs = sha1_to_uid_idx.get(k)
            if uid_idxs is None:
                continue
            n_seen += 1
            vec_full = np.array([v.get(n, 0.0) for n in all_fnames], dtype=np.float64)
            vec_sub = vec_full[col_idx]
            vec_sub_scaled = ss.transform(vec_sub[None, :])[0]
            z = pca.transform(vec_sub_scaled[None, :])[0].astype(np.float64)
            for ui in uid_idxs:
                counts[ui] += 1
                delta = z - mean_acc[ui]
                mean_acc[ui] = mean_acc[ui] + delta / counts[ui]
                delta2 = z - mean_acc[ui]
                M2_acc[ui] = M2_acc[ui] + delta * delta2
            if n_seen % 50_000 == 0:
                gc.collect()
                n_with_rev = int((counts > 0).sum())
                log(f"    streaming: {n_seen} matched, "
                    f"users_with_reviews={n_with_rev}/{n_users}")
    log(f"  streaming done: {n_seen} matched, "
        f"users_with_reviews={int((counts > 0).sum())}/{n_users}")

    # --- 7. Compute mu / sigma_diag (shrinkage) ---
    valid_mask = counts >= MIN_REVIEWS_FOR_PER_USER
    n_valid = int(valid_mask.sum())
    safe_counts = np.maximum(counts, 1).astype(np.float64)
    var_pop = M2_acc / safe_counts[:, None]
    var_per_user_mean = var_pop.mean(axis=1, keepdims=True)
    var_shrink = (1 - LAMBDA) * var_pop + LAMBDA * var_per_user_mean
    sigma_diag_arr = np.maximum(var_shrink, VAR_EPS)
    sigma_diag_mean_per_user = sigma_diag_arr.mean(axis=1)
    mu_arr = mean_acc / safe_counts[:, None]
    r_floor_per_user = (sigma_diag_arr <= VAR_EPS).mean(axis=1)

    # --- 8. Inlier check (Q3): self-consistency d²(z, G_u) ≤ R_95² ---
    R_95_SQ = R_95 ** 2
    inlier_count = np.zeros(n_users, dtype=np.int64)
    seen_count = np.zeros(n_users, dtype=np.int64)
    log(f"  inlier check (R_95²={R_95_SQ:.2f})...")
    n_inlier_seen = 0
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = rec["k"]
            v = rec["v"]
            uid_idxs = sha1_to_uid_idx.get(k)
            if uid_idxs is None:
                continue
            n_inlier_seen += 1
            vec_full = np.array([v.get(n, 0.0) for n in all_fnames], dtype=np.float64)
            vec_sub = vec_full[col_idx]
            vec_sub_scaled = ss.transform(vec_sub[None, :])[0]
            z = pca.transform(vec_sub_scaled[None, :])[0].astype(np.float64)
            for ui in uid_idxs:
                if not valid_mask[ui]:
                    continue
                seen_count[ui] += 1
                mu_u = mu_arr[ui]
                sigma_u = sigma_diag_arr[ui]
                d_sq = np.sum((z - mu_u) ** 2 / sigma_u)
                if d_sq <= R_95_SQ:
                    inlier_count[ui] += 1
            if n_inlier_seen % 100_000 == 0:
                log(f"    inlier: {n_inlier_seen} seen, "
                    f"users_seen={int((seen_count > 0).sum())}")
    inlier_frac = np.where(seen_count > 0, inlier_count / seen_count, 0.0)
    log(f"  inlier check done: {n_inlier_seen} sentences seen")
    if (seen_count > 0).any():
        valid_inlier = inlier_frac[seen_count > 0]
        log(f"  inlier_frac: min={valid_inlier.min():.3f}, "
            f"median={np.median(valid_inlier):.3f}, "
            f"max={valid_inlier.max():.3f}")

    # --- 9. 3-class quality gate (Q1 σ_mean / Q2 r_floor / Q3 inlier_frac) ---
    q1_mask = sigma_diag_mean_per_user >= MIN_SIGMA_MEAN
    q2_mask = r_floor_per_user <= MAX_R_FLOOR
    q3_mask = inlier_frac >= MIN_INLIER_FRAC
    quality_mask = q1_mask & q2_mask & q3_mask & valid_mask
    n_quality = int(quality_mask.sum())
    n_q1 = int((q1_mask & valid_mask).sum())
    n_q2 = int((q2_mask & valid_mask).sum())
    n_q3 = int((q3_mask & valid_mask).sum())
    log(f"\n  3-class quality gate:")
    log(f"    Q1 σ_mean≥{MIN_SIGMA_MEAN}: {n_q1}/{n_valid} pass "
        f"({n_q1/max(1,n_valid)*100:.1f}%)")
    log(f"    Q2 r_floor≤{MAX_R_FLOOR}: {n_q2}/{n_valid} pass "
        f"({n_q2/max(1,n_valid)*100:.1f}%)")
    log(f"    Q3 inlier_frac≥{MIN_INLIER_FRAC}: {n_q3}/{n_valid} pass "
        f"({n_q3/max(1,n_valid)*100:.1f}%)")
    log(f"    ALL Q1&Q2&Q3: {n_quality}/{n_valid} pass "
        f"({n_quality/max(1,n_valid)*100:.1f}%)")

    user_gaussians_new: dict = {}
    for uid_idx, uid in enumerate(target_user_list):
        if not quality_mask[uid_idx]:
            continue
        mu = mu_arr[uid_idx]
        sd = sigma_diag_arr[uid_idx]
        user_gaussians_new[uid] = {
            "mu": mu.tolist(),
            "sigma_diag": sd.tolist(),
            "n_reviews": user_n_reviews.get(uid, 0),
            "n_words": user_n_words.get(uid, 0),
            "n_sentences": int(counts[uid_idx]),
            "sigma_mean": float(sigma_diag_mean_per_user[uid_idx]),
            "r_floor": float(r_floor_per_user[uid_idx]),
            "inlier_frac": float(inlier_frac[uid_idx]),
            "source": "per_user",
        }
    log(f"  new users passing 3-class gate: {len(user_gaussians_new)}")

    # Free intermediates
    del mean_acc, M2_acc, var_pop, var_shrink, sigma_diag_arr
    del mu_arr, inlier_count, seen_count, inlier_frac, sha1_to_uid_idx
    gc.collect()

    # Return ss/pca as markers — None when reusing cached (caller will use cached_*)
    if cached_scaler_mean is not None:
        return (user_gaussians_new, None, None, all_fnames, fnames_sub)
    else:
        return (user_gaussians_new, ss, pca, all_fnames, fnames_sub)


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