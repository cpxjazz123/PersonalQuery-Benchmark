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
    """Stage 3: per-user Gaussian fitting (PCA48 + Welford online + shrinkage)."""
    # --- Cache detection ---
    sig_payload = json.dumps({
        "PCA_DIM": PCA_DIM,
        "LAMBDA": LAMBDA,
        "VAR_EPS": VAR_EPS,
        "MIN_REVIEWS_FOR_PER_USER": MIN_REVIEWS_FOR_PER_USER,
        # 用户指令 2026-08-29: 3-class Gaussian quality gate (不再用 n_sentences 过滤)
        "MIN_SIGMA_MEAN": MIN_SIGMA_MEAN,
        "MAX_R_FLOOR": MAX_R_FLOOR,
        "R_95": R_95,
        "MIN_INLIER_FRAC": MIN_INLIER_FRAC,
        "MIN_N_QUALITY_USERS_PER_ASIN": MIN_N_QUALITY_USERS_PER_ASIN,
        "ASINS_IN": str(ASINS_IN),
        "ASINS_IN_mtime": ASINS_IN.stat().st_mtime if ASINS_IN.exists() else 0,
        "FEAT_CACHE": str(FEAT_CACHE),
        "FEAT_CACHE_mtime": FEAT_CACHE.stat().st_mtime if FEAT_CACHE.exists() else 0,
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

    log("[build_user] === Phase 2: per-user Gaussian (Stage 3) ===")

    log("\n=== 1. Loading target users ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    target_users = set()
    user_to_asins = collections.defaultdict(set)
    for a in asin_data:
        for uid in a["users_sampled"]:
            target_users.add(uid)
            user_to_asins[uid].add(a["asin"])
    log(f"  target users: {len(target_users)}")

    log("\n=== 3. Scanning review corpus ===")
    user_review_texts = collections.defaultdict(list)
    n_records = 0
    for line in gzip.open(REVIEW_GZ, "rt", encoding="utf-8"):
        r = json.loads(line)
        uid = r.get("reviewerID") or r.get("user_id")
        if uid in target_users and r.get("text"):
            user_review_texts[uid].append(r["text"])
        n_records += 1
        if n_records % 2_000_000 == 0:
            log(f"    {n_records/1e6:.1f}M records, {len(user_review_texts)} users found")
    log(f"  total records: {n_records}")
    log(f"  users with reviews: {len(user_review_texts)}")

    review_counts = [len(v) for v in user_review_texts.values()]
    if review_counts:
        log(f"  review count: min={min(review_counts)}, "
            f"mean={sum(review_counts)/len(review_counts):.1f}, "
            f"max={max(review_counts)}")
    n_high = sum(1 for c in review_counts if c >= MIN_REVIEWS_FOR_PER_USER)
    log(f"  users with >={MIN_REVIEWS_FOR_PER_USER} reviews (per_user Gaussian): {n_high}")
    n_low = sum(1 for c in review_counts if c < MIN_REVIEWS_FOR_PER_USER and c >= 1)
    log(f"  users with 1-2 reviews: {n_low}")
    n_zero = len(target_users) - len(user_review_texts)
    log(f"  users with 0 reviews: {n_zero}")

    log("\n=== 4. Loading feature cache (lazy: only seen_keys) ===")
    seen_keys = set()
    all_fnames_set: set = set()
    # Sample fnames from first 1000 entries only — per_sentence_features_v2
    # produces consistent 182d keys, so sampling avoids 1.68M-entry full pass
    # (~10GB set overhead → OOM). 1000 entries is plenty for canonical keys.
    FNAMES_SAMPLE_N = 1000
    n_fnames_samples = 0
    if FEAT_CACHE.exists():
        n_load_errors = 0
        try:
            with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        n_load_errors += 1
                        continue
                    seen_keys.add(rec["k"])
                    if n_fnames_samples < FNAMES_SAMPLE_N:
                        all_fnames_set.update(rec["v"].keys())
                        n_fnames_samples += 1
        except (EOFError, gzip.BadGzipFile, zlib.error) as e:
            log(f"  WARN cache gzip stream truncated ({type(e).__name__}: {e!r}), "
                f"loaded {len(seen_keys)} keys + {n_load_errors} corrupt lines")
    log(f"  cache keys loaded: {len(seen_keys)} (lazy mode)")
    all_fnames = sorted(all_fnames_set)
    log(f"  canonical fnames (from first {n_fnames_samples} entries): {len(all_fnames)}")

    log("\n=== 5. Extracting spaCy features for user review texts ===")
    all_sents = []
    sent_to_user = []
    for uid, texts in user_review_texts.items():
        for t in texts:
            t = t.replace("\n", " ").strip()
            if t:
                all_sents.append(t)
                sent_to_user.append(uid)
    log(f"  total reviews (no length threshold, no sentence split): {len(all_sents)}")

    new_sents = [s for s in all_sents if feat_key(s) not in seen_keys]
    log(f"  unique sentences: {len(set(all_sents))}, new to extract: {len(new_sents)}")

    feat_map = {}
    if new_sents:
        import spacy
        from syntactic_features import per_sentence_features_v2
        nlp = spacy.load("en_core_web_sm")
        for comp in ("ner", "lemmatizer", "attribute_ruler"):
            if comp in nlp.pipe_names:
                nlp.disable_pipe(comp)
        N_PROCESS = 3
        BATCH_SIZE = 256
        CHUNK_FLUSH = 25_000
        log(f"  extracting features for {len(new_sents)} new sentences "
            f"(n_process={N_PROCESS}, batch={BATCH_SIZE}, "
            f"chunked append every {CHUNK_FLUSH})...")
        new_unique = sorted(set(new_sents))
        if not FEAT_CACHE.exists():
            with gzip.open(FEAT_CACHE, "wt", encoding="utf-8") as f:
                f.write("# user-review sentence features (key=sha1(text), v=182d dict)\n")

        n_processed = 0
        with gzip.open(FEAT_CACHE, "at", encoding="utf-8") as f_cache:
            for chunk_start in range(0, len(new_unique), CHUNK_FLUSH):
                chunk = new_unique[chunk_start:chunk_start + CHUNK_FLUSH]
                for i, doc in enumerate(
                    nlp.pipe(chunk, batch_size=BATCH_SIZE, n_process=N_PROCESS)
                ):
                    s = chunk[i]
                    k = feat_key(s)
                    try:
                        feats = per_sentence_features_v2(doc)
                        feats = feats if feats is not None else {}
                    except Exception:
                        feats = {}
                    numeric = {n: float(v) for n, v in feats.items()
                               if isinstance(v, (int, float))}
                    # all_fnames collected in Section 4 (canonical keys from FEAT_CACHE)
                    filtered = {n: numeric.get(n, 0.0) for n in all_fnames}
                    feat_map[k] = filtered
                    seen_keys.add(k)
                    f_cache.write(json.dumps({"k": k, "v": filtered}) + "\n")
                    n_processed += 1
                f_cache.flush()
                del chunk
                gc.collect()
                log(f"    flushed chunk {chunk_start + CHUNK_FLUSH}/{len(new_unique)} "
                    f"(mem-safe + gc.collect)")
        log(f"  saved cache: {len(feat_map)} entries (chunked append done)")

    del all_sents, sent_to_user, feat_map
    gc.collect()
    log("  Stage 5 intermediates freed (all_sents / sent_to_user / feat_map)")

    log("\n=== 5b. Fit StandardScaler + PCA48 on FEAT_CACHE (target users' sentences) ===")
    # User directive 2026-08-29: 不再用 10K cache (sentences_for_rewrite_10k.jsonl)。
    # StandardScaler + PCA48 改成 fit on FEAT_CACHE 里 target users 的句子。
    # select_query 从 user_gaussians.json 读 scaler/PCA 投影 query, 不再调
    # _syntax_subspace_prepare()。
    # all_fnames 在 Section 4 已从 FEAT_CACHE 收集, 此处直接复用。

    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.decomposition import PCA as _PCA

    target_shas = set()
    for uid_idxs in []:
        pass  # populated below via sha1_to_uid_idx (defined later)

    feat_records_for_fit: list = []
    N_FIT_MAX = 5000
    n_target_seen = 0
    log(f"  FEAT_CACHE: {len(all_fnames)} feature dims (canonical, "
        f"collected in Section 4)")

    # F3_CoreStruct subset (drop n-gram prefixes + semantic tags)
    EXCL_PREFIXES = ("open_", "close_", "posbg_", "postg_", "depbg_")
    EXCL_EXACT = ("opener", "stype", "has_passive", "is_interrog", "has_cond",
                  "acl", "advcl", "ccomp", "xcomp", "relcl")
    fnames_sub = [n for n in all_fnames
                  if not any(n.startswith(p) for p in EXCL_PREFIXES)
                  and n not in EXCL_EXACT]
    col_idx = [all_fnames.index(n) for n in fnames_sub]
    log(f"  F3_CoreStruct: {len(fnames_sub)} / {len(all_fnames)} features")

    log("\n=== 6. Computing z_user per user (Welford online) ===")
    MIN_NONZERO_FEATS = 0

    target_user_list = sorted(target_users)
    uid_to_idx = {u: i for i, u in enumerate(target_user_list)}
    n_users = len(target_user_list)
    log(f"  target users ordered: {n_users}")

    user_n_reviews: dict = {}
    user_n_words: dict = {}
    sha1_to_uid_idx: dict = collections.defaultdict(list)
    n_reviews_total = 0
    n_reviews_empty = 0
    for uid, texts in user_review_texts.items():
        user_n_reviews[uid] = len(texts)
        user_n_words[uid] = sum(len(t.split()) for t in texts)
        uid_idx = uid_to_idx.get(uid)
        if uid_idx is None:
            continue
        for t in texts:
            t = t.replace("\n", " ").strip()
            if not t:
                n_reviews_empty += 1
                continue
            sha1_to_uid_idx[feat_key(t)].append(uid_idx)
            n_reviews_total += 1
    log(f"  sha1_to_uid_idx built: {len(sha1_to_uid_idx)} unique keys, "
        f"{n_reviews_total} reviews (empty={n_reviews_empty})")
    del user_review_texts
    gc.collect()
    log("  user_review_texts freed")

    counts = np.zeros(n_users, dtype=np.int64)
    mean_acc = np.zeros((n_users, PCA_DIM), dtype=np.float64)
    M2_acc = np.zeros((n_users, PCA_DIM), dtype=np.float64)

    # ---- 6a. Fit StandardScaler + PCA48 on target users' FEAT_CACHE entries ----
    target_shas = set(sha1_to_uid_idx.keys())
    log(f"  target_shas: {len(target_shas)} unique keys")
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
    log(f"  StandardScaler fitted on {len(fit_records)} target-user sentences "
        f"(F3 {len(fnames_sub)}d → PCA{PCA_DIM}, "
        f"EV={pca.explained_variance_ratio_.sum():.4f})")
    del X_fit_full, X_fit_sub, X_fit_sub_scaled, fit_records
    gc.collect()

    # ---- 6a-cache. Welford streaming 缓存 (用户指令 2026-08-29) ----
    # 缓存 Section 6b streaming 的输出 (counts / mean_acc / M2_acc)。下次 Phase 2
    # 只改 Section 7 阈值时跳过 ~13 min streaming。Signature 包含所有影响 streaming
    # 输出的因素: target users, target sha1s, FEAT_CACHE mtime, scaler params, PCA params,
    # fnames, col_idx, MIN_NONZERO_FEATS。
    sorted_shas = sorted(target_shas)
    sha_bytes = "|".join(sorted_shas).encode("utf-8")
    user_bytes = "|".join(target_user_list).encode("utf-8")
    fnames_all_bytes = "|".join(all_fnames).encode("utf-8")
    fnames_sub_bytes = "|".join(fnames_sub).encode("utf-8")
    col_idx_bytes = np.asarray(col_idx, dtype=np.int32).tobytes()
    scaler_bytes = ss.mean_.tobytes() + ss.scale_.tobytes()
    pca_bytes = pca.components_.tobytes() + pca.mean_.tobytes()
    sig_payload = b"".join([
        b"PCA_DIM=" + str(PCA_DIM).encode() + b"\n",
        b"MIN_NONZERO_FEATS=" + str(MIN_NONZERO_FEATS).encode() + b"\n",
        b"FEAT_CACHE_mtime=" + str(FEAT_CACHE.stat().st_mtime).encode() + b"\n",
        b"n_users=" + str(len(target_user_list)).encode() + b"\n",
        b"n_target_shas=" + str(len(target_shas)).encode() + b"\n",
        b"user_hash=" + hashlib.sha1(user_bytes).hexdigest()[:16].encode() + b"\n",
        b"sha_hash=" + hashlib.sha1(sha_bytes).hexdigest()[:16].encode() + b"\n",
        b"fnames_all_hash=" + hashlib.sha1(fnames_all_bytes).hexdigest()[:16].encode() + b"\n",
        b"fnames_sub_hash=" + hashlib.sha1(fnames_sub_bytes).hexdigest()[:16].encode() + b"\n",
        b"col_idx_hash=" + hashlib.sha1(col_idx_bytes).hexdigest()[:16].encode() + b"\n",
        b"scaler_hash=" + hashlib.sha1(scaler_bytes).hexdigest()[:16].encode() + b"\n",
        b"pca_hash=" + hashlib.sha1(pca_bytes).hexdigest()[:16].encode() + b"\n",
    ])
    welford_sig = hashlib.sha1(sig_payload).hexdigest()[:12]
    del sig_payload, sha_bytes, user_bytes, fnames_all_bytes, fnames_sub_bytes
    del col_idx_bytes, scaler_bytes, pca_bytes
    gc.collect()

    welford_cache_hit = False
    if WELFORD_CHECKPOINT.exists() and WELFORD_SIG_JSON.exists():
        try:
            cached_sig = json.load(open(WELFORD_SIG_JSON)).get("sig")
            if cached_sig == welford_sig:
                welford_cache_hit = True
                log(f"  Welford checkpoint CACHE HIT (sig={welford_sig}), loading...")
                npz = np.load(WELFORD_CHECKPOINT)
                counts = npz["counts"]
                mean_acc = npz["mean_acc"]
                M2_acc = npz["M2_acc"]
                npz.close()
                log(f"  loaded counts.shape={counts.shape}, "
                    f"mean_acc.shape={mean_acc.shape}")
            else:
                log(f"  Welford checkpoint signature mismatch "
                    f"(cached={cached_sig}, current={welford_sig}), recomputing...")
        except Exception as e:
            log(f"  Welford checkpoint load failed ({type(e).__name__}: {e!r}), "
                f"recomputing...")

    # ---- 6b. Welford streaming using fitted scaler + PCA ----
    n_cache_seen = 0
    n_cache_matched = 0
    n_skip_sparse = 0
    if welford_cache_hit:
        # Cache 已加载, 跳过 streaming。仅恢复 log 用的统计量 (与 streaming 末尾一致)。
        n_cache_matched = int((counts > 0).sum())  # 占位,只为日志
        log(f"  [Welford CACHE] skipping streaming, counts.sum={int(counts.sum())}, "
            f"users_with_reviews={int((counts > 0).sum())}/{n_users}")
    elif FEAT_CACHE.exists():
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
                    k = rec["k"]
                    v = rec["v"]
                    n_cache_seen += 1
                    uid_idxs = sha1_to_uid_idx.get(k)
                    if uid_idxs is None:
                        continue
                    n_cache_matched += 1
                    nonzero_count = sum(1 for val in v.values() if val != 0.0)
                    if nonzero_count < MIN_NONZERO_FEATS:
                        n_skip_sparse += len(uid_idxs)
                        continue
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
                    if n_cache_matched % 200_000 == 0:
                        gc.collect()
                        n_with_rev = int((counts > 0).sum())
                        log(f"    streaming cache: {n_cache_matched}/{n_cache_seen} matched, "
                            f"sha1_to_uid_idx remaining: {len(sha1_to_uid_idx)}, "
                            f"users with reviews: {n_with_rev}/{n_users}")
        except (EOFError, gzip.BadGzipFile, zlib.error) as e:
            log(f"  WARN streaming cache truncated ({type(e).__name__}: {e!r}), "
                f"seen {n_cache_seen} entries")
    n_skip_no_feats = sum(len(v) for v in sha1_to_uid_idx.values())
    log(f"  streaming cache done: {n_cache_seen} entries seen, "
        f"{n_cache_matched} matched, "
        f"users with reviews: {int((counts > 0).sum())}/{n_users}, "
        f"n_skip_sparse: {n_skip_sparse}, n_skip_no_feats: {n_skip_no_feats}")
    # 用户指令 2026-08-29: sha1_to_uid_idx 不删 — Section 7b self-consistency inlier check
    # 还要复用 (iterate FEAT_CACHE → look up uid_idxs → compute d²(z, G_u))。在 7b 之后释放。
    log(f"  sha1_to_uid_idx retained for Section 7b inlier check ({len(sha1_to_uid_idx)} keys)")

    # 用户指令 2026-08-29: 缓存 Welford 输出 (counts / mean_acc / M2_acc) + signature。
    # 下次 Phase 2 只改 Section 7 阈值时跳过 streaming 节省 ~13 min。
    # 仅在非 cache-hit 且 streaming 完成时保存 (避免重复覆盖)。
    if not welford_cache_hit:
        log(f"  saving Welford checkpoint to {WELFORD_CHECKPOINT} (sig={welford_sig})...")
        np.savez(WELFORD_CHECKPOINT,
                 counts=counts.astype(np.int32),
                 mean_acc=mean_acc.astype(np.float32),
                 M2_acc=M2_acc.astype(np.float32))
        with open(WELFORD_SIG_JSON, "w", encoding="utf-8") as f:
            json.dump({"sig": welford_sig}, f)
        log(f"  Welford checkpoint saved (npz + sig)")

    log("\n=== 7. Computing per-user Gaussian (mu / sigma_diag) ===")
    user_gaussians = {}
    missing_users = []

    valid_mask = counts >= MIN_REVIEWS_FOR_PER_USER
    n_valid = int(valid_mask.sum())
    log(f"  users meeting MIN_REVIEWS_FOR_PER_USER={MIN_REVIEWS_FOR_PER_USER}: {n_valid}")

    safe_counts = np.maximum(counts, 1).astype(np.float64)
    var_pop = M2_acc / safe_counts[:, None]
    var_per_user_mean = var_pop.mean(axis=1, keepdims=True)
    var_shrink = (1 - LAMBDA) * var_pop + LAMBDA * var_per_user_mean
    sigma_diag_arr = np.maximum(var_shrink, VAR_EPS)
    sigma_diag_mean_per_user = sigma_diag_arr.mean(axis=1)
    mu_arr = mean_acc / safe_counts[:, None]
    # Q2 指标: r_floor = #{σ_d ≤ VAR_EPS}/48 — 触底维度比例
    r_floor_per_user = (sigma_diag_arr <= VAR_EPS).mean(axis=1)
    log(f"  σ_mean: min={sigma_diag_mean_per_user[valid_mask].min():.4f}, "
        f"median={np.median(sigma_diag_mean_per_user[valid_mask]):.4f}, "
        f"max={sigma_diag_mean_per_user[valid_mask].max():.4f}")
    log(f"  r_floor (Q2): min={r_floor_per_user[valid_mask].min():.4f}, "
        f"median={np.median(r_floor_per_user[valid_mask]):.4f}, "
        f"max={r_floor_per_user[valid_mask].max():.4f}")

    # 用户指令 2026-08-29: Section 7b — Self-consistency inlier check
    # Q3 = #{自己句子 d²(z, G_u) ≤ R_95}/N ≥ MIN_INLIER_FRAC。
    # 这一指标直接量化 "Gaussian 是不是可靠地描述用户自己的句子", 完全不依赖评论数。
    log(f"\n=== 7b. Self-consistency: per-user inlier fraction ===")
    R_95_SQ = R_95 ** 2
    inlier_count = np.zeros(n_users, dtype=np.int64)
    seen_count = np.zeros(n_users, dtype=np.int64)
    n_seen = 0
    log(f"  iterating FEAT_CACHE for inlier check (R_95²={R_95_SQ:.2f})...")
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
                if not valid_mask[ui]:
                    continue
                seen_count[ui] += 1
                mu_u = mu_arr[ui]
                sigma_u = sigma_diag_arr[ui]
                d_sq = np.sum((z - mu_u) ** 2 / sigma_u)
                if d_sq <= R_95_SQ:
                    inlier_count[ui] += 1
    inlier_frac = np.where(seen_count > 0, inlier_count / seen_count, 0.0)
    log(f"  inlier check done: {n_seen} matched sentences")
    if (seen_count > 0).any():
        valid_inlier = inlier_frac[seen_count > 0]
        log(f"  inlier_frac: min={valid_inlier.min():.3f}, "
            f"median={np.median(valid_inlier):.3f}, "
            f"max={valid_inlier.max():.3f}")

    # 用户指令 2026-08-29: Section 7c — 3-class Gaussian quality gate
    # Q1 non-degenerate: σ_mean ≥ MIN_SIGMA_MEAN (0.01)
    # Q2 non-floored:    r_floor ≤ MAX_R_FLOOR (0.3)
    # Q3 self-consistent: inlier_frac ≥ MIN_INLIER_FRAC (0.5)
    # 不使用评论数 (n_u) 作为质量信号。
    q1_mask = sigma_diag_mean_per_user >= MIN_SIGMA_MEAN
    q2_mask = r_floor_per_user <= MAX_R_FLOOR
    q3_mask = inlier_frac >= MIN_INLIER_FRAC
    quality_mask = q1_mask & q2_mask & q3_mask & valid_mask
    n_quality = int(quality_mask.sum())
    n_q1 = int((q1_mask & valid_mask).sum())
    n_q2 = int((q2_mask & valid_mask).sum())
    n_q3 = int((q3_mask & valid_mask).sum())
    log(f"\n=== 7c. 3-class quality gate ===")
    log(f"  Q1 σ_mean≥{MIN_SIGMA_MEAN}: {n_q1} pass ({n_q1/n_valid*100:.1f}%)")
    log(f"  Q2 r_floor≤{MAX_R_FLOOR}: {n_q2} pass ({n_q2/n_valid*100:.1f}%)")
    log(f"  Q3 inlier_frac≥{MIN_INLIER_FRAC}: {n_q3} pass ({n_q3/n_valid*100:.1f}%)")
    log(f"  ALL Q1&Q2&Q3: {n_quality} pass ({n_quality/n_valid*100:.1f}% of valid)")

    n_written = 0
    n_skipped_degenerate = 0
    n_skip_q1 = 0
    n_skip_q2 = 0
    n_skip_q3 = 0
    for uid_idx, uid in enumerate(target_user_list):
        if not valid_mask[uid_idx]:
            missing_users.append((uid, user_n_reviews.get(uid, 0), int(counts[uid_idx])))
            continue
        if not quality_mask[uid_idx]:
            n_skipped_degenerate += 1
            if not q1_mask[uid_idx]:
                n_skip_q1 += 1
            elif not q2_mask[uid_idx]:
                n_skip_q2 += 1
            elif not q3_mask[uid_idx]:
                n_skip_q3 += 1
            continue
        mu = mu_arr[uid_idx]
        sd = sigma_diag_arr[uid_idx]
        user_gaussians[uid] = {
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
        n_written += 1
    log(f"  per-user Gaussians written: {n_written}")
    log(f"  skipped (3-class gate fail): {n_skipped_degenerate} "
        f"(Q1 σ_mean fail: {n_skip_q1}, Q2 r_floor fail: {n_skip_q2}, "
        f"Q3 inlier fail: {n_skip_q3})")

    if missing_users:
        log(f"  WARN {len(missing_users)}/{n_users} users lack per-user Gaussian "
            f"(counts<{MIN_REVIEWS_FOR_PER_USER}):")
        for uid, n_reviews, n_sents in missing_users[:10]:
            log(f"      {uid[:12]}... reviews={n_reviews} sents={n_sents}")

    # Free inlier intermediates + sha1_to_uid_idx (used by Section 7b, no longer needed)
    del counts, mean_acc, M2_acc, var_pop, var_shrink, sigma_diag_arr
    del mu_arr, inlier_count, seen_count, inlier_frac
    if 'sha1_to_uid_idx' in dir():
        sha1_to_uid_idx.clear()
        del sha1_to_uid_idx
    gc.collect()

    log("\n=== 8. Saving ===")
    GAUSSIANS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GAUSSIANS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 8.5: per-user Gaussian + StandardScaler + PCA48 from "
                                "Baby_Products review corpus, F3_CoreStruct spaCy features "
                                "(Welford online accumulation). "
                                "用户指令 2026-08-29: 不再用 10K cache, scaler/PCA 改 fit on "
                                "FEAT_CACHE 里 target users 的句子, select_query 直接从这里读。"
                                "用户指令 2026-08-29: 3-class Gaussian quality gate (Q1 σ_mean ≥ "
                                f"{MIN_SIGMA_MEAN}, Q2 r_floor ≤ {MAX_R_FLOOR}, "
                                f"Q3 inlier_frac ≥ {MIN_INLIER_FRAC}); 不再使用 review count "
                                "作为质量信号。"),
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
                "N_FIT_MAX": N_FIT_MAX,
                "cache_signature": sig_hash,
            },
            # StandardScaler fit on target users' FEAT_CACHE (F3 subset)
            "scaler_mean": ss.mean_.tolist(),
            "scaler_scale": ss.scale_.tolist(),
            # PCA48 fit on scaled F3 subset
            "pca_components": pca.components_.tolist(),
            "pca_explained_variance": pca.explained_variance_.tolist(),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pca_mean": pca.mean_.tolist(),
            # Feature order (full + F3 subset)
            "feature_names_ordered": all_fnames,
            "fnames_f3": fnames_sub,
            # Per-user Gaussian (Mahalanobis input)
            "users": user_gaussians,
            "n_users_with_gaussian": len(user_gaussians),
            "n_users_skipped": len(target_users) - len(user_gaussians),
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {GAUSSIANS_OUT}")

    # 用户指令 2026-08-29: Phase 1 cohort 是按 raw 评论者数 ≥MIN_USERS_PER_ASIN=2 粗筛,
    # Gaussian 质量门槛在 Phase 2 末尾 step_post_filter_cohort_by_gaussian_quality()
    # 重新判定 — 只保留有 ≥MIN_N_QUALITY_USERS_PER_ASIN 个 Gaussian 质量达标用户的 ASIN。
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