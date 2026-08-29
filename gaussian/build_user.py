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
    MIN_REVIEWS_FOR_PER_USER, PCA_DIM, REVIEW_GZ, VAR_EPS, log, feat_key,
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

    log("\n=== 1. Loading PCA48 ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]
    log(f"  scaler mean shape: {scaler.mean_.shape}, fnames: {len(fnames)}")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    log("\n=== 2. Loading target users ===")
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
        except (EOFError, gzip.BadGzipFile, zlib.error) as e:
            log(f"  WARN cache gzip stream truncated ({type(e).__name__}: {e!r}), "
                f"loaded {len(seen_keys)} keys + {n_load_errors} corrupt lines")
    log(f"  cache keys loaded: {len(seen_keys)} (lazy mode)")

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
        N_PROCESS = 2
        BATCH_SIZE = 128
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
                    filtered = {n: numeric.get(n, 0.0) for n in fnames}
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

    n_cache_seen = 0
    n_cache_matched = 0
    n_skip_sparse = 0
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
                    k = rec["k"]
                    v = rec["v"]
                    n_cache_seen += 1
                    uid_idxs = sha1_to_uid_idx.pop(k, None)
                    if uid_idxs is None:
                        continue
                    n_cache_matched += 1
                    nonzero_count = sum(1 for val in v.values() if val != 0.0)
                    if nonzero_count < MIN_NONZERO_FEATS:
                        n_skip_sparse += len(uid_idxs)
                        continue
                    vec = np.array([v.get(n, 0.0) for n in fnames], dtype=np.float64)
                    vec_scaled = scaler.transform(vec[None, :])[0]
                    z = pca.transform(vec_scaled[None, :])[0].astype(np.float64)
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
    sha1_to_uid_idx.clear()
    del sha1_to_uid_idx
    gc.collect()

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

    n_written = 0
    for uid_idx, uid in enumerate(target_user_list):
        if not valid_mask[uid_idx]:
            missing_users.append((uid, user_n_reviews.get(uid, 0), int(counts[uid_idx])))
            continue
        mu = mean_acc[uid_idx]
        sd = sigma_diag_arr[uid_idx]
        user_gaussians[uid] = {
            "mu": mu.tolist(),
            "sigma_diag": sd.tolist(),
            "n_reviews": user_n_reviews.get(uid, 0),
            "n_words": user_n_words.get(uid, 0),
            "n_sentences": int(counts[uid_idx]),
            "source": "per_user",
        }
        n_written += 1
    log(f"  per-user Gaussians written: {n_written}")

    if missing_users:
        log(f"  WARN {len(missing_users)}/{n_users} users lack per-user Gaussian "
            f"(counts<{MIN_REVIEWS_FOR_PER_USER}):")
        for uid, n_reviews, n_sents in missing_users[:10]:
            log(f"      {uid[:12]}... reviews={n_reviews} sents={n_sents}")

    del counts, mean_acc, M2_acc, var_pop, var_shrink, sigma_diag_arr
    gc.collect()

    log("\n=== 8. Saving ===")
    GAUSSIANS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GAUSSIANS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 8.5: per-user Gaussian from Baby_Products review "
                                "corpus, PCA48 spaCy features (Welford online accumulation)"),
                "PCA_DIM": PCA_DIM,
                "LAMBDA": LAMBDA,
                "VAR_EPS": VAR_EPS,
                "MIN_REVIEWS_FOR_PER_USER": MIN_REVIEWS_FOR_PER_USER,
                "cache_signature": sig_hash,
            },
            "users": user_gaussians,
            "n_users_with_gaussian": len(user_gaussians),
            "n_users_skipped": len(target_users) - len(user_gaussians),
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