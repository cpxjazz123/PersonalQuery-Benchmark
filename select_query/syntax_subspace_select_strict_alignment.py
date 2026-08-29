"""Stage 2 + Stage 4 — spaCy features + Strict Personalized Alignment Selection.

合并 Stage 2 features 和 Stage 4 strict alignment 到单一脚本(用户指令 2026-08-29):

Stage 2 (spaCy features):
  - 读 result/gen_query/pool.json → pool queries
  - 缺失 query → spaCy nlp.pipe(batch) 抽 318d 句法特征
  - 写 stage7b_query_features.jsonl.gz(本目录下,不是 scratch2)
  - Stage 3 metadata 同步读取此 cache (Phase 2 Gaussian 用 user review texts 扩充)

Stage 4 strict alignment:
  - 用户原话:
    > 选到"在 cohort 里最像目标用户",并且"绝对上也确实落在目标用户正常句法范围内"的 Query。
    >
    > 严格候选集:
    >   Q_strict = {q: M(q,u) > 0 AND d_self(q,u) ≤ R_95}
    > 选 M 最大的那条:
    >   q*_u = argmax_q∈Q_strict M(q,u)
    >
    > 如果没有任何 Query 同时满足两个条件,就标记为 no_strict_candidate。
  - 两步 gate + 一步排序:
    1. M(q,u) > 0              (target user Rank@1 in cohort)
    2. d_self(q,u) ≤ R_95       (query 落在真实用户历史 95% Mahalanobis 距离内)
    3. argmax M under Q_strict
  - 用户指令 2026-08-29: 用 full Gaussian (mu + sigma_diag from user_gaussians.json)
    算 Mahalanobis 距离, 取代之前 10K-derived L2 means。
  - R_95 = sqrt(chi2.ppf(0.95, 48)) = 8.073 (Mahalanobis P95, d=48)
  - 复用 K=200 pool + F3_CoreStruct + PCA48 + whitening 已有 infrastructure。

输出:
  select_query/stage7b_query_features.jsonl.gz (Stage 2 cache, in-place)
  scratch2/stage8_5_selection.json (Stage 4 main)
  scratch2/stage8_5_selection_stats.json (Stage 4 stats)
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_OUT, POOL_IN,
    PCA_DIM, PCA_SEED, log, feat_key,
)
# User instruction 2026-08-29: inlined select_feature_names / fit_pca here
# because syntax_subspace_repr_sweep was archived (用户指令 "1 script per
# directory"). 旧 import 已被删除.

# F3_CoreStruct: drop sequence n-gram features (open_/close_/posbg_/postg_/depbg_)
# and semantic tags (opener/stype/has_passive/is_interrog/has_cond/acl/advcl/
# ccomp/xcomp/relcl) — keep only structural core (depth, n_clause, etc.).
_EXCLUDE_PREFIXES = ("open_", "close_", "posbg_", "postg_", "depbg_")
_EXCLUDE_EXACT = (
    "opener", "stype", "has_passive", "is_interrog", "has_cond",
    "acl", "advcl", "ccomp", "xcomp", "relcl",
)


def select_feature_names(all_names: list[str], exclude_prefixes: tuple,
                         exclude_exact: tuple) -> list[str]:
    """Apply prefix/exact filter, return ordered feature names."""
    keep = []
    for n in all_names:
        if any(n.startswith(p) for p in exclude_prefixes):
            continue
        if n in exclude_exact:
            continue
        keep.append(n)
    return keep


def fit_pca(X_train, n_dim, seed):
    """Fit PCA(d) with given seed, return pca, sqrt_lambda."""
    pca = PCA(n_components=n_dim, random_state=seed)
    pca.fit(X_train)
    return pca, np.sqrt(pca.explained_variance_)


# User instruction 2026-08-29: full Gaussian (mu + sigma_diag) Mahalanobis.
# R_95 = sqrt(chi2.ppf(0.95, 48)) = 8.073 (theoretical P95 for d=48 dim).
# 替换之前 10K-derived L2 means + R_95=11.308。
R_95_PERCENTILE = 8.073  # Mahalanobis P95, d=48
from scipy.stats import chi2
_R_95_SQ = chi2.ppf(0.95, PCA_DIM)  # = 65.17, Mahal² ≤ R²_95 = 65.17

POOL_IN_LOCAL = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K200_F3pca48_full.json"
# 用户指令 2026-08-29: FEAT_CACHE 改到 select_query/ 目录下(Stage 3 metadata 也同步)
FEAT_CACHE_LOCAL = str(FEAT_CACHE)  # from syntax_subspace_utils
# Main pipeline (用户指令 2026-08-28): 直接覆盖 canonical paths,
# 让 Stage 5 retrieval 默认读取 strict alignment 数据。
# 用户指令 2026-08-29 (sweep): 支持环境变量覆盖 SEL_OUT / STATS_OUT, 让 sweep 跑多阈值
# 不互相覆盖 stage8_5_selection.json。每个 threshold 用 SEL_OUT_SUFFIX (e.g. "t0.7")。
import os as _os
_SUFFIX = _os.environ.get("SEL_OUT_SUFFIX", "")
SEL_OUT = (
    f"/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/"
    f"stage8_5_selection{_SUFFIX}.json"
)
STATS_OUT = (
    f"/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/"
    f"stage8_5_selection_stats{_SUFFIX}.json"
)


def load_query_features(path: str) -> dict[str, np.ndarray]:
    """Load sha1(text) -> 182d features from sentence features cache.
    用户指令 2026-08-29: 容错 zlib + JSON 错误 (chunked append 模式下
    cache 文件可能被强杀在 gzip block 中间).
    """
    import hashlib
    log(f"  Loading sentence features cache {path} ...")
    feat_lookup: dict[str, np.ndarray] = {}
    n_header = 0
    n_skip = 0
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    n_header += 1
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    # 末尾 line 可能被截断 (chunked append + kill)
                    n_skip += 1
                    continue
                if "k" in e and "v" in e:
                    feat_lookup[e["k"]] = np.array(list(e["v"].values()), dtype=np.float64)
    except (EOFError, gzip.BadGzipFile, zlib.error) as exc:
        log(f"  ⚠ cache gzip stream truncated ({type(exc).__name__}), "
            f"loaded {len(feat_lookup)} entries + skipped {n_skip} corrupt")
    log(f"  feat_lookup (sha1 -> 182d) size: {len(feat_lookup)}, header lines: {n_header}")
    return feat_lookup


def sha1_of(text: str) -> str:
    """SHA1 of lowercased text (matches stage7b_query_features cache key format)."""
    import hashlib
    return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()


def stage_features():
    """Stage 2 — extract spaCy 182d features for pool queries.

    用户指令 2026-08-29: 从 gen_query/syntax_subspace_pool_regen.py --stage features
    迁移过来, cache 文件路径同步移到 select_query/ 目录下。
    Stage 3 Gaussian (gaussian/build_user.py) 通过 FEAT_CACHE 常量同步读这个文件。

    用户指令 2026-08-29 (10K 移除): 不再调 _syntax_subspace_prepare()。
    canonical fnames 直接从已有 FEAT_CACHE 条目的 keys 推断,
    若 cache 为空则以第一条新记录作为 canonical, 保证后续 Phase 2/Stage 4
    PCA48 投影维度一致。
    """
    log("=== STAGE 2 — FEATURES ===")

    log(f"loading {POOL_IN}")
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    all_queries = []
    for asin, qs in pools.items():
        for q in qs:
            all_queries.append(q["query"])
    log(f"  total pool queries: {len(all_queries)}")

    # Build set of needed keys (queries in current pool) so we only keep those entries
    # (FEAT_CACHE has ~1.68M historical entries, but current pool only has ~219K — saves ~8× memory)
    needed_keys = set(feat_key(q) for q in all_queries)
    log(f"  needed_keys (current pool): {len(needed_keys)}")

    feat_map = {}
    if Path(FEAT_CACHE_LOCAL).exists():
        with gzip.open(FEAT_CACHE_LOCAL, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                if rec["k"] in needed_keys:
                    feat_map[rec["k"]] = rec["v"]
    log(f"  cache keys (filtered): {len(feat_map)}")

    # canonical fnames: from existing FEAT_CACHE entries (sorted); empty cache → defer to first new record
    fnames: list[str] = []
    if feat_map:
        fnames_set: set = set()
        for v in feat_map.values():
            fnames_set.update(v.keys())
        fnames = sorted(fnames_set)
    log(f"  canonical fnames (from FEAT_CACHE): {len(fnames)}")

    missing_q = [q for q in all_queries if feat_key(q) not in feat_map]
    log(f"  missing: {len(missing_q)}")

    if not missing_q:
        log("  no new features needed")
        return

    import spacy
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
    from syntactic_features import per_sentence_features_v2
    nlp = spacy.load("en_core_web_sm")
    # 用户指令 2026-08-27: 关闭 features 用不到的 spaCy 组件, 提速 30-40%
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    log(f"  extracting features for {len(missing_q)} queries via spaCy pipe (n_process=8, batch=512)...")
    new_unique = sorted(set(missing_q))
    new_entries = []
    n_skip = 0
    docs = list(nlp.pipe(new_unique, batch_size=512, n_process=8))
    for i, doc in enumerate(docs):
        q = new_unique[i]
        k = feat_key(q)
        try:
            feats = per_sentence_features_v2(doc)
            feats = feats if feats is not None else {}
        except Exception:
            n_skip += 1
            continue
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        # If FEAT_CACHE was empty, bootstrap fnames from first new record
        if not fnames:
            fnames = sorted(numeric.keys())
        filtered = {n: numeric.get(n, 0.0) for n in fnames}
        feat_map[k] = filtered
        new_entries.append({"k": k, "v": filtered})
        if (i + 1) % 1000 == 0:
            log(f"    {i + 1}/{len(new_unique)}")

    log(f"  extracted: {len(new_entries)}, skipped: {n_skip}")

    with gzip.open(FEAT_CACHE_LOCAL, "wt", encoding="utf-8") as f:
        f.write("# spaCy 182d sentence features (key=sha1(text), v=filtered dict)\n")
        for k, v in feat_map.items():
            f.write(json.dumps({"k": k, "v": v}) + "\n")
    log(f"  saved cache: {len(feat_map)} entries → {FEAT_CACHE_LOCAL}")


def main():
    log_start = time.time()
    log("=== Stage 2 + Stage 4 strict alignment: spaCy features + Strict Personalized Selection ===")
    log(f"  R_95 (hard threshold) = {R_95_PERCENTILE:.3f}")
    log(f"  Gate: M(q,u) > 0 AND d_self(q,u) ≤ R_95")
    log(f"  Selection: argmax M under gate (strict personalized subset)")

    # ---- Stage 2 — spaCy features (cache to select_query/) ----
    stage_features()

    # ---- 1. Load pool K=200 ----
    log("\n=== 1. Loading K=200 pool ===")
    pool_data = json.load(open(POOL_IN_LOCAL))
    asin_pool: dict[str, list[dict]] = pool_data["pools"]
    log(f"  pool ASINs: {len(asin_pool)}")
    log(f"  total queries: {pool_data['n_total_strict']}")

    # ---- 2 + 3. Load scaler + PCA48 + per-user Gaussian (from user_gaussians.json) ----
    # User directive 2026-08-29: 删 _syntax_subspace_prepare() + sentences_for_rewrite_10k.jsonl。
    # Phase 2 自己 fit StandardScaler + PCA48 on FEAT_CACHE (target users' sentences),
    # 把 scaler_mean/scale + pca_components + fnames 写进 user_gaussians.json。
    # select_query 直接从这里读, 不再调 10K cache。
    log("\n=== 2+3. Loading scaler + PCA48 + per-user Gaussian (from user_gaussians.json) ===")
    gauss_data = json.load(open(GAUSSIANS_OUT))

    ss = StandardScaler()
    ss.mean_ = np.array(gauss_data["scaler_mean"], dtype=np.float64)
    ss.scale_ = np.array(gauss_data["scaler_scale"], dtype=np.float64)
    ss.n_features_in_ = len(ss.mean_)

    pca = PCA(n_components=PCA_DIM)
    pca.components_ = np.array(gauss_data["pca_components"], dtype=np.float64)
    pca.explained_variance_ = np.array(gauss_data["pca_explained_variance"], dtype=np.float64)
    pca.explained_variance_ratio_ = np.array(
        gauss_data["pca_explained_variance_ratio"], dtype=np.float64
    )
    pca.mean_ = np.array(gauss_data["pca_mean"], dtype=np.float64)
    pca.n_components_ = PCA_DIM
    pca.n_features_in_ = len(pca.mean_)

    sqrt_lambda = np.sqrt(pca.explained_variance_)
    all_fnames = gauss_data["feature_names_ordered"]
    fnames_sub = gauss_data["fnames_f3"]
    col_idx = [all_fnames.index(n) for n in fnames_sub]

    # 用户指令 2026-08-29 (sweep 优化): 当 STAGE4_USER_FILTER=1 时, 只保留 cohort 实际
    # 用到的 user Gaussian (从 stage8_5_asins.json 的 users_sampled 取并集)。
    # 否则加载全 1.22M 用户 (~12GB peak, 在 32GB cgroup 里只能跑单 worker)。
    users_gauss_full = gauss_data["users"]
    if os.environ.get("STAGE4_USER_FILTER") == "1":
        # 先读 cohort, 收集 needed uids
        asins_path = ASINS_IN if "_SUFFIX_BU" not in globals() else SCRATCH / f"stage8_5_asins{os.environ.get('ASINS_OUT_SUFFIX', '')}.json"
        if "ASINS_OUT_SUFFIX" in os.environ:
            from pathlib import Path as _P
            asins_path = _P("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades") / f"stage8_5_asins{os.environ['ASINS_OUT_SUFFIX']}.json"
        cohort_for_filter = json.load(open(asins_path))
        needed_uids = set()
        for e in cohort_for_filter.get("asins", []):
            needed_uids.update(e.get("users_sampled", []))
        log(f"  cohort users_sampled (unique): {len(needed_uids)}")
        users_gauss = {u: g for u, g in users_gauss_full.items() if u in needed_uids}
        log(f"  filtered Gaussian cache: {len(users_gauss)} / {len(users_gauss_full)} "
            f"users (saved ~{len(users_gauss_full)-len(users_gauss)} entries)")
        del users_gauss_full, gauss_data
        import gc; gc.collect()
    else:
        users_gauss = users_gauss_full
    log(f"  F3: {len(fnames_sub)} features, scaler.mean_.shape={ss.mean_.shape}, "
        f"PCA{PCA_DIM}: cumvar={pca.explained_variance_ratio_.sum():.4f}")
    log(f"  users with full Gaussian: {len(users_gauss)}")

    # ---- 5. Load ASINs with cohort users_sampled ----
    log("\n=== 5. Loading stage8_5_asins.json ===")
    asin_data_full = json.load(open(ASINS_IN))["asins"]
    asin_data = [a for a in asin_data_full if a["asin"] in asin_pool]
    log(f"  ASINs with K=200 pool: {len(asin_data)} / {len(asin_data_full)}")

    # ---- 6. Load query features cache (sha1 -> 182d) ----
    log("\n=== 6. Loading sentence features cache (sha1 -> 182d) ===")
    feat_lookup = load_query_features(FEAT_CACHE_LOCAL)

    # ---- 7. Selection loop ----
    log("\n=== 7. Strict personalized selection (per asin, per user) ===")
    selection_entries = []
    n_no_pool_for_user = 0
    n_no_user_for_asin = 0          # 用户指令 2026-08-29: 初始化, 否则 n_users==0 分支 raise UnboundLocalError
    n_strict_personalized = 0       # NEW: M>0 AND d_self ≤ R_95
    n_no_strict_candidate = 0      # NEW: no candidate passes both gates
    n_missing_gauss = 0             # user not in user_gauss metadata table

    for entry in asin_data:
        asin = entry["asin"]
        strict_pool = asin_pool[asin]  # already filtered by Stage 1 strict
        if not strict_pool:
            for uid in entry["users_sampled"]:
                if uid not in users_gauss:
                    n_missing_gauss += 1
                    meta_source, meta_n_reviews = None, None
                else:
                    meta_source = users_gauss[uid]["source"]
                    meta_n_reviews = users_gauss[uid]["n_reviews"]
                n_no_pool_for_user += 1
                selection_entries.append({
                    "asin": asin, "user_id": uid,
                    "selection_method": "no_pool", "selected": None,
                    "selected_distance": None, "selected_margin": None,
                    "n_candidates": 0,
                    "user_source": meta_source,
                })
            continue
        # Collect user cohort (full Gaussian: mu + sigma_diag)
        asin_users = []
        for uid in entry["users_sampled"]:
            if uid not in users_gauss:
                # 用户不在 Phase 2 全量 Gaussian 表中 → 无法算 Mahalanobis → 跳过
                n_missing_gauss += 1
                continue
            mu = np.array(users_gauss[uid]["mu"], dtype=np.float64)
            sigma_diag = np.array(users_gauss[uid]["sigma_diag"], dtype=np.float64)
            meta_source = users_gauss[uid]["source"]
            meta_n_reviews = users_gauss[uid]["n_reviews"]
            asin_users.append((uid, mu, sigma_diag,
                               {"source": meta_source, "n_reviews": meta_n_reviews}))

        n_users = len(asin_users)
        if n_users == 0:
            # 边界: 该 ASIN 没有合格用户 (全部 <2 句子)。记录 no_user_for_asin 后 continue
            n_no_user_for_asin += 1
            log(f"  no eligible users for asin {asin}, skip")
            continue
        n_queries = len(strict_pool)

        # ---- 7a. Compute whitened L2 matrix ||z̃_q − μ̃_u|| ----
        # Get query whitened features via sha1(text) lookup
        Z_q_list = []
        valid_q_idx = []
        for qi, q_entry in enumerate(strict_pool):
            text = q_entry.get("query", "")
            sha = sha1_of(text)
            full_feat = feat_lookup.get(sha)
            if full_feat is None or len(full_feat) != 182:
                continue
            # Subset to F3_CoreStruct columns
            x_sub = full_feat[col_idx]
            x_sub_scaled = ss.transform(x_sub.reshape(1, -1))
            z_white = pca.transform(x_sub_scaled) / sqrt_lambda
            Z_q_list.append(z_white[0])
            valid_q_idx.append(qi)
        if not Z_q_list:
            for uid, mu, sigma_diag, gauss_info in asin_users:
                n_no_pool_for_user += 1
                selection_entries.append({
                    "asin": asin, "user_id": uid,
                    "selection_method": "no_query_features", "selected": None,
                    "selected_distance": None, "selected_margin": None,
                    "n_candidates": 0,
                    "user_source": gauss_info["source"],
                })
            continue
        Z_q_arr = np.stack(Z_q_list, axis=0)  # (n_valid_q, 48)
        n_valid_q = Z_q_arr.shape[0]
        cohort_mus = np.stack([au[1] for au in asin_users], axis=0)  # (n_users, 48)
        cohort_sigmas = np.stack([au[2] for au in asin_users], axis=0)  # (n_users, 48)
        # ---- 7a. Mahalanobis distance matrix d_Mahal(q, u) ----
        # d²_Mahal = sum_d ((z_q - mu_u)² / sigma_u²)
        diff = cohort_mus[:, None, :] - Z_q_arr[None, :, :]  # (n_users, n_valid_q, 48)
        mahal_sq = (diff ** 2 / cohort_sigmas[:, None, :]).sum(axis=2)  # (n_users, n_valid_q)
        mahal_dist = np.sqrt(np.maximum(mahal_sq, 0.0))  # numerical floor

        # ---- 7b. Compute M_Mahal per (user, query) ----
        # d_self = mahal_dist[ui, qi]
        # d_other = min over v != ui of mahal_dist[v, qi]
        d_other_mahal = np.zeros((n_users, n_valid_q))
        for qi in range(n_valid_q):
            d_q = mahal_dist[:, qi]
            sorted_for_q = np.sort(d_q)
            is_self_min = (d_q == sorted_for_q[0])
            if len(sorted_for_q) >= 2:
                d_other_mahal[:, qi] = np.where(is_self_min, sorted_for_q[1], sorted_for_q[0])
            else:
                # 单 user cohort: M=0 (no other user to compare)
                d_other_mahal[:, qi] = sorted_for_q[0]
        M_mahal = d_other_mahal - mahal_dist  # (n_users, n_valid_q)

        # ---- 7c. NEW GATE: M>0 AND d_self ≤ R_95 (Mahalanobis P95 = 8.073) ----
        gate_strict = (M_mahal > 0) & (mahal_dist <= R_95_PERCENTILE)

        # ---- 7d. Per-user selection ----
        for ui, (uid, mu, sigma_diag, gauss_info) in enumerate(asin_users):
            source = gauss_info["source"]

            cand_strict_mask = gate_strict[ui]
            if not cand_strict_mask.any():
                # No candidate passes both gates → mark no_strict_candidate
                # DO NOT fallback to margin-max or nearest
                best_qi = int(np.argmin(mahal_dist[ui]))
                best_dist = float(mahal_dist[ui][best_qi])
                best_margin = float(M_mahal[ui][best_qi])
                method = "no_strict_candidate"
                n_no_strict_candidate += 1
                selected_q = None
            else:
                avail_M = np.where(cand_strict_mask, M_mahal[ui], -np.inf)
                best_qi = int(np.argmax(avail_M))
                best_dist = float(mahal_dist[ui][best_qi])
                best_margin = float(M_mahal[ui][best_qi])
                method = "strict_personalized_mahal_margin_max"
                n_strict_personalized += 1
                # Map best_qi back to original strict_pool index
                orig_qi = valid_q_idx[best_qi]
                selected_q = strict_pool[orig_qi]

            selection_entries.append({
                "asin": asin,
                "user_id": uid,
                "selection_method": method,
                "selected": selected_q,
                "selected_distance": best_dist,
                "selected_margin": best_margin,
                "n_candidates": int(cand_strict_mask.sum()),
                "user_source": source,
            })

    # ---- 8. Save ----
    log("\n=== 8. Saving selection ===")
    Path(SEL_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(SEL_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 strict alignment: Strict Personalized Alignment Selection. "
                                "Two-gate: M(q,u) > 0 (target user Rank@1 in Mahalanobis) AND "
                                "d_self(q,u) ≤ R_95=8.073 (Mahalanobis P95, sqrt(chi2(0.95, 48))). "
                                "Then argmax M under gate. "
                                "If no candidate passes both, mark 'no_strict_candidate' "
                                "WITHOUT fallback to margin-max."),
                "feature_subset": "F3_CoreStruct",
                "pca_dim": PCA_DIM,
                "R_95_threshold": R_95_PERCENTILE,
                "distance_metric": "mahalanobis (mu + sigma_diag from user_gaussians.json)",
                "gate_conditions": ["M > 0", f"d_self <= R_95={R_95_PERCENTILE} (Mahalanobis P95)"],
                "selection_logic": "argmax M under (M > 0 AND d_self ≤ R_95)",
                "no_strict_candidate_handling": "no fallback — mark as no_strict_candidate",
            },
            "n_entries": len(selection_entries),
            "entries": selection_entries,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SEL_OUT}")

    # ---- 9. Stats ----
    by_method = collections.Counter(e["selection_method"] for e in selection_entries)
    log(f"\n  Selection method counts:")
    for m, n in by_method.most_common():
        log(f"    {m}: {n}")

    selected_entries = [e for e in selection_entries
                        if e["selection_method"] == "strict_personalized_mahal_margin_max"]
    no_strict = [e for e in selection_entries if e["selection_method"] == "no_strict_candidate"]

    log("\n  Stats on strict_personalized selected (n={}):".format(len(selected_entries)))
    if selected_entries:
        sel_d = np.array([e["selected_distance"] for e in selected_entries])
        sel_m = np.array([e["selected_margin"] for e in selected_entries])
        sel_n_cands = np.array([e["n_candidates"] for e in selected_entries])
        log(f"    d_self mean = {sel_d.mean():.3f}, median = {np.median(sel_d):.3f}")
        log(f"    M mean = {sel_m.mean():.3f}, median = {np.median(sel_m):.3f}")
        log(f"    n_candidates (in Q_strict) mean = {sel_n_cands.mean():.1f}, "
            f"median = {np.median(sel_n_cands):.1f}")

    # Save stats
    stats_data = {
        "n_pairs": len(selection_entries),
        "n_missing_user_gauss_metadata": n_missing_gauss,
        "n_no_user_for_asin": n_no_user_for_asin,
        "by_selection_method": dict(by_method),
        "strict_personalized": {
            "n": len(selected_entries),
            "selected_distance_mean": float(np.mean([e["selected_distance"] for e in selected_entries])) if selected_entries else None,
            "selected_distance_median": float(np.median([e["selected_distance"] for e in selected_entries])) if selected_entries else None,
            "margin_mean": float(np.mean([e["selected_margin"] for e in selected_entries])) if selected_entries else None,
            "margin_median": float(np.median([e["selected_margin"] for e in selected_entries])) if selected_entries else None,
            "n_candidates_in_strict_subset_mean": float(np.mean([e["n_candidates"] for e in selected_entries])) if selected_entries else None,
        },
        "no_strict_candidate": {
            "n": len(no_strict),
            "fraction": float(len(no_strict) / len(selection_entries)) if selection_entries else 0,
            "selected_distance_mean_when_no_strict": float(np.mean([e["selected_distance"] for e in no_strict])) if no_strict else None,
        },
    }
    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {STATS_OUT}")
    log(f"\n=== Stage 4 strict alignment complete ({time.time() - log_start:.1f}s) ===")


if __name__ == "__main__":
    main()