"""Syntax Subspace — Stage 3 (user Gaussians).

gaussian/ 只保留"计算用户先验分布"逻辑:从 Baby_Products review corpus
为每个 target user 构建 per-user Mahalanobis 高斯 (PCA48 + 收缩,
LAMBDA=0.1, VAR_EPS=1e-3, MIN_REVIEWS_FOR_PER_USER=1)。

不设 fallback 链:若 user 无 Gaussian 直接 raise,让上游数据问题显式暴露。

用法:
  python gaussian/syntax_subspace_user_gaussians.py

I/O 路径:
  输入: stage8_5_asins.json (target users + ASIN)
        data/Baby_Products_2023.jsonl.gz (review corpus)
        stage7b_query_features.jsonl.gz (spaCy 182d features cache)
  输出: stage8_5_user_gaussians.json
        (per-user mu/sigma_diag)

共享工具 (log, feat_key, paths, hyperparams, _syntax_subspace_prepare) 来自:
  common/syntax_subspace_utils.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

# Ensure common/ is on sys.path so we can import the shared utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_OUT, REPO_ROOT, REVIEW_GZ,
    LAMBDA, MIN_REVIEWS_FOR_PER_USER, PCA_DIM, VAR_EPS, log, feat_key,
)

# Stage 3 also imports per_sentence_features_v2 from common/syntactic_features
sys.path.insert(0, str(REPO_ROOT / "common"))


def stage_user_gaussians():
    log("=== STAGE 3 — USER GAUSSIANS ===")

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
    # 用户指令 2026-08-27: 字段提取与 attribute_extraction/build_dataset.py 一致,
    # 否则上游 ≥20 reviews 过滤过的 user 在 Stage 3 找不到 Gaussian
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
    log(f"  users with ≥{MIN_REVIEWS_FOR_PER_USER} reviews (per_user Gaussian): {n_high}")
    n_low = sum(1 for c in review_counts if c < MIN_REVIEWS_FOR_PER_USER and c >= 1)
    log(f"  users with 1-2 reviews: {n_low}")
    n_zero = len(target_users) - len(user_review_texts)
    log(f"  users with 0 reviews: {n_zero}")

    log("\n=== 4. Loading feature cache ===")
    feat_map = {}
    if FEAT_CACHE.exists():
        with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                feat_map[rec["k"]] = rec["v"]
    log(f"  cache loaded: {len(feat_map)} features")

    log("\n=== 5. Extracting spaCy features for user review texts ===")
    # 用户指令 2026-08-27: 不设切句阈值, 用整条 review 作为 1 个 sample。
    # 原因: 86 个 short_users (mean_wc<3) 的 review 多是 1-2 词短句,
    # 切句后平均 nz=0-7 全 0 污染; 整条 review 至少含 token-based features。
    # 抽完特征后按 non-zero count >= K 过滤, 1-2 词 review 大多被自然过滤。
    all_sents = []
    sent_to_user = []
    for uid, texts in user_review_texts.items():
        for t in texts:
            t = t.replace("\n", " ").strip()
            if t:  # 任意非空 review 都保留, 后面按特征过滤
                all_sents.append(t)
                sent_to_user.append(uid)
    log(f"  total reviews (no length threshold, no sentence split): {len(all_sents)}")

    new_sents = [s for s in all_sents if feat_key(s) not in feat_map]
    log(f"  unique sentences: {len(set(all_sents))}, new to extract: {len(new_sents)}")

    if new_sents:
        import spacy
        from syntactic_features import per_sentence_features_v2
        nlp = spacy.load("en_core_web_sm")
        # 用户指令 2026-08-27: 关闭 features 用不到的 spaCy 组件, 提速 30-40%
        for comp in ("ner", "lemmatizer", "attribute_ruler"):
            if comp in nlp.pipe_names:
                nlp.disable_pipe(comp)
        log(f"  extracting features for {len(new_sents)} new sentences (n_process=8, batch=512)...")
        new_unique = sorted(set(new_sents))
        docs = list(nlp.pipe(new_unique, batch_size=512, n_process=8))
        for i, doc in enumerate(docs):
            s = new_unique[i]
            k = feat_key(s)
            try:
                feats = per_sentence_features_v2(doc)
                feats = feats if feats is not None else {}
            except Exception:
                feats = {}
            numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
            filtered = {n: numeric.get(n, 0.0) for n in fnames}
            feat_map[k] = filtered
            if (i + 1) % 5000 == 0:
                log(f"    {i + 1}/{len(new_unique)}")
        with gzip.open(FEAT_CACHE, "wt", encoding="utf-8") as f:
            f.write("# user-review sentence features (key=sha1(text), v=182d dict)\n")
            for k, v in feat_map.items():
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        log(f"  saved cache: {len(feat_map)} entries")

    log("\n=== 6. Computing z_user per user (mean-pool) ===")
    # 用户指令 2026-08-27: K=0 不过滤 nz (只过滤 feats is None)。
    # 原因: 7 个纯 1 词 review user 即使 K=1 也只过 0.66 条 (1 词 review 96.7% nz=0);
    # 如果不过滤 nz, 这些 user 的全 0 z 进 Gaussian 后 σ²_diag=VAR_EPS,
    # Mahalanobis 全 0 等价于 random selection (符合"用户无 personal style"的现实)。
    # LAMBDA=0.1 收缩 var 到 var.mean(),让 0 方差维度不发散。
    MIN_NONZERO_FEATS = 0
    user_z_list = collections.defaultdict(list)
    n_skip_no_feats = 0
    n_skip_sparse = 0
    for s, uid in zip(all_sents, sent_to_user):
        feats = feat_map.get(feat_key(s))
        if not feats:
            n_skip_no_feats += 1
            continue
        nonzero_count = sum(1 for v in feats.values() if v != 0.0)
        if nonzero_count < MIN_NONZERO_FEATS:
            n_skip_sparse += 1
            continue
        vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
        vec_scaled = scaler.transform(vec[None, :])[0]
        z = pca.transform(vec_scaled[None, :])[0]
        user_z_list[uid].append(z)
    log(f"  reviews projected: {sum(len(v) for v in user_z_list.values())}, "
        f"skipped(no_feats): {n_skip_no_feats}, skipped(sparse<{MIN_NONZERO_FEATS}): {n_skip_sparse}")

    log("\n=== 7. Building global pooled variance ===")
    all_z = []
    for uid, zs in user_z_list.items():
        all_z.extend(zs)
    all_z = np.stack(all_z, axis=0)
    global_var = all_z.var(axis=0)
    log(f"  global var shape: {global_var.shape}, mean: {global_var.mean():.4f}")

    log("\n=== 8. Building per-user Gaussians ===")
    user_gaussians = {}
    missing_users = []

    for uid in target_users:
        zs = user_z_list.get(uid, [])
        n_reviews = len(user_review_texts.get(uid, []))
        n_words = sum(len(t.split()) for t in user_review_texts.get(uid, []))

        # 用户指令 2026-08-27: 去掉 fallback 逻辑, 上游 build_dataset.py 已用
        # MIN_REVIEWS_PER_USER=20 过滤, 每个 target_user 都应有足够 sentences。
        # 数据不足直接 raise, 让上游数据问题显式暴露。
        if len(zs) < MIN_REVIEWS_FOR_PER_USER:
            missing_users.append((uid, n_reviews, len(zs)))
            continue

        Z = np.stack(zs, axis=0)
        mu = Z.mean(axis=0)
        var = Z.var(axis=0)
        var_shrink = (1 - LAMBDA) * var + LAMBDA * var.mean()
        sigma_diag = np.maximum(var_shrink, VAR_EPS)

        user_gaussians[uid] = {
            "mu": mu.tolist(),
            "sigma_diag": sigma_diag.tolist(),
            "n_reviews": n_reviews,
            "n_words": n_words,
            "n_sentences": len(zs),
            "source": "per_user",
        }

    if missing_users:
        log(f"  ❌ ERROR: {len(missing_users)}/{len(target_users)} users lack per-user Gaussian:")
        for uid, n_reviews, n_sents in missing_users[:20]:
            log(f"      {uid[:12]}... reviews={n_reviews} sents={n_sents}")
        raise RuntimeError(
            f"上游过滤条件不足: {len(missing_users)} users have <{MIN_REVIEWS_FOR_PER_USER} sentences. "
            f"需修 attribute_extraction/build_dataset.py 的 user_id 字段提取逻辑, "
            f"或降低 MIN_REVIEWS_PER_USER。"
        )

    log(f"  per-user Gaussians: {len(user_gaussians)}")
    log(f"  (no fallback chain: upstream MIN_REVIEWS_PER_USER=20 保证所有 user 都有 Gaussian)")

    log("\n=== 9. Saving ===")
    GAUSSIANS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GAUSSIANS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5: per-user Gaussian from Baby_Products review corpus, PCA48 spaCy features",
                "PCA_DIM": PCA_DIM,
                "LAMBDA": LAMBDA,
                "VAR_EPS": VAR_EPS,
                "MIN_REVIEWS_FOR_PER_USER": MIN_REVIEWS_FOR_PER_USER,
            },
            "users": user_gaussians,
            "n_users_with_gaussian": len(user_gaussians),
            "n_users_skipped": len(target_users) - len(user_gaussians),
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {GAUSSIANS_OUT}")


def main():
    parser = argparse.ArgumentParser(description="Syntax Subspace — gaussian/ Stage 3 user Gaussians")
    args = parser.parse_args()
    log("=== syntax_subspace_user_gaussians.py ===")
    stage_user_gaussians()


if __name__ == "__main__":
    main()