#!/usr/bin/env python3
"""按评论数 bucket 评估 Gaussian 拟合质量：Mahalanobis ↓ / NLL / 协方差条件数。

Bucket 调整到实际数据范围: [15], [16-18], [19-21], [22-24], [25-27], [28-30]
每 bucket 抽 100 用户，输出 raw 318d + PCA-32/64/128。
"""
from __future__ import annotations
import gzip
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

# 实际数据 rc 在 15-30
BUCKETS = [
    ("15", lambda rc: rc == 15),
    ("16-18", lambda rc: 16 <= rc <= 18),
    ("19-21", lambda rc: 19 <= rc <= 21),
    ("22-24", lambda rc: 22 <= rc <= 24),
    ("25-27", lambda rc: 25 <= rc <= 27),
    ("28-30", lambda rc: 28 <= rc <= 30),
]
N_SAMPLE = 100
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def sent_key(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()


def load_all():
    # 用户评论数
    users = json.load(open(SCRATCH / "stage1_filtered_users_reviews_10000u.json"))
    user_rc = {u["user_id"]: u["review_count"] for u in users}
    print(f"用户: {len(user_rc)}, rc范围: {min(user_rc.values())}-{max(user_rc.values())}")

    # 句子
    sent_to_uid = {}
    with open(SCRATCH / "sentences_for_rewrite_10k.jsonl") as f:
        for line in f:
            r = json.loads(line)
            sent_to_uid[r["sentence_text"]] = r["user_id"]
    print(f"句子: {len(sent_to_uid)}")

    # 特征
    feat_map = {}
    with gzip.open(SCRATCH / "sentences_318d_cache.jsonl.gz", "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            numeric = {}
            for k, v in rec["v"].items():
                if isinstance(v, (int, float)):
                    numeric[k] = float(v)
            feat_map[rec["k"]] = numeric
    print(f"特征: {len(feat_map)}")

    # 全局特征矩阵
    sample_keys = None
    rows_list = []
    for text, uid in sent_to_uid.items():
        k = sent_key(text)
        f = feat_map.get(k)
        if f:
            if sample_keys is None:
                sample_keys = list(f.keys())
            rows_list.append([f[kk] for kk in sample_keys])
    all_mat = np.stack(rows_list, axis=0).astype(np.float64)
    print(f"全局特征矩阵: {all_mat.shape}, feat_dim={len(sample_keys)}")
    return user_rc, sent_to_uid, feat_map, all_mat, sample_keys


def per_user_metrics(X, global_mean, global_cov_inv, eps=1e-6):
    """计算单个用户的 Mahalanobis / NLL / CondNum。

    X: [n, d] per-sentence features
    global_mean: [d]
    global_cov_inv: [d, d]
    """
    if X.shape[0] == 0:
        return np.nan, np.nan, np.nan
    n, d = X.shape
    mu_u = X.mean(axis=0)
    diff = mu_u - global_mean[:d]

    # Mahalanobis
    mahal = float(np.sqrt(max(float(diff @ global_cov_inv[:d, :d] @ diff), 0)))

    # Covariance
    if n > 1:
        cov_u = np.cov(X, rowvar=False)
        # Condition number
        eigvals = np.linalg.eigvalsh(cov_u)
        pos_eig = eigvals[eigvals > eps]
        cond = float(pos_eig.max() / pos_eig.min()) if len(pos_eig) >= 2 else np.nan
        # NLL: sentences under user's own Gaussian N(mu_u, cov_u)
        # Add small regularization to cov_u for numerical stability
        cov_u_reg = cov_u + eps * np.eye(d)
        sign, logdet_cov = np.linalg.slogdet(cov_u_reg)
        if sign > 0 and np.isfinite(logdet_cov):
            diff2 = diff.reshape(1, d)
            pinv_cov = np.linalg.pinv(cov_u_reg)
            quad = float(diff2 @ pinv_cov @ diff2.T)
            nll = 0.5 * (d * np.log(2 * np.pi) + logdet_cov + quad)
        else:
            nll = np.nan
    else:
        cond = np.nan
        nll = np.nan

    return mahal, nll, cond


def main():
    t0 = time.time()
    user_rc, sent_to_uid, feat_map, all_mat, feat_keys = load_all()

    feat_dim = len(feat_keys)
    global_mean = all_mat.mean(axis=0)
    global_cov = np.cov(all_mat, rowvar=False)
    global_cov_inv = np.linalg.pinv(global_cov)

    # 预处理: 文本→用户列表
    uid_to_texts = {}
    for text, uid in sent_to_uid.items():
        uid_to_texts.setdefault(uid, []).append(text)

    results = {}
    for bucket_name, bucket_fn in BUCKETS:
        eligible = [u for u, rc in user_rc.items() if bucket_fn(rc) and u in uid_to_texts]
        sample_n = min(N_SAMPLE, len(eligible))
        if sample_n == 0:
            print(f"\n{bucket_name}: 无用户")
            continue
        sampled = random.sample(eligible, sample_n)

        # 收集该 bucket 用户的所有句子特征
        bucket_X = []
        for uid in sampled:
            for text in uid_to_texts[uid]:
                k = sent_key(text)
                f = feat_map.get(k)
                if f:
                    bucket_X.append([f[kk] for kk in feat_keys])

        # 对每个用户单独计算指标
        mahas, nlls, conds = [], [], []
        for uid in sampled:
            vecs = []
            for text in uid_to_texts[uid]:
                k = sent_key(text)
                f = feat_map.get(k)
                if f:
                    vecs.append([f[kk] for kk in feat_keys])
            if not vecs:
                mahas.append(np.nan); nlls.append(np.nan); conds.append(np.nan)
                continue
            X = np.stack(vecs, axis=0).astype(np.float64)
            m, n, c = per_user_metrics(X, global_mean, global_cov_inv)
            mahas.append(m); nlls.append(n); conds.append(c)

        mahas = np.array(mahas)
        nlls = np.array(nlls)
        conds = np.array(conds)
        valid = ~np.isnan(mahas)
        print(f"\n{'='*60}")
        print(f"Bucket {bucket_name}: eligible={len(eligible)}, n={valid.sum()}")
        r_raw = {
            "maha_mean": float(np.mean(mahas[valid])),
            "maha_std": float(np.std(mahas[valid])),
            "nll_mean": float(np.mean(nlls[valid])),
            "cond_num_mean": float(np.mean(conds[valid])),
        }
        print(f"  Raw {feat_dim}d: Mahalanobis={r_raw['maha_mean']:.2f}±{r_raw['maha_std']:.2f}, "
              f"NLL={r_raw['nll_mean']:.1f}, CondNum={r_raw['cond_num_mean']:.0f}")
        results[bucket_name] = {"raw": r_raw}

        # PCA versions
        for pca_d in [32, 64, 128]:
            pca = PCA(n_components=pca_d, random_state=42)
            pca.fit(all_mat)
            reduced = pca.transform(all_mat)  # [N, pca_d]
            gmean_pca = reduced.mean(axis=0)
            gcov_pca = np.cov(reduced, rowvar=False)
            gcov_pca_inv = np.linalg.pinv(gcov_pca)

            # Transform each user's sentences
            mahas_p, nlls_p, conds_p = [], [], []
            for uid in sampled:
                vecs = []
                for text in uid_to_texts[uid]:
                    k = sent_key(text)
                    f = feat_map.get(k)
                    if f:
                        vecs.append([f[kk] for kk in feat_keys])
                if not vecs:
                    mahas_p.append(np.nan); nlls_p.append(np.nan); conds_p.append(np.nan)
                    continue
                X_raw = np.stack(vecs, axis=0).astype(np.float64)
                X_pca = pca.transform(X_raw).astype(np.float64)
                m, n, c = per_user_metrics(X_pca, gmean_pca, gcov_pca_inv)
                mahas_p.append(m); nlls_p.append(n); conds_p.append(c)

            mahas_p = np.array(mahas_p)
            nlls_p = np.array(nlls_p)
            conds_p = np.array(conds_p)
            valid_p = ~np.isnan(mahas_p)
            r_pca = {
                "maha_mean": float(np.mean(mahas_p[valid_p])),
                "maha_std": float(np.std(mahas_p[valid_p])),
                "nll_mean": float(np.mean(nlls_p[valid_p])),
                "cond_num_mean": float(np.mean(conds_p[valid_p])),
            }
            print(f"  PCA-{pca_d}: Mahalanobis={r_pca['maha_mean']:.2f}±{r_pca['maha_std']:.2f}, "
                  f"NLL={r_pca['nll_mean']:.1f}, CondNum={r_pca['cond_num_mean']:.0f}")
            results[bucket_name][f"PCA_{pca_d}"] = r_pca

    out = SCRATCH / "gaussian_quality_by_bucket.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] → {out}")
    print(f"[total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
