#!/usr/bin/env python3
"""Generate-then-Rank: 给每个候选 query 打分 → 选 top-1.

加载已训练 VADES (SentenceEncoder + UserDistributionTableGMM K=2 D=20),
加载 candidates.jsonl, 用 spaCy 提取 20 维 clause features,
用训练时拟合的 StandardScaler 标准化, encoder forward 得 mu_query,
对每个 user 计算 GMM-mix-weighted user_mu, 候选 score = -||mu_query - user_mu||².

Outputs:
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/scored.jsonl
  每行: {user_id, asin, candidate, mu_query (20d), score, gt_similarity (to gt)
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/top1.jsonl
  每行: {user_id, asin, top1_candidate, top1_score, n_candidates}
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SYNTAX_DIR = REPO_ROOT / "syntactic_analysis"
sys.path.insert(0, str(SYNTAX_DIR))
from extract_clause_features_single_query import extract_clause_features  # noqa: E402

# 复用 gaussian_vades.py 里的 SentenceEncoder / UserDistributionTableDisentangled
sys.path.insert(0, str(REPO_ROOT))
from gaussian.gaussian_vades import (  # noqa: E402
    SentenceEncoder, UserDistributionTableDisentangled,
)

# ==== 选择 user table 类型: disentangle vs gmm ====
USER_TABLE_CLASS = UserDistributionTableDisentangled
COVARIANCE_MODE_TAG = "disentangled"

INPUT_DIM = 20
HIDDEN_DIM = 64
LATENT_DIM = 20
GMM_K = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
TEST_CASES = EXPERIMENT_DIR / "test_cases.jsonl"
CANDIDATES = EXPERIMENT_DIR / "candidates.jsonl"
SCORED_OUT = EXPERIMENT_DIR / "scored.jsonl"
TOP1_OUT = EXPERIMENT_DIR / "top1.jsonl"

# VADES 训练产物 (默认 disentangle)
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
SENTENCE_FILE = VADES_DIR / "vades_disentangled_v2_train10_holdout10_sentences.jsonl"
USER_PROFILE_FILE = VADES_DIR / "vades_disentangled_v2_train10_holdout10_user_profiles.jsonl"
ENCODER_CKPT = VADES_DIR / "vades_encoder.pt"
USER_TABLE_CKPT = VADES_DIR / "vades_user_table.pt"

# 训练 summary 里固化的 feature_names
FEATURE_NAMES = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]


def fit_scaler_from_training_sentences() -> StandardScaler:
    """用训练时同样的数据 + 字段顺序拟合 StandardScaler."""
    log = lambda m: print(f"[rank] {m}", flush=True)
    log(f"读训练句子 → fit StandardScaler: {SENTENCE_FILE}")
    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    feature_matrix = np.asarray(
        [[float(row["features"][name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    scaler = StandardScaler()
    scaler.fit(feature_matrix)
    log(f"  n_train={len(rows)} 张, mean[:5]={scaler.mean_[:5].tolist()}, scale[:5]={scaler.scale_[:5].tolist()}")
    return scaler


def load_user_profiles() -> tuple[dict[str, np.ndarray], int]:
    """加载 user_id → user_mu (mix-weighted, K 维度 collapse 后的 20-dim)."""
    log = lambda m: print(f"[rank] {m}", flush=True)
    log(f"读 user profiles: {USER_PROFILE_FILE}")
    raw: dict[str, list[list[float]]] = {}
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            row = json.loads(line)
            uid = row["user_id"]
            vec = np.asarray(row["user_mu"], dtype=np.float64)
            raw[uid] = vec
    log(f"  n_users={len(raw)}, dim={len(next(iter(raw.values())))}")
    return raw, len(raw)


def build_model() -> tuple[SentenceEncoder, UserDistributionTableDisentangled]:
    encoder = SentenceEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    enc_sd = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    ut_sd = torch.load(USER_TABLE_CKPT, map_location=DEVICE, weights_only=False)
    # 从 state_dict 推 n_clusters (style_centers 第一维)
    n_clusters = int(ut_sd["style_centers"].shape[0])
    num_users = int(ut_sd["user_offsets"].shape[0])
    latent_dim = int(ut_sd["style_centers"].shape[1])
    placeholder_cluster = torch.zeros(num_users, dtype=torch.long)
    placeholder_cluster[0] = n_clusters - 1  # 让 max()=n_clusters-1 → constructor 建出 n_clusters 个 style_center
    user_table = UserDistributionTableDisentangled(
        num_users=num_users, latent_dim=latent_dim,
        user_cluster_ids=placeholder_cluster,
        style_anchors=None,
    )
    encoder.load_state_dict(enc_sd)
    user_table.load_state_dict(ut_sd)
    encoder.to(DEVICE).eval()
    user_table.to(DEVICE).eval()
    return encoder, user_table


def load_test_cases() -> dict[tuple[str, str], dict]:
    cases: dict[tuple[str, str], dict] = {}
    with TEST_CASES.open() as f:
        for line in f:
            row = json.loads(line)
            cases[(row["user_id"], row["asin"])] = row
    return cases


def load_candidates() -> dict[tuple[str, str], list[str]]:
    """把所有 (user, asin) 的候选聚合为一个长 list."""
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    with CANDIDATES.open() as f:
        for line in f:
            row = json.loads(line)
            grouped[(row["user_id"], row["asin"])].extend(row["candidates"])
    return grouped


def extract_features_for_candidates(candidates: list[str], scaler: StandardScaler) -> np.ndarray:
    """批量 spaCy 解析 → 20-dim 特征 → 标准化."""
    log = lambda m: print(f"[rank] {m}", flush=True)
    log(f"  batch extract features for {len(candidates)} candidates ...")
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    feats = []
    n_fail = 0
    for doc in nlp.pipe(candidates, batch_size=128):
        try:
            # extract_clause_features_from_doc 自动转为 doc
            from extract_clause_features_single_query import extract_clause_features_from_doc
            f = extract_clause_features_from_doc(doc, doc.text)
            row = [float(f[name]) for name in FEATURE_NAMES]
        except Exception as e:
            n_fail += 1
            row = [0.0] * INPUT_DIM
        feats.append(row)
    log(f"    parsed={len(candidates) - n_fail}, failed={n_fail}")
    X = np.asarray(feats, dtype=np.float64)
    X_scaled = scaler.transform(X)
    return X_scaled


def encode_to_mu(encoder: SentenceEncoder, X_scaled: np.ndarray) -> np.ndarray:
    """encoder forward → mu (均值). 入参 shape [N, 20]."""
    with torch.no_grad():
        X_t = torch.as_tensor(X_scaled, dtype=torch.float32, device=DEVICE)
        mu, _logvar, _recon = encoder(X_t)
        return mu.detach().cpu().numpy()


def get_user_mixed_mu(user_table: UserDistributionTableDisentangled, user_idx: int) -> np.ndarray:
    """对 user u, 拿 disentangle mode 的 user_mu (= style + offset)."""
    with torch.no_grad():
        all_idx = torch.arange(user_table.num_users, device=DEVICE)
        user_mu, _logvar = user_table(all_idx)
        return user_mu[user_idx].detach().cpu().numpy()


def main() -> None:
    log = lambda m: print(f"[rank] {m}", flush=True)
    log("=" * 70)
    log("Generate-then-Rank: 用 VADES user_mu 给候选打分排序")
    log("=" * 70)

    scaler = fit_scaler_from_training_sentences()
    encoder, user_table = build_model()
    user_profiles, n_users = load_user_profiles()
    user_id_to_idx = {uid: i for i, uid in enumerate(sorted(user_profiles.keys()))}
    log(f"user_id_to_idx 字典建好")

    test_cases = load_test_cases()
    log(f"test_cases={len(test_cases)}")
    candidates = load_candidates()
    log(f"候选 (user, asin) = {len(candidates)} 个")

    SCORED_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCORED_OUT.unlink(missing_ok=True)
    TOP1_OUT.parent.mkdir(parents=True, exist_ok=True)
    TOP1_OUT.unlink(missing_ok=True)

    # 提前解 user_mu (disentangle: style + offset 直接), 300 用户的 packed 矩阵
    with torch.no_grad():
        all_idx = torch.arange(user_table.num_users, device=DEVICE)
        um, _ul = user_table(all_idx)
        user_mu_table = um.cpu().numpy()
    log(f"user_mu_table shape={user_mu_table.shape}")

    n_total_scored = 0
    n_total_top1 = 0
    score_stats = defaultdict(list)
    with SCORED_OUT.open("w", encoding="utf-8") as f_score, TOP1_OUT.open("w", encoding="utf-8") as f_top1:
        for case_idx, ((uid, asin), cands) in enumerate(candidates.items()):
            case_meta = test_cases.get((uid, asin), {})
            gt_reviews: list[str] = case_meta.get("gt_reviews", [])
            user_idx = user_id_to_idx[uid]
            if user_idx >= n_users:
                continue
            user_mu = user_mu_table[user_idx]

            X_scaled = extract_features_for_candidates(cands, scaler)
            mu_query = encode_to_mu(encoder, X_scaled)
            # 距离 (欧式)
            diff = mu_query - user_mu[None, :]
            dist_sq = (diff ** 2).sum(axis=1)
            scores = -dist_sq  # 越大越像

            # 写 scored.jsonl
            for cand, mu, sc in zip(cands, mu_query, scores):
                f_score.write(json.dumps({
                    "user_id": uid,
                    "asin": asin,
                    "candidate": cand,
                    "mu_query": mu.tolist(),
                    "score": float(sc),
                    "dist_sq": float(-sc),
                }, ensure_ascii=False) + "\n")
                score_stats[uid].append(float(sc))
                n_total_scored += 1

            # 选 top-1
            top1_idx = int(np.argmax(scores))
            top1_record = {
                "user_id": uid,
                "asin": asin,
                "top1_candidate": cands[top1_idx],
                "top1_score": float(scores[top1_idx]),
                "top1_dist_sq": float(-scores[top1_idx]),
                "n_candidates": len(cands),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "score_max": float(scores.max()),
                "score_min": float(scores.min()),
                "score_median": float(np.median(scores)),
            }
            f_top1.write(json.dumps(top1_record, ensure_ascii=False) + "\n")
            n_total_top1 += 1
            if case_idx < 5 or case_idx % 10 == 0:
                log(f"  case={case_idx+1}/{len(candidates)} ({uid[:8]}...): n_cands={len(cands)}, "
                    f"top1='{cands[top1_idx][:60]}', score={scores[top1_idx]:.3f}")

    log(f"完成: scored={n_total_scored}, top1={n_total_top1}")
    log(f"已写入 {SCORED_OUT}")
    log(f"已写入 {TOP1_OUT}")


if __name__ == "__main__":
    main()
