#!/usr/bin/env python3
"""Raw-space VADES candidate scorer — Mahalanobis 后约束器.

输入:
  - candidates.jsonl: 每行 {user_id, asin, candidate_query, attrs?, reference_query?}
  - {TAG}_user_profiles.jsonl (vades_prototype_3000u_v6_raw): 提供 user_mu / user_logvar
  - {TAG}_sentences.jsonl: 提供 feat_mean / feat_std (标准化参数, 从 training set 全句统计)

输出:
  - scored_candidates.jsonl: 每行加 {feat, maha_target, maha_min_cross, rank, attr_pass, semantic_sim, calibrated}
  - summary.json: 整体指标 (avg maha, pass rate, etc.)

设计要点:
  - candidate features 提取复用 extract_clause_features_single_query.py::extract_clause_features
  - 标准化用训练时统计的 feat_mean / feat_std (与 build_raw_space_vades.py 保持一致)
  - 距离: Mahalanobis 用 target 用户的 user_logvar 作为协方差 (Bayesian 后验)
  - 属性完整性: 5 个字段 [Brand, Item Weight, Product Dimensions, Color, Material] 全部出现
  - 语义相似度: token overlap (Jaccard) + character-level overlap, 不调用 LLM
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
from extract_clause_features_single_query import extract_clause_features  # noqa: E402

VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
TAG = "vades_prototype_3000u_v6_raw"
USER_PROFILE_FILE = VADES_DIR / f"{TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{TAG}_sentences.jsonl"

# 属性完整性: 5 个字段
ATTR_FIELDS = ["Brand", "Item Weight", "Product Dimensions", "Color", "Material"]


def load_user_profiles() -> Tuple[Dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 user profiles + 训练集的标准化参数."""
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    user_mu = np.array([p["user_mu"] for p in profiles], dtype=np.float32)
    user_logvar = np.array([p["user_logvar"] for p in profiles], dtype=np.float32)

    # 训练集统计的标准化参数
    feat_rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            feat_rows.append([float(r["features"][k]) for k in list(r["features"].keys())])
    feat_array = np.array(feat_rows, dtype=np.float32)
    feat_mean = feat_array.mean(axis=0)
    feat_std = feat_array.std(axis=0) + 1e-9
    return user_id_to_idx, user_mu, user_logvar, feat_mean, feat_std


def extract_features(query_text: str, feature_names: List[str]) -> np.ndarray:
    """从 query text 提取 20d features; 返回标准化后的 20d 向量."""
    feat_dict = extract_clause_features(query_text)
    # 跟训练 feature_names 对齐顺序
    return np.array([float(feat_dict[name]) for name in feature_names], dtype=np.float32)


def maha_one_target(query_norm: np.ndarray, user_mu_n: np.ndarray, user_logvar: np.ndarray) -> float:
    """D(q, target_u) = Σ_k (q[k] - mu[k])^2 / exp(logvar[k])."""
    diff = query_norm - user_mu_n
    return float(((diff ** 2) * np.exp(-user_logvar)).sum())


def check_attr_pass(query_text: str, attrs: Dict[str, str]) -> Tuple[bool, List[str]]:
    """属性完整性: query 文本中提到 5 个 attribute 的 value.

    Returns:
        pass_all: bool
        missing: list of missing attribute keys
    """
    q_lower = query_text.lower()
    missing = []
    for k in ATTR_FIELDS:
        v = attrs.get(k, "").strip()
        if not v:
            missing.append(k)
            continue
        # 取 v 的首个有意义 token 作为 needle (>=3 字符)
        # 对于 "1.5 pounds" 这种取 "pounds"; 对于 "Blue" 取 "blue"
        v_clean = re.sub(r"[^a-zA-Z0-9 ]", " ", v).strip()
        tokens = [t for t in v_clean.split() if len(t) >= 3]
        if not tokens:
            missing.append(k)
            continue
        # 至少 1 个 token 出现在 query 中
        if not any(t.lower() in q_lower for t in tokens):
            missing.append(k)
    return len(missing) == 0, missing


def semantic_similarity(q1: str, q2: str) -> float:
    """简单的 token Jaccard 相似度 + 字符级 Jaccard 加权."""
    if not q1 or not q2:
        return 0.0
    toks1 = set(re.findall(r"\w+", q1.lower()))
    toks2 = set(re.findall(r"\w+", q2.lower()))
    if not toks1 or not toks2:
        return 0.0
    tok_jacc = len(toks1 & toks2) / len(toks1 | toks2)
    # 字符级 Jaccard (3-grams)
    def ngrams(s, n=3):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    ng1 = ngrams(q1.lower())
    ng2 = ngrams(q2.lower())
    if not ng1 or not ng2:
        char_jacc = 0.0
    else:
        char_jacc = len(ng1 & ng2) / len(ng1 | ng2)
    return 0.6 * tok_jacc + 0.4 * char_jacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True,
                    help="candidates.jsonl, 每行 {user_id, asin, candidate_query, attrs?, reference_query?}")
    ap.add_argument("--output", type=Path, required=True,
                    help="输出 scored jsonl 路径")
    ap.add_argument("--summary", type=Path, default=None,
                    help="输出 summary json 路径 (可选)")
    ap.add_argument("--limit", type=int, default=None,
                    help="限制处理候选数 (调试用)")
    args = ap.parse_args()

    log = lambda m: print(f"[maha-scorer] {m}", flush=True)
    log("=" * 70)
    log(f"Candidate Mahalanobis scorer")
    log("=" * 70)
    log(f"输入: {args.candidates}")

    # === Load profiles ===
    user_id_to_idx, user_mu, user_logvar, feat_mean, feat_std = load_user_profiles()
    log(f"  num_users: {len(user_id_to_idx)}")
    log(f"  user_mu shape: {user_mu.shape}")

    # 标准化 user_mu (跟 query 标准化一致)
    user_mu_norm = (user_mu - feat_mean) / feat_std
    log(f"  feat_mean range: [{feat_mean.min():.2f}, {feat_mean.max():.2f}]")
    log(f"  feat_std range: [{feat_std.min():.2f}, {feat_std.max():.2f}]")

    # === Process candidates ===
    feature_names = list(json.loads(SENTENCE_FILE.open().readline())["features"].keys())
    log(f"  feature_names: {feature_names[:5]}... ({len(feature_names)} dims)")

    n_total = 0
    n_attr_pass = 0
    n_extracted = 0
    maha_target_list = []
    maha_min_cross_list = []
    semantic_sim_list = []
    rank_list = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.candidates.open() as fin, args.output.open("w") as fout:
        for line in fin:
            if args.limit and n_total >= args.limit:
                break
            row = json.loads(line)
            n_total += 1
            q = row.get("candidate_query", "")
            target_uid = row.get("user_id", "")
            target_idx = user_id_to_idx.get(target_uid, -1)
            attrs = row.get("attrs", {})
            ref_q = row.get("reference_query", "")

            # === 1. 提取 features ===
            try:
                feat_raw = extract_features(q, feature_names)
                n_extracted += 1
            except Exception as e:
                fout.write(json.dumps({
                    **row,
                    "error": f"feature_extraction_failed: {type(e).__name__}: {e}",
                }) + "\n")
                continue
            feat_norm = (feat_raw - feat_mean) / feat_std

            # === 2. Mahalanobis 距离 (target + cross) ===
            if target_idx >= 0:
                maha_target = maha_one_target(feat_norm, user_mu_norm[target_idx], user_logvar[target_idx])
                # cross: vs 其他用户 (用各自的 user_logvar)
                diff_cross = feat_norm[None, :] - user_mu_norm  # [U, 20]
                maha_all = (diff_cross ** 2 * np.exp(-user_logvar)).sum(axis=1)  # [U]
                maha_all = maha_all.copy()
                maha_all[target_idx] = np.inf  # 排除 target
                maha_min_cross = float(maha_all.min())
                rank = int((maha_all < maha_target).sum()) + 1  # target 排在多少名 (1-indexed)
            else:
                maha_target = None
                maha_min_cross = None
                rank = None

            # === 3. 属性完整性 ===
            attr_pass, missing = check_attr_pass(q, attrs)
            if attr_pass:
                n_attr_pass += 1

            # === 4. 语义相似度 ===
            sem_sim = semantic_similarity(q, ref_q) if ref_q else None

            # === 记录 ===
            scored = {
                **row,
                "feat_norm": feat_norm.tolist(),
                "maha_target": maha_target,
                "maha_min_cross": maha_min_cross,
                "rank_in_target_user_space": rank,
                "attr_pass": attr_pass,
                "attr_missing": missing,
                "semantic_sim": sem_sim,
                "calibrated": (maha_target is not None and maha_target < maha_min_cross if maha_min_cross is not None else False),
            }
            fout.write(json.dumps(scored) + "\n")

            if maha_target is not None:
                maha_target_list.append(maha_target)
                maha_min_cross_list.append(maha_min_cross)
                rank_list.append(rank)
            if sem_sim is not None:
                semantic_sim_list.append(sem_sim)

    # === Summary ===
    summary = {
        "n_total": n_total,
        "n_features_extracted": n_extracted,
        "n_attr_pass": n_attr_pass,
        "attr_pass_rate": n_attr_pass / n_total if n_total else 0,
    }
    if maha_target_list:
        summary["maha_target_mean"] = float(np.mean(maha_target_list))
        summary["maha_target_std"] = float(np.std(maha_target_list))
        summary["maha_min_cross_mean"] = float(np.mean(maha_min_cross_list))
        summary["rank_median"] = float(np.median(rank_list))
        summary["rank_mean"] = float(np.mean(rank_list))
        summary["rank_p25"] = float(np.percentile(rank_list, 25))
        summary["rank_p75"] = float(np.percentile(rank_list, 75))
        summary["calibrated_pass_rate"] = float(np.mean([t < c for t, c in zip(maha_target_list, maha_min_cross_list)]))
    if semantic_sim_list:
        summary["semantic_sim_mean"] = float(np.mean(semantic_sim_list))

    log("=" * 70)
    log("汇总")
    log("=" * 70)
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"已写入 {args.output}")

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        log(f"已写入 {args.summary}")


if __name__ == "__main__":
    main()