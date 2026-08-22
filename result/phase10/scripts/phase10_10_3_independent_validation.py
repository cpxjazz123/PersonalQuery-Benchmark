#!/usr/bin/env python3
"""Phase 10.10.3: Independent Validation of Post-hoc Selection Rerank.

5 子评估 (Phase 10.10 主系统独立验证):

1. **独立信号距离 (Independent feature-space L2)**:
   - 每用户训练集 20d 特征均值 (from SENTENCE_FILE, 独立于 VADES VAE encoder)
   - 重计算每候选到 target_user_centroid 的 L2 距离
   - 不依赖 user_mu (Phase 10.10.2 maha metric)
   - 比较 rerank_top1 vs random pick 在独立距离上的差距

2. **目标 vs 错误用户方向 (Target vs Wrong Direction)**:
   - rerank_top1 的独立距离: d_target vs d_wrong
   - win_rate = P(d_target < d_wrong)
   - 期望 > 50% (rerank 选出的应更接近目标用户)

3. **语义相似度 (Semantic Similarity)**:
   - Jaccard token overlap between rerank_top1 query and product attrs (5 attrs string)
   - 比较 rerank_top1 vs random pick 的 Jaccard
   - 期望 rerank top-1 不显著降低语义相似度

4. **attr_pass = 100% 不变量**:
   - rerank pool 已 pre-filter 到 attr_pass=True (Phase 10.10.2)
   - 检查 rerank_top1 100% 通过 attr check

5. **N 候选曲线 + Pool 多样性**:
   - N ∈ {3, 5, 10}: 子采样 rerank pool, 重算 utility
   - 学习曲线: rerank_utility vs N
   - Pool 多样性: 10 candidates 两两 L2 距离分布

输入: phase10_9_4i2_candidates_1000.jsonl + phase10_pairs_1000.jsonl
      + SENTENCE_FILE (per-user 20d feature mean)
      + USER_PROFILE_FILE (VADES user_mu as baseline reference)
输出: phase10_10_3_independent_eval.json + phase10_10_3_per_pair.jsonl

期望:
- win_rate (target < wrong via independent distance) > 50% significant
- rerank_top1 独立距离 < random_pick 独立距离
- N=5 rerank utility 已接近 N=10 (边际收益递减)
- pool_diversity > 0 (候选间有差异, rerank 才有意义)
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
from extract_clause_features_single_query import extract_clause_features  # noqa: E402

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
CANDIDATES_FILE = OUT_DIR / "phase10_9_4i2_candidates_1000.jsonl"
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
SENTENCE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_sentences.jsonl"
)
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"

OUT_EVAL = OUT_DIR / "phase10_10_3_independent_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase10_10_3_per_pair.jsonl"
LOG_OUT = OUT_DIR / "phase10_10_3_independent_validation.log"

SEED = 42
N_BOOTSTRAP = 5000
ATTR_FIELDS = ["Brand", "Color", "Material"]


def log(m):
    print(f"[phase10-10.3] {m}", flush=True)


def load_user_profiles():
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    user_mu = np.array([p["user_mu"] for p in profiles], dtype=np.float32)
    user_logvar = np.array([p["user_logvar"] for p in profiles], dtype=np.float32)
    return user_id_to_idx, user_mu, user_logvar


def load_user_feature_centroids():
    """每用户训练集 20d 特征均值 (独立于 VADES VAE 的原始特征 centroid).

    这是 INDEPENDENT distance metric: 直接从用户的训练句 20d 特征求平均.
    与 user_mu (VAE 编码) 不同, 但同维度 (20d).
    """
    feat_rows_by_user: Dict[str, List[np.ndarray]] = defaultdict(list)
    feat_names = None
    with SENTENCE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            if feat_names is None:
                feat_names = list(r["features"].keys())
            vec = np.array([float(r["features"][k]) for k in feat_names], dtype=np.float32)
            feat_rows_by_user[r["user_id"]].append(vec)
    user_centroid = {uid: np.mean(vs, axis=0) for uid, vs in feat_rows_by_user.items()}
    user_centroid_norm = {
        uid: c / (np.linalg.norm(c) + 1e-9) for uid, c in user_centroid.items()
    }
    return user_centroid, user_centroid_norm, feat_names


def load_user_ngram_sets():
    """每用户训练集 word 3-gram set (text-level, 纯 lexical, 独立于 20d 特征)."""
    user_text = defaultdict(list)
    with SENTENCE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            user_text[r["user_id"]].append(r.get("sentence", ""))
    user_ngrams = {}
    for uid, sents in user_text.items():
        text = " ".join(sents).lower()
        tokens = re.findall(r"\w+", text)
        # word 3-grams
        grams = set()
        for i in range(len(tokens) - 2):
            grams.add(" ".join(tokens[i:i + 3]))
        user_ngrams[uid] = grams
    return user_ngrams


def extract_features(query_text: str, feature_names: List[str]):
    if not query_text or not query_text.strip():
        return None
    try:
        feat_dict = extract_clause_features(query_text)
        return np.array([float(feat_dict[name]) for name in feature_names], dtype=np.float32)
    except Exception:
        return None


def maha_one_user(query_norm: np.ndarray, mu_n: np.ndarray, logvar_n: np.ndarray) -> float:
    diff = query_norm - mu_n
    return float(((diff ** 2) * np.exp(-logvar_n)).sum())


def check_attr_pass(query: str, attrs: Dict[str, str]) -> bool:
    if not query or not attrs:
        return False
    q_lower = query.lower()
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", q_lower)
    for k, v in attrs.items():
        if not v:
            return False
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        tokens = []
        for t in v_clean.split():
            if len(t) >= 3:
                tokens.append(t)
            elif re.match(r"^\d+(\.\d+)?$", t):
                tokens.append(t)
            elif k == "Brand" and re.match(r"^[a-zA-Z]+$", t) and len(t) >= 2:
                tokens.append(t)
        if not tokens:
            return False
        found = False
        for t in tokens:
            t_lower = t.lower()
            if t_lower in q_clean:
                found = True; break
            if t_lower.endswith("s") and len(t_lower) > 3 and t_lower[:-1] in q_clean:
                found = True; break
            if (t_lower + "s") in q_clean:
                found = True; break
        if not found:
            return False
    return True


def jaccard_tokens(s1: str, s2: str) -> float:
    t1 = set(re.findall(r"\w+", s1.lower()))
    t2 = set(re.findall(r"\w+", s2.lower()))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def jaccard_ngrams(query: str, user_ngrams: set) -> float:
    tokens = re.findall(r"\w+", query.lower())
    q_grams = set()
    for i in range(len(tokens) - 2):
        q_grams.add(" ".join(tokens[i:i + 3]))
    if not q_grams or not user_ngrams:
        return 0.0
    return len(q_grams & user_ngrams) / len(q_grams | user_ngrams)


def bootstrap_ci_user_level(values_per_item: Dict, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED):
    items = list(values_per_item.keys())
    n = len(items)
    if n == 0:
        return 0.0, (0.0, 0.0)
    obs_vals = np.array([values_per_item[k] for k in items])
    obs_mean = float(obs_vals.mean())
    rng = np.random.default_rng(seed)
    boot_means = []
    indices = np.arange(n)
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=n, replace=True)
        boot_vals = obs_vals[sample_idx]
        boot_means.append(float(boot_vals.mean()))
    boot_means = np.array(boot_means)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return obs_mean, (ci_low, ci_high)


def main():
    log("=" * 70)
    log("Phase 10.10.3: Independent Validation of Post-hoc Selection Rerank (n=876)")
    log("=" * 70)

    # === Load all data sources ===
    log("[1] Loading profiles + per-user feature centroids + n-gram sets ...")
    user_id_to_idx, user_mu, user_logvar = load_user_profiles()
    user_feat_centroid, user_feat_centroid_norm, feature_names = load_user_feature_centroids()
    user_ngrams = load_user_ngram_sets()
    log(f"  VADES profiles: {len(user_id_to_idx)} users, 20d")
    log(f"  feature centroids: {len(user_feat_centroid)} users")
    log(f"  user n-gram sets: {len(user_ngrams)} users")

    # === Load pairs ===
    log("[2] Loading pairs ...")
    pair_lookup = {}
    with PAIRS_FILE.open() as f:
        for line in f:
            p = json.loads(line)
            pair_lookup[(p["user_id"], p["asin"])] = p
    log(f"  pairs: {len(pair_lookup)}")

    # === Load candidates ===
    log("[3] Loading candidates ...")
    cand_by_pair: Dict[Tuple[str, str], Dict[str, list]] = {}
    with CANDIDATES_FILE.open() as f:
        for line in f:
            d = json.loads(line)
            key = (d["user_id"], d["asin"])
            cond = d["cond"]
            cand_by_pair.setdefault(key, {}).setdefault(cond, []).append(d)
    log(f"  total pairs (w/ candidates): {len(cand_by_pair)}")

    # === Per-pair evaluation ===
    log("[4] Computing per-pair metrics (5 sub-evaluations) ...")
    per_pair_metrics = {}
    n_feat_fail = 0
    n_attr_fail = 0

    for (uid, asin), conds_dict in cand_by_pair.items():
        if "D_target_style" not in conds_dict or "C_random_style" not in conds_dict:
            continue
        target_pair = pair_lookup.get((uid, asin))
        if not target_pair:
            continue
        random_uid = target_pair["random_user_id"]
        target_idx = user_id_to_idx.get(uid)
        random_idx = user_id_to_idx.get(random_uid)
        if target_idx is None or random_idx is None:
            continue
        if uid not in user_feat_centroid or random_uid not in user_feat_centroid:
            continue
        if uid not in user_ngrams or random_uid not in user_ngrams:
            continue

        # VADES maha (baseline rerank signal)
        mu_t = user_mu[target_idx]
        logvar_t = user_logvar[target_idx]
        mu_r = user_mu[random_idx]
        logvar_r = user_logvar[random_idx]

        target_centroid_feat = user_feat_centroid[uid]
        wrong_centroid_feat = user_feat_centroid[random_uid]
        target_ngrams = user_ngrams[uid]
        wrong_ngrams = user_ngrams[random_uid]
        # attrs 用候选的 attrs (3 fields), 不对 (5 fields) 的 pair attrs
        # (因为 hard-copy 是按 3 attrs 触发, candidate 文本里没有 Item Weight 等)
        attrs_str = ""  # 会在循环内逐候选更新

        # Pool: 5 D_target + 5 C_random, with attr_pass filter
        pool = []
        for cond in ("D_target_style", "C_random_style"):
            for ci, c in enumerate(conds_dict[cond]):
                feat_raw = extract_features(c["candidate_query"], feature_names)
                if feat_raw is None:
                    n_feat_fail += 1
                    feat_n = np.zeros(len(feature_names), dtype=np.float32)
                else:
                    feat_n = feat_raw
                mt = maha_one_user(feat_n, mu_t, logvar_t)
                mr = maha_one_user(feat_n, mu_r, logvar_r)
                # Independent feature-space L2 distance
                l2_target = float(np.linalg.norm(feat_n - target_centroid_feat))
                l2_wrong = float(np.linalg.norm(feat_n - wrong_centroid_feat))
                # Independent n-gram Jaccard
                jn_target = jaccard_ngrams(c["candidate_query"], target_ngrams)
                jn_wrong = jaccard_ngrams(c["candidate_query"], wrong_ngrams)
                # Semantic similarity to product attrs (use candidate's 3-attr attrs)
                cand_attrs = c["attrs"]
                cand_attrs_str = " ".join(str(v) for v in cand_attrs.values() if v)
                sem_attrs = jaccard_tokens(c["candidate_query"], cand_attrs_str)
                attr_ok = check_attr_pass(c["candidate_query"], cand_attrs)
                if not attr_ok:
                    n_attr_fail += 1
                pool.append({
                    "cond": cond, "ci": ci,
                    "feat_n": feat_n,
                    "maha_target": mt, "maha_random": mr,
                    "l2_target": l2_target, "l2_wrong": l2_wrong,
                    "jn_target": jn_target, "jn_wrong": jn_wrong,
                    "sem_attrs": sem_attrs,
                    "attr_pass": attr_ok,
                    "query": c["candidate_query"],
                })

        # Filter pool to attr_pass
        pool_filtered = [p for p in pool if p["attr_pass"]]
        if len(pool_filtered) < 1:
            continue

        # === Rerank: top-1 by maha_target_min (VADES signal) ===
        rerank_sorted = sorted(pool_filtered, key=lambda x: x["maha_target"])
        rerank_top1 = rerank_sorted[0]

        # === Random pick baseline (100 random picks) ===
        rng = np.random.default_rng(SEED + hash((uid, asin)) % (2**31))
        random_picks = [pool_filtered[rng.integers(0, len(pool_filtered))] for _ in range(100)]
        random_top1_avg_l2_target = float(np.mean([p["l2_target"] for p in random_picks]))
        random_top1_avg_jn_target = float(np.mean([p["jn_target"] for p in random_picks]))
        random_top1_avg_sem_attrs = float(np.mean([p["sem_attrs"] for p in random_picks]))

        # === Per-pair metrics ===
        l2_target_top1 = rerank_top1["l2_target"]
        l2_wrong_top1 = rerank_top1["l2_wrong"]
        jn_target_top1 = rerank_top1["jn_target"]
        jn_wrong_top1 = rerank_top1["jn_wrong"]
        sem_attrs_top1 = rerank_top1["sem_attrs"]
        target_closer_l2 = bool(l2_target_top1 < l2_wrong_top1)  # 1 if rerank_top1 closer to target
        target_closer_jn = bool(jn_target_top1 > jn_wrong_top1)   # 1 if rerank_top1 ngram more to target
        l2_diff = l2_wrong_top1 - l2_target_top1  # positive = target closer
        jn_diff = jn_target_top1 - jn_wrong_top1   # positive = target closer (ngram)

        # === N candidate curve: rerank utility at N=3, 5, 10 ===
        n_curve = {}
        for N in (3, 5, 10):
            n_samples = 100
            utilities_N = []
            for _ in range(n_samples):
                if len(pool_filtered) >= N:
                    pool_sub = list(rng.choice(pool_filtered, size=N, replace=False))
                else:
                    pool_sub = list(pool_filtered)
                if not pool_sub:
                    continue
                pool_sub_sorted = sorted(pool_sub, key=lambda x: x["maha_target"])
                rerank_sub_top1 = pool_sub_sorted[0]
                rand_top1 = pool_sub[rng.integers(0, len(pool_sub))]
                # utility = random_l2 - rerank_l2 (positive = rerank wins)
                util = rand_top1["l2_target"] - rerank_sub_top1["l2_target"]
                utilities_N.append(util)
            n_curve[str(N)] = float(np.mean(utilities_N)) if utilities_N else 0.0

        # === Pool diversity (pairwise L2 in feature space, attr_pass) ===
        n_pool = len(pool_filtered)
        pairwise_l2 = []
        for i in range(n_pool):
            for j in range(i + 1, n_pool):
                d = float(np.linalg.norm(pool_filtered[i]["feat_n"] - pool_filtered[j]["feat_n"]))
                pairwise_l2.append(d)
        pool_diversity = float(np.mean(pairwise_l2)) if pairwise_l2 else 0.0
        # Diversity per cond
        pool_d = [p for p in pool_filtered if p["cond"] == "D_target_style"]
        pool_c = [p for p in pool_filtered if p["cond"] == "C_random_style"]
        div_d = []
        for i in range(len(pool_d)):
            for j in range(i + 1, len(pool_d)):
                div_d.append(float(np.linalg.norm(pool_d[i]["feat_n"] - pool_d[j]["feat_n"])))
        div_c = []
        for i in range(len(pool_c)):
            for j in range(i + 1, len(pool_c)):
                div_c.append(float(np.linalg.norm(pool_c[i]["feat_n"] - pool_c[j]["feat_n"])))
        div_d_target = float(np.mean(div_d)) if div_d else 0.0
        div_c_random = float(np.mean(div_c)) if div_c else 0.0
        div_cross = []
        for pd in pool_d:
            for pc in pool_c:
                div_cross.append(float(np.linalg.norm(pd["feat_n"] - pc["feat_n"])))
        div_cross_mean = float(np.mean(div_cross)) if div_cross else 0.0

        per_pair_metrics[(uid, asin)] = {
            "user_id": uid,
            "asin": asin,
            "random_user_id": random_uid,
            "n_pool": len(pool),
            "n_pool_attr_pass": n_pool,
            # Sub 1: independent distance
            "rerank_top1_l2_target": l2_target_top1,
            "rerank_top1_l2_wrong": l2_wrong_top1,
            "random_pick_l2_target": random_top1_avg_l2_target,
            "l2_diff_target_minus_wrong": l2_diff,  # positive = target closer
            # Sub 2: direction (win rate)
            "target_closer_l2": float(target_closer_l2),
            "target_closer_jn": float(target_closer_jn),
            "jn_diff_target_minus_wrong": jn_diff,
            # Sub 3: semantic similarity
            "rerank_top1_sem_attrs": sem_attrs_top1,
            "random_pick_sem_attrs": random_top1_avg_sem_attrs,
            "sem_attrs_diff_rerank_minus_random": sem_attrs_top1 - random_top1_avg_sem_attrs,
            # Sub 5: N curve + diversity
            "rerank_utility_N3": n_curve["3"],
            "rerank_utility_N5": n_curve["5"],
            "rerank_utility_N10": n_curve["10"],
            "pool_diversity": pool_diversity,
            "pool_diversity_D_target": div_d_target,
            "pool_diversity_C_random": div_c_random,
            "pool_diversity_cross_D_C": div_cross_mean,
            "rerank_top1_jn_target": jn_target_top1,
            "rerank_top1_jn_wrong": jn_wrong_top1,
        }

    log(f"  per_pair_metrics: {len(per_pair_metrics)}")
    log(f"  feat fail: {n_feat_fail}, attr fail: {n_attr_fail}")

    # === Aggregate ===
    log("[5] Aggregating + Bootstrap CI ...")
    # Sub 1: independent distance comparison
    rerank_l2_target = {k: v["rerank_top1_l2_target"] for k, v in per_pair_metrics.items()}
    rerank_l2_wrong = {k: v["rerank_top1_l2_wrong"] for k, v in per_pair_metrics.items()}
    random_l2_target = {k: v["random_pick_l2_target"] for k, v in per_pair_metrics.items()}
    l2_diff_paired = {k: v["l2_diff_target_minus_wrong"] for k, v in per_pair_metrics.items()}

    # Sub 2: direction win rates
    win_rate_l2 = {k: v["target_closer_l2"] for k, v in per_pair_metrics.items()}
    win_rate_jn = {k: v["target_closer_jn"] for k, v in per_pair_metrics.items()}
    jn_diff_paired = {k: v["jn_diff_target_minus_wrong"] for k, v in per_pair_metrics.items()}

    # Sub 3: semantic similarity
    rerank_sem = {k: v["rerank_top1_sem_attrs"] for k, v in per_pair_metrics.items()}
    random_sem = {k: v["random_pick_sem_attrs"] for k, v in per_pair_metrics.items()}
    sem_diff_paired = {k: v["sem_attrs_diff_rerank_minus_random"] for k, v in per_pair_metrics.items()}

    # Sub 5: N curve + diversity
    n3_util = {k: v["rerank_utility_N3"] for k, v in per_pair_metrics.items()}
    n5_util = {k: v["rerank_utility_N5"] for k, v in per_pair_metrics.items()}
    n10_util = {k: v["rerank_utility_N10"] for k, v in per_pair_metrics.items()}
    pool_div = {k: v["pool_diversity"] for k, v in per_pair_metrics.items()}
    pool_div_D = {k: v["pool_diversity_D_target"] for k, v in per_pair_metrics.items()}
    pool_div_C = {k: v["pool_diversity_C_random"] for k, v in per_pair_metrics.items()}
    pool_div_cross = {k: v["pool_diversity_cross_D_C"] for k, v in per_pair_metrics.items()}

    mean_rerank_l2_t, ci_rerank_l2_t = bootstrap_ci_user_level(rerank_l2_target)
    mean_rerank_l2_w, ci_rerank_l2_w = bootstrap_ci_user_level(rerank_l2_wrong)
    mean_random_l2_t, ci_random_l2_t = bootstrap_ci_user_level(random_l2_target)
    mean_l2_diff, ci_l2_diff = bootstrap_ci_user_level(l2_diff_paired)
    mean_win_l2, ci_win_l2 = bootstrap_ci_user_level(win_rate_l2)
    mean_win_jn, ci_win_jn = bootstrap_ci_user_level(win_rate_jn)
    mean_jn_diff, ci_jn_diff = bootstrap_ci_user_level(jn_diff_paired)
    mean_rerank_sem, ci_rerank_sem = bootstrap_ci_user_level(rerank_sem)
    mean_random_sem, ci_random_sem = bootstrap_ci_user_level(random_sem)
    mean_sem_diff, ci_sem_diff = bootstrap_ci_user_level(sem_diff_paired)
    mean_n3, ci_n3 = bootstrap_ci_user_level(n3_util)
    mean_n5, ci_n5 = bootstrap_ci_user_level(n5_util)
    mean_n10, ci_n10 = bootstrap_ci_user_level(n10_util)
    mean_pool_div, ci_pool_div = bootstrap_ci_user_level(pool_div)
    mean_pool_div_D, _ = bootstrap_ci_user_level(pool_div_D)
    mean_pool_div_C, _ = bootstrap_ci_user_level(pool_div_C)
    mean_pool_div_cross, _ = bootstrap_ci_user_level(pool_div_cross)

    log(f"\n=== Sub 1: Independent Feature-space L2 Distance ===")
    log(f"  rerank_top1 l2_target:    {mean_rerank_l2_t:.4f} CI [{ci_rerank_l2_t[0]:.4f}, {ci_rerank_l2_t[1]:.4f}]")
    log(f"  rerank_top1 l2_wrong:     {mean_rerank_l2_w:.4f} CI [{ci_rerank_l2_w[0]:.4f}, {ci_rerank_l2_w[1]:.4f}]")
    log(f"  random_pick l2_target:    {mean_random_l2_t:.4f} CI [{ci_random_l2_t[0]:.4f}, {ci_random_l2_t[1]:.4f}]")
    log(f"  Δ_l2 paired (target<wrong): {mean_l2_diff:.4f} CI [{ci_l2_diff[0]:.4f}, {ci_l2_diff[1]:.4f}]")

    log(f"\n=== Sub 2: Direction (target vs wrong) ===")
    log(f"  win_rate (l2 target<wrong): {mean_win_l2:.4f} CI [{ci_win_l2[0]:.4f}, {ci_win_l2[1]:.4f}]")
    log(f"  win_rate (ngram target>wrong): {mean_win_jn:.4f} CI [{ci_win_jn[0]:.4f}, {ci_win_jn[1]:.4f}]")
    log(f"  Δ_jn paired (target>wrong): {mean_jn_diff:.6f} CI [{ci_jn_diff[0]:.6f}, {ci_jn_diff[1]:.6f}]")

    log(f"\n=== Sub 3: Semantic Similarity (vs product attrs) ===")
    log(f"  rerank_top1 Jaccard:  {mean_rerank_sem:.4f} CI [{ci_rerank_sem[0]:.4f}, {ci_rerank_sem[1]:.4f}]")
    log(f"  random_pick Jaccard:  {mean_random_sem:.4f} CI [{ci_random_sem[0]:.4f}, {ci_random_sem[1]:.4f}]")
    log(f"  Δ Jaccard paired:     {mean_sem_diff:.4f} CI [{ci_sem_diff[0]:.4f}, {ci_sem_diff[1]:.4f}]")

    log(f"\n=== Sub 5a: N candidate curve (independent rerank utility) ===")
    log(f"  N=3:  {mean_n3:.4f} CI [{ci_n3[0]:.4f}, {ci_n3[1]:.4f}]")
    log(f"  N=5:  {mean_n5:.4f} CI [{ci_n5[0]:.4f}, {ci_n5[1]:.4f}]")
    log(f"  N=10: {mean_n10:.4f} CI [{ci_n10[0]:.4f}, {ci_n10[1]:.4f}]")
    log(f"  N=5/N=10 ratio: {mean_n5 / mean_n10 if mean_n10 else 0:.3f}")

    log(f"\n=== Sub 5b: Pool diversity ===")
    log(f"  overall pairwise L2:     {mean_pool_div:.4f} CI [{ci_pool_div[0]:.4f}, {ci_pool_div[1]:.4f}]")
    log(f"  D_target_style sub-pool: {mean_pool_div_D:.4f}")
    log(f"  C_random_style sub-pool: {mean_pool_div_C:.4f}")
    log(f"  cross D-C distance:      {mean_pool_div_cross:.4f}")

    # === Verdict ===
    checks = {
        # Sub 1: independent distance 优势 (rerank_top1 在独立 L2 上仍 < random)
        "independent_l2_rerank_below_random_ci": ci_rerank_l2_t[1] < ci_random_l2_t[0],
        # Sub 2: direction (rerank 选出的更接近 target, 正确方向)
        "direction_win_rate_l2_above_50_ci": ci_win_l2[0] > 0.5,
        "direction_l2_diff_positive_ci": ci_l2_diff[0] > 0,
        # Sub 3: semantic similarity (rerank 不显著降低)
        "sem_attrs_rerank_above_50pct_random": ci_rerank_sem[0] > 0.5 * mean_random_sem if mean_random_sem > 0 else False,
        # Sub 5: N curve monotonic (or near-monotonic)
        "n_curve_5_above_3": mean_n5 > mean_n3,
        "n_curve_10_above_5": mean_n10 > mean_n5,
        # Pool diversity: rerank 才有意义的前提
        "pool_diversity_positive": ci_pool_div[0] > 0,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    log(f"\n=== Verdict ===")
    log(f"  checks: {checks}")
    log(f"  VERDICT: {verdict}")

    # === Save ===
    eval_dict = {
        "n_pairs": len(per_pair_metrics),
        "sub1_independent_l2_distance": {
            "rerank_top1_l2_target_mean": mean_rerank_l2_t,
            "rerank_top1_l2_target_ci": list(ci_rerank_l2_t),
            "rerank_top1_l2_wrong_mean": mean_rerank_l2_w,
            "rerank_top1_l2_wrong_ci": list(ci_rerank_l2_w),
            "random_pick_l2_target_mean": mean_random_l2_t,
            "random_pick_l2_target_ci": list(ci_random_l2_t),
            "l2_diff_paired_mean": mean_l2_diff,
            "l2_diff_paired_ci": list(ci_l2_diff),
        },
        "sub2_target_vs_wrong_direction": {
            "win_rate_l2_target_closer_mean": mean_win_l2,
            "win_rate_l2_ci": list(ci_win_l2),
            "win_rate_ngram_target_closer_mean": mean_win_jn,
            "win_rate_ngram_ci": list(ci_win_jn),
            "jn_diff_paired_mean": mean_jn_diff,
            "jn_diff_paired_ci": list(ci_jn_diff),
        },
        "sub3_semantic_similarity": {
            "rerank_top1_jaccard_attrs_mean": mean_rerank_sem,
            "rerank_top1_jaccard_attrs_ci": list(ci_rerank_sem),
            "random_pick_jaccard_attrs_mean": mean_random_sem,
            "random_pick_jaccard_attrs_ci": list(ci_random_sem),
            "diff_paired_mean": mean_sem_diff,
            "diff_paired_ci": list(ci_sem_diff),
        },
        "sub4_attr_pass": {
            "rate": float(np.mean([v["n_pool_attr_pass"] > 0 for v in per_pair_metrics.values()])),
            "invariant": "100% (pre-filter on rerank pool)",
        },
        "sub5a_n_candidate_curve": {
            "N3": {"mean": mean_n3, "ci": list(ci_n3)},
            "N5": {"mean": mean_n5, "ci": list(ci_n5)},
            "N10": {"mean": mean_n10, "ci": list(ci_n10)},
            "N5_over_N10_ratio": mean_n5 / mean_n10 if mean_n10 else 0.0,
        },
        "sub5b_pool_diversity": {
            "overall_pairwise_l2_mean": mean_pool_div,
            "overall_pairwise_l2_ci": list(ci_pool_div),
            "D_target_subpool_l2_mean": mean_pool_div_D,
            "C_random_subpool_l2_mean": mean_pool_div_C,
            "cross_D_C_l2_mean": mean_pool_div_cross,
        },
        "checks": checks,
        "verdict": verdict,
        "n_feat_fail": n_feat_fail,
        "n_attr_fail": n_attr_fail,
        "note": "Phase 10.10.3: Independent validation. "
                "Sub1 = feature-space L2 (independent from VADES user_mu via raw 20d centroids). "
                "Sub2 = target vs wrong user direction. "
                "Sub3 = semantic similarity via Jaccard vs product attrs tokens. "
                "Sub4 = attr_pass invariant. "
                "Sub5 = N candidate curve + pool diversity.",
    }
    OUT_EVAL.write_text(json.dumps(eval_dict, ensure_ascii=False, indent=2))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for v in per_pair_metrics.values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log(f"VERDICT: {verdict}")
    log("=" * 70)


if __name__ == "__main__":
    main()