#!/usr/bin/env python3
"""Phase 1 — Syntax Prototype Extraction (T=34 user PCA48 whitened space).

用户指令 2026-08-30: K-sweep 已证 F3=86.9% 在 K=50/100/200 都 flat,
  瓶颈不是 pool 太小, 是 generation distribution 没覆盖真实用户句法空间.
  下一步: Syntax-Prototype Guided Generation.

本脚本(Phase 1):
  1. 收集 T=34 用户的真实历史句子 (sentences_for_rewrite_10k.jsonl, 104K 句)
  2. 抽取 182d features (sentences_318d_cache.jsonl.gz)
  3. F3_CoreStruct 过滤 + StandardScaler + PCA48 投影
     (用 user_gaussians.json 里的 scaler / pca 参数, 与 Stage 4 同空间)
  4. KMeans(K ∈ {8, 16, 32}) 在 PCA48 whitened space 聚类
  5. 对每个 centroid, 找最近 5 句, 抽 abstract structural pattern (dep tree depth,
     POS ratios, clause counts) → 用作 Phase 2 LLM prompt 的 abstract guidance
  6. 写 prototypes JSON: centroid + structural description + sample texts (anonymized,
     不带 user_id, 仅供 generation prompt 参考结构, 不直接复制)

**输入**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/sentences_for_rewrite_10k.jsonl
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/sentences_318d_cache.jsonl.gz
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_quality_strict_users.json
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians.json

**输出**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/syntax_prototypes_K{8,16,32}.json
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/syntax_prototypes_summary.json

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

SENT_IN = SCRATCH / "sentences_for_rewrite_10k.jsonl"
FEAT_CACHE = SCRATCH / "sentences_318d_cache.jsonl.gz"
STRICT_USERS = SCRATCH / "stage8_5_quality_strict_users.json"
GAUSSIANS = SCRATCH / "stage8_5_user_gaussians.json"

K_VALUES = [8, 16, 32]
N_SAMPLE_PER_PROTO = 5
RANDOM_STATE = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [prototype] {msg}", flush=True)


def feat_key(text: str) -> str:
    """Match the cache key format used elsewhere in the pipeline."""
    return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()


def load_sentences_for_strict_users() -> list[dict]:
    """Collect all sentences authored by T=34 strict users."""
    with open(STRICT_USERS, "r", encoding="utf-8") as f:
        strict_set = set(json.load(f)["users"])
    log(f"  T=34 strict users: {len(strict_set)}")
    sentences = []
    n_total = 0
    n_match = 0
    n_too_short = 0
    with open(SENT_IN, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_total += 1
            if rec["user_id"] not in strict_set:
                continue
            n_match += 1
            text = rec.get("sentence_text", "").strip()
            wc = rec.get("word_count", len(text.split()))
            if wc < 5 or wc > 60:  # match Stage 1 strict 5-layer filter upper bound
                n_too_short += 1
                continue
            sentences.append({
                "user_id": rec["user_id"],
                "asin": rec["asin"],
                "text": text,
                "word_count": wc,
            })
    log(f"  total sentences: {n_total}, T=34 user match: {n_match}, "
        f"after length filter (5≤wc≤60): {len(sentences)} (dropped {n_too_short})")
    return sentences


def load_features(texts: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Load 184d features from sentences_318d_cache.jsonl.gz for given texts.
    Returns (X [N, 184], matched_texts, missing_texts).
    """
    needed = set(feat_key(t) for t in texts)
    log(f"  needed features: {len(needed)} unique keys")
    feat_map: dict[str, dict] = {}
    t0 = time.time()
    n_skip = 0
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skip += 1
                continue
            k = rec.get("k", "")
            if k in needed and k not in feat_map:
                feat_map[k] = rec.get("v", {})
    log(f"  loaded cache in {time.time() - t0:.1f}s, matched: {len(feat_map)}/{len(needed)}, "
        f"skipped (corrupt): {n_skip}")
    return feat_map


def project_to_pca48(feat_map: dict[str, dict],
                     sentences: list[dict],
                     gauss_data: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Project sentences → F3 subset → StandardScaler → PCA48.
    Returns (X_pca48 [N, 48], X_whitened [N, 48], valid_idx).
    Whitened = (X_pca48 - 0) / sqrt(lambda)  (pca mean is in input space, not here).
    """
    # user_gaussians.json already stores scaler/pca fit on F3_103d (not 182d):
    #   scaler_mean/scale: 103d, pca_components: [48, 103], pca_mean: 103d
    # So we directly use these — no col_idx_f3 selection needed.
    fnames_f3 = gauss_data["fnames_f3"]  # 103 names (order matches scaler/pca)
    scaler_mean = np.array(gauss_data["scaler_mean"], dtype=np.float64)  # [103]
    scaler_scale = np.array(gauss_data["scaler_scale"], dtype=np.float64)  # [103]
    pca_mean_f3 = np.array(gauss_data["pca_mean"], dtype=np.float64)  # [103]
    pca_components_f3 = np.array(gauss_data["pca_components"], dtype=np.float64)  # [48, 103]
    pca_ev = np.array(gauss_data["pca_explained_variance"], dtype=np.float64)  # [48]
    sqrt_lambda = np.sqrt(pca_ev)

    X_raw = []
    valid_idx = []
    for i, s in enumerate(sentences):
        feats = feat_map.get(feat_key(s["text"]))
        if not feats:
            continue
        vec = np.array([float(feats.get(n, 0.0)) for n in fnames_f3], dtype=np.float64)
        X_raw.append(vec)
        valid_idx.append(i)
    if not X_raw:
        raise RuntimeError("no sentences had cached features")
    X_raw = np.stack(X_raw, axis=0)
    log(f"  F3 raw: {X_raw.shape}")

    X_scaled = (X_raw - scaler_mean) / scaler_scale
    X_pca = (X_scaled - pca_mean_f3) @ pca_components_f3.T  # [N, 48]
    X_white = X_pca / sqrt_lambda  # whitened space — what Mahalanobis uses
    log(f"  PCA48: {X_pca.shape}, whitened: {X_white.shape}")
    return X_pca, X_white, valid_idx


def extract_abstract_pattern(texts: list[str]) -> dict:
    """Extract abstract structural pattern from sample texts (no user_id,
    no verbatim content). Used as guidance for Phase 2 prompt.
    Returns counts/ratios of:
      - token counts (min/median/max)
      - sentence structure (avg n_clause, n_coord, n_subj)
      - opener type (first word: I/this/it/these/the)
      - clause composition (presence of advcl / relcl / ccomp / coord)
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    docs = list(nlp.pipe(texts, batch_size=64, n_process=4))

    n_tok_list = []
    n_clause_list = []
    n_coord_list = []
    n_advcl_list = []
    n_relcl_list = []
    n_ccomp_list = []
    opener_counts: dict[str, int] = {}

    for doc in docs:
        if len(doc) == 0:
            continue
        n_tok = len(doc)
        n_tok_list.append(n_tok)

        # clause subtree counts via dep labels
        n_clause = sum(1 for t in doc if t.dep_ in ("advcl", "relcl", "ccomp", "acl", "xcomp"))
        n_clause_list.append(n_clause)
        n_coord = sum(1 for t in doc if t.dep_ == "conj")
        n_coord_list.append(n_coord)
        n_advcl = sum(1 for t in doc if t.dep_ == "advcl")
        n_advcl_list.append(n_advcl)
        n_relcl = sum(1 for t in doc if t.dep_ == "relcl")
        n_relcl_list.append(n_relcl)
        n_ccomp = sum(1 for t in doc if t.dep_ == "ccomp")
        n_ccomp_list.append(n_ccomp)

        # opener type
        first = doc[0].text.lower().strip(".,!?")
        if first.startswith("i "):
            opener = "first_person_I"
        elif first in ("this", "these", "the"):
            opener = "demonstrative"
        elif first in ("it", "they"):
            opener = "pronoun_subject"
        elif first.endswith("ly") or first in ("really", "truly", "honestly"):
            opener = "adverb_fronted"
        else:
            opener = "other"
        opener_counts[opener] = opener_counts.get(opener, 0) + 1

    def stat(arr):
        if not arr:
            return None
        s = sorted(arr)
        n = len(s)
        return {"min": s[0], "median": s[n // 2], "max": s[-1], "mean": sum(s) / n}

    pattern = {
        "n_samples": len(docs),
        "n_tok": stat(n_tok_list),
        "n_clause": stat(n_clause_list),
        "n_coord": stat(n_coord_list),
        "n_advcl": stat(n_advcl_list),
        "n_relcl": stat(n_relcl_list),
        "n_ccomp": stat(n_ccomp_list),
        "opener_distribution": opener_counts,
    }
    return pattern


def cluster_and_describe(X_white: np.ndarray, X_pca: np.ndarray,
                         sentences: list[dict], valid_idx: list[int],
                         K: int) -> dict:
    """KMeans in whitened PCA48 space; per centroid: abstract pattern + 5 nearest
    sample texts (anonymized: drop user_id, keep asin + text)."""
    log(f"  KMeans K={K} on {X_white.shape} ...")
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_white)
    log(f"  KMeans done, cluster sizes: min={int(np.bincount(labels).min())}, "
        f"max={int(np.bincount(labels).max())}, "
        f"median={int(np.median(np.bincount(labels)))}")

    centroids_white = km.cluster_centers_

    prototypes = []
    for c in range(K):
        member_mask = labels == c
        member_indices = np.where(member_mask)[0]
        centroid_w = centroids_white[c]

        # Distance from each member to centroid (whitened)
        dists_to_centroid = np.linalg.norm(X_white[member_indices] - centroid_w, axis=1)
        # Nearest N
        nearest_local = np.argsort(dists_to_centroid)[:N_SAMPLE_PER_PROTO]
        nearest_global = member_indices[nearest_local]

        sample_texts = [sentences[valid_idx[i]]["text"] for i in nearest_global]

        pattern = extract_abstract_pattern(sample_texts)

        proto = {
            "cluster_id": c,
            "centroid_whitened": centroid_w.tolist(),
            "n_members": int(member_mask.sum()),
            "stats_in_pca48": {
                "member_mean_pca48": X_pca[member_indices].mean(axis=0).tolist(),
                "member_std_pca48": X_pca[member_indices].std(axis=0).tolist(),
            },
            "abstract_pattern": pattern,
            "sample_texts_anonymized": sample_texts,  # only text, no user_id
        }
        prototypes.append(proto)
        log(f"    cluster {c}: n={pattern['n_samples']} samples, "
            f"members={member_mask.sum()}, opener={pattern['opener_distribution']}")

    return {
        "K": K,
        "n_total_sentences": int(X_white.shape[0]),
        "KMeans_inertia": float(km.inertia_),
        "cluster_sizes": [int(x) for x in np.bincount(labels).tolist()],
        "prototypes": prototypes,
    }


def main():
    log("=== Phase 1 — Syntax Prototype Extraction ===")
    log(f"  K values: {K_VALUES}")
    log(f"  random_state: {RANDOM_STATE}")

    # ---- Load sentences ----
    log("\n=== Step 1: load T=34 user sentences ===")
    sentences = load_sentences_for_strict_users()

    # ---- Load features cache ----
    log("\n=== Step 2: load 184d features cache ===")
    feat_map = load_features([s["text"] for s in sentences])

    # ---- Load Gaussian metadata for scaler/PCA ----
    log("\n=== Step 3: load user_gaussians metadata ===")
    with open(GAUSSIANS, "r", encoding="utf-8") as f:
        gauss_data = json.load(f)
    log(f"  fnames_f3: {len(gauss_data['fnames_f3'])}, "
        f"feature_names_ordered: {len(gauss_data['feature_names_ordered'])}")

    # ---- Project to PCA48 whitened space ----
    log("\n=== Step 4: project to F3 + PCA48 whitened ===")
    X_pca, X_white, valid_idx = project_to_pca48(feat_map, sentences, gauss_data)

    # ---- Cluster for each K ----
    log(f"\n=== Step 5: KMeans K ∈ {K_VALUES} ===")
    results = {}
    for K in K_VALUES:
        log(f"\n  ---- K={K} ----")
        results[K] = cluster_and_describe(X_white, X_pca, sentences, valid_idx, K)
        out = SCRATCH / f"syntax_prototypes_K{K}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results[K], f, ensure_ascii=False, indent=2)
        log(f"  wrote → {out}")

    # ---- Aggregate summary ----
    log("\n=== Step 6: aggregate summary ===")
    summary = {
        "description": (
            "Phase 1 syntax prototype extraction. KMeans on T=34 user sentences "
            "in F3_CoreStruct + StandardScaler + PCA48 whitened space (same as Stage 4 "
            "Mahalanobis computation). Per prototype: centroid + abstract structural "
            "pattern (dep tree / opener distribution / clause counts) + 5 nearest "
            "anonymized sample texts (no user_id)."
        ),
        "thresholds": {"cv_nll_threshold": 34, "validation_precision": 0.9821},
        "n_strict_users": len(set(s["user_id"] for s in sentences)),
        "n_sentences_input": len(sentences),
        "n_sentences_with_features": len(valid_idx),
        "K_values": K_VALUES,
        "per_K": {
            str(K): {
                "K": results[K]["K"],
                "n_sentences_projected": results[K]["n_total_sentences"],
                "KMeans_inertia": results[K]["KMeans_inertia"],
                "cluster_sizes": results[K]["cluster_sizes"],
                "n_prototypes": len(results[K]["prototypes"]),
            }
            for K in K_VALUES
        },
        "next_step": (
            "Phase 2: Prototype-guided pool regen. For each ASIN, prompt LLM with "
            "abstract structural descriptions (not verbatim reviews) per prototype, "
            "K_per_proto candidates per ASIN. Total K_pool = sum_K_syntax × K_per_proto."
        ),
    }
    summary_out = REPO_ROOT / "result" / "gaussian" / "syntax_prototypes_summary.json"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {summary_out}")


if __name__ == "__main__":
    main()
