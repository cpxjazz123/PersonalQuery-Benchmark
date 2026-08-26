"""Syntax Subspace — 单主脚本,合并 Stage 8.5/8.5V/10K 全流程.

这是一个**编排器(orchestrator)**:将原本拆分的 7 个脚本合并到同一文件,
通过 section marker 清晰区分各阶段,各阶段可独立运行也可以 `main()` 一键
跑完。

=== Stages (顺序) ===

  STAGE 1 — POOL REGEN      : 为 100 个 ASIN 各生成 K=50 共享候选池 (via HTTP vLLM)
                              原: syntax_subspace_stage8_5_regen.py

  STAGE 2 — FEATURES        : 抽取 spaCy 182d 特征并缓存到 JSONL.gz
                              原: syntax_subspace_stage8_5_features.py

  STAGE 3 — USER GAUSSIANS  : 从 review corpus 构建 per-user 高斯 (PCA48 + 收缩)
                              原: syntax_subspace_stage8_5_user_gaussians.py

  STAGE 4 — MAHA SELECT     : 对每 (asin, user) 选 Mahalanobis 最小候选 (selected/random/farthest)
                              原: syntax_subspace_stage8_5_select.py

  STAGE 5 — RETRIEVAL       : bm25s + GPU MiniLM, 算 rank/RR/hit@10
                              原: syntax_subspace_stage8_5_retrieval.py

  STAGE 6 — VOLATILITY      : V_low / V_user / V_high 经验波动率标定
                              原: syntax_subspace_stage8_5v_volatility.py

  STAGE 7 — A1 SELECT       : mean_t50 + reject-repeat,提升 unique (Stage 10K SOTA)
                              原: syntax_subspace_stage10k_selection.py

=== 用法 ===

  # 跑单阶段
  python gaussian/syntax_subspace_main.py --stage pool_regen
  python gaussian/syntax_subspace_main.py --stage features
  python gaussian/syntax_subspace_main.py --stage user_gaussians
  python gaussian/syntax_subspace_main.py --stage select
  python gaussian/syntax_subspace_main.py --stage retrieval
  python gaussian/syntax_subspace_main.py --stage volatility
  python gaussian/syntax_subspace_main.py --stage a1_select

  # 跑全部
  python gaussian/syntax_subspace_main.py --stage all

=== I/O 路径 ===

  scratch: /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades
  输入:
    stage8_5_asins.json
    Baby_Products_2023.jsonl.gz (review corpus)
    meta_Baby_Products_2023.jsonl.gz (ASIN metadata corpus)
  输出:
    stage8_5_pool.json           (Stage 1)
    stage7b_query_features.jsonl.gz  (Stage 2 + Stage 3 共用, append-only)
    stage8_5_user_gaussians.json (Stage 3)
    stage8_5_selection.json      (Stage 4)
    stage8_5_selection_stats.json (Stage 4)
    stage8_5_retrieval_per_query.json (Stage 5)
    stage8_5_retrieval_summary.json  (Stage 5)
    stage8_5v_retrieval.json     (Stage 6)
    stage8_5v_volatility.json    (Stage 6)
    stage10k_selection.json      (Stage 7)
    stage10k_summary.json        (Stage 7)

共享工具 (_syntax_subspace_prepare, RESIDUAL_SCRATCH, log) 来自:
  gaussian/syntax_subspace_utils.py

"""

from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import requests

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
ASINS_IN = SCRATCH / "stage8_5_asins.json"
POOL_OUT = SCRATCH / "stage8_5_pool.json"
POOL_IN = POOL_OUT
FEAT_CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"
GAUSSIANS_OUT = SCRATCH / "stage8_5_user_gaussians.json"
GAUSSIANS_IN = GAUSSIANS_OUT
SELECTION_OUT = SCRATCH / "stage8_5_selection.json"
SELECTION_IN = SELECTION_OUT
SELECTION_STATS_OUT = SCRATCH / "stage8_5_selection_stats.json"
REVIEW_GZ = REPO_ROOT / "data/Baby_Products_2023.jsonl.gz"
META_FILE = REPO_ROOT / "data/meta_Baby_Products_2023.jsonl.gz"
RETRIEVAL_PER_QUERY_OUT = SCRATCH / "stage8_5_retrieval_per_query.json"
RETRIEVAL_SUMMARY_OUT = SCRATCH / "stage8_5_retrieval_summary.json"
VOLATILITY_PER_QUERY_OUT = SCRATCH / "stage8_5v_retrieval.json"
VOLATILITY_SUMMARY_OUT = SCRATCH / "stage8_5v_volatility.json"
A1_SELECTION_OUT = SCRATCH / "stage10k_selection.json"
A1_SUMMARY_OUT = SCRATCH / "stage10k_summary.json"

VLLM_URL = "http://localhost:8800/v1/completions"
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
SEED = 2024

PCA_DIM = 48
PCA_SEED = 2024
LAMBDA = 0.1
VAR_EPS = 1e-3
MIN_REVIEWS_FOR_PER_USER = 3
K_POOL = 50
TEMP = 0.7
MAX_TOKENS = 80
K_SET = 8
MIN_LEN = 5
PRIMARY_LEN_DELTA = 2
LEN_BAND_FALLBACK = 5
THRESHOLD_PCT = 50

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


# ===========================================================================
# STAGE 1 — POOL REGEN
# ===========================================================================

GEN_SYSTEM_TMPL_NATURAL = (
    "You are an Amazon shopper writing a search query. Use EXACTLY the {N_INPUT} "
    "attribute values listed below verbatim (mention each value once). DO NOT add any "
    "product type (no bottle/clothes/toy/candle/tumbler), use case, personal "
    "context, or inferred property (do not turn a Color into a scent, a "
    "Material into a function, or a Style into a product class). DO NOT "
    "include attribute field names (no 'Brand:', 'material_type:', "
    "'material_composition:', 'main category', 'Style:') in the query — "
    "only the values. Each value keeps the meaning of its attribute name. "
    "Write a natural sentence (any length is fine). Output ONLY the query, no preamble.\n\n"
    "Attributes ({N_INPUT}):\n{ATTRIBUTES}"
)


def build_user_content(attrs: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in attrs.items())


def make_prompt(attrs: dict, n_input: int, k: int = 0) -> str:
    base = GEN_SYSTEM_TMPL_NATURAL.format(
        N_INPUT=n_input,
        ATTRIBUTES=build_user_content(attrs),
    )
    if k > 0:
        base += f"\n(variant {k})"
    return base


def batch_generate_vllm(prompts: List[str], temp: float = TEMP, max_tokens: int = MAX_TOKENS) -> List[str]:
    outputs = []
    full_prompts = []
    for p in prompts:
        full_prompts.append(
            f"system\n{p}\n"
            f"user\n\n"
            f"assistant\n"
        )
    bs = 64
    for i in range(0, len(prompts), bs):
        chunk = full_prompts[i: i + bs]
        try:
            resp = requests.post(
                VLLM_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": chunk,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "top_p": 0.95 if temp > 0 else 1.0,
                },
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            for choice in data["choices"]:
                outputs.append(choice["text"].strip())
        except Exception as e:
            log(f"  batch error: {e!r}, falling back to single requests")
            for single_prompt in chunk:
                try:
                    r = requests.post(
                        VLLM_URL,
                        json={
                            "model": MODEL_NAME,
                            "prompt": [single_prompt],
                            "temperature": temp,
                            "max_tokens": max_tokens,
                            "top_p": 0.95 if temp > 0 else 1.0,
                        },
                        timeout=60,
                    )
                    r.raise_for_status()
                    rj = r.json()
                    outputs.append(rj["choices"][0]["text"].strip())
                except Exception as _e:
                    log(f"  single fallback failed: {_e!r}")
                    outputs.append("")
    return outputs


def count_attrs_covered(text: str, attrs: dict) -> int:
    if not text:
        return 0
    text_lower = text.lower()
    covered = 0
    for k, v in attrs.items():
        if v and str(v).lower() in text_lower:
            covered += 1
    return covered


def has_invalid_punct(text: str) -> bool:
    if not text:
        return True
    bad_patterns = [
        r"^(here|this|below|sure|okay|ok)[,:]",
        r"^attribute[s]?:",
        r"^brand:",
        r"^color:",
        r"^material:",
        r"^style:",
    ]
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower) for p in bad_patterns)


def n_tokens_simple(text: str) -> int:
    return len(text.split())


def stage_pool_regen():
    log(f"=== STAGE 1 — POOL REGEN: K_POOL={K_POOL} × 100 ASINs ===")

    log(f"loading {ASINS_IN}")
    asin_data = json.load(open(ASINS_IN))["asins"]
    log(f"  {len(asin_data)} ASINs")

    log("building prompts...")
    all_prompts = []
    asin_idx = []
    for entry in asin_data:
        a = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        for k in range(K_POOL):
            prompt = make_prompt(attrs, n_input, k=k)
            all_prompts.append(prompt)
            asin_idx.append((a, k))

    log(f"generating {len(all_prompts)} pool queries via vLLM...")
    all_outputs = batch_generate_vllm(all_prompts, temp=TEMP, max_tokens=MAX_TOKENS)
    log(f"  got {len(all_outputs)} outputs")

    pools = collections.defaultdict(list)
    strict_counts = {}
    n_total_strict = 0

    for (a, k), out in zip(asin_idx, all_outputs):
        entry = next(e for e in asin_data if e["asin"] == a)
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        text = out.strip() if out else ""
        if not text:
            continue
        n_cov = count_attrs_covered(text, attrs)
        invalid = has_invalid_punct(text)
        is_strict = (n_cov == n_input) and (not invalid)
        if is_strict:
            n_total_strict += 1
            strict_counts[a] = strict_counts.get(a, 0) + 1
        pools[a].append({
            "k": k,
            "query": text,
            "strict": is_strict,
            "attrs_covered": n_cov,
            "invalid": invalid,
            "n_tok": n_tokens_simple(text),
        })

    log(f"\n=== Pool stats ===")
    log(f"  total queries: {len(all_outputs)}")
    log(f"  strict: {n_total_strict} ({n_total_strict / max(1, len(all_outputs)) * 100:.1f}%)")
    log(f"  ASINs: {len(pools)}")
    if strict_counts:
        log(f"  strict per ASIN: min={min(strict_counts.values())}, "
            f"max={max(strict_counts.values())}, "
            f"mean={sum(strict_counts.values()) / len(strict_counts):.1f}")
    log(f"  ASINs with ≥10 strict: {sum(1 for v in strict_counts.values() if v >= 10)}")
    log(f"  ASINs with ≥20 strict: {sum(1 for v in strict_counts.values() if v >= 20)}")

    POOL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5: SHARED candidate pool per ASIN (K=50, attrs-only prompt)",
                "K_POOL": K_POOL,
                "TEMP": TEMP,
                "MAX_TOKENS": MAX_TOKENS,
                "SEED": SEED,
                "MODEL_NAME": MODEL_NAME,
            },
            "strict_counts": strict_counts,
            "n_asins": len(pools),
            "n_total_strict": n_total_strict,
            "pools": dict(pools),
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {POOL_OUT}")


# ===========================================================================
# STAGE 2 — FEATURES
# ===========================================================================

def stage_features():
    log("=== STAGE 2 — FEATURES ===")

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    fnames = P["feature_names_ordered"]
    log(f"  fnames: {len(fnames)}")

    log(f"loading {POOL_IN}")
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    all_queries = []
    for asin, qs in pools.items():
        for q in qs:
            all_queries.append(q["query"])
    log(f"  total pool queries: {len(all_queries)}")

    feat_map = {}
    if FEAT_CACHE.exists():
        with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                feat_map[rec["k"]] = rec["v"]
    log(f"  cache keys: {len(feat_map)}")

    missing_q = [q for q in all_queries if feat_key(q) not in feat_map]
    log(f"  missing: {len(missing_q)}")

    if not missing_q:
        log("  no new features needed")
        return

    import spacy
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
    from main import per_sentence_features_v2
    nlp = spacy.load("en_core_web_sm")

    log(f"  extracting features for {len(missing_q)} queries via spaCy pipe (n_process=8, batch=256)...")
    new_unique = sorted(set(missing_q))
    new_entries = []
    n_skip = 0
    docs = list(nlp.pipe(new_unique, batch_size=256, n_process=8))
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
        filtered = {n: numeric.get(n, 0.0) for n in fnames}
        feat_map[k] = filtered
        new_entries.append({"k": k, "v": filtered})
        if (i + 1) % 1000 == 0:
            log(f"    {i + 1}/{len(new_unique)}")

    log(f"  extracted: {len(new_entries)}, skipped: {n_skip}")

    with gzip.open(FEAT_CACHE, "wt", encoding="utf-8") as f:
        f.write("# spaCy 182d sentence features (key=sha1(text), v=filtered dict)\n")
        for k, v in feat_map.items():
            f.write(json.dumps({"k": k, "v": v}) + "\n")
    log(f"  saved cache: {len(feat_map)} entries")


# ===========================================================================
# STAGE 3 — USER GAUSSIANS
# ===========================================================================

def stage_user_gaussians():
    log("=== STAGE 3 — USER GAUSSIANS ===")

    log("\n=== 1. Loading PCA48 ===")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

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
        uid = r.get("user_id", "")
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
    log(f"  users with 1-2 reviews (global var fallback): {n_low}")
    n_zero = len(target_users) - len(user_review_texts)
    log(f"  users with 0 reviews (asin centroid fallback): {n_zero}")

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
    all_sents = []
    sent_to_user = []
    for uid, texts in user_review_texts.items():
        for t in texts:
            for s in t.replace("\n", " ").split(". "):
                s = s.strip()
                if s and len(s.split()) >= 3:
                    all_sents.append(s)
                    sent_to_user.append(uid)
    log(f"  total sentences: {len(all_sents)}")

    new_sents = [s for s in all_sents if feat_key(s) not in feat_map]
    log(f"  unique sentences: {len(set(all_sents))}, new to extract: {len(new_sents)}")

    if new_sents:
        import spacy
        sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
        from main import per_sentence_features_v2
        nlp = spacy.load("en_core_web_sm")
        log(f"  extracting features for {len(new_sents)} new sentences (n_process=8, batch=256)...")
        new_unique = sorted(set(new_sents))
        docs = list(nlp.pipe(new_unique, batch_size=256, n_process=8))
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
    user_z_list = collections.defaultdict(list)
    n_skip = 0
    for s, uid in zip(all_sents, sent_to_user):
        feats = feat_map.get(feat_key(s))
        if not feats:
            n_skip += 1
            continue
        vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
        vec_scaled = scaler.transform(vec[None, :])[0]
        z = pca.transform(vec_scaled[None, :])[0]
        user_z_list[uid].append(z)
    log(f"  sentences projected: {len(all_sents) - n_skip}, skipped: {n_skip}")

    log("\n=== 7. Building global pooled variance ===")
    all_z = []
    for uid, zs in user_z_list.items():
        all_z.extend(zs)
    all_z = np.stack(all_z, axis=0)
    global_var = all_z.var(axis=0)
    log(f"  global var shape: {global_var.shape}, mean: {global_var.mean():.4f}")

    log("\n=== 8. Building per-user Gaussians ===")
    user_gaussians = {}
    n_per_user = 0
    n_fallback = 0

    for uid in target_users:
        zs = user_z_list.get(uid, [])
        n_reviews = len(user_review_texts.get(uid, []))
        n_words = sum(len(t.split()) for t in user_review_texts.get(uid, []))

        if len(zs) >= MIN_REVIEWS_FOR_PER_USER:
            Z = np.stack(zs, axis=0)
            mu = Z.mean(axis=0)
            var = Z.var(axis=0)
            var_shrink = (1 - LAMBDA) * var + LAMBDA * var.mean()
            sigma_diag = np.maximum(var_shrink, VAR_EPS)
            source = "per_user"
            n_per_user += 1
        elif len(zs) >= 1:
            Z = np.stack(zs, axis=0)
            mu = Z.mean(axis=0)
            sigma_diag = np.maximum(global_var, VAR_EPS)
            source = "global_var_fallback"
            n_fallback += 1
        else:
            continue

        user_gaussians[uid] = {
            "mu": mu.tolist(),
            "sigma_diag": sigma_diag.tolist(),
            "n_reviews": n_reviews,
            "n_words": n_words,
            "n_sentences": len(zs),
            "source": source,
        }

    log(f"  per-user Gaussians: {n_per_user}")
    log(f"  global-var fallback: {n_fallback}")
    log(f"  no reviews (skip): {len(target_users) - len(user_gaussians)}")

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
            "global_var": global_var.tolist(),
            "users": user_gaussians,
            "n_users_with_gaussian": len(user_gaussians),
            "n_users_skipped": len(target_users) - len(user_gaussians),
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {GAUSSIANS_OUT}")


# ===========================================================================
# STAGE 4 — MAHA SELECT
# ===========================================================================

def mahalanobis_sq(z: np.ndarray, mu: np.ndarray, sigma_diag: np.ndarray) -> float:
    diff = z - mu
    return float((diff * diff / sigma_diag).sum())


def stage_select():
    log("=== STAGE 4 — MAHA SELECT ===")

    log("\n=== 1. Loading PCA48 ===")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 ready")

    log("\n=== 2. Loading inputs ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]
    global_var = np.array(gauss_data["global_var"])
    log(f"  ASINs: {len(asin_data)}, pools: {len(pools)}")
    log(f"  user Gaussians: {len(users_gauss)}")

    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  features: {len(feat_map)}")

    log("\n=== 3. Projecting pool queries ===")
    pool_z = {}
    miss = 0
    for asin, qs in pools.items():
        zs_for_asin = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs_for_asin.append((z, q))
        pool_z[asin] = zs_for_asin
    log(f"  ASINs with pool_z: {len(pool_z)}, missing features: {miss}")

    log("\n=== 4. ASIN centroid fallback ===")
    asin_centroid = {}
    for asin, zqs in pool_z.items():
        zs = np.stack([z for z, _ in zqs], axis=0)
        asin_centroid[asin] = zs.mean(axis=0)
    log(f"  ASIN centroids: {len(asin_centroid)}")

    log("\n=== 5. Selection per (asin, user) ===")
    rng = random.Random(SEED)

    selection_entries = []
    n_mahal = 0
    n_asin_fallback = 0
    n_random_fallback = 0

    for entry in asin_data:
        asin = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        zqs = pool_z.get(asin, [])
        n_pool_strict = sum(1 for z, q in zqs if q["strict"])

        c_asin = asin_centroid.get(asin)
        if c_asin is None:
            log(f"  WARNING: no centroid for {asin}, skipping users")
            continue

        for uid in entry["users_sampled"]:
            if uid in users_gauss:
                mu = np.array(users_gauss[uid]["mu"])
                sigma = np.array(users_gauss[uid]["sigma_diag"])
                source = users_gauss[uid]["source"]
                n_reviews = users_gauss[uid]["n_reviews"]
            else:
                mu = c_asin
                sigma = np.maximum(global_var, 1e-3)
                source = "asin_centroid_fallback"
                n_reviews = 0

            strict_zqs = [(z, q) for z, q in zqs if q["strict"]]
            if not strict_zqs:
                strict_zqs = zqs
            if not strict_zqs:
                selection_entries.append({
                    "asin": asin,
                    "user_id": uid,
                    "attrs_used": attrs,
                    "selection_method": "no_pool",
                    "selected": None,
                    "random": None,
                    "farthest": None,
                    "selected_distance": None,
                    "random_distance": None,
                    "farthest_distance": None,
                    "n_candidates": 0,
                    "user_source": source,
                    "n_reviews": n_reviews,
                })
                continue

            distances = np.array([mahalanobis_sq(z, mu, sigma) for z, _ in strict_zqs])
            best_idx = int(np.argmin(distances))
            worst_idx = int(np.argmax(distances))

            selected_q = strict_zqs[best_idx][1]
            farthest_q = strict_zqs[worst_idx][1]

            rng_u = random.Random(hash(uid) & 0xffffffff)
            random_idx = rng_u.randint(0, len(strict_zqs) - 1)
            random_q = strict_zqs[random_idx][1]

            if source == "per_user" or source == "global_var_fallback":
                method = "mahal_min"
                n_mahal += 1
            else:
                method = "asin_centroid_fallback"
                n_asin_fallback += 1

            selection_entries.append({
                "asin": asin,
                "user_id": uid,
                "attrs_used": attrs,
                "selection_method": method,
                "selected": selected_q,
                "random": random_q,
                "farthest": farthest_q,
                "selected_distance": float(distances[best_idx]),
                "random_distance": float(distances[random_idx]),
                "farthest_distance": float(distances[worst_idx]),
                "n_candidates": len(strict_zqs),
                "user_source": source,
                "n_reviews": n_reviews,
            })

    log(f"  total entries: {len(selection_entries)}")
    log(f"    mahal_min: {n_mahal}")
    log(f"    asin_centroid_fallback: {n_asin_fallback}")

    SELECTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5: Mahalanobis selection from shared pool",
                "SEED": SEED,
            },
            "n_entries": len(selection_entries),
            "n_mahal": n_mahal,
            "n_asin_fallback": n_asin_fallback,
            "n_random_fallback": n_random_fallback,
            "entries": selection_entries,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {SELECTION_OUT}")

    log("\n=== 7. Validation ===")
    from scipy.stats import wilcoxon
    mahal_entries = [e for e in selection_entries if e["selected_distance"] is not None]
    selected = np.array([e["selected_distance"] for e in mahal_entries])
    random_d = np.array([e["random_distance"] for e in mahal_entries])
    farthest = np.array([e["farthest_distance"] for e in mahal_entries])

    log(f"  N pairs: {len(mahal_entries)}")
    log(f"  selected: mean={selected.mean():.3f}, std={selected.std():.3f}, median={np.median(selected):.3f}")
    log(f"  random:   mean={random_d.mean():.3f}, std={random_d.std():.3f}, median={np.median(random_d):.3f}")
    log(f"  farthest: mean={farthest.mean():.3f}, std={farthest.std():.3f}, median={np.median(farthest):.3f}")

    w_sr, p_sr = wilcoxon(selected, random_d, alternative="less")
    log(f"  Wilcoxon selected < random: W={w_sr:.1f}, p={p_sr:.4g}")
    w_sf, p_sf = wilcoxon(selected, farthest, alternative="less")
    log(f"  Wilcoxon selected < farthest: W={w_sf:.1f}, p={p_sf:.4g}")
    w_rf, p_rf = wilcoxon(random_d, farthest, alternative="less")
    log(f"  Wilcoxon random < farthest: W={w_rf:.1f}, p={p_rf:.4g}")

    by_source = collections.defaultdict(lambda: {"selected": [], "random": [], "farthest": []})
    for e in mahal_entries:
        s = e["user_source"]
        by_source[s]["selected"].append(e["selected_distance"])
        by_source[s]["random"].append(e["random_distance"])
        by_source[s]["farthest"].append(e["farthest_distance"])

    log(f"\n=== Per-source breakdown ===")
    for src, d in by_source.items():
        sel = np.array(d["selected"])
        rnd = np.array(d["random"])
        log(f"  {src} (n={len(d['selected'])}):")
        log(f"    selected mean={sel.mean():.3f}, random mean={rnd.mean():.3f}, "
            f"diff={(sel - rnd).mean():+.3f}, %sel<rnd={100 * (sel < rnd).mean():.1f}%")

    with open(SELECTION_STATS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "n_pairs": len(mahal_entries),
            "selected_mean": float(selected.mean()),
            "selected_std": float(selected.std()),
            "random_mean": float(random_d.mean()),
            "random_std": float(random_d.std()),
            "farthest_mean": float(farthest.mean()),
            "farthest_std": float(farthest.std()),
            "wilcoxon_selected_vs_random": {"W": float(w_sr), "p_value": float(p_sr), "alternative": "less"},
            "wilcoxon_selected_vs_farthest": {"W": float(w_sf), "p_value": float(p_sf), "alternative": "less"},
            "wilcoxon_random_vs_farthest": {"W": float(w_rf), "p_value": float(p_rf), "alternative": "less"},
            "per_source": {
                src: {
                    "n": len(d["selected"]),
                    "selected_mean": float(np.mean(d["selected"])),
                    "random_mean": float(np.mean(d["random"])),
                    "farthest_mean": float(np.mean(d["farthest"])),
                    "pct_selected_lt_random": float(100 * (np.array(d["selected"]) < np.array(d["random"])).mean()),
                }
                for src, d in by_source.items()
            },
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {SELECTION_STATS_OUT}")


# ===========================================================================
# STAGE 5 — RETRIEVAL
# ===========================================================================

def build_meta_corpus():
    asin_to_doc = {}
    log(f"  loading metadata from {META_FILE}")
    with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            asin = r.get("parent_asin", "").strip()
            if not asin:
                continue
            parts = []
            t = r.get("title", "").strip()
            if t:
                parts.append(t)
            desc = r.get("description", [])
            if isinstance(desc, list):
                desc = " ".join(desc)
            elif not isinstance(desc, str):
                desc = ""
            desc = desc.strip()
            if desc:
                parts.append(desc)
            feats = r.get("features", [])
            if isinstance(feats, list):
                feats = " ".join(feats)
            if feats:
                parts.append(feats[:500])
            doc = " | ".join(parts).strip()
            if doc:
                asin_to_doc[asin] = doc[:1000]
    log(f"  loaded {len(asin_to_doc)} ASIN docs")
    return asin_to_doc


def stage_retrieval():
    log("=== STAGE 5 — RETRIEVAL ===")

    log("\n=== 1. Loading selection ===")
    selection = json.load(open(SELECTION_IN))
    entries = selection["entries"]
    log(f"  {len(entries)} entries")

    query_records = []
    for i, e in enumerate(entries):
        for variant in ("selected", "random", "farthest"):
            q = e[variant]
            if q is None:
                continue
            query_records.append({
                "entry_idx": i,
                "asin": e["asin"],
                "user_id": e["user_id"],
                "variant": variant,
                "selection_method": e["selection_method"],
                "user_source": e["user_source"],
                "query": q["query"],
                "n_tok": q["n_tok"],
                "attrs_covered": q["attrs_covered"],
                "strict": q["strict"],
            })
    log(f"  total queries to retrieve: {len(query_records)}")

    log("\n=== 2. Building ASIN metadata corpus ===")
    asin_to_doc = build_meta_corpus()
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus size: {len(asins)} ASINs")

    queries = [r["query"] for r in query_records]
    target_indices = np.array([asin_to_idx.get(r["asin"], -1) for r in query_records])
    n_missing = int((target_indices < 0).sum())
    if n_missing:
        log(f"  WARNING: {n_missing} queries have missing target ASINs in corpus")

    log("\n=== 3. BM25 retrieval (bm25s) ===")
    import bm25s
    corpus_texts = [asin_to_doc[a] for a in asins]
    log(f"  tokenizing {len(corpus_texts)} corpus texts...")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    log(f"  corpus tokenized in {time.time() - t0:.1f}s")

    log(f"  building bm25s index...")
    t0 = time.time()
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    log(f"  index built in {time.time() - t0:.1f}s")

    log(f"  tokenizing {len(queries)} queries...")
    t0 = time.time()
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    log(f"  queries tokenized in {time.time() - t0:.1f}s")

    log(f"  retrieving (k=all)...")
    t0 = time.time()
    results = retriever.retrieve(query_tokens, k=len(corpus_texts), show_progress=False)
    log(f"  retrieved in {time.time() - t0:.1f}s")

    log(f"  computing target ranks...")
    t0 = time.time()
    bm25_results = []
    bm25_doc_ids = results.documents
    for i in range(len(queries)):
        tgt_idx = target_indices[i]
        if tgt_idx < 0:
            bm25_results.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        sorted_docs = bm25_doc_ids[i]
        positions = np.where(sorted_docs == tgt_idx)[0]
        if len(positions) == 0:
            bm25_results.append({"rank": None, "RR": 0.0, "hit10": 0})
        else:
            rank = int(positions[0]) + 1
            bm25_results.append({
                "rank": rank,
                "RR": 1.0 / rank,
                "hit10": 1 if rank <= 10 else 0,
            })
    log(f"  ranks computed in {time.time() - t0:.1f}s")

    log("\n=== 4. MiniLM retrieval (GPU) ===")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  MiniLM device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    log(f"  encoding {len(corpus_texts)} corpus docs via MiniLM (GPU, batch=512)...")
    t0 = time.time()
    doc_embeds = model.encode(
        corpus_texts,
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(f"  docs encoded in {time.time() - t0:.1f}s, shape: {doc_embeds.shape}")

    log(f"  encoding {len(queries)} queries via MiniLM (GPU)...")
    t0 = time.time()
    q_embeds = model.encode(
        queries,
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(f"  queries encoded in {time.time() - t0:.1f}s")
    doc_embeds_gpu = torch.from_numpy(doc_embeds).cuda()
    q_embeds_gpu = torch.from_numpy(q_embeds).cuda()

    log(f"  computing cosine similarities on GPU...")
    t0 = time.time()
    sims = q_embeds_gpu @ doc_embeds_gpu.T
    log(f"  matmul done in {time.time() - t0:.1f}s")
    log(f"  sorting similarities...")
    t0 = time.time()
    minilm_results = []
    for i in range(sims.shape[0]):
        tgt_idx = target_indices[i]
        if tgt_idx < 0:
            minilm_results.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        sc = sims[i]
        rank = int((sc > sc[tgt_idx]).sum().item()) + 1
        minilm_results.append({
            "rank": rank,
            "RR": 1.0 / rank,
            "hit10": 1 if rank <= 10 else 0,
        })
    log(f"  ranks computed in {time.time() - t0:.1f}s")

    del doc_embeds_gpu, q_embeds_gpu, sims
    torch.cuda.empty_cache()

    log("\n=== 5. Saving ===")
    for r, bm25_r, minilm_r in zip(query_records, bm25_results, minilm_results):
        r["bm25_rank"] = bm25_r["rank"]
        r["bm25_RR"] = bm25_r["RR"]
        r["bm25_hit10"] = bm25_r["hit10"]
        r["minilm_rank"] = minilm_r["rank"]
        r["minilm_RR"] = minilm_r["RR"]
        r["minilm_hit10"] = minilm_r["hit10"]

    with open(RETRIEVAL_PER_QUERY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5 retrieval per query (bm25s + GPU MiniLM) on ASIN meta corpus",
                "corpus_size": len(asins),
            },
            "n_queries": len(query_records),
            "queries": query_records,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {RETRIEVAL_PER_QUERY_OUT}")

    log("\n=== 6. Summary by variant ===")
    by_variant = collections.defaultdict(lambda: {
        "bm25_RR": [], "minilm_RR": [], "bm25_hit10": [], "minilm_hit10": [],
        "bm25_rank": [], "minilm_rank": [],
    })
    for r in query_records:
        v = r["variant"]
        by_variant[v]["bm25_RR"].append(r["bm25_RR"])
        by_variant[v]["minilm_RR"].append(r["minilm_RR"])
        by_variant[v]["bm25_hit10"].append(r["bm25_hit10"])
        by_variant[v]["minilm_hit10"].append(r["minilm_hit10"])
        if r["bm25_rank"]:
            by_variant[v]["bm25_rank"].append(r["bm25_rank"])
        if r["minilm_rank"]:
            by_variant[v]["minilm_rank"].append(r["minilm_rank"])

    summary = {}
    for v in ("selected", "random", "farthest"):
        d = by_variant[v]
        summary[v] = {
            "n": len(d["bm25_RR"]),
            "bm25_MRR": float(np.mean(d["bm25_RR"])),
            "bm25_Hit@10": float(np.mean(d["bm25_hit10"])),
            "bm25_mean_rank": float(np.mean(d["bm25_rank"])) if d["bm25_rank"] else None,
            "minilm_MRR": float(np.mean(d["minilm_RR"])),
            "minilm_Hit@10": float(np.mean(d["minilm_hit10"])),
            "minilm_mean_rank": float(np.mean(d["minilm_rank"])) if d["minilm_rank"] else None,
        }

    log(f"\n=== Variant summary ===")
    for v, s in summary.items():
        bm25_rank_str = f"{s['bm25_mean_rank']:.1f}" if s["bm25_mean_rank"] else "n/a"
        minilm_rank_str = f"{s['minilm_mean_rank']:.1f}" if s["minilm_mean_rank"] else "n/a"
        log(f"  {v} (n={s['n']}):")
        log(f"    BM25:   MRR={s['bm25_MRR']:.4f}, Hit@10={s['bm25_Hit@10']*100:.1f}%, mean_rank={bm25_rank_str}")
        log(f"    MiniLM: MRR={s['minilm_MRR']:.4f}, Hit@10={s['minilm_Hit@10']*100:.1f}%, mean_rank={minilm_rank_str}")

    with open(RETRIEVAL_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {RETRIEVAL_SUMMARY_OUT}")


# ===========================================================================
# STAGE 6 — VOLATILITY
# ===========================================================================

def stage_volatility():
    log("=== STAGE 6 — VOLATILITY ===")

    log("\n=== 1. Loading inputs ===")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 ready, EV={pca.explained_variance_ratio_.sum():.4f}")

    selection = json.load(open(SELECTION_IN))
    entries = selection["entries"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]

    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  loaded {len(feat_map)} features")

    log("\n=== 2. Projecting pool queries ===")
    pool_z = {}
    miss = 0
    for asin, qs in pools.items():
        zs = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs.append({"z": z, "q": q["query"], "n_tok": q["n_tok"], "strict": q["strict"], "k": q["k"]})
        pool_z[asin] = zs
    log(f"  pool_z: {len(pool_z)} ASINs, missed {miss} features")

    log("\n=== 3. Building V_low / V_user / V_high query sets ===")
    asin_query_sets = {}
    n_insufficient_low = 0
    n_insufficient_high = 0
    n_total = 0
    for entry in entries:
        asin = entry["asin"]
        if asin not in pool_z:
            continue
        cands = [c for c in pool_z[asin] if c["strict"] and c["n_tok"] >= MIN_LEN]
        if len(cands) < K_SET * 2:
            continue
        zs = np.array([c["z"] for c in cands])
        n = len(cands)
        dist = np.zeros((n, n))
        for i in range(n):
            diff = zs - zs[i]
            dist[i] = np.linalg.norm(diff, axis=1)
        mean_dist = (dist.sum(axis=1) - np.diag(dist)) / (n - 1)
        sorted_idx = np.argsort(mean_dist)

        def length_match(anchor_idx, max_diff):
            ok = []
            a_tok = cands[anchor_idx]["n_tok"]
            for j in range(n):
                if abs(cands[j]["n_tok"] - a_tok) <= max_diff:
                    ok.append(j)
            return ok

        anchor_lo = sorted_idx[0]
        ok_idx_lo = []
        for max_diff in [PRIMARY_LEN_DELTA, LEN_BAND_FALLBACK, 10]:
            ok_idx_lo = length_match(anchor_lo, max_diff)
            if len(ok_idx_lo) >= K_SET:
                break
        if len(ok_idx_lo) < K_SET:
            n_insufficient_low += 1
            continue
        ok_dist_lo = dist[anchor_lo][ok_idx_lo]
        sorted_ok_lo = np.array(ok_idx_lo)[np.argsort(ok_dist_lo)][:K_SET]
        low_qs = [cands[i]["q"] for i in sorted_ok_lo]

        anchor_hi = sorted_idx[-1]
        ok_idx_hi = []
        for max_diff in [PRIMARY_LEN_DELTA, LEN_BAND_FALLBACK, 10]:
            ok_idx_hi = length_match(anchor_hi, max_diff)
            if len(ok_idx_hi) >= K_SET:
                break
        if len(ok_idx_hi) < K_SET:
            n_insufficient_high += 1
            continue
        ok_dist_hi = dist[anchor_hi][ok_idx_hi]
        sorted_ok_hi = np.array(ok_idx_hi)[np.argsort(-ok_dist_hi)][:K_SET]
        high_qs = [cands[i]["q"] for i in sorted_ok_hi]

        user_queries = {}
        for e_idx, e in enumerate(entries):
            if e["asin"] != asin:
                continue
            sel = e["selected"]
            if sel:
                user_queries[e["user_id"]] = sel["query"]
        if not user_queries:
            continue
        asin_query_sets[asin] = {
            "low": low_qs,
            "high": high_qs,
            "user_queries": user_queries,
        }
        n_total += 1

    log(f"  ASINs with all 3 sets: {n_total}")
    log(f"    insufficient low: {n_insufficient_low}")
    log(f"    insufficient high: {n_insufficient_high}")

    log("\n=== 4. Building ASIN metadata corpus ===")
    asin_to_doc = {}
    with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            asin = r.get("parent_asin", "").strip()
            if not asin:
                continue
            parts = []
            t = r.get("title", "").strip()
            if t:
                parts.append(t)
            desc = r.get("description", [])
            if isinstance(desc, list):
                desc = " ".join(desc)
            elif not isinstance(desc, str):
                desc = ""
            desc = desc.strip()
            if desc:
                parts.append(desc)
            feats = r.get("features", [])
            if isinstance(feats, list):
                feats = " ".join(feats)
            if feats:
                parts.append(feats[:500])
            doc = " | ".join(parts).strip()
            if doc:
                asin_to_doc[asin] = doc[:1000]
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus: {len(asins)} ASINs")

    log("\n=== 5. Building flat query list ===")
    flat_queries = []
    for asin, sets in asin_query_sets.items():
        for set_type in ("low", "high"):
            for q in sets[set_type]:
                flat_queries.append({
                    "asin": asin,
                    "set_type": set_type,
                    "user_id": None,
                    "query": q,
                })
        for uid, q in sets["user_queries"].items():
            flat_queries.append({
                "asin": asin,
                "set_type": "user",
                "user_id": uid,
                "query": q,
            })
    log(f"  total queries: {len(flat_queries)}")

    log("\n=== 6. BM25 retrieval (bm25s) ===")
    import bm25s
    corpus_texts = [asin_to_doc[a] for a in asins]
    queries = [r["query"] for r in flat_queries]
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    t0 = time.time()
    results = retriever.retrieve(query_tokens, k=len(asins), show_progress=False)
    log(f"  bm25s retrieve done in {time.time() - t0:.1f}s")

    bm25_ranks = []
    for i, r in enumerate(flat_queries):
        tgt_idx = asin_to_idx.get(r["asin"], -1)
        if tgt_idx < 0:
            bm25_ranks.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        sorted_docs = results.documents[i]
        positions = np.where(sorted_docs == tgt_idx)[0]
        if len(positions) == 0:
            bm25_ranks.append({"rank": None, "RR": 0.0, "hit10": 0})
        else:
            rank = int(positions[0]) + 1
            bm25_ranks.append({"rank": rank, "RR": 1.0 / rank, "hit10": 1 if rank <= 10 else 0})

    log("\n=== 7. MiniLM retrieval (GPU) ===")
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    log(f"  encoding {len(corpus_texts)} corpus docs...")
    t0 = time.time()
    doc_embeds = model.encode(corpus_texts, batch_size=512, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)
    log(f"  docs encoded in {time.time() - t0:.1f}s")
    log(f"  encoding {len(queries)} queries...")
    t0 = time.time()
    q_embeds = model.encode(queries, batch_size=512, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
    log(f"  queries encoded in {time.time() - t0:.1f}s")
    doc_embeds_gpu = torch.from_numpy(doc_embeds).cuda()
    q_embeds_gpu = torch.from_numpy(q_embeds).cuda()
    sims = q_embeds_gpu @ doc_embeds_gpu.T
    log(f"  matmul done")

    minilm_ranks = []
    for i, r in enumerate(flat_queries):
        tgt_idx = asin_to_idx.get(r["asin"], -1)
        if tgt_idx < 0:
            minilm_ranks.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        sc = sims[i]
        rank = int((sc > sc[tgt_idx]).sum().item()) + 1
        minilm_ranks.append({"rank": rank, "RR": 1.0 / rank, "hit10": 1 if rank <= 10 else 0})

    del doc_embeds_gpu, q_embeds_gpu, sims
    torch.cuda.empty_cache()

    log("\n=== 8. Aggregating volatility metrics ===")
    per_query = []
    for r, bm25_r, minilm_r in zip(flat_queries, bm25_ranks, minilm_ranks):
        per_query.append({**r, "bm25_rank": bm25_r["rank"], "bm25_RR": bm25_r["RR"],
                          "bm25_hit10": bm25_r["hit10"],
                          "minilm_rank": minilm_r["rank"], "minilm_RR": minilm_r["RR"],
                          "minilm_hit10": minilm_r["hit10"]})
    with open(VOLATILITY_PER_QUERY_OUT, "w", encoding="utf-8") as f:
        json.dump({"config": {"description": "Stage 8.5.V volatility calibration"},
                   "queries": per_query}, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {VOLATILITY_PER_QUERY_OUT}")

    grouped = collections.defaultdict(lambda: {"bm25_RR": [], "minilm_RR": []})
    for r in per_query:
        key = (r["asin"], r["set_type"])
        grouped[key]["bm25_RR"].append(r["bm25_RR"])
        grouped[key]["minilm_RR"].append(r["minilm_RR"])

    def variance_stats(arr):
        a = np.array(arr)
        if len(a) < 2:
            return {"std": None, "iqr": None, "gap": None, "mean": float(a.mean()) if len(a) else None, "n": len(a)}
        return {
            "std": float(a.std()),
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "gap": float(a.max() - a.min()),
            "mean": float(a.mean()),
            "n": len(a),
        }

    asin_stats = {}
    for asin, sets in asin_query_sets.items():
        asin_stats[asin] = {}
        for set_type in ("low", "user", "high"):
            data = grouped[(asin, set_type)]
            asin_stats[asin][set_type] = {
                "bm25": variance_stats(data["bm25_RR"]),
                "minilm": variance_stats(data["minilm_RR"]),
            }

    def collect_var(metric_key, retriever, set_type):
        vals = []
        for asin, stats in asin_stats.items():
            v = stats[set_type][retriever][metric_key]
            if v is not None:
                vals.append(v)
        return np.array(vals) if vals else np.array([])

    summary_table = {}
    for retriever in ("bm25", "minilm"):
        summary_table[retriever] = {}
        for metric in ("std", "iqr", "gap", "mean"):
            row = {}
            for set_type in ("low", "user", "high"):
                v = collect_var(metric, retriever, set_type)
                row[set_type] = {
                    "mean": float(v.mean()) if len(v) else None,
                    "std": float(v.std()) if len(v) else None,
                    "n_asins": len(v),
                }
            if metric == "std":
                user_v = collect_var("std", retriever, "user")
                low_v = collect_var("std", retriever, "low")
                high_v = collect_var("std", retriever, "high")
                nvs = []
                for u, l, h in zip(user_v, low_v, high_v):
                    if h - l > 1e-6:
                        nvs.append((u - l) / (h - l))
                row["NV_mean"] = float(np.mean(nvs)) if nvs else None
                row["NV_std"] = float(np.std(nvs)) if nvs else None
            summary_table[retriever][metric] = row

    log(f"\n=== Volatility Summary Table ===")
    log(f"{'Retriever':<10} {'Metric':<8} {'V_low':<12} {'V_user':<12} {'V_high':<12} {'NV':<10}")
    for retriever in ("bm25", "minilm"):
        for metric in ("std", "iqr", "gap"):
            row = summary_table[retriever][metric]
            v_low = row["low"]["mean"]
            v_user = row["user"]["mean"]
            v_high = row["high"]["mean"]
            nv_str = f"{summary_table[retriever]['std'].get('NV_mean', 0):.3f}" if metric == "std" else "-"
            log(f"  {retriever:<10} {metric:<8} {v_low:<12.4f} {v_user:<12.4f} {v_high:<12.4f} {nv_str:<10}")

    log("\n=== 9. Saving summary ===")
    with open(VOLATILITY_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5.V volatility calibration summary",
                "K_SET": K_SET,
                "MAX_LEN_DIFF": 3,
                "PRIMARY_LEN_DELTA": PRIMARY_LEN_DELTA,
            },
            "summary_table": summary_table,
            "n_asins_with_sets": n_total,
            "n_insufficient_low": n_insufficient_low,
            "n_insufficient_high": n_insufficient_high,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {VOLATILITY_SUMMARY_OUT}")


# ===========================================================================
# STAGE 7 — A1 SELECTION
# ===========================================================================

def maha_diag_batch(z_arr: np.ndarray, mu_arr: np.ndarray, sigma_arr: np.ndarray) -> np.ndarray:
    diff = z_arr[:, None, :] - mu_arr[None, :, :]
    return np.sum(diff ** 2 / sigma_arr[None, :, :], axis=2)


def stage_a1_select():
    log("=== STAGE 7 — A1 SELECTION ===")

    log("\n=== 1. Loading user Gaussians ===")
    g_data = json.load(open(GAUSSIANS_IN))
    base_users = g_data["users"]
    log(f"  base users: {len(base_users)}")
    user_mu = {}
    user_sigma = {}
    for uid, u in base_users.items():
        user_mu[uid] = np.array(u["mu"], dtype=np.float64)
        user_sigma[uid] = np.array(u["sigma_diag"], dtype=np.float64)

    log("\n=== 2. Loading scaler + PCA48 ===")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    feature_names = P["feature_names_ordered"]

    feat_cache_path = FEAT_CACHE
    feat_map = {}
    with gzip.open(feat_cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  feat_map: {len(feat_map)}")

    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(P["X_scaled"]), size=min(5000, len(P["X_scaled"])), replace=False)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=PCA_SEED)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    log("\n=== 3. Projecting pool queries ===")
    pool_data = json.load(open(POOL_IN))
    pool_by_asin = pool_data["pools"]
    log(f"  n asins: {len(pool_by_asin)}")

    pool_z_by_asin = {}
    n_projected = 0
    n_skipped = 0
    for asin, queries in pool_by_asin.items():
        zs = []
        for q in queries:
            text = q["query"]
            k = hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()
            feats = feat_map.get(k)
            if not feats:
                zs.append(None)
                n_skipped += 1
                continue
            numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
            vec = np.array([numeric.get(n, 0.0) for n in feature_names], dtype=np.float64)
            vec_scaled = scaler.transform(vec[None, :])[0]
            vec_pca = pca.transform(vec_scaled[None, :])[0]
            zs.append(vec_pca)
            n_projected += 1
        pool_z_by_asin[asin] = zs
    log(f"  projected: {n_projected}, skipped: {n_skipped}")

    log("\n=== 4. Loading (asin, user) pairs ===")
    sel_data = json.load(open(SELECTION_IN))
    asin_users = collections.defaultdict(list)
    for e in sel_data["entries"]:
        asin = e["asin"]
        uid = e["user_id"]
        if uid not in user_mu:
            continue
        if uid not in asin_users[asin]:
            asin_users[asin].append(uid)
    log(f"  n asins with users: {len(asin_users)}")

    log("\n=== 5. Computing margins + running strategies ===")

    from sklearn.cluster import KMeans

    def run_strategy(z_arr: np.ndarray, mu_arr: np.ndarray, sigma_arr: np.ndarray,
                     valid_users: list, strategy: str):
        n_q = z_arr.shape[0]
        n_u = len(valid_users)
        D = maha_diag_batch(z_arr, mu_arr, sigma_arr)
        margins = np.zeros((n_u, n_q))
        d_selfs = np.zeros((n_u, n_q))
        for ui in range(n_u):
            d_self = D[:, ui]
            d_others = np.delete(D, ui, axis=1)
            d_mean_other = np.mean(d_others, axis=1)
            margin = d_mean_other - d_self
            margins[ui] = margin
            d_selfs[ui] = d_self

        if strategy == "mean_t50":
            selected = []
            for ui in range(n_u):
                d_self = d_selfs[ui]
                margin = margins[ui]
                threshold = float(np.percentile(d_self, THRESHOLD_PCT))
                valid = d_self <= threshold
                if not valid.any():
                    idx = int(np.argmax(margin))
                else:
                    m = np.where(valid, margin, -np.inf)
                    idx = int(np.argmax(m))
                selected.append(idx)
            return selected, margins, d_selfs

        elif strategy == "A1_reject_repeat":
            selected = []
            picked = set()
            for ui in range(n_u):
                d_self = d_selfs[ui]
                margin = margins[ui]
                threshold = float(np.percentile(d_self, THRESHOLD_PCT))
                valid = d_self <= threshold
                m = np.where(valid, margin, -np.inf)
                ranked = np.argsort(-m)
                chosen = None
                for c in ranked:
                    if c not in picked:
                        chosen = int(c)
                        break
                if chosen is None:
                    chosen = int(ranked[0])
                selected.append(chosen)
                picked.add(chosen)
            return selected, margins, d_selfs

        elif strategy.startswith("C3_kmeans_k"):
            k = int(strategy.rsplit("_k", 1)[1])
            user_clusters = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(mu_arr)
            selected = [0] * n_u
            used_global = set()
            for c in range(k):
                cluster_mask = (user_clusters == c)
                cluster_user_idxs = np.where(cluster_mask)[0]
                if len(cluster_user_idxs) == 0:
                    continue
                for local_i, ui in enumerate(cluster_user_idxs):
                    d_self = d_selfs[ui]
                    margin = margins[ui]
                    threshold = float(np.percentile(d_self, THRESHOLD_PCT))
                    valid = d_self <= threshold
                    m = np.where(valid, margin, -np.inf)
                    ranked = np.argsort(-m)
                    chosen = None
                    for cand in ranked:
                        if cand not in used_global:
                            chosen = int(cand)
                            break
                    if chosen is None:
                        chosen = int(ranked[0])
                    selected[ui] = chosen
                    used_global.add(chosen)
            return selected, margins, d_selfs

        else:
            raise ValueError(f"unknown strategy: {strategy}")

    strategies = [
        "mean_t50",
        "A1_reject_repeat",
        "C3_kmeans_k2",
        "C3_kmeans_k3",
        "C3_kmeans_k4",
        "C3_kmeans_k5",
    ]

    per_asin_results = {}
    rnd_dist_per_asin = {}

    for ai, (asin, users) in enumerate(asin_users.items()):
        if ai % 10 == 0:
            log(f"  [{ai}/{len(asin_users)}] asin={asin}, n_users={len(users)}")
        zs = pool_z_by_asin.get(asin, [])
        valid_idx = [i for i, z in enumerate(zs) if z is not None]
        if len(valid_idx) < 5 or len(users) < 2:
            continue
        z_arr = np.stack([zs[i] for i in valid_idx])
        valid_users = [u for u in users if u in user_mu]
        if len(valid_users) < 2:
            continue

        mu_arr = np.stack([user_mu[uid] for uid in valid_users])
        sigma_arr = np.stack([user_sigma[uid] for uid in valid_users])

        D = maha_diag_batch(z_arr, mu_arr, sigma_arr)
        rnd_per_user = [float(np.mean(D[:, ui])) for ui in range(len(valid_users))]
        rnd_dist_per_asin[asin] = np.mean(rnd_per_user)

        asin_strategies = {}
        for strategy in strategies:
            sel_idx, margins, d_selfs = run_strategy(z_arr, mu_arr, sigma_arr,
                                                     valid_users, strategy)
            sel_d_self = [float(d_selfs[ui, sel_idx[ui]]) for ui in range(len(valid_users))]
            asin_strategies[strategy] = {
                "selected_idx": [int(i) for i in sel_idx],
                "selected_d_self": sel_d_self,
                "n_unique": len(set(sel_idx)),
            }
        per_asin_results[asin] = asin_strategies

    log("\n=== 6. Aggregating metrics ===")
    summary = {}
    for strategy in strategies:
        uniqs = []
        sel_dists = []
        for asin, res in per_asin_results.items():
            uniqs.append(res[strategy]["n_unique"])
            sel_dists.extend(res[strategy]["selected_d_self"])
        rnd_mean = np.mean(list(rnd_dist_per_asin.values())) if rnd_dist_per_asin else 0.0
        summary[strategy] = {
            "n_asins": len(per_asin_results),
            "unique_mean": float(np.mean(uniqs)) if uniqs else 0.0,
            "unique_median": float(np.median(uniqs)) if uniqs else 0.0,
            "unique_min": int(min(uniqs)) if uniqs else 0,
            "unique_max": int(max(uniqs)) if uniqs else 0,
            "sel_dist_mean": float(np.mean(sel_dists)) if sel_dists else 0.0,
            "rnd_dist_mean": float(rnd_mean),
            "sel_rnd_ratio": float(np.mean(sel_dists) / rnd_mean) if rnd_mean > 0 else None,
        }

    log(f"\n{'strategy':<25} {'unique_mean':>12} {'unique_med':>11} {'sel_dist':>10} {'rnd_dist':>10} {'ratio':>7}")
    for strategy in strategies:
        s = summary[strategy]
        log(f"  {strategy:<23} {s['unique_mean']:>11.2f} {s['unique_median']:>11.1f} "
            f"{s['sel_dist_mean']:>9.1f} {s['rnd_dist_mean']:>9.1f} {s['sel_rnd_ratio']:>6.2f}")

    with open(A1_SELECTION_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 10K selection strategies",
                "PCA_DIM": PCA_DIM,
                "THRESHOLD_PCT": THRESHOLD_PCT,
                "strategies": strategies,
            },
            "per_asin": per_asin_results,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {A1_SELECTION_OUT}")

    with open(A1_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {A1_SUMMARY_OUT}")

    log("\n=== Stage 10K complete ===")


# ===========================================================================
# MAIN
# ===========================================================================

STAGE_FUNCTIONS = {
    "pool_regen": stage_pool_regen,
    "features": stage_features,
    "user_gaussians": stage_user_gaussians,
    "select": stage_select,
    "retrieval": stage_retrieval,
    "volatility": stage_volatility,
    "a1_select": stage_a1_select,
}

ALL_STAGES = list(STAGE_FUNCTIONS.keys())


def main():
    parser = argparse.ArgumentParser(description="Syntax Subspace Main Pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=ALL_STAGES + ["all"],
        help="Which stage to run",
    )
    args = parser.parse_args()

    log(f"=== syntax_subspace_main.py — stage={args.stage} ===")

    if args.stage == "all":
        for stage_name in ALL_STAGES:
            log(f"\n>>> Running stage: {stage_name}")
            STAGE_FUNCTIONS[stage_name]()
    else:
        STAGE_FUNCTIONS[args.stage]()


if __name__ == "__main__":
    main()