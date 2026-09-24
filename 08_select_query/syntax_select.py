#!/usr/bin/env python3
"""Query selection: per-user trainable svd_mlp Gaussian + logp_delta=2.0 gate.

Pipeline (canonical, 2026-09-17):
  1. 加载 svd_mlp trainable Gaussian (Stage 04b user_gaussian_stats_trainable.json)
  2. 加载 SVD components + StyleMLP encoder (Stage 03b)
  3. spaCy.pipe 编码 pool queries -> SVD 256d -> MLP 64d
  4. 对每个 query: 计算对所有 cohort 用户的 log p(z | N(μ_u, diag(σ_u²)))
  5. best-fit user = argmax logp; 保留 logp ≥ max_logp - LOGP_DELTA 的所有 query
  6. 输出 per-ASIN kept queries

复用资产:
  - SVD components: pcfg_cache/svd_components.npz (256d)
  - MLP encoder:    pcfg_cache/svd_mlp_encoder.pt (256→128→64)
  - Vocab:          pcfg_cache/vocab.json
  - Stage 04b:      result/04_gaussian/user_gaussian_stats_trainable.json (21521 users × 64d)
  - Pool queries:   result/07_gen_query/pool_queries.json
  - ASIN→users:     result/02_user_review_sentence_extract/asin_to_users.json

用法 (Rule 3: 无参数):
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  $PY 08_select_query/syntax_select_mahalanobis_gate.py

输出:
  - result/08_select_query/selected_queries_svdmlp.json
  - result/08_select_query/selection_stats_svdmlp.json
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import spacy
import torch
import torch.nn as nn
from scipy.stats import chi2 as _chi2_dist

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "03_spacy_encode"))
from syntax_encoder import StyleMLP  # cohort3 MLP (F.normalize + ReLU)
_cohort3_StyleMLP = StyleMLP
StyleMLP = _cohort3_StyleMLP  # override alias so downstream 'StyleMLP(...)' uses cohort3 MLP

POOL_PATH = REPO_ROOT / "result/07_gen_query/pool_queries.json"
# 用户指令 2026-09-23: asin_to_users 改 pkl-only (Stage 02 已切换).
ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract/asin_to_users_baby.pkl"
# 用户指令 2026-09-23: 3 个 category 各自一份 (Baby / Musical / Video_Games),
# main() 改为串行跑 3 个 domain, 产物写到 result/08_select_query/<subdir>/.
CATEGORY_INPUTS = [
    # (category_key, subdir)
    ("Baby",                "baby"),
    ("Musical_Instruments", "musical"),
    ("Video_Games",         "video_games"),
]

# --- svd_mlp canonical config (Rule 3: hardcoded) ---
DEVICE = "cuda:0"
ROW_NORMALIZE = False
LATENT_DIM = 16                   # cohort3 MLP output dim (this run)
GATE_MODE = "per_user_top1"     # per-user Top-1 with chi²_{D,0.95} fit + user-vs-competitor margin
LOGP_DELTA = 2.0                  # unused under per_user_top1 (kept for legacy reference)
GATE_Q = 0.05                     # unused under per_user_top1
# Chi² percentile gate (controls uniqueness radius): sweep over [0.75, 0.55,
# 0.35, 0.15, 0.005, 0.95]. The hard uniqueness rule D²(u*)≤chi² AND
# ∀v≠u*, D²(v)>chi² becomes stricter as q shrinks (smaller sphere).
D2_CHI2_Q95 = float(_chi2_dist.ppf(0.95, LATENT_DIM))   # chi²_{16, 0.95} theoretical
                                  # pool queries now sit on the review manifold so the theoretical
                                  # threshold becomes meaningful again.
# Sweep grid: percentile q of chi²_{D=32}; corresponding D² thresholds.
CHI2_SWEEP_Q = [0.95, 0.75, 0.55, 0.35, 0.15, 0.005]
CHI2_SWEEP_THRESHOLDS = {q: float(_chi2_dist.ppf(q, LATENT_DIM))
                         for q in CHI2_SWEEP_Q}
                                  # pool queries now sit on the review manifold so the theoretical
                                  # threshold becomes meaningful again.
MARGIN_MIN = 0.0                  # min user-vs-competitor margin (log_p[u*] - max_{v≠u*} log_p[v])
SEED_SVDMLP = 42
HARD_MIN_QUERIES_PER_ASIN = 1     # logp_delta mode 下保留至少 1 query/ASIN
BATCH_SIZE = 256                  # spaCy pipe batch size
SVD_COMPONENTS = "/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/svd_components.npz"
SVD_NPZ = SVD_COMPONENTS
SVD_DIM = 256
MLP_ENCODER = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort2_mlp16_30_30ep.pt"
MLP_PT = MLP_ENCODER
CORAL_ASIN_PATH = "/home/wlia0047/hj82_scratch2/wenyu/coral_asin_cohort2_mlp16_30/coral_asin_cohort2_mlp16_30.npz"
CORAL_ART_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/coral_asin_cohort2_mlp16_30")
CORAL_ART_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_UIDS_PATH = REPO_ROOT / "result/03_spacy_encode/cohort3_trained_uids.json"
STRICT3_NPZ = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/strict3_embeddings.npz")
Z_PROFILE_POOL = Path("/home/wlia0047/hj82_scratch2/wenyu/coral_cohort3mlp16_30/z_profile_review_cohort3mlp16_30.npz")
ASIN_TO_USERS = REPO_ROOT / "result/02_user_review_sentence_extract/asin_to_users_baby.pkl"
EPSILON_REL = 0.1              # 2026-09-19: 加大 Tikhonov regularization 让 A 接近 identity, 避免 query z 被过度压缩
COND_THRESHOLD = 1e3
MIN_USERS_PER_ASIN = 1
MIN_QUERIES_PER_ASIN = 1
CORAL_MODE = "asin"
APPLY_CORAL = True
SMOKE = os.environ.get("STAGE08_ALIGNMENT_SMOKE") == "1"
SMOKE_ASIN_LIMIT = 5
VOCAB_PATH = "/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/vocab.json"
GAUSSIAN_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_cohort3mlp16_30_lowrankdiag_rank1.json"
OUT_PATH = REPO_ROOT / "result/08_select_query/selected_queries.json"
OUT_STATS_PATH = REPO_ROOT / "result/08_select_query/selection_stats.json"
SPACY_MODEL = "en_core_web_sm"


def _svdmlp_log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _svdmlp_extract_struct_rules(doc):
    """从 spaCy Doc 提取 dependency + POS rules (D4/D3/P3), 与 syntax_pcfg_pipeline.py 同源。"""
    n = len(doc)
    if n < 3:
        return []
    pos = [t.pos_ for t in doc]
    heads_abs = [t.head.i for t in doc]
    deps = [t.dep_ for t in doc]
    rs = set()
    for i in range(n):
        h = heads_abs[i]
        if h == i:
            continue
        gh = heads_abs[h]
        gp_pos = pos[gh] if gh != h else "ROOT"
        rs.add(f"D4|{gp_pos}|{pos[h]}|{deps[i]}|{pos[i]}")
        rs.add(f"D3|{pos[h]}|{deps[i]}|{pos[i]}")
    for i in range(n - 2):
        rs.add(f"P3|{pos[i]}|{pos[i+1]}|{pos[i+2]}")
    return list(rs)


extract_struct_rules = _svdmlp_extract_struct_rules  # alias for svdmlp encoding pipeline




def _svdmlp_query_cache_key(
        vocab: list[str],
        cohort_filter_signature: str,
        queries: list[str],
        flat_records: list | None = None) -> str:
    """Fingerprint encoder inputs and the exact ordered query pool."""
    def _vocab_fingerprint(values):
        h = hashlib.sha256()
        h.update(b"vocab-v1\n")
        for value in values:
            h.update(len(value).to_bytes(8, "big"))
            h.update(value.encode("utf-8"))
        return h.hexdigest()

    h = hashlib.sha256()
    h.update(b"pool_query_cache_v2\n")
    h.update(_vocab_fingerprint(vocab).encode())
    svd_path = Path(SVD_COMPONENTS)
    if svd_path.exists():
        with open(svd_path, "rb") as f:
            h.update(f.read(1024))
            try:
                f.seek(-1024, 2)
                h.update(f.read(1024))
            except OSError:
                pass
    h.update(str(SVD_DIM).encode())
    h.update(cohort_filter_signature.encode())
    for index, query in enumerate(queries):
        if flat_records is not None:
            h.update(str(flat_records[index][0]).encode("utf-8"))
            h.update(b"\0")
        h.update(query.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _svdmlp_query_cache_path(cache_key: str) -> Path:
    return Path("/home/wlia0047/hj82_scratch2/wenyu/pool_query_cache") / (
        f"qcache_{cache_key[:16]}.npz")


def _svdmlp_encode_queries(
    queries: list[str],
    vocab: list[str],
    Vt: np.ndarray,
    mlp: StyleMLP,
    nlp,
    cohort_filter_signature: str | None = None,
    asin_alignment: dict | None = None,
    flat_records: list | None = None,
) -> tuple[np.ndarray, list[tuple[int, dict[int, float]]]]:
    """queries -> 64d z (svd_mlp encoder 完整流程).

    流程:
      1) spaCy.pipe (batch=BATCH_SIZE) -> docs
      2) extract_struct_rules -> 每句 set of rule_strs
      3) rule_str -> vocab index -> sparse row (1, V)
      4) row-normalize -> SVD 投影 (256d) -> MLP (64d)

    Cache: spaCy parse + sparse build + SVD-z projection are memoized to
    /home/wlia0047/hj82_scratch2/wenyu/pool_query_cache/. The MLP forward +
    CORAL alignment always run fresh because they depend on the encoder
    weights and CORAL alignment matrix (cheaper to recompute than to cache).
    """
    V = len(vocab)
    rule2idx = {r: i for i, r in enumerate(vocab)}
    N = len(queries)

    cache_path = None
    if cohort_filter_signature is not None:
        cache_key = _svdmlp_query_cache_key(
            vocab, cohort_filter_signature, queries, flat_records)
        cache_path = _svdmlp_query_cache_path(cache_key)

    z_svd: np.ndarray | None = None
    rows_data: list[tuple[int, dict[int, float]]] | None = None

    if cache_path is not None and cache_path.exists():
        try:
            npz = np.load(cache_path, allow_pickle=False)
            # Sanity: cache n_rows must equal N (matches cohort-filtered queries
            # input length). If vocab/SVD/cohort changed, cache_key differs and
            # this file is not even opened.
            cached_n_rows = int(npz["z_svd"].shape[0])
            if cached_n_rows == N:
                z_svd = np.asarray(npz["z_svd"], dtype=np.float32)
                flat_idx_arr = np.asarray(npz["flat_idx_arr"], dtype=np.int64)
                # rows_data is downstream consumed only for its first element
                # (flat_idx_for_row). Counts dict is unused on cache-hit path,
                # so we skip the O(N) dict rebuild and store empty counts.
                # This drops cache-hit latency from ~30s to <1s.
                rows_data = [(int(fi), {}) for fi in flat_idx_arr]
                _svdmlp_log(
                    f"  spaCy + SVD cache hit: {cache_path.name} "
                    f"({len(rows_data)} rows, z_svd={z_svd.shape})")
            else:
                _svdmlp_log(
                    f"  spaCy + SVD cache n_rows drift "
                    f"(cached={cached_n_rows}, current={N}); rebuilding")
        except (OSError, KeyError, ValueError) as e:
            _svdmlp_log(f"  spaCy + SVD cache load failed ({type(e).__name__}); "
                        f"rebuilding: {e}")

    if z_svd is None:
        _svdmlp_log(f"  encoding {N} queries with spaCy batch=2048 n_process=16")
        t0 = time.time()
        rows_data = []
        # 2026-09-18: 64-core 机器,n_process=16 + batch=2048,277K queries 跑 ~3500 docs/s
        for di, doc in enumerate(nlp.pipe(queries, batch_size=2048, n_process=16)):
            rules = extract_struct_rules(doc)
            counts: dict[int, float] = {}
            for r in rules:
                idx = rule2idx.get(r)
                if idx is None:
                    continue
                counts[idx] = counts.get(idx, 0) + 1.0
            if counts:
                rows_data.append((di, counts))
            if di % 10000 == 0 and di > 0:
                _svdmlp_log(f"    spacy {di}/{N}  "
                    f"elapsed={time.time() - t0:.1f}s")
        _svdmlp_log(f"  spacy + rule extraction: {len(rows_data)}/{N} non-empty "
            f"elapsed={time.time() - t0:.1f}s")

        # build sparse matrix
        data, indices, indptr = [], [], [0]
        for _, counts in rows_data:
            for k, v in counts.items():
                indices.append(k)
                data.append(v)
            indptr.append(len(indices))
        X = sp.csr_matrix(
            (np.asarray(data, dtype=np.float32),
             np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int32)),
            shape=(len(rows_data), V),
        )
        _svdmlp_log(f"  sparse X: {X.shape} nnz={X.nnz}")

        # row normalize
        if ROW_NORMALIZE:
            norms = np.asarray(sp.linalg.norm(X, axis=1)).ravel()
            norms[norms == 0] = 1.0
            inv = sp.diags(1.0 / norms)
            Xn = inv @ X
        else:
            Xn = X

        # SVD 投影
        z_svd = (Xn @ Vt.T).astype(np.float32)
        _svdmlp_log(f"  SVD projection: {z_svd.shape}")

        # Persist cache (atomically)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp.npz")
            # Re-build sparse in CSR for storage (compact)
            # rows_data is per-row counts dict; we don't need to persist the
            # sparse matrix itself — we re-derive counts from row_indices/row_data.
            flat_idx_arr = np.asarray([r[0] for r in rows_data], dtype=np.int64)
            row_indices_lens = [len(list(r[1].keys())) for r in rows_data]
            row_data_lens = row_indices_lens
            max_len = max(row_indices_lens) if row_indices_lens else 0
            # Pad row indices / data with -1 / 0 to fixed length for storage
            row_indices_padded = np.full(
                (len(rows_data), max_len), -1, dtype=np.int32
            )
            row_data_padded = np.zeros(
                (len(rows_data), max_len), dtype=np.float32
            )
            for i, (_, counts) in enumerate(rows_data):
                keys = list(counts.keys())
                vals = list(counts.values())
                row_indices_padded[i, :len(keys)] = keys
                row_data_padded[i, :len(vals)] = vals
            queries_text_hash_unused = None  # retained for legacy npz readers
            np.savez_compressed(
                tmp,
                z_svd=z_svd,
                flat_idx_arr=flat_idx_arr,
                row_indices=row_indices_padded,
                row_data=row_data_padded,
            )
            os.replace(tmp, cache_path)
            _svdmlp_log(f"  wrote cache → {cache_path} "
                f"({cache_path.stat().st_size // 1024} KB)")

    # MLP encode (batch) — always fresh
    z_all = np.zeros((z_svd.shape[0], LATENT_DIM), dtype=np.float32)
    mlp.eval()
    with torch.no_grad():
        for s in range(0, z_svd.shape[0], 1024):
            e = min(s + 1024, z_svd.shape[0])
            xb = torch.from_numpy(z_svd[s:e]).to(DEVICE)
            z_all[s:e] = mlp(xb).cpu().numpy()
    _svdmlp_log(f"  MLP encode done, shape {z_all.shape}")

    if APPLY_CORAL:
        if CORAL_MODE != "asin":
            raise ValueError("CORAL_MODE must be 'asin'; global alignment is forbidden")
        if asin_alignment is None or flat_records is None:
            raise ValueError("per-ASIN CORAL requires asin_alignment + flat_records")
        asin_keys = asin_alignment["asin_keys"]
        A_per = asin_alignment["A_per_asin"]
        muq_per = asin_alignment["mu_q_per_asin"]
        mur_per = asin_alignment["mu_r_per_asin"]
        asin_to_idx = {str(a): i for i, a in enumerate(asin_keys)}
        missing = sorted({
            str(flat_records[rec_idx][0])
            for rec_idx, _ in rows_data
            if str(flat_records[rec_idx][0]) not in asin_to_idx
        })
        if missing:
            raise RuntimeError(
                f"per-ASIN CORAL missing {len(missing)} encoded ASINs; "
                f"examples={missing[:10]}")
        for row_i, (rec_idx, _) in enumerate(rows_data):
            asin = str(flat_records[rec_idx][0])
            ai = asin_to_idx[asin]
            z_all[row_i] = (
                mur_per[ai] + (z_all[row_i] - muq_per[ai]) @ A_per[ai].T
            )
        _svdmlp_log(
            f"  CORAL per-ASIN applied: {len(rows_data)} rows aligned, "
            "0 fallback rows")
    return z_all, rows_data


def _svdmlp_load_gaussian() -> tuple[dict, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 cohort3mlp 32d lowrank+diag Gaussian; 预计算 mu(N,32), lambdas(N,K_RANK),
    V(N,32,K_RANK), sigma_diag_sq(N,32), uid_order(N,)."""
    _svdmlp_log(f"loading {GAUSSIAN_PATH}")
    with open(GAUSSIAN_PATH) as f:
        d = json.load(f)
    cfg = d["config"]
    if cfg.get("encoder_source") != "cohort2_mlp16_30_30ep":
        raise ValueError(
            f"gaussian cfg encoder_source={cfg.get('encoder_source')}, "
            "expected cohort2_mlp16_30_30ep")
    if int(cfg["syntax_dim"]) != LATENT_DIM:
        raise ValueError(
            f"gaussian syntax_dim={cfg['syntax_dim']}, script LATENT_DIM={LATENT_DIM}")
    k_rank = int(cfg.get("k_rank", 1))
    users = d["users"]
    uid_order = sorted(users.keys())
    N = len(uid_order)
    mu = np.zeros((N, LATENT_DIM), dtype=np.float32)
    lambdas = np.zeros((N, k_rank), dtype=np.float32)
    V = np.zeros((N, LATENT_DIM, k_rank), dtype=np.float32)
    sigma_diag_sq = np.zeros((N, LATENT_DIM), dtype=np.float32)
    d2_q = np.zeros(N, dtype=np.float32)
    for i, uid in enumerate(uid_order):
        u = users[uid]
        mu[i] = np.asarray(u["mu"], dtype=np.float32)
        lambdas[i] = np.asarray(u["lambdas"], dtype=np.float32)
        V[i] = np.asarray(u["V"], dtype=np.float32)  # stored as (D, K_RANK)
        sigma_diag_sq[i] = np.asarray(u["sigma_diag_sq"], dtype=np.float32)
        d2_q[i] = float(u["d2_q95"])
    uid_idx = {uid: i for i, uid in enumerate(uid_order)}
    return uid_idx, uid_order, mu, lambdas, V, sigma_diag_sq, d2_q


def _fit_coral_one(z_q_asin: np.ndarray, z_r_asin: np.ndarray,
                   epsilon_rel: float = EPSILON_REL,
                   return_cond: bool = False):
    """Fit regularized per-ASIN CORAL A = C_r^{1/2} C_q^{-1/2}."""
    K = z_q_asin.shape[1]
    mu_q = z_q_asin.mean(axis=0).astype(np.float32)
    mu_r = z_r_asin.mean(axis=0).astype(np.float32)

    def _covariance(x: np.ndarray) -> np.ndarray:
        if x.shape[0] <= 1:
            return np.zeros((K, K), dtype=np.float64)
        cov = np.asarray(np.cov(x.astype(np.float64), rowvar=False), dtype=np.float64)
        if cov.ndim == 0:
            return np.zeros((K, K), dtype=np.float64)
        return cov.reshape(K, K)

    Cq = _covariance(z_q_asin)
    Cr = _covariance(z_r_asin)
    eps = max(float(np.trace(Cq) / K) * epsilon_rel, 1e-6)
    Cr_reg = Cr + eps * np.eye(K, dtype=np.float64)
    Cq_reg = Cq + eps * np.eye(K, dtype=np.float64)
    w_r, V_r = np.linalg.eigh((Cr_reg + Cr_reg.T) / 2)
    w_q, V_q = np.linalg.eigh((Cq_reg + Cq_reg.T) / 2)
    w_r = np.clip(w_r, 0.0, None)
    w_q = np.clip(w_q, 0.0, None)
    Sr = V_r @ np.diag(np.sqrt(w_r)) @ V_r.T
    Sq_inv = V_q @ np.diag(1.0 / np.clip(np.sqrt(w_q), 1e-12, None)) @ V_q.T
    A = Sr @ Sq_inv
    U, singular_values, Vh = np.linalg.svd(A, full_matrices=False)
    if not np.isfinite(singular_values).all() or singular_values[0] <= 0:
        raise np.linalg.LinAlgError("non-finite or zero per-ASIN CORAL transform")
    singular_values = np.maximum(
        singular_values, singular_values[0] / COND_THRESHOLD)
    A = ((U * singular_values) @ Vh).astype(np.float32)
    if not np.isfinite(A).all() or not np.isfinite(mu_q).all() or not np.isfinite(mu_r).all():
        raise np.linalg.LinAlgError("non-finite per-ASIN CORAL parameters")
    if return_cond:
        return A, mu_q, mu_r, float(np.linalg.cond(A.astype(np.float64)))
    return A, mu_q, mu_r


def _fit_coral_ensure_artifacts(pool: dict, vocab: list[str],
                                Vt: np.ndarray, mlp, fitted_uid_set: set[str]
                                ) -> dict:
    """Fit and persist one regularized CORAL transform for every candidate ASIN."""
    _svdmlp_log("=== Per-ASIN CORAL fit (global fallback disabled) ===")

    with open(GAUSSIAN_PATH) as f:
        gaussian_users = json.load(f)["users"]
    review_uid_order = sorted(gaussian_users)
    z_r_user = np.asarray(
        [gaussian_users[uid]["mu"] for uid in review_uid_order],
        dtype=np.float32,
    )
    real_uid_to_zr_row = {uid: i for i, uid in enumerate(review_uid_order)}
    _svdmlp_log(f"  Gaussian profile-review means: {z_r_user.shape}")

    rule2idx = {r: i for i, r in enumerate(vocab)}
    flat_records: list[tuple[str, str, int]] = [
        (asin, query, local_idx)
        for asin, queries in pool.items()
        for local_idx, query in enumerate(queries)
    ]
    queries_text = [record[1] for record in flat_records]
    _svdmlp_log(
        f"  encoding {len(queries_text)} pool queries for per-ASIN fitting")

    nlp_local = spacy.load(SPACY_MODEL, disable=["ner", "textcat", "lemmatizer"])
    query_asin_kept: list[str] = []
    rows_data: list[tuple[int, dict[int, float]]] = []
    t0 = time.time()
    checkpoint_every = max(1, len(flat_records) // 20)
    for rec_idx, doc in enumerate(
            nlp_local.pipe(queries_text, batch_size=2048, n_process=16)):
        asin = flat_records[rec_idx][0]
        idx_set = {
            rule2idx[rule]
            for rule in _svdmlp_extract_struct_rules(doc)
            if rule in rule2idx
        }
        if idx_set:
            query_asin_kept.append(asin)
            rows_data.append((rec_idx, dict.fromkeys(idx_set, 1.0)))
        if (rec_idx + 1) % checkpoint_every == 0:
            elapsed = time.time() - t0
            _svdmlp_log(
                f"    spaCy {rec_idx + 1}/{len(flat_records)} "
                f"elapsed={elapsed:.1f}s")
    if not rows_data:
        raise RuntimeError("no candidate query produced structural features")

    n_rows = len(rows_data)
    row_lens = [len(counts) for _, counts in rows_data]
    data = np.ones(sum(row_lens), dtype=np.float32)
    indices = np.concatenate([
        np.fromiter(counts.keys(), dtype=np.int32)
        for _, counts in rows_data
    ])
    indptr = np.zeros(n_rows + 1, dtype=np.int32)
    np.cumsum(row_lens, out=indptr[1:])
    X = sp.csr_matrix((data, indices, indptr), shape=(n_rows, len(vocab)))
    z_svd = (X @ Vt.T).astype(np.float32)
    mlp.eval()
    with torch.no_grad():
        z_q_all = np.empty((n_rows, LATENT_DIM), dtype=np.float32)
        for start in range(0, n_rows, 1024):
            end = min(start + 1024, n_rows)
            xb = torch.from_numpy(z_svd[start:end]).to(DEVICE)
            z_q_all[start:end] = mlp(xb).cpu().numpy()
    _svdmlp_log(f"  encoded fitting queries: {z_q_all.shape}")

    asin_to_qrows: dict[str, list[int]] = {}
    for row_idx, asin in enumerate(query_asin_kept):
        asin_to_qrows.setdefault(asin, []).append(row_idx)

    with open(ASIN_TO_USERS, "rb") as f:
        asin_to_users_raw = pickle.load(f)
    asin_to_ruser: dict[str, list[int]] = {}
    for asin, uids in asin_to_users_raw.items():
        r_idx = [
            real_uid_to_zr_row[uid]
            for uid in uids
            if uid in real_uid_to_zr_row
        ]
        if r_idx:
            asin_to_ruser[str(asin)] = r_idx

    candidate_asins = set(pool)
    missing_queries = sorted(candidate_asins - set(asin_to_qrows))
    missing_reviews = sorted(candidate_asins - set(asin_to_ruser))
    if missing_queries or missing_reviews:
        raise RuntimeError(
            "cannot align every candidate ASIN: "
            f"{len(missing_queries)} lack query embeddings "
            f"(examples={missing_queries[:10]}), "
            f"{len(missing_reviews)} lack cohort review means "
            f"(examples={missing_reviews[:10]})")

    out_keys: list[str] = []
    out_A: list[np.ndarray] = []
    out_mu_q: list[np.ndarray] = []
    out_mu_r: list[np.ndarray] = []
    for index, asin in enumerate(sorted(candidate_asins), start=1):
        q_idx = asin_to_qrows[asin]
        r_idx = asin_to_ruser[asin]
        if len(q_idx) < MIN_QUERIES_PER_ASIN or len(r_idx) < MIN_USERS_PER_ASIN:
            raise RuntimeError(
                f"cannot fit per-ASIN CORAL for {asin}: "
                f"queries={len(q_idx)}, review_users={len(r_idx)}")
        A, mu_q, mu_r, condition = _fit_coral_one(
            z_q_all[q_idx], z_r_user[r_idx], return_cond=True)
        if not np.isfinite(condition) or condition > COND_THRESHOLD * (1 + 1e-5):
            raise RuntimeError(
                f"unstable per-ASIN CORAL for {asin}: cond={condition:g}")
        out_keys.append(asin)
        out_A.append(A)
        out_mu_q.append(mu_q)
        out_mu_r.append(mu_r)
        if index % max(1, len(candidate_asins) // 20) == 0:
            _svdmlp_log(
                f"    CORAL fit {index}/{len(candidate_asins)} ASINs")

    if set(out_keys) != candidate_asins:
        raise RuntimeError("per-ASIN CORAL coverage differs from candidate pool")
    asin_alignment = {
        "asin_keys": np.asarray(out_keys, dtype=object),
        "A_per_asin": np.stack(out_A, axis=0),
        "mu_q_per_asin": np.stack(out_mu_q, axis=0),
        "mu_r_per_asin": np.stack(out_mu_r, axis=0),
    }
    np.savez(
        CORAL_ASIN_PATH,
        asin_keys=asin_alignment["asin_keys"],
        A_per_asin=asin_alignment["A_per_asin"],
        mu_q_per_asin=asin_alignment["mu_q_per_asin"],
        mu_r_per_asin=asin_alignment["mu_r_per_asin"],
        min_users_per_asin=np.int32(MIN_USERS_PER_ASIN),
        min_queries_per_asin=np.int32(MIN_QUERIES_PER_ASIN),
        cond_threshold=np.float32(COND_THRESHOLD),
        epsilon=np.float32(EPSILON_REL),
        dim=np.int32(LATENT_DIM),
    )
    _svdmlp_log(
        f"  fitted and saved per-ASIN CORAL for {len(out_keys)} ASINs "
        f"→ {Path(CORAL_ASIN_PATH).name}")
    return asin_alignment


def main_task_body() -> None:
    _svdmlp_log(f"device={DEVICE}  GATE_MODE={GATE_MODE}  "
        f"LOGP_DELTA={LOGP_DELTA}  GATE_Q={GATE_Q}  LATENT_DIM={LATENT_DIM}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    _svdmlp_log("loading vocab, svd components, mlp encoder")
    vocab = json.load(open(VOCAB_PATH))
    _svdmlp_log(f"  vocab: {len(vocab)} rules")
    with np.load(SVD_COMPONENTS) as npz:
        Vt = np.asarray(npz["Vt"], dtype=np.float32)
    _svdmlp_log(f"  Vt: {Vt.shape}  row_normalize={ROW_NORMALIZE}")
    ckpt = torch.load(MLP_ENCODER, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        mlp_cfg = ckpt.get("config", {})
    else:
        # cohort3 ckpt is raw state_dict (saved via torch.save(enc.state_dict(), ...))
        state_dict = ckpt
        mlp_cfg = {}
    mlp = StyleMLP().to(DEVICE)
    mlp.load_state_dict(state_dict)
    _svdmlp_log(f"  mlp: in={mlp_cfg.get('svd_dim')} hidden={mlp_cfg.get('hidden')} "
        f"out={mlp_cfg.get('out_dim')}")

    _svdmlp_log("loading pool queries")
    pool_doc = json.load(open(POOL_PATH))
    # Accept both the canonical {"pool": ...} wrapper and the direct
    # dictionary emitted by the transformers fallback. Normalize candidate
    # records to query strings for the selection pipeline.
    pool_raw = pool_doc.get("pool", pool_doc) if isinstance(pool_doc, dict) else {}
    pool = {}
    for asin, candidates in pool_raw.items():
        normalized = []
        for candidate in candidates:
            text = candidate.get("text") if isinstance(candidate, dict) else candidate
            if isinstance(text, str) and text.strip():
                normalized.append(text.strip())
        if normalized:
            pool[asin] = normalized
    if SMOKE:
        pool = dict(list(pool.items())[:SMOKE_ASIN_LIMIT])
        if not pool:
            raise RuntimeError("Stage08 alignment smoke pool is empty")
        _svdmlp_log(
            f"  SMOKE pool: {len(pool)} ASINs, "
            f"{sum(map(len, pool.values()))} queries")
    asin_to_users = pickle.load(open(ASIN_USERS_PATH, "rb"))
    _svdmlp_log(f"  pool: {len(pool)} ASINs")

    _svdmlp_log("loading cohort3mlp lowrank+diag Gaussian")
    uid_idx, uid_order, mu, lambdas, V_eig, sigma_diag_sq, d2_q95 = _svdmlp_load_gaussian()
    _svdmlp_log(f"  fitted users: {len(uid_idx)}")

    # Pre-filter: only keep queries whose ASIN has at least one fitted cohort
    # user. Queries for ASINs with no cohort coverage cannot win any user and
    # would be filtered out by `cand_idx` later — skipping spaCy here saves
    # ~75% of encoding cost on the cohort3mlp variant.
    _svdmlp_log("flattening queries (cohort-filtered)")
    fitted_uid_set = set(uid_idx.keys())
    flat_records: list[tuple[str, str, int, str]] = []  # (asin, query, local_idx, qstr)
    n_filtered_asins = 0
    for asin, queries in pool.items():
        cohort_users = [u for u in asin_to_users.get(asin, [])
                        if u in fitted_uid_set]
        if not cohort_users:
            n_filtered_asins += 1
            continue
        for li, q in enumerate(queries):
            flat_records.append((asin, q, li, q))
    _svdmlp_log(
        f"  cohort-keep: {len(flat_records)} queries "
        f"({n_filtered_asins}/{len(pool)} ASINs had zero cohort users, "
        f"skipped before spaCy)")
    if n_filtered_asins:
        raise RuntimeError(
            f"cannot score every candidate ASIN: {n_filtered_asins} have no "
            "fitted cohort user; global fallback is forbidden")

    nlp = spacy.load(SPACY_MODEL, disable=["ner", "textcat", "lemmatizer"])
    queries_text = [r[1] for r in flat_records]
    # Cohort filter signature: hash of sorted fitted-uid-set so cache invalidates
    # if the cohort coverage changes (e.g. new cohort3 encoder produces a
    # different trained-uid whitelist).
    cohort_sig = hashlib.sha256(
        ("\n".join(sorted(fitted_uid_set))).encode("utf-8")
    ).hexdigest()
    if APPLY_CORAL and CORAL_MODE == "asin":
        asin_alignment = _fit_coral_ensure_artifacts(
            pool, vocab, Vt, mlp, fitted_uid_set)
        _svdmlp_log(
            f"  per-ASIN CORAL: {len(asin_alignment['asin_keys'])}/"
            f"{len(pool)} candidate ASINs aligned")
    z_all, rows_data = _svdmlp_encode_queries(
        queries_text, vocab, Vt, mlp, nlp,
        cohort_filter_signature=cohort_sig,
        asin_alignment=asin_alignment,
        flat_records=flat_records,
    )

    # 还原 row -> flat_records index
    flat_idx_for_row = [rec_idx for rec_idx, _ in rows_data]

    # 预计算每用户的 log-norm 常数 (low-rank + diag Gaussian):
    #   log|Σ| = log|diag(σ_d²)| + log|I + V^T diag(1/σ_d²) V diag(λ)|
    # 使用 Sylvester 行列式引理;额外加 d log 2π 项。
    inv_sigma_diag = 1.0 / sigma_diag_sq                          # (N, D)
    K_RANK = lambdas.shape[1]
    log_norm_per_user = np.zeros(len(uid_order), dtype=np.float64)
    for n in range(len(uid_order)):
        lnd_diag = np.log(sigma_diag_sq[n]).sum()                 # log|diag(σ_d²)|
        M = (V_eig[n].T * inv_sigma_diag[n]) @ V_eig[n]           # (K, K)
        lnd_correction = np.linalg.slogdet(np.eye(K_RANK) + M * lambdas[n])[1]
        log_norm_per_user[n] = 0.5 * (
            lnd_diag + lnd_correction + LATENT_DIM * np.log(2 * np.pi)
        )

    _svdmlp_log(f"per-query best-fit user + per-user margin Top-1 sweep "
                f"(D² thresholds q∈{CHI2_SWEEP_Q}, "
                f"corresponding D²∈{[round(CHI2_SWEEP_THRESHOLDS[q],2) for q in CHI2_SWEEP_Q]})")
    # per_asin → query × (u*, logp_u*, d2_u*, margin, n_inside_per_q, query_text)
    # n_inside_per_q: dict {q: count} of cohort users whose D² ≤ chi²_{D,q}
    # Precompute everything per query once; sweep multiple q thresholds by
    # re-deriving gate decisions.
    per_asin_records: dict[str, list[dict]] = {}
    n_no_user = 0
    selection_started = time.time()
    selection_total = len(flat_idx_for_row)
    selection_progress_every = max(1, (selection_total + 19) // 20)
    _svdmlp_log(f"  query scoring start: {selection_total} encoded queries")
    for ri, (z_row, flat_idx) in enumerate(zip(z_all, flat_idx_for_row)):
        asin, q_text, local_idx, _ = flat_records[flat_idx]
        user_list = asin_to_users.get(asin, [])
        cand_idx = [uid_idx[u] for u in user_list if u in uid_idx]
        if not cand_idx:
            n_no_user += 1
            if (ri + 1) % selection_progress_every == 0 or ri + 1 == selection_total:
                _svdmlp_log(f"    query scoring {ri+1}/{selection_total} "
                            f"({100 * (ri+1) / selection_total:.1f}%), "
                            f"no_user={n_no_user}, "
                            f"elapsed={time.time() - selection_started:.1f}s")
            continue
        cand_idx_arr = np.asarray(cand_idx, dtype=np.int64)
        diff = z_row[None, :] - mu[cand_idx_arr]                          # (n_cand, D)
        cand_V = V_eig[cand_idx_arr]                                       # (n_cand, D, K_RANK)
        cand_l = lambdas[cand_idx_arr]                                     # (n_cand, K_RANK)
        cand_sd = sigma_diag_sq[cand_idx_arr]                              # (n_cand, D)
        low = np.einsum("cd,cdk->ck", diff, cand_V)
        d2_low = (low ** 2 / cand_l).sum(axis=-1)
        proj_back = np.einsum("ck,cdk->cd", low, cand_V)
        r = diff - proj_back
        d2_diag = (r ** 2 / cand_sd).sum(axis=-1)
        d2 = d2_low + d2_diag
        log_p = -0.5 * (d2 + log_norm_per_user[cand_idx_arr])
        best_k = int(np.argmax(log_p))
        # margin vs runner-up (any other user, not just top-1)
        sorted_logp = np.sort(log_p)[::-1]
        margin = float(sorted_logp[0] - sorted_logp[1]) if len(sorted_logp) >= 2 else float("inf")
        n_inside_per_q = {
            q: int(np.sum(d2 <= thr))
            for q, thr in CHI2_SWEEP_THRESHOLDS.items()
        }
        per_asin_records.setdefault(asin, []).append({
            "logp": float(log_p[best_k]),
            "d2": float(d2[best_k]),
            "user_idx": int(cand_idx_arr[best_k]),
            "margin": margin,
            "n_inside_per_q": n_inside_per_q,
            "query": q_text,
        })
        if (ri + 1) % selection_progress_every == 0 or ri + 1 == selection_total:
            _svdmlp_log(f"    query scoring {ri+1}/{selection_total} "
                        f"({100 * (ri+1) / selection_total:.1f}%), "
                        f"no_user={n_no_user}, "
                        f"elapsed={time.time() - selection_started:.1f}s")

    _svdmlp_log(f"  queries with candidate: {sum(len(v) for v in per_asin_records.values())}/{len(flat_records)}")

    # Diagnostic: d² distribution on best-fit per-query (before gates) for gate tuning
    all_d2 = np.concatenate([np.asarray([c["d2"] for c in v]) for v in per_asin_records.values()]) if per_asin_records else np.array([])
    if len(all_d2) > 0:
        _svdmlp_log(
            f"  d²(u*) diag: n={len(all_d2)} "
            f"min={float(all_d2.min()):.2f} "
            f"p50={float(np.percentile(all_d2, 50)):.2f} "
            f"p90={float(np.percentile(all_d2, 90)):.2f} "
            f"p95={float(np.percentile(all_d2, 95)):.2f} "
            f"p99={float(np.percentile(all_d2, 99)):.2f} "
            f"max={float(all_d2.max()):.2f}"
        )

    # Sweep chi² quantile q. For each q we run the gate:
    #   - D²(q, u*) ≤ chi²_{D, q} AND ∀ v≠u*, D²(q, v) > chi²_{D, q} (uniqueness)
    #   - margin ≥ MARGIN_MIN
    #   - bucket by (asin, user_idx), pick max logp per bucket
    sweep_results: dict[str, dict] = {}
    for q in CHI2_SWEEP_Q:
        thr = CHI2_SWEEP_THRESHOLDS[q]
        kept: dict[str, list[str]] = {}
        selections_block: list = []
        drops = {"above_d2": 0, "below_margin": 0, "multi_user_inside": 0}
        for asin, cands in per_asin_records.items():
            bucket: dict[int, list[dict]] = {}
            for c in cands:
                if c["d2"] > thr:
                    drops["above_d2"] += 1
                    continue
                n_in = c["n_inside_per_q"][q]
                if n_in > 1:
                    drops["multi_user_inside"] += 1
                    continue
                if c["margin"] < MARGIN_MIN:
                    drops["below_margin"] += 1
                    continue
                bucket.setdefault(c["user_idx"], []).append(c)
            per_user_picks: list[dict] = []
            for uid_idx_in_asin, group in bucket.items():
                best = max(group, key=lambda c: (c["logp"], c["margin"]))
                per_user_picks.append(best)
            if not per_user_picks:
                continue
            kept[asin] = [c["query"] for c in per_user_picks]
            users_block = [
                {
                    "uid": uid_order[c["user_idx"]],
                    "query": c["query"],
                    "logp": c["logp"],
                    "d2": c["d2"],
                    "margin": c["margin"],
                }
                for c in per_user_picks
            ]
            selections_block.append({
                "asin": asin,
                "users": users_block,
                "max_pair_cos": None,
            })

        counts = [len(v) for v in kept.values()]
        n_ge1 = sum(1 for c in counts if c >= 1)
        n_ge2 = sum(1 for c in counts if c >= 2)
        n_ge3 = sum(1 for c in counts if c >= 3)
        n_ge5 = sum(1 for c in counts if c >= 5)
        _svdmlp_log(
            f"  q={q:.3f} (D²≤{thr:.3f}): "
            f"kept={sum(len(v) for v in kept.values())} "
            f"ASINs={len(kept)} "
            f"(≥1u={n_ge1}, ≥2u={n_ge2}, ≥3u={n_ge3}, ≥5u={n_ge5}) "
            f"drops={drops}")
        sweep_results[q] = {
            "threshold": thr,
            "kept": kept,
            "selections": selections_block,
            "stats": {
                "q": q,
                "d2_threshold": thr,
                "n_kept_queries": sum(len(v) for v in kept.values()),
                "n_total_queries": len(flat_records),
                "n_total_asin": len(pool),
                "n_asin_ge1": n_ge1,
                "n_asin_ge2": n_ge2,
                "n_asin_ge3": n_ge3,
                "n_asin_ge5": n_ge5,
                "drops": drops,
            },
        }

    canonical_q = 0.95
    if canonical_q not in sweep_results:
        canonical_q = CHI2_SWEEP_Q[0]
    canonical = sweep_results[canonical_q]

    out = {
        "config": {
            "source": "cohort3mlp32_lowrankdiag",
            "latent_dim": LATENT_DIM,
            "gate_mode": GATE_MODE,
            "d2_chi2_q_default": canonical_q,
            "d2_chi2_threshold_default": canonical["threshold"],
            "chi2_sweep_q": CHI2_SWEEP_Q,
            "chi2_sweep_thresholds": CHI2_SWEEP_THRESHOLDS,
            "margin_min": MARGIN_MIN,
            "gaussian_path": str(GAUSSIAN_PATH),
            "svd_components": str(SVD_COMPONENTS),
            "mlp_encoder": str(MLP_ENCODER),
            "n_pool_asin": len(pool),
            "n_total_queries": len(flat_records),
            "n_fitted_users": len(uid_idx),
            "device": DEVICE,
            "selection_rule": (
                f"per-(asin,user) Top-1 with hard uniqueness: best_fit=user u* "
                f"requires (1) D²(q,u*) ≤ chi²_{LATENT_DIM},q where q is the "
                f"sweep percentile, (2) ∀ v≠u*, D²(q,v) > chi²_{LATENT_DIM},q "
                f"(query lies in EXACTLY ONE user's q-region; drop if multi-user), "
                f"(3) margin = logp(q|u*) - max_{{v≠u*}} logp(q|v) ≥ MARGIN_MIN; "
                f"tie-break: max logp then max margin."
            ),
            "d2_formula": "lowrank+diag: d2 = Σ_i (v_i·diff)²/λ_i + r²/σ_d², r = diff - (diff·V)V^T",
            "log_norm_formula": "0.5 * (log|diag(σ_d²)| + slogdet(I + V^T diag(1/σ_d²) V diag(λ)) + D log 2π)",
        },
        "kept": canonical["kept"],
        "selections": canonical["selections"],
        "sweep_results": {
            str(q): {
                "threshold": sweep_results[q]["threshold"],
                "stats": sweep_results[q]["stats"],
            }
            for q in CHI2_SWEEP_Q
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    _svdmlp_log(f"DONE wrote {OUT_PATH} (canonical q={canonical_q})")

    summary_stats = {
        "gate_mode": GATE_MODE,
        "latent_dim": LATENT_DIM,
        "chi2_sweep_q": CHI2_SWEEP_Q,
        "chi2_sweep_thresholds": CHI2_SWEEP_THRESHOLDS,
        "margin_min": MARGIN_MIN,
        "n_total_queries": len(flat_records),
        "n_total_asin": len(pool),
        "n_no_user": n_no_user,
        "per_q": [sweep_results[q]["stats"] for q in CHI2_SWEEP_Q],
    }
    with open(OUT_STATS_PATH, "w") as f:
        json.dump(summary_stats, f, indent=2)
    _svdmlp_log(f"DONE wrote {OUT_STATS_PATH}")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """用户指令 2026-09-23: 串行运行 3 个 category.

    每个 category 重新绑定该脚本使用的路径常量为 category-specific 路径,
    然后调原 main_task_body() (保持原有逻辑不动). 产物写到
    result/<stage>/<baby|musical|video_games>/ 子目录.
    """
    global SENT_CACHE, UID_TO_SENTS, ASIN_USERS_PATH, ATTRIBUTES_PATH, META_FILE, OUT_DIR, OUT_PATH, OUT_STATS_PATH, POOL_PATH, GAUSSIAN_PATH, ASIN_TO_USERS, SVD_COMPONENTS, SVD_NPZ, MLP_ENCODER, MLP_PT, VOCAB_PATH, TRAINED_UIDS_PATH, STRICT3_NPZ, CORAL_ART_DIR, CORAL_ASIN_PATH  # noqa
    # backup current (Baby) defaults
    saved = {
        k: v for k, v in globals().items()
        if k in {"SENT_CACHE", "UID_TO_SENTS", "ASIN_USERS_PATH", "ATTRIBUTES_PATH",
                 "META_FILE", "OUT_DIR", "OUT_PATH", "OUT_STATS_PATH", "POOL_PATH",
                 "GAUSSIAN_PATH", "ASIN_TO_USERS"}
        and isinstance(v, Path)
    }
    base_out = REPO_ROOT / "result" / Path(__file__).parent.name
    for category, subdir in CATEGORY_INPUTS:
        _svdmlp_log(f"\n========== [{category}] (subdir={subdir}) ==========")
        # Reset all known category-dependent paths to point at the per-category subdir.
        cache_dir = Path("/home/wlia0047/hj82_scratch2/wenyu") / f"pcfg_cache_{subdir}"
        SVD_COMPONENTS = str(cache_dir / "svd_components.npz")
        SVD_NPZ = SVD_COMPONENTS
        VOCAB_PATH = str(cache_dir / "vocab.json")
        MLP_ENCODER = str(REPO_ROOT / "result/03_spacy_encode" / subdir /
                          "cohort2_mlp16_30_30ep.pt")
        MLP_PT = MLP_ENCODER
        TRAINED_UIDS_PATH = (REPO_ROOT / "result/03_spacy_encode" / subdir /
                             "cohort3_trained_uids.json")
        STRICT3_NPZ = cache_dir / "strict3_embeddings.npz"
        CORAL_ART_DIR = (Path("/home/wlia0047/hj82_scratch2/wenyu") /
                         f"coral_asin_cohort2_mlp16_30_{subdir}")
        CORAL_ART_DIR.mkdir(parents=True, exist_ok=True)
        CORAL_ASIN_PATH = str(CORAL_ART_DIR / "coral_asin_cohort2_mlp16_30.npz")
        if "SENT_CACHE" in saved:
            SENT_CACHE = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "UID_TO_SENTS" in saved:
            UID_TO_SENTS = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "ASIN_USERS_PATH" in saved:
            ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract" / f"asin_to_users_{subdir}.pkl"
        if "ATTRIBUTES_PATH" in saved:
            ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction" / f"product_attributes_{subdir}.pkl"
        if "META_FILE" in saved:
            META_FILE = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data") / {
                "baby": "meta_Baby_Products_2023.jsonl",
                "musical": "meta_Musical_Instruments.jsonl",
                "video_games": "meta_Video_Games.jsonl",
            }[subdir]
        if "OUT_DIR" in saved:
            OUT_DIR = base_out / subdir
        if "OUT_PATH" in saved:
            OUT_PATH = base_out / subdir / saved["OUT_PATH"].name
        if "OUT_STATS_PATH" in saved:
            OUT_STATS_PATH = base_out / subdir / saved["OUT_STATS_PATH"].name
        if SMOKE:
            smoke_dir = (Path("/home/wlia0047/hj82_scratch2/wenyu") /
                         "stage08_alignment_smoke" / subdir)
            OUT_PATH = smoke_dir / "selected_queries.json"
            OUT_STATS_PATH = smoke_dir / "selection_stats.json"
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            _svdmlp_log(
                f"  SMOKE enabled: max {SMOKE_ASIN_LIMIT} ASINs; "
                f"outputs in {smoke_dir}")
        if "POOL_PATH" in saved:
            POOL_PATH = REPO_ROOT / "result/07_gen_query" / subdir / saved["POOL_PATH"].name
        if "GAUSSIAN_PATH" in saved:
            GAUSSIAN_PATH = REPO_ROOT / "result/04_gaussian" / subdir / saved["GAUSSIAN_PATH"].name
        if "ASIN_TO_USERS" in saved:
            ASIN_TO_USERS = REPO_ROOT / "result/02_user_review_sentence_extract" / f"asin_to_users_{subdir}.pkl"
        OUT_DIR.mkdir(parents=True, exist_ok=True) if "OUT_DIR" in saved else None
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True) if "OUT_PATH" in saved else None
        OUT_STATS_PATH.parent.mkdir(parents=True, exist_ok=True) if "OUT_STATS_PATH" in saved else None
        POOL_PATH.parent.mkdir(parents=True, exist_ok=True) if "POOL_PATH" in saved else None
        GAUSSIAN_PATH.parent.mkdir(parents=True, exist_ok=True) if "GAUSSIAN_PATH" in saved else None
        try:
            main_task_body()
        except Exception as e:
            _svdmlp_log(f"[{category}] FAILED: {e!r}")
            raise
    # Restore Baby defaults (for import compatibility with downstream).
    for k, v in saved.items():
        globals()[k] = v


if __name__ == "__main__":
    main()
