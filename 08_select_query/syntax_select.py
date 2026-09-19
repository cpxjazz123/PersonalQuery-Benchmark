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
ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract/asin_to_users.json"

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
CORAL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/coral_cohort3mlp16_30/coral_cohort3mlp16_30.npz"
CORAL_ASIN_PATH = "/home/wlia0047/hj82_scratch2/wenyu/coral_asin_cohort2_mlp16_30/coral_asin_cohort2_mlp16_30.npz"
CORAL_ART_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/coral_asin_cohort2_mlp16_30")
CORAL_ART_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_UIDS_PATH = REPO_ROOT / "result/03_spacy_encode/cohort3_trained_uids.json"
STRICT3_NPZ = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/strict3_embeddings.npz")
Z_PROFILE_POOL = Path("/home/wlia0047/hj82_scratch2/wenyu/coral_cohort3mlp16_30/z_profile_review_cohort3mlp16_30.npz")
ASIN_TO_USERS = REPO_ROOT / "result/02_user_review_sentence_extract/asin_to_users.json"
EPSILON_REL = 0.1              # 2026-09-19: 加大 Tikhonov regularization 让 A 接近 identity, 避免 query z 被过度压缩
COND_THRESHOLD = 1e3
MIN_USERS_PER_ASIN = 1
MIN_QUERIES_PER_ASIN = 1
CORAL_MODE = "asin"   # "global" | "asin" — per-ASIN alignment when asin
APPLY_CORAL = True
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


class _svdmlp_StyleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


StyleMLP = _svdmlp_StyleMLP


def _svdmlp_query_cache_key(vocab: list[str], cohort_filter_signature: str) -> str:
    """Build a stable cache key for the spaCy + SVD-z pipeline.

    Inputs (cheap to compute, captures all sources of cache invalidation):
      - vocab fingerprint (Stage 03 PCFG rules) so cache invalidates if vocab changes
      - SVD fingerprint (svd_components.npz head/tail bytes) so cache invalidates
        if SVD dim/components change
      - cohort_filter_signature: caller-supplied stable hash of the cohort filter
        (e.g. sorted fitted-uid-set). Cache invalidates if cohort coverage changes.

    We intentionally do NOT include queries text in the key: the cohort filter
    already restricts which queries are processed (the rest are dropped before
    spaCy), so the per-query text is implicitly fixed by the filter signature.
    """
    def _vocab_fingerprint(values):
        h = hashlib.sha256()
        h.update(b"vocab-v1\n")
        for v in values:
            h.update(len(v).to_bytes(8, "big"))
            h.update(v.encode("utf-8"))
        return h.hexdigest()

    h = hashlib.sha256()
    h.update(b"pool_query_cache_v1\n")
    h.update(_vocab_fingerprint(vocab).encode())
    # SVD fingerprint: cheap surrogate via head + tail bytes of svd_components.npz
    svd_path = Path(SVD_COMPONENTS)
    if svd_path.exists():
        with open(svd_path, "rb") as f:
            head = f.read(1024)
            try:
                f.seek(-1024, 2)
                tail = f.read(1024)
            except OSError:
                tail = b""
        h.update(head)
        h.update(tail)
    h.update(b"svd_dim=")
    h.update(str(SVD_DIM).encode())
    h.update(b"cohort_filter=")
    h.update(cohort_filter_signature.encode())
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
        cache_key = _svdmlp_query_cache_key(vocab, cohort_filter_signature)
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
        if CORAL_MODE == "asin":
            if asin_alignment is None or flat_records is None:
                raise ValueError("CORAL_MODE=asin requires asin_alignment + flat_records")
            asin_keys = asin_alignment["asin_keys"]          # (n_asin,) object str
            A_per = asin_alignment["A_per_asin"]              # (n_asin, 16, 16)
            muq_per = asin_alignment["mu_q_per_asin"]         # (n_asin, 16)
            mur_per = asin_alignment["mu_r_per_asin"]         # (n_asin, 16)
            global_A = asin_alignment["global_A"]
            global_muq = asin_alignment["global_mu_q"]
            global_mur = asin_alignment["global_mu_r"]
            asin_to_idx = {str(a): i for i, a in enumerate(asin_keys)}
            # rows_data order = filtered (non-empty) flat_records
            n_fallback = 0
            n_asin_used = 0
            for row_i, (rec_idx, _) in enumerate(rows_data):
                asin = str(flat_records[rec_idx][0])
                ai = asin_to_idx.get(asin)
                if ai is None:
                    A_a, muq_a, mur_a = global_A, global_muq, global_mur
                    n_fallback += 1
                else:
                    A_a = A_per[ai]
                    muq_a = muq_per[ai]
                    mur_a = mur_per[ai]
                    n_asin_used += 1
                z_all[row_i] = mur_a + (z_all[row_i] - muq_a) @ A_a.T
            _svdmlp_log(f"  CORAL per-ASIN applied: {n_asin_used} rows per-ASIN, "
                        f"{n_fallback} rows fallback (global A); "
                        f"||A-I||_F={float(np.linalg.norm(global_A - np.eye(LATENT_DIM, dtype=np.float32))):.3f}")
        else:
            if not Path(CORAL_PATH).exists():
                raise FileNotFoundError(
                    f"APPLY_CORAL=True but {CORAL_PATH} missing; "
                    "regenerate the global CORAL npz and rerun")
            c = np.load(CORAL_PATH)
            mu_q = c["mu_query"].astype(np.float32)
            mu_r = c["mu_review"].astype(np.float32)
            A = c["A"].astype(np.float32)
            z_all = mu_r[None, :] + (z_all - mu_q[None, :]) @ A.T
            _svdmlp_log(f"  CORAL applied: ||A-I||={float(np.linalg.norm(A - np.eye(LATENT_DIM, dtype=np.float32))):.3f}, "
                        f"ε={float(c['epsilon']):.6f}")
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
    """Fit A = C_r^{1/2} C_q^{-1/2}, return (A, mu_q, mu_r[, cond_A])."""
    K = z_q_asin.shape[1]
    mu_q = z_q_asin.mean(axis=0).astype(np.float32)
    mu_r = z_r_asin.mean(axis=0).astype(np.float32)
    Cq = np.cov(z_q_asin.astype(np.float64), rowvar=False) + 0.0
    Cr = np.cov(z_r_asin.astype(np.float64), rowvar=False) + 0.0
    eps = float(np.trace(Cq) / K) * epsilon_rel
    Cr_reg = Cr + eps * np.eye(K, dtype=np.float64)
    Cq_reg = Cq + eps * np.eye(K, dtype=np.float64)
    w_r, V_r = np.linalg.eigh((Cr_reg + Cr_reg.T) / 2)
    w_q, V_q = np.linalg.eigh((Cq_reg + Cq_reg.T) / 2)
    w_r = np.clip(w_r, 0.0, None)
    w_q = np.clip(w_q, 0.0, None)
    Sr = V_r @ np.diag(np.sqrt(w_r)) @ V_r.T
    Sq_inv = V_q @ np.diag(1.0 / np.clip(np.sqrt(w_q), 1e-12, None)) @ V_q.T
    A = (Sr @ Sq_inv).astype(np.float32)
    if return_cond:
        return A, mu_q, mu_r, float(np.linalg.cond(A.astype(np.float64)))
    return A, mu_q, mu_r


def _fit_coral_ensure_artifacts(pool: dict, vocab: list[str],
                                Vt: np.ndarray, mlp, fitted_uid_set: set[str]
                                ) -> dict:
    """If CORAL_ASIN_PATH missing or pool_query_cache key mismatch → run full fit,
    write both coral_asin_*.npz AND qcache_*.npz. Otherwise load + return."""
    cohort_sig = hashlib.sha256(
        ("\n".join(sorted(fitted_uid_set))).encode("utf-8")
    ).hexdigest()
    cache_key = _svdmlp_query_cache_key(vocab, cohort_sig)
    cache_path = _svdmlp_query_cache_path(cache_key)

    need_coral = not Path(CORAL_ASIN_PATH).exists()
    need_cache = not cache_path.exists()
    if not need_coral and not need_cache:
        _svdmlp_log(f"  loading per-ASIN CORAL alignment from {Path(CORAL_ASIN_PATH).name}")
        ca = np.load(CORAL_ASIN_PATH, allow_pickle=True)
        return {
            "asin_keys": ca["asin_keys"],
            "A_per_asin": ca["A_per_asin"],
            "mu_q_per_asin": ca["mu_q_per_asin"],
            "mu_r_per_asin": ca["mu_r_per_asin"],
            "global_A": ca["global_A"],
            "global_mu_q": ca["global_mu_q"],
            "global_mu_r": ca["global_mu_r"],
        }
    # 2026-09-18: 如果只有 cache 缺失 (pool 扩展加了 ASINs), 但 CORAL npz 还在,
    # 跳过 coral fit, 只重建 cache (per-ASIN A 保留原值, 新 ASIN 走 fallback global A)。
    if not need_coral and need_cache:
        _svdmlp_log(
            f"  CORAL npz exists ({Path(CORAL_ASIN_PATH).name}) but cache missing; "
            f"rebuild cache only (preserve existing per-ASIN CORAL)")
        # Caller will load existing coral AFTER this function returns;
        # we return None and let _svdmlp_main load npz directly.
        ca = np.load(CORAL_ASIN_PATH, allow_pickle=True)
        return {
            "asin_keys": ca["asin_keys"],
            "A_per_asin": ca["A_per_asin"],
            "mu_q_per_asin": ca["mu_q_per_asin"],
            "mu_r_per_asin": ca["mu_r_per_asin"],
            "global_A": ca["global_A"],
            "global_mu_q": ca["global_mu_q"],
            "global_mu_r": ca["global_mu_r"],
            "_rebuild_cache_only": True,
        }

    _svdmlp_log("=== Per-ASIN CORAL auto-fit (cache/npz missing) ===")

    # Load z_profile_review
    zrp = np.load(Z_PROFILE_POOL)
    z_r_user = zrp["z"].astype(np.float32)            # (n_user, K)
    uid_idx_r = zrp["uid_idx"]                         # index into strict3 uid_list
    _svdmlp_log(f"  z_profile_review: {z_r_user.shape}")

    # Build flat_records in pool order, plus rule2idx
    rule2idx = {r: i for i, r in enumerate(vocab)}
    flat_records: list[tuple[str, str, int]] = []
    for asin, queries in pool.items():
        for li, q in enumerate(queries):
            flat_records.append((asin, q, li))
    queries_text = [r[1] for r in flat_records]
    _svdmlp_log(f"  re-encoding {len(queries_text)} queries (spaCy n_process=8 batched)...")

    nlp_local = spacy.load(SPACY_MODEL, disable=["ner", "textcat", "lemmatizer"])
    query_asin_kept: list[str] = []
    rows_data: list[tuple[int, dict[int, float]]] = []
    t0 = time.time()
    n_total = len(flat_records)
    checkpoint_every = max(1, n_total // 20)
    for rec_idx, doc in enumerate(nlp_local.pipe(queries_text, batch_size=2048, n_process=16)):
        a_str = flat_records[rec_idx][0]
        idx_set = {rule2idx[r] for r in _svdmlp_extract_struct_rules(doc) if r in rule2idx}
        if idx_set:
            query_asin_kept.append(a_str)
            rows_data.append((rec_idx, dict.fromkeys(idx_set, 1.0)))
        if (rec_idx + 1) % checkpoint_every == 0:
            elapsed = time.time() - t0
            rate = (rec_idx + 1) / elapsed
            _svdmlp_log(f"    spaCy {rec_idx+1}/{n_total}  "
                        f"elapsed={elapsed:.1f}s  rate={rate:.1f} docs/s")
    _svdmlp_log(f"    query_asin_kept rows = {len(query_asin_kept)}")

    # Sparse + SVD projection
    n_rows = len(rows_data)
    V_dim = Vt.shape[1]
    row_lens = [len(d) for _, d in rows_data]
    data = np.ones(sum(row_lens), dtype=np.float32)
    indices = np.concatenate([np.fromiter(d.keys(), dtype=np.int32) for _, d in rows_data])
    indptr = np.zeros(n_rows + 1, dtype=np.int32)
    cursor = 0
    for i, n in enumerate(row_lens):
        indptr[i] = cursor
        cursor += n
    indptr[-1] = cursor
    X = sp.csr_matrix((data, indices, indptr), shape=(n_rows, V_dim))
    z_svd = (X @ Vt.T).astype(np.float32)
    _svdmlp_log(f"  z_svd shape={z_svd.shape}")

    # Persist Stage 8 cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.npz")
    flat_idx_arr = np.asarray([r[0] for r in rows_data], dtype=np.int64)
    max_len = max(row_lens) if row_lens else 0
    row_indices_padded = np.full((n_rows, max_len), -1, dtype=np.int32)
    row_data_padded = np.zeros((n_rows, max_len), dtype=np.float32)
    for i, (_, counts) in enumerate(rows_data):
        keys = list(counts.keys())
        vals = list(counts.values())
        row_indices_padded[i, :len(keys)] = keys
        row_data_padded[i, :len(vals)] = vals
    np.savez_compressed(
        tmp, z_svd=z_svd, flat_idx_arr=flat_idx_arr,
        row_indices=row_indices_padded, row_data=row_data_padded,
    )
    os.replace(tmp, cache_path)
    _svdmlp_log(f"  wrote pool_query_cache → {cache_path.name} "
                f"({cache_path.stat().st_size // 1024} KB, key={cache_key[:16]})")

    # MLP encode
    mlp.eval()
    with torch.no_grad():
        z_q_all = np.empty((n_rows, LATENT_DIM), dtype=np.float32)
        for s in range(0, n_rows, 1024):
            xb = torch.from_numpy(z_svd[s:s + 1024]).to(DEVICE)
            z_q_all[s:s + 1024] = mlp(xb).cpu().numpy()
    _svdmlp_log(f"  encoded z_query_pool: {z_q_all.shape}")
    del z_svd

    # asin -> query row indices
    asin_to_qrows: dict[str, list[int]] = {}
    for i, asin in enumerate(query_asin_kept):
        asin_to_qrows.setdefault(asin, []).append(i)

    # asin -> cohort3-trained user indices in z_r_user
    asin_to_users_raw = json.load(open(ASIN_TO_USERS))
    trained_uid_set = set(json.load(open(TRAINED_UIDS_PATH)))
    npz_strict3 = np.load(STRICT3_NPZ, allow_pickle=False)
    uid_list = np.asarray(npz_strict3["uid_list"], dtype=object)
    real_uid_to_zr_row: dict[str, int] = {}
    for i in range(len(uid_idx_r)):
        real_uid = str(uid_list[int(uid_idx_r[i])])
        if real_uid in trained_uid_set:
            real_uid_to_zr_row[real_uid] = i
    _svdmlp_log(f"  cohort3-trained users with z_r_user: {len(real_uid_to_zr_row)}")
    asin_to_ruser: dict[str, list[int]] = {}
    for asin, uids in asin_to_users_raw.items():
        r_idx_list = [real_uid_to_zr_row[u] for u in uids if u in real_uid_to_zr_row]
        if r_idx_list:
            asin_to_ruser[asin] = r_idx_list
    _svdmlp_log(f"  ASINs with cohort3 review users: {len(asin_to_ruser)}")

    valid_asins = sorted(set(asin_to_qrows.keys()) & set(asin_to_ruser.keys()))
    _svdmlp_log(f"  ASINs with both queries AND cohort reviews: {len(valid_asins)}")

    # Per-ASIN CORAL fit
    out_keys: list[str] = []
    out_A: list[np.ndarray] = []
    out_mu_q: list[np.ndarray] = []
    out_mu_r: list[np.ndarray] = []
    n_fitted = 0
    n_skip_cond = 0
    n_skip_linalg = 0
    n_skip_short = 0
    for asin in valid_asins:
        q_idx = asin_to_qrows[asin]
        r_idx = asin_to_ruser[asin]
        if len(q_idx) < MIN_QUERIES_PER_ASIN or len(r_idx) < MIN_USERS_PER_ASIN:
            n_skip_short += 1
            continue
        z_q_a = z_q_all[q_idx]
        z_r_a = z_r_user[r_idx]
        try:
            A_a, muq_a, mur_a, cond_a = _fit_coral_one(
                z_q_a, z_r_a, return_cond=True)
        except np.linalg.LinAlgError:
            n_skip_linalg += 1
            continue
        if not np.isfinite(cond_a) or cond_a > COND_THRESHOLD:
            n_skip_cond += 1
            continue
        out_keys.append(asin)
        out_A.append(A_a)
        out_mu_q.append(muq_a)
        out_mu_r.append(mur_a)
        n_fitted += 1
    _svdmlp_log(f"  fitted A_asin for {n_fitted} ASINs "
                f"(skip short={n_skip_short}, cond>{COND_THRESHOLD:g}={n_skip_cond}, "
                f"linalg={n_skip_linalg})")

    # Global fallback A: 2026-09-18 限定只用 reuse_asins 子集的 queries 算 (避免新 ASIN
    # 拉偏全局分布导致 d² 大幅退化)。reuse_asins 从 pool_queries.json config 读。
    reuse_asin_list: list[str] = []
    try:
        _pool_doc = json.load(open(POOL_PATH))
        reuse_asin_list = list(_pool_doc.get("config", {}).get("reuse_asin_list", []))
    except (OSError, KeyError, ValueError):
        reuse_asin_list = []
    reuse_row_set = set()
    if reuse_asin_list:
        reuse_asin_set = set(reuse_asin_list)
        for row_i, asin in enumerate(query_asin_kept):
            if asin in reuse_asin_set:
                reuse_row_set.add(row_i)
        if reuse_row_set:
            z_q_global = z_q_all[sorted(reuse_row_set)]
            _svdmlp_log(f"  fitting global fallback CORAL on reuse subset ({len(reuse_row_set)} rows / "
                        f"{len(reuse_asin_list)} reuse ASINs)...")
        else:
            z_q_global = z_q_all
    else:
        z_q_global = z_q_all
    A_global, mu_q_g, mu_r_g = _fit_coral_one(z_q_global, z_r_user)

    n_asin = len(out_keys)
    asin_alignment = {
        "asin_keys": np.asarray(out_keys, dtype=object),
        "A_per_asin": (np.stack(out_A, axis=0) if n_asin > 0
                       else np.zeros((0, LATENT_DIM, LATENT_DIM), dtype=np.float32)),
        "mu_q_per_asin": (np.stack(out_mu_q, axis=0) if n_asin > 0
                          else np.zeros((0, LATENT_DIM), dtype=np.float32)),
        "mu_r_per_asin": (np.stack(out_mu_r, axis=0) if n_asin > 0
                          else np.zeros((0, LATENT_DIM), dtype=np.float32)),
        "global_A": A_global,
        "global_mu_q": mu_q_g,
        "global_mu_r": mu_r_g,
    }
    np.savez(
        CORAL_ASIN_PATH,
        asin_keys=asin_alignment["asin_keys"],
        A_per_asin=asin_alignment["A_per_asin"],
        mu_q_per_asin=asin_alignment["mu_q_per_asin"],
        mu_r_per_asin=asin_alignment["mu_r_per_asin"],
        global_A=asin_alignment["global_A"],
        global_mu_q=asin_alignment["global_mu_q"],
        global_mu_r=asin_alignment["global_mu_r"],
        min_users_per_asin=np.int32(MIN_USERS_PER_ASIN),
        min_queries_per_asin=np.int32(MIN_QUERIES_PER_ASIN),
        cond_threshold=np.float32(COND_THRESHOLD),
        epsilon=np.float32(EPSILON_REL),
        dim=np.int32(LATENT_DIM),
    )
    _svdmlp_log(f"  wrote → {Path(CORAL_ASIN_PATH).name} "
                f"({Path(CORAL_ASIN_PATH).stat().st_size // 1024} KB)")
    return asin_alignment


def _svdmlp_main() -> None:
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
    mlp = StyleMLP(
        in_dim=mlp_cfg.get("svd_dim") or mlp_cfg.get("in_dim") or SVD_DIM,
        hidden=mlp_cfg.get("hidden") or 128,
        out_dim=mlp_cfg.get("out_dim") or LATENT_DIM,
    ).to(DEVICE)
    mlp.load_state_dict(state_dict)
    _svdmlp_log(f"  mlp: in={mlp_cfg.get('svd_dim')} hidden={mlp_cfg.get('hidden')} "
        f"out={mlp_cfg.get('out_dim')}")

    _svdmlp_log("loading pool queries")
    pool = json.load(open(POOL_PATH))["pool"]
    asin_to_users = json.load(open(ASIN_USERS_PATH))
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

    nlp = spacy.load(SPACY_MODEL, disable=["ner", "textcat", "lemmatizer"])
    queries_text = [r[1] for r in flat_records]
    # Cohort filter signature: hash of sorted fitted-uid-set so cache invalidates
    # if the cohort coverage changes (e.g. new cohort3 encoder produces a
    # different trained-uid whitelist).
    cohort_sig = hashlib.sha256(
        ("\n".join(sorted(fitted_uid_set))).encode("utf-8")
    ).hexdigest()
    asin_alignment: dict | None = None
    if APPLY_CORAL and CORAL_MODE == "asin":
        asin_alignment = _fit_coral_ensure_artifacts(
            pool, vocab, Vt, mlp, fitted_uid_set)
        _svdmlp_log(
            f"  per-ASIN CORAL: {len(asin_alignment['asin_keys'])} ASINs "
            f"(global fallback A ready)")
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
    for ri, (z_row, flat_idx) in enumerate(zip(z_all, flat_idx_for_row)):
        asin, q_text, local_idx, _ = flat_records[flat_idx]
        user_list = asin_to_users.get(asin, [])
        cand_idx = [uid_idx[u] for u in user_list if u in uid_idx]
        if not cand_idx:
            n_no_user += 1
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


if __name__ == "__main__":
    _svdmlp_main()
