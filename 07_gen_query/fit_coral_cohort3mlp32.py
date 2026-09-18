#!/usr/bin/env python3
"""P0: Fit global CORAL alignment between pool-query and profile-review 32d latent.

Inputs (both reused from existing artifacts, NOT regenerated):
  - profile review 32d embeddings: reuses Stage 4 cohort3mlp encoding of
    strict3 profile_idx × SVD-z × cohort3 MLP encoder (= 985 cohort3-trained
    users' profile sentences in 32d).
  - pool query 32d embeddings: encoded once from pool_queries.json via
    spaCy + SVD + cohort3 MLP. Cached so Stage 08 can reuse without re-spacy.

Algorithm:
  μ_q, Σ  : mean and covariance over pool query z_q.
  μ_r, C_r : mean and covariance over profile review z_r.
  ε = trace(Σ)/32 * 1e-4  (relative regularization).
  Σ' = Σ + ε I
  C_r' = C_r + ε I
  A   = C_r'^{1/2} Σ'^{-1/2}

Align any new query z_q:
  z_aligned = μ_r + A @ (z_q - μ_q)

No F.normalize after CORAL — keep the exact latent scale the Gaussian was fit on.

Outputs:
  - z_query_pool_cohort3mlp32.npz   (z_q, asin_idx, local_idx)
  - z_profile_review_cohort3mlp32.npz (z_r, uid_idx)  [optional, derived from cache]
  - coral_cohort3mlp32.npz         (mu_query, mu_review, A, epsilon, dim)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/03_spacy_encode")

import torch
from train_real_asin_cohort3 import StyleMLP

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
ART_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/coral_cohort3mlp32")
ART_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_WEIGHTS_CANDIDATES = [
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp32_30_30ep.pt"),
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp30_30ep.pt"),
    Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp30_10ep.pt"),
]
STRICT3_NPZ = CACHE_DIR / "strict3_embeddings.npz"
USER_N_SENTS = CACHE_DIR / "user_n_sents.json"
SVD_COMPONENTS = CACHE_DIR / "svd_components.npz"
SENT_VECTORS = CACHE_DIR / "sent_vectors.npz"
POOL_PATH = REPO_ROOT / "result/07_gen_query/pool_queries.json"

K_DIM = 32
COHORT3_TRAINED_UIDS = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_trained_uids.json")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42

OUT_Z_PROFILE = ART_DIR / "z_profile_review_cohort3mlp32.npz"
OUT_Z_QUERY = ART_DIR / "z_query_pool_cohort3mlp32.npz"
OUT_CORAL = ART_DIR / "coral_cohort3mlp32.npz"

EPSILON_REL = 1e-4   # regularization relative to trace(C)/dim


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_encoder() -> torch.nn.Module:
    weights_path = next((p for p in ENCODER_WEIGHTS_CANDIDATES if p.exists()), None)
    if weights_path is None:
        raise FileNotFoundError(f"no encoder weights in {ENCODER_WEIGHTS_CANDIDATES}")
    enc = StyleMLP().to(DEVICE)
    enc.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    enc.eval()
    log(f"  loaded encoder from {weights_path}")
    return enc


def load_sparse_sent_vectors() -> sp.csr_matrix:
    d = np.load(SENT_VECTORS, allow_pickle=True)
    return sp.csr_matrix(
        (d["data"].astype(np.float32), d["indices"].astype(np.int32), d["indptr"].astype(np.int32)),
        shape=tuple(d["shape"]),
    )


def encode_rows(z_all: np.ndarray, rows: np.ndarray, enc, bs: int = 4096) -> np.ndarray:
    out = np.empty((len(rows), K_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            idx = rows[s:s + bs]
            z_t = torch.from_numpy(z_all[idx]).to(DEVICE)
            out[s:s + bs] = enc(z_t).cpu().numpy()
    return out


def encode_profile_review(z_all: np.ndarray, enc, trained_uid_idx_set: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """Encode strict3 profile_idx rows, restrict to cohort3-trained uids.

    Returns (z_profile_review, kept_uid_idx) where kept_uid_idx is a sorted
    int array of user indices that contributed at least one profile row.
    """
    npz = np.load(STRICT3_NPZ, allow_pickle=False)
    profile_idx = np.asarray(npz["profile_idx"], dtype=np.int64)
    uid_list = np.asarray(npz["uid_list"])
    with open(USER_N_SENTS) as f:
        user_n_sents = [int(n) for n in json.load(f)]
    row2uid = np.repeat(np.arange(len(uid_list), dtype=np.int64), user_n_sents)
    prof_uid = row2uid[profile_idx]
    mask = np.isin(prof_uid, np.asarray(sorted(trained_uid_idx_set), dtype=np.int64))
    kept_rows = profile_idx[mask]
    kept_uids = prof_uid[mask]
    log(f"  encoding {len(kept_rows)} profile review rows (cohort3-trained users)…")
    z = encode_rows(z_all, kept_rows, enc)
    # Average per user
    order = np.argsort(kept_uids, kind="stable")
    z_sorted = z[order]
    u_sorted = kept_uids[order]
    unique_uids, inverse = np.unique(u_sorted, return_inverse=True)
    n_users = len(unique_uids)
    z_user = np.zeros((n_users, K_DIM), dtype=np.float32)
    np.add.at(z_user, inverse, z_sorted)
    counts = np.bincount(inverse, minlength=n_users).astype(np.float32)
    z_user /= counts[:, None]
    log(f"  per-user profile review embeddings: {z_user.shape} (mean of {int(counts.mean())} rows/user)")
    return z_user, unique_uids


def encode_pool_queries(z_all: np.ndarray, enc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode pool queries via spaCy + SVD + cohort3 MLP."""
    import spacy

    pool = json.load(open(POOL_PATH))["pool"]
    log(f"  pool: {len(pool)} ASINs")
    flat_records: list[tuple[str, str, int]] = []
    for asin, queries in pool.items():
        for li, q in enumerate(queries):
            flat_records.append((asin, q, li))
    queries_text = [r[1] for r in flat_records]

    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "lemmatizer"])
    log(f"  encoding {len(queries_text)} pool queries via spaCy…")

    # Reuse the same spaCy → PCFG rules → SVD pipeline as Stage 8.
    # PCFG rules from Stage 02/03 — we re-import the rule extractor by
    # loading vocab.json (vocab mapping for SVD).
    VOCAB_PATH = CACHE_DIR / "vocab.json"
    vocab = json.load(open(VOCAB_PATH))
    rule2idx = {r: i for i, r in enumerate(vocab)}
    # Use the canonical 08_select_query rules extractor indirectly by
    # re-importing its helpers:
    sys.path.insert(0, str(REPO_ROOT / "08_select_query"))
    from syntax_select_mahalanobis_gate import _svdmlp_extract_struct_rules
    # SVD components Vt (K x V) — note: rule count = vocab size
    with np.load(SVD_COMPONENTS) as npz_svd:
        Vt = np.asarray(npz_svd["Vt"], dtype=np.float32)
    log(f"  Vt: {Vt.shape}")

    rows_data = []
    for doc in nlp.pipe(queries_text, batch_size=256):
        idx_set = set()
        for rule in _svdmlp_extract_struct_rules(doc):
            ri = rule2idx.get(rule)
            if ri is not None:
                idx_set.add(ri)
        if idx_set:
            rows_data.append((np.fromiter(idx_set, dtype=np.int32), len(idx_set)))
    if not rows_data:
        raise RuntimeError("no pool query has any rule after filtering")

    n_rows = len(rows_data)
    V = Vt.shape[1]
    data = np.ones(sum(rd[1] for rd in rows_data), dtype=np.float32)
    indices = np.concatenate([rd[0] for rd in rows_data])
    indptr = np.zeros(n_rows + 1, dtype=np.int32)
    cursor = 0
    for i, (_, n) in enumerate(rows_data):
        indptr[i] = cursor
        cursor += n
    indptr[-1] = cursor
    X = sp.csr_matrix((data, indices, indptr), shape=(n_rows, V))
    log(f"  sparse X: {X.shape} nnz={X.nnz}")

    z_svd = (X @ Vt.T).astype(np.float32)
    z_q = encode_rows(z_svd, np.arange(n_rows, dtype=np.int64), enc)
    log(f"  encoded {z_q.shape} queries")
    del z_svd

    flat_idx_for_row = list(range(len(rows_data)))
    # Records with no rules filtered out — keep alignment with flat_records.
    keep = [i for i, rd in enumerate(rows_data)]
    flat_records_kept = [flat_records[i] for i in keep]
    z_q_kept = z_q[keep]
    flat_idx_for_row_kept = list(range(len(flat_records_kept)))
    asin_idx = np.asarray(
        [hash(asin) % (2**31) for asin, _, _ in flat_records_kept], dtype=np.int64
    )
    return z_q_kept, asin_idx, np.asarray([li for _, _, li in flat_records_kept], dtype=np.int64)


def fit_coral(z_r: np.ndarray, z_q: np.ndarray) -> dict:
    """Fit A = C_r^{1/2} C_q^{-1/2} with epsilon regularization."""
    mu_r = z_r.mean(axis=0)
    mu_q = z_q.mean(axis=0)
    Cr = np.cov(z_r, rowvar=False) + 0.0
    Cq = np.cov(z_q, rowvar=False) + 0.0
    eps = float(np.trace(Cq) / Cq.shape[0]) * EPSILON_REL
    Cr_reg = Cr + eps * np.eye(Cr.shape[0], dtype=np.float64)
    Cq_reg = Cq + eps * np.eye(Cq.shape[0], dtype=np.float64)
    # Symmetric eigendecomposition → matrix square root via eigvals.
    w_r, V_r = np.linalg.eigh((Cr_reg + Cr_reg.T) / 2)
    w_q, V_q = np.linalg.eigh((Cq_reg + Cq_reg.T) / 2)
    # Clip negative eigenvalues to 0 (PSD).
    w_r = np.clip(w_r, 0.0, None)
    w_q = np.clip(w_q, 0.0, None)
    Sr = V_r @ np.diag(np.sqrt(w_r)) @ V_r.T       # Cr^{1/2}
    Sq_inv = V_q @ np.diag(1.0 / np.sqrt(np.clip(w_q, 1e-12, None))) @ V_q.T
    A = Sr @ Sq_inv
    return {
        "mu_review": mu_r.astype(np.float32),
        "mu_query": mu_q.astype(np.float32),
        "A": A.astype(np.float32),
        "epsilon": np.float32(eps),
        "dim": K_DIM,
    }


def main() -> None:
    t0 = time.time()
    log("=== CORAL alignment cohort3mlp32 (profile review ↔ pool query) ===")

    enc = load_encoder()
    with np.load(SVD_COMPONENTS) as npz_svd:
        Vt = np.asarray(npz_svd["Vt"], dtype=np.float32)
    log(f"  Vt: {Vt.shape}")
    X = load_sparse_sent_vectors()
    log(f"  X: {X.shape}")
    z_all = (X @ Vt.T).astype(np.float32)
    del X

    # Load cohort3-trained uid whitelist (1,579 unique uids).
    with open(COHORT3_TRAINED_UIDS) as f:
        trained_uid_set = set(json.load(f))
    npz = np.load(STRICT3_NPZ, allow_pickle=False)
    uid_list = np.asarray(npz["uid_list"])
    trained_uid_idx_set = {i for i, u in enumerate(uid_list) if u in trained_uid_set}
    log(f"  cohort3-trained users: {len(trained_uid_idx_set)}")

    # Profile review embeddings
    log("== profile review 32d ==")
    z_profile_review, kept_uids = encode_profile_review(z_all, enc, trained_uid_idx_set)
    np.savez(OUT_Z_PROFILE, z=z_profile_review.astype(np.float32), uid_idx=kept_uids)
    log(f"  wrote → {OUT_Z_PROFILE}")

    # Pool query embeddings
    log("== pool query 32d ==")
    z_query, asin_idx, local_idx = encode_pool_queries(z_all, enc)
    np.savez(OUT_Z_QUERY, z=z_query.astype(np.float32),
             asin_idx=asin_idx, local_idx=local_idx)
    log(f"  wrote → {OUT_Z_QUERY}")

    # CORAL fit
    log("== CORAL fit ==")
    coral = fit_coral(z_profile_review.astype(np.float64),
                      z_query.astype(np.float64))
    np.savez(OUT_CORAL, **coral)
    log(f"  wrote → {OUT_CORAL}")
    log(f"  ε = {coral['epsilon']:.6f} (relative={EPSILON_REL})")
    log(f"  ||A - I||_F = {np.linalg.norm(coral['A'] - np.eye(K_DIM, dtype=np.float32)):.3f}")

    # Quick sanity: pre/post mean distance to review manifold
    sample = z_query[:5].astype(np.float64)
    A = coral["A"].astype(np.float64)
    mu_q = coral["mu_query"].astype(np.float64)
    mu_r = coral["mu_review"].astype(np.float64)
    aligned = mu_r[None, :] + (sample - mu_q[None, :]) @ A.T     # (5, 32)
    # "Distance to review manifold" = L2 to nearest review mean (rough proxy)
    review_mean = z_profile_review[:50].astype(np.float64).mean(axis=0)
    d2_pre = ((sample - review_mean) ** 2).sum(axis=1)
    d2_post = ((aligned - review_mean) ** 2).sum(axis=1)
    log(f"  sample mean-distance pre:  {d2_pre.tolist()}")
    log(f"  sample mean-distance post: {d2_post.tolist()}")

    # Full distribution: per-query distance to nearest review mean (cohort3-trained only)
    log("== full distribution: nearest-review-mean distance, pre vs post CORAL ==")
    review_z = z_profile_review.astype(np.float64)
    # chunk over z_query to avoid N×M blow-up in memory; report quantiles
    B = 4096
    d_pre = np.empty(len(z_query), dtype=np.float64)
    d_post = np.empty(len(z_query), dtype=np.float64)
    aligned_all = mu_r[None, :] + (z_query.astype(np.float64) - mu_q[None, :]) @ A.T
    for s in range(0, len(z_query), B):
        e = min(s + B, len(z_query))
        zq = z_query[s:e].astype(np.float64)
        za = aligned_all[s:e]
        # nearest mean (use squared L2)
        # for speed, compute against a coarse sub-sample of 200 reviews
        sub = review_z[::max(1, len(review_z) // 200)]
        d2_pre_mat = ((zq[:, None, :] - sub[None, :, :]) ** 2).sum(axis=-1)
        d2_post_mat = ((za[:, None, :] - sub[None, :, :]) ** 2).sum(axis=-1)
        d_pre[s:e] = d2_pre_mat.min(axis=1)
        d_post[s:e] = d2_post_mat.min(axis=1)
    for q in (50, 90, 95, 99):
        log(f"  pre  p{q}: {np.percentile(d_pre, q):.3f}  "
            f"post p{q}: {np.percentile(d_post, q):.3f}  "
            f"Δ mean: {float(np.mean(d_pre - d_post)):.3f}")

    log(f"=== CORAL fit DONE in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()