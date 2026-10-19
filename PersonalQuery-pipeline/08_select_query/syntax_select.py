#!/usr/bin/env python3
"""Query selection: per-user svd_mlp Gaussian (Stage 04 cohort3mlp16_30 lowrank+diag) + logp_delta=2.0 gate.

Pipeline (canonical, 2026-09-17):
  1. Load Stage 04 cohort3mlp16_30_lowrankdiag_rank1 Gaussian (mu, lambdas, V, sigma_diag_sq)
  2. Load SVD components + StyleMLP encoder (Stage 03b)
  3. spaCy.pipe encode pool queries -> SVD 256d -> MLP 64d
  4. For each query: compute log p(z | N(mu_u, diag(sigma_u^2))) against every cohort user
  5. best-fit user = argmax logp; keep queries with logp >= max_logp - LOGP_DELTA
  6. Output per-ASIN kept queries

Reused assets:
  - SVD components: pcfg_cache/svd_components.npz (256d)
  - MLP encoder:    pcfg_cache/svd_mlp_encoder.pt (256->128->64)
  - Vocab:          pcfg_cache/vocab.json
  - Stage 04b:      n/a (Stage 04b trainable Gaussian artifact deprecated; only cohort3mlp16_30_lowrankdiag is consumed)
  - Pool queries:   result/07_gen_query/pool_queries.json
  - ASIN→users:     result/02_user_review_sentence_extract/asin_to_users.json

Usage (Rule 3: no args):
  cd $REPO_ROOT
  $PY 08_select_query/syntax_select_mahalanobis_gate.py

Outputs:
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

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("PQ_SCRATCH", str(REPO_ROOT / "scratch")))
DATA_DIR = Path(os.environ.get("PQ_DATA_DIR", str(REPO_ROOT / "data")))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "03_spacy_encode"))
from syntax_encoder import StyleMLP  # cohort3 MLP (F.normalize + ReLU)
_cohort3_StyleMLP = StyleMLP
StyleMLP = _cohort3_StyleMLP  # override alias so downstream 'StyleMLP(...)' uses cohort3 MLP

POOL_PATH = REPO_ROOT / "result/07_gen_query/pool_queries.json"
ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract/asin_to_users_baby.pkl"
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
MAX_USERS_PER_ASIN = 2
UNIQUENESS_MAX_USER_OVERLAP = 2
# Chi² percentile gate (controls uniqueness radius): sweep over [0.75, 0.55,
# 0.35, 0.15, 0.005, 0.95]. The hard uniqueness rule D²(u*)≤chi² AND
# ∀v≠u*, D²(v)>chi² becomes stricter as q shrinks (smaller sphere).
D2_CHI2_Q95 = float(_chi2_dist.ppf(0.95, LATENT_DIM))   # chi²_{16, 0.95} theoretical
                                  # pool queries now sit on the review manifold so the theoretical
                                  # threshold becomes meaningful again.
# Sweep grid: percentile q of chi²_{D=32}; corresponding D² thresholds.
CHI2_SWEEP_Q = [0.95, 0.75, 0.55, 0.35, 0.15, 0.05, 0.005]
CHI2_SWEEP_THRESHOLDS = {q: float(_chi2_dist.ppf(q, LATENT_DIM))
                         for q in CHI2_SWEEP_Q}
D2_OBSERVED_SWEEP = [0.50, 0.75, 0.90, 0.95, 0.99]   # observed d^2 percentiles
                                  # pool queries now sit on the review manifold so the theoretical
                                  # threshold becomes meaningful again.
MARGIN_MIN = 0.0                  # min user-vs-competitor margin (log_p[u*] - max_{v≠u*} log_p[v])
SEED_SVDMLP = 42
HARD_MIN_QUERIES_PER_ASIN = 1     # keep at least 1 query/ASIN in logp_delta mode
BATCH_SIZE = 256                  # spaCy pipe batch size
SVD_COMPONENTS = str(SCRATCH / "pcfg_cache" / "svd_components.npz")
SVD_NPZ = SVD_COMPONENTS
SVD_DIM = 256
MLP_ENCODER = str(REPO_ROOT / "result/03_spacy_encode/cohort2_mlp16_30_30ep.pt")
MLP_PT = MLP_ENCODER
TRAINED_UIDS_PATH = REPO_ROOT / "result/03_spacy_encode/cohort3_trained_uids.json"
STRICT3_NPZ = SCRATCH / "pcfg_cache/strict3_embeddings.npz"
ASIN_TO_USERS = REPO_ROOT / "result/02_user_review_sentence_extract/asin_to_users_baby.pkl"
SMOKE = os.environ.get("STAGE08_ALIGNMENT_SMOKE") == "1"
SMOKE_ASIN_LIMIT = 5
# Stage 28 sets this to dump the frozen per-candidate scoring table.
CANDIDATE_DUMP: Path | None = None
VOCAB_PATH = str(SCRATCH / "pcfg_cache/vocab.json")
GAUSSIAN_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_cohort3mlp16_30_lowrankdiag_rank1.json"
OUT_PATH = REPO_ROOT / "result/08_select_query/selected_queries.json"
OUT_STATS_PATH = REPO_ROOT / "result/08_select_query/selection_stats.json"
SPACY_MODEL = "en_core_web_sm"


def _svdmlp_log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _svdmlp_extract_struct_rules(doc):
    """Extract dependency + POS rules (D4/D3/P3) from a spaCy Doc; same source as syntax_pcfg_pipeline.py."""
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
    return SCRATCH / "pool_query_cache" / (
        f"qcache_{cache_key[:16]}.npz")


def _svdmlp_encode_queries(
    queries: list[str],
    vocab: list[str],
    Vt: np.ndarray,
    mlp: StyleMLP,
    nlp,
    cohort_filter_signature: str | None = None,
    flat_records: list | None = None,
) -> tuple[np.ndarray, list[tuple[int, dict[int, float]]]]:
    """queries -> 64d z (full svd_mlp encoder pipeline).

    Steps:
      1) spaCy.pipe (batch=BATCH_SIZE) -> docs
      2) extract_struct_rules -> per-sentence set of rule_strs
      3) rule_str -> vocab index -> sparse row (1, V)
      4) row-normalize -> SVD projection (256d) -> MLP (64d)

    Cache: spaCy parse + sparse build + SVD-z projection are memoized to
    $SCRATCH/pool_query_cache/. The MLP forward always runs fresh because it
    depends on the encoder weights (cheaper to recompute than to cache).
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

    return z_all, rows_data


def _svdmlp_load_gaussian() -> tuple[dict, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cohort3mlp 32d lowrank+diag Gaussian; precompute mu(N,32), lambdas(N,K_RANK),
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


def main_task_body() -> dict:
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

    uid_to_sents_path = globals().get("UID_TO_SENTS")
    if uid_to_sents_path is None or not Path(uid_to_sents_path).exists():
        # Fallback: derive from category subdir by inspecting OUT_PATH
        subdir = OUT_PATH.parent.name
        uid_to_sents_path = (REPO_ROOT / "result/02_user_review_sentence_extract"
                             / f"uid_to_sentences_{subdir}.pkl")
    SENT_CACHE_LOADED = pickle.load(open(uid_to_sents_path, "rb"))

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
        if len(cohort_users) > MAX_USERS_PER_ASIN:
            sent_cache = SENT_CACHE_LOADED
            cohort_users = sorted(
                cohort_users,
                key=lambda u: (-len(sent_cache.get(u, [])), u)
            )[:MAX_USERS_PER_ASIN]
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
    z_all, rows_data = _svdmlp_encode_queries(
        queries_text, vocab, Vt, mlp, nlp,
        cohort_filter_signature=cohort_sig,
        flat_records=flat_records,
    )

    flat_idx_for_row = [rec_idx for rec_idx, _ in rows_data]

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
        n_inside_per_q.update({
            f"obs:{int(q*100)}": 0
            for q in D2_OBSERVED_SWEEP
        })
        per_asin_records.setdefault(asin, []).append({
            "logp": float(log_p[best_k]),
            "d2": float(d2[best_k]),
            "user_idx": int(cand_idx_arr[best_k]),
            "margin": margin,
            "n_inside_per_q": n_inside_per_q,
            "query": q_text,
            "local_idx": int(local_idx),
            "z_idx": int(ri),
        })
        if (ri + 1) % selection_progress_every == 0 or ri + 1 == selection_total:
            _svdmlp_log(f"    query scoring {ri+1}/{selection_total} "
                        f"({100 * (ri+1) / selection_total:.1f}%), "
                        f"no_user={n_no_user}, "
                        f"elapsed={time.time() - selection_started:.1f}s")

    _svdmlp_log(f"  queries with candidate: {sum(len(v) for v in per_asin_records.values())}/{len(flat_records)}")

    # Optional per-candidate dump for Stage 28: the exact frozen per-query
    # scores (best-fit user, D2, positive margin and region-overlap counts) that
    # every selector arm must share. Nothing here changes the scoring itself.
    if CANDIDATE_DUMP is not None:
        CANDIDATE_DUMP.parent.mkdir(parents=True, exist_ok=True)
        n_dumped = 0
        z_rows = []
        with open(CANDIDATE_DUMP, "w") as fh:
            for _asin, _cands in per_asin_records.items():
                for _c in _cands:
                    _nin = _c["n_inside_per_q"]
                    fh.write(json.dumps({
                        "asin": _asin,
                        "query": _c["query"],
                        "local_idx": _c["local_idx"],
                        "user_idx": _c["user_idx"],
                        "logp": _c["logp"],
                        "d2": _c["d2"],
                        "margin": _c["margin"],
                        "n_inside_95": _nin.get(0.95),
                        "n_inside_75": _nin.get(0.75),
                        "z_idx": _c["z_idx"],
                    }) + "\n")
                    z_rows.append(z_all[_c["z_idx"]])
                    n_dumped += 1
        # Row-aligned 16-d embeddings so downstream stages can re-score the very
        # same candidates under an exact Gaussian without re-encoding.
        np.save(str(CANDIDATE_DUMP.with_suffix(".z.npy")),
                np.asarray(z_rows, dtype=np.float32))
        _svdmlp_log(f"  candidate dump -> {CANDIDATE_DUMP} ({n_dumped} rows) "
                    f"+ {CANDIDATE_DUMP.with_suffix('.z.npy').name}")

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

    if len(all_d2) > 0:
        D2_OBSERVED_THRESHOLDS = {
            q: float(np.percentile(all_d2, q * 100))
            for q in D2_OBSERVED_SWEEP
        }
        _svdmlp_log(
            f"  observed d² percentile thresholds: " +
            ", ".join(f"p{int(q*100)}={D2_OBSERVED_THRESHOLDS[q]:.3f}"
                       for q in D2_OBSERVED_SWEEP)
        )
    else:
        D2_OBSERVED_THRESHOLDS = {}

    # Sweep chi² quantile q. For each q we run the gate:
    #   - D²(q, u*) ≤ chi²_{D, q} AND ∀ v≠u*, D²(q, v) > chi²_{D, q} (uniqueness)
    #   - margin ≥ MARGIN_MIN
    #   - bucket by (asin, user_idx), pick max logp per bucket
    sweep_results: dict[str, dict] = {}

    def _run_gate(thr_key, thr_value, label):
        kept: dict[str, list[str]] = {}
        selections_block: list = []
        supports: dict[str, dict[int, list[dict]]] = {}
        drops = {"above_d2": 0, "below_margin": 0, "multi_user_inside": 0}
        is_obs_key = isinstance(thr_key, str) and thr_key.startswith("obs:")
        for asin, cands in per_asin_records.items():
            bucket: dict[int, list[dict]] = {}
            for c in cands:
                if c["d2"] > thr_value:
                    drops["above_d2"] += 1
                    continue
                if is_obs_key:
                    pass
                else:
                    n_in = c["n_inside_per_q"][thr_key]
                    if n_in > UNIQUENESS_MAX_USER_OVERLAP:
                        drops["multi_user_inside"] += 1
                        continue
                if c["margin"] < MARGIN_MIN:
                    drops["below_margin"] += 1
                    continue
                bucket.setdefault(c["user_idx"], []).append(c)
            # Pre-Top-1 eligible support: every candidate that cleared all three
            # gates, grouped by best-fit user. Stage 27 resamples from this
            # instead of the single retained Top-1 query.
            if bucket:
                supports[asin] = bucket
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
            f"  {label} (D²≤{thr_value:.3f}): "
            f"kept={sum(len(v) for v in kept.values())} "
            f"ASINs={len(kept)} "
            f"(≥1u={n_ge1}, ≥2u={n_ge2}, ≥3u={n_ge3}, ≥5u={n_ge5}) "
            f"drops={drops}")
        return {
            "threshold": thr_value,
            "kept": kept,
            "selections": selections_block,
            "supports": supports,
            "stats": {
                "label": label,
                "d2_threshold": thr_value,
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

    # Theoretical chi² sweep
    for q in CHI2_SWEEP_Q:
        thr = CHI2_SWEEP_THRESHOLDS[q]
        sweep_results[str(q)] = _run_gate(q, thr, f"chi2q={q:.3f}")
    # 2026-09-25: observed-d² percentile sweep (empirical calibration)
    for q in D2_OBSERVED_SWEEP:
        thr = D2_OBSERVED_THRESHOLDS[q]
        key = f"obs:{int(q*100)}"
        sweep_results[key] = _run_gate(key, thr, f"obs_p{int(q*100)}")

    canonical_key = "0.95"
    canonical = sweep_results[canonical_key]
    canonical_q = canonical_key

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
            k: {
                "threshold": v["threshold"],
                "stats": v["stats"],
            }
            for k, v in sweep_results.items()
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
        "d2_observed_sweep_q": D2_OBSERVED_SWEEP,
        "d2_observed_thresholds": D2_OBSERVED_THRESHOLDS,
        "margin_min": MARGIN_MIN,
        "n_total_queries": len(flat_records),
        "n_total_asin": len(pool),
        "n_no_user": n_no_user,
        "per_q": [v["stats"] for v in sweep_results.values()],
    }
    with open(OUT_STATS_PATH, "w") as f:
        json.dump(summary_stats, f, indent=2)
    _svdmlp_log(f"DONE wrote {OUT_STATS_PATH}")
    # Exposed for Stage 27: the canonical gate's pre-Top-1 eligible supports.
    # The canonical artifacts written above are untouched.
    return {
        "subdir": OUT_PATH.parent.name,
        "threshold": canonical["threshold"],
        "supports": canonical["supports"],
        "kept": canonical["kept"],
        "uid_order": uid_order,
        "n_total_queries": len(flat_records),
        "n_total_asin": len(pool),
        "n_no_user": n_no_user,
        "stats": summary_stats,
    }



# ============================================================================
# Canonical Stage 08 (Stage 32 P6): query-feasibility-aware strict USER-PAIR C3
# ============================================================================
# Historical main_task_body and its approximate scorer are retained ONLY for
# experimental replay through explicit main(out_root=...); main() is P6.
# This delegates the actual C3 predicate and Gaussian calculation to the
# already-tested Stage 31/32 primitives.  No hand-copied Gaussian scorer.
P6_EXPECTED_ASINS = {"baby": 199, "musical": 371, "video_games": 1257}


def _main_p6_canonical() -> None:
    """Recompute P6 in canonical Stage 08 paths from the frozen candidate dump.

    The dump contains Stage 08's encoded Stage 07 queries, with source indices;
    Stage 32 used this exact dump. We verify every source query against the
    *current* Stage 07 pool before replacing downstream-consumed outputs, so a
    modified pool cannot silently reuse outdated embeddings. The original
    source encoder and fitted Gaussians are NOT retrained.
    """
    import importlib

    sys.path.insert(0, str(REPO_ROOT / "32_query_feasibility_pair_selection"))
    s32 = importlib.import_module("feasibility_pair_selection")
    s31 = s32.s31
    ExactGaussian = s32.ExactGaussian
    load_asin_users = s32.load_asin_users

    out_root = (SCRATCH / "stage08_p6_smoke" if SMOKE
                else REPO_ROOT / "result/08_select_query")
    _svdmlp_log("Canonical Stage 08 = Stage 32 P6 feasibility-aware pair C3")
    _svdmlp_log("Pair-conditional C3, NOT global all-user strict exclusion")

    # Prepare and check all three domains BEFORE writing any canonical output.
    pending = []
    for _category, sub in CATEGORY_INPUTS:
        dumped = s31.load_dump(sub)
        if not dumped:
            raise RuntimeError(f"{sub}: Stage 30 encoded candidate dump is empty")
        pool_path = REPO_ROOT / "result/07_gen_query" / sub / "pool_queries.json"
        with open(pool_path, encoding="utf-8") as f:
            pool_doc = json.load(f)
        raw = pool_doc.get("pool", pool_doc)
        if not isinstance(raw, dict):
            raise ValueError(f"{sub}: Stage 07 pool must be a dict")
        pool = {}
        for asin, items in raw.items():
            texts = []
            for item in items:
                q = item.get("text") if isinstance(item, dict) else item
                if isinstance(q, str) and q.strip():
                    texts.append(q.strip())
            if texts:
                pool[asin] = texts
        for asin, candidates in dumped.items():
            if asin not in pool:
                raise RuntimeError(f"{sub}/{asin}: candidate ASIN absent from Stage 07 pool")
            seen_idx = set()
            for c in candidates:
                i = int(c["local_idx"])
                if i in seen_idx or not (0 <= i < len(pool[asin])):
                    raise RuntimeError(f"{sub}/{asin}: invalid or duplicate local_idx={i}")
                seen_idx.add(i)
                if c["query"].strip() != pool[asin][i]:
                    raise RuntimeError(
                        f"{sub}/{asin}/{i}: stale Stage 30 embedding dump; "
                        "regenerate candidate dump from the current Stage 07 pool")
        g = ExactGaussian(sub)
        a2u = load_asin_users(sub)
        picked = []
        n_pairs_examined = 0
        n_zero_feasible = 0
        for asin, cands in dumped.items():
            ctx = s31.build_ctx(g, asin, cands, a2u)
            if ctx is None or len(ctx["uids"]) < 2:
                continue
            d_b, bad = s31.pair_bd_matrix(g, ctx["users"])
            if bad:
                raise RuntimeError(f"{sub}/{asin}: nonfinite Gaussian pair distances")
            pair_records, feasible = s32.enumerate_pairs(ctx, d_b)
            n_pairs_examined += len(pair_records)
            if not feasible:
                n_zero_feasible += 1
                continue
            i, j = s32.pick_p6_stable(d_b, feasible, ctx["uids"])
            selected = s31.conditional_select(ctx, [i, j])
            if not s31.outcome_of(selected, ctx)["ge2_distinct_text"]:
                raise AssertionError(f"{sub}/{asin}: P6 selected an infeasible pair")
            users = []
            for col in (i, j):
                row = selected[col]
                other = j if col == i else i
                cand = ctx["cands"][row]
                users.append({
                    "uid": ctx["uids"][col],
                    "query": cand["query"],
                    "logp": float(ctx["LOGP"][row, col]),
                    "d2": float(ctx["D2"][row, col]),
                    "margin": float(ctx["LOGP"][row, col] - ctx["LOGP"][row, other]),
                    "source_local_idx": int(cand["local_idx"]),
                })
            if users[0]["uid"] == users[1]["uid"] or users[0]["query"] == users[1]["query"]:
                raise AssertionError(f"{sub}/{asin}: non-distinct P6 user or query")
            picked.append({"asin": asin, "users": users, "max_pair_cos": None})
            if SMOKE and len(picked) >= SMOKE_ASIN_LIMIT:
                break

        expected = P6_EXPECTED_ASINS[sub]
        if not SMOKE and len(picked) != expected:
            raise RuntimeError(
                f"{sub}: expected {expected} Stage 32 P6 ASINs, got {len(picked)}; "
                "canonical outputs left untouched")
        if not picked:
            raise RuntimeError(f"{sub}: no P6 selections available")
        chosen = {block["asin"]: [u["query"] for u in block["users"]] for block in picked}
        doc = {
            "config": {
                "source": "stage32_p6_query_feasibility_aware",
                "selection_method": "P6_max_BD_feasible_pair",
                "cohort_semantics": "strict C3 within the selected user pair, not all original users",
                "gaussian_scoring": "exact normalized 16D multivariate Gaussian",
                "d2_gate": float(s31.TAU_95),
                "margin_min": float(s31.MARGIN_MIN),
                "n_inside_required_within_pair": 1,
                "candidate_source": "Stage 30 encoded dump checked against current Stage 07 query indices/text",
                "n_selected_asins": len(picked),
                "n_selected_user_query_cells": 2 * len(picked),
                "smoke": bool(SMOKE),
            },
            "kept": chosen,
            "selections": picked,
        }
        stats = {
            "selection_method": "P6_max_BD_feasible_pair",
            "n_asins": len(picked),
            "n_users_queries": 2 * len(picked),
            "n_pairs_examined": n_pairs_examined,
            "n_asins_zero_feasible": n_zero_feasible,
            "expected_n_asins_full": expected,
            "full_reproduction": not SMOKE,
        }
        pending.append((sub, doc, stats))

    # Archive + fingerprint the historical canonical outputs BEFORE replacing
    # them, so the pre-migration baseline stays directly recoverable and the
    # provenance of the replacement is auditable.
    import hashlib as _hashlib
    archive_root = SCRATCH / "stage08_canonical_archive"
    archived = {}
    for sub, _doc, _stats in pending:
        for name in ("selected_queries.json", "selection_stats.json"):
            src = REPO_ROOT / "result/08_select_query" / sub / name
            if not src.exists():
                continue
            digest = _hashlib.sha256(src.read_bytes()).hexdigest()
            dst_dir = archive_root / sub
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            archived[f"{sub}/{name}"] = {"sha256": digest, "bytes": src.stat().st_size,
                                         "archived_to": str(dst)}
    if archived:
        _svdmlp_log(f"archived {len(archived)} historical Stage 08 artifacts -> "
                    f"{archive_root}")

    # Only publish the new canonical Stage 08 artifacts after complete
    # schema/count/source validation of the entire frozen three-domain run.
    for sub, doc, stats in pending:
        doc["config"]["archived_previous"] = {
            k: v for k, v in archived.items() if k.startswith(f"{sub}/")}
        out_dir = out_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in (("selected_queries.json", doc),
                              ("selection_stats.json", stats)):
            target = out_dir / name
            temporary = target.with_suffix(target.suffix + ".writing")
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(temporary, target)
        _svdmlp_log(f"Stage 08 P6 [{sub}]: {stats['n_asins']} products / "
                    f"{stats['n_users_queries']} user Queries -> {out_dir}")


# ============================================================================
# Entry point
# ============================================================================

def main(out_root: Path | None = None, collect: dict | None = None,
         candidate_dump_root: Path | None = None) -> None:
    if out_root is None and collect is None and candidate_dump_root is None:
        _main_p6_canonical()
        return
    """Run the 3 categories (Baby / Musical / Video_Games) serially.

    For each category, rebind the script path constants to the
    category-specific paths, then call the original main_task_body()
    (logic unchanged). Outputs go to result/<stage>/<baby|musical|video_games>/.

    ``out_root`` redirects that output tree, so Stage 27 can replay the
    canonical scoring without touching the released artifacts. ``collect``
    receives ``{subdir: main_task_body() result}`` for the pre-Top-1 supports.
    """
    global SENT_CACHE, UID_TO_SENTS, ASIN_USERS_PATH, ATTRIBUTES_PATH, META_FILE, OUT_DIR, OUT_PATH, OUT_STATS_PATH, POOL_PATH, GAUSSIAN_PATH, ASIN_TO_USERS, SVD_COMPONENTS, SVD_NPZ, MLP_ENCODER, MLP_PT, VOCAB_PATH, TRAINED_UIDS_PATH, STRICT3_NPZ, CANDIDATE_DUMP  # noqa
    # backup current (Baby) defaults
    saved = {
        k: v for k, v in globals().items()
        if k in {"SENT_CACHE", "UID_TO_SENTS", "ASIN_USERS_PATH", "ATTRIBUTES_PATH",
                 "META_FILE", "OUT_DIR", "OUT_PATH", "OUT_STATS_PATH", "POOL_PATH",
                 "GAUSSIAN_PATH", "ASIN_TO_USERS"}
        and isinstance(v, Path)
    }
    base_out = (Path(out_root) if out_root is not None
                else REPO_ROOT / "result" / Path(__file__).parent.name)
    for category, subdir in CATEGORY_INPUTS:
        _svdmlp_log(f"\n========== [{category}] (subdir={subdir}) ==========")
        CANDIDATE_DUMP = (Path(candidate_dump_root) / subdir / "candidates.jsonl"
                          if candidate_dump_root is not None else None)
        # Reset all known category-dependent paths to point at the per-category subdir.
        cache_dir = SCRATCH / f"pcfg_cache_{subdir}"
        SVD_COMPONENTS = str(cache_dir / "svd_components.npz")
        SVD_NPZ = SVD_COMPONENTS
        VOCAB_PATH = str(cache_dir / "vocab.json")
        MLP_ENCODER = str(REPO_ROOT / "result/03_spacy_encode" / subdir /
                          "cohort2_mlp16_30_30ep.pt")
        MLP_PT = MLP_ENCODER
        TRAINED_UIDS_PATH = (REPO_ROOT / "result/03_spacy_encode" / subdir /
                             "cohort3_trained_uids.json")
        STRICT3_NPZ = cache_dir / "strict3_embeddings.npz"
        if "SENT_CACHE" in saved:
            SENT_CACHE = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "UID_TO_SENTS" in saved:
            UID_TO_SENTS = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "ASIN_USERS_PATH" in saved:
            ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract" / f"asin_to_users_{subdir}.pkl"
        if "ATTRIBUTES_PATH" in saved:
            ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction" / f"product_attributes_{subdir}.pkl"
        if "META_FILE" in saved:
            META_FILE = DATA_DIR / {
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
            smoke_dir = (SCRATCH /
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
            result = main_task_body()
        except Exception as e:
            _svdmlp_log(f"[{category}] FAILED: {e!r}")
            raise
        if collect is not None:
            collect[subdir] = result
    # Restore Baby defaults (for import compatibility with downstream).
    for k, v in saved.items():
        globals()[k] = v
    CANDIDATE_DUMP = None


if __name__ == "__main__":
    main()
