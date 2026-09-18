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

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import spacy
import torch
import torch.nn as nn

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "03_spacy_encode"))
from train_real_asin_cohort3 import StyleMLP  # cohort3 MLP (F.normalize + ReLU)
_cohort3_StyleMLP = StyleMLP
StyleMLP = _cohort3_StyleMLP  # override alias so downstream 'StyleMLP(...)' uses cohort3 MLP

POOL_PATH = REPO_ROOT / "result/07_gen_query/pool_queries.json"
ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract/asin_to_users.json"

# --- svd_mlp canonical config (Rule 3: hardcoded) ---
DEVICE = "cuda:0"
ROW_NORMALIZE = False
LATENT_DIM = 32                   # cohort3 MLP output dim (this run)
GATE_MODE = "per_user_top1"     # per-user Top-1 with chi²_{D,0.95} fit + user-vs-competitor margin
LOGP_DELTA = 2.0                  # unused under per_user_top1 (kept for legacy reference)
GATE_Q = 0.05                     # unused under per_user_top1
D2_CHI2_Q95 = 46.1943             # chi²_{32, 0.95}: 95% Gaussian fit gate (theoretical) — CORAL-aligned
                                  # pool queries now sit on the review manifold so the theoretical
                                  # threshold becomes meaningful again.
MARGIN_MIN = 0.0                  # min user-vs-competitor margin (log_p[u*] - max_{v≠u*} log_p[v])
SEED_SVDMLP = 42
HARD_MIN_QUERIES_PER_ASIN = 1     # logp_delta mode 下保留至少 1 query/ASIN
BATCH_SIZE = 256                  # spaCy pipe batch size
SVD_COMPONENTS = "/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/svd_components.npz"
SVD_NPZ = SVD_COMPONENTS
SVD_DIM = 256
MLP_ENCODER = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort3_mlp32_30_30ep.pt"
MLP_PT = MLP_ENCODER
CORAL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/coral_cohort3mlp32/coral_cohort3mlp32.npz"
APPLY_CORAL = True
VOCAB_PATH = "/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/vocab.json"
GAUSSIAN_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_cohort3mlp32_lowrankdiag_rank1.json"
OUT_PATH = REPO_ROOT / "result/08_select_query/selected_queries_cohort3mlp32_lowrankdiag.json"
OUT_STATS_PATH = REPO_ROOT / "result/08_select_query/selection_stats_cohort3mlp32_lowrankdiag.json"
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


def _svdmlp_encode_queries(
    queries: list[str],
    vocab: list[str],
    Vt: np.ndarray,
    mlp: StyleMLP,
    nlp,
) -> tuple[np.ndarray, list[tuple[int, dict[int, float]]]]:
    """queries -> 64d z (svd_mlp encoder 完整流程).

    流程:
      1) spaCy.pipe (batch=BATCH_SIZE) -> docs
      2) extract_struct_rules -> 每句 set of rule_strs
      3) rule_str -> vocab index -> sparse row (1, V)
      4) row-normalize -> SVD 投影 (256d) -> MLP (64d)
    """
    V = len(vocab)
    rule2idx = {r: i for i, r in enumerate(vocab)}
    N = len(queries)
    _svdmlp_log(f"  encoding {N} queries with spaCy batch={BATCH_SIZE}")
    t0 = time.time()
    rows_data: list[tuple[int, dict[int, float]]] = []
    for batch_start in range(0, N, BATCH_SIZE):
        batch_texts = queries[batch_start:batch_start + BATCH_SIZE]
        for di, doc in enumerate(nlp.pipe(batch_texts, batch_size=BATCH_SIZE)):
            rules = extract_struct_rules(doc)
            counts: dict[int, float] = {}
            for r in rules:
                idx = rule2idx.get(r)
                if idx is None:
                    continue
                counts[idx] = counts.get(idx, 0) + 1.0
            if counts:
                rows_data.append((batch_start + di, counts))
        if (batch_start // BATCH_SIZE) % 20 == 0:
            _svdmlp_log(f"    spacy {batch_start + len(batch_texts)}/{N}  "
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

    # MLP encode (batch)
    z_all = np.zeros((z_svd.shape[0], LATENT_DIM), dtype=np.float32)
    mlp.eval()
    with torch.no_grad():
        for s in range(0, z_svd.shape[0], 1024):
            e = min(s + 1024, z_svd.shape[0])
            xb = torch.from_numpy(z_svd[s:e]).to(DEVICE)
            z_all[s:e] = mlp(xb).cpu().numpy()
    _svdmlp_log(f"  MLP encode done, shape {z_all.shape}, elapsed={time.time() - t0:.1f}s")
    if APPLY_CORAL:
        if not Path(CORAL_PATH).exists():
            raise FileNotFoundError(
                f"APPLY_CORAL=True but {CORAL_PATH} missing; "
                "run 07_gen_query/fit_coral_cohort3mlp32.py first")
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
    if cfg.get("encoder_source") != "cohort3_mlp32_30ep":
        raise ValueError(
            f"gaussian cfg encoder_source={cfg.get('encoder_source')}, "
            "expected cohort3_mlp32_30ep")
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

    # 把所有 query 收集起来一起编码
    _svdmlp_log("flattening queries")
    flat_records: list[tuple[str, str, int, str]] = []  # (asin, query, local_idx, qstr)
    for asin, queries in pool.items():
        for li, q in enumerate(queries):
            flat_records.append((asin, q, li, q))
    _svdmlp_log(f"  total queries: {len(flat_records)}")

    nlp = spacy.load(SPACY_MODEL, disable=["ner", "textcat", "lemmatizer"])
    queries_text = [r[1] for r in flat_records]
    z_all, rows_data = _svdmlp_encode_queries(queries_text, vocab, Vt, mlp, nlp)

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

    _svdmlp_log(f"per-query best-fit user + per-user margin Top-1 (D² ≤ chi²_{LATENT_DIM},0.95={D2_CHI2_Q95})")
    # per_asin → query × (u*, logp_u*, d2_u*, all_users_logp list, query_text)
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
        per_asin_records.setdefault(asin, []).append({
            "logp": float(log_p[best_k]),
            "d2": float(d2[best_k]),
            "user_idx": int(cand_idx_arr[best_k]),
            "margin": margin,
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

    # Per-(asin, user) Top-1 selection with 95% Gaussian fit + margin gate
    # (a) filter queries with d²(u*) ≤ chi²_{D,0.95} AND margin ≥ MARGIN_MIN
    # (b) bucket by (asin, user_idx)
    # (c) per bucket pick the query with the largest log_p
    kept: dict[str, list[str]] = {}
    selections_block: list = []
    drops = {"above_d2": 0, "below_margin": 0}
    for asin, cands in per_asin_records.items():
        # Group by (user_idx, query) and keep best d2 per group (one user may be best
        # for many queries).
        bucket: dict[int, list[dict]] = {}
        for c in cands:
            if c["d2"] > D2_CHI2_Q95:
                drops["above_d2"] += 1
                continue
            if c["margin"] < MARGIN_MIN:
                drops["below_margin"] += 1
                continue
            bucket.setdefault(c["user_idx"], []).append(c)
        # Per bucket: pick the single query with the largest logp (= best fit on
        # the per-user Gaussian); keep ties by max margin.
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

    _svdmlp_log(f"  selected: {sum(len(v) for v in kept.values())}/{len(flat_records)} queries")

    _svdmlp_log(f"  selected: {sum(len(v) for v in kept.values())}/{len(flat_records)} queries")
    _svdmlp_log(f"  ASIN with >=1 kept: {len(kept)}/{len(pool)}")
    _svdmlp_log(f"  drops: {drops}, n_no_user={n_no_user}")

    # ASIN>=K 分布
    counts = [len(v) for v in kept.values()]
    n_ge1 = sum(1 for c in counts if c >= 1)
    n_ge2 = sum(1 for c in counts if c >= 2)
    n_ge3 = sum(1 for c in counts if c >= 3)
    n_ge5 = sum(1 for c in counts if c >= 5)

    out = {
        "config": {
            "source": "cohort3mlp32_lowrankdiag",
            "latent_dim": LATENT_DIM,
            "gate_mode": GATE_MODE,
            "d2_chi2_q95": D2_CHI2_Q95,
            "margin_min": MARGIN_MIN,
            "gaussian_path": str(GAUSSIAN_PATH),
            "svd_components": str(SVD_COMPONENTS),
            "mlp_encoder": str(MLP_ENCODER),
            "n_pool_asin": len(pool),
            "n_total_queries": len(flat_records),
            "n_fitted_users": len(uid_idx),
            "device": DEVICE,
            "selection_rule": (
                "per-(asin,user) Top-1: best_fit=user requires d²(q,u*) ≤ chi²_{D,0.95}; "
                "margin = logp(q|u*) - max_{v≠u*} logp(q|v) ≥ MARGIN_MIN; "
                "tie-break: max logp then max margin."
            ),
            "d2_formula": "lowrank+diag: d2 = Σ_i (v_i·diff)²/λ_i + r²/σ_d², r = diff - (diff·V)V^T",
            "log_norm_formula": "0.5 * (log|diag(σ_d²)| + slogdet(I + V^T diag(1/σ_d²) V diag(λ)) + D log 2π)",
        },
        "kept": kept,
        "selections": selections_block,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    _svdmlp_log(f"DONE wrote {OUT_PATH}")

    stats = {
        "gate_mode": GATE_MODE,
        "d2_chi2_q95": D2_CHI2_Q95,
        "margin_min": MARGIN_MIN,
        "n_kept_queries": sum(len(v) for v in kept.values()),
        "n_total_queries": len(flat_records),
        "n_total_asin": len(pool),
        "n_asin_ge1": n_ge1,
        "n_asin_ge2": n_ge2,
        "n_asin_ge3": n_ge3,
        "n_asin_ge5": n_ge5,
        "drops": drops,
    }
    with open(OUT_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    _svdmlp_log(f"DONE wrote {OUT_STATS_PATH}")


if __name__ == "__main__":
    _svdmlp_main()
