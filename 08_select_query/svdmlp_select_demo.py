#!/usr/bin/env python3
"""Stage 08 (svd_mlp) — 用 svd_mlp 64d trainable Gaussian 选 query。

Selection scheme (默认 LOGP_DELTA):
  完全使用拟合出的 Gaussian N(μ_u, σ_u²):
    log p(z|μ_u, σ_u²) = -0.5 * [Σ (z-μ)²/σ² + Σ log σ² + d log(2π)]
  - 对每 query, 在该 ASIN 用户 cohort 中找 logp 最大的 user (best-fit)
  - Per-ASIN 保留 logp ≥ max_logp - LOGP_DELTA 的所有 query (至少 1 条)
  - 不依赖经验 d2_q95 阈值, 真正"用上" σ (σ 通过 log_norm = Σ log σ² 显式惩罚)

Fallback (TG_GATE_MODE=d2):
  保留与早期版本的兼容性: 用经验 d2_q95 × GATE_Q 当阈值 (d² 直过滤)。
  通过环境变量切换:
    TG_GATE_MODE=logp_delta (default) 或 d2
    TG_LOGP_DELTA=2.0       (per-ASIN 相对 logp 阈值)
    TG_GATE_Q=0.95          (d2_q95 multiplier, 仅 d2 模式生效)

与 stage8 syntax_select_mahalanobis_gate 的差异:
  - Gaussian 来源: trainable svd_mlp (对角 σ, 64d) 而非 fit_per_user full Σ (16d)
  - 每个 query 编码: SVD 256d -> MLP 64d (svd_mlp_encoder) 而非 strict3 监督 encoder
  - 无 cohort gate (svd_mlp 没生成 cohort_gates), 用 logp 或 d2 在 owner ASIN 内过滤

流程:
  1. 加载 svd_mlp trainable Gaussian (user_gaussian_stats_trainable_svdmlp.json)
  2. 加载 svd_mlp encoder (svd_components.npz + svd_mlp_encoder.pt)
  3. 加载 pool_queries.json + asin_to_users.json
  4. spaCy.pipe 编码所有 pool queries -> 64d z
  5. 每 query -> ASIN 内 best user -> score = logp / d²
  6. 按 GATE_MODE 过滤; 输出 selected_queries_svdmlp.json + stats

硬编码配置 (Rule 3/4):
  - GATE_MODE = logp_delta (default)
  - LOGP_DELTA = 2.0 (per-ASIN logp 相对阈值)
  - GATE_Q = 0.95 (仅 d2 模式生效)
  - n_pool = 全部 11331 ASIN
  - DEVICE = cpu (环境无 cuda)

使用:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        08_select_query/svdmlp_select_demo.py \
        > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/trainable/svdmlp_select.log 2>&1 &
"""
from __future__ import annotations

import os

import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
SPACY_MODEL = "en_core_web_sm"

POOL_PATH = REPO_ROOT / "result/07_gen_query/pool_queries.json"
ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract/asin_to_users.json"
GAUSSIAN_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_trainable_svdmlp.json"
SVD_COMPONENTS = CACHE_DIR / "svd_components.npz"
MLP_ENCODER = CACHE_DIR / "svd_mlp_encoder.pt"
VOCAB_PATH = CACHE_DIR / "vocab.json"
OUT_PATH = REPO_ROOT / "result/08_select_query/selected_queries_svdmlp.json"
OUT_STATS_PATH = REPO_ROOT / "result/08_select_query/svdmlp_select_stats.json"

# 硬编码配置 (Rule 3)
GATE_MODE = os.environ.get("TG_GATE_MODE", "logp_delta")  # "logp_delta" | "d2"
LOGP_DELTA = float(os.environ.get("TG_LOGP_DELTA", "2.0"))  # per-ASIN logp 相对阈值
GATE_Q = float(os.environ.get("TG_GATE_Q", "0.95"))  # 仅 d2 模式: d2_q95 × GATE_Q 当阈值
LATENT_DIM = 64
SVD_DIM = 256
ROW_NORMALIZE = True
BATCH_SIZE = 256     # spaCy batch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Stage 03b 同款规则提取器 + StyleMLP
# ============================================================================

def extract_struct_rules(doc):
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


class StyleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


def encode_queries(
    queries: list[str],
    vocab: list[str],
    Vt: np.ndarray,
    mlp: StyleMLP,
    nlp,
) -> np.ndarray:
    """queries -> 64d z (svd_mlp encoder 完整流程).

    流程:
      1) spaCy.pipe (batch=BATCH_SIZE) -> docs
      2) extract_struct_rules -> 每句 set of rule_strs
      3) rule_str -> vocab index -> sparse row (1, V)
      4) row-normalize -> SVD 投影 (256d) -> MLP (64d) -> L2 norm
    """
    V = len(vocab)
    rule2idx = {r: i for i, r in enumerate(vocab)}
    N = len(queries)
    log(f"  encoding {N} queries with spaCy batch={BATCH_SIZE}")
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
            log(f"    spacy {batch_start + len(batch_texts)}/{N}  "
                f"elapsed={time.time() - t0:.1f}s")
    log(f"  spacy + rule extraction: {len(rows_data)}/{N} non-empty "
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
    log(f"  sparse X: {X.shape} nnz={X.nnz}")

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
    log(f"  SVD projection: {z_svd.shape}")

    # MLP encode (batch)
    z_all = np.zeros((z_svd.shape[0], LATENT_DIM), dtype=np.float32)
    mlp.eval()
    with torch.no_grad():
        for s in range(0, z_svd.shape[0], 1024):
            e = min(s + 1024, z_svd.shape[0])
            xb = torch.from_numpy(z_svd[s:e]).to(DEVICE)
            z_all[s:e] = mlp(xb).cpu().numpy()
    log(f"  MLP encode done, shape {z_all.shape}, elapsed={time.time() - t0:.1f}s")
    return z_all, rows_data


def load_gaussian() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """加载 svd_mlp trainable Gaussian; 预计算 mu(N,64), sigma2(N,64), uid_order(N,)."""
    log(f"loading {GAUSSIAN_PATH}")
    with open(GAUSSIAN_PATH) as f:
        d = json.load(f)
    cfg = d["config"]
    if cfg.get("source") != "svd_mlp":
        raise ValueError(
            f"gaussian cfg source={cfg.get('source')}, expected svd_mlp. "
            "rerun 04_gaussian/trainable_per_user_gaussian.py with TG_SOURCE=svd_mlp"
        )
    if cfg["latent_dim"] != LATENT_DIM:
        raise ValueError(
            f"gaussian latent_dim={cfg['latent_dim']}, script LATENT_DIM={LATENT_DIM}"
        )
    users = d["users"]
    uid_order = sorted(users.keys())
    N = len(uid_order)
    mu = np.zeros((N, LATENT_DIM), dtype=np.float32)
    sigma2 = np.zeros((N, LATENT_DIM), dtype=np.float32)
    d2_q = np.zeros(N, dtype=np.float32)
    for i, uid in enumerate(uid_order):
        u = users[uid]
        mu[i] = np.asarray(u["mu"], dtype=np.float32)
        sigma2[i] = np.asarray(u["sigma_diag"], dtype=np.float32)
        d2_q[i] = float(u["d2_q95"])  # 经验 val 95% 分位
    uid_idx = {uid: i for i, uid in enumerate(uid_order)}
    return uid_idx, uid_order, mu, sigma2, d2_q


def main() -> None:
    log(f"device={DEVICE}  GATE_MODE={GATE_MODE}  "
        f"LOGP_DELTA={LOGP_DELTA}  GATE_Q={GATE_Q}  LATENT_DIM={LATENT_DIM}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log("loading vocab, svd components, mlp encoder")
    vocab = json.load(open(VOCAB_PATH))
    log(f"  vocab: {len(vocab)} rules")
    with np.load(SVD_COMPONENTS) as npz:
        Vt = np.asarray(npz["Vt"], dtype=np.float32)
    log(f"  Vt: {Vt.shape}  row_normalize={ROW_NORMALIZE}")
    ckpt = torch.load(MLP_ENCODER, map_location=DEVICE, weights_only=False)
    mlp_cfg = ckpt.get("config", {})
    mlp = StyleMLP(
        in_dim=mlp_cfg.get("svd_dim") or mlp_cfg.get("in_dim") or SVD_DIM,
        hidden=mlp_cfg.get("hidden") or 128,
        out_dim=mlp_cfg.get("out_dim") or LATENT_DIM,
    ).to(DEVICE)
    mlp.load_state_dict(ckpt["state_dict"])
    log(f"  mlp: in={mlp_cfg.get('svd_dim')} hidden={mlp_cfg.get('hidden')} "
        f"out={mlp_cfg.get('out_dim')}")

    log("loading pool queries")
    pool = json.load(open(POOL_PATH))["pool"]
    asin_to_users = json.load(open(ASIN_USERS_PATH))
    log(f"  pool: {len(pool)} ASINs")

    log("loading trainable svd_mlp Gaussian")
    uid_idx, uid_order, mu, sigma2, d2_q95 = load_gaussian()
    log(f"  fitted users: {len(uid_idx)}")

    # 把所有 query 收集起来一起编码
    log("flattening queries")
    flat_records: list[tuple[str, str, int, str]] = []  # (asin, query, local_idx, qstr)
    for asin, queries in pool.items():
        for li, q in enumerate(queries):
            flat_records.append((asin, q, li, q))
    log(f"  total queries: {len(flat_records)}")

    nlp = spacy.load(SPACY_MODEL, disable=["ner", "textcat", "lemmatizer"])
    queries_text = [r[1] for r in flat_records]
    z_all, rows_data = encode_queries(queries_text, vocab, Vt, mlp, nlp)

    # 还原 row -> flat_records index
    flat_idx_for_row = [rec_idx for rec_idx, _ in rows_data]

    # 预计算每用户的 log-norm 常数: 0.5 * [Σ log σ² + d log 2π]
    log_norm_per_user = 0.5 * (
        np.log(sigma2).sum(axis=-1) + LATENT_DIM * np.log(2 * np.pi)
    )

    log(f"per-query best-fit user + selection (mode={GATE_MODE})")
    per_asin_records: dict[str, list[tuple[float, float, str]]] = {}
    # 每个 query -> (asin, logp, d2, query_text)
    n_no_user = 0
    for ri, (z_row, flat_idx) in enumerate(zip(z_all, flat_idx_for_row)):
        asin, q_text, local_idx, _ = flat_records[flat_idx]
        user_list = asin_to_users.get(asin, [])
        cand_idx = [uid_idx[u] for u in user_list if u in uid_idx]
        if not cand_idx:
            n_no_user += 1
            continue
        cand_idx_arr = np.asarray(cand_idx, dtype=np.int64)
        diff = z_row[None, :] - mu[cand_idx_arr]
        d2 = (diff ** 2 / sigma2[cand_idx_arr]).sum(axis=-1)
        log_p = -0.5 * (d2 + log_norm_per_user[cand_idx_arr])
        best_k = int(np.argmax(log_p))
        per_asin_records.setdefault(asin, []).append({
            "logp": float(log_p[best_k]),
            "d2": float(d2[best_k]),
            "user_idx": int(cand_idx_arr[best_k]),
            "query": q_text,
        })

    log(f"  queries with candidate: {sum(len(v) for v in per_asin_records.values())}/{len(flat_records)}")

    # Per-ASIN selection
    kept: dict[str, list[str]] = {}
    selections_block: list = []
    drops = {"logp_below_delta": 0, "above_d2": 0}
    if GATE_MODE == "logp_delta":
        for asin, cands in per_asin_records.items():
            cands_sorted = sorted(cands, key=lambda c: -c["logp"])
            max_logp = cands_sorted[0]["logp"]
            kept_for_asin = [
                c["query"] for c in cands_sorted if c["logp"] >= max_logp - LOGP_DELTA
            ]
            drops["logp_below_delta"] += sum(
                1 for c in cands_sorted if c["logp"] < max_logp - LOGP_DELTA
            )
            kept[asin] = kept_for_asin
            # build selections block (for stage11 compatibility)
            users_block = [
                {
                    "uid": uid_order[c["user_idx"]],
                    "query": c["query"],
                    "logp": c["logp"],
                    "d2": c["d2"],
                }
                for c in cands_sorted if c["logp"] >= max_logp - LOGP_DELTA
            ]
            if users_block:
                selections_block.append({
                    "asin": asin,
                    "users": users_block,
                    "max_pair_cos": None,
                })
    elif GATE_MODE == "d2":
        # Fallback: 经验 d2_q95 × GATE_Q 阈值 (保留向后兼容, 不推荐)
        log("  d2 mode: per-query best-fit d2 + d2_q95 threshold")
        kept = {}
        drops = {"above_d2": 0}
        for ri, (z_row, flat_idx) in enumerate(zip(z_all, flat_idx_for_row)):
            asin, q_text, local_idx, _ = flat_records[flat_idx]
            user_list = asin_to_users.get(asin, [])
            cand_idx = [uid_idx[u] for u in user_list if u in uid_idx]
            if not cand_idx:
                continue
            cand_idx_arr = np.asarray(cand_idx, dtype=np.int64)
            diff = z_row[None, :] - mu[cand_idx_arr]
            d2 = (diff ** 2 / sigma2[cand_idx_arr]).sum(axis=-1)
            best_k = int(np.argmin(d2))
            best_d2_val = float(d2[best_k])
            threshold = float(d2_q95[cand_idx_arr[best_k]]) * GATE_Q
            if best_d2_val > threshold:
                drops["above_d2"] += 1
                continue
            kept.setdefault(asin, []).append(q_text)
    else:
        raise ValueError(f"unknown GATE_MODE={GATE_MODE!r}")

    log(f"  selected: {sum(len(v) for v in kept.values())}/{len(flat_records)} queries")
    log(f"  ASIN with >=1 kept: {len(kept)}/{len(pool)}")
    log(f"  drops: {drops}, n_no_user={n_no_user}")

    # ASIN>=K 分布
    counts = [len(v) for v in kept.values()]
    n_ge1 = sum(1 for c in counts if c >= 1)
    n_ge2 = sum(1 for c in counts if c >= 2)
    n_ge3 = sum(1 for c in counts if c >= 3)
    n_ge5 = sum(1 for c in counts if c >= 5)

    out = {
        "config": {
            "source": "svd_mlp",
            "latent_dim": LATENT_DIM,
            "gate_mode": GATE_MODE,
            "logp_delta": LOGP_DELTA if GATE_MODE == "logp_delta" else None,
            "gate_q": GATE_Q if GATE_MODE == "d2" else None,
            "gaussian_path": str(GAUSSIAN_PATH),
            "svd_components": str(SVD_COMPONENTS),
            "mlp_encoder": str(MLP_ENCODER),
            "n_pool_asin": len(pool),
            "n_total_queries": len(flat_records),
            "n_fitted_users": len(uid_idx),
            "device": DEVICE,
        },
        "kept": kept,
        "selections": selections_block,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    log(f"DONE wrote {OUT_PATH}")

    stats = {
        "gate_mode": GATE_MODE,
        "n_kept_queries": sum(len(v) for v in kept.values()),
        "n_total_queries": len(flat_records),
        "n_total_asin": len(pool),
        "n_asin_ge1": n_ge1,
        "n_asin_ge2": n_ge2,
        "n_asin_ge3": n_ge3,
        "n_asin_ge5": n_ge5,
        "drops": drops,
    }
    if GATE_MODE == "logp_delta":
        stats["logp_delta"] = LOGP_DELTA
    else:
        stats["gate_q"] = GATE_Q
    with open(OUT_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    log(f"DONE wrote {OUT_STATS_PATH}")


if __name__ == "__main__":
    main()
