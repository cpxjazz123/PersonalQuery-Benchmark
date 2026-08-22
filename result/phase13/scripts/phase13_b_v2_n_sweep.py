#!/usr/bin/env python3
"""Phase 13.B-v2: N sweep on Gaussian fitting convergence.

Question: 多少 sentences/user 拟合的 σ 才不嘈杂? (Phase 15.D 验证 ||σ|| >> ||μ|| 让 ρ noise 爆炸)

Approach:
  1. 取 198 phase10 users (≥50 reviews in raw Amazon Baby_Products_2023)
  2. 抽 Qwen hidden states (layer 14, mean-pool) for 100 sentences per user
  3. 拟合 Gaussian with N ∈ {10, 20, 30, 50, 100} (前 N 句取前 N)
  4. 输出 ||σ||/||μ|| ratio per (user, N)
  5. 找 ratio 收敛点 (= plateau 起点)
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
RAW_REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"

QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
LAYER = 14
N_SENTS_PER_USER = 100
N_LEVELS = [10, 20, 30, 50, 100]
MIN_SENT_WORDS = 5
MAX_SENT_WORDS = 60
HIDDEN_BATCH = 32
HIDDEN_MAX_LEN = 160
SEED = 42

OUT_HIDDEN = OUT_DIR / "phase13_b_v2_user_hiddens_n100.npz"
OUT_RATIOS = OUT_DIR / "phase13_b_v2_ratio_per_user.csv"
OUT_SUMMARY = OUT_DIR / "phase13_b_v2_summary.json"
LOG_PATH = OUT_DIR / "phase13_b_v2.log"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_sents(text: str) -> list[str]:
    # 简单 punctuation 切句, 去掉 <br/> 等 HTML
    text = re.sub(r"<[^>]+>", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def main() -> None:
    log("=" * 70)
    log("Phase 13.B-v2: N sweep on Gaussian fitting convergence")
    log("=" * 70)

    np.random.seed(SEED)

    # === [1] Find 198 users (≥50 reviews) and extract sentences ===
    log("[1] Finding 198 users with ≥50 reviews in raw data ...")
    target_uids: set[str] = set()
    with PAIRS_FILE.open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                target_uids.add(d["user_id"])

    ur: dict[str, list[str]] = {u: [] for u in target_uids}
    with gzip.open(RAW_REVIEWS, "rt") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                uid = d.get("user_id")
                if uid in ur:
                    text = d.get("text", "")
                    if text:
                        ur[uid].append(text)

    ur50 = {u: v for u, v in ur.items() if len(v) >= 50}
    log(f"  users with ≥50 reviews: {len(ur50)}")

    # sentence split + filter
    user_sents: dict[str, list[str]] = {}
    for u, revs in ur50.items():
        sents = []
        for r in revs:
            for s in split_sents(r):
                wc = len(s.split())
                if MIN_SENT_WORDS <= wc <= MAX_SENT_WORDS:
                    sents.append(s)
        user_sents[u] = sents
    sent_counts = [len(v) for v in user_sents.values()]
    log(f"  sents per user: min={min(sent_counts)} median={sorted(sent_counts)[len(sent_counts)//2]} max={max(sent_counts)}")
    log(f"  total sentences to encode: {sum(min(N_SENTS_PER_USER, c) for c in sent_counts)}")

    # === [2] Load Qwen + extract layer 14 hidden states (mean-pool per sentence) ===
    log("[2] Loading Qwen (transformers backend) ...")
    os.environ["QWEN_MODEL_PATH"] = QWEN_MODEL_PATH
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import QwenLocalClient
    import torch
    client = QwenLocalClient(with_vllm=False)
    model = client._hidden_backend.model
    tok = client._hidden_backend.tokenizer
    device = client._hidden_backend.device
    log(f"  Qwen loaded on {device}")

    log("[3] Extracting layer 14 hidden states (mean-pool) ...")
    user_hiddens: dict[str, np.ndarray] = {}  # [N, H]
    t0 = time.time()
    for uidx, (uid, sents) in enumerate(user_sents.items()):
        sents_use = sents[:N_SENTS_PER_USER]
        N = len(sents_use)
        hiddens = np.zeros((N, model.config.hidden_size), dtype=np.float32)
        # batch encode
        for st in range(0, N, HIDDEN_BATCH):
            chunk = sents_use[st:st + HIDDEN_BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=HIDDEN_MAX_LEN).to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            # layer 14 hidden states, mean-pool over valid positions
            hs = out.hidden_states[LAYER]  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            mean_h = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, H]
            hiddens[st:st + len(chunk)] = mean_h.float().cpu().numpy()
        user_hiddens[uid] = hiddens
        if (uidx + 1) % 20 == 0:
            log(f"  {uidx + 1}/{len(user_sents)} users ({time.time() - t0:.1f}s)")

    log(f"  done ({time.time() - t0:.1f}s), caching to {OUT_HIDDEN}")
    # 用 object array 存储变长 hiddens (不同 user sentences 数不同)
    h_per_user = np.empty(len(user_hiddens), dtype=object)
    for i, uid in enumerate(user_hiddens.keys()):
        h_per_user[i] = user_hiddens[uid]
    np.savez(OUT_HIDDEN,
             user_ids=np.array(list(user_hiddens.keys())),
             hiddens=h_per_user,
             counts=np.array([len(v) for v in user_hiddens.values()]))

    # === [3] Free Qwen ===
    log("[4] Freeing Qwen ...")
    del model
    del client
    torch.cuda.empty_cache()

    # === [4] Fit Gaussian with N ∈ {10, 20, 30, 50, 100} (前 N 句取前 N) ===
    log("[5] Fitting Gaussian with N levels ...")
    user_ids = list(user_hiddens.keys())
    H = model.config.hidden_size if False else user_hiddens[user_ids[0]].shape[1]
    # 上面 model.config 已经被 del 释放; 改读 hiddens.shape
    H = user_hiddens[user_ids[0]].shape[1]
    log(f"  H = {H}")

    rows = []
    for N in N_LEVELS:
        log(f"  N={N}:")
        for uid in user_ids:
            h = user_hiddens[uid][:N]  # [N, H]
            mu = h.mean(axis=0)
            sigma = h.std(axis=0, ddof=0)  # per-dim std (与 phase13_b 一致)
            mu_norm = float(np.linalg.norm(mu))
            sigma_norm = float(np.linalg.norm(sigma))
            ratio = sigma_norm / (mu_norm + 1e-12)
            rows.append({
                "user_id": uid,
                "N": N,
                "mu_norm": mu_norm,
                "sigma_norm": sigma_norm,
                "ratio_sigma_over_mu": ratio,
            })
    log(f"  fits: {len(rows)}")

    # === [5] Write CSV ===
    log(f"[6] Writing CSV: {OUT_RATIOS}")
    with OUT_RATIOS.open("w") as f:
        f.write("user_id,N,mu_norm,sigma_norm,ratio_sigma_over_mu\n")
        for r in rows:
            f.write(f"{r['user_id']},{r['N']},{r['mu_norm']:.4f},{r['sigma_norm']:.4f},{r['ratio_sigma_over_mu']:.4f}\n")

    # === [6] Aggregate per N ===
    log("[7] Aggregate per N ...")
    summary: dict = {"phase": "13.B-v2", "n_users": len(user_ids), "N_levels": N_LEVELS, "per_N": {}}
    print(f"\n{'N':>4} {'mean_mu':>10} {'mean_sigma':>11} {'mean_ratio':>11} {'med_ratio':>10}")
    print("-" * 55)
    for N in N_LEVELS:
        ms = [r["mu_norm"] for r in rows if r["N"] == N]
        ss = [r["sigma_norm"] for r in rows if r["N"] == N]
        rs = [r["ratio_sigma_over_mu"] for r in rows if r["N"] == N]
        stats = {
            "mean_mu_norm": float(np.mean(ms)),
            "mean_sigma_norm": float(np.mean(ss)),
            "mean_ratio": float(np.mean(rs)),
            "median_ratio": float(np.median(rs)),
            "std_ratio": float(np.std(rs)),
        }
        summary["per_N"][N] = stats
        print(f"{N:>4} {stats['mean_mu_norm']:>10.3f} {stats['mean_sigma_norm']:>11.3f} {stats['mean_ratio']:>11.4f} {stats['median_ratio']:>10.4f}")

    # 找 plateau 起点 (ratio 变化 <5%)
    prev = None
    plateau_start = None
    for N in N_LEVELS:
        r = summary["per_N"][N]["mean_ratio"]
        if prev is not None:
            change = abs(r - prev) / prev
            if change < 0.05 and plateau_start is None:
                plateau_start = N
        prev = r
    summary["plateau_start_N"] = plateau_start
    print(f"\n→ ratio plateau starts at N={plateau_start}")

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  saved summary: {OUT_SUMMARY}")

    log("=" * 70)
    log("PHASE 13.B-v2 N SWEEP DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()