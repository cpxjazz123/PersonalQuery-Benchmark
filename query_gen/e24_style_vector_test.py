#!/usr/bin/env python3
"""E24 Phase D — 独立 50 用户 test 验证。

输入:
  - pareto_selection.json 中 chosen (X*, Y*, L*)
  - /home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/test_pool_50u.jsonl
  - /home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/test_rewrites.jsonl
  - /home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/test_hidden_L{24,26,27}.npz
  - grid_validity.json (dev baseline 对比)

输出:
  /home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/test_validation.json

每个 chosen cell 在 50 新用户上重新跑：
  - direction: own_minus_other
  - identity: top1, top5, AUC
  - stability: split_half_cos
并报告相对 dev 的退化率。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")

OUT_E24 = Path("/home/wlia0047/hj82_scratch2/wenyu/e24_style_vector")
PARETO_JSON = OUT_E24 / "pareto_selection.json"
GRID_JSON = OUT_E24 / "grid_validity.json"
TEST_POOL = OUT_E24 / "test_pool_50u.jsonl"
TEST_REW = OUT_E24 / "test_rewrites.jsonl"
TEST_HIDDEN = {24: OUT_E24 / "test_hidden_L24.npz",
               26: OUT_E24 / "test_hidden_L26.npz",
               27: OUT_E24 / "test_hidden_L27.npz"}
OUT_JSON = OUT_E24 / "test_validation.json"

N_HO = 8
N_SPLIT_HALF = 30
SEED = 7777


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unit(a, axis=-1):
    n = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / np.maximum(n, 1e-9)


def load_hidden(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    c = np.load(path, allow_pickle=True)
    return {str(t): np.asarray(v) for t, v in zip(c["texts"], c["vecs"])}


def main() -> None:
    t0 = time.time()
    if not PARETO_JSON.exists():
        raise FileNotFoundError(f"{PARETO_JSON} missing; run Phase C first")
    pareto = json.load(open(PARETO_JSON))
    chosen = pareto["chosen"]
    X, Y, L = chosen["X"], chosen["Y"], chosen["layer"]
    log(f"chosen: X={X} Y={Y} L={L} from pareto")

    if not TEST_POOL.exists():
        raise FileNotFoundError(f"{TEST_POOL} missing; run Phase A first")

    users = []
    with open(TEST_POOL) as f:
        for line in f:
            if line.strip():
                users.append(json.loads(line))
    log(f"loaded {len(users)} test users")

    rewrites: dict[str, str] = {}
    with open(TEST_REW) as f:
        for line in f:
            try:
                d = json.loads(line)
                rewrites[d["sentence"]] = d["rewrite"]
            except Exception:
                pass
    log(f"loaded {len(rewrites)} test rewrites")

    cache = load_hidden(TEST_HIDDEN[L])
    log(f"loaded {len(cache)} test hidden states at L{L}")

    # ---- per-user construction + held-out deltas ----
    rng = random.Random(SEED)
    user_constr: dict[str, list[str]] = {}
    user_ho: dict[str, list[str]] = {}
    for u in users:
        uid = u["user_id"]
        cs = [s for s in u["construction_sents"]
              if s in cache and s in rewrites and rewrites[s] in cache]
        ho = [s for s in u["heldout_sents"]
              if s in cache and s in rewrites and rewrites[s] in cache]
        user_constr[uid] = cs[:Y]
        user_ho[uid] = ho[:N_HO]
    keep = [uid for uid, in [u["user_id"] for u in users]
            if len(user_constr[uid]) >= max(3, Y // 2)
            and len(user_ho[uid]) >= N_HO]
    log(f"test users with enough data: {len(keep)}")

    # ---- held-out deltas ----
    ho_delta = []
    for uid in keep:
        ua = np.stack([cache[s] for s in user_ho[uid]])
        na = np.stack([cache[rewrites[s]] for s in user_ho[uid]])
        ho_delta.append(ua - na)
    ho_delta = np.stack(ho_delta)              # [U, N_HO, H]
    log(f"ho_delta: {ho_delta.shape}")

    # ---- style vectors ----
    style_vecs = []
    for uid in keep:
        cs = user_constr[uid]
        ua = np.stack([cache[s] for s in cs])
        na = np.stack([cache[rewrites[s]] for s in cs])
        style_vecs.append(ua.mean(0) - na.mean(0))
    S = np.stack(style_vecs)
    S_u = unit(S)

    # ---- metrics ----
    own_c, other_c = [], []
    top1, top5, ranks = 0, 0, []
    n_us = len(keep)
    for j in range(n_us):
        for i in range(N_HO):
            dq = unit(ho_delta[j, i])
            sims = dq @ S_u.T
            order = np.argsort(-sims)
            rank = int(np.where(order == j)[0][0]) + 1
            ranks.append(rank)
            top1 += (rank == 1)
            top5 += (rank <= 5)
            own_c.append(float(dq @ S_u[j]))
            other_c.append(float(np.mean(dq @ np.delete(S_u, j, axis=0).T)))
    n_total = len(ranks)
    auc = float(np.mean([1.0 - (r - 1) / max(n_us - 1, 1) for r in ranks]))

    # ---- split-half ----
    per_sent_delta = {}
    for uid in keep:
        cs = user_constr[uid]
        deltas = []
        for s in cs:
            try:
                deltas.append(cache[s] - cache[rewrites[s]])
            except KeyError:
                continue
        per_sent_delta[uid] = deltas

    sh_cos = []
    for _ in range(N_SPLIT_HALF):
        cos_per_user = []
        for uid in keep:
            ds = per_sent_delta[uid]
            if len(ds) < 4:
                continue
            idx = list(range(len(ds)))
            rng.shuffle(idx)
            a = np.mean([ds[k] for k in idx[:len(ds)//2]], axis=0)
            b = np.mean([ds[k] for k in idx[len(ds)//2:]], axis=0)
            cos_per_user.append(float(unit(a) @ unit(b)))
        if cos_per_user:
            sh_cos.append(np.mean(cos_per_user))

    test_metrics = {
        "n_users": n_us,
        "own_cos": round(float(np.mean(own_c)), 4),
        "other_cos": round(float(np.mean(other_c)), 4),
        "own_minus_other": round(
            float(np.mean(own_c) - np.mean(other_c)), 4),
        "top1": round(top1 / n_total, 4),
        "top5": round(top5 / n_total, 4),
        "auc": round(auc, 4),
        "self_rank_mean": round(float(np.mean(ranks)), 2),
        "split_half_cos": round(float(np.mean(sh_cos)), 4),
        "split_half_std": round(float(np.std(sh_cos)), 4),
    }
    log(f"TEST (X={X} Y={Y} L={L}): "
        f"n={n_us} own-other={test_metrics['own_minus_other']:+.3f} "
        f"top1={test_metrics['top1']:.3f} sh={test_metrics['split_half_cos']:.3f}")

    # ---- compare to dev baseline ----
    dev_metrics = None
    if GRID_JSON.exists():
        grid = json.load(open(GRID_JSON))
        key = f"X{X}_Y{Y}_L{L}"
        if key in grid["results"]:
            dev_metrics = grid["results"][key]
            log(f"DEV baseline ({key}): n={dev_metrics.get('n_users')} "
                f"own-other={dev_metrics.get('own_minus_other'):+.3f} "
                f"top1={dev_metrics.get('top1'):.3f} "
                f"sh={dev_metrics.get('split_half_cos'):.3f}")

    degradation = {}
    if dev_metrics:
        for k in ("own_minus_other", "top1", "top5", "auc", "split_half_cos"):
            d = dev_metrics.get(k)
            t = test_metrics.get(k)
            if d is not None and t is not None and abs(d) > 1e-6:
                degradation[k] = {
                    "dev": d, "test": t,
                    "rel_drop": round((t - d) / abs(d), 4)}

    out = {
        "version": "e24_test_v1",
        "chosen": chosen,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "degradation": degradation,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    log(f"DONE — wrote {OUT_JSON} ({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()