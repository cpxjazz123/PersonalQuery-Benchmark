#!/usr/bin/env python3
"""E21 v8: 6-cell grid across 3 groups (realistic / balanced / high-info).

Per user plan (2026-08-15):
  Realistic: (L=3,N=10) [30 sents], (L=5,N=10) [50 sents]
  Balanced:  (L=5,N=20) [100 sents], (L=8,N=15) [120 sents]
  High-info: (L=8,N=30) [240 sents], (L=10,N=20) [200 sents]
  Same pre-registered 4-gate as v6:
    test_users >= 150
    AUC >= 0.65
    Cohen d >= 0.5
    seed_pass >= 0.80
    perm_p_two (Bonferroni over evaluated cells) < 0.01
  Use corrected stats from v4/v6/v7.
  All syntactic parsing delegated to parse_sentences_to_features.parse_corpus.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from e20_lopo_v2 import ALL_FEATS, load_spacy_model
from extract_syntactic_features import ALL_FEATS_V2, user_features_v2
from parse_sentences_to_features import parse_corpus

# v9: use 318-dim features (vs v1/v8's 32-dim)
ALL_FEATS = ALL_FEATS_V2
USER_FEATS_FN = user_features_v2

REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
OUT = REPO_ROOT / "result" / "e21_v10_results.json"
LOG = REPO_ROOT / "result" / "e21_v10.log"
CACHE_PATH = REPO_ROOT / "result" / "cache" / "per_sentence_features.jsonl.gz"

SEED = 43
N_USERS = 5000
MIN_WORDS = 100
MIN_ASINS = 2
# v10: 12 cells — 微调 + 修复
CELLS = ((3, 15), (3, 20), (4, 15), (4, 20), (5, 15), (5, 18),
         (5, 22), (5, 25), (5, 30), (6, 20), (7, 20), (8, 20))
N_SEEDS = 30
N_PERM = 9999
DEV_FRAC = 0.5
AUC_THRESH = 0.65
COHEN_D_THRESH = 0.5
P_THRESH = 0.01
SEED_PASS_FRAC = 0.80
TEST_MIN_USERS = 150


def user_features(sent_feats: list[dict]) -> np.ndarray | None:
    """v9: 318-dim syntactic features (see extract_syntactic_features)."""
    return USER_FEATS_FN(sent_feats)


def norm_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)


def auc_self_vs_cross(d_self: np.ndarray, d_cross: np.ndarray) -> float:
    if len(d_self) == 0 or len(d_cross) == 0:
        return float("nan")
    auc = 0.0
    for x in d_self:
        auc += float((d_cross > x).sum())
    return auc / (len(d_self) * len(d_cross))


def cohen_d(d_cross: np.ndarray, d_self: np.ndarray) -> float:
    if len(d_cross) < 2 or len(d_self) < 2:
        return float("nan")
    delta = d_cross.mean() - d_self.mean()
    pooled = np.sqrt((d_cross.std(ddof=1) ** 2 + d_self.std(ddof=1) ** 2) / 2)
    if pooled == 0:
        return 0.0
    return float(delta / pooled)


def perm_null_delta(all_halves: list[np.ndarray], rng: np.random.Generator) -> float:
    """Legacy: single permutation. Use perm_null_deltas_batched instead."""
    n = len(all_halves)
    if n < 4:
        return 0.0
    idx = rng.permutation(n)
    if n % 2:
        idx = idx[:-1]
    a = np.stack([all_halves[i] for i in idx[0::2]])
    b = np.stack([all_halves[i] for i in idx[1::2]])
    diffs = np.linalg.norm(a - b, axis=1)
    return float(diffs.mean())


def perm_null_deltas_batched(all_halves: list[np.ndarray],
                              n_perm: int,
                              rng: np.random.Generator,
                              batch_size: int = 128,
                              log_every_batches: int = 16) -> np.ndarray:
    """Vectorized permutation null distribution. Returns [n_perm] array.

    Optimizations vs loop:
      - Pre-stack all halves into [M, D] once
      - Generate `batch_size` random permutations per batch via argsort
      - Single fancy-index + numpy L2 per batch (faster than 128 Python loops)
      - Memory: [batch_n, M/2, D] ≈ 333 MB per tensor at batch_n=128, D=318
    """
    X = np.stack(all_halves)  # [M, D]
    M, D = X.shape
    if M < 4:
        return np.zeros(n_perm, dtype=np.float32)
    M2 = M - (M % 2)
    half = M2 // 2

    null_deltas = np.empty(n_perm, dtype=np.float32)
    for batch_start in range(0, n_perm, batch_size):
        batch_end = min(batch_start + batch_size, n_perm)
        batch_n = batch_end - batch_start
        # Random permutation per perm in batch: argsort of random keys
        keys = rng.random((batch_n, M))
        all_idx = np.argsort(keys, axis=1)[:, :M2]  # [batch_n, M2]
        a_idx = all_idx[:, 0::2]  # [batch_n, half]
        b_idx = all_idx[:, 1::2]
        a = X[a_idx]  # [batch_n, half, D]
        b = X[b_idx]
        diffs = np.linalg.norm(a - b, axis=2)  # [batch_n, half]
        null_deltas[batch_start:batch_end] = diffs.mean(axis=1)
        batch_idx = batch_start // batch_size
        if batch_idx % log_every_batches == 0:
            print(f"      perm {batch_end}/{n_perm}", flush=True)
    return null_deltas


def eval_cell(cell_users: list[str],
              user_sents_L: dict[int, dict[str, list[dict]]],
              L: int,
              N: int,
              seeds: tuple[int, ...]) -> list[dict]:
    per_seed = []
    for sd in seeds:
        rs = np.random.default_rng(sd)
        # Pass 1: collect per-user (v1n, v2n, others_n_pairs) — defer L2
        # user_rows[i] = (v1n, v2n, list_of_vvn)
        user_rows: list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = []
        for u in cell_users:
            sfs = user_sents_L[L][u]
            if len(sfs) < 2 * N:
                continue
            idx = rs.permutation(len(sfs))[:2 * N]
            half1 = [sfs[i] for i in idx[:N]]
            half2 = [sfs[i] for i in idx[N:]]
            v1 = user_features(half1)
            v2 = user_features(half2)
            if v1 is None or v2 is None:
                continue
            v1n = norm_vec(v1)
            v2n = norm_vec(v2)
            others = [uu for uu in cell_users if uu != u]
            rs.shuffle(others)
            vvns: list[np.ndarray] = []
            for v_other in others:
                if len(vvns) >= 3:
                    break
                sfs_v = user_sents_L[L][v_other]
                if len(sfs_v) < N:
                    continue
                idx_v = rs.permutation(len(sfs_v))[:N]
                vv = user_features([sfs_v[i] for i in idx_v])
                if vv is None:
                    continue
                vvns.append(norm_vec(vv))
            user_rows.append((v1n, v2n, vvns))

        if not user_rows:
            continue
        # Pass 2: vectorized L2 — stack v1n/v2n then matrix diff + norm
        V1 = np.stack([r[0] for r in user_rows])  # [U, D]
        V2 = np.stack([r[1] for r in user_rows])  # [U, D]
        diffs_self = np.linalg.norm(V1 - V2, axis=1)  # [U]
        diffs_cross: list[float] = []
        for u_idx, (_, _, vvns) in enumerate(user_rows):
            v1n = user_rows[u_idx][0]
            if not vvns:
                continue
            VV = np.stack(vvns)  # [k, D]
            cs = np.linalg.norm(VV - v1n[None, :], axis=1)  # [k]
            diffs_cross.extend(cs.tolist())
        ds = diffs_self.astype(np.float64)
        dc = np.asarray(diffs_cross, dtype=np.float64)
        if len(ds) < 5 or len(dc) < 5:
            continue
        per_seed.append({
            "seed": int(sd),
            "auc": auc_self_vs_cross(ds, dc),
            "delta": float(dc.mean() - ds.mean()),
            "cohen_d": cohen_d(dc, ds),
            "d_self_mean": float(ds.mean()),
            "d_cross_mean": float(dc.mean()),
            "n_self": len(ds),
            "n_cross": len(dc),
        })
    return per_seed


def eval_permutation(cell_users: list[str],
                     user_sents_L: dict[int, dict[str, list[dict]]],
                     L: int,
                     N: int,
                     obs_delta: float,
                     n_perm: int,
                     rng_p: np.random.Generator) -> tuple[float, float]:
    all_halves = []
    for u in cell_users:
        sfs = user_sents_L[L][u]
        if len(sfs) < 2 * N:
            continue
        rs_fixed = np.random.default_rng(0)
        idx = rs_fixed.permutation(len(sfs))[:2 * N]
        h1 = [sfs[i] for i in idx[:N]]
        h2 = [sfs[i] for i in idx[N:]]
        v1 = user_features(h1)
        v2 = user_features(h2)
        if v1 is not None:
            all_halves.append(norm_vec(v1))
        if v2 is not None:
            all_halves.append(norm_vec(v2))
    if len(all_halves) < 4:
        return float("nan"), float("nan")
    print(f"      perm: {n_perm} iterations on {len(all_halves)} halves "
          f"(dim={all_halves[0].shape[0]})", flush=True)
    t_p = time.time()
    null_deltas = np.asarray([perm_null_delta(all_halves, rng_p)
                              for _ in range(n_perm)])
    print(f"      perm done (t={time.time() - t_p:.1f}s)", flush=True)
    p_one = (float((null_deltas <= obs_delta).sum()) + 1) / (len(null_deltas) + 1)
    null_center = float(null_deltas.mean())
    p_two = (float((np.abs(null_deltas - null_center) >=
                    abs(obs_delta - null_center)).sum()) + 1) / (len(null_deltas) + 1)
    return float(p_one), float(p_two)


def main() -> None:
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    wc: dict[str, int] = defaultdict(int)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            t = d.get("text") or ""
            if u:
                wc[u] += len(t.split())
    print(f"users scanned: {len(wc)} (t={time.time() - t0:.1f}s)", flush=True)

    cands = [u for u, w in wc.items() if w >= MIN_WORDS]
    rng.shuffle(cands)
    cands = cands[:N_USERS * 4]
    poolset = set(cands)
    print(f"word>={MIN_WORDS} candidates: {len(cands)}", flush=True)

    reviews: dict[str, list[tuple]] = defaultdict(list)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            if u not in poolset:
                continue
            t = (d.get("text") or "").strip()
            if t:
                reviews[u].append((d.get("parent_asin") or d.get("asin"), t[:2000]))
    for u in list(reviews):
        if len({r[0] for r in reviews[u]}) < MIN_ASINS:
            del reviews[u]
    print(f"users with >= {MIN_ASINS} products: {len(reviews)}", flush=True)

    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    print(f"  active spaCy pipes: {nlp.pipe_names}", flush=True)

    # Parse corpus via shared module (uses SHA1-keyed cache)
    user_texts: list[tuple[str, str]] = []
    for u, revs in reviews.items():
        for _, t in revs:
            user_texts.append((u, t))
    print(f"  parsing {len(user_texts)} reviews for {len(reviews)} users "
          f"(via parse_sentences_to_features)", flush=True)
    user_sents = parse_corpus(user_texts, CACHE_PATH, nlp=nlp,
                              batch_size=256, log_prefix="  ")
    print(f"users parsed: {len(user_sents)} (t={time.time() - t0:.1f}s)",
          flush=True)

    # Build per-L user_sents (each cell's L filter is independent)
    unique_L = sorted({L for L, _ in CELLS})
    user_sents_L: dict[int, dict[str, list[dict]]] = {}
    for L in unique_L:
        user_sents_L[L] = {}
        for u, sfs in user_sents.items():
            user_sents_L[L][u] = [s for s in sfs if s["n_tok"] >= L]

    # Compute eligible counts per (L, N) cell
    eligible_per_cell: dict[str, list[str]] = {}
    for L, N in CELLS:
        ckey = f"L={L}_N={N}"
        eligible_per_cell[ckey] = [u for u, sfs in user_sents_L[L].items()
                                   if len(sfs) >= 2 * N]
        print(f"  eligible for {ckey} (>=L={L} & >={2 * N} sents): "
              f"{len(eligible_per_cell[ckey])}", flush=True)

    # Subsample the largest eligible pool to N_USERS (deterministic via rng)
    biggest_ckey = max(eligible_per_cell, key=lambda k: len(eligible_per_cell[k]))
    if len(eligible_per_cell[biggest_ckey]) > N_USERS:
        rng.shuffle(eligible_per_cell[biggest_ckey])
        eligible_per_cell[biggest_ckey] = eligible_per_cell[biggest_ckey][:N_USERS]
        print(f"  subsampled {biggest_ckey} to {N_USERS}", flush=True)

    # For each cell, split into dev/test 50/50
    rng_split = np.random.default_rng(SEED + 100)
    splits: dict[str, tuple[list[str], list[str]]] = {}
    for L, N in CELLS:
        ckey = f"L={L}_N={N}"
        users_n = sorted(eligible_per_cell[ckey])
        perm = rng_split.permutation(len(users_n))
        n_dev = int(len(users_n) * DEV_FRAC)
        dev_idx = set(perm[:n_dev].tolist())
        dev_users_n = [u for i, u in enumerate(users_n) if i in dev_idx]
        test_users_n = [u for i, u in enumerate(users_n) if i not in dev_idx]
        splits[ckey] = (dev_users_n, test_users_n)
        print(f"  {ckey} dev={len(dev_users_n)} test={len(test_users_n)}",
              flush=True)

    seeds_dev = tuple(int(SEED + 1000 + i) for i in range(N_SEEDS))
    seeds_test = tuple(int(SEED + 5000 + i) for i in range(N_SEEDS))
    rng_perm = np.random.default_rng(SEED + 2000)

    grid_results: dict[str, dict] = {}
    for L, N in CELLS:
        ckey = f"L={L}_N={N}"
        cell_t0 = time.time()
        cell_dev, cell_test = splits[ckey]
        print(f"\n=== Cell {ckey}: dev={len(cell_dev)} test={len(cell_test)} ===",
              flush=True)
        if len(cell_dev) < 30 or len(cell_test) < TEST_MIN_USERS:
            grid_results[ckey] = {
                "eligible_dev": len(cell_dev),
                "eligible_test": len(cell_test),
                "skipped": "too few users",
            }
            continue

        per_seed_dev = eval_cell(cell_dev, user_sents_L, L, N, seeds_dev)
        if not per_seed_dev:
            grid_results[ckey] = {
                "eligible_dev": len(cell_dev),
                "eligible_test": len(cell_test),
                "skipped": "no seed produced >=5 pairs",
            }
            continue
        aucs = np.asarray([s["auc"] for s in per_seed_dev])
        deltas = np.asarray([s["delta"] for s in per_seed_dev])
        cohen_ds = np.asarray([s["cohen_d"] for s in per_seed_dev])
        seed_pass_frac = float((aucs >= AUC_THRESH).mean())
        obs_delta = float(deltas.mean())
        obs_cohen = float(cohen_ds.mean())
        print(f"  dev AUC={aucs.mean():.4f}+-{aucs.std():.4f}  "
              f"d={obs_cohen:.4f}  seed_pass={seed_pass_frac:.2f}  "
              f"delta={obs_delta:.5f}", flush=True)

        p_one, p_two = eval_permutation(cell_dev, user_sents_L, L, N,
                                        obs_delta, N_PERM, rng_perm)
        print(f"  perm p_one={p_one:.5f}  p_two={p_two:.5f}", flush=True)

        per_seed_test = eval_cell(cell_test, user_sents_L, L, N, seeds_test)
        aucs_t = np.asarray([s["auc"] for s in per_seed_test])
        deltas_t = np.asarray([s["delta"] for s in per_seed_test])
        cohen_ts = np.asarray([s["cohen_d"] for s in per_seed_test])
        test_seed_pass = float((aucs_t >= AUC_THRESH).mean())
        test_cohen = float(cohen_ts.mean())
        print(f"  test AUC={aucs_t.mean():.4f}  d={test_cohen:.4f}  "
              f"seed_pass={test_seed_pass:.2f}", flush=True)

        grid_results[ckey] = {
            "eligible_dev": len(cell_dev),
            "eligible_test": len(cell_test),
            "per_seed_dev": per_seed_dev,
            "per_seed_test": per_seed_test,
            "dev_auc_mean": round(float(aucs.mean()), 4),
            "dev_auc_std": round(float(aucs.std()), 4),
            "dev_cohen_d_mean": round(obs_cohen, 4),
            "dev_delta_mean": round(obs_delta, 5),
            "dev_seed_pass_frac": round(seed_pass_frac, 4),
            "test_auc_mean": round(float(aucs_t.mean()), 4),
            "test_auc_std": round(float(aucs_t.std()), 4),
            "test_cohen_d_mean": round(test_cohen, 4),
            "test_delta_mean": round(float(deltas_t.mean()), 5),
            "test_seed_pass_frac": round(test_seed_pass, 4),
            "perm_p_one": round(p_one, 5),
            "perm_p_two": round(p_two, 5),
            "runtime_sec": round(time.time() - cell_t0, 1),
        }

    n_cells = sum(1 for r in grid_results.values() if "perm_p_two" in r)
    for r in grid_results.values():
        if "perm_p_two" in r:
            r["perm_p_two_bonf"] = round(min(1.0, r["perm_p_two"] * n_cells), 5)

    for L, N in CELLS:
        ckey = f"L={L}_N={N}"
        if ckey not in grid_results or "dev_auc_mean" not in grid_results[ckey]:
            continue
        r = grid_results[ckey]
        passes = (
            r["eligible_test"] >= TEST_MIN_USERS
            and r["dev_auc_mean"] >= AUC_THRESH
            and r["dev_cohen_d_mean"] >= COHEN_D_THRESH
            and r["dev_seed_pass_frac"] >= SEED_PASS_FRAC
            and r["perm_p_two_bonf"] < P_THRESH
        )
        r["all_gates_pass"] = bool(passes)

    best_cell = None
    # best_cell = max test Cohen d among all_gates_pass cells
    go_cells = [(L, N, grid_results[f"L={L}_N={N}"].get("test_cohen_d_mean", 0.0))
                for L, N in CELLS
                if grid_results.get(f"L={L}_N={N}", {}).get("all_gates_pass")]
    if go_cells:
        best_cell = list(sorted(go_cells, key=lambda x: -x[2])[0][:2])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "version": "v10 (fine-tune around (5,20) + fix low seed_pass cells)",
        "purpose": "12-cell grid: (3,15)(3,20)(4,15)(4,20)(5,15)(5,18)(5,22)(5,25)(5,30)(6,20)(7,20)(8,20)",
        "seed": SEED,
        "cells": [{"L": L, "N": N, "total_sents": 2 * N} for L, N in CELLS],
        "n_perm": N_PERM,
        "n_seeds": N_SEEDS,
        "n_users_parsed": len(user_sents),
        "eligible_per_cell_before_subsample": {
            f"L={L}_N={N}": len(eligible_per_cell[f"L={L}_N={N}"]) for L, N in CELLS
        },
        "grid": grid_results,
        "best_cell": best_cell,
        "gates": {"test_users_min": TEST_MIN_USERS,
                  "auc_min": AUC_THRESH, "cohen_d_min": COHEN_D_THRESH,
                  "seed_pass_min": SEED_PASS_FRAC,
                  "perm_p_two_bonf_max": P_THRESH},
        "runtime_sec": round(time.time() - t0, 1),
    }, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT} (t={time.time() - t0:.1f}s)", flush=True)
    print(f"best_cell: {best_cell}", flush=True)


if __name__ == "__main__":
    main()