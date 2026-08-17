#!/usr/bin/env python3
"""E23 P0 — validity analysis of extracted style vectors.

Questions answered:
  A. What do the hidden activations look like? (norms, Δ magnitude, dim activity)
  B. Is the style vector statistically reliable?
       B1 split-half reliability: style vector from half the pairs vs the other
          half (per user, averaged over random splits) — vs inter-user cosine
       B2 shuffle-pair null: permute user/neutral pairing within a user —
          does the mean_diff direction survive shuffling?
       B3 between-user distinctness + leave-one-pair-out identity retrieval:
          does s_u rank the held-out pair's Δ higher than other users' s_v?
  C. Method agreement: cosine(mean_diff, logreg, pca) per user per layer
  D. Which layer is best (reliability + retrieval)?

All computation on the cached hidden vectors; no GPU / no generation needed.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
OUT_DIR = REPO_ROOT / "result" / "e23_style_vector"
VEC_NPZ = OUT_DIR / "style_vectors_100u.npz"
HIDDEN_CACHE = OUT_DIR / "hidden_cache.npz"
ANALYSIS_JSON = OUT_DIR / "analysis.json"

SEED = 12345
N_SPLITS = 30          # random half-splits per user (B1)
N_SHUFFLES = 30        # pairing permutations per user (B2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


def main() -> None:
    t0 = time.time()
    rng = random.Random(SEED)

    v = np.load(VEC_NPZ, allow_pickle=True)
    users = [str(u) for u in v["users"]]
    sentences = [str(s) for s in v["sentences"]]
    layers = [int(l) for l in v["layers"]]
    n_per_user = [int(n) for n in v["n_per_user"]]
    sv = {m: v[m] for m in ["mean_diff", "logreg", "pca"]}
    U = len(users)
    L = len(layers)
    H = sv["mean_diff"].shape[-1]

    h = np.load(HIDDEN_CACHE, allow_pickle=True)
    hid_texts = [str(t) for t in h["texts"]]
    hid = h["vecs"]
    hid_map = {t: hid[i] for i, t in enumerate(hid_texts)}

    with open(OUT_DIR / "rewrites.jsonl", "r", encoding="utf-8") as f:
        rewrites = {json.loads(line)["sentence"]: json.loads(line)["rewrite"]
                    for line in f}

    # ---- per-user Δ arrays [n_i, L, H] ----
    offs = np.cumsum([0] + n_per_user)
    user_delta: list[np.ndarray] = []
    user_acts: list[np.ndarray] = []
    neutral_acts: list[np.ndarray] = []
    for ui in range(U):
        seg = sentences[offs[ui]:offs[ui + 1]]
        ua = np.stack([hid_map[s] for s in seg])
        na = np.stack([hid_map[rewrites[s]] for s in seg])
        user_acts.append(ua)
        neutral_acts.append(na)
        user_delta.append(ua - na)
    n_avail = np.array([len(d) for d in user_delta], dtype=int)
    log(f"users={U}, pairs per user: min={n_avail.min()} max={n_avail.max()} "
        f"(need >=2 for split-half)")

    # ================= A. activation structure =================
    log("A. activation structure...")
    a_struct: dict = {}
    for li, l in enumerate(layers):
        ua = np.concatenate([a[:, li] for a in user_acts])
        na = np.concatenate([a[:, li] for a in neutral_acts])
        dl = np.concatenate([d[:, li] for d in user_delta])
        a_struct[f"layer{l}"] = {
            "act_norm_mean": round(float(np.linalg.norm(ua, axis=1).mean()), 2),
            "neutral_norm_mean": round(
                float(np.linalg.norm(na, axis=1).mean()), 2),
            "delta_norm_mean": round(float(np.linalg.norm(dl, axis=1).mean()), 2),
            "delta_over_act_norm": round(float(
                np.linalg.norm(dl, axis=1).mean()
                / np.linalg.norm(ua, axis=1).mean()), 4),
            "delta_std_mean": round(float(dl.std(axis=1).mean()), 3),
            "dims_abs_gt_1std": round(float(
                (np.abs(dl) > dl.std(axis=0, keepdims=True).mean()).mean()), 4),
            "cos_pair_mean": round(float(
                (unit(ua) * unit(na)).sum(axis=1).mean()), 4),
        }

    # ================= B1. split-half reliability =================
    log("B1. split-half reliability...")
    rel: dict[str, dict] = {}
    for li, l in enumerate(layers):
        cos_halves: list[float] = []
        for ui in range(U):
            n = n_avail[ui]
            if n < 4:
                continue
            for _ in range(N_SPLITS):
                idx = rng.sample(range(n), n // 2)
                m1 = user_delta[ui][idx][:, li].mean(axis=0)
                m2 = user_delta[ui][
                    [i for i in range(n) if i not in set(idx)]][:, li].mean(axis=0)
                cos_halves.append(float((unit(m1) * unit(m2)).sum()))
        # inter-user cosine (chance reference) at the same layer
        S = unit(sv["mean_diff"][:, li]) @ unit(sv["mean_diff"][:, li]).T
        off = S[~np.eye(U, dtype=bool)]
        rel[f"layer{l}"] = {
            "split_half_cos_mean": round(float(np.mean(cos_halves)), 4),
            "split_half_cos_std": round(float(np.std(cos_halves)), 4),
            "n_samples": len(cos_halves),
            "noise_null_cos": round(float(1.0 / np.sqrt(H)), 4),
            "inter_user_cos_mean": round(float(off.mean()), 4),
            "separation": round(float(np.mean(cos_halves)) - float(off.mean()), 4),
        }

    # ================= B2. cross-user mispairing null =================
    # Null hypothesis: s_u carries NO user-specific signal and is just the
    # generic "user-review style vs neutral" shift. Test: pair user u's
    # activations with ANOTHER user's neutral activations (breaks both content
    # and style pairing). If s_u were generic, cos(true, cross) ≈ 1; if it is
    # user-specific, the cross-paired direction should diverge.
    log("B2. cross-user mispairing null...")
    null: dict[str, dict] = {}
    for li, l in enumerate(layers):
        cos_true_vs_cross: list[float] = []
        norm_true = 0.0
        norm_cross = 0.0
        for ui in range(U):
            n = n_avail[ui]
            ua = user_acts[ui][:, li]
            na = neutral_acts[ui][:, li]
            true = (ua - na).mean(axis=0)
            norm_true += float(np.linalg.norm(true))
            for _ in range(N_SHUFFLES):
                v = rng.randrange(U)
                nv = n_avail[v]
                other = neutral_acts[v][:, li]
                idx = [rng.randrange(nv) for _ in range(n)]
                cross = (ua - other[idx]).mean(axis=0)
                cos_true_vs_cross.append(
                    float((unit(true) * unit(cross)).sum()))
                norm_cross += float(np.linalg.norm(cross))
        cs = np.array(cos_true_vs_cross)
        null[f"layer{l}"] = {
            "cos_true_vs_cross_user": round(float(cs.mean()), 4),
            "cos_true_vs_cross_user_std": round(float(cs.std()), 4),
            "p(c2s>0.5)": round(float((cs > 0.5).mean()), 4),
            "true_norm_mean": round(float(norm_true / U), 2),
            "cross_norm_mean": round(float(norm_cross / (U * N_SHUFFLES)), 2),
            "norm_ratio_true_over_cross": round(float(
                norm_true / max(norm_cross / N_SHUFFLES, 1e-9)), 3),
        }

    # ================= B3. leave-one-pair-out identity retrieval =================
    log("B3. identity retrieval (leave-one-pair-out)...")
    retr: dict[str, dict] = {}
    for li, l in enumerate(layers):
        full = np.stack([d[:, li].mean(axis=0) for d in user_delta])  # [U,H]
        hits1 = 0
        hits5 = 0
        n_test = 0
        per_user_hit = np.zeros(U)
        per_user_cnt = np.zeros(U)
        for ui in range(U):
            n = n_avail[ui]
            for i in range(n):
                # leave out pair (ui,i): adjusted vector for user ui
                adjusted = full.copy()
                if n > 1:
                    adjusted[ui] = (full[ui] * n - user_delta[ui][i, li]) / (n - 1)
                du = unit(user_delta[ui][i, li])
                sims = du @ unit(adjusted).T
                order = np.argsort(-sims)
                rank = int(np.where(order == ui)[0][0]) + 1
                hits1 += rank == 1
                hits5 += rank <= 5
                per_user_hit[ui] += rank == 1
                per_user_cnt[ui] += 1
                n_test += 1
        retr[f"layer{l}"] = {
            "top1": round(float(hits1 / n_test), 4),
            "top5": round(float(hits5 / n_test), 4),
            "chance_top1": round(1.0 / U, 4),
            "chance_top5": round(5.0 / U, 4),
            "n_test": n_test,
            "users_with_top1_hit": round(float((per_user_hit > 0).mean()), 4),
        }

    # ================= C. method agreement =================
    log("C. method agreement...")
    meth_agree: dict[str, dict] = {}
    for li, l in enumerate(layers):
        cu = unit(sv["mean_diff"][:, li]) @ unit(sv["logreg"][:, li]).T
        cp = unit(sv["mean_diff"][:, li]) @ unit(sv["pca"][:, li]).T
        clp = unit(sv["logreg"][:, li]) @ unit(sv["pca"][:, li]).T
        meth_agree[f"layer{l}"] = {
            "cos(mean_diff, logreg)": round(float(np.diag(cu).mean()), 4),
            "cos(mean_diff, pca)": round(float(np.diag(cp).mean()), 4),
            "cos(logreg, pca)": round(float(np.diag(clp).mean()), 4),
        }

    # ================= C2. common vs individual direction =================
    # s_u decomposes into a GLOBAL mean direction (shared by all users — the
    # generic "user review style vs neutral" shift) and the user-specific
    # residual. If most norm sits in the common direction, the vector is
    # mostly NOT individual.
    log("C2. common vs individual decomposition...")
    common: dict[str, dict] = {}
    for li, l in enumerate(layers):
        S = sv["mean_diff"][:, li]                       # [U,H]
        S0 = S - S.mean(axis=0, keepdims=True)           # residual (user-specific)
        g = S.mean(axis=0)
        gn = float(np.linalg.norm(g))
        # fraction of each user's norm along the common direction
        cos_common = np.abs(unit(S) @ unit(g))
        res_norms = np.linalg.norm(S0, axis=1)
        tot_norms = np.linalg.norm(S, axis=1)
        # variance decomposition (norm^2): share of per-user variance (||s_u||^2)
        # explained by the global common direction g (mean over users of s_u)
        var_common = float((g ** 2).sum()
                           / max(np.mean(np.sum(S ** 2, axis=1)), 1e-9))
        common[f"layer{l}"] = {
            "common_norm": round(gn, 2),
            "user_frac_norm_along_common_mean": round(
                float(cos_common.mean()), 4),
            "user_frac_norm_along_common_max": round(
                float(cos_common.max()), 4),
            "residual_norm_mean": round(float(res_norms.mean()), 2),
            "total_norm_mean": round(float(tot_norms.mean()), 2),
            "var_frac_in_common": round(var_common, 4),
        }
        # center user vectors; recompute inter-user cosine on the residual
        S0u = unit(S0)
        off0 = S0u @ S0u.T
        off0 = off0[~np.eye(U, dtype=bool)]
        common[f"layer{l}"]["residual_inter_user_cos_mean"] = round(
            float(off0.mean()), 4)

    result = {
        "version": "e23_style_vector_analysis_v1",
        "seed": SEED,
        "n_users": U,
        "n_layers": L,
        "layers": layers,
        "n_splits": N_SPLITS,
        "n_shuffles": N_SHUFFLES,
        "A_activation_structure": a_struct,
        "B1_split_half_reliability": rel,
        "B2_shuffle_pair_null": null,
        "B3_identity_retrieval": retr,
        "C_method_agreement": meth_agree,
        "C2_common_vs_individual": common,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(ANALYSIS_JSON, "w") as f:
        json.dump(result, f, indent=1)
    log("=== summary ===")
    for l in layers:
        r = rel[f"layer{l}"]
        n = null[f"layer{l}"]
        g = retr[f"layer{l}"]
        c2 = common[f"layer{l}"]
        log(f"L{l}: split-half {r['split_half_cos_mean']:.3f} "
            f"(noise-null {r['noise_null_cos']:.3f}, inter-user "
            f"{r['inter_user_cos_mean']:.3f}, sep {r['separation']:+.3f}) | "
            f"cross-user cos {n['cos_true_vs_cross_user']:.3f} "
            f"(norm ratio {n['norm_ratio_true_over_cross']:.2f}) | "
            f"identity top1 {g['top1']:.3f} top5 {g['top5']:.3f} "
            f"(chance {g['chance_top1']:.3f}/{g['chance_top5']:.3f}) | "
            f"common-var {c2['var_frac_in_common']:.2f} "
            f"resid-inter-user-cos {c2['residual_inter_user_cos_mean']:.3f}")
    log(f"DONE — wrote {ANALYSIS_JSON} ({result['runtime_sec']}s)")


if __name__ == "__main__":
    main()
