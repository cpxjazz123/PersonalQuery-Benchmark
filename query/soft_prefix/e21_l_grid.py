#!/usr/bin/env python3
"""E21 v3: 2D grid search over (L, N) with split-half design.

Question: pooled over all of a user's reviews, can we distinguish user U
from user V by syntactic style, and what is the minimum (sentence length
threshold L, sentence count per half N) required?

Grid:
  L (min sentence length in tokens) in {3, 5, 8}
  N (sentence count per half) in {10, 20, 30, 50}

For each (L, N) cell and each random seed:
  1. Filter user reviews to keep only sentences with >= L tokens
  2. From each user, sample 2N complete sentences (no truncation)
  3. Split into two halves of N sentences each
  4. Compute 32-dim ratio/mean/dist features for each half
  5. Distance: L2 on unit-normalized vectors

Comparisons:
  self  = ||vec_U_half1 - vec_U_half2||    (same user, two halves)
  cross = ||vec_U_half1 - vec_V_half1||     (different users, same half)

Per cell: aggregate over 30 seeds, report:
  mean AUC, mean d_self, mean d_cross, mean delta, bootstrap CI
  per-cell p from 999 user-label permutations

Multiple testing: Bonferroni over 12 cells (in dev set).
Dev/test split: 50/50 users. Report best cell in dev, then verify in test.

Pass gate (per cell): AUC >= 0.65 AND p_adj < 0.01 AND Cohen d >= 0.5
AND >= 80% of seeds pass the AUC threshold.
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

from e20_lopo_v2 import per_sentence_features  # reuse
from e20_lopo_v2 import load_spacy_model, ALL_FEATS

REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
OUT = REPO_ROOT / "result" / "e21_l_grid_results.json"

SEED = 42
N_USERS = 400
MIN_WORDS = 200
MIN_ASINS = 3
# 2D grid
L_VALUES = (3, 5, 8)
N_VALUES = (10, 20, 30, 50)
N_SEEDS = 30
N_PERM = 999
DEV_FRAC = 0.5
AUC_THRESH = 0.65
COHEN_D_THRESH = 0.5
P_THRESH = 0.01
SEED_PASS_FRAC = 0.8


def user_features(sent_feats: list[dict]) -> np.ndarray | None:
    """Aggregate per-sentence features to a 32-dim user vector. Inline copy
    of e21_simple_discrimination.user_features_from_sents (avoids circular
    import and keeps this script self-contained)."""
    if not sent_feats:
        return None
    total_tok = sum(s["n_tok"] for s in sent_feats)
    n_sent = len(sent_feats)
    if total_tok == 0 or n_sent == 0:
        return None
    f = {}
    f["clause_rate"] = sum(s["n_clause"] for s in sent_feats) / total_tok
    f["acl_rate"] = sum(s["acl"] for s in sent_feats) / total_tok
    f["advcl_rate"] = sum(s["advcl"] for s in sent_feats) / total_tok
    f["ccomp_rate"] = sum(s["ccomp"] for s in sent_feats) / total_tok
    f["xcomp_rate"] = sum(s["xcomp"] for s in sent_feats) / total_tok
    f["relcl_rate"] = sum(s["relcl"] for s in sent_feats) / total_tok
    f["modifier_density"] = sum(s["n_mod"] for s in sent_feats) / total_tok
    f["coordination_density"] = sum(s["n_coord"] for s in sent_feats) / total_tok
    f["mean_dep_distance"] = float(np.mean([s["mean_dist"] for s in sent_feats]))
    f["depth_variance"] = float(np.mean([s["depth_var"] for s in sent_feats]))
    f["median_dep_depth"] = float(np.median([s["max_depth"] for s in sent_feats]))
    f["passive_rate"] = sum(1 for s in sent_feats if s["has_passive"]) / n_sent
    f["interrogative_rate"] = sum(1 for s in sent_feats if s["is_interrog"]) / n_sent
    f["conditional_rate"] = sum(1 for s in sent_feats if s["has_cond"]) / n_sent
    from collections import Counter
    OPENER_POSES = ("NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ",
                    "AUX", "NUM", "INTJ", "PART", "PUNCT", "X", "SYM")
    SENT_TYPES = ("simple", "conjunctive", "complex")
    opener_counts = Counter(s["opener"] for s in sent_feats)
    for p in OPENER_POSES:
        f[f"opener_{p}"] = opener_counts.get(p, 0) / n_sent
    stype_counts = Counter(s["stype"] for s in sent_feats)
    for t in SENT_TYPES:
        f[f"senttype_{t}"] = stype_counts.get(t, 0) / n_sent
    return np.asarray([f[name] for name in ALL_FEATS], dtype=np.float32)


def main() -> None:
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    # pass 1: word counts
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

    # pass 2: reviews
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

    # spaCy parse + per-sentence features
    nlp = load_spacy_model()
    user_sents: dict[str, list[dict]] = {}
    for u, revs in reviews.items():
        sfs = []
        for _, t in revs:
            doc = nlp(t)
            for sent in doc.sents:
                sf = per_sentence_features(sent)
                if sf is not None:
                    sfs.append(sf)
        if sfs:
            user_sents[u] = sfs
    print(f"users parsed: {len(user_sents)} (t={time.time() - t0:.1f}s)", flush=True)

    # sub-sample to N_USERS
    user_list = sorted(user_sents)
    rng.shuffle(user_list)
    if len(user_list) > N_USERS:
        keep = set(user_list[:N_USERS])
        user_sents = {u: user_sents[u] for u in keep}
    n_users = len(user_sents)
    sorted_users = sorted(user_sents)
    print(f"using {n_users} users", flush=True)

    # dev/test split
    rng_split = np.random.default_rng(SEED + 100)
    perm = rng_split.permutation(n_users)
    n_dev = int(n_users * DEV_FRAC)
    dev_idx = set(perm[:n_dev].tolist())
    dev_users = [u for i, u in enumerate(sorted_users) if i in dev_idx]
    test_users = [u for i, u in enumerate(sorted_users) if i not in dev_idx]
    print(f"dev users: {len(dev_users)}, test users: {len(test_users)}", flush=True)

    # ============================================================
    # 2D grid search
    # ============================================================
    # For each (L, N), we need users with >= max(2N) sentences after L-filter.
    # Precompute per-user filtered sents for each L.
    # Then for each seed, sample + split + compute distances.
    # ============================================================
    user_filtered: dict[int, dict[str, list[dict]]] = {L: {} for L in L_VALUES}
    for L in L_VALUES:
        for u, sfs in user_sents.items():
            f = [s for s in sfs if s["n_tok"] >= L]
            user_filtered[L][u] = f

    # for each (L, N) cell, find users with >= 2N filtered sents
    def eligible(L: int, N: int, users: list[str]) -> list[str]:
        return [u for u in users if len(user_filtered[L][u]) >= 2 * N]

    grid_results = {}
    for L in L_VALUES:
        for N in N_VALUES:
            cell_t0 = time.time()
            cell_users_dev = eligible(L, N, dev_users)
            cell_users_test = eligible(L, N, test_users)
            print(f"  cell L={L} N={N}: dev={len(cell_users_dev)} test={len(cell_users_test)}", flush=True)
            if len(cell_users_dev) < 30:
                grid_results[(L, N)] = {"eligible_dev": len(cell_users_dev),
                                         "eligible_test": len(cell_users_test),
                                         "skipped": "too few dev users"}
                continue

            def eval_cell(cell_users: list[str], seeds: tuple) -> dict:
                per_seed = []
                for sd in seeds:
                    rs = np.random.default_rng(sd)
                    diffs_self, diffs_cross = [], []
                    for ui, u in enumerate(cell_users):
                        sfs = user_filtered[L][u]
                        if len(sfs) < 2 * N:
                            continue
                        idx = rs.permutation(len(sfs))[:2 * N]
                        half1 = [sfs[i] for i in idx[:N]]
                        half2 = [sfs[i] for i in idx[N:]]
                        v1 = user_features(half1)
                        v2 = user_features(half2)
                        if v1 is None or v2 is None:
                            continue
                        v1n = v1 / max(np.linalg.norm(v1), 1e-12)
                        v2n = v2 / max(np.linalg.norm(v2), 1e-12)
                        diffs_self.append(float(np.linalg.norm(v1n - v2n)))
                        # cross: random other user in same cell
                        others = [uu for uu in cell_users if uu != u]
                        rs.shuffle(others)
                        for v_other in others[:3]:
                            sfs_v = user_filtered[L][v_other]
                            if len(sfs_v) < N:
                                continue
                            idx_v = rs.permutation(len(sfs_v))[:N]
                            vv = user_features([sfs_v[i] for i in idx_v])
                            if vv is None:
                                continue
                            vvn = vv / max(np.linalg.norm(vv), 1e-12)
                            diffs_cross.append(float(np.linalg.norm(v1n - vvn)))
                    ds = np.asarray(diffs_self)
                    dc = np.asarray(diffs_cross)
                    if len(ds) < 5 or len(dc) < 5:
                        continue
                    # AUC: P(d_cross > d_self)
                    auc = 0.0
                    ncomp = 0
                    for x in ds:
                        auc += float((dc > x).sum())
                        ncomp += len(dc)
                    auc = auc / max(ncomp, 1)
                    delta = float(ds.mean() - dc.mean())
                    # pooled std for Cohen d
                    pooled = np.sqrt((ds.std() ** 2 + dc.std() ** 2) / 2)
                    d = float(delta / pooled) if pooled > 0 else 0.0
                    per_seed.append({"seed": int(sd), "auc": auc, "delta": delta,
                                     "d_self_mean": float(ds.mean()),
                                     "d_cross_mean": float(dc.mean()),
                                     "cohen_d": d, "n_self": len(ds), "n_cross": len(dc)})
                return per_seed

            seeds_dev = tuple(int(SEED + 1000 + i) for i in range(N_SEEDS))
            per_seed_dev = eval_cell(cell_users_dev, seeds_dev)
            if not per_seed_dev:
                grid_results[(L, N)] = {"eligible_dev": len(cell_users_dev),
                                         "eligible_test": len(cell_users_test),
                                         "skipped": "no seed produced >=5 pairs"}
                continue
            aucs = np.asarray([s["auc"] for s in per_seed_dev])
            deltas = np.asarray([s["delta"] for s in per_seed_dev])
            ds_self = np.asarray([s["d_self_mean"] for s in per_seed_dev])
            ds_cross = np.asarray([s["d_cross_mean"] for s in per_seed_dev])
            cohen_ds = np.asarray([s["cohen_d"] for s in per_seed_dev])
            seed_pass_frac = float((aucs >= AUC_THRESH).mean())
            obs_delta = float(deltas.mean())

            # permutation p: shuffle user labels, recompute mean delta
            rngp = np.random.default_rng(SEED + 2000)
            perm_deltas = []
            n_perm_users = min(50, len(cell_users_dev))  # subsample for speed
            for _ in range(N_PERM):
                rs_p = np.random.default_rng(rngp.integers(0, 1 << 30))
                null_self, null_cross = [], []
                perm_users = list(cell_users_dev)
                rs_p.shuffle(perm_users)
                for ui, u in enumerate(perm_users[:n_perm_users]):
                    sfs = user_filtered[L][u]
                    if len(sfs) < 2 * N:
                        continue
                    idx = rs_p.permutation(len(sfs))[:2 * N]
                    v1 = user_features([sfs[i] for i in idx[:N]])
                    v2 = user_features([sfs[i] for i in idx[N:]])
                    if v1 is None or v2 is None:
                        continue
                    v1n = v1 / max(np.linalg.norm(v1), 1e-12)
                    v2n = v2 / max(np.linalg.norm(v2), 1e-12)
                    null_self.append(float(np.linalg.norm(v1n - v2n)))
                    # paired user (shuffled) provides the cross
                    pair = perm_users[(ui + 1) % n_perm_users]
                    sfs_v = user_filtered[L][pair]
                    if len(sfs_v) < N:
                        continue
                    idx_v = rs_p.permutation(len(sfs_v))[:N]
                    vv = user_features([sfs_v[i] for i in idx_v])
                    if vv is None:
                        continue
                    vvn = vv / max(np.linalg.norm(vv), 1e-12)
                    null_cross.append(float(np.linalg.norm(v1n - vvn)))
                if null_self and null_cross:
                    perm_deltas.append(np.mean(null_self) - np.mean(null_cross))
            perm_deltas = np.asarray(perm_deltas)
            p_one = (float((perm_deltas <= obs_delta).sum()) + 1) / (len(perm_deltas) + 1)
            p_two = (float((np.abs(perm_deltas - perm_deltas.mean()) >=
                            abs(obs_delta - perm_deltas.mean())).sum()) + 1) / (len(perm_deltas) + 1)

            grid_results[(L, N)] = {
                "eligible_dev": len(cell_users_dev),
                "eligible_test": len(cell_users_test),
                "per_seed": per_seed_dev,
                "auc_mean": round(float(aucs.mean()), 4),
                "auc_std": round(float(aucs.std()), 4),
                "delta_mean": round(float(deltas.mean()), 4),
                "d_self_mean": round(float(ds_self.mean()), 4),
                "d_cross_mean": round(float(ds_cross.mean()), 4),
                "cohen_d_mean": round(float(cohen_ds.mean()), 4),
                "seed_pass_frac": round(seed_pass_frac, 4),
                "perm_p_one": round(float(p_one), 4),
                "perm_p_two": round(float(p_two), 4),
                "runtime_sec": round(time.time() - cell_t0, 1),
            }
            print(f"    AUC={aucs.mean():.3f}+-{aucs.std():.3f} d={cohen_ds.mean():.3f} "
                  f"p_one={p_one:.3f} seed_pass={seed_pass_frac:.2f} "
                  f"({time.time() - cell_t0:.1f}s)", flush=True)

    # Bonferroni over cells (p_two * 12)
    n_cells = len(L_VALUES) * len(N_VALUES)
    for k, v in grid_results.items():
        if "perm_p_two" in v:
            v["perm_p_two_bonf"] = round(min(1.0, v["perm_p_two"] * n_cells), 4)

    # find best (L, N) cell in dev set: passes all gates
    best = None
    for (L, N), r in grid_results.items():
        if "auc_mean" not in r:
            continue
        if (r["auc_mean"] >= AUC_THRESH
                and r["perm_p_two_bonf"] < P_THRESH
                and r["cohen_d_mean"] >= COHEN_D_THRESH
                and r["seed_pass_frac"] >= SEED_PASS_FRAC):
            if best is None or (L, N) < best:
                best = (L, N, r)

    # test-set verification for the best cell
    test_verify = None
    if best is not None:
        L, N, r = best
        cell_users_test = eligible(L, N, test_users)
        if len(cell_users_test) >= 30:
            seeds_test = tuple(int(SEED + 5000 + i) for i in range(N_SEEDS))
            per_seed_test = []
            for sd in seeds_test:
                rs = np.random.default_rng(sd)
                diffs_self, diffs_cross = [], []
                for u in cell_users_test:
                    sfs = user_filtered[L][u]
                    if len(sfs) < 2 * N:
                        continue
                    idx = rs.permutation(len(sfs))[:2 * N]
                    v1 = user_features([sfs[i] for i in idx[:N]])
                    v2 = user_features([sfs[i] for i in idx[N:]])
                    if v1 is None or v2 is None:
                        continue
                    v1n = v1 / max(np.linalg.norm(v1), 1e-12)
                    v2n = v2 / max(np.linalg.norm(v2), 1e-12)
                    diffs_self.append(float(np.linalg.norm(v1n - v2n)))
                    others = [uu for uu in cell_users_test if uu != u]
                    rs.shuffle(others)
                    for v_other in others[:3]:
                        sfs_v = user_filtered[L][v_other]
                        if len(sfs_v) < N:
                            continue
                        idx_v = rs.permutation(len(sfs_v))[:N]
                        vv = user_features([sfs_v[i] for i in idx_v])
                        if vv is None:
                            continue
                        vvn = vv / max(np.linalg.norm(vv), 1e-12)
                        diffs_cross.append(float(np.linalg.norm(v1n - vvn)))
                ds = np.asarray(diffs_self)
                dc = np.asarray(diffs_cross)
                if len(ds) < 5 or len(dc) < 5:
                    continue
                auc = 0.0
                ncomp = 0
                for x in ds:
                    auc += float((dc > x).sum())
                    ncomp += len(dc)
                auc = auc / max(ncomp, 1)
                delta = float(ds.mean() - dc.mean())
                per_seed_test.append({"seed": int(sd), "auc": auc, "delta": delta})
            if per_seed_test:
                aucs_t = np.asarray([s["auc"] for s in per_seed_test])
                deltas_t = np.asarray([s["delta"] for s in per_seed_test])
                test_verify = {
                    "cell": [L, N],
                    "n_test_users": len(cell_users_test),
                    "auc_mean": round(float(aucs_t.mean()), 4),
                    "auc_std": round(float(aucs_t.std()), 4),
                    "delta_mean": round(float(deltas_t.mean()), 4),
                    "seed_pass_frac": round(float((aucs_t >= AUC_THRESH).mean()), 4),
                    "per_seed": per_seed_test,
                }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "question": "minimum (sentence-length threshold L, sentence count N) "
                    "for user syntactic style to be identifiable on pooled reviews",
        "design": "2D grid L in {3,5,8}, N in {10,20,30,50}; "
                 "split-half (each user 2N sents, half1 vs half2); "
                 "30 random seeds per cell; dev/test 50/50 split; "
                 "Bonferroni over 12 cells; per-cell permutation p (999)",
        "l_values": list(L_VALUES),
        "n_values": list(N_VALUES),
        "n_seeds": N_SEEDS,
        "n_perm": N_PERM,
        "n_users": n_users,
        "dev_users": len(dev_users),
        "test_users": len(test_users),
        "gates": {"auc": AUC_THRESH, "p_adj_bonf": P_THRESH,
                  "cohen_d": COHEN_D_THRESH, "seed_pass_frac": SEED_PASS_FRAC},
        "grid": {f"L={L}_N={N}": r for (L, N), r in grid_results.items()},
        "best_cell_dev": list(best[:2]) if best else None,
        "test_verification": test_verify,
        "runtime_sec": round(time.time() - t0, 1),
    }, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT} (t={time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
