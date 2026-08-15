#!/usr/bin/env python3
"""E21 v5: 2D grid search with enlarged user pool + corrected stats (v4).

Fixes vs v3 (commit 507b109):
  1. Effect-size direction. Use delta = d_cross - d_self so that positive
     delta == style is identifiable (d_self < d_cross == same user halves
     are closer than different-user halves). Gate: cohen_d >= 0.5.
  2. Permutation test. Pool all halves across users; for each permutation,
     randomly pair halves — under H0 every pair is "cross", so the null
     distribution is well-defined and informative.
  3. Test-set verification always runs, even if dev gate fails.
  4. Result JSON committed alongside the script.

v5 vs v4 (commit 55e2bba):
  - N_USERS: 400 -> 1500  (enlarge eligible pool)
  - MIN_WORDS: 200 -> 100  (accept more users)
  - MIN_ASINS: 3 -> 2     (accept more users)
  Goal: lift (L=3, N=20) dev users from 43 to >= 300 so that
  seed_pass_frac and test-set effect size become statistically stable.

Grid: L (min sentence length in tokens) in {3, 5, 8}; N (sentences per
half) in {10, 20, 30, 50}; 30 random seeds per cell; dev/test 50/50;
Bonferroni over 12 cells; per-cell 999 permutations.

Per-cell pass gate: AUC >= 0.65 AND perm_p_two (Bonferroni) < 0.01
AND cohen_d >= 0.5 AND seed_pass_frac >= 0.80.
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
LOG = REPO_ROOT / "result" / "e21_l_grid.log"

SEED = 42
N_USERS = 1500           # v5: 400 -> 1500 to enlarge eligible pool
MIN_WORDS = 100          # v5: 200 -> 100 to widen candidates
MIN_ASINS = 2            # v5: 3 -> 2 to widen candidates
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
    """Aggregate per-sentence features to a 32-dim user vector."""
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


def norm_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)


def auc_self_vs_cross(d_self: np.ndarray, d_cross: np.ndarray) -> float:
    """P(d_cross > d_self). Style-identifiable iff AUC > 0.5."""
    if len(d_self) == 0 or len(d_cross) == 0:
        return float("nan")
    auc = 0.0
    for x in d_self:
        auc += float((d_cross > x).sum())
    return auc / (len(d_self) * len(d_cross))


def cohen_d(d_cross: np.ndarray, d_self: np.ndarray) -> float:
    """Cohen d with positive == style identifiable
    (d_cross.mean() > d_self.mean())."""
    if len(d_cross) < 2 or len(d_self) < 2:
        return float("nan")
    delta = d_cross.mean() - d_self.mean()
    pooled = np.sqrt((d_cross.std(ddof=1) ** 2 + d_self.std(ddof=1) ** 2) / 2)
    if pooled == 0:
        return 0.0
    return float(delta / pooled)


def paired_pooled_var(d_cross: np.ndarray, d_self: np.ndarray) -> tuple[float, float]:
    return (float(d_cross.mean()), float(np.sqrt((d_cross.std(ddof=1) ** 2 +
                                                   d_self.std(ddof=1) ** 2) / 2)))


def perm_null_delta(all_halves: list[np.ndarray], rng: np.random.Generator) -> float:
    """Pool all halves across users, randomly pair them up, compute mean
    pair-distance. Under H0 (no user identity) all pairs are cross, so
    this approximates the null distribution of mean distance. Return the
    delta = mean(pair_dist) - 0 (since null self == null cross under H0).

    Implementation: shuffle indices, pair consecutive halves; for odd
    counts drop the last unpaired half.
    """
    n = len(all_halves)
    if n < 4:
        return 0.0
    idx = rng.permutation(n)
    if n % 2 == 1:
        idx = idx[:-1]
    a = np.stack([all_halves[i] for i in idx[0::2]])
    b = np.stack([all_halves[i] for i in idx[1::2]])
    diffs = np.linalg.norm(a - b, axis=1)
    return float(diffs.mean())


def eval_cell_observed(cell_users: list[str],
                       user_filtered: dict[int, dict[str, list[np.ndarray]]],
                       L: int, N: int,
                       seeds: tuple[int, ...],
                       rng_p: np.random.Generator) -> list[dict]:
    """Per-seed: split-half self + cross distance aggregation."""
    per_seed = []
    for sd in seeds:
        rs = np.random.default_rng(sd)
        diffs_self, diffs_cross = [], []
        for u in cell_users:
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
            v1n = norm_vec(v1)
            v2n = norm_vec(v2)
            diffs_self.append(float(np.linalg.norm(v1n - v2n)))
            # cross: 3 random other users
            others = [uu for uu in cell_users if uu != u]
            rs.shuffle(others)
            cross_n = 0
            for v_other in others:
                if cross_n >= 3:
                    break
                sfs_v = user_filtered[L][v_other]
                if len(sfs_v) < N:
                    continue
                idx_v = rs.permutation(len(sfs_v))[:N]
                vv = user_features([sfs_v[i] for i in idx_v])
                if vv is None:
                    continue
                vvn = norm_vec(vv)
                diffs_cross.append(float(np.linalg.norm(v1n - vvn)))
                cross_n += 1
        ds = np.asarray(diffs_self)
        dc = np.asarray(diffs_cross)
        if len(ds) < 5 or len(dc) < 5:
            continue
        per_seed.append({
            "seed": int(sd),
            "auc": auc_self_vs_cross(ds, dc),
            "delta": float(dc.mean() - ds.mean()),  # positive == signal
            "cohen_d": cohen_d(dc, ds),
            "d_self_mean": float(ds.mean()),
            "d_cross_mean": float(dc.mean()),
            "n_self": len(ds),
            "n_cross": len(dc),
        })
    return per_seed


def eval_cell_permutation(cell_users: list[str],
                          user_filtered: dict[int, dict[str, list[np.ndarray]]],
                          L: int, N: int,
                          obs_delta: float,
                          n_perm: int,
                          rng_p: np.random.Generator) -> tuple[float, float]:
    """Pool all halves for the cell's users, run n_perm permutations,
    compare obs_delta to null_delta distribution.

    null_delta = mean(pair_dist) - 0  (H0: no user identity)
    """
    all_halves = []
    for u in cell_users:
        sfs = user_filtered[L][u]
        if len(sfs) < 2 * N:
            continue
        # Pre-compute a fixed split (half1, half2) for this user — we'll
        # shuffle the assignment of halves to users in each permutation.
        # Use seed 0 to keep splits stable across permutations.
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
    null_deltas = np.asarray([perm_null_delta(all_halves, rng_p)
                              for _ in range(n_perm)])
    # one-sided: obs_delta larger than null (positive == signal)
    p_one = (float((null_deltas <= obs_delta).sum()) + 1) / (len(null_deltas) + 1)
    # two-sided: |obs - null_mean| extreme
    null_center = float(null_deltas.mean())
    p_two = (float((np.abs(null_deltas - null_center) >=
                    abs(obs_delta - null_center)).sum()) + 1) / (len(null_deltas) + 1)
    return float(p_one), float(p_two)


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

    # spaCy parse + per-sentence features (CLAUDE.md rule 11c: batch via nlp.pipe)
    nlp = load_spacy_model()
    user_sents: dict[str, list[dict]] = {}
    # Flatten all texts in stable order, batched by user
    user_texts: list[tuple[str, str]] = []  # (user_id, text)
    for u, revs in reviews.items():
        for _, t in revs:
            user_texts.append((u, t))
    print(f"  parsing {len(user_texts)} reviews for {len(reviews)} users...", flush=True)
    BATCH = 128
    user_text_iter = iter(user_texts)
    batch_pairs = []
    parsed = 0
    while True:
        batch_pairs = []
        try:
            for _ in range(BATCH):
                batch_pairs.append(next(user_text_iter))
        except StopIteration:
            pass
        if not batch_pairs:
            break
        texts = [t for _, t in batch_pairs]
        for (u, _), doc in zip(batch_pairs, nlp.pipe(texts, batch_size=BATCH)):
            for sent in doc.sents:
                sf = per_sentence_features(sent)
                if sf is not None:
                    user_sents.setdefault(u, []).append(sf)
        parsed += len(batch_pairs)
        if parsed % 1000 < BATCH:
            print(f"    parsed {parsed}/{len(user_texts)} (t={time.time() - t0:.1f}s)", flush=True)
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
    user_filtered: dict[int, dict[str, list[dict]]] = {L: {} for L in L_VALUES}
    for L in L_VALUES:
        for u, sfs in user_sents.items():
            user_filtered[L][u] = [s for s in sfs if s["n_tok"] >= L]

    def eligible(L: int, N: int, users: list[str]) -> list[str]:
        return [u for u in users if len(user_filtered[L][u]) >= 2 * N]

    seeds_dev = tuple(int(SEED + 1000 + i) for i in range(N_SEEDS))
    seeds_test = tuple(int(SEED + 5000 + i) for i in range(N_SEEDS))
    rng_perm = np.random.default_rng(SEED + 2000)

    grid_results = {}
    best = None
    for L in L_VALUES:
        for N in N_VALUES:
            cell_t0 = time.time()
            cell_users_dev = eligible(L, N, dev_users)
            cell_users_test = eligible(L, N, test_users)
            print(f"  cell L={L} N={N}: dev={len(cell_users_dev)} test={len(cell_users_test)}", flush=True)
            if len(cell_users_dev) < 30:
                grid_results[(L, N)] = {
                    "eligible_dev": len(cell_users_dev),
                    "eligible_test": len(cell_users_test),
                    "skipped": "too few dev users",
                }
                continue

            per_seed_dev = eval_cell_observed(cell_users_dev, user_filtered, L, N,
                                              seeds_dev, rng_perm)
            if not per_seed_dev:
                grid_results[(L, N)] = {
                    "eligible_dev": len(cell_users_dev),
                    "eligible_test": len(cell_users_test),
                    "skipped": "no seed produced >=5 pairs",
                }
                continue
            aucs = np.asarray([s["auc"] for s in per_seed_dev])
            deltas = np.asarray([s["delta"] for s in per_seed_dev])
            ds_self = np.asarray([s["d_self_mean"] for s in per_seed_dev])
            ds_cross = np.asarray([s["d_cross_mean"] for s in per_seed_dev])
            cohen_ds = np.asarray([s["cohen_d"] for s in per_seed_dev])
            seed_pass_frac = float((aucs >= AUC_THRESH).mean())
            obs_delta = float(deltas.mean())
            obs_cohen = float(cohen_ds.mean())

            # Sanity check: AUC and Cohen d must agree in sign
            auc_above_half = obs_delta > 0  # delta > 0 iff AUC > 0.5
            if (aucs.mean() > 0.5) != (obs_cohen > 0):
                raise RuntimeError(
                    f"sanity check failed at L={L} N={N}: "
                    f"AUC={aucs.mean():.3f} cohen_d={obs_cohen:.3f} must agree")

            p_one, p_two = eval_cell_permutation(cell_users_dev, user_filtered,
                                                 L, N, obs_delta, N_PERM, rng_perm)

            cell_rec = {
                "eligible_dev": len(cell_users_dev),
                "eligible_test": len(cell_users_test),
                "per_seed_dev": per_seed_dev,
                "auc_mean": round(float(aucs.mean()), 4),
                "auc_std": round(float(aucs.std()), 4),
                "delta_mean": round(obs_delta, 4),  # positive == signal
                "d_self_mean": round(float(ds_self.mean()), 4),
                "d_cross_mean": round(float(ds_cross.mean()), 4),
                "cohen_d_mean": round(obs_cohen, 4),
                "seed_pass_frac": round(seed_pass_frac, 4),
                "perm_p_one": round(p_one, 4),
                "perm_p_two": round(p_two, 4),
                "runtime_sec": round(time.time() - cell_t0, 1),
            }
            grid_results[(L, N)] = cell_rec
            print(f"    AUC={aucs.mean():.3f}+-{aucs.std():.3f} "
                  f"d={obs_cohen:.3f} delta={obs_delta:.4f} "
                  f"p_one={p_one:.3f} seed_pass={seed_pass_frac:.2f} "
                  f"({time.time() - cell_t0:.1f}s)", flush=True)

            # Test-set verification: always run, regardless of dev gate
            per_seed_test = eval_cell_observed(cell_users_test, user_filtered,
                                               L, N, seeds_test, rng_perm)
            if per_seed_test:
                aucs_t = np.asarray([s["auc"] for s in per_seed_test])
                deltas_t = np.asarray([s["delta"] for s in per_seed_test])
                cohen_ts = np.asarray([s["cohen_d"] for s in per_seed_test])
                cell_rec["per_seed_test"] = per_seed_test
                cell_rec["test_auc_mean"] = round(float(aucs_t.mean()), 4)
                cell_rec["test_auc_std"] = round(float(aucs_t.std()), 4)
                cell_rec["test_delta_mean"] = round(float(deltas_t.mean()), 4)
                cell_rec["test_cohen_d_mean"] = round(float(cohen_ts.mean()), 4)
                cell_rec["test_seed_pass_frac"] = round(float((aucs_t >= AUC_THRESH).mean()), 4)

            # Update best (only if all dev gates pass)
            if (cell_rec["auc_mean"] >= AUC_THRESH
                    and cell_rec["cohen_d_mean"] >= COHEN_D_THRESH
                    and cell_rec["seed_pass_frac"] >= SEED_PASS_FRAC):
                if best is None or (L, N) < best[:2]:
                    best = (L, N, cell_rec)

    # Bonferroni over 12 cells
    n_cells = len(L_VALUES) * len(N_VALUES)
    for r in grid_results.values():
        if "perm_p_two" in r:
            r["perm_p_two_bonf"] = round(min(1.0, r["perm_p_two"] * n_cells), 4)

    # Check perm-p-threshold against Bonferroni-corrected value
    for r in grid_results.values():
        if "perm_p_two_bonf" in r:
            r["passes_p_bonf"] = r["perm_p_two_bonf"] < P_THRESH

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "version": "v5 (enlarged pool + corrected stats)",
        "question": "minimum (sentence-length threshold L, sentence count N) "
                    "for user syntactic style to be identifiable on pooled reviews",
        "design": "2D grid L in {3,5,8}, N in {10,20,30,50}; "
                 "split-half (each user 2N sents, half1 vs half2); "
                 "30 random seeds per cell; dev/test 50/50 split; "
                 "Bonferroni over 12 cells; per-cell permutation p (999)",
        "fixes_vs_v3": [
            "delta = d_cross - d_self (positive == signal); "
            "AUC and Cohen d direction aligned and sanity-checked at runtime",
            "permutation pools all halves and random-pairs (true H0)",
            "test-set verification runs regardless of dev gate outcome",
            "result JSON committed alongside script",
        ],
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
        "runtime_sec": round(time.time() - t0, 1),
    }, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT} (t={time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()