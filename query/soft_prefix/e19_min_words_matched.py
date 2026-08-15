#!/usr/bin/env python3
"""E19: matched min-words sweep — product-split halves, product-type-matched
negatives, residualized syntax features.

Replaces the sentence-alternating + random-negative design:

  positive : sim(profile_U, audit_U)   profile/audit = disjoint product halves
  negative : sim(profile_U, audit_V)   V commented on an overlapping product
                                        type, same word-count bin, similar
                                        rating distribution
  residual : per-sentence syntax features regressed on [1, sent_len,
             review_len, rating, product_type_onehot]; residuals pooled into
             user-half vectors (20 residual features + 10 opener histogram)
  stats    : per threshold T: delta-z, user-clustered bootstrap 95% CI,
             user-label permutation p (within matched set), same-user AUC,
             BH-FDR across thresholds, two product-split seeds must both pass
  pass     : CI_low > 0 AND p_adj < 0.01 AND dZ >= 0.2 AND AUC >= 0.60
             AND reproducible under both split seeds
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from user_stat_vector import FEATURES20, OPENER_CLASSES, opener_class_of  # noqa: E402
from extract_clause_features_single_query import load_spacy_model, extract_clause_features_from_doc  # noqa: E402

REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"
OUT = REPO_ROOT / "result" / "e19_matched_sweep.json"
SEED = 42
SPLIT_SEEDS = [42, 7]
N_PER_BIN = 300
BINS = [(48, 72), (80, 120), (120, 180), (192, 288), (360, 540), (720, 1080)]  # around T=60..900
N_MATCHED = 3
MIN_SENTS_PER_HALF = 3
MIN_ASINS = 3
N_PERM = 999
N_BOOT = 999
N_TYPE_DUMMY = 100
FEATS = FEATURES20


def build_vec(resid: list[np.ndarray], openers: list[int]) -> np.ndarray:
    fmean = np.mean(resid, axis=0) if resid else np.zeros(len(FEATS), dtype=np.float32)
    n = float(np.linalg.norm(fmean))
    if n > 1e-12:
        fmean = fmean / n
    ohist = np.zeros(len(OPENER_CLASSES), dtype=np.float32)
    if openers:
        for c in openers:
            ohist[c] += 1
        ohist = ohist / len(openers)
    return np.concatenate([fmean, ohist]).astype(np.float32)


def fisher_z(r: float) -> float:
    r = max(min(r, 0.9999), -0.9999)
    return 0.5 * float(np.log((1 + r) / (1 - r)))


def rho(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    rng = np.random.default_rng(SEED)

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
    print(f"users scanned: {len(wc)}", flush=True)

    # bin sampling by true total words
    bin_users: dict[tuple, list[str]] = {}
    for (lo, hi) in BINS:
        cands = [u for u, w in wc.items() if lo <= w < hi]
        rng.shuffle(cands)
        bin_users[(lo, hi)] = cands[:N_PER_BIN]
        print(f"bin [{lo},{hi}): candidates={len(cands)} sampled={N_PER_BIN}", flush=True)
    pool = sorted({u for v in bin_users.values() for u in v})
    poolset = set(pool)

    # pass 2: extract reviews (user, asin, rating, text)
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
                reviews[u].append((d.get("parent_asin") or d.get("asin"), d.get("rating") or 0.0, t))
    # keep users with >= MIN_ASINS distinct products
    for u in list(reviews):
        if len({r[0] for r in reviews[u]}) < MIN_ASINS:
            del reviews[u]
    print(f"users with >= {MIN_ASINS} products: {len(reviews)}", flush=True)

    # meta: asin -> product type (categories[0:2] or main_category)
    ptype: dict[str, str] = {}
    with gzip.open(META, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            a = d.get("parent_asin")
            if not a:
                continue
            cats = d.get("categories") or []
            ptype[a] = "/".join(cats[:2]) if cats else (d.get("main_category") or "Unknown")
    print(f"meta product types: {len(ptype)}", flush=True)

    # one spaCy pass: per-sentence features + covars
    nlp = load_spacy_model()
    flat = [(u, a, r, t[:2000]) for u in reviews for (a, r, t) in reviews[u]]
    rows: list[dict] = []
    for (u, a, rating, t), doc in zip(flat, nlp.pipe([f[3] for f in flat], batch_size=256)):
        rev_len = len(t.split())
        for sent in doc.sents:
            toks = [tk for tk in sent if not tk.is_punct and not tk.is_space]
            if len(toks) < 3:
                continue
            try:
                ex = extract_clause_features_from_doc(sent, sent.text)
                feat = np.asarray([float(ex.get(k, 0.0)) for k in FEATS], dtype=np.float32)
            except Exception:
                continue
            op = OPENER_CLASSES.index(opener_class_of(toks[0].text))
            rows.append({"u": u, "a": a, "rating": rating, "sent_len": len(toks),
                         "rev_len": rev_len, "feat": feat, "opener": op})
    print(f"sentences parsed: {len(rows)}", flush=True)

    # product-type one-hot (top types by frequency)
    ty = defaultdict(int)
    for r in rows:
        ty[ptype.get(r["a"], "Unknown")] += 1
    top_types = [t for t, _ in sorted(ty.items(), key=lambda x: -x[1])[:N_TYPE_DUMMY]]
    type_idx = {t: i for i, t in enumerate(top_types)}
    print(f"top product types used: {len(top_types)} (coverage {sum(ty[t] for t in top_types) / max(len(rows), 1):.2f})", flush=True)

    # residualize: feat ~ 1 + sent_len + rev_len + rating + type_onehot
    X = []
    Y = []
    for r in rows:
        x = [1.0, r["sent_len"], r["rev_len"], r["rating"]]
        one = np.zeros(len(top_types))
        ti = type_idx.get(ptype.get(r["a"], "Unknown"))
        if ti is not None:
            one[ti] = 1.0
        x.extend(one.tolist())
        X.append(x)
        Y.append(r["feat"])
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid_all = Y - X @ beta
    print(f"residualized {len(FEATS)} features on {X.shape[1]} covars", flush=True)
    for i, r in enumerate(rows):
        r["resid"] = resid_all[i]

    # split-half by PRODUCT with two seeds; per (T, seed): stats
    results = []
    for (lo, hi) in BINS:
        T = (lo + hi) // 2
        users = sorted(set(bin_users[(lo, hi)]) & set(reviews.keys()))
        for seed in SPLIT_SEEDS:
            rngs = np.random.default_rng(seed)
            same, diff = [], []
            ok_users = 0
            for u in users:
                # product halves
                asins = sorted({r["a"] for r in rows if r["u"] == u})
                rngs.shuffle(asins)
                n2 = len(asins) // 2
                a_set, b_set = set(asins[:n2]), set(asins[n2:])
                def half_vec(aset):
                    fr = [r["resid"] for r in rows if r["u"] == u and r["a"] in aset]
                    op = [r["opener"] for r in rows if r["u"] == u and r["a"] in aset]
                    if len(fr) < MIN_SENTS_PER_HALF:
                        return None
                    return build_vec(fr, op)
                va, vb = half_vec(a_set), half_vec(b_set)
                if va is None or vb is None:
                    continue
                ok_users += 1
                # U's audit-half mean rating bucket (floor ±1 star)
                u_ratings = [r["rating"] for r in rows if r["u"] == u and r["a"] in b_set]
                if not u_ratings:
                    continue
                u_bucket = int(np.floor(np.mean(u_ratings)))
                same.append(fisher_z(rho(va, vb)))
                # matched negatives: same bin, product-type overlap with U's
                # audit half, AND same rating bucket on V's audit half
                u_types = {ptype.get(a, "Unknown") for a in b_set}
                # precompute V's audit half + rating bucket (cache per (U, seed))
                v_audit: dict[str, tuple[set, int]] = {}
                for v in users:
                    if v == u:
                        continue
                    asins_v = sorted({r4["a"] for r4 in rows if r4["u"] == v})
                    n2v = len(asins_v) // 2
                    v_b = set(asins_v[n2v:])
                    v_ratings = [r["rating"] for r in rows if r["u"] == v and r["a"] in v_b]
                    if not v_ratings:
                        continue
                    v_audit[v] = (v_b, int(np.floor(np.mean(v_ratings))))
                cands = []
                for v, (v_b, v_bucket) in v_audit.items():
                    if v_bucket != u_bucket:
                        continue
                    v_types = {ptype.get(a, "Unknown") for a in v_b}
                    if v_types & u_types:
                        cands.append(v)
                if not cands:
                    continue
                rngs.shuffle(cands)
                for v in cands[:N_MATCHED]:
                    v_b, _ = v_audit[v]
                    vv = half_vec(v_b)
                    if vv is not None:
                        diff.append(fisher_z(rho(va, vv)))
            if len(same) < 20 or len(diff) < 20:
                results.append({"T": T, "seed": seed, "n_users": ok_users, "n_same": len(same),
                                "n_diff": len(diff), "passed": False, "note": "insufficient pairs"})
                print(f"T={T} seed={seed}: only {ok_users} users / {len(same)} same-pairs", flush=True)
                continue
            same_a, diff_a = np.asarray(same), np.asarray(diff)
            obs = float(same_a.mean() - diff_a.mean())
            rngb = np.random.default_rng(seed * 1000 + 1)
            boot = []
            n_u = len(same_a)
            for _ in range(N_BOOT):
                ids = rngb.integers(0, n_u, size=n_u)
                boot.append(same_a[ids].mean() - diff_a[rngb.integers(0, len(diff_a), size=n_u)].mean())
            lo_, hi_ = np.percentile(boot, [2.5, 97.5])
            poolz = np.concatenate([same_a, diff_a])
            cnt = 0
            for _ in range(N_PERM):
                pv = rngb.permutation(poolz)
                if pv[:n_u].mean() - pv[n_u:].mean() >= obs:
                    cnt += 1
            p = (cnt + 1) / (N_PERM + 1)
            # same-user AUC
            auc = 0.0
            ncmp = 0
            for s in same_a:
                auc += float((diff_a < s).sum())
                ncmp += len(diff_a)
            auc = auc / max(ncmp, 1)
            d_z = obs
            results.append({"T": T, "seed": seed, "n_users": ok_users, "n_same": len(same_a),
                            "n_diff": len(diff_a), "mean_delta_z": round(obs, 4),
                            "ci95": [round(lo_, 4), round(hi_, 4)], "p": round(p, 4),
                            "auc": round(auc, 4),
                            "passed_raw": bool(lo_ > 0 and p < 0.01 and d_z >= 0.2 and auc >= 0.6)})
            print(f"T={T} seed={seed}: users={ok_users} dZ={obs:.3f} CI=[{lo_:.3f},{hi_:.3f}] "
                  f"p={p:.4f} auc={auc:.3f}", flush=True)

    # BH-FDR across thresholds per seed, then final gate (both seeds)
    for seed in SPLIT_SEEDS:
        rs = [r for r in results if r["seed"] == seed and "p" in r]
        ps = np.asarray([r["p"] for r in rs])
        m = len(ps)
        order = np.argsort(ps)
        ranked = np.zeros(m)
        for i, j in enumerate(order):
            ranked[j] = ps[j] * m / (i + 1)
        ranked = np.minimum.accumulate(ranked[order][np.argsort(np.argsort(ps))])
        for r, r_adj in zip(rs, ranked):
            r["p_adj"] = round(float(r_adj), 4)
            r["passed"] = bool(r["ci95"][0] > 0 and r_adj < 0.01 and r["mean_delta_z"] >= 0.2
                               and r["auc"] >= 0.6)
    # both-seed gate
    for T in sorted({r["T"] for r in results}):
        rs = [r for r in results if r["T"] == T]
        both = all(r.get("passed", False) for r in rs) and len(rs) == len(SPLIT_SEEDS)
        print(f"T={T}: both-seed pass = {both}")
        for r in rs:
            r["both_seed_pass"] = both

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"question": "min TRUE per-user review words with product-split halves, "
                           "product-type-matched negatives, residualized features",
               "design": "positive=sim(profile_U,audit_U) [disjoint product halves]; "
                         "negative=sim(profile_U,audit_V) with V sharing product type, same word bin; "
                         "residuals = syntax_feat ~ sent_len+rev_len+rating+type_onehot; "
                         "gate: CI_low>0, p_adj<0.01 (BH-FDR per seed), dZ>=0.2, AUC>=0.60, "
                         "two split seeds both pass",
               "bins": [[b[0], b[1]] for b in BINS], "n_per_bin": N_PER_BIN,
               "n_matched": N_MATCHED, "split_seeds": SPLIT_SEEDS,
               "top_type_dummies": len(top_types), "type_coverage": round(sum(ty[t] for t in top_types) / max(len(rows), 1), 3),
               "results": results,
               "min_passed": next((r for r in results if r.get("both_seed_pass")), None)},
              open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
