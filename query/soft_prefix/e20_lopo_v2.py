#!/usr/bin/env python3
"""E20 v2: ratio/mean/distribution syntactic features + LOPO + conditional perm.

Replaces E20 v1 (FEATURES20 = absolute counts, unstable with word count)
with ratio/mean features per the #20 comment feedback:

  clause_rate               = total_clauses / total_tokens
  subordinator_type_dist    = {acl,advcl,ccomp,xcomp,relcl} / total_tokens
  modifier_density          = modifiers / total_tokens  (already in extractor)
  coordination_density      = coordinations / total_tokens
  mean_dep_distance         (already mean)
  depth_variance            (already mean)
  median_dep_depth          = per-sentence median of max dep depth
  opener_pos_dist           = histogram of first token's POS, normalized
  sentence_type_dist        = simple/conjunctive/complex ratios
  passive_rate              = fraction of sentences with nsubjpass/auxpass
  conditional_rate          = fraction of sentences containing if/when/unless
  interrogative_rate        = fraction of sentences ending in '?'

Feature stability filter: per user, compute corr(feat, sentence_count);
drop features with |corr| > 0.2. Plus LOPO and conditional permutation
within (product_cluster, rating_bucket).

Hard-coded config; runs with genrec_env python (spacy 3.8.14).
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from user_stat_vector import load_spacy_model  # noqa: E402

REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"
OUT = REPO_ROOT / "result" / "e20_lopo_v2_results.json"
CACHE = REPO_ROOT / "result" / "e20_v2_rows_cache.npz"

SEED = 42
MIN_ASINS = 3
N_USERS = 300
MIN_SENTS_TOTAL = 10           # per-user minimum (sentence count, not word count)
MIN_SENTS_PER_PRODUCT = 3
N_PERM = 99
N_LOPO_BOOT = 199
RATING_BUCKET = 1.0
STABILITY_CORR_THRESH = 0.2    # MVP: drop feats with |corr(feat, sent_count)| > 0.2

# Subordinator dep types for clause counting
CLAUSE_DEPS = ("acl", "advcl", "ccomp", "xcomp", "relcl")
# POS labels for opener distribution (universal tags; small subset for sparsity)
OPENER_POSES = ("NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ",
                "AUX", "NUM", "INTJ", "PART", "PUNCT", "X", "SYM")
# Sentence type buckets
SENT_TYPES = ("simple", "conjunctive", "complex")
# Rate/mean/distribution feature names
RATE_FEATS = [
    "clause_rate",
    "acl_rate", "advcl_rate", "ccomp_rate", "xcomp_rate", "relcl_rate",
    "modifier_density",
    "coordination_density",
    "mean_dep_distance", "depth_variance", "median_dep_depth",
    "passive_rate", "interrogative_rate", "conditional_rate",
]
DIST_FEATS = (
    [f"opener_{p}" for p in OPENER_POSES]
    + [f"senttype_{t}" for t in SENT_TYPES]
)
ALL_FEATS = RATE_FEATS + DIST_FEATS


def per_sentence_features(sent_doc) -> dict:
    """Extract ratio/mean/distribution features at the sentence level.

    Returns a dict of per-sentence primitives. User-level aggregation is done
    separately (sum-based for rates, mean for means, hist for dists).
    """
    toks = [t for t in sent_doc if not t.is_punct and not t.is_space]
    n_tok = len(toks)
    if n_tok < 3:
        return None
    n_clause = sum(1 for t in sent_doc if t.dep_ in CLAUSE_DEPS)
    n_coord = sum(1 for t in sent_doc if t.dep_ in ("conj", "cc"))
    n_mod = sum(1 for t in sent_doc if t.dep_ in (
        "amod", "advmod", "nmod", "appos", "nummod", "poss", "det"))
    # dependency distance
    dists = [abs(t.head.i - t.i) for t in sent_doc if t.head.i != t.i]
    mean_dist = float(np.mean(dists)) if dists else 0.0
    # depth (tree height per token)
    def height(token, memo):
        if token in memo:
            return memo[token]
        children = [c for c in token.children]
        if not children:
            memo[token] = 1
            return 1
        h = 1 + max(height(c, memo) for c in children)
        memo[token] = h
        return h
    memo = {}
    depths = [height(t, memo) for t in sent_doc]
    max_depth = max(depths) if depths else 0
    # depth_variance
    mean_depth = sum(depths) / len(depths) if depths else 0
    depth_var = sum((d - mean_depth) ** 2 for d in depths) / len(depths) if depths else 0
    # opener
    first_real = next((t for t in sent_doc if not t.is_space), None)
    opener = first_real.pos_ if first_real is not None else "X"
    # sentence type
    if n_clause > 0:
        stype = "complex"
    elif n_coord > 0:
        stype = "conjunctive"
    else:
        stype = "simple"
    # passive
    has_passive = any(t.dep_ in ("nsubjpass", "auxpass") for t in sent_doc)
    # interrogative
    is_interrog = sent_doc.text.rstrip().endswith("?")
    # conditional markers
    txt_lower = sent_doc.text.lower()
    has_cond = any(m in txt_lower for m in (" if ", " when ", " unless ", " whenever "))
    return {
        "n_tok": n_tok,
        "n_clause": n_clause,
        "n_coord": n_coord,
        "n_mod": n_mod,
        "mean_dist": mean_dist,
        "max_depth": max_depth,
        "depth_var": depth_var,
        "opener": opener,
        "stype": stype,
        "has_passive": has_passive,
        "is_interrog": is_interrog,
        "has_cond": has_cond,
        "acl": sum(1 for t in sent_doc if t.dep_ == "acl"),
        "advcl": sum(1 for t in sent_doc if t.dep_ == "advcl"),
        "ccomp": sum(1 for t in sent_doc if t.dep_ == "ccomp"),
        "xcomp": sum(1 for t in sent_doc if t.dep_ == "xcomp"),
        "relcl": sum(1 for t in sent_doc if t.dep_ == "relcl"),
    }


def user_feature_vector(sent_feats: list[dict]) -> np.ndarray | None:
    """Aggregate per-sentence features to a fixed-length user vector."""
    if not sent_feats:
        return None
    total_tok = sum(s["n_tok"] for s in sent_feats)
    n_sent = len(sent_feats)
    if total_tok == 0 or n_sent == 0:
        return None
    # rate features
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
    # dist features
    opener_counts = Counter(s["opener"] for s in sent_feats)
    for p in OPENER_POSES:
        f[f"opener_{p}"] = opener_counts.get(p, 0) / n_sent
    stype_counts = Counter(s["stype"] for s in sent_feats)
    for t in SENT_TYPES:
        f[f"senttype_{t}"] = stype_counts.get(t, 0) / n_sent
    return np.asarray([f[name] for name in ALL_FEATS], dtype=np.float32)


def per_product_feature_vector(sent_feats: list[dict]) -> np.ndarray | None:
    """Same as user-level but per (user, product) for LOPO. Includes product
    mean subtraction for within-product standardization (Task 1)."""
    return user_feature_vector(sent_feats)


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

    cands = [u for u, w in wc.items() if w >= 200]
    rng.shuffle(cands)
    cands = cands[:N_USERS * 4]
    poolset = set(cands)
    print(f"word>=200 candidates: {len(cands)}", flush=True)

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
                reviews[u].append((d.get("parent_asin") or d.get("asin"),
                                   d.get("rating") or 0.0, t))
    for u in list(reviews):
        if len({r[0] for r in reviews[u]}) < MIN_ASINS:
            del reviews[u]
    print(f"users with >= {MIN_ASINS} products: {len(reviews)}", flush=True)

    # meta
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

    # spaCy parse + per-sentence feature extraction
    nlp = load_spacy_model()
    flat = [(u, a, r, t[:2000]) for u in reviews for (a, r, t) in reviews[u]]
    print(f"to parse: {len(flat)} reviews", flush=True)
    # store per-sentence features keyed by (u, a)
    up_sentfeats: dict[tuple, list[dict]] = defaultdict(list)
    for (u, a, rating, t), doc in zip(flat, nlp.pipe([f[3] for f in flat], batch_size=256)):
        for sent in doc.sents:
            sf = per_sentence_features(sent)
            if sf is not None:
                up_sentfeats[(u, a)].append(sf)
    print(f"sentences parsed: {sum(len(v) for v in up_sentfeats.values())} "
          f"(t={time.time() - t0:.1f}s)", flush=True)

    # sub-sample users
    user_list = sorted({u for u, _ in up_sentfeats})
    rng.shuffle(user_list)
    if len(user_list) > N_USERS:
        keep = set(user_list[:N_USERS])
        up_sentfeats = {k: v for k, v in up_sentfeats.items() if k[0] in keep}
    print(f"using {len({k[0] for k in up_sentfeats})} users", flush=True)

    # filter: keep users with >= MIN_ASINS products and each product >= MIN_SENTS_PER_PRODUCT
    user_products: dict[str, list[str]] = defaultdict(list)
    for (u, a), sfs in up_sentfeats.items():
        if len(sfs) >= MIN_SENTS_PER_PRODUCT:
            user_products[u].append(a)
    user_products = {u: ps for u, ps in user_products.items() if len(ps) >= MIN_ASINS}
    # also require user total sentences >= MIN_SENTS_TOTAL
    user_total_sents = {u: sum(len(up_sentfeats[(u, a)]) for a in user_products[u])
                        for u in user_products}
    user_products = {u: ps for u, ps in user_products.items()
                     if user_total_sents[u] >= MIN_SENTS_TOTAL}
    print(f"users usable for LOPO (>= {MIN_ASINS} products, >= {MIN_SENTS_TOTAL} sents): "
          f"{len(user_products)}", flush=True)

    # ============================================================
    # Task 1: within-product standardization
    # ============================================================
    # product-level mean vector (over all users' sentences on that product)
    prod_sfs: dict[str, list[dict]] = defaultdict(list)
    for (u, a), sfs in up_sentfeats.items():
        if a in user_products.get(u, []):
            prod_sfs[a].extend(sfs)
    prod_mean: dict[str, np.ndarray] = {}
    for a, sfs in prod_sfs.items():
        v = user_feature_vector(sfs)
        if v is not None:
            prod_mean[a] = v
    print(f"product means: {len(prod_mean)}", flush=True)

    # per-(user,product) residual = user_vec - product_mean
    up_vec: dict[tuple, np.ndarray] = {}
    for (u, a), sfs in up_sentfeats.items():
        if a not in user_products.get(u, []):
            continue
        v = per_product_feature_vector(sfs)
        if v is not None and a in prod_mean:
            up_vec[(u, a)] = v - prod_mean[a]
    print(f"per-(user,product) residual vectors: {len(up_vec)}", flush=True)

    # ============================================================
    # Feature stability: per-user corr(feat, sent_count) across products
    # ============================================================
    # For each user, compute per-product sent_count and feature value; corr.
    user_prod_data: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (u, a), v in up_vec.items():
        n_s = len(up_sentfeats[(u, a)])
        user_prod_data[u].append((n_s, v))
    # stability: corr per feature
    feat_corr: dict[str, float] = {}
    for fi, fname in enumerate(ALL_FEATS):
        corrs = []
        for u, lst in user_prod_data.items():
            if len(lst) < 3:
                continue
            ns = np.asarray([x[0] for x in lst], dtype=np.float32)
            fv = np.asarray([x[1][fi] for x in lst], dtype=np.float32)
            if np.std(ns) == 0 or np.std(fv) == 0:
                continue
            r = float(np.corrcoef(ns, fv)[0, 1])
            if np.isfinite(r):
                corrs.append(r)
        feat_corr[fname] = float(np.mean(corrs)) if corrs else 0.0
    stable_feats = [f for f, c in feat_corr.items() if abs(c) < STABILITY_CORR_THRESH]
    unstable_feats = [f for f, c in feat_corr.items() if abs(c) >= STABILITY_CORR_THRESH]
    print(f"stable features (|corr|<{STABILITY_CORR_THRESH}): {len(stable_feats)}/{len(ALL_FEATS)}", flush=True)
    print(f"  unstable: {unstable_feats}", flush=True)

    # LOPO uses stable features only
    stable_idx = [ALL_FEATS.index(f) for f in stable_feats]
    feat_dim = len(stable_idx)
    # normalize per-feature to unit norm
    feat_norms: dict[str, float] = {}
    for f in stable_feats:
        vals = []
        for u, lst in user_prod_data.items():
            for n_s, v in lst:
                vals.append(v[ALL_FEATS.index(f)])
        vals = np.asarray(vals)
        feat_norms[f] = float(np.std(vals)) if np.std(vals) > 0 else 1.0

    def lopo_vec(u, ps):
        v = np.stack([up_vec[(u, p)][stable_idx] for p in ps], axis=0).mean(axis=0)
        return v / max(np.linalg.norm(v), 1e-12)

    def normalize(v):
        out = v.copy()
        for i, f in enumerate(stable_feats):
            out[i] = out[i] / feat_norms[f]
        return out / max(np.linalg.norm(out), 1e-12)

    # ============================================================
    # Task 3: LOPO
    # ============================================================
    rng_lopo = np.random.default_rng(SEED + 1)
    users_for_lopo = sorted(user_products.keys())
    diffs_self = []
    diffs_cross = []
    for u in users_for_lopo:
        products = sorted(user_products[u])
        held = products[rng_lopo.integers(0, len(products))]
        train_ps = [p for p in products if p != held]
        v_self = lopo_vec(u, train_ps)
        v_held_raw = up_vec[(u, held)][stable_idx]
        v_held = normalize(v_held_raw)
        d_self = float(np.linalg.norm(v_self - v_held))
        diffs_self.append(d_self)
        # cross: 3 random other users with similar #products
        for _ in range(3):
            v_other = rng_lopo.choice([uu for uu in users_for_lopo if uu != u])
            other_ps = sorted(user_products[v_other])
            rng_lopo.shuffle(other_ps)
            train_v = other_ps[:max(len(train_ps), 1)]
            if len(train_v) < MIN_ASINS:
                continue
            v_cross = lopo_vec(v_other, train_v)
            d_cross = float(np.linalg.norm(v_cross - v_held))
            diffs_cross.append(d_cross)
    diffs_self = np.asarray(diffs_self)
    diffs_cross = np.asarray(diffs_cross)
    obs_delta = float(diffs_self.mean() - diffs_cross.mean())
    print(f"LOPO: n={len(diffs_self)} d_self={diffs_self.mean():.4f} "
          f"d_cross={diffs_cross.mean():.4f} delta={obs_delta:.4f}", flush=True)

    # bootstrap CI
    rngb = np.random.default_rng(SEED + 2)
    boot = []
    for _ in range(N_LOPO_BOOT):
        ids = rngb.integers(0, len(diffs_self), size=len(diffs_self))
        boot.append(diffs_self[ids].mean() - diffs_cross[rngb.integers(0, len(diffs_cross), size=len(diffs_self))].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # AUC
    auc = 0.0
    ncmp = 0
    for d in diffs_self:
        auc += float((diffs_cross > d).sum())
        ncmp += len(diffs_cross)
    auc = auc / max(ncmp, 1)

    # ============================================================
    # Task 6: conditional permutation
    # ============================================================
    a_set = sorted({a for u in user_products for a in user_products[u]})
    prod_cluster = {a: ptype.get(a, "Unknown") for a in a_set}
    user_bucket: dict[str, int] = {}
    for u in users_for_lopo:
        all_sfs = [sf for a in user_products[u] for sf in up_sentfeats[(u, a)]]
        if not all_sfs:
            continue
        # use average rating for the user's products (cheap proxy: from flat)
        rs = [r for (uu, a, r, t) in flat if uu == u and a in user_products[u]]
        if rs:
            user_bucket[u] = int(np.floor(np.mean(rs) / RATING_BUCKET))

    perm_deltas = []
    rngp = np.random.default_rng(SEED + 3)
    for _ in range(N_PERM):
        null_diffs = []
        for u in users_for_lopo:
            held = user_products[u][0]
            cluster = prod_cluster.get(held, "Unknown")
            bucket = user_bucket.get(u, 3)
            cands = [uu for uu in users_for_lopo if uu != u
                     and user_bucket.get(uu, -1) == bucket
                     and any(prod_cluster.get(p, "") == cluster
                             for p in user_products.get(uu, []))]
            if not cands:
                continue
            u_perm = rngp.choice(cands)
            products = sorted(user_products[u_perm])
            held_p = products[0]
            train_ps = products[1:]
            if len(train_ps) < 2:
                continue
            v_perm = lopo_vec(u_perm, train_ps)
            v_held_p_raw = up_vec[(u_perm, held_p)][stable_idx]
            v_held_p = normalize(v_held_p_raw)
            null_diffs.append(float(np.linalg.norm(v_perm - v_held_p)))
        if null_diffs:
            perm_deltas.append(np.mean(null_diffs))
    perm_deltas = np.asarray(perm_deltas)
    perm_delta_vec = perm_deltas - diffs_cross.mean()
    obs_delta_signed = obs_delta
    null_mean = float(perm_delta_vec.mean()) if len(perm_delta_vec) else 0.0
    p_one_lower = (float((perm_delta_vec <= obs_delta_signed).sum()) + 1) / (len(perm_delta_vec) + 1)
    abs_null = np.abs(perm_delta_vec - null_mean)
    abs_obs_perm = abs(obs_delta_signed - null_mean)
    p_two = (float((abs_null >= abs_obs_perm).sum()) + 1) / (len(perm_delta_vec) + 1)

    # ============================================================
    # Output
    # ============================================================
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "question": "are ratio/mean/distribution syntactic features identifiable "
                    "on held-out products (LOPO) under within-product standardization?",
        "design": "Task1 feat-product_mean=resid; stability filter "
                 "|corr(feat, sent_count_per_product)|<0.2; Task3 LOPO "
                 "d_self<d_cross; Task6 conditional permute user label within "
                 "(product_cluster, rating_bucket)",
        "feat_dim_raw": len(ALL_FEATS),
        "feat_dim_stable": feat_dim,
        "stable_features": stable_feats,
        "unstable_features": unstable_feats,
        "feat_corr_with_sentcount": {f: round(c, 4) for f, c in feat_corr.items()},
        "n_users": len(diffs_self), "n_cross_pairs": len(diffs_cross),
        "n_lopo_boot": N_LOPO_BOOT, "n_perm": N_PERM,
        "min_asins": MIN_ASINS, "min_sents_total": MIN_SENTS_TOTAL,
        "lopo": {
            "d_self_mean": round(float(diffs_self.mean()), 4),
            "d_cross_mean": round(float(diffs_cross.mean()), 4),
            "delta": round(obs_delta, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "auc_self_lt_cross": round(float(auc), 4),
        },
        "permutation": {
            "n_valid": int(len(perm_deltas)),
            "null_delta_mean": round(null_mean, 4),
            "obs_delta": round(obs_delta, 4),
            "p_one_lower": round(float(p_one_lower), 4),
            "p_two_sided": round(float(p_two), 4),
        },
        "runtime_sec": round(time.time() - t0, 1),
    }, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT} (t={time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
