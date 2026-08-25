"""Stage 9G-1.5 v2 — Rebuild user Gaussians at 3 levels WITH CACHING.

Speed-up vs v1: cache clause/dep/connective features by sha1(text) so each
sentence is only parsed once. Re-runs are O(read-cache).

Output: /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9g_user_gaussians_3levels.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9g_user_gaussians.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9g_user_gaussians_run.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
RESIDUAL_SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
SENTS_PATH = RESIDUAL_SCRATCH / "sentences_for_rewrite_10k.jsonl"
FEAT_CACHE_182D = RESIDUAL_SCRATCH / "sentences_318d_cache.jsonl.gz"
ENRICHED_CACHE = SCRATCH / "stage9g_clause_conn_cache.jsonl.gz"
REVIEW_GZ = REPO_ROOT / "data/Baby_Products_2023.jsonl.gz"
ASINS_IN = SCRATCH / "stage8_5_asins.json"
USER_GAUSSIANS_OUT = SCRATCH / "stage9g_user_gaussians_3levels.json"
USER_GAUSSIANS_ORIG = SCRATCH / "stage8_5_user_gaussians.json"

SEED = 2024
PCA_DIM = 48
LAMBDA = 0.1
VAR_EPS = 1e-3
MIN_REVIEWS_FOR_PER_USER = 3
N_TRAIN = 5000


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


CLAUSE_DEPS = {"acl", "advcl", "ccomp", "relcl", "xcomp"}
TOP_DEP_RELS = [
    "nsubj", "nsubjpass", "obj", "iobj", "obl", "amod", "advmod",
    "prep", "pobj", "det", "compound", "conj", "cc", "aux", "auxpass",
    "punct", "case", "mark", "acl", "relcl", "advcl",
]
CONNECTIVE_WORDS = [
    "which", "that", "who", "whom", "whose",
    "because", "since", "while", "although", "when", "if",
    "for", "with", "in", "of", "by", "at", "on", "from", "to", "about",
]


def extract_clause_dependency_fast(doc) -> dict:
    """Vectorized-ish: extract clause/dep features in one pass."""
    feats = {}
    # Pre-compute counts in single pass
    dep_count = Counter()
    pos_amod = []
    pos_prep = []
    pos_clause = []
    pos_conj = []
    n_tok = 0
    for t in doc:
        dep_count[t.dep_] += 1
        n_tok += 1
        if t.dep_ == "amod":
            pos_amod.append((t.i, t.head.i))
        if t.dep_ == "prep":
            pos_prep.append(t.i)
        if t.dep_ in CLAUSE_DEPS:
            pos_clause.append((t, t.head))
        if t.dep_ in ("conj", "cc"):
            pos_conj.append(t.i)

    n_acl = dep_count.get("acl", 0)
    n_advcl = dep_count.get("advcl", 0)
    n_ccomp = dep_count.get("ccomp", 0)
    n_relcl = dep_count.get("relcl", 0)
    n_xcomp = dep_count.get("xcomp", 0)
    total_clauses = n_acl + n_advcl + n_ccomp + n_relcl + n_xcomp

    feats["acl_count"] = n_acl
    feats["advcl_count"] = n_advcl
    feats["ccomp_count"] = n_ccomp
    feats["relcl_count"] = n_relcl
    feats["xcomp_count"] = n_xcomp
    feats["total_clause_count"] = total_clauses
    if total_clauses > 0:
        feats["subordinate_clause_ratio"] = (n_advcl + n_ccomp + n_xcomp) / total_clauses
        feats["relative_clause_ratio"] = n_relcl / total_clauses
    else:
        feats["subordinate_clause_ratio"] = 0.0
        feats["relative_clause_ratio"] = 0.0
    feats["main_sub_clause_ratio"] = (1.0 if total_clauses == 0 else n_tok / max(total_clauses, 1)) / max(n_tok, 1)

    # Clause depth via chain walk (only for clause tokens, typically few)
    def chain_depth(token):
        d = 0
        current = token
        while current.dep_ in CLAUSE_DEPS and current.head != current:
            d += 1
            current = current.head
        return d

    if pos_clause:
        depths = [chain_depth(t) for t, _ in pos_clause]
        feats["clause_depth_max"] = max(depths)
        feats["clause_depth_mean"] = float(np.mean(depths))
    else:
        feats["clause_depth_max"] = 0
        feats["clause_depth_mean"] = 0.0

    if len(pos_conj) >= 2:
        spans = np.diff(pos_conj)
        feats["coordination_span_mean"] = float(np.mean(spans))
        feats["coordination_span_max"] = int(np.max(spans))
    else:
        feats["coordination_span_mean"] = 0.0
        feats["coordination_span_max"] = 0

    n_prep = dep_count.get("prep", 0)
    feats["prep_count"] = n_prep
    feats["prep_avg_attachment_distance"] = 0.0  # skip for speed

    modifiers_before = 0
    modifiers_after = 0
    for ti, hi in pos_amod:
        if ti < hi:
            modifiers_before += 1
        else:
            modifiers_after += 1
    total_mod = modifiers_before + modifiers_after
    if total_mod > 0:
        feats["modifier_before_head_ratio"] = modifiers_before / total_mod
        feats["modifier_after_head_ratio"] = modifiers_after / total_mod
    else:
        feats["modifier_before_head_ratio"] = 0.0
        feats["modifier_after_head_ratio"] = 0.0

    feats["passive_count"] = dep_count.get("nsubjpass", 0) + dep_count.get("auxpass", 0)
    feats["copula_count"] = dep_count.get("cop", 0)
    for d in TOP_DEP_RELS:
        feats[f"depnorm_{d}"] = dep_count.get(d, 0) / max(n_tok, 1)

    has_nsubj = dep_count.get("nsubj", 0) > 0
    has_obj = (dep_count.get("obj", 0) + dep_count.get("iobj", 0)) > 0
    feats["has_svo_pattern"] = int(has_nsubj and has_obj)
    return feats


def extract_connective(text: str) -> dict:
    tl = text.lower().strip()
    feats = {}
    for w in CONNECTIVE_WORDS:
        pattern = r"\b" + re.escape(w) + r"\b"
        feats[f"cnt_{w}"] = len(re.findall(pattern, tl))
    return feats


def load_or_build_enriched_cache(texts):
    """Load clause+connective features from cache; build missing via spaCy."""
    log(f"  Loading enriched cache from {ENRICHED_CACHE}")
    cache = {}
    if ENRICHED_CACHE.exists():
        with gzip.open(ENRICHED_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[rec["k"]] = rec["v"]

    # Find missing
    missing = []
    for t in texts:
        k = feat_key(t)
        if k not in cache:
            missing.append(t)
    log(f"  cache: {len(cache)} entries, missing: {len(missing)}")

    if not missing:
        return cache

    # Dedupe missing
    unique_missing = list(set(missing))
    log(f"  unique missing: {len(unique_missing)}")

    # Parse + extract
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    t0 = time.time()
    docs = list(nlp.pipe(unique_missing, batch_size=512, n_process=1))
    log(f"  parsed {len(docs)} docs in {time.time() - t0:.1f}s")

    t0 = time.time()
    new_records = []
    for text, doc in zip(unique_missing, docs):
        clause = extract_clause_dependency_fast(doc)
        conn = extract_connective(text)
        rec = {"k": feat_key(text), "clause": clause, "connective": conn}
        new_records.append(rec)
        cache[rec["k"]] = {"clause": clause, "connective": conn}
    log(f"  extracted features in {time.time() - t0:.1f}s")

    # Append to cache
    with gzip.open(ENRICHED_CACHE, "at", encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec) + "\n")
    log(f"  appended {len(new_records)} to cache (total {len(cache)})")

    return cache


def main():
    log("=== Stage 9G-1.5 v2 — Rebuild user Gaussians with CACHING ===")

    # === 1. Load existing user Gaussians (Base182) ===
    log("\n=== 1. Loading existing user Gaussians ===")
    ug_orig = json.load(open(USER_GAUSSIANS_ORIG))
    users_with_gaussian = set(ug_orig["users"].keys())
    log(f"  users with Gaussian: {len(users_with_gaussian)}")

    # === 2. Load ASIN data ===
    asin_data = json.load(open(ASINS_IN))["asins"]
    target_users = set()
    for a in asin_data:
        for grp in a.get("users_sampled", []):
            if isinstance(grp, list):
                target_users.update(grp)
            else:
                target_users.add(grp)

    # === 3. Get user review sentences (just texts, dedupe) ===
    log("\n=== 2. Loading user review sentences ===")
    user_review_texts = collections.defaultdict(list)
    n_records = 0
    with gzip.open(REVIEW_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            n_records += 1
            if n_records > 5e7:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            uid = r.get("user_id") or r.get("reviewerID") or r.get("reviewer_id")
            if uid in users_with_gaussian:
                txt = r.get("text") or r.get("reviewText") or r.get("summary") or ""
                if txt:
                    user_review_texts[uid].append(txt)
            if n_records % 500000 == 0:
                log(f"    {n_records/1e6:.1f}M records")
    log(f"  users with reviews: {len(user_review_texts)}")

    # === 4. Sentence split ===
    log("\n=== 3. Sentence splitting ===")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    all_sents = []
    sent_to_user = []
    for uid, texts in user_review_texts.items():
        if uid not in users_with_gaussian:
            continue
        docs = list(nlp.pipe(texts, batch_size=64))
        for doc in docs:
            for s in doc.sents:
                t = s.text.strip()
                if len(t.split()) >= 3:
                    all_sents.append(t)
                    sent_to_user.append(uid)
    log(f"  total user review sentences: {len(all_sents)}")

    # === 5. Load training sentences ===
    log("\n=== 4. Loading training sentences ===")
    train_sents_raw = []
    with open(SENTS_PATH) as f:
        for line in f:
            try:
                r = json.loads(line)
                train_sents_raw.append(r.get("sentence_text", ""))
            except Exception:
                continue
    rng = np.random.default_rng(42)
    train_idx_chosen = rng.choice(len(train_sents_raw), size=min(N_TRAIN, len(train_sents_raw)), replace=False)
    train_sents = [train_sents_raw[i] for i in train_idx_chosen]
    log(f"  training sentences: {len(train_sents)}")

    # === 6. Build enriched cache for training + user review sentences ===
    log("\n=== 5. Building enriched feature cache ===")
    all_texts_for_cache = list(set(all_sents + train_sents))
    enriched_cache = load_or_build_enriched_cache(all_texts_for_cache)

    # === 7. Build feature matrices at each level ===
    log("\n=== 6. Building feature matrices ===")

    # Load 182d feature cache
    feat_map_182d = {}
    with gzip.open(FEAT_CACHE_182D, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map_182d[rec["k"]] = rec["v"]
    log(f"  182d cache: {len(feat_map_182d)} entries")

    # Get ordered feature names
    base_182_names = None
    for s in train_sents:
        f = feat_map_182d.get(feat_key(s))
        if f:
            base_182_names = sorted([n for n, v in f.items() if isinstance(v, (int, float))])
            break
    if not base_182_names:
        log("  ERROR: no 182d features found")
        return
    log(f"  Base182 numeric: {len(base_182_names)}")

    # Get clause/conn names from first enriched cache entry
    sample_ec = next(iter(enriched_cache.values()))
    clause_names = sorted(sample_ec["clause"].keys())
    conn_names = sorted(sample_ec["connective"].keys())
    log(f"  +Clause/Dep: {len(clause_names)}")
    log(f"  +Connective: {len(conn_names)}")

    def build_matrix_for_level(texts, level):
        """Build feature matrix at given level."""
        X = []
        miss = 0
        for t in texts:
            k = feat_key(t)
            f182 = feat_map_182d.get(k, {})
            ec = enriched_cache.get(k)
            if not ec:
                miss += 1
                # Use zeros for missing
            vec = [float(f182.get(n, 0.0) or 0.0) for n in base_182_names]
            if level in ("clause", "conn") and ec:
                vec += [float(ec["clause"].get(n, 0.0) or 0.0) for n in clause_names]
            elif level in ("clause", "conn"):
                vec += [0.0] * len(clause_names)
            if level == "conn" and ec:
                vec += [float(ec["connective"].get(n, 0.0) or 0.0) for n in conn_names]
            elif level == "conn":
                vec += [0.0] * len(conn_names)
            X.append(vec)
        if miss > 0:
            log(f"    missing enriched features: {miss}/{len(texts)}")
        return np.array(X, dtype=np.float64)

    log("  building train matrices...")
    X_train_base = build_matrix_for_level(train_sents, "base")
    X_train_clause = build_matrix_for_level(train_sents, "clause")
    X_train_conn = build_matrix_for_level(train_sents, "conn")
    log(f"  X_train_base: {X_train_base.shape}")
    log(f"  X_train_clause: {X_train_clause.shape}")
    log(f"  X_train_conn: {X_train_conn.shape}")

    log("  building user matrices...")
    X_user_clause = build_matrix_for_level(all_sents, "clause")
    X_user_conn = build_matrix_for_level(all_sents, "conn")
    log(f"  X_user_clause: {X_user_clause.shape}")
    log(f"  X_user_conn: {X_user_conn.shape}")

    # === 8. Build user Gaussians ===
    log("\n=== 7. Building user Gaussians ===")
    user_gaussians_3l = {"base": {}, "clause": {}, "conn": {}}

    # BASE: keep existing
    for uid, info in ug_orig["users"].items():
        user_gaussians_3l["base"][uid] = {
            "mu": info["mu"],
            "sigma_diag": info["sigma_diag"],
            "n_reviews": info.get("n_reviews"),
            "n_words": info.get("n_words"),
            "source": info.get("source"),
        }
    log(f"  base: kept {len(user_gaussians_3l['base'])} (from Stage 8.5)")

    # CLAUSE / CONN: rebuild
    for level, X_tr, X_us in [
        ("clause", X_train_clause, X_user_clause),
        ("conn", X_train_conn, X_user_conn),
    ]:
        scaler = StandardScaler()
        scaler.fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_us_s = scaler.transform(X_us)

        pca = PCA(n_components=PCA_DIM, random_state=SEED)
        pca.fit(X_tr_s)
        Z_user = pca.transform(X_us_s)

        user_z = collections.defaultdict(list)
        for z, uid in zip(Z_user, sent_to_user):
            user_z[uid].append(z)

        all_z = np.stack([z for zs in user_z.values() for z in zs], axis=0)
        global_var = all_z.var(axis=0)

        for uid in users_with_gaussian:
            zs = user_z.get(uid, [])
            n_reviews = len(user_review_texts.get(uid, []))
            n_words = sum(len(t.split()) for t in user_review_texts.get(uid, []))

            if len(zs) >= MIN_REVIEWS_FOR_PER_USER:
                Z = np.stack(zs, axis=0)
                mu = Z.mean(axis=0)
                var = Z.var(axis=0)
                var_shrunk = (1 - LAMBDA) * var + LAMBDA * global_var
                var_shrunk = np.maximum(var_shrunk, VAR_EPS)
                user_gaussians_3l[level][uid] = {
                    "mu": mu.tolist(),
                    "sigma_diag": var_shrunk.tolist(),
                    "n_reviews": n_reviews,
                    "n_words": n_words,
                    "source": "per_user",
                }
            elif len(zs) >= 1:
                mu = zs[0]
                var_shrunk = np.maximum(global_var.copy(), VAR_EPS)
                user_gaussians_3l[level][uid] = {
                    "mu": mu.tolist(),
                    "sigma_diag": var_shrunk.tolist(),
                    "n_reviews": n_reviews,
                    "n_words": n_words,
                    "source": "marginal",
                }
        log(f"  {level}: built {len(user_gaussians_3l[level])}")

    # === 9. Save ===
    log("\n=== 8. Saving ===")
    out_data = {
        "config": {
            "SEED": SEED, "PCA_DIM": PCA_DIM, "LAMBDA": LAMBDA,
            "N_TRAIN": N_TRAIN, "MIN_REVIEWS_FOR_PER_USER": MIN_REVIEWS_FOR_PER_USER,
            "description": "User Gaussians at 3 feature levels; base=Stage 8.5, clause/conn=rebuilt with enriched features",
        },
        "feature_names": {
            "base": base_182_names,
            "clause": clause_names,
            "conn": conn_names,
        },
        "users": user_gaussians_3l,
    }
    with open(USER_GAUSSIANS_OUT, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    log(f"  wrote → {USER_GAUSSIANS_OUT}")

    log("\n=== Stage 9G-1.5 v2 complete ===")


if __name__ == "__main__":
    main()