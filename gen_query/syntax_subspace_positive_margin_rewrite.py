"""Phase 7.B — Positive-Margin Source Rewrite (2026-08-31).

User critique (2026-08-31, CRITICAL):
- Phase 7.A's "structural NO-GO" over-claims. centroid-nearest (argmin d_self)
  != most user-distinctive (argmax M = d_other - d_self). A sentence at
  d_self=5.0 with d_other=4.7 (M=-0.3) is NOT evidence that no M>0 sentence
  exists for that user (6.E.4 shows 95.7-98.6% of users have >50% real
  sentences with M>0 under the same-ASIN anchor).
- Anchor mismatch: 6.E.4 uses LOO mu_{u,-s} (d_self_med ~4.35) while 7.A
  uses production full mu_u (rep_d_self 4.99-5.51). Difference 0.6-1.1.

Design (per user proposal):
1. For ALL user historical sentences (excl target ASIN, product kw,
   length 8-60): compute M_local(s) under BOTH anchors with the EXACT
   6.E.4 competitor set (same-ASIN + quality gate + BD-separated greedy
   max-min, T_B=2.0):
     M_full = d_other - ||z_s - mu_u_production||    (7.A anchor)
     M_LOO  = d_other - ||z_s - mu_{u,-s}||          (6.E.4 anchor)
2. Reconciliation: P(M_full>0) vs P(M_LOO>0) per user. If M_full<0 but
   M_LOO>0 → 6.E.4/7.A mismatch is ANCHOR. If both <0 → selection bias.
3. Pick positive source: filter M_LOO>0, s* = argmax M_LOO (also report
   Score = M_LOO - LAMBDA*d_self_LOO variant).
4. Minimal rewrite (C_rewrite from positive source) + A_free control,
   K=4 each, Qwen2-7B vLLM batched.
5. Audit under 6.E.4 anchor:
     M_rewrite_LOO / M_rewrite_full
     d_drift = ||z_rew - z_source||
     dF3     = ||f3_rew - f3_source||_1 (103d)
     dep_arc_jaccard (pos-generalized dependency arcs)
     clause delta (F3 clause dims) + token overlap

Decision:
- M_rewrite stays >0 at high rate → rewrite route WORKS; 7.A failure
  was selection bias (centroid-nearest).
- M_source>0 but M_rewrite<0 stable → LLM rewrite destroys user-specific
  PCA48 geometry (strong causal evidence).

Per Rule 18 this is the SMOKE run: N=5 users x K=4 x 2 cond = 40 gens.
"""
import collections
import gzip
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import spacy

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
sys.path.insert(0, str(REPO_ROOT))

# === Hardcoded paths (Rule 3) ===
DATA_DIR = Path("/fs04/ar57/wenyu/PersoanlQuery/data")
REVIEWS_GZ = DATA_DIR / "Baby_Products_2023.jsonl.gz"
META_GZ = DATA_DIR / "meta_Baby_Products_2023.jsonl.gz"
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"
GAUSS_CV_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians_cv.json"
ASINS_JSON = SCRATCH / "gaussian_vades" / "stage8_5_asins.json"
PATTRS_PATH = REPO_ROOT / "result" / "product_attributes.json"
PHASE7A_JSON = REPO_ROOT / "result" / "gen_query" / "phase7a_representative_sentences.json"
LOG_PATH = SCRATCH / "logs" / "phase7b_positive_margin_rewrite.log"
RESULT_PATH = REPO_ROOT / "result" / "gen_query" / "phase7b_positive_margin_rewrite.json"

# === Config (Rule 3) ===
USERS = [
    "AERFSIGUZKWIES3W3FRBVUNX4RUA",
    "AFOTTSAZYNXVVEZ5IV24QOP5QOWQ",
    "AFYB7O3AY4KNFJ466V2KSOQB2ODQ",
    "AHRQVI734AF32IXI37RUF7P56KMQ",
    "AFRGJMSUHGKN6E36WF7NUXIQVXVQ",
]
MIN_TOKENS = 8
MAX_TOKENS = 60
CV_INLIER_MIN = 0.9
CV_NLL_MAX = 39.0
T_B = 2.0            # BD-separated threshold (6.E.4 same)
SEED = 42
DTYPE = np.float32

# Generation
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 120
VLLM_PORT = 8800
K_PER_SOURCE = 4
LAMBDA = 0.15        # Score = M - LAMBDA*d_self (typicality tiebreak)
MAX_GEN_WORDS = 60   # post-gen length cap


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] [phase7b] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# === 6.E.4 helpers (verbatim logic) ===
def bhattacharyya_distance_diag(mu_i, sigma_i, mu_j, sigma_j):
    mu_i = np.asarray(mu_i, dtype=np.float64)
    mu_j = np.asarray(mu_j, dtype=np.float64)
    sigma_i = np.asarray(sigma_i, dtype=np.float64)
    sigma_j = np.asarray(sigma_j, dtype=np.float64)
    diff = mu_i - mu_j
    sigma_avg = 0.5 * (sigma_i + sigma_j)
    term1 = 0.125 * np.sum(diff * diff / np.maximum(sigma_avg, 1e-12))
    term2 = 0.5 * np.sum(np.log(np.maximum(sigma_avg, 1e-12))
                         - 0.5 * np.log(np.maximum(sigma_i, 1e-12))
                         - 0.5 * np.log(np.maximum(sigma_j, 1e-12)))
    return float(term1 + term2)


def greedy_maxmin_in_group(users_data, t_b):
    """Greedy max-min BD separation within a group (6.E.4 same).
    users_data: list of (uid, mu, sigma_diag). Returns selected indices."""
    n = len(users_data)
    if n == 0:
        return [], None
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = bhattacharyya_distance_diag(
                users_data[i][1], users_data[i][2],
                users_data[j][1], users_data[j][2],
            )
            dists[i, j] = dists[j, i] = d
    rng = np.random.default_rng(SEED)
    first = int(rng.integers(0, n))
    selected = [first]
    remaining = set(range(n)) - {first}
    while remaining:
        min_d_to_selected = np.array(
            [min(dists[i, j] for j in selected) for i in remaining]
        )
        best = max(zip(min_d_to_selected, remaining), key=lambda x: x[0])[1]
        if min_d_to_selected[list(remaining).index(best)] < t_b:
            break
        selected.append(best)
        remaining.discard(best)
    return sorted(selected), dists


# === Data loading (7.A.1 same filters) ===
def load_user_target_mapping():
    with open(PHASE7A_JSON) as f:
        d = json.load(f)
    return {r["user_id"]: r["target_asin"] for r in d["per_user"]
            if "target_asin" in r and r["user_id"] in USERS}


def load_target_product_keywords(target_asins):
    target_kw = {}
    with gzip.open(META_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            asin = r.get("parent_asin") or r.get("asin")
            if asin not in target_asins:
                continue
            title = r.get("title", "")
            brand = r.get("store", "") or r.get("brand", "")
            details = r.get("details", {})
            if isinstance(details, dict):
                brand = details.get("Brand", brand)
            kw = set()
            for s in (title, brand):
                for w in re.findall(r"\b[a-zA-Z]{3,}\b", s.lower()):
                    kw.add(w)
            generic = {"the", "and", "for", "with", "this", "that", "from",
                       "baby", "item", "use", "your", "you", "are", "can"}
            kw = kw - generic
            target_kw[asin] = kw
    return target_kw


def load_user_reviews(user_ids):
    user_reviews = collections.defaultdict(list)
    target_set = set(user_ids)
    n_records = 0
    n_match = 0
    with gzip.open(REVIEWS_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = r.get("reviewerID") or r.get("user_id")
            if uid in target_set and r.get("text"):
                asin = r.get("asin") or r.get("parent_asin")
                user_reviews[uid].append((r["text"], asin))
                n_match += 1
            n_records += 1
            if n_records % 2_000_000 == 0:
                log(f"    {n_records/1e6:.1f}M records scanned, {n_match} matched")
    log(f"  scanned {n_records} records, matched {n_match} reviews")
    return user_reviews


def filter_sentences(reviews, target_asin, target_kw):
    out = []
    for text, asin in reviews:
        if asin == target_asin:
            continue
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        for s in sents:
            s = s.strip()
            if not s:
                continue
            n_tok = len(s.split())
            if n_tok < MIN_TOKENS or n_tok > MAX_TOKENS:
                continue
            s_lower = s.lower()
            if target_kw and any(kw in s_lower for kw in target_kw):
                continue
            out.append(s)
    return out


# === Projection (batched) ===
def project_batch(texts, nlp, all_fnames, fnames_sub, col_idx,
                  scaler_mean, scaler_scale, pca_components, pca_mean):
    """Return (z48: (n,48), f3: (n,103), ok_mask: (n,) bool)."""
    from common.syntactic_features import per_sentence_features_v2 as psf
    n = len(texts)
    z48 = np.zeros((n, 48), dtype=np.float64)
    f3 = np.zeros((n, len(fnames_sub)), dtype=np.float64)
    ok = np.zeros(n, dtype=bool)
    fnames_arr = np.array(all_fnames)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=128, n_process=4)):
        sents = list(doc.sents)
        if not sents:
            continue
        feats_per_sent = []
        for s in sents:
            f = psf(s)
            if f is not None:
                feats_per_sent.append(f)
        if not feats_per_sent:
            continue
        all_keys = set()
        for f in feats_per_sent:
            all_keys.update(f.keys())
        mean_feats = {}
        for k in all_keys:
            vals = [f.get(k, 0.0) for f in feats_per_sent]
            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
                try:
                    mean_feats[k] = float(np.mean(vals))
                except (TypeError, ValueError):
                    pass
        vec_full = np.array([mean_feats.get(nm, 0.0) for nm in all_fnames], dtype=np.float64)
        v = vec_full[col_idx]
        v_scaled = (v - scaler_mean) / scaler_scale
        z48[i] = (v_scaled - pca_mean) @ pca_components.T
        f3[i] = v
        ok[i] = True
    return z48, f3, ok


# === Generation (7.A smoke same: /v1/chat/completions, batched threads) ===
def build_prompt_free(attrs):
    attrs_str = ", ".join(attrs)
    return (
        "You are a real parent looking for a product on Amazon.\n"
        f"The product has these attributes: {attrs_str}.\n"
        "Write a single natural shopping query (15-30 words) in your own voice.\n"
        "Be specific and personal. Output only the query."
    )


def build_prompt_rewrite(attrs, source_sentence):
    attrs_str = ", ".join(attrs)
    return (
        "You are a real parent looking for a product on Amazon.\n"
        f"The product has these attributes: {attrs_str}.\n"
        "TASK: Modify the sentence below to describe the new product.\n"
        "STRICT RULES:\n"
        "  - Keep the SAME function words (although, because, when, that, etc.)\n"
        "  - Keep the SAME clause order and dependency structure\n"
        "  - Keep the SAME sentence length\n"
        "  - ONLY substitute content words (products, properties, functions)\n"
        f"SOURCE: \"{source_sentence}\"\n"
        "OUTPUT: The rewritten sentence only (no explanation)."
    )


def vllm_generate(prompts):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    model_name = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
    url = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    results = [None] * len(prompts)
    BATCH = 32
    MAX_WORKERS = 32
    MAX_RETRIES = 3

    def _call(idx_p):
        idx, p = idx_p
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": p}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_NEW_TOKENS,
            "top_p": 0.95,
        }
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=60)
                if r.status_code == 200:
                    d = r.json()
                    return idx, d["choices"][0]["message"]["content"].strip()
                time.sleep(2.0 * (attempt + 1))
            except Exception:
                time.sleep(2.0 * (attempt + 1))
        return idx, ""

    t0 = time.time()
    n_done = 0
    for s in range(0, len(prompts), BATCH):
        chunk = list(enumerate(prompts[s:s + BATCH], start=s))
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for idx, out in ex.map(_call, chunk):
                results[idx] = out
                n_done += 1
        log(f"  batch {s//BATCH+1}: {n_done}/{len(prompts)} done ({time.time()-t0:.1f}s)")
    return results


# === Syntax-preservation metrics ===
def dep_arc_set(doc):
    """Pos-generalized dependency arc set: {(head_pos, dep, child_pos)}."""
    arcs = set()
    for tok in doc:
        if tok.dep_ == "ROOT":
            continue
        arcs.add((tok.head.pos_, tok.dep_, tok.pos_))
    return arcs


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def clause_dims_f3():
    return ["n_clause", "n_advcl", "n_relcl", "n_ccomp", "n_coord",
            "n_conj", "n_advmod", "n_mod", "n_aux", "n_auxpass"]


def main():
    log("=== Phase 7.B — Positive-Margin Source Rewrite (SMOKE) ===")
    log(f"  users={len(USERS)}, K={K_PER_SOURCE}, conds=[A_free, C_rewrite], T_B={T_B}")
    log(f"  LAMBDA={LAMBDA}, CV gate=[{CV_INLIER_MIN},{CV_NLL_MAX}]")

    # 1. user → target ASIN (from 7.A output)
    user_target = load_user_target_mapping()
    log(f"\n  user→target: {user_target}")

    # 2. target product keywords
    log(f"\n  Loading target product keywords from {META_GZ}...")
    target_kw = load_target_product_keywords(set(user_target.values()))
    for a, kw in target_kw.items():
        log(f"    {a}: {len(kw)} kw")

    # 3. user reviews + sentence filtering
    log(f"\n  Loading reviews from {REVIEWS_GZ}...")
    user_reviews = load_user_reviews(USERS)
    user_sents = {}
    for u in USERS:
        cands = filter_sentences(user_reviews.get(u, []),
                                 user_target[u], target_kw.get(user_target[u], set()))
        user_sents[u] = cands
        log(f"    {u[:12]}: {len(cands)} candidate sentences")

    # 4. Load PCA48 + production μ + CV quality
    log(f"\n  Loading PCA48 + μ from {GAUSS_PATH}...")
    with open(GAUSS_PATH) as f:
        gdoc = json.load(f)
    all_fnames = gdoc["feature_names_ordered"]
    fnames_sub = gdoc["fnames_f3"]
    scaler_mean = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
    pca_components = np.asarray(gdoc["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gdoc["pca_mean"], dtype=np.float64)
    col_idx = np.array([all_fnames.index(n) for n in fnames_sub], dtype=np.int64)
    users_dict = gdoc["users"]
    f3_idx = {n: i for i, n in enumerate(fnames_sub)}

    log(f"  Loading CV quality from {GAUSS_CV_PATH}...")
    with open(GAUSS_CV_PATH) as f:
        cv_data = json.load(f)
    cv_users = cv_data["users"]

    log(f"  Loading ASIN cohort from {ASINS_JSON}...")
    with open(ASINS_JSON) as f:
        asin_data = json.load(f)
    # 6.E.4 same dual structure: uid → asins AND asin → uids
    user_to_asins = collections.defaultdict(list)
    asin_to_users_map = collections.defaultdict(list)
    for entry in asin_data["asins"]:
        a = entry["asin"]
        for uid in entry["users_sampled"]:
            user_to_asins[uid].append(a)
            asin_to_users_map[a].append(uid)

    # 5. spaCy
    log(f"\n  Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    # 6. Per user: project candidates, build 6.E.4 competitor, dual-anchor M
    log(f"\n=== Step A: per-user candidate projection + dual-anchor M ===")
    per_user = []
    all_sources = {}
    for u in USERS:
        texts = user_sents[u]
        log(f"\n  {u[:12]} ({len(texts)} cands): projecting...")
        z48, f3, ok = project_batch(
            texts, nlp, all_fnames, fnames_sub, col_idx,
            scaler_mean, scaler_scale, pca_components, pca_mean,
        )
        n_ok = int(ok.sum())
        log(f"    projected {n_ok}/{len(texts)} ok")
        if n_ok == 0:
            log(f"    [ERROR] no sentences projected for {u}")
            continue
        zs = z48[ok]
        fs = f3[ok]
        ts = [t for t, o in zip(texts, ok) if o]

        # production μ (7.A anchor)
        prod_mu = np.asarray(users_dict[u]["mu"], dtype=np.float32)
        # LOO μ: mean of candidate z (excl self)
        loo_mu_all = zs.mean(axis=0)

        # --- 6.E.4 competitor: same-ASIN + quality + BD greedy max-min ---
        asin_users_data = []
        for asin in user_to_asins.get(u, []):
            for other in asin_to_users_map.get(asin, []):
                if other == u:
                    continue
                uc = cv_users.get(other)
                if uc is None:
                    continue
                if uc.get("cv_inlier_frac") is None or uc.get("cv_nll") is None:
                    continue
                if uc["cv_inlier_frac"] < CV_INLIER_MIN or uc["cv_nll"] > CV_NLL_MAX:
                    continue
                mu = np.asarray(users_dict[other]["mu"], dtype=np.float64)
                sigma = np.asarray(cv_users[other]["sigma_diag"], dtype=np.float64)
                if mu.shape == (48,) and sigma.shape == (48,):
                    asin_users_data.append((other, mu, sigma))
        # dedup by uid
        seen = set()
        dedup = []
        for rec in asin_users_data:
            if rec[0] not in seen:
                seen.add(rec[0])
                dedup.append(rec)
        asin_users_data = dedup

        if len(asin_users_data) >= 2:
            sel_idx, _ = greedy_maxmin_in_group(asin_users_data, T_B)
            comp = np.array([asin_users_data[i][1] for i in sel_idx], dtype=np.float64)
        elif len(asin_users_data) == 1:
            comp = np.array([asin_users_data[0][1]], dtype=np.float64)
        else:
            comp = np.zeros((0, 48), dtype=np.float64)
        log(f"    competitor: {len(asin_users_data)} same-ASIN+Q → {len(comp)} BD-separated")

        if len(comp) == 0:
            log(f"    [ERROR] no competitor for {u}")
            continue

        # --- dual-anchor M per sentence ---
        d_other = np.linalg.norm(zs[:, None, :] - comp[None, :, :], axis=2).min(axis=1)
        d_full = np.linalg.norm(zs - prod_mu[None, :].astype(np.float64), axis=1)
        d_loo = np.zeros(n_ok)
        m_full = np.zeros(n_ok)
        m_loo = np.zeros(n_ok)
        for i in range(n_ok):
            others = [j for j in range(n_ok) if j != i]
            mu_loo_i = zs[others].mean(axis=0) if others else loo_mu_all
            d_loo[i] = float(np.linalg.norm(zs[i] - mu_loo_i))
            m_full[i] = d_other[i] - d_full[i]
            m_loo[i] = d_other[i] - d_loo[i]

        # cache for Step C audit (rewrite μ_LOO = full candidate mean, 6.E.4 same)
        _COMP_CACHE[u] = comp.astype(np.float64)
        _LOO_MU_CACHE[u] = loo_mu_all.astype(np.float32)

        p_full = float(np.mean(m_full > 0))
        p_loo = float(np.mean(m_loo > 0))
        log(f"    P(M_full>0)={p_full*100:.1f}%  P(M_LOO>0)={p_loo*100:.1f}%")
        log(f"    d_self_full med={np.median(d_full):.3f}  d_self_LOO med={np.median(d_loo):.3f}")

        # --- positive source selection (6.E.4 anchor) ---
        cand = [(i, m_loo[i]) for i in range(n_ok) if m_loo[i] > 0]
        if not cand:
            log(f"    [WARN] NO positive-margin sentence for {u} (M_LOO>0)")
            # still keep best-score sentence for diagnosis but flag it
            scores = [(i, m_loo[i] - LAMBDA * d_loo[i]) for i in range(n_ok)]
            scores.sort(key=lambda x: -x[1])
            best_i = scores[0][0]
            best_score = scores[0][1]
        else:
            best_i, _ = max(cand, key=lambda x: x[1])
            best_score = m_loo[best_i] - LAMBDA * d_loo[best_i]

        src_rec = {
            "user_id": u,
            "n_sentences": n_ok,
            "n_pos_margin": len(cand),
            "p_M_full_gt0": p_full,
            "p_M_LOO_gt0": p_loo,
            "d_self_full_med": float(np.median(d_full)),
            "d_self_LOO_med": float(np.median(d_loo)),
            "d_other_nearest": float(np.min(d_other)),
            "n_competitors": int(len(comp)),
            "source": ts[best_i],
            "source_idx": int(best_i),
            "source_M_LOO": float(m_loo[best_i]),
            "source_M_full": float(m_full[best_i]),
            "source_d_self_LOO": float(d_loo[best_i]),
            "source_d_self_full": float(d_full[best_i]),
            "source_d_other": float(d_other[best_i]),
            "source_score": float(best_score),
            "z48_source": zs[best_i].astype(np.float32).tolist(),
            "f3_source": fs[best_i].astype(np.float64).tolist(),
        }
        per_user.append(src_rec)
        all_sources[u] = src_rec
        log(f"    SOURCE (M_LOO={m_loo[best_i]:.2f}): \"{ts[best_i][:110]}\"")

    n_users_ok = len(per_user)
    log(f"\n  users with positive source: {n_users_ok}/{len(USERS)}")

    # 7. Generate: A_free control + C_rewrite from positive source
    log(f"\n=== Step B: generation (A_free + C_rewrite, K={K_PER_SOURCE}) ===")
    with open(PATTRS_PATH) as f:
        pattrs = json.load(f)
    # attrs: reuse 7.A smoke's attr extraction (flat dict format)
    from gen_query.syntax_subspace_minimal_rewrite_smoke import get_top_n_attrs
    import requests as _rq
    try:
        r = _rq.get(f"http://localhost:{VLLM_PORT}/v1/models", timeout=5)
        log(f"  vLLM: {r.status_code}")
    except Exception as e:
        log(f"  [ERROR] vLLM not ready: {e}")
        return

    prompt_recs = []
    for u in USERS:
        if u not in all_sources:
            continue
        attrs = get_top_n_attrs(user_target[u], pattrs, n=5)
        src = all_sources[u]["source"]
        for k in range(K_PER_SOURCE):
            prompt_recs.append({"user_id": u, "condition": "A_free",
                                "k": k, "attrs": attrs, "source": src,
                                "prompt": build_prompt_free(attrs)})
            prompt_recs.append({"user_id": u, "condition": "C_rewrite",
                                "k": k, "attrs": attrs, "source": src,
                                "prompt": build_prompt_rewrite(attrs, src)})
    log(f"  {len(prompt_recs)} prompts")
    outputs = vllm_generate([p["prompt"] for p in prompt_recs])
    for rec, out in zip(prompt_recs, outputs):
        rec["output"] = out or ""

    # 8. Audit: project outputs, M under both anchors, syntax preservation
    log(f"\n=== Step C: audit ===")
    gen_texts = [p["output"] for p in prompt_recs]
    z_gen, f3_gen, ok_gen = project_batch(
        gen_texts, nlp, all_fnames, fnames_sub, col_idx,
        scaler_mean, scaler_scale, pca_components, pca_mean,
    )
    log(f"  projected {int(ok_gen.sum())}/{len(gen_texts)} outputs")

    # dep-arc sets for sources and outputs (spaCy per-sentence)
    src_docs = {u: next(nlp.pipe([all_sources[u]["source"]])) for u in all_sources}
    out_docs = list(nlp.pipe(gen_texts, batch_size=128))

    audit = []
    for pi, rec in enumerate(prompt_recs):
        u = rec["user_id"]
        if not ok_gen[pi]:
            continue
        src = all_sources[u]
        mu_loo = loo_mu_for(u, per_user)  # stored below
        prod_mu = np.asarray(users_dict[u]["mu"], dtype=np.float32)
        # competitor for this user (reuse comp array — stored in per_user)
        comp_arr = np.asarray(comp_for(u, per_user), dtype=np.float64)
        d_other = float(np.linalg.norm(z_gen[pi] - comp_arr, axis=1).min())
        d_full = float(np.linalg.norm(z_gen[pi] - prod_mu.astype(np.float64)))
        d_loo = float(np.linalg.norm(z_gen[pi] - mu_loo.astype(np.float64)))
        d_drift = float(np.linalg.norm(z_gen[pi] - np.asarray(src["z48_source"])))
        dF3 = float(np.linalg.norm(f3_gen[pi] - np.asarray(src["f3_source"]), ord=1))
        # clause dims delta
        clause_dims = clause_dims_f3()
        cd_delta = {}
        for cname in clause_dims:
            if cname in f3_idx:
                cd_delta[cname] = float(f3_gen[pi][f3_idx[cname]] - src["f3_source"][f3_idx[cname]])
        n_tok_gen = len(gen_texts[pi].split())
        # dep-arc jaccard vs source
        jac = jaccard(dep_arc_set(src_docs[u]), dep_arc_set(out_docs[pi]))
        # token overlap
        toks_src = set(re.findall(r"[a-z]+", src["source"].lower()))
        toks_gen = set(re.findall(r"[a-z]+", rec["output"].lower()))
        tok_ov = len(toks_src & toks_gen) / max(1, len(toks_gen))
        audit.append({
            "user_id": u,
            "condition": rec["condition"],
            "k": rec["k"],
            "output": rec["output"],
            "M_LOO": float(d_other - d_loo),
            "M_full": float(d_other - d_full),
            "d_self_LOO": d_loo,
            "d_self_full": d_full,
            "d_other": d_other,
            "d_drift": d_drift,
            "dF3_l1": dF3,
            "dep_arc_jaccard": jac,
            "tok_overlap": tok_ov,
            "n_tok": n_tok_gen,
            "clause_delta": cd_delta,
        })

    # 9. Aggregate per condition
    log(f"\n=== RESULTS ===")
    agg = {}
    for cond in ("A_free", "C_rewrite"):
        rows = [a for a in audit if a["condition"] == cond]
        if not rows:
            continue
        m_loo = [r["M_LOO"] for r in rows]
        m_full = [r["M_full"] for r in rows]
        agg[cond] = {
            "n": len(rows),
            "M_LOO_med": float(np.median(m_loo)),
            "P_M_LOO_gt0": float(np.mean([m > 0 for m in m_loo])),
            "M_full_med": float(np.median(m_full)),
            "P_M_full_gt0": float(np.mean([m > 0 for m in m_full])),
            "d_drift_med": float(np.median([r["d_drift"] for r in rows])),
            "dF3_l1_med": float(np.median([r["dF3_l1"] for r in rows])),
            "dep_arc_jaccard_med": float(np.median([r["dep_arc_jaccard"] for r in rows])),
            "tok_overlap_med": float(np.median([r["tok_overlap"] for r in rows])),
        }
        log(f"  [{cond}] n={agg[cond]['n']}  M_LOO_med={agg[cond]['M_LOO_med']:.3f}  "
            f"P(M_LOO>0)={agg[cond]['P_M_LOO_gt0']*100:.1f}%")
        log(f"           d_drift_med={agg[cond]['d_drift_med']:.3f}  "
            f"dF3_med={agg[cond]['dF3_l1_med']:.1f}  "
            f"dep_jaccard={agg[cond]['dep_arc_jaccard_med']:.3f}  "
            f"tok_ov={agg[cond]['tok_overlap_med']:.3f}")

    # reconciliation verdict
    rec_rows = [r for r in per_user]
    n_anchor_diff = sum(1 for r in rec_rows
                        if r["p_M_full_gt0"] < 0.5 and r["p_M_LOO_gt0"] >= 0.5)
    verdict = (
        f"{n_users_ok} users with positive source. P(M_full>0) vs P(M_LOO>0) "
        f"differ for {n_anchor_diff}/{len(rec_rows)} users → "
        + ("ANCHOR MISMATCH confirmed" if n_anchor_diff > 0
           else "no anchor mismatch (both low → selection bias)")
    )
    log(f"\n=== RECONCILIATION: {verdict} ===")

    out = {
        "config": {
            "phase": "7.B smoke",
            "users": USERS,
            "k_per_source": K_PER_SOURCE,
            "conditions": ["A_free", "C_rewrite"],
            "t_b": T_B,
            "cv_gate": [CV_INLIER_MIN, CV_NLL_MAX],
            "lambda": LAMBDA,
            "competitor": "6.E.4 same-ASIN + Q + BD-separated (T_B=2.0)",
            "anchors": {"M_full": "production mu_u (7.A)", "M_LOO": "LOO mu_{u,-s} (6.E.4)"},
        },
        "n_users": len(per_user),
        "per_user": [{k: v for k, v in r.items() if k not in ("z48_source", "f3_source")}
                     for r in per_user],
        "reconciliation_verdict": verdict,
        "per_condition_stats": agg,
        "audit": audit,
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\n  wrote → {RESULT_PATH}")
    log(f"\n=== DONE ===")


# small helpers to keep comp arrays around per user (populated in main)
_COMP_CACHE = {}


def loo_mu_for(u, per_user):
    """μ_LOO for user u: mean over that user's candidate z (computed in Step A)."""
    global _LOO_MU_CACHE
    return _LOO_MU_CACHE[u]


def comp_for(u, per_user):
    global _COMP_CACHE
    return _COMP_CACHE[u]


_LOO_MU_CACHE = {}


if __name__ == "__main__":
    main()
