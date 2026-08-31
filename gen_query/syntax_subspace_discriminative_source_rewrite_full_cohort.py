"""
Phase 7.J.3 — Full cv_users + per-ASIN Rewrite Test

Faithfully reproduces 7.I's evaluator by loading 80 target ASINs (not just
40 anchor ASINs of selected users) and the full per-ASIN cohort from
`asin_to_qual[a][:USERS_PER_ASIN]`. This gives every selected user enough
z_all sentences AND enough cohort peers to be evaluated legitimately.

Step 1: SOURCE IDENTITY (re-evaluate 7.I gold disc sents)
  Expected P(M > 0) ≈ 100% (sanity gate; must pass before rewrite eval)

Step 2: REWRITE (C_rewrite 4-rule, top-K disc sents per user)
  Compute M_rewrite and d_self_rewrite

Step 3: REWRITE RETENTION RATE (RR)
  RR = # rewrites with M>0 AND d_self ≤ R_95 / # valid rewrites
"""

import collections
import gzip
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


# ===== constants =====
# SMOKE: small cohort to verify evaluator reproduces 7.I's identity.
# Toggle by SMOKE_MODE.
import os as _os
SMOKE_MODE = bool(int(_os.environ.get("PHASE7J3_SMOKE", "1")))
K_PER_USER = 3
MIN_TOKENS = 8
MAX_TOKENS = 60
MIN_QUERY_TOKENS = 6
MAX_QUERY_TOKENS = 60
R_95_THEORETICAL = 8.073
USERS_PER_ASIN = 12       # mirror 7.I
N_ASIN = 1 if SMOKE_MODE else 80               # mirror 7.I
TARGET_RR = 0.80
ATTR_COVERAGE_MIN = 0.95

# Inputs
DISC_POOL = Path(
    "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/phase7i_discriminative_source_pool.json"
)
CV_SLIM = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_cv_users_slim.json")
ASINS_PATH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json")
PATTRS = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/product_attributes.json")
SCALER_NPZ = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_shared_scaler.npz")
DATA_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/data")
RAW_REVIEWS = DATA_ROOT / "Baby_Products_2023.jsonl"
REVIEWS_GZ = DATA_ROOT / "Baby_Products_2023.jsonl.gz"

# Q-gate (mirror 7.I)
CV_INLIER_MIN = 0.85
CV_NLL_MAX = 34.0
N_REVIEWS_MIN = 15

# vLLM
VLLM_PORT = 8800
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 80

BAD_ATTR_KEYS = frozenset({
    "domestic shipping", "international shipping", "country of origin",
    "is discontinued by manufacturer", "manufacturer", "item model number",
    "package dimensions", "item weight", "department", "asin",
    "date first available", "best sellers rank", "customer reviews",
    "shipping weight", "shipping", "manufacturer recommended age",
    "batteries required", "batteries included", "imported",
})

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
RESULT_DIR = REPO_ROOT / "result" / "gen_query"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = RESULT_DIR / "phase7j3_full_cohort_rewrite.json"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [7J3] {msg}", flush=True)


def split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def attrs_to_str(attrs):
    return "; ".join(attrs) if attrs else "(no attributes available)"


def get_top_n_attrs(asin: str, pattrs: dict, n: int = 5):
    pa = pattrs.get(asin, {})
    items = []
    for k, v in pa.items():
        if not isinstance(v, str):
            continue
        if any(c.isdigit() for c in v):
            continue
        if k.lower() in BAD_ATTR_KEYS:
            continue
        items.append(f"{k}: {v}")
    return items[:n]


def attr_coverage(query: str, attrs):
    if not attrs:
        return 0.0
    q_low = query.lower()
    hit = 0
    for a in attrs:
        val = a.split(":", 1)[1].strip().lower() if ":" in a else a.lower()
        if len(val) >= 4 and val[:8] in q_low:
            hit += 1
    return hit / len(attrs)


def first_word_check(query: str) -> bool:
    q_low = query.strip().lower()
    return not q_low.startswith(("here:", "brand:", "attribute:", "product:", "query:", "answer:"))


def project_query(query: str, nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                  pca_components, pca_mean, psf):
    sents = split_sentences(query)
    valid = [s for s in sents if MIN_QUERY_TOKENS <= len(s.split()) <= MAX_QUERY_TOKENS]
    if not valid:
        return None
    text = valid[0]
    doc = next(nlp.pipe([text], batch_size=1, n_process=1))
    sds = list(doc.sents)
    if not sds:
        return None
    feats = [psf(s) for s in sds if psf(s) is not None]
    if not feats:
        return None
    all_keys = set()
    for f in feats:
        all_keys.update(f.keys())
    mean_feats = {}
    for k in all_keys:
        vals = [f.get(k, 0.0) for f in feats]
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            try:
                mean_feats[k] = float(np.mean(vals))
            except (TypeError, ValueError):
                pass
    vec_full = np.array([mean_feats.get(nm, 0.0) for nm in all_fnames], dtype=np.float64)
    vec = vec_full[col_idx]
    vec_std = (vec - scaler_mean) / scaler_scale
    z48 = (vec_std - pca_mean) @ pca_components.T
    return z48


def build_prompt_rewrite(attrs, source_sentence: str) -> str:
    return (
        "You are a real parent looking for a product on Amazon.\n"
        f"The product has these attributes: {attrs_to_str(attrs)}.\n"
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
    url = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    results = [None] * len(prompts)
    MAX_RETRIES = 3

    def _call(idx_p):
        idx, p = idx_p
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": p}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_NEW_TOKENS,
            "top_p": 0.95,
        }
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=60)
                r.raise_for_status()
                results[idx] = r.json()["choices"][0]["message"]["content"].strip()
                return
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    results[idx] = f"ERROR: {e}"
                else:
                    time.sleep(2.0 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_call, list(enumerate(prompts))))
    return results


def main():
    log("=== Phase 7.J.3 — Full cv_users + per-ASIN Rewrite Test ===")
    log("  Faithful 7.I reproduction: μ_u from z_all (incl. anchor), per-ASIN cohort.")

    # ---- Load 7.I pool (for selected_users list, anchor ASIN) ----
    pool = json.load(open(DISC_POOL))
    selected_users = pool["selected_users"]
    selected_uids = {u["uid"]: u for u in selected_users}
    log(f"  loaded 7.I pool: {len(selected_users)} selected users")

    # ---- Load cv_users + asins ----
    log(f"  loading {CV_SLIM.name}")
    cv_users = json.load(open(CV_SLIM))
    log(f"  loaded slim cv_users: {len(cv_users)} users")
    log(f"  loading {ASINS_PATH.name}")
    asins_doc = json.load(open(ASINS_PATH))
    # 7.I's structure: asins_doc["asins"] = [{asin, users_sampled: [...]}, ...]
    asin_to_users_raw = {}
    if isinstance(asins_doc.get("asins"), list):
        for entry in asins_doc["asins"]:
            a = entry.get("asin")
            ulist = entry.get("users_sampled", entry.get("users", []))
            if a and ulist:
                asin_to_users_raw[a] = list(ulist)
    else:
        asin_to_users_raw = asins_doc.get("asin_to_users", {})
    log(f"  loaded {len(asin_to_users_raw)} ASINs from asins doc")
    # Build selected_uid → anchor ASIN map from 7.I pool
    selected_anchor = {u["uid"]: u["asin"] for u in selected_users}
    # 7.I uses anchor ASINs from its 52-user pool.  Expand cohort to ASINs that
    # contain at least one 7.I-selected user AND have ≥USERS_PER_ASIN qual users.
    asin_to_qual = {}
    for a, ulist in asin_to_users_raw.items():
        qual = []
        for u in ulist:
            cv = cv_users.get(u)
            if cv is None:
                continue
            inl = cv.get("inlier_frac", cv.get("cv_inlier_frac"))
            nll = cv.get("cv_nll")
            nrev = cv.get("n_reviews", 0)
            if inl is None or nll is None:
                continue
            if inl < CV_INLIER_MIN or nll > CV_NLL_MAX or nrev < N_REVIEWS_MIN:
                continue
            qual.append(u)
        if len(qual) >= USERS_PER_ASIN:
            asin_to_qual[a] = qual
    # Anchor ASINs = union of selected_users' anchors
    anchor_asins = set(selected_anchor.values())
    # target_asins: 7.I anchor ASINs that ALSO have ≥USERS_PER_ASIN qual users
    # (so we have a meaningful per-ASIN cohort)
    target_asins_pool = sorted(
        [a for a in anchor_asins if a in asin_to_qual],
        key=lambda a: -len(asin_to_qual[a]),
    )
    target_asins = target_asins_pool[:N_ASIN]
    log(f"  ASINs with >={USERS_PER_ASIN} qual users: {len(asin_to_qual)}")
    log(f"  7.I selected user anchor ASINs: {len(anchor_asins)}")
    log(f"  anchors with >={USERS_PER_ASIN} qual cohort: {len(target_asins_pool)}")
    log(f"  selected {len(target_asins)} target ASINs (anchor-priority)")
    for a in target_asins[:5]:
        log(f"    {a}: {len(asin_to_qual[a])} qual users")

    # Build target user set = all users in any of the selected ASINs' cohorts
    # PLUS the 7.I selected_users (they must appear in their anchor ASIN's cohort)
    target_user_set = set()
    asin_to_cohort = {}
    for a in target_asins:
        cohort = list(asin_to_qual[a][:USERS_PER_ASIN])
        # ensure 7.I selected user for this ASIN is in cohort
        sel_for_a = [u["uid"] for u in selected_users if u["asin"] == a]
        for su in sel_for_a:
            if su not in cohort:
                cohort.insert(0, su)
        # dedupe preserving order
        seen = set()
        cohort = [u for u in cohort if not (u in seen or seen.add(u))]
        asin_to_cohort[a] = cohort[:USERS_PER_ASIN]
        target_user_set.update(cohort)
    log(f"  cohort users (unique): {len(target_user_set)}")

    # ---- Load attrs ----
    pattrs = json.load(open(PATTRS))

    # ---- Load scaler/PCA48 ----
    gdoc = np.load(SCALER_NPZ, allow_pickle=True)
    all_fnames = list(gdoc["feature_names_ordered"])
    fnames_f3 = list(gdoc["fnames_f3"])
    scaler_mean = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
    pca_components = np.asarray(gdoc["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gdoc["pca_mean"], dtype=np.float64)
    gdoc.close()
    col_idx = [all_fnames.index(nm) for nm in fnames_f3]

    import spacy
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2 as psf
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    # ---- Re-scan reviews for ALL cohort users on ALL target ASINs ----
    target_asin_set = set(target_asins)
    user_reviews = collections.defaultdict(list)
    log(f"  scanning reviews for {len(target_user_set)} users × {len(target_asin_set)} ASINs...")
    n_records = 0
    if RAW_REVIEWS.exists():
        open_fn = lambda: open(RAW_REVIEWS, "r", encoding="utf-8")
    else:
        open_fn = lambda: gzip.open(REVIEWS_GZ, "rt", encoding="utf-8")
    with open_fn() as f:
        for line in f:
            r = json.loads(line)
            uid = r.get("reviewerID") or r.get("user_id")
            asin = r.get("asin") or r.get("parent_asin")
            if uid in target_user_set and asin in target_asin_set and r.get("text"):
                user_reviews[uid].append((r["text"], asin))
            n_records += 1
            if n_records % 2_000_000 == 0:
                log(f"    {n_records/1e6:.1f}M scanned, "
                    f"{len(user_reviews)}/{len(target_user_set)} users seen")
    log(f"  scanned {n_records}, found reviews for {len(user_reviews)}/{len(target_user_set)} users")

    # ---- Build z_all per cohort user ----
    log("  building z_all per cohort user...")
    user_z_all = {}
    for uid in target_user_set:
        revs = user_reviews.get(uid, [])
        if not revs:
            continue
        sents = []
        for t, _ in revs:
            for s in split_sentences(t):
                toks = s.split()
                if MIN_TOKENS <= len(toks) <= MAX_TOKENS:
                    sents.append(s)
        zs = []
        seen = set()
        for s in sents:
            k = s.strip().lower()
            if k in seen:
                continue
            seen.add(k)
            z = project_query(s, nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                              pca_components, pca_mean, psf)
            if z is not None:
                zs.append(z)
        if len(zs) >= 2:
            user_z_all[uid] = np.asarray(zs, dtype=np.float64)
    log(f"  z_all built for {len(user_z_all)} cohort users")

    # ---- Per-ASIN: collect μ_v from all cohort users on this ASIN, fit per user ----
    log("  fitting μ_u per ASIN cohort user...")
    user_mu = {}
    user_R95 = {}
    user_asin = {}
    for a in target_asins:
        cohort_uids = [u for u in asin_to_cohort[a] if u in user_z_all]
        if len(cohort_uids) < 2:
            continue
        # compute cohort_mu stack
        cohort_mu = np.stack([user_z_all[u].mean(axis=0) for u in cohort_uids], axis=0)
        uid_to_idx = {u: i for i, u in enumerate(cohort_uids)}
        for uid in cohort_uids:
            Z = user_z_all[uid]
            mu = Z.mean(axis=0)
            if len(Z) >= 5:
                d_self = np.linalg.norm(Z - mu, axis=1)
                R_95 = float(np.quantile(d_self, 0.95))
            else:
                R_95 = R_95_THEORETICAL
            user_mu[(uid, a)] = (mu, cohort_mu, uid_to_idx)
            user_R95[uid] = R_95
            user_asin[uid] = a

    # ---- Eligibility: selected user must be in user_mu AND have ≥2 cohort peers
    # cohort peers = other users on same ASIN (excluding self)
    eligible = []
    for u in selected_users:
        uid = u["uid"]
        a = u["asin"]
        if (uid, a) not in user_mu:
            continue
        _, cohort_mu, uid_to_idx = user_mu[(uid, a)]
        n_peers = cohort_mu.shape[0] - 1  # exclude self at uid_to_idx[uid]
        if n_peers < 1:
            continue
        eligible.append((uid, a, n_peers))
    log(f"  eligible selected users (have μ in their ASIN cohort + ≥1 peer): {len(eligible)}")

    # ---- Helper: per-ASIN eval ----
    def eval_z(z, uid, a):
        mu, cohort_mu, uid_to_idx = user_mu[(uid, a)]
        self_idx = uid_to_idx[uid]
        d_all = np.linalg.norm(cohort_mu - z, axis=1)
        d_self = float(d_all[self_idx])
        d_other_min = float(np.min(np.delete(d_all, self_idx)))
        return d_other_min - d_self, d_self, d_other_min, user_R95[uid]

    # ---- Step A: source identity ----
    log("  [A] source identity: re-evaluating disc sents through 7.I-faithful evaluator...")
    src_M_pass = 0
    src_d_self_pass = 0
    src_strict = 0
    src_n = 0
    for uid, a, n_peers in eligible:
        u_doc = selected_uids[uid]
        for s in u_doc["discriminative_sentences"]:
            z = project_query(s["text"], nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                              pca_components, pca_mean, psf)
            if z is None:
                continue
            M, d_self, d_other, R_95 = eval_z(z, uid, a)
            src_n += 1
            if M > 0:
                src_M_pass += 1
            if d_self <= R_95:
                src_d_self_pass += 1
            if M > 0 and d_self <= R_95:
                src_strict += 1
    log(f"  [A] P(M>0)={src_M_pass}/{src_n} = {src_M_pass/src_n:.1%}, "
        f"P(d_self≤R_95)={src_d_self_pass}/{src_n} = {src_d_self_pass/src_n:.1%}, "
        f"STRICT={src_strict}/{src_n} = {src_strict/src_n:.1%}")

    # ---- Gate: refuse rewrite if source identity fails ----
    P_src = src_strict / src_n if src_n else 0.0
    if P_src < 0.80:
        log(f"  [GATE] source identity {P_src:.1%} < 80% — aborting before rewrite.")
        out = {
            "summary": {
                "config": {"description": "7.J.3 aborted: source identity < 80%"},
                "n_eligible_users": len(eligible),
                "source_identity_check": {
                    "n_sents": src_n,
                    "P_strict": P_src,
                },
                "rewrite_eval": None,
            }
        }
        with open(RESULT_PATH, "w") as f:
            json.dump(out, f, indent=2)
        log(f"  wrote → {RESULT_PATH}")
        return

    # ---- Step B: build rewrite prompts ----
    log(f"  [B] building rewrite prompts (top-{K_PER_USER} disc sents per eligible user)...")
    work = []
    for uid, a, n_peers in eligible:
        attrs = get_top_n_attrs(a, pattrs, n=5)
        if not attrs:
            continue
        u_doc = selected_uids[uid]
        sents_sorted = sorted(u_doc["discriminative_sentences"], key=lambda s: -s["M"])[:K_PER_USER]
        for s in sents_sorted:
            prompt = build_prompt_rewrite(attrs, s["text"])
            work.append({
                "uid": uid,
                "asin": a,
                "attrs": attrs,
                "source_text": s["text"],
                "source_M": s["M"],
                "source_d_self": s["d_self"],
                "n_cohort_peers": n_peers,
                "prompt": prompt,
            })
    log(f"  [B] built {len(work)} prompts ({len({w['uid'] for w in work})} users × {K_PER_USER} sents)")

    log(f"  [B] calling vLLM ({len(work)} prompts)...")
    t0 = time.time()
    outputs = vllm_generate([w["prompt"] for w in work])
    log(f"  [B] vLLM done in {time.time()-t0:.1f}s")

    # ---- Step C: rewrite eval ----
    log("  [C] evaluating rewrites...")
    rw_n = 0
    rw_valid = 0
    rw_M_pass = 0
    rw_d_self_pass = 0
    rw_strict = 0
    rw_attr_pass = 0
    per_user = collections.defaultdict(lambda: {
        "n": 0, "n_valid": 0, "n_M_pass": 0, "n_d_self_pass": 0,
        "n_strict": 0, "n_attr_pass": 0,
    })
    rows = []
    for w, out in zip(work, outputs):
        uid = w["uid"]
        a = w["asin"]
        per_user[uid]["n"] += 1
        if out is None or out.startswith("ERROR"):
            rows.append({**w, "rewrite": None, "valid": False, "reason": "vllm_error"})
            continue
        rewrite = out.strip()
        n_tok = len(rewrite.split())
        if not (MIN_QUERY_TOKENS <= n_tok <= MAX_QUERY_TOKENS):
            rows.append({**w, "rewrite": rewrite, "valid": False, "reason": f"len={n_tok}"})
            continue
        if not first_word_check(rewrite):
            rows.append({**w, "rewrite": rewrite, "valid": False, "reason": "first_word"})
            continue
        z = project_query(rewrite, nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                          pca_components, pca_mean, psf)
        if z is None:
            rows.append({**w, "rewrite": rewrite, "valid": False, "reason": "project_fail"})
            continue
        rw_valid += 1
        per_user[uid]["n_valid"] += 1
        cov = attr_coverage(rewrite, w["attrs"])
        attr_pass = cov >= ATTR_COVERAGE_MIN
        if attr_pass:
            rw_attr_pass += 1
            per_user[uid]["n_attr_pass"] += 1
        M, d_self, d_other, R_95 = eval_z(z, uid, a)
        M_pass = M > 0
        d_self_pass = d_self <= R_95
        strict = M_pass and d_self_pass
        if M_pass:
            rw_M_pass += 1
            per_user[uid]["n_M_pass"] += 1
        if d_self_pass:
            rw_d_self_pass += 1
            per_user[uid]["n_d_self_pass"] += 1
        if strict:
            rw_strict += 1
            per_user[uid]["n_strict"] += 1
        rows.append({
            **w,
            "rewrite": rewrite,
            "valid": True,
            "n_tokens": n_tok,
            "attr_coverage": cov,
            "M_query": M,
            "d_self_query": d_self,
            "d_other_min_query": d_other,
            "R_95_user": R_95,
            "attr_pass": attr_pass,
            "M_pass": M_pass,
            "d_self_pass": d_self_pass,
            "strict_pass": strict,
        })

    summary = {
        "config": {
            "description": ("Phase 7.J.3: full cv_users + per-ASIN cohort rewrite test. "
                            "Faithful 7.I reproduction."),
            "K_PER_USER": K_PER_USER,
            "N_ASIN": N_ASIN,
            "USERS_PER_ASIN": USERS_PER_ASIN,
            "ATTR_COVERAGE_MIN": ATTR_COVERAGE_MIN,
            "TARGET_RR": TARGET_RR,
            "R_95_THEORETICAL": R_95_THEORETICAL,
        },
        "n_selected_users": len(selected_users),
        "n_target_asins": len(target_asins),
        "n_cohort_users_total": len(target_user_set),
        "n_cohort_users_with_z_all": len(user_z_all),
        "n_eligible_users": len(eligible),
        "n_prompts_total": len(work),
        "source_identity_check": {
            "n_sents": src_n,
            "P_M_pos": src_M_pass / src_n if src_n else 0.0,
            "P_d_self_le_R95": src_d_self_pass / src_n if src_n else 0.0,
            "P_strict": P_src,
        },
        "rewrite_eval": {
            "n_valid": rw_valid,
            "n_M_pass": rw_M_pass,
            "n_d_self_pass": rw_d_self_pass,
            "n_strict": rw_strict,
            "n_attr_pass": rw_attr_pass,
            "RR_strict": rw_strict / rw_valid if rw_valid else 0.0,
            "P_M_pos": rw_M_pass / rw_valid if rw_valid else 0.0,
            "P_d_self_le_R95": rw_d_self_pass / rw_valid if rw_valid else 0.0,
        },
        "per_user_summary": {
            uid: dict(stats) | {
                "RR": (stats["n_strict"] / stats["n_valid"]) if stats["n_valid"] else 0.0,
            }
            for uid, stats in per_user.items()
        },
    }
    out = {"summary": summary, "per_rewrite_rows": rows}
    log(f"  wrote → {RESULT_PATH}")
    with open(RESULT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    log("")
    log("=== OVERALL ===")
    log(f"  cohort users: {len(target_user_set)} | with z_all: {len(user_z_all)}")
    log(f"  eligible selected users: {len(eligible)}")
    log(f"  prompts: {len(work)}, valid: {rw_valid}")
    log(f"  [A] SOURCE P(M>0)={src_M_pass/src_n:.1%}, STRICT={src_strict/src_n:.1%}")
    log(f"  [C] REWRITE P(M>0)={rw_M_pass}/{rw_valid}={rw_M_pass/rw_valid:.1%}, "
        f"P(d_self≤R_95)={rw_d_self_pass/rw_valid:.1%}, "
        f"STRICT={rw_strict}/{rw_valid}={rw_strict/rw_valid:.1%}")
    log(f"  Δ (rewrite - source) M_pass: {rw_M_pass/rw_valid - src_M_pass/src_n:+.1%}")
    log(f"  Δ (rewrite - source) STRICT: {rw_strict/rw_valid - src_strict/src_n:+.1%}")
    log(f"  target RR ≥ {TARGET_RR:.0%}")
    if rw_strict / rw_valid >= TARGET_RR:
        log("  GO — RR meets target.")
    elif rw_strict / rw_valid >= 0.50:
        log("  PARTIAL — RR > 50%; investigate per-user breakdown.")
    else:
        log("  NO-GO — RR < 50%; LLM systematically destroys user syntax.")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
