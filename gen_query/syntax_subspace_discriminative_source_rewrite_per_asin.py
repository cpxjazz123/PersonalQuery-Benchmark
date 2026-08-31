"""
Phase 7.J.1 — Per-ASIN-Cohort Rewrite Test (smoke)

Uses 7.I's EXACT evaluator definitions to legitimately answer the question
"does LLM rewrite destroy user syntax?":

  - μ_u = mean(z_all_u)   [user's ALL history sentences, excluding anchor ASIN a]
  - R_95 = 95th pct of d_self(z_all_u), fallback 8.073 if n<5
  - cohort = other users who reviewed ASIN a  [per-ASIN, not merged]

Pipeline:
  1. Load 7.I discriminative pool (52 users / 1255 sents)
  2. For each user u, re-scan reviews and rebuild z_all_u (excluding anchor ASIN)
  3. Per-ASIN: gather cohort user μ's (other selected users who reviewed same ASIN)
  4. Re-evaluate source disc sents through this evaluator → expect ≈100% M>0
  5. LLM rewrite (C_rewrite 4-rule) on top-K sents per user
  6. Evaluate rewrites through SAME evaluator → compare M_pass source vs M_pass rewrite

Smoke: 52 users × 3 sents = 156 prompts. Filter to users with ≥2 cohort peers.
Targets: source P(M>0) ≈ 100%, then RR (rewrite) measured honestly.
"""

import collections
import gzip
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


# ===== constants =====
K_PER_USER = 3
MIN_TOKENS = 8
MAX_TOKENS = 60
MIN_QUERY_TOKENS = 6
MAX_QUERY_TOKENS = 60
R_95_THEORETICAL = 8.073
MIN_COHORT_PEERS = 1  # require at least 1 other selected user on same ASIN
MIN_Z_ALL_SENTS = 2  # require ≥2 z_all to fit μ_u (else skip)
MIN_HIST_TOKENS = 6  # for building z_all from history (relaxed from 8 to capture short sentences)

# Inputs
DISC_POOL = Path(
    "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/phase7i_discriminative_source_pool.json"
)
PATTRS = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/product_attributes.json")
SCALER_NPZ = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_shared_scaler.npz")
DATA_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/data")
RAW_REVIEWS = DATA_ROOT / "Baby_Products_2023.jsonl"
REVIEWS_GZ = DATA_ROOT / "Baby_Products_2023.jsonl.gz"

# vLLM
VLLM_PORT = 8800
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 80

# Bad attribute keys (consistent with 7.E / 7.J)
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
RESULT_PATH = RESULT_DIR / "phase7j1_per_asin_rewrite.json"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [7J1] {msg}", flush=True)


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
    log("=== Phase 7.J.1 — Per-ASIN-Cohort Rewrite Test ===")
    log(f"  K_PER_USER={K_PER_USER}, MIN_COHORT_PEERS={MIN_COHORT_PEERS}")
    log("  Uses 7.I's exact evaluator: μ_u = mean(z_all_u), cohort = per-ASIN peers.")

    # ---- Load 7.I pool ----
    pool = json.load(open(DISC_POOL))
    selected_users = pool["selected_users"]
    user_asin = {u["uid"]: u["asin"] for u in selected_users}
    asin_to_users = collections.defaultdict(list)
    for u in selected_users:
        asin_to_users[u["asin"]].append(u["uid"])
    log(f"  loaded {len(selected_users)} users across {len(asin_to_users)} ASINs")
    log(f"  ASIN sizes: min={min(len(v) for v in asin_to_users.values())}, "
        f"max={max(len(v) for v in asin_to_users.values())}, "
        f"median={int(np.median([len(v) for v in asin_to_users.values()]))}")

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

    # ---- Re-scan reviews for ALL selected users (mirror 7.I exactly) ----
    target_user_set = set(user_asin.keys())
    target_asin_set = set(user_asin.values())
    user_reviews = collections.defaultdict(list)
    log(f"  scanning reviews for {len(target_user_set)} users...")
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
                log(f"    {n_records/1e6:.1f}M scanned, {len(user_reviews)}/{len(target_user_set)} users seen")
    log(f"  scanned {n_records}, found reviews for {len(user_reviews)}/{len(target_user_set)} users")

    # ---- Build z_all_u per user (exclude anchor ASIN a) and fit μ_u, R_95 ----
    log("  building z_all_u (excluding anchor ASIN) per user...")
    user_z_all = {}      # uid -> np.ndarray (n, 48)
    user_mu = {}         # uid -> np.ndarray (48,)
    user_R95 = {}        # uid -> float
    user_asin_for = {}   # uid -> anchor ASIN
    for uid, anchor_asin in user_asin.items():
        user_asin_for[uid] = anchor_asin
        revs = user_reviews.get(uid, [])
        sents = []
        for t, a in revs:
            if a == anchor_asin:
                continue
            for s in split_sentences(t):
                toks = s.split()
                if MIN_HIST_TOKENS <= len(toks) <= MAX_TOKENS:
                    sents.append(s)
        if len(sents) < MIN_Z_ALL_SENTS:
            continue
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
        if len(zs) < MIN_Z_ALL_SENTS:
            continue
        Z = np.asarray(zs, dtype=np.float64)
        mu = Z.mean(axis=0)
        if len(zs) >= 5:
            d_self = np.linalg.norm(Z - mu, axis=1)
            R_95 = float(np.quantile(d_self, 0.95))
        else:
            R_95 = R_95_THEORETICAL
        user_z_all[uid] = Z
        user_mu[uid] = mu
        user_R95[uid] = R_95
    log(f"  fitted μ_u for {len(user_mu)} users (had z_all ≥ {MIN_Z_ALL_SENTS})")

    # ---- Filter to users with ≥ MIN_COHORT_PEERS same-ASIN selected peers ----
    eligible_users = []
    for uid in user_mu:
        a = user_asin_for[uid]
        peers = [v for v in asin_to_users.get(a, []) if v != uid and v in user_mu]
        if len(peers) >= MIN_COHORT_PEERS:
            eligible_users.append((uid, a, peers))
    log(f"  eligible users (≥{MIN_COHORT_PEERS} cohort peer): {len(eligible_users)}")

    if not eligible_users:
        log("  NO eligible users — abort.")
        return

    # ---- Pre-compute per-ASIN cohort mu stacks ----
    asin_cohort_mu = {}
    for uid, a, peers in eligible_users:
        stack = np.stack([user_mu[uid]] + [user_mu[p] for p in peers], axis=0)
        asin_cohort_mu[a] = stack  # rows: [self, peer1, peer2, ...]
    asin_self_idx = {a: 0 for a in asin_cohort_mu}  # self always at row 0

    # ---- Helper: per-ASIN evaluation ----
    def eval_z(z, uid):
        """Returns (M, d_self, d_other_min, R_95, cohort_size) using per-ASIN cohort."""
        a = user_asin_for[uid]
        mu_u = user_mu[uid]
        cohort = asin_cohort_mu[a]  # row 0 = self
        d_all = np.linalg.norm(cohort - z, axis=1)
        self_idx = asin_self_idx[a]
        d_self = float(d_all[self_idx])
        d_other = float(np.min(np.delete(d_all, self_idx)))
        M = d_other - d_self
        return M, d_self, d_other, user_R95[uid], cohort.shape[0] - 1

    # ---- Step A: source identity check (re-eval disc sents through this evaluator) ----
    log("  [A] source identity: re-evaluating disc sents through per-ASIN evaluator...")
    src_M_pass = 0
    src_d_self_pass = 0
    src_strict = 0
    src_n = 0
    for uid, a, peers in eligible_users:
        for s in next(u for u in selected_users if u["uid"] == uid)["discriminative_sentences"]:
            z = project_query(s["text"], nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                              pca_components, pca_mean, psf)
            if z is None:
                continue
            M, d_self, d_other, R_95, n_peers = eval_z(z, uid)
            src_n += 1
            if M > 0:
                src_M_pass += 1
            if d_self <= R_95:
                src_d_self_pass += 1
            if M > 0 and d_self <= R_95:
                src_strict += 1
    log(f"  [A] source P(M>0)={src_M_pass}/{src_n} = {src_M_pass/src_n:.1%}, "
        f"P(d_self≤R_95)={src_d_self_pass}/{src_n} = {src_d_self_pass/src_n:.1%}, "
        f"STRICT={src_strict}/{src_n} = {src_strict/src_n:.1%}")

    # ---- Step B: rewrite with C_rewrite ----
    log(f"  [B] building rewrite prompts (top-{K_PER_USER} disc sents per eligible user)...")
    work = []
    for uid, a, peers in eligible_users:
        attrs = get_top_n_attrs(a, pattrs, n=5)
        if not attrs:
            continue
        u_doc = next(u for u in selected_users if u["uid"] == uid)
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
                "prompt": prompt,
            })
    log(f"  [B] built {len(work)} prompts ({len({w['uid'] for w in work})} users × {K_PER_USER} sents)")

    log(f"  [B] calling vLLM...")
    t0 = time.time()
    outputs = vllm_generate([w["prompt"] for w in work])
    log(f"  [B] vLLM done in {time.time()-t0:.1f}s")

    # ---- Step C: evaluate rewrites ----
    log("  [C] evaluating rewrites...")
    rw_n = 0
    rw_valid = 0
    rw_attr_pass = 0
    rw_M_pass = 0
    rw_d_self_pass = 0
    rw_strict = 0
    per_user = collections.defaultdict(lambda: {
        "n": 0, "n_valid": 0, "n_M_pass": 0, "n_d_self_pass": 0,
        "n_strict": 0, "n_attr_pass": 0,
    })
    rows = []
    for w, out in zip(work, outputs):
        uid = w["uid"]
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
        rw_n += 1
        rw_valid += 1
        per_user[uid]["n_valid"] += 1
        cov = attr_coverage(rewrite, w["attrs"])
        attr_pass = cov >= 0.95
        if attr_pass:
            rw_attr_pass += 1
            per_user[uid]["n_attr_pass"] += 1
        M, d_self, d_other, R_95, n_peers = eval_z(z, uid)
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
            "n_cohort_peers": n_peers,
            "attr_pass": attr_pass,
            "M_pass": M_pass,
            "d_self_pass": d_self_pass,
            "strict_pass": strict,
        })

    summary = {
        "config": {
            "description": ("Phase 7.J.1 per-ASIN-cohort rewrite test. "
                            "μ_u = mean(z_all_u), cohort = same-ASIN peers (7.I def)."),
            "K_PER_USER": K_PER_USER,
            "MIN_COHORT_PEERS": MIN_COHORT_PEERS,
            "MIN_Z_ALL_SENTS": MIN_Z_ALL_SENTS,
            "R_95_THEORETICAL": R_95_THEORETICAL,
        },
        "n_selected_users_total": len(selected_users),
        "n_users_fitted": len(user_mu),
        "n_eligible_users": len(eligible_users),
        "n_prompts_total": len(work),
        "source_identity_check": {
            "n_sents": src_n,
            "P_M_pos": src_M_pass / src_n if src_n else 0.0,
            "P_d_self_le_R95": src_d_self_pass / src_n if src_n else 0.0,
            "P_strict": src_strict / src_n if src_n else 0.0,
        },
        "rewrite_eval": {
            "n_valid": rw_valid,
            "n_attr_pass": rw_attr_pass,
            "n_M_pass": rw_M_pass,
            "n_d_self_pass": rw_d_self_pass,
            "n_strict": rw_strict,
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
    log(f"  eligible users: {len(eligible_users)} / fitted {len(user_mu)} / pool {len(selected_users)}")
    log(f"  prompts: {len(work)}, valid: {rw_valid}")
    log(f"  [A] SOURCE P(M>0)={src_M_pass/src_n:.1%}, STRICT={src_strict/src_n:.1%} "
        f"(sanity; should be ≈100%)")
    log(f"  [B-C] REWRITE P(M>0)={rw_M_pass}/{rw_valid}={rw_M_pass/rw_valid:.1%}, "
        f"P(d_self≤R_95)={rw_d_self_pass}/{rw_valid}={rw_d_self_pass/rw_valid:.1%}, "
        f"STRICT={rw_strict}/{rw_valid}={rw_strict/rw_valid:.1%}")
    log(f"  Δ (rewrite - source) M_pass: {rw_M_pass/rw_valid - src_M_pass/src_n:+.1%}")
    log(f"  Δ (rewrite - source) STRICT: {rw_strict/rw_valid - src_strict/src_n:+.1%}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
