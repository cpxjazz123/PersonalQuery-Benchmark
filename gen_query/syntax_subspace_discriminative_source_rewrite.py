"""
Phase 7.J — Discriminative-Source Rewrite (smoke)

For each of the 52 syntax-identifiable users (from 7.I), take top-K
discriminative source sentences (verified M>0 AND d_self<=R_95), rewrite them
into shopping queries for the user's anchor ASIN using a 4-rule "syntax-
preserving" prompt, and measure **Rewrite Retention Rate**:

    RR = # strict-pass rewritten queries / # valid rewrites

A strict-pass rewrite must satisfy BOTH:
  (a) Semantic quality:  attr coverage >= 95%, unsupported semantic content <= 5%
  (b) Syntax preservation: M_query > 0  AND  d_self_query <= R_95(user)

Smoke config: 52 users x 3 sentences = 156 prompts (1 trial each).
Targets: RR >= 80%, attr_complete >= 95%, unsupported <= 5%.

NOTE: 7.J is INTENTIONALLY in-sample (mirroring 7.I's cohort construction).
     The 7.I pool is the gold reference; 7.J asks whether LLM can rewrite
     toward that bar. NOT held-out generalization proof.
"""

import collections
import gzip
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


# ===== constants =====
SEED = 42
N_TRIALS_PER_SENT = 1              # smoke: 1 trial per source sentence
K_PER_USER = 3                      # smoke: 3 sents per user (52 x 3 = 156)
MIN_TOKENS = 8
MAX_TOKENS = 60
MIN_QUERY_TOKENS = 6
MAX_QUERY_TOKENS = 60
ATTR_COVERAGE_MIN = 0.95           # user goal 2026-08-31
UNSUPPORTED_MAX = 0.05
R_95_THEORETICAL = 8.073           # sqrt(chi2(0.95, 48))
TARGET_RR = 0.80                   # user goal 2026-08-31

# Discriminative-source pool (7.I output)
DISC_POOL = Path(
    "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/phase7i_discriminative_source_pool.json"
)
# Product attributes (per-ASIN dict)
PATTRS = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/product_attributes.json")
# PCA48 / scaler params (7.C.1 shared scaler cache, npz format)
SCALER_NPZ = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_shared_scaler.npz")

# Raw reviews
DATA_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/data")
RAW_REVIEWS = DATA_ROOT / "Baby_Products_2023.jsonl"
REVIEWS_GZ = DATA_ROOT / "Baby_Products_2023.jsonl.gz"

# vLLM
VLLM_PORT = 8800
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 80

# Bad attribute keys (consistent with 7.E)
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
RESULT_PATH = RESULT_DIR / "phase7j_discriminative_source_rewrite.json"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [7J] {msg}", flush=True)


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
    """Fraction of attribute VALUES (>=4 chars) that appear in the query."""
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


# ---------- C_rewrite prompt: syntax-preserving rewrite ----------
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


# ---------- vLLM ----------
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


# ---------- PCA48 projection (mirror 7.C.1 / 7.E) ----------
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
    # full-dim vector (matches all_fnames order); then subset to F3 (103d)
    vec_full = np.array([mean_feats.get(nm, 0.0) for nm in all_fnames], dtype=np.float64)
    vec = vec_full[col_idx]
    vec_std = (vec - scaler_mean) / scaler_scale
    z48 = (vec_std - pca_mean) @ pca_components.T
    return z48


# ---------- main ----------
def main():
    log("=== Phase 7.J — Discriminative-Source Rewrite (smoke) ===")
    log(f"  K_PER_USER={K_PER_USER}, N_TRIALS_PER_SENT={N_TRIALS_PER_SENT}")
    log(f"  targets: RR >= {TARGET_RR:.0%}, attr_complete >= {ATTR_COVERAGE_MIN:.0%}, unsupported <= {UNSUPPORTED_MAX:.0%}")

    # Load 7.I discriminative pool
    log(f"  loading {DISC_POOL.name}")
    pool = json.load(open(DISC_POOL))
    selected_users = pool["selected_users"]
    log(f"  loaded {len(selected_users)} syntax-identifiable users "
        f"with {pool['summary']['n_discriminative_sentences_total']} disc sents")

    # Load attrs
    log(f"  loading {PATTRS.name}")
    pattrs = json.load(open(PATTRS))

    # Load scaler + PCA48 params
    log(f"  loading {SCALER_NPZ.name}")
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

    # ---- Fit per-user μ_u (in-sample) from 7.I's confirmed disc sentence
    #      embeddings. This is the SAME μ_u that 7.I used to mark the
    #      sentences as discriminative (M>0 ∧ d_self ≤ R_95). 7.J asks
    #      whether LLM rewrites can hit that bar.
    log("  fitting per-user μ_u (in-sample) from disc sentence embeddings...")
    user_mu = {}
    user_R95 = {}
    for u in selected_users:
        disc_texts = [s["text"] for s in u["discriminative_sentences"]]
        zs = []
        for t in disc_texts:
            z = project_query(t, nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                              pca_components, pca_mean, psf)
            if z is not None:
                zs.append(z)
        if len(zs) < 2:
            continue
        Z = np.asarray(zs, dtype=np.float64)
        mu = Z.mean(axis=0)
        if len(zs) >= 5:
            d_self = np.linalg.norm(Z - mu, axis=1)
            R_95 = float(np.quantile(d_self, 0.95))
        else:
            R_95 = R_95_THEORETICAL
        user_mu[u["uid"]] = mu
        user_R95[u["uid"]] = R_95
    log(f"  fitted μ_u for {len(user_mu)} users")

    # cohort μ for M_query (other users = competitors)
    cohort_uids = [u["uid"] for u in selected_users if u["uid"] in user_mu]
    cohort_mu = np.stack([user_mu[uid] for uid in cohort_uids], axis=0)
    cohort_uid_idx = {uid: i for i, uid in enumerate(cohort_uids)}

    # ---- Build prompt list (top-K disc sents per user) ----
    log("  building rewrite prompts...")
    work = []
    for u in selected_users:
        uid = u["uid"]
        asin = u["asin"]
        if uid not in user_mu:
            continue
        attrs = get_top_n_attrs(asin, pattrs, n=5)
        if not attrs:
            continue
        sents_sorted = sorted(u["discriminative_sentences"], key=lambda s: -s["M"])[:K_PER_USER]
        for s in sents_sorted:
            prompt = build_prompt_rewrite(attrs, s["text"])
            work.append({
                "uid": uid,
                "asin": asin,
                "attrs": attrs,
                "source_text": s["text"],
                "source_M": s["M"],
                "source_d_self": s["d_self"],
                "source_d_other": s["d_other"],
                "prompt": prompt,
            })
    log(f"  built {len(work)} prompts "
        f"({len({w['uid'] for w in work})} users × {K_PER_USER} sents)")

    # ---- vLLM batched generate ----
    log(f"  calling vLLM ({len(work)} prompts, 16 workers)...")
    t0 = time.time()
    prompts = [w["prompt"] for w in work]
    outputs = vllm_generate(prompts)
    log(f"  vLLM done in {time.time()-t0:.1f}s")

    # ---- Eval each rewrite ----
    log("  evaluating rewrites (semantic + syntax)...")
    n_valid = 0
    n_attr_pass = 0
    n_unsupp_violations = 0
    n_M_pass = 0
    n_d_self_pass = 0
    n_strict = 0
    per_user = collections.defaultdict(lambda: {
        "n_prompts": 0, "n_valid": 0, "n_attr_pass": 0,
        "n_M_pass": 0, "n_d_self_pass": 0, "n_strict": 0,
    })
    rows = []

    for w, out in zip(work, outputs):
        uid = w["uid"]
        per_user[uid]["n_prompts"] += 1
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
        n_valid += 1
        per_user[uid]["n_valid"] += 1
        cov = attr_coverage(rewrite, w["attrs"])
        unsupported = 1.0 - cov  # upper-bound heuristic
        attr_pass = cov >= ATTR_COVERAGE_MIN
        unsupp_pass = unsupported <= UNSUPPORTED_MAX
        if attr_pass:
            n_attr_pass += 1
            per_user[uid]["n_attr_pass"] += 1
        if not unsupp_pass:
            n_unsupp_violations += 1
        # M_query against cohort peers
        mu_u = user_mu[uid]
        d_self = float(np.linalg.norm(z - mu_u))
        self_idx = cohort_uid_idx[uid]
        d_all = np.linalg.norm(cohort_mu - z, axis=1)
        d_other_min = float(np.min(np.delete(d_all, self_idx)))
        M = d_other_min - d_self
        R_95 = user_R95[uid]
        M_pass = M > 0
        d_self_pass = d_self <= R_95
        if M_pass:
            n_M_pass += 1
            per_user[uid]["n_M_pass"] += 1
        if d_self_pass:
            n_d_self_pass += 1
            per_user[uid]["n_d_self_pass"] += 1
        strict = attr_pass and unsupp_pass and M_pass and d_self_pass
        if strict:
            n_strict += 1
            per_user[uid]["n_strict"] += 1
        rows.append({
            **w,
            "rewrite": rewrite,
            "valid": True,
            "n_tokens": n_tok,
            "attr_coverage": cov,
            "unsupported": unsupported,
            "M_query": M,
            "d_self_query": d_self,
            "d_other_min_query": d_other_min,
            "R_95_user": R_95,
            "attr_pass": attr_pass,
            "unsupp_pass": unsupp_pass,
            "M_pass": M_pass,
            "d_self_pass": d_self_pass,
            "strict_pass": strict,
        })

    rr_overall = n_strict / n_valid if n_valid else 0.0
    summary = {
        "config": {
            "description": ("Phase 7.J smoke: discriminative-source rewrite retention. "
                            "In-sample μ_u (NOT held-out)."),
            "K_PER_USER": K_PER_USER,
            "N_TRIALS_PER_SENT": N_TRIALS_PER_SENT,
            "ATTR_COVERAGE_MIN": ATTR_COVERAGE_MIN,
            "UNSUPPORTED_MAX": UNSUPPORTED_MAX,
            "R_95_THEORETICAL": R_95_THEORETICAL,
            "TARGET_RR": TARGET_RR,
        },
        "n_users_in_pool": len(selected_users),
        "n_users_with_fit": len(user_mu),
        "n_prompts_total": len(work),
        "n_valid_rewrites": n_valid,
        "n_attr_pass": n_attr_pass,
        "n_unsupported_violations": n_unsupp_violations,
        "n_M_pass": n_M_pass,
        "n_d_self_pass": n_d_self_pass,
        "n_strict_pass": n_strict,
        "RR_overall_strict": rr_overall,
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
    log(f"  prompts: {len(work)}")
    log(f"  valid rewrites: {n_valid} ({n_valid/len(work):.0%})")
    log(f"  attr_complete >= 95%: {n_attr_pass} ({n_attr_pass/n_valid if n_valid else 0:.0%})")
    log(f"  unsupported violations: {n_unsupp_violations}")
    log(f"  M_query > 0: {n_M_pass} ({n_M_pass/n_valid if n_valid else 0:.0%})")
    log(f"  d_self <= R_95: {n_d_self_pass} ({n_d_self_pass/n_valid if n_valid else 0:.0%})")
    log(f"  STRICT pass: {n_strict} ({rr_overall:.0%})")
    log(f"  target: RR >= {TARGET_RR:.0%}")
    if rr_overall >= TARGET_RR:
        log("  GO — rewrite retention rate meets target; consider full 1255-sent run.")
    elif rr_overall >= 0.50:
        log("  PARTIAL — RR > 50%; investigate per-user breakdown before scaling.")
    else:
        log("  NO-GO — RR < 50%; LLM cannot reliably preserve user syntax even with 4-rule prompt.")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
