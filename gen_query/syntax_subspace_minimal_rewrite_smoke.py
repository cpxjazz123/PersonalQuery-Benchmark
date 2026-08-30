"""Phase 7 — Minimal syntax-preserving rewrite (SMOKE).

User directive 2026-08-31:
Phase 6 系列证明 LLM 从零生成无法 reach user-specific PCA48 location
(d_self ~5.7 vs real LOO 4.5, 27% too generic, P(M_local>0)=0%).

新方向: 不要从零生成, 而是
  s* = argmin_i ||z(s_i) - μ_u||  (centroid-nearest representative)
  保留原句的 function words + dependency structure + clause order
  只替换 content words (target ASIN attrs)

3 条件对照 (per (user, target_asin) pair):
  A. Free generation — Phase 6.D 基线 (no exemplar, no constraint)
  B. Exemplar prompting — 看 representative, 写新句子 (Phase 4.C-style)
  C. Syntax-preserving rewrite — 直接改 representative (minimal change)

关键 metrics:
  d_drift = ||z_rewrite - z_source||      (preservation)
  d_self  = ||z_rewrite - μ_u||           (user-specific)
  M_local = d_nearest_same_ASIN - d_self  (the REAL question)

Decision:
  GO          P(M_local>0) C > A by ≥20pp AND d_drift_C < d_drift_A
  PARTIAL-GO  P(M_local>0) C > A but d_drift_C ≈ d_drift_A
  NO-GO       C ≈ A in M_local
  STRONG NO-GO C ≈ B (rewrite no better than exemplar)

Smoke 配置 (N_SMOKE=5, K_PER_PAIR=2):
  5 users × 1 target_asin × 3 conditions × 2 candidates = 30 generations
  Expected runtime: <10 min (after vLLM ready)

Output:
  result/gen_query/phase7_rewrite_smoke.json  (paired audit)
"""
import gzip
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"
SENT_IN = SCRATCH / "gaussian_vades" / "sentences_for_rewrite_10k.jsonl"
ASINS_JSON = SCRATCH / "gaussian_vades" / "stage8_5_asins.json"
PATTRS_PATH = REPO_ROOT / "result" / "product_attributes.json"
PILOT_JSON = SCRATCH / "logs" / "phase6d_pilot.json"
LOG_PATH = SCRATCH / "logs" / "phase7_rewrite_smoke.log"
RESULT_PATH = REPO_ROOT / "result" / "gen_query" / "phase7_rewrite_smoke.json"

# Smoke config (Rule 3: hardcoded)
N_SMOKE = 5
K_PER_PAIR = 2
CONDITIONS = ["A_free", "B_exemplar", "C_rewrite"]
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 120
R_95 = 9.0
VLLM_PORT = 8800
MIN_WORDS = 8  # 排除过短句子
MAX_WORDS = 40  # 排除过长句子
CV_INLIER_MIN = 0.9
CV_NLL_MAX = 39.0


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] [phase7] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def project_query_to_pca48(query_text, nlp, all_fnames, fnames_sub, col_idx,
                            scaler_mean, scaler_scale, pca_components, pca_mean):
    """Extract F3 → sub → scale → PCA48."""
    doc = nlp(query_text)
    sents = list(doc.sents)
    if not sents:
        return None
    feats_per_sent = []
    for sent in sents:
        f = per_sentence_features_v2(sent)
        if f is not None:
            feats_per_sent.append(f)
    if not feats_per_sent:
        return None
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
    vec_full = np.array([mean_feats.get(n, 0.0) for n in all_fnames], dtype=np.float64)
    vec_sub = vec_full[col_idx]
    vec_sub_scaled = (vec_sub - scaler_mean) / scaler_scale
    candidate_z = (vec_sub_scaled - pca_mean) @ pca_components.T
    return candidate_z.astype(np.float32)


def per_sentence_features_v2(sent):
    sys.path.insert(0, str(REPO_ROOT))
    from common.syntactic_features import per_sentence_features_v2 as _psf
    return _psf(sent)


def get_top_n_attrs(asin, pattrs, n=5):
    """从 product_attributes.json (flat dict {asin: {key: value}}) 取 top-N attrs.

    Skip keys with numeric values per Rule 5 (Stage 1 EXCLUDE_NUMERIC_ATTRS).
    Prefer semantic keys (Brand/Color/Material/...).
    """
    SKIP_KEYS = {
        "Average Rating", "Price", "Rating Number", "Item model number",
        "Batteries required", "Is Discontinued By Manufacturer",
        "Date First Available", "Item Weight", "Maximum weight recommendation",
        "Minimum weight recommendation", "Package Dimensions", "Product Dimensions",
        "Number Of Items", "ASIN", "UPC",
    }
    NUMERIC_KW = {"price", "average rating", "rating number", "item weight",
                  "item model number", "date first available",
                  "package dimensions", "product dimensions",
                  "minimum weight recommendation", "maximum weight recommendation"}
    PREF = ["Brand", "Color", "Material", "Material Type", "Style",
            "Fabric Type", "Frame Material", "Pattern", "Theme", "Size",
            "Age Range (Description)", "Target gender", "Special Feature",
            "Main Category", "Manufacturer", "Harness type", "Form Factor",
            "Item Weight", "Shape"]

    pa = pattrs.get(asin, {})
    if not pa:
        return [f"<ATTR{i}>" for i in range(n)]
    out = []
    real = []
    for k, v in pa.items():
        if v is None:
            continue
        s = str(v).strip()
        if s in ("", "[]", "{}", "None"):
            continue
        if any(nk in k.lower() for nk in NUMERIC_KW):
            continue
        if k in SKIP_KEYS:
            continue
        real.append((k, v))
    real.sort(key=lambda kv: (PREF.index(kv[0]) if kv[0] in PREF else len(PREF), kv[0]))
    for k, v in real[:n]:
        out.append(f"{k}: {v}")
    while len(out) < n:
        out.append(f"<ATTR{len(out)}>")
    return out[:n]


def build_prompt_free(attrs):
    """A. Free generation — no exemplar, no constraint."""
    attrs_str = ", ".join(attrs)
    return (
        "You are a real parent looking for a product on Amazon.\n"
        f"The product has these attributes: {attrs_str}.\n"
        "Write a single natural shopping query (15-30 words) in your own voice.\n"
        "Be specific and personal. Output only the query."
    )


def build_prompt_exemplar(attrs, source_sentence):
    """B. Exemplar prompting — show representative, ask LLM to write similar new one."""
    attrs_str = ", ".join(attrs)
    return (
        "You are a real parent looking for a product on Amazon.\n"
        f"The product has these attributes: {attrs_str}.\n"
        "Here is an example of how you typically write:\n"
        f"  \"{source_sentence}\"\n"
        "Write a single new shopping query (15-30 words) in your own voice,\n"
        "matching the writing style of the example but referring to the new product.\n"
        "Output only the query."
    )


def build_prompt_rewrite(attrs, source_sentence):
    """C. Syntax-preserving rewrite — minimal change to source."""
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


PROMPT_BUILDERS = {
    "A_free": build_prompt_free,
    "B_exemplar": build_prompt_exemplar,
    "C_rewrite": build_prompt_rewrite,
}


def vllm_generate(prompts, model_name=None):
    """Batched vLLM generation via /v1/chat/completions (parallel ThreadPool)."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    if model_name is None:
        model_name = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
    url = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    results = [None] * len(prompts)
    BATCH = 32
    MAX_WORKERS = 32
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0

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
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                else:
                    log(f"    [vLLM ERROR] idx={idx}: {e}")
        return idx, ""

    t0 = time.time()
    n_done = 0
    for s in range(0, len(prompts), BATCH):
        chunk = list(enumerate(prompts[s:s+BATCH], start=s))
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(_call, item) for item in chunk]
            for f in futs:
                idx, txt = f.result()
                results[idx] = txt
        n_done += len(chunk)
        rate = n_done / (time.time() - t0 + 1e-9)
        log(f"    [vLLM] {n_done}/{len(prompts)} ({rate:.1f}/s)")
    return results


def main():
    log("=== Phase 7 — Minimal Syntax-Preserving Rewrite (SMOKE) ===")
    log(f"  N_SMOKE={N_SMOKE}, K_PER_PAIR={K_PER_PAIR}")
    log(f"  CONDITIONS={CONDITIONS}")
    log(f"  R_95={R_95}, T={TEMPERATURE}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}")

    # 1. Load pilot users
    log(f"\n  Loading pilot users from {PILOT_JSON}...")
    with open(PILOT_JSON) as f:
        pilot = json.load(f)
    pilot_users = [r["user_id"] for r in pilot["results"]]
    log(f"  pilot users: {len(pilot_users)}")

    # 2. Load PCA48 + scaler + user μ
    log(f"\n  Loading PCA + scaler from {GAUSS_PATH}...")
    with open(GAUSS_PATH) as f:
        gdoc = json.load(f)
    all_fnames = gdoc["feature_names_ordered"]
    fnames_sub = gdoc["fnames_f3"]
    scaler_mean = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
    pca_components = np.asarray(gdoc["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gdoc["pca_mean"], dtype=np.float64)
    col_idx = np.array([all_fnames.index(n) for n in fnames_sub], dtype=np.int64)
    user_mus = gdoc["users"]
    log(f"  loaded: scaler={scaler_mean.shape}, pca={pca_components.shape}, μ={len(user_mus)} users")

    # 3. Load sentences for each pilot user
    log(f"\n  Loading sentences from {SENT_IN}...")
    user_sentences = defaultdict(list)
    with open(SENT_IN) as f:
        for line in f:
            d = json.loads(line)
            if d["user_id"] in set(pilot_users):
                user_sentences[d["user_id"]].append(d)
    for u in pilot_users:
        log(f"    {u}: {len(user_sentences[u])} sentences, asins={set(s['asin'] for s in user_sentences[u])}")

    # 4. Load attrs
    log(f"\n  Loading product attributes...")
    with open(PATTRS_PATH) as f:
        pattrs = json.load(f)

    # 5. Load ASIN cohort for target asin selection
    log(f"\n  Loading ASIN cohort...")
    with open(ASINS_JSON) as f:
        asin_data = json.load(f)
    user_to_target_asins = {}
    for asin_rec in asin_data["asins"]:
        a = asin_rec["asin"]
        for u in asin_rec["users_sampled"]:
            user_to_target_asins.setdefault(u, []).append(a)

    # 6. For each pilot user, find representative sentence
    log(f"\n=== Step 1: Find representative sentence (argmin ||z(s) - μ_u||) ===")
    log(f"  Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    # Pick top-N_SMOKE users with at least 1 cohort asin different from primary
    user_primary = {}
    for u in pilot_users:
        sents = user_sentences[u]
        if sents:
            user_primary[u] = sents[0]["asin"]
    eligible = []
    for u in pilot_users:
        if u not in user_mus:
            continue
        cohort = user_to_target_asins.get(u, [])
        primary = user_primary.get(u)
        diff = [a for a in cohort if a != primary]
        if diff:
            eligible.append(u)
        if len(eligible) >= N_SMOKE:
            break
    log(f"  eligible users: {len(eligible)}, picked: {eligible}")

    # 7. Project each user's sentences to PCA48, find argmin to μ_u
    representative = {}
    for u in eligible:
        user_mu = np.asarray(user_mus[u]["mu"], dtype=np.float32)
        sents = user_sentences[u]
        log(f"\n  {u}: projecting {len(sents)} sentences...")
        best_d = float("inf")
        best_s = None
        best_z = None
        for s in sents:
            text = s["sentence_text"]
            wc = s["word_count"]
            if wc < MIN_WORDS or wc > MAX_WORDS:
                continue
            try:
                z = project_query_to_pca48(
                    text, nlp, all_fnames, fnames_sub, col_idx,
                    scaler_mean, scaler_scale, pca_components, pca_mean,
                )
            except Exception as e:
                continue
            if z is None:
                continue
            d = float(np.linalg.norm(z - user_mu))
            if d < best_d:
                best_d = d
                best_s = s
                best_z = z
        if best_s is None:
            log(f"    {u}: NO valid sentence found in range [{MIN_WORDS}, {MAX_WORDS}]")
            continue
        representative[u] = {
            "sentence_text": best_s["sentence_text"],
            "source_asin": best_s["asin"],
            "word_count": best_s["word_count"],
            "d_self": best_d,
            "z48": best_z.tolist(),
        }
        log(f"    {u}: argmin sentence: wc={best_s['word_count']}, d_self={best_d:.3f}, "
            f"asin={best_s['asin']}, text={best_s['sentence_text'][:120]}")

    # 8. For each user, pick 1 target asin from cohort
    pairs = []
    for u in eligible:
        if u not in representative:
            continue
        primary = user_primary.get(u)
        cohort = user_to_target_asins.get(u, [])
        diff = [a for a in cohort if a != primary]
        target_asin = diff[0] if diff else None
        if target_asin is None:
            continue
        attrs = get_top_n_attrs(target_asin, pattrs, n=5)
        pairs.append({
            "user_id": u,
            "target_asin": target_asin,
            "target_attrs": attrs,
            "representative": representative[u],
        })
    log(f"\n  Total (user, target_asin) pairs: {len(pairs)}")
    for p in pairs:
        log(f"    {p['user_id']} → {p['target_asin']}: attrs={p['target_attrs']}, rep d_self={p['representative']['d_self']:.3f}")

    # 9. Build prompts × 3 conditions × K_PER_PAIR candidates
    log(f"\n=== Step 2: Build prompts for {len(pairs)} pairs × 3 conditions × {K_PER_PAIR} ===")
    prompt_records = []  # (pair_idx, condition, cand_idx, prompt, source_sentence)
    for pi, p in enumerate(pairs):
        for cond in CONDITIONS:
            builder = PROMPT_BUILDERS[cond]
            for k in range(K_PER_PAIR):
                if cond == "A_free":
                    prompt = builder(p["target_attrs"])
                else:
                    prompt = builder(p["target_attrs"], p["representative"]["sentence_text"])
                prompt_records.append({
                    "pair_idx": pi,
                    "user_id": p["user_id"],
                    "target_asin": p["target_asin"],
                    "condition": cond,
                    "cand_idx": k,
                    "source_sentence": p["representative"]["sentence_text"],
                    "prompt": prompt,
                })
    log(f"  Total prompts to generate: {len(prompt_records)}")

    # 10. Batch vLLM generation
    log(f"\n=== Step 3: vLLM batch generation (T={TEMPERATURE}) ===")
    log(f"  Checking vLLM health...")
    import requests
    try:
        r = requests.get(f"http://localhost:{VLLM_PORT}/v1/models", timeout=5)
        log(f"  vLLM status: {r.status_code}, models: {r.json()}")
    except Exception as e:
        log(f"  [ERROR] vLLM not ready: {e}")
        return

    prompts_list = [pr["prompt"] for pr in prompt_records]
    log(f"  Generating {len(prompts_list)} prompts...")
    t0 = time.time()
    outputs = vllm_generate(prompts_list)
    dt = time.time() - t0
    log(f"  Generation done in {dt:.1f}s ({dt/len(prompts_list):.2f}s/prompt)")

    for pr, out in zip(prompt_records, outputs):
        pr["output"] = out or ""

    # 11. Re-extract F3 → PCA48 for each output
    log(f"\n=== Step 4: Re-extract F3 → PCA48 for generated outputs ===")
    user_mu_arr = {}
    for u in eligible:
        if u in user_mus:
            user_mu_arr[u] = np.asarray(user_mus[u]["mu"], dtype=np.float32)

    # Build local competitor set
    log(f"  Loading CV quality + ASIN cohort...")
    cv_json = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians_cv.json"
    with open(cv_json) as f:
        cv_data = json.load(f)
    cv_users = cv_data["users"]
    high_quality = set()
    for u, info in cv_users.items():
        ci = info.get("cv_inlier_frac")
        cn = info.get("cv_nll")
        if ci is not None and cn is not None:
            if ci >= CV_INLIER_MIN and cn <= CV_NLL_MAX:
                high_quality.add(u)

    asin_to_users = defaultdict(set)
    for asin_rec in asin_data["asins"]:
        a = asin_rec["asin"]
        for u in asin_rec["users_sampled"]:
            asin_to_users[a].add(u)

    # Pre-compute local competitor μ arrays per pair (MEMORY EFFICIENT)
    log(f"  Building local competitor μ arrays per pair...")
    pair_local_mu = {}
    for pi, p in enumerate(pairs):
        target_asin = p["target_asin"]
        local_comp_ids = (asin_to_users.get(target_asin, set()) & high_quality) - {p["user_id"]}
        local_mus = []
        for c in local_comp_ids:
            if c in user_mus:
                local_mus.append(user_mus[c]["mu"])
        if local_mus:
            pair_local_mu[pi] = np.array(local_mus, dtype=np.float32)
            log(f"    pair {pi} ({p['user_id']} → {target_asin}): {len(local_mus)} local comps")
        else:
            pair_local_mu[pi] = None
            log(f"    pair {pi} ({p['user_id']} → {target_asin}): NO local comps")

    # 12. Compute distances per condition (memory efficient)
    log(f"\n=== Step 5: Compute d_drift, d_self, M_local per condition ===")
    for pr in prompt_records:
        u = pr["user_id"]
        pi = pr["pair_idx"]
        target_asin = pr["target_asin"]
        out_text = pr["output"]
        if not out_text:
            pr["z48"] = None
            pr["d_drift"] = None
            pr["d_self"] = None
            pr["M_local"] = None
            pr["d_nearest_other_local"] = None
            continue
        # Project generated query
        try:
            z = project_query_to_pca48(
                out_text, nlp, all_fnames, fnames_sub, col_idx,
                scaler_mean, scaler_scale, pca_components, pca_mean,
            )
        except Exception as e:
            log(f"    [ERROR] {u} {pr['condition']} k={pr['cand_idx']}: {e}")
            pr["z48"] = None
            continue
        if z is None:
            pr["z48"] = None
            continue
        pr["z48"] = z.tolist()
        # d_drift = ||z_rewrite - z_source||
        rep = representative.get(u, {})
        if rep and "z48" in rep:
            z_source = np.asarray(rep["z48"], dtype=np.float32)
            pr["d_drift"] = float(np.linalg.norm(z - z_source))
        else:
            pr["d_drift"] = None
        # d_self = ||z - μ_u||
        if u in user_mu_arr:
            pr["d_self"] = float(np.linalg.norm(z - user_mu_arr[u]))
        else:
            pr["d_self"] = None
        # d_nearest_other_local (same-ASIN high-quality competitors) — ONLY, skip global
        local_arr = pair_local_mu.get(pi)
        if local_arr is not None and len(local_arr) > 0:
            diffs_l = local_arr - z[None, :]
            dists_l = np.sqrt((diffs_l ** 2).sum(axis=1))
            pr["d_nearest_other_local"] = float(dists_l.min())
            pr["M_local"] = pr["d_nearest_other_local"] - pr["d_self"]
        else:
            pr["d_nearest_other_local"] = None
            pr["M_local"] = None

    # 13. Per-condition aggregate
    log(f"\n=== Step 6: Per-condition aggregate ===")
    cond_stats = {}
    for cond in CONDITIONS:
        d_drift_list = [pr["d_drift"] for pr in prompt_records
                        if pr["condition"] == cond and pr["d_drift"] is not None]
        d_self_list = [pr["d_self"] for pr in prompt_records
                       if pr["condition"] == cond and pr["d_self"] is not None]
        M_local_list = [pr["M_local"] for pr in prompt_records
                        if pr["condition"] == cond and pr["M_local"] is not None]
        cond_stats[cond] = {
            "n_generated": sum(1 for pr in prompt_records if pr["condition"] == cond and pr.get("output")),
            "n_with_z48": len(d_drift_list),
            "d_drift_median": round(statistics.median(d_drift_list), 3) if d_drift_list else None,
            "d_self_median": round(statistics.median(d_self_list), 3) if d_self_list else None,
            "M_local_median": round(statistics.median(M_local_list), 3) if M_local_list else None,
            "P_M_local_gt0": round(100 * sum(1 for m in M_local_list if m > 0) / max(len(M_local_list), 1), 1),
        }
        log(f"  {cond}:")
        log(f"    n_generated={cond_stats[cond]['n_generated']}, n_with_z48={cond_stats[cond]['n_with_z48']}")
        log(f"    d_drift_median={cond_stats[cond]['d_drift_median']}")
        log(f"    d_self_median={cond_stats[cond]['d_self_median']}")
        log(f"    M_local_median={cond_stats[cond]['M_local_median']}")
        log(f"    P(M_local>0)={cond_stats[cond]['P_M_local_gt0']}%")

    # 14. Per-pair paired comparison
    log(f"\n=== Step 7: Per-pair comparison ===")
    pair_compare = []
    for pi, p in enumerate(pairs):
        per_pair = {
            "user_id": p["user_id"],
            "target_asin": p["target_asin"],
            "rep_d_self": p["representative"]["d_self"],
            "rep_text": p["representative"]["sentence_text"][:120],
            "per_condition": {},
        }
        for cond in CONDITIONS:
            prs = [pr for pr in prompt_records if pr["pair_idx"] == pi and pr["condition"] == cond]
            M_l = [pr["M_local"] for pr in prs if pr["M_local"] is not None]
            d_d = [pr["d_drift"] for pr in prs if pr["d_drift"] is not None]
            d_s = [pr["d_self"] for pr in prs if pr["d_self"] is not None]
            per_pair["per_condition"][cond] = {
                "M_local_median": round(statistics.median(M_l), 3) if M_l else None,
                "d_drift_median": round(statistics.median(d_d), 3) if d_d else None,
                "d_self_median": round(statistics.median(d_s), 3) if d_s else None,
                "outputs": [pr["output"][:150] for pr in prs],
            }
        pair_compare.append(per_pair)
        log(f"  {p['user_id']} → {p['target_asin']}:")
        for cond in CONDITIONS:
            cdat = per_pair["per_condition"][cond]
            log(f"    {cond}: M={cdat['M_local_median']}, d_drift={cdat['d_drift_median']}, d_self={cdat['d_self_median']}")
            for o in cdat["outputs"]:
                log(f"      output: {o[:140]}")

    # 15. Save
    summary = {
        "config": {
            "n_smoke": N_SMOKE,
            "k_per_pair": K_PER_PAIR,
            "conditions": CONDITIONS,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "r_95_threshold": R_95,
            "cv_inlier_min": CV_INLIER_MIN,
            "cv_nll_max": CV_NLL_MAX,
        },
        "eligible_users": eligible,
        "n_pairs": len(pairs),
        "n_total_generations": len(prompt_records),
        "per_condition_stats": cond_stats,
        "per_pair_compare": pair_compare,
        "representative_sentences": {u: {k: v for k, v in rep.items() if k != "z48"}
                                     for u, rep in representative.items()},
        "key_finding": (
            f"Phase 7 smoke: 3 conditions × {len(pairs)} pairs × {K_PER_PAIR} candidates. "
            f"d_drift_med (rewrite)={cond_stats['C_rewrite']['d_drift_median']} vs "
            f"(exemplar)={cond_stats['B_exemplar']['d_drift_median']} vs "
            f"(free)={cond_stats['A_free']['d_drift_median']}. "
            f"P(M_local>0): free={cond_stats['A_free']['P_M_local_gt0']}%, "
            f"exemplar={cond_stats['B_exemplar']['P_M_local_gt0']}%, "
            f"rewrite={cond_stats['C_rewrite']['P_M_local_gt0']}%."
        ),
    }
    os.makedirs(RESULT_PATH.parent, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  wrote → {RESULT_PATH}")
    log(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
