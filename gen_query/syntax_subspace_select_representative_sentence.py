"""Phase 7.A.1 — Select centroid-nearest representative sentence per user.

User proposal (2026-08-31):
  s* = argmin_i ||z(s_i) - μ_u||
  exclude: target ASIN, 商品信息, 过短/残缺

For each (user, target_ASIN) pair from the Phase 6.F.1 cohort:
1. Read all user reviews from Baby_Products_2023.jsonl.gz
2. Filter:
   - Exclude reviews where asin == target_ASIN
   - Filter length 8-60 tokens
   - Filter sentences containing target product keywords (brand + title tokens)
3. Project to PCA48 → compute d_i = ||z_i - μ_u||
4. Pick s* = argmin d_i

Output: result/gen_query/phase7a_representative_sentences.json
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

# Make sure project root on path so `common.X` imports work under nohup (Rule 11)
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

# Hardcoded paths (per Rule 3)
DATA_DIR = Path("/fs04/ar57/wenyu/PersoanlQuery/data")
REVIEWS_GZ = DATA_DIR / "Baby_Products_2023.jsonl.gz"
META_GZ = DATA_DIR / "meta_Baby_Products_2023.jsonl.gz"

GAUSSIANS_JSON = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians.json"
ASINS_JSON = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json"
PHASE6D_PILOT = "/home/wlia0047/hj82_scratch2/wenyu/logs/phase6d_pilot.json"

LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/logs/phase7a1_select_repr.log"
RESULT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/phase7a_representative_sentences.json"

# Filters
MIN_TOKENS = 8
MAX_TOKENS = 60


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] [phase7a1] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_user_target_mapping():
    """Build user_id → target_asin mapping for the 7-user cohort via attrs overlap."""
    with open(PHASE6D_PILOT) as f:
        pilot = json.load(f)
    with open(ASINS_JSON) as f:
        asin_data = json.load(f)

    user_to_asin_candidates = collections.defaultdict(list)
    for asin_rec in asin_data["asins"]:
        for u in asin_rec.get("users_sampled", []):
            user_to_asin_candidates[u].append((asin_rec["asin"], asin_rec.get("attrs_used", {})))

    user_target = {}
    for r in pilot["results"]:
        u = r["user_id"]
        target_vals = set(str(x).lower() for x in r.get("attrs", []))
        best_asin = None
        best_overlap = 0
        for asin, attrs_dict in user_to_asin_candidates.get(u, []):
            if isinstance(attrs_dict, dict):
                attr_vals = set(str(v).lower() for v in attrs_dict.values())
            else:
                attr_vals = set(str(x).lower() for x in attrs_dict)
            overlap = len(target_vals & attr_vals)
            if overlap > best_overlap:
                best_overlap = overlap
                best_asin = asin
        if best_asin and best_overlap >= 1:
            user_target[u] = best_asin
    return user_target


def load_target_product_keywords(target_asins):
    """For each target ASIN, load title + brand as keyword set."""
    target_kw = {}
    with gzip.open(META_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("parent_asin") in target_asins or r.get("asin") in target_asins:
                asin = r.get("parent_asin") or r.get("asin")
                if asin in target_asins:
                    title = r.get("title", "")
                    brand = r.get("store", "") or r.get("brand", "")
                    details = r.get("details", {})
                    if isinstance(details, dict):
                        brand = details.get("Brand", brand)
                    # Extract keywords: words ≥3 chars, lowercase, alpha
                    kw = set()
                    for s in (title, brand):
                        for w in re.findall(r"\b[a-zA-Z]{3,}\b", s.lower()):
                            kw.add(w)
                    # Drop generic stopwords (basic)
                    generic = {"the", "and", "for", "with", "this", "that", "from",
                               "baby", "item", "use", "your", "you", "are", "can"}
                    kw = kw - generic
                    target_kw[asin] = kw
    return target_kw


def load_user_reviews(user_ids):
    """Scan Baby_Products, collect (text, asin) for target user_ids."""
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
    log(f"  scanned {n_records} records, matched {n_match} reviews for {len(user_reviews)} users")
    return user_reviews


def filter_sentences(reviews, target_asin, target_kw):
    """Filter sentences: exclude target ASIN, exclude product keywords, length 8-60."""
    out = []
    for text, asin in reviews:
        if asin == target_asin:
            continue
        # Split into sentences (period/question/exclamation)
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        for s in sents:
            s = s.strip()
            if not s:
                continue
            n_tok = len(s.split())
            if n_tok < MIN_TOKENS or n_tok > MAX_TOKENS:
                continue
            s_lower = s.lower()
            # Filter sentences containing target product keywords
            if target_kw and any(kw in s_lower for kw in target_kw):
                continue
            out.append(s)
    return out


def project_sentence_to_pca48(sent, nlp, all_fnames, fnames_sub, col_idx,
                              scaler_mean, scaler_scale, pca_components, pca_mean):
    """spaCy → F3 (mean over sents) → scaler → PCA48. Returns candidate_z (48,) or None."""
    from common.syntactic_features import per_sentence_features_v2
    doc = nlp(sent)
    sents = list(doc.sents)
    if not sents:
        return None
    feats_per_sent = []
    for s in sents:
        f = per_sentence_features_v2(s)
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


def main():
    log("=== Phase 7.A.1 — Select centroid-nearest representative sentence ===")

    # 1. User → target ASIN mapping (7-user cohort)
    user_target = load_user_target_mapping()
    log(f"\n  User → target ASIN mapping (overlap ≥1):")
    for u, a in user_target.items():
        log(f"    {u}: {a}")
    pilot_users = list(user_target.keys())
    target_asins = set(user_target.values())

    # 2. Load target product keywords (brand + title)
    log(f"\n  Loading target product keywords from {META_GZ}...")
    target_kw = load_target_product_keywords(target_asins)
    for asin, kw in target_kw.items():
        log(f"    {asin}: {len(kw)} keywords (sample: {list(kw)[:8]})")

    # 3. Load user reviews
    log(f"\n  Loading reviews for {len(pilot_users)} users from {REVIEWS_GZ}...")
    user_reviews = load_user_reviews(pilot_users)

    # 4. Load PCA48 model
    log(f"\n  Loading PCA48 from {GAUSSIANS_JSON}...")
    with open(GAUSSIANS_JSON) as f:
        gdoc = json.load(f)
    all_fnames = gdoc["feature_names_ordered"]
    fnames_sub = gdoc["fnames_f3"]
    scaler_mean = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
    pca_components = np.asarray(gdoc["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gdoc["pca_mean"], dtype=np.float64)
    col_idx = np.array([all_fnames.index(n) for n in fnames_sub], dtype=np.int64)
    user_mus = gdoc["users"]
    log(f"  loaded: all_fnames={len(all_fnames)}, fnames_sub={len(fnames_sub)}, "
        f"PCA={pca_components.shape}")

    # 5. Load spaCy
    log(f"\n  Loading spaCy en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    # 6. Per user: filter → project → pick centroid-nearest
    log(f"\n=== Per-user representative sentence selection ===")
    per_user_results = []
    for u in pilot_users:
        target_asin = user_target[u]
        reviews = user_reviews.get(u, [])
        log(f"\n  {u} (target={target_asin}): {len(reviews)} reviews")
        # Filter sentences
        cands = filter_sentences(reviews, target_asin, target_kw.get(target_asin, set()))
        log(f"    filtered to {len(cands)} candidate sentences (excl target ASIN + product kw + length)")
        if not cands:
            per_user_results.append({"user_id": u, "target_asin": target_asin,
                                     "n_reviews": len(reviews),
                                     "n_filtered_sentences": 0,
                                     "error": "no sentences after filtering"})
            continue
        user_mu = user_mus.get(u, {}).get("mu", None)
        if user_mu is None:
            log(f"    no μ for {u}, skipping")
            continue
        user_mu_arr = np.array(user_mu, dtype=np.float32)

        # Project all candidates → compute d_i
        scored = []
        n_fail = 0
        n_exc = 0
        n_none = 0
        first_error = None
        for s in cands:
            try:
                z = project_sentence_to_pca48(
                    s, nlp, all_fnames, fnames_sub, col_idx,
                    scaler_mean, scaler_scale, pca_components, pca_mean,
                )
            except Exception as e:
                n_exc += 1
                n_fail += 1
                if first_error is None:
                    first_error = f"{type(e).__name__}: {e}"
                    log(f"    [DEBUG] first exception on: \"{s[:80]}\" → {first_error}")
                continue
            if z is None:
                n_none += 1
                n_fail += 1
                if first_error is None:
                    first_error = "z is None (likely empty spaCy or no features)"
                    log(f"    [DEBUG] first None on: \"{s[:80]}\"")
                continue
            d = float(np.linalg.norm(z - user_mu_arr))
            scored.append({"text": s, "d_i": d, "z": z.tolist()})
        log(f"    projected: {len(scored)}/{len(cands)} ok "
            f"(exceptions={n_exc}, none={n_none}, first_error={first_error})")

        if not scored:
            per_user_results.append({"user_id": u, "target_asin": target_asin,
                                     "n_reviews": len(reviews),
                                     "n_filtered_sentences": len(cands),
                                     "error": "all projection failed"})
            continue

        # Pick centroid-nearest (min d_i)
        scored.sort(key=lambda r: r["d_i"])
        top5 = scored[:5]
        best = scored[0]
        log(f"    best d_i = {best['d_i']:.3f}, text: \"{best['text'][:120]}\"")
        for i, t in enumerate(top5[1:5], start=2):
            log(f"    top{i} d_i = {t['d_i']:.3f}, text: \"{t['text'][:80]}...\"")

        per_user_results.append({
            "user_id": u,
            "target_asin": target_asin,
            "n_reviews": len(reviews),
            "n_filtered_sentences": len(cands),
            "n_projected_ok": len(scored),
            "n_projected_failed": n_fail,
            "representative_sentence": best["text"],
            "d_i_min": best["d_i"],
            "top5_distances": [round(t["d_i"], 3) for t in top5],
            "top5_texts_preview": [t["text"][:120] for t in top5],
        })

    # 7. Summary
    n_success = sum(1 for r in per_user_results if "representative_sentence" in r)
    n_fail = len(per_user_results) - n_success
    log(f"\n=== SUMMARY ===")
    log(f"  n users: {len(per_user_results)}")
    log(f"  n with representative: {n_success}")
    log(f"  n failed: {n_fail}")
    if n_success:
        d_i_mins = [r["d_i_min"] for r in per_user_results if "d_i_min" in r]
        log(f"  d_i_min median: {statistics.median(d_i_mins):.3f}")
        log(f"  d_i_min min: {min(d_i_mins):.3f}, max: {max(d_i_mins):.3f}")

    summary = {
        "config": {
            "min_tokens": MIN_TOKENS,
            "max_tokens": MAX_TOKENS,
            "cohort": "Phase 6.F.1 7-user subset with attrs overlap ≥1",
        },
        "n_users": len(per_user_results),
        "n_success": n_success,
        "n_fail": n_fail,
        "per_user": per_user_results,
        "key_finding": (
            f"Phase 7.A.1: selected centroid-nearest representative sentence for "
            f"{n_success}/{len(per_user_results)} users. Median d_i_min = "
            f"{statistics.median(d_i_mins):.3f}." if n_success else "all users failed"
        ),
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  wrote → {RESULT_PATH}")
    log(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
