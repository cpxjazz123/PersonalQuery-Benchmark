"""
Phase 7.J.2 — 7.I vs 7.J evaluator divergence diagnostic

For each 7.I disc sentence, compute z48 once, then evaluate under THREE
different evaluators:

  (A) 7.I evaluator:
        mu_u = mean(z_all_u)        [user's ALL history, excl anchor ASIN]
        R_95 = 95th pct d_self(z_all_u)  [fallback 8.073 if n<5]
        cohort = per-ASIN peers (other selected users on same ASIN)

  (B) 7.J.0 evaluator (cross-ASIN merged):
        mu_u = mean(z_disc_u)       [only the disc sents, biased to eval set]
        R_95 = 95th pct d_self(z_disc_u)  [fallback 8.073 if n<5]
        cohort = ALL 39 selected users across all ASINs

  (C) 7.J.1 evaluator (per-ASIN, but disc-only mu):
        mu_u = mean(z_disc_u)
        R_95 = same
        cohort = per-ASIN peers

For each sentence, report d_self, d_other, M under each evaluator.
Find where 7.I says M>0 but 7.J.0 says M<0 (or vice versa), and identify
which term (mu_u fit set? cohort composition? R_95?) flips the sign.
"""

import collections
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np


# ===== constants =====
MIN_TOKENS = 8
MAX_TOKENS = 60
MIN_QUERY_TOKENS = 6
MAX_QUERY_TOKENS = 60
R_95_THEORETICAL = 8.073

DISC_POOL = Path(
    "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/phase7i_discriminative_source_pool.json"
)
SCALER_NPZ = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_shared_scaler.npz")
DATA_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/data")
RAW_REVIEWS = DATA_ROOT / "Baby_Products_2023.jsonl"
REVIEWS_GZ = DATA_ROOT / "Baby_Products_2023.jsonl.gz"

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
RESULT_DIR = REPO_ROOT / "result" / "gen_query"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = RESULT_DIR / "phase7j2_evaluator_divergence.json"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [7J2] {msg}", flush=True)


def split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


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


def main():
    log("=== Phase 7.J.2 — 7.I vs 7.J evaluator divergence diagnostic ===")

    # ---- Load 7.I pool ----
    pool = json.load(open(DISC_POOL))
    selected_users = pool["selected_users"]
    user_asin = {u["uid"]: u["asin"] for u in selected_users}
    asin_to_users = collections.defaultdict(list)
    for u in selected_users:
        asin_to_users[u["asin"]].append(u["uid"])

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

    # ---- Re-scan reviews, build z_all_u (anchor ASIN excluded) ----
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

    # ---- Build z_all_u (ALL history, INCLUDING anchor ASIN — mirrors 7.I) ----
    # 7.I's build_user_fullset does NOT exclude anchor ASIN; it scans all reviews.
    log("  building z_all_u per user (ALL reviews incl. anchor ASIN, mirrors 7.I)...")
    user_z_all = {}
    for uid in user_asin:
        revs = user_reviews.get(uid, [])
        sents = []
        for t, _ in revs:  # 7.I: ignore asin, scan all reviews
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
        if zs:
            user_z_all[uid] = np.asarray(zs, dtype=np.float64)
    log(f"  z_all built for {len(user_z_all)} users")

    # ---- Build z_disc_u (the disc sents we want to evaluate) ----
    # Use the SAME z's 7.I used (re-project from text for consistency)
    log("  re-projecting disc sents for z_disc_u...")
    user_z_disc = {}
    user_disc_text = {}
    for u in selected_users:
        zs = []
        texts = []
        for s in u["discriminative_sentences"]:
            z = project_query(s["text"], nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                              pca_components, pca_mean, psf)
            if z is not None:
                zs.append(z)
                texts.append(s["text"])
        if zs:
            user_z_disc[u["uid"]] = np.asarray(zs, dtype=np.float64)
            user_disc_text[u["uid"]] = texts

    # ---- Compute mu_u and R_95 under both fit strategies ----
    user_mu_all = {}   # 7.I's mu_u = mean(z_all_u)
    user_R95_all = {}
    user_mu_disc = {}  # 7.J.0/1's mu_u = mean(z_disc_u)
    user_R95_disc = {}

    for uid in user_z_all:
        Z = user_z_all[uid]
        mu = Z.mean(axis=0)
        user_mu_all[uid] = mu
        if len(Z) >= 5:
            user_R95_all[uid] = float(np.quantile(np.linalg.norm(Z - mu, axis=1), 0.95))
        else:
            user_R95_all[uid] = R_95_THEORETICAL
    for uid in user_z_disc:
        Z = user_z_disc[uid]
        mu = Z.mean(axis=0)
        user_mu_disc[uid] = mu
        if len(Z) >= 5:
            user_R95_disc[uid] = float(np.quantile(np.linalg.norm(Z - mu, axis=1), 0.95))
        else:
            user_R95_disc[uid] = R_95_THEORETICAL

    # ---- Build cohort mu stacks under two definitions ----
    # A. per-ASIN (7.I / 7.J.1)
    asin_to_fitted_users = {
        a: [u for u in us if u in user_mu_all]
        for a, us in asin_to_users.items()
    }
    asin_cohort_mu_all = {a: np.stack([user_mu_all[u] for u in us], axis=0)
                          for a, us in asin_to_fitted_users.items() if len(us) >= 1}
    asin_self_idx = {a: {u: i for i, u in enumerate(us)}
                     for a, us in asin_to_fitted_users.items() if len(us) >= 1}

    # B. cross-ASIN merged (7.J.0)
    all_fitted = [u for u in user_mu_disc]
    merged_mu_disc = np.stack([user_mu_disc[u] for u in all_fitted], axis=0)

    # ---- Evaluate each disc sentence under THREE evaluators ----
    rows = []
    n_total_sents = 0
    n_eval = 0
    n_skip_no_peers = 0
    n_skip_no_z = 0

    for u in selected_users:
        uid = u["uid"]
        a = u["asin"]
        if uid not in user_z_disc:
            continue
        disc_z = user_z_disc[uid]
        disc_text = user_disc_text[uid]
        # use the projected z (same as 7.I used to claim M>0)
        for s_7i, z, text in zip(u["discriminative_sentences"], disc_z, disc_text):
            n_total_sents += 1
            # ----- (A) 7.I evaluator: mu_all, per-ASIN cohort -----
            if uid in user_mu_all and a in asin_cohort_mu_all:
                mu_A = user_mu_all[uid]
                cohort_A = asin_cohort_mu_all[a]
                self_idx_A = asin_self_idx[a][uid]
                d_all_A = np.linalg.norm(cohort_A - z, axis=1)
                d_self_A = float(d_all_A[self_idx_A])
                others_A = np.delete(d_all_A, self_idx_A)
                if len(others_A) == 0:
                    n_skip_no_peers += 1
                    continue
                d_other_A = float(np.min(others_A))
                M_A = d_other_A - d_self_A
                R_95_A = user_R95_all[uid]
            else:
                M_A = d_self_A = d_other_A = R_95_A = None

            # ----- (B) 7.J.0 evaluator: mu_disc, merged 39u cohort -----
            if uid in user_mu_disc:
                mu_B = user_mu_disc[uid]
                d_self_B = float(np.linalg.norm(z - mu_B))
                idx_B = all_fitted.index(uid)
                d_all_B = np.linalg.norm(merged_mu_disc - z, axis=1)
                others_B = np.delete(d_all_B, idx_B)
                d_other_B = float(np.min(others_B))
                M_B = d_other_B - d_self_B
                R_95_B = user_R95_disc[uid]
            else:
                M_B = d_self_B = d_other_B = R_95_B = None

            # ----- (C) 7.J.1 evaluator: mu_disc, per-ASIN cohort -----
            if uid in user_mu_disc and a in asin_cohort_mu_all:
                # use mu_disc (different from A's mu_all)
                # need to project user's mu_disc onto the per-ASIN cohort built from mu_all
                # — for fair comparison, use the per-ASIN cohort built from same mu type as 7.J.1:
                asin_users_disc = [v for v in asin_to_users[a] if v in user_mu_disc]
                if len(asin_users_disc) >= 2:
                    cohort_C = np.stack([user_mu_disc[v] for v in asin_users_disc], axis=0)
                    self_idx_C = asin_users_disc.index(uid)
                    d_all_C = np.linalg.norm(cohort_C - z, axis=1)
                    d_self_C = float(d_all_C[self_idx_C])
                    others_C = np.delete(d_all_C, self_idx_C)
                    if len(others_C) == 0:
                        d_self_C = d_other_C = M_C = R_95_C = None
                    else:
                        d_other_C = float(np.min(others_C))
                        M_C = d_other_C - d_self_C
                        R_95_C = user_R95_disc[uid]
                else:
                    d_self_C = d_other_C = M_C = R_95_C = None
            else:
                d_self_C = d_other_C = M_C = R_95_C = None

            rows.append({
                "uid": uid,
                "asin": a,
                "text": text,
                "source_M_7I": s_7i["M"],  # what 7.I reported
                "A_7I": {"M": M_A, "d_self": d_self_A, "d_other": d_other_A,
                         "R_95": R_95_A, "mu_set": "z_all", "cohort": "per-ASIN"},
                "B_7J0": {"M": M_B, "d_self": d_self_B, "d_other": d_other_B,
                          "R_95": R_95_B, "mu_set": "z_disc", "cohort": "merged-39u"},
                "C_7J1": {"M": M_C, "d_self": d_self_C, "d_other": d_other_C,
                          "R_95": R_95_C, "mu_set": "z_disc", "cohort": "per-ASIN"},
            })
            n_eval += 1

    # ---- Aggregate: how often does sign flip? ----
    n_A_pos = sum(1 for r in rows if r["A_7I"]["M"] is not None and r["A_7I"]["M"] > 0)
    n_B_pos = sum(1 for r in rows if r["B_7J0"]["M"] is not None and r["B_7J0"]["M"] > 0)
    n_C_pos = sum(1 for r in rows if r["C_7J1"]["M"] is not None and r["C_7J1"]["M"] > 0)
    n_7I_pos = sum(1 for r in rows if r["source_M_7I"] > 0)

    # Sign-flip analysis
    flips_AB = sum(1 for r in rows
                   if r["A_7I"]["M"] is not None and r["B_7J0"]["M"] is not None
                   and r["A_7I"]["M"] > 0 and r["B_7J0"]["M"] <= 0)
    flips_AC = sum(1 for r in rows
                   if r["A_7I"]["M"] is not None and r["C_7J1"]["M"] is not None
                   and r["A_7I"]["M"] > 0 and r["C_7J1"]["M"] <= 0)
    n_BC_both = sum(1 for r in rows
                    if r["B_7J0"]["M"] is not None and r["C_7J1"]["M"] is not None)
    n_AB_both = sum(1 for r in rows
                    if r["A_7I"]["M"] is not None and r["B_7J0"]["M"] is not None)

    # Δ decomposition: which term dominates the flip?
    # For each flipped row, report (Δd_self, Δd_other) from A→B
    flip_decomp = []
    for r in rows:
        if r["A_7I"]["M"] is None or r["B_7J0"]["M"] is None:
            continue
        if r["A_7I"]["M"] > 0 and r["B_7J0"]["M"] <= 0:
            flip_decomp.append({
                "uid": r["uid"], "asin": r["asin"],
                "text": r["text"][:80],
                "A": r["A_7I"], "B": r["B_7J0"],
                "Δd_self": r["B_7J0"]["d_self"] - r["A_7I"]["d_self"],
                "Δd_other": r["B_7J0"]["d_other"] - r["A_7I"]["d_other"],
                "ΔM": r["B_7J0"]["M"] - r["A_7I"]["M"],
            })

    summary = {
        "config": {
            "description": ("Phase 7.J.2: same z, three evaluators (7.I / 7.J.0 / 7.J.1). "
                           "Identifies whether sign-flip comes from mu_set (z_all vs z_disc) "
                           "or cohort composition (per-ASIN vs merged)."),
        },
        "n_disc_sents_total": n_total_sents,
        "n_evaluated": n_eval,
        "P_M_pos_7I_reported": n_7I_pos / n_eval if n_eval else 0.0,
        "P_M_pos_A_re_evaluated": n_A_pos / n_eval if n_eval else 0.0,
        "P_M_pos_B_re_evaluated": n_B_pos / n_eval if n_eval else 0.0,
        "P_M_pos_C_re_evaluated": n_C_pos / n_eval if n_eval else 0.0,
        "n_AB_both_evaluated": n_AB_both,
        "n_BC_both_evaluated": n_BC_both,
        "sign_flip_A_to_B": flips_AB,
        "sign_flip_A_to_C": flips_AC,
        "n_skipped_no_peers": n_skip_no_peers,
    }
    out = {
        "summary": summary,
        "per_sentence_rows": rows,
        "flip_decomposition_A_to_B": flip_decomp[:30],  # sample
    }
    log(f"  wrote → {RESULT_PATH}")
    with open(RESULT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    log("")
    log("=== OVERALL ===")
    log(f"  total disc sents: {n_total_sents}, evaluated: {n_eval}")
    log(f"  P(M>0) 7.I reported:       {n_7I_pos/n_eval:.1%} ({n_7I_pos}/{n_eval})")
    log(f"  P(M>0) A (z_all, per-ASIN): {n_A_pos/n_eval:.1%} ({n_A_pos}/{n_eval})")
    log(f"  P(M>0) B (z_disc, merged):  {n_B_pos/n_eval:.1%} ({n_B_pos}/{n_eval})")
    log(f"  P(M>0) C (z_disc, per-ASIN): {n_C_pos/n_eval:.1%} ({n_C_pos}/{n_eval})")
    log(f"  Sign flips A→B (gold→fail): {flips_AB}")
    log(f"  Sign flips A→C (gold→fail): {flips_AC}")
    log("")
    if flips_AB > 0:
        # dominant flip source: which term changes more?
        mean_d_self = np.mean([f["Δd_self"] for f in flip_decomp])
        mean_d_other = np.mean([f["Δd_other"] for f in flip_decomp])
        mean_dM = np.mean([f["ΔM"] for f in flip_decomp])
        log(f"  On {flips_AB} A→B flips: mean Δd_self={mean_d_self:+.3f}, "
            f"mean Δd_other={mean_d_other:+.3f}, mean ΔM={mean_dM:+.3f}")
        if abs(mean_d_other) > abs(mean_d_self):
            log("  → DOMINANT cause: d_other dropped (cohort composition change "
                "made peers more similar to z than expected).")
        else:
            log("  → DOMINANT cause: d_self rose (μ_u change made user less central).")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
