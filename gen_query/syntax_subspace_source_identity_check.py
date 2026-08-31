"""
Phase 7.J.0 — Source Identity Sanity Check

Verify that the 156 source sentences (selected by 7.I as gold discriminative:
M > 0 AND d_self <= R_95) STILL pass M > 0 AND d_self <= R_95 when re-evaluated
through 7.J's EXACT same pipeline:

  - mu_u fit on the user's disc sentence embeddings (in-sample)
  - R_95 = per-user 95th pct of d_self (>=5 sents), else theoretical 8.073
  - cohort competitors = ALL 52 selected users' mu_u

EXPECTED: P(M_source > 0) ~ 100%, P(d_self_source <= R_95) ~ 100%.

If observed rate is much below 100%, then 7.I's "gold" and 7.J's "evaluator"
Gaussian/R_95/competitor definitions are inconsistent. Must reconcile before
any rewrite-prompt tuning. The "LLM destroyed the syntax" conclusion is only
valid if this sanity check passes first.
"""

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np


# ===== constants (must match 7.J exactly) =====
MIN_TOKENS = 8
MAX_TOKENS = 60
MIN_QUERY_TOKENS = 6
MAX_QUERY_TOKENS = 60
R_95_THEORETICAL = 8.073
K_PER_USER = 3

DISC_POOL = Path(
    "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/phase7i_discriminative_source_pool.json"
)
SCALER_NPZ = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_shared_scaler.npz")
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
RESULT_DIR = REPO_ROOT / "result" / "gen_query"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = RESULT_DIR / "phase7j0_source_identity_check.json"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [7J0] {msg}", flush=True)


def split_sentences(text: str):
    import re
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
    log("=== Phase 7.J.0 — Source Identity Sanity Check ===")
    log("  Re-evaluating 7.I gold disc sents through 7.J's exact pipeline.")
    log("  Expected: P(M>0) ≈ 100%, P(d_self ≤ R_95) ≈ 100%")

    pool = json.load(open(DISC_POOL))
    selected_users = pool["selected_users"]
    log(f"  loaded {len(selected_users)} users, "
        f"{pool['summary']['n_discriminative_sentences_total']} disc sents total")

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

    # ---- Step 1: fit mu_u and R_95 per user from ALL their disc sents
    #              (mirrors 7.J step exactly) ----
    log("  fitting per-user mu_u from disc sentence embeddings...")
    user_mu = {}
    user_R95 = {}
    user_disc_z = {}  # store all z48 for later re-eval
    user_disc_text = {}
    for u in selected_users:
        disc = u["discriminative_sentences"]
        zs = []
        texts = []
        for s in disc:
            z = project_query(s["text"], nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                              pca_components, pca_mean, psf)
            if z is not None:
                zs.append(z)
                texts.append(s["text"])
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
        user_disc_z[u["uid"]] = Z
        user_disc_text[u["uid"]] = texts
    log(f"  fitted mu_u for {len(user_mu)} users")

    # cohort for M
    cohort_uids = [uid for uid in user_mu]
    cohort_mu = np.stack([user_mu[uid] for uid in cohort_uids], axis=0)
    cohort_uid_idx = {uid: i for i, uid in enumerate(cohort_uids)}

    # ---- Step 2: re-evaluate EACH disc sentence (including those that fed
    #              into mu_u itself — i.e. leave-one-out vs leave-all-in)
    #              through the 7.J evaluator ----
    log("  re-evaluating disc sentences (in-sample, mu includes self)...")
    n_total = 0
    n_valid = 0
    n_M_pass = 0
    n_d_self_pass = 0
    n_strict = 0
    rows = []

    for uid in cohort_uids:
        Z = user_disc_z[uid]
        texts = user_disc_text[uid]
        mu = user_mu[uid]
        R_95 = user_R95[uid]
        for zi, ti in zip(Z, texts):
            n_total += 1
            z = zi
            d_self = float(np.linalg.norm(z - mu))
            self_idx = cohort_uid_idx[uid]
            d_all = np.linalg.norm(cohort_mu - z, axis=1)
            d_other_min = float(np.min(np.delete(d_all, self_idx)))
            M = d_other_min - d_self
            M_pass = M > 0
            d_self_pass = d_self <= R_95
            strict = M_pass and d_self_pass
            if M_pass:
                n_M_pass += 1
            if d_self_pass:
                n_d_self_pass += 1
            if strict:
                n_strict += 1
            rows.append({
                "uid": uid,
                "text": ti,
                "M_source": M,
                "d_self_source": d_self,
                "d_other_min_source": d_other_min,
                "R_95_user": R_95,
                "M_pass": M_pass,
                "d_self_pass": d_self_pass,
                "strict_pass": strict,
            })
    n_valid = n_total  # all disc sents are "valid" by construction

    P_M = n_M_pass / n_valid if n_valid else 0.0
    P_d_self = n_d_self_pass / n_valid if n_valid else 0.0
    P_strict = n_strict / n_valid if n_valid else 0.0

    log("")
    log("=== IDENTITY SANITY ===")
    log(f"  re-evaluated {n_valid} disc sentences "
        f"(mu_u fit on these same sents, in-sample)")
    log(f"  P(M > 0): {n_M_pass}/{n_valid} = {P_M:.1%}")
    log(f"  P(d_self <= R_95): {n_d_self_pass}/{n_valid} = {P_d_self:.1%}")
    log(f"  STRICT pass (M>0 AND d_self<=R_95): {n_strict}/{n_valid} = {P_strict:.1%}")
    log("")
    log(f"  expected ≈100% (gold sents by construction).")
    if P_strict >= 0.95:
        log("  PASS — 7.I gold is consistent with 7.J evaluator. "
            "→ LLM rewrite (9/153=6%) genuinely destroyed user syntax.")
        verdict = "PASS"
    elif P_strict >= 0.80:
        log("  PARTIAL — small drift between 7.I filter and 7.J eval; "
            "investigate per-user before drawing conclusions.")
        verdict = "PARTIAL"
    else:
        log("  FAIL — 7.I and 7.J Gaussian / R_95 / competitor definitions are "
            "inconsistent. Must reconcile before any rewrite conclusion.")
        verdict = "FAIL"

    # distribution diagnostics
    Ms = [r["M_source"] for r in rows]
    ds = [r["d_self_source"] for r in rows]
    Rs = [r["R_95_user"] for r in rows]
    import statistics
    summary = {
        "config": {
            "description": "Phase 7.J.0 source identity sanity check. "
                           "Re-evaluates 7.I gold sents through 7.J's exact pipeline.",
            "K_PER_USER": K_PER_USER,
            "MIN_QUERY_TOKENS": MIN_QUERY_TOKENS,
            "MAX_QUERY_TOKENS": MAX_QUERY_TOKENS,
            "R_95_THEORETICAL": R_95_THEORETICAL,
        },
        "n_users_fitted": len(user_mu),
        "n_sentences_evaluated": n_valid,
        "P_M_pos": P_M,
        "P_d_self_le_R95": P_d_self,
        "P_strict": P_strict,
        "M_distribution": {
            "min": float(min(Ms)), "median": float(statistics.median(Ms)),
            "mean": float(statistics.mean(Ms)), "max": float(max(Ms)),
        },
        "d_self_distribution": {
            "min": float(min(ds)), "median": float(statistics.median(ds)),
            "mean": float(statistics.mean(ds)), "max": float(max(ds)),
        },
        "R_95_distribution": {
            "min": float(min(Rs)), "median": float(statistics.median(Rs)),
            "mean": float(statistics.mean(Rs)), "max": float(max(Rs)),
        },
        "verdict": verdict,
    }
    out = {"summary": summary, "per_sentence_rows": rows}
    log(f"  wrote → {RESULT_PATH}")
    with open(RESULT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    log("=== DONE ===")


if __name__ == "__main__":
    main()
