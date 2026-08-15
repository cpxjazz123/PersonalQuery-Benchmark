#!/usr/bin/env python3
"""E17 Step 1: user style-vector contract.

Profile/audit separation, leakage-free vector computation and split-half
stability (user-clustered bootstrap 95% CI lower bound > 0, permutation
p < 0.01). Outputs e17_style_vector_contract.json, e17_split_manifest.json
and a SHA-256 hash of the canonical vector store.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from user_stat_vector import FEATURES20, OPENER_CLASSES, opener_class_of  # noqa: E402
from extract_clause_features_single_query import load_spacy_model, extract_clause_features_from_doc  # noqa: E402

CATEGORY = "Baby_Products"
STAGE1 = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / CATEGORY / "stage1_filtered_users_reviews.json"
TRAIN_ASINS = set(json.load(open(REPO_ROOT / "result" / "personal_query" / "e14_multiproduct" / "baby_dataset.json")) and
                  [r["asin"] for r in json.load(open(REPO_ROOT / "result" / "personal_query" / "e14_multiproduct" / "baby_dataset.json"))])
TRAIN_USERS = {r["user_id"] for r in json.load(open(REPO_ROOT / "result" / "personal_query" / "e14_multiproduct" / "baby_dataset.json"))}
SEED = 42
N_CONTRACT_USERS = 400
MIN_GROUPS = 6          # >= 3 profile + 3 audit
MIN_SENTS_PER_HALF = 4
MAX_SENTENCES = 12      # per half (vector from max_sentences sentences)
OUT = REPO_ROOT / "result" / "personal_query" / "e17"
FEATS = FEATURES20
FEAT_IDX = {k: i for i, k in enumerate(FEATS)}


def build_vec(sent_feats: list[np.ndarray], openers: list[int]) -> np.ndarray:
    fmean = np.mean(sent_feats, axis=0) if sent_feats else np.zeros(len(FEATS), dtype=np.float32)
    n = float(np.linalg.norm(fmean))
    if n > 1e-12:
        fmean = fmean / n
    ohist = np.zeros(len(OPENER_CLASSES), dtype=np.float32)
    if openers:
        for c in openers:
            ohist[c] += 1
        ohist = ohist / len(openers)
    return np.concatenate([fmean, ohist]).astype(np.float32)  # len(FEATS)+10


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = json.load(open(STAGE1))

    # 1) candidate pool: users NOT in the training set, with >= MIN_GROUPS
    #    product groups, and at least one group whose ASIN is not in train
    cands = []
    for u in data["users"]:
        uid = u["user_id"]
        if uid in TRAIN_USERS:
            continue
        groups = [r for r in u.get("results", []) if r.get("target_reviews")]
        if len(groups) < MIN_GROUPS:
            continue
        if not any(r["asin"] not in TRAIN_ASINS for r in groups):
            continue
        cands.append(u)
    print(f"candidate pool: {len(cands)} non-train users with >= {MIN_GROUPS} groups")
    chosen = list(rng.choice(cands, size=min(N_CONTRACT_USERS, len(cands)), replace=False))

    # 2) per-user: profile (first half of groups, excluding train-ASIN groups)
    #    vs audit (second half). Deterministic per-user order.
    profiles: dict[str, list[str]] = {}
    audits: dict[str, list[dict]] = {}
    for u in chosen:
        groups = [r for r in u["results"] if r.get("target_reviews")]
        groups = [r for r in groups if r["asin"] not in TRAIN_ASINS]  # leak guard
        if not groups:
            continue
        n = len(groups)
        profile_groups, audit_groups = groups[: n // 2], groups[n // 2:]
        ptexts = [t for r in profile_groups for t in r["target_reviews"] if t.strip()]
        profiles[u["user_id"]] = ptexts
        audits[u["user_id"]] = audit_groups
    users = [u["user_id"] for u in chosen if u["user_id"] in profiles]

    # 3) single spaCy pass: sentences + features + opener per user, then
    #    split-half A/B by sentence order (stable per-user seed)
    nlp = load_spacy_model()
    all_texts = [(uid, t[:1000]) for uid in users for t in profiles[uid]]
    sent_feats: dict[str, list[np.ndarray]] = defaultdict(list)
    openers: dict[str, list[int]] = defaultdict(list)
    for (uid, t), doc in zip(all_texts, nlp.pipe([t for _, t in all_texts], batch_size=256)):
        for sent in doc.sents:
            toks = [tok for tok in sent if not tok.is_punct and not tok.is_space]
            if len(toks) < 3:
                continue
            try:
                ex = extract_clause_features_from_doc(sent, sent.text)
                sent_feats[uid].append(np.asarray([float(ex.get(k, 0.0)) for k in FEATS], dtype=np.float32))
            except Exception:
                continue
            openers[uid].append(OPENER_CLASSES.index(opener_class_of(toks[0].text)))
    print(f"users with profile sentences: {len(sent_feats)}")

    # 4) split-half vectors A/B (alternating by sentence index)
    vecA, vecB = {}, {}
    for uid in users:
        sf, op = sent_feats.get(uid, []), openers.get(uid, [])
        if len(sf) < 2 * MIN_SENTS_PER_HALF:
            continue
        ia = slice(0, len(sf), 2)  # deterministic alternating split
        A = build_vec(sf[ia], [op[i] for i in range(len(op)) if i % 2 == 0])
        B = build_vec(sf[1:][ia], [op[i] for i in range(len(op)) if i % 2 == 1])
        vecA[uid], vecB[uid] = A, B
    stable = sorted(vecA)
    print(f"stable users (>= {2 * MIN_SENTS_PER_HALF} sentences): {len(stable)}")

    # 5) user-clustered bootstrap + permutation: r(A_u, B_u) vs r(A_u, B_v)
    def rho(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def fisher_z(r):
        r = max(min(r, 0.9999), -0.9999)
        return 0.5 * np.log((1 + r) / (1 - r))

    same = {u: fisher_z(rho(vecA[u], vecB[u])) for u in stable}
    diff = {}
    for u in stable:
        others = rng.choice([v for v in stable if v != u], min(3, len(stable) - 1), replace=False)
        diff[u] = [fisher_z(rho(vecA[u], vecB[o])) for o in others]
    obs = float(np.mean(list(same.values())) - np.mean([d for ds in diff.values() for d in ds]))

    rngb = np.random.default_rng(0)
    z_diff_boot = []
    for _ in range(999):
        ids = rngb.choice(stable, size=len(stable), replace=True)
        s = np.mean([same[u] for u in ids])
        d = np.mean([d for u in ids for d in diff[u]])
        z_diff_boot.append(s - d)
    lo, hi = np.percentile(z_diff_boot, [2.5, 97.5])
    ci_lower = lo
    # label permutation on observations (keeping group sizes): shuffle the
    # pool of (same, diff) z-values, recompute the statistic
    pool = np.concatenate([np.array([same[u] for u in stable]),
                           np.array([d for ds in diff.values() for d in ds])])
    n_same = len(stable)
    n_perm = 999
    cnt = 0
    for _ in range(n_perm):
        pv = rngb.permutation(pool)
        s_p, d_p = pv[:n_same].mean(), pv[n_same:].mean()
        if s_p - d_p >= obs:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    print(f"split-half stability: meanΔz(same-diff)={obs:.3f} "
          f"boot95%CI=[{lo:.3f},{hi:.3f}] p={p:.4f} ({len(stable)} users)")
    passed1 = bool(ci_lower > 0 and p < 0.01)

    # 6) full profile vector (all profile sentences) + hash
    vecs = {}
    for uid in users:
        sf, op = sent_feats.get(uid, []), openers.get(uid, [])
        if len(sf) < MIN_SENTS_PER_HALF:
            continue
        vecs[uid] = build_vec(sf, op).tolist()
    n_users = len(vecs)
    canonical = json.dumps({"n_users": n_users, "dim": len(FEATS) + len(OPENER_CLASSES),
                            "vectors": {u: [round(float(x), 6) for x in v] for u, v in sorted(vecs.items())}},
                           sort_keys=True, separators=(",", ":"))
    vhash = hashlib.sha256(canonical.encode()).hexdigest()

    OUT.mkdir(parents=True, exist_ok=True)
    contract = {
        "version": "e17-step1", "category": CATEGORY, "seed": SEED,
        "n_users": n_users,
        "leakage_guard": {
            "train_users_excluded": True, "n_train_users": len(TRAIN_USERS),
            "train_asins_excluded_from_profile": True,
            "n_train_asins": len(TRAIN_ASINS),
            "audit_groups_not_in_profile": True,
            "target_query_features_not_used": True,
        },
        "vector_spec": {"features": FEATS, "openers": OPENER_CLASSES,
                        "dim": len(FEATS) + len(OPENER_CLASSES),
                        "max_sentences_per_half": MAX_SENTENCES,
                        "normalization": "L2 on 20 feature means; opener histogram normalized"},
        "split_half_stability": {
            "users": len(stable), "mean_delta_z_same_minus_diff": round(obs, 4),
            "bootstrap95_ci": [round(lo, 4), round(hi, 4)],
            "permutation_p": round(p, 4), "passed_lower_ci_gt_0_and_p_lt_0_01": passed1,
        },
        "vector_store_hash": vhash,
        "vector_store": str(OUT / "e17_style_vectors.json"),
        "manifest": str(OUT / "e17_split_manifest.json"),
    }
    json.dump(contract, open(OUT / "e17_style_vector_contract.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"n_users": n_users, "dim": len(FEATS) + len(OPENER_CLASSES),
               "vectors": vecs}, open(OUT / "e17_style_vectors.json", "w"))
    manifest = {"version": "e17-step1", "seed": SEED,
                "profile_rule": "first half of product groups (sorted by file order); train-ASIN groups removed",
                "audit_rule": "second half of product groups; not used in vector computation",
                "n_profile_sentences_per_user": {u: len(sent_feats.get(u, [])) for u in users},
                "n_audit_groups_per_user": {u: len(audits[u]) for u in audits if u in users},
                "users_with_vector": users,
                "excluded_train_users": sorted(TRAIN_USERS)[:5] + ["..."],
                "excluded_train_asin_count": len(TRAIN_ASINS)}
    json.dump(manifest, open(OUT / "e17_split_manifest.json", "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}: vectors={n_users}, hash={vhash[:16]}..., Check-1 passed={passed1}")


if __name__ == "__main__":
    main()
