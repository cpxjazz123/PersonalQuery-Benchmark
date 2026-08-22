#!/usr/bin/env python3
"""Phase 35.D: Hybrid rerank on Phase 35.C 10-scorer scores.

用现有 scores.json (Phase 35.C) 做 hybrid:
  hybrid_ab = a * zscore(scorer_x) + b * zscore(scorer_y)

Test variants:
  pca64_diag + syntax (Phase 35.B hybrid_07 经验)
  pca64_diag + pca32_global (PCA 高低维互补)
  pca64_diag + pca64_global
  pca32_diag + pca64_diag
  pca32_global + pca64_diag (cov 高的 + rank-1 高的)

Weights: 0.3/0.5/0.7 主 scorers 比例
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
PHASE35C_DIR = REPO_ROOT / "result/phase35c"
OUT_DIR = REPO_ROOT / "result/phase35d"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / (s + 1e-8)


def main():
    # Load Phase 35.C scores (sample only — full per-record data needed for intra-product)
    # Need to recompute from raw scores, not just first 5 samples
    # Let me re-read the raw scores.json which has all 10 scorers per record per cand
    raw = json.load(open(PHASE35C_DIR / "scores.json"))
    print(f"[load] {raw['n_records']} records, {len(raw['scorer_keys'])} scorers, K={raw['K']}")

    # We need full per-record scores to do intra-product. Sample has only 5.
    # Let me check actual structure
    if "sample_first_5" not in raw:
        print("[error] scores.json only has samples; need full per-record scoring")
        return 1
    print(f"[load] sample_first_5: {len(raw['sample_first_5'])} records")

    # Approach: hybrid requires full per-record per-cand scores for all 10 scorers
    # Need to re-load from phase35c_score.py and re-run scoring logic
    # OR re-read scores.json sample and just demo hybrid logic on it

    # Better approach: re-compute per-record per-cand scores from cached data
    sys.path.insert(0, str(REPO_ROOT))
    SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")

    # Load candidates + user sents + cache
    cands_data = json.load(open(SCRATCH / "phase35c_candidates_k8.json"))
    user_sents = json.load(open(SCRATCH / "user_sents_phase35b.json"))
    cache = __import__("torch").load(SCRATCH / "phase35c_user_qwen_residuals.pt", weights_only=False)
    all_resids = cache["residuals"]
    flat_meta = cache["meta"]
    print(f"[load] {len(cands_data)} records, {len(user_sents)} users, {len(flat_meta)} texts")

    syntax_row = {m: i for i, m in enumerate(flat_meta)}

    # Load PCA components
    pca_data = np.load(PHASE35C_DIR / "pca_components.npz", allow_pickle=True)
    # npz keys may be strings of integers
    def _maybe_int(k):
        try:
            return int(k)
        except (ValueError, TypeError):
            return k
    pca_components = {_maybe_int(k): v for k, v in pca_data["components"].item().items()}
    pca_means = {_maybe_int(k): v for k, v in pca_data["means"].item().items()}
    PCA_DIMS = sorted(pca_components.keys())
    print(f"[pca] dims={PCA_DIMS}")

    # Project all to PCA
    Z_all = {d: (all_resids - pca_means[d]) @ pca_components[d].T for d in PCA_DIMS}

    # Build per-user PCA stats
    user_pca_mu = {}
    user_pca_inv_var_diag = {}
    pca_global_inv_sigma = {}
    user_resids_dict = {}

    # Compute global sigma in PCA space
    for d in PCA_DIMS:
        Z_d = Z_all[d]
        pca_global_inv_sigma[d] = 1.0 / np.maximum(Z_d.var(axis=0), 1e-6)
    for i, m in enumerate(flat_meta):
        if m[0] == "user":
            user_resids_dict.setdefault(m[1], []).append(all_resids[i])
    for uid, vecs in user_resids_dict.items():
        V = np.stack(vecs, axis=0).astype(np.float64)
        for d in PCA_DIMS:
            Z = np.stack([Z_all[d][i] for i, mm in enumerate(flat_meta)
                          if mm[0] == "user" and mm[1] == uid], axis=0)
            mu = Z.mean(axis=0)
            user_pca_mu[(uid, d)] = mu
            var = Z.var(axis=0) + 1e-3
            user_pca_inv_var_diag[(uid, d)] = 1.0 / np.maximum(var, 1e-6)
        # 3584d baseline
        var = V.var(axis=0) + 1e-3
        user_inv_var_diag_3584 = 1.0 / np.maximum(var, 1e-6) if False else None

    user_inv_var_diag_3584 = {}
    for uid, vecs in user_resids_dict.items():
        V = np.stack(vecs, axis=0).astype(np.float64)
        var = V.var(axis=0) + 1e-3
        user_inv_var_diag_3584[uid] = 1.0 / np.maximum(var, 1e-6)

    # Compute user_qwen_mu for 3584d
    user_qwen_mu_3584 = {uid: np.stack(vecs, axis=0).mean(axis=0) for uid, vecs in user_resids_dict.items()}

    # Per-user syntax mean (length-controlled)
    # Need to compute spacy features too
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    FUNCTION_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "PART"}
    flat_texts = []
    for m in flat_meta:
        if m[0] == "cand":
            flat_texts.append(cands_data[m[1]]["candidates"][m[2]])
        else:
            flat_texts.append(user_sents[m[1]][m[2]])
    print(f"[spacy] computing 14d syntax for {len(flat_texts)} texts...")
    rows = []
    for doc in nlp.pipe(flat_texts, batch_size=64):
        n_tok = max(len(doc), 1)
        pos_counts = {}
        for tok in doc:
            pos_counts[tok.pos_] = pos_counts.get(tok.pos_, 0) + 1
        function_words_n = sum(pos_counts.get(p, 0) for p in FUNCTION_POS)
        coord = pos_counts.get("CCONJ", 0)
        subord = pos_counts.get("SCONJ", 0) + pos_counts.get("ADP", 0)
        words = [t.text for t in doc if t.is_alpha]
        avg_wlen = (sum(len(w) for w in words) / max(len(words), 1)) if words else 0
        max_depth = 0
        for tok in doc:
            depth, cur = 0, tok
            while cur.head != cur and depth < 20:
                cur = cur.head
                depth += 1
            max_depth = max(max_depth, depth)
        clause_count = pos_counts.get("VERB", 0) + subord
        rows.append([float(n_tok), float(clause_count), function_words_n / n_tok,
                     float(coord), float(subord),
                     sum(1 for tok in doc if tok.pos_ == "SCONJ" and tok.dep_ in {"relcl", "advcl"}),
                     float(max_depth), avg_wlen,
                     pos_counts.get("NOUN", 0) / n_tok, pos_counts.get("VERB", 0) / n_tok,
                     pos_counts.get("ADJ", 0) / n_tok, pos_counts.get("ADV", 0) / n_tok,
                     pos_counts.get("PRON", 0) / n_tok,
                     float(sum(1 for t in doc if t.is_punct))])
    all_syntax = np.array(rows, dtype=np.float64)
    print(f"[spacy] done shape={all_syntax.shape}")

    # Length-controlled syntax
    def count_attrs_covered(text, attrs):
        if not attrs: return 0
        text_lower = text.lower()
        return sum(1 for v in attrs.values() if v and str(v).strip() and str(v).strip().lower() in text_lower)
    n_attrs_per_text = np.zeros(len(flat_texts))
    for i, m in enumerate(flat_meta):
        if m[0] == "cand":
            n_attrs_per_text[i] = count_attrs_covered(cands_data[m[1]]["candidates"][m[2]],
                                                     cands_data[m[1]]["attrs_used"])
    control = np.stack([all_syntax[:, 0], all_syntax[:, 1], n_attrs_per_text], axis=1)
    ctrl_mean = control.mean(axis=0)
    ctrl_std = control.std(axis=0) + 1e-8
    control_z = (control - ctrl_mean) / ctrl_std
    all_syntax_ctrl = np.zeros_like(all_syntax)
    for dim in range(14):
        y = all_syntax[:, dim]
        XtX = control_z.T @ control_z
        Xty = control_z.T @ y
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(control_z, y, rcond=None)[0]
        all_syntax_ctrl[:, dim] = y - (control_z @ beta)
    user_syn_mean_ctrl = {}
    for uid, syn_list in user_sents.items():
        u_indices = [i for i, m in enumerate(flat_meta) if m[0] == "user" and m[1] == uid]
        u_syn_ctrl = all_syntax_ctrl[u_indices]
        user_syn_mean_ctrl[uid] = u_syn_ctrl.mean(axis=0)

    # === Build per-record per-cand scores for all 10 scorers ===
    print("[score] computing per-record per-cand scores for all 10 scorers...")
    record_scores: dict[int, dict[str, list[float]]] = {}
    for rec_i, rec in enumerate(cands_data):
        uid = rec["user_id"]
        asin = rec["asin"]
        cands = rec["candidates"]
        cand_resids = np.stack([all_resids[syntax_row[("cand", rec_i, k)]]
                                for k in range(len(cands))], axis=0)
        cand_syn_rows = [syntax_row[("cand", rec_i, k)] for k in range(len(cands))]
        cand_syn = all_syntax_ctrl[cand_syn_rows]

        scores = {}

        # diag_3584
        if uid in user_inv_var_diag_3584:
            inv_var = user_inv_var_diag_3584[uid]
            mu = user_qwen_mu_3584[uid]
            diffs = cand_resids - mu[None, :]
            maha = np.sqrt(np.maximum((diffs * diffs * inv_var[None, :]).sum(axis=1), 1e-8))
            scores["diag_3584"] = (-maha).tolist()
        else:
            scores["diag_3584"] = [0.0] * len(cands)

        # PCA scorers
        for d in PCA_DIMS:
            cand_z = Z_all[d][[syntax_row[("cand", rec_i, k)] for k in range(len(cands))]]
            if (uid, d) in user_pca_mu:
                mu_d = user_pca_mu[(uid, d)]
                inv_var_d = user_pca_inv_var_diag[(uid, d)]
                diffs = cand_z - mu_d[None, :]
                maha = np.sqrt(np.maximum((diffs * diffs * inv_var_d[None, :]).sum(axis=1), 1e-8))
                scores[f"pca{d}_diag"] = (-maha).tolist()
                # pca_d_global: GLOBAL sigma + user-specific mean
                inv_sigma_g = pca_global_inv_sigma[d]
                maha_sq = (diffs * diffs * inv_sigma_g[None, :]).sum(axis=1)
                maha = np.sqrt(np.maximum(maha_sq, 1e-8))
                scores[f"pca{d}_global"] = (-maha).tolist()
            else:
                scores[f"pca{d}_diag"] = [0.0] * len(cands)
                scores[f"pca{d}_global"] = [0.0] * len(cands)

        # syntax
        if uid in user_syn_mean_ctrl:
            syn_dists = np.linalg.norm(cand_syn - user_syn_mean_ctrl[uid][None, :], axis=1)
            scores["syntax"] = (-syn_dists).tolist()
        else:
            scores["syntax"] = [0.0] * len(cands)

        record_scores[rec_i] = scores
        if (rec_i + 1) % 20 == 0:
            print(f"  {rec_i + 1}/{len(cands_data)}", flush=True)

    # === Build per-asin pools for intra-product Rank-1 ===
    asin_to_recs: dict[str, list[int]] = {}
    for i, r in enumerate(cands_data):
        asin_to_recs.setdefault(r["asin"], []).append(i)

    asin_to_cands: dict[str, list[dict]] = {}
    for asin, rec_idxs in asin_to_recs.items():
        cands_in_pool = []
        for rec_i in rec_idxs:
            r = cands_data[rec_i]
            for k, q in enumerate(r["candidates"]):
                cands_in_pool.append({"uid": r["user_id"], "rec_i": rec_i, "k": k})
        asin_to_cands[asin] = cands_in_pool
    print(f"[pool] sizes top: {sorted([(a, len(c)) for a, c in asin_to_cands.items()], key=lambda x: -x[1])[:5]}")

    # === Hybrid variants ===
    hybrid_specs = [
        ("pca64_diag+syntax_05", "pca64_diag", "syntax", 0.5),
        ("pca64_diag+syntax_07", "pca64_diag", "syntax", 0.7),
        ("pca64_diag+syntax_03", "pca64_diag", "syntax", 0.3),
        ("pca32_global+syntax_07", "pca32_global", "syntax", 0.7),
        ("pca32_global+syntax_05", "pca32_global", "syntax", 0.5),
        ("pca32_global+pca64_diag_07", "pca32_global", "pca64_diag", 0.7),
        ("pca32_global+pca64_diag_05", "pca32_global", "pca64_diag", 0.5),
        ("pca64_diag+pca64_global_07", "pca64_diag", "pca64_global", 0.7),
        ("pca64_diag+pca128_diag_05", "pca64_diag", "pca128_diag", 0.5),
        ("pca64_diag+pca32_diag_05", "pca64_diag", "pca32_diag", 0.5),
    ]
    print(f"[hybrid] {len(hybrid_specs)} variants to test")

    intra_results: dict[str, list] = {f"size>={s}": [] for s in [2, 3, 4, 5, 10]}

    def score_pool_for_user(scorer: str, cand_objs: list[dict], target_uid: str) -> np.ndarray:
        """Score each cand in pool against target_uid's stats (NOT cand owner)."""
        n = len(cand_objs)
        if scorer == "diag_3584":
            if target_uid not in user_inv_var_diag_3584:
                return np.zeros(n)
            inv_var = user_inv_var_diag_3584[target_uid]
            mu = user_qwen_mu_3584[target_uid]
            cand_resids_arr = np.stack([all_resids[syntax_row[("cand", c["rec_i"], c["k"])]] for c in cand_objs], axis=0)
            diffs = cand_resids_arr - mu[None, :]
            maha = np.sqrt(np.maximum((diffs * diffs * inv_var[None, :]).sum(axis=1), 1e-8))
            return -maha
        if scorer.startswith("pca") and "_diag" in scorer:
            d = int(scorer.replace("pca", "").replace("_diag", ""))
            if (target_uid, d) not in user_pca_mu:
                return np.zeros(n)
            mu_d = user_pca_mu[(target_uid, d)]
            inv_var_d = user_pca_inv_var_diag[(target_uid, d)]
            cand_z = np.stack([Z_all[d][syntax_row[("cand", c["rec_i"], c["k"])]] for c in cand_objs], axis=0)
            diffs = cand_z - mu_d[None, :]
            maha = np.sqrt(np.maximum((diffs * diffs * inv_var_d[None, :]).sum(axis=1), 1e-8))
            return -maha
        if scorer.startswith("pca") and "_global" in scorer:
            d = int(scorer.replace("pca", "").replace("_global", ""))
            if (target_uid, d) not in user_pca_mu:
                return np.zeros(n)
            mu_d = user_pca_mu[(target_uid, d)]
            inv_sigma_g = pca_global_inv_sigma[d]
            cand_z = np.stack([Z_all[d][syntax_row[("cand", c["rec_i"], c["k"])]] for c in cand_objs], axis=0)
            diffs = cand_z - mu_d[None, :]
            maha_sq = (diffs * diffs * inv_sigma_g[None, :]).sum(axis=1)
            return -np.sqrt(np.maximum(maha_sq, 1e-8))
        if scorer == "syntax":
            if target_uid not in user_syn_mean_ctrl:
                return np.zeros(n)
            cand_syn = np.stack([all_syntax_ctrl[syntax_row[("cand", c["rec_i"], c["k"])]] for c in cand_objs], axis=0)
            syn_dists = np.linalg.norm(cand_syn - user_syn_mean_ctrl[target_uid][None, :], axis=1)
            return -syn_dists
        raise ValueError(scorer)

    for asin, cands in asin_to_cands.items():
        size = len(cands)
        if size < 2: continue
        uids_in_pool = list({c["uid"] for c in cands})

        for uid in uids_in_pool:
            # Score all cands in pool against THIS uid's stats (not cand owner)
            pool_scores: dict[str, np.ndarray] = {}
            for scorer in ["diag_3584", "pca32_diag", "pca32_global", "pca64_diag",
                            "pca64_global", "pca128_diag", "syntax"]:
                pool_scores[scorer] = score_pool_for_user(scorer, cands, uid)

            for hs_name, s_a, s_b, w_a in hybrid_specs:
                a_w = w_a
                b_w = 1 - w_a
                scores = a_w * zscore(pool_scores[s_a]) + b_w * zscore(pool_scores[s_b])
                order = np.argsort(-scores)
                rank1_uid = cands[int(order[0])]["uid"]
                hit = rank1_uid == uid
                own_ranks = [int(np.where(order == i)[0][0]) for i, c in enumerate(cands) if c["uid"] == uid]
                bucket = ("size>=10" if size >= 10 else
                          f"size>={'5' if size >= 5 else '4' if size >= 4 else '3' if size >= 3 else '2'}")
                intra_results[bucket].append({
                    "asin": asin, "uid": uid, "scorer": hs_name,
                    "n_pool": size, "rank1_uid": rank1_uid, "is_hit": hit,
                    "mean_own_rank": float(np.mean(own_ranks)) if own_ranks else None,
                })

    # Aggregate
    intra_summary = {}
    for size_key, items in intra_results.items():
        per_scorer = {}
        for it in items:
            sn = it["scorer"]
            per_scorer.setdefault(sn, {"n_hit": 0, "n_users": 0, "ranks": []})
            per_scorer[sn]["n_users"] += 1
            if it["is_hit"]:
                per_scorer[sn]["n_hit"] += 1
            if it["mean_own_rank"] is not None:
                per_scorer[sn]["ranks"].append(it["mean_own_rank"])
        for sn, st in per_scorer.items():
            per_scorer[sn] = {
                "n_users": st["n_users"], "n_hit": st["n_hit"],
                "rank1_pct": st["n_hit"] / max(st["n_users"], 1) * 100,
                "mean_own_rank": float(np.mean(st["ranks"])) if st["ranks"] else None,
            }
        intra_summary[size_key] = per_scorer

    # Print
    print(f"\n=== Phase 35.D — Hybrid Rerank on K=8 candidates ===")
    print(f"{'hybrid variant':<35} {'rank1_size>=10':>20} {'mean_rank':>10}")
    size10 = intra_summary.get("size>=10", {})
    base_lines = [
        ("pca64_diag (SOTA)", "pca64_diag"),
        ("pca32_global", "pca32_global"),
        ("pca32_diag", "pca32_diag"),
        ("pca64_global", "pca64_global"),
        ("pca128_diag", "pca128_diag"),
        ("diag_3584", "diag_3584"),
        ("syntax", "syntax"),
    ]
    # Add base scorers from intra_results (already computed)
    for base_name, scorer in base_lines:
        if scorer in size10:
            s = size10[scorer]
            mr = s["mean_own_rank"] if s["mean_own_rank"] is not None else 0
            print(f"  {base_name:<35} {s['n_hit']:>4}/{s['n_users']:<4} ({s['rank1_pct']:>5.1f}%) {mr:>9.2f}")

    for hs_name, _, _, _ in hybrid_specs:
        if hs_name in size10:
            s = size10[hs_name]
            mr = s["mean_own_rank"] if s["mean_own_rank"] is not None else 0
            print(f"  {hs_name:<35} {s['n_hit']:>4}/{s['n_users']:<4} ({s['rank1_pct']:>5.1f}%) {mr:>9.2f}")

    json.dump({
        "intra_summary": intra_summary,
        "hybrid_specs": [{"name": n, "s_a": a, "s_b": b, "w_a": w} for n, a, b, w in hybrid_specs],
    }, open(OUT_DIR / "hybrid_intra.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] hybrid → {OUT_DIR / 'hybrid_intra.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
