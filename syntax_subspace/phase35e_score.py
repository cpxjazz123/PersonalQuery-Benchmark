#!/usr/bin/env python3
"""Phase 35.C: PCA + Shrinkage + K=8 — 验证用户在最高优先级建议。

Pipeline:
  K=8 candidates (Phase 35.C gen) → 一次性 batched Qwen + spacy → PCA subspace
  对比 4 种 scorer:
    S_diag: diagonal Maha (Phase 35.B baseline)
    S_pca_diag: PCA + diagonal Maha (用户路线 #1)
    S_pca_lw:  PCA + Ledoit-Wolf shrinkage covariance Maha (用户路线 #1 推荐)
    S_pca_global: PCA + GLOBAL sigma + user-specific mean (用户路线 #1 最推荐)
  + Oracle@K: 上限(理想 reranker 选最好 candidate)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
OUT_DIR = REPO_ROOT / "result/phase35e"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_JSON = SCRATCH / "phase35e_candidates_k16.json"
USER_SENTS_JSON = SCRATCH / "user_sents_phase35b.json"
RESIDUAL_CACHE = SCRATCH / "phase35e_user_qwen_residuals.pt"

OUT_SCORES = OUT_DIR / "scores.json"
OUT_INTRA = OUT_DIR / "intra_product_rank1.json"
OUT_EVAL = OUT_DIR / "eval_summary.json"
OUT_PCA_INFO = OUT_DIR / "pca_components.npz"

K_SAMPLES = 16
PCA_DIMS = [int(d) for d in os.environ.get("P35C_PCA_DIMS", "32,64,128").split(",")]


def compute_syntax_features(texts: list[str]) -> np.ndarray:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    FUNCTION_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "PART"}
    rows = []
    for doc in nlp.pipe(texts, batch_size=64):
        n_tok = max(len(doc), 1)
        pos_counts = {}
        for tok in doc:
            pos_counts[tok.pos_] = pos_counts.get(tok.pos_, 0) + 1
        function_words_n = sum(pos_counts.get(p, 0) for p in FUNCTION_POS)
        func_ratio = function_words_n / n_tok
        coord = pos_counts.get("CCONJ", 0)
        subord = pos_counts.get("SCONJ", 0) + pos_counts.get("ADP", 0)
        rel_count = sum(1 for tok in doc if tok.pos_ == "SCONJ" and
                        tok.dep_ in {"relcl", "advcl"})
        max_depth = 0
        for tok in doc:
            depth, cur = 0, tok
            while cur.head != cur and depth < 20:
                cur = cur.head
                depth += 1
            max_depth = max(max_depth, depth)
        words = [t.text for t in doc if t.is_alpha]
        avg_wlen = (sum(len(w) for w in words) / max(len(words), 1)) if words else 0
        noun_r = pos_counts.get("NOUN", 0) / n_tok
        verb_r = pos_counts.get("VERB", 0) / n_tok
        adj_r = pos_counts.get("ADJ", 0) / n_tok
        adv_r = pos_counts.get("ADV", 0) / n_tok
        pron_r = pos_counts.get("PRON", 0) / n_tok
        punct = sum(1 for t in doc if t.is_punct)
        clause_count = pos_counts.get("VERB", 0) + subord
        rows.append([float(n_tok), float(clause_count), func_ratio, float(coord),
                     float(subord), float(rel_count), float(max_depth), avg_wlen,
                     noun_r, verb_r, adj_r, adv_r, pron_r, float(punct)])
    return np.array(rows, dtype=np.float64)


def count_attrs_covered(text: str, attrs: dict) -> int:
    if not attrs:
        return 0
    text_lower = text.lower()
    return sum(1 for v in attrs.values() if v and str(v).strip()
               and str(v).strip().lower() in text_lower)


def qwen_forward_residuals(texts: list[str], client, batch_size: int = 8) -> np.ndarray:
    out_vecs = []
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk = texts[i:i + batch_size]
        tok = client._hidden_backend.tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(client._hidden_backend.model.device)
        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = client._hidden_backend.model(
                **tok, output_hidden_states=True, return_dict=True
            )
        torch.cuda.synchronize()
        last_h = outputs.hidden_states[-1]
        mask = tok["attention_mask"].unsqueeze(-1).float()
        mean_pooled = (last_h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out_vecs.append(mean_pooled.cpu().numpy().astype(np.float32))
        del outputs, last_h
        torch.cuda.empty_cache()
        done = min(i + batch_size, n)
        if done % 64 == 0 or done == n:
            print(f"  [qwen] {done}/{n}", flush=True)
    return np.concatenate(out_vecs, axis=0)


def ledoit_wolf_shrinkage(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage estimator: returns (shrunk_cov, shrinkage)."""
    n, d = X.shape
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / max(n - 1, 1)  # sample cov
    mu = np.trace(S) / d
    S_target = mu * np.eye(d)  # identity scaled by avg variance
    delta = S - S_target
    # Ledoit-Wolf: shrinkage = sum(var(s_ij)) / sum(delta_ij^2)
    # Numerator: var of sample cov entries
    sq_S = (Xc ** 2).T @ (Xc ** 2) / max(n - 1, 1) - S ** 2  # var per entry
    num = np.sum(sq_S)
    den = np.sum(delta ** 2) + 1e-12
    rho = np.clip(num / den, 0.0, 1.0)
    shrunk = (1 - rho) * S_target + rho * S
    return shrunk, float(rho)


def main():
    t0 = time.time()
    print("[load] candidates + user sents...", flush=True)
    cands_data = json.load(open(CANDIDATES_JSON))
    user_sents = json.load(open(USER_SENTS_JSON))
    print(f"[load] {len(cands_data)} records, {len(user_sents)} users")

    # Flatten texts (candidates + user sents)
    flat_texts = []
    flat_meta = []
    for rec_i, rec in enumerate(cands_data):
        for k, q in enumerate(rec["candidates"]):
            flat_texts.append(q)
            flat_meta.append(("cand", rec_i, k))
    for uid, sents in user_sents.items():
        for s_i, s in enumerate(sents):
            flat_texts.append(s)
            flat_meta.append(("user", uid, s_i))
    print(f"[flatten] {len(flat_texts)} texts (cands + user sents)")

    # === Step 1: Qwen forward ONCE ===
    cache_data = None
    if RESIDUAL_CACHE.exists():
        try:
            cache_data = torch.load(RESIDUAL_CACHE, weights_only=False)
            if len(cache_data.get("residuals", [])) == len(flat_texts):
                print(f"[step-1] cache hit: {len(cache_data['residuals'])} residuals")
            else:
                print(f"[step-1] cache stale, recompute")
                cache_data = None
        except Exception:
            cache_data = None
    if cache_data is None:
        print("[step-1] loading Qwen...")
        sys.path.insert(0, str(REPO_ROOT))
        from llm_client import create_qwen_local_client
        client = create_qwen_local_client(with_vllm=False)
        print(f"[step-1] Qwen loaded, {time.time()-t0:.1f}s")
        all_resids = qwen_forward_residuals(flat_texts, client, batch_size=8)
        print(f"[step-1] all_resids shape={all_resids.shape}, {time.time()-t0:.1f}s")
        torch.save({"residuals": all_resids, "meta": flat_meta}, RESIDUAL_CACHE)
    else:
        all_resids = cache_data["residuals"]

    # Split residuals
    cand_resids_dict: dict[tuple[int, int], np.ndarray] = {}
    user_resids_dict: dict[str, list[np.ndarray]] = {}
    for i, meta in enumerate(flat_meta):
        if meta[0] == "cand":
            cand_resids_dict[(meta[1], meta[2])] = all_resids[i]
        else:
            user_resids_dict.setdefault(meta[1], []).append(all_resids[i])

    # === Step 2: spacy syntax ONCE ===
    print(f"[step-2] spacy nlp.pipe on {len(flat_texts)} texts...")
    all_syntax = compute_syntax_features(flat_texts)
    print(f"[step-2] all_syntax shape={all_syntax.shape}, {time.time()-t0:.1f}s")
    cand_syntax_dict: dict[tuple[int, int], np.ndarray] = {}
    user_syntax_dict: dict[str, list[np.ndarray]] = {}
    for i, meta in enumerate(flat_meta):
        if meta[0] == "cand":
            cand_syntax_dict[(meta[1], meta[2])] = all_syntax[i]
        else:
            user_syntax_dict.setdefault(meta[1], []).append(all_syntax[i])

    # === Step 3: PCA on union of all residuals ===
    # Stack all cands + user sents, fit PCA on this
    all_resids_stack = all_resids
    print(f"[step-3] PCA on {all_resids_stack.shape}, dims={PCA_DIMS}")
    pca_components = {}
    pca_means = {}
    for d in PCA_DIMS:
        pca = PCA(n_components=d, random_state=42)
        Z = pca.fit_transform(all_resids_stack)
        pca_components[d] = pca.components_.astype(np.float32)  # (d, 3584)
        pca_means[d] = pca.mean_.astype(np.float32)  # (3584,)
        evr = pca.explained_variance_ratio_.sum()
        print(f"  PCA-{d}: evr_sum={evr:.4f}")

    # Project everything to PCA spaces
    Z_all = {}
    for d in PCA_DIMS:
        Z_all[d] = (all_resids_stack - pca_means[d]) @ pca_components[d].T  # (n, d)
        Z_all[d] = Z_all[d].astype(np.float32)
    print(f"[step-3] Z_all built for d={list(Z_all.keys())}, {time.time()-t0:.1f}s")

    # Save PCA components for reuse
    np.savez_compressed(OUT_PCA_INFO,
                        means={d: pca_means[d] for d in PCA_DIMS},
                        components={d: pca_components[d] for d in PCA_DIMS},
                        evr={d: float((Z_all[d].std(axis=0) ** 2).sum() / (all_resids_stack.std(axis=0) ** 2).sum()) for d in PCA_DIMS})
    print(f"[step-3] saved PCA → {OUT_PCA_INFO}")

    # === Step 4: per-user PCA-space statistics (3 variants) ===
    # S_diag on raw 3584d (baseline)
    user_inv_var_diag_3584: dict[str, np.ndarray] = {}
    # S_pca_diag: per-user diagonal in PCA-d
    user_pca_inv_var_diag: dict[tuple[str, int], np.ndarray] = {}
    # S_pca_lw: per-user Ledoit-Wolf shrunk cov in PCA-d
    user_pca_lw_inv: dict[tuple[str, int], np.ndarray] = {}
    user_pca_lw_rho: dict[tuple[str, int], float] = {}
    # S_pca_global: global sigma + user-specific mean
    user_pca_mu: dict[tuple[str, int], np.ndarray] = {}
    pca_global_inv_sigma: dict[int, np.ndarray] = {}  # global diag Maha

    for d in PCA_DIMS:
        Z = Z_all[d]
        # Build user Z dict
        user_Z_dict: dict[str, np.ndarray] = {}
        for i, meta in enumerate(flat_meta):
            if meta[0] == "user":
                user_Z_dict.setdefault(meta[1], []).append(Z[i])

        # Compute global stats in PCA-d
        Z_global_mean = Z.mean(axis=0)
        Z_global_var = Z.var(axis=0)
        pca_global_inv_sigma[d] = 1.0 / np.maximum(Z_global_var, 1e-6)

        for uid, zs in user_Z_dict.items():
            V = np.stack(zs, axis=0)
            mu_d = V.mean(axis=0)
            user_pca_mu[(uid, d)] = mu_d

            # Diagonal
            if V.shape[0] >= 2:
                var_diag_d = V.var(axis=0) + 1e-3
            else:
                var_diag_d = np.ones(d) * 1e-3
            user_pca_inv_var_diag[(uid, d)] = 1.0 / np.maximum(var_diag_d, 1e-6)

            # Ledoit-Wolf
            if V.shape[0] >= 2:
                shrunk_cov, rho = ledoit_wolf_shrinkage(V.astype(np.float64))
                shrunk_cov += 1e-6 * np.eye(d)
                try:
                    user_pca_lw_inv[(uid, d)] = np.linalg.inv(shrunk_cov).astype(np.float32)
                    user_pca_lw_rho[(uid, d)] = rho
                except np.linalg.LinAlgError:
                    user_pca_lw_inv[(uid, d)] = np.diag(1.0 / np.maximum(np.diag(shrunk_cov), 1e-6)).astype(np.float32)
                    user_pca_lw_rho[(uid, d)] = rho
            else:
                user_pca_lw_inv[(uid, d)] = np.eye(d, dtype=np.float32)
                user_pca_lw_rho[(uid, d)] = 1.0

    # 3584d baseline
    for uid, vecs in user_resids_dict.items():
        V = np.stack(vecs, axis=0).astype(np.float64)
        var_diag = V.var(axis=0) + 1e-3
        user_inv_var_diag_3584[uid] = 1.0 / np.maximum(var_diag, 1e-6)

    print(f"[step-4] user stats ready, {time.time()-t0:.1f}s")

    # === Step 5: pre-compute user_sents PCA projections for syntax weighting ===
    # Length-controlled syntax in 14d (same as Phase 35.B)
    n_attrs_per_text = np.zeros(len(flat_texts))
    for i, meta in enumerate(flat_meta):
        if meta[0] == "cand":
            rec_i = meta[1]
            attrs = cands_data[rec_i]["attrs_used"]
            q = cands_data[rec_i]["candidates"][meta[2]]
            n_attrs_per_text[i] = count_attrs_covered(q, attrs)
        else:
            n_attrs_per_text[i] = 0
    control = np.stack([all_syntax[:, 0], all_syntax[:, 1], n_attrs_per_text], axis=1)
    ctrl_mean = control.mean(axis=0)
    ctrl_std = control.std(axis=0) + 1e-8
    control_z = (control - ctrl_mean) / ctrl_std
    all_syntax_ctrl = np.zeros_like(all_syntax, dtype=np.float64)
    for dim in range(14):
        y = all_syntax[:, dim].astype(np.float64)
        XtX = control_z.T @ control_z
        Xty = control_z.T @ y
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(control_z, y, rcond=None)[0]
        all_syntax_ctrl[:, dim] = y - (control_z @ beta)
    user_syn_mean_ctrl: dict[str, np.ndarray] = {}
    syntax_row: dict[tuple, int] = {m: i for i, m in enumerate(flat_meta)}
    for uid, syn_list in user_syntax_dict.items():
        u_indices = [i for i, m in enumerate(flat_meta) if m[0] == "user" and m[1] == uid]
        u_syn_ctrl = all_syntax_ctrl[u_indices]
        user_syn_mean_ctrl[uid] = u_syn_ctrl.mean(axis=0)
    print(f"[step-5] length-ctrl syntax ready")

    # === Step 6: Per-record per-candidate scores (4 scorers × N PCA dims) ===
    # Scorer names: diag_3584, pca{32,64,128}_diag, pca{32,64,128}_lw, pca{32,64,128}_global
    scorer_keys = ["diag_3584"]
    for d in PCA_DIMS:
        scorer_keys.append(f"pca{d}_diag")
        scorer_keys.append(f"pca{d}_lw")
        scorer_keys.append(f"pca{d}_global")

    print(f"[step-6] {len(scorer_keys)} scorers: {scorer_keys}")

    # Per-record scores dict: record_idx -> {scorer: [cand_score]}
    record_scores: dict[int, dict[str, list[float]]] = {}
    record_attrs_cov: dict[int, list[int]] = {}
    record_full_cov_mask: dict[int, list[bool]] = {}

    syntax_cand_cache: dict[tuple[int, int], np.ndarray] = {}

    for rec_i, rec in enumerate(cands_data):
        uid = rec["user_id"]
        asin = rec["asin"]
        attrs = rec["attrs_used"]
        cands = rec["candidates"]
        n_attr = len(attrs)
        coverages = np.array([count_attrs_covered(c, attrs) for c in cands])
        full_cov_mask = coverages == n_attr

        record_attrs_cov[rec_i] = coverages.tolist()
        record_full_cov_mask[rec_i] = full_cov_mask.tolist()

        cand_resids = np.stack([cand_resids_dict[(rec_i, k)] for k in range(len(cands))], axis=0)

        scores_for_rec: dict[str, list[float]] = {}

        # 1) diag_3584 (Phase 35.B baseline)
        if uid in user_inv_var_diag_3584:
            inv_var = user_inv_var_diag_3584[uid]
            mu = np.stack([user_resids_dict[uid]], axis=0).mean(axis=0) if False else None
            mu = np.stack(user_resids_dict[uid], axis=0).mean(axis=0)
            diffs = cand_resids - mu[None, :]
            maha = np.sqrt(np.maximum((diffs * diffs * inv_var[None, :]).sum(axis=1), 1e-8))
            scores_for_rec["diag_3584"] = (-maha).tolist()
        else:
            scores_for_rec["diag_3584"] = [0.0] * len(cands)

        # 2) PCA scorers
        for d in PCA_DIMS:
            cand_z = Z_all[d][[syntax_row[("cand", rec_i, k)] for k in range(len(cands))]]

            if (uid, d) in user_pca_mu:
                mu_d = user_pca_mu[(uid, d)]
                # pca_d_diag
                inv_var_d = user_pca_inv_var_diag[(uid, d)]
                diffs = cand_z - mu_d[None, :]
                maha = np.sqrt(np.maximum((diffs * diffs * inv_var_d[None, :]).sum(axis=1), 1e-8))
                scores_for_rec[f"pca{d}_diag"] = (-maha).tolist()

                # pca_d_lw (full LW shrunk cov inverse)
                inv_sigma_lw = user_pca_lw_inv[(uid, d)]
                maha_sq = np.einsum("nd,de,ne->n", diffs.astype(np.float64),
                                    inv_sigma_lw.astype(np.float64), diffs.astype(np.float64))
                maha = np.sqrt(np.maximum(maha_sq, 1e-8))
                scores_for_rec[f"pca{d}_lw"] = (-maha).tolist()

                # pca_d_global (global sigma + user mean)
                inv_sigma_g = pca_global_inv_sigma[d]
                maha_sq = (diffs * diffs * inv_sigma_g[None, :]).sum(axis=1)
                maha = np.sqrt(np.maximum(maha_sq, 1e-8))
                scores_for_rec[f"pca{d}_global"] = (-maha).tolist()
            else:
                for k in [f"pca{d}_diag", f"pca{d}_lw", f"pca{d}_global"]:
                    scores_for_rec[k] = [0.0] * len(cands)

        # syntax 14d (for hybrid)
        if uid in user_syn_mean_ctrl:
            cand_syn_rows = [syntax_row[("cand", rec_i, k)] for k in range(len(cands))]
            cand_syn_ctrl_arr = all_syntax_ctrl[cand_syn_rows]
            syn_dists = np.linalg.norm(cand_syn_ctrl_arr - user_syn_mean_ctrl[uid][None, :], axis=1)
            scores_for_rec["syntax"] = (-syn_dists).tolist()
        else:
            scores_for_rec["syntax"] = [0.0] * len(cands)

        record_scores[rec_i] = scores_for_rec
        if (rec_i + 1) % 20 == 0 or rec_i == len(cands_data) - 1:
            print(f"[step-6] {rec_i + 1}/{len(cands_data)} records scored", flush=True)

    # === Step 7: Per-record selection + Oracle upper bound ===
    # Oracle@K: best candidate per record (max coverage, fallback to best cov)
    oracle_top1_idx: dict[int, int] = {}
    for rec_i, rec in enumerate(cands_data):
        covs = record_attrs_cov[rec_i]
        full = record_full_cov_mask[rec_i]
        # Oracle: max coverage; ties broken by syntax closest to user mean
        max_cov = max(covs)
        candidates_at_max = [k for k, c in enumerate(covs) if c == max_cov]
        # pick the one with best syntax among max-cov
        syn_scores = record_scores[rec_i].get("syntax", [0.0] * len(rec["candidates"]))
        best = max(candidates_at_max, key=lambda k: syn_scores[k])
        oracle_top1_idx[rec_i] = best

    # Save per-record scores (only first 5 records for inspection)
    save_data = {
        "n_records": len(cands_data),
        "K": K_SAMPLES,
        "scorer_keys": scorer_keys,
        "oracle_top1_idx": oracle_top1_idx,
        "sample_first_5": [
            {
                "rec_i": rec_i,
                "user_id": cands_data[rec_i]["user_id"],
                "asin": cands_data[rec_i]["asin"],
                "attrs_used": cands_data[rec_i]["attrs_used"],
                "coverages": record_attrs_cov[rec_i],
                "scores": {k: v for k, v in record_scores[rec_i].items()},
                "oracle_idx": oracle_top1_idx[rec_i],
            }
            for rec_i in range(min(5, len(cands_data)))
        ],
    }
    json.dump(save_data, open(OUT_SCORES, "w"), indent=2, ensure_ascii=False)
    print(f"[save] scores → {OUT_SCORES}")

    # === Step 8: Intra-product Rank-1 ===
    asin_to_recs: dict[str, list[int]] = {}
    for i, r in enumerate(cands_data):
        asin_to_recs.setdefault(r["asin"], []).append(i)

    asin_to_cands: dict[str, list[dict]] = {}
    for asin, rec_idxs in asin_to_recs.items():
        cands_in_pool = []
        for rec_i in rec_idxs:
            r = cands_data[rec_i]
            for k, q in enumerate(r["candidates"]):
                cands_in_pool.append({
                    "uid": r["user_id"],
                    "text": q,
                    "resid": cand_resids_dict[(rec_i, k)],
                    "z": {d: Z_all[d][syntax_row[("cand", rec_i, k)]] for d in PCA_DIMS},
                    "syntax_ctrl": all_syntax_ctrl[syntax_row[("cand", rec_i, k)]],
                })
        asin_to_cands[asin] = cands_in_pool
    pool_sizes = sorted([(a, len(c)) for a, c in asin_to_cands.items()], key=lambda x: -x[1])
    print(f"\n[step-8] pool sizes top 10: {pool_sizes[:10]}")

    # Build candidate score arrays per scorer (per asin pool, per cand)
    intra_results: dict[str, list] = {f"size>={s}": [] for s in [2, 3, 4, 5, 10]}

    # Scorer function generators
    def score_pool(scorer_name: str, cand_z_dict: dict[int, np.ndarray], uid: str) -> np.ndarray:
        """Compute scores for all cands in pool for given scorer."""
        if scorer_name == "diag_3584":
            inv_var = user_inv_var_diag_3584[uid]
            mu = np.stack(user_resids_dict[uid], axis=0).mean(axis=0)
            cand_resids = np.stack([c["resid"] for c in cand_z_dict["cands"]], axis=0)
            diffs = cand_resids - mu[None, :]
            maha = np.sqrt(np.maximum((diffs * diffs * inv_var[None, :]).sum(axis=1), 1e-8))
            return -maha
        if scorer_name.startswith("pca") and "_diag" in scorer_name:
            d = int(scorer_name.replace("pca", "").replace("_diag", ""))
            inv_var_d = user_pca_inv_var_diag[(uid, d)]
            mu_d = user_pca_mu[(uid, d)]
            cand_z_arr = np.stack([c["z"][d] for c in cand_z_dict["cands"]], axis=0)
            diffs = cand_z_arr - mu_d[None, :]
            maha = np.sqrt(np.maximum((diffs * diffs * inv_var_d[None, :]).sum(axis=1), 1e-8))
            return -maha
        if scorer_name.startswith("pca") and "_lw" in scorer_name:
            d = int(scorer_name.replace("pca", "").replace("_lw", ""))
            inv_sigma = user_pca_lw_inv[(uid, d)]
            mu_d = user_pca_mu[(uid, d)]
            cand_z_arr = np.stack([c["z"][d] for c in cand_z_dict["cands"]], axis=0)
            diffs = cand_z_arr - mu_d[None, :]
            maha_sq = np.einsum("nd,de,ne->n", diffs, inv_sigma, diffs)
            return -np.sqrt(np.maximum(maha_sq, 1e-8))
        if scorer_name.startswith("pca") and "_global" in scorer_name:
            d = int(scorer_name.replace("pca", "").replace("_global", ""))
            inv_sigma = pca_global_inv_sigma[d]
            mu_d = user_pca_mu[(uid, d)]
            cand_z_arr = np.stack([c["z"][d] for c in cand_z_dict["cands"]], axis=0)
            diffs = cand_z_arr - mu_d[None, :]
            maha_sq = (diffs * diffs * inv_sigma[None, :]).sum(axis=1)
            return -np.sqrt(np.maximum(maha_sq, 1e-8))
        if scorer_name == "syntax":
            u_mean = user_syn_mean_ctrl[uid]
            cand_syn = np.stack([c["syntax_ctrl"] for c in cand_z_dict["cands"]], axis=0)
            return -np.linalg.norm(cand_syn - u_mean[None, :], axis=1)
        raise ValueError(f"Unknown scorer: {scorer_name}")

    for asin, cands in asin_to_cands.items():
        size = len(cands)
        if size < 2:
            continue
        uids_in_pool = list({c["uid"] for c in cands})
        for uid in uids_in_pool:
            if uid not in user_inv_var_diag_3584:
                continue

            cand_z_dict = {"cands": cands}

            for scorer_name in scorer_keys + ["syntax"]:
                if scorer_name == "syntax" and uid not in user_syn_mean_ctrl:
                    continue
                scores = score_pool(scorer_name, cand_z_dict, uid)
                order = np.argsort(-scores)
                rank1_idx = int(order[0])
                rank1_uid = cands[rank1_idx]["uid"]
                hit = rank1_uid == uid
                own_ranks = [i for i, c in enumerate(cands) if c["uid"] == uid]
                own_ranks_pos = [int(np.where(order == r)[0][0]) for r in own_ranks]

                if size >= 10:
                    bucket_key = "size>=10"
                elif size >= 5:
                    bucket_key = "size>=5"
                elif size >= 4:
                    bucket_key = "size>=4"
                elif size >= 3:
                    bucket_key = "size>=3"
                else:
                    bucket_key = "size>=2"

                intra_results[bucket_key].append({
                    "asin": asin, "uid": uid, "scorer": scorer_name,
                    "n_pool": size, "rank1_uid": rank1_uid, "is_hit": hit,
                    "own_cand_ranks": own_ranks_pos,
                    "mean_own_rank": float(np.mean(own_ranks_pos)) if own_ranks_pos else None,
                })

            # Oracle: per-user oracle (best coverage, ties by syntax)
            user_own_cands = [k for k, c in enumerate(cands) if c["uid"] == uid]
            user_own_coverages = []
            for k_rec in user_own_cands:
                # Find which rec this cand belongs to
                pass  # we don't have coverage info at pool level; use syntax best
            # Oracle from cands owned by this user: pick one with best syntax distance
            u_mean_syn = user_syn_mean_ctrl.get(uid, np.zeros(14))
            own_syn_scores = []
            for k in user_own_cands:
                dist = np.linalg.norm(cands[k]["syntax_ctrl"] - u_mean_syn)
                own_syn_scores.append(-dist)
            if own_syn_scores:
                oracle_order = np.argsort(-np.array(own_syn_scores))
                oracle_rank_pos = int(oracle_order[0])
                oracle_hit = oracle_rank_pos == 0  # if best own cand is the pool-best-by-syntax, oracle rank-1
                # Track: does ANY owned cand exist with full coverage? Use record-level data
                # Simpler: oracle rank = (1 / (1 + best_own_rank_pos)) ... use mean_own_rank
                bucket_key = "size>=10" if size >= 10 else f"size>={'5' if size >= 5 else '4' if size >= 4 else '3' if size >= 3 else '2'}"
                intra_results[bucket_key].append({
                    "asin": asin, "uid": uid, "scorer": "oracle_syntax",
                    "n_pool": size, "rank1_uid": cands[user_own_cands[oracle_order[0]]]["uid"] if user_own_cands else None,
                    "is_hit": oracle_hit,
                    "own_cand_ranks": [int(r) for r in oracle_order],
                    "mean_own_rank": float(np.mean(oracle_order)) if len(oracle_order) > 0 else None,
                })

    # Aggregate
    intra_summary = {}
    for size_key, items in intra_results.items():
        per_scorer = {}
        for it in items:
            sn = it["scorer"]
            if sn not in per_scorer:
                per_scorer[sn] = {"n_users": 0, "n_hit": 0, "ranks": []}
            per_scorer[sn]["n_users"] += 1
            if it["is_hit"]:
                per_scorer[sn]["n_hit"] += 1
            if it["mean_own_rank"] is not None:
                per_scorer[sn]["ranks"].append(it["mean_own_rank"])
        per_scorer_summary = {}
        for sn, stats in per_scorer.items():
            per_scorer_summary[sn] = {
                "n_users": stats["n_users"],
                "n_hit": stats["n_hit"],
                "rank1_pct": stats["n_hit"] / max(stats["n_users"], 1) * 100,
                "mean_own_rank": float(np.mean(stats["ranks"])) if stats["ranks"] else None,
            }
        intra_summary[size_key] = per_scorer_summary

    json.dump({
        "pool_sizes_top10": pool_sizes[:10],
        "intra_summary": intra_summary,
        "n_intra": {k: len(v) for k, v in intra_results.items()},
    }, open(OUT_INTRA, "w"), indent=2, ensure_ascii=False)
    print(f"[save] intra-product → {OUT_INTRA}")

    # === Step 9: Print summary ===
    print(f"\n=== Phase 35.C — K=8 + PCA + Shrinkage + Global sigma ===")
    print(f"{'scorer':<22} {'rank1_size>=10':>20} {'mean_rank':>12}")
    size10 = intra_summary.get("size>=10", {})
    for sn in scorer_keys + ["syntax", "oracle_syntax"]:
        if sn in size10:
            s = size10[sn]
            mr = s["mean_own_rank"] if s["mean_own_rank"] is not None else 0
            print(f"  {sn:<22} {s['n_hit']:>4}/{s['n_users']:<4} ({s['rank1_pct']:>5.1f}%) {mr:>10.2f}")

    # Coverage summary
    full_cov_counts = {k: 0 for k in scorer_keys + ["baseline_cand0", "syntax"]}
    for rec_i in range(len(cands_data)):
        mask = record_full_cov_mask[rec_i]
        if mask[0]:
            full_cov_counts["baseline_cand0"] += 1
        for k in scorer_keys + ["syntax"]:
            sel = int(np.argmax(record_scores[rec_i][k]))
            if mask[sel]:
                full_cov_counts[k] += 1
    print(f"\n=== Coverage ({len(cands_data)} records) ===")
    print(f"{'scorer':<22} {'full_cov':>15}")
    for k, n in full_cov_counts.items():
        print(f"  {k:<22} {n:>4} ({n/max(len(cands_data),1)*100:>5.1f}%)")

    json.dump({
        "intra_summary": intra_summary,
        "coverage": full_cov_counts,
        "n_records": len(cands_data),
    }, open(OUT_EVAL, "w"), indent=2, ensure_ascii=False)
    print(f"[save] eval → {OUT_EVAL}")
    print(f"\n[total] {time.time() - t0:.1f}s")
    return 0


def user_qwen_mu_wrapper(user_pca_mu: dict, uid: str, d: int) -> bool:
    """Helper to check if user has PCA stats."""
    return (uid, d) in user_pca_mu


if __name__ == "__main__":
    sys.exit(main())
