#!/usr/bin/env python3
"""Phase 35 Full Pipeline: 4-condition rerank with S_residual + intra-product Rank-1.

输入:
  result/phase34/candidates.json (20 records × K=4)
  /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/syntax_gaussian_layer_20.npz
  /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/user_sents_per_user.json

Step:
  A. Compute Qwen 768d mean-pool residual for each user self sentence (per user, cached)
  B. Compute Qwen 768d mean-pool residual for each candidate
  C. Per-user mu_user_resid, Sigma_user_resid + Maha S_residual
  D. Per-candidate 14d syntax features + S_syntax (log prob under Layer 20 syntax Gaussian)
  E. Length-controlled syntax: orthogonalize 14d syntax by [length, clause_count, attr_count]
  F. 4 conditions rerank:
     - baseline: cand[0]
     - residual: argmax S_residual within full_cov
     - syntax: argmax S_syntax within full_cov
     - hybrid(α): argmax α*z(S_residual) + (1-α)*z(S_syntax) within full_cov, α ∈ {0.5}
  G. Intra-product Rank-1 (same asin pool, rank user true sentence by score)

输出:
  result/phase35/scores.json — per-record, per-candidate scores + selections
  result/phase35/intra_product_rank1.json — intra-product metrics per condition
  result/phase35/eval_summary.json — aggregate comparison + length-controlled syntax
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import multivariate_normal

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
OUT_DIR = REPO_ROOT / "result/phase35"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_JSON = Path(os.environ.get("P35_CANDIDATES", str(REPO_ROOT / "result/phase34/candidates.json")))
SYNTAX_GAUSSIAN_NPZ = SCRATCH / "syntax_gaussian_layer_20.npz"
USER_SENTS_JSON = Path(os.environ.get("P35_USER_SENTS", str(SCRATCH / "user_sents_per_user.json")))
USER_RESIDUAL_CACHE = SCRATCH / "user_qwen_residuals.pt"  # per-sentence cached

OUT_SCORES = OUT_DIR / "scores.json"
OUT_INTRA = OUT_DIR / "intra_product_rank1.json"
OUT_EVAL = OUT_DIR / "eval_summary.json"

# Qwen hidden (mean-pool last layer, 768d for 0.5B or 3584d for 7B - we use last layer's hidden_size)
QWEN_LAYER_FOR_RESID = -1  # last layer
# Layer 26 alpha=0 (no injection) = baseline generation, but for SCORING we use the hidden state

# Function word set
FUNCTION_WORDS = {"the", "a", "an", "and", "or", "but", "if", "because", "as",
                  "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did",
                  "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
                  "my", "your", "his", "its", "our", "their",
                  "this", "that", "these", "those",
                  "in", "on", "at", "with", "from", "to", "for", "of", "by", "into",
                  "about", "between", "through", "during", "before", "after",
                  "above", "below", "up", "down", "out", "off", "over", "under"}


def function_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w in FUNCTION_WORDS}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


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


def qwen_mean_pool_residuals(texts: list[str], client, batch_size: int = 8) -> np.ndarray:
    """Compute Qwen hidden last-layer mean-pool residual for each text.

    Returns: (n_texts, hidden_dim) numpy float32.
    """
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
        print(f"  [qwen-resid] {done}/{n} batches done", flush=True)
    return np.concatenate(out_vecs, axis=0)


def main():
    t0 = time.time()
    candidates_data = json.load(open(CANDIDATES_JSON))
    user_sents = json.load(open(USER_SENTS_JSON))
    print(f"[load] {len(candidates_data)} candidate records, {len(user_sents)} users with exemplars")

    # === Step A: Load Qwen (compute residual for users + candidates) ===
    print("[step-A] loading Qwen for residual extraction...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    print(f"[step-A] Qwen loaded, {time.time()-t0:.1f}s")

    # === Step B: Per-user self sentence Qwen mean-pool (cache) ===
    cache = {}
    if USER_RESIDUAL_CACHE.exists() and os.environ.get("P35_FORCE_RERESIDUAL", "0") != "1":
        try:
            cache = torch.load(USER_RESIDUAL_CACHE, weights_only=False)
            print(f"[step-B] cache hit: {len(cache)} users")
        except Exception:
            cache = {}
    all_user_sents_flat = []
    sent_uids_flat = []
    for uid, sents in user_sents.items():
        if uid not in cache or len(cache[uid]) != len(sents):
            for s in sents:
                all_user_sents_flat.append(s)
                sent_uids_flat.append(uid)
    if all_user_sents_flat:
        print(f"[step-B] computing Qwen residual for {len(all_user_sents_flat)} user sents (batch=16)...")
        new_resids = qwen_mean_pool_residuals(all_user_sents_flat, client, batch_size=8)
        # group by uid
        per_uid = {}
        for s, uid, r in zip(all_user_sents_flat, sent_uids_flat, new_resids):
            per_uid.setdefault(uid, []).append(r)
        cache.update(per_uid)
        torch.save(cache, USER_RESIDUAL_CACHE)
        print(f"[step-B] cached {len(cache)} users, {time.time()-t0:.1f}s")
    else:
        print(f"[step-B] all {len(cache)} users cached, no recompute needed")

    # per-user mu, Sigma from residuals (768d or 3584d)
    user_resid_mu: dict[str, np.ndarray] = {}
    user_resid_sigma: dict[str, np.ndarray] = {}
    user_resid_pool: dict[str, np.ndarray] = {}
    for uid, vecs in cache.items():
        V = np.stack(vecs, axis=0)  # (n, H)
        user_resid_pool[uid] = V
        user_resid_mu[uid] = V.mean(axis=0)
        if V.shape[0] >= 2:
            cov = np.cov(V, rowvar=False) + 1e-3 * np.eye(V.shape[1])
        else:
            cov = 1e-3 * np.eye(V.shape[1])
        user_resid_sigma[uid] = cov.astype(np.float32)
    print(f"[step-B] per-user mu/Sigma computed for {len(user_resid_mu)} users")

    # === Step C: Candidate Qwen mean-pool ===
    cand_records = []
    for rec in candidates_data:
        uid = rec["user_id"]
        asin = rec["asin"]
        attrs = rec["attrs_used"]
        cands = rec["candidates"]
        if not cands:
            continue
        cand_records.append({
            "user_id": uid, "asin": asin, "attrs": attrs, "candidates": cands,
        })
    all_cands_flat = []
    flat_idx = []
    for i, r in enumerate(cand_records):
        for k, q in enumerate(r["candidates"]):
            all_cands_flat.append(q)
            flat_idx.append((i, k))
    print(f"[step-C] computing Qwen residual for {len(all_cands_flat)} candidates...")
    cand_resids = qwen_mean_pool_residuals(all_cands_flat, client, batch_size=8)
    print(f"[step-C] cand_resids shape={cand_resids.shape}, {time.time()-t0:.1f}s")

    # === Step D: Syntax features (candidates + user self sents, ONE-TIME) ===
    print("[step-D] computing syntax features for all candidates...")
    cand_syntax = compute_syntax_features(all_cands_flat)
    print(f"[step-D] cand_syntax shape={cand_syntax.shape}")

    # All user self sentences (flatten across all needed users)
    all_user_sents_flat_list = []
    user_sents_uid_order = []
    for uid, sents in user_sents.items():
        for s in sents:
            all_user_sents_flat_list.append(s)
            user_sents_uid_order.append(uid)
    print(f"[step-D] computing syntax features for {len(all_user_sents_flat_list)} user self sents (one-time)...")
    user_sents_syntax = compute_syntax_features(all_user_sents_flat_list)
    print(f"[step-D] user_sents_syntax shape={user_sents_syntax.shape}")

    # Index user_sents syntax back to per-uid arrays
    user_sents_syntax_by_uid: dict[str, list[np.ndarray]] = {}
    idx = 0
    for uid, sents in user_sents.items():
        n = len(sents)
        user_sents_syntax_by_uid[uid] = [user_sents_syntax[idx + i] for i in range(n)]
        idx += n
    print(f"[step-D] cached user syntax for {len(user_sents_syntax_by_uid)} users")

    # per-record per-candidate S_residual and S_syntax
    # First: load syntax Gaussian from layer 20
    npz = np.load(SYNTAX_GAUSSIAN_NPZ, allow_pickle=True)
    user_ids_syn = [str(u) for u in npz["user_ids"]]
    syn_mu = npz["mu"]  # (n_users, n_keep)
    syn_sigma = npz["sigma"]
    syn_uid_to_idx = {u: i for i, u in enumerate(user_ids_syn)}
    print(f"[step-D] syntax Gaussian: {len(user_ids_syn)} users, n_keep={syn_mu.shape[1]}")

    # norm stats for raw 14d syntax features (computed from user self sents)
    all_user_sents_for_norm = []
    for uid, sents in user_sents.items():
        all_user_sents_for_norm.extend(sents)
    X_user_norm = compute_syntax_features(all_user_sents_for_norm)
    feat_mean = X_user_norm.mean(axis=0)
    feat_std = X_user_norm.std(axis=0) + 1e-8

    # === Step E: Length-controlled syntax (residualize 14d by [length, clause_count, attr_count]) ===
    # Strategy: regress 14d syntax features on [length, clause_count, attr_count], take residual
    # We need attr_count per candidate, and length/clause_count are in cand_syntax columns 0/1
    n_attrs_per_cand = np.zeros(len(all_cands_flat))
    for idx, (rec_i, k) in enumerate(flat_idx):
        q = cand_records[rec_i]["candidates"][k]
        n_attrs_per_cand[idx] = count_attrs_covered(q, cand_records[rec_i]["attrs"])
    # Build control matrix: [length, clause_count, attr_count] (3-dim)
    control = np.stack([
        cand_syntax[:, 0],   # length
        cand_syntax[:, 1],   # clause_count
        n_attrs_per_cand,
    ], axis=1).astype(np.float64)
    # control z-score
    ctrl_mean = control.mean(axis=0)
    ctrl_std = control.std(axis=0) + 1e-8
    control_z = (control - ctrl_mean) / ctrl_std

    # Orthogonalize each syntax feature column by control_z (linear regression, take residual)
    cand_syntax_ctrl = np.zeros_like(cand_syntax, dtype=np.float64)
    for d in range(cand_syntax.shape[1]):
        y = cand_syntax[:, d].astype(np.float64)
        XtX = control_z.T @ control_z
        Xty = control_z.T @ y
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(control_z, y, rcond=None)[0]
        y_pred = control_z @ beta
        cand_syntax_ctrl[:, d] = y - y_pred

    # === Step F: Per-record per-candidate scores ===
    n_records = len(cand_records)
    score_data = []
    for rec_i, rec in enumerate(cand_records):
        uid = rec["user_id"]
        asin = rec["asin"]
        attrs = rec["attrs"]
        cands = rec["candidates"]
        n_attr = len(attrs)
        coverages = np.array([count_attrs_covered(c, attrs) for c in cands])
        full_cov_mask = coverages == n_attr

        # S_residual via Maha distance (Qwen last-layer mean-pool residual)
        if uid in user_resid_mu:
            mu_u = user_resid_mu[uid]
            sigma_u = user_resid_sigma[uid]
            inv_sigma = np.linalg.pinv(sigma_u)
            diffs = []
            for k in range(len(cands)):
                idx = flat_idx.index((rec_i, k))
                r_k = cand_resids[idx]
                d = r_k - mu_u
                maha_sq = float(d @ inv_sigma @ d)
                maha = np.sqrt(max(maha_sq, 1e-8))
                diffs.append(-maha)
            scores_residual = np.array(diffs)
        else:
            scores_residual = np.zeros(len(cands))

        # S_syntax (Layer 20 syntax Gaussian log-prob)
        syn_idx = syn_uid_to_idx.get(uid)
        if syn_idx is not None:
            mu_s = syn_mu[syn_idx]
            sigma_s = syn_sigma[syn_idx]
            try:
                rv = multivariate_normal(mean=mu_s, cov=sigma_s)
                cand_syntax_z = (cand_syntax - feat_mean) / feat_std
                # only keep KEEP dims (mu/sigma already on KEEP subspace, so use all 14 to be safe — actually no, logpdf dim must match)
                # Phase 2 outputs mu/sigma in KEEP dims only, so we need to project cand_syntax_z to KEEP dims
                # But we don't have the projection matrix here. Workaround: full 14d syntax Gaussian directly (refit on user self sents)
                # Use length-controlled syntax features here too
                scores_syntax = np.zeros(len(cands))
                # Skip Phase 2 syntax Gaussian (dim mismatch), use length-controlled 14d syntax distance to user self mean
            except Exception:
                scores_syntax = np.zeros(len(cands))
        else:
            scores_syntax = np.zeros(len(cands))

        # S_syntax: length-controlled 14d syntax distance to user self mean (negative dist = higher score)
        u_sents = user_sents.get(uid, [])
        u_sents_syn = user_sents_syntax_by_uid.get(uid, [])
        if u_sents_syn:
            X_u_raw = np.array(u_sents_syn)
            # same control orthogonalization for user self sents
            u_ctrl = np.stack([X_u_raw[:, 0], X_u_raw[:, 1],
                               np.array([count_attrs_covered(s, {}) for s in u_sents])], axis=1)
            u_ctrl_mean = u_ctrl.mean(axis=0)
            u_ctrl_std = u_ctrl.std(axis=0) + 1e-8
            u_ctrl_z = (u_ctrl - u_ctrl_mean) / u_ctrl_std
            X_u_ctrl = np.zeros_like(X_u_raw)
            for d in range(X_u_raw.shape[1]):
                y = X_u_raw[:, d].astype(np.float64)
                XtX = u_ctrl_z.T @ u_ctrl_z
                Xty = u_ctrl_z.T @ y
                try:
                    beta = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(u_ctrl_z, y, rcond=None)[0]
                X_u_ctrl[:, d] = y - (u_ctrl_z @ beta)
            u_mean_ctrl = X_u_ctrl.mean(axis=0)
            # distance from each cand length-controlled syntax to user mean
            cand_ctrl_rows = []
            for k in range(len(cands)):
                idx = flat_idx.index((rec_i, k))
                cand_ctrl_rows.append(cand_syntax_ctrl[idx])
            cand_ctrl_arr = np.array(cand_ctrl_rows)
            syn_dists = np.linalg.norm(cand_ctrl_arr - u_mean_ctrl[None, :], axis=1)
            scores_syntax = -syn_dists
        else:
            scores_syntax = np.zeros(len(cands))

        # Coverage mask
        scores_syntax_filtered = scores_syntax.copy()
        scores_residual_filtered = scores_residual.copy()
        if full_cov_mask.any():
            scores_syntax_filtered[~full_cov_mask] = -1e9
            scores_residual_filtered[~full_cov_mask] = -1e9

        # 4 conditions
        sel_baseline = 0
        sel_residual = int(np.argmax(scores_residual_filtered))
        sel_syntax = int(np.argmax(scores_syntax_filtered))

        # hybrid z-score
        def zscore(x):
            s = x.std()
            return (x - x.mean()) / (s + 1e-8)
        z_res = zscore(scores_residual)
        z_syn = zscore(scores_syntax)
        scores_hybrid_05 = 0.5 * z_res + 0.5 * z_syn
        scores_hybrid_05_filtered = scores_hybrid_05.copy()
        if full_cov_mask.any():
            scores_hybrid_05_filtered[~full_cov_mask] = -1e9
        sel_hybrid_05 = int(np.argmax(scores_hybrid_05_filtered))

        scores_hybrid_07 = 0.7 * z_res + 0.3 * z_syn
        scores_hybrid_07_filtered = scores_hybrid_07.copy()
        if full_cov_mask.any():
            scores_hybrid_07_filtered[~full_cov_mask] = -1e9
        sel_hybrid_07 = int(np.argmax(scores_hybrid_07_filtered))

        if (rec_i + 1) % 10 == 0 or rec_i == len(cand_records) - 1:
            print(f"[step-F] {rec_i + 1}/{len(cand_records)} records scored", flush=True)

        score_data.append({
            "user_id": uid,
            "asin": asin,
            "attrs": attrs,
            "n_attrs": n_attr,
            "candidates": cands,
            "coverages": coverages.tolist(),
            "scores_residual": scores_residual.tolist(),
            "scores_syntax": scores_syntax.tolist(),
            "scores_hybrid_05": scores_hybrid_05.tolist(),
            "scores_hybrid_07": scores_hybrid_07.tolist(),
            "sel_baseline": sel_baseline,
            "sel_residual": sel_residual,
            "sel_syntax": sel_syntax,
            "sel_hybrid_05": sel_hybrid_05,
            "sel_hybrid_07": sel_hybrid_07,
            "sel_baseline_cov": int(coverages[sel_baseline]),
            "sel_residual_cov": int(coverages[sel_residual]),
            "sel_syntax_cov": int(coverages[sel_syntax]),
            "sel_hybrid_05_cov": int(coverages[sel_hybrid_05]),
            "sel_hybrid_07_cov": int(coverages[sel_hybrid_07]),
        })

    # === Step G: Save scores ===
    json.dump({"n_records": len(score_data), "records": score_data},
              open(OUT_SCORES, "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] scores → {OUT_SCORES}")

    # === Step H: Intra-product Rank-1 ===
    # 同 asin pool: 收集所有 (uid, cand, cand_resid, cand_syntax)
    # 对每 user u in pool, 用 4 个 scorer 重算所有 candidate vs u 的 score
    # 排序 → top-1 candidate's uid == u ? rank-1 hit
    asin_to_records: dict[str, list[int]] = {}
    for i, r in enumerate(score_data):
        asin_to_records.setdefault(r["asin"], []).append(i)
    pool_sizes = sorted([(a, len(ix)) for a, ix in asin_to_records.items()], key=lambda x: -x[1])
    print(f"\n[step-H] intra-product pool sizes: top 10 = {pool_sizes[:10]}")

    # global cand residuals (cand_resids indexed by flat_idx order)
    cand_global = []  # list of (uid, cand_text, cand_resid, cand_syntax_ctrl)
    for rec_i, rec in enumerate(cand_records):
        for k, q in enumerate(rec["candidates"]):
            idx = flat_idx.index((rec_i, k))
            cand_global.append({
                "uid": rec["user_id"],
                "text": q,
                "asin": rec["asin"],
                "resid": cand_resids[idx],
                "syntax": cand_syntax_ctrl[idx],
            })

    # Group by asin
    asin_to_cands: dict[str, list[dict]] = {}
    for c in cand_global:
        asin_to_cands.setdefault(c["asin"], []).append(c)

    print("[step-H] computing intra-product Rank-1 (per-user re-score)...")
    intra_results: dict[str, list] = {f"size>={s}": [] for s in [2, 3, 4, 5]}
    for asin, cands in asin_to_cands.items():
        size = len(cands)
        if size < 2:
            continue  # skip pool size < 2 (cannot rank) — proper handling: don't put in size>=4 bucket
        # unique users in pool
        uids_in_pool = list({c["uid"] for c in cands})
        for uid in uids_in_pool:
            # per scorer, rank all candidates by score vs THIS user u
            if uid not in user_resid_mu:
                continue
            mu_u = user_resid_mu[uid]
            sigma_u = user_resid_sigma[uid]
            inv_sigma = np.linalg.pinv(sigma_u)
            # user self length-controlled syntax mean (use cached syntax features)
            u_sents_syn = user_sents_syntax_by_uid.get(uid, [])
            if u_sents_syn:
                X_u_raw = np.array(u_sents_syn)
                u_sents_list = user_sents.get(uid, [])
                u_ctrl = np.stack([X_u_raw[:, 0], X_u_raw[:, 1],
                                   np.array([count_attrs_covered(s, {}) for s in u_sents_list])], axis=1)
                u_ctrl_mean = u_ctrl.mean(axis=0)
                u_ctrl_std = u_ctrl.std(axis=0) + 1e-8
                u_ctrl_z = (u_ctrl - u_ctrl_mean) / u_ctrl_std
                X_u_ctrl = np.zeros_like(X_u_raw)
                for d in range(X_u_raw.shape[1]):
                    y = X_u_raw[:, d].astype(np.float64)
                    XtX = u_ctrl_z.T @ u_ctrl_z
                    Xty = u_ctrl_z.T @ y
                    try:
                        beta = np.linalg.solve(XtX, Xty)
                    except np.linalg.LinAlgError:
                        beta = np.linalg.lstsq(u_ctrl_z, y, rcond=None)[0]
                    X_u_ctrl[:, d] = y - (u_ctrl_z @ beta)
                u_mean_ctrl = X_u_ctrl.mean(axis=0)
            else:
                u_mean_ctrl = np.zeros(cand_syntax_ctrl.shape[1])

            # S_residual for all cands in pool
            resid_stack = np.stack([c["resid"] for c in cands], axis=0)
            diffs = resid_stack - mu_u[None, :]
            maha = np.sqrt(np.maximum(np.einsum("nd,de,ne->n", diffs, inv_sigma, diffs), 1e-8))
            s_res = -maha

            # S_syntax (length-controlled distance to u_mean_ctrl, neg)
            syn_stack = np.stack([c["syntax"] for c in cands], axis=0)
            syn_dists = np.linalg.norm(syn_stack - u_mean_ctrl[None, :], axis=1)
            s_syn = -syn_dists

            # z-score
            def zscore(x):
                s = x.std()
                return (x - x.mean()) / (s + 1e-8)
            z_r = zscore(s_res)
            z_s = zscore(s_syn)
            s_hyb_05 = 0.5 * z_r + 0.5 * z_s
            s_hyb_07 = 0.7 * z_r + 0.3 * z_s

            for scorer_name, scores in [
                ("residual", s_res), ("syntax", s_syn),
                ("hybrid_05", s_hyb_05), ("hybrid_07", s_hyb_07),
            ]:
                # rank descending; rank-1 hit if top candidate's uid == u
                order = np.argsort(-scores)
                rank1_idx = int(order[0])
                rank1_uid = cands[rank1_idx]["uid"]
                hit = rank1_uid == uid
                # find own candidate rank
                own_ranks = [i for i, c in enumerate(cands) if c["uid"] == uid]
                own_ranks_pos = [int(np.where(order == r)[0][0]) for r in own_ranks]
                intra_results[f"size>={min(size, 5)}"].append({
                    "asin": asin, "uid": uid, "scorer": scorer_name,
                    "n_pool": size, "rank1_uid": rank1_uid, "is_hit": hit,
                    "own_cand_ranks": own_ranks_pos,
                    "mean_own_rank": float(np.mean(own_ranks_pos)) if own_ranks_pos else None,
                })

    # aggregate
    intra_summary = {}
    for size_key, items in intra_results.items():
        n = len(items)
        n_hit = sum(1 for it in items if it["is_hit"])
        # mean_own_rank
        mean_ranks = [it["mean_own_rank"] for it in items if it["mean_own_rank"] is not None]
        intra_summary[size_key] = {
            "n_users": n,
            "n_hit": n_hit,
            "rank1_pct": n_hit / max(n, 1) * 100,
            "mean_own_rank": float(np.mean(mean_ranks)) if mean_ranks else None,
            "per_scorer": {},
        }
        # per-scorer breakdown
        for scorer in ["residual", "syntax", "hybrid_05", "hybrid_07"]:
            sub = [it for it in items if it["scorer"] == scorer]
            n_h = sum(1 for it in sub if it["is_hit"])
            mean_r = [it["mean_own_rank"] for it in sub if it["mean_own_rank"] is not None]
            intra_summary[size_key]["per_scorer"][scorer] = {
                "n_users": len(sub),
                "n_hit": n_h,
                "rank1_pct": n_h / max(len(sub), 1) * 100,
                "mean_own_rank": float(np.mean(mean_r)) if mean_r else None,
            }

    json.dump({
        "pool_sizes_top10": pool_sizes[:10],
        "intra_summary": intra_summary,
        "intra_detail": intra_results,
    }, open(OUT_INTRA, "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] intra-product → {OUT_INTRA}")

    # === Step I: Aggregate eval summary ===
    # per-condition: full_cov, length-controlled syntax distance (if we have ref)
    summary = {}
    for cond in ["baseline", "residual", "syntax", "hybrid_05", "hybrid_07"]:
        full_cov_n = sum(1 for r in score_data if r[f"sel_{cond}_cov"] == r["n_attrs"])
        summary[cond] = {
            "n_records": len(score_data),
            "n_full_cov": full_cov_n,
            "full_cov_pct": full_cov_n / max(len(score_data), 1) * 100,
        }

    json.dump({
        "summary": summary,
        "intra_summary": intra_summary,
        "pool_sizes_top10": pool_sizes[:10],
    }, open(OUT_EVAL, "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] eval_summary → {OUT_EVAL}")

    # print summary
    print(f"\n=== Phase 35 4-condition Rerank Summary ===")
    print(f"{'condition':<10} {'full_cov':>12}")
    for cond, s in summary.items():
        print(f"{cond:<10} {s['n_full_cov']:>4}/{s['n_records']} ({s['full_cov_pct']:>5.1f}%)")
    print(f"\n=== Intra-product Rank-1 ===")
    for k, s in intra_summary.items():
        print(f"{k:<8} n_users={s['n_users']}, rank1={s['n_hit']} ({s['rank1_pct']:.1f}%)")

    print(f"\n[total] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
