#!/usr/bin/env python3
"""Phase 35.B Minimal: 一次性 batched Qwen + spacy,避免重复调用。

输入:
  /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/phase35b_candidates.json (74 records × K=4)
  /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/user_sents_phase35b.json (74 user exemplars)

Pipeline (all batched, no per-record loop):
  1. Flatten all 296 candidates + all 370 user sents → ONE list
  2. Qwen forward once → 666 × 3584d mean-pool residuals
  3. spacy nlp.pipe once → 666 × 14d syntax features
  4. Per-user mu/sigma from user residuals
  5. Per-record per-candidate scores (vectorized):
     - S_residual: -Maha (precomputed inv_sigma)
     - S_syntax: -||syntax_ctrl - user_mean_syntax_ctrl|| (vectorized)
  6. Intra-product Rank-1: per asin pool, per user, rank all cands by 4 scorers

输出:
  result/phase35b/scores.json
  result/phase35b/intra_product_rank1.json
  result/phase35b/eval_summary.json
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
OUT_DIR = REPO_ROOT / "result/phase35b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_JSON = SCRATCH / "phase35b_candidates.json"
USER_SENTS_JSON = SCRATCH / "user_sents_phase35b.json"
USER_RESIDUAL_CACHE = SCRATCH / "phase35b_user_qwen_residuals.pt"

OUT_SCORES = OUT_DIR / "scores.json"
OUT_INTRA = OUT_DIR / "intra_product_rank1.json"
OUT_EVAL = OUT_DIR / "eval_summary.json"

# Function words
FUNCTION_WORDS = {"the", "a", "an", "and", "or", "but", "if", "because", "as",
                  "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did",
                  "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
                  "my", "your", "his", "its", "our", "their",
                  "this", "that", "these", "those",
                  "in", "on", "at", "with", "from", "to", "for", "of", "by", "into"}


def compute_syntax_features(texts: list[str]) -> np.ndarray:
    """ONE call to spacy nlp.pipe, returns (n, 14) syntax features."""
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
    """ONE call to Qwen forward for all texts, returns (n, 3584) mean-pool last layer."""
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
        print(f"  [qwen] {done}/{n} batches done", flush=True)
    return np.concatenate(out_vecs, axis=0)


def main():
    t0 = time.time()

    # === Load data ===
    print("[load] candidates + user sents...", flush=True)
    cands_data = json.load(open(CANDIDATES_JSON))
    user_sents = json.load(open(USER_SENTS_JSON))
    print(f"[load] {len(cands_data)} records, {len(user_sents)} users", flush=True)

    # === Flatten all texts (candidates + user sents) ===
    flat_texts = []
    flat_meta = []  # ('cand', rec_i, k) or ('user', uid, sent_i)
    for rec_i, rec in enumerate(cands_data):
        for k, q in enumerate(rec["candidates"]):
            flat_texts.append(q)
            flat_meta.append(("cand", rec_i, k))
    for uid, sents in user_sents.items():
        for s_i, s in enumerate(sents):
            flat_texts.append(s)
            flat_meta.append(("user", uid, s_i))
    print(f"[flatten] {len(flat_texts)} texts (cands + user sents)", flush=True)

    # === Step 1: Qwen forward ONCE for all 666 texts ===
    cache_data = None
    if USER_RESIDUAL_CACHE.exists():
        try:
            cache_data = torch.load(USER_RESIDUAL_CACHE, weights_only=False)
            if len(cache_data.get("residuals", [])) == len(flat_texts):
                print(f"[step-1] cache hit: {len(cache_data['residuals'])} residuals")
            else:
                print(f"[step-1] cache stale ({len(cache_data.get('residuals', []))} != {len(flat_texts)}), recompute")
                cache_data = None
        except Exception:
            cache_data = None
    if cache_data is None:
        print("[step-1] loading Qwen...")
        sys.path.insert(0, str(REPO_ROOT))
        from llm_client import create_qwen_local_client
        client = create_qwen_local_client(with_vllm=False)
        print(f"[step-1] Qwen loaded, {time.time()-t0:.1f}s")
        print(f"[step-1] Qwen forward {len(flat_texts)} texts (batched, ONE call per chunk)...")
        all_resids = qwen_forward_residuals(flat_texts, client, batch_size=8)
        print(f"[step-1] all_resids shape={all_resids.shape}, {time.time()-t0:.1f}s")
        torch.save({"residuals": all_resids, "meta": flat_meta}, USER_RESIDUAL_CACHE)
        print(f"[step-1] cached → {USER_RESIDUAL_CACHE}")
    else:
        all_resids = cache_data["residuals"]

    # Split residuals back to cand / user
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

    # === Step 3: per-user mu/sigma (Qwen residual) + pre-compute diagonal Maha ===
    user_qwen_mu: dict[str, np.ndarray] = {}
    user_qwen_inv_sigma_diag: dict[str, np.ndarray] = {}  # 1/sigma_diag for fast Maha
    user_qwen_sigma_full: dict[str, np.ndarray] = {}  # full cov, only used for Cholesky solve if needed
    for uid, vecs in user_resids_dict.items():
        V = np.stack(vecs, axis=0)
        mu = V.mean(axis=0)
        user_qwen_mu[uid] = mu
        if V.shape[0] >= 2:
            cov = np.cov(V, rowvar=False).astype(np.float64)  # (3584, 3584)
            cov += 1e-3 * np.eye(V.shape[1], dtype=np.float64)
        else:
            cov = 1e-3 * np.eye(V.shape[1], dtype=np.float64)
        user_qwen_sigma_full[uid] = cov
        # Diagonal Maha: var_diag = diag(cov), maha^2 = sum (x-mu)^2 / var_diag
        var_diag = np.diag(cov).astype(np.float64)
        user_qwen_inv_sigma_diag[uid] = 1.0 / np.maximum(var_diag, 1e-6)
    print(f"[step-3] {len(user_qwen_mu)} users with mu/sigma (diag Maha precomputed)")

    # === Step 4: Length-controlled syntax ===
    # Build control matrix for ALL texts: [length, clause_count, attr_count]
    n_attrs_per_text = np.zeros(len(flat_texts))
    for i, meta in enumerate(flat_meta):
        if meta[0] == "cand":
            rec_i = meta[1]
            attrs = cands_data[rec_i]["attrs_used"]
            q = cands_data[rec_i]["candidates"][meta[2]]
            n_attrs_per_text[i] = count_attrs_covered(q, attrs)
        else:
            n_attrs_per_text[i] = 0  # user sents have no attrs context
    control = np.stack([all_syntax[:, 0], all_syntax[:, 1], n_attrs_per_text], axis=1)
    ctrl_mean = control.mean(axis=0)
    ctrl_std = control.std(axis=0) + 1e-8
    control_z = (control - ctrl_mean) / ctrl_std
    # orthogonalize each of 14 syntax features
    all_syntax_ctrl = np.zeros_like(all_syntax, dtype=np.float64)
    for d in range(14):
        y = all_syntax[:, d].astype(np.float64)
        XtX = control_z.T @ control_z
        Xty = control_z.T @ y
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(control_z, y, rcond=None)[0]
        all_syntax_ctrl[:, d] = y - (control_z @ beta)
    print(f"[step-4] all_syntax_ctrl shape={all_syntax_ctrl.shape}")

    # Per-user mean of length-controlled syntax (only user sents)
    user_syn_mean_ctrl: dict[str, np.ndarray] = {}
    for uid, syn_list in user_syntax_dict.items():
        # find global indices for this user's sents
        u_indices = [i for i, m in enumerate(flat_meta) if m[0] == "user" and m[1] == uid]
        u_syn_ctrl = all_syntax_ctrl[u_indices]
        user_syn_mean_ctrl[uid] = u_syn_ctrl.mean(axis=0)
    print(f"[step-4] {len(user_syn_mean_ctrl)} users with syntax mean")

    # Pre-build global index → row mapping (avoid O(n) flat_meta.index())
    syntax_row: dict[tuple, int] = {}
    for i, m in enumerate(flat_meta):
        syntax_row[m] = i

    # === Step 5: Per-record per-candidate scores (vectorized where possible) ===
    score_data = []
    for rec_i, rec in enumerate(cands_data):
        uid = rec["user_id"]
        asin = rec["asin"]
        attrs = rec["attrs_used"]
        cands = rec["candidates"]
        n_attr = len(attrs)
        coverages = np.array([count_attrs_covered(c, attrs) for c in cands])
        full_cov_mask = coverages == n_attr

        # S_residual: -Maha(cand_resid, user_qwen_mu) per cand (DIAGONAL Maha for speed)
        if uid in user_qwen_mu:
            mu_u = user_qwen_mu[uid]
            inv_var_diag = user_qwen_inv_sigma_diag[uid]  # (3584,)
            cand_resids_arr = np.stack([cand_resids_dict[(rec_i, k)] for k in range(len(cands))], axis=0)
            diffs = cand_resids_arr - mu_u[None, :]  # (n_cands, 3584)
            maha_sq = (diffs * diffs * inv_var_diag[None, :]).sum(axis=1)
            maha = np.sqrt(np.maximum(maha_sq, 1e-8))
            scores_residual = -maha
        else:
            scores_residual = np.zeros(len(cands))

        # S_syntax: -||cand_syntax_ctrl - user_syn_mean_ctrl||
        if uid in user_syn_mean_ctrl:
            cand_syn_rows = [syntax_row[("cand", rec_i, k)] for k in range(len(cands))]
            cand_syn_ctrl_arr = all_syntax_ctrl[cand_syn_rows]
            syn_dists = np.linalg.norm(cand_syn_ctrl_arr - user_syn_mean_ctrl[uid][None, :], axis=1)
            scores_syntax = -syn_dists
        else:
            scores_syntax = np.zeros(len(cands))

        # Coverage-filtered selections
        scores_residual_filt = scores_residual.copy()
        scores_syntax_filt = scores_syntax.copy()
        if full_cov_mask.any():
            scores_residual_filt[~full_cov_mask] = -1e9
            scores_syntax_filt[~full_cov_mask] = -1e9

        # Hybrid z-score
        def zscore(x):
            s = x.std()
            return (x - x.mean()) / (s + 1e-8)
        z_res = zscore(scores_residual)
        z_syn = zscore(scores_syntax)
        scores_hybrid_05 = 0.5 * z_res + 0.5 * z_syn
        scores_hybrid_07 = 0.7 * z_res + 0.3 * z_syn
        scores_hybrid_05_filt = scores_hybrid_05.copy()
        scores_hybrid_07_filt = scores_hybrid_07.copy()
        if full_cov_mask.any():
            scores_hybrid_05_filt[~full_cov_mask] = -1e9
            scores_hybrid_07_filt[~full_cov_mask] = -1e9

        sel_baseline = 0
        sel_residual = int(np.argmax(scores_residual_filt))
        sel_syntax = int(np.argmax(scores_syntax_filt))
        sel_hybrid_05 = int(np.argmax(scores_hybrid_05_filt))
        sel_hybrid_07 = int(np.argmax(scores_hybrid_07_filt))

        score_data.append({
            "user_id": uid,
            "asin": asin,
            "attrs_used": attrs,
            "n_attrs": n_attr,
            "intra_product_size": rec.get("intra_product_size", 1),
            "candidates": cands,
            "coverages": coverages.tolist(),
            "scores_residual": scores_residual.tolist(),
            "scores_syntax": scores_syntax.tolist(),
            "scores_hybrid_05": scores_hybrid_05.tolist(),
            "scores_hybrid_07": scores_hybrid_07.tolist(),
            "selections": {
                "baseline": {"idx": sel_baseline, "cov": int(coverages[sel_baseline]), "q": cands[sel_baseline]},
                "residual": {"idx": sel_residual, "cov": int(coverages[sel_residual]), "q": cands[sel_residual]},
                "syntax": {"idx": sel_syntax, "cov": int(coverages[sel_syntax]), "q": cands[sel_syntax]},
                "hybrid_05": {"idx": sel_hybrid_05, "cov": int(coverages[sel_hybrid_05]), "q": cands[sel_hybrid_05]},
                "hybrid_07": {"idx": sel_hybrid_07, "cov": int(coverages[sel_hybrid_07]), "q": cands[sel_hybrid_07]},
            },
        })
        if (rec_i + 1) % 10 == 0 or rec_i == len(cands_data) - 1:
            print(f"[step-5] {rec_i + 1}/{len(cands_data)} records scored", flush=True)

    # Save scores
    json.dump({"n_records": len(score_data), "records": score_data},
              open(OUT_SCORES, "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] scores → {OUT_SCORES}")

    # === Step 6: Intra-product Rank-1 ===
    asin_to_recs: dict[str, list[int]] = {}
    for i, r in enumerate(score_data):
        asin_to_recs.setdefault(r["asin"], []).append(i)

    # Build global cand pool per asin (use pre-built syntax_row dict, NOT list.index)
    asin_to_cands: dict[str, list[dict]] = {}
    for asin, rec_idxs in asin_to_recs.items():
        cands_in_pool = []
        for rec_i in rec_idxs:
            r = score_data[rec_i]
            for k, q in enumerate(r["candidates"]):
                cands_in_pool.append({
                    "uid": r["user_id"],
                    "text": q,
                    "resid": cand_resids_dict[(rec_i, k)],
                    "syntax_ctrl": all_syntax_ctrl[syntax_row[("cand", rec_i, k)]],
                })
        asin_to_cands[asin] = cands_in_pool
    pool_sizes = sorted([(a, len(c)) for a, c in asin_to_cands.items()], key=lambda x: -x[1])
    print(f"\n[step-6] pool sizes top 10: {pool_sizes[:10]}")

    print("[step-6] computing intra-product Rank-1...")
    intra_results: dict[str, list] = {f"size>={s}": [] for s in [2, 3, 4, 5, 10]}
    for asin, cands in asin_to_cands.items():
        size = len(cands)
        if size < 2:
            continue
        uids_in_pool = list({c["uid"] for c in cands})
        for uid in uids_in_pool:
            if uid not in user_qwen_mu:
                continue
            mu_u = user_qwen_mu[uid]
            inv_var_diag = user_qwen_inv_sigma_diag[uid]  # (3584,)
            u_mean_ctrl = user_syn_mean_ctrl.get(uid, np.zeros(14))

            # vectorized diagonal Maha + syntax for all cands in pool
            resid_stack = np.stack([c["resid"] for c in cands], axis=0)
            diffs = resid_stack - mu_u[None, :]
            maha_sq = (diffs * diffs * inv_var_diag[None, :]).sum(axis=1)
            maha = np.sqrt(np.maximum(maha_sq, 1e-8))
            s_res = -maha

            syn_stack = np.stack([c["syntax_ctrl"] for c in cands], axis=0)
            syn_dists = np.linalg.norm(syn_stack - u_mean_ctrl[None, :], axis=1)
            s_syn = -syn_dists

            def zscore(x):
                s = x.std()
                return (x - x.mean()) / (s + 1e-8)
            z_r = zscore(s_res)
            z_s = zscore(s_syn)
            s_h05 = 0.5 * z_r + 0.5 * z_s
            s_h07 = 0.7 * z_r + 0.3 * z_s

            for scorer_name, scores in [
                ("residual", s_res), ("syntax", s_syn),
                ("hybrid_05", s_h05), ("hybrid_07", s_h07),
            ]:
                order = np.argsort(-scores)
                rank1_idx = int(order[0])
                rank1_uid = cands[rank1_idx]["uid"]
                hit = rank1_uid == uid
                own_ranks = [i for i, c in enumerate(cands) if c["uid"] == uid]
                own_ranks_pos = [int(np.where(order == r)[0][0]) for r in own_ranks]
                # bucket assignment: size in {2,3,4,5,10}
                if size >= 10:
                    bucket_key = "size>=10"
                elif size >= 5:
                    bucket_key = "size>=5"
                elif size >= 4:
                    bucket_key = "size>=4"
                elif size >= 3:
                    bucket_key = "size>=3"
                elif size >= 2:
                    bucket_key = "size>=2"
                else:
                    continue
                intra_results[bucket_key].append({
                    "asin": asin, "uid": uid, "scorer": scorer_name,
                    "n_pool": size, "rank1_uid": rank1_uid, "is_hit": hit,
                    "own_cand_ranks": own_ranks_pos,
                    "mean_own_rank": float(np.mean(own_ranks_pos)) if own_ranks_pos else None,
                })

    intra_summary = {}
    for size_key, items in intra_results.items():
        n = len(items)
        n_hit = sum(1 for it in items if it["is_hit"])
        mean_ranks = [it["mean_own_rank"] for it in items if it["mean_own_rank"] is not None]
        intra_summary[size_key] = {
            "n_users": n,
            "n_hit": n_hit,
            "rank1_pct": n_hit / max(n, 1) * 100,
            "mean_own_rank": float(np.mean(mean_ranks)) if mean_ranks else None,
            "per_scorer": {},
        }
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

    # === Step 7: Eval summary ===
    summary = {}
    for cond in ["baseline", "residual", "syntax", "hybrid_05", "hybrid_07"]:
        full_cov_n = sum(1 for r in score_data if r["selections"][cond]["cov"] == r["n_attrs"])
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

    print(f"\n=== Phase 35.B 4-condition Rerank (size>=10 subset, 74 records) ===")
    print(f"{'condition':<10} {'full_cov':>12}")
    for cond, s in summary.items():
        print(f"{cond:<10} {s['n_full_cov']:>4}/{s['n_records']} ({s['full_cov_pct']:>5.1f}%)")

    print(f"\n=== Intra-product Rank-1 ===")
    for k, s in intra_summary.items():
        print(f"{k:<10} n_users={s['n_users']:>4}, rank1={s['n_hit']:>4} ({s['rank1_pct']:>5.1f}%), mean_rank={s['mean_own_rank']}")
        for scorer, ss in s["per_scorer"].items():
            print(f"  {scorer:<10} rank1={ss['n_hit']:>4}/{ss['n_users']:<4} ({ss['rank1_pct']:>5.1f}%) mean_rank={ss['mean_own_rank']}")

    print(f"\n[total] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
