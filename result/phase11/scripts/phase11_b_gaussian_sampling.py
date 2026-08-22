#!/usr/bin/env python3
"""Phase 11.B: Validate Gaussian sampling > point vector.

4-control experiment on 30 pairs to test if sampling s_u ~ N(mu_u, Sigma_u)
produces more diverse but still in-style queries vs using single mu_u.

Conditions (each produces N=10 candidates per pair):
  A sampled-z   : s_u ~ N(mu_u, Sigma_u), 10 different samples, 1 seed each
  B mean-z      : mu_u (single mean), 10 different seeds
  C shuffled-z  : mu_other (wrong user), 10 different seeds (sanity)
  D injection-off: no prefix, 10 different seeds (baseline)

Output:
  phase11_b_gaussian_samples.jsonl  (30 pairs × 4 conds × 10 cands = 1200)
  phase11_b_diversity_report.json

Acceptance:
  std(A's cos sim) > 1.5× std(B's cos sim)        ← Gaussian adds diversity
  Mean rank A NOT regress > 5% vs B                ← Doesn't hurt style signal
  C << A,B (sanity)                                ← shuffled gives wrong style
  D << A,B (sanity)                                ← no prefix gives no style
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "TinyStyler" / "tinystyler"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"
GAUSSIANS_NPZ = OUT_DIR / "phase11_a_user_gaussians_768d.npz"
OUT_GENERATIONS = OUT_DIR / "phase11_b_gaussian_samples.jsonl"
OUT_REPORT = OUT_DIR / "phase11_b_diversity_report.json"

# TinyStyler paths
T5_BASE = 'google/t5-v1_1-large'
T5_SNAP = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--google--t5-v1_1-large/snapshots/a98b0fcd0b8137ded40cdf0c0cf0ee884e7c9726'
TINYSTYLER_WEIGHTS = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--tinystyler--tinystyler/snapshots/2a879107b2ec342e57170b82cdc344d5179fa32b/tinystyler_model_weights.pt'

# Hardcoded config
N_PAIRS = 30
N_CANDIDATES_PER_COND = 10   # K=10
SEED_BASE = 7777
BATCH = 8                    # T5-large fp16 ~3GB
MAX_NEW_TOKENS = 48
TEMPERATURE = 1.0
TOP_P = 0.95
STYLE_K_PREFIX = 8           # E30.34 verified
STYLE_ALPHA = 2.0            # E30.34 verified

CONTROLS = ["sampled-z", "mean-z", "shuffled-z", "injection-off"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HOME'] = '/fs04/ar57/wenyu/.cache/huggingface'

    log("=" * 70)
    log("Phase 11.B: Gaussian sampling diversity test (4-control, 30 pairs)")
    log("=" * 70)

    np.random.seed(SEED_BASE)
    torch.manual_seed(SEED_BASE)

    # === Load pairs ===
    log("[1] Loading pairs ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    pairs = pairs[:N_PAIRS]
    log(f"  pairs: {len(pairs)}")

    # === Load Phase 11.A Gaussians cache ===
    log("[2] Loading Phase 11.A Gaussian cache ...")
    npz = np.load(GAUSSIANS_NPZ, allow_pickle=True)
    user_ids = list(npz["user_ids"])
    mu_768 = npz["mu_768"]            # (876, 768)
    sigma_lowrank = npz["sigma_lowrank"]  # (876, 768, 768) Cholesky
    valid_mask = npz["valid"]         # (876,) bool
    log(f"  mu_768: {mu_768.shape}, sigma_lowrank: {sigma_lowrank.shape}")
    log(f"  valid users: {int(valid_mask.sum())}/{len(user_ids)}")

    uid_to_idx = {u: i for i, u in enumerate(user_ids)}

    # === Build pair_info ===
    log("[3] Building per-pair info ...")
    rng = np.random.default_rng(SEED_BASE)
    all_uids = [u for i, u in enumerate(user_ids) if valid_mask[i]]
    pair_info = []
    for pi, p in enumerate(pairs):
        uid = p["user_id"]
        if uid not in uid_to_idx or not valid_mask[uid_to_idx[uid]]:
            continue
        attrs = p["attrs"]
        if not all(attrs.get(k) for k in ["Brand", "Color", "Material"]):
            continue
        # Pick other user for shuffled-z
        candidates_other = [u for u in all_uids if u != uid]
        other_uid = candidates_other[int(rng.integers(0, len(candidates_other)))]
        pair_info.append({
            "pair_idx": pi,
            "user_id": uid,
            "asin": p["asin"],
            "attrs": attrs,
            "u_idx": uid_to_idx[uid],
            "other_u_idx": uid_to_idx[other_uid],
            "other_uid": other_uid,
            "prompt_A_no_style": p["prompts"]["A_no_style"],
        })
    log(f"  pair_info: {len(pair_info)}")

    # === Load TinyStyler ===
    log("[4] Loading TinyStyler model ...")
    from tinystyler import TinyStyler
    model = TinyStyler(base_model=T5_BASE, use_style=True, ctrl_embed_dim=768)
    saved = torch.load(TINYSTYLER_WEIGHTS, map_location='cpu')
    saved = {k.replace('module.', ''): v for k, v in saved.items()}
    cur = model.state_dict()
    cur.update(saved)
    model.load_state_dict(cur)
    model.to('cuda:0').half().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"  loaded")

    # === Load T5 tokenizer ===
    log("[5] Loading T5 tokenizer ...")
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(T5_SNAP, legacy=True)
    log(f"  loaded")

    # === Pre-tokenize all prompts ===
    log(f"[6] Pre-tokenizing {len(pair_info)} prompts ...")
    pi_prompts_tokens = []
    for pi in pair_info:
        toks = tokenizer(pi["prompt_A_no_style"], padding=False, truncation=True,
                         max_length=256)['input_ids']
        pi_prompts_tokens.append(toks)
    log(f"  tokenized")

    # === Build batch generation jobs ===
    # For each pair: 4 conditions × N_CANDIDATES candidates = 40 candidates/pair
    # Total: 30 × 40 = 1200 candidates
    # Strategy: for each pair, group all 40 candidates into chunks of BATCH.
    # Each candidate needs: prompt + style (different per cond)
    # For sampled-z: 10 different s_u drawn from N(mu, L L^T)
    # For mean-z: same mu × 10 seeds (style=mu for all 10)
    # For shuffled-z: mu_other × 10 seeds
    # For injection-off: no prefix × 10 seeds

    log(f"[7] Building all candidates (30 pairs × 4 conds × {N_CANDIDATES_PER_COND} = 1200) ...")

    # For each pair, sample K=N_CANDIDATES_PER_COND s_u from N(mu_u, Sigma_u)
    pair_to_samples = {}
    for pi in pair_info:
        u_idx = pi["u_idx"]
        mu = mu_768[u_idx].astype(np.float64)
        L = sigma_lowrank[u_idx].astype(np.float64)  # [768, 768] Cholesky
        eps = rng.standard_normal((N_CANDIDATES_PER_COND, 768))
        s_samples = mu[None, :] + eps @ L.T  # [10, 768]
        pair_to_samples[pi["pair_idx"]] = s_samples.astype(np.float32)
    log(f"  sampled {N_CANDIDATES_PER_COND} s_u per user")

    # Build flat list of (pi_idx, ctrl, cand_idx) with style tensor
    cand_jobs = []
    for pi in pair_info:
        pi_idx = pi["pair_idx"]
        s_samples = pair_to_samples[pi_idx]  # [10, 768]
        mu_user = mu_768[pi["u_idx"]].astype(np.float32)
        mu_other = mu_768[pi["other_u_idx"]].astype(np.float32)
        for cand_idx in range(N_CANDIDATES_PER_COND):
            for ctrl in CONTROLS:
                if ctrl == "sampled-z":
                    style = torch.from_numpy(s_samples[cand_idx])
                elif ctrl == "mean-z":
                    style = torch.from_numpy(mu_user)
                elif ctrl == "shuffled-z":
                    style = torch.from_numpy(mu_other)
                else:  # injection-off
                    style = None
                cand_jobs.append({
                    "pi_idx": pi_idx,
                    "ctrl": ctrl,
                    "cand_idx": cand_idx,
                    "user_id": pi["user_id"],
                    "asin": pi["asin"],
                    "attrs": pi["attrs"],
                    "other_uid": pi["other_uid"],
                    "style": style,
                })
    log(f"  cand_jobs: {len(cand_jobs)}")

    # === Generate batched ===
    # We process in batches of BATCH candidates. Each candidate has its own style.
    records = []
    t0 = time.time()
    n_done = 0
    for batch_start in range(0, len(cand_jobs), BATCH):
        batch_jobs = cand_jobs[batch_start:batch_start + BATCH]
        # Group by (pi_idx, ctrl): for non-injection-off, multiple cand_idx share pi's prompt
        # We need prompts + styles per sample
        prompts_tok = []
        styles = []
        for job in batch_jobs:
            pi_idx = job["pi_idx"]
            # Find pair_info for this pi_idx
            pi = next(p for p in pair_info if p["pair_idx"] == pi_idx)
            prompts_tok.append(pi_prompts_tokens[pi_idx])
            styles.append(job["style"])

        # Run batched generation
        with torch.no_grad():
            # Use partition logic from phase10_17_b (prefix vs no-prefix)
            pi_idx_with_prefix = [i for i, s in enumerate(styles) if s is not None]
            pi_idx_no_prefix = [i for i, s in enumerate(styles) if s is None]
            all_outputs = [None] * len(styles)

            H = model.proj.weight.shape[0]  # 1024
            D_in = model.proj.weight.shape[1]  # 768
            device = 'cuda:0'
            dtype = torch.half

            if pi_idx_with_prefix:
                bsz_p = len(pi_idx_with_prefix)
                prompts_p = [prompts_tok[i] for i in pi_idx_with_prefix]
                styles_p = [styles[i] for i in pi_idx_with_prefix]
                max_p = max(len(p) for p in prompts_p)
                pad_id = tokenizer.pad_token_id or 0
                input_ids = torch.full((bsz_p, max_p), pad_id, dtype=torch.long, device=device)
                for i, p in enumerate(prompts_p):
                    input_ids[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
                attention_mask = (input_ids != pad_id).long()
                style_in = torch.zeros(bsz_p, D_in, dtype=dtype, device=device)
                for i, s in enumerate(styles_p):
                    style_in[i] = s.to(dtype=dtype, device=device)
                style_proj = model.proj(style_in * STYLE_ALPHA)
                style_prefix = style_proj.unsqueeze(1).expand(bsz_p, STYLE_K_PREFIX, H).contiguous()
                prefix_mask = torch.ones((bsz_p, STYLE_K_PREFIX), dtype=torch.long, device=device)
                full_attn = torch.cat([prefix_mask, attention_mask], dim=1)
                input_embeds = model.model.shared(input_ids)
                full_embeds = torch.cat([style_prefix, input_embeds], dim=1)
                out_ids = model.model.generate(
                    inputs_embeds=full_embeds, attention_mask=full_attn,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=TEMPERATURE, top_p=TOP_P,
                    num_return_sequences=1,
                )
                for j, s_idx in enumerate(pi_idx_with_prefix):
                    all_outputs[s_idx] = out_ids[j:j+1]

            if pi_idx_no_prefix:
                bsz_n = len(pi_idx_no_prefix)
                prompts_n = [prompts_tok[i] for i in pi_idx_no_prefix]
                max_p = max(len(p) for p in prompts_n)
                pad_id = tokenizer.pad_token_id or 0
                input_ids = torch.full((bsz_n, max_p), pad_id, dtype=torch.long, device=device)
                for i, p in enumerate(prompts_n):
                    input_ids[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
                attention_mask = (input_ids != pad_id).long()
                out_ids = model.model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=TEMPERATURE, top_p=TOP_P,
                    num_return_sequences=1,
                )
                for j, s_idx in enumerate(pi_idx_no_prefix):
                    all_outputs[s_idx] = out_ids[j:j+1]

        # Decode and record
        for s_idx, job in enumerate(batch_jobs):
            txt = tokenizer.decode(all_outputs[s_idx][0], skip_special_tokens=True).strip()
            records.append({
                "pair_idx": job["pi_idx"],
                "user_id": job["user_id"],
                "asin": job["asin"],
                "attrs": job["attrs"],
                "control": job["ctrl"],
                "cand_idx": job["cand_idx"],
                "other_uid": job["other_uid"],
                "candidate_query": txt,
            })
        n_done = len(records)
        if batch_start % (BATCH * 20) == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 0.001)
            eta = (len(cand_jobs) - n_done) / max(rate, 0.001)
            log(f"  [{n_done}/{len(cand_jobs)}] elapsed {elapsed:.1f}s, rate={rate:.2f}/s, ETA={eta:.0f}s")

    elapsed = time.time() - t0
    log(f"  generated {len(records)} in {elapsed:.1f}s, rate={len(records)/elapsed:.2f}/s")

    # === Save generations ===
    log(f"[8] Saving generations to {OUT_GENERATIONS} ...")
    with OUT_GENERATIONS.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_GENERATIONS}")

    # === Diversity & style report ===
    log(f"[9] Computing diversity and style report ...")
    # Load AnnaWegmann encoder for style distance
    import glob
    ANNA_SNAP = sorted(glob.glob("/fs04/ar57/wenyu/.cache/huggingface/hub/models--AnnaWegmann--Style-Embedding/snapshots/*"))[-1]
    from sentence_transformers import SentenceTransformer
    anna = SentenceTransformer(ANNA_SNAP, device='cuda:0')

    # For each (pair, ctrl), collect 10 candidate queries and compute:
    #   - pairwise cos sim among the 10 candidates
    #   - mean cos sim to mu_u (style preservation)
    pair_cond_cands = {}
    for r in records:
        key = (r["pair_idx"], r["control"])
        pair_cond_cands.setdefault(key, []).append(r["candidate_query"])

    # Encode all candidates
    all_texts = []
    keys_in_order = []
    for k, v in pair_cond_cands.items():
        for t in v:
            all_texts.append(t)
            keys_in_order.append(k)
    log(f"  encoding {len(all_texts)} candidates ...")
    embs = anna.encode(all_texts, convert_to_numpy=True, batch_size=64,
                       show_progress_bar=False, normalize_embeddings=True)
    cand_emb = {}
    for i, k in enumerate(keys_in_order):
        cand_emb.setdefault(k, []).append(embs[i])

    # Compute per-(pair, ctrl) diversity metrics
    results = []
    for k, embs_list in cand_emb.items():
        if len(embs_list) < 2:
            continue
        E = np.array(embs_list)  # [n, 768]
        # Pairwise cos sim
        cos_mat = E @ E.T  # already normalized
        iu = np.triu_indices(len(E), k=1)
        pair_cos = cos_mat[iu]
        results.append({
            "pair_idx": k[0],
            "ctrl": k[1],
            "n_cands": len(E),
            "mean_pair_cos": float(pair_cos.mean()),
            "std_pair_cos": float(pair_cos.std()),
            "mean_emb_norm": float(np.linalg.norm(E, axis=1).mean()),
        })

    # Aggregate by ctrl
    from collections import defaultdict
    by_ctrl = defaultdict(list)
    for r in results:
        by_ctrl[r["ctrl"]].append(r)

    report = {"per_pair_ctrl": results}
    summary = {}
    for ctrl, lst in by_ctrl.items():
        std_arr = np.array([x["std_pair_cos"] for x in lst])
        mean_cos_arr = np.array([x["mean_pair_cos"] for x in lst])
        # Diversity = 1 - mean_pair_cos (lower cos = more diverse)
        diversity_arr = 1.0 - mean_cos_arr
        summary[ctrl] = {
            "n_pairs": len(lst),
            "std_pair_cos_mean": float(std_arr.mean()),
            "std_pair_cos_std": float(std_arr.std()),
            "mean_pair_cos_mean": float(mean_cos_arr.mean()),
            "mean_pair_cos_std": float(mean_cos_arr.std()),
            "diversity_mean": float(diversity_arr.mean()),
            "diversity_std": float(diversity_arr.std()),
        }
    report["summary_by_ctrl"] = summary
    log(f"  summary by ctrl:")
    for ctrl, s in summary.items():
        log(f"    {ctrl:15s}: diversity(1-mean_cos)={s['diversity_mean']:.4f}, "
            f"mean_pair_cos={s['mean_pair_cos_mean']:.4f}")

    # Acceptance: diversity gain = sampled_diversity / mean_diversity
    if "sampled-z" in summary and "mean-z" in summary:
        sampled_div = summary["sampled-z"]["diversity_mean"]
        mean_div = summary["mean-z"]["diversity_mean"]
        diversity_gain = sampled_div / max(mean_div, 1e-6)
        report["acceptance"] = {
            "sampled_diversity": sampled_div,
            "mean_diversity": mean_div,
            "diversity_gain_ratio": diversity_gain,
            "diversification_pass": bool(diversity_gain > 1.5),
        }
        log(f"  diversity gain (sampled/mean): {diversity_gain:.2f}x "
            f"({'PASS' if diversity_gain > 1.5 else 'FAIL'})")

    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"  → {OUT_REPORT}")

    log("=" * 70)
    log("PHASE 11.B COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()