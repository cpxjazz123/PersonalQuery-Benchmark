#!/usr/bin/env python3
"""Phase 15.D: α × ρ sweep to find natural StyleVector generation.

User feedback (iter_039 followup): v1 α=1.0, ρ=0.5 too strong → language confusion,
reviewer-style commentary, character-level degradation. Need to find sweet spot:
  - Style margin still lifts (user task: "我们只关心风格")
  - Semantic preservation near D_off (don't regress content > 2%)
  - Qualitative: queries look natural (no乱码, no reviewer commentary)

Sweep grid:
  α ∈ {0.3, 0.5, 0.7}
  ρ ∈ {0.0, 0.3, 0.5}
  + D_off baseline (no hook)

Total: 9 StyleVector configs + 1 baseline = 10 conditions × 30 pairs × K=8 = 2400 records.
But generation is sequential (one Qwen session, hook toggled each config), so total time:
  - Qwen load: ~4 min (one-time)
  - Each config: ~70s (30 pairs × K=8)
  - 10 configs × 70s = ~12 min
  - Eval (CPU): ~1 min

Output:
  - phase15_d_alpha_rho_sweep.jsonl   (all 10 conds)
  - phase15_d_alpha_rho_sweep_meta.json
  - phase15_d_samples.json             (per-cond representative samples)
  - phase15_d_eval.json               (style margin + semantic per cond)
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_GAUSSIANS = OUT_DIR / "phase13_b_user_gaussians_qwen.npz"  # mean-pool v1
IN_PAIRS = OUT_DIR / "phase10_pairs_1000.jsonl"

OUT_QUERIES = OUT_DIR / "phase15_d_alpha_rho_sweep.jsonl"
OUT_META = OUT_DIR / "phase15_d_alpha_rho_sweep_meta.json"
OUT_SAMPLES = OUT_DIR / "phase15_d_samples.json"
OUT_EVAL = OUT_DIR / "phase15_d_eval.json"

QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

# === Hardcoded sweep ===
N_PAIRS = 30
K_CANDIDATES = 8
LAYER = 14
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 48
RANDOM_SEED = 42

# (alpha, rho) grid — drop α=0.7 (still too strong per 13.D v1 sample inspection)
SWEEP = [
    ("D_off", 0.0, 0.0),  # baseline, no hook
    ("A_a0.3_r0.0", 0.3, 0.0),
    ("A_a0.3_r0.3", 0.3, 0.3),
    ("A_a0.3_r0.5", 0.3, 0.5),
    ("A_a0.5_r0.0", 0.5, 0.0),
    ("A_a0.5_r0.3", 0.5, 0.3),
    ("A_a0.5_r0.5", 0.5, 0.5),
]

PROMPT_TEMPLATE = (
    "You are a search assistant. Generate a short shopping search query (5-15 words) "
    "for the product below. Use natural phrasing.\n\n"
    "Product attributes: {attrs}\n\n"
    "Search query:"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def make_hook(steer_vecs: list, steer_layer: int, steer_alpha: float):
    def _hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        for b in range(h.size(0)):
            v = steer_vecs[b]
            if v is None:
                continue
            v_dev = v.to(h.device).to(h.dtype)
            pos = h.size(1) - 1
            h[b, pos] = h[b, pos] + steer_alpha * v_dev
        if is_tuple:
            return (h,) + output[1:]
        return h
    return _hook


def get_module(model, layer: int):
    # Qwen2: model.model.layers[layer]
    return model.model.layers[layer]


def generate_for_config(client, model, tok, device, pairs: list, user_gauss: dict,
                        cond_name: str, alpha: float, rho: float, K: int) -> list[dict]:
    """Generate K candidates for each pair under a single (α, ρ) config.

    For D_off (alpha=0): no hook, no injection, just K independent samples.
    For A: hook installed; K samples each with independent ε (Gaussian sampling).
    """
    import torch  # local import for hook path
    log(f"  [{cond_name}] generating {len(pairs)} pairs × K={K} (α={alpha}, ρ={rho}) ...")
    is_baseline = (alpha == 0.0 and rho == 0.0)

    # Install hook (only matters if α > 0)
    handle = None
    if not is_baseline:
        layer_module = get_module(model, LAYER)
        # steer_vecs is set per-batch inside the loop
        steer_vecs = [None] * 1  # placeholder, replaced each batch
        handle = layer_module.register_forward_hook(make_hook(steer_vecs, LAYER, alpha))

    results: list[dict] = []
    t0 = time.time()
    try:
        for i, pair in enumerate(pairs):
            uid = pair["user_id"]
            asin = pair["asin"]
            attrs = pair["attrs"]
            attrs_str = ", ".join(f"{k}: {v}" for k, v in attrs.items() if v)
            prompt = PROMPT_TEMPLATE.format(attrs=attrs_str)
            # Build K repeats of the prompt for vLLM-style batch sampling
            prompts = [prompt] * K

            # Encode prompts (tokenize + repeat for K copies)
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(device)

            # Compute steer vector(s) — for D_off: None list
            if not is_baseline:
                if uid not in user_gauss:
                    log(f"    WARN: user {uid[:10]} not in Gaussian cache, skip")
                    continue
                mu = torch.tensor(user_gauss[uid]["mu"][LAYER], dtype=torch.float32,
                                  device=device)  # [H]
                std = torch.tensor(user_gauss[uid]["std"][LAYER], dtype=torch.float32,
                                   device=device)  # [H]

                # K independent ε, scaled by ρ
                if rho > 0:
                    eps = torch.randn(K, mu.size(0), device=device, dtype=torch.float32)
                    s = mu.unsqueeze(0) + rho * std.unsqueeze(0) * eps  # [K, H]
                else:
                    s = mu.unsqueeze(0).expand(K, -1)  # [K, H]
                # Subtract mean across K (mean-zero per-position) for per-batch diversity
                s = s - s.mean(dim=0, keepdim=True) + mu.unsqueeze(0)  # keep mean = mu
                steer_vecs[:] = [s[k] for k in range(K)]
                # Actually we need steer_vecs of length B
                # The hook reads steer_vecs[b] for b in range(B); B == K
                # So steer_vecs must be a list of K tensors
                steer_vecs.clear()
                steer_vecs.extend([s[k] for k in range(K)])

            # Generate
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            # Decode (skip prompt)
            input_len = enc["input_ids"].shape[1]
            for k in range(K):
                gen_ids = out[k][input_len:]
                gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                # Strip prompt remnant
                for prefix in ['Search query:', '"', 'Search Query:']:
                    if gen_text.startswith(prefix):
                        gen_text = gen_text[len(prefix):].strip()
                # Apply hard-copy post-processing (attrs → tail if missing)
                gen_text = append_missing_attrs(gen_text, attrs)
                results.append({
                    "user_id": uid,
                    "asin": asin,
                    "attrs": attrs,
                    "condition": cond_name,
                    "alpha": alpha,
                    "rho": rho,
                    "cand_local_idx": k,
                    "q_styled": gen_text,
                    "q_final_post": gen_text,
                })

            if (i + 1) % 10 == 0 or i + 1 == len(pairs):
                log(f"    {cond_name} {i + 1}/{len(pairs)} ({time.time()-t0:.1f}s)")
    finally:
        if handle is not None:
            handle.remove()

    return results


def append_missing_attrs(query: str, attrs: dict) -> str:
    """Hard-copy post-processing: append missing attrs at tail."""
    cand_lower = query.lower()
    missing = []
    for k, v in attrs.items():
        if not v or not str(v).strip():
            continue
        v_str = str(v).strip()
        if v_str.lower() not in cand_lower:
            missing.append(v_str)
    if missing:
        return query.rstrip(".,;: ") + ", with " + ", ".join(missing) + "."
    return query


def evaluate_all(results: list[dict]) -> dict:
    """Run style rerank + semantic sim per cond."""
    log("[EVAL] Loading encoders ...")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")

    # 768d AnnaWegmann
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("AnnaWegmann/Style-Embedding")
    style_model = AutoModel.from_pretrained("AnnaWegmann/Style-Embedding", trust_remote_code=True).to("cuda:0")
    style_model.eval()

    # Load user embs
    user_embs_data = np.load("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase10_user_embs_768d.npz",
                             allow_pickle=True)
    user_embs = user_embs_data["embs"]  # [876, 768]
    user_ids = user_embs_data["user_ids"]
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    user_embs_norm = l2_normalize(user_embs.astype(np.float32))

    # Group by (uid, cond)
    log("[EVAL] Encoding all candidates ...")
    cand_by_key = defaultdict(list)
    for r in results:
        cand_by_key[(r["user_id"], r["condition"])].append(r)

    all_texts = [r["q_final_post"] for r in results]
    encoded = []
    bs = 64
    with torch.no_grad():
        for st in range(0, len(all_texts), bs):
            chunk = all_texts[st:st + bs]
            inp = tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
            inp = {k: v.to("cuda:0") for k, v in inp.items()}
            o = style_model(**inp)
            emb = o.pooler_output.cpu().numpy()  # [B, 768]
            encoded.append(emb)
    cand_embs = np.concatenate(encoded, axis=0)
    log(f"  cand_embs: {cand_embs.shape}")

    # Per-cond margin (compute once cleanly)
    log("[EVAL] Computing per-cond margins ...")
    cond_to_margin_clean = defaultdict(list)
    cand_norm = l2_normalize(cand_embs.astype(np.float32))
    for r, i in zip(results, range(len(results))):
        uid = r["user_id"]
        if uid not in uid_to_idx:
            continue
        target_idx = uid_to_idx[uid]
        target_emb = user_embs_norm[target_idx]
        cos_target = float(np.dot(cand_norm[i], target_emb))
        off_idxs = [j for j in range(len(user_ids)) if j != target_idx]
        cos_off_mean = float(np.mean(cand_norm[i].dot(user_embs_norm[off_idxs].T)))
        margin = cos_target - cos_off_mean
        cond_to_margin_clean[r["condition"]].append(margin)

    # Per-cond best-of-K margin (max across K)
    cond_to_best_of_k = defaultdict(list)
    by_key = defaultdict(list)
    for i, r in enumerate(results):
        by_key[(r["user_id"], r["condition"])].append(i)
    for key, idx_list in by_key.items():
        cond_name = key[1]
        ms = [cond_to_margin_clean[cond_name][idx] for idx in idx_list]
        if ms:
            cond_to_best_of_k[cond_name].append(max(ms))

    # Semantic sim via sentence-bert
    log("[EVAL] Loading sentence-bert for semantic ...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda:0")

    # Per-pair attrs string
    pair_uids = sorted({r["user_id"] for r in results})
    pair_attrs_text = {}
    for r in results:
        if r["user_id"] in pair_attrs_text:
            continue
        attrs = r["attrs"]
        pair_attrs_text[r["user_id"]] = ", ".join(f"{k}: {v}" for k, v in attrs.items() if v)
    attr_texts = [pair_attrs_text[u] for u in pair_uids]
    attr_embs = sbert.encode(attr_texts, convert_to_numpy=True, normalize_embeddings=True)

    cand_texts = [r["q_final_post"] for r in results]
    sbert_cand = sbert.encode(cand_texts, convert_to_numpy=True, normalize_embeddings=True)
    uid_to_attr = {u: i for i, u in enumerate(pair_uids)}

    cond_to_sem = defaultdict(list)
    for i, r in enumerate(results):
        a = uid_to_attr[r["user_id"]]
        sem = float(np.dot(sbert_cand[i], attr_embs[a]))
        cond_to_sem[r["condition"]].append(sem)

    # Aggregate
    eval_out = {}
    for cond in sorted(set(r["condition"] for r in results)):
        ms = cond_to_margin_clean.get(cond, [])
        bks = cond_to_best_of_k.get(cond, [])
        ss = cond_to_sem.get(cond, [])
        eval_out[cond] = {
            "n_candidates": len(ms),
            "mean_margin": float(np.mean(ms)) if ms else 0.0,
            "mean_best_of_K_margin": float(np.mean(bks)) if bks else 0.0,
            "mean_semantic_sim": float(np.mean(ss)) if ss else 0.0,
        }
    return eval_out


def collect_samples(results: list[dict]) -> dict:
    """For each cond, pick 5 user × 1 representative query (k=0)."""
    samples = defaultdict(list)
    seen_users = defaultdict(set)
    for r in results:
        cond = r["condition"]
        uid = r["user_id"]
        if uid in seen_users[cond] or len(samples[cond]) >= 5:
            continue
        samples[cond].append({
            "user_id": uid[:12] + "...",
            "asin": r["asin"][:12] + "...",
            "attrs": r["attrs"],
            "q_final_post": r["q_final_post"][:200],
        })
        seen_users[cond].add(uid)
    return dict(samples)


def main():
    log("=" * 70)
    log("Phase 15.D: α × ρ sweep — find natural StyleVector generation")
    log("=" * 70)

    np.random.seed(RANDOM_SEED)
    import torch
    torch.manual_seed(RANDOM_SEED)

    # === [1] Load pairs and Gaussian cache ===
    log("[1] Loading pairs and Gaussian cache (mean-pool v1) ...")
    user_ids_all: list[str] = []
    with IN_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            user_ids_all.append(obj["user_id"])
    user_ids_all = sorted(set(user_ids_all))

    gdata = np.load(IN_GAUSSIANS, allow_pickle=True)
    g_user_ids = [str(u) for u in gdata["user_ids"]]
    gmu = gdata["mu"]      # [n_users, 28, 3584]
    gstd = gdata["std_diag"]
    user_gauss = {g_user_ids[i]: {"mu": gmu[i], "std": gstd[i]} for i in range(len(g_user_ids))}
    valid_users = [u for u in user_ids_all if u in user_gauss][:N_PAIRS]

    # Build pairs (uid, asin, attrs) — first valid pair per user
    pair_lookup: dict[str, dict] = {}
    with IN_PAIRS.open() as f:
        for line in f:
            obj = json.loads(line)
            if obj["user_id"] in pair_lookup:
                continue
            pair_lookup[obj["user_id"]] = obj
    pairs = [pair_lookup[u] for u in valid_users if u in pair_lookup]
    log(f"  pairs: {len(pairs)}")

    # === [2] Load Qwen ===
    log(f"[2] Loading Qwen2-7B (transformers backend, fp16) ...")
    os.environ.setdefault("QWEN_GPU_MEMORY_UTILIZATION", "0.5")
    os.environ.setdefault("QWEN_MODEL_PATH", QWEN_MODEL_PATH)
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=False)  # transformers only for hook
    tok = client._hidden_backend.tokenizer
    model = client._hidden_backend.model
    device = client._hidden_backend.device
    log(f"  loaded, device={device}")

    # === [3] Sweep over (α, ρ) configs ===
    log("[3] Sweep over (α, ρ) configs ...")
    all_results: list[dict] = []
    t_sweep = time.time()
    for cond_name, alpha, rho in SWEEP:
        log(f"\n--- Config: {cond_name} (α={alpha}, ρ={rho}) ---")
        results = generate_for_config(
            client, model, tok, device, pairs, user_gauss,
            cond_name, alpha, rho, K_CANDIDATES,
        )
        all_results.extend(results)
        log(f"  → {len(results)} records, sweep total {time.time()-t_sweep:.1f}s")
        # Persist incrementally
        with OUT_QUERIES.open("w") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # === [4] Save samples ===
    log("[4] Collecting samples per cond ...")
    samples = collect_samples(all_results)
    OUT_SAMPLES.write_text(json.dumps(samples, indent=2, ensure_ascii=False))

    # === [5] Evaluate ===
    log("[5] Running eval (style margin + semantic) ...")
    eval_out = evaluate_all(all_results)

    # === [6] Verdict + save ===
    log("[6] Computing verdict ...")
    d_off_margin = eval_out.get("D_off", {}).get("mean_best_of_K_margin", 0.0)
    d_off_sem = eval_out.get("D_off", {}).get("mean_semantic_sim", 0.0)

    summary = []
    for cond in sorted(eval_out.keys()):
        e = eval_out[cond]
        d_margin = e["mean_best_of_K_margin"] - d_off_margin
        d_sem = e["mean_semantic_sim"] - d_off_sem
        summary.append({
            "condition": cond,
            "mean_best_K_margin": e["mean_best_of_K_margin"],
            "margin_diff_vs_D_off": d_margin,
            "mean_semantic_sim": e["mean_semantic_sim"],
            "sem_diff_vs_D_off": d_sem,
        })
    # Verdict: lift AND semantic close to D_off
    verdict = "TBD"
    for s in summary:
        if s["condition"] == "D_off":
            continue
        if s["margin_diff_vs_D_off"] > 0.02 and abs(s["sem_diff_vs_D_off"]) < 0.02:
            verdict = f"CANDIDATE_NATURAL: {s['condition']}"
            break

    OUT_EVAL.write_text(json.dumps({
        "phase": "15.D",
        "alpha_rho_sweep": eval_out,
        "summary": summary,
        "verdict": verdict,
        "decision_logic": (
            "For each (α, ρ): best-of-K style margin > D_off + 0.02 "
            "AND |sem_diff_vs_D_off| < 0.02 → CANDIDATE_NATURAL"
        ),
    }, indent=2, ensure_ascii=False))

    OUT_META.write_text(json.dumps({
        "phase": "15.D",
        "n_configs": len(SWEEP),
        "n_pairs": len(pairs),
        "k": K_CANDIDATES,
        "layer": LAYER,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "elapsed_sec": time.time() - t_sweep,
    }, indent=2))

    log("\n" + "=" * 70)
    log("Phase 15.D COMPLETE")
    log("=" * 70)
    log(f"\nVERDICT: {verdict}")
    log("\nSummary:")
    for s in summary:
        log(f"  {s['condition']:15s}: best_K_margin={s['mean_best_K_margin']:.4f} "
            f"(Δ={s['margin_diff_vs_D_off']:+.4f}), "
            f"sem_sim={s['mean_semantic_sim']:.4f} "
            f"(Δ={s['sem_diff_vs_D_off']:+.4f})")


if __name__ == "__main__":
    main()