#!/usr/bin/env python3
"""Phase 14.B v2: Layer × α grid sweep — extract per-layer hiddens first.

Sweep layers ∈ {8, 14, 18, 22, 26} × α ∈ {0.5, 1.0} + D_off = 11 conds.
Batched multi-cond: 11 conds × K=8 = 88 rows per generate call.

Step 1: Re-extract per-sentence hidden states at 5 layers for 198 users × 30 sents
Step 2: Fit Gaussian (μ, σ_diag) per user per layer
Step 3: Sweep with batched multi-cond + per-layer hooks
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")

# Source data: raw Amazon + phase10 user_ids
RAW_REVIEWS_GZ = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/data/Baby_Products_2023.jsonl.gz")
USER_IDS_NPZ = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase13_b_v2_user_hiddens_n100.npz")
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"

QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
LAYERS_USED = [8, 14, 18, 22, 26]
N_FIT = 30
K = 8
N_PAIRS = 30
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 80
SEED = 42
BATCH_SIZE = 32
MIN_WORDS = 5
MAX_WORDS = 60

# Sweep config
SWEEP = [
    ("D_off",     None, 0.0),
    ("A8_a0.5",    8,   0.5),
    ("A8_a1.0",    8,   1.0),
    ("A14_a0.5",  14,   0.5),
    ("A14_a1.0",  14,   1.0),
    ("A18_a0.5",  18,   0.5),
    ("A18_a1.0",  18,   1.0),
    ("A22_a0.5",  22,   0.5),
    ("A22_a1.0",  22,   1.0),
    ("A26_a0.5",  26,   0.5),
    ("A26_a1.0",  26,   1.0),
]

PROMPT_TEMPLATE = (
    "You are helping a user write a shopping search query. "
    "Given the product attributes below, write a natural, fluent search query "
    "that includes all the key attributes. Output ONLY the query.\n\n"
    "Attributes: {attrs}\n\n"
    "Search query:"
)

# Outputs
HIDDEN_OUT = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
JSONL_OUT = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
META_OUT = OUT_DIR / "phase14_b_meta.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_missing_attrs(q: str, attrs_str: str) -> str:
    q_low = q.lower()
    missing = []
    for kv in attrs_str.split(","):
        kv = kv.strip()
        if ":" not in kv:
            continue
        _, v = kv.split(":", 1)
        v = v.strip()
        if v and v.lower() not in q_low:
            missing.append(v)
    if missing:
        q = q.rstrip(".") + ", " + ", ".join(missing) + "."
    return q


def extract_user_sentences(user_ids_set: set) -> dict[str, list[str]]:
    """Read raw reviews, return {uid: [sent1, sent2, ...]} for target users."""
    log(f"  Reading {RAW_REVIEWS_GZ.name} ...")
    user_sents: dict[str, list[str]] = {uid: [] for uid in user_ids_set}
    n_lines = 0
    with gzip.open(RAW_REVIEWS_GZ, "rt") as f:
        for line in f:
            n_lines += 1
            if n_lines % 1_000_000 == 0:
                log(f"    scanned {n_lines/1e6:.1f}M lines, collected {sum(len(v) for v in user_sents.values())} sents")
            try:
                r = json.loads(line)
            except Exception:
                continue
            uid = r.get("user_id") or r.get("reviewerID")
            if uid not in user_ids_set:
                continue
            text = (r.get("text") or r.get("reviewText") or "").strip()
            if not text:
                continue
            # Split into sentences (split on . ! ? ; + chinese full-width)
            import re
            sents = re.split(r"[.!?;。！？；]+", text)
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                w = len(s.split())
                if MIN_WORDS <= w <= MAX_WORDS:
                    user_sents[uid].append(s)
                    if len(user_sents[uid]) >= N_FIT:
                        # early exit per user (not perfect but efficient)
                        pass
    log(f"  Total scanned: {n_lines/1e6:.1f}M, collected: {sum(len(v) for v in user_sents.values())} sents, "
        f"users with ≥1: {sum(1 for v in user_sents.values() if v)}")
    return user_sents


def extract_hiddens_for_layers(model, tokenizer, device, user_sents: dict, layers: list[int]) -> dict[str, np.ndarray]:
    """For each user, extract hidden states at specified layers for each sentence."""
    import torch
    H = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 3584
    user_hiddens = {}
    sents_data = []
    sent_user_idx = []  # which user each sent belongs to
    user_idx_list = list(user_sents.keys())
    for ui, uid in enumerate(user_idx_list):
        sents = user_sents[uid][:N_FIT]
        sents_data.extend(sents)
        sent_user_idx.extend([ui] * len(sents))
    log(f"  Total sentences to encode: {len(sents_data)} for {len(user_idx_list)} users")

    # Init per-user hidden buffer
    for uid in user_idx_list:
        user_hiddens[uid] = np.zeros((N_FIT, len(layers), H), dtype=np.float16)

    # Encode in batches
    t0 = time.time()
    for st in range(0, len(sents_data), BATCH_SIZE):
        chunk = sents_data[st:st + BATCH_SIZE]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=128).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs_all = out.hidden_states  # tuple of [B, T, H] for each layer
        for li, layer_idx in enumerate(layers):
            hs = hs_all[layer_idx]  # [B, T, H]
            mask = enc["attention_mask"]  # [B, T]
            # Mean-pool over real tokens
            mask_f = mask.unsqueeze(-1).float()
            pooled = (hs.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # [B, H]
            # Assign back
            for bi in range(pooled.size(0)):
                global_idx = st + bi
                ui = sent_user_idx[global_idx]
                sent_pos = global_idx - sum(len(user_sents[user_idx_list[k]]) for k in range(ui))  # sent pos in this user
                sent_pos = min(sent_pos, N_FIT - 1)
                uid = user_idx_list[ui]
                user_hiddens[uid][sent_pos, li] = pooled[bi].half().cpu().numpy()
        if (st // BATCH_SIZE) % 20 == 0:
            log(f"    batch {st // BATCH_SIZE + 1}/{(len(sents_data) + BATCH_SIZE - 1) // BATCH_SIZE} "
                f"({time.time() - t0:.1f}s)")
    log(f"  Extraction done in {time.time() - t0:.1f}s")
    return user_hiddens


def extract_neutral_per_layer(model, tokenizer, device, neutral_texts: list[str], layers: list[int]) -> dict[int, np.ndarray]:
    """Extract neutral hidden states at specified layers."""
    import torch
    out_per_layer = {l: [] for l in layers}
    for st in range(0, len(neutral_texts), BATCH_SIZE):
        chunk = neutral_texts[st:st + BATCH_SIZE]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=128).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs_all = out.hidden_states
        mask = enc["attention_mask"]
        mask_f = mask.unsqueeze(-1).float()
        for li, layer_idx in enumerate(layers):
            hs = hs_all[layer_idx].float()
            pooled = (hs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            out_per_layer[layer_idx].append(pooled.cpu().numpy())
    return {l: np.concatenate(out_per_layer[l], axis=0).mean(axis=0) for l in layers}


def main() -> None:
    log("=" * 70)
    log("Phase 14.B v2: Layer × α grid sweep (extract per-layer + sweep)")
    log("=" * 70)
    np.random.seed(SEED)

    log("[1] Load target user_ids (198 from existing cache) ...")
    user_data = np.load(str(USER_IDS_NPZ), allow_pickle=True)
    target_uids = list(user_data["user_ids"])
    log(f"  target users: {len(target_uids)}")

    log("[2] Extract sentences per user from raw reviews ...")
    user_sents = extract_user_sentences(set(target_uids))
    valid_uids = [u for u in target_uids if len(user_sents.get(u, [])) >= N_FIT]
    log(f"  valid users with ≥{N_FIT} sents: {len(valid_uids)}")

    log("[3] Load Qwen ...")
    os.environ["QWEN_MODEL_PATH"] = QWEN_MODEL_PATH
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=False)
    model = client._hidden_backend.model
    tokenizer = client._hidden_backend.tokenizer
    device = client._hidden_backend.device
    log(f"  Qwen loaded on {device}, layers: {len(model.model.layers)}")

    log(f"[4] Extract per-sentence hiddens at layers {LAYERS_USED} ...")
    user_hiddens = extract_hiddens_for_layers(model, tokenizer, device,
                                                {u: user_sents[u][:N_FIT] for u in valid_uids},
                                                LAYERS_USED)

    # Save cache
    log(f"[5] Cache per-layer hiddens: {HIDDEN_OUT}")
    np.savez(HIDDEN_OUT,
             user_ids=np.array(valid_uids),
             hiddens=np.stack([user_hiddens[u] for u in valid_uids]),  # [U, N_FIT, n_layers, H]
             layers=np.array(LAYERS_USED))
    log(f"  cached: {HIDDEN_OUT}")

    log("[6] Extract neutral per-layer (use existing neutral texts if available) ...")
    n_d = np.load(OUT_DIR / "phase13_a_neutral_hiddens.npz", allow_pickle=True)
    n_vecs = n_d["vecs"].astype(np.float32)  # [N, 28, H]
    # For layers in LAYERS_USED, compute neutral mean across all neutral sentences
    neutral_per_layer = {}
    for layer_idx in LAYERS_USED:
        neutral_per_layer[layer_idx] = n_vecs[:, layer_idx, :].mean(axis=0)
    log(f"  neutral per layer computed for {list(neutral_per_layer.keys())}")

    log("[7] Fit Gaussian per user per layer ...")
    user_gauss = {}  # uid → {layer: {"mu", "sigma_diag"}}
    for uid in valid_uids:
        h = user_hiddens[uid]  # [N_FIT, n_layers, H]
        user_gauss[uid] = {}
        for li, layer_idx in enumerate(LAYERS_USED):
            layer_h = h[:, li, :].astype(np.float32)  # [N_FIT, H]
            neutral = neutral_per_layer[layer_idx]
            residual = layer_h - neutral[None, :]
            mu = residual.mean(axis=0)
            sigma = residual.std(axis=0, ddof=0)
            user_gauss[uid][layer_idx] = {"mu": mu, "sigma_diag": sigma}
    log(f"  Gaussians fit for {len(user_gauss)} users × {len(LAYERS_USED)} layers")

    # Load pairs
    log("[8] Load pairs ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    pairs = [p for p in pairs if p["user_id"] in user_gauss][:N_PAIRS]
    log(f"  filtered pairs: {len(pairs)}")

    log("[9] Register per-layer hooks ...")
    n_conds = len(SWEEP)
    rows_per_call = n_conds * K
    cond_layer = [spec[1] for spec in SWEEP]
    cond_alpha = [spec[2] for spec in SWEEP]
    cond_is_off = [spec[0] == "D_off" for spec in SWEEP]
    steer_per_cond: list = [None] * n_conds

    def make_layer_hook(layer_idx):
        def _hook(module, args, output):
            import torch
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            for cond_idx in range(n_conds):
                if cond_layer[cond_idx] != layer_idx:
                    continue
                v = steer_per_cond[cond_idx]
                if v is None:
                    continue
                v_dev = v.to(h.device).to(h.dtype)
                for k in range(K):
                    b = cond_idx * K + k
                    h[b, -1] = h[b, -1] + v_dev
            if is_tuple:
                return (h,) + output[1:]
            return h
        return _hook

    handles = []
    for layer_idx in LAYERS_USED:
        h = model.model.layers[layer_idx].register_forward_hook(make_layer_hook(layer_idx))
        handles.append(h)
    log(f"  hooks registered on layers {LAYERS_USED}")

    log("[10] Batched sweep ...")
    all_results = []
    t_start = time.time()
    try:
        for i, pair in enumerate(pairs):
            uid = pair["user_id"]
            asin = pair["asin"]
            attrs = pair["attrs"]
            attrs_str = ", ".join(f"{k}: {v}" for k, v in attrs.items() if v)
            prompt = PROMPT_TEMPLATE.format(attrs=attrs_str)
            prompts = [prompt] * rows_per_call

            import torch
            for cond_idx, spec in enumerate(SWEEP):
                cond_name, layer_idx, alpha = spec
                if cond_is_off[cond_idx]:
                    steer_per_cond[cond_idx] = None
                else:
                    g = user_gauss[uid][layer_idx]
                    mu = torch.tensor(g["mu"], dtype=torch.float32, device=device)
                    steer_per_cond[cond_idx] = alpha * mu

            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=256).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            input_len = enc["input_ids"].shape[1]
            row = 0
            for cond_idx, (cond_name, layer_idx, alpha) in enumerate(SWEEP):
                for k in range(K):
                    gen_ids = out[row][input_len:]
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    all_results.append({
                        "user_id": uid,
                        "asin": asin,
                        "attrs": attrs,
                        "condition": cond_name,
                        "layer": layer_idx,
                        "alpha": alpha,
                        "cand_local_idx": k,
                        "q_styled": gen_text,
                        "q_final_post": append_missing_attrs(gen_text, attrs_str),
                        "attrs_str": attrs_str,
                    })
                    row += 1
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (i + 1) * (len(pairs) - i - 1)
                log(f"    pair {i + 1}/{len(pairs)} ({elapsed:.1f}s elapsed, ETA {eta:.1f}s)")
    finally:
        for h in handles:
            h.remove()

    log(f"[11] Total records: {len(all_results)} (expected {len(pairs) * rows_per_call})")
    log(f"  sweep total: {time.time() - t_start:.1f}s")

    log(f"[12] Writing jsonl: {JSONL_OUT}")
    with JSONL_OUT.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    META_OUT.write_text(json.dumps({
        "phase": "14.B",
        "n_configs": n_conds,
        "config_names": [c[0] for c in SWEEP],
        "layers": [c[1] for c in SWEEP],
        "alphas": [c[2] for c in SWEEP],
        "n_pairs": len(pairs),
        "K": K,
        "N_fit": N_FIT,
        "layers_used": LAYERS_USED,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_total_records": len(all_results),
        "n_user_gaussians": len(user_gauss),
        "rows_per_call": rows_per_call,
    }, indent=2, ensure_ascii=False))
    log(f"  meta: {META_OUT}")

    log("=" * 70)
    log("PHASE 14.B SWEEP DONE — run eval next")
    log("=" * 70)


if __name__ == "__main__":
    main()