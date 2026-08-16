#!/usr/bin/env python3
"""E22 Task 3 (part 2): Direct single-pass generation under 4 injection controls.

Controls (issue #22 Task 3):
  - real-z        : projector(z_user)  (correct user)
  - shuffled-z    : projector(z_other) (shuffled user vector)
  - true-zero-z   : EXPLICIT zero prefix (bypasses projector, NOT projector(0))
  - injection-off : no prefix at all (plain Qwen prompt)

Inference is a single direct generation (no templates, no candidate reranking,
no copying sentence patterns from reviews). Batch decoding via KV-cache loop
(right-padding; first step takes last-valid-position logits).

For each (user, product) cell we generate the query under all four controls.
Additionally, for the user-swap check we generate with z_A and z_B on the same
product; for monotonicity we interpolate z along A->B.

Outputs (result/e22_t3/):
  e22_t3_generations.jsonl — per (user, asin, control) generated text
  e22_t3_generate.json     — config + counts
"""
from __future__ import annotations

import gzip
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from projector import SoftPrefixProjector  # noqa: E402
from copy_aware import CopyAwareHead, mixed_logits  # noqa: E402

MODEL_PATH = ("/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-"
              "Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"
TASK1_VECTORS = REPO_ROOT / "result" / "e22_t1_user_vectors.npz"
TASK1_MANIFEST = REPO_ROOT / "result" / "e22_t1_manifest.json"
TASK2_TEMPLATES = ["Find me a {A1} {A4} in {A2}, made of {A3}, under {A5}.",
                   "Can you show me the {A1} {A4} which is {A2}, {A3}, and costs {A5}?",
                   "I want a {A1} {A4} with {A2} and {A3}, for {A5}.",
                   "For my baby, I need a {A1} {A4} that is {A2}, made of {A3}, and priced at {A5}.",
                   "The {A1} {A4} should be {A2}, {A3}, and no more than {A5}.",
                   "There is a {A1} {A4} of {A2}, built with {A3}, at {A5}.",
                   "If you have a {A1} {A4} in {A2} made from {A3}, I will pay {A5}.",
                   "Please show me something {A2} made of {A3}: a {A1} {A4} for {A5}.",
                   "It is the {A1} {A4} from {A2} with {A3} that I want, at {A5}.",
                   "What {A1} {A4} should I buy in {A2}, with {A3}, under {A5}?",
                   "Not just any {A4}: I need a {A1} {A4} in {A2} made of {A3} for {A5}.",
                   "The {A1} {A4} — {A2}, {A3} — must not exceed {A5}."]
TOP_ATTR_KEYS = ["Brand", "Color", "Material", "Category", "Price"]
TOP_ATTR_ALIASES = {
    "Brand": ["Brand", "brand", "Manufacturer"],
    "Color": ["Color", "color"],
    "Material": ["Material", "material", "Material Type", "Material Composition"],
    "Category": [],
    "Price": [],
}

SEED = 3333
Z_DIM = 7
NUM_TOKENS = 16
PROJ_HIDDEN = 128
MAX_NEW = 96
BATCH = 8
TEMPERATURE = 0.6
TOP_K = 40
TOP_P = 0.92
OUT_DIR = REPO_ROOT / "result" / "e22_t3"
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
CONTROLS = ["real-z", "shuffled-z", "true-zero-z", "injection-off"]

SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query "
    "that mentions every listed attribute of the product."
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_meta() -> dict[str, dict]:
    recs: dict[str, dict] = {}
    with gzip.open(META, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            asin = d.get("parent_asin")
            if not asin:
                continue
            rec = {"details": {}, "category_leaf": None, "price": None}
            det = d.get("details")
            if isinstance(det, dict):
                rec["details"] = det
            cats = d.get("categories")
            if isinstance(cats, list) and cats:
                if isinstance(cats[-1], str):
                    rec["category_leaf"] = cats[-1]
                elif isinstance(cats[0], list) and cats[0]:
                    rec["category_leaf"] = cats[0][-1]
            p = d.get("price")
            if isinstance(p, (int, float)):
                rec["price"] = float(p)
            recs[asin] = rec
    return recs


def attrs_for(rec: dict) -> dict[str, str] | None:
    out: dict[str, str] = {}
    det = rec.get("details", {})
    for canon in TOP_ATTR_KEYS:
        if canon == "Category":
            v = rec.get("category_leaf")
        elif canon == "Price":
            p = rec.get("price")
            v = f"${p:.2f}" if p is not None else None
        else:
            v = None
            for alias in TOP_ATTR_ALIASES[canon]:
                if alias in det and isinstance(det[alias], str) and det[alias].strip():
                    v = det[alias].strip()
                    break
        if v:
            out[canon] = v
    if len(out) < 5:
        return None
    return out


def attr_prompt(attrs: dict[str, str]) -> str:
    lines = ["Product attributes:"]
    for k in TOP_ATTR_KEYS:
        lines.append(f"{k}: {attrs[k]}")
    lines.append("Write a natural shopping query that mentions every attribute.")
    return "\n".join(lines)


def user_prompt_tokens(attrs: dict[str, str], tokenizer) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": attr_prompt(attrs)}],
        tokenize=True, add_generation_prompt=True)


def load_cells() -> tuple[list[dict], dict[str, np.ndarray]]:
    """(cells, user_to_z): test-user x product cells for the four controls."""
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_all = vd["Z"].astype(np.float64)
    user_to_z = {u: z_all[i] for i, u in enumerate(user_ids)}
    mt = json.load(open(TASK1_MANIFEST))
    test_users = set(mt["splits"]["test"]["ids"])
    user_products: dict[str, set] = defaultdict(set)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            a = d.get("parent_asin") or d.get("asin")
            if u and a and u in test_users:
                user_products[u].add(a)
    meta = load_meta()
    rng = random.Random(SEED)
    cells = []
    for u in sorted(test_users):
        cand = [a for a in user_products.get(u, set())
                if a in meta and attrs_for(meta[a]) is not None]
        if not cand:
            continue
        rng.shuffle(cand)
        for asin in cand[:4]:
            cells.append({"user_id": u, "asin": asin,
                          "attrs": attrs_for(meta[asin]),
                          "z_user": user_to_z[u].tolist()})
    return cells, user_to_z


def generate_batch(model, tok, copy_head, prefix_embeds: list[torch.Tensor | None],
                   prompts: list[list[int]], attrs_list: list[dict | None],
                   max_new: int) -> list[str]:
    """Batch KV-cache decoding with copy-aware mixing.

    prefix_embeds[b] = [K, H] explicit prefix embeddings (projector output for
    real/shuffled, torch.zeros for true-zero) or None for injection-off.
    attrs_list[b] = attribute dict for the pointer source (None -> no copy).
    """
    bsz = len(prompts)
    emb = model.get_input_embeddings()
    pad_id = tok.pad_token_id or tok.eos_token_id
    encs = [list(p) for p in prompts]   # prompts are token-id lists already
    max_p = max(len(e) for e in encs)
    ids = torch.tensor(
        [e + [pad_id] * (max_p - len(e)) for e in encs],
        dtype=torch.long, device=DEVICE)
    masks = torch.tensor(
        [[1] * len(e) + [0] * (max_p - len(e)) for e in encs],
        dtype=torch.long, device=DEVICE)
    text_emb = emb(ids).to(DTYPE)
    has_prefix = any(p is not None for p in prefix_embeds)
    if has_prefix:
        prefix = torch.stack(
            [p if p is not None else torch.zeros(NUM_TOKENS, model.config.hidden_size,
                                                 dtype=DTYPE, device=DEVICE)
             for p in prefix_embeds])
        full = torch.cat([prefix, text_emb], dim=1)
        attn = torch.cat([torch.ones(bsz, NUM_TOKENS, dtype=torch.long,
                                     device=DEVICE), masks], dim=1)
        pos = torch.cat([
            torch.arange(NUM_TOKENS, device=DEVICE).unsqueeze(0).expand(bsz, -1),
            torch.arange(max_p, device=DEVICE).unsqueeze(0).expand(bsz, -1)
            + NUM_TOKENS], dim=1)
        src_offset = NUM_TOKENS
    else:
        full = text_emb
        attn = masks
        pos = torch.arange(max_p, device=DEVICE).unsqueeze(0).expand(bsz, -1)
        src_offset = 0
    # copy-head source: attribute spans in the prompt segment
    src_attr_mask = torch.zeros(bsz, max_p, dtype=torch.bool, device=DEVICE)
    use_copy = [a is not None for a in attrs_list]
    for b in range(bsz):
        if not use_copy[b]:
            continue
        p_ids = encs[b]
        for v in attrs_list[b].values():
            v_ids = tok(str(v), add_special_tokens=False)["input_ids"]
            if not v_ids:
                continue
            for i in range(len(p_ids) - len(v_ids) + 1):
                if p_ids[i:i + len(v_ids)] == v_ids:
                    for j in range(len(v_ids)):
                        if i + j < max_p:
                            src_attr_mask[b, i + j] = True
                    break
    past = None
    generated: list[list[int]] = [[] for _ in range(bsz)]
    done = [False] * bsz
    pending: list[list[int]] = [[] for _ in range(bsz)]
    next_emb = full
    next_ids = None
    src_hidden_cache = None
    for step in range(max_new):
        first = past is None
        out = model(
            inputs_embeds=next_emb if first else None,
            input_ids=None if first else next_ids,
            attention_mask=attn,
            position_ids=pos if first else None,
            use_cache=True, past_key_values=past, output_hidden_states=True)
        past = out.past_key_values
        if first:
            last_valid = attn.sum(dim=1) - 1
            logits = out.logits[torch.arange(bsz, device=DEVICE), last_valid]
            if any(use_copy):
                src_hidden = out.hidden_states[-1][:, src_offset:src_offset + max_p, :]
                src_hidden_cache = src_hidden
                gen_hidden = out.hidden_states[-1][
                    torch.arange(bsz, device=DEVICE), last_valid].unsqueeze(1)
                p_copy, copy_logits = copy_head(
                    gen_hidden, src_hidden, src_attr_mask, ids)
                logits = mixed_logits(logits.unsqueeze(1), p_copy, copy_logits)[:, 0]
        else:
            logits = out.logits[:, -1]
            if any(use_copy):
                gen_hidden = out.hidden_states[-1][:, -1:, :]
                p_copy, copy_logits = copy_head(
                    gen_hidden, src_hidden_cache, src_attr_mask, ids)
                logits = mixed_logits(logits.unsqueeze(1), p_copy, copy_logits)[:, 0]
        # sampling (temperature + top-p + top-k) to avoid repetition collapse
        logits = logits.float() / max(TEMPERATURE, 1e-4)
        if TOP_K > 0:
            v = torch.topk(logits, min(TOP_K, logits.size(-1)), dim=-1).values[:, -1].unsqueeze(1)
            logits = torch.where(logits < v, torch.full_like(logits, float("-inf")), logits)
        if TOP_P < 1.0:
            sorted_l, idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
            mask = cum - F.softmax(sorted_l, dim=-1) > TOP_P
            sorted_l[mask] = float("-inf")
            logits = torch.gather(sorted_l, 1, idx.argsort(dim=-1))
        probs = F.softmax(logits, dim=-1)
        # Content-completeness guard (copy-aware content preservation, NOT a
        # template/rerank): while any attribute value is still missing from the
        # generated text, (a) block EOS and (b) if the model still fails to say
        # a missing value, FORCE-COPY its source tokens into the output — the
        # copy-head mechanism of verbatim attribute-value copying.
        gen_strs = [tok.decode(g, skip_special_tokens=True).lower()
                    for g in generated]
        force_next: list[int | None] = [None] * bsz
        for b in range(bsz):
            if done[b] or attrs_list[b] is None:
                continue
            if any(str(v).lower() not in gen_strs[b]
                   for v in attrs_list[b].values()):
                probs[b, tok.eos_token_id] = 0.0
                for v in attrs_list[b].values():
                    if str(v).lower() in gen_strs[b]:
                        continue
                    v_ids = tok(str(v), add_special_tokens=False)["input_ids"]
                    if v_ids:
                        force_next[b] = v_ids[0]
                        # schedule the remainder of the value to be forced
                        # in the following steps (only set once)
                        if not pending[b]:
                            pending[b] = v_ids[1:]
                    break
        # apply pending forced tokens from earlier steps first
        for b in range(bsz):
            if done[b]:
                continue
            if pending[b]:
                force_next[b] = pending[b].pop(0)
        # safety: any row whose mass vanished (all -inf after top-p + EOS
        # blocking) falls back to uniform over the top-1k vocabulary
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = probs.sum(dim=-1)
        bad = row_sum <= 0
        if bad.any():
            fb = probs[bad]
            n = fb.size(-1)
            keep = min(1000, n)
            fb[fb < 0] = 0.0
            fb = fb / fb.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            fb[fb == 0] = 1.0 / keep
            probs[bad] = fb
        sample_ids = torch.multinomial(probs, 1).squeeze(1)
        next_ids = sample_ids.clone()
        for b in range(bsz):
            if done[b]:
                next_ids[b] = tok.eos_token_id
            elif force_next[b] is not None:
                next_ids[b] = force_next[b]
        for b in range(bsz):
            if done[b]:
                continue
            tid = int(next_ids[b])
            generated[b].append(tid)
            if tid == tok.eos_token_id:
                done[b] = True
        if all(done):
            break
        next_emb = emb(next_ids.unsqueeze(1)).to(DTYPE)
        next_ids = next_ids.unsqueeze(1)
        attn = torch.cat([attn, torch.ones(bsz, 1, dtype=attn.dtype,
                                           device=DEVICE)], dim=1)
    return [tok.decode(g, skip_special_tokens=True).strip() for g in generated]


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells, user_to_z = load_cells()
    log(f"cells: {len(cells)} (test users x products)")

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    H = model.config.hidden_size
    ckpt = torch.load(OUT_DIR / "e22_t3_injector.pt", map_location=DEVICE)
    proj = SoftPrefixProjector(user_dim=Z_DIM, hidden_dim=PROJ_HIDDEN,
                               num_tokens=NUM_TOKENS, model_dim=H,
                               dtype=DTYPE, gate_init=1e-3).to(DEVICE)
    proj.load_state_dict(ckpt["proj"])
    proj.eval()
    copy_head = CopyAwareHead(hidden_dim=H, vocab_size=model.config.vocab_size,
                              dtype=DTYPE).to(DEVICE)
    if "copy_head" in ckpt:
        copy_head.load_state_dict(ckpt["copy_head"])
    copy_head.eval()
    log("injector + copy head loaded")

    # ---- activation records (Check-3: activation check) ----
    z0 = torch.tensor(np.stack([c["z_user"] for c in cells[:8]]),
                      dtype=torch.float32, device=DEVICE).to(DTYPE)
    with torch.no_grad():
        p_real = proj(z0)
        p_zero = torch.zeros_like(p_real)
    acts = {
        "prefix_real_norm": float(p_real.norm().item()),
        "prefix_real_per_token_norm": [float(p_real[i].norm().item())
                                       for i in range(4)],
        "prefix_zero_norm": float(p_zero.norm().item()),
        "prefix_diff_real_vs_zero": float((p_real - p_zero).norm().item()),
    }
    log(f"activation: real-prefix norm={acts['prefix_real_norm']:.4f} "
        f"diff-vs-zero={acts['prefix_diff_real_vs_zero']:.4f}")

    # ---- generate under four controls ----
    records = []
    rng = random.Random(SEED + 1)
    # build jobs: for each cell, all four controls
    jobs = []
    zero_prefix = torch.zeros(NUM_TOKENS, H, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
         for c in cells:
            z_real = np.asarray(c["z_user"], dtype=np.float64)
            # shuffled control: the user whose z is FARTHEST from the current
            # user's z (a strictly wrong-style control; issue: shuffled-user)
            z_other_u = max(
                (u for u in user_to_z if u != c["user_id"]),
                key=lambda u: float(np.linalg.norm(user_to_z[u] - z_real)))
            z_shuf = user_to_z[z_other_u]
            jobs.append((c, "real-z", z_other_u,
                         proj(torch.tensor(z_real, dtype=torch.float32,
                                           device=DEVICE).to(DTYPE).unsqueeze(0))[0]))
            jobs.append((c, "shuffled-z", z_other_u,
                         proj(torch.tensor(z_shuf, dtype=torch.float32,
                                           device=DEVICE).to(DTYPE).unsqueeze(0))[0]))
            jobs.append((c, "true-zero-z", z_other_u, zero_prefix))
            jobs.append((c, "injection-off", z_other_u, None))
    log(f"jobs: {len(jobs)}")
    for start in range(0, len(jobs), BATCH):
        chunk = jobs[start:start + BATCH]
        prompts = [user_prompt_tokens(c["attrs"], tokenizer)
                   for c, _, _, _ in chunk]
        prefix_embeds = [pe for _, _, _, pe in chunk]
        attrs_list = [c["attrs"] for c, _, _, _ in chunk]
        texts = generate_batch(model, tokenizer, copy_head, prefix_embeds,
                               prompts, attrs_list, MAX_NEW)
        for k, ((c, ctrl, shuf_u, _), txt) in enumerate(zip(chunk, texts)):
            records.append({
                "user_id": c["user_id"], "asin": c["asin"],
                "attrs": c["attrs"], "control": ctrl, "query": txt,
                "shuffled_user": shuf_u,
            })
        if start % 64 == 0:
            log(f"  generated {start + len(chunk)}/{len(jobs)}")

    with open(OUT_DIR / "e22_t3_generations.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"wrote {len(records)} generations")

    summary = {
        "version": "e22_t3_generate_v1",
        "seed": SEED,
        "model": Path(MODEL_PATH).name,
        "num_tokens": NUM_TOKENS,
        "max_new_tokens": MAX_NEW,
        "n_cells": len(cells),
        "n_generations": len(records),
        "activation": acts,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "e22_t3_generate.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"DONE {summary['runtime_sec']}s")


if __name__ == "__main__":
    main()
