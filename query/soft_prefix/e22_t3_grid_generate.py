#!/usr/bin/env python3
"""E22 Task 3 Step 3: SCALE GRID generation diagnostics with FULL 318-dim z_user.

User directive (2026-08-16):
  - Input: FULL 318-dim z_user (projector trained with z_dim=318)
  - Generation: no 7-dim restriction
  - Evaluate: different Query LENGTHS x different Query COUNTS (aggregated)
  - Goal: find the minimal generation scale that stably carries user syntax.

For each (max_new_len, n_queries_per_user) cell, generate N test users x their
products; aggregate each user's generated-query 318-dim vectors and report:
  1. observability  — how many of 318 dims are non-zero in the aggregate
  2. stability      — repeat-generation (different seed) aggregate 318 vec
                       closeness for the same user
  3. fidelity       — generated aggregate vs user-review z_full (318) distance
                       relative to a random-user baseline
  4. controllability— swap z_A/z_B on the same product: aggregate syntax
                       follows the user (paired swap accuracy)
  5. content        — 5 attrs / digits / brand exact rates

Outputs (result/e22_t3/):
  e22_t3_grid_generations.jsonl — per cell per (user, asin, control, len, rep)
  e22_t3_grid_eval.json         — per-cell diagnostics
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
from e22_t3_generate import (  # noqa: E402
    user_prompt_tokens, load_cells, DTYPE, DEVICE, PROJ_HIDDEN, TOP_ATTR_KEYS,
)
from e22_t3_check3 import neutralize_content  # noqa: E402
from extract_syntactic_features import (  # noqa: E402
    per_sentence_features_v2, user_features_v2,
)
from extract_clause_features_single_query import load_spacy_model  # noqa: E402

MODEL_PATH = ("/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-"
              "Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
TASK1_VECTORS = REPO_ROOT / "result" / "e22_t1_user_vectors.npz"
TASK1_MANIFEST = REPO_ROOT / "result" / "e22_t1_manifest.json"
OUT_DIR = REPO_ROOT / "result" / "e22_t3"

SEED = 6666
Z_DIM = 318
NUM_TOKENS = 16
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.92
BATCH = 8
MAX_NEW_GRID = [16, 32, 64]      # query length cells
N_QUERIES_GRID = [2, 4, 8]       # aggregated query count cells
N_TEST_USERS = 120               # test users per cell (full test set = 306)
MAX_PRODS_PER_USER = 6           # max products sampled per user
# repetitions for stability estimation
N_REPS = 2

DTYPE_F = torch.bfloat16


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def user_z_full(user_ids: list[str], z_full: np.ndarray) -> dict[str, np.ndarray]:
    return {u: z_full[i] for i, u in enumerate(user_ids)}


def spacy318_full(text: str, attrs: dict, nlp, tm, ts) -> np.ndarray | None:
    qn = neutralize_content(text, attrs)
    doc = nlp(qn)
    sfs = [per_sentence_features_v2(s) for s in doc.sents]
    sfs = [s for s in sfs if s is not None]
    if not sfs:
        return None
    v = user_features_v2(sfs)
    if v is None:
        return None
    return ((v - tm) / ts).astype(np.float64)


def generate_batch(model, tok, copy_head, prefix_embeds, prompts,
                   attrs_list, max_new) -> list[str]:
    """Batch KV-cache decoding with copy mixing (content-preserving)."""
    bsz = len(prompts)
    emb = model.get_input_embeddings()
    pad_id = tok.pad_token_id or tok.eos_token_id
    encs = [list(p) for p in prompts]
    max_p = max(len(e) for e in encs)
    ids = torch.tensor([e + [pad_id] * (max_p - len(e)) for e in encs],
                       dtype=torch.long, device=DEVICE)
    masks = torch.tensor([[1] * len(e) + [0] * (max_p - len(e)) for e in encs],
                         dtype=torch.long, device=DEVICE)
    text_emb = emb(ids).to(DTYPE_F)
    has_prefix = any(p is not None for p in prefix_embeds)
    if has_prefix:
        prefix = torch.stack(
            [p if p is not None
             else torch.zeros(NUM_TOKENS, model.config.hidden_size,
                              dtype=DTYPE_F, device=DEVICE)
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
    # copy source attr mask
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
    next_emb = full
    next_ids = None
    src_hidden_cache = None
    for step in range(max_new):
        first = past is None
        out = model(inputs_embeds=next_emb if first else None,
                    input_ids=None if first else next_ids,
                    attention_mask=attn, position_ids=pos if first else None,
                    use_cache=True, past_key_values=past,
                    output_hidden_states=True)
        past = out.past_key_values
        if first:
            last_valid = attn.sum(dim=1) - 1
            logits = out.logits[torch.arange(bsz, device=DEVICE), last_valid]
            if any(use_copy):
                src_hidden = out.hidden_states[-1][
                    :, src_offset:src_offset + max_p, :]
                src_hidden_cache = src_hidden
                gen_hidden = out.hidden_states[-1][
                    torch.arange(bsz, device=DEVICE), last_valid].unsqueeze(1)
                p_copy, copy_logits = copy_head(
                    gen_hidden, src_hidden, src_attr_mask, ids)
                logits = mixed_logits(logits.unsqueeze(1), p_copy,
                                      copy_logits)[:, 0]
        else:
            logits = out.logits[:, -1]
            if any(use_copy):
                gen_hidden = out.hidden_states[-1][:, -1:, :]
                p_copy, copy_logits = copy_head(
                    gen_hidden, src_hidden_cache, src_attr_mask, ids)
                logits = mixed_logits(logits.unsqueeze(1), p_copy,
                                      copy_logits)[:, 0]
        logits = logits.float() / max(TEMPERATURE, 1e-4)
        if TOP_K > 0:
            v = torch.topk(logits, min(TOP_K, logits.size(-1)), dim=-1) \
                .values[:, -1].unsqueeze(1)
            logits = torch.where(logits < v,
                                 torch.full_like(logits, float("-inf")), logits)
        if TOP_P < 1.0:
            sorted_l, idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
            mask = cum - F.softmax(sorted_l, dim=-1) > TOP_P
            sorted_l[mask] = float("-inf")
            logits = torch.gather(sorted_l, 1, idx.argsort(dim=-1))
        probs = F.softmax(logits, dim=-1)
        gen_strs = [tok.decode(g, skip_special_tokens=True).lower()
                    for g in generated]
        for b in range(bsz):
            if done[b] or attrs_list[b] is None:
                continue
            if any(str(v).lower() not in gen_strs[b]
                   for v in attrs_list[b].values()):
                probs[b, tok.eos_token_id] = 0.0
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
        for b in range(bsz):
            if done[b]:
                continue
            tid = int(next_ids[b])
            generated[b].append(tid)
            if tid == tok.eos_token_id:
                done[b] = True
        if all(done):
            break
        next_emb = emb(next_ids.unsqueeze(1)).to(DTYPE_F)
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

    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_full = vd["Z_full"].astype(np.float64)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    user_to_zfull = user_z_full(user_ids, z_full)
    mt = json.load(open(TASK1_MANIFEST))
    test_users = sorted(mt["splits"]["test"]["ids"])
    rng = random.Random(SEED)
    rng.shuffle(test_users)
    test_users = test_users[:N_TEST_USERS]
    log(f"test users: {len(test_users)}")

    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    # cells: (user, product) from load_cells (reviews + meta)
    cells, _ = load_cells()
    cell_map: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        if c["user_id"] in test_users:
            # z_user in cells is 7-dim (Task-1 filtered); rebuild with FULL
            # 318-dim z_user for the 318-dim projector
            c["z_user"] = user_to_zfull[c["user_id"]].tolist()
            cell_map[c["user_id"]].append(c)
    # cap products per user
    for u in test_users:
        rng.shuffle(cell_map[u])
        cell_map[u] = cell_map[u][:MAX_PRODS_PER_USER]
    log(f"cells per user: {[len(cell_map[u]) for u in test_users[:5]]}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE_F, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    H = model.config.hidden_size
    ckpt = torch.load(OUT_DIR / "e22_t3_injector.pt", map_location=DEVICE)
    proj = SoftPrefixProjector(user_dim=Z_DIM, hidden_dim=PROJ_HIDDEN,
                               num_tokens=NUM_TOKENS, model_dim=H,
                               dtype=DTYPE_F, gate_init=0.1).to(DEVICE)
    proj.load_state_dict(ckpt["proj"])
    proj.eval()
    copy_head = CopyAwareHead(hidden_dim=H, vocab_size=model.config.vocab_size,
                              dtype=DTYPE_F).to(DEVICE)
    if "copy_head" in ckpt:
        copy_head.load_state_dict(ckpt["copy_head"])
    copy_head.eval()
    log("injector (318) + copy head loaded")

    # ---- generate per (len, rep) — real-z only (controllability uses
    # shuffled-z of the same cell; generate both) ----
    records = []
    jobs = []  # (user, cell, control, max_new, rep)
    for mn in MAX_NEW_GRID:
        for u in test_users:
            for c in cell_map[u]:
                for rep in range(N_REPS):
                    jobs.append((c, "real-z", mn, rep))
                    jobs.append((c, "shuffled-z", mn, rep))
    log(f"jobs: {len(jobs)}")

    zero_prefix = torch.zeros(NUM_TOKENS, H, dtype=DTYPE_F, device=DEVICE)
    rng2 = random.Random(SEED + 1)
    with torch.no_grad():
        for start in range(0, len(jobs), BATCH):
            chunk = jobs[start:start + BATCH]
            prompts, prefix_embeds, attrs_list, meta4 = [], [], [], []
            for c, ctrl, mn, rep in chunk:
                z_real = np.asarray(c["z_user"], dtype=np.float64)
                if ctrl == "real-z":
                    zz = z_real
                else:
                    # shuffled: farthest other test user z
                    o = max([x for x in test_users if x != c["user_id"]],
                            key=lambda x: float(
                                np.linalg.norm(user_to_zfull[x] - z_real)))
                    zz = user_to_zfull[o]
                prompts.append(user_prompt_tokens(c["attrs"], tok))
                if ctrl == "real-z":
                    pe = proj(torch.tensor(zz, dtype=torch.float32,
                                           device=DEVICE).to(DTYPE_F)
                              .unsqueeze(0))[0]
                else:
                    pe = proj(torch.tensor(zz, dtype=torch.float32,
                                           device=DEVICE).to(DTYPE_F)
                              .unsqueeze(0))[0]
                prefix_embeds.append(pe)
                attrs_list.append(c["attrs"])
                meta4.append((c, ctrl, mn, rep))
            texts = generate_batch(model, tok, copy_head, prefix_embeds,
                                   prompts, attrs_list, chunk[0][2])
            for (c, ctrl, mn, rep), txt in zip(meta4, texts):
                records.append({
                    "user_id": c["user_id"], "asin": c["asin"],
                    "attrs": c["attrs"], "control": ctrl,
                    "max_new": mn, "rep": rep, "query": txt,
                })
            if start % 320 == 0:
                log(f"  generated {start + len(chunk)}/{len(jobs)}")

    with open(OUT_DIR / "e22_t3_grid_generations.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"wrote {len(records)} grid generations; "
        f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
