#!/usr/bin/env python3
"""Phase 10.9.4f v2: CopyAwareHead Retrain (加速版).

设计动机 (diag_missing_attr_source.py):
  - 55 candidates 缺属性 = 100% GEN_FAIL
  - 字符级错误 (截断/替代/顺序错) + 语义替换 → copy head 3 epochs 训练不够

加速策略 (CLAUDE.md Rule 5f):
  - batch_size 2 → 4 (steps 减半)
  - 预编码 dataset.pkl 缓存 (tokenize 只做一次)
  - 跳过 LM warm-up (gate bias=-3 直接 sigmoid 0.05, 起点足够保守)
  - 8 epochs (vs 10) — 节省 2 epochs

ckpt:
  - 输入: phase10_9_2_projector.pt (frozen)
  - 输出: phase10_9_4f_copyhead.pt
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PROJ_CKPT = OUT_DIR / "phase10_9_2_projector.pt"
PREF_FILE = OUT_DIR / "preference_pairs_60.jsonl"
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)

OUT_COPYHEAD = OUT_DIR / "phase10_9_4f_copyhead.pt"
OUT_META = OUT_DIR / "phase10_9_4f_retrain_meta.json"
LOG_OUT = OUT_DIR / "phase10_9_4f_retrain_copyhead.log"
CACHE_PATH = OUT_DIR / "phase10_9_4f_train_cache.pkl"

# === 硬编码 ===
ATTR_FIELDS = ["Brand", "Color", "Material"]
SEED = 42
EPOCHS = 8
BATCH_SIZE = 4  # 2 → 4 (steps 减半)
LR = 3e-4
COPY_LAMBDA = 2.0  # 强 copy 权重
MAX_QUERY_LEN = 80
NUM_TOKENS = 8

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
)


def log(m):
    print(f"[phase10-9.4f] {m}", flush=True)


def find_attr_value_spans(prompt_str: str, attrs_used: Dict[str, str]):
    spans = {}
    used = []
    items = sorted(((k, str(v)) for k, v in attrs_used.items() if str(v)),
                   key=lambda kv: -len(kv[1]))
    for key, val in items:
        v = val.strip()
        if not v:
            continue
        idx = 0
        while True:
            pos = prompt_str.find(v, idx)
            if pos < 0:
                break
            ok = True
            for (us, ue) in used:
                if not (ue <= pos or pos + len(v) <= us):
                    ok = False; break
            if ok:
                used.append((pos, pos + len(v)))
                spans.setdefault(key, []).append((pos, pos + len(v)))
            idx = pos + 1
    return spans


def attr_token_spans(prompt_str: str, attrs_used: Dict[str, str], tokenizer):
    enc = tokenizer(prompt_str, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    char_spans = find_attr_value_spans(prompt_str, attrs_used)
    out = {}
    for key, spans in char_spans.items():
        key_spans = []
        for (cs, ce) in spans:
            tok = []
            for i, (os_, oe) in enumerate(offsets):
                if os_ < ce and oe > cs:
                    tok.append(i)
            if tok:
                key_spans.append([tok[0], tok[-1]])
        if key_spans:
            out[key] = key_spans
    return out, ids, offsets


def find_query_attr_pos(query: str, attrs_used: Dict[str, str], tokenizer):
    enc = tokenizer(query, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    char_spans = find_attr_value_spans(query, attrs_used)
    pos_set = set()
    for key, spans in char_spans.items():
        for (cs, ce) in spans:
            for i, (os_, oe) in enumerate(offsets):
                if os_ < ce and oe > cs:
                    pos_set.add(i)
    return pos_set


def build_attr_prompt_lines(attrs_used: Dict[str, str]) -> str:
    lines = ["Product attributes:"]
    def _sort_key(k):
        if k.startswith("A") and k[1:].isdigit():
            return (0, int(k[1:]))
        return (1, k)
    for key in sorted(attrs_used, key=_sort_key):
        lines.append(f"{key}: {attrs_used[key]}")
    lines.append("Write a natural shopping query that mentions every attribute.")
    return "\n".join(lines)


def filter_attrs(attrs: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in attrs.items() if k in ATTR_FIELDS and v}


SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query "
    "that mentions every listed attribute of the product by its exact value "
    "(brand name, weight, dimensions, color, material). Keep the query under 25 words."
)


def build_or_load_cache(rows, z_map, tokenizer):
    """一次性预编码所有 record, 缓存到 pkl."""
    if CACHE_PATH.exists():
        log(f"  loading cache: {CACHE_PATH}")
        with CACHE_PATH.open("rb") as f:
            return pickle.load(f)

    log(f"  building cache: {len(rows)} rows ...")
    cache = []
    for i, row in enumerate(rows):
        uid = row["user_id"]
        attrs_5 = row["attrs"]
        attrs = filter_attrs(attrs_5)
        prompt_str = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_attr_prompt_lines(attrs)},
            ],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        spans, _, _ = attr_token_spans(prompt_str, attrs, tokenizer)
        src_attr_mask = torch.zeros(len(prompt_ids), dtype=torch.bool)
        for spans_list in spans.values():
            for (a, b) in spans_list:
                src_attr_mask[a:b + 1] = True
        query = row["chosen_query"]
        target_ids = tokenizer.encode(query, add_special_tokens=False)[:MAX_QUERY_LEN]
        target_ids = target_ids + [tokenizer.eos_token_id]
        attr_pos = find_query_attr_pos(query, attrs, tokenizer)
        cache.append({
            "user_id": uid,
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "src_attr_mask": src_attr_mask,
            "attr_pos": list(attr_pos),
            "user_vec": torch.tensor(z_map.get(uid, np.zeros(40, dtype=np.float32)), dtype=torch.float32),
        })
    with CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f)
    log(f"  cache saved: {CACHE_PATH} ({len(cache)} records)")
    return cache


def collate(batch_list, pad_id):
    B = len(batch_list)
    p_max = max(len(b["prompt_ids"]) for b in batch_list)
    t_max = max(len(b["target_ids"]) for b in batch_list)

    prompt_ids = torch.full((B, p_max), pad_id, dtype=torch.long)
    target_ids = torch.full((B, t_max), -100, dtype=torch.long)
    src_attr_mask = torch.zeros((B, p_max), dtype=torch.bool)
    attention_mask = torch.zeros((B, p_max + t_max), dtype=torch.long)
    user_vecs = torch.zeros((B, 40), dtype=torch.float32)

    attr_pos_list = []
    for b, item in enumerate(batch_list):
        pl = len(item["prompt_ids"])
        prompt_ids[b, :pl] = item["prompt_ids"]
        src_attr_mask[b, :pl] = item["src_attr_mask"]
        tl = len(item["target_ids"])
        target_ids[b, :tl] = item["target_ids"]
        attention_mask[b, :pl + tl] = 1
        user_vecs[b] = item["user_vec"]
        attr_pos_list.append(item["attr_pos"])
    return {
        "prompt_ids": prompt_ids,
        "target_ids": target_ids,
        "src_attr_mask": src_attr_mask,
        "attention_mask": attention_mask,
        "user_vecs": user_vecs,
        "attr_pos": attr_pos_list,
    }


def main():
    log("=" * 70)
    log("Phase 10.9.4f v2: CopyAwareHead Retrain (8 epochs, bs=4, copy_lambda=2.0)")
    log("=" * 70)

    # === 1. Load Qwen ===
    log("[1] Loading Qwen2-7B (frozen) ...")
    from llm_client import _HiddenBackend
    try:
        _HiddenBackend.reset()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    backend = _HiddenBackend.get(QWEN_MODEL_PATH)
    model = backend.model
    tokenizer = backend.tokenizer
    DEVICE = next(model.parameters()).device
    VOCAB_SIZE = model.config.vocab_size
    MODEL_DIM = model.config.hidden_size
    log(f"  Qwen on {DEVICE}, dim={MODEL_DIM}, vocab={VOCAB_SIZE}")

    # === 2. Load projector (frozen) ===
    log("[2] Loading projector (frozen) ...")
    from query.soft_prefix.projector import SoftPrefixProjector
    from query.soft_prefix.copy_aware import CopyAwareHead

    ckpt = torch.load(PROJ_CKPT, map_location=DEVICE, weights_only=False)
    projector = SoftPrefixProjector(
        user_dim=ckpt["user_dim"], hidden_dim=ckpt["hidden_dim"],
        num_tokens=ckpt["num_tokens"], model_dim=ckpt["model_dim"],
        dtype=torch.float32, gate_init=ckpt["gate_init"],
    ).to(DEVICE)
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False
    log(f"  projector loaded + frozen")

    # === 3. Init CopyAwareHead ===
    log("[3] Init CopyAwareHead (trainable) ...")
    copy_head = CopyAwareHead(MODEL_DIM, VOCAB_SIZE, dtype=torch.bfloat16).to(DEVICE)
    with torch.no_grad():
        copy_head.gate[-1].bias.fill_(-3.0)  # sigmoid ≈ 0.05, 强 copy bias
    trainable = list(copy_head.parameters())
    log(f"  CopyAwareHead params: {sum(p.numel() for p in trainable)}")
    log(f"  gate init: bias=-3 (sigmoid ≈ 0.05)")

    # === 4. Load + pre-encode data ===
    log("[4] Loading training data + pre-encoding cache ...")
    rows = []
    with PREF_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            if "chosen_query" not in r or "attrs" not in r:
                continue
            attrs_f = filter_attrs(r["attrs"])
            if len(attrs_f) < 2:
                continue
            rows.append(r)
    log(f"  total rows: {len(rows)}")

    z_map = {}
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            row = json.loads(line)
            z_map[row["user_id"]] = np.concatenate([row["user_mu"], row["user_logvar"]]).astype(np.float32)
    log(f"  z_map: {len(z_map)} users")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(rows))
    n_val = max(4, len(rows) // 10)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    log(f"  train: {len(train_rows)}, val: {len(val_rows)}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # 先 build cache 再切 train/val
    all_cache = build_or_load_cache(rows, z_map, tokenizer)
    train_cache = [all_cache[i] for i in train_idx]
    val_cache = [all_cache[i] for i in val_idx]
    log(f"  cache ready: train={len(train_cache)}, val={len(val_cache)}")

    # === 5. Training loop ===
    log(f"[5] Training: {EPOCHS} epochs, bs={BATCH_SIZE}, copy_lambda={COPY_LAMBDA}")
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    n_steps = (len(train_cache) + BATCH_SIZE - 1) // BATCH_SIZE * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=n_steps, pct_start=0.1)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    copy_head.train()

    losses = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        # 每 epoch shuffle
        epoch_perm = np.random.default_rng(SEED + epoch).permutation(len(train_cache))
        epoch_cache = [train_cache[i] for i in epoch_perm]
        ep_lm = 0.0
        ep_ptr = 0.0
        n_batch = 0
        for step_start in range(0, len(epoch_cache), BATCH_SIZE):
            batch = epoch_cache[step_start:step_start + BATCH_SIZE]
            batch_dict = collate(batch, pad_id)

            prompt_ids = batch_dict["prompt_ids"].to(DEVICE)
            target_ids = batch_dict["target_ids"].to(DEVICE)
            attention_mask = batch_dict["attention_mask"].to(DEVICE)
            src_attr_mask = batch_dict["src_attr_mask"].to(DEVICE)
            user_vecs = batch_dict["user_vecs"].to(DEVICE)
            attr_pos_list = batch_dict["attr_pos"]

            text_emb = model.get_input_embeddings()(prompt_ids).to(model.dtype)
            target_emb = model.get_input_embeddings()(target_ids.clamp_min(0)).to(model.dtype)
            prefix = projector(user_vecs.float()).to(model.dtype)

            full_emb = torch.cat([prefix, text_emb, target_emb], dim=1)
            am = torch.cat([
                torch.ones((prompt_ids.size(0), NUM_TOKENS), dtype=attention_mask.dtype, device=DEVICE),
                attention_mask,
            ], dim=1)
            src_mask_full = torch.cat([
                torch.zeros((prompt_ids.size(0), NUM_TOKENS), dtype=torch.bool, device=DEVICE),
                src_attr_mask,
            ], dim=1)

            with torch.no_grad():
                out = model(inputs_embeds=full_emb, attention_mask=am, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1]
            gen_logits = out.logits

            src_len = prompt_ids.size(1)
            src_hidden = hidden[:, NUM_TOKENS:NUM_TOKENS + src_len, :]
            tgt_len = target_ids.size(1)
            tgt_hidden = hidden[:, NUM_TOKENS + src_len:, :]
            tgt_gen_logits = gen_logits[:, NUM_TOKENS + src_len:, :]

            # 注意: 这里是 src_attr_mask (B, S=prompt_len), 不是 src_mask_full (B, S+NUM_TOKENS)
            # copy_head.forward 期望 src_attr_mask 与 src_hidden 维度对齐
            p_copy, copy_logits = copy_head(tgt_hidden, src_hidden,
                                            src_attr_mask, prompt_ids)

            lm_loss = F.cross_entropy(
                tgt_gen_logits.reshape(-1, VOCAB_SIZE).float(),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

            T = target_ids.size(1)
            # gate_target[t] = 1 if p_copy_shift at position t (predicting target_ids[t+1])
            # should COPY. attr_pos_list[b] are token indices in FULL target_ids
            # (incl. EOS). To supervise "at position t, copy if next token is attr",
            # shift attr_pos by -1 (only positions with attr_pos >= 1).
            gate_target = torch.zeros(T - 1, dtype=torch.float32, device=DEVICE)
            for b in range(target_ids.size(0)):
                for pos in attr_pos_list[b]:
                    if pos >= 1 and pos - 1 < T - 1:
                        gate_target[pos - 1] = 1.0
            p_copy_shift = p_copy[:, :-1, 0]
            ptr_losses = []
            for b in range(target_ids.size(0)):
                tgt_b = target_ids[b, 1:]
                mask = tgt_b != -100
                if mask.sum() == 0:
                    continue
                pl = F.binary_cross_entropy_with_logits(
                    p_copy_shift[b][mask], gate_target.to(p_copy_shift.dtype)[mask]
                )
                ptr_losses.append(pl)
            ptr_loss = torch.stack(ptr_losses).mean() if ptr_losses else torch.tensor(0.0, device=DEVICE)

            loss = lm_loss + COPY_LAMBDA * ptr_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            opt.step()
            sched.step()

            ep_lm += float(lm_loss.detach())
            ep_ptr += float(ptr_loss.detach())
            n_batch += 1

            del full_emb, out, hidden, gen_logits, src_hidden, tgt_hidden, tgt_gen_logits
            torch.cuda.empty_cache()

        avg_lm = ep_lm / max(n_batch, 1)
        avg_ptr = ep_ptr / max(n_batch, 1)
        log(f"  epoch {epoch + 1}/{EPOCHS} done: lm={avg_lm:.4f} ptr={avg_ptr:.4f} "
            f"({time.time() - t0:.1f}s)")
        losses.append({"epoch": epoch + 1, "lm": avg_lm, "ptr": avg_ptr})

    # === 6. Save copy_head ===
    log("[6] Saving copy head ...")
    torch.save(copy_head.state_dict(), OUT_COPYHEAD)
    log(f"  saved → {OUT_COPYHEAD}")

    meta = {
        "n_train": len(train_cache),
        "n_val": len(val_cache),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "copy_lambda": COPY_LAMBDA,
        "gate_init_bias": -3.0,
        "attr_fields": ATTR_FIELDS,
        "proj_ckpt": str(PROJ_CKPT),
        "final_losses": losses,
        "elapsed_sec": time.time() - t0,
        "optimizations": ["batch_size=4", "pre-encode pkl cache", "skip warm-up", "8 epochs"],
        "approach": "8 epochs, batch_size=4, copy_lambda=2.0, gate_init=-3 (sigmoid 0.05)",
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log(f"训练完成: 末 epoch lm={losses[-1]['lm']:.4f} ptr={losses[-1]['ptr']:.4f}")
    log("=" * 70)


if __name__ == "__main__":
    main()
