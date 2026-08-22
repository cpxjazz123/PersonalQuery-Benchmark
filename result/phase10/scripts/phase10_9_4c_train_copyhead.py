#!/usr/bin/env python3
"""Phase 10.9.4c: 快速训练 CopyAwareHead (用于 phase10_9_2 projector).

设计:
  - 冻结 phase10_9_2_projector.pt (前面 DPO 训练好的 prefix projector)
  - 训练新的 CopyAwareHead:
    * 数据: preference_pairs_60.jsonl 的 chosen_query + 3 attrs (Brand/Color/Material)
    * 损失: LM cross-entropy + copy gate BCE (attribute token 应 copy, 其他应 generate)
    * Epochs: 3, batch_size: 2, lr: 3e-4 (与 E28 copy_aware_train.py 默认一致)
  - 输出:
    * phase10_9_4c_copyhead.pt (CopyAwareHead state_dict)
    * phase10_9_4c_train_meta.json (loss 曲线 + 数据统计)
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

OUT_COPYHEAD = OUT_DIR / "phase10_9_4c_copyhead.pt"
OUT_META = OUT_DIR / "phase10_9_4c_train_meta.json"
LOG_OUT = OUT_DIR / "phase10_9_4c_train_copyhead.log"

# === 硬编码 ===
ATTR_FIELDS = ["Brand", "Color", "Material"]   # 2026-08-19: 移除数字属性
SEED = 42
EPOCHS = 3
BATCH_SIZE = 2
LR = 3e-4
COPY_LAMBDA = 0.5
MAX_QUERY_LEN = 80
NUM_TOKENS = 8

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
)


def log(m):
    print(f"[phase10-9.4c] {m}", flush=True)


def find_attr_value_spans(prompt_str: str, attrs_used: Dict[str, str]):
    """字符 span [start, end), longest first."""
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
    """在 query 文本中, 哪些 token span 落在 attribute value 上."""
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
    # 排序: 优先按 A1/A2/... 风格 (int(k[1:])), 否则按 key 字母序
    def _sort_key(k):
        if k.startswith("A") and k[1:].isdigit():
            return (0, int(k[1:]))
        return (1, k)
    for key in sorted(attrs_used, key=_sort_key):
        lines.append(f"{key}: {attrs_used[key]}")
    lines.append("Write a natural shopping query that mentions every attribute.")
    return "\n".join(lines)


def filter_attrs(attrs: Dict[str, str]) -> Dict[str, str]:
    """只保留 ATTR_FIELDS 中的字段."""
    return {k: v for k, v in attrs.items() if k in ATTR_FIELDS and v}


SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query "
    "that mentions every listed attribute of the product by its exact value "
    "(brand name, weight, dimensions, color, material). Keep the query under 25 words."
)


class CopyDataset(Dataset):
    def __init__(self, rows, z_map, tokenizer):
        self.rows = rows
        self.z_map = z_map
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        uid = row["user_id"]
        attrs_5 = row["attrs"]
        attrs = filter_attrs(attrs_5)
        prompt_str = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_attr_prompt_lines(attrs)},
            ],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        spans, _, _ = attr_token_spans(prompt_str, attrs, self.tokenizer)
        src_attr_mask = torch.zeros(len(prompt_ids), dtype=torch.bool)
        for spans_list in spans.values():
            for (a, b) in spans_list:
                src_attr_mask[a:b + 1] = True
        query = row["chosen_query"]
        target_ids = self.tokenizer.encode(query, add_special_tokens=False)[:MAX_QUERY_LEN]
        target_ids = target_ids + [self.tokenizer.eos_token_id]
        attr_pos = find_query_attr_pos(query, attrs, self.tokenizer)
        return {
            "user_id": uid,
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "src_attr_mask": src_attr_mask,
            "attr_pos": list(attr_pos),
            "user_vec": torch.tensor(self.z_map.get(uid, np.zeros(40, dtype=np.float32)), dtype=torch.float32),
        }


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
        # attr_pos 是相对 target_ids 的索引 (不是整个 sequence)
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
    log("Phase 10.9.4c: 快速训练 CopyAwareHead (3 attrs)")
    log("=" * 70)

    # === 1. Load Qwen ===
    log("[1] Loading Qwen2-7B (frozen, base 推理 only) ...")
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
    log(f"  projector loaded + frozen (epoch={ckpt.get('epoch', '?')})")

    # === 3. Init CopyAwareHead (trainable) ===
    log("[3] Init CopyAwareHead (trainable) ...")
    copy_head = CopyAwareHead(MODEL_DIM, VOCAB_SIZE, dtype=torch.bfloat16).to(DEVICE)
    # Gate 初始化: 默认 last layer = 0 → sigmoid(0) = 0.5
    # 为了让训练从低起点开始, 把 bias 设为 -2 → sigmoid(-2) ≈ 0.12 (保守起步)
    with torch.no_grad():
        copy_head.gate[-1].bias.fill_(-2.0)
    trainable = list(copy_head.parameters())
    log(f"  CopyAwareHead params: {sum(p.numel() for p in trainable)}")
    log(f"  gate init: bias=-2 (sigmoid ≈ 0.12)")

    # === 4. Load training data ===
    log("[4] Loading training data ...")
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

    # z_map
    z_map = {}
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            row = json.loads(line)
            z_map[row["user_id"]] = np.concatenate([row["user_mu"], row["user_logvar"]]).astype(np.float32)
    log(f"  z_map: {len(z_map)} users")

    # train/val split
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(rows))
    n_val = max(4, len(rows) // 10)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    log(f"  train: {len(train_rows)}, val: {len(val_rows)}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train_ds = CopyDataset(train_rows, z_map, tokenizer)
    val_ds = CopyDataset(val_rows, z_map, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=lambda b: collate(b, pad_id))
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: collate(b, pad_id))

    # === 5. Optimizer + training loop ===
    log("[5] Training copy head ...")
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    n_steps = len(train_dl) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=n_steps, pct_start=0.1)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    copy_head.train()

    losses = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        ep_lm = 0.0
        ep_ptr = 0.0
        for step, batch in enumerate(train_dl):
            prompt_ids = batch["prompt_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            src_attr_mask = batch["src_attr_mask"].to(DEVICE)
            user_vecs = batch["user_vecs"].to(DEVICE)
            attr_pos_list = batch["attr_pos"]

            # forward
            text_emb = model.get_input_embeddings()(prompt_ids).to(model.dtype)
            target_emb = model.get_input_embeddings()(target_ids.clamp_min(0)).to(model.dtype)
            prefix = projector(user_vecs.float()).to(model.dtype)  # [B, K, H] — projector expects float32, output bf16

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
            hidden = out.hidden_states[-1]  # [B, K+P+T, D]
            gen_logits = out.logits  # [B, K+P+T, V]

            # source hidden = prompt part
            src_len = prompt_ids.size(1)
            src_hidden = hidden[:, NUM_TOKENS:NUM_TOKENS + src_len, :]

            # target 部分 hidden (length T)
            tgt_len = target_ids.size(1)
            tgt_hidden = hidden[:, NUM_TOKENS + src_len:, :]  # [B, T, D]
            tgt_gen_logits = gen_logits[:, NUM_TOKENS + src_len:, :]

            # copy head on target
            p_copy, copy_logits = copy_head(tgt_hidden, src_hidden,
                                            src_attr_mask, prompt_ids)

            # 准备 copy source ids (prompt 部分)
            # copy_logits scatter by src_ids, 但 src_ids 只在 prompt 范围内
            # copy_aware CopyAwareHead 的 scatter 期望 src_ids shape [B, S] = prompt_ids
            # 已对齐

            # LM loss (cross_entropy over target, ignore -100)
            lm_loss = F.cross_entropy(
                tgt_gen_logits.reshape(-1, VOCAB_SIZE).float(),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

            # copy gate loss: target 位置在 attr_pos 列表里 → p_copy → 1, 其他 → 0
            T = target_ids.size(1)
            gate_target = torch.zeros(T - 1, dtype=torch.float32, device=DEVICE)
            for b in range(target_ids.size(0)):
                for pos in attr_pos_list[b]:
                    if pos < T - 1:
                        gate_target[pos] = 1.0
            # shift: 预测 next token (target[:, 1:])
            p_copy_shift = p_copy[:, :-1, 0]  # [B, T-1]
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

            if (step + 1) % 5 == 0:
                log(f"  epoch {epoch + 1}/{EPOCHS} step {step + 1}/{len(train_dl)} "
                    f"lm={float(lm_loss):.4f} ptr={float(ptr_loss):.4f}")

            del full_emb, out, hidden, gen_logits, src_hidden, tgt_hidden, tgt_gen_logits
            torch.cuda.empty_cache()

        log(f"  epoch {epoch + 1}/{EPOCHS} done: lm={ep_lm / max(len(train_dl), 1):.4f} "
            f"ptr={ep_ptr / max(len(train_dl), 1):.4f} ({time.time() - t0:.1f}s)")
        losses.append({"epoch": epoch + 1, "lm": ep_lm / max(len(train_dl), 1), "ptr": ep_ptr / max(len(train_dl), 1)})

    # === 6. Save copy_head ===
    log("[6] Saving copy head ...")
    torch.save(copy_head.state_dict(), OUT_COPYHEAD)
    log(f"  saved → {OUT_COPYHEAD}")

    meta = {
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "copy_lambda": COPY_LAMBDA,
        "final_losses": losses,
        "gate_init_bias": -2.0,
        "attr_fields": ATTR_FIELDS,
        "proj_ckpt": str(PROJ_CKPT),
        "elapsed_sec": time.time() - t0,
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log(f"训练完成: 末 epoch lm={losses[-1]['lm']:.4f} ptr={losses[-1]['ptr']:.4f}")
    log("=" * 70)


if __name__ == "__main__":
    main()