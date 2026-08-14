#!/usr/bin/env python3
"""E14-P — Copy-aware free-form query generation: training.

The model generates NATURAL queries (no placeholder template) while price,
brand and other immutable attribute strings are copied verbatim from the
product-attribute input through a pointer/copy distribution (P-series).

Loss:
  L = CE(mixed distribution over target) + lambda * pointer CE
      (target positions belonging to an attribute value must attend to that
      attribute's input span)

User style condition: VADES user_mu -> soft prefix (statistics only, no
review sentences), entering ONLY the generation distribution — the copy path
depends exclusively on the product attribute spans (P7).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query"))
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))

from user_style_vectors import (  # noqa: E402
    UserVectorProvider,
    load_vades_profiles,
)
from projector import SoftPrefixProjector  # noqa: E402
from copy_aware import (  # noqa: E402
    CopyAwareHead,
    build_attr_prompt_lines,
    attr_token_spans,
    mixed_logits,
)
from opener_stats import (  # noqa: E402
    build_opener_stats_for_users,
    get_opener_stats,
)

QWEN_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
HIDDEN_DIM = 3584  # overridden by the loaded model's config.hidden_size
SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query "
    "that mentions every listed attribute of the product."
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_messages(attrs_used: Dict[str, str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_attr_prompt_lines(attrs_used)},
    ]


def target_attr_token_positions(target_query: str, attrs_used: Dict[str, str], tokenizer) -> List[int]:
    """Token positions in the ENCODED target query that belong to attribute
    values (exact + variant-tolerant matching)."""
    enc = tokenizer(target_query, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    positions = [False] * n
    sys.path.insert(0, str(REPO_ROOT / "04_query"))
    from common.attribute_helpers import _find_variant_token_spans
    for key, value in attrs_used.items():
        if not value:
            continue
        spans = _find_variant_token_spans(target_query, str(value))
        if not spans:
            import re
            spans = [m.span() for m in re.finditer(re.escape(str(value)), target_query)]
        for (cs, ce) in spans:
            for i, (os_, oe) in enumerate(offsets):
                if os_ < ce and oe > cs:
                    positions[i] = True
    return [i for i, p in enumerate(positions) if p]


def copy_pointer_targets(target_ids: List[int], attr_positions: List[int], n_src: int) -> torch.Tensor:
    """Pointer target: for target positions belonging to an attribute value,
    the pointer should attend to the whole source (relaxed: uniform over
    source). We use a uniform pointer target over the source so the model
    learns "copy from the attribute input" without needing exact source
    spans during training; exact span supervision (P5) is a stricter follow-up.
    """
    targets = -100 * torch.ones(len(target_ids), dtype=torch.long)
    targets[attr_positions] = 0  # class 0 = "copy from source" (uniform)
    return targets


class CopyDataset(Dataset):
    def __init__(self, rows, provider, tokenizer, max_query_len=80, opener_cache=None, category=None):
        self.rows = rows
        self.provider = provider
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.opener_cache = opener_cache
        self.category = category

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        attrs = row["attrs_used"]
        vec, has_vector = self.provider.get(row["user_id"])
        if self.opener_cache is not None and vec is not None:
            mu = np.asarray(vec, dtype=np.float32)
            n = float(np.linalg.norm(mu))
            if n > 1e-12:
                mu = mu / n
            vec = np.concatenate([mu, get_opener_stats(self.category, row["user_id"], self.opener_cache)]).astype(np.float32)
        prompt_str = self.tokenizer.apply_chat_template(
            build_messages(attrs), tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        # prompt-side attribute token spans (for the copy source)
        p_spans = attr_token_spans(prompt_str, attrs, self.tokenizer)
        src_attr_mask = torch.zeros(len(prompt_ids), dtype=torch.bool)
        for spans in p_spans.values():
            for (a, b) in spans:
                src_attr_mask[a:b + 1] = True
        query = row["y_plus_query"]
        target_ids = self.tokenizer.encode(query, add_special_tokens=False)[: self.max_query_len]
        target_ids = target_ids + [self.tokenizer.eos_token_id]
        attr_pos = target_attr_token_positions(query, attrs, self.tokenizer)
        return {
            "user_id": row["user_id"],
            "prompt_ids": prompt_ids,
            "target_ids": target_ids,
            "src_attr_mask": src_attr_mask,
            "attr_pos": attr_pos,
            "user_vec": np.zeros(self.provider.vector_dim, dtype=np.float32) if vec is None else vec,
            "has_vector": has_vector,
        }


def collate(batch, pad_token_id):
    max_p = max(len(s["prompt_ids"]) for s in batch)
    max_t = max(len(s["target_ids"]) for s in batch)
    prompts, targets, masks, src_masks, vecs = [], [], [], [], []
    for s in batch:
        pn = max_p - len(s["prompt_ids"])
        tn = max_t - len(s["target_ids"])
        prompts.append(s["prompt_ids"] + [pad_token_id] * pn)
        targets.append(s["target_ids"] + [-100] * tn)
        masks.append([1] * len(s["prompt_ids"]) + [0] * pn + [1] * len(s["target_ids"]) + [0] * tn)
        sm = torch.zeros(max_p, dtype=torch.bool)
        sm[: len(s["src_attr_mask"])] = s["src_attr_mask"]
        src_masks.append(sm)
        vecs.append(s["user_vec"])
    return {
        "prompt_ids": torch.tensor(prompts, dtype=torch.long),
        "target_ids": torch.tensor(targets, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "src_attr_mask": torch.stack(src_masks),
        "user_vecs": torch.tensor(np.stack(vecs), dtype=torch.float32),
        "attr_pos": [s["attr_pos"] for s in batch],
    }


class CopyAwareModel(nn.Module):
    def __init__(self, base_model, projector, copy_head, device: str, num_tokens: int = 4):
        super().__init__()
        self.base = base_model
        self.projector = projector
        self.copy_head = copy_head
        self.device = device
        self.num_tokens = num_tokens
        self.base.train()

    def forward(self, batch):
        prompt_ids = batch["prompt_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        src_attr_mask = batch["src_attr_mask"].to(self.device)
        user_vecs = batch["user_vecs"].to(self.device)

        text_emb = self.base.get_input_embeddings()(prompt_ids)
        target_dtype = next(self.base.parameters()).dtype
        text_emb = text_emb.to(target_dtype)
        target_emb = self.base.get_input_embeddings()(target_ids.clamp_min(0)).to(target_dtype)
        if self.num_tokens > 0:
            prefix = self.projector(user_vecs.to(target_dtype))
            full_emb = torch.cat([prefix, text_emb, target_emb], dim=1)
            am = torch.cat([
                torch.ones((prompt_ids.size(0), self.num_tokens), dtype=attention_mask.dtype, device=self.device),
                attention_mask,
            ], dim=1)
            src_mask_full = torch.cat([
                torch.zeros((prompt_ids.size(0), self.num_tokens), dtype=torch.bool, device=self.device),
                src_attr_mask,
            ], dim=1)
        else:
            full_emb = torch.cat([text_emb, target_emb], dim=1)
            am = attention_mask
            src_mask_full = src_attr_mask

        # LM over prompt+target, hidden states for the whole sequence
        out = self.base(
            inputs_embeds=full_emb,
            attention_mask=am,
            labels=None,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states[-1]  # [B, K+P+T, D]
        gen_logits = out.logits          # [B, K+P+T, V]

        # source hidden = prompt part (attribute tokens)
        src_len = prompt_ids.size(1)
        src_hidden = hidden[:, self.num_tokens: self.num_tokens + src_len, :]
        src_mask = src_mask_full[:, self.num_tokens: self.num_tokens + src_len] if self.num_tokens > 0 else src_attr_mask

        # generation positions: all positions of the target part (shift by one)
        T = target_ids.size(1)
        gen_hidden = hidden[:, self.num_tokens + src_len: self.num_tokens + src_len + T, :]
        p_copy, copy_logits = self.copy_head(gen_hidden, src_hidden, src_mask, prompt_ids)

        # mixed logits on target positions
        mixed = mixed_logits(gen_logits[:, self.num_tokens + src_len: self.num_tokens + src_len + T, :], p_copy, copy_logits)

        # LM CE on the mixed distribution (only target tokens)
        shift_mixed = mixed[:, :-1, :].reshape(-1, mixed.size(-1))
        shift_labels = target_ids[:, 1:].reshape(-1)
        lm_loss = F_cross_entropy(shift_mixed, shift_labels, ignore_index=-100)

        # copy gate loss: BCE supervising p_copy on target positions that
        # belong to an attribute value (should copy) vs others (should generate).
        ptr_targets = batch["attr_pos"]
        gate_target = torch.zeros(T, dtype=torch.float32, device=self.device)
        counts = torch.zeros(T, dtype=torch.float32, device=self.device)
        for b in range(target_ids.size(0)):
            for pos in ptr_targets[b]:
                if pos < T:
                    gate_target[pos] += 1.0
                    counts[pos] += 1.0
        gate_target = (gate_target > 0).float()
        gate_target = gate_target[1:]  # shift: predict next token
        valid = torch.ones_like(gate_target)
        p_copy_shift = p_copy[:, :-1, 0]  # [B, T-1]
        losses = []
        for b in range(target_ids.size(0)):
            tgt_b = target_ids[b, 1:]
            mask = tgt_b != -100
            if mask.sum() == 0:
                continue
            l = torch.nn.functional.binary_cross_entropy_with_logits(
                p_copy_shift[b][mask], gate_target[mask])
            losses.append(l)
        ptr_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)
        return lm_loss, ptr_loss


def F_cross_entropy(logits, targets, ignore_index=-100):
    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Baby_Products")
    ap.add_argument("--base_model", default=QWEN_PATH)
    ap.add_argument("--dataset_path", default="", help="multi-product dataset json (user_id, asin, attrs)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_tokens", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--copy_lambda", type=float, default=0.5)
    ap.add_argument("--gate_init", type=float, default=1e-3, help="soft-prefix gate alpha init (E14: style strength)")
    ap.add_argument("--max_records", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset_path:
        raise ValueError("multi-product dataset has no natural query targets; train on the 04_query candidates (copy-aware) and use the dataset only for evaluation generation")
        records_p = REPO_ROOT / "result" / "personal_query" / "04_query" / args.category / "query_by_syntax_depth_no_depth_check_10.json"
        with open(records_p) as f:
            records = json.load(f)
        rows = []
        for rec in records:
            attrs = (rec.get("syntax_depth_query") or {}).get("attrs_used")
            candidates = rec.get("syntax_depth_queries", [])
            if not attrs:
                for cand in candidates:
                    if cand.get("attrs_used"):
                        attrs = cand["attrs_used"]
                        break
            if not attrs:
                continue
            for cand in candidates:
                q = cand.get("query", "")
                if q:
                    rows.append({"user_id": rec["user_id"], "asin": rec["asin"], "attrs_used": attrs, "y_plus_query": q})
    if args.max_records:
        rows = rows[: args.max_records]
    log(f"copy-aware rows: {len(rows)}")

    profiles = load_vades_profiles(args.category)
    split_ids = sorted({r["user_id"] for r in rows})
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(split_ids))
    n_test = max(1, int(round(len(split_ids) * args.test_frac)))
    test_users = {split_ids[int(i)] for i in idx[:n_test]}
    train_rows = [r for r in rows if r["user_id"] not in test_users]
    test_rows = [r for r in rows if r["user_id"] in test_users]
    log(f"train rows={len(train_rows)} test rows={len(test_rows)} users={len(split_ids)}")

    provider = UserVectorProvider("vades", profiles, [r["user_id"] for r in rows], seed=args.seed)
    opener_cache = build_opener_stats_for_users(args.category, [r["user_id"] for r in rows])
    USER_VEC_DIM = provider.vector_dim + len(get_opener_stats(args.category, next(iter(opener_cache)), opener_cache)) if opener_cache else provider.vector_dim
    log(f"user condition dim = {USER_VEC_DIM} (VADES {provider.vector_dim} + opener stats)")

    log("loading Qwen2-7B ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    HIDDEN_DIM = base.config.hidden_size
    log(f"base model: {args.base_model} hidden_dim={HIDDEN_DIM} vocab={base.config.vocab_size}")
    if args.lora:
        from peft import LoraConfig, get_peft_model
        base = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=2 * args.lora_r,
                                               target_modules=["q_proj", "v_proj"], lora_dropout=0.05))
        log(f"LoRA enabled: {sum(p.numel() for p in base.parameters() if p.requires_grad)} trainable")
    else:
        log("LoRA disabled (copy-aware baseline: projector only)")

    projector = SoftPrefixProjector(user_dim=USER_VEC_DIM, hidden_dim=128, num_tokens=args.num_tokens,
                                    model_dim=HIDDEN_DIM, dtype=torch.bfloat16,
                                    gate_init=args.gate_init).to("cuda:0")
    log(f"gate_init={args.gate_init}")
    vocab_size = base.get_output_embeddings().weight.size(0)
    copy_head = CopyAwareHead(HIDDEN_DIM, vocab_size, dtype=torch.bfloat16).to("cuda:0")
    log(f"copy head vocab_size={vocab_size}")
    model = CopyAwareModel(base, projector, copy_head, device="cuda:0", num_tokens=args.num_tokens).to("cuda:0")
    if hasattr(base, "active_peft_config") or "peft" in type(base).__module__:
        for name, p in base.named_parameters():
            if "lora_" not in name and p.requires_grad:
                p.requires_grad = False
    assert any(p.requires_grad for p in model.projector.parameters()), "projector must be trainable"
    assert any(p.requires_grad for p in model.copy_head.parameters()), "copy head must be trainable"
    log("assert OK: projector + copy head trainable")

    ds = CopyDataset(train_rows, provider, tokenizer, opener_cache=opener_cache, category=args.category)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    total = len(dl) * args.epochs
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=min(20, total // 10), num_training_steps=total)
    log(f"trainable params={sum(p.numel() for p in trainable)} total_steps={total}")

    model.train()
    step = 0
    for epoch in range(args.epochs):
        ep_lm = 0.0
        ep_ptr = 0.0
        for batch in dl:
            lm_loss, ptr_loss = model(batch)
            loss = lm_loss + args.copy_lambda * ptr_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            opt.step()
            sched.step()
            ep_lm += float(lm_loss.detach())
            ep_ptr += float(ptr_loss.detach())
            step += 1
            if step % 20 == 0:
                log(f"  step {step}/{total} lm={float(lm_loss.detach()):.4f} ptr={float(ptr_loss.detach()):.4f}")
        log(f"  epoch {epoch + 1}/{args.epochs} lm={ep_lm / max(len(dl), 1):.4f} ptr={ep_ptr / max(len(dl), 1):.4f}")

    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), ckpt / "projector.pt")
    torch.save(model.copy_head.state_dict(), ckpt / "copy_head.pt")
    tokenizer.save_pretrained(ckpt / "tokenizer")
    if args.lora:
        base.save_pretrained(ckpt / "lora_adapter")
    with open(ckpt / "config.json", "w") as f:
        json.dump({
            "category": args.category, "num_tokens": args.num_tokens, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
            "test_frac": args.test_frac, "lora": args.lora, "lora_r": args.lora_r,
            "copy_lambda": args.copy_lambda, "gate_init": args.gate_init, "base_model_path": args.base_model,
            "user_dim": USER_VEC_DIM, "model_dim": HIDDEN_DIM,
            "n_train_rows": len(train_rows), "n_test_rows": len(test_rows),
            "vector_encoder": "VADES-lite diagonal 20d user_mu (statistics only)",
        }, f, indent=2)
    with open(ckpt / "split_manifest.json", "w") as f:
        json.dump({"test_users": sorted(test_users), "train_users": sorted(set(split_ids) - test_users)}, f, indent=2)
    log(f"saved {ckpt}")
    log("=== done ===")


if __name__ == "__main__":
    main()
