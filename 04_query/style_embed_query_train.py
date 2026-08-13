#!/usr/bin/env python3
"""Style-Embedding Injection Query Generator — Training.

Inject user style vector into Qwen2-7B hidden states to train the model to
generate queries aligned with each user's writing style.

Architecture:
- User style vector: mean of user's review sentence embeddings (E5-large, 1024-dim)
- Projection: Linear(1024 → 3584) maps style vector to LLM hidden dim
- Injection point: concat projected vector as soft prefix to input embeddings
- LoRA: train q_proj, v_proj on attention layers

Training:
- Data: (user_style_vec, asin_desc, gold_query) tuples from 04_query output
- Loss: causal LM CE on gold_query tokens
- Optimizer: AdamW lr=1e-4 with cosine schedule

Outputs:
- lora_weights/: PEFT adapter weights
- style_projection.pt: Linear projection layer weights
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

QWEN_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
HIDDEN_DIM = 3584  # Qwen2-7B hidden dim
USER_DIM = 1024  # E5-large embedding dim


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_04_query(category: str, base: str) -> List[Dict]:
    """Load (user_id, asin, query_text) tuples from 04_query output."""
    p = Path(base) / "result/personal_query/04_query" / category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(p) as f:
        records = json.load(f)
    out = []
    for r in records:
        # Each record has syntax_depth_queries (list of 10); use the first one
        queries = r.get("syntax_depth_queries", [])
        if not queries:
            continue
        query_text = queries[0].get("query", "").strip()
        if not query_text:
            continue
        out.append({
            "user_id": r["user_id"],
            "asin": r["asin"],
            "query_text": query_text,
        })
    return out


def compute_user_style_vecs(
    user_ids: List[str], category: str, base: str
) -> Dict[str, np.ndarray]:
    """For each user, compute mean of their review sentence embeddings via E5.

    Returns dict user_id -> 1024-dim numpy array (L2-normalized).
    """
    log(f"loading reviews for {len(user_ids)} users in {category}...")
    reviews_p = Path(base) / "result/personal_query/01_preference_extraction" / category / "stage1_filtered_users_reviews.json"
    with open(reviews_p) as f:
        data = json.load(f)
    user_reviews: Dict[str, List[str]] = {}
    for rec in data["users"]:
        uid = rec.get("user_id")
        if uid in user_ids:
            for r in rec.get("results", []):
                for txt in r.get("target_reviews", []):
                    if txt and isinstance(txt, str) and txt.strip():
                        user_reviews.setdefault(uid, []).append(txt[:1000])

    target_users = set(user_ids)
    found_users = {uid for uid in user_reviews if uid in target_users}
    log(f"  found reviews for {len(found_users)}/{len(target_users)} users")

    # Encode via E5
    log("loading E5 model...")
    os.environ.setdefault("HF_HOME", "/home/wlia0047/.cache/huggingface/hub")
    from sentence_transformers import SentenceTransformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e5 = SentenceTransformer("intfloat/e5-large-v2", device=str(device))

    user_vecs: Dict[str, np.ndarray] = {}
    for i, uid in enumerate(user_ids):
        reviews = user_reviews.get(uid, [])
        if not reviews:
            # Fallback: random init
            user_vecs[uid] = np.random.randn(USER_DIM).astype(np.float32)
            continue
        embs = e5.encode(reviews[:32], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        user_vecs[uid] = embs.mean(axis=0).astype(np.float32)
        if (i + 1) % 100 == 0:
            log(f"  encoded {i + 1}/{len(user_ids)} users")
    log(f"  computed {len(user_vecs)} user style vectors")
    return user_vecs


class StyleDataset(Dataset):
    """Per sample: user_style_vec, asin prompt, gold query text."""

    def __init__(
        self,
        records: List[Dict],
        user_vecs: Dict[str, np.ndarray],
        tokenizer,
        max_query_len: int = 64,
    ):
        self.records = records
        self.user_vecs = user_vecs
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        vec = self.user_vecs.get(r["user_id"], np.zeros(USER_DIM, dtype=np.float32))
        # Build asin prompt (short)
        prompt = f"Generate a shopping query for ASIN {r['asin']}."
        # Encode prompt + query
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        query_ids = self.tokenizer.encode(r["query_text"], add_special_tokens=False)[: self.max_query_len]
        # Causal LM: input = prompt + query, label = -100 (prompt) + query
        input_ids = prompt_ids + query_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + query_ids + [self.tokenizer.eos_token_id]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "user_vec": vec,
        }


def collate(batch, pad_token_id: int):
    max_len = max(len(s["input_ids"]) for s in batch)
    input_ids = []
    labels = []
    user_vecs = []
    for s in batch:
        ids = s["input_ids"]
        lbls = s["labels"]
        pad_n = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_n)
        labels.append(lbls + [-100] * pad_n)
        user_vecs.append(s["user_vec"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "user_vecs": torch.tensor(np.stack(user_vecs), dtype=torch.float32),
    }


class StyleInjectedModel(nn.Module):
    """Qwen2-7B + LoRA + style vector injection at embedding layer."""

    def __init__(self, base_model, user_dim: int = USER_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.base = base_model
        # Style projection: user_dim -> hidden_dim
        self.style_proj = nn.Linear(user_dim, hidden_dim, bias=False)
        nn.init.normal_(self.style_proj.weight, std=0.02)
        # Match base model dtype
        target_dtype = next(self.base.parameters()).dtype
        self.style_proj = self.style_proj.to(target_dtype)
        # Freeze base; PEFT handles LoRA on attention
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, user_vecs: torch.Tensor, labels: torch.Tensor = None):
        # Embed input tokens via PEFT-compatible interface
        embed_fn = self.base.get_input_embeddings()
        inputs_embeds = embed_fn(input_ids)
        # Cast to bf16 to match base model dtype
        target_dtype = next(self.base.parameters()).dtype
        inputs_embeds = inputs_embeds.to(target_dtype)
        # Project user vectors and prepend as soft prefix (in matching dtype)
        style_prefix = self.style_proj(user_vecs.to(target_dtype)).unsqueeze(1)
        inputs_embeds = torch.cat([style_prefix, inputs_embeds], dim=1)
        # Adjust labels: prepend -100 for the style prefix
        if labels is not None:
            prefix_labels = torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([prefix_labels, labels], dim=1)
        # Adjust attention_mask (all 1s for prefix)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        out = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out


def train_category(
    category: str,
    base: str,
    out_dir: Path,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-4,
    lora_r: int = 8,
    max_records: int = 0,
):
    log(f"\n=== {category} ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    log("loading 04_query data...")
    records = load_04_query(category, base)
    if max_records and max_records < len(records):
        records = records[:max_records]
    log(f"  {len(records)} training records")

    log("computing user style vectors...")
    user_vecs = compute_user_style_vecs([r["user_id"] for r in records], category, base)

    log("loading Qwen2-7B + LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=2 * lora_r,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    base_model = get_peft_model(base_model, lora_cfg)
    base_model.print_trainable_parameters()
    model = StyleInjectedModel(base_model).to("cuda:0")

    log("building dataset/dataloader...")
    ds = StyleDataset(records, user_vecs, tokenizer)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        num_workers=0,
    )

    # Optimizer over LoRA + style projection
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    total_steps = len(dl) * epochs
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=total_steps)
    log(f"  total_steps={total_steps}")

    log("training...")
    model.train()
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batch = 0
        for batch in dl:
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                user_vecs=batch["user_vecs"],
                labels=batch["labels"],
            )
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            opt.step()
            sched.step()
            epoch_loss += float(loss.detach())
            n_batch += 1
            step += 1
            if step % 20 == 0:
                log(f"  step {step}/{total_steps} loss={float(loss.detach()):.4f}")
        avg = epoch_loss / max(n_batch, 1)
        log(f"  epoch {epoch + 1}/{epochs} avg_loss={avg:.4f}")

    log("saving LoRA adapter + style projection...")
    base_model.save_pretrained(out_dir / "lora_adapter")
    tokenizer.save_pretrained(out_dir / "lora_adapter")
    torch.save(model.style_proj.state_dict(), out_dir / "style_projection.pt")
    # Save user_vecs for inference
    np.savez(out_dir / "user_vecs.npz", **user_vecs)
    log(f"saved to {out_dir}")
    log("=== done ===")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"])
    ap.add_argument("--base", default="/home/wlia0047/ar57/wenyu/PersoanlQuery")
    ap.add_argument("--out_dir", default="/home/wlia0047/hj82_scratch2/wenyu/RAG/style_embed_models")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--max_records", type=int, default=0, help="0=all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.category
    train_category(
        category=args.category,
        base=args.base,
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()