#!/usr/bin/env python3
"""Leakage-upper-bound diagnostic for style-conditioned generation.

Condition vector = the TARGET query's own 20-dim syntactic features (leakage).
If even this oracle condition cannot make the generated syntax match the
condition, the architecture (soft-prefix -> syntax) is the bottleneck; if it
can, the problem is user-vector information, not transfer.

Quick protocol: train copy-aware (free generation + copy head) with condition
= target sentence features; generate; measure condition->output feature
correlation / match. Hardcoded config.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))

from copy_aware import CopyAwareHead, mixed_logits  # noqa: E402
from copy_aware_train import build_messages  # noqa: E402
from user_stat_vector import FEATURES20  # noqa: E402
from extract_clause_features_single_query import load_spacy_model, extract_clause_features  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
RECORDS = REPO_ROOT / "result" / "personal_query" / "04_query" / "Baby_Products" / "query_by_syntax_depth_no_depth_check_10.json"
OUT = REPO_ROOT / "result" / "personal_query" / "e17_leakage_diag"
EPOCHS = 10
BATCH = 8
SEED = 42
STYLE_DIM = 20


class LeakDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = self.tokenizer.apply_chat_template(build_messages(r["attrs"]),
                                                    tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(r["y"], add_special_tokens=False) + [self.tokenizer.eos_token_id]
        return {
            "prompt_ids": prompt_ids, "target_ids": target_ids,
            "cond": r["cond"], "attrs": r["attrs"],
        }


def collate(batch, pad):
    max_p = max(len(b["prompt_ids"]) for b in batch)
    max_t = max(len(b["target_ids"]) for b in batch)
    prompts, targets, masks, conds = [], [], [], []
    for b in batch:
        pn = max_p - len(b["prompt_ids"])
        tn = max_t - len(b["target_ids"])
        prompts.append(b["prompt_ids"] + [pad] * pn)
        targets.append(b["target_ids"] + [-100] * tn)
        masks.append([1] * len(b["prompt_ids"]) + [0] * pn + [1] * len(b["target_ids"]) + [0] * tn)
        conds.append(b["cond"])
    return {"prompt_ids": torch.tensor(prompts), "target_ids": torch.tensor(targets),
            "attention_mask": torch.tensor(masks), "conds": torch.tensor(conds, dtype=torch.float32)}


def main() -> None:
    torch.manual_seed(SEED)
    nlp = load_spacy_model()
    recs = json.load(open(RECORDS))
    rows = []
    for rec in recs:
        attrs = (rec.get("syntax_depth_query") or {}).get("attrs_used")
        if not attrs:
            continue
        for cand in rec.get("syntax_depth_queries", []):
            q = cand.get("query", "")
            if not q:
                continue
            f = extract_clause_features(q)
            cond = [float(f.get(k, 0.0)) for k in FEATURES20]
            rows.append({"attrs": attrs, "y": q, "cond": cond})
    print(f"rows: {len(rows)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                device_map="cuda:0", trust_remote_code=True)
    for p in base.parameters():
        p.requires_grad = False
    from peft import LoraConfig, get_peft_model
    base = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16,
                                           target_modules=["q_proj", "v_proj"]))
    H = base.config.hidden_size
    proj = nn.Linear(STYLE_DIM, H).to("cuda:0").to(torch.bfloat16)
    copy_head = CopyAwareHead(H, base.config.vocab_size, dtype=torch.bfloat16).to("cuda:0")

    ds = LeakDataset(rows, tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    trainable = [p for p in list(proj.parameters()) + list(copy_head.parameters())
                 if p.requires_grad] + [p for p in base.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4)

    base.train()
    for epoch in range(EPOCHS):
        tot = 0.0
        for b in dl:
            pids = b["prompt_ids"].to("cuda:0")
            tids = b["target_ids"].to("cuda:0")
            am = b["attention_mask"].to("cuda:0")
            conds = b["conds"].to("cuda:0")
            cond_vec = proj(conds.to(torch.bfloat16))  # [B, H] per-token bias
            emb_m = base.get_input_embeddings()
            text_emb = emb_m(pids) + cond_vec.unsqueeze(1)
            target_emb = emb_m(tids.clamp_min(0))
            full = torch.cat([text_emb, target_emb], dim=1)
            am2 = am
            out = base(inputs_embeds=full, attention_mask=am2, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1]
            src_len = pids.size(1)
            src_hidden = hidden[:, :src_len, :]
            T = tids.size(1)
            gen_hidden = hidden[:, src_len: src_len + T, :]
            src_mask = torch.zeros(src_hidden.shape[:2], dtype=torch.bool, device="cuda:0")
            p_copy, copy_logits = copy_head(gen_hidden, src_hidden, src_mask, pids)
            gen_logits = out.logits[:, src_len: src_len + T, :]
            mixed = mixed_logits(gen_logits, p_copy, copy_logits)
            loss = torch.nn.functional.cross_entropy(
                mixed[:, :-1, :].reshape(-1, mixed.size(-1)), tids[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        print(f"epoch {epoch + 1}: lm={tot / max(len(dl), 1):.4f}", flush=True)

    # eval: condition->output feature match (correlation between condition
    # dims and generated feature dims across rows)
    base.eval()
    pred_feats, cond_feats = [], []
    n_empty = 0
    with torch.no_grad():
        for r in rows[:30]:
            prompt = tok.apply_chat_template(build_messages(r["attrs"]), tokenize=False, add_generation_prompt=True)
            ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            cond_vec = proj(torch.tensor([r["cond"]], dtype=torch.bfloat16, device="cuda:0"))
            emb = base.get_input_embeddings()(ids) + cond_vec.unsqueeze(1)
            full = emb
            am = torch.ones(full.shape[:2], dtype=torch.long, device="cuda:0")
            # manual stepwise decoding (generate+inputs_embeds desyncs cache)
            with torch.no_grad():
                out = base(inputs_embeds=full, attention_mask=am, use_cache=True, past_key_values=None)
            past = out.past_key_values
            next_id = torch.argmax(out.logits[0, -1])
            g = []
            am2 = torch.cat([am, torch.ones((1, 1), dtype=torch.long, device="cuda:0")], dim=1)
            for _ in range(32):
                if next_id.item() == tok.eos_token_id:
                    break
                with torch.no_grad():
                    out = base(input_ids=next_id.unsqueeze(0).unsqueeze(0), attention_mask=am2,
                               past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = torch.argmax(out.logits[0, -1])
                g.append(int(next_id))
                am2 = torch.cat([am2, torch.ones((1, 1), dtype=torch.long, device="cuda:0")], dim=1)
            q = tok.decode(g, skip_special_tokens=True)
            if not q.strip():
                n_empty += 1
                continue
            f = extract_clause_features(q)
            pred_feats.append([float(f.get(k, 0.0)) for k in FEATURES20])
            cond_feats.append(r["cond"])
    print(f"eval: {len(pred_feats)} non-empty / 30 (empty={n_empty})", flush=True)
    if not pred_feats:
        print("ALL EMPTY — generation failed", flush=True)
        return
    P = np.array(pred_feats)
    C = np.array(cond_feats)
    # per-dim correlation between condition and output
    corr = np.array([np.corrcoef(C[:, i], P[:, i])[0, 1] if np.std(C[:, i]) > 0 else 0.0
                     for i in range(STYLE_DIM)])
    mean_corr = float(np.nanmean(np.abs(corr)))
    # syntax-distance: ||scaled_pred - scaled_cond|| relative to ||cond||
    rel_err = float(np.linalg.norm(P - C) / max(np.linalg.norm(C), 1e-9))
    print(f"LEAKAGE DIAG: mean|corr|={mean_corr:.3f} rel_err={rel_err:.3f}")
    print("top-correlated dims:", {FEATURES20[i]: round(float(corr[i]), 3) for i in np.argsort(-np.abs(corr))[:5]})
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"mean_abs_corr": mean_corr, "rel_err": rel_err,
               "per_dim_corr": {FEATURES20[i]: float(corr[i]) for i in range(STYLE_DIM)}},
              open(OUT / "leakage_diag.json", "w"), indent=2)


if __name__ == "__main__":
    main()
