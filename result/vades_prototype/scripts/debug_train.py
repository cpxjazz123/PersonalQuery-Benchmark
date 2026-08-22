#!/usr/bin/env python3
"""Debug: 跑一个 batch, 把 mask 和 logp shape 全部打印, 看为什么 lpc=lpr=0."""
import sys
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
import json
import torch
import numpy as np
import torch.nn.functional as F
from phase10_train_projector import (
    encode_inputs, compute_logp, build_chat_prompts,
    PrefPairDataset, collate_pairs, NUM_TOKENS, USER_DIM,
)
from llm_client import _HiddenBackend
from query.soft_prefix.projector import SoftPrefixProjector

QWEN_MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

# 加载 Qwen
print("[load] Qwen ...")
backend = _HiddenBackend.get(QWEN_MODEL_PATH)
model = backend.model
tokenizer = backend.tokenizer
DEVICE = next(model.parameters()).device
print(f"[load] done, DEVICE={DEVICE}")

# 加载 projector
proj = SoftPrefixProjector(USER_DIM, 128, NUM_TOKENS, model.config.hidden_size,
                            dtype=torch.float32).to(DEVICE)
proj.train()

# 加载 data
records = []
with open("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/preference_pairs.jsonl") as f:
    for line in f:
        records.append(json.loads(line))
ds = PrefPairDataset(records[:2])  # 2 pairs = 4 forward
batch = collate_pairs([ds[0], ds[1]])
print(f"[batch] z_u shape={batch['z_u'].shape}, prompts len={len(batch['prompt_chat'])}, queries len={len(batch['query_text'])}")
print(f"[batch] is_chosen={batch['is_chosen']}")

SYSTEM = "You are a shopping query writer. Write one short natural shopping query that mentions every listed attribute of the product by its exact value (brand name, weight, dimensions, color, material). Keep the query under 25 words."
prompts = [build_chat_prompts(tokenizer, SYSTEM, p) for p in batch["prompt_chat"]]
input_ids, attention_mask, query_mask = encode_inputs(tokenizer, prompts, batch["query_text"], DEVICE)
print(f"[enc] input_ids shape={input_ids.shape}")
print(f"[enc] query_mask sum per row={query_mask.sum(dim=1).tolist()}  (应为每行 query token 数, 期望 ~10-20)")
print(f"[enc] query_mask[0] = {query_mask[0].int().tolist()[:50]}...")
# 看看实际 query token 在哪
import json
with open("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/preference_pairs.jsonl") as f:
    r = json.loads(f.readline())
q_ids = tokenizer.encode(r["chosen_query"], add_special_tokens=False)
print(f"[debug] chosen_query={r['chosen_query']!r}")
print(f"[debug] q_ids={q_ids}")
print(f"[debug] input_ids[0, 270:320]={input_ids[0, 270:320].tolist()}")
print(f"[debug] query_mask[0, 270:320]={query_mask[0, 270:320].int().tolist()}")
print(f"[debug] input_ids[0] nonzero count = {(input_ids[0] != 151643).sum().item()}  (151643 是 pad_token)")

z_u = batch["z_u"].to(DEVICE).to(torch.float32)
prefix = proj(z_u).to(model.dtype)
emb_m = model.get_input_embeddings()
text_emb = emb_m(input_ids).to(model.dtype)
full_emb = torch.cat([prefix, text_emb], dim=1)
full_am = torch.cat([torch.ones((full_emb.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE), attention_mask], dim=1)
full_qm = torch.cat([torch.zeros((full_emb.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE), query_mask], dim=1)

print(f"[full] full_emb shape={full_emb.shape}, full_qm shape={full_qm.shape}")

avg_lp = compute_logp(model, full_emb, full_am, full_qm, input_ids, num_prefix_tokens=NUM_TOKENS)
print(f"[logp] avg_lp = {avg_lp}")
print(f"[logp] avg_lp shape = {avg_lp.shape}")

# 手动重算, 验证
out = model(inputs_embeds=full_emb, attention_mask=full_am)
K = NUM_TOKENS
logits = out.logits[:, K-1:K-1+input_ids.size(1), :]
print(f"[manual] logits shape={logits.shape}")
targets = input_ids
print(f"[manual] targets[0,:20]={targets[0,:20].tolist()}")
print(f"[manual] targets[0,20:50]={targets[0,20:50].tolist()}")
log_probs = F.log_softmax(logits.float(), dim=-1)
chosen_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
print(f"[manual] chosen_lp[0,:20]={chosen_lp[0,:20].tolist()}")
print(f"[manual] chosen_lp[0,:20] mean={chosen_lp[0,:20].mean().item():.4f}")
text_qm = query_mask  # already L
print(f"[manual] text_qm sum per row={text_qm.sum(dim=1).tolist()}")
