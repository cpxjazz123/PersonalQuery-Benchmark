#!/usr/bin/env python3
"""Phase 10.4: 训练 20d Raw-VADES → soft-prefix projector (DPO 风格).

设计:
  - 输入: preference_pairs.jsonl (Phase 10.3 输出)
  - 每个 pair: (user_mu[20d] + log_sigma[20d]) -> 40d z_u
  - SoftPrefixProjector(z_u) -> [B, K, H] prefix
  - Qwen2-7B forward with prefix (inputs_embeds 注入, 冻结)
  - Loss:
      L_pref = -logsigmoid(beta * (log_p_chosen - log_p_rejected))
      L_lm   = (NLL_chosen + NLL_rejected) / 2  (辅助, 防止语言崩溃)
      L      = L_pref + lambda_lm * L_lm
  - 验证: 同样接口 forward val chosen/rejected, 输出 validation loss

走 llm_client.py::_HiddenBackend (transformers 路径, frozen Qwen, 显存独立
于 vLLM backend; 见 AGENTS.md Rule 8: 业务侧不允许直接 import transformers,
因此本脚本只能通过 QwenLocalClient.get_hidden_states / 直接读取
_HiddenBackend.model 来访问 — 但 _HiddenBackend 是 llm_client.py 内部类,
所以走 llm_client.py 暴露的入口。

实施注意:
  - 只训练 projector (~3.7M params). Qwen2-7B (7B params) 全 frozen.
  - inputs_embeds 注入 prefix 时 prefix 是 leaf with grad, 梯度只回传到 projector.
  - 损失只用 chosen/rejected query 文本的 average token log-prob (single-pass
    per query, 不需要 SFT 阶段).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

# === 硬编码配置 ===
USER_DIM = 40           # 20d mu + 20d log_sigma
HIDDEN_DIM = 128
NUM_TOKENS = 8
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
BETA = 0.5              # DPO temperature (Phase 10.6.3 提升: 0.1 → 0.5, 加速对比)
LAMBDA_LM = 0.005       # 辅助 LM loss 权重 (Phase 10.6.3 降低: 0.05 → 0.005, 避免压制 DPO)
LR = 1e-3
EPOCHS = 80             # Phase 10.6.3: 5 → 80 (audit 验证 80 epochs 收敛到 acc 0.94)
BATCH_SIZE = 2          # DPO: 每 batch 2 条 pair (chosen + rejected = 4 forward per pair)
GRAD_ACCUM = 4          # 梯度累积 → effective batch = 8 pairs
MAX_PROMPT_LEN = 480
MAX_QUERY_LEN = 64
SEED = 42
MIN_MARGIN = 0.0        # 剔除 margin < 0 的负样本 (Phase 10.6.3 审计: 1 个负 margin pair)

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PREF_PAIRS_FILE = OUT_DIR / "preference_pairs.jsonl"
CKPT_FILE = OUT_DIR / "phase10_6_3_projector.pt"
META_FILE = OUT_DIR / "phase10_6_3_projector_meta.json"

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
)


class PrefPairDataset(Dataset):
    def __init__(self, records: List[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        z_u = torch.tensor(r["mu_u"] + r["log_sigma_u"], dtype=torch.float32)
        return {
            "z_u": z_u,
            "prompt_no_style": r["prompt_no_style"],
            "prompt_target_style": r["prompt_target_style"],
            "chosen_query": r["chosen_query"],
            "rejected_query": r["rejected_query"],
        }


def collate_pairs(batch: List[dict]) -> dict:
    """每个 pair 内部展开为 chosen/rejected 两行, batched 后 forward."""
    z_us, prompt_chat, query_text, is_chosen = [], [], [], []
    for r in batch:
        z = r["z_u"]
        for prompt, query, flag in [
            (r["prompt_target_style"], r["chosen_query"], 1.0),
            (r["prompt_target_style"], r["rejected_query"], 0.0),
        ]:
            z_us.append(z)
            prompt_chat.append(prompt)
            query_text.append(query)
            is_chosen.append(flag)
    return {
        "z_u": torch.stack(z_us),
        "prompt_chat": prompt_chat,
        "query_text": query_text,
        "is_chosen": torch.tensor(is_chosen, dtype=torch.float32),
    }


def build_chat_prompts(tokenizer, system_text: str, user_attrs_text: str) -> str:
    """构造 Qwen chat prompt (system + user), 跟 phase10_generate.py 一致."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_attrs_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_inputs(tokenizer, prompt_chats: List[str], queries: List[str], device: str):
    """把 prompt + query 拼成完整输入序列, 返回 (input_ids, attention_mask, query_mask).

    关键: 单独 encode prompt 和 query, 然后拼接 (避免 BPE 在边界变).
    query_mask 在 query token 位置为 1, 用于计算 LM loss (只对 query 部分算 NLL).
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # 单独 encode prompt 和 query, 然后拼接
    prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_chats]
    query_ids_list = [tokenizer.encode(q, add_special_tokens=False) for q in queries]
    full_ids_list = [p_ids + q_ids for p_ids, q_ids in zip(prompt_ids_list, query_ids_list)]

    # right-pad for LM forward (we want query tokens at the end)
    max_len = min(MAX_PROMPT_LEN + MAX_QUERY_LEN, max(len(x) for x in full_ids_list))
    B = len(full_ids_list)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    query_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for b, (p_ids, q_ids) in enumerate(zip(prompt_ids_list, query_ids_list)):
        full_ids = (p_ids + q_ids)[:max_len]
        L = len(full_ids)
        input_ids[b, :L] = torch.tensor(full_ids, dtype=torch.long, device=device)
        attention_mask[b, :L] = 1
        p_len = min(len(p_ids), L)
        query_mask[b, p_len:L] = 1

    return input_ids, attention_mask, query_mask


def compute_logp(model, input_embeds: torch.Tensor, attention_mask: torch.Tensor,
                 query_mask: torch.Tensor, target_ids: torch.Tensor,
                 num_prefix_tokens: int = 0) -> torch.Tensor:
    """计算每个样本 query token 的 average log-prob (作为 DPO log_p).

    关键 alignment:
      full_emb = cat([prefix (K), text_emb (L)])
      full_emb 长度 = K + L
      model 输出 logits 长度 = K + L - 1 (predict 下一个 token)
      logits[:, K-1:K-1+L] = 在 text 部分每个 token 位置上的"下一个 token"预测
        → logits[b, K-1+i] 是 input_embeds[b, K+i] (text 第 i 个 token) 的预测分布
      target_ids 是 text 部分的 token ids, 长度 L
      text_query_mask = query_mask[:, K:K+L] (query 在 text 部分的位置)
    """
    out = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
    K = num_prefix_tokens
    L = target_ids.size(1)
    logits = out.logits[:, K-1:K-1+L, :].contiguous()  # [B, L, V]
    targets = target_ids.contiguous()                   # [B, L]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    chosen_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [B, L]
    text_query_mask = query_mask[:, K:K+L].contiguous()  # [B, L]
    mask = text_query_mask.to(chosen_lp.dtype)
    n_tokens = mask.sum(dim=1).clamp(min=1.0)
    avg_lp = (chosen_lp * mask).sum(dim=1) / n_tokens
    return avg_lp


def main():
    # 硬编码 (CLAUDE.md Rule 3 禁止 argparse 传参运行; env vars 作为可选覆盖)
    args_epochs = int(os.environ.get("PHASE10_EPOCHS", str(EPOCHS)))
    args_lr = float(os.environ.get("PHASE10_LR", str(LR)))
    args_batch = int(os.environ.get("PHASE10_BATCH", str(BATCH_SIZE)))
    args_beta = float(os.environ.get("PHASE10_BETA", str(BETA)))
    args_lambda_lm = float(os.environ.get("PHASE10_LAMBDA_LM", str(LAMBDA_LM)))

    log = lambda m: print(f"[phase10-train] {m}", flush=True)
    log("=" * 70)
    log(f"Phase 10.4: 训练 20d Raw-VADES → soft-prefix projector (DPO)")
    log("=" * 70)
    log(f"  epochs={args_epochs}, lr={args_lr}, batch_size={args_batch}, "
        f"beta={args_beta}, lambda_lm={args_lambda_lm}")

    if not PREF_PAIRS_FILE.exists():
        raise FileNotFoundError(
            f"{PREF_PAIRS_FILE} 不存在; 请先运行 build_preference_pairs.py"
        )

    # === 1. 加载 preference pairs ===
    records = []
    with PREF_PAIRS_FILE.open() as f:
        for line in f:
            records.append(json.loads(line))
    log(f"  preference pairs: {len(records)}")
    if not records:
        raise ValueError("preference_pairs.jsonl 为空")

    # === 1.5 过滤负 margin pairs (Phase 10.6.3: 1 个负 margin pair 会拉低 DPO 信号) ===
    records = [r for r in records if (r.get("maha_rejected", 0) - r.get("maha_chosen", 0)) >= MIN_MARGIN]
    log(f"  过滤 min_margin={MIN_MARGIN} 后: {len(records)} pairs")

    # 80/20 split
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(records))
    n_val = max(1, len(records) // 5)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    log(f"  train: {len(train_records)}, val: {len(val_records)}")

    train_ds = PrefPairDataset(train_records)
    train_loader = DataLoader(
        train_ds, batch_size=args_batch, shuffle=True,
        collate_fn=collate_pairs, num_workers=0, drop_last=True,
    )

    # === 2. 加载 Qwen + projector ===
    log(f"加载 Qwen2-7B Hidden backend (frozen) ...")
    from llm_client import _HiddenBackend  # 业务代码不允许 import transformers; 通过 llm_client.py 内部 _HiddenBackend 访问 transformers 模型
    backend = _HiddenBackend.get(QWEN_MODEL_PATH)
    model = backend.model
    tokenizer = backend.tokenizer
    model_dim = model.config.hidden_size
    log(f"  model_dim={model_dim}, dtype={backend.dtype}")

    from query.soft_prefix.projector import SoftPrefixProjector
    projector = SoftPrefixProjector(
        user_dim=USER_DIM,
        hidden_dim=HIDDEN_DIM,
        num_tokens=NUM_TOKENS,
        model_dim=model_dim,
        dtype=torch.float32,  # 训练用 fp32 更稳; 注入时 cast
        gate_init=1e-3,
    ).to(DEVICE)
    n_params = projector.num_parameters(trainable_only=True)
    log(f"  projector trainable params: {n_params:,}")

    optim = torch.optim.AdamW(projector.parameters(), lr=args_lr, weight_decay=0.0)

    # === 3. 训练循环 ===
    SYSTEM = (
        "You are a shopping query writer. Write one short natural shopping query "
        "that mentions every listed attribute of the product."
    )

    def run_batch(batch: dict, train: bool) -> dict:
        z_u = batch["z_u"].to(DEVICE).to(torch.float32)
        prompt_chats = [
            build_chat_prompts(tokenizer, SYSTEM, p) for p in batch["prompt_chat"]
        ]
        target_queries = batch["query_text"]
        is_chosen = batch["is_chosen"].to(DEVICE)

        input_ids, attention_mask, query_mask = encode_inputs(
            tokenizer, prompt_chats, target_queries, DEVICE,
        )

        # projector forward → prefix (trainable)
        prefix = projector(z_u).to(DTYPE)  # [B, K, H]
        emb_m = model.get_input_embeddings()
        text_emb = emb_m(input_ids).to(DTYPE)  # [B, L, H]
        full_emb = torch.cat([prefix, text_emb], dim=1)
        full_am = torch.cat(
            [torch.ones((full_emb.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE),
             attention_mask], dim=1,
        )

        # 用 projector prefix 时的 log_p_chosen / log_p_rejected
        # query_mask 在 prefix 注入前已经处理, 但 prefix 不参与 query_mask,
        # 而 query_mask 长度 = L (无 prefix). 我们 forward 用 full_emb (L+K),
        # 所以 query_mask 要 right-pad 到 L+K
        full_query_mask = torch.cat(
            [torch.zeros((full_emb.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE),
             query_mask], dim=1,
        )

        if train:
            avg_lp = compute_logp(model, full_emb, full_am, full_query_mask, input_ids,
                                  num_prefix_tokens=NUM_TOKENS)
        else:
            with torch.no_grad():
                avg_lp = compute_logp(model, full_emb, full_am, full_query_mask, input_ids,
                                      num_prefix_tokens=NUM_TOKENS)

        # 把 avg_lp 分成 chosen / rejected
        is_c_mask = (is_chosen > 0.5).bool()
        if is_c_mask.sum() == 0 or (~is_c_mask).sum() == 0:
            return None
        log_p_c = avg_lp[is_c_mask].mean()
        log_p_r = avg_lp[~is_c_mask].mean()

        # NLL loss (辅助, 防止语言崩溃)
        nll_c = -log_p_c
        nll_r = -log_p_r
        lm_loss = (nll_c + nll_r) / 2

        # DPO loss
        pref_loss = -F.logsigmoid(args_beta * (log_p_c - log_p_r))

        total = pref_loss + args_lambda_lm * lm_loss
        return {
            "total": total,
            "pref": pref_loss.detach().item(),
            "lm": lm_loss.detach().item(),
            "log_p_c": log_p_c.detach().item(),
            "log_p_r": log_p_r.detach().item(),
        }

    best_val = math.inf
    history = []
    for epoch in range(args_epochs):
        projector.train()
        ep_metrics = {"pref": [], "lm": [], "log_p_c": [], "log_p_r": [], "total": []}
        t0 = time.time()
        for batch in train_loader:
            try:
                out = run_batch(batch, train=True)
            except torch.cuda.OutOfMemoryError as e:
                log(f"    [OOM] {e}; skip batch (input too long)")
                torch.cuda.empty_cache()
                continue
            if out is None:
                continue
            (out["total"] / GRAD_ACCUM).backward()
            ep_metrics["total"].append(out["total"].detach().item())
            ep_metrics["pref"].append(out["pref"])
            ep_metrics["lm"].append(out["lm"])
            ep_metrics["log_p_c"].append(out["log_p_c"])
            ep_metrics["log_p_r"].append(out["log_p_r"])

            if (len(ep_metrics["total"])) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                optim.step()
                optim.zero_grad()

        # 任何残留梯度清零
        optim.zero_grad()

        train_pref = float(np.mean(ep_metrics["pref"] or [0]))
        train_lm = float(np.mean(ep_metrics["lm"] or [0]))
        train_lpc = float(np.mean(ep_metrics["log_p_c"] or [0]))
        train_lpr = float(np.mean(ep_metrics["log_p_r"] or [0]))

        # 验证
        val_pref = val_lm = val_lpc = val_lpr = float("nan")
        if val_records:
            projector.eval()
            val_ds = PrefPairDataset(val_records)
            vm = {"pref": [], "lm": [], "log_p_c": [], "log_p_r": []}
            with torch.no_grad():
                for vi in range(0, len(val_records), args_batch):
                    sub_indices = list(range(vi, min(vi + args_batch, len(val_records))))
                    sub_batch = collate_pairs([val_ds[i] for i in sub_indices])
                    out = run_batch(sub_batch, train=False)
                    if out is None:
                        continue
                    vm["pref"].append(out["pref"])
                    vm["lm"].append(out["lm"])
                    vm["log_p_c"].append(out["log_p_c"])
                    vm["log_p_r"].append(out["log_p_r"])
            val_pref = float(np.mean(vm["pref"] or [float("nan")]))
            val_lm = float(np.mean(vm["lm"] or [float("nan")]))
            val_lpc = float(np.mean(vm["log_p_c"] or [float("nan")]))
            val_lpr = float(np.mean(vm["log_p_r"] or [float("nan")]))

        log(f"  [epoch {epoch+1}/{args_epochs}] "
            f"train_pref={train_pref:.4f} train_lm={train_lm:.4f} "
            f"lpc={train_lpc:.3f} lpr={train_lpr:.3f} | "
            f"val_pref={val_pref:.4f} val_lm={val_lm:.4f} "
            f"lpc={val_lpc:.3f} lpr={val_lpr:.3f} | "
            f"{time.time()-t0:.1f}s")
        history.append({
            "epoch": epoch + 1, "train_pref": train_pref, "train_lm": train_lm,
            "train_lpc": train_lpc, "train_lpr": train_lpr,
            "val_pref": val_pref, "val_lm": val_lm,
            "val_lpc": val_lpc, "val_lpr": val_lpr,
        })

        # 保存 best (按 val_pref 最小)
        val_total = val_pref + args_lambda_lm * val_lm
        if not math.isnan(val_pref) and val_total < best_val:
            best_val = val_total
            torch.save({
                "state_dict": projector.state_dict(),
                "user_dim": USER_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_tokens": NUM_TOKENS,
                "model_dim": model_dim,
                "epoch": epoch + 1,
                "val_pref": val_pref,
            }, CKPT_FILE)
            log(f"    [ckpt] saved (val_total={val_total:.4f})")

    # === 4. 保存 meta ===
    meta = {
        "n_records": len(records),
        "n_train": len(train_records),
        "n_val": len(val_records),
        "epochs": args_epochs,
        "lr": args_lr,
        "batch_size": args_batch,
        "beta": args_beta,
        "lambda_lm": args_lambda_lm,
        "grad_accum": GRAD_ACCUM,
        "user_dim": USER_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_tokens": NUM_TOKENS,
        "model_dim": model_dim,
        "n_params": n_params,
        "ckpt_file": str(CKPT_FILE),
        "history": history,
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"已写入 {META_FILE}")
    log("=" * 70)
    log(f"训练完成. best_val_total={best_val:.4f}")
    log("=" * 70)


if __name__ == "__main__":
    main()
