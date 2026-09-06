#!/usr/bin/env python3
"""SFT pipeline for query generation (完全 SFT, 不再 GRPO).

用户 2026-09-05 决策: 完全 SFT 路线 (替换之前的 GRPO fresh-base 路线).
等用户提供训练样本后, 在 _load_sft_dataset() 处接入.

Pipeline:
  Stage A: 加载样本 -> {prompt, response} 对
  Stage B: LoRA SFT on Qwen2.5-0.5B-Instruct (masked answer loss)
  Stage C: 独立 ASIN 配对评估 (base vs SFT)

输入 (等用户提供):
  训练样本格式: list[dict], 每项含 "prompt" + "response"
  prompt:    "You are shopping for a product with these 5 known attributes:..."
  response:  已组织好的合格 query (cov=5, no rep, no extra)

输出:
  result/06_training_model/sft_lora/                  (LoRA adapter)
  result/06_training_model/sft_training_log.json      (loss 曲线 + 配置)
  result/06_training_model/eval_sft_vs_base.json      (Stage C 配对评估)

用法 (Rule 3: no args, env vars only):
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  SFT_TRAIN_SMOKE=1     $PY 06_training_model/sft_pipeline.py
  SFT_TRAIN_FULL=1      $PY 06_training_model/sft_pipeline.py
  SFT_EVAL_INDEPENDENT=1 $PY 06_training_model/sft_pipeline.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "result/06_training_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QWEN_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction/product_attributes.json"
SFT_ADAPTER_DIR = OUT_DIR / "sft_lora"
os.environ.setdefault("HF_HOME", "/home/wlia0047/hj82_scratch2/wenyu/hf_cache")

# --- Hardcoded hyperparams (Rule 3) ---
SFT_SEED = 42
SFT_LORA_R = 8
SFT_LORA_ALPHA = 16
SFT_LEARNING_RATE = 2e-4
SFT_NUM_EPOCHS = 3
SFT_BATCH_SIZE = 4
SFT_GRAD_ACCUM = 2
SFT_MAX_SEQ_LEN = 256
SFT_MAX_NEW_TOKENS = 40
SFT_SAMPLE_TEMP = 0.7
SFT_TOP_P = 0.95
SFT_SMOKE_N_PAIRS = 32           # smoke 用 32 对
SFT_FULL_N_PAIRS = 100           # full 用 100 对 (用户 query_samples_100.json 提供)
SFT_EVAL_N_INDEPENDENT = 10
SFT_EVAL_N_CANDS = 5
SFT_EVAL_MAX_EXTRAS = 3          # 用户 2026-09-05 决策: 容忍 "I'd"/"like" 等填充词
                                 # (用户样本 query 是长句模板, 词数 20-30, ≤1 太严)
                                 # 训练时仍用 ≤1 (硬约束目标), eval 时用 ≤3 (实际可达)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Prompt builder (复用 GRPO 的, 因为用户已确认新 prompt 设计)
# ============================================================================
def _build_sft_prompt(attrs: Dict[str, str]) -> str:
    """用户 2026-09-05 确认的 prompt: 明确'用已知属性表达搜索需求'任务.

    末尾加 Qwen2.5 的 eos_token `<|im_end|>`, 让 model 学'生成 response 后立即停'.
    否则 model 会继续生成 'Query: ... Answer: ...' 等 system echo 形式.
    """
    attrs_text = "\n".join(f"- {k}: {v}" for k, v in attrs.items())
    return (
        f"You are shopping for a product with these 5 known attributes:\n"
        f"{attrs_text}\n\n"
        f"Write one short natural shopping search query (5-20 words) that "
        f"uses these attributes as keywords. Do NOT ask questions about the "
        f"attributes (they are already known). Do NOT invent or add new "
        f"product information not listed above.\n\n"
        f"Query:<|im_end|>\n"
    )


# ============================================================================
# Dataset loading — 等用户给样本
# ============================================================================
def _load_sft_dataset(n_pairs: int) -> List[Dict[str, str]]:
    """加载 SFT 训练样本.

    用户 2026-09-05 提供: /home/wlia0047/ar57/wenyu/PersoanlQuery/query_samples_100.json
    格式: JSON array [{id, attributes: {k: v}, query: str}, ...]
    """
    # 优先路径: 用户提供
    user_path = REPO_ROOT / "query_samples_100.json"
    if user_path.exists():
        log(f"  loading SFT dataset from: {user_path}")
        with open(user_path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data)}")
        log(f"  loaded {len(data)} pairs (will subsample to {n_pairs})")
        rng = random.Random(SFT_SEED)
        rng.shuffle(data)
        return data[:n_pairs]

    # Fallback 占位路径 (其他用户接入位置)
    candidate_paths = [
        REPO_ROOT / "result/06_training_model/sft_samples_user_provided.jsonl",
        REPO_ROOT / "result/06_training_model/sft_samples.jsonl",
    ]
    for p in candidate_paths:
        if p.exists():
            log(f"  loading SFT dataset from: {p}")
            with open(p) as f:
                data = [json.loads(line) for line in f if line.strip()]
            log(f"  loaded {len(data)} pairs (will subsample to {n_pairs})")
            rng = random.Random(SFT_SEED)
            rng.shuffle(data)
            return data[:n_pairs]

    raise FileNotFoundError(
        f"No SFT dataset found. Expected {user_path} or one of: "
        f"{[str(p) for p in candidate_paths]}."
    )


def _normalize_sample(row: Dict) -> Dict[str, str]:
    """归一化样本为 {prompt, response} 格式.

    支持:
      - {'prompt': str, 'response': str}
      - {'attrs': {k: v}, 'query': str}
      - {'attributes': {k: v}, 'query': str}  (用户 query_samples_100.json 格式)
    """
    if "prompt" in row and "response" in row:
        return {"prompt": row["prompt"], "response": row["response"]}
    attrs = row.get("attrs") or row.get("attributes")
    query = row.get("query") or row.get("response")
    if attrs and query:
        return {"prompt": _build_sft_prompt(attrs), "response": query}
    raise ValueError(f"Unrecognized sample format: {list(row.keys())}")


# ============================================================================
# Stage B: LoRA SFT training
# ============================================================================
def stage_b_sft_train(smoke: bool = True) -> None:
    """LoRA SFT on Qwen2.5-0.5B-Instruct with masked answer loss.

    Standard SFT setup:
      - Tokenize prompt + response
      - Labels = -100 for prompt tokens, response token ids for answer tokens
      - LoRA on q,k,v,o,gate,up,down projections (r=8)
      - AdamW, lr=2e-4, 3 epochs
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    n_pairs = SFT_SMOKE_N_PAIRS if smoke else SFT_FULL_N_PAIRS
    log("=== stage_b_sft_train (smoke={}) ===".format(smoke))
    log(f"  n_pairs={n_pairs} epochs={SFT_NUM_EPOCHS} lr={SFT_LEARNING_RATE} "
        f"lora_r={SFT_LORA_R} batch={SFT_BATCH_SIZE} grad_accum={SFT_GRAD_ACCUM}")

    # --- Load dataset ---
    raw = _load_sft_dataset(n_pairs)
    samples = [_normalize_sample(r) for r in raw]
    log(f"  normalized {len(samples)} samples")

    # --- Tokenizer ---
    tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- Load base model + LoRA ---
    log(f"  loading base: {QWEN_PATH}")
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True)
    lora_cfg = LoraConfig(
        r=SFT_LORA_R, lora_alpha=SFT_LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=SFT_LEARNING_RATE, weight_decay=0.0,
    )

    # --- Tokenize with prompt-mask labels ---
    def _encode(samples_batch):
        prompts = [s["prompt"] for s in samples_batch]
        responses = [s["response"] for s in samples_batch]
        # 用户 2026-09-05: 在 response 末尾追加 eos, 让 model 学"生成完 response 立即停",
        # 避免续写 "Query: ... Answer: ..." 等 system echo.
        # prompt 末尾已含 <|im_end|> (见 _build_sft_prompt), response 末尾也加一个.
        full_texts = [p + r + tok.eos_token for p, r in zip(prompts, responses)]
        enc = tok(full_texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=SFT_MAX_SEQ_LEN).to("cuda:0")
        labels = enc["input_ids"].clone()
        # Mask prompt tokens: re-encode prompts alone to find their lengths
        prompt_encs = tok(prompts, return_tensors=None, padding=False, add_special_tokens=False)
        for i, p_ids in enumerate(prompt_encs["input_ids"]):
            p_len = len(p_ids)
            labels[i, :p_len] = -100
        # Also mask padding
        labels[enc["attention_mask"] == 0] = -100
        return enc["input_ids"], enc["attention_mask"], labels

    # --- Training loop ---
    rng = random.Random(SFT_SEED)
    rng.shuffle(samples)
    n = len(samples)
    train_log = []
    global_step = 0
    for epoch in range(SFT_NUM_EPOCHS):
        rng.shuffle(samples)
        for start in range(0, n, SFT_BATCH_SIZE):
            batch = samples[start:start + SFT_BATCH_SIZE]
            input_ids, attn, labels = _encode(batch)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / SFT_GRAD_ACCUM
            loss.backward()
            if (global_step + 1) % SFT_GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            if global_step % max(1, n // (SFT_BATCH_SIZE * 4)) == 0:
                log(f"  epoch {epoch+1}/{SFT_NUM_EPOCHS} step {global_step} "
                    f"loss={float(out.loss):.4f}")
                train_log.append({"epoch": epoch, "step": global_step,
                                  "loss": float(out.loss)})

    # --- Save adapter ---
    SFT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(SFT_ADAPTER_DIR))
    tok.save_pretrained(str(SFT_ADAPTER_DIR))
    log(f"  saved adapter -> {SFT_ADAPTER_DIR}")

    # --- Save training log ---
    log_path = OUT_DIR / "sft_training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "config": {
                "smoke": smoke, "n_pairs": n_pairs,
                "epochs": SFT_NUM_EPOCHS, "lr": SFT_LEARNING_RATE,
                "lora_r": SFT_LORA_R, "lora_alpha": SFT_LORA_ALPHA,
                "batch_size": SFT_BATCH_SIZE, "grad_accum": SFT_GRAD_ACCUM,
            },
            "train_log": train_log,
        }, f, indent=2, ensure_ascii=False)
    log(f"  saved log -> {log_path}")
    log("=== stage_b_sft_train DONE ===")


# ============================================================================
# Stage C: Independent ASIN paired eval (base vs SFT)
# ============================================================================
def stage_c_eval_independent():
    """Evaluate SFT adapter vs fresh base on 5 NEW held-out ASINs."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common.grpo_reward import check_content, compute_fluency, tokenize

    log("=== stage_c_eval_independent ===")

    with open(ATTRIBUTES_PATH) as f:
        attrs_all = json.load(f)
    train_path = OUT_DIR / "sft_train_asins.json"
    if not train_path.exists():
        # 排除 GRPO 训练集 (避免 overlap)
        grpo_path = OUT_DIR / "grpo_train_asins.json"
        if grpo_path.exists():
            with open(grpo_path) as f:
                grpo_obj = json.load(f)
            excluded = set(grpo_obj.get("train_asins", []))
        else:
            excluded = set()
    else:
        with open(train_path) as f:
            excluded = set(json.load(f).get("train_asins", []))

    candidates = [a for a in sorted(attrs_all.keys()) if a not in excluded
                  and len(list(attrs_all[a].items())[:5]) >= 5]
    rng = random.Random(SFT_SEED + 99)
    rng.shuffle(candidates)
    eval_asins = candidates[:SFT_EVAL_N_INDEPENDENT]
    log(f"  eval_asins: {eval_asins}")
    log(f"  excluded: {len(excluded)}")

    tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log("  [base] loading Qwen2.5-0.5B-Instruct ...")
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    has_adapter = (SFT_ADAPTER_DIR / "adapter_config.json").exists()
    sft = None
    if has_adapter:
        log("  [sft] loading base + SFT adapter ...")
        sft_base = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH, torch_dtype=torch.bfloat16,
            device_map="cuda:0", trust_remote_code=True)
        sft = PeftModel.from_pretrained(sft_base, str(SFT_ADAPTER_DIR))
        sft.eval()
        for p in sft.parameters():
            p.requires_grad_(False)

    def _gen(model, asins):
        results = []
        # Qwen2.5 eos_token 是 <|im_end|> (151645). prompt 末尾已加 <|im_end|>,
        # model 应学会生成 response 后立即 emit eos 停. 用 eos_token_id 强制停.
        eos_id = tok.eos_token_id
        for a in asins:
            attrs = dict(list(attrs_all[a].items())[:5])
            prompt = _build_sft_prompt(attrs)
            enc = tok([prompt], return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to("cuda:0")
            torch.manual_seed(SFT_SEED)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=SFT_MAX_NEW_TOKENS,
                    do_sample=True, temperature=SFT_SAMPLE_TEMP,
                    top_p=SFT_TOP_P, pad_token_id=tok.pad_token_id,
                    num_return_sequences=SFT_EVAL_N_CANDS,
                    eos_token_id=eos_id,
                )
            prompt_len = enc["input_ids"].shape[1]
            gen_part = out[:, prompt_len:]
            cands = []
            for k in range(SFT_EVAL_N_CANDS):
                text = tok.decode(gen_part[k], skip_special_tokens=True).split("\n")[0].strip()
                cands.append(text)
            results.append({"asin": a, "attrs": attrs, "cands": cands})
        return results

    base_results = _gen(base, eval_asins)
    log("  [base] done")
    sft_results = _gen(sft, eval_asins) if sft else []

    def _score(results):
        scored = []
        for r in results:
            per_asin = {"asin": r["asin"], "attrs": r["attrs"], "cands": []}
            for c in r["cands"]:
                cov, miss, rep, extras = check_content(c, r["attrs"])
                F = compute_fluency(c)
                n_words = len(tokenize(c))
                # 用户 2026-09-05: eval 阈值 ≤3 容忍 "I'd"/"like" 填充词
                content_pass = (len(miss) == 0 and len(rep) == 0
                                and len(extras) <= SFT_EVAL_MAX_EXTRAS)
                per_asin["cands"].append({
                    "text": c, "cov": cov, "missing": miss,
                    "repeated": rep, "extras": extras,
                    "pass": content_pass, "F": F, "n_words": n_words,
                })
            scored.append(per_asin)
        return scored

    base_scored = _score(base_results)
    sft_scored = _score(sft_results) if sft_results else []

    def _agg(scored):
        agg = {"cov": [], "extras": [], "F": [], "pass": [], "n_words": [],
               "rep_rate": []}
        for r in scored:
            for c in r["cands"]:
                agg["cov"].append(c["cov"])
                agg["extras"].append(len(c["extras"]))
                agg["F"].append(c["F"])
                agg["pass"].append(1 if c["pass"] else 0)
                agg["n_words"].append(c["n_words"])
                agg["rep_rate"].append(1 if c["repeated"] else 0)
        import numpy as np
        return {k: (float(np.mean(v)) if v else 0.0,
                    float(np.std(v)) if v else 0.0)
                for k, v in agg.items()}

    base_agg = _agg(base_scored)
    sft_agg = _agg(sft_scored) if sft_scored else {}

    log(f"  [base] cov={base_agg['cov'][0]:.3f} pass={base_agg['pass'][0]:.3f} "
        f"extras={base_agg['extras'][0]:.2f}")
    if sft_agg:
        log(f"  [sft]  cov={sft_agg['cov'][0]:.3f} pass={sft_agg['pass'][0]:.3f} "
            f"extras={sft_agg['extras'][0]:.2f}")

    out_path = OUT_DIR / "eval_sft_vs_base.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"n_eval_asin": SFT_EVAL_N_INDEPENDENT,
                       "n_cands": SFT_EVAL_N_CANDS},
            "eval_asins": eval_asins,
            "base_summary": {k: {"mean": v[0], "std": v[1]}
                              for k, v in base_agg.items()},
            "sft_summary": {k: {"mean": v[0], "std": v[1]}
                             for k, v in sft_agg.items()} if sft_agg else {},
            "base_scored": base_scored,
            "sft_scored": sft_scored,
        }, f, indent=2, ensure_ascii=False)
    log(f"  saved -> {out_path}")
    log("=== stage_c_eval_independent DONE ===")


# ============================================================================
# Main dispatch (Rule 3: no args, env vars only)
# ============================================================================
def main_pipeline():
    if os.environ.get("SFT_TRAIN_SMOKE", "0") == "1":
        stage_b_sft_train(smoke=True)
        return
    if os.environ.get("SFT_TRAIN_FULL", "0") == "1":
        stage_b_sft_train(smoke=False)
        return
    if os.environ.get("SFT_EVAL_INDEPENDENT", "0") == "1":
        stage_c_eval_independent()
        return
    log("No SFT_* env var set. Set one of: "
        "SFT_TRAIN_SMOKE=1 / SFT_TRAIN_FULL=1 / SFT_EVAL_INDEPENDENT=1")


if __name__ == "__main__":
    main_pipeline()
