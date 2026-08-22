#!/usr/bin/env python3
"""Phase 10.5 Primary Evaluation: 用训练好的 projector 走 Qwen 直生成 (无 rerank).

架构:
  1. 加载 Phase 10.4 训练好的 projector (40d → K=8 × 3584, fp32)
  2. 加载 Qwen2-7B (走 vLLM, with_vllm=True, 用于 generate)
  3. 30 user × 1 product × 4 condition × 5 candidate = 600 calls
     - A) no_style:      无 prefix, baseline
     - B) zero_prefix:   训练好的 projector 但输入 z=0 (sanity: 看 projector 学到没)
     - C) random_style:  projector 输入 random 用户的 z_u
     - D) target_style:  projector 输入 target 用户的 z_u
  4. 每条生成 → 强制 attr 完整 → 计算 maha_target (vs target user)
  5. 配对 bootstrap CI on D vs C 差值 (95% CI 不跨 0 = 风格生效)
  6. 属性完整率 ≥0.99 (强制)
  7. semantic_sim (vs reference) 三组对比

不使用 Mahalanobis rerank — primary evaluation 走 LLM 直生成.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PAIRS_FILE = OUT_DIR / "phase10_pairs.jsonl"
PROJ_CKPT = OUT_DIR / "phase10_6_3_projector.pt"
OUT_CANDS = OUT_DIR / "phase10_eval_candidates.jsonl"
OUT_META = OUT_DIR / "phase10_eval_meta.json"

# === 走本地 Qwen (必须用 llm_client.py) ===
os.environ.setdefault("LLM_CLIENT", "qwen_local")
from llm_client import create_qwen_local_client  # noqa: E402

# === Mahalanobis scorer ===
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
TAG = "vades_prototype_3000u_v6_raw"
USER_PROFILE_FILE = VADES_DIR / f"{TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{TAG}_sentences.jsonl"

ATTR_FIELDS = ["Brand", "Item Weight", "Product Dimensions", "Color", "Material"]
N_CANDIDATES_PER_COND = 5
CONDITIONS = ["A_no_style", "B_zero_prefix", "C_random_style", "D_target_style"]
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
TOP_P = 0.95

SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query "
    "that mentions every listed attribute of the product by its exact value "
    "(brand name, weight, dimensions, color, material). Keep the query under 25 words."
)


# === helpers ===
def load_user_profiles() -> Tuple[Dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    user_mu = np.array([p["user_mu"] for p in profiles], dtype=np.float32)
    user_logvar = np.array([p["user_logvar"] for p in profiles], dtype=np.float32)

    # 训练集标准化参数
    feat_rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            feat_rows.append([float(r["features"][k]) for k in list(r["features"].keys())])
    feat_array = np.array(feat_rows, dtype=np.float32)
    feat_mean = feat_array.mean(axis=0)
    feat_std = feat_array.std(axis=0) + 1e-9

    with SENTENCE_FILE.open() as f:
        first = json.loads(f.readline())
    feature_names = list(first["features"].keys())
    return user_id_to_idx, user_mu, user_logvar, feat_mean, feat_std, feature_names


def maha_one(q_norm: np.ndarray, mu_n: np.ndarray, logvar: np.ndarray) -> float:
    diff = q_norm - mu_n
    return float(((diff ** 2) * np.exp(-logvar)).sum())


def check_attr_pass(query: str, attrs: Dict[str, str]) -> bool:
    """5 attrs 全部出现在 query 中 (容忍复数变体、数字小数点、Brand 短 token).

    见 phase10_generate.py 同样实现 (保留 ".", "/", "-", Brand 短 token).
    """
    import re
    q_lower = query.lower()
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", q_lower)
    for k, v in attrs.items():
        if not v:
            return False
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        all_tokens = v_clean.split()
        tokens = []
        for t in all_tokens:
            if len(t) >= 3:
                tokens.append(t)
            elif re.match(r"^\d+(\.\d+)?$", t):
                tokens.append(t)
            elif k == "Brand" and re.match(r"^[a-zA-Z]+$", t) and len(t) >= 2:
                tokens.append(t)
        if not tokens:
            return False
        found = False
        for t in tokens:
            t_lower = t.lower()
            if t_lower in q_clean:
                found = True
                break
            if t_lower.endswith("s") and len(t_lower) > 3 and t_lower[:-1] in q_clean:
                found = True
                break
            if (t_lower + "s") in q_clean:
                found = True
                break
        if not found:
            return False
    return True


def append_missing_attrs(query: str, attrs: Dict[str, str]) -> str:
    import re
    if check_attr_pass(query, attrs):
        return query
    missing = []
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", query.lower())
    for k, v in attrs.items():
        if not v:
            continue
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        all_tokens = v_clean.split()
        tokens = []
        for t in all_tokens:
            if len(t) >= 3:
                tokens.append(t)
            elif re.match(r"^\d+(\.\d+)?$", t):
                tokens.append(t)
        found = False
        for t in tokens:
            t_lower = t.lower()
            if t_lower in q_clean:
                found = True
                break
            if t_lower.endswith("s") and len(t_lower) > 3 and t_lower[:-1] in q_clean:
                found = True
                break
            if (t_lower + "s") in q_clean:
                found = True
                break
        if not found:
            words = str(v).split()[:4]
            cleaned = " ".join(w.rstrip(":;,.") for w in words).strip()
            if cleaned:
                missing.append(f"{k} {cleaned}")
    if not missing:
        return query
    if len(missing) == 1:
        clause = f"with {missing[0]}"
    elif len(missing) == 2:
        clause = f"with {missing[0]} and {missing[1]}"
    else:
        clause = "with " + ", ".join(missing[:-1]) + f", and {missing[-1]}"
    return query.rstrip(" .") + f", {clause}."


def semantic_sim(q1: str, q2: str) -> float:
    import re
    if not q1 or not q2:
        return 0.0
    toks1 = set(re.findall(r"\w+", q1.lower()))
    toks2 = set(re.findall(r"\w+", q2.lower()))
    if not toks1 or not toks2:
        return 0.0
    return len(toks1 & toks2) / len(toks1 | toks2)


def main():
    log = lambda m: print(f"[phase10-eval] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.5 Primary Evaluation (无 rerank)")
    log("=" * 70)

    # === 1. 加载 pairs + user profiles + 标准化参数 ===
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    log(f"  test pairs: {len(pairs)}")

    user_id_to_idx, user_mu, user_logvar, feat_mean, feat_std, feature_names = load_user_profiles()
    log(f"  users in profiles: {len(user_id_to_idx)}")

    user_mu_norm = (user_mu - feat_mean) / feat_std

    # === 2. 加载 projector ===
    if not PROJ_CKPT.exists():
        raise FileNotFoundError(
            f"{PROJ_CKPT} 不存在; 请先运行 phase10_train_projector.py"
        )
    ckpt = torch.load(PROJ_CKPT, map_location="cpu", weights_only=False)
    from query.soft_prefix.projector import SoftPrefixProjector

    projector = SoftPrefixProjector(
        user_dim=ckpt["user_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_tokens=ckpt["num_tokens"],
        model_dim=ckpt["model_dim"],
        dtype=torch.float32,
        gate_init=1e-3,
    )
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval()
    log(f"  projector loaded: user_dim={ckpt['user_dim']}, K={ckpt['num_tokens']}, "
        f"model_dim={ckpt['model_dim']}, epoch={ckpt['epoch']}")
    # projector 先放到 cpu, 等加载完 Qwen 后再搬到 cuda
    projector = projector.to("cpu")

    # === 3. 加载 Qwen + tokenizer (用于生成 + 注入 prefix) ===
    # 因为要走 inputs_embeds 注入 prefix, 必须用 _HiddenBackend (transformers),
    # 不能走 vLLM (vLLM 不暴露 inputs_embeds 接口).
    # _HiddenBackend 是 llm_client.py 内部已封装 transformers 的入口,
    # 业务侧直接调用, 不在业务代码里 import transformers (AGENTS.md Rule 8).
    from llm_client import _HiddenBackend
    QWEN_MODEL_PATH = os.environ.get(
        "QWEN_MODEL_PATH",
        "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
    )
    backend = _HiddenBackend.get(QWEN_MODEL_PATH)
    model = backend.model
    tokenizer = backend.tokenizer
    DEVICE = next(model.parameters()).device
    projector = projector.to(DEVICE)
    log(f"  Qwen loaded on {DEVICE}, dtype={backend.dtype}, projector moved to {DEVICE}")

    # === 4. 构造每个 (user, asin, cond) 的 prompt + prefix ===
    # 走 system prompt + user attrs + style desc (与 phase10_generate 一致)
    rng = np.random.RandomState(2026)

    results = []
    n_total = 0
    n_attr_pass = 0

    # 由于 projector 输入到 Qwen inputs_embeds 是 transformers-only,
    # 我们手写生成循环: prefix 注入 + model.generate(inputs_embeds=...)
    # 实际上 forward 用 inputs_embeds + KV cache 跟 model.generate(inputs_embeds=...) 都行
    # 这里为了批量, 直接写 prefix 前缀 batch forward.

    emb_m = model.get_input_embeddings()

    def generate_with_prefix(prompts: List[str], prefix: torch.Tensor, n_per: int = N_CANDIDATES_PER_COND):
        """prefix: [B, K, H]; 对每个 prompt 生成 n_per 个候选 (用 temperature sampling).

        返回: List[List[str]] (B x n_per)
        """
        B = len(prompts)
        # 构造 prompt ids (right-padding)
        # 注意: apply_chat_template 给出完整 prompt, 我们用 tokenizer 重新 tokenize 后 right-pad
        chat_prompts = []
        for p in prompts:
            cp = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            chat_prompts.append(cp)
        prompt_ids_list = [tokenizer.encode(cp, add_special_tokens=False) for cp in chat_prompts]
        max_p = max(len(x) for x in prompt_ids_list)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        ids = torch.full((B, max_p), pad_id, dtype=torch.long, device=DEVICE)
        am = torch.zeros((B, max_p), dtype=torch.long, device=DEVICE)
        for b, pl in enumerate(prompt_ids_list):
            ids[b, max_p - len(pl):] = torch.tensor(pl, dtype=torch.long, device=DEVICE)
            am[b, max_p - len(pl):] = 1

        text_emb = emb_m(ids).to(model.dtype)
        full_emb = torch.cat([prefix.to(model.dtype), text_emb], dim=1)
        full_am = torch.cat(
            [torch.ones((B, prefix.size(1)), dtype=torch.long, device=DEVICE), am], dim=1,
        )
        # 生成
        with torch.no_grad():
            out = model.generate(
                inputs_embeds=full_emb,
                attention_mask=full_am,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # out: [B, n_new] - 直接 decode
        decoded = []
        for b in range(B):
            text = tokenizer.decode(out[b].tolist(), skip_special_tokens=True).strip()
            # 截断到第一个换行
            text = text.split("\n")[0]
            decoded.append(text)
        return decoded

    # 4 个 condition 的 prompt 模板
    def build_cond_prompts(attrs: dict, target_style_desc: str, random_style_desc: str) -> Dict[str, str]:
        attr_text = "\n".join(f"{k}: {v}" for k, v in attrs.items())
        return {
            "A_no_style": (
                f"Product attributes:\n{attr_text}\n\n"
                f"Write a short shopping query that includes every attribute."
            ),
            "B_zero_prefix": (  # prompt 加 style desc 但 prefix 强制为 0 (sanity)
                f"Product attributes:\n{attr_text}\n\n"
                f"{target_style_desc}\n\n"
                f"Write a short shopping query that includes every attribute."
            ),
            "C_random_style": (
                f"Product attributes:\n{attr_text}\n\n"
                f"{random_style_desc}\n\n"
                f"Write a short shopping query that includes every attribute."
            ),
            "D_target_style": (
                f"Product attributes:\n{attr_text}\n\n"
                f"{target_style_desc}\n\n"
                f"Write a short shopping query that includes every attribute."
            ),
        }

    # === 5. 主循环 (BATCH 加速: 同 cond × 5 candidate 一次 forward) ===
    # 每个 pair: 4 cond × 5 candidate = 20 candidates; 现在合并成 4 个 forward (每个 B=5)
    # 实际 4 cond prompt 长度差异不大, 也可以 cond 一起 batch (B=20), 但 prefix 难拼接
    # 先做每个 cond 内 batch=5, 后续可继续优化
    OUT_CANDS.parent.mkdir(parents=True, exist_ok=True)
    fout = OUT_CANDS.open("w")
    t0 = time.time()
    for pi, pair in enumerate(pairs):
        if pi % 5 == 0:
            log(f"  [{pi+1}/{len(pairs)}] user={pair['user_id'][:10]} asin={pair['asin']} "
                f"({time.time()-t0:.1f}s)")
        attrs = pair["attrs"]
        prompts = build_cond_prompts(
            attrs, pair["target_style_desc"], pair["random_style_desc"],
        )

        # 准备 4 个 cond 的 z_u
        target_idx = user_id_to_idx[pair["user_id"]]
        target_z = np.concatenate([user_mu[target_idx], user_logvar[target_idx]])
        random_idx = user_id_to_idx[pair["random_user_id"]]
        random_z = np.concatenate([user_mu[random_idx], user_logvar[random_idx]])

        cond_z = {
            "A_no_style": None,
            "B_zero_prefix": np.zeros_like(target_z),
            "C_random_style": random_z,
            "D_target_style": target_z,
        }

        for cond in CONDITIONS:
            zs = cond_z[cond]
            if zs is None:
                # A_no_style: 每个 candidate 共享 prefix=0 (batch 共享)
                prefix_one = torch.zeros(
                    (1, ckpt["num_tokens"], ckpt["model_dim"]),
                    dtype=torch.float32, device=DEVICE,
                )
            else:
                with torch.no_grad():
                    z_t = torch.tensor(zs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    prefix_one = projector(z_t)  # [1, K, H]

            # === BATCH: 一次 forward 生成 5 个 candidate (同 cond, 同 prompt, 同 prefix) ===
            # 不加 prefix noise! 之前用户指出: "额外噪声可能直接把目标 prefix 推离用户方向"
            # 候选多样性通过 model.generate(do_sample=True) 自身的 sampling randomness 提供
            # 4 个 cond 共享同一随机种子候选预算 (model.generate 不指定 seed 时由 generator 决定)
            BATCH_N = N_CANDIDATES_PER_COND
            prefix = prefix_one.expand(BATCH_N, -1, -1).contiguous()  # 完全相同 prefix, 5 行

            try:
                decoded = generate_with_prefix([prompts[cond]] * BATCH_N, prefix)
                raw_qs = decoded
            except torch.cuda.OutOfMemoryError as e:
                log(f"    [OOM] {e}; fallback to batch=1")
                torch.cuda.empty_cache()
                raw_qs = []
                for ci in range(BATCH_N):
                    try:
                        raw_q = generate_with_prefix([prompts[cond]], prefix[ci:ci+1])[0]
                    except Exception as e2:
                        log(f"    [WARN] single generate failed: {e2}")
                        raw_q = ""
                    raw_qs.append(raw_q)
            except Exception as e:
                log(f"    [WARN] batch generate failed: {e}; use empty")
                raw_qs = [""] * BATCH_N

            for ci, raw_q in enumerate(raw_qs):
                kept_q = append_missing_attrs(raw_q, attrs)
                attr_pass = check_attr_pass(kept_q, attrs)
                n_total += 1
                if attr_pass:
                    n_attr_pass += 1
                rec = {
                    "user_id": pair["user_id"],
                    "asin": pair["asin"],
                    "random_user_id": pair["random_user_id"],
                    "cond": cond,
                    "candidate_index": ci,
                    "candidate_query": kept_q,
                    "candidate_query_raw": raw_q,
                    "attrs": attrs,
                    "attr_pass": attr_pass,
                }
                results.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    fout.close()
    log(f"已写入 {OUT_CANDS} ({len(results)} 条)")

    # === 6. 输出 ===
    OUT_CANDS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CANDS.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"已写入 {OUT_CANDS} ({len(results)} 条)")

    # === 7. 计算 maha_target per cond ===
    from syntactic_analysis.extract_clause_features_single_query import extract_clause_features
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

    # 按 (user, asin, cond) group
    by_cond = {c: [] for c in CONDITIONS}
    by_pair = {}  # (user, asin) -> {cond: [maha_target]}
    for r in results:
        if not r["attr_pass"]:
            continue
        try:
            feat_dict = extract_clause_features(r["candidate_query"])
            feat_raw = np.array([float(feat_dict[n]) for n in feature_names], dtype=np.float32)
            feat_n = (feat_raw - feat_mean) / feat_std
            target_idx = user_id_to_idx[r["user_id"]]
            mt = maha_one(feat_n, user_mu_norm[target_idx], user_logvar[target_idx])
        except Exception:
            mt = float("inf")
        by_cond[r["cond"]].append(mt)
        key = (r["user_id"], r["asin"])
        by_pair.setdefault(key, {}).setdefault(r["cond"], []).append(mt)

    # === 8. 配对 bootstrap CI on D - C 差值 ===
    diffs = []  # D - C per (user, asin)
    for key, cdict in by_pair.items():
        if "D_target_style" in cdict and "C_random_style" in cdict:
            # 取每 cond 的均值代表这对
            d = np.mean(cdict["D_target_style"])
            c = np.mean(cdict["C_random_style"])
            diffs.append(d - c)
    diffs = np.array(diffs)
    n_boot = 1000
    rng_b = np.random.RandomState(0)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng_b.randint(0, len(diffs), size=len(diffs))
        boot_diffs.append(np.mean(diffs[idx]))
    boot_diffs = np.array(boot_diffs)
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    # === 9. 汇总 ===
    meta = {
        "n_total": n_total,
        "n_attr_pass": n_attr_pass,
        "attr_pass_rate": n_attr_pass / max(n_total, 1),
        "n_pairs_with_d_and_c": int(len(diffs)),
        "conditions": CONDITIONS,
        "mean_maha_target_by_cond": {c: float(np.mean(v)) if v else None for c, v in by_cond.items()},
        "median_maha_target_by_cond": {c: float(np.median(v)) if v else None for c, v in by_cond.items()},
        "mean_sem_diff_D_minus_C": float(diffs.mean()) if len(diffs) else None,
        "boot_diff_mean": float(boot_diffs.mean()) if len(boot_diffs) else None,
        "boot_diff_ci95": [float(ci_low), float(ci_high)] if len(diffs) else None,
        "D_less_than_C": bool(diffs.mean() < 0) if len(diffs) else None,
        "ckpt_path": str(PROJ_CKPT),
        "ckpt_epoch": ckpt.get("epoch"),
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"已写入 {OUT_META}")

    log("=" * 70)
    log("Phase 10.5 汇总:")
    log(f"  attr_pass_rate: {meta['attr_pass_rate']:.4f}")
    for c in CONDITIONS:
        log(f"  {c}: mean_maha={meta['mean_maha_target_by_cond'][c]:.2f} "
            f"median={meta['median_maha_target_by_cond'][c]:.2f}")
    log(f"  D-C mean: {meta['mean_sem_diff_D_minus_C']:.2f}")
    log(f"  D-C 95% bootstrap CI: {meta['boot_diff_ci95']}")
    log(f"  D < C: {meta['D_less_than_C']}")
    log("=" * 70)


if __name__ == "__main__":
    main()
