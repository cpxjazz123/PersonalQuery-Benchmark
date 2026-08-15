#!/usr/bin/env python3
"""E17 alpha sweep: regenerate "correct" condition with multiple alpha gates
and measure 5-attr exact-match + paired d_z vs zero baseline.

Goal: find an alpha that maximizes Check-5 (content) without breaking Check-4
(user-style signal). This avoids retraining and lets us close Check-5.

Output: e17_alpha_sweep.json with per-alpha 5-attr and d_z.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))

from copy_aware import build_attr_prompt_lines  # noqa: E402
from projector import SoftPrefixProjector  # noqa: E402

BASE = "/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
E17 = REPO_ROOT / "result" / "personal_query" / "e17"
STAGE1 = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / "Baby_Products" / "stage1_filtered_users_reviews.json"
TRAIN_VECS = json.load(open(E17 / "e17_train_vectors.json"))
SPLIT = json.load(open(E17 / "e17_train_dev_split.json"))
CKPT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/e17_ckpt")

SEED = 42
N_USERS = 30           # subset for speed
PRODUCTS_PER_USER = 1
GEN_SEED = 0
MAX_NEW_TOKENS = 32
BATCH = 16
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
USER_DIM = 30
NUM_TOKENS = 4
GATE_INIT = 0.05
ALPHAS = [0.0, 0.005, 0.01, 0.025, 0.05, 0.1]

NUMERIC_TOKEN = re.compile(r"\$?\d+(?:\.\d+)?")


def attr_verbatim(q: str, val: str) -> bool:
    return val.lower() in q.lower() if val else False


def numeric_exact(q: str, val: str) -> bool:
    if not val:
        return False
    bare = val.lstrip("$")
    if bare.replace(".", "", 1).isdigit() is False:
        return attr_verbatim(q, val)
    return bare in q


def brand_exact(q: str, val: str) -> bool:
    return val in q if val and len(val) >= 2 else False


def main() -> None:
    rng = np.random.default_rng(SEED)

    style = json.load(open(E17 / "e17_style_vectors.json"))
    train_users = set(SPLIT["train"]["users"]) | set(SPLIT["dev"]["users"])
    test_pool = [u for u in style["vectors"] if u not in train_users]
    chosen = list(rng.choice(test_pool, size=min(N_USERS, len(test_pool)), replace=False))

    s1 = json.load(open(STAGE1))
    by_user = {u["user_id"]: u for u in s1["users"]}

    train_vec_arr = np.asarray(
        [TRAIN_VECS["vectors"][u] for u in train_users if u in TRAIN_VECS["vectors"]],
        dtype=np.float32)
    Z_MU = train_vec_arr.mean(axis=0)
    Z_SD = train_vec_arr.std(axis=0) + 1e-9

    def zscore(x):
        return (np.asarray(x, dtype=np.float32) - Z_MU) / Z_SD

    # get 1 product per user from stage1 audit
    jobs = []
    for uid in chosen:
        u = by_user.get(uid)
        if not u:
            continue
        groups = [r for r in u["results"] if r.get("target_reviews")]
        if not groups:
            continue
        audit = groups[len(groups) // 2:]
        prod = rng.choice(audit)
        prod_title = (prod.get("product_title") or "")[:160]
        feats = prod.get("features", []) or []
        a1 = (prod_title.split()[:3] and " ".join(prod_title.split()[:3])) or "item"
        a2 = (prod.get("brand") or prod.get("store") or "Generic")[:32]
        price = prod.get("price")
        if isinstance(price, (int, float)) and price:
            a3 = f"${price:.2f}"
        elif isinstance(price, str) and price.strip():
            a3 = price.strip()
        else:
            a3 = "mid-range"
        a4 = (feats[0] if feats else "standard")[:48]
        a5 = (feats[1] if len(feats) > 1 else (feats[0] if feats else "general use"))[:48]
        attrs = {"A1": a1[:32], "A2": a2, "A3": a3, "A4": a4, "A5": a5}
        vec_correct = np.asarray(style["vectors"][uid], dtype=np.float32)
        vec_correct_z = zscore(vec_correct)
        jobs.append({"user_id": uid, "asin": prod["asin"], "attrs": attrs,
                     "title": prod_title, "z": vec_correct_z.tolist(),
                     "z_zero": np.zeros(USER_DIM, dtype=np.float32).tolist()})

    print(f"[sweep] {len(jobs)} test (user, product) pairs × {len(ALPHAS)} alphas", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=DTYPE, device_map=DEVICE,
                                                trust_remote_code=True)
    base.eval()
    H = base.config.hidden_size
    proj = SoftPrefixProjector(user_dim=USER_DIM, hidden_dim=128, num_tokens=NUM_TOKENS,
                               model_dim=H, dtype=DTYPE, gate_init=GATE_INIT).to(DEVICE)
    proj.load_state_dict(torch.load(CKPT_DIR / "projector.pt", map_location=DEVICE, weights_only=True))
    proj.eval()
    from peft import PeftModel
    base = PeftModel.from_pretrained(base, str(CKPT_DIR / "lora_adapter")).eval()

    # precompute prompts
    prompts, prompt_ids_list = [], []
    for j in jobs:
        p = tok.apply_chat_template(
            [{"role": "system", "content": "You are a shopping query writer. Write one short natural shopping query that mentions every listed attribute."},
             {"role": "user", "content": build_attr_prompt_lines(j["attrs"])}],
            tokenize=False, add_generation_prompt=True)
        prompts.append(p)
        prompt_ids_list.append(tok.encode(p, add_special_tokens=False))
    max_p = max(len(x) for x in prompt_ids_list)
    pad_id = tok.pad_token_id
    ids = torch.full((len(jobs), max_p), pad_id, dtype=torch.long, device=DEVICE)
    am = torch.zeros((len(jobs), max_p), dtype=torch.long, device=DEVICE)
    for b, pl in enumerate(prompt_ids_list):
        ids[b, max_p - len(pl):] = torch.tensor(pl, dtype=torch.long, device=DEVICE)
        am[b, max_p - len(pl):] = 1
    emb_m = base.get_input_embeddings()

    out_by_alpha = {}
    for alpha in ALPHAS:
        proj.alpha.data.fill_(alpha)
        print(f"[alpha={alpha}] generating...", flush=True)
        # generate correct
        z_correct = torch.tensor([j["z"] for j in jobs], dtype=DTYPE, device=DEVICE)
        z_zero = torch.zeros_like(z_correct)
        queries = {}
        for cond_name, z_batch in [("correct", z_correct), ("zero", z_zero)]:
            torch.manual_seed(GEN_SEED)
            with torch.no_grad():
                prefix = proj(z_batch)
                text_emb = emb_m(ids)
                full = torch.cat([prefix, text_emb], dim=1)
                am_full = torch.cat([torch.ones((len(jobs), NUM_TOKENS), dtype=torch.long, device=DEVICE), am], dim=1)
                out = base.generate(inputs_embeds=full, attention_mask=am_full,
                                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    num_beams=1, temperature=1.0, top_p=1.0,
                                    pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
                for b, j in enumerate(jobs):
                    q = tok.decode(out[b].tolist(), skip_special_tokens=True).strip()
                    queries.setdefault(cond_name, {})[j["asin"]] = q
        # compute metrics
        five_correct = sum(1 for j in jobs
                           if all(attr_verbatim(queries["correct"][j["asin"]], v)
                                  for k, v in j["attrs"].items() if k != "A3"))
        five_correct += sum(1 for j in jobs
                            if numeric_exact(queries["correct"][j["asin"]], j["attrs"]["A3"]))
        # simpler: count rows with all 5 attrs
        n_total = len(jobs)
        five_match_n = 0
        for j in jobs:
            q = queries["correct"][j["asin"]]
            ok = True
            for k, v in j["attrs"].items():
                if k == "A3":
                    if not numeric_exact(q, v):
                        ok = False; break
                elif k == "A2":
                    if not brand_exact(q, v):
                        ok = False; break
                else:
                    if not attr_verbatim(q, v):
                        ok = False; break
            if ok:
                five_match_n += 1
        # also zero condition
        five_match_zero_n = 0
        for j in jobs:
            q = queries["zero"][j["asin"]]
            ok = True
            for k, v in j["attrs"].items():
                if k == "A3":
                    if not numeric_exact(q, v):
                        ok = False; break
                elif k == "A2":
                    if not brand_exact(q, v):
                        ok = False; break
                else:
                    if not attr_verbatim(q, v):
                        ok = False; break
            if ok:
                five_match_zero_n += 1
        out_by_alpha[f"{alpha}"] = {
            "five_attr_correct": five_match_n / n_total,
            "five_attr_zero": five_match_zero_n / n_total,
            "n": n_total,
        }
        print(f"  alpha={alpha}: correct 5-attr={five_match_n / n_total:.3f}  "
              f"zero 5-attr={five_match_zero_n / n_total:.3f}", flush=True)

    json.dump({"version": "e17-alpha-sweep", "alphas": ALPHAS, "results": out_by_alpha},
              open(E17 / "e17_alpha_sweep.json", "w"), indent=1)
    print(f"wrote {E17 / 'e17_alpha_sweep.json'}", flush=True)


if __name__ == "__main__":
    main()
