#!/usr/bin/env python3
"""E17 Step 4: Direct (no best-of-N) generation on test users.

For each of >=100 test users, sample 3 unseen audit-group products (build A1..A5
from product_title / brand / features / price; A1=type, A2=brand, A3=price,
A4=appearance, A5=use_case). For each (user, product) pair, generate ONE query
under four conditions sharing the same prompt, checkpoint and seed:

    correct       z_u
    shuffled      z_u with a user-level permutation
    global_mean   mean of training-user vectors
    zero          all-zero vector

No re-ranking, no best-of-N, no candidate filtering. Output one JSONL row per
generation so style/content eval can re-derive every metric from raw samples.

Outputs:
  - e17_test_samples.jsonl       one row per (user, product, condition, seed)
  - e17_step4_meta.json          run meta (n_users, n_products, hash of vectors)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from copy_aware import build_attr_prompt_lines  # noqa: E402
from projector import SoftPrefixProjector  # noqa: E402
from user_stat_vector import FEATURES20  # noqa: E402
from extract_clause_features_single_query import load_spacy_model, extract_clause_features  # noqa: E402

BASE = "/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
E17 = REPO_ROOT / "result" / "personal_query" / "e17"
STAGE1 = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / "Baby_Products" / "stage1_filtered_users_reviews.json"
TRAIN_VECS = json.load(open(E17 / "e17_train_vectors.json"))
OUT_JSONL = E17 / "e17_test_samples.jsonl"
OUT_META = E17 / "e17_step4_meta.json"

SEED = 42
N_TEST_USERS = 100          # >= 100 per issue
PRODUCTS_PER_USER = 3       # >= 3 per issue
GEN_SEED = 0
MAX_NEW_TOKENS = 32
BATCH = 16
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
USER_DIM = 30
NUM_TOKENS = 4
GATE_INIT = 0.05

CONDITIONS = ["correct", "shuffled", "global_mean", "zero"]


# ---------------------------------------------------------------------------
# Build A1..A5 from raw product fields
# ---------------------------------------------------------------------------
def build_attrs(prod: dict, nlp=None) -> dict:
    """Heuristic A1=product_type, A2=brand, A3=price, A4=appearance, A5=use_case."""
    title = prod.get("product_title", "") or ""
    features = prod.get("features", []) or []
    brand = prod.get("brand") or prod.get("store") or "Generic"
    price = prod.get("price")
    if isinstance(price, (int, float)) and price:
        a3 = f"${price:.2f}"
    elif isinstance(price, str) and price.strip():
        a3 = price.strip()
    else:
        a3 = "mid-range"
    # product_type: first noun chunk or first 2-3 words of title
    a1 = _first_noun_chunk(title, nlp) or " ".join(title.split()[:3]) or "item"
    a1 = a1[:32]
    # appearance: from features[0] (size/pack/color/...) or "standard"
    a4 = features[0] if features else "standard"
    a4 = re.sub(r"^\W+|\W+$", "", a4)[:48] or "standard"
    # use_case: from features[-1] or second feature, else "general use"
    a5 = (features[1] if len(features) > 1 else (features[0] if features else "general use"))
    a5 = re.sub(r"^\W+|\W+$", "", a5)[:48] or "general use"
    return {"A1": a1, "A2": brand[:32], "A3": a3, "A4": a4, "A5": a5}


def _first_noun_chunk(text: str, nlp) -> str:
    if not text or nlp is None:
        return ""
    try:
        doc = nlp(text[:200])
        for chunk in doc.noun_chunks:
            t = chunk.text.strip()
            if t and len(t) > 1:
                return t
    except Exception:
        return ""
    return ""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)

    print("[load] vectors + style-vec split", flush=True)
    style = json.load(open(E17 / "e17_style_vectors.json"))
    train_split = json.load(open(E17 / "e17_train_dev_split.json"))
    train_users = set(train_split["train"]["users"]) | set(train_split["dev"]["users"])
    test_pool = [u for u in style["vectors"] if u not in train_users]
    print(f"  test pool (style-vec, not in train/dev): {len(test_pool)}", flush=True)
    chosen_users = list(rng.choice(test_pool, size=min(N_TEST_USERS, len(test_pool)), replace=False))

    # build per-user product attrs from stage1 audit groups
    print("[load] stage1 for audit groups", flush=True)
    s1 = json.load(open(STAGE1))
    by_user = {u["user_id"]: u for u in s1["users"]}
    train_asins_in_dataset = {r["asin"] for r in
        json.load(open(REPO_ROOT / "result/personal_query/e14_multiproduct/baby_dataset.json"))}
    nlp = load_spacy_model()
    user_products: dict[str, list[dict]] = {}
    for uid in chosen_users:
        u = by_user.get(uid)
        if not u:
            continue
        groups = [r for r in u["results"] if r.get("target_reviews")]
        # audit = second half (same rule as step 1)
        if not groups:
            continue
        n = len(groups)
        audit = groups[n // 2:]
        # also exclude any asin that the training set saw (issue says "unseen products")
        cands = [r for r in audit if r["asin"] not in train_asins_in_dataset] or audit
        chosen = list(rng.choice(cands, size=min(PRODUCTS_PER_USER, len(cands)), replace=False))
        for prod in chosen:
            attrs = build_attrs(prod, nlp)
            user_products.setdefault(uid, []).append({
                "asin": prod["asin"],
                "title": (prod.get("product_title") or "")[:160],
                "attrs": attrs,
            })
    final_users = [u for u in chosen_users if u in user_products and len(user_products[u]) >= 1]
    print(f"  test users with products: {len(final_users)}", flush=True)

    # global mean (over training-user vectors from step3a, not contract / test)
    # The 400 contract vectors in e17_style_vectors.json are HELD-OUT (test side),
    # so they must NOT contribute to the "global mean" reference. Use the
    # training-side vectors from e17_train_vectors.json (computed in step3a
    # over users with non-target profile reviews).
    train_vec_arr = np.asarray(
        [TRAIN_VECS["vectors"][u] for u in train_users if u in TRAIN_VECS["vectors"]],
        dtype=np.float32)
    if len(train_vec_arr) == 0:
        # fall back to all available training vectors
        train_vec_arr = np.asarray(list(TRAIN_VECS["vectors"].values()), dtype=np.float32)
    global_mean = train_vec_arr.mean(axis=0)
    # CRITICAL: training script z-scored vectors with (x - mu) / sd over training
    # users; inference MUST use the SAME mu/sd or projector receives OOD inputs.
    Z_MU = train_vec_arr.mean(axis=0)
    Z_SD = train_vec_arr.std(axis=0) + 1e-9
    def zscore(x):
        return (np.asarray(x, dtype=np.float32) - Z_MU) / Z_SD
    global_mean_z = zscore(global_mean)  # apply z-score so projector gets train-distribution input
    print(f"  global mean computed over {len(train_vec_arr)} train-user vectors "
          f"(from e17_train_vectors.json); z-scored with train mu/sd", flush=True)

    # build a user-level permutation table (each user gets a fixed shuffle)
    perm_table: dict[str, np.ndarray] = {}
    for uid in final_users:
        perm_table[uid] = rng.permutation(USER_DIM)

    # ---- load model + projector ----
    print("[load] Qwen2.5-1.5B", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=DTYPE, device_map=DEVICE,
                                                trust_remote_code=True)
    base.eval()
    H = base.config.hidden_size
    proj = SoftPrefixProjector(user_dim=USER_DIM, hidden_dim=128, num_tokens=NUM_TOKENS,
                               model_dim=H, dtype=DTYPE, gate_init=GATE_INIT).to(DEVICE)
    proj.eval()

    # ---- direct generation (no best-of-N) ----
    # per-condition vectors and per-row attrs
    jobs: list[dict] = []
    for uid in final_users:
        vec_correct_raw = np.asarray(style["vectors"][uid], dtype=np.float32)
        vec_correct_z = zscore(vec_correct_raw)  # project into train distribution
        vec_shuffled_z = vec_correct_z[perm_table[uid]]
        for prod in user_products[uid]:
            for cond in CONDITIONS:
                if cond == "correct":
                    z = vec_correct_z
                elif cond == "shuffled":
                    z = vec_shuffled_z
                elif cond == "global_mean":
                    z = global_mean_z
                else:
                    z = np.zeros(USER_DIM, dtype=np.float32)
                jobs.append({"user_id": uid, "asin": prod["asin"], "cond": cond,
                             "attrs": prod["attrs"], "title": prod["title"], "z": z.tolist()})
    print(f"[gen] {len(jobs)} generations (users={len(final_users)} × products={PRODUCTS_PER_USER} × conds={len(CONDITIONS)})", flush=True)

    # write JSONL in streaming form
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_empty = 0
    torch.manual_seed(GEN_SEED)
    with open(OUT_JSONL, "w") as fout:
        for i in range(0, len(jobs), BATCH):
            chunk = jobs[i:i + BATCH]
            zs = torch.tensor([j["z"] for j in chunk], dtype=DTYPE, device=DEVICE)
            with torch.no_grad():
                prefix = proj(zs)  # [B, K, H]
                # build prompts
                prompts, prompt_ids_list = [], []
                for j in chunk:
                    p = tok.apply_chat_template(
                        [{"role": "system", "content": "You are a shopping query writer. Write one short natural shopping query that mentions every listed attribute."},
                         {"role": "user", "content": build_attr_prompt_lines(_lower_keys(j["attrs"]))}],
                        tokenize=False, add_generation_prompt=True)
                    prompts.append(p)
                    prompt_ids_list.append(tok.encode(p, add_special_tokens=False))
                max_p = max(len(x) for x in prompt_ids_list)
                pad_id = tok.pad_token_id
                # left-pad for generation
                ids = torch.full((len(chunk), max_p), pad_id, dtype=torch.long, device=DEVICE)
                am = torch.zeros((len(chunk), max_p), dtype=torch.long, device=DEVICE)
                for b, pl in enumerate(prompt_ids_list):
                    ids[b, max_p - len(pl):] = torch.tensor(pl, dtype=torch.long, device=DEVICE)
                    am[b, max_p - len(pl):] = 1
                # build inputs_embeds with soft prefix
                emb_m = base.get_input_embeddings()
                text_emb = emb_m(ids)  # [B, T, H]
                full = torch.cat([prefix, text_emb], dim=1)
                am_full = torch.cat([torch.ones((len(chunk), NUM_TOKENS), dtype=torch.long, device=DEVICE), am], dim=1)
                # generate
                out = base.generate(inputs_embeds=full, attention_mask=am_full,
                                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    num_beams=1, temperature=1.0, top_p=1.0,
                                    pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
                # out has [B, num_new_tokens]; decode each
                for b, j in enumerate(chunk):
                    new_ids = out[b].tolist()
                    text = tok.decode(new_ids, skip_special_tokens=True).strip()
                    if not text:
                        n_empty += 1
                    else:
                        n_ok += 1
                    row = {
                        "user_id": j["user_id"], "asin": j["asin"], "cond": j["cond"],
                        "attrs": j["attrs"], "title": j["title"],
                        "z_hash": hashlib.sha256(bytes(np.asarray(j["z"]).tobytes())).hexdigest()[:16],
                        "seed": GEN_SEED,
                        "query": text,
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i // BATCH) % 5 == 0:
                print(f"  gen {i + len(chunk)}/{len(jobs)}  ok={n_ok} empty={n_empty}", flush=True)
    print(f"[done] wrote {OUT_JSONL}  ok={n_ok} empty={n_empty}", flush=True)

    # meta
    vec_hash = hashlib.sha256(json.dumps(style["vectors"], sort_keys=True).encode()).hexdigest()
    meta = {
        "version": "e17-step4",
        "n_users": len(final_users),
        "n_products_per_user": PRODUCTS_PER_USER,
        "n_conditions": len(CONDITIONS),
        "n_generations": len(jobs),
        "n_ok": n_ok, "n_empty": n_empty,
        "seed": SEED, "gen_seed": GEN_SEED, "batch": BATCH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "vector_store_hash": vec_hash,
        "vector_dim": USER_DIM, "num_tokens": NUM_TOKENS, "gate_init": GATE_INIT,
        "conditions": CONDITIONS,
        "no_best_of_n": True,
        "single_seed_per_condition": True,
    }
    json.dump(meta, open(OUT_META, "w"), indent=1, ensure_ascii=False)
    print(f"[meta] {OUT_META}", flush=True)


def _lower_keys(attrs: dict) -> dict:
    # copy_aware uses sorted by int(k[1:]) — keep the A1..A5 form
    return {k: v for k, v in attrs.items()}


if __name__ == "__main__":
    main()
