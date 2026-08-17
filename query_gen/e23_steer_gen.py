#!/usr/bin/env python3
"""E23 P0 — steering injection experiment.

Question: does adding alpha * s_u (mean_diff style vector, layer 26) to the
generated-token hidden state during decoding push the model's output toward
user u's style?

Protocol (per user u of the 100 extracted style vectors):
  1. decode a FIXED neutral prompt with steering alpha in {0, .5, 1, 2, 4}
     at layer 26 (hook on generated-token positions, left-padded batch)
  2. encode the generated text -> mean-pooled hidden at layer 26 (h_gen)
  3. metrics:
     a. movement = h_gen(alpha) - h_gen(0); cos(movement, s_u)
        -> does steering move the representation ALONG the style vector?
     b. cos(h_gen(alpha), s_u) vs alpha
     c. identity retrieval: rank users by cos(h_gen(alpha), s_v);
        top-1 accuracy of s_u should rise with alpha (chance = 1/100)
     d. distance to user's own sentence mean vs neutral mean in act space
     e. cross-steering control: alpha=2 with ANOTHER user's vector ->
        retrieval should flip to that other user
  4. qualitative examples (alpha=0 vs alpha=2 vs user's real sentence)

Hardcoded config, no CLI args. Run: python query_gen/e23_steer_gen.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
OUT_DIR = REPO_ROOT / "result" / "e23_style_vector"
VEC_NPZ = OUT_DIR / "style_vectors_100u.npz"
HIDDEN_CACHE = OUT_DIR / "hidden_cache.npz"
GEN_JSONL = OUT_DIR / "steer_gen.jsonl"
GEN_JSON = OUT_DIR / "steer_gen.json"

SEED = 7777
MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
DTYPE = torch.bfloat16
DEVICE = "cuda:0"

STEER_LAYER = 26            # 0-indexed; e22_t3 used -2 (== 26)
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]
MAX_NEW = 64
GEN_BATCH = 16
N_CROSS = 20                # users for the cross-steering control
CROSS_ALPHA = 2.0
N_USERS_MAX = 100

PROMPT = ("Write a one-sentence review of the product you bought. "
          "Reply with only the review sentence.")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v = np.load(VEC_NPZ, allow_pickle=True)
    users = [str(u) for u in v["users"]]
    layers = [int(l) for l in v["layers"]]
    li = layers.index(STEER_LAYER)
    # inject the RAW mean_diff vector (norm ~100 at L26), not the unit vector:
    # activation norms are ~460-800, so a unit vector at alpha=1 moves the
    # hidden state by ~0.2% — far too small to change decoding (this is why
    # e22_t3 steering showed chance-level effects). alpha is a multiple of the
    # raw mean_diff norm.
    s_raw = v["mean_diff"][:, li].astype(np.float32)         # [U, H]
    s_unit = v["mean_diff_norm"][:, li].astype(np.float32)   # [U, H] for metrics
    U = len(users)
    log(f"users={U}, layer={STEER_LAYER}, alphas={ALPHAS}, "
        f"mean_diff norm at L26 = {np.linalg.norm(s_raw, axis=1).mean():.1f}")

    # ---- per-user activation means (user sentences vs neutral rewrites) ----
    sentences = [str(s) for s in v["sentences"]]
    n_per_user = [int(n) for n in v["n_per_user"]]
    with open(OUT_DIR / "rewrites.jsonl", "r", encoding="utf-8") as f:
        rewrites = {json.loads(line)["sentence"]: json.loads(line)["rewrite"]
                    for line in f}
    h = np.load(HIDDEN_CACHE, allow_pickle=True)
    hid_map = {str(t): v for t, v in zip(h["texts"], h["vecs"])}
    offs = np.cumsum([0] + n_per_user)
    user_act_mean = np.zeros((U, s_raw.shape[1]), dtype=np.float32)
    neutral_act_mean = np.zeros((U, s_raw.shape[1]), dtype=np.float32)
    for ui in range(U):
        seg = sentences[offs[ui]:offs[ui + 1]]
        ua = np.stack([hid_map[s] for s in seg])
        na = np.stack([hid_map[rewrites[s]] for s in seg])
        user_act_mean[ui] = ua[:, li].mean(axis=0)
        neutral_act_mean[ui] = na[:, li].mean(axis=0)
    log("activation means built")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"        # h[:, -1] is a real token in prefill
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    n_layers = model.config.num_hidden_layers
    log(f"model loaded, {n_layers} layers, steers at layer {STEER_LAYER}")

    # ---- steering hook: add alpha * s to the last position of each row ----
    # hook state (sv_batch, alpha) set per generate call
    hook_state = {"sv": None, "alpha": 0.0}

    def steer_hook(module, args, output):
        h = output[0]
        pos = h.size(1) - 1
        sv = hook_state["sv"]
        alpha = hook_state["alpha"]
        for b in range(h.size(0)):
            if sv[b] is not None:
                h[b, pos] = h[b, pos] + alpha * sv[b]
        return (h,) + output[1:]

    hook = model.model.layers[STEER_LAYER].register_forward_hook(steer_hook)

    @torch.no_grad()
    def gen_batch(prompts: list[str], sv_list, alpha: float) -> list[str]:
        hook_state["sv"] = sv_list
        hook_state["alpha"] = alpha
        encs = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        max_l = max(len(e) for e in encs)
        # MANUAL LEFT-padding: pos = T-1 must be the last REAL token of every
        # row in the prefill forward, so the steering hook hits all samples.
        ids = torch.tensor(
            [[tok.pad_token_id] * (max_l - len(e)) + e for e in encs],
            dtype=torch.long, device=DEVICE)
        attn = torch.tensor([[0] * (max_l - len(e)) + [1] * len(e)
                             for e in encs], dtype=torch.long, device=DEVICE)
        gen = model.generate(
            input_ids=ids, attention_mask=attn, max_new_tokens=MAX_NEW,
            do_sample=False, temperature=1.0,  # greedy: isolate steering
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            use_cache=True)
        outs = []
        for b, e in enumerate(encs):
            row = gen[b, len(e):]
            if tok.eos_token_id in row:
                row = row[:row.tolist().index(tok.eos_token_id)]
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
        return outs

    @torch.no_grad()
    def pooled_hidden(texts: list[str], bs: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=160).to(DEVICE)
            o = model(**enc, use_cache=False, output_hidden_states=True)
            hv = o.hidden_states[STEER_LAYER]
            m = enc["attention_mask"].unsqueeze(-1).to(hv.dtype)
            p = (hv * m).sum(1) / m.sum(1)
            vecs.append(p.float().cpu().numpy())
        return np.concatenate(vecs, axis=0)

    # ---- generation: (user, alpha, target) -> text ----
    existing = {}
    if GEN_JSONL.exists():
        with open(GEN_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                existing[(d["user"], d["alpha"], d["target"])] = d["text"]
    log(f"cached generations: {len(existing)}")

    def need_run(u: str, alpha: float, target: str) -> bool:
        return (u, alpha, target) not in existing

    tasks = []
    for u in users:
        for a in ALPHAS:
            tasks.append((u, a, "self"))
    rng = random.Random(SEED)
    cross_users = rng.sample(users, N_CROSS)
    cross_targets = {}
    for u in cross_users:
        others = [x for x in users if x != u]
        cross_targets[u] = rng.choice(others)
        tasks.append((u, CROSS_ALPHA, "cross"))
    todo = [t for t in tasks if need_run(*t)]
    log(f"tasks total={len(tasks)}, to generate={len(todo)}")

    for start in range(0, len(todo), GEN_BATCH):
        chunk = todo[start:start + GEN_BATCH]
        prompts = [PROMPT] * len(chunk)
        sv_list = []
        for u, a, tg in chunk:
            if a == 0.0:
                sv_list.append(None)
            else:
                ui = users.index(u)
                if tg == "self":
                    sv_list.append(torch.from_numpy(s_raw[ui]).to(DTYPE).to(DEVICE))
                else:
                    vi = users.index(cross_targets[u])
                    sv_list.append(torch.from_numpy(s_raw[vi]).to(DTYPE).to(DEVICE))
        texts = gen_batch(prompts, sv_list, chunk[0][1])
        with open(GEN_JSONL, "a", encoding="utf-8") as f:
            for (u, a, tg), t in zip(chunk, texts):
                f.write(json.dumps({"user": u, "alpha": a, "target": tg,
                                    "text": t}) + "\n")
                existing[(u, a, tg)] = t
        log(f"  generated {min(start + GEN_BATCH, len(todo))}/{len(todo)}")
    hook.remove()

    # ---- encode all generated texts at layer 26 ----
    gen_texts = [existing[(u, a, tg)] for u, a, tg in tasks]
    h_gen = pooled_hidden(gen_texts)                          # [T, H]
    h_gen_map = {}
    for (u, a, tg), hv in zip(tasks, h_gen):
        h_gen_map[(u, a, tg)] = hv
    log("generated texts encoded")

    # ---- metrics ----
    suu = s_unit / np.maximum(np.linalg.norm(s_unit, axis=1, keepdims=True), 1e-9)

    def cos(a, b):
        a = a / max(float(np.linalg.norm(a)), 1e-9)
        b = b / max(float(np.linalg.norm(b)), 1e-9)
        return float(a @ b)

    met = {"layer": STEER_LAYER, "prompt": PROMPT, "max_new": MAX_NEW}
    per_alpha = {}
    for a in ALPHAS:
        if a == 0.0:
            continue
        move_cos = []
        su_cos = []
        n_below0 = 0
        for ui, u in enumerate(users):
            h0 = h_gen_map[(u, 0.0, "self")]
            ha = h_gen_map[(u, a, "self")]
            mv = ha - h0
            c = cos(mv, suu[ui])
            move_cos.append(c)
            su_cos.append(cos(ha, suu[ui]))
            n_below0 += c < 0
        per_alpha[f"alpha{a}"] = {
            "movement_cos_mean": round(float(np.mean(move_cos)), 4),
            "movement_cos_std": round(float(np.std(move_cos)), 4),
            "frac_movement_cos_gt_0": round(
                float(np.mean([c > 0 for c in move_cos])), 4),
            "h_gen_cos_s_u_mean": round(float(np.mean(su_cos)), 4),
            "h_gen_cos_s_u_baseline(alpha0)": round(
                float(np.mean([cos(h_gen_map[(u, 0.0, "self")], suu[ui])
                               for ui, u in enumerate(users)])), 4),
        }
    met["per_alpha"] = per_alpha

    # identity retrieval: rank users by cos(h_gen, s_v)
    retr = {}
    for a in ALPHAS:
        top1 = 0
        top5 = 0
        n = 0
        for ui, u in enumerate(users):
            hg = h_gen_map[(u, a, "self")]
            sims = hg / max(np.linalg.norm(hg), 1e-9) @ suu.T
            order = np.argsort(-sims)
            rank = int(np.where(order == ui)[0][0]) + 1
            top1 += rank == 1
            top5 += rank <= 5
            n += 1
        retr[f"alpha{a}"] = {
            "top1": round(float(top1 / n), 4),
            "top5": round(float(top5 / n), 4),
            "chance": round(1.0 / U, 4),
        }
    met["identity_retrieval"] = retr

    # cross-steering control: does retrieval flip to the injected other user?
    cross = []
    for u in cross_users:
        vi = users.index(cross_targets[u])
        hg = h_gen_map[(u, CROSS_ALPHA, "cross")]
        sims = hg / max(np.linalg.norm(hg), 1e-9) @ suu.T
        rank_v = int(np.where(np.argsort(-sims) == vi)[0][0]) + 1
        rank_u = int(np.where(np.argsort(-sims) == users.index(u))[0][0]) + 1
        cross.append({"user": u, "other": cross_targets[u],
                      "rank_other": rank_v, "rank_self": rank_u})
    met["cross_steering"] = {
        "alpha": CROSS_ALPHA,
        "n": len(cross),
        "other_top1": round(float(np.mean([c["rank_other"] == 1 for c in cross])), 4),
        "other_top5": round(float(np.mean([c["rank_other"] <= 5 for c in cross])), 4),
        "self_rank_mean": round(float(np.mean([c["rank_self"] for c in cross])), 2),
    }

    # distance to user's own activation region
    dist = {}
    for a in ALPHAS:
        d_user = []
        d_neutral = []
        for ui, u in enumerate(users):
            hg = h_gen_map[(u, a, "self")]
            d_user.append(float(np.linalg.norm(hg - user_act_mean[ui])))
            d_neutral.append(float(np.linalg.norm(hg - neutral_act_mean[ui])))
        dist[f"alpha{a}"] = {
            "dist_to_user_mean": round(float(np.mean(d_user)), 2),
            "dist_to_neutral_mean": round(float(np.mean(d_neutral)), 2),
        }
    met["activation_distance"] = dist

    # qualitative examples: first 8 users
    examples = []
    for u in users[:8]:
        ui = users.index(u)
        examples.append({
            "user": u,
            "real_sentence": sentences[offs[ui]],
            "alpha0": existing[(u, 0.0, "self")],
            "alpha1": existing[(u, 1.0, "self")],
            "alpha4": existing[(u, 4.0, "self")],
        })
    met["examples"] = examples

    met["runtime_sec"] = round(time.time() - t0, 1)
    with open(GEN_JSON, "w") as f:
        json.dump(met, f, indent=1)
    log("=== summary ===")
    for a in ALPHAS:
        if a == 0:
            continue
        pa = per_alpha[f"alpha{a}"]
        r = retr[f"alpha{a}"]
        d = dist[f"alpha{a}"]
        log(f"alpha={a}: move_cos={pa['movement_cos_mean']:.3f} "
            f"(frac>0 {pa['frac_movement_cos_gt_0']:.2f}) | "
            f"h_gen~s_u {pa['h_gen_cos_s_u_mean']:.3f} | "
            f"identity top1={r['top1']:.3f} top5={r['top5']:.3f} | "
            f"d_user={d['dist_to_user_mean']:.1f} "
            f"d_neutral={d['dist_to_neutral_mean']:.1f}")
    c = met["cross_steering"]
    log(f"cross (alpha={CROSS_ALPHA}): other_top1={c['other_top1']:.3f} "
        f"other_top5={c['other_top5']:.3f} self_rank_mean={c['self_rank_mean']}")
    log(f"DONE — wrote {GEN_JSON} ({met['runtime_sec']}s)")


if __name__ == "__main__":
    main()
