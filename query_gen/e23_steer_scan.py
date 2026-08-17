#!/usr/bin/env python3
"""E23 P0 — steering injection sweep: layers x alpha.

Fix the two bugs that made the first steer run look like a null result:
  1. MANUAL LEFT-padding in the decode batch (prefill pos = last REAL token)
  2. inject the RAW mean_diff vector (norm ~66-109, ~15-25% of activation
     norm) instead of the unit vector (norm=1 -> ~0.2% perturbation)

Sweep: layers {8,16,20,24,26,27} x alpha {0.25,0.5,1,2,4} on 20 users.
Metrics per (layer, alpha):
  - move_cos = cos(h_gen(l,a) - h_gen(0), s_u)  (steering moves ALONG s_u?)
  - h_gen~s_u cosine
  - text quality: repeated-token fraction (injection too big => gibberish)

Hardcoded config, no CLI args. Run: python query_gen/e23_steer_scan.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
OUT_DIR = REPO_ROOT / "result" / "e23_style_vector"
VEC_NPZ = OUT_DIR / "style_vectors_100u.npz"
HIDDEN_CACHE = OUT_DIR / "hidden_cache.npz"
SCAN_JSONL = OUT_DIR / "steer_scan.jsonl"
SCAN_JSON = OUT_DIR / "steer_scan.json"

SEED = 7777
MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
DTYPE = torch.bfloat16
DEVICE = "cuda:0"

LAYERS_SCAN = [8, 16, 20, 24, 26, 27]
ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]
N_USERS = 20
MAX_NEW = 64
GEN_BATCH = 16

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
    users = [str(u) for u in v["users"]][:N_USERS]
    layers_all = [int(l) for l in v["layers"]]
    s_raw_all = {l: v["mean_diff"][:, layers_all.index(l)].astype(np.float32)
                 for l in LAYERS_SCAN}
    U = len(users)
    log(f"users={U}, layers={LAYERS_SCAN}, alphas={ALPHAS}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

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

    hooks = [model.model.layers[l].register_forward_hook(steer_hook)
             for l in LAYERS_SCAN]

    @torch.no_grad()
    def gen_batch(prompts: list[str], sv_list, alpha: float,
                  layer: int) -> list[str]:
        hook_state["sv"] = sv_list
        hook_state["alpha"] = alpha
        encs = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        max_l = max(len(e) for e in encs)
        # manual LEFT-padding so pos=T-1 is the last REAL token for every row
        ids = torch.tensor(
            [[tok.pad_token_id] * (max_l - len(e)) + e for e in encs],
            dtype=torch.long, device=DEVICE)
        attn = torch.tensor([[0] * (max_l - len(e)) + [1] * len(e)
                             for e in encs], dtype=torch.long, device=DEVICE)
        gen = model.generate(
            input_ids=ids, attention_mask=attn, max_new_tokens=MAX_NEW,
            do_sample=False, temperature=1.0,
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
    def pooled_hidden(texts: list[str], layer: int, bs: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=160).to(DEVICE)
            o = model(**enc, use_cache=False, output_hidden_states=True)
            hv = o.hidden_states[layer]
            m = enc["attention_mask"].unsqueeze(-1).to(hv.dtype)
            p = (hv * m).sum(1) / m.sum(1)
            vecs.append(p.float().cpu().numpy())
        return np.concatenate(vecs, axis=0)

    # ---- tasks: (user, layer, alpha) ----
    existing = {}
    if SCAN_JSONL.exists():
        with open(SCAN_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                existing[(d["user"], d["layer"], d["alpha"])] = d["text"]
    tasks = [(u, l, a) for u in users for l in LAYERS_SCAN for a in ALPHAS]
    tasks += [(u, 0, 0.0) for u in users]     # baseline (no steering)
    todo = [t for t in tasks if t not in existing]
    log(f"tasks={len(tasks)}, cached={len(existing)}, to generate={len(todo)}")

    for start in range(0, len(todo), GEN_BATCH):
        chunk = todo[start:start + GEN_BATCH]
        sv_list = []
        for u, l, a in chunk:
            if a == 0.0 or l == 0:
                sv_list.append(None)
            else:
                sv_list.append(torch.from_numpy(
                    s_raw_all[l][users.index(u)]).to(DTYPE).to(DEVICE))
        texts = gen_batch([PROMPT] * len(chunk), sv_list,
                          chunk[0][2], chunk[0][1])
        with open(SCAN_JSONL, "a", encoding="utf-8") as f:
            for (u, l, a), t in zip(chunk, texts):
                f.write(json.dumps({"user": u, "layer": l, "alpha": a,
                                    "text": t}) + "\n")
                existing[(u, l, a)] = t
        log(f"  generated {min(start + GEN_BATCH, len(todo))}/{len(todo)}")
    for hk in hooks:
        hk.remove()

    # ---- encode all generated texts at every scan layer ----
    h_map: dict[tuple, np.ndarray] = {}
    for l in LAYERS_SCAN:
        texts = [existing[(u, l, a)] for u, a in
                 [(u, a) for u in users for a in ALPHAS]]
        # baseline (no steering) is layer-independent
        base_texts = [existing[(u, 0, 0.0)] for u in users]
        all_texts = texts + base_texts
        hv = pooled_hidden(all_texts, l)
        for i, (u, a) in enumerate([(u, a) for u in users for a in ALPHAS]):
            h_map[(u, l, a)] = hv[i]
        for i, u in enumerate(users):
            h_map[(u, l, 0.0)] = hv[len(texts) + i]
    log("encoded")

    # ---- metrics per (layer, alpha) ----
    def cos(a, b):
        a = a / max(float(np.linalg.norm(a)), 1e-9)
        b = b / max(float(np.linalg.norm(b)), 1e-9)
        return float(a @ b)

    def repeat_frac(t: str) -> float:
        words = t.split()
        if len(words) < 2:
            return 0.0
        bigrams = [f"{w} {words[i+1]}" for i, w in enumerate(words[:-1])]
        return 1.0 - len(set(bigrams)) / max(1, len(bigrams))

    grid = {}
    for l in LAYERS_SCAN:
        su = s_raw_all[l]
        suu = su / np.maximum(np.linalg.norm(su, axis=1, keepdims=True), 1e-9)
        grid[f"layer{l}"] = {}
        for a in ALPHAS:
            move_cos = []
            hcos = []
            reps = []
            for ui, u in enumerate(users):
                h0 = h_map[(u, l, 0.0)]
                ha = h_map[(u, l, a)]
                move_cos.append(cos(ha - h0, suu[ui]))
                hcos.append(cos(ha, suu[ui]))
                reps.append(repeat_frac(existing[(u, l, a)]))
            grid[f"layer{l}"][f"alpha{a}"] = {
                "move_cos": round(float(np.mean(move_cos)), 4),
                "frac_move_cos_gt0": round(float(np.mean([c > 0 for c in move_cos])), 4),
                "h_gen_cos_s_u": round(float(np.mean(hcos)), 4),
                "repeat_frac": round(float(np.mean(reps)), 4),
            }
        # baseline h_gen~s_u at alpha=0
        base = []
        for ui, u in enumerate(users):
            base.append(cos(h_map[(u, l, 0.0)], suu[ui]))
        grid[f"layer{l}"]["baseline_cos"] = round(float(np.mean(base)), 4)

    examples = []
    for u in users[:5]:
        ex = {"user": u, "alpha0": existing[(u, 0, 0.0)]}
        for l in LAYERS_SCAN[:2]:
            ex[f"l{l}_a{ALPHAS[0]}"] = existing[(u, l, ALPHAS[0])]
        ex[f"l{LAYERS_SCAN[-1]}_a{ALPHAS[-1]}"] = existing[
            (u, LAYERS_SCAN[-1], ALPHAS[-1])]
        examples.append(ex)

    result = {
        "version": "e23_steer_scan_v1",
        "seed": SEED,
        "n_users": U,
        "layers": LAYERS_SCAN,
        "alphas": ALPHAS,
        "prompt": PROMPT,
        "grid": grid,
        "examples": examples,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(SCAN_JSON, "w") as f:
        json.dump(result, f, indent=1)
    log("=== grid (move_cos / frac>0 / repeat) ===")
    header = "        " + "".join(f"a{a:<10}" for a in ALPHAS)
    log(header)
    for l in LAYERS_SCAN:
        row = f"L{l:<5} "
        for a in ALPHAS:
            g = grid[f"layer{l}"][f"alpha{a}"]
            row += f"{g['move_cos']:+.3f}/{g['frac_move_cos_gt0']:.2f}/{g['repeat_frac']:.2f} "
        log(row)
    log(f"DONE — wrote {SCAN_JSON} ({result['runtime_sec']}s)")


if __name__ == "__main__":
    main()
