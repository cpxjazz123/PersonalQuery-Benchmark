#!/usr/bin/env python3
"""E22 Task 3 (part 1): Train Qwen latent-space injection for Query syntax control.

Issue #22 Task 3 — 注入 Qwen 潜空间并验证 Query 句法控制:
  - projector maps z_user (7-dim) -> soft prefix (K tokens, model_dim), injected
    at positions [0, K) of the Qwen input.
  - training objective (issue):
      (a) Query CONTENT generation loss (CE over templated training queries);
      (b) syntax-encoder STYLE alignment loss (encoder output vs z_user);
      (c) contrastive loss: correct user beats shuffled-user.
  - style probe: a small MLP on pooled Qwen hidden states predicts z_user
    (the Task-2 bridge generalised to Qwen hidden states).
  - inference (part 3): single direct generation, no templates/reranking.

The projector gate is trained from a near-zero init so that at initialization
the prefix is (almost) a no-op; "true-zero" control is an EXPLICIT zero prefix
(bypassing the projector), never projector(0).

Outputs (result/e22_t3/):
  e22_t3_train.json         — training config + loss curve + checkpoint hash
  e22_t3_injector.pt        — projector + style probe state dict
  e22_t3_train_samples.jsonl — (user, asin, attrs, target_query) samples
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from projector import SoftPrefixProjector  # noqa: E402
from copy_aware import (  # noqa: E402
    CopyAwareHead,
    attr_token_spans,
    mixed_logits,
)

# ---------------------------------------------------------------- config
MODEL_PATH = ("/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-"
              "Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"
TASK1_VECTORS = REPO_ROOT / "result" / "e22_t1_user_vectors.npz"
TASK1_MANIFEST = REPO_ROOT / "result" / "e22_t1_manifest.json"
TASK2_TEMPLATES = [
    "Find me a {A1} {A4} in {A2}, made of {A3}, under {A5}.",
    "Can you show me the {A1} {A4} which is {A2}, {A3}, and costs {A5}?",
    "I want a {A1} {A4} with {A2} and {A3}, for {A5}.",
    "For my baby, I need a {A1} {A4} that is {A2}, made of {A3}, and priced at {A5}.",
    "The {A1} {A4} should be {A2}, {A3}, and no more than {A5}.",
    "There is a {A1} {A4} of {A2}, built with {A3}, at {A5}.",
    "If you have a {A1} {A4} in {A2} made from {A3}, I will pay {A5}.",
    "Please show me something {A2} made of {A3}: a {A1} {A4} for {A5}.",
    "It is the {A1} {A4} from {A2} with {A3} that I want, at {A5}.",
    "What {A1} {A4} should I buy in {A2}, with {A3}, under {A5}?",
    "Not just any {A4}: I need a {A1} {A4} in {A2} made of {A3} for {A5}.",
    "The {A1} {A4} — {A2}, {A3} — must not exceed {A5}.",
]
TOP_ATTR_KEYS = ["Brand", "Color", "Material", "Category", "Price"]
TOP_ATTR_ALIASES = {
    "Brand": ["Brand", "brand", "Manufacturer"],
    "Color": ["Color", "color"],
    "Material": ["Material", "material", "Material Type", "Material Composition"],
    "Category": [],
    "Price": [],
}

SEED = 3333
Z_DIM = 318             # FULL 318-dim z_user (user directive 2026-08-16)
NUM_TOKENS = 16              # soft prefix length (8 was too weak to shift syntax)
PROJ_HIDDEN = 128
GATE_INIT = 0.1              # E12 gate; must grow so the prefix affects decoding
LR = 2e-4
N_EPOCHS = 8
N_EPOCHS_PATIENCE = 3       # early stop: 3 epochs without dev improvement
BATCH = 16                   # 2B rows/step; 20GB/80GB at BATCH=6
MAX_LEN = 96
MAX_SAMPLES_PER_USER = 4
TEMPLATES_PER_SAMPLE = 4
WEIGHT_DECAY = 1e-2
# loss weights (issue: content + style + contrastive)
W_STYLE = 3.0
W_CMP = 2.0
CMP_MARGIN = 0.3
W_COPY_POINTER = 0.2      # pointer supervision on copied attribute tokens
OUT_DIR = REPO_ROOT / "result" / "e22_t3"
DTYPE = torch.bfloat16
DEVICE = "cuda:0"

SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query "
    "that mentions every listed attribute of the product."
)


class StyleProbe(nn.Module):
    """FROZEN Step-1 syntax probe: Qwen hidden states -> 318-dim syntax vector.
    Loaded from e22_t3_syntax_probe.pt (trained on neutralized query hidden
    states, labels from spaCy). NOT trained here — Task-3 style loss flows
    through it into the projector."""

    def __init__(self, hidden_dim: int, z_dim: int, dtype=torch.float32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, z_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.to(dtype)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled.float())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_meta() -> dict[str, dict]:
    recs: dict[str, dict] = {}
    with gzip.open(META, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            asin = d.get("parent_asin")
            if not asin:
                continue
            rec = {"details": {}, "category_leaf": None, "price": None}
            det = d.get("details")
            if isinstance(det, dict):
                rec["details"] = det
            cats = d.get("categories")
            if isinstance(cats, list) and cats:
                if isinstance(cats[-1], str):
                    rec["category_leaf"] = cats[-1]
                elif isinstance(cats[0], list) and cats[0]:
                    rec["category_leaf"] = cats[0][-1]
            p = d.get("price")
            if isinstance(p, (int, float)):
                rec["price"] = float(p)
            recs[asin] = rec
    return recs


def attrs_for(rec: dict) -> dict[str, str] | None:
    out: dict[str, str] = {}
    det = rec.get("details", {})
    for canon in TOP_ATTR_KEYS:
        if canon == "Category":
            v = rec.get("category_leaf")
        elif canon == "Price":
            p = rec.get("price")
            v = f"${p:.2f}" if p is not None else None
        else:
            v = None
            for alias in TOP_ATTR_ALIASES[canon]:
                if alias in det and isinstance(det[alias], str) and det[alias].strip():
                    v = det[alias].strip()
                    break
        if v:
            out[canon] = v
    if len(out) < 5:
        return None
    return out


def attr_prompt(attrs: dict[str, str]) -> str:
    lines = ["Product attributes:"]
    for k in TOP_ATTR_KEYS:
        lines.append(f"{k}: {attrs[k]}")
    lines.append("Write a natural shopping query that mentions every attribute.")
    return "\n".join(lines)


def render_templates(attrs: dict[str, str]) -> list[str]:
    sv = [attrs[k] for k in TOP_ATTR_KEYS]
    return [t.format(A1=sv[0], A2=sv[1], A3=sv[2], A4=sv[3], A5=sv[4])
            for t in TASK2_TEMPLATES]


def build_samples() -> tuple[list[dict], list[dict], list[dict]]:
    """(train, dev, test) samples: {user_id, asin, attrs, target_queries}."""
    log("Loading Task 1 vectors + manifest...")
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_all = vd["Z_full"].astype(np.float64)      # [n, 318] FULL standardized
    user_to_z = {u: z_all[i] for i, u in enumerate(user_ids)}
    mt = json.load(open(TASK1_MANIFEST))
    splits = {s: set(mt["splits"][s]["ids"]) for s in ("train", "dev", "test")}

    log("Indexing (user, asin)...")
    user_products: dict[str, set] = defaultdict(set)
    target = set(user_ids)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            a = d.get("parent_asin") or d.get("asin")
            if u and a and u in target:
                user_products[u].add(a)

    meta = load_meta()
    log(f"meta asins: {len(meta)}")

    rng = random.Random(SEED)
    samples: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    users_by_split = {s: [u for u in user_ids if u in splits[s]] for s in splits}
    # For style supervision the target query must be the CHOSEN template (the
    # one whose 7-dim syntax vector is closest to the user's z_user — Task 2
    # bridge logic), chosen PER (user, product), so the prefix learns "this
    # user's syntax direction".
    from e22_t2_syntax_encoder_bridge import neutralize_content, load_spacy_model
    nlp = load_spacy_model()
    from extract_syntactic_features import per_sentence_features_v2, user_features_v2
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)

    # pre-build per-product template renders (12 each), batch-parse once
    prod_attrs: dict[str, dict] = {}
    for u in user_ids:
        for a in user_products.get(u, set()):
            if a in meta and a not in prod_attrs:
                at = attrs_for(meta[a])
                if at is not None:
                    prod_attrs[a] = at
    all_renders: list[str] = []
    render_of: dict[str, tuple[str, int]] = {}
    for a, at in prod_attrs.items():
        sv = [at[k] for k in TOP_ATTR_KEYS]
        for ti, t in enumerate(TASK2_TEMPLATES):
            q = t.format(A1=sv[0], A2=sv[1], A3=sv[2], A4=sv[3], A5=sv[4])
            render_of[q] = (a, ti)
            all_renders.append(q)
    q318_cache: dict[str, np.ndarray] = {}
    Q318_CACHE = OUT_DIR / "e22_t3_template318_cache.npz"
    if Q318_CACHE.exists():
        cc = np.load(Q318_CACHE, allow_pickle=True)
        q318_cache = {q: v for q, v in zip(cc["queries"], cc["vecs"])}
        log(f"  loaded template-318 cache: {len(q318_cache)}")
    neu_texts = [neutralize_content(q, prod_attrs[a])
                 for q, (a, _) in render_of.items()]
    todo_q = [q for q, (a, _) in render_of.items() if q not in q318_cache]
    if todo_q:
        neu_todo = [neutralize_content(q, prod_attrs[a])
                    for q, (a, _) in render_of.items() if q in todo_q]
        neu_docs = list(nlp.pipe(neu_todo, batch_size=512))
        for q, doc in zip(todo_q, neu_docs):
            sfs = [per_sentence_features_v2(s) for s in doc.sents]
            sfs = [s for s in sfs if s is not None]
            if sfs:
                v = user_features_v2(sfs)
                if v is not None:
                    q318_cache[q] = ((v - tm) / ts).astype(np.float64)
        np.savez_compressed(
            Q318_CACHE,
            queries=np.asarray(list(q318_cache.keys())),
            vecs=np.stack(list(q318_cache.values())))
        log(f"  wrote template-318 cache: {len(q318_cache)}")

    chosen_by_user_prod: dict[tuple[str, str], str] = {}
    for split in ("train", "dev", "test"):
        for u in users_by_split[split]:
            z = user_to_z[u]
            cand = [a for a in user_products.get(u, set())
                    if a in prod_attrs]
            if not cand:
                continue
            rng.shuffle(cand)
            cnt = 0
            for asin in cand:
                attrs = prod_attrs[asin]
                # choose the syntax-closest template for THIS (user, product)
                sv = [attrs[k] for k in TOP_ATTR_KEYS]
                best_t, best_d = None, None
                for ti, t in enumerate(TASK2_TEMPLATES):
                    q = t.format(A1=sv[0], A2=sv[1], A3=sv[2], A4=sv[3], A5=sv[4])
                    q7 = q318_cache.get(q)
                    if q7 is None:
                        continue
                    d = float(np.linalg.norm(q7 - z))
                    if best_d is None or d < best_d:
                        best_d, best_t = d, q
                if best_t is None:
                    continue
                samples[split].append({
                    "user_id": u, "asin": asin, "z_user": z.tolist(),
                    "attrs": attrs, "target_queries": [best_t]})
                cnt += 1
                if cnt >= MAX_SAMPLES_PER_USER:
                    break
    for s in samples:
        log(f"  {s}: {len(samples[s])} samples")
    return samples["train"], samples["dev"], samples["test"]


def encode_prompt(tokenizer, attrs: dict[str, str], target: str):
    """Returns (prompt_ids, assistant_ids): token IDs of the system+user turns
    and of the assistant target query, so CE can be restricted to the target."""
    msgs_no_tgt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": attr_prompt(attrs)},
    ]
    prompt = tokenizer.apply_chat_template(
        msgs_no_tgt, tokenize=True, add_generation_prompt=True)
    tgt_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + \
        [tokenizer.eos_token_id]
    return prompt, tgt_ids


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_samples, dev_samples, test_samples = build_samples()

    log(f"Loading model {Path(MODEL_PATH).name}...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    H = model.config.hidden_size
    for p in model.parameters():
        p.requires_grad = False

    proj = SoftPrefixProjector(user_dim=Z_DIM, hidden_dim=PROJ_HIDDEN,
                               num_tokens=NUM_TOKENS, model_dim=H,
                               dtype=DTYPE, gate_init=GATE_INIT).to(DEVICE)
    probe = StyleProbe(hidden_dim=H, z_dim=Z_DIM).to(DEVICE)
    # load the FROZEN Step-1 syntax probe (Qwen hidden -> 318-dim)
    probe_path = OUT_DIR / "e22_t3_syntax_probe.pt"
    probe.load_state_dict(torch.load(probe_path, map_location=DEVICE))
    for p in probe.parameters():
        p.requires_grad = False
    probe.eval()
    log(f"frozen syntax probe loaded from {probe_path.name}")
    copy_head = CopyAwareHead(hidden_dim=H, vocab_size=model.config.vocab_size,
                              dtype=DTYPE).to(DEVICE)
    params = list(proj.parameters()) + list(copy_head.parameters())
    # alpha gets its own (higher) LR group; exclude from the main group
    main_params = [p for p in params if p is not proj.alpha]
    n_param = sum(p.numel() for p in params)
    log(f"trainable: projector {sum(p.numel() for p in proj.parameters())} "
        f"(probe frozen) "
        f"+ copy_head {sum(p.numel() for p in copy_head.parameters())} "
        f"= {n_param}")

    opt = torch.optim.AdamW(
        [{"params": main_params, "lr": LR},
         {"params": [proj.alpha], "lr": LR * 10.0}],
        weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=N_EPOCHS * len(train_samples) // BATCH)

    # deterministic dev subset for early stopping
    dev_data = dev_samples[:64]
    dev_losses: list[float] = []
    best_dev = float("inf")
    best_ep = -1

    emb = model.get_input_embeddings()
    history: dict[str, list[float]] = {"train_loss": [], "dev_loss": []}

    def run_batch(batch: list[dict], train: bool) -> tuple[float, float, float, float, float]:
        """Returns (loss_content, loss_style, loss_cmp, loss_pointer, total).

        ONE Qwen forward over [real; shuffled] concatenated rows (2B samples,
        duplicated text embeddings) — halves the forward cost vs two calls.
        """
        n = len(batch)
        z = torch.tensor(np.stack([b["z_user"] for b in batch]),
                         dtype=torch.float32, device=DEVICE)
        z_shuf = z[torch.randperm(n)]
        # prefixes for real (rows 0..n) and shuffled (rows n..2n)
        z_cat = torch.cat([z, z_shuf], dim=0).to(DTYPE)
        prefix_cat = proj(z_cat)                    # [2B, K, H]
        # build prompts + targets: INPUT = prompt_ids + target_ids (teacher
        # forcing); CE computed on the target segment only.
        encs = []
        for b in batch:
            tgt = random.choice(b["target_queries"])
            encs.append((b, tgt, encode_prompt(tok, b["attrs"], tgt)))
        max_p = max(len(e[2][0]) for e in encs)
        max_t = max(len(e[2][1]) for e in encs)
        ids = []
        attn = []
        tgt_len = []
        for b, tgt, (p_ids, a_ids) in encs:
            seq = p_ids + a_ids
            pad_n = (max_p + max_t) - len(seq)
            ids.append(F.pad(torch.tensor(seq, dtype=torch.long),
                             (0, pad_n), value=tok.pad_token_id))
            attn.append([1] * len(seq) + [0] * pad_n)
            tgt_len.append(len(a_ids))
        ids = torch.stack(ids).to(DEVICE)
        attn = torch.tensor(attn, dtype=torch.long, device=DEVICE)
        max_len = max_p + max_t
        text_emb = emb(ids).to(DTYPE)
        # duplicate text for the concatenated batch
        ids2 = torch.cat([ids, ids], dim=0)
        attn2 = torch.cat([attn, attn], dim=0)
        text_emb2 = torch.cat([text_emb, text_emb], dim=0)
        # positions: prefix 0..K, then prompt+target K..K+max_len
        pos = torch.arange(max_len, device=DEVICE).unsqueeze(0).expand(2 * n, -1) \
            + NUM_TOKENS
        full = torch.cat([prefix_cat, text_emb2], dim=1)
        attn_full = torch.cat([torch.ones(2 * n, NUM_TOKENS, dtype=torch.long,
                                          device=DEVICE), attn2], dim=1)
        pos_full = torch.cat([torch.arange(NUM_TOKENS, device=DEVICE).unsqueeze(0)
                              .expand(2 * n, -1), pos], dim=1)

        with torch.set_grad_enabled(train):
            out = model(inputs_embeds=full, attention_mask=attn_full,
                        position_ids=pos_full, use_cache=False,
                        output_hidden_states=True)
        gen_logits = out.logits                       # [2B, K+max_len, V]
        hs = out.hidden_states[-1]                    # [2B, K+max_len, H]

        # split back
        hsr, hss = hs[:n], hs[n:]
        genr, gens = gen_logits[:n], gen_logits[n:]

        # --- copy head on the REAL rows only (content supervision) ---
        src_offset = NUM_TOKENS
        src_hidden = hsr[:, src_offset:src_offset + max_p, :]
        src_ids = ids                                  # [B, max_len]
        src_attr_mask = torch.zeros(n, max_p, dtype=torch.bool, device=DEVICE)
        for b in range(n):
            p_ids = encs[b][2][0]
            for v in encs[b][0]["attrs"].values():
                v_ids = tok(str(v), add_special_tokens=False)["input_ids"]
                if not v_ids:
                    continue
                for i in range(len(p_ids) - len(v_ids) + 1):
                    if p_ids[i:i + len(v_ids)] == v_ids:
                        for j in range(len(v_ids)):
                            if i + j < max_p:
                                src_attr_mask[b, i + j] = True
                        break
        gen_start = NUM_TOKENS + max_p
        gen_hidden = hsr[:, gen_start:gen_start + max_t - 1, :]
        p_copy, copy_logits = copy_head(gen_hidden, src_hidden,
                                        src_attr_mask, src_ids[:, :max_p])
        tgt_gen_logits = genr[:, gen_start - 1:gen_start - 1 + max_t - 1, :]
        mixed = mixed_logits(tgt_gen_logits, p_copy, copy_logits)
        tgt_labels = ids[:, max_p:max_p + max_t - 1]
        tgt_labels = torch.where(
            attn[:, max_p:max_p + max_t - 1] > 0,
            tgt_labels, torch.full_like(tgt_labels, -100))
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss_content = loss_fct(mixed.reshape(-1, mixed.size(-1)),
                                tgt_labels.reshape(-1))

        # pointer loss (exact span supervision, real rows only)
        ptr_target = torch.zeros(n, max_t - 1, max_p, device=DEVICE)
        ptr_mask = torch.zeros(n, max_t - 1, dtype=torch.bool, device=DEVICE)
        for b in range(n):
            span_by_tok: dict[int, list[int]] = {}
            p_ids = encs[b][2][0]
            for v in encs[b][0]["attrs"].values():
                v_ids = tok(str(v), add_special_tokens=False)["input_ids"]
                if not v_ids:
                    continue
                for i in range(len(p_ids) - len(v_ids) + 1):
                    if p_ids[i:i + len(v_ids)] == v_ids:
                        for j in range(len(v_ids)):
                            span_by_tok[v_ids[j]] = list(range(i, i + len(v_ids)))
                        break
            tgt_ids_b = ids[b, max_p:max_p + max_t - 1].tolist()
            for k, tid in enumerate(tgt_ids_b):
                span = span_by_tok.get(tid)
                if span:
                    for s in span:
                        if s < max_p:
                            ptr_target[b, k, s] = 1.0 / len(span)
                    ptr_mask[b, k] = True
        ptr_mask = ptr_mask & (attn[:, max_p:max_p + max_t - 1] > 0)
        ptr_logits = copy_head.pointer_logits(gen_hidden, src_hidden,
                                              src_attr_mask)
        if ptr_mask.any():
            safe = ptr_logits.clone()
            row_inf = (~torch.isfinite(safe)).all(dim=-1, keepdim=True)
            safe = torch.where(row_inf, torch.zeros_like(safe), safe)
            ptr_log_sm = safe.log_softmax(dim=-1)
            attr_mask_exp = src_attr_mask.unsqueeze(1).expand_as(ptr_log_sm)
            ptr_log_sm = torch.where(attr_mask_exp, ptr_log_sm,
                                     torch.zeros_like(ptr_log_sm))
            masked = torch.where(ptr_mask.unsqueeze(-1),
                                 ptr_log_sm, torch.zeros_like(ptr_log_sm))
            loss_pointer = -(masked * ptr_target).sum() / max(1.0, ptr_mask.sum().item())
        else:
            loss_pointer = torch.tensor(0.0, device=DEVICE)

        # style loss via the FROZEN syntax probe (Qwen hidden -> 318-dim):
        # (a) prefix-segment pooled hidden; (b) target-segment pooled hidden.
        pooled_pre = hsr[:, :NUM_TOKENS, :].mean(dim=1)
        z_pred_pre = probe(pooled_pre)
        loss_style_pre = F.mse_loss(z_pred_pre, z)
        t_start = NUM_TOKENS + max_p
        t_end = min(t_start + max_t, hs.size(1))
        pooled_tgt = hsr[:, t_start:t_end, :].mean(dim=1)
        z_pred_tgt = probe(pooled_tgt)
        loss_style_tgt = F.mse_loss(z_pred_tgt, z)
        loss_style = loss_style_pre + loss_style_tgt

        # contrastive: correct user must beat shuffled user (prefix encoding)
        pooled_s = hss[:, :NUM_TOKENS, :].mean(dim=1)
        z_pred_s = probe(pooled_s)
        d_real = F.pairwise_distance(z_pred_pre, z, p=2)
        d_shuf = F.pairwise_distance(z_pred_s, z, p=2)
        loss_cmp = torch.clamp(d_real - d_shuf + CMP_MARGIN, min=0).mean()

        total = (loss_content + W_STYLE * loss_style + W_CMP * loss_cmp
                 + W_COPY_POINTER * torch.clamp(loss_pointer, max=10.0))
        return (float(loss_content.item()), float(loss_style.item()),
                float(loss_cmp.item()), float(loss_pointer.item()), total)

    log("Training...")
    step = 0
    for ep in range(N_EPOCHS):
        rng = random.Random(SEED + ep)
        order = list(range(len(train_samples)))
        rng.shuffle(order)
        ep_loss = 0.0
        for start in range(0, len(order), BATCH):
            batch = [train_samples[i] for i in order[start:start + BATCH]]
            if len(batch) < 2:
                break
            opt.zero_grad()
            lc, ls, lc2, lp, total = run_batch(batch, train=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            ep_loss += total.item()
            step += 1
            if step % 100 == 0:
                log(f"  ep{ep} step{step}: content={lc:.3f} style={ls:.3f} "
                    f"cmp={lc2:.3f} ptr={lp:.3f} total={total.item():.3f}")
        history["train_loss"].append(round(ep_loss / max(1, len(order) // BATCH), 4))
        # dev
        dev_tot = 0.0
        for start in range(0, len(dev_data), BATCH):
            batch = dev_data[start:start + BATCH]
            with torch.no_grad():
                lc, ls, lc2, lp, total = run_batch(batch, train=False)
            dev_tot += total.item()
        dev_m = dev_tot / max(1, len(dev_data) // BATCH)
        dev_losses.append(round(dev_m, 4))
        history["dev_loss"].append(round(dev_m, 4))
        log(f"  ep{ep}: train={history['train_loss'][-1]} dev={dev_m:.4f}")
        if dev_m < best_dev:
            best_dev = dev_m
            best_ep = ep
            torch.save({"proj": proj.state_dict(),
                        "copy_head": copy_head.state_dict()},
                       OUT_DIR / "e22_t3_injector_best.pt")
        # early stopping: stop if dev did not improve for N_EPOCHS_PATIENCE
        if ep - best_ep >= N_EPOCHS_PATIENCE:
            log(f"  early stop at ep{ep} (no dev improvement for "
                f"{N_EPOCHS_PATIENCE} epochs)")
            break

    torch.save({"proj": proj.state_dict(),
                "copy_head": copy_head.state_dict()},
               OUT_DIR / "e22_t3_injector.pt")
    sha = hashlib.sha256((OUT_DIR / "e22_t3_injector.pt").read_bytes()).hexdigest()
    log(f"injector saved sha256={sha[:16]}")

    with open(OUT_DIR / "e22_t3_train_samples.jsonl", "w") as f:
        for b in train_samples + dev_samples + test_samples:
            f.write(json.dumps({"user_id": b["user_id"], "asin": b["asin"],
                                "attrs": b["attrs"],
                                "target_queries": b["target_queries"]},
                               ensure_ascii=False) + "\n")

    summary = {
        "version": "e22_t3_train_v1",
        "seed": SEED,
        "model": Path(MODEL_PATH).name,
        "model_sha256_short": "n/a",
        "z_dim": Z_DIM, "num_tokens": NUM_TOKENS,
        "proj_hidden": PROJ_HIDDEN, "gate_init": GATE_INIT,
        "lr": LR, "epochs": N_EPOCHS, "batch": BATCH,
        "early_stop_patience": N_EPOCHS_PATIENCE,
        "loss_weights": {"style": W_STYLE, "cmp": W_CMP, "margin": CMP_MARGIN},
        "n_train_samples": len(train_samples),
        "n_dev_samples": len(dev_samples),
        "n_test_samples": len(test_samples),
        "best_dev_loss": best_dev,
        "history": history,
        "injector_sha256": sha,
        "n_trainable_params": n_param,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "e22_t3_train.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"DONE {summary['runtime_sec']}s")


if __name__ == "__main__":
    main()
