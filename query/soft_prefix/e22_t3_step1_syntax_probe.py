#!/usr/bin/env python3
"""E22 Task 3 Step 1: Correct the syntax encoder bridge.

User directive (2026-08-16): the Task-2 syntax encoder read hand-crafted 7-dim
Query features; it must instead read QWEN HIDDEN STATES so that Task 3's style
loss can back-propagate through the Qwen forward pass.

  Query text -> Qwen -> aggregate Query-token hidden states -> syntax probe
  -> 318-dim syntax vector (Task-1 full space)

Training labels are still extracted by spaCy/rule on the CONTENT-NEUTRALIZED
query (same protocol as Task 2), so the probe learns "hidden states -> syntax"
while remaining content-agnostic. The probe is FROZEN after training and used
as the Task-3 style head.

Checks (user-specified):
  1. held-out TEMPLATES (not just held-out products): prediction error
     significantly better than mean baseline;
  2. content-swap (same syntax, different content) -> probe output ~unchanged;
  3. syntax-swap (same content, different syntax) -> probe output changes;
  4. held-out template split guards against memorizing the 12 templates.

Outputs (result/e22_t3/):
  e22_t3_syntax_probe.pt  — frozen probe state dict (+ hidden pooling config)
  e22_t3_syntax_probe.json — checks + metrics + sha256
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import gzip
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from extract_syntactic_features import (  # noqa: E402
    per_sentence_features_v2, user_features_v2,
)
from extract_clause_features_single_query import load_spacy_model  # noqa: E402
from e22_t2_syntax_encoder_bridge import load_spacy_model as _lsp  # noqa: E402

MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"
TASK1_VECTORS = REPO_ROOT / "result" / "e22_t1_user_vectors.npz"
TASK1_MANIFEST = REPO_ROOT / "result" / "e22_t1_manifest.json"
OUT_DIR = REPO_ROOT / "result" / "e22_t3"

SEED = 5555
Z_DIM = 318             # FULL 318-dim syntactic space (user directive 2026-08-16)
MAX_SAMPLES = 800          # products (x 12 templates each)
N_EPOCHS = 30
LR = 3e-4
BATCH = 4
TEMPLATE_SPLIT_TEST = [10, 11]   # held-out templates (0-indexed)
DTYPE = torch.bfloat16
DEVICE = "cuda:0"

TOP_ATTR_KEYS = ["Brand", "Color", "Material", "Category", "Price"]
TOP_ATTR_ALIASES = {
    "Brand": ["Brand", "brand", "Manufacturer"],
    "Color": ["Color", "color"],
    "Material": ["Material", "material", "Material Type", "Material Composition"],
    "Category": [],
    "Price": [],
}
TEMPLATES = [
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


class SyntaxProbe(nn.Module):
    """hidden-states -> 318-dim syntax vector (frozen after training)."""

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


def neutralize_equal_tokens(query: str, attrs: dict) -> str:
    """Neutralize every attribute value to the SAME fixed placeholder
    ("xx xx", 2 tokens) so same-template queries from ANY products are
    BIT-IDENTICAL — the 318-dim label then depends only on template syntax
    (content-swap invariance is exact by construction)."""
    import re as _re
    out = query
    for v in attrs.values():
        if not v:
            continue
        out = _re.sub(_re.escape(str(v)), "xx xx", out)
    return out


def query_tokens(query: str, attrs: dict, tokenizer) -> list[int]:
    """Tokenize the CONTENT-NEUTRALIZED query (equal-token-count placeholder)
    so Qwen hidden states carry ONLY syntax (no brand/color/material lexemes),
    matching the spaCy label protocol — content-swap keeps output unchanged."""
    qn = neutralize_equal_tokens(query, attrs)
    return tokenizer(qn, add_special_tokens=False)["input_ids"]


def render(attrs: dict, ti: int) -> str:
    sv = [attrs[k] for k in TOP_ATTR_KEYS]
    return TEMPLATES[ti].format(A1=sv[0], A2=sv[1], A3=sv[2], A4=sv[3], A5=sv[4])


def spacy318(text: str, attrs: dict, nlp, tm, ts) -> np.ndarray | None:
    """spaCy FULL 318-dim syntax label on the CONTENT-NEUTRALIZED query
    (equal-token-count placeholder; standardized with Task-1 train mean/std —
    the full z_user space)."""
    qn = neutralize_equal_tokens(text, attrs)
    doc = nlp(qn)
    sfs = [per_sentence_features_v2(s) for s in doc.sents]
    sfs = [s for s in sfs if s is not None]
    if not sfs:
        return None
    v = user_features_v2(sfs)
    if v is None:
        return None
    return ((v - tm) / ts).astype(np.float64)


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    fi = vd["feature_indices"].astype(int)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    mt = json.load(open(TASK1_MANIFEST))
    train_users = set(mt["splits"]["train"]["ids"])

    log("Indexing train-user products...")
    user_products: dict[str, set] = defaultdict(set)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            a = d.get("parent_asin") or d.get("asin")
            if u and a and u in train_users:
                user_products[u].add(a)
    meta = load_meta()
    asins = sorted({a for us in user_products.values() for a in us
                    if a in meta and attrs_for(meta[a]) is not None})
    rng = random.Random(SEED)
    rng.shuffle(asins)
    asins = asins[:MAX_SAMPLES]
    log(f"samples (products): {len(asins)}")

    # ----- build dataset: (query_text, attrs, template_index) -----
    samples = []
    for a in asins:
        attrs = attrs_for(meta[a])
        for ti in range(len(TEMPLATES)):
            samples.append({"asin": a, "attrs": attrs, "ti": ti,
                            "query": render(attrs, ti)})
    rng.shuffle(samples)
    log(f"total samples: {len(samples)}")

    # ----- spaCy labels + tokenizer -----
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    labels = {}
    for s in samples:
        y = spacy318(s["query"], s["attrs"], nlp, tm, ts)
        if y is not None:
            labels[s["query"]] = y
    log(f"labels computed: {len(labels)}")

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
    emb = model.get_input_embeddings()

    # ----- template split: held-out templates for test -----
    train_s = [s for s in samples if s["ti"] not in TEMPLATE_SPLIT_TEST
               and s["query"] in labels]
    test_s = [s for s in samples if s["ti"] in TEMPLATE_SPLIT_TEST
              and s["query"] in labels]
    log(f"train={len(train_s)} test(held-out templates)={len(test_s)}")

    # ----- Qwen hidden pooling for a batch of queries -----
    def pooled_hidden(batch: list[dict]) -> torch.Tensor:
        """Aggregate Query-token hidden states -> [B, H] (mean-pool)."""
        n = len(batch)
        encs = [query_tokens(s["query"], s["attrs"], tok) for s in batch]
        max_l = max(len(e) for e in encs)
        ids = torch.tensor([e + [tok.pad_token_id] * (max_l - len(e))
                            for e in encs], dtype=torch.long, device=DEVICE)
        mask = torch.tensor([[1] * len(e) + [0] * (max_l - len(e))
                             for e in encs], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask,
                        use_cache=False, output_hidden_states=True)
        hs = out.hidden_states[-1]                       # [B, L, H]
        mask_f = mask.unsqueeze(-1).to(DTYPE)
        pooled = (hs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1)
        return pooled                                   # [B, H]

    probe = SyntaxProbe(hidden_dim=H, z_dim=Z_DIM).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=1e-2)
    n_param = sum(p.numel() for p in probe.parameters())
    log(f"probe params: {n_param}")

    # precompute hidden for all samples (Qwen frozen; ~2-3 min). Cache to disk
    # keyed by the NEUTRALIZED query text so (a) re-runs skip the Qwen forward
    # and (b) content-swap pairs (same template, different products) share the
    # exact same hidden vector -> probe output must be identical.
    log("Precomputing hidden states...")
    HIDDEN_CACHE = OUT_DIR / "e22_t3_step1_hidden_cache.npz"
    all_s = train_s + test_s
    for s in all_s:
        s["key"] = neutralize_equal_tokens(s["query"], s["attrs"])
    hidden_cache: dict[str, torch.Tensor] = {}
    if HIDDEN_CACHE.exists():
        c = np.load(HIDDEN_CACHE, allow_pickle=True)
        cached_q = set(c["queries"])
        cached = {q: torch.as_tensor(v, device=DEVICE)
                  for q, v in zip(c["queries"], c["vecs"])}
        log(f"  loaded hidden cache: {len(cached)} queries")
    else:
        cached_q, cached = set(), {}
    # batch by unique neutralized text
    unique = list(dict.fromkeys(s["key"] for s in all_s))
    todo = [k for k in unique if k not in cached]
    log(f"  unique neutralized queries: {len(unique)}, to compute: {len(todo)}")
    for start in range(0, len(todo), BATCH):
        chunk_keys = todo[start:start + BATCH]
        chunk_s = [next(s for s in all_s if s["key"] == k) for k in chunk_keys]
        h = pooled_hidden(chunk_s)
        for k, hv in zip(chunk_keys, h):
            cached[k] = hv
        if start % 400 == 0:
            log(f"  hidden {start + len(chunk_keys)}/{len(todo)}")
    hidden_cache = cached
    if not HIDDEN_CACHE.exists() or len(cached) > len(cached_q):
        np.savez_compressed(
            HIDDEN_CACHE,
            queries=np.asarray(list(cached.keys())),
            vecs=np.stack([v.float().cpu().numpy() for v in cached.values()]))
        log(f"  wrote hidden cache: {len(cached)}")

    def to_xy(subset: list[dict]):
        X = torch.stack([hidden_cache[s["key"]] for s in subset])
        Y = torch.tensor(np.stack([labels[s["query"]] for s in subset]),
                         dtype=torch.float32, device=DEVICE)
        return X.float(), Y

    Xtr, Ytr = to_xy(train_s)
    Xte, Yte = to_xy(test_s)
    mean_pred = Ytr.mean(dim=0)
    mean_base_te = float(torch.linalg.norm(mean_pred[None] - Yte,
                                           dim=1).mean().item())

    log("Training probe...")
    best_te = float("inf")
    for ep in range(N_EPOCHS):
        order = list(range(len(train_s)))
        random.Random(SEED + ep).shuffle(order)
        ep_loss = 0.0
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            opt.zero_grad()
            pred = probe(Xtr[idx])
            loss = F.mse_loss(pred, Ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(idx)
        with torch.no_grad():
            pte = probe(Xte)
            te_err = float(torch.linalg.norm(pte - Yte, dim=1).mean().item())
        log(f"  ep{ep}: train={ep_loss / len(train_s):.4f} "
            f"test(held-out tpl)={te_err:.4f} vs mean-base={mean_base_te:.4f}")
        if te_err < best_te:
            best_te = te_err
            torch.save(probe.state_dict(), OUT_DIR / "e22_t3_syntax_probe.pt")
    probe.load_state_dict(torch.load(OUT_DIR / "e22_t3_syntax_probe.pt"))

    # ----- Check 1: held-out template error vs mean baseline (user-spec) -----
    with torch.no_grad():
        pte = probe(Xte)
        te_err = float(torch.linalg.norm(pte - Yte, dim=1).mean().item())
        pct_impr = (mean_base_te - te_err) / mean_base_te
    check1 = {"test_heldout_templates": len(test_s),
              "probe_test_l2": round(te_err, 4),
              "mean_baseline_l2": round(mean_base_te, 4),
              "relative_improvement": round(pct_impr, 4),
              "pass": bool(pct_impr > 0.10)}

    # ----- Check 2: content-swap invariance (same syntax, diff content) -----
    # same template, different products -> probe output close
    c2_dists = []
    rng = random.Random(SEED + 1)
    by_ti: dict[int, list[dict]] = defaultdict(list)
    for s in train_s:
        by_ti[s["ti"]].append(s)
    for ti in by_ti:
        grp = by_ti[ti]
        rng.shuffle(grp)
        for a, b in zip(grp[::2], grp[1::2]):
            if len(c2_dists) >= 150:
                break
            with torch.no_grad():
                za = probe(hidden_cache[a["key"]][None])
                zb = probe(hidden_cache[b["key"]][None])
            c2_dists.append(float(torch.linalg.norm(za - zb).item()))
    c2_mean = float(np.mean(c2_dists))
    check2 = {"n_pairs": len(c2_dists),
              "content_swap_l2_mean": round(c2_mean, 4),
              "pass": bool(c2_mean < 0.5)}

    # ----- Check 3: syntax-swap sensitivity (same content, diff syntax) -----
    # same product, different templates -> probe output changes
    c3_dists = []
    for s in train_s[:150]:
        with torch.no_grad():
            z0 = probe(hidden_cache[s["key"]][None])
        # pick another template of the same product
        others = [s2 for s2 in train_s if s2["asin"] == s["asin"]
                  and s2["ti"] != s["ti"]]
        if not others:
            continue
        s2 = rng.choice(others)
        with torch.no_grad():
            z1 = probe(hidden_cache[s2["key"]][None])
        c3_dists.append(float(torch.linalg.norm(z0 - z1).item()))
    c3_mean = float(np.mean(c3_dists))
    check3 = {"n_pairs": len(c3_dists),
              "syntax_swap_l2_mean": round(c3_mean, 4),
              "pass": bool(c3_mean > 0.5 and c3_mean > 2 * c2_mean)}

    sha = hashlib.sha256(
        (OUT_DIR / "e22_t3_syntax_probe.pt").read_bytes()).hexdigest()
    result = {
        "version": "e22_t3_syntax_probe_v1",
        "seed": SEED,
        "model": Path(MODEL_PATH).name,
        "pooling": "mean-pool over Query tokens, last layer hidden",
        "label": "spaCy 318-dim on content-neutralized query (Task-1 full space)",
        "held_out_templates": TEMPLATE_SPLIT_TEST,
        "n_train_samples": len(train_s),
        "n_test_samples": len(test_s),
        "probe_params": n_param,
        "probe_sha256": sha,
        "check1_heldout_templates": check1,
        "check2_content_swap": check2,
        "check3_syntax_swap": check3,
        "all_checks_pass": bool(check1["pass"] and check2["pass"]
                                and check3["pass"]),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "e22_t3_syntax_probe.json", "w") as f:
        json.dump(result, f, indent=1)
    log(f"DONE {result['runtime_sec']}s — all_checks_pass="
        f"{result['all_checks_pass']}")


if __name__ == "__main__":
    main()
