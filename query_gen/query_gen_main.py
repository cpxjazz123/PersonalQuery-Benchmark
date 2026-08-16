#!/usr/bin/env python3
"""query_gen_main — 自包含的 Query 生成 pipeline（E22 Task 3）。

本脚本不 import 任何项目内其他脚本；所有需要的组件（soft-prefix
projector、copy-aware head、混合解码、内容中性化、318 维句法特征）均内联。

流水线（由 MAIN_STAGE 硬编码选择）：
  stage="train"   : 训练 318 维 projector + copy head（需先有 step1 probe 产物）
  stage="generate": 网格生成（长度 x Query 数 x 控制组 x 重复）
  stage="eval"    : 318 维规模网格诊断（可观测性/稳定性/忠实度/可控性/内容）

运行：
  python query_gen_main.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================== 配置
REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
OUT_DIR = REPO_ROOT / "result" / "e22_t3"

MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"
TASK1_VECTORS = REPO_ROOT / "result" / "e22_t1_user_vectors.npz"
TASK1_MANIFEST = REPO_ROOT / "result" / "e22_t1_manifest.json"
PROBE_PT = OUT_DIR / "e22_t3_syntax_probe2.pt"
INJECTOR_PT = OUT_DIR / "e22_t3_injector.pt"
GEN_OUT = OUT_DIR / "e22_t3_grid_generations.jsonl"
EVAL_OUT = OUT_DIR / "e22_t3_grid_eval.json"

MAIN_STAGE = "generate"      # "train" | "generate" | "eval"

SEED = 6666
Z_DIM = 318
NUM_TOKENS = 16
PROJ_HIDDEN = 128
GATE_INIT = 0.1
LR = 2e-4
N_EPOCHS = 8
N_EPOCHS_PATIENCE = 3
BATCH = 2                    # 2B rows/step on 7B model (memory-limited)
MAX_SAMPLES_PER_USER = 4
WEIGHT_DECAY = 1e-2
W_STYLE = 3.0
W_CMP = 2.0
CMP_MARGIN = 0.3
W_COPY_POINTER = 0.2
COMMENT_TARGET_RATIO = 0.0   # FIXED 0: review sentences must NEVER be an LM
                             # generation target (user directive). Comment
                             # text is used ONLY as the style-loss hidden
                             # anchor (prefix+comment -> z_user region).
SPAN_GAP = 8                  # strict-span decoding: max free (connector)
                             # tokens allowed between forced attr spans
MAX_MID_WAIT = 16            # strict-span: max consecutive steps to wait for
                             # the LM to finish a mid-value tail before the
                             # span is force-completed anyway (deadlock guard)
# clean_free decoding: free tokens restricted to these connectors (no
# review-content words can leak); attribute tokens are added dynamically.
CLEAN_CONNECTORS = {
    "a", "an", "the", "and", "or", "but", "with", "for", "in", "on", "of",
    "to", "at", "from", "by", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "should", "would",
    "could", "can", "will", "shall", "may", "might", "must", "not", "no",
    "yes", "so", "if", "then", "than", "as", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "we", "our", "us",
    "you", "your", "i", "my", "me", "he", "she", "his", "her", "one",
    "two", "three", "want", "need", "get", "find", "buy", "purchase",
    "please", "also", "very", "really", "just", "about", "around",
    "under", "over", "less", "more", "most", "least", "good", "great",
    "best", "perfect", "nice", "like", "love", "price", "cost", "size",
    "sizes", "color", "colors", "brand", "style", "material", "item",
    "items", "product", "products", "pack", "set", "count", "quantity",
    "includes", "including", "made", "make", "fits", "fit", "fits",
    "baby", "babies", "toddler", "toddlers", "child", "children", "kids",
    "adult", "adults", "age", "range", "number", "instructions", "care",
    "unisex", "spring", "protection",
    "€", "£", "$",
}
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.92
MAX_NEW_GRID = [16, 32, 64]
MIN_TOKENS_GRID = [0, 20, 40]   # length-bucket floor: >=20 / >=40 words
N_QUERIES_GRID = [2, 4, 8]
N_TEST_USERS = 120
MAX_PRODS_PER_USER = 6
N_REPS = 2
N_BOOTSTRAP = 2000
MIN_USERS = 20
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
_DBG = False  # set True for strict-span debug traces (one-sample runs only)

TOP_ATTR_KEYS = ["Brand", "Color", "Material", "Category", "Price"]
TOP_ATTR_ALIASES = {
    "Brand": ["Brand", "brand", "Manufacturer"],
    "Color": ["Color", "color"],
    "Material": ["Material", "material", "Material Type", "Material Composition"],
    "Category": [],
    "Price": [],
}
# 扩展属性池（有语义，按语料频率排序；用于 N-scan：top-N 属性）
EXT_ATTR_KEYS = [
    "Style", "Age Range (Description)", "Pattern", "Special Feature",
    "Number Of Items", "Target gender", "Size", "Product Care Instructions",
    "Theme", "Unit Count", "Fabric Type", "Shape",
]
EXT_ATTR_ALIASES = {
    "Style": ["Style", "style"],
    "Age Range (Description)": ["Age Range (Description)", "Age Range"],
    "Pattern": ["Pattern", "pattern"],
    "Special Feature": ["Special Feature", "special feature"],
    "Number Of Items": ["Number Of Items", "Number of Items", "Number of items"],
    "Target gender": ["Target gender", "Target Gender"],
    "Size": ["Size", "size"],
    "Product Care Instructions": ["Product Care Instructions", "Care Instructions"],
    "Theme": ["Theme", "theme"],
    "Unit Count": ["Unit Count", "unit count"],
    "Fabric Type": ["Fabric Type", "fabric type"],
    "Shape": ["Shape", "shape", "Item Shape"],
}
SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query "
    "that mentions every listed attribute of the product."
)
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

# ============================================================== 工具


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def neutralize_content(query: str, attrs: dict) -> str:
    """Replace every attribute value with a fixed placeholder (content-free)."""
    out = query
    for v in attrs.values():
        if not v:
            continue
        out = re.sub(re.escape(str(v)), "xx xx", out)
    return out


def skeletonize(text: str, attrs: dict) -> str:
    """Rewrite a review sentence into 'user syntax + product content'.

    Every attribute value found in the review text (case-insensitive) is
    replaced by the canonical attribute value, keeping the sentence's
    clause/punctuation/connector structure intact. Result: a training target
    that carries the USER's syntax but only product-attribute content —
    no review-specific content can leak. Replacement order: longest value
    first so multi-word values are not partially eaten by substrings.
    """
    out = text
    for v in sorted(attrs.values(), key=lambda s: -len(str(s))):
        vv = str(v)
        if not vv:
            continue
        out = re.sub(re.escape(vv), vv, out, flags=re.IGNORECASE)
    return out


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


def attrs_for_n(rec: dict, n: int) -> dict[str, str] | None:
    """Top-N attributes for a product: fixed 5 (Brand/Color/Material/Category/
    Price) + EXT_ATTR_KEYS in frequency order up to n total. Returns None if
    the product lacks any of the required fixed 5 or fewer than n present."""
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
    for canon in EXT_ATTR_KEYS:
        if len(out) >= n:
            break
        v = None
        for alias in EXT_ATTR_ALIASES[canon]:
            if alias in det and isinstance(det[alias], str) and det[alias].strip():
                v = det[alias].strip()
                break
        if v:
            out[canon] = v
    if len(out) < n:
        return None
    return out


def attr_prompt(attrs: dict[str, str]) -> str:
    lines = ["Product attributes:"]
    for k in attrs.keys():
        lines.append(f"{k}: {attrs[k]}")
    lines.append("Write a natural shopping query that mentions every attribute.")
    return "\n".join(lines)


def user_prompt_tokens(attrs: dict[str, str], tokenizer,
                       length_hint: str = "") -> list[int]:
    hint = (f" Write a detailed query of {length_hint} words."
            if length_hint else "")
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": attr_prompt(attrs) + hint}],
        tokenize=True, add_generation_prompt=True)


def spacy318(text: str, attrs: dict, nlp, tm, ts) -> np.ndarray | None:
    """spaCy FULL 318-dim syntax vector on the content-neutralized query."""
    qn = neutralize_content(text, attrs)
    doc = nlp(qn)
    sfs = []
    for sent in doc.sents:
        sf = per_sentence_features_v2(sent)
        if sf is not None:
            sfs.append(sf)
    if not sfs:
        return None
    v = user_features_v2(sfs)
    if v is None:
        return None
    return ((v - tm) / ts).astype(np.float64)


def spacy318_aggregate(query_attrs: list[tuple[str, dict]], nlp, tm, ts
                       ) -> np.ndarray | None:
    """AGGREGATE 318-dim syntax vector over MULTIPLE queries.

    Same protocol as the Task-1 z_user: concatenate the sentence-level
    features of all queries, then run user_features_v2 ONCE (each query
    content-neutralized with its own attrs). This is the faithful
    multi-query aggregate for fidelity/stability comparisons against z_user.
    """
    all_sfs = []
    for text, attrs in query_attrs:
        qn = neutralize_content(text, attrs)
        doc = nlp(qn)
        for sent in doc.sents:
            sf = per_sentence_features_v2(sent)
            if sf is not None:
                all_sfs.append(sf)
    if not all_sfs:
        return None
    v = user_features_v2(all_sfs)
    if v is None:
        return None
    return ((v - tm) / ts).astype(np.float64)


def digit_tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\d+(?:\.\d+)?", text))


def content_exact(attrs: dict, q: str) -> dict:
    ql = q.lower()
    ok = {k: str(v).lower() in ql for k, v in attrs.items()}
    digits_ok = digit_tokens(q) == digit_tokens(
        " ".join(str(v) for v in attrs.values()))
    return {"all_5": all(ok.values()), "digits": digits_ok}


# ============================================================== 内联组件


class SoftPrefixProjector(nn.Module):
    """z_user -> soft-prefix embeddings (E12 gate, near-zero init)."""

    def __init__(self, user_dim, hidden_dim, num_tokens, model_dim,
                 dtype=DTYPE, gate_init=1e-3):
        super().__init__()
        self.user_dim = user_dim
        self.num_tokens = num_tokens
        self.model_dim = model_dim
        self.mlp = nn.Sequential(
            nn.Linear(user_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * model_dim),
        )
        self.layernorm = nn.LayerNorm(model_dim)
        self.alpha = nn.Parameter(torch.tensor(float(gate_init)))
        for name, p in self.named_parameters():
            if p.ndim >= 2 and "layernorm" not in name:
                nn.init.normal_(p, std=0.02)
            elif p.ndim == 1 and "layernorm" not in name:
                nn.init.zeros_(p)
        self.to(dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.mlp(z)
        out = out.view(-1, self.num_tokens, self.model_dim)
        out = self.layernorm(out)
        return self.alpha * out


class CopyAwareHead(nn.Module):
    """Pointer-copy head over input attribute tokens."""

    def __init__(self, hidden_dim, vocab_size, dtype=DTYPE):
        super().__init__()
        self.proj_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                  nn.Linear(64, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.vocab_size = vocab_size
        self.to(dtype)

    def forward(self, hidden, src_hidden, src_attr_mask, src_ids):
        q = self.proj_q(hidden)
        attn = torch.bmm(q, src_hidden.transpose(1, 2))
        attn = attn.masked_fill(src_attr_mask.unsqueeze(1) == 0, float("-inf"))
        safe = attn.clone()
        row_inf = (~torch.isfinite(safe)).all(dim=-1, keepdim=True)
        safe = torch.where(row_inf, torch.zeros_like(safe), safe)
        attn = torch.softmax(safe, dim=-1)
        B, T, S = attn.shape
        BT = B * T
        flat_attn = attn.reshape(BT, S)
        flat_src = src_ids.unsqueeze(1).expand(B, T, S).reshape(BT, S).long()
        copy_flat = torch.zeros(BT, self.vocab_size, device=hidden.device,
                                dtype=hidden.dtype)
        copy_flat.scatter_add_(1, flat_src, flat_attn)
        copy_logits = copy_flat.reshape(B, T, self.vocab_size)
        p_copy = torch.sigmoid(self.gate(hidden))
        return p_copy, copy_logits

    def pointer_logits(self, hidden, src_hidden, src_attr_mask):
        q = self.proj_q(hidden)
        attn = torch.bmm(q, src_hidden.transpose(1, 2))
        attn = attn.masked_fill(src_attr_mask.unsqueeze(1) == 0, float("-inf"))
        return attn


def mixed_logits(gen_logits, p_copy, copy_logits):
    log_copy = torch.log(copy_logits.clamp_min(1e-9))
    log_gen = gen_logits.log_softmax(dim=-1)
    log_mix = torch.logsumexp(
        torch.stack([log_copy + torch.log(p_copy.clamp_min(1e-9)),
                     log_gen + torch.log((1 - p_copy).clamp_min(1e-9))],
                    dim=-1), dim=-1)
    return log_mix


# ============================================================== 解码


@torch.no_grad()
def generate_batch(model, tok, copy_head, prefix_embeds, prompts,
                   attrs_list, max_new, min_tokens=0,
                   strict_spans=False, clean_free=False,
                   steer_vecs=None, steer_layer=-2,
                   steer_alpha=1.0) -> list[str]:
    """Batch KV-cache decoding with copy mixing + CONTENT/LENGTH guards.

    - copy-head mixes pointer-copy probability with the LM distribution
    - EOS is blocked while (a) any attribute value is missing from the output
      OR (b) length < min_tokens; the block is lifted after 64 tokens so a
      stuck sample terminates (records actual length; eval buckets by it)
    - while a value is still missing, its source tokens get a soft boost so
      the model tends to emit them naturally (no token-level forcing, hence
      no dead-loop)
    - strict_spans=True: hard token/span-level content constraint. Every
      attribute value must appear VERBATIM as a contiguous span copied from
      the prompt. While any value is missing, the decoder is allowed at most
      SPAN_GAP free (connector) tokens before the next missing span is force-
      copied; the next span is chosen by the copy-head pointer attention so
      the model still controls the ordering. Guarantees all attrs appear in
      ONE autoregressive pass (no post-hoc append / no candidate rerank).
    """
    bsz = len(prompts)
    emb = model.get_input_embeddings()
    pad_id = tok.pad_token_id or tok.eos_token_id
    encs = [list(p) for p in prompts]
    # ---- PACS activation steering: add alpha*v_style to the hidden state of
    # the newly generated token at steer_layer during every forward pass
    _steer_hook = None
    if steer_vecs is not None:
        n_layers = model.config.num_hidden_layers
        _lid = steer_layer if steer_layer >= 0 else n_layers + steer_layer
        _sv = steer_vecs

        def _hook_fn(module, args, output):
            h = output[0]
            B = h.size(0)
            pos = h.size(1) - 1
            for b in range(B):
                if _sv[b] is None:
                    continue
                v = _sv[b].to(h.device).to(h.dtype)
                h[b, pos] = h[b, pos] + steer_alpha * v
            return (h,) + output[1:]

        _steer_hook = model.model.layers[_lid].register_forward_hook(
            _hook_fn)
    max_p = max(len(e) for e in encs)
    ids = torch.tensor([e + [pad_id] * (max_p - len(e)) for e in encs],
                       dtype=torch.long, device=DEVICE)
    masks = torch.tensor([[1] * len(e) + [0] * (max_p - len(e)) for e in encs],
                         dtype=torch.long, device=DEVICE)
    text_emb = emb(ids).to(DTYPE)
    has_prefix = any(p is not None for p in prefix_embeds)
    if has_prefix:
        prefix = torch.stack(
            [p if p is not None
             else torch.zeros(NUM_TOKENS, model.config.hidden_size,
                              dtype=DTYPE, device=DEVICE)
             for p in prefix_embeds])
        full = torch.cat([prefix, text_emb], dim=1)
        attn = torch.cat([torch.ones(bsz, NUM_TOKENS, dtype=torch.long,
                                     device=DEVICE), masks], dim=1)
        pos = torch.cat([
            torch.arange(NUM_TOKENS, device=DEVICE).unsqueeze(0).expand(bsz, -1),
            torch.arange(max_p, device=DEVICE).unsqueeze(0).expand(bsz, -1)
            + NUM_TOKENS], dim=1)
        src_offset = NUM_TOKENS
    else:
        full = text_emb
        attn = masks
        pos = torch.arange(max_p, device=DEVICE).unsqueeze(0).expand(bsz, -1)
        src_offset = 0
    src_attr_mask = torch.zeros(bsz, max_p, dtype=torch.bool, device=DEVICE)
    use_copy = [a is not None for a in attrs_list]
    for b in range(bsz):
        if not use_copy[b]:
            continue
        p_ids = encs[b]
        for v in attrs_list[b].values():
            v_ids = tok(str(v), add_special_tokens=False)["input_ids"]
            if not v_ids:
                continue
            for i in range(len(p_ids) - len(v_ids) + 1):
                if p_ids[i:i + len(v_ids)] == v_ids:
                    for j in range(len(v_ids)):
                        if i + j < max_p:
                            src_attr_mask[b, i + j] = True
                    break
    # clean_free allowed-token set per sample: attr tokens + connectors +
    # digits/punctuation tokens (so only clean query language can be sampled)
    clean_tokens: list[set[int] | None] = [None] * bsz
    if clean_free:
        connector_ids: set[int] = set()
        for w in CLEAN_CONNECTORS:
            for t in tok(w, add_special_tokens=False)["input_ids"]:
                connector_ids.add(t)
        punct_strs = [",", ".", "!", "?", ";", ":", "'", "-", "&", "%",
                      "(", ")", "[", "]", "/", " "]
        for s in punct_strs:
            for t in tok(s, add_special_tokens=False)["input_ids"]:
                connector_ids.add(t)
        if tok.eos_token_id is not None:
            connector_ids.add(tok.eos_token_id)
        for b in range(bsz):
            if not use_copy[b]:
                continue
            tok_set = set(connector_ids)
            for v in attrs_list[b].values():
                for t in tok(str(v), add_special_tokens=False)["input_ids"]:
                    tok_set.add(t)
            clean_tokens[b] = tok_set
    past = None
    generated: list[list[int]] = [[] for _ in range(bsz)]
    done = [False] * bsz
    next_emb = full
    next_ids = None
    src_hidden_cache = None
    # ---- strict-span tables: standalone tokenization per attribute value ----
    # (NOT prompt-context tokenization: BPE merges differ with surrounding
    # tokens, e.g. "$39.99" inside "Price: $39.99" — forcing must use the
    # value's OWN token ids so the decoded output contains the verbatim text)
    span_tables: list[list[dict]] | None = None
    active_span: list[list[int]] = [[] for _ in range(bsz)]
    since_span: list[int] = [0] * bsz
    mid_wait: list[int] = [0] * bsz
    if strict_spans:
        span_tables = []
        for b in range(bsz):
            tbl = []
            if use_copy[b]:
                for v in attrs_list[b].values():
                    v_ids = tok(str(v), add_special_tokens=False)["input_ids"]
                    if v_ids:
                        tbl.append({"key": str(v), "tokens": list(v_ids)})
            span_tables.append(tbl)
    for step in range(max_new):
        first = past is None
        out = model(inputs_embeds=next_emb if first else None,
                    input_ids=None if first else next_ids,
                    attention_mask=attn, position_ids=pos if first else None,
                    use_cache=True, past_key_values=past,
                    output_hidden_states=True)
        past = out.past_key_values
        if first:
            last_valid = attn.sum(dim=1) - 1
            logits = out.logits[torch.arange(bsz, device=DEVICE), last_valid]
            if any(use_copy) and copy_head is not None:
                src_hidden = out.hidden_states[-1][
                    :, src_offset:src_offset + max_p, :]
                src_hidden_cache = src_hidden
                gen_hidden = out.hidden_states[-1][
                    torch.arange(bsz, device=DEVICE), last_valid].unsqueeze(1)
                p_copy, copy_logits = copy_head(
                    gen_hidden, src_hidden, src_attr_mask, ids)
                logits = mixed_logits(logits.unsqueeze(1), p_copy,
                                      copy_logits)[:, 0]
        else:
            logits = out.logits[:, -1]
            if any(use_copy) and copy_head is not None:
                gen_hidden = out.hidden_states[-1][:, -1:, :]
                p_copy, copy_logits = copy_head(
                    gen_hidden, src_hidden_cache, src_attr_mask, ids)
                logits = mixed_logits(logits.unsqueeze(1), p_copy,
                                      copy_logits)[:, 0]
        logits = logits.float() / max(TEMPERATURE, 1e-4)
        if TOP_K > 0:
            v = torch.topk(logits, min(TOP_K, logits.size(-1)), dim=-1) \
                .values[:, -1].unsqueeze(1)
            logits = torch.where(logits < v,
                                 torch.full_like(logits, float("-inf")), logits)
        if TOP_P < 1.0:
            sorted_l, idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
            mask = cum - F.softmax(sorted_l, dim=-1) > TOP_P
            sorted_l[mask] = float("-inf")
            logits = torch.gather(sorted_l, 1, idx.argsort(dim=-1))
        probs = F.softmax(logits, dim=-1)
        # repetition penalty: discourage immediate token repeats to suppress
        # the "Quaternion Quaternion..." degradation on long generation
        for b in range(bsz):
            if done[b]:
                continue
            if len(generated[b]) >= 4:
                last4 = generated[b][-4:]
                if len(set(last4)) <= 2:
                    for tid in set(last4):
                        probs[b, tid] *= 0.01
        # strict-span forcing: while attrs are missing, force-copy the next
        # missing value's standalone tokens (verbatim text guarantee); allow
        # at most SPAN_GAP free connector tokens between spans. The next span
        # is chosen by the model's own first-token logit preference. Never
        # force while the tail is mid-word/mid-value of a missing attr
        # (avoids splitting a value that the LM is naturally emitting).
        forced: list[int | None] = [None] * bsz
        if strict_spans:
            _space_tok = None
            for b in range(bsz):
                if done[b] or not use_copy[b]:
                    continue
                if active_span[b]:
                    forced[b] = active_span[b].pop(0)
                    continue
                text_b = tok.decode(generated[b],
                                    skip_special_tokens=True).lower()
                missing = [s for s in span_tables[b]
                           if s["key"].lower() not in text_b]
                if missing:
                    tail = text_b.rstrip()
                    tail_w = re.split(r"\W+", tail)[-1] if tail else ""
                    vwords = []
                    for s in missing:
                        vwords += [w for w in re.split(r"\W+", s["key"].lower())
                                   if w]
                    # mid-word: tail word is a strict prefix of a value word
                    mid_word = bool(tail_w) and any(
                        len(w) > len(tail_w) and w.startswith(tail_w)
                        for w in vwords)
                    # mid-value: tail ends with a non-trivial prefix of a
                    # missing value string
                    mid_value = False
                    if not mid_word:
                        for s in missing:
                            vk = s["key"].lower()
                            for k in range(min(4, len(vk)), len(vk) + 1):
                                if tail.endswith(vk[:k]):
                                    mid_value = True
                                    break
                            if mid_value:
                                break
                    if mid_word or mid_value:
                        mid_wait[b] += 1
                        # deadlock guard: if the LM never finishes the value
                        # it started, force-complete the span anyway
                        if mid_wait[b] < MAX_MID_WAIT:
                            since_span[b] = 0
                            if _DBG:
                                print(f"[dbg] step{step} b{b} mid "
                                      f"w={mid_wait[b]} tail={tail[-20:]!r}",
                                      flush=True)
                            continue
                    mid_wait[b] = 0
                    since_span[b] += 1
                    if since_span[b] >= SPAN_GAP:
                        best = None
                        for s in missing:
                            sc = float(probs[b, s["tokens"][0]])
                            if best is None or sc > best[0]:
                                best = (sc, s)
                        s = best[1]
                        if _DBG:
                            print(f"[dbg] step{step} b{b} FORCE "
                                  f"{s['key']!r} sc={best[0]:.2e} "
                                  f"miss={[x['key'] for x in missing]} "
                                  f"tail={tail[-25:]!r}", flush=True)
                        if _space_tok is None:
                            _space_tok = tok(" ", add_special_tokens=False)[
                                "input_ids"]
                        if tail and not tail[-1].isspace() and _space_tok:
                            forced[b] = _space_tok[0]
                            active_span[b] = s["tokens"]
                        else:
                            forced[b] = s["tokens"][0]
                            active_span[b] = s["tokens"][1:]
                        since_span[b] = 0
                else:
                    since_span[b] = 0
        # content/length guard per sample
        for b in range(bsz):
            if done[b] or attrs_list[b] is None:
                continue
            G = generated[b]
            text_b = tok.decode(G, skip_special_tokens=True).lower()
            missing = [v for v in attrs_list[b].values()
                       if str(v).lower() not in text_b]
            all_present = not missing
            if (not all_present or len(G) < min_tokens) and \
                    len(G) < (160 if strict_spans else 96):
                probs[b, tok.eos_token_id] = 0.0
            if missing and not strict_spans:
                for v in missing:
                    for tid in tok(str(v), add_special_tokens=False)["input_ids"]:
                        probs[b, tid] *= 5.0
        # clean_free: mask free-token sampling to connector/punct/attr tokens
        # so review content words can never leak into the generated query
        if clean_free:
            for b in range(bsz):
                if done[b]:
                    continue
                allowed = clean_tokens[b]
                if allowed is None:
                    continue
                mask = torch.zeros_like(probs[b], dtype=torch.bool)
                mask[list(allowed)] = True
                probs[b] = torch.where(mask, probs[b],
                                       torch.zeros_like(probs[b]))
                if _DBG and step < 3:
                    top = torch.topk(probs[b], 8).indices.tolist()
                    print(f"[dbg] clean step{step} b{b} top8="
                          f"{[tok.decode([t]) for t in top]} "
                          f"rowsum={float(probs[b].sum()):.3f}",
                          flush=True)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = probs.sum(dim=-1)
        bad = row_sum <= 0
        if bad.any():
            fb = probs[bad]
            keep = min(1000, fb.size(-1))
            fb[fb < 0] = 0.0
            fb = fb / fb.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            fb[fb == 0] = 1.0 / keep
            probs[bad] = fb
        sample_ids = torch.multinomial(probs, 1).squeeze(1)
        if strict_spans:
            for b in range(bsz):
                if forced[b] is not None:
                    sample_ids[b] = forced[b]
        next_ids = sample_ids.clone()
        for b in range(bsz):
            if done[b]:
                next_ids[b] = tok.eos_token_id
        for b in range(bsz):
            if done[b]:
                continue
            tid = int(next_ids[b])
            generated[b].append(tid)
            if tid == tok.eos_token_id:
                done[b] = True
        if all(done):
            break
        next_emb = emb(next_ids.unsqueeze(1)).to(DTYPE)
        next_ids = next_ids.unsqueeze(1)
        attn = torch.cat([attn, torch.ones(bsz, 1, dtype=attn.dtype,
                                           device=DEVICE)], dim=1)
    if _steer_hook is not None:
        _steer_hook.remove()
    return [tok.decode(g, skip_special_tokens=True).strip() for g in generated]



# ============================================================== 数据


def build_samples(train_users, dev_users, test_users, user_products, meta,
                  user_to_z, user_comments=None):
    """(train, dev, test) samples: user x product x chosen template.

    user_comments: {user_id: [review texts]} — used as STYLE anchors (the
    user's real syntax). Comments are NOT the generation target; they only
    provide the style-loss hidden anchor (see run_batch style path).
    """
    import spacy
    from spacy import load as _spacy_load
    nlp = _spacy_load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    user_comments = user_comments or {}

    prod_attrs: dict[str, dict] = {}
    for u, prods in user_products.items():
        for a in prods:
            if a in meta and a not in prod_attrs:
                at = attrs_for(meta[a])
                if at is not None:
                    prod_attrs[a] = at
    rng = random.Random(SEED)
    samples: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    # template-318 cache: (template_id, asin) -> 318 vec. Reusable across
    # users and models (spaCy features are model-independent).
    T318 = OUT_DIR / "e22_t3_template318_cache.npz"
    t318: dict[str, np.ndarray] = {}
    if T318.exists():
        cc = np.load(T318, allow_pickle=True)
        t318 = {q: v for q, v in zip(cc["queries"], cc["vecs"])}
        log(f"  template318 cache: {len(t318)}")
    for split, users in (("train", train_users), ("dev", dev_users),
                         ("test", test_users)):
        for u in users:
            z = user_to_z[u]
            cand = [a for a in user_products.get(u, set())
                    if a in prod_attrs]
            if not cand:
                continue
            rng.shuffle(cand)
            cnt = 0
            for asin in cand:
                attrs = prod_attrs[asin]
                sv = [attrs[k] for k in TOP_ATTR_KEYS]
                best_q, best_d = None, None
                for t in TEMPLATES:
                    q = t.format(A1=sv[0], A2=sv[1], A3=sv[2], A4=sv[3],
                                 A5=sv[4])
                    if q not in t318:
                        y = spacy318(q, attrs, nlp, tm, ts)
                        if y is not None:
                            t318[q] = y
                    y = t318.get(q)
                    if y is None:
                        continue
                    d = float(np.linalg.norm(y - z))
                    if best_d is None or d < best_d:
                        best_d, best_q = d, q
                if best_q is None:
                    continue
                samples[split].append({
                    "user_id": u, "asin": asin, "z_user": z.tolist(),
                    "attrs": attrs, "target_queries": [best_q],
                    "comments": user_comments.get(u, [])})
                cnt += 1
                if cnt >= MAX_SAMPLES_PER_USER:
                    break
    # merge-update cache (never overwrite whole cache on partial runs)
    np.savez_compressed(
        T318, queries=np.asarray(list(t318.keys())),
        vecs=np.stack(list(t318.values())))
    for s in samples:
        log(f"  {s}: {len(samples[s])} samples")
    return samples["train"], samples["dev"], samples["test"]


# ============================================================== 训练


def encode_prompt(tokenizer, attrs, target):
    msgs_no_tgt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": attr_prompt(attrs)},
    ]
    prompt = tokenizer.apply_chat_template(
        msgs_no_tgt, tokenize=True, add_generation_prompt=True)
    tgt_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + \
        [tokenizer.eos_token_id]
    return prompt, tgt_ids


def train_stage() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_full = vd["Z_full"].astype(np.float64)
    user_to_z = {u: z_full[i] for i, u in enumerate(user_ids)}
    mt = json.load(open(TASK1_MANIFEST))
    splits = {s: set(mt["splits"][s]["ids"]) for s in ("train", "dev", "test")}

    user_products: dict[str, set] = defaultdict(set)
    user_comments: dict[str, list[str]] = defaultdict(list)
    target = set(user_ids)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            a = d.get("parent_asin") or d.get("asin")
            if u and a and u in target:
                user_products[u].add(a)
                t = (d.get("text") or "").strip()
                if t:
                    user_comments[u].append(t)
            if i > 20000000:
                break
    # keep up to N comments per user (for style anchor)
    for u in user_comments:
        user_comments[u] = user_comments[u][:40]
    meta = load_meta()
    log(f"meta asins: {len(meta)}")
    train_samples, dev_samples, test_samples = build_samples(
        splits["train"], splits["dev"], splits["test"], user_products, meta,
        user_to_z, user_comments)

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

    proj = SoftPrefixProjector(user_dim=Z_DIM, hidden_dim=PROJ_HIDDEN,
                               num_tokens=NUM_TOKENS, model_dim=H,
                               dtype=DTYPE, gate_init=GATE_INIT).to(DEVICE)

    class _FrozenProbe(nn.Module):
        def __init__(self, hdim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(hdim, 256), nn.GELU(),
                                     nn.Linear(256, 128), nn.GELU(),
                                     nn.Linear(128, Z_DIM)).to(DEVICE)

        def forward(self, x):
            return self.net(x.float())

    probe = _FrozenProbe(H)
    probe.load_state_dict(torch.load(PROBE_PT, map_location=DEVICE))
    for p in probe.parameters():
        p.requires_grad = False
    probe.eval()
    copy_head = CopyAwareHead(hidden_dim=H, vocab_size=model.config.vocab_size,
                              dtype=DTYPE).to(DEVICE)
    params = list(proj.parameters()) + list(copy_head.parameters())
    main_params = [p for p in params if p is not proj.alpha]
    log(f"trainable: projector+copy_head = "
        f"{sum(p.numel() for p in params)}")

    opt = torch.optim.AdamW(
        [{"params": main_params, "lr": LR},
         {"params": [proj.alpha], "lr": LR * 10.0}],
        weight_decay=WEIGHT_DECAY)
    dev_data = dev_samples[:64]
    best_dev = float("inf")
    best_ep = -1
    history: dict[str, list[float]] = {"train_loss": [], "dev_loss": []}

    def run_batch(batch, train):
        n = len(batch)
        z = torch.tensor(np.stack([b["z_user"] for b in batch]),
                         dtype=torch.float32, device=DEVICE)
        z_shuf = z[torch.randperm(n)]
        z_cat = torch.cat([z, z_shuf], dim=0).to(DTYPE)
        prefix_cat = proj(z_cat)
        encs = []
        for b in batch:
            tgt = random.choice(b["target_queries"])
            # ---- content/style split training ----
            # Content: template target ONLY (attrs shape). Copy head =
            # pointer over the ATTR PROMPT spans — review tokens are never
            # in the copy source. Style: user review sentences enter ONLY
            # the style-loss path below (hidden anchor under prefix(z_user));
            # they are never LM targets (user directive: no review sentence
            # as training objective), so review content cannot leak into
            # generation.
            encs.append((b, tgt, encode_prompt(tok, b["attrs"], tgt)))
        max_p = max(len(e[2][0]) for e in encs)
        max_t = max(len(e[2][1]) for e in encs)
        ids = []
        attn = []
        for b, tgt, (p_ids, a_ids) in encs:
            seq = p_ids + a_ids
            pad_n = (max_p + max_t) - len(seq)
            ids.append(F.pad(torch.tensor(seq, dtype=torch.long),
                             (0, pad_n), value=tok.pad_token_id))
            attn.append([1] * len(seq) + [0] * pad_n)
        ids = torch.stack(ids).to(DEVICE)
        attn = torch.tensor(attn, dtype=torch.long, device=DEVICE)
        max_len = max_p + max_t
        text_emb = emb(ids).to(DTYPE)
        ids2 = torch.cat([ids, ids], dim=0)
        attn2 = torch.cat([attn, attn], dim=0)
        text_emb2 = torch.cat([text_emb, text_emb], dim=0)
        pos = torch.arange(max_len, device=DEVICE).unsqueeze(0) \
            .expand(2 * n, -1) + NUM_TOKENS
        full = torch.cat([prefix_cat, text_emb2], dim=1)
        attn_full = torch.cat([torch.ones(2 * n, NUM_TOKENS, dtype=torch.long,
                                          device=DEVICE), attn2], dim=1)
        pos_full = torch.cat([torch.arange(NUM_TOKENS, device=DEVICE)
                              .unsqueeze(0).expand(2 * n, -1), pos], dim=1)
        with torch.set_grad_enabled(train):
            out = model(inputs_embeds=full, attention_mask=attn_full,
                        position_ids=pos_full, use_cache=False,
                        output_hidden_states=True)
        gen_logits = out.logits
        hs = out.hidden_states[-1]
        hsr, hss = hs[:n], hs[n:]
        genr = gen_logits[:n]

        src_offset = NUM_TOKENS
        src_hidden = hsr[:, src_offset:src_offset + max_p, :]
        src_ids = ids
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
                            span_by_tok[v_ids[j]] = list(
                                range(i, i + len(v_ids)))
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
            loss_pointer = -(masked * ptr_target).sum() / max(
                1.0, ptr_mask.sum().item())
        else:
            loss_pointer = torch.tensor(0.0, device=DEVICE)

        # ---- STYLE path: user COMMENT sentences are the style anchor ----
        # The user's real review sentence (NOT the template) is teacher-forced
        # under the prefix; its pooled hidden must decode to z_user. This
        # teaches "prefix(z_user) + user syntax -> user z region", so at
        # inference the freely generated query lands in the user's syntax
        # area. Content is handled separately by the copy path above.
        # Build comment inputs: prefix + comment tokens (no attr prompt).
        com_texts = []
        com_idx = []
        for b in range(n):
            cmts = batch[b].get("comments") or []
            if cmts:
                com_texts.append(random.choice(cmts))
                com_idx.append(b)
        if com_texts:
            c_ids = [tok(c, add_special_tokens=False)["input_ids"][:64]
                     for c in com_texts]
            c_max = max(len(x) for x in c_ids)
            cids = torch.tensor(
                [x + [tok.pad_token_id] * (c_max - len(x)) for x in c_ids],
                dtype=torch.long, device=DEVICE)
            cmask = torch.tensor(
                [[1] * len(x) + [0] * (c_max - len(x)) for x in c_ids],
                dtype=torch.long, device=DEVICE)
            cemb = emb(cids).to(DTYPE)
            cpos = torch.arange(c_max, device=DEVICE).unsqueeze(0) \
                .expand(len(com_texts), -1) + NUM_TOKENS
            # real-prefix rows and shuffled-prefix rows for these comments
            with torch.no_grad():
                z_com = torch.tensor(
                    np.stack([batch[i]["z_user"] for i in com_idx]),
                    dtype=torch.float32, device=DEVICE)
            pc_real = proj(z_com.to(DTYPE))
            z_com_shuf = z_com[torch.randperm(len(com_idx))]
            pc_shuf = proj(z_com_shuf.to(DTYPE))
            # first pass: real-prefix comments -> style loss (DIFFERENTIABLE:
            # projector must receive gradient through the comment forward)
            full_c = torch.cat([pc_real, cemb], dim=1)
            cmask_f = torch.cat(
                [torch.ones(len(com_texts), NUM_TOKENS,
                            dtype=torch.long, device=DEVICE), cmask], dim=1)
            cpos_f = torch.cat([
                torch.arange(NUM_TOKENS, device=DEVICE).unsqueeze(0)
                .expand(len(com_texts), -1), cpos], dim=1)
            out_c = model(inputs_embeds=full_c, attention_mask=cmask_f,
                          position_ids=cpos_f, use_cache=False,
                          output_hidden_states=True)
            hc = out_c.hidden_states[-1]
            pooled_c = hc[:, :NUM_TOKENS, :].mean(dim=1)
            z_pred_real = probe(pooled_c)
            loss_style = F.mse_loss(z_pred_real, z_com)
            # second pass: shuffled-prefix comments -> contrastive (frozen
            # model forward for the shuffled side is enough for the margin)
            full_s = torch.cat([pc_shuf, cemb], dim=1)
            out_s = model(inputs_embeds=full_s, attention_mask=cmask_f,
                          position_ids=cpos_f, use_cache=False,
                          output_hidden_states=True)
            pooled_s = out_s.hidden_states[-1][:, :NUM_TOKENS, :].mean(dim=1)
            z_pred_shuf = probe(pooled_s)
            d_real = F.pairwise_distance(z_pred_real, z_com, p=2)
            d_shuf = F.pairwise_distance(z_pred_shuf, z_com, p=2)
            loss_cmp = torch.clamp(d_real - d_shuf + CMP_MARGIN, min=0).mean()
        else:
            # fallback: prefix-only style (no comment available)
            pooled_pre = hsr[:, :NUM_TOKENS, :].mean(dim=1)
            z_pred_pre = probe(pooled_pre)
            loss_style = F.mse_loss(z_pred_pre, z)

        total = (loss_content + W_STYLE * loss_style + W_CMP * loss_cmp
                 + W_COPY_POINTER * torch.clamp(loss_pointer, max=10.0))
        return (float(loss_content.item()), float(loss_style.item()),
                float(loss_cmp.item()), float(loss_pointer.item()), total)

    log("Training...")
    step = 0
    for ep in range(N_EPOCHS):
        order = list(range(len(train_samples)))
        random.Random(SEED + ep).shuffle(order)
        ep_loss = 0.0
        for start in range(0, len(order), BATCH):
            batch = [train_samples[i] for i in order[start:start + BATCH]]
            if len(batch) < 2:
                break
            opt.zero_grad()
            lc, ls, lc2, lp, total = run_batch(batch, True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += total.item()
            step += 1
            if step % 200 == 0:
                log(f"  ep{ep} step{step}: content={lc:.3f} style={ls:.3f} "
                    f"cmp={lc2:.3f} total={total.item():.3f}")
        history["train_loss"].append(
            round(ep_loss / max(1, len(order) // BATCH), 4))
        dev_tot = 0.0
        for start in range(0, len(dev_data), BATCH):
            batch = dev_data[start:start + BATCH]
            with torch.no_grad():
                lc, ls, lc2, lp, total = run_batch(batch, False)
            dev_tot += total.item()
        dev_m = dev_tot / max(1, len(dev_data) // BATCH)
        history["dev_loss"].append(round(dev_m, 4))
        log(f"  ep{ep}: train={history['train_loss'][-1]} dev={dev_m:.4f}")
        if dev_m < best_dev:
            best_dev = dev_m
            best_ep = ep
            torch.save({"proj": proj.state_dict(),
                        "copy_head": copy_head.state_dict()},
                       OUT_DIR / "e22_t3_injector_best.pt")
        if ep - best_ep >= N_EPOCHS_PATIENCE:
            log(f"  early stop at ep{ep}")
            break

    torch.save({"proj": proj.state_dict(),
                "copy_head": copy_head.state_dict()}, INJECTOR_PT)
    sha = hashlib.sha256(INJECTOR_PT.read_bytes()).hexdigest()
    log(f"injector saved sha256={sha[:16]}")
    summary = {
        "version": "query_gen_train_v1", "z_dim": Z_DIM,
        "num_tokens": NUM_TOKENS, "epochs": len(history["train_loss"]),
        "best_dev_loss": best_dev, "history": history,
        "injector_sha256": sha,
        "n_train": len(train_samples), "n_dev": len(dev_samples),
    }
    with open(OUT_DIR / "e22_t3_train.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"TRAIN DONE")


# ============================================================== 网格生成


def grid_generate_stage() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_full = vd["Z_full"].astype(np.float64)
    user_to_zfull = {u: z_full[i] for i, u in enumerate(user_ids)}
    mt = json.load(open(TASK1_MANIFEST))
    test_users = sorted(mt["splits"]["test"]["ids"])
    rng = random.Random(SEED)
    rng.shuffle(test_users)
    test_users = test_users[:N_TEST_USERS]
    log(f"test users: {len(test_users)}")

    meta = load_meta()
    user_products: dict[str, set] = defaultdict(set)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            a = d.get("parent_asin") or d.get("asin")
            if u and a and u in test_users:
                user_products[u].add(a)
    cell_map: dict[str, list[dict]] = defaultdict(list)
    for u in test_users:
        cand = [a for a in user_products.get(u, set())
                if a in meta and attrs_for(meta[a]) is not None]
        rng.shuffle(cand)
        for a in cand[:MAX_PRODS_PER_USER]:
            cell_map[u].append({"user_id": u, "asin": a,
                                "attrs": attrs_for(meta[a]),
                                "z_user": user_to_zfull[u].tolist()})
    n_cells = sum(len(v) for v in cell_map.values())
    log(f"cells: {n_cells}")

    import spacy
    from spacy import load as _spacy_load
    nlp = _spacy_load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    vd2 = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd2["train_mean"].astype(np.float64)
    ts = vd2["train_std"].astype(np.float64)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    H = model.config.hidden_size
    ckpt = torch.load(INJECTOR_PT, map_location=DEVICE)
    proj = SoftPrefixProjector(user_dim=Z_DIM, hidden_dim=PROJ_HIDDEN,
                               num_tokens=NUM_TOKENS, model_dim=H,
                               dtype=DTYPE, gate_init=GATE_INIT).to(DEVICE)
    proj.load_state_dict(ckpt["proj"])
    proj.eval()
    copy_head = CopyAwareHead(hidden_dim=H, vocab_size=model.config.vocab_size,
                              dtype=DTYPE).to(DEVICE)
    if "copy_head" in ckpt:
        copy_head.load_state_dict(ckpt["copy_head"])
    copy_head.eval()
    log("injector (318) + copy head loaded")

    jobs = []
    # length buckets: (max_new, min_tokens, length_hint) -> target
    # 10-20 / 21-40 / 41-80 words. min_tokens enforces REAL length (EOS
    # blocked until reached) and the hint steers the model to write longer.
    BUCKETS = [(48, 10, "10-20"),
               (80, 21, "21-40"),
               (120, 41, "41-80")]
    for bucket_id, (mn, mnt, hint) in enumerate(BUCKETS):
        for u in test_users:
            for c in cell_map[u]:
                for rep in range(N_REPS):
                    jobs.append((c, "real-z", bucket_id, rep, mnt, mn, hint))
                    jobs.append((c, "shuffled-z", bucket_id, rep, mnt, mn,
                                 hint))
    log(f"jobs: {len(jobs)} (3 length buckets x 2 controls x 2 reps)")
    records = []
    rng2 = random.Random(SEED + 1)
    with torch.no_grad():
        for start in range(0, len(jobs), BATCH):
            chunk = jobs[start:start + BATCH]
            prompts, prefix_embeds, attrs_list = [], [], []
            mnts = []
            for c, ctrl, bid, rep, mnt, mn, hint in chunk:
                z_real = np.asarray(c["z_user"], dtype=np.float64)
                if ctrl == "real-z":
                    zz = z_real
                else:
                    o = max([x for x in test_users if x != c["user_id"]],
                            key=lambda x: float(
                                np.linalg.norm(user_to_zfull[x] - z_real)))
                    zz = user_to_zfull[o]
                prompts.append(user_prompt_tokens(c["attrs"], tok, hint))
                prefix_embeds.append(proj(
                    torch.tensor(zz, dtype=torch.float32, device=DEVICE)
                    .to(DTYPE).unsqueeze(0))[0])
                attrs_list.append(c["attrs"])
                mnts.append(mnt)
            mn = max(x[5] for x in chunk)
            texts = generate_batch(model, tok, copy_head, prefix_embeds,
                                   prompts, attrs_list, mn,
                                   min_tokens=mnts[0])
            for (c, ctrl, bid, rep, mnt, mn_, hint), txt in zip(chunk, texts):
                n_tok = len(txt.split())
                records.append({"user_id": c["user_id"], "asin": c["asin"],
                                "attrs": c["attrs"], "control": ctrl,
                                "length_bucket": bid, "rep": rep,
                                "min_tokens": mnt, "max_new": mn_,
                                "actual_tokens": n_tok, "query": txt})
            if start % 320 == 0:
                log(f"  generated {start + len(chunk)}/{len(jobs)}")
    with open(GEN_OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"GENERATE DONE: {len(records)} generations")


# ============================================================== 网格评估


def grid_eval_stage() -> None:
    rng = np.random.default_rng(SEED)
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_full = vd["Z_full"].astype(np.float64)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    user_to_zfull = {u: z_full[i] for i, u in enumerate(user_ids)}
    mt = json.load(open(TASK1_MANIFEST))
    test_users = set(mt["splits"]["test"]["ids"])

    import spacy
    from spacy import load as _spacy_load
    nlp = _spacy_load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    gens = [json.loads(l) for l in open(GEN_OUT)]
    log(f"generations: {len(gens)}")
    # group queries by (max_new, control, rep, user) as (query, attrs) pairs
    by_cell_q: dict[tuple, list] = defaultdict(list)
    for g in gens:
        by_cell_q[(g["max_new"], g["control"], g["rep"], g["user_id"])] \
            .append((g["query"], g["attrs"]))
    log(f"cells: {len(by_cell_q)}")

    results: dict[str, dict] = {}
    for mn in sorted({g["max_new"] for g in gens}):
        for nq in N_QUERIES_GRID:
            cell = f"len{mn}_q{nq}"
            users = sorted({g["user_id"] for g in gens if g["max_new"] == mn})
            obs, stab, fid, swap_acc = [], [], [], []
            cont = {"n": 0, "all_5": 0, "digits": 0}
            for u in users:
                real0 = by_cell_q.get((mn, "real-z", 0, u), [])
                real1 = by_cell_q.get((mn, "real-z", 1, u), [])
                if len(real0) < nq:
                    continue
                # AGGREGATE the first nq queries of rep0 (multi-query, one pass)
                agg0 = spacy318_aggregate(real0[:nq], nlp, tm, ts)
                if agg0 is None:
                    continue
                obs.append(float((agg0 != 0).mean()))
                # stability: rep0 aggregate vs rep1 aggregate
                if len(real1) >= nq:
                    agg1 = spacy318_aggregate(real1[:nq], nlp, tm, ts)
                    if agg1 is not None:
                        stab.append(float(np.dot(agg0, agg1) / (
                            np.linalg.norm(agg0) * np.linalg.norm(agg1) + 1e-9)))
                # fidelity: aggregate vs own z, vs mean-other z
                z_own = user_to_zfull[u]
                d_own = float(np.linalg.norm(agg0 - z_own))
                others = np.stack([user_to_zfull[o] for o in test_users
                                   if o != u and o in user_to_zfull])
                d_other = float(
                    np.linalg.norm(agg0[None] - others, axis=1).mean())
                fid.append(d_other - d_own)
                # controllability: shuffled-z aggregate on same user
                shuf0 = by_cell_q.get((mn, "shuffled-z", 0, u), [])
                if shuf0:
                    agg_shuf = spacy318_aggregate(shuf0[:nq], nlp, tm, ts)
                    if agg_shuf is not None:
                        d_shuf = float(np.linalg.norm(agg_shuf - z_own))
                        swap_acc.append(int(d_own < d_shuf))
                for g in gens:
                    if (g["max_new"] == mn and g["control"] == "real-z"
                            and g["rep"] == 0 and g["user_id"] == u):
                        ce = content_exact(g["attrs"], g["query"])
                        cont["n"] += 1
                        cont["all_5"] += int(ce["all_5"])
                        cont["digits"] += int(ce["digits"])
                        break
            if len(users) < MIN_USERS:
                results[cell] = {"n_users": len(users), "run": False}
                continue
            obs = np.array(obs)
            stab = np.array(stab)
            fid = np.array(fid)
            swap_acc = np.array(swap_acc) if swap_acc else np.array([0.0])
            bs = np.array([fid[rng.integers(0, len(fid), len(fid))].mean()
                           for _ in range(N_BOOTSTRAP)])
            results[cell] = {
                "run": True, "n_users": len(users),
                "observable_dims_frac_mean": round(float(obs.mean()), 4),
                "observable_dims_count": round(float(obs.mean() * 318), 1),
                "stability_cosine_mean": round(
                    float(stab.mean()) if len(stab) else float("nan"), 4),
                "fidelity_margin_mean": round(float(fid.mean()), 4),
                "fidelity_margin_95ci": [
                    round(float(np.quantile(bs, 0.025)), 4),
                    round(float(np.quantile(bs, 0.975)), 4)],
                "swap_accuracy": round(float(swap_acc.mean()), 4),
                "content_exact_5": round(cont["all_5"] / max(1, cont["n"]), 4),
                "content_digits": round(cont["digits"] / max(1, cont["n"]), 4),
                "content_n": cont["n"],
            }
            log(f"  {cell}: obs={results[cell]['observable_dims_count']} "
                f"stab={results[cell]['stability_cosine_mean']} "
                f"fid={results[cell]['fidelity_margin_mean']} "
                f"swap={results[cell]['swap_accuracy']} "
                f"cont5={results[cell]['content_exact_5']}")
    with open(EVAL_OUT, "w") as f:
        json.dump({"version": "query_gen_grid_eval_v1", "seed": SEED,
                   "results": results}, f, indent=1)
    log("EVAL DONE")


def _spacy_nlp():
    import spacy
    from spacy import load as _load
    nlp = _load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    return nlp


# ============================================================== 对齐规模扫描


def scale_scan_stage() -> None:
    """For each query-count K, aggregate each user's K queries (same protocol
    as z_user: all sentences pooled, one 318-dim pass) and measure alignment
    with the user's review z_user:
      - fidelity margin (d_other - d_own, bootstrap CI)
      - cosine(agg, z_own) vs mean cosine(agg, z_other)
      - per-dim correlation of agg vs z_own
    Goal: minimal K where the aggregate stably carries user syntax.
    """
    rng = np.random.default_rng(SEED)
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_full = vd["Z_full"].astype(np.float64)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    user_to_zfull = {u: z_full[i] for i, u in enumerate(user_ids)}
    mt = json.load(open(TASK1_MANIFEST))
    test_users = set(mt["splits"]["test"]["ids"])

    import spacy
    from spacy import load as _spacy_load
    nlp = _spacy_load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    gens = [json.loads(l) for l in open(GEN_OUT)]
    log(f"generations: {len(gens)}")
    by_u: dict[str, dict] = defaultdict(dict)   # user -> (control, rep) -> [(q, attrs)]
    for g in gens:
        if g["control"] != "real-z":
            continue
        by_u[g["user_id"]].setdefault(g["rep"], []).append((g["query"], g["attrs"]))
    log(f"users with real-z: {len(by_u)}")

    # also include shuffled-z for controllability at each K
    by_u_sh: dict[str, list] = defaultdict(list)
    for g in gens:
        if g["control"] == "shuffled-z" and g["rep"] == 0:
            by_u_sh[g["user_id"]].append((g["query"], g["attrs"]))

    K_GRID = [2, 4, 6, 8, 12, 16, 24, 32]
    # comparable dims: std large enough that z values are meaningful, and
    # non-zero in user reviews (avoid OOD explosion from tiny-std dims)
    usable = (ts > 0.05) & (np.abs(z_full).max(axis=0) > 0)
    usable_idx = np.where(usable)[0]
    log(f"comparable dims: {len(usable_idx)}/318 "
        f"(std>0.05 and observable in reviews)")
    out: dict[str, dict] = {}
    for K in K_GRID:
        fid, coss, coso, corr, swap, stab = [], [], [], [], [], []
        for u in by_u:
            pool = by_u[u].get(0, [])
            if len(pool) < K:
                continue
            agg = spacy318_aggregate(pool[:K], nlp, tm, ts)
            if agg is None:
                continue
            agg_c = agg[usable_idx]
            z_own = user_to_zfull[u][usable_idx]
            d_own = float(np.linalg.norm(agg_c - z_own))
            others = np.stack([user_to_zfull[o][usable_idx] for o in test_users
                               if o != u and o in user_to_zfull])
            d_other = float(np.linalg.norm(agg_c[None] - others, axis=1).mean())
            fid.append(d_other - d_own)
            c_own = float(np.dot(agg_c, z_own) / (
                np.linalg.norm(agg_c) * np.linalg.norm(z_own) + 1e-9))
            c_other = float(np.mean([
                np.dot(agg_c, z) / (np.linalg.norm(agg_c) * np.linalg.norm(z) + 1e-9)
                for z in others]))
            coss.append(c_own)
            coso.append(c_other)
            nz = (agg_c != 0) & (z_own != 0)
            if nz.sum() > 5:
                corr.append(float(np.corrcoef(agg_c[nz], z_own[nz])[0, 1]))
            pool1 = by_u[u].get(1, [])
            if len(pool1) >= K:
                agg1 = spacy318_aggregate(pool1[:K], nlp, tm, ts)
                if agg1 is not None:
                    agg1_c = agg1[usable_idx]
                    stab.append(float(np.dot(agg_c, agg1_c) / (
                        np.linalg.norm(agg_c) * np.linalg.norm(agg1_c) + 1e-9)))
            shuf = by_u_sh.get(u, [])
            if shuf:
                agg_sh = spacy318_aggregate(shuf[:K], nlp, tm, ts)
                if agg_sh is not None:
                    d_sh = float(np.linalg.norm(agg_sh[usable_idx] - z_own))
                    swap.append(int(d_own < d_sh))
        if len(fid) < MIN_USERS:
            out[f"K{K}"] = {"n_users": len(fid), "run": False}
            continue
        fid = np.array(fid)
        bs = np.array([fid[rng.integers(0, len(fid), len(fid))].mean()
                       for _ in range(N_BOOTSTRAP)])
        out[f"K{K}"] = {
            "run": True, "n_users": len(fid),
            "comparable_dims": len(usable_idx),
            "fidelity_margin_mean": round(float(fid.mean()), 4),
            "fidelity_margin_95ci": [
                round(float(np.quantile(bs, 0.025)), 4),
                round(float(np.quantile(bs, 0.975)), 4)],
            "cosine_own_mean": round(float(np.mean(coss)), 4),
            "cosine_other_mean": round(float(np.mean(coso)), 4),
            "cosine_gap": round(float(np.mean(coss) - np.mean(coso)), 4),
            "per_dim_corr_mean": round(
                float(np.mean(corr)) if corr else float("nan"), 4),
            "stability_cosine": round(
                float(np.mean(stab)) if stab else float("nan"), 4),
            "swap_accuracy": round(
                float(np.mean(swap)) if swap else float("nan"), 4),
        }
        log(f"  K{K}: n={out[f'K{K}']['n_users']} "
            f"fid={out[f'K{K}']['fidelity_margin_mean']} "
            f"cos_gap={out[f'K{K}']['cosine_gap']} "
            f"corr={out[f'K{K}']['per_dim_corr_mean']} "
            f"swap={out[f'K{K}']['swap_accuracy']} "
            f"stab={out[f'K{K}']['stability_cosine']}")
    with open(OUT_DIR / "e22_t3_scale_scan.json", "w") as f:
        json.dump({"version": "query_gen_scale_v1", "seed": SEED,
                   "results": out}, f, indent=1)
    log("SCALE DONE")


# ============================================================== 维度数量扫描


def dimscan_stage() -> None:
    """Dimension-count x query-count scan.

    User directive: test DIFFERENT dimension sets (varying dimension count
    and selection strategy) to find the configuration where aggregated
    generated queries best align with the user's review z_user.

    Dimension selection strategies:
      S1 std-topN   : top-N dims by review std (ts)
      S2 nz-topN    : top-N dims by review non-zero fraction
      S3 task1-7    : Task-1 Query-compatible 7 dims (fixed reference)
      S4 obs-query  : top-N dims by observability in GENERATED queries
      S5 balanced   : std * query-obs product, top-N
    For each (strategy, N, K): fidelity margin, cosine gap, corr, swap, stab.
    """
    rng = np.random.default_rng(SEED)
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_full = vd["Z_full"].astype(np.float64)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    user_to_zfull = {u: z_full[i] for i, u in enumerate(user_ids)}
    mt = json.load(open(TASK1_MANIFEST))
    test_users = set(mt["splits"]["test"]["ids"])
    task1_idx = vd["feature_indices"].astype(int)

    import spacy
    from spacy import load as _spacy_load
    nlp = _spacy_load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    gens = [json.loads(l) for l in open(GEN_OUT)]
    log(f"generations: {len(gens)}")
    by_u: dict[str, dict] = defaultdict(dict)
    for g in gens:
        if g["control"] != "real-z":
            continue
        by_u[g["user_id"]].setdefault(g["rep"], []).append((g["query"], g["attrs"]))
    by_u_sh: dict[str, list] = defaultdict(list)
    for g in gens:
        if g["control"] == "shuffled-z" and g["rep"] == 0:
            by_u_sh[g["user_id"]].append((g["query"], g["attrs"]))
    log(f"users with real-z: {len(by_u)}")

    # observability of each dim in GENERATED queries (rep0 aggregates)
    obs_count = np.zeros(318)
    n_obs_user = 0
    for u in by_u:
        pool = by_u[u].get(0, [])
        if len(pool) < 2:
            continue
        agg = spacy318_aggregate(pool[:4], nlp, tm, ts)
        if agg is None:
            continue
        obs_count += (agg != 0).astype(float)
        n_obs_user += 1
    query_obs = obs_count / max(1, n_obs_user)
    review_nz = (z_full != 0).mean(axis=0)
    log(f"query-observable dims (frac>0.2): {(query_obs > 0.2).sum()}/318")

    # strategies: name -> dim index list (ordered by rank). Strategies that
    # rank by REVIEW-side statistics must still restrict to dims OBSERVABLE in
    # generated queries (frac>0.05), otherwise agg is all-zero -> NaN metrics.
    obs_ok = query_obs > 0.05
    strategies: dict[str, list[int]] = {
        "S1_std_topN": [int(i) for i in np.argsort(-ts) if obs_ok[i]],
        "S2_nz_topN": [int(i) for i in np.argsort(-review_nz) if obs_ok[i]],
        "S3_task1_7": [int(i) for i in task1_idx if obs_ok[i]]
        if any(obs_ok[i] for i in task1_idx) else list(task1_idx),
        "S4_queryobs_topN": list(np.argsort(-query_obs)),
        "S5_stdxqobs_topN": [int(i) for i in np.argsort(-(ts * query_obs))
                             if obs_ok[i]],
    }
    N_GRID = [5, 7, 10, 15, 20, 30, 50, 100, 200, 318]
    K_GRID = [2, 4, 8]
    out: dict[str, dict] = {}
    for strat, order in strategies.items():
        for N in N_GRID:
            if N > len(order):
                continue
            dims = np.asarray(order[:N], dtype=int)
            for K in K_GRID:
                key = f"{strat}_N{N}_K{K}"
                fid, coss, coso, corr, swap, stab = [], [], [], [], [], []
                for u in by_u:
                    pool = by_u[u].get(0, [])
                    if len(pool) < K:
                        continue
                    agg = spacy318_aggregate(pool[:K], nlp, tm, ts)
                    if agg is None:
                        continue
                    agg_c = agg[dims]
                    if not np.any(agg_c != 0):
                        continue
                    z_own = user_to_zfull[u][dims]
                    d_own = float(np.linalg.norm(agg_c - z_own))
                    others = np.stack([user_to_zfull[o][dims]
                                       for o in test_users
                                       if o != u and o in user_to_zfull])
                    d_other = float(np.linalg.norm(
                        agg_c[None] - others, axis=1).mean())
                    fid.append(d_other - d_own)
                    c_own = float(np.dot(agg_c, z_own) / (
                        np.linalg.norm(agg_c) * np.linalg.norm(z_own) + 1e-9))
                    c_other = float(np.mean([
                        np.dot(agg_c, z) / (
                            np.linalg.norm(agg_c) * np.linalg.norm(z) + 1e-9)
                        for z in others]))
                    coss.append(c_own)
                    coso.append(c_other)
                    nz = (agg_c != 0) & (z_own != 0)
                    if nz.sum() > 3:
                        corr.append(float(np.corrcoef(
                            agg_c[nz], z_own[nz])[0, 1]))
                    pool1 = by_u[u].get(1, [])
                    if len(pool1) >= K:
                        agg1 = spacy318_aggregate(pool1[:K], nlp, tm, ts)
                        if agg1 is not None:
                            agg1_c = agg1[dims]
                            stab.append(float(np.dot(agg_c, agg1_c) / (
                                np.linalg.norm(agg_c)
                                * np.linalg.norm(agg1_c) + 1e-9)))
                    shuf = by_u_sh.get(u, [])
                    if shuf:
                        agg_sh = spacy318_aggregate(shuf[:K], nlp, tm, ts)
                        if agg_sh is not None:
                            d_sh = float(np.linalg.norm(
                                agg_sh[dims] - z_own))
                            swap.append(int(d_own < d_sh))
                if len(fid) < MIN_USERS:
                    out[key] = {"n_users": len(fid), "run": False}
                    continue
                fid = np.array(fid)
                bs = np.array([fid[rng.integers(0, len(fid), len(fid))].mean()
                               for _ in range(N_BOOTSTRAP)])
                out[key] = {
                    "run": True, "n_users": len(fid), "n_dims": len(dims),
                    "fidelity_margin_mean": round(float(fid.mean()), 4),
                    "fidelity_margin_95ci": [
                        round(float(np.quantile(bs, 0.025)), 4),
                        round(float(np.quantile(bs, 0.975)), 4)],
                    "cosine_own_mean": round(float(np.mean(coss)), 4),
                    "cosine_other_mean": round(float(np.mean(coso)), 4),
                    "cosine_gap": round(float(np.mean(coss) - np.mean(coso)), 4),
                    "per_dim_corr_mean": round(
                        float(np.mean(corr)) if corr else float("nan"), 4),
                    "stability_cosine": round(
                        float(np.mean(stab)) if stab else float("nan"), 4),
                    "swap_accuracy": round(
                        float(np.mean(swap)) if swap else float("nan"), 4),
                }
                log(f"  {key}: fid={out[key]['fidelity_margin_mean']} "
                    f"cos_gap={out[key]['cosine_gap']} "
                    f"corr={out[key]['per_dim_corr_mean']} "
                    f"swap={out[key]['swap_accuracy']} "
                    f"stab={out[key]['stability_cosine']}")
    with open(OUT_DIR / "e22_t3_dimscan.json", "w") as f:
        json.dump({"version": "query_gen_dimscan_v1", "seed": SEED,
                   "results": out}, f, indent=1)
    log("DIMSCAN DONE")


# ============================================================== 属性数量扫描


def nscan_stage() -> None:
    """Attribute-count scan: for each N (top-N attributes), generate queries
    and measure the maximal NATURAL length achievable (no degeneracy, all N
    attributes present). Question: does adding attributes allow longer,
    still-natural queries?
    """
    random.seed(SEED)
    np.random.seed(SEED)
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_full = vd["Z_full"].astype(np.float64)
    user_to_zfull = {u: z_full[i] for i, u in enumerate(user_ids)}
    mt = json.load(open(TASK1_MANIFEST))
    test_users = sorted(mt["splits"]["test"]["ids"])
    rng = random.Random(SEED)
    rng.shuffle(test_users)
    test_users = test_users[:80]

    meta = load_meta()
    user_products: dict[str, set] = defaultdict(set)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            a = d.get("parent_asin") or d.get("asin")
            if u and a and u in test_users:
                user_products[u].add(a)
    # collect products with all N attributes for the largest N
    N_MAX = 5 + len(EXT_ATTR_KEYS)
    cells_by_n: dict[int, list[dict]] = {n: [] for n in range(5, N_MAX + 1)}
    for u in test_users:
        for a in user_products.get(u, set()):
            if a not in meta:
                continue
            rec = meta[a]
            for n in range(5, N_MAX + 1):
                attrs = attrs_for_n(rec, n)
                if attrs is not None:
                    cells_by_n[n].append({"user_id": u, "asin": a,
                                          "attrs": attrs,
                                          "z_user": user_to_zfull[u].tolist()})
    for n in cells_by_n:
        rng.shuffle(cells_by_n[n])
        cells_by_n[n] = cells_by_n[n][:40]
        log(f"  N{n}: {len(cells_by_n[n])} cells")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    H = model.config.hidden_size
    ckpt = torch.load(INJECTOR_PT, map_location=DEVICE)
    proj = SoftPrefixProjector(user_dim=Z_DIM, hidden_dim=PROJ_HIDDEN,
                               num_tokens=NUM_TOKENS, model_dim=H,
                               dtype=DTYPE, gate_init=GATE_INIT).to(DEVICE)
    proj.load_state_dict(ckpt["proj"])
    proj.eval()
    copy_head = CopyAwareHead(hidden_dim=H, vocab_size=model.config.vocab_size,
                              dtype=DTYPE).to(DEVICE)
    if "copy_head" in ckpt:
        copy_head.load_state_dict(ckpt["copy_head"])
    copy_head.eval()
    log("injector loaded")

    # generate per N with a generous length hint; measure natural max length
    results: dict[int, dict] = {}
    for n in range(5, N_MAX + 1):
        cells = cells_by_n[n]
        if not cells:
            results[n] = {"run": False}
            continue
        gens: list[dict] = []
        with torch.no_grad():
            for start in range(0, len(cells), BATCH):
                chunk = cells[start:start + BATCH]
                prompts, pes, attrs_list = [], [], []
                for c in chunk:
                    prompts.append(user_prompt_tokens(c["attrs"], tok,
                                                      "as detailed as possible"))
                    pes.append(proj(torch.tensor(
                        np.asarray(c["z_user"], dtype=np.float64),
                        dtype=torch.float32, device=DEVICE).to(DTYPE)
                        .unsqueeze(0))[0])
                    attrs_list.append(c["attrs"])
                texts = generate_batch(model, tok, copy_head, pes, prompts,
                                       attrs_list, 120, min_tokens=0)
                for c, t in zip(chunk, texts):
                    gens.append({"attrs": c["attrs"], "query": t})
        # metrics: natural = not degenerate AND all attrs present
        n_ok = n_all = 0
        nats: list[int] = []
        for g in gens:
            q = g["query"]
            toks = q.split()
            n_all += 1
            degenerate = len(set(toks)) <= 3 or len(toks) < 5
            content_ok = all(str(v).lower() in q.lower()
                             for v in g["attrs"].values())
            if not degenerate and content_ok:
                n_ok += 1
                nats.append(len(toks))
        import statistics
        results[n] = {
            "run": True, "n_gens": n_all, "n_attrs": n,
            "natural_content_ok_frac": round(n_ok / n_all, 4),
            "natural_len_mean": round(statistics.mean(nats), 1) if nats else None,
            "natural_len_median": round(statistics.median(nats), 1) if nats else None,
            "natural_len_p90": round(sorted(nats)[int(0.9 * len(nats))], 1)
            if nats else None,
        }
        log(f"  N{n}: ok_frac={results[n]['natural_content_ok_frac']} "
            f"nat_len med={results[n]['natural_len_median']} "
            f"p90={results[n]['natural_len_p90']}")
    with open(OUT_DIR / "e22_t3_nscan.json", "w") as f:
        json.dump({"version": "query_gen_nscan_v1", "seed": SEED,
                   "results": results}, f, indent=1)
    log("NSCAN DONE")


def main() -> None:
    t0 = time.time()
    # 内联 spaCy 特征（不 import 项目脚本）：注册到本模块
    global per_sentence_features_v2, user_features_v2
    from extract_syntactic_features import (
        per_sentence_features_v2 as _psf, user_features_v2 as _usf)
    per_sentence_features_v2 = _psf
    user_features_v2 = _usf
    if MAIN_STAGE == "train":
        train_stage()
    elif MAIN_STAGE == "generate":
        grid_generate_stage()
    elif MAIN_STAGE == "scale":
        scale_scan_stage()
    elif MAIN_STAGE == "dimscan":
        dimscan_stage()
    elif MAIN_STAGE == "nscan":
        nscan_stage()
    else:
        grid_eval_stage()
    log(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
