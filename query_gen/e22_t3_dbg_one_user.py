#!/usr/bin/env python3
"""Debug: single user AE7RO546AH with strict-span generation + dbg traces."""
from __future__ import annotations

import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

import query_gen_main as _qgm
from query_gen_main import (
    SoftPrefixProjector, CopyAwareHead, user_prompt_tokens, attrs_for_n,
    load_meta, DTYPE, DEVICE, NUM_TOKENS, PROJ_HIDDEN, Z_DIM, GATE_INIT,
    generate_batch, TASK1_VECTORS, REVIEWS, INJECTOR_PT,
)
from extract_clause_features_single_query import load_spacy_model
from extract_syntactic_features import per_sentence_features_v2, user_features_v2
_qgm.per_sentence_features_v2 = per_sentence_features_v2
_qgm.user_features_v2 = user_features_v2

TARGET_USER = "AE7RO546AH3XZFXLWNSOB6NL5QXQ"
ASIN = "B0B5JP4MGX"


def main() -> None:
    torch.manual_seed(5555)
    random.seed(5555)
    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    meta = load_meta()
    attrs = attrs_for_n(meta[ASIN], 10)
    texts = []
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("user_id") == TARGET_USER and \
                    (d.get("parent_asin") or d.get("asin")) == ASIN:
                t = (d.get("text") or "").strip()
                if t:
                    texts.append(t)
            if i > 20000000:
                break
    print(f"reviews: {len(texts)}", flush=True)
    sfs = []
    for doc in nlp.pipe(texts[:50], batch_size=64):
        for s in doc.sents:
            sf = per_sentence_features_v2(s)
            if sf is not None:
                sfs.append(sf)
    v = user_features_v2(sfs)
    z = ((v - tm) / ts).astype(np.float64)
    print(f"z: {z.shape}", flush=True)

    tok = AutoTokenizer.from_pretrained(INJECTOR_PT.parent if False else
                                        "/home/wlia0047/hj82_scratch2/wenyu/"
                                        "RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
                                        trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/"
        "Qwen2-7B-Instruct", torch_dtype=DTYPE, device_map=DEVICE,
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
    print("injector loaded", flush=True)

    prompt = [user_prompt_tokens(attrs, tok)]
    with torch.no_grad():
        pe = proj(torch.tensor(z, dtype=torch.float32,
                               device=DEVICE).to(DTYPE).unsqueeze(0))[0]
    text = generate_batch(model, tok, copy_head, [pe], prompt, [attrs],
                          128, strict_spans=True)[0]
    print("RESULT:", repr(text), flush=True)


if __name__ == "__main__":
    main()
