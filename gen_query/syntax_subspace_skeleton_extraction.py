#!/usr/bin/env python3
"""Phase 6.D.1 — Syntax skeleton extraction (smoke v1).

用户 2026-08-30 22:20 提案 (PCA-guided syntax template retrieval pipeline):
不教 LLM 理解 PCA48,而是从真实 corpus 提取 delexicalized syntax skeleton,
让 LLM 只负责 fill-in-the-blank。

骨架提取规则:
- 保留 first-person pronouns (I/my) — style 个性化
- 保留 CONJ/SCONJ/AUX/DET/VERB/PRON — function words 保留结构
- 替换 NOUN/PROPN → ATTR_SLOT  (产品词位置)
- 替换 ADJ → ADJ_SLOT          (修饰词位置)
- 替换 ADV → ADV_SLOT          (状语位置)
- 替换 NUM → NUM_SLOT          (数字位置)
- 保留全部 punctuation/case/dep relations

输出:
- syntax_skeleton_pool.jsonl (skeleton_str, pca48_z, token_count, dep_tree_str)
- skeleton_48d_embeddings.npy (N, 48)

Smoke: 1000 train queries, ~30s
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
DATA_DIR = SCRATCH / "pca48_dataset"
OUT_DIR = SCRATCH / "syntax_skeleton"
LOG_DIR = SCRATCH / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Smoke config ===
N_QUERIES = 1000
SEED = 42
LOG_EVERY = 100

# === POS-based delexicalization ===
SLOT_MAP = {
    "NOUN": "ATTR_SLOT",
    "PROPN": "ATTR_SLOT",
    "ADJ": "ADJ_SLOT",
    "ADV": "ADV_SLOT",
    "NUM": "NUM_SLOT",
}

PRESERVE_KEEP_FIRST_PERSON = True  # I/my/me/mine


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [skel_extract] {msg}", flush=True)


def token_to_skeleton(token_text: str, pos: str, is_first_person: bool = False) -> str:
    """Delexicalize token by POS tag.

    Returns skeleton token (slot placeholder or preserved token).
    """
    if PRESERVE_KEEP_FIRST_PERSON and is_first_person:
        return token_text.lower()
    if pos in SLOT_MAP:
        return SLOT_MAP[pos]
    return token_text.lower()


def extract_skeleton(doc):
    """Extract (skeleton_str, dep_tree_str, token_count) from spaCy doc.

    Returns:
      skeleton_str: "I need ATTR_SLOT SCONJ PRON ADJ VB ADJ"
      dep_tree_str: "nsubj(I,need) det(need,a) dobj(need,ATTR_SLOT) ..."
      token_count: int
    """
    tokens = []
    dep_rels = []
    for tok in doc:
        # Skip punctuation for token count, but include in dep tree
        if tok.is_punct or tok.is_space:
            continue
        is_fp = tok.text.lower() in {"i", "my", "me", "mine", "we", "our", "us"}
        skel_token = token_to_skeleton(tok.text, tok.pos_, is_first_person=is_fp)
        tokens.append(skel_token)
        dep_rels.append(f"{tok.dep_}({tok.i},{skel_token})")

    skeleton_str = " ".join(tokens)
    dep_tree_str = " | ".join(dep_rels)
    token_count = len(tokens)
    return skeleton_str, dep_tree_str, token_count


def main():
    log("=== Phase 6.D.1 — Syntax skeleton extraction (SMOKE v1) ===")
    log(f"  N_QUERIES={N_QUERIES}, N_slot_types={len(SLOT_MAP)}")

    # Load train.jsonl
    samples = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    log(f"  loaded {len(samples)} training samples")

    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(samples), size=min(N_QUERIES, len(samples)), replace=False)

    # Load spaCy
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if "textcat" in nlp.pipe_names:
        nlp.remove_pipe("textcat")

    # Extract skeletons
    skeletons = []
    pca48_embeddings = []
    t0 = time.time()
    for k, idx in enumerate(indices):
        s = samples[idx]
        s_i = s["s_i"]
        z_48 = s["z_48"]

        doc = nlp(s_i)
        skel_str, dep_tree, tok_count = extract_skeleton(doc)

        skeletons.append({
            "sample_idx": int(idx),
            "user_id": s.get("user_id") or s.get("user"),
            "asin": s.get("asin"),
            "original_query": s_i,
            "skeleton_str": skel_str,
            "dep_tree_str": dep_tree,
            "token_count": tok_count,
            "slot_count": skel_str.count("_SLOT"),
            "z_48": z_48,
        })
        pca48_embeddings.append(z_48)

        if (k + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t0
            avg = elapsed / (k + 1)
            eta = (len(indices) - k - 1) * avg
            log(f"  extracted {k+1}/{len(indices)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    pca48_arr = np.array(pca48_embeddings, dtype=np.float32)
    log(f"  total extracted: {len(skeletons)}  total time: {time.time()-t0:.0f}s")

    # Save
    pool_path = OUT_DIR / f"syntax_skeleton_pool_n{N_QUERIES}.jsonl"
    with open(pool_path, "w") as f:
        for sk in skeletons:
            f.write(json.dumps(sk) + "\n")
    log(f"  saved skeleton pool → {pool_path}")

    emb_path = OUT_DIR / f"skeleton_48d_embeddings_n{N_QUERIES}.npy"
    np.save(emb_path, pca48_arr)
    log(f"  saved PCA48 embeddings → {emb_path}  shape={pca48_arr.shape}")

    # Quick stats
    skel_strs = [s["skeleton_str"] for s in skeletons]
    unique_skel = len(set(skel_strs))
    avg_tok = np.mean([s["token_count"] for s in skeletons])
    avg_slot = np.mean([s["slot_count"] for s in skeletons])
    log(f"\n=== SKELETON POOL STATS ===")
    log(f"  total: {len(skeletons)}")
    log(f"  unique skeletons: {unique_skel} ({unique_skel/len(skeletons)*100:.1f}%)")
    log(f"  avg token count: {avg_tok:.1f}")
    log(f"  avg SLOT count per skeleton: {avg_slot:.1f}")

    # Print 5 example skeletons for verification
    log(f"\n=== EXAMPLE SKELETONS (sanity check) ===")
    for sk in skeletons[:5]:
        log(f"  original: {sk['original_query']}")
        log(f"  skeleton: {sk['skeleton_str']}")
        log(f"  slots: {sk['slot_count']}/{sk['token_count']} tokens")
        log(f"  PCA48 norm: {float(np.linalg.norm(sk['z_48'])):.2f}")
        log("")


if __name__ == "__main__":
    main()