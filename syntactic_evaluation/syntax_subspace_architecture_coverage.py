"""Stage 5 architecture coverage: 4-arch FULL retrieval on v6m strict alignment.

按用户指令 2026-08-28,补齐 retrieval architecture 集合到 4 类 FULL retrieval:
  1. Lexical sparse:       BM25 (bm25s lucene) — full corpus
  2. Learned sparse:       SPLADE (naver/splade-cocondenser-ensembledistil) — full corpus sparse scoring
  3. Dense bi-encoder:     BGE-base-en-v1.5 (BAAI) — full corpus dense cosine
  4. Late interaction:     ColBERTv2 (colbert-ir/colbertv2.0) — full corpus late-interaction maxsim
     (768d BERT hidden → 128d linear projection loaded from safetensors)

(Cross-encoder dropped — full corpus CE on 217K is impractical: ~230h compute.
 See [[stage5-cross-encoder-rerank]] for prior BM25 top-100 + MS-MARCO-MiniLM rerank ablation.)

输入: /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection.json (v6m 3781 strict)
输出: /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_arch4_per_query.json
      /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_arch4_summary.json
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASIN_TO_DOC_CACHE, META_FILE, log,
)

SEL_IN = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection.json")
PER_QUERY_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_arch4_per_query.json")
SUMMARY_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_arch4_summary.json")
EMBED_CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/arch4_embeds")

BM25_TOPK_FOR_CE = 100


def build_meta_corpus():
    if ASIN_TO_DOC_CACHE.exists():
        t0 = time.time()
        asin_to_doc = json.load(open(ASIN_TO_DOC_CACHE, encoding="utf-8"))
        log(f"  ✓ asin_to_doc cache hit ({len(asin_to_doc)} ASINs, {time.time() - t0:.2f}s)")
        return asin_to_doc
    asin_to_doc = {}
    log(f"  loading metadata from {META_FILE}")
    with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            asin = r.get("parent_asin", "").strip()
            if not asin:
                continue
            parts = []
            t = r.get("title", "").strip()
            if t:
                parts.append(t)
            desc = r.get("description", [])
            if isinstance(desc, list):
                desc = " ".join(desc)
            elif not isinstance(desc, str):
                desc = ""
            desc = desc.strip()
            if desc:
                parts.append(desc)
            feats = r.get("features", [])
            if isinstance(feats, list):
                feats = " ".join(feats)
            if feats:
                parts.append(feats[:500])
            doc = " | ".join(parts).strip()
            if doc:
                asin_to_doc[asin] = doc
    return asin_to_doc


def rank_target_in_sorted(target_idx: int, sorted_docs: np.ndarray, max_k: int) -> int:
    """Find rank of target_idx in sorted_docs. Return max_k+1 if not found."""
    if target_idx < 0:
        return -1
    positions = np.where(sorted_docs == target_idx)[0]
    if len(positions) == 0:
        return max_k + 1
    return int(positions[0]) + 1


def rr_hit_from_rank(rank: int) -> dict:
    if rank < 0:
        return {"rank": None, "RR": 0.0, "hit1": 0, "hit5": 0, "hit10": 0}
    return {"rank": rank, "RR": 1.0 / rank,
            "hit1": 1 if rank == 1 else 0,
            "hit5": 1 if rank <= 5 else 0,
            "hit10": 1 if rank <= 10 else 0}


# ===========================================================================
# A1: BM25 full
# ===========================================================================
def bm25_full_retrieve(queries: list[str], corpus_texts: list[str], target_indices: np.ndarray,
                       bm25_k: int = 20000) -> list[dict]:
    import bm25s
    log("\n=== A1 BM25 (Lexical sparse) ===")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    log(f"  corpus tokenized in {time.time() - t0:.1f}s")
    log("  building bm25s index...")
    t0 = time.time()
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    log(f"  index built in {time.time() - t0:.1f}s")

    t0 = time.time()
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    BM25_BATCH = 1000
    results = [None] * len(queries)

    def _slice_tok(tok, s, e):
        return type(tok)(tok.ids[s:e], tok.vocab)

    for s in range(0, len(queries), BM25_BATCH):
        e = min(s + BM25_BATCH, len(queries))
        sub_tokens = _slice_tok(query_tokens, s, e)
        sub_res = retriever.retrieve(sub_tokens, k=bm25_k, show_progress=False)
        for i in range(e - s):
            gi = s + i
            tgt_idx = int(target_indices[gi])
            sorted_docs = sub_res.documents[i]
            rank = rank_target_in_sorted(tgt_idx, sorted_docs, bm25_k)
            results[gi] = rr_hit_from_rank(rank)
    log(f"  retrieved in {time.time() - t0:.1f}s")
    return results


def bm25_topk(queries: list[str], corpus_texts: list[str], k: int) -> list[list[tuple[int, float]]]:
    import bm25s
    log(f"\n=== BM25 top-{k} (for downstream rerank) ===")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    log(f"  corpus tokenized in {time.time() - t0:.1f}s")
    log("  building bm25s index...")
    t0 = time.time()
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    log(f"  index built in {time.time() - t0:.1f}s")

    t0 = time.time()
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    BM25_BATCH = 1000
    out = [None] * len(queries)

    def _slice_tok(tok, s, e):
        return type(tok)(tok.ids[s:e], tok.vocab)

    for s in range(0, len(queries), BM25_BATCH):
        e = min(s + BM25_BATCH, len(queries))
        sub_tokens = _slice_tok(query_tokens, s, e)
        sub_res = retriever.retrieve(sub_tokens, k=k, show_progress=False)
        for i in range(e - s):
            gi = s + i
            docs = sub_res.documents[i]
            scores = sub_res.scores[i]
            out[gi] = [(int(docs[j]), float(scores[j])) for j in range(len(docs))]
    log(f"  retrieved in {time.time() - t0:.1f}s")
    return out


# ===========================================================================
# A2: SPLADE full
# ===========================================================================
def splade_encode(model, tok, texts: list[str], max_length: int = 128, batch_size: int = 128) -> torch.Tensor:
    """bf16 + large batch for A100. Returns (N, V) fp16 sparse-ish."""
    chunks: list[torch.Tensor] = []
    n_texts = len(texts)
    n_done = 0
    for s in range(0, n_texts, batch_size):
        e = min(s + batch_size, n_texts)
        batch_texts = texts[s:e]
        inputs = tok(batch_texts, return_tensors="pt", truncation=True,
                     max_length=max_length, padding=True).to(model.device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits  # (B, L, V) bf16
        rep = torch.log(1 + torch.relu(logits.float()))  # fp32 stability
        rep = rep * inputs.attention_mask.unsqueeze(-1)
        rep = rep.max(dim=1)[0]  # (B, V)
        chunks.append(rep.cpu().half())
        del logits, rep, inputs
        torch.cuda.empty_cache()
        n_done += e - s
        if n_done % (batch_size * 50) == 0 or n_done == n_texts:
            log(f"    encoded {n_done}/{n_texts} ({100*n_done/n_texts:.1f}%)")
    return torch.cat(chunks, dim=0)


def splade_full_retrieve(queries: list[str], corpus_texts: list[str],
                         target_indices: np.ndarray) -> list[dict]:
    """Full SPLADE scoring: (3781, V) @ (V, 217K) → scores, then rank."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    log("\n=== A2 SPLADE (Learned sparse, naver/splade-cocondenser-ensembledistil) ===")
    cache_dir = EMBED_CACHE_DIR / "splade"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = "naver/splade-cocondenser-ensembledistil"

    tok = AutoTokenizer.from_pretrained(name, cache_dir='/home/wlia0047/hj82_scratch2/wenyu/hf_cache')
    model = AutoModelForMaskedLM.from_pretrained(name, cache_dir='/home/wlia0047/hj82_scratch2/wenyu/hf_cache').to("cuda").eval()

    corpus_cache = cache_dir / "corpus_splade.pt"
    query_cache = cache_dir / "query_splade.pt"

    if corpus_cache.exists():
        log(f"  ✓ corpus SPLADE cache hit ({corpus_cache.stat().st_size / 1e9:.2f} GB)")
        splade_corpus = torch.load(corpus_cache, weights_only=True)
    else:
        log(f"  encoding {len(corpus_texts)} corpus with SPLADE (bf16, batch=128, max_length=128)...")
        t0 = time.time()
        splade_corpus = splade_encode(model, tok, corpus_texts)
        log(f"  encoded in {time.time() - t0:.1f}s, shape={splade_corpus.shape}")
        torch.save(splade_corpus, corpus_cache)

    if query_cache.exists():
        splade_queries = torch.load(query_cache, weights_only=True)
    else:
        log(f"  encoding {len(queries)} queries with SPLADE...")
        splade_queries = splade_encode(model, tok, queries)
        torch.save(splade_queries, query_cache)

    # Score: q_rep @ corpus_rep.T → (Q, N) on GPU
    log(f"  matmul Q × N on GPU...")
    t0 = time.time()
    q_gpu = splade_queries.cuda().float()  # (Q, V)
    c_gpu = splade_corpus.cuda().float()  # (N, V)
    results = []
    Q_BATCH = 500
    for s in range(0, q_gpu.shape[0], Q_BATCH):
        e = min(s + Q_BATCH, q_gpu.shape[0])
        sims = q_gpu[s:e] @ c_gpu.T  # (q_b, N)
        # Per-query: find rank of target_idx
        for j in range(e - s):
            gi = s + j
            tgt_idx = int(target_indices[gi])
            if tgt_idx < 0:
                results.append({"rank": None, "RR": 0.0, "hit1": 0, "hit5": 0, "hit10": 0})
                continue
            sc = sims[j]
            rank = int((sc > sc[tgt_idx]).sum().item()) + 1
            results.append({"rank": rank, "RR": 1.0 / rank,
                           "hit1": 1 if rank == 1 else 0,
                           "hit5": 1 if rank <= 5 else 0,
                           "hit10": 1 if rank <= 10 else 0})
        del sims
    del q_gpu, c_gpu
    torch.cuda.empty_cache()
    log(f"  matmul done in {time.time() - t0:.1f}s")
    del model, tok, splade_corpus, splade_queries
    torch.cuda.empty_cache()
    return results


# ===========================================================================
# A3: BGE dense
# ===========================================================================
def bge_full_retrieve(queries: list[str], corpus_texts: list[str], target_indices: np.ndarray) -> tuple[list[dict], np.ndarray]:
    from sentence_transformers import SentenceTransformer
    log("\n=== A3 BGE-base-en-v1.5 (Dense bi-encoder) ===")
    cache_dir = EMBED_CACHE_DIR / "bge_base_v15"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")

    corpus_cache = cache_dir / "corpus_embeds.npy"
    query_cache = cache_dir / "query_embeds.npy"

    if corpus_cache.exists():
        corpus_embeds = np.load(corpus_cache)
        log(f"  ✓ corpus embeds cache ({corpus_embeds.shape})")
    else:
        log(f"  encoding corpus...")
        t0 = time.time()
        corpus_embeds = model.encode(corpus_texts, batch_size=128, show_progress_bar=False,
                                     convert_to_numpy=True, normalize_embeddings=True)
        log(f"  encoded in {time.time() - t0:.1f}s")
        np.save(corpus_cache, corpus_embeds)

    if query_cache.exists():
        q_embeds = np.load(query_cache)
    else:
        log(f"  encoding {len(queries)} queries...")
        q_embeds = model.encode(queries, batch_size=512, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
        np.save(query_cache, q_embeds)

    log(f"  matmul + ranks on GPU...")
    t0 = time.time()
    corpus_gpu = torch.from_numpy(corpus_embeds).cuda()
    q_gpu = torch.from_numpy(q_embeds).cuda()
    results = []
    BATCH = 2000
    tgt_all = torch.as_tensor(target_indices, device="cuda", dtype=torch.long)
    for s in range(0, q_gpu.shape[0], BATCH):
        e = min(s + BATCH, q_gpu.shape[0])
        sims = q_gpu[s:e] @ corpus_gpu.T
        tgt_chunk = tgt_all[s:e]
        for j in range(sims.shape[0]):
            tgt_idx = int(tgt_chunk[j].item())
            if tgt_idx < 0:
                results.append({"rank": None, "RR": 0.0, "hit1": 0, "hit5": 0, "hit10": 0})
                continue
            sc = sims[j]
            rank = int((sc > sc[tgt_idx]).sum().item()) + 1
            results.append({"rank": rank, "RR": 1.0 / rank,
                           "hit1": 1 if rank == 1 else 0,
                           "hit5": 1 if rank <= 5 else 0,
                           "hit10": 1 if rank <= 10 else 0})
        del sims
    del corpus_gpu, q_gpu, tgt_all
    torch.cuda.empty_cache()
    log(f"  matmul done in {time.time() - t0:.1f}s")
    return results, q_embeds


# ===========================================================================
# A4: ColBERTv2 FULL late-interaction (768d → 128d linear projection)
# ===========================================================================
def load_colbert_projection(snapshot_dir: Path) -> torch.Tensor:
    """Load linear.weight (128, 768) from ColBERTv2 safetensors."""
    import safetensors.torch as st
    snap_str = str(snapshot_dir) + "/"
    sd = st.load_file(snap_str + "model.safetensors")
    return sd["linear.weight"].float()  # (128, 768)


def colbert_encode_all(
    model, tok, proj_weight: torch.Tensor, texts: list[str],
    max_length: int = 64, batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode texts to (n_tokens, 128) ColBERT reps.

    Returns:
      reps: (N, max_L, 128) fp16 padded (only valid tokens per-row are meaningful)
      valid_lens: (N,) actual token counts
    Memory: 217K × 64 × 128 × 2B = 3.5 GB.
    """
    max_L = max_length
    proj = proj_weight.cuda()  # (128, 768)
    D_out = proj.shape[0]
    reps_arr = np.zeros((len(texts), max_L, D_out), dtype=np.float16)
    valid_lens = np.zeros(len(texts), dtype=np.int32)
    n_done = 0
    t0 = time.time()
    for s in range(0, len(texts), batch_size):
        e = min(s + batch_size, len(texts))
        batch_texts = texts[s:e]
        inputs = tok(batch_texts, return_tensors="pt", truncation=True, max_length=max_length,
                     padding="max_length").to(model.device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model(**inputs).last_hidden_state  # (B, L, 768)
        # Apply linear projection: (B, L, 768) @ (768, 128) → (B, L, 128)
        projected = hidden.float() @ proj.T  # (B, L, 128) fp32
        # L2 normalize per token (ColBERT style)
        norms = projected.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        projected = projected / norms  # (B, L, 128)
        # Move to CPU as fp16
        proj_cpu = projected.cpu().half().numpy()  # (B, L, 128)
        for j in range(e - s):
            L = int(inputs.attention_mask[j].sum().item())
            if L == 0:
                L = 1
            reps_arr[s + j, :L] = proj_cpu[j, :L]
            valid_lens[s + j] = L
        del hidden, projected, inputs
        torch.cuda.empty_cache()
        n_done += e - s
        if n_done % (batch_size * 50) == 0 or n_done == len(texts):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(texts) - n_done) / rate if rate > 0 else 0
            log(f"    encoded {n_done}/{len(texts)} ({100*n_done/len(texts):.1f}%) rate={rate:.0f}/s eta={eta:.0f}s")
    return reps_arr, valid_lens


def colbert_full_retrieve(
    queries: list[str], corpus_texts: list[str], target_indices: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    """ColBERTv2 FULL retrieval: maxsim query × all docs on GPU."""
    from transformers import AutoModel, AutoTokenizer
    log("\n=== A4 ColBERTv2 (Late interaction, FULL corpus maxsim) ===")
    cache_dir = EMBED_CACHE_DIR / "colbertv2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = "colbert-ir/colbertv2.0"
    snap_dir = Path("/home/wlia0047/hj82_scratch2/wenyu/hf_cache/models--colbert-ir--colbertv2.0/snapshots") / \
               sorted((Path("/home/wlia0047/hj82_scratch2/wenyu/hf_cache/models--colbert-ir--colbertv2.0/snapshots").iterdir()))[0].name

    tok = AutoTokenizer.from_pretrained(name, cache_dir='/home/wlia0047/hj82_scratch2/wenyu/hf_cache')
    model = AutoModel.from_pretrained(name, cache_dir='/home/wlia0047/hj82_scratch2/wenyu/hf_cache').to("cuda").eval()

    proj_weight = load_colbert_projection(snap_dir)  # (128, 768) fp32

    corpus_reps_file = cache_dir / "corpus_reps.npy"
    corpus_lens_file = cache_dir / "corpus_lens.npy"
    query_reps_file = cache_dir / "query_reps.npy"
    query_lens_file = cache_dir / "query_lens.npy"

    if corpus_reps_file.exists():
        corpus_reps = np.load(corpus_reps_file)
        corpus_lens = np.load(corpus_lens_file)
        log(f"  ✓ corpus reps cache {corpus_reps.shape} (lens {corpus_lens.shape})")
    else:
        log(f"  encoding {len(corpus_texts)} corpus with ColBERTv2 (max_length=64)...")
        corpus_reps, corpus_lens = colbert_encode_all(model, tok, proj_weight, corpus_texts, max_length=64, batch_size=128)
        np.save(corpus_reps_file, corpus_reps)
        np.save(corpus_lens_file, corpus_lens)

    if query_reps_file.exists():
        query_reps = np.load(query_reps_file)
        query_lens = np.load(query_lens_file)
        log(f"  ✓ query reps cache {query_reps.shape}")
    else:
        log(f"  encoding {len(queries)} queries with ColBERTv2 (max_length=32)...")
        query_reps, query_lens = colbert_encode_all(model, tok, proj_weight, queries, max_length=32, batch_size=256)
        np.save(query_reps_file, query_reps)
        np.save(query_lens_file, query_lens)

    # GPU scoring: per query, full corpus maxsim
    log(f"  GPU maxsim Q × N...")
    t0 = time.time()
    corpus_gpu = torch.from_numpy(corpus_reps).cuda()  # (N, Ld, 128) fp16
    # We need Ld_per_doc — use corpus_lens to mask
    # Precompute: for each query, compute maxsim against all docs (slow if naive).
    # Smart: compute per-query token vs all doc tokens: (Lq, 128) × (N, Ld, 128) → (N, Lq, Ld) — OOM.
    # Strategy: for each query, score in chunks of N candidates (e.g. 4096 at a time).
    results = []
    n_done = 0
    Q = query_reps.shape[0]
    N = corpus_reps.shape[0]
    for qi in range(Q):
        tgt_idx = int(target_indices[qi])
        if tgt_idx < 0:
            results.append({"rank": None, "RR": 0.0, "hit1": 0, "hit5": 0, "hit10": 0})
            continue
        Lq = int(query_lens[qi])
        qt = torch.from_numpy(query_reps[qi, :Lq]).cuda().float()  # (Lq, 128) fp32
        # Compute scores in chunks of N candidates (load doc reps into GPU per chunk)
        all_scores = np.empty(N, dtype=np.float32)
        CHUNK_N = 4096
        for s in range(0, N, CHUNK_N):
            e = min(s + CHUNK_N, N)
            d_chunk = corpus_gpu[s:e].float()  # (chunk, Ld, 128)
            Ld = d_chunk.shape[1]
            # Mask: zero out tokens beyond valid lens per doc
            lens_chunk = corpus_lens[s:e]
            mask = torch.arange(Ld, device="cuda")[None, :] < torch.as_tensor(lens_chunk, device="cuda")[:, None]
            d_chunk = d_chunk * mask.unsqueeze(-1)  # zero out invalid tokens
            # maxsim: for each (q_token, d_token), dot product, then max over d_tokens, sum over q_tokens
            # (Lq, 128) × (chunk, Ld, 128) → (chunk, Lq, Ld)
            sim = torch.einsum("qd,cld->cql", qt, d_chunk)  # (chunk, Lq, Ld)
            max_per_q = sim.max(dim=-1)[0]  # (chunk, Lq)
            all_scores[s:e] = max_per_q.sum(dim=-1).cpu().numpy()
            del d_chunk, sim, max_per_q, mask
        # Rank: target_idx's score
        tgt_score = all_scores[tgt_idx]
        # Avoid counting target itself: mask it out
        all_scores[tgt_idx] = -np.inf
        better = int((all_scores > tgt_score).sum())
        rank = better + 1
        results.append({"rank": rank, "RR": 1.0 / rank,
                       "hit1": 1 if rank == 1 else 0,
                       "hit5": 1 if rank <= 5 else 0,
                       "hit10": 1 if rank <= 10 else 0})
        n_done += 1
        if n_done % 100 == 0 or n_done == Q:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (Q - n_done) / rate if rate > 0 else 0
            log(f"    scored {n_done}/{Q} ({100*n_done/Q:.1f}%) rate={rate:.1f} q/s eta={eta:.0f}s")
        del qt

    log(f"  GPU scoring done in {time.time() - t0:.1f}s")
    del corpus_gpu, model, tok
    torch.cuda.empty_cache()
    return results, query_reps.reshape(Q, -1)  # flat for sim09


# ===========================================================================
# A5: MS-MARCO CE rerank on BM25 top-100
# ===========================================================================
def ce_msmarco_retrieve(
    queries: list[str], corpus_texts: list[str], target_indices: np.ndarray,
    bm25_topk_per_q: list[list[tuple[int, float]]],
) -> tuple[list[dict], np.ndarray]:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    log("\n=== A5 MS-MARCO-MiniLM Cross-encoder rerank (BM25 top-100) ===")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256, device="cuda")

    log(f"  building (query, doc) pairs for {len(queries)} queries × top-{BM25_TOPK_FOR_CE}...")
    all_pairs = []
    pair_offsets = [0] * (len(queries) + 1)
    for qi, q in enumerate(queries):
        for cand_idx, _ in bm25_topk_per_q[qi]:
            all_pairs.append((q, corpus_texts[cand_idx]))
        pair_offsets[qi + 1] = pair_offsets[qi] + len(bm25_topk_per_q[qi])
    log(f"  total pairs: {len(all_pairs)}")

    log(f"  predicting CE scores in batches...")
    t0 = time.time()
    CE_BATCH = 128
    all_scores = []
    for s in range(0, len(all_pairs), CE_BATCH):
        e = min(s + CE_BATCH, len(all_pairs))
        sc = ce.predict(all_pairs[s:e], show_progress_bar=False, convert_to_numpy=True)
        all_scores.extend(sc.tolist())
    log(f"  predicted in {time.time() - t0:.1f}s")

    log(f"  reranking per query...")
    results = []
    for qi in range(len(queries)):
        tgt_idx = int(target_indices[qi])
        if tgt_idx < 0:
            results.append({"rank": None, "RR": 0.0, "hit1": 0, "hit5": 0, "hit10": 0})
            continue
        cands = bm25_topk_per_q[qi]
        start, end = pair_offsets[qi], pair_offsets[qi + 1]
        ce_scores = all_scores[start:end]
        order = sorted(range(len(cands)), key=lambda j: -ce_scores[j])
        sorted_docs = [cands[j][0] for j in order]
        positions = np.where(np.array(sorted_docs) == tgt_idx)[0]
        if len(positions) == 0:
            rank = BM25_TOPK_FOR_CE + 1
        else:
            rank = int(positions[0]) + 1
        results.append({"rank": rank, "RR": 1.0 / rank,
                       "hit1": 1 if rank == 1 else 0,
                       "hit5": 1 if rank <= 5 else 0,
                       "hit10": 1 if rank <= 10 else 0})

    log(f"  encoding queries for sim09 (using MiniLM)...")
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    q_embeds = minilm.encode(queries, batch_size=512, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
    del ce, minilm
    torch.cuda.empty_cache()
    return results, q_embeds


# ===========================================================================
# Volatility (sim09)
# ===========================================================================
def compute_volatility_sim09(
    per_query: list[dict], rank_key: str, rr_key: str,
    q_embeds: np.ndarray, sim_threshold: float = 0.9,
) -> dict:
    by_asin: dict[str, list] = collections.defaultdict(list)
    for r in per_query:
        by_asin[r["asin"]].append(r)
    qid_to_embed = {id(r): q_embeds[i] for i, r in enumerate(per_query)}

    f1_list, f5_list, f10_list = [], [], []
    rr_std_list = []
    n_asins_used = 0
    for asin, qs in by_asin.items():
        if len(qs) < 2:
            continue
        ranks = [q.get(rank_key) for q in qs]
        if any(r is None for r in ranks):
            continue
        hit1 = [1 if r == 1 else 0 for r in ranks]
        hit5 = [1 if r <= 5 else 0 for r in ranks]
        hit10 = [1 if r <= 10 else 0 for r in ranks]
        rrs = [float(q.get(rr_key) or 0.0) for q in qs]
        rr_std = float(np.std(rrs, ddof=0)) if len(rrs) >= 2 else None
        embeds = np.stack([qid_to_embed[id(q)] for q in qs], axis=0)
        sim_mat = embeds @ embeds.T

        def flip_rate(hit_labels):
            n_q = len(hit_labels)
            used, disagree = 0, 0
            for i in range(n_q):
                for j in range(i + 1, n_q):
                    if sim_mat[i, j] >= sim_threshold:
                        used += 1
                        if hit_labels[i] != hit_labels[j]:
                            disagree += 1
            if used == 0:
                return None
            return disagree / used

        for hit_labels, lst in [(hit1, f1_list), (hit5, f5_list), (hit10, f10_list)]:
            v = flip_rate(hit_labels)
            if v is not None:
                lst.append(v)
        if rr_std is not None:
            rr_std_list.append(rr_std)
        n_asins_used += 1

    return {
        "n_asins": n_asins_used,
        "Hit@1_FlipRate_mean": float(np.mean(f1_list)) if f1_list else None,
        "Hit@5_FlipRate_mean": float(np.mean(f5_list)) if f5_list else None,
        "Hit@10_FlipRate_mean": float(np.mean(f10_list)) if f10_list else None,
        "RR_Std_mean": float(np.mean(rr_std_list)) if rr_std_list else None,
        "RR_Std_median": float(np.median(rr_std_list)) if rr_std_list else None,
    }


def main():
    log_start = time.time()
    log("=== Stage 5 architecture coverage: 5-arch FULL retrieval on v6m strict alignment ===")

    # ---- 1. Load v6m selection ----
    log("\n=== 1. Loading v6m selection ===")
    selection = json.load(open(SEL_IN))
    entries = selection["entries"]
    query_records = []
    for i, e in enumerate(entries):
        q = e["selected"]
        if q is None:
            continue
        query_records.append({
            "entry_idx": i,
            "asin": e["asin"],
            "user_id": e["user_id"],
            "selection_method": e["selection_method"],
            "selected_distance": e["selected_distance"],
            "selected_margin": e["selected_margin"],
            "user_source": e["user_source"],
            "query": q["query"],
            "n_tok": q["n_tok"],
            "attrs_covered": q["attrs_covered"],
            "strict": q["strict"],
        })
    log(f"  strict queries: {len(query_records)}")

    # ---- 2. Build corpus ----
    log("\n=== 2. Building ASIN corpus ===")
    asin_to_doc = build_meta_corpus()
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus size: {len(asins)} ASINs")

    queries = [r["query"] for r in query_records]
    target_indices = np.array([asin_to_idx.get(r["asin"], -1) for r in query_records])
    corpus_texts = [asin_to_doc[a] for a in asins]

    # ---- 3. A1: BM25 full ----
    bm25_results = bm25_full_retrieve(queries, corpus_texts, target_indices)

    # ---- 4. A2: SPLADE full ----
    splade_results = splade_full_retrieve(queries, corpus_texts, target_indices)

    # ---- 5. A3: BGE dense full ----
    bge_results, bge_q_embeds = bge_full_retrieve(queries, corpus_texts, target_indices)

    # ---- 6. A4: ColBERTv2 full late-interaction ----
    colbert_results, colbert_q_flat = colbert_full_retrieve(queries, corpus_texts, target_indices)

    # ---- 7. Save per-query ----
    log("\n=== 7. Saving per-query ===")
    results_by_name = {
        "bm25": bm25_results,
        "splade": splade_results,
        "bge_base_v15": bge_results,
        "colbertv2": colbert_results,
    }
    for i, r in enumerate(query_records):
        for name, rs in results_by_name.items():
            res = rs[i]
            r[f"{name}_rank"] = res["rank"]
            r[f"{name}_RR"] = res["RR"]
            r[f"{name}_hit1"] = res["hit1"]
            r[f"{name}_hit5"] = res["hit5"]
            r[f"{name}_hit10"] = res["hit10"]

    PER_QUERY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PER_QUERY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 5 architecture coverage (4-arch) on v6m strict alignment",
                "architectures": ["bm25 (lexical sparse)", "splade (learned sparse)", "bge-base-v1.5 (dense bi-encoder)", "colbertv2 (late interaction)"],
                "selection_file": str(SEL_IN),
                "corpus_size": len(asins),
            },
            "n_queries": len(query_records),
            "queries": query_records,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {PER_QUERY_OUT}")

    # ---- 8. Headline ----
    log("\n=== 8. Headline (per-query mean) ===")
    headline = {}
    for name, rs in results_by_name.items():
        headline[name] = {
            "RR_mean": float(np.mean([r[f"{name}_RR"] for r in query_records])),
            "RR_median": float(np.median([r[f"{name}_RR"] for r in query_records])),
            "hit1": float(np.mean([r[f"{name}_hit1"] for r in query_records])),
            "hit5": float(np.mean([r[f"{name}_hit5"] for r in query_records])),
            "hit10": float(np.mean([r[f"{name}_hit10"] for r in query_records])),
        }
        h = headline[name]
        log(f"  {name:<22} RR={h['RR_mean']:.4f}  hit@1={h['hit1']:.4f}  hit@5={h['hit5']:.4f}  hit@10={h['hit10']:.4f}")

    # ---- 9. Volatility ----
    log("\n=== 9. Volatility sim09 ===")
    q_embeds_map = {
        "bm25": bge_q_embeds,        # BM25: no native embeds, use BGE for sim09
        "splade": bge_q_embeds,
        "bge_base_v15": bge_q_embeds,
        "colbertv2": colbert_q_flat,  # flattened (Lq*128)
    }
    volatility = {}
    for name in results_by_name:
        qe = q_embeds_map.get(name)
        vol = compute_volatility_sim09(query_records, f"{name}_rank", f"{name}_RR", qe)
        volatility[name] = vol
        log(f"  {name:<22} Hit@1_flip={vol['Hit@1_FlipRate_mean']}  Hit@5_flip={vol['Hit@5_FlipRate_mean']}  Hit@10_flip={vol['Hit@10_FlipRate_mean']}  RR_Std={vol['RR_Std_mean']}")

    summary_data = {
        "config": {
            "description": "Stage 5 architecture coverage (4-arch) on v6m strict alignment",
            "architectures": {
                "lexical_sparse": "BM25 (bm25s lucene k1=1.5 b=0.75, k=20000)",
                "learned_sparse": "SPLADE (naver/splade-cocondenser-ensembledistil) full Q×N sparse score",
                "dense_biencoder": "BGE-base-en-v1.5 (BAAI, 768d, L2-normalized) full Q×N cosine",
                "late_interaction": "ColBERTv2 (colbert-ir/colbertv2.0) full corpus maxsim with 768→128 linear projection",
            },
            "selection_file": str(SEL_IN),
        },
        "headline_per_query_mean": headline,
        "volatility_sim09": volatility,
    }
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")

    log("\n=== Final Headline (4-arch FULL retrieval comparison on v6m strict alignment) ===")
    arch_order = ["bm25", "splade", "bge_base_v15", "colbertv2"]
    arch_label = {
        "bm25": "BM25 (lex)",
        "splade": "SPLADE (ls)",
        "bge_base_v15": "BGE-base (dense)",
        "colbertv2": "ColBERTv2 (late-int)",
    }
    header = f"{'architecture':<22} {'RR':>7} {'hit@1':>6} {'hit@5':>6} {'hit@10':>7} {'Hit@1_flip':>11} {'Hit@5_flip':>11} {'Hit@10_flip':>12} {'RR_Std':>8}"
    log(header)
    log("-" * len(header))
    for name in arch_order:
        h = headline[name]
        v = volatility[name]
        h1 = f"{v['Hit@1_FlipRate_mean'] * 100:>10.2f}%" if v.get('Hit@1_FlipRate_mean') is not None else f"{'n/a':>11}"
        h5 = f"{v['Hit@5_FlipRate_mean'] * 100:>10.2f}%" if v.get('Hit@5_FlipRate_mean') is not None else f"{'n/a':>11}"
        h10 = f"{v['Hit@10_FlipRate_mean'] * 100:>11.2f}%" if v.get('Hit@10_FlipRate_mean') is not None else f"{'n/a':>12}"
        rrs = f"{v['RR_Std_mean']:.4f}" if v.get('RR_Std_mean') is not None else f"{'n/a':>8}"
        log(f"{arch_label[name]:<22} {h['RR_mean']:>7.4f} {h['hit1']:>6.4f} {h['hit5']:>6.4f} {h['hit10']:>7.4f} {h1} {h5} {h10} {rrs}")
    log(f"\n=== Stage 5 architecture coverage complete ({time.time() - log_start:.1f}s) ===")


if __name__ == "__main__":
    main()