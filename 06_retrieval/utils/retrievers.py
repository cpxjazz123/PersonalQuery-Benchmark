#!/usr/bin/env python3
"""
Stage 13: Base Retrieval Models

Contains all base retriever classes:
- BM25: Traditional sparse retrieval
- DenseRetriever: ANCE/MiniLM dense retrieval
- E5Retriever: E5-large-v2 dense retrieval
- BGERetriever: BGE-large-en dense retrieval
- ColBERTRetriever: Token-level late interaction
- HybridRetriever: BM25 + Dense score fusion hybrid retrieval
- TFIDFRetriever: TF-IDF baseline
- DirichletPriorRetriever: Query likelihood with Dirichlet smoothing
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
import threading
import pickle
import json
import math
import subprocess
from datetime import datetime
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import log_with_timestamp, build_document_text

# Global lock for thread-safe model inference
# Prevents CUDA/GIL deadlocks when multiple threads call model.encode() simultaneously
_model_inference_lock = threading.Lock()


def _resolve_local_hf_snapshot(repo_id: str) -> str:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise RuntimeError("HF_HOME must be set for offline Hugging Face loading")
    repo_cache_dir = os.path.join(hf_home, "models--" + repo_id.replace("/", "--"))
    ref_file = os.path.join(repo_cache_dir, "refs", "main")
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Hugging Face ref file not found for {repo_id}: {ref_file}")
    with open(ref_file, "r") as f:
        snapshot_hash = f.read().strip()
    snapshot_dir = os.path.join(repo_cache_dir, "snapshots", snapshot_hash)
    if not os.path.exists(snapshot_dir):
        raise FileNotFoundError(f"Hugging Face snapshot not found for {repo_id}: {snapshot_dir}")
    return snapshot_dir


def _load_local_sentence_transformer(repo_id: str, device: torch.device, label: str):
    model_path = _resolve_local_hf_snapshot(repo_id)
    log_with_timestamp(f"  Loading {label} model from local snapshot: {model_path}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path, device=device)
    log_with_timestamp(f"  Using device: {device}")
    tokenizer = getattr(model, "tokenizer", None)
    tokenizer_path = getattr(tokenizer, "name_or_path", model_path)
    log_with_timestamp(f"  Model path: {tokenizer_path}")
    log_with_timestamp(f"  HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    return model


def _require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for retrieval models in this codepath")
    return torch.device("cuda")

try:
    import bm25s
except ImportError:
    log_with_timestamp("WARNING: bm25s not installed. BM25 will not work properly.")
    bm25s = None


class BM25:
    """BM25 retrieval model using bm25s library with proper tokenization"""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.retriever = None
        self.doc_asins = []
        self.corpus = []
        self.corpus_to_asin = {}
        
    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build BM25 index using bm25s with proper tokenization"""
        log_with_timestamp("  Building BM25 index using bm25s...")
        
        if bm25s is None:
            raise RuntimeError("bm25s library is required but not installed")
        
        self.corpus = []
        self.doc_asins = []
        self.corpus_to_asin = {}
        
        for doc in documents:
            asin = doc.get('asin', '')
            self.doc_asins.append(asin)
            text = build_document_text(doc, all_metadata)
            self.corpus.append(text)
            self.corpus_to_asin[text] = asin
        
        log_with_timestamp(f"  Tokenizing {len(self.corpus)} documents...")
        tokenized_corpus = bm25s.tokenize(
            self.corpus,
            lower=True,
            stopwords="english",
            show_progress=False
        )
        
        self.retriever = bm25s.BM25(corpus=self.corpus)
        self.retriever.index(tokenized_corpus)
        
        log_with_timestamp(f"  BM25 index built with {len(self.corpus)} docs using bm25s")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 with bm25s library"""
        if self.retriever is None:
            return []
        
        tokenized_query = bm25s.tokenize(
            query,
            lower=True,
            stopwords="english",
            show_progress=False
        )
        
        results, scores = self.retriever.retrieve(tokenized_query, k=top_k)
        
        output = []
        retrieved_texts = results[0]
        scores_list = scores[0].tolist()
        
        for doc_text, score in zip(retrieved_texts, scores_list):
            asin = self.corpus_to_asin.get(doc_text, '')
            if asin:
                output.append((asin, float(score)))
        
        return output


class HybridRetriever:
    """Hybrid retrieval combining BM25 sparse scores with Dense embedding scores via weighted score fusion.

    Runs both BM25 and a dense retriever (default: E5-large-v2), normalizes their
    scores to [0, 1] via min-max scaling, and returns a weighted combination.

    Default weights: 0.4 BM25 + 0.6 Dense (empirically tuned for product search).
    """

    def __init__(self, bm25_weight: float = 0.4, dense_weight: float = 0.6,
                 dense_model_name: str = "intfloat/e5-large-v2"):
        """
        Args:
            bm25_weight: Weight for normalized BM25 scores (default 0.4).
            dense_weight: Weight for normalized Dense scores (default 0.6).
            dense_model_name: HuggingFace model ID for the dense retriever.
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.dense_model_name = dense_model_name
        self.bm25: BM25 = BM25()
        self.dense: "E5Retriever" = None  # type: ignore[assignment]
        self.doc_ids: List[str] = []
        self.is_fitted: bool = False

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Initialize BOTH BM25 and Dense (E5) indices from the same document set.

        Args:
            documents: List of document dicts, each must contain an 'asin' key.
            all_metadata: Optional metadata dict keyed by ASIN for enriched text.
        """
        log_with_timestamp("  Building Hybrid (BM25 + E5 Dense) index...")
        self.doc_ids = [doc.get('asin', '') for doc in documents]

        self.bm25.fit(documents, all_metadata)

        self.dense = E5Retriever(model_name=self.dense_model_name)
        self.dense.fit(documents, all_metadata)

        self.is_fitted = True
        log_with_timestamp(
            f"  Hybrid index built with {len(self.doc_ids)} docs "
            f"(weights: BM25={self.bm25_weight}, Dense={self.dense_weight})"
        )

    @staticmethod
    def _normalize_scores(scores: List[Tuple[str, float]]) -> Dict[str, float]:
        """Min-max normalize raw scores to [0, 1].

        Args:
            scores: List of (asin, raw_score) tuples.

        Returns:
            Dict mapping asin -> normalized score in [0, 1].
        """
        if not scores:
            return {}

        raw = [s for _, s in scores]
        min_s = min(raw)
        max_s = max(raw)
        score_range = max_s - min_s

        if score_range < 1e-12:
            # All scores identical -> uniform mid-point
            return {asin: 0.5 for asin, _ in scores}

        return {asin: (score - min_s) / score_range for asin, score in scores}

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Run both retrievers, fuse normalized scores, return merged top-k.

        Score fusion:
            combined = bm25_weight * norm_bm25 + dense_weight * norm_dense
        Missing results from either side receive 0.0 for that component.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (asin, fused_score) sorted descending by score.
        """
        if not self.is_fitted:
            return []

        fetch_k = max(top_k * 3, 50)
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        dense_results = self.dense.search(query, top_k=fetch_k)

        # Handle edge cases: one or both empty
        if not bm25_results and not dense_results:
            return []
        if not bm25_results:
            return dense_results[:top_k]
        if not dense_results:
            return bm25_results[:top_k]

        # Normalize each score set to [0, 1]
        bm25_norm = self._normalize_scores(bm25_results)
        dense_norm = self._normalize_scores(dense_results)

        # Union of all candidate ASINs
        all_asins = set(bm25_norm.keys()) | set(dense_norm.keys())

        # Weighted fusion (missing component -> 0.0)
        fused: List[Tuple[str, float]] = []
        for asin in all_asins:
            b_score = bm25_norm.get(asin, 0.0)
            d_score = dense_norm.get(asin, 0.0)
            combined = self.bm25_weight * b_score + self.dense_weight * d_score
            fused.append((asin, float(combined)))

        fused.sort(key=lambda x: -x[1])
        return fused[:top_k]


class DenseRetriever:
    """Dense retrieval using sentence-transformers"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None  # 存储所有元数据
        self.device = _require_cuda_device()

    def _get_model(self):
        if self.model is None:
            self.model = _load_local_sentence_transformer(self.model_name, self.device, "dense")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build dense index using enhanced document text"""
        log_with_timestamp("  Building dense index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata  # 保存元数据引用
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        embeddings = model.encode(texts, show_progress_bar=True, batch_size=2048)
        
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        
        if self.device.type == 'cuda':
            embeddings = embeddings.to(self.device)
        
        self.doc_embeddings = embeddings
        log_with_timestamp(f"  Dense index built with {len(self.doc_ids)} docs (device: {self.device})")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using dense embeddings

        Args:
            query: Query text

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode([query])

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = np.asarray(query_embedding.detach().tolist(), dtype=np.float32)

        return query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding

    def encode_queries(self, queries: List[str], batch_size: int = 1024) -> np.ndarray:
        """Encode multiple queries in batch for efficiency

        Args:
            queries: List of query texts
            batch_size: Number of queries to process at once

        Returns:
            numpy array of shape (len(queries), embedding_dim)
        """
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            embeddings = model.encode(
                queries,
                batch_size=batch_size,
                show_progress_bar=False
            )

        if isinstance(embeddings, torch.Tensor):
            embeddings = np.asarray(embeddings.detach().tolist(), dtype=np.float32)

        return embeddings

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using dense vectors with optimized top-k extraction"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode([query])

        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)
        else:
            query_embedding = query_embedding.to(self.device)

        doc_embeddings = self.doc_embeddings
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        
        results = [(self.doc_ids[idx], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices.tolist())]
        
        return results


class E5Retriever:
    """E5-large-v2: SOTA embedding model with instruction-based retrieval (支持滑动窗口，不截断)"""
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None  # 可能是 tensor 或 list of tensors (多窗口)
        self.doc_ids = []
        self.all_metadata = None
        self.device = _require_cuda_device()

        # 滑动窗口配置
        self.max_seq_length = 512  # E5 模型的最大长度
        self.window_stride = 256  # 窗口步长（重叠50%）

    def _get_model(self):
        if self.model is None:
            self.model = _load_local_sentence_transformer(self.model_name, self.device, "E5")
        return self.model

    def _add_instruction(self, text: str, is_query: bool = False) -> str:
        """E5 需要添加 instruction 前缀"""
        if is_query:
            return "query: " + text
        else:
            return "passage: " + text

    def _encode_text_with_sliding_window(self, text: str, add_prefix: bool = True):
        """
        使用滑动窗口编码文本，保证不截断

        Args:
            text: 输入文本
            add_prefix: 是否添加 passage: 前缀

        Returns:
            如果文本 <= max_seq_length: 单个 embedding [dim]
            如果文本 > max_seq_length: 多个窗口的 embeddings [num_windows, dim]
        """
        model = self._get_model()

        # 添加前缀
        if add_prefix:
            text_with_prefix = self._add_instruction(text, is_query=False)
        else:
            text_with_prefix = text

        # 使用 tokenizer 获取真实 token 数量（不截断）
        tokens = model.tokenizer(
            text_with_prefix,
            truncation=False,
            return_tensors='pt'
        )

        num_tokens = len(tokens['input_ids'][0])

        # 如果文本较短，直接编码
        if num_tokens <= self.max_seq_length:
            result = model.encode(
                [text_with_prefix],
                convert_to_tensor=True,
                show_progress_bar=False
            )
            return result[0]
        else:
            # 长文本：使用滑动窗口
            # 计算窗口数量
            num_windows = (num_tokens - self.max_seq_length) // self.window_stride + 1

            window_embeddings = []
            for i in range(num_windows):
                # 计算窗口的 token 范围
                start_token = i * self.window_stride
                end_token = min(start_token + self.max_seq_length, num_tokens)

                # 提取窗口的 token IDs
                window_tokens = {
                    'input_ids': tokens['input_ids'][0][start_token:end_token].unsqueeze(0),
                    'attention_mask': tokens['attention_mask'][0][start_token:end_token].unsqueeze(0)
                }

                # 解码回文本
                window_text = model.tokenizer.decode(
                    window_tokens['input_ids'][0],
                    skip_special_tokens=True
                )

                # 编码这个窗口
                window_embedding = model.encode(
                    [window_text],
                    convert_to_tensor=True,
                    show_progress_bar=False
                )[0]

                window_embeddings.append(window_embedding)

            # 返回所有窗口的 embeddings（堆叠成 tensor）
            return torch.stack(window_embeddings)

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """
        构建 E5 索引（使用滑动窗口，保证不截断）
        """
        from tqdm import tqdm
        log_with_timestamp("  Building E5-large-v2 index with sliding window (no truncation)...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        texts = [build_document_text(doc, all_metadata) for doc in documents]

        doc_embeddings_list = []
        single_window_indices = []
        multi_window_indices = []
        window_stats = {'single_window': 0, 'multi_window': 0, 'max_windows': 0}

        for i, text in tqdm(enumerate(texts), total=len(texts), desc="  E5 Encoding"):
            doc_emb = self._encode_text_with_sliding_window(text, add_prefix=True)

            if doc_emb.dim() == 1:
                window_stats['single_window'] += 1
                single_window_indices.append(i)
            else:
                window_stats['multi_window'] += 1
                multi_window_indices.append(i)
                num_windows = doc_emb.shape[0]
                window_stats['max_windows'] = max(window_stats['max_windows'], num_windows)

            doc_embeddings_list.append(doc_emb)

        self.single_window_indices = single_window_indices
        self.multi_window_indices = multi_window_indices

        if single_window_indices:
            single_embeddings = torch.stack([doc_embeddings_list[i] for i in single_window_indices])
            if self.device.type == 'cuda':
                single_embeddings = single_embeddings.to(self.device)
            self.single_window_embeddings = single_embeddings
        else:
            self.single_window_embeddings = None
        
        self.doc_embeddings = doc_embeddings_list

        log_with_timestamp(f"  E5 index built with {len(self.doc_ids)} docs:")
        log_with_timestamp(f"    - Single window: {window_stats['single_window']} docs")
        log_with_timestamp(f"    - Multi window: {window_stats['multi_window']} docs")
        log_with_timestamp(f"    - Max windows per doc: {window_stats['max_windows']}")
        log_with_timestamp(f"    - Device: {self.device}")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using E5 embeddings

        Args:
            query: Query text

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            query_with_prefix = self._add_instruction(query, is_query=True)
            query_embedding = model.encode(
                [query_with_prefix],
                convert_to_tensor=True
            )[0]

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = np.asarray(query_embedding.detach().tolist(), dtype=np.float32)

        return query_embedding

    def encode_queries(self, queries: List[str], batch_size: int = 1024) -> np.ndarray:
        """Encode multiple queries in batch for efficiency

        Args:
            queries: List of query texts
            batch_size: Number of queries to process at once

        Returns:
            numpy array of shape (len(queries), embedding_dim)
        """
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        queries_with_prefix = [self._add_instruction(q, is_query=True) for q in queries]

        with _model_inference_lock:
            embeddings = model.encode(
                queries_with_prefix,
                batch_size=batch_size,
                convert_to_tensor=True
            )

        if isinstance(embeddings, torch.Tensor):
            embeddings = np.asarray(embeddings.detach().tolist(), dtype=np.float32)

        return embeddings

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using E5 with batch processing for single-window documents"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            query_with_prefix = self._add_instruction(query, is_query=True)
            query_embedding = model.encode(
                [query_with_prefix],
                convert_to_tensor=True
            )[0]

        query_embedding = query_embedding.to(self.device)

        from sentence_transformers import util
        all_scores = [0.0] * len(self.doc_ids)
        
        if hasattr(self, 'single_window_embeddings') and self.single_window_embeddings is not None:
            batch_scores = util.cos_sim(query_embedding, self.single_window_embeddings)[0]
            for idx, doc_idx in enumerate(self.single_window_indices):
                all_scores[doc_idx] = batch_scores[idx].item()
        
        for i in self.multi_window_indices:
            doc_emb = self.doc_embeddings[i].to(self.device)
            window_scores = util.cos_sim(query_embedding, doc_emb)[0]
            all_scores[i] = window_scores.max().item()

        results = [(self.doc_ids[i], all_scores[i]) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


class BGERetriever:
    """BGE-large-en: SOTA embedding model for English retrieval (支持滑动窗口，不截断)"""
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None  # 可能是 tensor 或 list of tensors (多窗口)
        self.doc_ids = []
        self.all_metadata = None
        self.device = _require_cuda_device()

        # 滑动窗口配置
        self.max_seq_length = 512  # BGE 模型的最大长度
        self.window_stride = 256  # 窗口步长（重叠50%）

    def _get_model(self):
        if self.model is None:
            self.model = _load_local_sentence_transformer(self.model_name, self.device, "BGE")
        return self.model

    def _add_instruction(self, text: str, is_query: bool = False) -> str:
        """BGE 推荐添加 instruction 前缀"""
        if is_query:
            return "Represent this sentence for searching relevant passages: " + text
        else:
            return text

    def _encode_text_with_sliding_window(self, text: str, pool_windows: str = 'mean'):
        """
        使用滑动窗口编码文本，保证不截断

        Args:
            text: 输入文本（BGE 文档不需要前缀）
            pool_windows: 多窗口池化策略
              - 'mean': 对所有窗口 embeddings 求平均（默认，用于快速检索）
              - 'max': 对所有窗口 embeddings 求最大值
              - 'first': 仅使用第一个窗口
              - None: 返回所有窗口 embeddings（用于精确排名）

        Returns:
            总是返回单个 embedding [dim]
            （当 pool_windows 不为 None 时）
        """
        model = self._get_model()

        # 使用 tokenizer 获取真实 token 数量（不截断）
        tokens = model.tokenizer(
            text,
            truncation=False,
            return_tensors='pt'
        )

        num_tokens = len(tokens['input_ids'][0])

        # 如果文本较短，直接编码
        if num_tokens <= self.max_seq_length:
            embedding = model.encode(
                [text],
                convert_to_tensor=True,
                show_progress_bar=False
            )[0]
            return embedding
        else:
            # 长文本：使用滑动窗口
            # 计算窗口数量
            num_windows = (num_tokens - self.max_seq_length) // self.window_stride + 1

            window_embeddings = []
            for i in range(num_windows):
                # 计算窗口的 token 范围
                start_token = i * self.window_stride
                end_token = min(start_token + self.max_seq_length, num_tokens)

                # 提取窗口的 token IDs
                window_tokens = {
                    'input_ids': tokens['input_ids'][0][start_token:end_token].unsqueeze(0),
                    'attention_mask': tokens['attention_mask'][0][start_token:end_token].unsqueeze(0)
                }

                # 解码回文本
                window_text = model.tokenizer.decode(
                    window_tokens['input_ids'][0],
                    skip_special_tokens=True
                )

                # 编码这个窗口
                window_embedding = model.encode(
                    [window_text],
                    convert_to_tensor=True,
                    show_progress_bar=False
                )[0]

                window_embeddings.append(window_embedding)

            # 池化多个窗口的 embeddings
            stacked = torch.stack(window_embeddings)
            
            if pool_windows is None:
                return stacked
            elif pool_windows == 'mean':
                return torch.mean(stacked, dim=0)
            elif pool_windows == 'max':
                return torch.max(stacked, dim=0)[0]
            elif pool_windows == 'first':
                return stacked[0]
            else:
                return torch.mean(stacked, dim=0)

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build BGE index with FULLY BATCHED encoding - GPU maximized"""
        log_with_timestamp("  Building BGE-large-en index with FULLY BATCHED encoding...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # 直接批量编码（debug显示绝大多数文档 < 512 tokens，直接encode即可）
        # 注意：bge-large 模型较大，batch_size 需要适度以避免 CUDA OOM
        batch_size = 256
        log_with_timestamp(f"  Batch encoding {len(texts)} docs (batch_size={batch_size})...")

        all_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        self.doc_embeddings = all_embeddings

        log_with_timestamp(f"  BGE index built with {len(self.doc_ids)} docs:")
        log_with_timestamp(f"    - Embeddings shape: {self.doc_embeddings.shape} (device: {self.device})")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using BGE embeddings

        Args:
            query: Query text

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            query_with_prefix = self._add_instruction(query, is_query=True)
            query_embedding = model.encode(
                [query_with_prefix],
                convert_to_tensor=True
            )[0]

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = np.asarray(query_embedding.detach().tolist(), dtype=np.float32)

        return query_embedding

    def encode_queries(self, queries: List[str], batch_size: int = 1024) -> np.ndarray:
        """Encode multiple queries in batch for efficiency

        Args:
            queries: List of query texts
            batch_size: Number of queries to process at once

        Returns:
            numpy array of shape (len(queries), embedding_dim)
        """
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        queries_with_prefix = [self._add_instruction(q, is_query=True) for q in queries]

        with _model_inference_lock:
            embeddings = model.encode(
                queries_with_prefix,
                batch_size=batch_size,
                convert_to_tensor=True
            )

        if isinstance(embeddings, torch.Tensor):
            embeddings = np.asarray(embeddings.detach().tolist(), dtype=np.float32)

        return embeddings

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BGE with batched similarity computation (optimized with window pooling)"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()
        
        model = self._get_model()

        with _model_inference_lock:
            query_with_prefix = self._add_instruction(query, is_query=True)
            query_embedding = model.encode(
                [query_with_prefix],
                convert_to_tensor=True,
            )[0]

        query_embedding = query_embedding.to(self.device)

        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]

        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        
        results = [(self.doc_ids[idx], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices.tolist())]
        
        return results


COLBERTV2_MODEL_NAME = "colbert-ir/colbertv2.0"
COLBERTV2_NBITS = 2
COLBERTV2_TARGET_NUM_PARTITIONS = 65536
COLBERTV2_KMEANS_NITERS = 4
COLBERTV2_EXPERIMENT_NAME = "colbertv2_index"


def parse_cuda_version_text(version_text: str) -> Tuple[int, int]:
    marker = "release "
    marker_index = version_text.find(marker)
    if marker_index < 0:
        raise ValueError(f"Unable to parse CUDA release from nvcc output: {version_text}")

    version_part = version_text[marker_index + len(marker):].split(",", 1)[0].strip()
    pieces = version_part.split(".")
    if len(pieces) < 2:
        raise ValueError(f"Unable to parse CUDA major/minor from nvcc output: {version_text}")

    return int(pieces[0]), int(pieces[1])


def get_nvcc_cuda_version(nvcc_path: str) -> Tuple[int, int]:
    result = subprocess.run(
        [nvcc_path, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return parse_cuda_version_text(result.stdout)


def get_torch_cuda_version() -> Tuple[int, int]:
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError("PyTorch CUDA version is unavailable; ColBERTv2 indexing requires CUDA PyTorch.")

    pieces = cuda_version.split(".")
    if len(pieces) < 2:
        raise RuntimeError(f"Unable to parse PyTorch CUDA version: {cuda_version}")

    return int(pieces[0]), int(pieces[1])


def get_gcc_major_version(compiler_path: str) -> int:
    result = subprocess.run(
        [compiler_path, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    first_line = result.stdout.splitlines()[0]
    marker = "(GCC) "
    marker_index = first_line.find(marker)
    if marker_index < 0:
        raise ValueError(f"Unable to parse GCC version from compiler output: {first_line}")

    version_text = first_line[marker_index + len(marker):].split()[0]
    return int(version_text.split(".", 1)[0])


def candidate_cuda_roots_for_colbert() -> List[str]:
    candidate_roots = []

    for key in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(key)
        if root and root not in candidate_roots:
            candidate_roots.append(root)

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        inferred_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc_path)))
        if inferred_root not in candidate_roots:
            candidate_roots.append(inferred_root)

    usr_local = "/usr/local"
    if os.path.isdir(usr_local):
        for entry_name in sorted(os.listdir(usr_local)):
            if entry_name.startswith("cuda-"):
                root = os.path.join(usr_local, entry_name)
                if root not in candidate_roots:
                    candidate_roots.append(root)

    return candidate_roots


def select_cuda_toolkit_for_colbert_extension_build() -> None:
    torch_cuda_major, torch_cuda_minor = get_torch_cuda_version()
    candidate_roots = candidate_cuda_roots_for_colbert()
    if not candidate_roots:
        raise RuntimeError("No CUDA toolkit candidates found for ColBERT CUDA extension build.")

    compatible_roots = []
    rejected_roots = []
    for root in candidate_roots:
        nvcc_path = os.path.join(root, "bin", "nvcc")
        if not os.path.exists(nvcc_path):
            rejected_roots.append((root, "missing nvcc"))
            continue

        try:
            cuda_major, cuda_minor = get_nvcc_cuda_version(nvcc_path)
        except Exception as e:
            rejected_roots.append((root, f"nvcc version parse failed: {type(e).__name__}: {e}"))
            continue

        if cuda_major != torch_cuda_major:
            rejected_roots.append(
                (root, f"CUDA {cuda_major}.{cuda_minor} does not match PyTorch CUDA major {torch_cuda_major}")
            )
            continue

        compatible_roots.append((root, cuda_major, cuda_minor))

    if not compatible_roots:
        raise RuntimeError(
            "No CUDA toolkit with the same major version as PyTorch CUDA was found. "
            f"PyTorch CUDA={torch.version.cuda}; rejected candidates={rejected_roots}"
        )

    compatible_roots.sort(key=lambda item: (abs(item[2] - torch_cuda_minor), -item[2], item[0]))
    selected_root, selected_major, selected_minor = compatible_roots[0]

    os.environ["CUDA_HOME"] = selected_root
    os.environ["CUDA_PATH"] = selected_root
    selected_bin = os.path.join(selected_root, "bin")
    path_entries = [path for path in os.environ.get("PATH", "").split(":") if path]
    if selected_bin in path_entries:
        path_entries.remove(selected_bin)
    path_entries.insert(0, selected_bin)
    os.environ["PATH"] = ":".join(path_entries)

    from torch.utils import cpp_extension
    cpp_extension.CUDA_HOME = selected_root

    log_with_timestamp(
        f"[DEBUG] ColBERTv2: selected CUDA toolkit {selected_root} "
        f"(nvcc CUDA {selected_major}.{selected_minor}, PyTorch CUDA {torch.version.cuda})"
    )


def configure_host_compiler_for_colbert_extension_build() -> None:
    cc_path = "/usr/bin/gcc"
    cxx_path = "/usr/bin/g++"

    if not os.path.exists(cc_path):
        raise RuntimeError(f"Required CUDA host C compiler not found: {cc_path}")
    if not os.path.exists(cxx_path):
        raise RuntimeError(f"Required CUDA host C++ compiler not found: {cxx_path}")

    cc_major = get_gcc_major_version(cc_path)
    cxx_major = get_gcc_major_version(cxx_path)
    max_supported_major = 13
    if cc_major > max_supported_major:
        raise RuntimeError(f"{cc_path} GCC major version {cc_major} exceeds CUDA 12.5 support limit {max_supported_major}")
    if cxx_major > max_supported_major:
        raise RuntimeError(f"{cxx_path} GCC major version {cxx_major} exceeds CUDA 12.5 support limit {max_supported_major}")

    os.environ["CC"] = cc_path
    os.environ["CXX"] = cxx_path
    log_with_timestamp(
        f"[DEBUG] ColBERTv2: selected host compilers CC={cc_path} (GCC {cc_major}), "
        f"CXX={cxx_path} (GCC {cxx_major})"
    )


def should_skip_existing_file(path: str) -> bool:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        log_with_timestamp(f"[DEBUG] Skip existing file: {path}")
        return True
    return False


def write_colbertv2_collection_tsv(documents: List[Dict], all_metadata: Dict, collection_path: str) -> None:
    log_with_timestamp("[DEBUG] ColBERTv2: collection TSV writing started")
    log_with_timestamp(f"Writing ColBERTv2 collection TSV to {collection_path}...")
    with open(collection_path, 'w', encoding='utf-8') as f:
        for pid, doc in enumerate(documents):
            text = build_document_text(doc, all_metadata)
            text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
            if not text:
                raise ValueError(f"Empty document text for pid={pid}, asin={doc.get('asin', '')}")
            f.write(f"{pid}\t{text}\n")
            if (pid + 1) % 50000 == 0:
                log_with_timestamp(f"[DEBUG] ColBERTv2: wrote {pid + 1}/{len(documents)} rows")
    log_with_timestamp(f"[DEBUG] ColBERTv2: collection TSV writing complete, total_rows={len(documents)}")


def patch_colbert_encoding_progress() -> None:
    log_with_timestamp("[DEBUG] ColBERTv2: installing encoding progress hook")
    from colbert.indexing.collection_encoder import CollectionEncoder
    from colbert.utils.utils import batch

    def encode_passages_with_progress(self, passages):
        total_passages = len(passages)
        batch_size = self.config.index_bsize * 50
        total_batches = max(1, math.ceil(total_passages / batch_size))
        log_with_timestamp(
            f"[DEBUG] ColBERTv2: encoding hook active, total_passages={total_passages}, "
            f"batch_size={batch_size}, total_batches={total_batches}"
        )

        if total_passages == 0:
            return None, None

        from colbert.infra.run import Run
        import torch

        Run().print(f"#> Encoding {total_passages} passages..")

        with torch.inference_mode():
            embs, doclens = [], []
            for batch_idx, passages_batch in enumerate(batch(passages, batch_size), start=1):
                processed_after = min(batch_idx * batch_size, total_passages)
                pct_after = processed_after * 100.0 / total_passages
                log_with_timestamp(
                    f"[DEBUG] ColBERTv2: encoding batch {batch_idx}/{total_batches}, "
                    f"progress={processed_after}/{total_passages} ({pct_after:.2f}%)"
                )

                embs_, doclens_ = self.checkpoint.docFromText(
                    passages_batch,
                    bsize=self.config.index_bsize,
                    keep_dims="flatten",
                    showprogress=(not self.use_gpu),
                    pool_factor=self.config.pool_factor,
                    clustering_mode=self.config.clustering_mode,
                    protected_tokens=self.config.protected_tokens,
                )
                embs.append(embs_)
                doclens.extend(doclens_)

                if batch_idx == total_batches:
                    log_with_timestamp(
                        f"[DEBUG] ColBERTv2: encoding finished {processed_after}/{total_passages} passages "
                        f"({pct_after:.2f}%)"
                    )

            embs = torch.cat(embs)

        return embs, doclens

    CollectionEncoder.encode_passages = encode_passages_with_progress


def patch_colbert_indexing_params() -> None:
    log_with_timestamp(
        f"[DEBUG] ColBERTv2: overriding indexing params: "
        f"partitions={COLBERTV2_TARGET_NUM_PARTITIONS}, kmeans_niters={COLBERTV2_KMEANS_NITERS}"
    )
    from colbert.indexing.collection_indexer import CollectionIndexer
    import numpy as np

    original_train_kmeans = CollectionIndexer._train_kmeans

    def setup_with_fixed_partitions(self):
        if self.config.resume:
            if self._try_load_plan():
                if self.verbose > 1:
                    from colbert.infra.run import Run
                    Run().print_main(f"#> Loaded plan from {self.plan_path}:")
                    Run().print_main(f"#> num_chunks = {self.num_chunks}")
                    Run().print_main(f"#> num_partitions = {self.num_partitions}")
                    Run().print_main(f"#> num_embeddings_est = {self.num_embeddings_est}")
                    Run().print_main(f"#> avg_doclen_est = {self.avg_doclen_est}")
                return

        self.num_chunks = int(np.ceil(len(self.collection) / self.collection.get_chunksize()))
        sampled_pids = self._sample_pids()
        avg_doclen_est = self._sample_embeddings(sampled_pids)

        num_passages = len(self.collection)
        self.num_embeddings_est = num_passages * avg_doclen_est
        self.num_partitions = COLBERTV2_TARGET_NUM_PARTITIONS

        if self.verbose > 0:
            from colbert.infra.run import Run
            Run().print_main(f"Creating {self.num_partitions:,} partitions.")
            Run().print_main(f"*Estimated* {int(self.num_embeddings_est):,} embeddings.")

        self._save_plan()

    CollectionIndexer.setup = setup_with_fixed_partitions

    def train_kmeans_with_checkpoint(self, sample, shared_lists):
        centroids = original_train_kmeans(self, sample, shared_lists)

        checkpoint_dir = os.path.join(self.config.index_path_, "kmeans_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        centroids_path = os.path.join(
            checkpoint_dir,
            f"iter_{self.config.kmeans_niters:03d}_centroids.pt",
        )
        metadata_path = os.path.join(
            checkpoint_dir,
            f"iter_{self.config.kmeans_niters:03d}_metadata.json",
        )

        if not should_skip_existing_file(centroids_path):
            torch.save(centroids.cpu(), centroids_path)
            log_with_timestamp(f"[DEBUG] ColBERTv2: saved kmeans centroids checkpoint to {centroids_path}")

        if not should_skip_existing_file(metadata_path):
            metadata = {
                "kmeans_niters": self.config.kmeans_niters,
                "num_partitions": self.num_partitions,
                "num_centroids": int(centroids.size(0)),
                "dim": int(centroids.size(1)),
                "dtype": str(centroids.dtype),
                "created_at": datetime.now().isoformat(),
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            log_with_timestamp(f"[DEBUG] ColBERTv2: saved kmeans metadata checkpoint to {metadata_path}")

        return centroids

    CollectionIndexer._train_kmeans = train_kmeans_with_checkpoint


def validate_cuda_toolkit_for_colbert() -> None:
    log_with_timestamp("[DEBUG] ColBERTv2: validating CUDA toolkit for extension build")

    nvcc_path = shutil.which("nvcc")
    cuda_home = os.environ.get("CUDA_HOME")
    cuda_path = os.environ.get("CUDA_PATH")
    candidate_roots = []
    candidate_headers = []

    for root in (cuda_home, cuda_path):
        if root and root not in candidate_roots:
            candidate_roots.append(root)

    if nvcc_path:
        inferred_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc_path)))
        if inferred_root not in candidate_roots:
            candidate_roots.append(inferred_root)

    for root in candidate_roots:
        standard_header = os.path.join(root, "include", "cuda_runtime.h")
        if standard_header not in candidate_headers:
            candidate_headers.append(standard_header)

        targets_root = os.path.join(root, "targets")
        if os.path.isdir(targets_root):
            for target_name in sorted(os.listdir(targets_root)):
                target_header = os.path.join(
                    targets_root, target_name, "include", "cuda_runtime.h"
                )
                if target_header not in candidate_headers:
                    candidate_headers.append(target_header)

    existing_headers = [path for path in candidate_headers if os.path.exists(path)]

    log_with_timestamp(f"[DEBUG] ColBERTv2: CUDA_HOME={cuda_home}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: CUDA_PATH={cuda_path}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: nvcc_path={nvcc_path}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: candidate_cuda_roots={candidate_roots}")

    if not nvcc_path:
        raise RuntimeError(
            "ColBERT 索引构建需要可执行的 nvcc，但当前环境中未找到。"
            "请在 sbatch_wrapper --gpu 作业环境中加载完整 CUDA toolkit。"
        )

    if not candidate_roots:
        raise RuntimeError(
            "ColBERT 索引构建需要 CUDA toolkit，但当前环境中既没有 CUDA_HOME/CUDA_PATH，"
            "也无法从 nvcc 推断 CUDA 安装目录。"
        )

    if not existing_headers:
        raise RuntimeError(
            "ColBERT 在建索引时会即时编译 CUDA 扩展，但当前环境缺少 cuda_runtime.h。"
            f"已检查路径: {candidate_headers}。"
            "请确保作业环境加载了完整 CUDA toolkit，并正确设置 CUDA_HOME。"
            "如果使用 conda 打包的 CUDA，请确认 targets/<arch>/include/cuda_runtime.h 存在。"
        )

    log_with_timestamp(f"[DEBUG] ColBERTv2: detected cuda_runtime.h at {existing_headers[0]}")


def configure_cuda_env_for_colbert_extension_build() -> None:
    candidate_roots = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(key)
        if root and root not in candidate_roots:
            candidate_roots.append(root)

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        inferred_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc_path)))
        if inferred_root not in candidate_roots:
            candidate_roots.append(inferred_root)

    include_dirs = []

    def add_include_dir(path: str) -> None:
        if os.path.isdir(path) and path not in include_dirs:
            include_dirs.append(path)

    for root in candidate_roots:
        standard_include = os.path.join(root, "include")
        add_include_dir(standard_include)
        add_include_dir(os.path.join(standard_include, "cccl"))

        targets_root = os.path.join(root, "targets")
        if os.path.isdir(targets_root):
            for target_name in sorted(os.listdir(targets_root)):
                target_include = os.path.join(targets_root, target_name, "include")
                add_include_dir(target_include)
                add_include_dir(os.path.join(target_include, "cccl"))

    if not include_dirs:
        raise RuntimeError(
            "无法为 ColBERT CUDA 扩展编译配置头文件目录：未找到有效的 CUDA include 路径。"
        )

    for env_key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        existing = os.environ.get(env_key, "")
        paths = [path for path in existing.split(":") if path]
        for include_dir in reversed(include_dirs):
            if include_dir not in paths:
                paths.insert(0, include_dir)
        os.environ[env_key] = ":".join(paths)

    log_with_timestamp(f"[DEBUG] ColBERTv2: configured CUDA include dirs={include_dirs}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: CPATH={os.environ.get('CPATH')}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: CPLUS_INCLUDE_PATH={os.environ.get('CPLUS_INCLUDE_PATH')}")


def _get_torch_extension_build_root() -> str:
    from torch.utils.cpp_extension import get_default_build_root

    root_extensions_directory = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root_extensions_directory is None:
        root_extensions_directory = get_default_build_root()
        if torch.version.hip is not None:
            accelerator_str = f"rocm{torch.version.hip.replace('.', '')}"
        elif torch.version.cuda is not None:
            accelerator_str = f"cu{torch.version.cuda.replace('.', '')}"
        else:
            accelerator_str = "cpu"
        python_version = f"py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, 'abiflags', '')}"
        root_extensions_directory = os.path.join(root_extensions_directory, f"{python_version}_{accelerator_str}")

    return root_extensions_directory


def remove_stale_colbert_extension_locks() -> None:
    build_root = _get_torch_extension_build_root()
    extension_names = ("decompress_residuals_cpp", "packbits_cpp")

    for extension_name in extension_names:
        lock_path = os.path.join(build_root, extension_name, "lock")
        if not os.path.exists(lock_path):
            continue
        os.remove(lock_path)
        log_with_timestamp(f"[DEBUG] ColBERTv2: removed stale lock file {lock_path}")


def preflight_colbert_cuda_extension_build() -> None:
    log_with_timestamp("[DEBUG] ColBERTv2: CUDA extension preflight build started")
    remove_stale_colbert_extension_locks()

    include_candidates = []
    for env_key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        include_candidates.extend([path for path in os.environ.get(env_key, "").split(":") if path])

    thrust_header_candidates = [
        os.path.join(include_dir, "thrust", "complex.h")
        for include_dir in include_candidates
    ]
    existing_thrust_headers = [path for path in thrust_header_candidates if os.path.exists(path)]
    if not existing_thrust_headers:
        raise RuntimeError(
            "ColBERT CUDA 扩展预检查失败：未找到 thrust/complex.h。"
            f"已检查路径: {thrust_header_candidates}"
        )

    log_with_timestamp(f"[DEBUG] ColBERTv2: detected thrust header at {existing_thrust_headers[0]}")

    from colbert.indexing.codecs.residual import ResidualCodec

    try:
        ResidualCodec.try_load_torch_extensions(use_gpu=True)
    except Exception as e:
        log_with_timestamp(f"[ERROR] ColBERTv2: CUDA extension preflight failed: {type(e).__name__}: {e}")
        raise

    log_with_timestamp("[DEBUG] ColBERTv2: CUDA extension preflight build complete")


def get_colbertv2_output_root(doc_hash: str, category_config: Dict) -> str:
    if not doc_hash:
        raise ValueError("ColBERTv2 output root requires a non-empty document hash")
    return os.path.join(category_config['retriever_cache_dir'], f"colbertv2_{doc_hash}")


def get_colbertv2_index_path(output_root: str) -> str:
    return os.path.join(output_root, COLBERTV2_EXPERIMENT_NAME, "indexes", COLBERTV2_EXPERIMENT_NAME)


def get_colbertv2_cache_paths(doc_hash: str, category_config: Dict) -> Dict[str, str]:
    output_root = get_colbertv2_output_root(doc_hash, category_config)
    index_path = get_colbertv2_index_path(output_root)
    return {
        'index_root': output_root,
        'index_path': index_path,
        'index_metadata': os.path.join(index_path, "metadata.json"),
        'plan': os.path.join(index_path, "plan.json"),
        'centroids': os.path.join(index_path, "centroids.pt"),
        'ivf': os.path.join(index_path, "ivf.pid.pt"),
        'doc_ids': os.path.join(output_root, "doc_ids.pkl"),
        'metadata': os.path.join(output_root, "metadata.pkl"),
        'manifest': os.path.join(output_root, "build_manifest.json"),
    }


def cache_exists_colbertv2(doc_hash: str, category_config: Dict) -> bool:
    paths = get_colbertv2_cache_paths(doc_hash, category_config)
    required_files = ['index_metadata', 'plan', 'centroids', 'ivf', 'doc_ids', 'metadata', 'manifest']
    return all(os.path.exists(paths[key]) and os.path.getsize(paths[key]) > 0 for key in required_files)


def validate_colbertv2_cache(doc_hash: str, category_config: Dict, n_documents: int) -> Tuple[bool, str]:
    paths = get_colbertv2_cache_paths(doc_hash, category_config)

    required_files = ['index_metadata', 'plan', 'centroids', 'ivf', 'doc_ids', 'metadata', 'manifest']
    for key in required_files:
        if not os.path.exists(paths[key]):
            return False, f"Missing required ColBERTv2 file: {key} ({paths[key]})"
        if os.path.getsize(paths[key]) == 0:
            return False, f"Empty ColBERTv2 file: {key} ({paths[key]})"

    try:
        with open(paths['doc_ids'], 'rb') as f:
            doc_ids = pickle.load(f)

        if len(doc_ids) != n_documents:
            return False, f"doc_ids count ({len(doc_ids)}) != expected document count ({n_documents})"

        if len(doc_ids) != len(set(doc_ids)):
            duplicates = len(doc_ids) - len(set(doc_ids))
            return False, f"Found {duplicates} duplicate ColBERTv2 doc_ids"

        with open(paths['manifest'], 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        manifest_hash = manifest.get('document_hash')
        if manifest_hash != doc_hash:
            return False, f"Manifest document_hash ({manifest_hash}) != current document_hash ({doc_hash})"

        manifest_count = manifest.get('num_documents')
        if manifest_count != n_documents:
            return False, f"Manifest num_documents ({manifest_count}) != expected document count ({n_documents})"

        log_with_timestamp(
            f"[VALIDATE] colbertv2: index_path={paths['index_path']}, doc_ids={len(doc_ids)}, all checks passed ✓"
        )
        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def build_colbertv2_index(documents: List[Dict], all_metadata: Dict, output_root: str) -> str:
    log_with_timestamp("[DEBUG] ColBERTv2: build started")
    if not torch.cuda.is_available():
        raise RuntimeError("ColBERTv2 indexing requires CUDA")

    nranks = torch.cuda.device_count()
    if nranks < 1:
        raise RuntimeError("CUDA is available but no visible GPU ranks were detected")

    select_cuda_toolkit_for_colbert_extension_build()
    configure_host_compiler_for_colbert_extension_build()
    validate_cuda_toolkit_for_colbert()
    configure_cuda_env_for_colbert_extension_build()
    preflight_colbert_cuda_extension_build()

    log_with_timestamp("[DEBUG] ColBERTv2: importing ColBERT modules")
    try:
        from colbert.infra import Run, RunConfig, ColBERTConfig
        from colbert import Indexer
    except Exception as e:
        log_with_timestamp(f"[ERROR] ColBERTv2: import failed: {type(e).__name__}: {e}")
        raise

    log_with_timestamp(f"[DEBUG] ColBERTv2: model checkpoint={COLBERTV2_MODEL_NAME}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
    log_with_timestamp(f"[DEBUG] ColBERTv2: TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")

    os.makedirs(output_root, exist_ok=True)
    log_with_timestamp(f"[DEBUG] ColBERTv2: output directory ready: {output_root}")
    collection_path = os.path.join(output_root, "collection.tsv")
    write_colbertv2_collection_tsv(documents, all_metadata, collection_path)

    log_with_timestamp(f"[DEBUG] ColBERTv2: detected CUDA device count={nranks}")
    log_with_timestamp(
        f"Building ColBERTv2 index with model={COLBERTV2_MODEL_NAME}, nbits={COLBERTV2_NBITS}, "
        f"nranks={nranks}, partitions={COLBERTV2_TARGET_NUM_PARTITIONS}, "
        f"kmeans_niters={COLBERTV2_KMEANS_NITERS}"
    )
    log_with_timestamp(f"ColBERTv2 output root: {output_root}")

    patch_colbert_encoding_progress()
    patch_colbert_indexing_params()
    log_with_timestamp("[DEBUG] ColBERTv2: entering ColBERT Run context")
    with Run().context(
        RunConfig(
            nranks=nranks,
            experiment=COLBERTV2_EXPERIMENT_NAME,
            avoid_fork_if_possible=True,
            root=output_root,
        )
    ):
        config = ColBERTConfig(
            nbits=COLBERTV2_NBITS,
            kmeans_niters=COLBERTV2_KMEANS_NITERS,
            avoid_fork_if_possible=True,
            root=output_root,
        )
        log_with_timestamp(
            f"[DEBUG] ColBERTv2: config ready: nbits={config.nbits}, "
            f"kmeans_niters={config.kmeans_niters}, "
            f"avoid_fork_if_possible={config.avoid_fork_if_possible}, root={config.root}"
        )
        indexer = Indexer(checkpoint=COLBERTV2_MODEL_NAME, config=config)
        log_with_timestamp("[DEBUG] ColBERTv2: indexer constructed, starting index()")
        indexer.index(name=COLBERTV2_EXPERIMENT_NAME, collection=collection_path, overwrite=True)
        log_with_timestamp("[DEBUG] ColBERTv2: index() returned successfully")

    log_with_timestamp("[DEBUG] ColBERTv2: build complete")
    return collection_path


def save_colbertv2_build_artifacts(
    output_root: str,
    documents: List[Dict],
    all_metadata: Dict,
    doc_hash: str,
    collection_path: str,
    category_name: str,
) -> None:
    log_with_timestamp("[DEBUG] ColBERTv2: artifact saving started")
    doc_ids_path = os.path.join(output_root, "doc_ids.pkl")
    metadata_path = os.path.join(output_root, "metadata.pkl")
    manifest_path = os.path.join(output_root, "build_manifest.json")

    doc_ids = [doc.get('asin', '') for doc in documents]
    log_with_timestamp(f"[DEBUG] ColBERTv2: saving doc_ids to {doc_ids_path}")
    with open(doc_ids_path, 'wb') as f:
        pickle.dump(doc_ids, f)

    log_with_timestamp(f"[DEBUG] ColBERTv2: saving metadata to {metadata_path}")
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_metadata, f)

    manifest = {
        "category": category_name,
        "model_name": COLBERTV2_MODEL_NAME,
        "nbits": COLBERTV2_NBITS,
        "num_partitions": COLBERTV2_TARGET_NUM_PARTITIONS,
        "kmeans_niters": COLBERTV2_KMEANS_NITERS,
        "experiment": COLBERTV2_EXPERIMENT_NAME,
        "document_hash": doc_hash,
        "num_documents": len(documents),
        "output_root": output_root,
        "collection_path": collection_path,
        "created_at": datetime.now().isoformat(),
    }
    log_with_timestamp(f"[DEBUG] ColBERTv2: saving manifest to {manifest_path}")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log_with_timestamp(f"Saved ColBERTv2 doc_ids: {doc_ids_path}")
    log_with_timestamp(f"Saved ColBERTv2 metadata: {metadata_path}")
    log_with_timestamp(f"Saved ColBERTv2 manifest: {manifest_path}")
    log_with_timestamp("[DEBUG] ColBERTv2: artifact saving complete")


def build_colbertv2_retriever_index(
    documents: List[Dict],
    doc_hash: str,
    all_metadata: Dict,
    category_name: str,
    category_config: Dict,
) -> str:
    output_root = get_colbertv2_output_root(doc_hash, category_config)
    collection_path = build_colbertv2_index(documents, all_metadata, output_root)
    save_colbertv2_build_artifacts(output_root, documents, all_metadata, doc_hash, collection_path, category_name)

    if os.path.exists(collection_path):
        log_with_timestamp(f"[DEBUG] ColBERTv2: removing temporary collection file: {collection_path}")
        os.remove(collection_path)

    log_with_timestamp(f"ColBERTv2 index build completed under: {output_root}")
    return output_root


class ColBERTRetriever:
    """ColBERTv2: 使用官方 ColBERT Indexer/Searcher 进行高效索引构建和搜索

    使用官方 ColBERT 的 Indexer 构建量化索引 (2-bit)，大幅节省显存。
    索引文件保存到磁盘，搜索时加载索引进行 Late Interaction 搜索。
    """
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", nbits: int = 2):
        """
        Args:
            model_name: ColBERT 模型名称
            nbits: 量化位数，默认 2-bit（大幅节省存储空间）
        """
        self.model_name = model_name
        self.nbits = nbits
        self.model = None
        self.tokenizer = None
        self.doc_ids = []  # 原始 asin 列表
        self.all_metadata = None
        self.device = _require_cuda_device()

        # 索引相关
        self.index_path = None
        self.checkpoint_path = None
        self.searcher = None

        # 查询编码相关
        self.max_query_length = 512

        # ID 映射：integer_id -> asin
        self._pid_to_asin = {}
        self._asin_to_pid = {}

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading ColBERTv2 model: {self.model_name}")
            from transformers import AutoModel, AutoTokenizer
            model_path = _resolve_local_hf_snapshot(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
            self.model = self.model.to(self.device)
            self.model.eval()
        return self.model

    def _encode_text(self, text: str):
        """编码文本为 token-level embeddings"""
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state[0]

        attention_mask_expanded = attention_mask[0].unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings = token_embeddings * attention_mask_expanded

        return token_embeddings

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """
        使用 ColBERT 官方 Indexer 构建量化索引

        索引保存到磁盘，不占用显存。
        """
        log_with_timestamp("  Building ColBERTv2 index with official Indexer...")

        # 清理 GPU 缓存
        log_with_timestamp("  Clearing GPU cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 构建 ID 映射：integer pid <-> asin
        self._pid_to_asin = {i: asin for i, asin in enumerate(self.doc_ids)}
        self._asin_to_pid = {asin: i for i, asin in enumerate(self.doc_ids)}

        # 创建临时目录用于 ColBERT 索引
        import tempfile
        import shutil
        self._temp_dir = tempfile.mkdtemp(prefix="colbert_")
        doc_tsv_path = os.path.join(self._temp_dir, "doc.tsv")

        # 写入文档 TSV 格式（使用整数 pid）
        log_with_timestamp(f"  Writing documents to TSV: {doc_tsv_path}")
        texts = [build_document_text(doc, all_metadata) for doc in documents]
        with open(doc_tsv_path, 'w', encoding='utf-8') as f:
            for pid, text in enumerate(texts):
                # 清理文本：移除换行符、tab符和多余空白
                text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
                if not text:  # 如果文本为空，使用占位符
                    text = f"product {pid}"
                # ColBERT TSV 格式: pid\ttext (pid 必须是整数)
                f.write(f"{pid}\t{text}\n")

        # 使用 ColBERT 官方 Indexer
        from colbert.infra import Run, RunConfig, ColBERTConfig
        from colbert import Indexer
        from colbert.data import Collection
        from tqdm import tqdm
        import threading
        import time

        log_with_timestamp(f"  Building ColBERT index with nbits={self.nbits}...")

        # 设置索引路径
        self.index_path = os.path.join(self._temp_dir, "index")

        nranks = torch.cuda.device_count()
        if nranks < 1:
            raise RuntimeError("CUDA is required and at least one GPU must be visible")
        exp_name = "colbert_index"

        # 创建 tqdm 进度条来跟踪索引进度
        pbar = tqdm(total=100, desc="  ColBERT indexing", unit="%", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        # ColBERT 索引是同步的，使用线程更新进度（因为 ColBERT 内部不提供回调）
        progress_value = [0]
        stop_flag = [False]

        def update_progress():
            """模拟进度更新（实际进度由 ColBERT 内部控制）"""
            while not stop_flag[0]:
                time.sleep(5)
                if progress_value[0] < 85:  # 编码阶段进度
                    progress_value[0] += 3
                    pbar.update(3)
                elif progress_value[0] < 95:  # 聚类阶段
                    progress_value[0] += 2
                    pbar.update(2)

        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()

        try:
            with Run().context(RunConfig(nranks=nranks, experiment=exp_name, root=self._temp_dir)):
                config = ColBERTConfig(nbits=self.nbits, root=self._temp_dir)
                checkpoint_path = _resolve_local_hf_snapshot(self.model_name)
                indexer = Indexer(checkpoint=checkpoint_path, config=config)
                indexer.index(name=exp_name, collection=doc_tsv_path, overwrite=True)

            # 完成
            stop_flag[0] = True
            pbar.update(100 - pbar.n)
            pbar.close()
        except Exception as e:
            stop_flag[0] = True
            pbar.close()
            raise e

        log_with_timestamp(f"  ColBERT index built and saved to: {self.index_path}")
        log_with_timestamp(f"  Total documents indexed: {len(self.doc_ids)}")

        # 清理临时文件（保留索引）
        if os.path.exists(doc_tsv_path):
            os.remove(doc_tsv_path)

    def _get_searcher(self):
        """获取 ColBERT Searcher（延迟加载）"""
        if self.searcher is None:
            from colbert.infra import Run, RunConfig, ColBERTConfig
            from colbert import Searcher

            log_with_timestamp(f"  Loading ColBERT searcher from: {self.index_path}")

            exp_name = "colbert_index"
            with Run().context(RunConfig(experiment=exp_name, root=self._temp_dir)):
                config = ColBERTConfig(root=self._temp_dir)
                self.searcher = Searcher(index=exp_name, config=config)

        return self.searcher

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        使用 ColBERT 官方 Searcher 进行 Late Interaction 搜索
        """
        searcher = self._get_searcher()

        # 使用 ColBERT Searcher 搜索
        results = searcher.search(query, k=top_k)

        # 解析结果: [(integer_pid, rank, score), ...]
        # ColBERT 返回的是 integer pid，需要映射回 asin
        output = []
        for pid, rank, score in zip(results[0], results[1], results[2]):
            asin = self._pid_to_asin.get(pid, str(pid))
            output.append((asin, float(score)))

        # 按分数降序排序
        output.sort(key=lambda x: -x[1])
        return output[:top_k]


class TFIDFRetriever:
    """TF-IDF retrieval as a baseline with FAISS acceleration"""
    def __init__(self):
        self.doc_vectors = None
        self.feature_names = None
        self.doc_ids = []
        self.all_metadata = None
        self.faiss_index = None
        self._doc_vectors_dense = None

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build TF-IDF index with FAISS"""
        log_with_timestamp("  Building TF-IDF index with FAISS...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        import faiss
        import numpy as np

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # 构建 TF-IDF 向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),  # 使用 unigrams 和 bigrams
            sublinear_tf=True
        )

        self.doc_vectors = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # 转换为密集向量并进行L2归一化（用于余弦相似度）
        self._doc_vectors_dense = self.doc_vectors.toarray().astype('float32')
        # 行归一化：每个文档向量的模为1，余弦相似度 = 内积
        norms = np.linalg.norm(self._doc_vectors_dense, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8  # 避免除零
        self._doc_vectors_dense = self._doc_vectors_dense / norms

        # 创建FAISS索引（内积索引，因为向量已归一化）
        d = self._doc_vectors_dense.shape[1]  # 特征维度
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(self._doc_vectors_dense)

        log_with_timestamp(f"  TF-IDF index built with FAISS: {len(self.doc_ids)} docs, {d} features")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using TF-IDF with FAISS"""
        import numpy as np

        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')[0]
        # 归一化查询向量
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        query_vector = query_vector.reshape(1, -1)

        # 使用FAISS搜索
        scores, indices = self.faiss_index.search(query_vector, top_k)

        results = [(self.doc_ids[idx], float(scores[0][i])) for i, idx in enumerate(indices[0])]
        return results


class DirichletPriorRetriever:
    """Query Likelihood with Dirichlet Prior smoothing"""
    def __init__(self, mu: float = 2000):
        """
        Args:
            mu: Dirichlet prior 参数，控制文档先验的权重
                 较小的 mu (500-1000) 更信任文档，较大的 mu (2000-5000) 更信任集合语言模型
        """
        self.mu = mu
        self.doc_probs = []  # 每个文档的词概率分布
        self.collection_probs = {}  # 集合语言模型
        self.vocab = set()
        self.doc_ids = []
        self.all_metadata = None

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """构建 Dirichlet Prior 索引"""
        log_with_timestamp(f"  Building Dirichlet Prior index (μ={self.mu})...")

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # 使用增强的文档文本构建
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # Tokenize 所有文档
        tokenized_docs = [text.lower().split() for text in texts]

        # 构建词汇表
        self.vocab = set()
        for doc_tokens in tokenized_docs:
            self.vocab.update(doc_tokens)

        # 计算集合语言模型 P(w|C)
        from collections import Counter, defaultdict
        collection_counts = Counter()
        total_collection_tokens = 0

        for doc_tokens in tokenized_docs:
            collection_counts.update(doc_tokens)
            total_collection_tokens += len(doc_tokens)

        self.collection_probs = {
            word: count / total_collection_tokens
            for word, count in collection_counts.items()
        }

        # 计算每个文档的语言模型 P(w|D)
        self.doc_probs = []
        self.doc_lengths = []

        for doc_tokens in tokenized_docs:
            doc_length = len(doc_tokens)
            self.doc_lengths.append(doc_length)

            doc_counts = Counter(doc_tokens)
            doc_prob = {
                word: count / doc_length
                for word, count in doc_counts.items()
            }
            self.doc_probs.append(doc_prob)

        avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths)
        log_with_timestamp(f"  Dirichlet Prior index built with {len(self.doc_ids)} docs, "
                          f"{len(self.vocab)} vocab, avg doc length: {avg_doc_len:.1f}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用 Dirichlet Prior 搜索"""
        query_tokens = query.lower().split()

        avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths)

        scores = []
        for doc_id, doc_prob in enumerate(self.doc_probs):
            doc_len = self.doc_lengths[doc_id]

            # Dirichlet Prior 平滑
            # P(w|Q,D) = (|D| / (|D| + μ)) * P(w|D) + (μ / (|D| + μ)) * P(w|C)
            score = 0.0
            for token in query_tokens:
                # 文档概率
                p_w_d = doc_prob.get(token, 0.0)
                # 集合概率
                p_w_c = self.collection_probs.get(token, 1e-10)  # 小常数平滑

                # Dirichlet Prior 插值
                p_smoothed = (doc_len / (doc_len + self.mu)) * p_w_d + \
                            (self.mu / (doc_len + self.mu)) * p_w_c

                # 累加对数概率（避免下溢）
                if p_smoothed > 0:
                    score += np.log(p_smoothed)
                else:
                    score += np.log(1e-10)

            scores.append((self.doc_ids[doc_id], score))

        # 按分数降序排序
        results = sorted(scores, key=lambda x: -x[1])
        return results[:top_k]


class GritLMRetriever:
    """
    GritLM: GritLM-7B model for knowledge base retrieval.

    Uses the official gritlm package for encoding with proper instruction prefixes.
    Reference: STaRK implementation (https://github.com/snap-stanford/STaRK)
              https://github.com/allenai/gritlm

    Standard GritLM instruction format:
    - Query: "<|user|>\nGiven a product search query, retrieve relevant product descriptions.\n<|embed|>\n"
    - Document: "<|user|>\nGiven a product description, retrieve relevant web search queries.\n<|embed|>\n"
    """
    def __init__(self, model_name: str = "GritLM/GritLM-7B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = _require_cuda_device()
        # 标准 GritLM instruction 格式
        self.query_instruction = "Given a product search query, retrieve relevant product descriptions."
        self.doc_instruction = "Given a product description, retrieve relevant web search queries."
        # Pooling method (与 gritlm 包一致)
        self.pooling_method = 'mean'
        self.normalized = True

    def _get_query_instruction(self) -> str:
        """获取查询指令格式"""
        return f"<|user|>\n{self.query_instruction}\n<|embed|>\n"

    def _get_doc_instruction(self) -> str:
        """获取文档指令格式"""
        return f"<|user|>\n{self.doc_instruction}\n<|embed|>\n"

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading GritLM model: {self.model_name}")
            from gritlm import GritLM
            # 参考 gritlm 包的实现：不使用 local_files_only，让 transformers 自动处理缓存
            self.model = GritLM(
                self.model_name,
                mode='unified',
                pooling_method=self.pooling_method,
                normalized=self.normalized,
                torch_dtype=torch.bfloat16,
                is_inference=True,
            )
            log_with_timestamp(f"  GritLM loaded on {self.device}")
            log_with_timestamp(f"  Model name: {self.model_name}")
            log_with_timestamp(f"  HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
        return self.model

    def _encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text using GritLM with standard query instruction prefix."""
        model = self._get_model()
        embedding = model.encode_queries(
            [text],
            instruction=self._get_query_instruction(),
            max_length=max_length,
        )
        return torch.from_numpy(embedding[0])

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build GritLM index"""
        log_with_timestamp(f"  Building GritLM index with {len(documents)} docs...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # Build document texts
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # 参考 gritlm 包的 encode_corpus 方法进行批量编码
        log_with_timestamp(f"  Encoding {len(texts)} documents...")
        batch_size = 8  # 与 gritlm 包默认值 256 相比减小以适应 7B 模型

        all_embeddings = []
        for start_index in tqdm(range(0, len(texts), batch_size), desc="GritLM encoding"):
            batch_texts = texts[start_index:start_index + batch_size]
            batch_embeddings = model.encode_corpus(
                batch_texts,
                instruction=self._get_doc_instruction(),
                max_length=512,
            )
            all_embeddings.append(torch.from_numpy(batch_embeddings))

        self.doc_embeddings = torch.cat(all_embeddings, dim=0).float()
        log_with_timestamp(f"  GritLM index built with {len(self.doc_ids)} docs")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding vector"""
        embedding = self._encode_text(query)
        return np.asarray(embedding.detach().tolist(), dtype=np.float32)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using GritLM"""
        query_embedding = self._encode_text(query)
        query_embedding = query_embedding.unsqueeze(0).to(self.device)  # [1, hidden_dim]

        # Cosine similarity
        doc_embeddings = self.doc_embeddings.to(self.device)
        query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

        scores = torch.mm(query_norm, doc_norm.T)[0]

        results = [(self.doc_ids[i], scores[i].item()) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


class ANCERetriever:
    """
    ANCE (Approximate Nearest Neighbor Negative Contrastive Learning) retriever.

    ANCE uses a contrastive learning approach with negative sampling to learn
    high-quality dense embeddings for retrieval.

    Enhanced with:
    - Document instruction prefix (passage: ) for better query-document alignment
    - Sliding window encoding to avoid 512 token truncation
    """
    def __init__(self, model_name: str = "castorini/ance-msmarco-passage"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = _require_cuda_device()

        # 滑动窗口配置
        self.max_seq_length = 512
        self.window_stride = 256

        # 单窗口和多窗口文档的索引
        self.single_window_indices = []
        self.multi_window_indices = []
        self.single_window_embeddings = None

    def _get_model(self):
        if self.model is None:
            self.model = _load_local_sentence_transformer(self.model_name, self.device, "ANCE-compatible")
        return self.model

    def _add_instruction(self, text: str, is_query: bool = False) -> str:
        """ANCE 使用 query: 和 passage: instruction 前缀"""
        if is_query:
            return "query: " + text
        else:
            return "passage: " + text

    def _encode_text_with_sliding_window(self, text: str, add_prefix: bool = True):
        """
        使用滑动窗口编码文本，保证不截断

        Args:
            text: 输入文本
            add_prefix: 是否添加 passage: 前缀

        Returns:
            如果文本 <= max_seq_length: 单个 embedding [dim]
            如果文本 > max_seq_length: 多个窗口的 embeddings [num_windows, dim]
        """
        model = self._get_model()

        # 添加前缀
        if add_prefix:
            text_with_prefix = self._add_instruction(text, is_query=False)
        else:
            text_with_prefix = text

        # 使用 tokenizer 获取真实 token 数量（不截断）
        tokens = model.tokenizer(
            text_with_prefix,
            truncation=False,
            return_tensors='pt'
        )

        num_tokens = len(tokens['input_ids'][0])

        # 如果文本较短，直接编码
        if num_tokens <= self.max_seq_length:
            result = model.encode(
                [text_with_prefix],
                convert_to_tensor=True,
                show_progress_bar=False
            )
            return result[0]
        else:
            # 长文本：使用滑动窗口
            num_windows = (num_tokens - self.max_seq_length) // self.window_stride + 1

            window_embeddings = []
            for i in range(num_windows):
                start_token = i * self.window_stride
                end_token = min(start_token + self.max_seq_length, num_tokens)

                window_tokens = {
                    'input_ids': tokens['input_ids'][0][start_token:end_token].unsqueeze(0),
                    'attention_mask': tokens['attention_mask'][0][start_token:end_token].unsqueeze(0)
                }

                window_text = model.tokenizer.decode(
                    window_tokens['input_ids'][0],
                    skip_special_tokens=True
                )

                window_embedding = model.encode(
                    [window_text],
                    convert_to_tensor=True,
                    show_progress_bar=False
                )[0]

                window_embeddings.append(window_embedding)

            return torch.stack(window_embeddings)

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build ANCE index with BATCHED encoding for efficiency.

        Two-phase approach:
        1. Fast tokenization scan: separate single-window vs multi-window docs
        2. Batch encode: single-window docs batched, multi-window docs grouped by num_windows
        """
        log_with_timestamp("  Building ANCE-compatible index with sliding window (per-doc processing like E5/BGE)...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        # Per-doc sliding window processing (like E5/BGE)
        doc_embeddings_list = []
        single_window_indices = []
        multi_window_indices = []
        window_stats = {'single_window': 0, 'multi_window': 0, 'max_windows': 0}

        for i, text in tqdm(enumerate(texts), total=len(texts), desc="  ANCE Encoding"):
            doc_emb = self._encode_text_with_sliding_window(text, add_prefix=True)

            if doc_emb.dim() == 1:
                window_stats['single_window'] += 1
                single_window_indices.append(i)
            else:
                window_stats['multi_window'] += 1
                multi_window_indices.append(i)
                num_windows = doc_emb.shape[0]
                window_stats['max_windows'] = max(window_stats['max_windows'], num_windows)

            doc_embeddings_list.append(doc_emb)

        self.doc_embeddings = doc_embeddings_list
        self.single_window_indices = single_window_indices
        self.multi_window_indices = multi_window_indices

        if single_window_indices:
            single_embeddings = torch.stack([doc_embeddings_list[i] for i in single_window_indices])
            if self.device.type == 'cuda':
                single_embeddings = single_embeddings.to(self.device)
            self.single_window_embeddings = single_embeddings
        else:
            self.single_window_embeddings = None

        log_with_timestamp(f"  ANCE index built with {len(self.doc_ids)} docs:")
        log_with_timestamp(f"    - Single window: {window_stats['single_window']} docs")
        log_with_timestamp(f"    - Multi window: {window_stats['multi_window']} docs (max {window_stats['max_windows']} windows)")
        log_with_timestamp(f"    - Device: {self.device}")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using ANCE embeddings

        Args:
            query: Query text

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            query_with_prefix = self._add_instruction(query, is_query=True)
            query_embedding = model.encode(
                [query_with_prefix],
                convert_to_tensor=True
            )[0]

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = np.asarray(query_embedding.detach().tolist(), dtype=np.float32)

        return query_embedding

    def encode_queries(self, queries: List[str], batch_size: int = 1024) -> np.ndarray:
        """Encode multiple queries in batch for efficiency

        Args:
            queries: List of query texts
            batch_size: Number of queries to process at once

        Returns:
            numpy array of shape (len(queries), embedding_dim)
        """
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        queries_with_prefix = [self._add_instruction(q, is_query=True) for q in queries]

        with _model_inference_lock:
            embeddings = model.encode(
                queries_with_prefix,
                batch_size=batch_size,
                convert_to_tensor=True
            )

        if isinstance(embeddings, torch.Tensor):
            embeddings = np.asarray(embeddings.detach().tolist(), dtype=np.float32)

        return embeddings

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using ANCE with sliding window support"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            query_with_prefix = self._add_instruction(query, is_query=True)
            query_embedding = model.encode(
                [query_with_prefix],
                convert_to_tensor=True
            )[0]

        # Ensure query is on GPU for fast computation
        query_embedding = query_embedding.to(self.device)

        from sentence_transformers import util
        all_scores = [0.0] * len(self.doc_ids)

        # 单窗口文档：批量计算
        if hasattr(self, 'single_window_embeddings') and self.single_window_embeddings is not None:
            single_emb = self.single_window_embeddings
            # Move to GPU for computation if not already there
            single_emb = single_emb.to(self.device)
            batch_scores = util.cos_sim(query_embedding, single_emb)[0]
            for idx, doc_idx in enumerate(self.single_window_indices):
                all_scores[doc_idx] = batch_scores[idx].item()

        # 多窗口文档：取所有窗口的最大分数
        for i in self.multi_window_indices:
            doc_emb = self.doc_embeddings[i]
            # Move to GPU for computation if not already there
            doc_emb = doc_emb.to(self.device)
            window_scores = util.cos_sim(query_embedding, doc_emb)[0]
            all_scores[i] = window_scores.max().item()

        results = [(self.doc_ids[i], all_scores[i]) for i in range(len(self.doc_ids))]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


class STARRetriever:
    """
    STAR (STaRK) retriever for dense passage retrieval.
    
    Using BAAI/bge-base-en-v1.5 as a publicly available alternative to STAR.
    This model is also trained on large-scale retrieval data with contrastive learning.
    """
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = _require_cuda_device()

    def _get_model(self):
        if self.model is None:
            self.model = _load_local_sentence_transformer(self.model_name, self.device, "STAR-compatible")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build STAR index using enhanced document text"""
        log_with_timestamp("  Building STAR-compatible index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=512)
        log_with_timestamp(f"  STAR index built with {len(self.doc_ids)} docs")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query using STAR embeddings

        Args:
            query: Query text

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode(
                [query]
            )

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = np.asarray(query_embedding.detach().tolist(), dtype=np.float32)

        return query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding

    def encode_queries(self, queries: List[str], batch_size: int = 1024) -> np.ndarray:
        """Encode multiple queries in batch for efficiency

        Args:
            queries: List of query texts
            batch_size: Number of queries to process at once

        Returns:
            numpy array of shape (len(queries), embedding_dim)
        """
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            embeddings = model.encode(
                queries,
                batch_size=batch_size,
                show_progress_bar=False
            )

        if isinstance(embeddings, torch.Tensor):
            embeddings = np.asarray(embeddings.detach().tolist(), dtype=np.float32)

        return embeddings

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using STAR embeddings"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode(
                [query]
            )
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)
        
        doc_embeddings = self.doc_embeddings
        if isinstance(doc_embeddings, np.ndarray):
            doc_embeddings = torch.from_numpy(doc_embeddings).float().to(query_embedding.device)
        if torch.is_tensor(doc_embeddings):
            if doc_embeddings.device != query_embedding.device:
                doc_embeddings = doc_embeddings.to(query_embedding.device)
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Optimized: Use torch.topk instead of full sort (O(n log k) vs O(n log n))
        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        results = [(self.doc_ids[idx.item()], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices)]
        return results


class MiniLMRetriever:
    """
    MiniLM retriever using sentence-transformers/all-MiniLM-L6-v2.
    
    A lightweight, fast model good for retrieval tasks.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = _require_cuda_device()

    def _get_model(self):
        if self.model is None:
            self.model = _load_local_sentence_transformer(self.model_name, self.device, "MiniLM")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build MiniLM index using enhanced document text"""
        log_with_timestamp("  Building MiniLM index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=2048)
        log_with_timestamp(f"  MiniLM index built with {len(self.doc_ids)} docs")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding vector"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode(
                [query]
            )

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = np.asarray(query_embedding.detach().tolist(), dtype=np.float32)

        return query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding

    def encode_queries(self, queries: List[str], batch_size: int = 1024) -> np.ndarray:
        """Encode multiple queries in batch for efficiency

        Args:
            queries: List of query texts
            batch_size: Number of queries to process at once

        Returns:
            numpy array of shape (len(queries), embedding_dim)
        """
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            embeddings = model.encode(
                queries,
                batch_size=batch_size,
                show_progress_bar=False
            )

        if isinstance(embeddings, torch.Tensor):
            embeddings = np.asarray(embeddings.detach().tolist(), dtype=np.float32)

        return embeddings

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using MiniLM embeddings"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()

        model = self._get_model()

        with _model_inference_lock:
            query_embedding = model.encode([query])

        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)

        doc_embeddings = self.doc_embeddings
        if isinstance(doc_embeddings, np.ndarray):
            doc_embeddings = torch.from_numpy(doc_embeddings).float().to(query_embedding.device)
        if torch.is_tensor(doc_embeddings):
            if doc_embeddings.device != query_embedding.device:
                doc_embeddings = doc_embeddings.to(query_embedding.device)

        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Optimized: Use torch.topk instead of full sort (O(n log k) vs O(n log n))
        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        results = [(self.doc_ids[idx.item()], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices)]
        return results


class MultiQAMiniLMRetriever:
    """
    MultiQA-MiniLM retriever using sentence-transformers/multi-qa-MiniLM-L6-cos-v1.
    
    Optimized for QA tasks using cosine similarity, used in PBR project.
    """
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = _require_cuda_device()

    def _get_model(self):
        if self.model is None:
            self.model = _load_local_sentence_transformer(self.model_name, self.device, "MultiQA-MiniLM")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build MultiQA-MiniLM index using enhanced document text"""
        log_with_timestamp("  Building MultiQA-MiniLM index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=2048)
        log_with_timestamp(f"  MultiQA-MiniLM index built with {len(self.doc_ids)} docs")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding vector"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()
        
        model = self._get_model()
        
        with _model_inference_lock:
            query_embedding = model.encode([query])
        
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = np.asarray(query_embedding.detach().tolist(), dtype=np.float32)
        
        return query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using MultiQA-MiniLM embeddings"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()
        
        model = self._get_model()
        
        with _model_inference_lock:
            query_embedding = model.encode([query])
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)
        
        doc_embeddings = self.doc_embeddings
        if isinstance(doc_embeddings, np.ndarray):
            doc_embeddings = torch.from_numpy(doc_embeddings).float().to(query_embedding.device)
        if torch.is_tensor(doc_embeddings):
            if doc_embeddings.device != query_embedding.device:
                doc_embeddings = doc_embeddings.to(query_embedding.device)
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Optimized: Use torch.topk instead of full sort (O(n log k) vs O(n log n))
        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        results = [(self.doc_ids[idx.item()], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices)]
        return results


class MPNetRetriever:
    """
    MPNet retriever using sentence-transformers/all-mpnet-base-v2.
    
    A more powerful model than MiniLM, better retrieval quality.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = _require_cuda_device()

    def _get_model(self):
        if self.model is None:
            self.model = _load_local_sentence_transformer(self.model_name, self.device, "MPNet")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build MPNet index using enhanced document text"""
        log_with_timestamp("  Building MPNet index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        self.doc_embeddings = model.encode(texts, show_progress_bar=True, batch_size=2048)
        log_with_timestamp(f"  MPNet index built with {len(self.doc_ids)} docs")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding vector"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()
        
        model = self._get_model()
        
        with _model_inference_lock:
            query_embedding = model.encode([query])
        
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = np.asarray(query_embedding.detach().tolist(), dtype=np.float32)
        
        return query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using MPNet embeddings"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()
        
        model = self._get_model()
        
        with _model_inference_lock:
            query_embedding = model.encode([query])
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)
        
        doc_embeddings = self.doc_embeddings
        if isinstance(doc_embeddings, np.ndarray):
            doc_embeddings = torch.from_numpy(doc_embeddings).float().to(query_embedding.device)
        if torch.is_tensor(doc_embeddings):
            if doc_embeddings.device != query_embedding.device:
                doc_embeddings = doc_embeddings.to(query_embedding.device)
        
        from sentence_transformers import util
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Optimized: Use torch.topk instead of full sort (O(n log k) vs O(n log n))
        topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.doc_ids)))
        results = [(self.doc_ids[idx.item()], topk_values[i].item()) 
                   for i, idx in enumerate(topk_indices)]
        return results


class FAISSRetriever:
    """FAISS-accelerated dense retrieval with GPU support"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 use_gpu: bool = True, nlist: int = 100, nprobe: int = 10):
        self.model_name = model_name
        self.model = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = _require_cuda_device()
        self.use_gpu = use_gpu
        self.index = None
        
        self.nlist = nlist
        self.nprobe = nprobe

    def _get_model(self):
        if self.model is None:
            self.model = _load_local_sentence_transformer(self.model_name, self.device, "dense")
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build FAISS index with IVFFlat for fast retrieval"""
        log_with_timestamp("  Building FAISS index...")
        model = self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata
        texts = [build_document_text(doc, all_metadata) for doc in documents]

        embeddings = model.encode(texts, show_progress_bar=True, batch_size=2048)
        embeddings = embeddings.astype('float32')
        
        embedding_dim = embeddings.shape[1]
        log_with_timestamp(f"  Embedding dimension: {embedding_dim}")
        
        import faiss

        log_with_timestamp(f"  Creating FAISS index (nlist={self.nlist}, nprobe={self.nprobe})...")

        quantizer = faiss.IndexFlatIP(embedding_dim)
        self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)

        log_with_timestamp("  Training FAISS index...")
        self.index.train(embeddings)

        log_with_timestamp("  Adding embeddings to FAISS index...")
        self.index.add(embeddings)
        self.index.nprobe = self.nprobe

        log_with_timestamp("  Moving FAISS index to GPU...")
        res = faiss.StandardGpuResources()
        self.index_gpu = faiss.index_cpu_to_gpu(res, 0, self.index)
        log_with_timestamp("  ✓ FAISS GPU index ready")

        log_with_timestamp(f"  FAISS index built with {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using FAISS index"""
        if not hasattr(self, 'device'):
            self.device = _require_cuda_device()
        
        model = self._get_model()
        
        with _model_inference_lock:
            query_embedding = model.encode([query])
        
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-10)
        
        if not self.use_gpu or not hasattr(self, 'index_gpu'):
            raise RuntimeError("FAISS GPU index is not initialized")

        distances, indices = self.index_gpu.search(query_embedding, k=min(top_k + 10, len(self.doc_ids)))

        results = [(self.doc_ids[int(idx)], float(dist))
                  for idx, dist in zip(indices[0], distances[0]) if idx != -1]
        return results[:top_k]


class CachedRetriever:
    """Wrapper that uses pre-computed query embeddings from cache - GPU optimized"""

    def __init__(self, base_retriever, cache_file: str = None):
        self.base_retriever = base_retriever
        self.cache = {}
        self._doc_embeddings_cached = None  # 缓存的 doc embeddings tensor
        self._doc_embeddings_device = None  # 记录 device

        if cache_file and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            log_with_timestamp(f"  [CachedRetriever] Loaded {len(self.cache)} cached queries")

    def _get_doc_embeddings_gpu(self):
        """获取 GPU 上的 doc embeddings（带缓存，避免重复创建tensor）"""
        # 快速路径：如果已有缓存，直接返回
        if self._doc_embeddings_cached is not None:
            return self._doc_embeddings_cached

        # 获取 base_retriever 的 embeddings
        doc_embeddings = self.base_retriever.doc_embeddings

        device = torch.device('cuda')
        if isinstance(doc_embeddings, np.ndarray):
            self._doc_embeddings_cached = torch.from_numpy(doc_embeddings).float().to(device)
        elif isinstance(doc_embeddings, torch.Tensor):
            if doc_embeddings.device.type != 'cuda':
                self._doc_embeddings_cached = doc_embeddings.to(device)
            else:
                self._doc_embeddings_cached = doc_embeddings
        else:
            self._doc_embeddings_cached = torch.from_numpy(np.array(doc_embeddings)).float().to(device)
        self._doc_embeddings_device = device

        return self._doc_embeddings_cached

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using cached embedding"""
        if query in self.cache:
            query_embedding = self.cache[query]
            if isinstance(query_embedding, np.ndarray):
                query_embedding = torch.from_numpy(query_embedding).float()

            # 获取 GPU 上的 doc embeddings
            doc_embeddings = self._get_doc_embeddings_gpu()

            # 确保 query embedding 在同一设备上
            device = self._doc_embeddings_device
            if query_embedding.device != device:
                query_embedding = query_embedding.to(device)

            # 归一化并计算余弦相似度（与 GritLMRetriever.search() 一致）
            query_norm = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
            doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

            # GPU 矩阵乘法计算相似度
            scores = torch.mm(query_norm, doc_norm.T)[0]

            # 取 top k
            topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(self.base_retriever.doc_ids)))

            return [(self.base_retriever.doc_ids[idx.item()], topk_values[i].item())
                    for i, idx in enumerate(topk_indices)]

        raise RuntimeError("Query embedding is missing from cache")

    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """批量搜索 - 更高效的处理多个查询"""
        if not queries:
            return []

        # 收集所有需要计算的查询
        uncached_queries = [q for q in queries if q not in self.cache]
        if uncached_queries:
            raise RuntimeError(f"{len(uncached_queries)} queries are missing from cache")

        # 获取 GPU 上的 doc embeddings
        doc_embeddings = self._get_doc_embeddings_gpu()
        device = self._doc_embeddings_device

        # 批量转换 query embeddings
        query_embeddings = []
        for q in queries:
            q_emb = self.cache[q]
            if isinstance(q_emb, np.ndarray):
                q_emb = torch.from_numpy(q_emb).float()
            query_embeddings.append(q_emb)

        # Stack and move to GPU
        query_tensor = torch.stack(query_embeddings).to(device)

        # 归一化并计算余弦相似度（与 GritLMRetriever.search() 一致）
        query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
        doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

        # GPU 批量矩阵乘法
        scores = torch.mm(query_norm, doc_norm.T)

        # 批量取 top k
        results = []
        for i, q in enumerate(queries):
            topk_values, topk_indices = torch.topk(scores[i], k=min(top_k, len(self.base_retriever.doc_ids)))
            results.append([(self.base_retriever.doc_ids[idx.item()], topk_values[j].item())
                          for j, idx in enumerate(topk_indices)])

        return results


class SPLADERetriever:
    """SPLADE++: Sparse Retrieval with BERT-based term weighting

    SPLADE++ uses BERT to compute importance weights for each term,
    producing a high-dimensional sparse vector representation.
    """
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.doc_ids = []
        self.all_metadata = None
        self.device = _require_cuda_device()
        self.doc_vectors = None  # Sparse vectors stored as dicts: {term_idx: weight}

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading SPLADE++ model: {self.model_name}")
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            model_path = _resolve_local_hf_snapshot(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True)
            self.model = self.model.half()  # Use FP16 to reduce memory
            self.model = self.model.to(self.device)
            self.model.eval()
            log_with_timestamp(f"  SPLADE++ model loaded (FP16)")

    def _encode_text(self, text: str) -> Dict[str, float]:
        """Encode text to sparse vector (term -> weight)"""
        import torch.nn.functional as F

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

            # SPLADE++: ReLU(log(1 + x)) + max pooling
            # Apply ReLU and log to get sparse weights
            weights = F.relu(logits)
            weights = torch.log(1 + weights)

            # Max pooling over sequence length dimension
            weights, _ = weights.max(dim=1)  # [batch, vocab_size]

            # Convert to sparse dict (only keep non-zero values)
            sparse_vec = {}
            for idx, w in enumerate(weights[0]):
                if w.item() > 0.01:  # Threshold for sparsity
                    sparse_vec[idx] = w.item()

            return sparse_vec

    def encode_query(self, query: str) -> Dict[str, float]:
        """Encode query to sparse vector for retrieval

        Args:
            query: Query text

        Returns:
            Dict mapping term_idx -> weight (sparse representation)
        """
        self._get_model()
        return self._encode_text(query)

    def encode_queries(self, queries: List[str], batch_size: int = 1024) -> List[Dict[str, float]]:
        """Encode multiple queries in batch for efficiency

        Args:
            queries: List of query texts
            batch_size: Number of queries to process at once

        Returns:
            List of dicts mapping term_idx -> weight (sparse representation)
        """
        self._get_model()
        import torch.nn.functional as F

        results = []
        n_batches = (len(queries) + batch_size - 1) // batch_size
        threshold = 0.01

        for batch_idx in tqdm(range(n_batches), desc="SPLADE Enc"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(queries))
            batch_queries = queries[batch_start:batch_end]

            inputs = self.tokenizer(
                batch_queries,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # [batch, seq_len, vocab_size]

                weights = F.relu(logits)
                weights = torch.log(1 + weights)
                weights, _ = weights.max(dim=1)  # [batch, vocab_size]

                # Efficient sparse vector construction - process each query
                for b in range(weights.shape[0]):
                    w_b = weights[b]
                    mask_b = w_b > threshold
                    # Use nonzero directly with as_tuple=True for efficiency
                    indices = mask_b.nonzero(as_tuple=True)[0]
                    vals = w_b[mask_b]
                    # Build dict efficiently using GPU tensors only
                    sparse_vec = dict(zip(indices.tolist(), vals.tolist()))
                    results.append(sparse_vec)

        return results

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """Build SPLADE++ index"""
        log_with_timestamp("  Building SPLADE++ index...")
        self._get_model()

        self.doc_ids = [doc.get('asin', '') for doc in documents]
        self.all_metadata = all_metadata

        # Batch processing for GPU efficiency
        import torch.nn.functional as F
        batch_size = 128  # Increased with FP16
        doc_vectors = []

        # Process documents in batches, building texts on-the-fly
        n_batches = (len(documents) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(n_batches), desc="  SPLADE Encoding"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(documents))
            batch_docs = documents[batch_start:batch_end]

            # Build texts for this batch
            batch_texts = [build_document_text(doc, all_metadata) for doc in batch_docs]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # [batch, seq_len, vocab_size]

                # SPLADE++: ReLU(log(1 + x)) + max pooling
                weights = F.relu(logits)
                weights = torch.log(1 + weights)
                # Max pooling over sequence length dimension: [batch, vocab_size]
                weights, _ = weights.max(dim=1)

                # Convert each row in batch to sparse dict efficiently
                # Only transfer non-zero terms from GPU to CPU (massive speedup)
                threshold = 0.01
                mask = weights > threshold
                indices = mask.nonzero(as_tuple=False)  # [num_nonzero, 2] (batch_idx, vocab_idx)
                nonzero_values = weights[mask]  # [num_nonzero]

                # Group by batch index
                batch_indices = indices[:, 0]  # batch idx for each nonzero term
                vocab_indices = indices[:, 1]  # vocab idx for each nonzero term

                # Split into per-document sparse vectors on CPU
                for b in range(weights.shape[0]):
                    doc_mask = batch_indices == b
                    doc_vocab = vocab_indices[doc_mask].tolist()
                    doc_vals = nonzero_values[doc_mask].tolist()
                    doc_vectors.append(dict(zip(doc_vocab, doc_vals)))

            if (batch_end) % 10000 == 0:
                log_with_timestamp(f"  SPLADE Encoding: {batch_end}/{len(documents)}")

        self.doc_vectors = doc_vectors
        log_with_timestamp(f"  SPLADE++ index built with {len(self.doc_ids)} docs")

    def _encode_query(self, query: str) -> Dict[str, float]:
        """Encode query to sparse vector"""
        self._get_model()
        return self._encode_text(query)

    def search(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Search using SPLADE++ sparse vectors with dot product

        Uses inverted index for efficient sparse computation:
        - Build inverted index (term -> [(doc_idx, weight)]) on first call
        - Subsequent calls reuse the cached index
        """
        if self.doc_vectors is None:
            log_with_timestamp("  ERROR: SPLADE index not built!")
            return [[] for _ in queries]

        from collections import defaultdict

        # Build inverted index on first call (lazy initialization)
        if not hasattr(self, '_inverted_index') or self._inverted_index is None:
            log_with_timestamp(f"  [SPLADE] Building inverted index for {len(self.doc_vectors)} docs...")
            self._inverted_index: Dict[int, List[Tuple[int, float]]] = {}
            for doc_idx, d_vec in enumerate(self.doc_vectors):
                for term_id, weight in d_vec.items():
                    if term_id not in self._inverted_index:
                        self._inverted_index[term_id] = []
                    self._inverted_index[term_id].append((doc_idx, weight))
            log_with_timestamp(f"  [SPLADE] Inverted index built with {len(self._inverted_index)} terms")

        # Encode all queries
        query_vectors = [self._encode_query(q) for q in queries]

        results = []
        for q_vec in query_vectors:
            # Compute scores using inverted index (only docs with matching terms)
            scores = defaultdict(float)
            for term_id, q_weight in q_vec.items():
                if term_id in self._inverted_index:
                    for doc_idx, d_weight in self._inverted_index[term_id]:
                        scores[doc_idx] += q_weight * d_weight

            # Sort and return top k
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            results.append([(self.doc_ids[doc_idx], score) for doc_idx, score in sorted_scores[:top_k]])

        return results
