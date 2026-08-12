#!/usr/bin/env python3
"""
Build all retriever indices for full-scale evaluation.
Extracts index-building logic from main evaluation script.
Only builds indices, does NOT evaluate.
"""

import os
import sys
import pickle
import hashlib
import threading
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime

# 确保 HF_HOME 和 HF_HUB_CACHE 指向正确的缓存目录
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
if "HF_HUB_CACHE" not in os.environ:
    os.environ["HF_HUB_CACHE"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"

# ColBERTv2 checkpoint is not fully cached in this environment, so allow online resolution.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Add utils path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import utils

log_with_timestamp = utils.log_with_timestamp
load_product_metadata = utils.load_product_metadata

# Import retriever utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from utils import retrievers

# ============ 配置加载 ============
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_category_config

CATEGORY_NAME = "Pet_Supplies"
CAT_CONFIG = get_category_config(CATEGORY_NAME)
COLBERTV2_TORCH_EXTENSIONS_BASE_DIR = "/home/wlia0047/ar57_scratch/wenyu/torch_extensions"


def get_colbert_torch_extension_arch_tag() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("ColBERTv2 index building requires a CUDA GPU for torch extension arch detection")
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def prepare_colbert_torch_extensions_dir() -> str:
    ext_dir = os.path.join(
        COLBERTV2_TORCH_EXTENSIONS_BASE_DIR,
        f"colbertv2_cuda125_{get_colbert_torch_extension_arch_tag()}"
    )
    os.makedirs(ext_dir, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = ext_dir
    os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "True"
    log_with_timestamp(f"[BOOT] ColBERT TORCH_EXTENSIONS_DIR = {ext_dir}")
    return ext_dir


COLBERTV2_TORCH_EXTENSIONS_DIR = prepare_colbert_torch_extensions_dir()

def load_fullscale_metadata(metadata_file: str) -> Dict:
    """Load full metadata"""
    log_with_timestamp(f"Loading metadata from {metadata_file}...")
    metadata, _ = load_product_metadata(metadata_file, None)
    return metadata


def build_fullscale_documents(category: str, metadata: Dict) -> Tuple[List[Dict], Set[str]]:
    """Build full-scale document set"""
    log_with_timestamp(f"Building {len(metadata)} documents from metadata...")
    
    documents = []
    asins = set()
    
    for idx, (asin, meta) in enumerate(metadata.items()):
        if (idx + 1) % 50000 == 0:
            log_with_timestamp(f"  Processed {idx + 1}/{len(metadata)}")
        
        doc = meta.copy()
        doc['asin'] = asin
        documents.append(doc)
        asins.add(asin)
    
    log_with_timestamp(f"Built document list: {len(documents)} documents")
    return documents, asins


def compute_document_hash(documents: List[Dict]) -> str:
    """Compute hash of document set to detect changes"""
    doc_ids = sorted([doc.get('asin', '') for doc in documents])
    hash_input = '|'.join(doc_ids)
    return hashlib.md5(hash_input.encode()).hexdigest()


def get_cache_paths(retriever_name: str, doc_hash: str, cache_dir: str) -> Dict[str, str]:
    """Get cache file paths for a retriever"""
    DENSE_RETRIEVERS = ['minilm', 'star', 'e5', 'bge', 'gritlm', 'ance']
    # ColBERT 使用 token-level embeddings，需要特殊处理，不使用简单的 numpy 数组
    COLBERT_RETRIEVERS = ['colbert']
    SPARSE_RETRIEVERS = ['bm25', 'splade']

    if retriever_name in DENSE_RETRIEVERS:
        base_path = os.path.join(cache_dir, f"{retriever_name}_{doc_hash}")
        return {
            'config': f"{base_path}_config.pkl",
            'embeddings': f"{base_path}_embeddings.npy",
            'doc_ids': f"{base_path}_doc_ids.pkl",
            'metadata': f"{base_path}_metadata.pkl",
        }
    elif retriever_name == 'colbertv2':
        return retrievers.get_colbertv2_cache_paths(doc_hash, CAT_CONFIG)
    elif retriever_name in COLBERT_RETRIEVERS:
        # ColBERT 使用 pickle 保存（因为是 token-level 可变长 embeddings）
        return {
            'pickle': os.path.join(cache_dir, f"{retriever_name}_{doc_hash}.pkl")
        }
    else:
        return {
            'pickle': os.path.join(cache_dir, f"{retriever_name}_{doc_hash}.pkl")
        }


def cache_exists(retriever_name: str, doc_hash: str, cache_dir: str) -> bool:
    """Check if retriever cache already exists"""
    DENSE_RETRIEVERS = ['minilm', 'star', 'e5', 'bge', 'gritlm', 'ance']
    COLBERT_RETRIEVERS = ['colbert']

    if retriever_name in DENSE_RETRIEVERS:
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        return os.path.exists(paths['config']) and os.path.exists(paths['embeddings'])
    elif retriever_name == 'colbertv2':
        return retrievers.cache_exists_colbertv2(doc_hash, CAT_CONFIG)
    elif retriever_name in COLBERT_RETRIEVERS:
        # ColBERT 使用 pickle 保存
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        return os.path.exists(paths['pickle'])
    else:
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        return os.path.exists(paths['pickle'])


def validate_retriever_cache(retriever_name: str, doc_hash: str, cache_dir: str, n_documents: int) -> Tuple[bool, str]:
    """Validate retriever cache integrity

    Returns:
        (is_valid, error_message)
    """
    DENSE_RETRIEVERS = ['minilm', 'star', 'e5', 'bge', 'gritlm', 'ance']
    COLBERT_RETRIEVERS = ['colbert']

    log_with_timestamp(f"[VALIDATE] Checking cache integrity for {retriever_name}...")

    if retriever_name in DENSE_RETRIEVERS:
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)

        # 检查文件是否存在
        required_files = ['config', 'embeddings', 'doc_ids', 'metadata']
        for key in required_files:
            if key not in paths or not os.path.exists(paths[key]):
                return False, f"Missing required file: {key}"

        # 检查文件是否为空
        for key in required_files:
            if os.path.getsize(paths[key]) == 0:
                return False, f"Empty file: {key}"

        # 检查 embeddings 和 doc_ids 数量是否匹配
        try:
            embeddings = np.load(paths['embeddings'], mmap_mode='r')
            n_embeddings = embeddings.shape[0]

            with open(paths['doc_ids'], 'rb') as f:
                doc_ids = pickle.load(f)
            n_doc_ids = len(doc_ids)

            if n_embeddings != n_doc_ids:
                return False, f"Embeddings count ({n_embeddings}) != doc_ids count ({n_doc_ids})"

            if n_embeddings != n_documents:
                return False, f"Embeddings count ({n_embeddings}) != expected document count ({n_documents})"

            # 检查 doc_ids 是否有重复
            if len(doc_ids) != len(set(doc_ids)):
                duplicates = len(doc_ids) - len(set(doc_ids))
                return False, f"Found {duplicates} duplicate doc_ids"

            log_with_timestamp(f"[VALIDATE] {retriever_name}: embeddings={n_embeddings}, doc_ids={n_doc_ids}, all checks passed ✓")
            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    elif retriever_name == 'colbertv2':
        return retrievers.validate_colbertv2_cache(doc_hash, CAT_CONFIG, n_documents)

    elif retriever_name in COLBERT_RETRIEVERS:
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)

        if not os.path.exists(paths['pickle']):
            return False, f"Missing pickle file"

        if os.path.getsize(paths['pickle']) == 0:
            return False, f"Empty pickle file"

        try:
            with open(paths['pickle'], 'rb') as f:
                data = pickle.load(f)

            # 检查是否为有效对象
            if data is None:
                return False, "Pickle data is None"

            log_with_timestamp(f"[VALIDATE] {retriever_name}: pickle file valid ✓")
            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    else:
        # BM25 等其他检索器
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)

        if not os.path.exists(paths['pickle']):
            return False, f"Missing pickle file"

        if os.path.getsize(paths['pickle']) == 0:
            return False, f"Empty pickle file"

        try:
            with open(paths['pickle'], 'rb') as f:
                data = pickle.load(f)

            # 检查是否有 search 方法
            if not hasattr(data, 'search'):
                return False, f"BM25 object missing 'search' method"

            log_with_timestamp(f"[VALIDATE] {retriever_name}: pickle file valid ✓")
            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"


def _normalize_embeddings_for_save(embeddings, retriever_name: str) -> np.ndarray:
    """
    Normalize embeddings for saving, handling variable-length cases (e.g., E5 multi-window).
    
    For mixed-shape embeddings (some 1D, some 2D), average-pool multi-window embeddings
    to create uniform 2D shape (n_docs, embedding_dim).
    
    Args:
        embeddings: List or array of embeddings
        retriever_name: Name of retriever (for logging)
    
    Returns:
        Normalized numpy array with shape (n_docs, embedding_dim)
    """
    log_with_timestamp(f"[DEBUG] _normalize_embeddings_for_save called for {retriever_name}")
    log_with_timestamp(f"[DEBUG] embeddings type: {type(embeddings)}")
    if isinstance(embeddings, list):
        log_with_timestamp(f"[DEBUG] embeddings is list with {len(embeddings)} items")
    
    if not isinstance(embeddings, list):
        # Already a tensor or array
        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32)
        elif hasattr(embeddings, 'detach'):
            return np.asarray(embeddings.detach().tolist(), dtype=np.float32)
        else:
            return embeddings.numpy().astype(np.float32)
    
    # Handle list of embeddings (possibly with mixed shapes)
    normalized = []
    has_multiwindow = False
    
    for i, emb in enumerate(embeddings):
        # Convert to numpy
        log_with_timestamp(f"[DEBUG] Processing embedding {i}: type={type(emb)}")
        if hasattr(emb, 'detach'):
            emb_np = np.asarray(emb.detach().tolist(), dtype=np.float32)
        elif isinstance(emb, np.ndarray):
            emb_np = emb
        else:
            emb_np = emb.numpy()
        
        log_with_timestamp(f"[DEBUG] emb[{i}] shape={emb_np.shape}, ndim={emb_np.ndim}")
        
        # Handle multi-window embeddings (2D) vs single-window (1D)
        if emb_np.ndim == 2:
            # Multi-window: average-pool across windows
            has_multiwindow = True
            emb_pooled = emb_np.mean(axis=0)  # [num_windows, dim] -> [dim]
            log_with_timestamp(f"[DEBUG] emb[{i}] multi-window: {emb_np.shape} -> {emb_pooled.shape}")
            normalized.append(emb_pooled)
        elif emb_np.ndim == 1:
            # Single-window: keep as is
            log_with_timestamp(f"[DEBUG] emb[{i}] single-window: kept as {emb_np.shape}")
            normalized.append(emb_np)
        else:
            raise ValueError(f"Unexpected embedding shape: {emb_np.shape}")
    
    if has_multiwindow and retriever_name == 'e5':
        log_with_timestamp(f"  [INFO] E5: Applied average pooling to multi-window embeddings for uniform shape")
    
    # Stack into (n_docs, embedding_dim)
    log_with_timestamp(f"[DEBUG] Stacking {len(normalized)} normalized embeddings...")
    embeddings_np = np.stack(normalized, axis=0).astype(np.float32)
    log_with_timestamp(f"[DEBUG] Final embeddings shape: {embeddings_np.shape}")
    return embeddings_np


def save_retriever_to_cache(retriever_name: str, doc_hash: str, retriever: object, cache_dir: str) -> bool:
    """Save retriever to disk cache. Returns True if successful."""
    DENSE_RETRIEVERS = ['minilm', 'star', 'e5', 'bge', 'gritlm', 'ance']
    COLBERT_RETRIEVERS = ['colbert']
    SPARSE_RETRIEVERS = ['bm25', 'splade']

    log_with_timestamp(f"[DEBUG] save_retriever_to_cache called for {retriever_name}")

    try:
        if retriever_name in DENSE_RETRIEVERS:
            log_with_timestamp(f"[CACHE_SAVE] Saving {retriever_name} with separated embeddings...")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
            log_with_timestamp(f"[DEBUG] Paths: {paths}")

            log_with_timestamp(f"[DEBUG] Checking for doc_embeddings...")
            if hasattr(retriever, 'doc_embeddings') and retriever.doc_embeddings is not None:
                log_with_timestamp(f"[DEBUG] doc_embeddings found, normalizing...")
                embeddings = retriever.doc_embeddings
                log_with_timestamp(f"[DEBUG] embeddings type: {type(embeddings)}")

                # Use robust normalization that handles variable-length embeddings
                log_with_timestamp(f"[DEBUG] Calling _normalize_embeddings_for_save...")
                embeddings_np = _normalize_embeddings_for_save(embeddings, retriever_name)
                log_with_timestamp(f"[DEBUG] Normalization complete, shape: {embeddings_np.shape}")

                log_with_timestamp(f"[DEBUG] Saving embeddings to {paths['embeddings']}...")
                np.save(paths['embeddings'], embeddings_np)
                log_with_timestamp(f"[DEBUG] Embeddings saved successfully")
                size_gb = embeddings_np.nbytes / (1024**3)
                log_with_timestamp(f"  → Embeddings: {paths['embeddings']} ({size_gb:.2f}GB)")
                log_with_timestamp(f"  → Shape: {embeddings_np.shape}")

                log_with_timestamp(f"[DEBUG] Clearing doc_embeddings from memory...")
                retriever.doc_embeddings = None

            # 删除 model（对于 GritLM 必须在 pickle 之前删除，因为其底层 MistralForCausalLM 无法 pickle）
            log_with_timestamp(f"[DEBUG] Clearing model from retriever before pickle...")
            if hasattr(retriever, 'model') and retriever.model is not None:
                del retriever.model
                retriever.model = None
                log_with_timestamp(f"[DEBUG] Model cleared from retriever")

            log_with_timestamp(f"[DEBUG] Saving config to {paths['config']}...")
            with open(paths['config'], 'wb') as f:
                pickle.dump(retriever, f)
            log_with_timestamp(f"  → Config: {paths['config']}")
            log_with_timestamp(f"[DEBUG] Config saved")

            if hasattr(retriever, 'doc_ids'):
                log_with_timestamp(f"[DEBUG] Saving doc_ids...")
                with open(paths['doc_ids'], 'wb') as f:
                    pickle.dump(retriever.doc_ids, f)
                log_with_timestamp(f"[DEBUG] doc_ids saved")

            if hasattr(retriever, 'all_metadata'):
                log_with_timestamp(f"[DEBUG] Saving metadata...")
                with open(paths['metadata'], 'wb') as f:
                    pickle.dump(retriever.all_metadata, f)
                log_with_timestamp(f"[DEBUG] metadata saved")

            log_with_timestamp(f"[DEBUG] Dense retriever save complete")

            # 释放 GPU 模型权重和缓存
            log_with_timestamp(f"[MEMORY] Releasing {retriever_name} model from GPU...")
            if hasattr(retriever, 'model') and retriever.model is not None:
                del retriever.model
                retriever.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log_with_timestamp(f"[MEMORY] GPU memory released for {retriever_name}")

            return True
        elif retriever_name in COLBERT_RETRIEVERS:
            # ColBERT 使用 token-level embeddings，需要特殊处理
            log_with_timestamp(f"[CACHE_SAVE] Saving {retriever_name} with token-level embeddings (as pickle)...")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)

            # ColBERT embeddings are list of tensors (possibly nested for multi-window)
            # 保留 GPU 张量后保存
            if hasattr(retriever, 'doc_embeddings') and retriever.doc_embeddings is not None:
                log_with_timestamp(f"[DEBUG] Keeping ColBERT embeddings on GPU before saving...")
                doc_embeddings_gpu = []
                for i, emb in enumerate(retriever.doc_embeddings):
                    if isinstance(emb, list):
                        # Multi-window: list of tensors
                        doc_embeddings_gpu.append([w.detach() for w in emb])
                    else:
                        # Single tensor
                        doc_embeddings_gpu.append(emb.detach())
                retriever.doc_embeddings = doc_embeddings_gpu
                log_with_timestamp(f"[DEBUG] ColBERT embeddings kept on GPU")

            log_with_timestamp(f"[DEBUG] Saving ColBERT to {paths['pickle']}...")
            with open(paths['pickle'], 'wb') as f:
                pickle.dump(retriever, f)

            log_with_timestamp(f"[DEBUG] File saved, getting size...")
            size_mb = os.path.getsize(paths['pickle']) / (1024 * 1024)
            log_with_timestamp(f"  → {paths['pickle']} ({size_mb:.1f}MB)")
            log_with_timestamp(f"[DEBUG] ColBERT save complete")

            # 释放 GPU 模型权重和缓存
            log_with_timestamp(f"[MEMORY] Releasing {retriever_name} model from GPU...")
            if hasattr(retriever, 'model') and retriever.model is not None:
                del retriever.model
                retriever.model = None
            if hasattr(retriever, 'tokenizer') and retriever.tokenizer is not None:
                del retriever.tokenizer
                retriever.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log_with_timestamp(f"[MEMORY] GPU memory released for {retriever_name}")

            return True
        elif retriever_name in SPARSE_RETRIEVERS:
            # BM25 和 SPLADE 都使用 pickle 保存
            log_with_timestamp(f"[CACHE_SAVE] Saving {retriever_name} (sparse retriever)...")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
            log_with_timestamp(f"[DEBUG] Saving to {paths['pickle']}...")

            # SPLADE 需要在 pickle 前删除模型（FP16 CUDA tensor 无法 pickle）
            # 但保留 doc_vectors（已转换为 CPU float dict）
            if retriever_name == 'splade' and hasattr(retriever, 'model') and retriever.model is not None:
                log_with_timestamp(f"[DEBUG] Clearing SPLADE model/tokenizer before pickle...")
                if hasattr(retriever, 'model') and retriever.model is not None:
                    del retriever.model
                    retriever.model = None
                if hasattr(retriever, 'tokenizer') and retriever.tokenizer is not None:
                    del retriever.tokenizer
                    retriever.tokenizer = None
                log_with_timestamp(f"[DEBUG] SPLADE model/tokenizer cleared, doc_vectors preserved")

            with open(paths['pickle'], 'wb') as f:
                pickle.dump(retriever, f)

            log_with_timestamp(f"[DEBUG] File saved, getting size...")
            size_mb = os.path.getsize(paths['pickle']) / (1024 * 1024)
            log_with_timestamp(f"  → {paths['pickle']} ({size_mb:.1f}MB)")
            log_with_timestamp(f"[DEBUG] Sparse retriever save complete")
            return True
        else:
            log_with_timestamp(f"[CACHE_SAVE] Saving {retriever_name}...")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
            log_with_timestamp(f"[DEBUG] Saving to {paths['pickle']}...")

            with open(paths['pickle'], 'wb') as f:
                pickle.dump(retriever, f)

            log_with_timestamp(f"[DEBUG] File saved, getting size...")
            size_mb = os.path.getsize(paths['pickle']) / (1024 * 1024)
            log_with_timestamp(f"  → {paths['pickle']} ({size_mb:.1f}MB)")
            log_with_timestamp(f"[DEBUG] Retriever save complete")
            return True
    except Exception as e:
        log_with_timestamp(f"[ERROR] Error saving {retriever_name}: {type(e).__name__}: {e}")
        import traceback
        log_with_timestamp(traceback.format_exc())
        return False


def build_retriever(retriever_name: str, documents: List[Dict], doc_hash: str, cache_dir: str, all_metadata: Dict = None) -> Tuple[bool, str]:
    """Build and save retriever to cache. Returns (success, cache_path_or_error)"""
    log_with_timestamp(f"[DEBUG] build_retriever: Creating {retriever_name}...")
    try:
        if retriever_name == 'colbertv2':
            if all_metadata is None:
                raise ValueError("ColBERTv2 build requires all_metadata")
            retrievers.preflight_colbert_cuda_extension_build()
            output_root = retrievers.build_colbertv2_retriever_index(
                documents,
                doc_hash,
                all_metadata,
                CATEGORY_NAME,
                CAT_CONFIG,
            )
            return True, output_root

        log_with_timestamp(f"[DEBUG] Creating retriever instance...")
        if retriever_name == 'bm25':
            retriever = retrievers.BM25()
            log_with_timestamp(f"[DEBUG] BM25 instance created")
        elif retriever_name == 'bge':
            retriever = retrievers.BGERetriever()
            log_with_timestamp(f"[DEBUG] BGERetriever instance created")
        elif retriever_name == 'e5':
            retriever = retrievers.E5Retriever()
            log_with_timestamp(f"[DEBUG] E5Retriever instance created")
        elif retriever_name == 'minilm':
            retriever = retrievers.MiniLMRetriever()
            log_with_timestamp(f"[DEBUG] MiniLMRetriever instance created")
        elif retriever_name == 'star':
            retriever = retrievers.STARRetriever()
            log_with_timestamp(f"[DEBUG] STARRetriever instance created")
        elif retriever_name == 'gritlm':
            retriever = retrievers.GritLMRetriever()
            log_with_timestamp(f"[DEBUG] GritLMRetriever instance created")
        elif retriever_name == 'ance':
            retriever = retrievers.ANCERetriever()
            log_with_timestamp(f"[DEBUG] ANCERetriever instance created")
        elif retriever_name == 'colbert':
            retriever = retrievers.ColBERTRetriever()
            log_with_timestamp(f"[DEBUG] ColBERTRetriever instance created")
        elif retriever_name == 'splade':
            retriever = retrievers.SPLADERetriever()
            log_with_timestamp(f"[DEBUG] SPLADERetriever instance created")
        else:
            return False, f"Unknown retriever type: {retriever_name}"
        
        log_with_timestamp(f"[DEBUG] Calling retriever.fit() on {len(documents)} documents...")
        retriever.fit(documents, all_metadata)
        log_with_timestamp(f"[DEBUG] retriever.fit() completed")
        
        log_with_timestamp(f"[DEBUG] Calling save_retriever_to_cache()...")
        if not save_retriever_to_cache(retriever_name, doc_hash, retriever, cache_dir):
            log_with_timestamp(f"[DEBUG] save_retriever_to_cache() returned False")
            return False, "Failed to save cache"
        
        log_with_timestamp(f"[DEBUG] save_retriever_to_cache() succeeded")
        paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
        cache_path = paths.get('config') or paths.get('pickle', '')
        return True, cache_path
    except Exception as e:
        import traceback
        log_with_timestamp(f"[ERROR] Exception in build_retriever: {type(e).__name__}: {e}")
        log_with_timestamp(traceback.format_exc())
        return False, f"{type(e).__name__}: {e}"


def main():
    """Main: Build all retriever indices"""
    setup_logging()
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("BUILD ALL RETRIEVER INDICES - STARTING")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"[DEBUG] Python version: {__import__('sys').version}")
    log_with_timestamp(f"[DEBUG] Current time: {__import__('datetime').datetime.now()}")
    
    category = CATEGORY_NAME
    log_with_timestamp(f"[DEBUG] Category: {category}")

    # Load full-scale metadata
    metadata_file = CAT_CONFIG['metadata_cache_file']
    log_with_timestamp(f"[DEBUG] Metadata file path: {metadata_file}")
    log_with_timestamp(f"[DEBUG] Metadata file exists: {os.path.exists(metadata_file)}")

    if os.path.exists(metadata_file):
        log_with_timestamp(f"[DEBUG] Loading metadata from cache...")
        try:
            with open(metadata_file, 'rb') as f:
                log_with_timestamp(f"[DEBUG] Opened metadata file successfully")
                metadata = pickle.load(f)
                log_with_timestamp(f"[DEBUG] Metadata loaded: {len(metadata)} items")
        except Exception as e:
            log_with_timestamp(f"[ERROR] Failed to load metadata: {type(e).__name__}: {e}")
            import traceback
            log_with_timestamp(traceback.format_exc())
            raise
    else:
        log_with_timestamp("[DEBUG] Metadata cache not found, loading from raw data...")
        raw_metadata_file = CAT_CONFIG['raw_corpus_file']
        log_with_timestamp(f"[DEBUG] Raw metadata file: {raw_metadata_file}")
        metadata = load_fullscale_metadata(raw_metadata_file)
    
    # Build documents
    log_with_timestamp(f"[DEBUG] Starting document building...")
    documents, asins = build_fullscale_documents(category, metadata)
    log_with_timestamp(f"[DEBUG] Document building complete")
    log_with_timestamp(f"Total documents: {len(documents)}, Total ASINs: {len(asins)}")
    
    # Compute document hash for cache keys
    log_with_timestamp(f"[DEBUG] Computing document hash...")
    doc_hash = compute_document_hash(documents)
    cache_dir = CAT_CONFIG['retriever_cache_dir']
    log_with_timestamp(f"[DEBUG] Creating cache directory...")
    os.makedirs(cache_dir, exist_ok=True)
    log_with_timestamp(f"Document hash: {doc_hash}")
    log_with_timestamp(f"Cache directory: {cache_dir}")
    
    # Define retrievers to build (按参数量从小到大排序: minilm < star < e5 < bge < gritlm < ance)
    DENSE_RETRIEVERS = [
        'minilm', 'star', 'e5', 'bge',
        # 'gritlm',
        'ance',
    ]
    COLBERT_RETRIEVERS = ['colbertv2']  # 使用 ColBERTv2 原生 late-interaction 索引
    SPARSE_RETRIEVERS = ['bm25', 'splade']
    ALL_RETRIEVERS = DENSE_RETRIEVERS + COLBERT_RETRIEVERS + SPARSE_RETRIEVERS

    log_with_timestamp(f"\nBuilding {len(ALL_RETRIEVERS)} retrievers:")
    log_with_timestamp(f"  Dense: {DENSE_RETRIEVERS}")
    log_with_timestamp(f"  ColBERT: {COLBERT_RETRIEVERS}")
    log_with_timestamp(f"  Sparse: {SPARSE_RETRIEVERS}")
    
    # Build each retriever
    log_with_timestamp(f"[DEBUG] Starting retriever build loop...")
    start_time = datetime.now()
    results = {}
    
    for retriever_name in ALL_RETRIEVERS:
        log_with_timestamp(f"[DEBUG] === Processing retriever: {retriever_name} ===")
        log_with_timestamp(f"\n[BUILD] {retriever_name}")
        log_with_timestamp(f"[DEBUG] Checking cache for {retriever_name}...")
        
        if cache_exists(retriever_name, doc_hash, cache_dir):
            log_with_timestamp(f"[CACHE_EXISTS] {retriever_name} cache already exists")
            paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
            if retriever_name in DENSE_RETRIEVERS:
                log_with_timestamp(f"  → {paths['config']}")
                log_with_timestamp(f"  → {paths['embeddings']}")
            elif retriever_name in COLBERT_RETRIEVERS:
                log_with_timestamp(f"  → {paths['index_root']}")
                log_with_timestamp(f"  → {paths['index_path']}")
            else:
                log_with_timestamp(f"  → {paths['pickle']}")

            # 验证缓存完整性
            is_valid, error_msg = validate_retriever_cache(retriever_name, doc_hash, cache_dir, len(documents))
            if not is_valid:
                log_with_timestamp(f"[CACHE_INVALID] {retriever_name} cache validation failed: {error_msg}")
                log_with_timestamp(f"[CACHE_INVALID] Will rebuild {retriever_name}...")
                # 缓存无效，需要重建
                should_rebuild = True
            else:
                log_with_timestamp(f"[CACHE_VALID] {retriever_name} cache integrity verified ✓")
                results[retriever_name] = {'status': 'cached', 'time': 0}
                log_with_timestamp(f"[DEBUG] {retriever_name} skipped (cached)")
                should_rebuild = False
        else:
            should_rebuild = True

        if should_rebuild:
            # 在构建 ColBERT 之前，显式清理 GPU 缓存
            if retriever_name in COLBERT_RETRIEVERS and torch.cuda.is_available():
                log_with_timestamp(f"[MEMORY] ColBERT build requested, clearing GPU cache...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                log_with_timestamp(f"[MEMORY] GPU cache cleared")

            log_with_timestamp(f"[CACHE_NOT_FOUND] Building {retriever_name}...")
            log_with_timestamp(f"[DEBUG] Calling build_retriever({retriever_name})...")
            retriever_start = datetime.now()
            try:
                success, cache_path = build_retriever(retriever_name, documents, doc_hash, cache_dir, metadata)
                log_with_timestamp(f"[DEBUG] build_retriever returned: success={success}")
            except Exception as e:
                log_with_timestamp(f"[ERROR] Exception in build_retriever: {type(e).__name__}: {e}")
                import traceback
                log_with_timestamp(traceback.format_exc())
                success = False
                cache_path = str(e)
            
            elapsed = (datetime.now() - retriever_start).total_seconds()
            log_with_timestamp(f"[DEBUG] Elapsed time: {elapsed:.1f}s")
            
            if success:
                log_with_timestamp(f"[BUILD_SUCCESS] {retriever_name} built in {elapsed:.1f}s")
                paths = get_cache_paths(retriever_name, doc_hash, cache_dir)
                if retriever_name in DENSE_RETRIEVERS:
                    log_with_timestamp(f"  → {paths['config']}")
                    log_with_timestamp(f"  → {paths['embeddings']}")
                elif retriever_name in COLBERT_RETRIEVERS:
                    log_with_timestamp(f"  → {paths['index_root']}")
                    log_with_timestamp(f"  → {paths['index_path']}")
                else:
                    log_with_timestamp(f"  → {paths['pickle']}")
                results[retriever_name] = {'status': 'success', 'time': elapsed}
            else:
                log_with_timestamp(f"[BUILD_FAILED] {retriever_name}: {cache_path}")
                results[retriever_name] = {'status': 'failed', 'error': cache_path}
                if retriever_name in COLBERT_RETRIEVERS:
                    log_with_timestamp(f"[WARNING] ColBERTv2 build failed but continuing: {cache_path}")
    
    # Summary
    log_with_timestamp(f"[DEBUG] Build loop complete, generating summary...")
    total_time = (datetime.now() - start_time).total_seconds()
    log_with_timestamp("\n" + "=" * 80)
    log_with_timestamp("BUILD SUMMARY")
    log_with_timestamp("=" * 80)
    
    for retriever_name, result in results.items():
        status = result['status']
        if status == 'success':
            time = result['time']
            log_with_timestamp(f"  ✓ {retriever_name:15} - Built in {time:7.1f}s")
        elif status == 'cached':
            log_with_timestamp(f"  ⚡ {retriever_name:15} - Already cached (skipped)")
        else:
            error = result['error']
            log_with_timestamp(f"  ✗ {retriever_name:15} - ERROR: {error}")
    
    log_with_timestamp(f"\nTotal time: {total_time:.1f}s")
    successful = sum(1 for r in results.values() if r['status'] in ['success', 'cached'])
    log_with_timestamp(f"Ready: {successful}/{len(ALL_RETRIEVERS)} (built + cached)")
    failed = sum(1 for r in results.values() if r['status'] == 'failed')
    if failed > 0:
        log_with_timestamp(f"Failed: {failed}/{len(ALL_RETRIEVERS)}")
    
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"[DEBUG] Main function complete")


def setup_logging():
    """Setup logging directory"""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)


if __name__ == '__main__':
    main()
    log_with_timestamp("当前任务已完成，请做下一个任务的指示。")
