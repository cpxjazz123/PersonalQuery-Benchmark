"""
Stage 13 Retrieval Utilities

This package provides core utilities for the retrieval evaluation pipeline.
"""

from .utils import (
    log_with_timestamp,
    load_product_metadata,
    load_reviews_for_products,
    build_document_text,
    evaluate_retriever,
    compute_metrics,
    compute_enhanced_metrics,
    compute_noise_robustness,
    compute_percentile_stats,
    compute_aggregate_metrics,
    compute_dcg,
    compute_cg,
    compute_err,
    compute_rbp,
    compute_r_precision,
    compute_bpref,
    compute_novelty
)

from .retrievers import (
    BM25,
    ColBERTRetriever,
    DenseRetriever,
    E5Retriever,
    BGERetriever,
    TFIDFRetriever
)

# from .hybrid import HybridRetriever  # File not found
from .reranker_bert import BERTReRanker

__all__ = [
    # Core utilities
    'log_with_timestamp',
    'load_product_metadata',
    'load_reviews_for_products',
    'build_document_text',
    'evaluate_retriever',
    'compute_metrics',
    'compute_enhanced_metrics',
    'compute_noise_robustness',
    'compute_percentile_stats',
    'compute_aggregate_metrics',
    # New comprehensive metrics
    'compute_dcg',
    'compute_cg',
    'compute_err',
    'compute_rbp',
    'compute_r_precision',
    'compute_bpref',
    'compute_novelty',

    # Retriever classes
    'BM25',
    'ColBERTRetriever',
    'DenseRetriever',
    'E5Retriever',
    'BGERetriever',
    'TFIDFRetriever',
    # 'HybridRetriever',  # File not found
    'BERTReRanker',
]