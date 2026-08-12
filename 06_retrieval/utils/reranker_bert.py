#!/usr/bin/env python3
"""
Stage 13: BERT Cross-Encoder Reranker

Contains BERT-based reranking class:
- BERTReRanker: BERT cross-encoder reranker for top-k results
"""

import torch
from typing import List, Dict, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import log_with_timestamp, build_document_text


class BERTReRanker:
    """BERT-based cross-encoder reranker for top-k results"""
    def __init__(self, base_retriever, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 50):
        """
        Args:
            base_retriever: 底层检索器（BM25, Dense等）
            model_name: BERT cross-encoder 模型
            top_k: 从底层检索器获取多少候选用于重排序
        """
        self.base_retriever = base_retriever
        self.model_name = model_name
        self.top_k = top_k
        self.model = None
        self.device = torch.device('cuda')

    def _get_model(self):
        if self.model is None:
            log_with_timestamp(f"  Loading BERT reranker: {self.model_name}")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
        return self.model

    def fit(self, documents: List[Dict[str, str]], all_metadata: Dict[str, Dict] = None):
        """训练底层检索器"""
        log_with_timestamp("  Building BERT Reranker index...")
        # 训练底层检索器
        if hasattr(self.base_retriever, 'all_metadata'):
            self.base_retriever.fit(documents, all_metadata)
        else:
            self.base_retriever.fit(documents)

        # 保存文档用于重排序
        self.documents = documents
        self.all_metadata = all_metadata
        self.doc_ids = [doc.get('asin', '') for doc in documents]

        log_with_timestamp(f"  BERT Reranker index built with {len(self.doc_ids)} docs")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用 BERT 重排序"""
        # Step 1: 使用底层检索器获取候选
        candidates = self.base_retriever.search(query, top_k=self.top_k)

        if not candidates:
            return []

        # Step 2: 使用 BERT cross-encoder 重排序
        model = self._get_model()

        # 准备 query-document pairs
        doc_texts = []
        for asin, _ in candidates:
            # 找到对应的文档
            doc_idx = self.doc_ids.index(asin)
            doc_text = build_document_text(self.documents[doc_idx], self.all_metadata)
            # 截断文档以避免超长（BERT max 512 tokens）
            doc_text = ' '.join(doc_text.split()[:300])  # 大约 400-500 tokens
            doc_texts.append(doc_text)

        # 构造 pairs
        pairs = [[query, doc_text] for doc_text in doc_texts]

        # BERT 打分
        scores = model.predict(pairs)

        # 组合结果
        reranked_results = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        reranked_results.sort(key=lambda x: -x[1])

        return reranked_results[:top_k]
