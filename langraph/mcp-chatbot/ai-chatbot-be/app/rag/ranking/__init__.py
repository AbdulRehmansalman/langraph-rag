"""
Reranking Services
==================
Document reranking for improved retrieval quality.

Production reranking strategies:
- CrossEncoderReranker: High-precision neural reranking (recommended)
- RRFReranker: Reciprocal Rank Fusion for ensemble retrieval

Usage:
    from app.rag.ranking import CrossEncoderReranker, RRFReranker, RerankingMethod

    # Cross-encoder for precision reranking
    reranker = CrossEncoderReranker()
    reranked_docs = reranker.rerank(query, documents, top_k=5)

    # RRF for combining multiple retrieval strategies
    rrf = RRFReranker()
    fused_docs = rrf.fuse_rankings([list1, list2, list3], top_k=10)
"""

from app.rag.ranking.base import BaseReranker, RerankingMethod
from app.rag.ranking.cross_encoder import CrossEncoderReranker
from app.rag.ranking.rrf_reranker import RRFReranker

__all__ = [
    "BaseReranker",
    "RerankingMethod",
    "CrossEncoderReranker",
    "RRFReranker",
]
