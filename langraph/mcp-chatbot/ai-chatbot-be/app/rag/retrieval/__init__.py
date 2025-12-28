"""
Retrieval Services
==================
Vector and hybrid retrieval with reranking support.

Production retrieval components:
- RetrieverService: Multi-strategy retriever with reranking
- RetrievalStrategy: SIMILARITY, MMR, HYBRID, ENSEMBLE
- RerankingMethod: NONE, CROSS_ENCODER, RECIPROCAL_RANK_FUSION
- HyDERetriever: Hypothetical Document Embeddings retriever
- ContextualCompressionRetriever: LLM-based document compression

Usage:
    from app.rag.retrieval import create_retriever, RetrievalStrategy, RerankingMethod

    retriever = create_retriever(
        user_id="user-123",
        strategy=RetrievalStrategy.ENSEMBLE,
        reranking=RerankingMethod.CROSS_ENCODER,
        top_k=5
    )
    docs = retriever.invoke("What is the refund policy?")
"""

from app.rag.retrieval.retriever import (
    RetrieverService,
    RetrievalStrategy,
    RetrievalConfig,
    HyDERetriever,
    ContextualCompressionRetriever,
    create_retriever,
)
from app.rag.retrieval.vector_store import VectorStoreService
from app.rag.ranking import RerankingMethod

__all__ = [
    "RetrieverService",
    "RetrievalStrategy",
    "RetrievalConfig",
    "RerankingMethod",
    "VectorStoreService",
    "HyDERetriever",
    "ContextualCompressionRetriever",
    "create_retriever",
]
