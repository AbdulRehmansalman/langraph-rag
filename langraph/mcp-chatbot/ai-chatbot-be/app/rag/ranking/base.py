"""
Base Reranker
=============
Abstract interface for document reranking strategies.

Rerankers improve retrieval quality by re-scoring documents
based on their relevance to the query after initial retrieval.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from langchain_core.documents import Document


class RerankingMethod(Enum):
    """
    Available reranking methods.

    - NONE: No reranking, use retrieval scores directly
    - CROSS_ENCODER: Neural cross-encoder for high-precision scoring
    - RECIPROCAL_RANK_FUSION: Combine multiple ranked lists
    """
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    RECIPROCAL_RANK_FUSION = "rrf"


class BaseReranker(ABC):
    """
    Abstract base class for document rerankers.

    Rerankers take a query and a list of documents, and return
    the documents sorted by relevance to the query with updated
    scores in metadata.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        Rerank documents based on query relevance.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            List of reranked documents with updated metadata:
            - rerank_score: The reranking score
            - original_score: The original retrieval score
        """
        pass

    def __call__(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """Allow calling the reranker directly."""
        return self.rerank(query, documents, top_k)
