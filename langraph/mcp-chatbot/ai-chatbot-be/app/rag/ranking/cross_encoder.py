"""
Cross-Encoder Reranker
======================
High-precision neural reranking using cross-encoder models.

Cross-encoders process query-document pairs together, enabling
more accurate relevance scoring than bi-encoders (separate embeddings).

The model jointly encodes the query and document, allowing it to
capture fine-grained semantic relationships that bi-encoders miss.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document

from app.rag.ranking.base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Production cross-encoder reranker with lazy model loading.

    Uses MS MARCO fine-tuned model optimized for passage ranking.
    Gracefully falls back to original order if model unavailable.

    Features:
    - Lazy initialization (loads model on first use)
    - Batch scoring for efficiency
    - Score normalization and metadata attachment
    - Graceful fallback if model fails to load
    """

    # MS MARCO fine-tuned model for passage ranking
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
                       Default: ms-marco-MiniLM-L-6-v2 (fast, accurate)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._initialized = False

    def _init_model(self) -> None:
        """
        Lazy-load cross-encoder model on first use.

        This defers the heavy model loading until actually needed,
        reducing startup time for applications that may not use reranking.
        """
        if self._initialized:
            return

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self._initialized = True
            logger.info(f"CrossEncoder model loaded: {self.model_name}")

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "CrossEncoder reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
            self._model = None
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            self._model = None
            self._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder relevance scores.

        The cross-encoder scores each query-document pair together,
        providing more accurate relevance assessment than comparing
        separate embeddings.

        Args:
            query: The search query
            documents: List of candidate documents to rerank
            top_k: Number of top documents to return

        Returns:
            List of reranked documents with metadata:
            - rerank_score: Cross-encoder relevance score
            - original_score: Original retrieval score (preserved)
        """
        # Ensure model is loaded
        self._init_model()

        # Fallback if model unavailable or no documents
        if not self._model or not documents:
            return documents[:top_k]

        try:
            # Create query-document pairs for batch scoring
            pairs = [(query, doc.page_content) for doc in documents]

            # Score all pairs in batch (efficient GPU/CPU utilization)
            scores = self._model.predict(pairs)

            # Combine documents with scores
            doc_scores = list(zip(documents, scores))

            # Sort by cross-encoder score (descending = most relevant first)
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Attach scores to metadata and return top_k
            reranked_docs = []
            for doc, score in doc_scores[:top_k]:
                # Preserve original score, add rerank score
                doc.metadata["rerank_score"] = float(score)
                doc.metadata["original_score"] = doc.metadata.get("score", 0.0)
                reranked_docs.append(doc)

            logger.debug(
                f"Reranked {len(documents)} documents → {len(reranked_docs)} "
                f"(top score: {reranked_docs[0].metadata['rerank_score']:.3f})"
            )
            return reranked_docs

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original order on error
            return documents[:top_k]

    @property
    def is_available(self) -> bool:
        """Check if the cross-encoder model is available and loaded."""
        self._init_model()
        return self._model is not None

    @property
    def model_info(self) -> str:
        """Get information about the loaded model."""
        if self.is_available:
            return f"CrossEncoder: {self.model_name}"
        return "CrossEncoder: Not available"
