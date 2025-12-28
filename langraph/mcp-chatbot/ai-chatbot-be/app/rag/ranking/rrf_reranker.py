"""
Reciprocal Rank Fusion (RRF) Reranker
=====================================
Combines multiple ranked lists into a single unified ranking.

RRF is particularly effective for ensemble retrieval where multiple
strategies (vector similarity, keyword search, MMR) return different
orderings of the same documents.

Algorithm:
    RRF_score(d) = Σ 1/(k + rank(d)) for each ranking list

Where k is a constant (default 60) that determines how much weight
is given to higher-ranked items.

Reference: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
           Reciprocal rank fusion outperforms condorcet and individual
           rank learning methods.
"""

import logging
from typing import List, Dict, Any

from langchain_core.documents import Document

from app.rag.ranking.base import BaseReranker

logger = logging.getLogger(__name__)


class RRFReranker(BaseReranker):
    """
    Reciprocal Rank Fusion for combining multiple retrieval strategies.

    RRF assigns scores based on rank position rather than absolute scores,
    making it robust to score scale differences between retrieval methods.

    This is ideal for ensemble retrieval where you want to combine:
    - Vector similarity results
    - Keyword/BM25 results
    - MMR (diversity-focused) results

    Features:
    - Score-agnostic fusion (uses ranks, not scores)
    - Handles duplicate documents across lists
    - Tracks which sources contributed to each result
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF reranker.

        Args:
            k: Constant in RRF formula (higher = smoother ranking)
               Default 60 is standard from the original paper.
               - Lower k (e.g., 10): More weight to top-ranked items
               - Higher k (e.g., 100): More uniform weighting
        """
        self.k = k

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        Rerank documents using RRF scores from their rank metadata.

        This method expects documents to have 'rank' in their metadata,
        typically set by the retrieval strategies before calling rerank.

        Args:
            query: Search query (unused, kept for interface consistency)
            documents: Documents with rank metadata
            top_k: Number of top results to return

        Returns:
            Documents sorted by RRF score with metadata:
            - rrf_score: The computed RRF score
        """
        if not documents:
            return []

        # Calculate RRF scores
        rrf_scores: Dict[str, Dict[str, Any]] = {}

        for doc in documents:
            # Use content hash as document identifier for deduplication
            doc_id = doc.metadata.get("chunk_hash") or hash(doc.page_content[:200])
            rank = doc.metadata.get("rank", 1)

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"doc": doc, "score": 0.0}

            # RRF formula: 1 / (k + rank)
            rrf_scores[doc_id]["score"] += 1.0 / (self.k + rank)

        # Sort by RRF score (descending)
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # Attach scores to metadata and return top_k
        result_docs = []
        for item in sorted_results[:top_k]:
            doc = item["doc"]
            doc.metadata["rrf_score"] = item["score"]
            result_docs.append(doc)

        logger.debug(f"RRF reranked {len(documents)} → {len(result_docs)} documents")
        return result_docs

    def fuse_rankings(
        self,
        ranked_lists: List[List[Document]],
        top_k: int = 5
    ) -> List[Document]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion.

        This is the primary method for ensemble retrieval, combining
        results from multiple retrieval strategies into a single ranking.

        Args:
            ranked_lists: List of document lists, each pre-sorted by
                         their respective retrieval strategy's relevance
            top_k: Number of top results to return

        Returns:
            Documents ranked by combined RRF score with metadata:
            - rrf_score: The combined RRF score
            - fusion_sources: List of source indices that contributed

        Example:
            ```python
            rrf = RRFReranker()

            # Results from different strategies
            similarity_results = retriever.similarity_search(query)
            keyword_results = retriever.keyword_search(query)
            mmr_results = retriever.mmr_search(query)

            # Fuse into single ranking
            fused = rrf.fuse_rankings(
                [similarity_results, keyword_results, mmr_results],
                top_k=10
            )
            ```
        """
        if not ranked_lists:
            return []

        rrf_scores: Dict[str, Dict[str, Any]] = {}

        for list_idx, ranked_list in enumerate(ranked_lists):
            for rank, doc in enumerate(ranked_list, start=1):
                # Use content hash for document identification
                doc_id = doc.metadata.get("chunk_hash") or hash(doc.page_content[:200])

                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {
                        "doc": doc,
                        "score": 0.0,
                        "sources": []
                    }

                # RRF formula: 1 / (k + rank)
                rrf_scores[doc_id]["score"] += 1.0 / (self.k + rank)
                rrf_scores[doc_id]["sources"].append(list_idx)

        # Sort by combined RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # Build result list with metadata
        result_docs = []
        for item in sorted_results[:top_k]:
            doc = item["doc"]
            doc.metadata["rrf_score"] = item["score"]
            doc.metadata["fusion_sources"] = item["sources"]
            result_docs.append(doc)

        logger.info(
            f"RRF fused {len(ranked_lists)} lists "
            f"({sum(len(l) for l in ranked_lists)} total docs) → "
            f"{len(result_docs)} results"
        )
        return result_docs
