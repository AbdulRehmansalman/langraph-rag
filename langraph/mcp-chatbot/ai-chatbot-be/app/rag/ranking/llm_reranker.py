"""
LLM-based Reranker
==================
Uses LLM to assess document relevance to query.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.rag.ranking.base import BaseReranker

logger = logging.getLogger(__name__)


class LLMReranker(BaseReranker):
    """
    LLM-based reranker for semantic relevance scoring.
    Uses the LLM to assess relevance of each document.
    """

    def __init__(self, llm=None):
        self.llm = llm

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents using LLM relevance assessment."""
        if not self.llm or not documents:
            return documents[:top_k]

        try:
            scored_docs = []
            for doc in documents:
                score = self._score_document(query, doc)
                doc.metadata["rerank_score"] = score
                scored_docs.append((doc, score))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:top_k]]

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return documents[:top_k]

    def _score_document(self, query: str, doc: Document) -> float:
        """Score a single document for relevance."""
        prompt = f"""Rate the relevance of this document to the query on a scale of 0-10.
Only respond with a number.

Query: {query}

Document: {doc.page_content[:500]}

Relevance score (0-10):"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            score_text = response.content.strip()
            score = float(score_text) / 10.0
            return min(max(score, 0), 1)
        except Exception:
            return 0.5
