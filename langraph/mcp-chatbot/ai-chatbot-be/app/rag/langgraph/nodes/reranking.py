"""
Context Reranking Node
======================

Reranks and compresses retrieved documents:
- Cross-encoder reranking for precision
- Context compression to fit token limits
- Source attribution maintenance
"""

import logging
import time
from typing import Any

from app.rag.langgraph.state import RAGState

logger = logging.getLogger(__name__)

# Configuration
MAX_CONTEXT_TOKENS = 4000
MAX_DOCUMENTS = 5


def _estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)."""
    return len(text) // 4


def _compress_context(
    documents: list[dict[str, Any]],
    max_tokens: int = MAX_CONTEXT_TOKENS
) -> tuple[str, int]:
    """
    Compress documents into context string within token limit.

    Args:
        documents: List of document dicts
        max_tokens: Maximum tokens allowed

    Returns:
        Tuple of (context string, token count)
    """
    context_parts = []
    total_tokens = 0

    for i, doc in enumerate(documents):
        content = doc.get("content", "")
        source = doc.get("source", "Unknown")
        score = doc.get("score", 0.0)

        # Format document with citation marker
        doc_text = f"[{i + 1}] Source: {source} (score: {score:.2f})\n{content}\n"
        doc_tokens = _estimate_tokens(doc_text)

        if total_tokens + doc_tokens <= max_tokens:
            context_parts.append(doc_text)
            total_tokens += doc_tokens
        else:
            # Truncate document to fit
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 100:  # Only add if meaningful
                truncated_content = content[:remaining_tokens * 4]
                doc_text = f"[{i + 1}] Source: {source} (score: {score:.2f})\n{truncated_content}...\n"
                context_parts.append(doc_text)
                total_tokens = max_tokens
            break

    return "\n".join(context_parts), total_tokens


async def context_reranking_node(state: RAGState) -> dict[str, Any]:
    """
    Rerank documents and compress context.

    This node:
    1. Applies cross-encoder reranking
    2. Selects top documents
    3. Compresses context to token limit
    4. Maintains source attribution

    Args:
        state: Current graph state

    Returns:
        Updated state with reranked documents and compressed context
    """
    start_time = time.time()
    logger.info("Starting context reranking")

    documents = state.get("retrieved_documents", [])
    query = state.get("enhanced_query") or state.get("cleaned_query", "")

    if not documents:
        logger.warning("No documents to rerank")
        return {
            "reranked_documents": [],
            "compressed_context": "",
            "context_token_count": 0,
            "current_node": "context_reranking",
            "next_node": "response_generation",
        }

    reranked_docs = documents.copy()

    # Try to apply cross-encoder reranking
    try:
        from app.rag.ranking import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        # Convert to format expected by reranker
        from langchain_core.documents import Document
        langchain_docs = [
            Document(
                page_content=doc.get("content", ""),
                metadata=doc.get("metadata", {})
            )
            for doc in documents
        ]

        # Rerank
        reranked_langchain = reranker.rerank(query, langchain_docs, top_k=MAX_DOCUMENTS)

        # Convert back with updated scores
        reranked_docs = []
        for i, doc in enumerate(reranked_langchain):
            original = documents[i] if i < len(documents) else {}
            reranked_docs.append({
                **original,
                "content": doc.page_content,
                "score": doc.metadata.get("score", original.get("score", 0.0)),
                "metadata": doc.metadata,
            })

        logger.info(f"Reranked {len(reranked_docs)} documents with cross-encoder")

    except Exception as e:
        logger.warning(f"Cross-encoder reranking failed, using original order: {e}")
        # Fall back to sorting by existing scores
        reranked_docs = sorted(
            documents,
            key=lambda x: x.get("score", 0.0),
            reverse=True
        )[:MAX_DOCUMENTS]

    # Compress context
    compressed_context, token_count = _compress_context(reranked_docs)

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Reranking complete: {len(reranked_docs)} docs, "
        f"{token_count} tokens, {duration_ms:.1f}ms"
    )

    # Determine next node
    query_analysis = state.get("query_analysis", {})
    if query_analysis.get("requires_human_review", False):
        next_node = "human_review"
    else:
        next_node = "response_generation"

    return {
        "reranked_documents": reranked_docs,
        "compressed_context": compressed_context,
        "context_token_count": token_count,
        "current_node": "context_reranking",
        "next_node": next_node,
        "metrics": {
            **state.get("metrics", {}),
            "reranking_applied": True,
            "documents_retrieved": len(reranked_docs),
        },
    }
