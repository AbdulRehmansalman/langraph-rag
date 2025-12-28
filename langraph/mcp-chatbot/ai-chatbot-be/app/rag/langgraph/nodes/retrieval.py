"""
Retrieval Nodes
===============

Parallel retrieval from multiple sources:
- Vector similarity search
- Keyword/BM25 search
- Metadata filtering
- Result merging and deduplication
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, Optional

from langchain_core.documents import Document

from app.rag.langgraph.state import RAGState, DocumentChunk

logger = logging.getLogger(__name__)

# Timeout for individual retrieval operations
RETRIEVAL_TIMEOUT = 5.0  # seconds
MAX_RETRIES = 3
RETRY_DELAYS = [1.0, 2.0, 4.0]  # Exponential backoff


def _document_to_chunk(doc: Document, index: int) -> dict[str, Any]:
    """Convert LangChain Document to DocumentChunk dict."""
    return DocumentChunk(
        id=doc.metadata.get("document_id", f"doc_{index}"),
        content=doc.page_content,
        source=doc.metadata.get("source", doc.metadata.get("filename", "unknown")),
        score=doc.metadata.get("score", 0.0),
        metadata=doc.metadata,
        page_number=doc.metadata.get("page_number"),
        chunk_index=doc.metadata.get("chunk_index", index),
    ).model_dump()


def _compute_content_hash(content: str) -> str:
    """Compute hash for deduplication."""
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _deduplicate_documents(
    documents: list[dict[str, Any]],
    similarity_threshold: float = 0.95
) -> list[dict[str, Any]]:
    """Remove duplicate and near-duplicate documents."""
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        content = doc.get("content", "")
        content_hash = _compute_content_hash(content)

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)

    return unique_docs


async def _retry_with_backoff(
    func,
    *args,
    max_retries: int = MAX_RETRIES,
    **kwargs
) -> Any:
    """Execute function with retry and exponential backoff."""
    last_error = None

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning(
                    f"Retrieval attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries} retrieval attempts failed: {e}")

    raise last_error


async def vector_search_node(state: RAGState) -> dict[str, Any]:
    """
    Perform vector similarity search.

    Args:
        state: Current graph state

    Returns:
        Updated state with vector search results
    """
    start_time = time.time()
    logger.info("Starting vector search")

    query = state.get("enhanced_query") or state.get("cleaned_query", "")
    user_id = state.get("user_id")
    document_ids = state.get("document_ids")

    results = []

    try:
        from app.rag.retrieval import create_retriever, RetrievalStrategy

        retriever = create_retriever(
            user_id=user_id,
            document_ids=document_ids,
            strategy=RetrievalStrategy.SIMILARITY,
            top_k=10,
        )

        async def do_search():
            return await asyncio.to_thread(retriever.invoke, query)

        docs = await asyncio.wait_for(
            _retry_with_backoff(do_search),
            timeout=RETRIEVAL_TIMEOUT * MAX_RETRIES
        )

        results = [_document_to_chunk(doc, i) for i, doc in enumerate(docs)]
        logger.info(f"Vector search found {len(results)} documents")

    except asyncio.TimeoutError:
        logger.error(f"Vector search timeout after {RETRIEVAL_TIMEOUT}s")
    except Exception as e:
        logger.error(f"Vector search error: {e}")

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Vector search complete in {duration_ms:.1f}ms")

    return {
        "vector_search_results": results,
    }


async def keyword_search_node(state: RAGState) -> dict[str, Any]:
    """
    Perform keyword/BM25 search.

    Args:
        state: Current graph state

    Returns:
        Updated state with keyword search results
    """
    start_time = time.time()
    logger.info("Starting keyword search")

    query = state.get("enhanced_query") or state.get("cleaned_query", "")
    user_id = state.get("user_id")
    document_ids = state.get("document_ids")
    query_analysis = state.get("query_analysis", {})
    keywords = query_analysis.get("keywords", [])

    results = []

    try:
        from app.rag.retrieval.vector_store import VectorStoreService

        vector_store = VectorStoreService()

        # Use keywords for more targeted search
        search_query = " ".join(keywords) if keywords else query

        async def do_search():
            return await asyncio.to_thread(
                vector_store.keyword_search,
                search_query,
                user_id=user_id,
                document_ids=document_ids,
                top_k=10,
            )

        docs = await asyncio.wait_for(
            _retry_with_backoff(do_search),
            timeout=RETRIEVAL_TIMEOUT * MAX_RETRIES
        )

        results = [_document_to_chunk(doc, i) for i, doc in enumerate(docs)]
        logger.info(f"Keyword search found {len(results)} documents")

    except asyncio.TimeoutError:
        logger.error(f"Keyword search timeout after {RETRIEVAL_TIMEOUT}s")
    except Exception as e:
        logger.error(f"Keyword search error: {e}")

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Keyword search complete in {duration_ms:.1f}ms")

    return {
        "keyword_search_results": results,
    }


async def parallel_retrieval_node(state: RAGState) -> dict[str, Any]:
    """
    Execute multiple retrieval strategies in parallel.

    This is a fan-out node that runs:
    - Vector similarity search
    - Keyword search
    - Metadata filtering (optional)

    Args:
        state: Current graph state

    Returns:
        Updated state with all retrieval results
    """
    start_time = time.time()
    logger.info("Starting parallel retrieval")

    retrieval_attempts = state.get("retrieval_attempts", 0) + 1

    # Run retrievals in parallel
    vector_task = asyncio.create_task(vector_search_node(state))
    keyword_task = asyncio.create_task(keyword_search_node(state))

    # Wait for all with timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(vector_task, keyword_task, return_exceptions=True),
            timeout=RETRIEVAL_TIMEOUT * 2
        )
    except asyncio.TimeoutError:
        logger.error("Parallel retrieval timeout")
        results = [{}, {}]

    # Extract results
    vector_results = {}
    keyword_results = {}

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Retrieval task error: {result}")
            continue
        if "vector_search_results" in result:
            vector_results = result
        elif "keyword_search_results" in result:
            keyword_results = result

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Parallel retrieval complete in {duration_ms:.1f}ms")

    return {
        "vector_search_results": vector_results.get("vector_search_results", []),
        "keyword_search_results": keyword_results.get("keyword_search_results", []),
        "retrieval_attempts": retrieval_attempts,
        "current_node": "parallel_retrieval",
        "next_node": "merge_results",
    }


async def merge_retrieval_results_node(state: RAGState) -> dict[str, Any]:
    """
    Merge and deduplicate results from all retrieval sources.

    Applies Reciprocal Rank Fusion (RRF) for combining ranked lists.

    Args:
        state: Current graph state

    Returns:
        Updated state with merged documents
    """
    start_time = time.time()
    logger.info("Merging retrieval results")

    vector_results = state.get("vector_search_results", [])
    keyword_results = state.get("keyword_search_results", [])
    metadata_results = state.get("metadata_filter_results", [])

    # Combine all results
    all_docs = []
    all_docs.extend(vector_results)
    all_docs.extend(keyword_results)
    all_docs.extend(metadata_results)

    if not all_docs:
        logger.warning("No documents retrieved from any source")
        return {
            "retrieved_documents": [],
            "current_node": "merge_results",
            "next_node": "quality_assessment",
        }

    # Apply RRF scoring
    doc_scores = {}
    k = 60  # RRF constant

    for rank, doc in enumerate(vector_results):
        doc_id = doc.get("id", _compute_content_hash(doc.get("content", "")))
        rrf_score = 1.0 / (k + rank + 1)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score * 0.5  # Vector weight
        doc_scores[f"{doc_id}_doc"] = doc

    for rank, doc in enumerate(keyword_results):
        doc_id = doc.get("id", _compute_content_hash(doc.get("content", "")))
        rrf_score = 1.0 / (k + rank + 1)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score * 0.5  # Keyword weight
        if f"{doc_id}_doc" not in doc_scores:
            doc_scores[f"{doc_id}_doc"] = doc

    # Sort by combined score
    scored_docs = []
    for doc_id, score in doc_scores.items():
        if not doc_id.endswith("_doc"):
            doc = doc_scores.get(f"{doc_id}_doc")
            if doc:
                doc["score"] = score
                scored_docs.append(doc)

    scored_docs.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Deduplicate
    unique_docs = _deduplicate_documents(scored_docs)

    # Limit to top results
    top_docs = unique_docs[:10]

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Merged {len(all_docs)} docs into {len(top_docs)} unique docs "
        f"in {duration_ms:.1f}ms"
    )

    return {
        "retrieved_documents": top_docs,
        "current_node": "merge_results",
        "next_node": "quality_assessment",
    }
