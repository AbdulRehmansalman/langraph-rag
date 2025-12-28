"""
Retrieval Quality Assessment Node
=================================

Evaluates retrieved documents for:
- Relevance scoring
- Quality threshold checks
- Decision to reformulate query or proceed
"""

import logging
import time
from typing import Any

from app.rag.langgraph.state import RAGState

logger = logging.getLogger(__name__)

# Quality thresholds
MIN_QUALITY_SCORE = 0.6
MIN_DOCUMENTS = 1
MAX_REFORMULATION_ATTEMPTS = 2


def _calculate_average_score(documents: list[dict[str, Any]]) -> float:
    """Calculate average relevance score of documents."""
    if not documents:
        return 0.0

    scores = [doc.get("score", 0.0) for doc in documents]
    return sum(scores) / len(scores)


def _assess_document_quality(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess overall quality of retrieved documents."""
    if not documents:
        return {
            "quality_score": 0.0,
            "has_sufficient_documents": False,
            "top_score": 0.0,
            "score_variance": 0.0,
            "recommendation": "reformulate",
        }

    scores = [doc.get("score", 0.0) for doc in documents]
    avg_score = sum(scores) / len(scores)
    top_score = max(scores)
    min_score = min(scores)
    score_variance = top_score - min_score

    has_sufficient = len(documents) >= MIN_DOCUMENTS and top_score >= MIN_QUALITY_SCORE

    if has_sufficient and avg_score >= MIN_QUALITY_SCORE:
        recommendation = "proceed"
    elif top_score >= MIN_QUALITY_SCORE:
        recommendation = "proceed_with_caution"
    else:
        recommendation = "reformulate"

    return {
        "quality_score": avg_score,
        "has_sufficient_documents": has_sufficient,
        "top_score": top_score,
        "score_variance": score_variance,
        "recommendation": recommendation,
    }


async def quality_assessment_node(state: RAGState) -> dict[str, Any]:
    """
    Assess retrieval quality and decide next action.

    This node:
    1. Calculates quality metrics for retrieved documents
    2. Determines if quality meets threshold
    3. Decides whether to proceed, reformulate, or use fallback

    Args:
        state: Current graph state

    Returns:
        Updated state with quality assessment
    """
    start_time = time.time()
    logger.info("Starting quality assessment")

    documents = state.get("retrieved_documents", [])
    retrieval_attempts = state.get("retrieval_attempts", 0)
    correction_attempts = state.get("correction_attempts", 0)

    # Assess document quality
    assessment = _assess_document_quality(documents)
    quality_score = assessment["quality_score"]
    recommendation = assessment["recommendation"]

    logger.info(
        f"Quality assessment: score={quality_score:.2f}, "
        f"docs={len(documents)}, recommendation={recommendation}"
    )

    # Determine next node based on quality
    if recommendation == "reformulate" and correction_attempts < MAX_REFORMULATION_ATTEMPTS:
        # Poor quality, try reformulating query
        next_node = "query_reformulation"
        logger.info("Quality below threshold, reformulating query")
    elif not documents:
        # No documents at all, go to fallback
        next_node = "response_generation"  # Will use fallback response
        logger.warning("No documents retrieved, using fallback")
    else:
        # Proceed with reranking
        next_node = "context_reranking"

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Quality assessment complete in {duration_ms:.1f}ms")

    return {
        "retrieval_quality_score": quality_score,
        "current_node": "quality_assessment",
        "next_node": next_node,
        "response_metadata": {
            **state.get("response_metadata", {}),
            "quality_assessment": assessment,
        },
    }


async def query_reformulation_node(state: RAGState) -> dict[str, Any]:
    """
    Reformulate query when retrieval quality is poor.

    Args:
        state: Current graph state

    Returns:
        Updated state with reformulated query
    """
    start_time = time.time()
    logger.info("Starting query reformulation")

    original_query = state.get("enhanced_query") or state.get("cleaned_query", "")
    correction_attempts = state.get("correction_attempts", 0) + 1
    query_analysis = state.get("query_analysis", {})

    try:
        from app.services.llm_factory import llm_factory
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = llm_factory.create_llm()

        reformulate_prompt = ChatPromptTemplate.from_messages([
            ("system", """The initial document search did not return good results.
Reformulate the query to be more specific or use different terminology.
Focus on key concepts and try alternative phrasings.
Output ONLY the reformulated query."""),
            ("human", """Original query: {query}
Keywords: {keywords}

Reformulated query:"""),
        ])

        chain = reformulate_prompt | llm | StrOutputParser()
        reformulated = await chain.ainvoke({
            "query": original_query,
            "keywords": ", ".join(query_analysis.get("keywords", [])),
        })
        reformulated = reformulated.strip()

        logger.info(f"Query reformulated: {original_query[:30]} -> {reformulated[:30]}")

    except Exception as e:
        logger.error(f"Query reformulation error: {e}")
        reformulated = original_query

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Query reformulation complete in {duration_ms:.1f}ms")

    return {
        "enhanced_query": reformulated,
        "reformulated_query": reformulated,
        "correction_attempts": correction_attempts,
        "current_node": "query_reformulation",
        "next_node": "parallel_retrieval",  # Retry retrieval
    }
