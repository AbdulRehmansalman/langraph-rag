"""
Response Formatting Node
========================

Polishes final output with:
- Citation formatting
- Metadata addition
- Related questions generation
- Structure formatting
"""

import logging
import re
import time
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage

from app.rag.langgraph.state import RAGState

logger = logging.getLogger(__name__)


def _format_citations(response: str, citations: list[dict]) -> str:
    """
    Ensure citations are consistently formatted.

    Converts various citation formats to [1], [2] style.
    """
    # Response already uses [n] format, just ensure consistency
    formatted = response

    # Add citation list at end if there are citations
    if citations:
        formatted += "\n\n---\n**Sources:**\n"
        for citation in citations:
            idx = citation.get("index", 0)
            source = citation.get("source", "Unknown")
            formatted += f"[{idx}] {source}\n"

    return formatted


def _generate_related_questions(
    query: str,
    response: str,
    documents: list[dict],
) -> list[str]:
    """
    Generate related follow-up questions.

    Simple heuristic-based generation (can be enhanced with LLM).
    """
    related = []

    # Extract key topics from documents
    topics = set()
    for doc in documents[:3]:
        content = doc.get("content", "")
        # Extract capitalized phrases as potential topics
        matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        topics.update(matches[:3])

    # Generate questions based on topics
    templates = [
        "What are the details about {}?",
        "How does {} work?",
        "Can you explain more about {}?",
    ]

    for topic in list(topics)[:3]:
        if len(topic) > 3 and topic.lower() not in query.lower():
            template = templates[len(related) % len(templates)]
            related.append(template.format(topic))

    return related[:3]  # Limit to 3 suggestions


async def response_formatting_node(state: RAGState) -> dict[str, Any]:
    """
    Format and enrich the final response.

    This node:
    1. Formats citations consistently
    2. Adds metadata (confidence, sources, timestamp)
    3. Generates related question suggestions
    4. Structures complex responses

    Args:
        state: Current graph state

    Returns:
        Updated state with formatted response
    """
    start_time = time.time()
    logger.info("Starting response formatting")

    response = state.get("generated_response", "")
    citations = state.get("citations", [])
    documents = state.get("reranked_documents", [])
    query = state.get("original_query", "")
    confidence = state.get("confidence_score", 0.0)

    # Format citations
    formatted_response = response
    if citations and not state.get("fallback_used", False):
        formatted_response = _format_citations(response, citations)

    # Generate related questions
    related_questions = []
    if documents and not state.get("fallback_used", False):
        related_questions = _generate_related_questions(query, response, documents)

    # Build response metadata
    response_metadata = {
        **state.get("response_metadata", {}),
        "confidence_score": confidence,
        "sources_count": len(citations),
        "timestamp": datetime.utcnow().isoformat(),
        "documents_used": len(documents),
        "query_type": state.get("query_analysis", {}).get("query_type", "unknown"),
        "retrieval_quality": state.get("retrieval_quality_score", 0.0),
        "verification_passed": state.get("verification_passed", False),
    }

    # Add grounding info if available
    if state.get("is_grounded") is not None:
        response_metadata["is_grounded"] = state["is_grounded"]

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Response formatting complete in {duration_ms:.1f}ms")

    return {
        "final_response": formatted_response,
        "related_questions": related_questions,
        "response_metadata": response_metadata,
        "current_node": "response_formatting",
        "next_node": "logging",
        "messages": [AIMessage(content=formatted_response)],
    }
