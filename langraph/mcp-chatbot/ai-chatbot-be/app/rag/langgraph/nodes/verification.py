"""
Self-Correction & Verification Node
====================================

Validates response quality through:
- Grounding verification (claims match context)
- Citation verification
- Hallucination detection
- Confidence scoring
"""

import logging
import re
import time
from typing import Any

from app.rag.langgraph.state import RAGState

logger = logging.getLogger(__name__)

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.7
MAX_CORRECTION_ATTEMPTS = 2

# Uncertainty indicators
UNCERTAINTY_PHRASES = [
    "i don't know",
    "i'm not sure",
    "i cannot find",
    "the context doesn't",
    "not mentioned",
    "unclear",
    "uncertain",
    "might be",
    "possibly",
    "perhaps",
]


def _check_grounding(response: str, context: str) -> tuple[bool, float]:
    """
    Check if response is grounded in context.

    Returns:
        Tuple of (is_grounded, grounding_score)
    """
    if not context:
        return False, 0.0

    # Simple word overlap check
    response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
    context_words = set(re.findall(r'\b\w{4,}\b', context.lower()))

    if not response_words:
        return True, 1.0

    overlap = response_words.intersection(context_words)
    overlap_ratio = len(overlap) / len(response_words)

    # Threshold for grounding
    is_grounded = overlap_ratio >= 0.3
    return is_grounded, min(overlap_ratio * 2, 1.0)


def _detect_uncertainty(response: str) -> float:
    """
    Detect uncertainty signals in response.

    Returns:
        Uncertainty score (0-1, higher = more uncertain)
    """
    response_lower = response.lower()
    uncertainty_count = sum(
        1 for phrase in UNCERTAINTY_PHRASES
        if phrase in response_lower
    )

    # Normalize by response length
    word_count = len(response.split())
    if word_count < 10:
        return 0.0

    uncertainty_ratio = uncertainty_count / (word_count / 50)
    return min(uncertainty_ratio, 1.0)


def _verify_citations(
    response: str,
    citations: list[dict],
    documents: list[dict]
) -> tuple[bool, float]:
    """
    Verify that citations in response match documents.

    Returns:
        Tuple of (all_valid, validity_score)
    """
    citation_pattern = r"\[(\d+)\]"
    used_citations = set(int(m) for m in re.findall(citation_pattern, response))

    if not used_citations:
        # No citations used - might be acceptable for simple responses
        return True, 0.8

    valid_citations = 0
    for idx in used_citations:
        if 0 < idx <= len(documents):
            valid_citations += 1

    validity_score = valid_citations / len(used_citations) if used_citations else 1.0
    all_valid = validity_score == 1.0

    return all_valid, validity_score


def _calculate_confidence(
    response: str,
    context: str,
    retrieval_scores: list[float],
    grounding_score: float,
    uncertainty_score: float,
) -> float:
    """
    Calculate overall confidence score.

    Factors:
    - Retrieval quality (40%)
    - Response-context alignment (40%)
    - Uncertainty signals (20%)
    """
    # Retrieval score component
    if retrieval_scores:
        retrieval_component = sum(retrieval_scores) / len(retrieval_scores)
    else:
        retrieval_component = 0.0

    # Alignment component (already calculated as grounding_score)
    alignment_component = grounding_score

    # Uncertainty penalty
    uncertainty_penalty = uncertainty_score * 0.3

    # Weighted combination
    confidence = (
        retrieval_component * 0.4 +
        alignment_component * 0.4 +
        (1 - uncertainty_penalty) * 0.2
    )

    return round(min(max(confidence, 0.0), 1.0), 2)


async def self_correction_node(state: RAGState) -> dict[str, Any]:
    """
    Verify and potentially correct the generated response.

    This node:
    1. Checks if response is grounded in context
    2. Verifies citations
    3. Detects potential hallucinations
    4. Calculates confidence score
    5. Routes to correction if quality is poor

    Args:
        state: Current graph state

    Returns:
        Updated state with verification results
    """
    start_time = time.time()
    logger.info("Starting self-correction verification")

    response = state.get("generated_response", "")
    context = state.get("compressed_context", "")
    citations = state.get("citations", [])
    documents = state.get("reranked_documents", [])
    correction_attempts = state.get("correction_attempts", 0)

    # Skip verification for fallback responses
    if state.get("fallback_used", False):
        return {
            "verification_passed": True,
            "confidence_score": 0.0,
            "is_grounded": False,
            "current_node": "self_correction",
            "next_node": "response_formatting",
        }

    # Check grounding
    is_grounded, grounding_score = _check_grounding(response, context)

    # Verify citations
    citations_valid, citation_score = _verify_citations(response, citations, documents)

    # Detect uncertainty
    uncertainty_score = _detect_uncertainty(response)

    # Calculate retrieval scores
    retrieval_scores = [doc.get("score", 0.0) for doc in documents]

    # Calculate overall confidence
    confidence = _calculate_confidence(
        response,
        context,
        retrieval_scores,
        grounding_score,
        uncertainty_score,
    )

    # Determine if verification passed
    verification_passed = (
        is_grounded and
        citations_valid and
        confidence >= MIN_CONFIDENCE_THRESHOLD
    )

    logger.info(
        f"Verification results: grounded={is_grounded}, "
        f"citations_valid={citations_valid}, confidence={confidence:.2f}, "
        f"passed={verification_passed}"
    )

    # Determine next action
    if verification_passed:
        next_node = "response_formatting"
    elif correction_attempts < MAX_CORRECTION_ATTEMPTS and not is_grounded:
        # Try query reformulation and re-retrieval
        next_node = "query_reformulation"
        correction_attempts += 1
        logger.info("Verification failed, attempting correction")
    else:
        # Accept response but with low confidence
        next_node = "response_formatting"
        logger.warning("Verification failed but max attempts reached")

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Self-correction complete in {duration_ms:.1f}ms")

    # Calculate hallucination score (inverse of grounding)
    hallucination_score = 1.0 - grounding_score if not is_grounded else 0.0

    return {
        "verification_passed": verification_passed,
        "confidence_score": confidence,
        "is_grounded": is_grounded,
        "hallucination_score": hallucination_score,
        "correction_attempts": correction_attempts,
        "current_node": "self_correction",
        "next_node": next_node,
        "response_metadata": {
            **state.get("response_metadata", {}),
            "verification": {
                "grounding_score": grounding_score,
                "citation_score": citation_score,
                "uncertainty_score": uncertainty_score,
                "passed": verification_passed,
            },
        },
    }
