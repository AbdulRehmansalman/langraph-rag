"""
Human Review Gate Node
======================

Implements human-in-the-loop review for sensitive queries:
- Interrupts execution for human approval
- Allows modification of context
- Logs all review decisions for audit
"""

import logging
import time
from datetime import datetime
from typing import Any

from langgraph.types import interrupt

from app.rag.langgraph.state import RAGState, HumanReviewStatus

logger = logging.getLogger(__name__)


async def human_review_node(state: RAGState) -> dict[str, Any]:
    """
    Human review gate for sensitive content.

    This node interrupts execution when queries involve:
    - PII (Personal Identifiable Information)
    - Medical advice
    - Financial advice
    - Legal matters

    The human can:
    - Approve the context as-is
    - Reject and stop processing
    - Modify the context before proceeding

    Args:
        state: Current graph state

    Returns:
        Updated state with review decision
    """
    start_time = time.time()
    logger.info("Entering human review gate")

    query_analysis = state.get("query_analysis", {})
    detected_topics = query_analysis.get("detected_topics", [])
    sensitivity_level = query_analysis.get("sensitivity_level", "none")

    # Check if human review was already completed
    human_review = state.get("human_review", {})
    if human_review.get("status") in [
        HumanReviewStatus.APPROVED.value,
        HumanReviewStatus.REJECTED.value,
        HumanReviewStatus.MODIFIED.value,
    ]:
        logger.info(f"Human review already completed: {human_review.get('status')}")
        if human_review.get("status") == HumanReviewStatus.REJECTED.value:
            return {
                "should_end": True,
                "final_response": "This query has been rejected by the review process.",
                "current_node": "human_review",
            }
        return {
            "current_node": "human_review",
            "next_node": "response_generation",
        }

    # Prepare review context
    review_context = {
        "query": state.get("original_query", ""),
        "enhanced_query": state.get("enhanced_query", ""),
        "detected_topics": detected_topics,
        "sensitivity_level": sensitivity_level,
        "documents": state.get("reranked_documents", [])[:3],  # First 3 for review
        "context_preview": state.get("compressed_context", "")[:500],
    }

    logger.info(
        f"Requesting human review for sensitive topics: {detected_topics}"
    )

    # Interrupt for human review
    # This will pause execution until human provides input
    review_decision = interrupt({
        "type": "human_review_required",
        "context": review_context,
        "options": ["approve", "reject", "modify"],
        "message": f"Review required for sensitive content: {', '.join(detected_topics)}",
    })

    # Process review decision
    decision_status = review_decision.get("decision", "reject")
    reviewer_id = review_decision.get("reviewer_id", "unknown")
    reason = review_decision.get("reason", "")
    modified_context = review_decision.get("modified_context")

    if decision_status == "approve":
        status = HumanReviewStatus.APPROVED
        next_node = "response_generation"
    elif decision_status == "modify":
        status = HumanReviewStatus.MODIFIED
        next_node = "response_generation"
    else:
        status = HumanReviewStatus.REJECTED
        next_node = None

    human_review_result = {
        "status": status.value,
        "reviewer_id": reviewer_id,
        "decision_time": datetime.utcnow().isoformat(),
        "reason": reason,
        "modified_context": modified_context,
    }

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Human review completed: status={status.value}, "
        f"reviewer={reviewer_id}, duration={duration_ms:.1f}ms"
    )

    result = {
        "human_review": human_review_result,
        "current_node": "human_review",
    }

    if status == HumanReviewStatus.REJECTED:
        result["should_end"] = True
        result["final_response"] = (
            "Your query has been reviewed and cannot be processed at this time. "
            f"Reason: {reason or 'Content requires further review.'}"
        )
    elif status == HumanReviewStatus.MODIFIED and modified_context:
        result["compressed_context"] = modified_context
        result["next_node"] = next_node
    else:
        result["next_node"] = next_node

    return result
