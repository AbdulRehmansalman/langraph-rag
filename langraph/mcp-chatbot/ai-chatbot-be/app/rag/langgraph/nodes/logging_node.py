"""
Logging & Monitoring Node
=========================

Records execution metrics and telemetry:
- Per-node latency tracking
- Token usage monitoring
- Quality metrics collection
- Error logging
- Prometheus metrics export
"""

import logging
import time
from datetime import datetime
from typing import Any, Optional

from app.rag.langgraph.state import RAGState, ExecutionMetrics, NodeMetrics

logger = logging.getLogger(__name__)

# Try to import prometheus client
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True

    # Define metrics
    RAG_REQUESTS_TOTAL = Counter(
        'rag_requests_total',
        'Total RAG requests',
        ['status', 'query_type']
    )
    RAG_LATENCY_SECONDS = Histogram(
        'rag_latency_seconds',
        'RAG request latency',
        ['node'],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    RAG_CONFIDENCE_SCORE = Gauge(
        'rag_confidence_score',
        'RAG response confidence score'
    )
    RAG_DOCUMENTS_RETRIEVED = Histogram(
        'rag_documents_retrieved',
        'Number of documents retrieved',
        buckets=[0, 1, 2, 5, 10, 20]
    )
    RAG_ERRORS_TOTAL = Counter(
        'rag_errors_total',
        'Total RAG errors',
        ['node', 'error_type']
    )

except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("Prometheus client not available, metrics disabled")


def _calculate_total_duration(state: RAGState) -> float:
    """Calculate total execution duration in milliseconds."""
    timestamp_str = state.get("timestamp", "")
    if timestamp_str:
        try:
            start_time = datetime.fromisoformat(timestamp_str)
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return duration
        except (ValueError, TypeError):
            pass
    return 0.0


def _collect_node_metrics(state: RAGState) -> list[dict]:
    """Collect metrics from all executed nodes."""
    metrics_data = state.get("metrics", {})
    node_metrics = metrics_data.get("node_metrics", [])
    return node_metrics


def _export_prometheus_metrics(state: RAGState) -> None:
    """Export metrics to Prometheus."""
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        query_analysis = state.get("query_analysis", {})
        query_type = query_analysis.get("query_type", "unknown")
        status = "success" if not state.get("has_error", False) else "error"

        # Request counter
        RAG_REQUESTS_TOTAL.labels(status=status, query_type=query_type).inc()

        # Confidence score
        confidence = state.get("confidence_score", 0.0)
        RAG_CONFIDENCE_SCORE.set(confidence)

        # Documents retrieved
        docs = state.get("retrieved_documents", [])
        RAG_DOCUMENTS_RETRIEVED.observe(len(docs))

        # Total latency
        duration_seconds = _calculate_total_duration(state) / 1000
        RAG_LATENCY_SECONDS.labels(node="total").observe(duration_seconds)

        # Error tracking
        for error in state.get("error_log", []):
            RAG_ERRORS_TOTAL.labels(
                node=error.get("node", "unknown"),
                error_type=error.get("error_type", "unknown")
            ).inc()

    except Exception as e:
        logger.error(f"Failed to export Prometheus metrics: {e}")


async def logging_node(state: RAGState) -> dict[str, Any]:
    """
    Log execution metrics and complete the pipeline.

    This node:
    1. Calculates final metrics
    2. Logs structured telemetry
    3. Exports Prometheus metrics
    4. Prepares final state

    Args:
        state: Current graph state

    Returns:
        Updated state with final metrics
    """
    start_time = time.time()
    logger.info("Starting logging and metrics collection")

    # Calculate total duration
    total_duration_ms = _calculate_total_duration(state)

    # Collect all metrics
    node_metrics = _collect_node_metrics(state)

    # Build final metrics
    final_metrics = ExecutionMetrics(
        total_duration_ms=total_duration_ms,
        node_metrics=[NodeMetrics(**m) for m in node_metrics] if node_metrics else [],
        total_tokens_used=state.get("context_token_count", 0),
        retrieval_count=state.get("retrieval_attempts", 0),
        documents_retrieved=len(state.get("retrieved_documents", [])),
        reranking_applied=state.get("metrics", {}).get("reranking_applied", False),
        tools_used=state.get("metrics", {}).get("tools_used", []),
        retry_count=state.get("correction_attempts", 0),
        cache_hit=state.get("metrics", {}).get("cache_hit", False),
    ).model_dump()

    # Export Prometheus metrics
    _export_prometheus_metrics(state)

    # Log structured summary
    query_analysis = state.get("query_analysis", {})
    log_data = {
        "thread_id": state.get("thread_id", ""),
        "user_id": state.get("user_id", ""),
        "query_type": query_analysis.get("query_type", "unknown"),
        "intent": query_analysis.get("intent", "unknown"),
        "duration_ms": total_duration_ms,
        "documents_retrieved": len(state.get("retrieved_documents", [])),
        "confidence_score": state.get("confidence_score", 0.0),
        "verification_passed": state.get("verification_passed", False),
        "fallback_used": state.get("fallback_used", False),
        "has_error": state.get("has_error", False),
        "error_count": len(state.get("error_log", [])),
    }

    # Log at appropriate level
    if state.get("has_error", False):
        logger.error(f"RAG pipeline completed with errors: {log_data}")
    elif state.get("confidence_score", 0) < 0.5:
        logger.warning(f"RAG pipeline completed with low confidence: {log_data}")
    else:
        logger.info(f"RAG pipeline completed successfully: {log_data}")

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Logging complete in {duration_ms:.1f}ms")

    return {
        "metrics": final_metrics,
        "current_node": "logging",
        "should_end": True,
    }
