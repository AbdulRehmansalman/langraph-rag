"""
Production RAG Chain (LangGraph Implementation)
===============================================

LangGraph-based RAG chain with full pipeline control.

Features:
- Stateful graph execution with checkpointing
- Parallel retrieval from multiple sources
- Self-correction and verification
- Human-in-the-loop review gates
- Token-by-token streaming generation
- Circuit breaker for fault tolerance
- Comprehensive observability

Usage:
    from app.rag.pipeline.chain import create_rag_chain

    rag_chain = create_rag_chain(user_id="user-123")
    response = await rag_chain.invoke("What is the refund policy?")

    # Or stream tokens
    async for token in rag_chain.stream("How do I cancel?"):
        print(token, end="")

Migration Note:
    This module now uses LangGraph internally while maintaining
    the same external API for backward compatibility.
"""

import asyncio
import logging
import time
import traceback
from typing import List, Optional, Dict, Any, AsyncIterator, Union

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from app.rag.models.schemas import RAGMode, RAGConfig, RAGResponse, RAGMetrics
from app.rag.utils.resilience import CircuitBreaker, with_retry

# Import LangGraph components
from app.rag.langgraph import create_rag_graph, RAGState
from app.rag.langgraph.state import create_initial_state
from app.rag.langgraph.graph import RAGAgent

# Keep legacy imports for backward compatibility
from app.rag.retrieval import (
    RetrievalStrategy,
    RerankingMethod,
)
from app.rag.pipeline.memory import get_memory_service

logger = logging.getLogger(__name__)


class LangGraphRAGChain:
    """
    Production RAG chain powered by LangGraph.

    This class wraps the LangGraph RAG agent while providing the same
    interface as the original RAGChain for backward compatibility.

    Pipeline Flow (LangGraph Nodes):
    1. Query Analysis - Classify and route query
    2. Query Enhancement - Rewrite and expand query
    3. Parallel Retrieval - Vector + keyword search
    4. Quality Assessment - Evaluate retrieval quality
    5. Context Reranking - Cross-encoder reranking
    6. Human Review Gate - Optional approval for sensitive content
    7. Response Generation - LLM response with citations
    8. Self-Correction - Verify grounding and confidence
    9. Response Formatting - Polish output
    10. Logging - Metrics and telemetry
    """

    def __init__(
        self,
        llm=None,  # Kept for backward compatibility, not directly used
        user_id: str = "default",
        document_ids: Optional[List[str]] = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        reranking_method: RerankingMethod = RerankingMethod.CROSS_ENCODER,
        top_k: int = 5,
        config: Optional[RAGConfig] = None,
        enable_human_review: bool = False,
        connection_string: Optional[str] = None,
    ):
        """
        Initialize LangGraph RAG chain.

        Args:
            llm: LangChain LLM instance (kept for compatibility, uses factory internally)
            user_id: User ID for memory and document filtering
            document_ids: Optional document IDs to limit search scope
            retrieval_strategy: How to retrieve documents
            reranking_method: How to rerank retrieved documents
            top_k: Number of documents to use for context
            config: RAG configuration options
            enable_human_review: Whether to enable human review gate
            connection_string: PostgreSQL connection string for checkpointing
        """
        self.user_id = user_id
        self.document_ids = document_ids
        self.config = config or RAGConfig()
        self.top_k = top_k
        self.retrieval_strategy = retrieval_strategy
        self.reranking_method = reranking_method

        # Initialize LangGraph agent
        self.agent = RAGAgent(
            user_id=user_id,
            document_ids=document_ids,
            enable_human_review=enable_human_review,
            connection_string=connection_string,
        )

        # Legacy components for compatibility
        self.memory_service = get_memory_service() if self.config.use_memory else None
        self.circuit_breaker = CircuitBreaker()
        self.metrics = RAGMetrics()

        logger.info(
            f"LangGraphRAGChain initialized: user={user_id}, "
            f"strategy={retrieval_strategy.value}, "
            f"human_review={enable_human_review}"
        )

    @with_retry(max_retries=3, delay=1.0)
    async def invoke(self, question: str, thread_id: Optional[str] = None) -> RAGResponse:
        """
        Process question and return answer with sources.

        Args:
            question: User's question
            thread_id: Optional thread ID for conversation continuity

        Returns:
            RAGResponse with answer, sources, confidence, and metadata
        """
        start_time = time.time()
        self.metrics.total_queries += 1

        # Circuit breaker check
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker open")
            return self._fallback_response(question, "Service temporarily unavailable")

        try:
            # Get chat history for state
            chat_history = ""
            if self.memory_service:
                chat_history = self.memory_service.format_for_prompt(self.user_id)

            # Invoke LangGraph agent
            result = await self.agent.invoke(
                query=question,
                thread_id=thread_id,
                chat_history=chat_history,
            )

            # Extract results
            answer = result.get("answer", "")
            confidence = result.get("confidence", 0.0)
            citations = result.get("citations", [])
            metadata = result.get("metadata", {})

            # Update memory
            if self.memory_service and answer:
                self.memory_service.add_exchange(
                    user_id=self.user_id,
                    user_message=question,
                    ai_message=answer,
                )

            # Build sources from citations
            sources = [
                {
                    "source": c.get("source", "Unknown"),
                    "score": c.get("score", 0.0),
                    "snippet": c.get("snippet", ""),
                }
                for c in citations
            ]

            # Build retrieval scores
            retrieval_scores = [c.get("score", 0.0) for c in citations]

            # Update metrics
            total_time = time.time() - start_time
            self.metrics.successful_queries += 1
            self.metrics.update_averages(
                metadata.get("retrieval_time", 0),
                metadata.get("generation_time", 0),
                confidence,
            )
            self.circuit_breaker.record_success()

            return RAGResponse(
                answer=answer,
                sources=sources,
                context_used=metadata.get("context_preview", ""),
                retrieval_scores=retrieval_scores,
                processing_time=total_time,
                confidence=confidence,
                query_rewritten=metadata.get("enhanced_query"),
                metadata={
                    "question_type": metadata.get("query_type", "unknown"),
                    "retrieval_quality": metadata.get("retrieval_quality", 0.0),
                    "verification_passed": metadata.get("verification_passed", False),
                    "num_documents": metadata.get("documents_used", 0),
                    "thread_id": result.get("thread_id", ""),
                    "langgraph": True,  # Flag indicating LangGraph execution
                },
                retrieved_chunks=[],  # Could be populated from detailed results
            )

        except Exception as e:
            self.metrics.failed_queries += 1
            self.circuit_breaker.record_failure()
            logger.error(f"LangGraph RAG chain error: {e}\n{traceback.format_exc()}")
            raise

    async def stream(self, question: str, thread_id: Optional[str] = None) -> AsyncIterator[str]:
        """
        Stream the response token by token.

        Args:
            question: User's question
            thread_id: Optional thread ID for conversation continuity

        Yields:
            Response tokens as they're generated
        """
        start_time = time.time()
        self.metrics.total_queries += 1

        # Circuit breaker check
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker open during stream")
            self.metrics.failed_queries += 1
            yield "I apologize, but the service is temporarily unavailable. Please try again later."
            return

        try:
            # Get chat history
            chat_history = ""
            if self.memory_service:
                chat_history = self.memory_service.format_for_prompt(self.user_id)

            full_response = ""

            # Stream from LangGraph agent
            async for event in self.agent.stream(
                query=question,
                thread_id=thread_id,
                chat_history=chat_history,
            ):
                event_type = event.get("type", "")

                if event_type == "token":
                    token = event.get("content", "")
                    full_response += token
                    yield token

                elif event_type == "status":
                    # Optionally emit status updates
                    # Could yield status markers if client supports them
                    pass

                elif event_type == "complete":
                    # Stream complete
                    pass

            # Update memory after streaming completes
            if self.memory_service and full_response:
                self.memory_service.add_exchange(
                    user_id=self.user_id,
                    user_message=question,
                    ai_message=full_response,
                )

            # Record success
            self.metrics.successful_queries += 1
            self.circuit_breaker.record_success()
            logger.info(f"Stream completed in {time.time() - start_time:.2f}s")

        except Exception as e:
            self.metrics.failed_queries += 1
            self.circuit_breaker.record_failure()
            logger.error(f"Stream error: {e}\n{traceback.format_exc()}")
            yield "\n\nI apologize, but an error occurred while generating the response."

    def invoke_sync(self, question: str) -> str:
        """
        Synchronous invoke for simple use cases.

        Args:
            question: User's question

        Returns:
            Answer string
        """
        try:
            # Run async invoke in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new loop
                import nest_asyncio
                nest_asyncio.apply()

            result = asyncio.run(self.invoke(question))
            return result.answer

        except Exception as e:
            logger.error(f"Sync invoke error: {e}")
            return "I apologize, but an error occurred while processing your request."

    def _fallback_response(self, question: str, reason: str) -> RAGResponse:
        """Generate fallback response when RAG fails."""
        return RAGResponse(
            answer=f"I apologize, but I'm unable to process your question at the moment. {reason}.",
            sources=[],
            context_used="",
            retrieval_scores=[],
            confidence=0.0,
            metadata={"fallback": True, "reason": reason},
        )

    def clear_memory(self) -> None:
        """Clear conversation memory for current user."""
        if self.memory_service:
            self.memory_service.clear_history(self.user_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current chain metrics."""
        metrics = self.metrics.to_dict()
        metrics["circuit_breaker_state"] = self.circuit_breaker.state
        metrics["implementation"] = "langgraph"
        return metrics


# Backward compatible alias
RAGChain = LangGraphRAGChain


def create_rag_chain(
    llm=None,
    user_id: str = "default",
    document_ids: Optional[List[str]] = None,
    config: Optional[RAGConfig] = None,
    use_langgraph: bool = True,  # Default to LangGraph
    enable_human_review: bool = False,
    connection_string: Optional[str] = None,
    **kwargs,
) -> Union[LangGraphRAGChain, "LegacyRAGChain"]:
    """
    Factory function to create a RAG chain instance.

    Args:
        llm: Language model (if None, creates from factory)
        user_id: User ID for memory and document filtering
        document_ids: Optional document IDs to limit search
        config: RAG configuration
        use_langgraph: If True, uses LangGraph implementation (default)
        enable_human_review: Enable human review gate (LangGraph only)
        connection_string: PostgreSQL connection string (LangGraph only)
        **kwargs: Additional arguments (retrieval_strategy, reranking_method, top_k)

    Returns:
        Configured RAG chain instance

    Example:
        ```python
        # LangGraph implementation (default)
        rag_chain = create_rag_chain(user_id="user-123")

        # With human review enabled
        rag_chain = create_rag_chain(
            user_id="user-123",
            enable_human_review=True,
            connection_string="postgresql://..."
        )

        # Invoke
        response = await rag_chain.invoke("What is the refund policy?")

        # Stream
        async for token in rag_chain.stream("How do I cancel?"):
            print(token, end="")
        ```
    """
    if config is None:
        from app.core.config import settings

        config = RAGConfig(
            mode=RAGMode.CONVERSATIONAL,
            use_memory=getattr(settings, "rag_use_memory", True),
            use_query_rewriting=getattr(settings, "rag_use_query_rewriting", True),
            max_context_length=getattr(settings, "rag_max_context_length", 4000),
            include_sources=getattr(settings, "rag_include_sources", True),
        )

    if use_langgraph:
        return LangGraphRAGChain(
            llm=llm,
            user_id=user_id,
            document_ids=document_ids,
            config=config,
            enable_human_review=enable_human_review,
            connection_string=connection_string,
            **kwargs,
        )
    else:
        # Fall back to legacy implementation if explicitly requested
        # Import here to avoid circular imports
        from app.rag.pipeline.legacy_chain import LegacyRAGChain
        return LegacyRAGChain(
            llm=llm,
            user_id=user_id,
            document_ids=document_ids,
            config=config,
            **kwargs,
        )


def get_llm_provider() -> str:
    """Get the current LLM provider name."""
    from app.services.llm_factory import llm_factory
    return llm_factory.get_current_provider()


# Export for backward compatibility
__all__ = [
    "RAGChain",
    "LangGraphRAGChain",
    "create_rag_chain",
    "get_llm_provider",
]
