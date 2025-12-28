"""
LangGraph RAG Agent Graph Assembly
==================================

Assembles all nodes into a complete RAG graph with:
- Conditional routing
- Parallel execution
- Human-in-the-loop interrupts
- PostgreSQL checkpointing
- Streaming support
"""

import logging
from typing import Any, AsyncIterator, Optional, Literal

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.rag.langgraph.state import RAGState, create_initial_state

# Import all nodes
from app.rag.langgraph.nodes.query_analysis import query_analysis_node
from app.rag.langgraph.nodes.query_enhancement import query_enhancement_node
from app.rag.langgraph.nodes.retrieval import (
    parallel_retrieval_node,
    merge_retrieval_results_node,
)
from app.rag.langgraph.nodes.quality_assessment import (
    quality_assessment_node,
    query_reformulation_node,
)
from app.rag.langgraph.nodes.reranking import context_reranking_node
from app.rag.langgraph.nodes.human_review import human_review_node
from app.rag.langgraph.nodes.generation import response_generation_node
from app.rag.langgraph.nodes.verification import self_correction_node
from app.rag.langgraph.nodes.formatting import response_formatting_node
from app.rag.langgraph.nodes.logging_node import logging_node

logger = logging.getLogger(__name__)


def _route_after_query_analysis(state: RAGState) -> str:
    """Route after query analysis based on query type."""
    query_analysis = state.get("query_analysis", {})
    query_type = query_analysis.get("query_type", "unknown")

    # Greetings don't need retrieval
    if query_type == "greeting":
        return "response_generation"

    # Unsafe content goes directly to formatting (with rejection message)
    if query_analysis.get("unsafe_content_detected", False):
        return "response_generation"

    # Normal flow continues to enhancement
    return "query_enhancement"


def _route_after_quality_assessment(state: RAGState) -> str:
    """Route after quality assessment."""
    quality_score = state.get("retrieval_quality_score", 0.0)
    correction_attempts = state.get("correction_attempts", 0)
    documents = state.get("retrieved_documents", [])

    # If no documents and we haven't tried reformulating
    if not documents and correction_attempts < 2:
        return "query_reformulation"

    # If quality is poor and we haven't tried reformulating
    if quality_score < 0.6 and correction_attempts < 2:
        return "query_reformulation"

    # Proceed to reranking
    return "context_reranking"


def _route_after_reranking(state: RAGState) -> str:
    """Route after context reranking."""
    query_analysis = state.get("query_analysis", {})

    # Check if human review is required
    if query_analysis.get("requires_human_review", False):
        return "human_review"

    return "response_generation"


def _route_after_human_review(state: RAGState) -> str:
    """Route after human review."""
    human_review = state.get("human_review", {})
    status = human_review.get("status", "not_required")

    if status == "rejected":
        return "response_formatting"  # Will use rejection message

    return "response_generation"


def _route_after_verification(state: RAGState) -> str:
    """Route after self-correction verification."""
    verification_passed = state.get("verification_passed", False)
    confidence = state.get("confidence_score", 0.0)
    correction_attempts = state.get("correction_attempts", 0)

    # If verification failed and we can still retry
    if not verification_passed and correction_attempts < 2 and confidence < 0.5:
        return "query_reformulation"

    # Proceed to formatting
    return "response_formatting"


def _should_continue(state: RAGState) -> str:
    """Determine if graph should continue or end."""
    if state.get("should_end", False):
        return END
    if state.get("has_error", False) and not state.get("fallback_used", False):
        return END
    return "continue"


class RAGGraphBuilder:
    """
    Builder class for constructing the RAG graph.

    Allows customization of nodes, checkpointing, and configuration.
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        enable_human_review: bool = True,
        enable_checkpointing: bool = True,
    ):
        """
        Initialize the graph builder.

        Args:
            user_id: User ID for document filtering and memory
            document_ids: Specific document IDs to search
            enable_human_review: Whether to enable human review gate
            enable_checkpointing: Whether to enable state checkpointing
        """
        self.user_id = user_id
        self.document_ids = document_ids
        self.enable_human_review = enable_human_review
        self.enable_checkpointing = enable_checkpointing
        self._checkpointer = None

    def with_postgres_checkpointer(self, connection_string: str) -> "RAGGraphBuilder":
        """
        Configure PostgreSQL checkpointer for persistence.

        Args:
            connection_string: PostgreSQL connection string

        Returns:
            Self for chaining
        """
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            self._checkpointer = PostgresSaver.from_conn_string(connection_string)
            logger.info("PostgreSQL checkpointer configured")
        except ImportError:
            logger.warning(
                "PostgresSaver not available, falling back to memory checkpointer"
            )
            self._checkpointer = MemorySaver()
        except Exception as e:
            logger.error(f"Failed to configure PostgreSQL checkpointer: {e}")
            self._checkpointer = MemorySaver()

        return self

    def with_memory_checkpointer(self) -> "RAGGraphBuilder":
        """
        Configure in-memory checkpointer.

        Returns:
            Self for chaining
        """
        self._checkpointer = MemorySaver()
        return self

    def build(self) -> StateGraph:
        """
        Build and return the complete RAG graph.

        Returns:
            Compiled StateGraph
        """
        # Create the graph
        graph = StateGraph(RAGState)

        # Add all nodes
        graph.add_node("query_analysis", query_analysis_node)
        graph.add_node("query_enhancement", query_enhancement_node)
        graph.add_node("parallel_retrieval", parallel_retrieval_node)
        graph.add_node("merge_results", merge_retrieval_results_node)
        graph.add_node("quality_assessment", quality_assessment_node)
        graph.add_node("query_reformulation", query_reformulation_node)
        graph.add_node("context_reranking", context_reranking_node)
        graph.add_node("response_generation", response_generation_node)
        graph.add_node("self_correction", self_correction_node)
        graph.add_node("response_formatting", response_formatting_node)
        graph.add_node("logging", logging_node)

        # Add human review node if enabled
        if self.enable_human_review:
            graph.add_node("human_review", human_review_node)

        # Set entry point
        graph.set_entry_point("query_analysis")

        # Add edges with conditional routing
        graph.add_conditional_edges(
            "query_analysis",
            _route_after_query_analysis,
            {
                "query_enhancement": "query_enhancement",
                "response_generation": "response_generation",
            }
        )

        graph.add_edge("query_enhancement", "parallel_retrieval")
        graph.add_edge("parallel_retrieval", "merge_results")
        graph.add_edge("merge_results", "quality_assessment")

        graph.add_conditional_edges(
            "quality_assessment",
            _route_after_quality_assessment,
            {
                "query_reformulation": "query_reformulation",
                "context_reranking": "context_reranking",
            }
        )

        graph.add_edge("query_reformulation", "parallel_retrieval")

        if self.enable_human_review:
            graph.add_conditional_edges(
                "context_reranking",
                _route_after_reranking,
                {
                    "human_review": "human_review",
                    "response_generation": "response_generation",
                }
            )
            graph.add_conditional_edges(
                "human_review",
                _route_after_human_review,
                {
                    "response_generation": "response_generation",
                    "response_formatting": "response_formatting",
                }
            )
        else:
            graph.add_edge("context_reranking", "response_generation")

        graph.add_edge("response_generation", "self_correction")

        graph.add_conditional_edges(
            "self_correction",
            _route_after_verification,
            {
                "query_reformulation": "query_reformulation",
                "response_formatting": "response_formatting",
            }
        )

        graph.add_edge("response_formatting", "logging")
        graph.add_edge("logging", END)

        # Get checkpointer
        checkpointer = self._checkpointer
        if checkpointer is None and self.enable_checkpointing:
            checkpointer = MemorySaver()

        # Compile the graph
        compiled = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["human_review"] if self.enable_human_review else None,
        )

        logger.info("RAG graph compiled successfully")
        return compiled


def create_rag_graph(
    user_id: Optional[str] = None,
    document_ids: Optional[list[str]] = None,
    enable_human_review: bool = False,  # Disabled by default for simplicity
    connection_string: Optional[str] = None,
) -> StateGraph:
    """
    Factory function to create a RAG graph.

    Args:
        user_id: User ID for document filtering
        document_ids: Specific document IDs to search
        enable_human_review: Whether to enable human review gate
        connection_string: PostgreSQL connection string for checkpointing

    Returns:
        Compiled StateGraph ready for execution

    Example:
        ```python
        graph = create_rag_graph(user_id="user-123")

        # Invoke
        result = await graph.ainvoke(
            create_initial_state("What is the refund policy?", user_id="user-123")
        )

        # Stream
        async for event in graph.astream_events(
            create_initial_state("How do I cancel?"),
            version="v2"
        ):
            print(event)
        ```
    """
    builder = RAGGraphBuilder(
        user_id=user_id,
        document_ids=document_ids,
        enable_human_review=enable_human_review,
    )

    if connection_string:
        builder.with_postgres_checkpointer(connection_string)
    else:
        builder.with_memory_checkpointer()

    return builder.build()


class RAGAgent:
    """
    High-level RAG agent interface.

    Provides a simple API for invoking the RAG graph.
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        enable_human_review: bool = False,
        connection_string: Optional[str] = None,
    ):
        """
        Initialize the RAG agent.

        Args:
            user_id: User ID for document filtering
            document_ids: Specific document IDs to search
            enable_human_review: Whether to enable human review
            connection_string: PostgreSQL connection string
        """
        self.user_id = user_id
        self.document_ids = document_ids
        self.graph = create_rag_graph(
            user_id=user_id,
            document_ids=document_ids,
            enable_human_review=enable_human_review,
            connection_string=connection_string,
        )

    async def invoke(
        self,
        query: str,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process a query and return the response.

        Args:
            query: User's question
            thread_id: Optional thread ID for conversation continuity
            **kwargs: Additional state overrides

        Returns:
            Dict containing response and metadata
        """
        initial_state = create_initial_state(
            query=query,
            user_id=self.user_id,
            document_ids=self.document_ids,
            thread_id=thread_id,
        )

        # Merge any additional kwargs
        for key, value in kwargs.items():
            if key in initial_state:
                initial_state[key] = value

        config = {"configurable": {"thread_id": initial_state["thread_id"]}}

        result = await self.graph.ainvoke(initial_state, config=config)

        return {
            "answer": result.get("final_response", ""),
            "confidence": result.get("confidence_score", 0.0),
            "citations": result.get("citations", []),
            "related_questions": result.get("related_questions", []),
            "metadata": result.get("response_metadata", {}),
            "thread_id": result.get("thread_id", ""),
        }

    async def stream(
        self,
        query: str,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream the response token by token.

        Args:
            query: User's question
            thread_id: Optional thread ID for conversation continuity
            **kwargs: Additional state overrides

        Yields:
            Dict containing token or status updates
        """
        initial_state = create_initial_state(
            query=query,
            user_id=self.user_id,
            document_ids=self.document_ids,
            thread_id=thread_id,
        )

        config = {"configurable": {"thread_id": initial_state["thread_id"]}}

        async for event in self.graph.astream_events(
            initial_state,
            config=config,
            version="v2",
        ):
            event_type = event.get("event", "")
            event_name = event.get("name", "")

            # Emit node status updates
            if event_type == "on_chain_start":
                yield {
                    "type": "status",
                    "node": event_name,
                    "status": "starting",
                }

            elif event_type == "on_chain_end":
                yield {
                    "type": "status",
                    "node": event_name,
                    "status": "complete",
                }

            # Emit LLM tokens
            elif event_type == "on_llm_stream":
                chunk = event.get("data", {}).get("chunk", "")
                if hasattr(chunk, "content"):
                    yield {
                        "type": "token",
                        "content": chunk.content,
                    }

            # Emit final result
            elif event_type == "on_chain_end" and event_name == "LangGraph":
                output = event.get("data", {}).get("output", {})
                yield {
                    "type": "complete",
                    "answer": output.get("final_response", ""),
                    "confidence": output.get("confidence_score", 0.0),
                    "citations": output.get("citations", []),
                }
