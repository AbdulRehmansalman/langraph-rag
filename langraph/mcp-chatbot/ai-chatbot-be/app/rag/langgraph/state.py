"""
LangGraph RAG State Schema
==========================

Comprehensive state structure for the RAG agent graph.
Maintains complete context throughout the conversation and retrieval process.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional, Sequence, TypedDict
from uuid import uuid4

from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Classification of query types for routing."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    OPINION = "opinion"
    CLARIFICATION = "clarification"
    GREETING = "greeting"
    UNKNOWN = "unknown"


class QueryIntent(str, Enum):
    """User intent classification."""
    SEARCH = "search"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    EXPLANATION = "explanation"
    CALCULATION = "calculation"
    VERIFICATION = "verification"
    NAVIGATION = "navigation"
    UNKNOWN = "unknown"


class RetrievalStrategy(str, Enum):
    """Document retrieval strategies."""
    SIMILARITY = "similarity"
    MMR = "mmr"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"
    HYDE = "hyde"


class SensitivityLevel(str, Enum):
    """Content sensitivity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HumanReviewStatus(str, Enum):
    """Human review gate status."""
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class DocumentChunk(BaseModel):
    """Retrieved document chunk with metadata."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(default="")
    source: str = Field(default="")
    score: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    timestamp: Optional[datetime] = None
    author: Optional[str] = None
    category: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class Citation(BaseModel):
    """Source citation for response."""
    index: int = Field(description="Citation number [1], [2], etc.")
    document_id: str = Field(default="")
    source: str = Field(default="")
    snippet: str = Field(default="")
    score: float = Field(default=0.0)
    page_number: Optional[int] = None


class QueryAnalysis(BaseModel):
    """Structured analysis of user query."""
    query_type: QueryType = QueryType.UNKNOWN
    intent: QueryIntent = QueryIntent.UNKNOWN
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    required_retrieval: bool = True
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    sensitivity_level: SensitivityLevel = SensitivityLevel.NONE
    requires_human_review: bool = False
    is_follow_up: bool = False
    complexity_score: float = 0.5
    unsafe_content_detected: bool = False
    detected_topics: list[str] = Field(default_factory=list)


class ErrorEntry(BaseModel):
    """Error log entry."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    node: str = ""
    error_type: str = ""
    message: str = ""
    recoverable: bool = True
    retry_count: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class NodeMetrics(BaseModel):
    """Metrics for a single node execution."""
    node_name: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None


class ExecutionMetrics(BaseModel):
    """Overall execution metrics."""
    total_duration_ms: float = 0.0
    node_metrics: list[NodeMetrics] = Field(default_factory=list)
    total_tokens_used: int = 0
    retrieval_count: int = 0
    documents_retrieved: int = 0
    reranking_applied: bool = False
    tools_used: list[str] = Field(default_factory=list)
    retry_count: int = 0
    cache_hit: bool = False


class UserFeedback(BaseModel):
    """User feedback on response."""
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    helpful: Optional[bool] = None
    accurate: Optional[bool] = None
    comment: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HumanReviewDecision(BaseModel):
    """Human review decision."""
    status: HumanReviewStatus = HumanReviewStatus.NOT_REQUIRED
    reviewer_id: Optional[str] = None
    decision_time: Optional[datetime] = None
    reason: Optional[str] = None
    modified_context: Optional[str] = None
    approved_documents: list[str] = Field(default_factory=list)
    rejected_documents: list[str] = Field(default_factory=list)


class RAGState(TypedDict, total=False):
    """
    Complete state schema for the RAG Agent graph.

    This state flows through all nodes and maintains the complete
    context of the conversation and retrieval process.

    Using TypedDict for LangGraph compatibility with proper
    message annotation for automatic message merging.
    """

    # === Conversation Management ===
    messages: Annotated[Sequence[AnyMessage], add_messages]
    thread_id: str
    timestamp: str  # ISO format string for serialization

    # === Query Processing ===
    original_query: str
    cleaned_query: str
    enhanced_query: str
    query_variations: list[str]
    query_analysis: dict[str, Any]  # QueryAnalysis as dict for serialization

    # === Retrieval ===
    retrieved_documents: list[dict[str, Any]]  # DocumentChunk as dicts
    vector_search_results: list[dict[str, Any]]
    keyword_search_results: list[dict[str, Any]]
    metadata_filter_results: list[dict[str, Any]]
    web_search_results: list[dict[str, Any]]
    retrieval_attempts: int
    retrieval_quality_score: float

    # === Context & Reranking ===
    reranked_documents: list[dict[str, Any]]
    compressed_context: str
    context_token_count: int

    # === Human Review ===
    human_review: dict[str, Any]  # HumanReviewDecision as dict

    # === Response Generation ===
    generated_response: str
    citations: list[dict[str, Any]]  # Citation as dicts
    confidence_score: float
    is_grounded: bool
    hallucination_score: float

    # === Verification & Correction ===
    verification_passed: bool
    correction_attempts: int
    reformulated_query: Optional[str]

    # === Final Output ===
    final_response: str
    related_questions: list[str]
    response_metadata: dict[str, Any]

    # === Error Handling ===
    error_log: list[dict[str, Any]]  # ErrorEntry as dicts
    has_error: bool
    fallback_used: bool

    # === User Context ===
    user_id: Optional[str]
    session_id: Optional[str]
    document_ids: Optional[list[str]]
    user_preferences: dict[str, Any]
    user_feedback: Optional[dict[str, Any]]

    # === Metrics & Monitoring ===
    metrics: dict[str, Any]  # ExecutionMetrics as dict
    current_node: str
    node_start_time: float

    # === Memory ===
    conversation_summary: Optional[str]
    relevant_history: list[dict[str, Any]]
    chat_history: str

    # === Routing ===
    next_node: Optional[str]
    should_end: bool


def create_initial_state(
    query: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    document_ids: Optional[list[str]] = None,
    thread_id: Optional[str] = None,
) -> RAGState:
    """
    Create initial state for a new RAG query.

    Args:
        query: User's query
        user_id: Optional user identifier
        session_id: Optional session identifier
        document_ids: Optional list of document IDs to search
        thread_id: Optional thread ID for conversation continuity

    Returns:
        Initialized RAGState
    """
    from langchain_core.messages import HumanMessage

    return RAGState(
        # Conversation
        messages=[HumanMessage(content=query)],
        thread_id=thread_id or str(uuid4()),
        timestamp=datetime.utcnow().isoformat(),

        # Query
        original_query=query,
        cleaned_query="",
        enhanced_query="",
        query_variations=[],
        query_analysis=QueryAnalysis().model_dump(),

        # Retrieval
        retrieved_documents=[],
        vector_search_results=[],
        keyword_search_results=[],
        metadata_filter_results=[],
        web_search_results=[],
        retrieval_attempts=0,
        retrieval_quality_score=0.0,

        # Context
        reranked_documents=[],
        compressed_context="",
        context_token_count=0,

        # Human Review
        human_review=HumanReviewDecision().model_dump(),

        # Response
        generated_response="",
        citations=[],
        confidence_score=0.0,
        is_grounded=True,
        hallucination_score=0.0,

        # Verification
        verification_passed=False,
        correction_attempts=0,
        reformulated_query=None,

        # Output
        final_response="",
        related_questions=[],
        response_metadata={},

        # Errors
        error_log=[],
        has_error=False,
        fallback_used=False,

        # User
        user_id=user_id,
        session_id=session_id,
        document_ids=document_ids,
        user_preferences={},
        user_feedback=None,

        # Metrics
        metrics=ExecutionMetrics().model_dump(),
        current_node="",
        node_start_time=0.0,

        # Memory
        conversation_summary=None,
        relevant_history=[],
        chat_history="",

        # Routing
        next_node=None,
        should_end=False,
    )


def add_error_to_state(
    state: RAGState,
    node: str,
    error_type: str,
    message: str,
    recoverable: bool = True,
    details: Optional[dict] = None
) -> RAGState:
    """Add an error entry to state."""
    error_entry = ErrorEntry(
        node=node,
        error_type=error_type,
        message=message,
        recoverable=recoverable,
        retry_count=state.get("retrieval_attempts", 0),
        details=details or {}
    ).model_dump()

    error_log = state.get("error_log", []).copy()
    error_log.append(error_entry)

    return {
        **state,
        "error_log": error_log,
        "has_error": not recoverable,
    }


def get_context_documents(state: RAGState) -> list[dict[str, Any]]:
    """Get the best documents for context from state."""
    if state.get("reranked_documents"):
        return state["reranked_documents"]
    return state.get("retrieved_documents", [])
