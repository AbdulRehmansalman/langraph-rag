"""
LangGraph RAG Agent Implementation
===================================

Production-grade RAG agent built with LangGraph featuring:
- Stateful conversation management
- Parallel retrieval from multiple sources
- Human-in-the-loop review gates
- Self-correction and verification
- Streaming support
- PostgreSQL checkpointing
- Comprehensive observability

Usage:
    from app.rag.langgraph import create_rag_graph, RAGState

    graph = create_rag_graph(user_id="user-123")

    # Invoke
    result = await graph.ainvoke({"messages": [HumanMessage(content="What is...")]}

    # Stream
    async for event in graph.astream_events({"messages": [...]}):
        print(event)
"""

from app.rag.langgraph.state import RAGState, QueryType, QueryIntent
from app.rag.langgraph.graph import create_rag_graph, RAGGraphBuilder

__all__ = [
    "RAGState",
    "QueryType",
    "QueryIntent",
    "create_rag_graph",
    "RAGGraphBuilder",
]
