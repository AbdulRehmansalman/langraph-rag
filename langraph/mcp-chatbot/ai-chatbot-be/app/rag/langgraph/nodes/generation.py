"""
Response Generation Node
========================

Generates responses using LLM with:
- Context-grounded answers
- Citation formatting
- Streaming support
- Tool integration
"""

import logging
import re
import time
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.rag.langgraph.state import RAGState, Citation, QueryType

logger = logging.getLogger(__name__)

# System prompts for different modes
SYSTEM_PROMPTS = {
    "strict": """You are a helpful assistant that answers questions based ONLY on the provided context.

CRITICAL RULES:
1. ONLY use information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question based on the available documents."
3. NEVER make up information or use external knowledge
4. Always cite your sources using [1], [2], etc. notation
5. Be concise but thorough

Context:
{context}""",

    "conversational": """You are a helpful assistant that answers questions based on the provided context.

Guidelines:
1. Use the provided context as your primary source
2. You may provide brief clarifications, but stay grounded in the context
3. Cite sources using [1], [2], etc. notation when referencing specific information
4. If the context is insufficient, acknowledge this clearly
5. Be conversational but accurate

Context:
{context}

Chat History:
{chat_history}""",

    "creative": """You are a helpful assistant that uses the provided context to answer questions.

Guidelines:
1. Use the context as your foundation
2. You may draw reasonable inferences from the context
3. Cite sources using [1], [2], etc. when referencing specific facts
4. Be engaging and helpful
5. If uncertain, indicate your confidence level

Context:
{context}""",
}

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "{question}"),
])

# Fallback response when no context available
FALLBACK_RESPONSE = (
    "I apologize, but I couldn't find relevant information in the available documents "
    "to answer your question. Could you please rephrase your question or ask about "
    "something else?"
)

# Greeting responses
GREETING_RESPONSES = [
    "Hello! How can I help you today?",
    "Hi there! What would you like to know?",
    "Hey! I'm here to help. What's on your mind?",
]


def _get_system_prompt(mode: str) -> str:
    """Get system prompt for the given mode."""
    return SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["conversational"])


def _extract_citations(response: str, documents: list[dict]) -> list[dict]:
    """Extract citations from response and match to documents."""
    citations = []
    citation_pattern = r"\[(\d+)\]"
    matches = re.findall(citation_pattern, response)

    seen_indices = set()
    for match in matches:
        index = int(match)
        if index not in seen_indices and 0 < index <= len(documents):
            seen_indices.add(index)
            doc = documents[index - 1]
            citations.append(Citation(
                index=index,
                document_id=doc.get("id", ""),
                source=doc.get("source", "Unknown"),
                snippet=doc.get("content", "")[:200],
                score=doc.get("score", 0.0),
            ).model_dump())

    return citations


async def response_generation_node(state: RAGState) -> dict[str, Any]:
    """
    Generate response using LLM.

    This node:
    1. Builds prompt with context and chat history
    2. Generates response using LLM
    3. Extracts citations
    4. Handles edge cases (no context, greetings)

    Args:
        state: Current graph state

    Returns:
        Updated state with generated response
    """
    start_time = time.time()
    logger.info("Starting response generation")

    query = state.get("original_query", "")
    context = state.get("compressed_context", "")
    chat_history = state.get("chat_history", "")
    documents = state.get("reranked_documents", [])
    query_analysis = state.get("query_analysis", {})
    query_type = query_analysis.get("query_type", QueryType.UNKNOWN.value)

    # Handle greetings
    if query_type == QueryType.GREETING.value:
        import random
        response = random.choice(GREETING_RESPONSES)
        return {
            "generated_response": response,
            "citations": [],
            "confidence_score": 1.0,
            "is_grounded": True,
            "current_node": "response_generation",
            "next_node": "response_formatting",
            "messages": [AIMessage(content=response)],
        }

    # Handle no context (fallback)
    if not context or not documents:
        logger.warning("No context available, using fallback response")
        return {
            "generated_response": FALLBACK_RESPONSE,
            "citations": [],
            "confidence_score": 0.0,
            "is_grounded": False,
            "fallback_used": True,
            "current_node": "response_generation",
            "next_node": "response_formatting",
            "messages": [AIMessage(content=FALLBACK_RESPONSE)],
        }

    # Handle unsafe content
    if query_analysis.get("unsafe_content_detected", False):
        response = (
            "I'm not able to assist with that request. "
            "Please ask a different question."
        )
        return {
            "generated_response": response,
            "citations": [],
            "confidence_score": 0.0,
            "is_grounded": True,
            "current_node": "response_generation",
            "next_node": "response_formatting",
            "messages": [AIMessage(content=response)],
        }

    try:
        from app.services.llm_factory import llm_factory
        llm = llm_factory.create_llm()

        # Determine mode based on query analysis
        mode = "conversational"
        if query_analysis.get("sensitivity_level", "none") in ["high", "critical"]:
            mode = "strict"

        system_prompt = _get_system_prompt(mode).format(
            context=context,
            chat_history=chat_history,
        )

        chain = RESPONSE_PROMPT | llm | StrOutputParser()
        response = await chain.ainvoke({
            "system_prompt": system_prompt,
            "question": query,
        })

        # Extract citations
        citations = _extract_citations(response, documents)

        logger.info(f"Generated response with {len(citations)} citations")

    except Exception as e:
        logger.error(f"Response generation error: {e}")
        response = (
            "I apologize, but I encountered an error while generating a response. "
            "Please try again."
        )
        citations = []

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Response generation complete in {duration_ms:.1f}ms")

    return {
        "generated_response": response,
        "citations": citations,
        "current_node": "response_generation",
        "next_node": "self_correction",
        "messages": [AIMessage(content=response)],
    }


async def stream_response_generation(
    state: RAGState
) -> AsyncIterator[dict[str, Any]]:
    """
    Stream response generation token by token.

    This is an alternative to the batch generation for streaming use cases.

    Args:
        state: Current graph state

    Yields:
        Token updates and final state
    """
    logger.info("Starting streaming response generation")

    query = state.get("original_query", "")
    context = state.get("compressed_context", "")
    chat_history = state.get("chat_history", "")
    documents = state.get("reranked_documents", [])
    query_analysis = state.get("query_analysis", {})

    # Handle edge cases first
    if query_analysis.get("query_type") == QueryType.GREETING.value:
        import random
        response = random.choice(GREETING_RESPONSES)
        yield {"token": response, "done": True}
        return

    if not context or not documents:
        yield {"token": FALLBACK_RESPONSE, "done": True}
        return

    try:
        from app.services.llm_factory import llm_factory
        llm = llm_factory.create_llm()

        mode = "conversational"
        system_prompt = _get_system_prompt(mode).format(
            context=context,
            chat_history=chat_history,
        )

        chain = RESPONSE_PROMPT | llm

        full_response = ""
        async for chunk in chain.astream({
            "system_prompt": system_prompt,
            "question": query,
        }):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_response += token
            yield {"token": token, "done": False}

        # Final yield with citations
        citations = _extract_citations(full_response, documents)
        yield {
            "token": "",
            "done": True,
            "full_response": full_response,
            "citations": citations,
        }

    except Exception as e:
        logger.error(f"Streaming generation error: {e}")
        yield {
            "token": "An error occurred while generating the response.",
            "done": True,
            "error": str(e),
        }
