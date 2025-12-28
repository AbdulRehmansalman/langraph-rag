"""
Query Enhancement Node
======================

Optimizes queries for better retrieval through:
- Context-aware query rewriting
- Query expansion with synonyms
- Multiple query variation generation
- Acronym expansion
"""

import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.rag.langgraph.state import RAGState

logger = logging.getLogger(__name__)

# Query rewriting prompt
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query optimization expert. Your task is to rewrite the user's query
to improve document retrieval. Consider the conversation history for context.

Guidelines:
- Make the query more specific and searchable
- Expand acronyms if present
- Add relevant context from chat history
- Keep the core intent intact
- Output ONLY the rewritten query, nothing else

If the query is already clear and specific, return it unchanged."""),
    ("human", """Chat History:
{chat_history}

Original Query: {query}

Rewritten Query:"""),
])

# Query variation prompt
VARIATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Generate 3 alternative versions of the query for diverse retrieval.
Each variation should capture the same intent but use different words/phrasings.
Output one variation per line, no numbering or bullets."""),
    ("human", "Query: {query}\n\nVariations:"),
])


async def query_enhancement_node(state: RAGState) -> dict[str, Any]:
    """
    Enhance query for improved retrieval.

    This node:
    1. Rewrites query with conversation context
    2. Generates query variations for diverse retrieval
    3. Prepares enhanced query for retrieval

    Args:
        state: Current graph state

    Returns:
        Updated state with enhanced query
    """
    start_time = time.time()
    logger.info("Starting query enhancement")

    cleaned_query = state.get("cleaned_query", "")
    if not cleaned_query:
        cleaned_query = state.get("original_query", "")

    chat_history = state.get("chat_history", "")
    query_analysis = state.get("query_analysis", {})

    # Check if enhancement is needed
    if not query_analysis.get("required_retrieval", True):
        logger.info("Retrieval not required, skipping enhancement")
        return {
            "enhanced_query": cleaned_query,
            "query_variations": [],
            "current_node": "query_enhancement",
            "next_node": "response_generation",
        }

    enhanced_query = cleaned_query
    query_variations = []

    try:
        # Get LLM for rewriting
        from app.services.llm_factory import llm_factory
        llm = llm_factory.create_llm()

        # Rewrite query if there's chat history and it's a follow-up
        if chat_history and query_analysis.get("is_follow_up", False):
            rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
            enhanced_query = await rewrite_chain.ainvoke({
                "chat_history": chat_history[-2000:],  # Limit history
                "query": cleaned_query,
            })
            enhanced_query = enhanced_query.strip()
            logger.info(f"Query rewritten: {cleaned_query[:50]} -> {enhanced_query[:50]}")

        # Generate variations for complex queries
        complexity = query_analysis.get("complexity_score", 0.5)
        if complexity > 0.5:
            try:
                variation_chain = VARIATION_PROMPT | llm | StrOutputParser()
                variations_text = await variation_chain.ainvoke({"query": enhanced_query})
                query_variations = [
                    v.strip() for v in variations_text.strip().split("\n")
                    if v.strip() and len(v.strip()) > 5
                ][:3]  # Limit to 3 variations
                logger.info(f"Generated {len(query_variations)} query variations")
            except Exception as e:
                logger.warning(f"Failed to generate variations: {e}")

    except Exception as e:
        logger.error(f"Query enhancement error: {e}")
        # Fall back to original query
        enhanced_query = cleaned_query

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Query enhancement complete in {duration_ms:.1f}ms")

    return {
        "enhanced_query": enhanced_query,
        "query_variations": query_variations,
        "current_node": "query_enhancement",
        "next_node": "parallel_retrieval",
    }
