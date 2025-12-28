"""
Query Analysis Node
===================

Analyzes incoming queries for:
- Query type classification (factual, analytical, conversational, etc.)
- Intent detection (search, comparison, summarization, etc.)
- Retrieval strategy selection
- Unsafe content detection
- Sensitivity level assessment
"""

import logging
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage

from app.rag.langgraph.state import (
    RAGState,
    QueryType,
    QueryIntent,
    RetrievalStrategy,
    SensitivityLevel,
    QueryAnalysis,
)

logger = logging.getLogger(__name__)

# Query type patterns
QUERY_TYPE_PATTERNS = {
    QueryType.FACTUAL: [
        r"\bwhat is\b", r"\bwho is\b", r"\bhow many\b", r"\bwhen did\b",
        r"\bwhere is\b", r"\bdefine\b", r"\bwhat are\b",
    ],
    QueryType.ANALYTICAL: [
        r"\bwhy\b", r"\bhow does\b", r"\bexplain\b", r"\banalyze\b",
        r"\bwhat causes\b", r"\breason for\b",
    ],
    QueryType.COMPARATIVE: [
        r"\bcompare\b", r"\bversus\b", r"\bvs\b", r"\bdifference between\b",
        r"\bbetter than\b", r"\bwhich is\b",
    ],
    QueryType.PROCEDURAL: [
        r"\bhow to\b", r"\bsteps to\b", r"\bguide\b", r"\bprocess of\b",
        r"\bprocedure\b", r"\binstructions\b",
    ],
    QueryType.OPINION: [
        r"\bshould i\b", r"\bis it better\b", r"\bwhat do you think\b",
        r"\brecommend\b", r"\badvice\b",
    ],
    QueryType.GREETING: [
        r"^hi\b", r"^hello\b", r"^hey\b", r"^good morning\b",
        r"^good afternoon\b", r"^good evening\b",
    ],
}

# Intent patterns
INTENT_PATTERNS = {
    QueryIntent.SEARCH: [r"\bfind\b", r"\bsearch\b", r"\blook for\b", r"\blocate\b"],
    QueryIntent.COMPARISON: [r"\bcompare\b", r"\bdifference\b", r"\bvs\b"],
    QueryIntent.SUMMARIZATION: [r"\bsummarize\b", r"\bsummary\b", r"\boverview\b"],
    QueryIntent.EXPLANATION: [r"\bexplain\b", r"\bwhy\b", r"\bhow\b"],
    QueryIntent.CALCULATION: [r"\bcalculate\b", r"\bcompute\b", r"\bhow much\b"],
    QueryIntent.VERIFICATION: [r"\bis it true\b", r"\bverify\b", r"\bconfirm\b"],
}

# Sensitive topic patterns
SENSITIVE_PATTERNS = {
    "medical": [r"\bmedical\b", r"\bhealth\b", r"\bdisease\b", r"\bsymptom\b", r"\bdiagnos\b"],
    "financial": [r"\binvest\b", r"\bstock\b", r"\btax\b", r"\bloan\b", r"\bcredit\b"],
    "legal": [r"\blegal\b", r"\blaw\b", r"\bcourt\b", r"\battorney\b", r"\bcontract\b"],
    "pii": [r"\bssn\b", r"\bsocial security\b", r"\bpassword\b", r"\bcredit card\b"],
}

# Unsafe content patterns
UNSAFE_PATTERNS = [
    r"\bhack\b", r"\bexploit\b", r"\billegal\b", r"\bharm\b",
    r"\bviolent\b", r"\bweapon\b", r"\bdrug\b",
]


def _clean_query(query: str) -> str:
    """Clean and normalize query text."""
    # Remove extra whitespace
    query = " ".join(query.split())
    # Remove control characters
    query = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", query)
    return query.strip()


def _detect_query_type(query: str) -> QueryType:
    """Detect the type of query."""
    query_lower = query.lower()

    for query_type, patterns in QUERY_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return query_type

    return QueryType.UNKNOWN


def _detect_intent(query: str) -> QueryIntent:
    """Detect user intent from query."""
    query_lower = query.lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return intent

    return QueryIntent.SEARCH  # Default to search


def _extract_keywords(query: str) -> list[str]:
    """Extract keywords from query."""
    # Simple keyword extraction - remove stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into",
        "about", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "or", "and", "but", "if", "because", "until",
        "while", "how", "why", "when", "where", "i", "me", "my", "you",
        "your", "it", "its", "we", "they", "them", "their",
    }

    words = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
    keywords = [w for w in words if w not in stopwords]

    return keywords[:10]  # Limit to 10 keywords


def _detect_sensitivity(query: str) -> tuple[SensitivityLevel, list[str]]:
    """Detect sensitivity level and topics."""
    query_lower = query.lower()
    detected_topics = []

    for topic, patterns in SENSITIVE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                detected_topics.append(topic)
                break

    if "pii" in detected_topics:
        return SensitivityLevel.CRITICAL, detected_topics
    elif "medical" in detected_topics or "legal" in detected_topics:
        return SensitivityLevel.HIGH, detected_topics
    elif "financial" in detected_topics:
        return SensitivityLevel.MEDIUM, detected_topics
    elif detected_topics:
        return SensitivityLevel.LOW, detected_topics

    return SensitivityLevel.NONE, detected_topics


def _detect_unsafe_content(query: str) -> bool:
    """Detect potentially unsafe content."""
    query_lower = query.lower()

    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, query_lower):
            return True

    return False


def _select_retrieval_strategy(
    query_type: QueryType,
    intent: QueryIntent,
    complexity: float
) -> RetrievalStrategy:
    """Select optimal retrieval strategy based on query characteristics."""
    # Complex analytical queries benefit from ensemble
    if query_type == QueryType.ANALYTICAL and complexity > 0.7:
        return RetrievalStrategy.ENSEMBLE

    # Factual queries work well with hybrid
    if query_type == QueryType.FACTUAL:
        return RetrievalStrategy.HYBRID

    # Comparative queries need diverse results
    if query_type == QueryType.COMPARATIVE:
        return RetrievalStrategy.MMR

    # Default to hybrid for best balance
    return RetrievalStrategy.HYBRID


def _calculate_complexity(query: str, keywords: list[str]) -> float:
    """Calculate query complexity score (0-1)."""
    # Factors: length, keyword count, question depth
    length_score = min(len(query) / 200, 1.0)
    keyword_score = min(len(keywords) / 8, 1.0)

    # Check for complex patterns
    complex_patterns = [
        r"\band\b.*\band\b",  # Multiple conjunctions
        r"\bif\b.*\bthen\b",  # Conditional
        r"\bbecause\b",      # Causal
        r"\balthough\b",     # Concessive
    ]

    pattern_score = 0
    for pattern in complex_patterns:
        if re.search(pattern, query.lower()):
            pattern_score += 0.25

    return min((length_score + keyword_score + pattern_score) / 3, 1.0)


def _is_follow_up(query: str, messages: list) -> bool:
    """Detect if query is a follow-up to previous conversation."""
    follow_up_patterns = [
        r"^(and|also|what about|how about|tell me more)",
        r"^(can you|could you|please)",
        r"\b(it|this|that|they|them)\b",  # Pronouns suggesting context
    ]

    query_lower = query.lower()
    for pattern in follow_up_patterns:
        if re.search(pattern, query_lower):
            return True

    # If there are previous messages, likely a follow-up
    return len(messages) > 1


async def query_analysis_node(state: RAGState) -> dict[str, Any]:
    """
    Analyze incoming query and prepare for routing.

    This node:
    1. Cleans and normalizes the query
    2. Classifies query type and intent
    3. Extracts keywords and entities
    4. Assesses sensitivity and safety
    5. Selects retrieval strategy

    Args:
        state: Current graph state

    Returns:
        Updated state with query analysis
    """
    start_time = time.time()
    logger.info("Starting query analysis")

    # Get query from messages
    messages = state.get("messages", [])
    if not messages:
        logger.warning("No messages in state")
        return {
            "has_error": True,
            "error_log": [{
                "node": "query_analysis",
                "error_type": "ValueError",
                "message": "No messages in state",
                "recoverable": False,
            }],
        }

    # Get the last human message
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            query = msg.content if hasattr(msg, "content") else str(msg)
            break

    if not query:
        query = state.get("original_query", "")

    if not query:
        logger.warning("No query found in messages or state")
        return {"has_error": True}

    # Clean query
    cleaned_query = _clean_query(query)

    # Detect query characteristics
    query_type = _detect_query_type(cleaned_query)
    intent = _detect_intent(cleaned_query)
    keywords = _extract_keywords(cleaned_query)
    sensitivity_level, detected_topics = _detect_sensitivity(cleaned_query)
    unsafe_content = _detect_unsafe_content(cleaned_query)
    is_follow_up = _is_follow_up(cleaned_query, messages)
    complexity = _calculate_complexity(cleaned_query, keywords)

    # Determine if human review is needed
    requires_human_review = (
        sensitivity_level in [SensitivityLevel.HIGH, SensitivityLevel.CRITICAL]
        or unsafe_content
    )

    # Select retrieval strategy
    retrieval_strategy = _select_retrieval_strategy(query_type, intent, complexity)

    # Build analysis result
    analysis = QueryAnalysis(
        query_type=query_type,
        intent=intent,
        keywords=keywords,
        entities=[],  # TODO: Add entity extraction
        required_retrieval=query_type not in [QueryType.GREETING],
        retrieval_strategy=retrieval_strategy,
        sensitivity_level=sensitivity_level,
        requires_human_review=requires_human_review,
        is_follow_up=is_follow_up,
        complexity_score=complexity,
        unsafe_content_detected=unsafe_content,
        detected_topics=detected_topics,
    )

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Query analysis complete: type={query_type.value}, "
        f"intent={intent.value}, duration={duration_ms:.1f}ms"
    )

    # Determine next node
    if query_type == QueryType.GREETING:
        next_node = "response_generation"
    elif unsafe_content:
        next_node = "response_formatting"  # Skip retrieval for unsafe content
    else:
        next_node = "query_enhancement"

    return {
        "original_query": query,
        "cleaned_query": cleaned_query,
        "query_analysis": analysis.model_dump(),
        "current_node": "query_analysis",
        "next_node": next_node,
        "node_start_time": start_time,
    }
