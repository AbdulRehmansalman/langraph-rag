"""
Query Processor
===============
Query preprocessing and enhancement for the RAG pipeline.

Features:
- Query cleaning and normalization
- Question type detection
- Entity and keyword extraction
- Query rewriting with conversation context
"""

import logging
from typing import Dict, Any, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.rag.utils.text_utils import STOPWORDS, extract_keywords, clean_text

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """
    Preprocesses queries for better retrieval.

    Features:
    - Query expansion with synonyms
    - Spell correction
    - Query classification
    - Intent detection
    """

    # Question type patterns for classification
    QUESTION_PATTERNS = {
        "factual": [
            "what is", "what are", "who is", "when did",
            "where is", "how many", "what do"
        ],
        "explanatory": ["why", "how does", "explain", "describe"],
        "comparative": ["compare", "difference between", "versus", "vs"],
        "procedural": ["how to", "steps to", "guide to", "process of"],
        "opinion": ["should i", "is it better", "what do you think"],
    }

    def __init__(self, llm=None):
        """
        Initialize query preprocessor.

        Args:
            llm: Optional LLM for advanced query rewriting
        """
        self.llm = llm

    def preprocess(self, query: str) -> Dict[str, Any]:
        """
        Preprocess query and extract metadata.

        Args:
            query: Raw user query

        Returns:
            Dict with processed query and metadata:
            - original: Original query
            - cleaned: Cleaned query
            - expanded: Expanded query (future use)
            - question_type: Detected question type
            - entities: Extracted entities
            - keywords: Extracted keywords
        """
        # Clean query
        cleaned = self._clean_query(query)

        # Detect question type
        question_type = self._detect_question_type(cleaned)

        # Extract key entities
        entities = self._extract_entities(cleaned)

        # Extract keywords
        keywords = extract_keywords(cleaned, stopwords=STOPWORDS, min_length=2)

        return {
            "original": query,
            "cleaned": cleaned,
            "expanded": cleaned,  # Placeholder for future query expansion
            "question_type": question_type,
            "entities": entities,
            "keywords": keywords,
        }

    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize query.

        Args:
            query: Raw query string

        Returns:
            Cleaned query string
        """
        # Clean using shared utility
        query = clean_text(query)

        # Ensure question ends with question mark if it's a question
        question_starters = ["what", "who", "when", "where", "why", "how"]
        if any(query.lower().startswith(w) for w in question_starters):
            if not query.endswith("?"):
                query += "?"

        return query

    def _detect_question_type(self, query: str) -> str:
        """
        Detect the type of question being asked.

        Args:
            query: Cleaned query string

        Returns:
            Question type string
        """
        query_lower = query.lower()

        for qtype, patterns in self.QUESTION_PATTERNS.items():
            if any(pattern in query_lower for pattern in patterns):
                return qtype

        return "general"

    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract key entities from query.

        Uses simple capitalization-based entity extraction.
        For production, consider using NER models.

        Args:
            query: Query string

        Returns:
            List of extracted entities
        """
        words = query.split()
        entities = [
            w for w in words
            if w and w[0].isupper() and len(w) > 2
        ]
        return entities

    async def rewrite_with_context(
        self,
        query: str,
        chat_history: str,
        llm=None
    ) -> str:
        """
        Rewrite query incorporating conversation context.

        Takes a follow-up question and conversation history,
        returns a standalone question suitable for retrieval.

        Args:
            query: Current query
            chat_history: Formatted conversation history
            llm: LLM to use (overrides instance LLM)

        Returns:
            Rewritten standalone query
        """
        if not chat_history:
            return query

        llm_to_use = llm or self.llm
        if not llm_to_use:
            return query

        prompt = ChatPromptTemplate.from_template(
            """Given the conversation history and a follow-up question, rephrase the follow-up question
to be a standalone question that captures the full context needed for retrieval.

Chat History:
{chat_history}

Follow Up Question: {question}

Standalone Question:"""
        )

        try:
            chain = prompt | llm_to_use | StrOutputParser()
            rewritten = await chain.ainvoke({
                "chat_history": chat_history,
                "question": query
            })
            return rewritten.strip()
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query

    def generate_query_variations(
        self,
        query: str,
        num_variations: int = 2
    ) -> List[str]:
        """
        Generate semantically similar query variations.

        Used for multi-query retrieval to improve recall.

        Args:
            query: Original query
            num_variations: Number of variations to generate

        Returns:
            List of query variations
        """
        variations = []
        query_lower = query.lower()

        # Question reformulations
        question_words = ["what", "how", "why", "when", "where", "who", "which"]

        for qw in question_words:
            if query_lower.startswith(qw):
                # Convert question to statement-like query
                words = query.split(maxsplit=1)
                if len(words) > 1:
                    variations.append(words[1].rstrip("?"))
                break

        # Extract key terms and create focused query
        keywords = extract_keywords(query, stopwords=STOPWORDS, min_length=3)

        if keywords:
            # Create a focused keyword query
            variations.append(" ".join(keywords))

            # Create variations with different term orders
            if len(keywords) >= 2:
                variations.append(" ".join(reversed(keywords)))

        return variations[:num_variations]
