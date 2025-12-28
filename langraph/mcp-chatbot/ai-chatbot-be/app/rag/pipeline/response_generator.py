"""
Response Generator
==================
Document-grounded response generation for the RAG pipeline.

Features:
- Strict grounding to prevent hallucination
- Source attribution
- Confidence scoring
- Multiple response modes
"""

import logging
from typing import List, Dict, Any, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.rag.models.schemas import RAGMode
from app.rag.utils.text_utils import STOPWORDS

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates responses with strict grounding to prevent hallucination.

    Features:
    - Strict grounding to document context
    - Anti-hallucination safeguards
    - Clear source attribution
    - Honest uncertainty acknowledgment
    """

    # System prompt for strict document-grounded responses
    SYSTEM_PROMPT = """You are a document-only assistant. You can ONLY answer using the DOCUMENT CONTEXT provided below.

STRICT RULES:
1. ONLY use information that appears in the DOCUMENT CONTEXT section
2. If text appears in the context, quote it or paraphrase it directly
3. If the context is empty or doesn't contain the answer, say: "I don't have this information in the provided documents."
4. NEVER use your own knowledge - you know NOTHING except what's in the context

YOUR PROCESS:
1. Read the DOCUMENT CONTEXT carefully
2. Look for sentences containing the keywords from the question
3. If you find relevant text, use ONLY that text to answer
4. Copy or closely paraphrase the actual document text

FORBIDDEN:
- Using any knowledge not in the DOCUMENT CONTEXT
- Making assumptions or inferences beyond what's written
- Adding extra details from your training
- Answering if the context doesn't contain relevant information"""

    def __init__(self, llm, mode: RAGMode = RAGMode.CONVERSATIONAL):
        """
        Initialize response generator.

        Args:
            llm: Language model for generation
            mode: RAG operation mode
        """
        self.llm = llm
        self.mode = mode

    def build_prompt(
        self,
        context: str,
        chat_history: str = "",
        question_type: str = "general"
    ) -> ChatPromptTemplate:
        """
        Build a STRICT document-only prompt.

        Args:
            context: Retrieved document context
            chat_history: Formatted conversation history
            question_type: Type of question (for future customization)

        Returns:
            ChatPromptTemplate ready for invocation
        """
        full_prompt = f"""{self.SYSTEM_PROMPT}

==================== DOCUMENT CONTEXT START ====================
{{context}}
==================== DOCUMENT CONTEXT END ======================

REMEMBER: You can ONLY use text from the DOCUMENT CONTEXT above.
If the answer is in the context, quote or paraphrase it.
If the answer is NOT in the context, say "I don't have this information in the provided documents."

Previous conversation:
{{chat_history}}"""

        return ChatPromptTemplate.from_messages([
            ("system", full_prompt),
            ("human", "{question}")
        ])

    def check_context_relevance(
        self,
        question: str,
        context: str
    ) -> Tuple[bool, float]:
        """
        Check if the context is relevant to the question.

        Uses keyword overlap to estimate relevance.

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Tuple of (is_relevant, relevance_score)
        """
        if not context or context.strip() == "No relevant documents found.":
            return False, 0.0

        # Simple keyword overlap check
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())

        # Remove stopwords
        question_keywords = question_words - STOPWORDS
        context_keywords = context_words - STOPWORDS

        if not question_keywords:
            return True, 0.5  # Default to relevant if no keywords

        overlap = len(question_keywords & context_keywords)
        relevance_score = overlap / len(question_keywords)

        return relevance_score > 0.1, relevance_score

    def calculate_confidence(
        self,
        answer: str,
        context: str,
        retrieval_scores: List[float]
    ) -> float:
        """
        Calculate confidence score for the response.

        Based on:
        - Retrieval scores
        - Answer-context overlap
        - Response coherence signals

        Args:
            answer: Generated answer
            context: Retrieved context
            retrieval_scores: Similarity scores from retrieval

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not retrieval_scores:
            return 0.0

        # Average retrieval score (normalized)
        avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)

        # Check for uncertainty signals in answer
        uncertainty_phrases = [
            "i don't know",
            "i'm not sure",
            "unclear",
            "uncertain",
            "no information",
            "cannot find",
            "not mentioned",
            "don't have information",
        ]
        has_uncertainty = any(
            phrase in answer.lower() for phrase in uncertainty_phrases
        )
        uncertainty_penalty = 0.3 if has_uncertainty else 0.0

        # Check for context-answer overlap (simple word overlap)
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        overlap_score = min(overlap * 2, 1.0)  # Cap at 1.0

        # Combine scores
        confidence = (
            avg_retrieval * 0.4 +
            overlap_score * 0.4 +
            (1 - uncertainty_penalty) * 0.2
        )

        return round(min(max(confidence, 0.0), 1.0), 2)

    def format_sources(
        self,
        documents: List[Document],
        max_preview: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Format source documents for response.

        Args:
            documents: List of retrieved documents
            max_preview: Maximum preview length

        Returns:
            List of formatted source dicts
        """
        sources = []
        for doc in documents:
            content = doc.page_content
            sources.append({
                "content": (
                    content[:max_preview] + "..."
                    if len(content) > max_preview
                    else content
                ),
                "source": doc.metadata.get("source", "Unknown"),
                "document_id": doc.metadata.get("document_id"),
                "score": (
                    doc.metadata.get("rerank_score") or
                    doc.metadata.get("score", 0)
                ),
                "chunk_index": doc.metadata.get("chunk_index"),
            })
        return sources


def format_context(
    docs: List[Document],
    max_context_length: int = 4000
) -> str:
    """
    Format retrieved documents into context string with deduplication.

    Args:
        docs: List of retrieved documents
        max_context_length: Maximum context length

    Returns:
        Formatted context string
    """
    if not docs:
        return "No relevant documents found."

    context_parts = []
    total_length = 0
    seen_content_hashes = set()

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", f"Document {i+1}")
        score = doc.metadata.get("rerank_score") or doc.metadata.get("score", 0)
        content = doc.page_content

        # Deduplicate by content hash (first 200 chars)
        content_hash = hash(content[:200])
        if content_hash in seen_content_hashes:
            continue
        seen_content_hashes.add(content_hash)

        # Truncate if exceeding max context length
        if total_length + len(content) > max_context_length:
            remaining = max_context_length - total_length
            if remaining > 100:
                content = content[:remaining] + "..."
            else:
                break

        context_parts.append(
            f"[Source: {source}] (Relevance: {score:.2f})\n{content}"
        )
        total_length += len(content)

    return "\n\n---\n\n".join(context_parts)
