"""
Conversation Memory Service
===========================
Production memory management for conversations.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.chat_history import BaseChatMessageHistory

from app.database.connection import SessionLocal
from app.database.models import ChatHistory

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """A single conversation message."""

    role: str  # "human" or "ai"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatabaseChatHistory(BaseChatMessageHistory):
    """
    Chat history backed by database.
    Implements LangChain's BaseChatMessageHistory for compatibility.
    """

    def __init__(self, user_id: str, session_id: Optional[str] = None):
        self.user_id = user_id
        self.session_id = session_id
        self._messages: List[BaseMessage] = []
        self._load_history()

    def _load_history(self, limit: int = 20):
        """Load recent history from database."""
        session = SessionLocal()

        try:
            query = (
                session.query(ChatHistory)
                .filter(ChatHistory.user_id == self.user_id)
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
            )

            results = query.all()

            # Convert to messages (reverse to get chronological order)
            self._messages = []
            for chat in reversed(results):
                self._messages.append(HumanMessage(content=chat.user_message))
                self._messages.append(AIMessage(content=chat.bot_response))

            logger.debug(f"Loaded {len(self._messages)} messages for user {self.user_id}")

        except Exception as e:
            logger.error(f"Error loading chat history: {str(e)}")
            self._messages = []

        finally:
            session.close()

    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages."""
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to history."""
        self._messages.append(message)

    def add_user_message(self, message: str) -> None:
        """Add a user message."""
        self._messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Add an AI message."""
        self._messages.append(AIMessage(content=message))

    def clear(self) -> None:
        """Clear in-memory messages (doesn't delete from DB)."""
        self._messages = []


class ConversationMemoryService:
    """
    Production conversation memory service.

    Features:
    - Buffer memory for recent context
    - Summary memory for long conversations
    - Database-backed persistence
    - Token-aware context management
    - Multi-session support
    """

    def __init__(
        self, max_messages: int = 10, max_tokens: int = 4000, summarize_threshold: int = 20
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold
        self._histories: Dict[str, DatabaseChatHistory] = {}

        logger.info(f"ConversationMemoryService initialized (max_messages={max_messages})")

    def get_history(self, user_id: str, session_id: Optional[str] = None) -> DatabaseChatHistory:
        """
        Get or create chat history for a user.

        Args:
            user_id: User ID
            session_id: Optional session ID for multi-session

        Returns:
            DatabaseChatHistory instance
        """
        key = f"{user_id}:{session_id or 'default'}"

        if key not in self._histories:
            self._histories[key] = DatabaseChatHistory(user_id, session_id)

        return self._histories[key]

    def get_context_messages(
        self, user_id: str, session_id: Optional[str] = None, include_summary: bool = True
    ) -> List[BaseMessage]:
        """
        Get context messages for the conversation.

        Args:
            user_id: User ID
            session_id: Session ID
            include_summary: Include conversation summary if available

        Returns:
            List of messages for context
        """
        history = self.get_history(user_id, session_id)
        messages = history.messages

        # Limit to recent messages
        if len(messages) > self.max_messages * 2:  # *2 because human+ai pairs
            # Get last N messages
            recent_messages = messages[-(self.max_messages * 2) :]

            # If we have a summary, prepend it
            if include_summary and len(messages) > self.summarize_threshold * 2:
                summary = self._get_or_create_summary(user_id, messages[: -len(recent_messages)])
                if summary:
                    return [
                        SystemMessage(content=f"Previous conversation summary: {summary}")
                    ] + recent_messages

            return recent_messages

        return messages

    def _get_or_create_summary(self, user_id: str, messages: List[BaseMessage]) -> Optional[str]:
        """Create a summary of older messages."""
        # For now, return a simple concatenation
        # In production, you'd use an LLM to summarize
        if not messages:
            return None

        summary_parts = []
        for msg in messages[-10:]:  # Summarize last 10 messages
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"{role}: {content}")

        return "\n".join(summary_parts)

    def add_exchange(
        self,
        user_id: str,
        user_message: str,
        ai_message: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a user-AI exchange to memory.

        Args:
            user_id: User ID
            user_message: User's message
            ai_message: AI's response
            session_id: Session ID
            metadata: Additional metadata
        """
        history = self.get_history(user_id, session_id)
        history.add_user_message(user_message)
        history.add_ai_message(ai_message)

        logger.debug(f"Added exchange to memory for user {user_id}")

    def clear_history(self, user_id: str, session_id: Optional[str] = None):
        """Clear conversation history from memory."""
        key = f"{user_id}:{session_id or 'default'}"

        if key in self._histories:
            self._histories[key].clear()
            del self._histories[key]

        logger.info(f"Cleared memory for user {user_id}")

    def format_for_prompt(self, user_id: str, session_id: Optional[str] = None) -> str:
        """
        Format conversation history for inclusion in a prompt.

        Args:
            user_id: User ID
            session_id: Session ID

        Returns:
            Formatted string of conversation history
        """
        messages = self.get_context_messages(user_id, session_id)

        if not messages:
            return ""

        formatted_parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_parts.append(f"Assistant: {msg.content}")
            elif isinstance(msg, SystemMessage):
                formatted_parts.append(f"[{msg.content}]")

        return "\n".join(formatted_parts)

    def get_stats(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics for a user."""
        history = self.get_history(user_id, session_id)
        messages = history.messages

        return {
            "total_messages": len(messages),
            "human_messages": sum(1 for m in messages if isinstance(m, HumanMessage)),
            "ai_messages": sum(1 for m in messages if isinstance(m, AIMessage)),
            "in_context": min(len(messages), self.max_messages * 2),
        }


# Global instance - lazy initialization
_memory_service: Optional[ConversationMemoryService] = None


def get_memory_service() -> ConversationMemoryService:
    """Get or create the global memory service."""
    global _memory_service
    if _memory_service is None:
        from app.core.config import settings

        _memory_service = ConversationMemoryService(
            max_messages=settings.max_instant_messages,
            max_tokens=settings.max_context_tokens,
            summarize_threshold=settings.summarization_threshold,
        )
    return _memory_service
