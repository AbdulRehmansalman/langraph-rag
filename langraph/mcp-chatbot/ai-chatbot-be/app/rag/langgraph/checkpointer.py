"""
PostgreSQL Checkpointer Configuration
======================================

Provides checkpointing setup for LangGraph state persistence.
Enables:
- Cross-session memory retrieval
- Time-travel debugging
- State recovery after failures
"""

import logging
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# Global checkpointer instance
_checkpointer = None


def get_postgres_checkpointer(connection_string: Optional[str] = None):
    """
    Get or create PostgreSQL checkpointer for state persistence.

    Args:
        connection_string: PostgreSQL connection URL

    Returns:
        PostgresSaver or MemorySaver as fallback
    """
    global _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    if connection_string:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            _checkpointer = PostgresSaver.from_conn_string(connection_string)
            logger.info("PostgreSQL checkpointer initialized")
            return _checkpointer

        except ImportError:
            logger.warning(
                "langgraph-checkpoint-postgres not installed. "
                "Install with: pip install langgraph-checkpoint-postgres"
            )
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL checkpointer: {e}")

    # Fallback to memory checkpointer
    logger.info("Using in-memory checkpointer (no persistence)")
    _checkpointer = MemorySaver()
    return _checkpointer


def get_memory_checkpointer():
    """Get in-memory checkpointer for development/testing."""
    return MemorySaver()


async def setup_postgres_schema(connection_string: str) -> bool:
    """
    Set up the required PostgreSQL schema for checkpointing.

    Args:
        connection_string: PostgreSQL connection URL

    Returns:
        True if successful, False otherwise
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        async with PostgresSaver.from_conn_string(connection_string) as saver:
            await saver.setup()
            logger.info("PostgreSQL checkpoint schema created successfully")
            return True

    except ImportError:
        logger.error("langgraph-checkpoint-postgres not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to set up PostgreSQL schema: {e}")
        return False


class CheckpointerManager:
    """
    Manager for handling checkpointer lifecycle and operations.

    Provides utilities for:
    - State snapshot management
    - Time-travel queries
    - State cleanup
    """

    def __init__(self, checkpointer):
        """
        Initialize checkpointer manager.

        Args:
            checkpointer: LangGraph checkpointer instance
        """
        self.checkpointer = checkpointer

    async def get_state_history(
        self,
        thread_id: str,
        limit: int = 10
    ) -> list[dict]:
        """
        Get state history for a thread (time-travel).

        Args:
            thread_id: Thread identifier
            limit: Maximum number of states to return

        Returns:
            List of historical states
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            states = []

            async for state in self.checkpointer.alist(config, limit=limit):
                states.append({
                    "checkpoint_id": state.config.get("checkpoint_id"),
                    "thread_id": thread_id,
                    "timestamp": state.metadata.get("created_at"),
                    "node": state.metadata.get("source"),
                })

            return states

        except Exception as e:
            logger.error(f"Failed to get state history: {e}")
            return []

    async def get_state_at_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str
    ) -> Optional[dict]:
        """
        Get state at a specific checkpoint (time-travel debugging).

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID

        Returns:
            State at that checkpoint or None
        """
        try:
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }

            state = await self.checkpointer.aget(config)
            if state:
                return state.values

            return None

        except Exception as e:
            logger.error(f"Failed to get state at checkpoint: {e}")
            return None

    async def delete_thread_history(self, thread_id: str) -> bool:
        """
        Delete all checkpoints for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            True if successful
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}

            # Some checkpointers support deletion
            if hasattr(self.checkpointer, "adelete"):
                await self.checkpointer.adelete(config)
                logger.info(f"Deleted history for thread {thread_id}")
                return True

            logger.warning("Checkpointer does not support deletion")
            return False

        except Exception as e:
            logger.error(f"Failed to delete thread history: {e}")
            return False
