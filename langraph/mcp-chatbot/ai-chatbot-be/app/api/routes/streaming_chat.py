"""
Streaming Chat Endpoint (LangGraph Implementation)
==================================================
Production SSE endpoint with enterprise guardrails.

Now powered by LangGraph for stateful graph execution with:
- Parallel retrieval from multiple sources
- Self-correction and verification
- Human-in-the-loop review gates (optional)
- Comprehensive observability

Security Features:
- JWT authentication required
- Pydantic request validation (1-10000 chars)
- Prompt injection detection (15+ patterns)
- Input sanitization (control chars, zero-width chars)
- Token count validation (max 2000)
- User-scoped document access

Streaming Contract Guarantees:
- START event: Always sent first
- RETRIEVAL event: Document search status
- GENERATING event: LLM generation status
- TOKEN events: Text chunks as they arrive
- PROGRESS events: Every 2s heartbeat
- COMPLETE event: Always sent on success
- ERROR event: Always sent on failure
- Timeout enforcement: Configurable (default 3600s)
- Graceful termination: Clean shutdown on errors

Usage:
    POST /api/chat/stream
    {
        "message": "What is the refund policy?",
        "document_ids": ["doc-123"],  // optional
        "stream_timeout": 3600,  // optional
        "thread_id": "uuid"  // optional, for conversation continuity
    }
"""

import asyncio
import logging
import time
from typing import Set

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.models.schemas import StreamingChatRequest
from app.api.dependencies.auth import get_current_user_id
from app.core.exceptions import ValidationException
from app.services.supabase_client import supabase_client
from app.validation.input import validate_and_sanitize
from app.streaming.sse import (
    StreamingManager,
    StatusEvent,
    StreamStatus,
    CompleteEvent,
    ErrorEvent,
    create_error_stream
)
from app.rag.pipeline.chain import create_rag_chain, get_llm_provider

router = APIRouter()
logger = logging.getLogger(__name__)

# Background task tracking to prevent garbage collection of pending saves
_background_tasks: Set[asyncio.Task] = set()
_MAX_BACKGROUND_TASKS = 1000  # Prevent unbounded memory growth


def _track_background_task(task: asyncio.Task) -> None:
    """
    Track background task and clean up when done with error logging.

    Args:
        task: asyncio.Task to track
    """
    # Cleanup old completed tasks if too many
    if len(_background_tasks) > _MAX_BACKGROUND_TASKS:
        completed = {t for t in _background_tasks if t.done()}
        _background_tasks.difference_update(completed)

    _background_tasks.add(task)

    def _on_done(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        # Log any unhandled exceptions from background tasks
        if not t.cancelled():
            exc = t.exception()
            if exc:
                logger.error(f"Background task failed: {exc}", exc_info=exc)

    task.add_done_callback(_on_done)


@router.post("/stream")
async def stream_chat_response(
    message: StreamingChatRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Stream chat response using Server-Sent Events (SSE).

    Enterprise Guardrails:
    - Pydantic validation (1-10000 chars, max 50 document IDs)
    - Prompt injection detection (15+ patterns)
    - Input sanitization (control chars, zero-width chars)
    - Token count validation (max 2000 tokens)

    Streaming Contract Guarantees:
    - START event: Always sent first
    - DATA events: Token chunks as they arrive
    - PROGRESS events: Periodic updates every 2s
    - COMPLETE event: Always sent on success
    - ERROR event: Always sent on failure
    - Timeout enforcement: Configurable (default 3600s)
    - Graceful termination: Clean shutdown on errors

    Args:
        message: StreamingChatRequest with message, optional document_ids, timeout
        user_id: Authenticated user ID from JWT token

    Returns:
        StreamingResponse with text/event-stream media type
    """
    # GUARDRAIL: Validate and sanitize request before streaming
    try:
        sanitized_message, validated_doc_ids = validate_and_sanitize(
            message.message,
            message.document_ids
        )
    except ValidationException as e:
        # Return validation error as SSE stream
        logger.warning(f"Validation error for user {user_id}: {e.message}")
        return StreamingResponse(
            iter([create_error_stream(e.message, "VALIDATION_ERROR")]),
            media_type="text/event-stream"
        )

    logger.info(
        f"Streaming request - User: {user_id}, "
        f"Message length: {len(sanitized_message)}, "
        f"Documents: {len(validated_doc_ids) if validated_doc_ids else 0}, "
        f"Timeout: {message.stream_timeout}s"
    )

    # Get LLM provider info for metadata
    llm_provider = get_llm_provider()

    async def event_generator():
        """
        Generate SSE events with contract guarantees.

        Contract:
        1. START event (status: starting)
        2. RETRIEVAL event (status: retrieving)
        3. GENERATING event (status: generating)
        4. TOKEN events (data chunks)
        5. PROGRESS events (every 2s)
        6. COMPLETE or ERROR event (always sent)
        """
        start_time = time.time()
        full_response = ""
        stream_completed = False

        try:
            # CONTRACT: START event - always sent first
            yield StatusEvent(data={
                "status": StreamStatus.STARTING,
                "message": "Initializing chat..."
            }).to_sse()

            await asyncio.sleep(0.05)  # Brief pause for UX

            # CONTRACT: RETRIEVAL event
            yield StatusEvent(data={
                "status": StreamStatus.RETRIEVING,
                "message": "Searching documents..." if validated_doc_ids else "Processing query..."
            }).to_sse()

            # Create RAG chain (auto-uses Bedrock in production)
            rag_chain = create_rag_chain(
                user_id=user_id,
                document_ids=validated_doc_ids
            )

            # Create streaming manager with configured timeout
            streaming_manager = StreamingManager(
                timeout=float(message.stream_timeout),
                heartbeat_interval=2.0,  # Frequent updates for smooth UX
                buffer_size=1  # Token-by-token streaming
            )

            # CONTRACT: GENERATING event
            yield StatusEvent(data={
                "status": StreamStatus.GENERATING,
                "message": "Generating response..."
            }).to_sse()

            # Get the stream from RAG chain (using sanitized input)
            stream = rag_chain.stream(sanitized_message)

            # CONTRACT: TOKEN and PROGRESS events
            async for event in streaming_manager.stream_with_timeout(stream, send_heartbeat=True):
                yield event.to_sse()

                # Accumulate full response for saving
                if event.type == "token":
                    full_response += event.data

                # Check if error event was sent (terminates stream)
                if event.type == "error":
                    stream_completed = True
                    break

            # CONTRACT: COMPLETE event (only if no error)
            if not stream_completed:
                total_time = time.time() - start_time
                yield CompleteEvent(data={
                    "total_time": round(total_time, 3),
                    "total_tokens": streaming_manager._tokens_sent,
                    "provider": llm_provider,
                    "status": "success"
                }).to_sse()
                stream_completed = True

                # Save to history asynchronously (tracked to prevent GC)
                save_task = asyncio.create_task(_save_streaming_history(
                    user_id=user_id,
                    message=sanitized_message,
                    document_ids=validated_doc_ids,
                    response=full_response,
                    response_time=total_time,
                    provider=llm_provider
                ))
                _track_background_task(save_task)

            logger.info(f"Stream completed for user {user_id} in {time.time() - start_time:.2f}s")

        except asyncio.TimeoutError:
            # CONTRACT: ERROR event on timeout
            logger.error(f"Stream timeout for user {user_id}")
            yield ErrorEvent(data={
                "message": f"Request timed out after {message.stream_timeout} seconds",
                "code": "TIMEOUT",
                "recoverable": False
            }).to_sse()
            stream_completed = True

        except Exception as e:
            # CONTRACT: ERROR event on any exception
            logger.error(f"Streaming error for user {user_id}: {str(e)}", exc_info=True)
            if not stream_completed:
                yield ErrorEvent(data={
                    "message": "An error occurred while streaming the response",
                    "code": "STREAMING_ERROR",
                    "recoverable": False
                }).to_sse()
                stream_completed = True

        finally:
            # GUARANTEE: Stream always terminates cleanly
            if not stream_completed:
                # Fallback ERROR event if nothing was sent
                yield ErrorEvent(data={
                    "message": "Stream terminated unexpectedly",
                    "code": "UNEXPECTED_TERMINATION",
                    "recoverable": False
                }).to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",  # Configure based on CORS settings
        }
    )


async def _save_streaming_history(
    user_id: str,
    message: str,
    document_ids: list[str] | None,
    response: str,
    response_time: float,
    provider: str
) -> None:
    """
    Save streaming chat history asynchronously.

    This runs in the background and doesn't block the stream completion.
    Errors are logged but don't affect the user experience.

    Args:
        user_id: User ID
        message: User's message (sanitized)
        document_ids: Document IDs used for retrieval
        response: Full generated response
        response_time: Total response time in seconds
        provider: LLM provider info
    """
    try:
        chat_data = {
            "user_id": user_id,
            "user_message": message,
            "bot_response": response,
            "document_ids": document_ids,
            "response_time": response_time,
            "has_documents": bool(document_ids),
            "sources_used": len(document_ids) if document_ids else 0,
            "provider": provider,
            "template_used": "langchain_streaming",
            "model_config": "production"
        }

        result = supabase_client.table("chat_history").insert(chat_data).execute()

        if result.data:
            logger.debug(f"Streaming history saved for user {user_id}")
        else:
            logger.warning(f"Failed to save streaming history for user {user_id}")

    except Exception as e:
        logger.error(f"Failed to save streaming history: {str(e)}", exc_info=True)
