"""
Document Processor Service
==========================
Production document processing pipeline using RAG services.
"""

import logging
from typing import List, Optional, Dict, Any
import io

from langchain_core.documents import Document

from app.services.supabase_client import supabase_client
from app.rag.documents.loader import DocumentLoaderService, document_loader
from app.rag.documents.splitter import TextSplitterService, text_splitter, ChunkingStrategy
from app.rag.retrieval.vector_store import VectorStoreService, get_vector_store

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Production document processor.

    Features:
    - Multi-format document loading
    - Intelligent text chunking
    - Embedding generation and storage
    - Progress tracking
    """

    def __init__(self):
        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self.vector_store = get_vector_store()

        logger.info("DocumentProcessor initialized with RAG services")

    async def process_document(
        self,
        document_id: str,
        file_content: bytes,
        content_type: str,
        filename: str = "document",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document through the full RAG pipeline.

        Args:
            document_id: Unique document ID
            file_content: Raw file bytes
            content_type: MIME type
            filename: Original filename
            metadata: Additional metadata

        Returns:
            Processing result with statistics
        """
        result = {
            "document_id": document_id,
            "success": False,
            "chunks_created": 0,
            "error": None,
        }

        try:
            logger.info(f"Processing document {document_id}: {filename} ({content_type})")

            # Step 1: Load document
            documents = self.document_loader.load_from_bytes(
                content=file_content,
                content_type=content_type,
                filename=filename,
                metadata={
                    "document_id": document_id,
                    **(metadata or {})
                }
            )

            if not documents:
                raise ValueError("No content extracted from document")

            logger.info(f"Loaded {len(documents)} document section(s)")

            # Step 2: Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            if not chunks:
                raise ValueError("No chunks created from document")

            logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Add to vector store (embeddings generated automatically)
            added_count = self.vector_store.add_documents(
                documents=chunks,
                document_id=document_id,
            )

            # Step 4: Update document status
            supabase_client.table("documents").update({
                "processed": True,
                "error": None
            }).eq("id", document_id).execute()

            result["success"] = True
            result["chunks_created"] = added_count
            result["sections_loaded"] = len(documents)

            logger.info(f"Successfully processed document {document_id}: {added_count} chunks")

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing document {document_id}: {error_msg}")

            # Update document with error
            supabase_client.table("documents").update({
                "processed": False,
                "error": error_msg
            }).eq("id", document_id).execute()

            result["error"] = error_msg
            return result

    async def reprocess_document(
        self,
        document_id: str,
        file_content: bytes,
        content_type: str,
        filename: str = "document"
    ) -> Dict[str, Any]:
        """
        Reprocess an existing document (deletes old embeddings first).

        Args:
            document_id: Document ID
            file_content: Raw file bytes
            content_type: MIME type
            filename: Original filename

        Returns:
            Processing result
        """
        # Delete existing embeddings
        deleted = self.vector_store.delete_document(document_id)
        logger.info(f"Deleted {deleted} existing embeddings for document {document_id}")

        # Process again
        return await self.process_document(
            document_id=document_id,
            file_content=file_content,
            content_type=content_type,
            filename=filename,
        )

    def delete_document_embeddings(self, document_id: str) -> int:
        """
        Delete all embeddings for a document.

        Args:
            document_id: Document ID

        Returns:
            Number of embeddings deleted
        """
        return self.vector_store.delete_document(document_id)

    def search_documents(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            user_id: User ID
            document_ids: Optional document IDs to search
            top_k: Number of results

        Returns:
            List of relevant chunks with metadata
        """
        results = self.vector_store.similarity_search(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            top_k=top_k,
        )

        return [
            {
                "content": r.content,
                "score": r.score,
                "document_id": r.document_id,
                "chunk_index": r.chunk_index,
                "metadata": r.metadata,
            }
            for r in results
        ]

    def get_document_chunks(
        self,
        document_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document ID
            limit: Maximum chunks to return

        Returns:
            List of chunks
        """
        results = self.vector_store.get_document_chunks(document_id, limit)

        return [
            {
                "content": r.content,
                "chunk_index": r.chunk_index,
                "metadata": r.metadata,
            }
            for r in results
        ]

    @staticmethod
    def is_supported_type(content_type: str) -> bool:
        """Check if content type is supported."""
        return DocumentLoaderService.is_supported(content_type)


# Global instance
doc_processor = DocumentProcessor()
