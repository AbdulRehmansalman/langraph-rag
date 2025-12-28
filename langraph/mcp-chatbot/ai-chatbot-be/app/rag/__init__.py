"""
RAG Services
============
Complete RAG pipeline for document Q&A.

Usage:
    from app.rag import create_rag_chain, RAGChain

    # Create with default settings (auto-detects environment for LLM)
    chain = create_rag_chain(user_id="user123")
    response = await chain.invoke("What is the document about?")

    # Or create with custom LLM
    from app.services.llm_factory import llm_factory
    llm = llm_factory.create_llm()
    chain = create_rag_chain(llm=llm, user_id="user123")
"""

# Document processing
from app.rag.documents.loader import DocumentLoaderService
from app.rag.documents.splitter import TextSplitterService

# Embeddings
from app.rag.embeddings.service import EmbeddingsService

# Retrieval
from app.rag.retrieval import (
    VectorStoreService,
    RetrieverService,
    RetrievalStrategy,
    RetrievalConfig,
    RerankingMethod,
    create_retriever,
)

# Pipeline
from app.rag.pipeline.chain import RAGChain, create_rag_chain, get_llm_provider
from app.rag.pipeline.memory import ConversationMemoryService

# Models
from app.rag.models.schemas import RAGMode, RAGConfig, RAGResponse

__all__ = [
    # Documents
    "DocumentLoaderService",
    "TextSplitterService",
    # Embeddings
    "EmbeddingsService",
    # Retrieval
    "VectorStoreService",
    "RetrieverService",
    "RetrievalStrategy",
    "RetrievalConfig",
    "RerankingMethod",
    "create_retriever",
    # Pipeline
    "RAGChain",
    "create_rag_chain",
    "get_llm_provider",
    "ConversationMemoryService",
    # Models
    "RAGMode",
    "RAGConfig",
    "RAGResponse",
]
