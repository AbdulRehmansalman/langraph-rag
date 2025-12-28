"""
Production Retriever Service
============================
Multi-strategy retrieval with RRF fusion and cross-encoder reranking.

Retrieval Strategies:
- SIMILARITY: Pure vector cosine similarity
- MMR: Maximal Marginal Relevance (relevance + diversity)
- HYBRID: Vector + keyword weighted combination
- ENSEMBLE: Multi-strategy with Reciprocal Rank Fusion

Reranking:
- Cross-encoder for high-precision relevance scoring
- RRF for combining multiple retrieval strategies

Features:
- User-scoped document filtering (multi-tenant)
- Result deduplication
- Configurable search weights
- LangChain BaseRetriever compatibility
"""

import logging
from typing import List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.messages import HumanMessage, SystemMessage

from app.rag.retrieval.vector_store import get_vector_store, SearchResult
from app.rag.ranking import BaseReranker, RerankingMethod, CrossEncoderReranker, RRFReranker

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """
    Available retrieval strategies.

    - SIMILARITY: Pure vector similarity search
    - MMR: Maximal Marginal Relevance (balances relevance and diversity)
    - HYBRID: Weighted combination of vector and keyword search
    - ENSEMBLE: Multiple strategies combined with RRF fusion
    """
    SIMILARITY = "similarity"
    MMR = "mmr"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


@dataclass
class RetrievalConfig:
    """
    Configuration for document retrieval.

    Attributes:
        strategy: Retrieval strategy to use
        top_k: Number of results to return
        score_threshold: Minimum similarity score (0-1)
        vector_weight: Weight for vector search in hybrid mode
        keyword_weight: Weight for keyword search in hybrid mode
        reranking_method: Method for reranking results
        rerank_top_k: Number of candidates to fetch before reranking
        fetch_k: Number of candidates for MMR diversity selection
        lambda_mult: MMR diversity parameter (0=max diversity, 1=max relevance)
        ensemble_weights: Weights for ensemble strategy sources
        deduplicate: Whether to remove duplicate documents
        include_metadata: Whether to include full metadata in results
    """
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 10
    score_threshold: float = 0.1

    # Search weights (for HYBRID strategy)
    vector_weight: float = 0.4
    keyword_weight: float = 0.6

    # Reranking configuration
    reranking_method: RerankingMethod = RerankingMethod.CROSS_ENCODER
    rerank_top_k: int = 20  # Fetch more candidates for reranking

    # MMR parameters
    fetch_k: int = 30
    lambda_mult: float = 0.5  # 0 = max diversity, 1 = max relevance

    # Ensemble weights (for ENSEMBLE strategy)
    ensemble_weights: dict = field(
        default_factory=lambda: {
            "similarity": 0.4,
            "mmr": 0.1,
            "keyword": 0.5
        }
    )

    # Options
    deduplicate: bool = True
    include_metadata: bool = True


class RetrieverService(BaseRetriever):
    """
    Production retriever with multi-strategy search and reranking.

    Implements LangChain BaseRetriever for compatibility with RAG chains.
    Supports user-scoped document filtering for multi-tenant applications.

    Pipeline:
    1. Execute retrieval strategy (similarity, MMR, hybrid, or ensemble)
    2. Deduplicate results (if enabled)
    3. Apply reranking (cross-encoder for precision)
    4. Return top_k results with metadata

    Example:
        ```python
        retriever = create_retriever(
            user_id="user-123",
            strategy=RetrievalStrategy.ENSEMBLE,
            reranking=RerankingMethod.CROSS_ENCODER,
            top_k=5
        )
        docs = retriever.invoke("What is the refund policy?")
        ```
    """

    # Pydantic model configuration
    user_id: str
    document_ids: Optional[List[str]] = None
    config: Optional[RetrievalConfig] = None
    llm: Any = None
    _vector_store: Any = None
    _reranker: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        config: Optional[RetrievalConfig] = None,
        llm=None,
        **kwargs
    ):
        """
        Initialize retriever service.

        Args:
            user_id: User ID for document filtering (multi-tenant support)
            document_ids: Optional list of specific document IDs to search
            config: Retrieval configuration
            llm: LLM instance (for advanced features like HyDE)
        """
        super().__init__(
            user_id=user_id,
            document_ids=document_ids,
            config=config or RetrievalConfig(),
            llm=llm,
            **kwargs
        )
        self._vector_store = get_vector_store()
        self._reranker = self._init_reranker()

        logger.info(
            f"RetrieverService initialized: user={user_id}, "
            f"strategy={self.config.strategy.value}, "
            f"reranking={self.config.reranking_method.value}"
        )

    def _init_reranker(self) -> Optional[BaseReranker]:
        """Initialize reranker based on configuration."""
        method = self.config.reranking_method

        if method == RerankingMethod.CROSS_ENCODER:
            return CrossEncoderReranker()
        elif method == RerankingMethod.RECIPROCAL_RANK_FUSION:
            return RRFReranker()
        return None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for query.

        This is the main entry point called by LangChain.

        Args:
            query: Search query
            run_manager: LangChain callback manager (optional)

        Returns:
            List of relevant documents with metadata
        """
        try:
            # Fetch more candidates if reranking is enabled
            fetch_k = self.config.rerank_top_k if self._reranker else self.config.top_k

            # Execute retrieval strategy
            if self.config.strategy == RetrievalStrategy.ENSEMBLE:
                documents = self._ensemble_retrieve(query, fetch_k)
            elif self.config.strategy == RetrievalStrategy.MMR:
                documents = self._mmr_retrieve(query, fetch_k)
            elif self.config.strategy == RetrievalStrategy.HYBRID:
                documents = self._hybrid_retrieve(query, fetch_k)
            else:
                documents = self._similarity_retrieve(query, fetch_k)

            # Deduplicate results
            if self.config.deduplicate:
                documents = self._deduplicate(documents)

            # Apply reranking (cross-encoder for precision)
            if self._reranker and documents:
                documents = self._reranker.rerank(
                    query=query,
                    documents=documents,
                    top_k=self.config.top_k
                )
            else:
                # Sort by score and truncate without reranking
                documents.sort(
                    key=lambda x: x.metadata.get("score", 0),
                    reverse=True
                )
                documents = documents[:self.config.top_k]

            logger.info(
                f"Retrieved {len(documents)} docs: "
                f"strategy={self.config.strategy.value}, "
                f"reranking={self.config.reranking_method.value}"
            )
            return documents

        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return []

    def _similarity_retrieve(self, query: str, top_k: int) -> List[Document]:
        """Pure vector similarity search."""
        results = self._vector_store.similarity_search(
            query=query,
            user_id=self.user_id,
            document_ids=self.document_ids,
            top_k=top_k,
            score_threshold=self.config.score_threshold
        )
        return self._results_to_documents(results)

    def _mmr_retrieve(self, query: str, top_k: int) -> List[Document]:
        """Maximal Marginal Relevance search (relevance + diversity)."""
        results = self._vector_store.mmr_search(
            query=query,
            user_id=self.user_id,
            document_ids=self.document_ids,
            top_k=top_k,
            fetch_k=self.config.fetch_k,
            lambda_mult=self.config.lambda_mult
        )
        return self._results_to_documents(results)

    def _hybrid_retrieve(self, query: str, top_k: int) -> List[Document]:
        """Hybrid vector + keyword search with weighted combination."""
        results = self._vector_store.hybrid_search(
            query=query,
            user_id=self.user_id,
            document_ids=self.document_ids,
            top_k=max(top_k, 20),
            vector_weight=self.config.vector_weight,
            keyword_weight=self.config.keyword_weight
        )
        return self._results_to_documents(results)

    def _ensemble_retrieve(self, query: str, top_k: int) -> List[Document]:
        """
        Ensemble retrieval: Multiple strategies combined with RRF.

        Runs similarity, MMR, and keyword searches, then fuses results
        using Reciprocal Rank Fusion for a unified ranking.
        """
        ranked_lists = []
        weights = self.config.ensemble_weights

        # Vector similarity search
        if weights.get("similarity", 0) > 0:
            sim_results = self._vector_store.similarity_search(
                query=query,
                user_id=self.user_id,
                document_ids=self.document_ids,
                top_k=top_k
            )
            sim_docs = self._results_to_documents(sim_results)
            for i, doc in enumerate(sim_docs):
                doc.metadata["retrieval_source"] = "similarity"
                doc.metadata["rank"] = i + 1
            ranked_lists.append(sim_docs)

        # MMR search (diversity)
        if weights.get("mmr", 0) > 0:
            mmr_results = self._vector_store.mmr_search(
                query=query,
                user_id=self.user_id,
                document_ids=self.document_ids,
                top_k=top_k,
                fetch_k=self.config.fetch_k
            )
            mmr_docs = self._results_to_documents(mmr_results)
            for i, doc in enumerate(mmr_docs):
                doc.metadata["retrieval_source"] = "mmr"
                doc.metadata["rank"] = i + 1
            ranked_lists.append(mmr_docs)

        # Keyword search
        if weights.get("keyword", 0) > 0:
            kw_results = self._vector_store._keyword_search(
                query=query,
                user_id=self.user_id,
                document_ids=self.document_ids,
                top_k=top_k
            )
            kw_docs = self._results_to_documents(kw_results)
            for i, doc in enumerate(kw_docs):
                doc.metadata["retrieval_source"] = "keyword"
                doc.metadata["rank"] = i + 1
            ranked_lists.append(kw_docs)

        # Fuse using RRF
        rrf = RRFReranker()
        return rrf.fuse_rankings(ranked_lists, top_k=top_k)

    def _results_to_documents(self, results: List[SearchResult]) -> List[Document]:
        """Convert SearchResult objects to LangChain Documents."""
        documents = []
        for result in results:
            metadata = {
                "document_id": result.document_id,
                "chunk_index": result.chunk_index,
                "score": result.score,
            }
            if self.config.include_metadata and result.metadata:
                metadata.update(result.metadata)

            documents.append(Document(
                page_content=result.content,
                metadata=metadata
            ))
        return documents

    def _deduplicate(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content hash."""
        if not documents:
            return documents

        seen = set()
        unique = []

        for doc in documents:
            # Use chunk_hash if available, otherwise hash content prefix
            content_hash = doc.metadata.get("chunk_hash") or hash(doc.page_content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(doc)

        return unique


class HyDERetriever:
    """
    Hypothetical Document Embeddings (HyDE) retriever.

    Generates a hypothetical answer to the query and uses it
    for retrieval, improving results for questions where the
    query terms don't directly match document vocabulary.

    Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    """

    PROMPT = """Write a detailed paragraph that answers the following question.
Write it as if you're quoting from a document that would contain this information.

Question: {query}

Hypothetical answer:"""

    def __init__(self, retriever: RetrieverService, llm):
        """
        Initialize HyDE retriever.

        Args:
            retriever: Base retriever service
            llm: LLM for generating hypothetical documents
        """
        self.retriever = retriever
        self.llm = llm

    async def retrieve(self, query: str) -> List[Document]:
        """
        Generate hypothetical document and retrieve similar real documents.

        Args:
            query: User's question

        Returns:
            List of relevant documents
        """
        hypothetical = await self._generate_hypothetical(query)
        if hypothetical:
            return self.retriever.invoke(hypothetical)
        return self.retriever.invoke(query)

    async def _generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical answer to use for retrieval."""
        prompt = self.PROMPT.format(query=query)
        try:
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=prompt),
            ]
            response = await self.llm.ainvoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"HyDE generation error: {e}")
            return ""


class ContextualCompressionRetriever:
    """
    Retriever that compresses retrieved documents to relevant parts.

    Uses an LLM to extract only the relevant portions of each document,
    reducing noise in the context provided to the final generation.
    """

    PROMPT = """Extract only the parts of the following text that are relevant to answering this question.

Question: {query}

Text:
{content}

Return only the relevant excerpts. If nothing is relevant, return "NOT_RELEVANT"."""

    def __init__(self, retriever: RetrieverService, llm):
        """
        Initialize contextual compression retriever.

        Args:
            retriever: Base retriever service
            llm: LLM for compression
        """
        self.retriever = retriever
        self.llm = llm

    async def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve and compress documents to relevant parts.

        Args:
            query: User's question

        Returns:
            List of compressed documents
        """
        documents = self.retriever.invoke(query)
        if not documents:
            return []

        compressed = []
        for doc in documents:
            content = await self._compress(query, doc.page_content)
            if content:
                compressed.append(Document(
                    page_content=content,
                    metadata={**doc.metadata, "compressed": True}
                ))
        return compressed

    async def _compress(self, query: str, content: str) -> str:
        """Compress content to relevant parts."""
        prompt = self.PROMPT.format(query=query, content=content[:2000])
        try:
            messages = [
                SystemMessage(content="You extract relevant information."),
                HumanMessage(content=prompt),
            ]
            response = await self.llm.ainvoke(messages)
            result = response.content if hasattr(response, "content") else str(response)
            return "" if "NOT_RELEVANT" in result else result.strip()
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return content


def create_retriever(
    user_id: str,
    document_ids: Optional[List[str]] = None,
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    top_k: int = 10,
    score_threshold: float = 0.1,
    reranking: RerankingMethod = RerankingMethod.CROSS_ENCODER,
    llm=None,
) -> RetrieverService:
    """
    Factory function to create a configured retriever.

    Args:
        user_id: User ID for document filtering
        document_ids: Optional specific document IDs to search
        strategy: Retrieval strategy (similarity, mmr, hybrid, ensemble)
        top_k: Number of results to return
        score_threshold: Minimum similarity score
        reranking: Reranking method (cross_encoder, rrf, none)
        llm: LLM instance for advanced features

    Returns:
        Configured RetrieverService

    Example:
        ```python
        retriever = create_retriever(
            user_id="user-123",
            strategy=RetrievalStrategy.ENSEMBLE,
            reranking=RerankingMethod.CROSS_ENCODER,
            top_k=5
        )
        docs = retriever.invoke("What is the return policy?")
        ```
    """
    config = RetrievalConfig(
        strategy=strategy,
        top_k=top_k,
        score_threshold=score_threshold,
        reranking_method=reranking,
        # Fetch 2x candidates for reranking, unless no reranking
        rerank_top_k=top_k * 2 if reranking != RerankingMethod.NONE else top_k,
    )

    return RetrieverService(
        user_id=user_id,
        document_ids=document_ids,
        config=config,
        llm=llm,
    )
