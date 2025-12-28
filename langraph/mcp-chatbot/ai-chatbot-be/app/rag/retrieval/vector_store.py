"""
Vector Store Service
====================
Production vector store with pgvector for PostgreSQL.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal, engine
from app.database.models import DocumentEmbedding, Document as DocumentModel
from app.rag.embeddings.service import get_embeddings_service

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with document content and metadata."""
    content: str
    metadata: Dict[str, Any]
    score: float
    document_id: str
    chunk_index: int


class VectorStoreService:
    """
    Production vector store using PostgreSQL with pgvector.

    Features:
    - Efficient vector similarity search
    - Metadata filtering
    - Hybrid search (vector + keyword)
    - Maximal Marginal Relevance (MMR)
    - Batch operations
    """

    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.embeddings_service = get_embeddings_service()
        logger.info(f"VectorStoreService initialized for collection: {collection_name}")

    def add_documents(
        self,
        documents: List[Document],
        document_id: str,
        batch_size: int = 100
    ) -> int:
        """
        Add documents to vector store.

        Args:
            documents: List of Document objects to add
            document_id: Parent document ID
            batch_size: Batch size for insertion

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings_service.embed_documents(texts)

        session = SessionLocal()
        added_count = 0

        try:
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]

                for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
                    chunk_index = i + j

                    embedding_record = DocumentEmbedding(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=doc.page_content,
                        embedding=embedding,
                        chunk_metadata=doc.metadata,
                    )
                    session.add(embedding_record)
                    added_count += 1

                session.commit()
                logger.debug(f"Added batch of {len(batch_docs)} embeddings")

            logger.info(f"Added {added_count} embeddings for document {document_id}")
            return added_count

        except Exception as e:
            session.rollback()
            logger.error(f"Error adding documents: {str(e)}")
            raise

        finally:
            session.close()

    def similarity_search(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar documents using vector similarity.

        Args:
            query: Search query
            user_id: User ID for filtering
            document_ids: Specific document IDs to search (optional)
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = self.embeddings_service.embed_query(query)

        session = SessionLocal()

        try:
            # Build query using pgvector's cosine distance
            # Note: pgvector uses distance (smaller = more similar), so we convert to similarity
            # Convert embedding to PostgreSQL vector string format
            embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

            if document_ids:
                # Search specific documents - cast document_ids to UUID array
                sql = text("""
                    SELECT
                        de.id,
                        de.document_id,
                        de.chunk_index,
                        de.content,
                        de.metadata,
                        1 - (de.embedding <=> CAST(:query_embedding AS vector)) as similarity
                    FROM document_embeddings de
                    JOIN documents d ON de.document_id = d.id
                    WHERE d.user_id = CAST(:user_id AS UUID)
                    AND de.document_id = ANY(CAST(:document_ids AS UUID[]))
                    ORDER BY de.embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :top_k
                """)

                results = session.execute(sql, {
                    "query_embedding": embedding_str,
                    "user_id": user_id,
                    "document_ids": document_ids,
                    "top_k": top_k,
                }).fetchall()

            else:
                # Search all user documents
                sql = text("""
                    SELECT
                        de.id,
                        de.document_id,
                        de.chunk_index,
                        de.content,
                        de.metadata,
                        1 - (de.embedding <=> CAST(:query_embedding AS vector)) as similarity
                    FROM document_embeddings de
                    JOIN documents d ON de.document_id = d.id
                    WHERE d.user_id = CAST(:user_id AS UUID)
                    ORDER BY de.embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :top_k
                """)

                results = session.execute(sql, {
                    "query_embedding": embedding_str,
                    "user_id": user_id,
                    "top_k": top_k,
                }).fetchall()

            # Convert to SearchResult objects
            search_results = []
            for row in results:
                similarity = float(row.similarity) if row.similarity else 0.0

                if similarity >= score_threshold:
                    search_results.append(SearchResult(
                        content=row.content,
                        metadata=row.metadata or {},
                        score=similarity,
                        document_id=str(row.document_id),
                        chunk_index=row.chunk_index,
                    ))

            logger.info(f"Found {len(search_results)} results for query (threshold: {score_threshold})")
            return search_results

        except Exception as e:
            logger.error(f"Similarity search error: {str(e)}")
            # Rollback the failed transaction before fallback
            try:
                session.rollback()
            except Exception:
                pass  # Session may already be invalidated
            # Fallback to Python-based similarity if pgvector query fails
            # Create fresh session for fallback to avoid state issues
            fallback_session = SessionLocal()
            try:
                return self._fallback_similarity_search(
                    query_embedding, user_id, document_ids, top_k, score_threshold, fallback_session
                )
            finally:
                fallback_session.close()

        finally:
            session.close()

    def _fallback_similarity_search(
        self,
        query_embedding: List[float],
        user_id: str,
        document_ids: Optional[List[str]],
        top_k: int,
        score_threshold: float,
        session: Session
    ) -> List[SearchResult]:
        """Fallback similarity search using Python."""
        logger.warning("Using fallback Python-based similarity search")
        import uuid as uuid_lib

        # Get all embeddings for user's documents
        query = session.query(DocumentEmbedding).join(DocumentModel)
        query = query.filter(DocumentModel.user_id == uuid_lib.UUID(user_id) if isinstance(user_id, str) else user_id)

        if document_ids:
            # Convert string UUIDs to UUID objects
            uuid_doc_ids = [uuid_lib.UUID(d) if isinstance(d, str) else d for d in document_ids]
            query = query.filter(DocumentEmbedding.document_id.in_(uuid_doc_ids))

        embeddings = query.all()

        if not embeddings:
            return []

        # Calculate similarities
        query_vec = np.array(query_embedding)
        results = []

        for emb in embeddings:
            if emb.embedding is not None:
                doc_vec = np.array(emb.embedding)

                # Cosine similarity
                similarity = float(
                    np.dot(query_vec, doc_vec) /
                    (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                )

                if similarity >= score_threshold:
                    results.append(SearchResult(
                        content=emb.content,
                        metadata=emb.chunk_metadata or {},
                        score=similarity,
                        document_id=str(emb.document_id),
                        chunk_index=emb.chunk_index,
                    ))

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def mmr_search(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[SearchResult]:
        """
        Maximal Marginal Relevance search for diverse results.

        Args:
            query: Search query
            user_id: User ID
            document_ids: Document IDs to search
            top_k: Number of final results
            fetch_k: Number of candidates to fetch
            lambda_mult: Diversity vs relevance trade-off (0=diverse, 1=relevant)

        Returns:
            List of diverse SearchResult objects
        """
        # First, get more candidates than needed
        candidates = self.similarity_search(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            top_k=fetch_k,
            score_threshold=0.0
        )

        if len(candidates) <= top_k:
            return candidates

        # Get query embedding
        query_embedding = np.array(self.embeddings_service.embed_query(query))

        # Get candidate embeddings
        candidate_embeddings = []
        for result in candidates:
            # Re-embed the content (could optimize by storing embeddings in results)
            emb = self.embeddings_service.embed_query(result.content)
            candidate_embeddings.append(np.array(emb))

        # MMR selection
        selected_indices = []
        candidate_indices = list(range(len(candidates)))

        while len(selected_indices) < top_k and candidate_indices:
            best_score = -float('inf')
            best_idx = -1

            for idx in candidate_indices:
                # Relevance to query
                relevance = candidates[idx].score

                # Maximum similarity to already selected
                if selected_indices:
                    max_sim = max(
                        float(np.dot(candidate_embeddings[idx], candidate_embeddings[sel_idx]) /
                              (np.linalg.norm(candidate_embeddings[idx]) *
                               np.linalg.norm(candidate_embeddings[sel_idx])))
                        for sel_idx in selected_indices
                    )
                else:
                    max_sim = 0

                # MMR score
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx != -1:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def hybrid_search(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 15,
        vector_weight: float = 0.4,
        keyword_weight: float = 0.6
    ) -> List[SearchResult]:
        """
        Comprehensive hybrid search combining vector similarity and keyword matching.
        Prioritizes keyword matches while using semantic similarity for related content.

        Args:
            query: Search query
            user_id: User ID
            document_ids: Document IDs to search
            top_k: Number of results
            vector_weight: Weight for vector similarity
            keyword_weight: Weight for keyword matching

        Returns:
            List of SearchResult with combined scores
        """
        logger.info(f"Hybrid search for: '{query}' (vector={vector_weight}, keyword={keyword_weight})")

        # Get vector search results - fetch many for comprehensive coverage
        vector_results = self.similarity_search(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            top_k=max(top_k * 3, 30)  # Fetch at least 30
        )
        logger.info(f"Vector search found {len(vector_results)} results")

        # Get keyword search results - comprehensive search across ALL document sections
        keyword_results = self._keyword_search(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            top_k=max(top_k * 4, 60)  # Fetch at least 60 to ensure coverage from all sections
        )
        logger.info(f"Keyword search found {len(keyword_results)} results")

        # Combine results - boost chunks that appear in both searches
        combined_scores: Dict[str, Tuple[SearchResult, float, bool, bool]] = {}

        # Add vector results
        for result in vector_results:
            key = f"{result.document_id}_{result.chunk_index}"
            vector_score = result.score * vector_weight
            combined_scores[key] = (result, vector_score, True, False)  # (result, score, in_vector, in_keyword)

        # Add/update with keyword results - keyword matches are important!
        for result in keyword_results:
            key = f"{result.document_id}_{result.chunk_index}"
            keyword_score = result.score * keyword_weight

            if key in combined_scores:
                # Found in both! Boost the score significantly
                existing_result, existing_score, _, _ = combined_scores[key]
                boost = 1.5  # 50% boost for appearing in both searches
                combined_scores[key] = (existing_result, (existing_score + keyword_score) * boost, True, True)
            else:
                # Keyword only - still important
                combined_scores[key] = (result, keyword_score, False, True)

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )

        # Update scores and return more results
        final_results = []
        for result, score, in_vector, in_keyword in sorted_results[:top_k]:
            result.score = score
            # Add metadata about search sources
            result.metadata["in_vector_search"] = in_vector
            result.metadata["in_keyword_search"] = in_keyword
            final_results.append(result)

        logger.info(f"Hybrid search returning {len(final_results)} combined results")
        return final_results

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract ALL meaningful words from query for matching.
        Only removes very basic stop words to ensure we catch important terms.
        """
        # Only remove the most basic stop words
        basic_stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'or', 'but', 'if', 'so', 'as',
            'what', 'which', 'who', 'whom', 'how', 'why', 'when', 'where',
            'i', 'me', 'my', 'you', 'your', 'he', 'she', 'it', 'we', 'they',
            'this', 'that', 'these', 'those',
        }

        # Extract words, convert to lowercase
        words = query.lower().split()
        keywords = [w.strip('.,!?;:"\'()[]{}') for w in words]
        # Keep words that are meaningful (longer than 1 char and not basic stop words)
        keywords = [w for w in keywords if w and len(w) > 1 and w not in basic_stop_words]

        return keywords

    def _keyword_search(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 15
    ) -> List[SearchResult]:
        """
        Comprehensive keyword search - finds ALL chunks containing keywords.
        Returns chunks from ALL sections of document (beginning, middle, end).
        Guarantees coverage across the entire document.
        """
        session = SessionLocal()

        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)

            if not keywords:
                # If no keywords extracted, use the whole query
                keywords = [query.lower().strip()]

            logger.info(f"Keyword search with terms: {keywords}")

            # Build parameters - NO LIMIT to find ALL matches first
            params = {
                "user_id": user_id,
            }

            # Build ILIKE conditions for each keyword
            ilike_conditions = []
            match_scores = []

            for i, kw in enumerate(keywords):
                params[f"kw_{i}"] = f"%{kw}%"
                ilike_conditions.append(f"LOWER(de.content) LIKE :kw_{i}")
                # Score: 1 point for each keyword found
                match_scores.append(f"CASE WHEN LOWER(de.content) LIKE :kw_{i} THEN 1.0 ELSE 0.0 END")

            # WHERE clause: match ANY keyword (OR)
            where_clause = " OR ".join(ilike_conditions)

            # Score: proportion of keywords matched
            score_calculation = f"({' + '.join(match_scores)}) / {len(keywords)}.0"

            # Document filter
            doc_filter = ""
            if document_ids:
                doc_filter = "AND de.document_id = ANY(CAST(:document_ids AS UUID[]))"
                params["document_ids"] = document_ids

            # First, find ALL chunks containing keywords (no limit)
            sql = text(f"""
                SELECT
                    de.id,
                    de.document_id,
                    de.chunk_index,
                    de.content,
                    de.metadata,
                    {score_calculation} as score
                FROM document_embeddings de
                JOIN documents d ON de.document_id = d.id
                WHERE d.user_id = CAST(:user_id AS UUID)
                {doc_filter}
                AND ({where_clause})
                ORDER BY de.chunk_index ASC
            """)

            all_results = session.execute(sql, params).fetchall()

            logger.info(f"Keyword search found {len(all_results)} TOTAL chunks containing keywords: {keywords}")

            if not all_results:
                return []

            # STRATEGY: Ensure chunks from ALL sections of document are included
            # Divide into sections: beginning (first 1/3), middle (second 1/3), end (last 1/3)
            total_chunks = len(all_results)
            results_by_index = sorted(all_results, key=lambda x: x.chunk_index)

            # Find max chunk index to understand document size
            max_chunk_idx = max(r.chunk_index for r in all_results)

            # Categorize chunks into sections based on their position
            beginning_threshold = max_chunk_idx // 3
            middle_threshold = (max_chunk_idx * 2) // 3

            beginning_chunks = [r for r in results_by_index if r.chunk_index <= beginning_threshold]
            middle_chunks = [r for r in results_by_index if beginning_threshold < r.chunk_index <= middle_threshold]
            end_chunks = [r for r in results_by_index if r.chunk_index > middle_threshold]

            logger.info(f"Document sections - Beginning: {len(beginning_chunks)}, Middle: {len(middle_chunks)}, End: {len(end_chunks)}")

            # Allocate slots proportionally but guarantee at least some from each section
            slots_per_section = max(top_k // 3, 3)  # At least 3 from each section

            selected_indices = set()
            final_results = []

            # PRIORITY 1: Add chunks from BEGINNING of document (page 1, early content)
            for row in beginning_chunks[:slots_per_section + 5]:  # Extra from beginning
                key = f"{row.document_id}_{row.chunk_index}"
                if key not in selected_indices:
                    selected_indices.add(key)
                    final_results.append(row)

            # PRIORITY 2: Add chunks from MIDDLE of document
            for row in middle_chunks[:slots_per_section]:
                if len(final_results) >= top_k:
                    break
                key = f"{row.document_id}_{row.chunk_index}"
                if key not in selected_indices:
                    selected_indices.add(key)
                    final_results.append(row)

            # PRIORITY 3: Add chunks from END of document
            for row in end_chunks[:slots_per_section]:
                if len(final_results) >= top_k:
                    break
                key = f"{row.document_id}_{row.chunk_index}"
                if key not in selected_indices:
                    selected_indices.add(key)
                    final_results.append(row)

            # PRIORITY 4: Fill remaining with highest scoring chunks from any section
            results_by_score = sorted(all_results, key=lambda x: x.score, reverse=True)
            for row in results_by_score:
                if len(final_results) >= top_k:
                    break
                key = f"{row.document_id}_{row.chunk_index}"
                if key not in selected_indices:
                    selected_indices.add(key)
                    final_results.append(row)

            # Sort final results by chunk_index for coherent context
            final_results = sorted(final_results, key=lambda x: x.chunk_index)

            logger.info(f"Keyword search returning {len(final_results)} results (sections: beginning={len([r for r in final_results if r.chunk_index <= beginning_threshold])}, middle={len([r for r in final_results if beginning_threshold < r.chunk_index <= middle_threshold])}, end={len([r for r in final_results if r.chunk_index > middle_threshold])})")

            return [
                SearchResult(
                    content=row.content,
                    metadata=row.metadata or {},
                    score=float(row.score) if row.score else 0.0,
                    document_id=str(row.document_id),
                    chunk_index=row.chunk_index,
                )
                for row in final_results
            ]

        except Exception as e:
            logger.error(f"Keyword search error: {str(e)}")
            return []

        finally:
            session.close()

    def delete_document(self, document_id: str) -> int:
        """
        Delete all embeddings for a document.

        Args:
            document_id: Document ID to delete

        Returns:
            Number of embeddings deleted
        """
        session = SessionLocal()

        try:
            result = session.query(DocumentEmbedding).filter(
                DocumentEmbedding.document_id == document_id
            ).delete()

            session.commit()
            logger.info(f"Deleted {result} embeddings for document {document_id}")
            return result

        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting document embeddings: {str(e)}")
            raise

        finally:
            session.close()

    def get_document_chunks(
        self,
        document_id: str,
        limit: int = 100
    ) -> List[SearchResult]:
        """Get all chunks for a specific document."""
        session = SessionLocal()

        try:
            embeddings = session.query(DocumentEmbedding).filter(
                DocumentEmbedding.document_id == document_id
            ).order_by(DocumentEmbedding.chunk_index).limit(limit).all()

            return [
                SearchResult(
                    content=emb.content,
                    metadata=emb.chunk_metadata or {},
                    score=1.0,
                    document_id=str(emb.document_id),
                    chunk_index=emb.chunk_index,
                )
                for emb in embeddings
            ]

        finally:
            session.close()


# Global instance - lazy initialization
_vector_store: Optional[VectorStoreService] = None


def get_vector_store() -> VectorStoreService:
    """Get or create the global vector store service."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store
