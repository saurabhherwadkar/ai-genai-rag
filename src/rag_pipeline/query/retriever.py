# =============================================================================
# RAG Pipeline - Retriever
# =============================================================================
# Retrieves relevant chunks from the vector store given a processed query.

import logging  # Import logging for retrieval operation tracking

from rag_pipeline.ingestion.embedding_generator import EmbeddingGenerator  # Embedding component
from rag_pipeline.models.pipeline_config import RetrievalConfig  # Import config model
from rag_pipeline.models.query_result import QueryResult  # Import result model
from rag_pipeline.vectorstore.vector_store_manager import VectorStoreManager  # Store component

logger = logging.getLogger(__name__)  # Create module-level logger instance


class Retriever:
    """Retrieves relevant chunks from the vector store given a processed query.

    Coordinates embedding the query and performing similarity search,
    then filters results by a minimum similarity threshold.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_generator: EmbeddingGenerator,
        config: RetrievalConfig,
    ) -> None:
        """Initialize the retriever with vector store and embedding components.

        Args:
            vector_store: The vector store to search for similar chunks.
            embedding_generator: Generator to embed the query text.
            config: RetrievalConfig with top_k and threshold settings.
        """
        self._vector_store = vector_store  # Store vector store reference
        self._embedding_generator = embedding_generator  # Store embedding generator reference
        self._config = config  # Store retrieval configuration
        self._logger = logger  # Store logger reference for this instance

    def retrieve(self, processed_query: str, top_k_override: int | None = None) -> list[QueryResult]:
        """Retrieve relevant chunks for the given processed query.

        Args:
            processed_query: Cleaned and validated query string.
            top_k_override: Optional override for number of results to fetch.

        Returns:
            List of QueryResult objects ranked by similarity.

        Raises:
            VectorStoreError: If retrieval from the vector store fails.
        """
        top_k = top_k_override if top_k_override is not None else self._config.top_k
        self._logger.debug("Retrieving for query: '%s'", processed_query[:50])  # Log query
        query_embedding = self._embed_query(processed_query)  # Step 1: Embed the query
        raw_results = self._vector_store.search_similar(  # Step 2: Search vector store
            query_embedding=query_embedding,  # Pass the query embedding
            top_k=top_k,  # Limit results to top_k
        )
        filtered_results = self._filter_by_threshold(raw_results)  # Step 3: Filter by score
        deduplicated = self._deduplicate_results(filtered_results)  # Step 4: Remove duplicates
        self._logger.info(  # Log retrieval summary
            "Retrieved %d results (from %d raw, %d after filter)",  # Format message
            len(deduplicated),  # Final result count
            len(raw_results),  # Raw results before filtering
            len(filtered_results),  # Results after threshold filter
        )
        return deduplicated  # Return the final result list

    def _embed_query(self, query: str) -> list[float]:
        """Generate an embedding vector for the query text.

        Args:
            query: The query text to embed.

        Returns:
            List of floats representing the query embedding vector.
        """
        embedding = self._embedding_generator.generate_embedding(query)  # Generate embedding
        self._logger.debug(  # Log embedding generation
            "Query embedded (dimension=%d)", len(embedding)  # Show embedding size
        )
        return embedding  # Return the query embedding vector

    def _filter_by_threshold(self, results: list[QueryResult]) -> list[QueryResult]:
        """Filter results that fall below the minimum similarity threshold.

        Args:
            results: List of QueryResult objects to filter.

        Returns:
            List with only results meeting the similarity threshold.
        """
        threshold = self._config.similarity_threshold  # Get configured threshold
        filtered = [  # Keep only results above threshold
            result for result in results  # Check each result
            if result.similarity_score >= threshold  # Compare score to threshold
        ]
        removed_count = len(results) - len(filtered)  # Calculate how many were removed
        if removed_count > 0:  # If any results were filtered out
            self._logger.debug(  # Log filtering action
                "Filtered %d results below threshold %.2f",  # Format message
                removed_count,  # Number of results removed
                threshold,  # The threshold value used
            )
        return filtered  # Return the filtered results

    def _deduplicate_results(self, results: list[QueryResult]) -> list[QueryResult]:
        """Remove duplicate chunks from results based on content similarity.

        Chunks with identical content (e.g., from overlapping regions)
        are deduplicated, keeping the one with the higher similarity score.

        Args:
            results: List of QueryResult objects to deduplicate.

        Returns:
            List with duplicate content removed.
        """
        seen_content = set()  # Track content hashes we've already seen
        unique_results = []  # Initialize list for unique results
        for result in results:  # Process each result
            # Use first 100 chars as content fingerprint for deduplication
            content_key = result.chunk.content[:100]  # Extract content fingerprint
            if content_key not in seen_content:  # If content not seen before
                seen_content.add(content_key)  # Mark content as seen
                unique_results.append(result)  # Add to unique results
        return unique_results  # Return deduplicated results
