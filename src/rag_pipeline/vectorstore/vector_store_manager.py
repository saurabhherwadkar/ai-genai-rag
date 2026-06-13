# =============================================================================
# RAG Pipeline - Vector Store Manager
# =============================================================================
# Manages the ChromaDB in-memory vector store for document storage and retrieval.

import logging  # Import logging for vector store operation tracking

from rag_pipeline.models.chunk import Chunk  # Import Chunk data model
from rag_pipeline.models.query_result import QueryResult  # Import QueryResult model
from rag_pipeline.utils.exceptions import VectorStoreError  # Import store exception

logger = logging.getLogger(__name__)  # Create module-level logger instance


class VectorStoreManager:
    """Manages the ChromaDB in-memory vector store for document storage and retrieval.

    Handles collection creation, chunk insertion with embeddings and metadata,
    similarity search, and collection lifecycle management.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | None = None,
        distance_metric: str = "cosine",
    ) -> None:
        """Initialize the vector store manager with collection settings.

        Args:
            collection_name: Name for the ChromaDB collection.
            persist_directory: Optional directory for persistent storage (None = in-memory).
            distance_metric: Distance function for similarity (cosine, l2, ip).
        """
        self._collection_name = collection_name  # Store the collection name
        self._persist_directory = persist_directory  # Store optional persistence path
        self._distance_metric = distance_metric  # Store the distance metric setting
        self._client = None  # Initialize ChromaDB client reference as None
        self._collection = None  # Initialize collection reference as None
        self._logger = logger  # Store logger reference for this instance

    def initialize_store(self) -> None:
        """Initialize the ChromaDB client and create or get the collection.

        Must be called before any other operations on the vector store.

        Raises:
            VectorStoreError: If initialization fails.
        """
        try:  # Attempt to initialize ChromaDB
            import chromadb  # Import ChromaDB lazily (only when needed)

            if self._persist_directory:  # If persistence is configured
                self._client = chromadb.PersistentClient(  # Create persistent client
                    path=self._persist_directory  # Use configured directory
                )
                self._logger.info(  # Log persistent mode initialization
                    "ChromaDB initialized with persistence at: %s",  # Format message
                    self._persist_directory,  # Include the persistence path
                )
            else:  # No persistence configured, use in-memory mode
                self._client = chromadb.Client()  # Create ephemeral in-memory client
                self._logger.info("ChromaDB initialized in-memory mode")  # Log mode
            self._create_or_get_collection()  # Create or retrieve the collection
        except Exception as error:  # Catch any initialization errors
            self._logger.error("Vector store initialization failed: %s", error)  # Log error
            raise VectorStoreError(  # Wrap in domain exception
                f"Failed to initialize vector store: {error}"  # Include error details
            ) from error  # Chain the original exception

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add embedded chunks to the vector store collection.

        Args:
            chunks: List of Chunk objects with embeddings to store.

        Returns:
            Number of chunks successfully added to the collection.

        Raises:
            VectorStoreError: If chunks cannot be added to the store.
        """
        self._ensure_initialized()  # Verify store is ready for operations
        if not chunks:  # Handle empty input list
            return 0  # Nothing to add
        # Filter to only chunks that have embeddings
        valid_chunks = [c for c in chunks if c.has_embedding]  # Keep only embedded chunks
        if not valid_chunks:  # If no chunks have embeddings
            self._logger.warning("No chunks with embeddings to add")  # Log warning
            return 0  # Nothing to add
        try:  # Attempt to add chunks to the collection
            ids = [chunk.chunk_id for chunk in valid_chunks]  # Extract chunk IDs
            embeddings = [chunk.embedding for chunk in valid_chunks]  # Extract embeddings
            documents = [chunk.content for chunk in valid_chunks]  # Extract text content
            metadatas = [  # Build metadata for each chunk
                self._prepare_metadata_for_storage(chunk)  # Format metadata for ChromaDB
                for chunk in valid_chunks  # Process each valid chunk
            ]
            self._collection.add(  # Add all chunks to the ChromaDB collection
                ids=ids,  # Unique identifiers for each chunk
                embeddings=embeddings,  # Vector embeddings for similarity search
                documents=documents,  # Original text for retrieval
                metadatas=metadatas,  # Metadata for filtering
            )
            self._logger.info(  # Log successful addition
                "Added %d chunks to collection '%s'",  # Format with count and name
                len(valid_chunks),  # Number of chunks added
                self._collection_name,  # Collection name
            )
            return len(valid_chunks)  # Return count of added chunks
        except Exception as error:  # Catch any storage errors
            self._logger.error("Failed to add chunks: %s", error)  # Log the error
            raise VectorStoreError(  # Wrap in domain exception
                f"Failed to add chunks to vector store: {error}"  # Include details
            ) from error  # Chain the original exception

    def search_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[QueryResult]:
        """Search for chunks most similar to the query embedding.

        Args:
            query_embedding: The embedding vector of the query.
            top_k: Maximum number of results to return.

        Returns:
            List of QueryResult objects ranked by similarity.

        Raises:
            VectorStoreError: If the search operation fails.
        """
        self._ensure_initialized()  # Verify store is ready for operations
        try:  # Attempt similarity search
            results = self._collection.query(  # Query the ChromaDB collection
                query_embeddings=[query_embedding],  # Pass query embedding as list
                n_results=top_k,  # Limit to top_k results
                include=["documents", "metadatas", "distances"],  # Include all fields
            )
            query_results = self._convert_results(results)  # Convert to QueryResult objects
            self._logger.debug(  # Log search results
                "Search returned %d results", len(query_results)  # Result count
            )
            return query_results  # Return the converted results
        except Exception as error:  # Catch any search errors
            self._logger.error("Similarity search failed: %s", error)  # Log the error
            raise VectorStoreError(  # Wrap in domain exception
                f"Similarity search failed: {error}"  # Include error details
            ) from error  # Chain the original exception

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a specific document.

        Args:
            document_id: The document ID whose chunks should be removed.

        Returns:
            Number of chunks deleted.

        Raises:
            VectorStoreError: If the delete operation fails.
        """
        self._ensure_initialized()  # Verify store is ready for operations
        try:  # Attempt to delete chunks by document ID
            # Query for all chunks with this document_id in metadata
            existing = self._collection.get(  # Get matching chunk IDs
                where={"document_id": document_id},  # Filter by document ID
                include=[],  # Only need IDs, not content
            )
            chunk_ids = existing["ids"]  # Extract the matching chunk IDs
            if chunk_ids:  # If any chunks were found for this document
                self._collection.delete(ids=chunk_ids)  # Delete the found chunks
            self._logger.info(  # Log deletion result
                "Deleted %d chunks for document %s",  # Format message
                len(chunk_ids),  # Number of chunks deleted
                document_id[:8],  # First 8 chars of document ID
            )
            return len(chunk_ids)  # Return count of deleted chunks
        except Exception as error:  # Catch any deletion errors
            self._logger.error("Delete operation failed: %s", error)  # Log the error
            raise VectorStoreError(  # Wrap in domain exception
                f"Failed to delete document chunks: {error}"  # Include details
            ) from error  # Chain the original exception

    def get_collection_stats(self) -> dict:
        """Get statistics about the current collection.

        Returns:
            Dictionary with collection statistics (count, name, etc.).
        """
        self._ensure_initialized()  # Verify store is ready for operations
        count = self._collection.count()  # Get total number of items in collection
        return {  # Build and return statistics dictionary
            "collection_name": self._collection_name,  # Name of the collection
            "total_chunks": count,  # Total number of stored chunks
            "distance_metric": self._distance_metric,  # Active distance metric
            "persist_directory": self._persist_directory,  # Persistence path or None
        }

    def clear_collection(self) -> None:
        """Remove all items from the collection.

        Raises:
            VectorStoreError: If the clear operation fails.
        """
        self._ensure_initialized()  # Verify store is ready for operations
        try:  # Attempt to clear the collection
            self._client.delete_collection(self._collection_name)  # Delete collection
            self._create_or_get_collection()  # Recreate empty collection
            self._logger.info(  # Log collection cleared
                "Collection '%s' cleared", self._collection_name  # Include name
            )
        except Exception as error:  # Catch any errors during clear
            self._logger.error("Failed to clear collection: %s", error)  # Log error
            raise VectorStoreError(  # Wrap in domain exception
                f"Failed to clear collection: {error}"  # Include details
            ) from error  # Chain the original exception

    def _create_or_get_collection(self) -> None:
        """Create a new collection or get existing one by name."""
        self._collection = self._client.get_or_create_collection(  # Get or create
            name=self._collection_name,  # Use configured collection name
            metadata={"hnsw:space": self._distance_metric},  # Set distance metric
        )
        self._logger.debug(  # Log collection ready
            "Collection '%s' ready (metric=%s)",  # Format message
            self._collection_name,  # Collection name
            self._distance_metric,  # Distance metric
        )

    def _ensure_initialized(self) -> None:
        """Verify that the store has been initialized before operations.

        Raises:
            VectorStoreError: If the store has not been initialized.
        """
        if self._client is None or self._collection is None:  # Check initialization state
            raise VectorStoreError(  # Raise error with guidance
                "Vector store not initialized. Call initialize_store() first."
            )

    def _prepare_metadata_for_storage(self, chunk: Chunk) -> dict:
        """Prepare chunk metadata for ChromaDB storage.

        ChromaDB only supports string, int, float, and bool metadata values.
        This method flattens and converts metadata to compatible types.

        Args:
            chunk: The Chunk whose metadata to prepare.

        Returns:
            Dictionary with ChromaDB-compatible metadata values.
        """
        metadata = {  # Build base metadata with guaranteed fields
            "document_id": chunk.document_id,  # Parent document reference
            "chunk_index": chunk.chunk_index,  # Position within document
            "start_char": chunk.start_char,  # Start character offset
            "end_char": chunk.end_char,  # End character offset
        }
        # Add compatible fields from chunk metadata
        for key, value in chunk.metadata.items():  # Iterate through chunk metadata
            if isinstance(value, (str, int, float, bool)):  # Check type compatibility
                metadata[key] = value  # Add compatible values directly
        return metadata  # Return the prepared metadata dictionary

    def _convert_results(self, raw_results: dict) -> list[QueryResult]:
        """Convert raw ChromaDB query results into QueryResult objects.

        Args:
            raw_results: Raw dictionary returned by ChromaDB's query method.

        Returns:
            List of QueryResult objects with chunks and scores.
        """
        query_results = []  # Initialize list for converted results
        if not raw_results or not raw_results.get("ids"):  # Check for empty results
            return query_results  # Return empty list
        ids = raw_results["ids"][0]  # Get IDs from first (only) query
        documents = raw_results["documents"][0]  # Get documents from first query
        distances = raw_results["distances"][0]  # Get distances from first query
        metadatas = raw_results["metadatas"][0]  # Get metadata from first query
        for rank, (chunk_id, doc, dist, meta) in enumerate(  # Enumerate with rank
            zip(ids, documents, distances, metadatas, strict=True), start=1  # Zip all fields
        ):
            # Convert distance to similarity (ChromaDB returns distances)
            similarity = 1.0 - dist if dist <= 1.0 else 0.0  # Convert distance to similarity
            chunk = Chunk(  # Reconstruct Chunk from stored data
                chunk_id=chunk_id,  # Use the stored chunk ID
                document_id=meta.get("document_id", ""),  # Get document ID from metadata
                content=doc,  # Use the stored document text
                chunk_index=meta.get("chunk_index", 0),  # Get chunk index from metadata
                start_char=meta.get("start_char", 0),  # Get start position from metadata
                end_char=meta.get("end_char", 0),  # Get end position from metadata
                metadata=meta,  # Attach all stored metadata
            )
            result = QueryResult(  # Create QueryResult with chunk and score
                chunk=chunk,  # The retrieved chunk
                similarity_score=similarity,  # Cosine similarity score
                rank=rank,  # Position rank in results
            )
            query_results.append(result)  # Add to results list
        return query_results  # Return all converted results
