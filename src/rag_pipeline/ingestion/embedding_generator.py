# =============================================================================
# RAG Pipeline - Embedding Generator
# =============================================================================
# Generates vector embeddings for text chunks using sentence-transformers.

import logging  # Import logging for embedding operation tracking

from rag_pipeline.models.chunk import Chunk  # Import Chunk data model
from rag_pipeline.models.pipeline_config import EmbeddingConfig  # Import config model
from rag_pipeline.utils.exceptions import EmbeddingError  # Import embedding exception

logger = logging.getLogger(__name__)  # Create module-level logger instance


class EmbeddingGenerator:
    """Generates vector embeddings for text chunks using sentence-transformers.

    Supports batch processing for memory efficiency and caches the loaded
    model to avoid repeated initialization. Embeddings can optionally be
    L2-normalized to unit vectors for cosine similarity computation.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize the embedding generator with model configuration.

        The model is loaded lazily on first use to avoid slow startup.

        Args:
            config: EmbeddingConfig specifying model name, batch size, etc.
        """
        self._config = config  # Store the embedding configuration
        self._model = None  # Initialize model reference as None (lazy loading)
        self._logger = logger  # Store logger reference for this instance
        self._embedding_dimension: int | None = None  # Cache the embedding size

    def generate_embedding(self, text: str) -> list[float]:
        """Generate a single embedding vector for the given text.

        Args:
            text: Input text string to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        validated_text = self._validate_input_text(text)  # Validate and clean input
        self._ensure_model_loaded()  # Load model if not already cached
        try:  # Attempt to generate the embedding
            embedding = self._model.encode(  # Call the model's encode method
                validated_text,  # Pass the validated text
                normalize_embeddings=self._config.normalize,  # Apply normalization setting
            )
            embedding_list = embedding.tolist()  # Convert numpy array to Python list
            return embedding_list  # Return the embedding as a list of floats
        except Exception as error:  # Catch any model inference errors
            self._logger.error("Embedding generation failed: %s", error)  # Log the error
            raise EmbeddingError(  # Wrap in domain exception
                f"Failed to generate embedding: {error}"  # Include error details
            ) from error  # Chain the original exception

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts efficiently.

        Processes texts in batches to optimize memory usage and throughput.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            EmbeddingError: If batch embedding generation fails.
        """
        if not texts:  # Handle empty input list
            return []  # Return empty list for empty input
        validated_texts = [  # Validate each text in the batch
            self._validate_input_text(text) for text in texts  # Apply validation to each
        ]
        self._ensure_model_loaded()  # Load model if not already cached
        try:  # Attempt batch encoding
            embeddings = self._model.encode(  # Call model's batch encode method
                validated_texts,  # Pass all validated texts at once
                batch_size=self._config.batch_size,  # Use configured batch size
                normalize_embeddings=self._config.normalize,  # Apply normalization setting
                show_progress_bar=False,  # Disable progress bar for clean logging
            )
            embeddings_list = embeddings.tolist()  # Convert numpy array to Python lists
            self._logger.debug(  # Log batch processing result
                "Generated %d embeddings (dim=%d)",  # Format with count and dimension
                len(embeddings_list),  # Number of embeddings generated
                len(embeddings_list[0]) if embeddings_list else 0,  # Dimension size
            )
            return embeddings_list  # Return list of embedding vectors
        except Exception as error:  # Catch any batch processing errors
            self._logger.error("Batch embedding failed: %s", error)  # Log the error
            raise EmbeddingError(  # Wrap in domain exception
                f"Batch embedding generation failed: {error}"  # Include error details
            ) from error  # Chain the original exception

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Generate embeddings for a list of chunks and return updated chunks.

        Creates new Chunk instances with embeddings attached (chunks are immutable).

        Args:
            chunks: List of Chunk objects to generate embeddings for.

        Returns:
            List of new Chunk objects with embeddings populated.
        """
        if not chunks:  # Handle empty input list
            return []  # Return empty list for empty input
        texts = [chunk.content for chunk in chunks]  # Extract text content from chunks
        embeddings = self.generate_embeddings_batch(texts)  # Generate all embeddings
        embedded_chunks = []  # Initialize list for chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings, strict=True):  # Pair chunks with embeddings
            new_chunk = chunk.with_embedding(embedding)  # Create chunk with embedding
            embedded_chunks.append(new_chunk)  # Add to results list
        self._logger.info(  # Log embedding completion summary
            "Embedded %d chunks successfully",  # Format message
            len(embedded_chunks),  # Number of chunks embedded
        )
        return embedded_chunks  # Return chunks with embeddings attached

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embedding vectors produced by the model.

        Returns:
            Integer representing the embedding vector dimension.
        """
        if self._embedding_dimension is not None:  # Return cached value if available
            return self._embedding_dimension  # Avoid recomputation
        self._ensure_model_loaded()  # Load model to determine dimension
        dimension = self._model.get_sentence_embedding_dimension()  # Query model dimension
        self._embedding_dimension = dimension  # Cache the dimension value
        return dimension  # Return the embedding dimension

    def _ensure_model_loaded(self) -> None:
        """Load the sentence-transformer model if not already in memory.

        Raises:
            EmbeddingError: If the model cannot be loaded.
        """
        if self._model is not None:  # Check if model is already loaded
            return  # Skip loading if model is cached
        try:  # Attempt to load the model
            from sentence_transformers import SentenceTransformer  # Import lazily

            self._logger.info(  # Log model loading start
                "Loading embedding model: %s", self._config.model_name  # Show model name
            )
            self._model = SentenceTransformer(  # Initialize the model
                self._config.model_name  # Use configured model identifier
            )
            self._logger.info("Embedding model loaded successfully")  # Log completion
        except Exception as error:  # Catch model loading failures
            self._logger.error(  # Log the loading failure
                "Failed to load embedding model %s: %s",  # Format with model and error
                self._config.model_name,  # The model that failed to load
                error,  # The error that occurred
            )
            raise EmbeddingError(  # Wrap in domain exception
                f"Cannot load embedding model '{self._config.model_name}': {error}"
            ) from error  # Chain the original exception

    def _validate_input_text(self, text: str) -> str:
        """Validate and clean input text before embedding.

        Args:
            text: Raw input text to validate.

        Returns:
            Cleaned text suitable for embedding.

        Raises:
            EmbeddingError: If text is empty after cleaning.
        """
        if not text:  # Check for None or empty string
            raise EmbeddingError("Cannot embed empty text")  # Raise with clear message
        cleaned = text.strip()  # Remove leading and trailing whitespace
        if not cleaned:  # Check if text was only whitespace
            raise EmbeddingError("Cannot embed whitespace-only text")  # Raise error
        return cleaned  # Return the cleaned text
