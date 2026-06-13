# =============================================================================
# RAG Pipeline - Query Processor
# =============================================================================
# Preprocesses and normalizes user queries before retrieval.

import logging  # Import logging for query processing tracking
import re  # Import regex for query text manipulation

from rag_pipeline.models.pipeline_config import RetrievalConfig  # Import config model
from rag_pipeline.security.input_sanitizer import InputSanitizer  # Import sanitizer
from rag_pipeline.utils.exceptions import QueryValidationError  # Import query exception

logger = logging.getLogger(__name__)  # Create module-level logger instance


class QueryProcessor:
    """Preprocesses and normalizes user queries before retrieval.

    Applies sanitization, normalization, and validation to ensure
    queries are clean and safe before being processed by the retriever.
    """

    def __init__(
        self, config: RetrievalConfig, input_sanitizer: InputSanitizer
    ) -> None:
        """Initialize the query processor with configuration and sanitizer.

        Args:
            config: RetrievalConfig with query length limits and settings.
            input_sanitizer: InputSanitizer for security validation.
        """
        self._config = config  # Store retrieval configuration
        self._sanitizer = input_sanitizer  # Store input sanitizer reference
        self._logger = logger  # Store logger reference for this instance

    def process_query(self, raw_query: str) -> str:
        """Process a raw user query through all preprocessing steps.

        Args:
            raw_query: The unprocessed user input query string.

        Returns:
            Cleaned, normalized, and validated query string.

        Raises:
            QueryValidationError: If the query fails validation.
        """
        self._logger.debug("Processing query: '%s'", raw_query[:50])  # Log truncated query
        sanitized = self._sanitizer.sanitize_query(raw_query)  # Apply security sanitization
        normalized = self._normalize_whitespace(sanitized)  # Normalize whitespace
        cleaned = self._remove_special_characters(normalized)  # Remove unnecessary chars
        self._validate_query_length(cleaned)  # Enforce length limits
        self._validate_not_empty(cleaned)  # Ensure query has substance
        self._logger.info(  # Log successful processing
            "Query processed: length=%d chars", len(cleaned)  # Show final length
        )
        return cleaned  # Return the fully processed query

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters into single spaces.

        Args:
            text: Input text with potential excessive whitespace.

        Returns:
            Text with normalized whitespace.
        """
        collapsed = re.sub(r"\s+", " ", text)  # Replace all whitespace runs with space
        return collapsed.strip()  # Remove leading and trailing whitespace

    def _remove_special_characters(self, text: str) -> str:
        """Remove characters that provide no semantic value for retrieval.

        Keeps alphanumeric, common punctuation, and whitespace.
        Removes exotic symbols that might confuse the embedding model.

        Args:
            text: Input text potentially containing special characters.

        Returns:
            Text with non-essential special characters removed.
        """
        # Keep letters, numbers, common punctuation, and whitespace
        cleaned = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)  # Remove exotic characters
        return cleaned  # Return the cleaned text

    def _validate_query_length(self, query: str) -> None:
        """Validate that the query does not exceed maximum length.

        Args:
            query: The query string to validate.

        Raises:
            QueryValidationError: If query exceeds maximum length.
        """
        max_length = self._config.max_query_length  # Get configured maximum
        if len(query) > max_length:  # Check if query exceeds limit
            self._logger.warning(  # Log the validation failure
                "Query exceeds max length: %d > %d", len(query), max_length  # Show lengths
            )
            raise QueryValidationError(  # Raise validation error
                f"Query length {len(query)} exceeds maximum of {max_length} characters"
            )

    def _validate_not_empty(self, query: str) -> None:
        """Validate that the query has meaningful content after processing.

        Args:
            query: The processed query string to validate.

        Raises:
            QueryValidationError: If query is empty after processing.
        """
        if not query or not query.strip():  # Check for empty content
            self._logger.warning("Query is empty after processing")  # Log the failure
            raise QueryValidationError(  # Raise validation error
                "Query is empty or contains only whitespace after processing"
            )
