# =============================================================================
# RAG Pipeline - Custom Exception Hierarchy
# =============================================================================
# Defines domain-specific exceptions for clear error handling throughout
# the pipeline. Each exception type maps to a specific failure domain.


class RAGPipelineError(Exception):
    """Base exception for all RAG pipeline errors.

    All custom exceptions in this project inherit from this class,
    allowing callers to catch all pipeline errors with a single handler.
    """

    pass  # No additional behavior needed for the base class


class DocumentLoadError(RAGPipelineError):
    """Raised when a document cannot be loaded from its source.

    Common causes: file not found, permission denied, encoding errors,
    unsupported file format, or corrupted file content.
    """

    pass  # Inherits message and traceback from RAGPipelineError


class ChunkingError(RAGPipelineError):
    """Raised when text chunking fails during document processing.

    Common causes: empty document content, invalid chunking configuration,
    or unexpected text structure that prevents proper splitting.
    """

    pass  # Inherits message and traceback from RAGPipelineError


class EmbeddingError(RAGPipelineError):
    """Raised when embedding generation fails.

    Common causes: model loading failure, invalid input text,
    out of memory during batch processing, or model inference errors.
    """

    pass  # Inherits message and traceback from RAGPipelineError


class VectorStoreError(RAGPipelineError):
    """Raised when vector store operations fail.

    Common causes: collection not found, dimension mismatch,
    storage capacity exceeded, or database corruption.
    """

    pass  # Inherits message and traceback from RAGPipelineError


class QueryValidationError(RAGPipelineError):
    """Raised when a query fails validation checks.

    Common causes: empty query, query exceeds maximum length,
    or query contains only whitespace/special characters.
    """

    pass  # Inherits message and traceback from RAGPipelineError


class ConfigurationError(RAGPipelineError):
    """Raised when configuration is missing, invalid, or malformed.

    Common causes: missing required config file, invalid YAML syntax,
    missing required configuration keys, or invalid parameter values.
    """

    pass  # Inherits message and traceback from RAGPipelineError


class SecurityViolationError(RAGPipelineError):
    """Raised when a security check detects a violation.

    Common causes: path traversal attempt, input injection detected,
    unauthorized file access, or prompt injection pattern matched.
    """

    pass  # Inherits message and traceback from RAGPipelineError
