# =============================================================================
# RAG Pipeline - Document Data Model
# =============================================================================
# Defines the immutable Document class representing a loaded source document.

from dataclasses import dataclass, field  # Import dataclass decorator and field factory
from datetime import UTC, datetime  # Import datetime for timestamp tracking


@dataclass(frozen=True)  # Frozen makes instances immutable after creation
class Document:
    """Immutable representation of a source document loaded into the pipeline.

    Each document has a unique ID, content, source path, and metadata.
    The frozen=True ensures documents cannot be accidentally modified after loading.
    """

    content: str  # The raw text content extracted from the source file
    source_path: str  # File path or URI where the document was loaded from
    document_id: str  # Unique identifier (UUID) assigned during loading
    metadata: dict = field(default_factory=dict)  # Extensible metadata dictionary
    loaded_at: str | None = None  # ISO format timestamp when document was loaded
    content_hash: str = ""  # SHA-256 hash of content for deduplication checks

    def __post_init__(self) -> None:  # Called after dataclass __init__ completes
        """Set the loaded_at timestamp if not already provided."""
        if self.loaded_at is None:  # Check if timestamp was not passed explicitly
            # Use object.__setattr__ because frozen dataclass prevents normal assignment
            object.__setattr__(  # Bypass frozen restriction for initialization only
                self,  # The instance to modify
                "loaded_at",  # The attribute name to set
                datetime.now(UTC).isoformat(),  # Current UTC time in ISO format
            )

    @property  # Define a computed property for content length
    def content_length(self) -> int:
        """Return the character count of the document content."""
        return len(self.content)  # Calculate and return the string length

    @property  # Define a computed property for word count
    def word_count(self) -> int:
        """Return the approximate word count of the document content."""
        return len(self.content.split())  # Split on whitespace and count tokens
