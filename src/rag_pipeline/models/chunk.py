# =============================================================================
# RAG Pipeline - Chunk Data Model
# =============================================================================
# Defines the Chunk class representing a segment of a document after splitting.

from dataclasses import dataclass, field  # Import dataclass decorator and field factory


@dataclass(frozen=True)  # Frozen ensures chunk immutability after creation
class Chunk:
    """A segment of a document after text splitting, ready for embedding.

    Each chunk maintains a reference to its parent document and tracks its
    position within the original text for source attribution.
    """

    chunk_id: str  # Unique identifier for this specific chunk
    document_id: str  # Reference to the parent document's unique ID
    content: str  # The text content of this chunk after splitting
    chunk_index: int  # Zero-based position of this chunk within the parent document
    start_char: int  # Character offset where this chunk starts in the original text
    end_char: int  # Character offset where this chunk ends in the original text
    metadata: dict = field(default_factory=dict)  # Inherited and chunk-specific metadata
    embedding: list[float] | None = None  # Vector embedding populated after embedding step

    @property  # Define a computed property for content length
    def content_length(self) -> int:
        """Return the character count of the chunk content."""
        return len(self.content)  # Calculate and return the string length

    @property  # Define a computed property to check if embedding exists
    def has_embedding(self) -> bool:
        """Check whether this chunk has been embedded."""
        return self.embedding is not None  # Return True if embedding has been set

    def with_embedding(self, embedding: list[float]) -> "Chunk":
        """Create a new Chunk instance with the given embedding attached.

        Since Chunk is frozen (immutable), this returns a new instance
        rather than modifying the existing one.
        """
        return Chunk(  # Create and return a new Chunk with all same fields
            chunk_id=self.chunk_id,  # Preserve the original chunk ID
            document_id=self.document_id,  # Preserve the parent document reference
            content=self.content,  # Preserve the text content
            chunk_index=self.chunk_index,  # Preserve the position index
            start_char=self.start_char,  # Preserve the start character offset
            end_char=self.end_char,  # Preserve the end character offset
            metadata=self.metadata,  # Preserve all metadata
            embedding=embedding,  # Attach the new embedding vector
        )
