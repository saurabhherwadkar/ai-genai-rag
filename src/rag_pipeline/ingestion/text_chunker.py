# =============================================================================
# RAG Pipeline - Text Chunker
# =============================================================================
# Splits documents into overlapping chunks suitable for embedding.

import logging  # Import logging for chunking operation tracking
import uuid  # Import uuid for generating unique chunk identifiers

from rag_pipeline.models.chunk import Chunk  # Import Chunk data model
from rag_pipeline.models.document import Document  # Import Document data model
from rag_pipeline.models.pipeline_config import ChunkingConfig  # Import config model
from rag_pipeline.utils.exceptions import ChunkingError  # Import chunking exception

logger = logging.getLogger(__name__)  # Create module-level logger instance


class TextChunker:
    """Splits documents into overlapping chunks suitable for embedding.

    Uses recursive character splitting with configurable boundaries.
    Ensures chunks respect sentence boundaries where possible and
    applies overlap between adjacent chunks for context continuity.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize the text chunker with chunking configuration.

        Args:
            config: ChunkingConfig specifying chunk size, overlap, and separators.
        """
        self._config = config  # Store the chunking configuration
        self._logger = logger  # Store logger reference for this instance

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks and return Chunk objects.

        Args:
            document: The Document to split into chunks.

        Returns:
            List of Chunk objects representing segments of the document.

        Raises:
            ChunkingError: If the document content is empty or chunking fails.
        """
        self._validate_document(document)  # Ensure document has valid content
        raw_chunks = self._split_text(document.content)  # Split text into raw strings
        raw_chunks = self._merge_small_chunks(raw_chunks)  # Merge chunks below minimum size
        overlapped_chunks = self._apply_overlap(raw_chunks)  # Apply overlap between chunks
        chunk_objects = self._create_chunk_objects(  # Convert strings to Chunk objects
            overlapped_chunks, document  # Pass chunks and source document
        )
        self._logger.info(  # Log chunking result summary
            "Document %s chunked into %d pieces (avg size=%d chars)",  # Format message
            document.document_id[:8],  # First 8 chars of document ID
            len(chunk_objects),  # Number of chunks created
            self._calculate_average_size(chunk_objects),  # Average chunk size
        )
        return chunk_objects  # Return the list of Chunk objects

    def _validate_document(self, document: Document) -> None:
        """Validate that the document has non-empty content for chunking.

        Args:
            document: The document to validate.

        Raises:
            ChunkingError: If document content is empty or whitespace-only.
        """
        if not document.content or not document.content.strip():  # Check for empty content
            raise ChunkingError(  # Raise error with document identification
                f"Document {document.document_id} has empty content, cannot chunk"
            )

    def _split_text(self, text: str) -> list[str]:
        """Split text into segments using the configured separator and size.

        First splits on the primary separator (e.g., paragraph breaks),
        then further splits any segments that exceed the chunk size.

        Args:
            text: The full text content to split.

        Returns:
            List of text segments (before overlap is applied).
        """
        # Split on the primary separator first (e.g., double newline for paragraphs)
        segments = text.split(self._config.separator)  # Split on configured separator
        chunks = []  # Initialize list to collect final chunks
        for segment in segments:  # Process each segment from the initial split
            segment = segment.strip()  # Remove leading/trailing whitespace
            if not segment:  # Skip empty segments resulting from consecutive separators
                continue  # Move to next segment
            if len(segment) <= self._config.chunk_size:  # If segment fits in one chunk
                chunks.append(segment)  # Add segment as-is
            else:  # Segment exceeds chunk size, needs further splitting
                sub_chunks = self._split_large_segment(segment)  # Split into smaller pieces
                chunks.extend(sub_chunks)  # Add all sub-chunks to the result
        return chunks  # Return all text segments

    def _split_large_segment(self, segment: str) -> list[str]:
        """Split a segment that exceeds chunk_size into smaller pieces.

        Attempts to split at sentence boundaries when possible.

        Args:
            segment: A text segment larger than the configured chunk size.

        Returns:
            List of smaller text segments within the chunk size limit.
        """
        chunks = []  # Initialize list for sub-chunks
        remaining = segment  # Track the remaining text to split
        while len(remaining) > self._config.chunk_size:  # While text exceeds chunk size
            split_point = self._find_split_point(remaining)  # Find best split position
            chunks.append(remaining[:split_point].strip())  # Add text before split point
            remaining = remaining[split_point:].strip()  # Update remaining text
        if remaining:  # If there's any remaining text after the loop
            chunks.append(remaining)  # Add the final piece
        return chunks  # Return all sub-chunks

    def _find_split_point(self, text: str) -> int:
        """Find the best position to split text near the chunk size boundary.

        Prefers splitting at sentence endings, then at word boundaries,
        and falls back to the exact chunk size if no better split exists.

        Args:
            text: The text to find a split point in.

        Returns:
            Character index at which to split the text.
        """
        target = self._config.chunk_size  # The ideal split position
        # Look for sentence-ending punctuation near the target position
        search_start = max(0, target - 100)  # Start searching 100 chars before target
        search_region = text[search_start:target]  # Extract the search region
        # Find the last sentence boundary in the search region
        for punct in (".", "!", "?"):  # Try each sentence-ending punctuation
            last_punct = search_region.rfind(punct)  # Find last occurrence
            if last_punct != -1:  # If punctuation found in search region
                return search_start + last_punct + 1  # Split after the punctuation
        # No sentence boundary found; try splitting at a space (word boundary)
        last_space = text[:target].rfind(" ")  # Find last space before target
        if last_space > 0:  # If a space was found (not at position 0)
            return last_space  # Split at the word boundary
        return target  # Fall back to splitting at exactly chunk_size

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap between adjacent chunks for context continuity.

        Each chunk (except the first) starts with characters from the
        end of the previous chunk, ensuring no information is lost at boundaries.

        Args:
            chunks: List of text segments without overlap.

        Returns:
            List of text segments with overlap applied between adjacent chunks.
        """
        if len(chunks) <= 1:  # No overlap needed for single chunk or empty list
            return chunks  # Return as-is
        overlap_size = self._config.chunk_overlap  # Get configured overlap size
        if overlap_size <= 0:  # If overlap is disabled (zero or negative)
            return chunks  # Return chunks without any overlap
        overlapped = [chunks[0]]  # First chunk stays unchanged
        for i in range(1, len(chunks)):  # Process each subsequent chunk
            previous_chunk = chunks[i - 1]  # Get the preceding chunk
            overlap_text = previous_chunk[-overlap_size:]  # Extract overlap from end
            overlapped_chunk = overlap_text + " " + chunks[i]  # Prepend overlap to current
            overlapped.append(overlapped_chunk)  # Add the overlapped chunk to result
        return overlapped  # Return chunks with overlap applied

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge chunks that fall below the minimum size threshold.

        Combines small adjacent chunks to avoid creating fragments
        that would generate poor quality embeddings.

        Args:
            chunks: List of text segments to check for minimum size.

        Returns:
            List with small chunks merged into their neighbors.
        """
        if not chunks:  # If input list is empty
            return chunks  # Return empty list
        min_size = self._config.min_chunk_size  # Get minimum chunk size threshold
        merged = []  # Initialize list for merged results
        buffer = ""  # Initialize buffer to accumulate small chunks
        for chunk in chunks:  # Process each chunk
            if buffer:  # If buffer has accumulated text from previous small chunks
                buffer = buffer + " " + chunk  # Append current chunk to buffer
            else:  # Buffer is empty, start fresh
                buffer = chunk  # Set buffer to current chunk
            if len(buffer) >= min_size:  # If buffer meets minimum size requirement
                merged.append(buffer)  # Add buffer to merged results
                buffer = ""  # Reset buffer for next accumulation
        if buffer:  # If buffer still has content after processing all chunks
            if merged:  # If there are already merged chunks
                merged[-1] = merged[-1] + " " + buffer  # Append to last merged chunk
            else:  # No merged chunks exist yet
                merged.append(buffer)  # Add buffer as the only chunk
        return merged  # Return the merged chunks list

    def _create_chunk_objects(
        self, chunk_texts: list[str], document: Document
    ) -> list[Chunk]:
        """Convert raw text segments into Chunk data objects.

        Args:
            chunk_texts: List of text strings to convert to Chunk objects.
            document: Source document for ID reference and metadata inheritance.

        Returns:
            List of Chunk objects with IDs, positions, and metadata.
        """
        chunks = []  # Initialize list for Chunk objects
        current_position = 0  # Track character position in original document
        for index, text in enumerate(chunk_texts):  # Enumerate chunks with index
            chunk_id = str(uuid.uuid4())  # Generate unique ID for this chunk
            # Find where this chunk text starts in the original document
            start_char = document.content.find(text[:50], current_position)  # Search for start
            if start_char == -1:  # If exact match not found (due to overlap modifications)
                start_char = current_position  # Use current tracking position
            end_char = start_char + len(text)  # Calculate end character position
            chunk = Chunk(  # Create the Chunk data object
                chunk_id=chunk_id,  # Assign the unique chunk identifier
                document_id=document.document_id,  # Reference the parent document
                content=text,  # Store the chunk text content
                chunk_index=index,  # Store the position index
                start_char=start_char,  # Store start character offset
                end_char=end_char,  # Store end character offset
                metadata=document.metadata.copy(),  # Copy parent document metadata
            )
            chunks.append(chunk)  # Add chunk to the results list
            current_position = start_char + len(text) // 2  # Advance position tracker
        return chunks  # Return all created Chunk objects

    def _calculate_average_size(self, chunks: list[Chunk]) -> int:
        """Calculate the average character size of a list of chunks.

        Args:
            chunks: List of Chunk objects to measure.

        Returns:
            Average character count per chunk, or 0 if list is empty.
        """
        if not chunks:  # Guard against empty list (division by zero)
            return 0  # Return zero for empty list
        total_chars = sum(chunk.content_length for chunk in chunks)  # Sum all chunk sizes
        return total_chars // len(chunks)  # Integer division for average
