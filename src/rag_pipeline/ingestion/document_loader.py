# =============================================================================
# RAG Pipeline - Document Loader
# =============================================================================
# Loads documents from various file formats into Document objects.

import hashlib  # Import hashlib for computing content hashes (SHA-256)
import json  # Import json for parsing JSON document files
import logging  # Import logging for load operation tracking
import uuid  # Import uuid for generating unique document identifiers
from pathlib import Path  # Import Path for cross-platform file path operations

from rag_pipeline.models.document import Document  # Import Document data model
from rag_pipeline.utils.exceptions import DocumentLoadError  # Import loading exception

logger = logging.getLogger(__name__)  # Create module-level logger instance


class DocumentLoader:
    """Loads documents from various file formats into Document objects.

    Supports plain text, Markdown, PDF, and JSON files. Each loaded document
    receives a unique ID and content hash for tracking and deduplication.
    """

    # Mapping of file extensions to their respective loader methods
    SUPPORTED_FORMATS = {  # Dictionary mapping extensions to handler method names
        ".txt": "_load_text_file",  # Plain text files
        ".md": "_load_text_file",  # Markdown files (treated as plain text)
        ".pdf": "_load_pdf_file",  # PDF documents via pypdf
        ".json": "_load_json_file",  # JSON files (extracts text content)
    }

    def __init__(self, supported_extensions: list[str]) -> None:
        """Initialize the document loader with allowed file extensions.

        Args:
            supported_extensions: List of file extensions to accept (e.g., ['.txt', '.md']).
        """
        self._supported_extensions = supported_extensions  # Store allowed extensions
        self._logger = logger  # Store logger reference for this instance

    def load_file(self, file_path: Path) -> Document:
        """Load a single file and return a Document object.

        Args:
            file_path: Path to the file to load.

        Returns:
            Document object containing the file's text content and metadata.

        Raises:
            DocumentLoadError: If the file cannot be loaded or parsed.
        """
        self._validate_file_path(file_path)  # Validate the file exists and is readable
        extension = file_path.suffix.lower()  # Get lowercase file extension
        self._validate_extension(extension)  # Check if extension is supported
        content = self._load_by_extension(file_path, extension)  # Load content using handler
        content_hash = self._compute_content_hash(content)  # Compute SHA-256 hash
        document_id = self._generate_document_id()  # Generate unique document ID
        metadata = self._build_metadata(file_path)  # Build metadata dictionary
        document = Document(  # Create the Document data object
            content=content,  # Set the extracted text content
            source_path=str(file_path),  # Store the source file path as string
            document_id=document_id,  # Assign the unique identifier
            metadata=metadata,  # Attach the metadata dictionary
            content_hash=content_hash,  # Store the content hash for deduplication
        )
        self._logger.info(  # Log successful document loading
            "Loaded document: %s (id=%s, chars=%d)",  # Format with key details
            file_path.name,  # Just the filename for readability
            document_id[:8],  # First 8 chars of UUID for brevity
            len(content),  # Character count of loaded content
        )
        return document  # Return the constructed Document object

    def load_directory(self, directory_path: Path, extensions: list[str]) -> list[Document]:
        """Load all supported files from a directory.

        Args:
            directory_path: Path to the directory to scan for documents.
            extensions: List of file extensions to include.

        Returns:
            List of Document objects loaded from the directory.

        Raises:
            DocumentLoadError: If the directory does not exist.
        """
        if not directory_path.exists():  # Check if directory exists
            raise DocumentLoadError(f"Directory not found: {directory_path}")  # Raise error
        if not directory_path.is_dir():  # Check if path is actually a directory
            raise DocumentLoadError(f"Path is not a directory: {directory_path}")  # Raise error
        documents = []  # Initialize list to collect loaded documents
        for extension in extensions:  # Iterate through each requested extension
            pattern = f"*{extension}"  # Build glob pattern for this extension
            matching_files = sorted(directory_path.glob(pattern))  # Find matching files
            for file_path in matching_files:  # Process each matching file
                try:  # Attempt to load each file individually
                    document = self.load_file(file_path)  # Load the file
                    documents.append(document)  # Add successfully loaded document
                except DocumentLoadError as error:  # Catch loading errors
                    self._logger.warning(  # Log warning but continue with other files
                        "Skipping file %s: %s", file_path.name, error  # Log filename and error
                    )
        self._logger.info(  # Log directory loading summary
            "Loaded %d documents from %s", len(documents), directory_path  # Summary message
        )
        return documents  # Return all successfully loaded documents

    def _validate_file_path(self, file_path: Path) -> None:
        """Validate that the file path exists and is readable.

        Args:
            file_path: Path to validate.

        Raises:
            DocumentLoadError: If the file does not exist or is not a file.
        """
        if not file_path.exists():  # Check if the file exists on disk
            raise DocumentLoadError(f"File not found: {file_path}")  # Raise with path info
        if not file_path.is_file():  # Check if path points to a regular file
            raise DocumentLoadError(f"Path is not a file: {file_path}")  # Raise with path info

    def _validate_extension(self, extension: str) -> None:
        """Validate that the file extension is in the supported list.

        Args:
            extension: Lowercase file extension including the dot.

        Raises:
            DocumentLoadError: If the extension is not supported.
        """
        if extension not in self._supported_extensions:  # Check against allowed list
            raise DocumentLoadError(  # Raise with details about supported formats
                f"Unsupported file extension: {extension}. "  # Show the rejected extension
                f"Supported: {self._supported_extensions}"  # Show valid extensions
            )

    def _load_by_extension(self, file_path: Path, extension: str) -> str:
        """Route to the appropriate loader method based on file extension.

        Args:
            file_path: Path to the file to load.
            extension: Lowercase file extension including the dot.

        Returns:
            Extracted text content from the file.
        """
        method_name = self.SUPPORTED_FORMATS.get(extension)  # Look up handler method name
        if method_name is None:  # If no handler registered for this extension
            raise DocumentLoadError(f"No loader for extension: {extension}")  # Raise error
        loader_method = getattr(self, method_name)  # Get the handler method reference
        return loader_method(file_path)  # Call the handler and return its result

    def _load_text_file(self, file_path: Path) -> str:
        """Load content from a plain text or markdown file.

        Args:
            file_path: Path to the text file to read.

        Returns:
            The full text content of the file.

        Raises:
            DocumentLoadError: If the file cannot be read or decoded.
        """
        try:  # Attempt to read the file with UTF-8 encoding
            content = file_path.read_text(encoding="utf-8")  # Read entire file as string
            return content  # Return the file content
        except UnicodeDecodeError as error:  # Handle encoding issues
            self._logger.error("Encoding error in %s: %s", file_path, error)  # Log the error
            raise DocumentLoadError(  # Wrap in domain exception
                f"Cannot decode file {file_path}: {error}"  # Include original error
            ) from error  # Chain the original exception

    def _load_pdf_file(self, file_path: Path) -> str:
        """Load text content from a PDF file using pypdf.

        Args:
            file_path: Path to the PDF file to read.

        Returns:
            Extracted text content from all pages concatenated.

        Raises:
            DocumentLoadError: If the PDF cannot be parsed.
        """
        try:  # Attempt to read and parse the PDF file
            from pypdf import PdfReader  # Import pypdf lazily (only when needed)

            reader = PdfReader(str(file_path))  # Create PDF reader instance
            pages_text = []  # Initialize list to collect text from each page
            for page in reader.pages:  # Iterate through all pages in the PDF
                page_text = page.extract_text()  # Extract text from current page
                if page_text:  # Only add non-empty page text
                    pages_text.append(page_text)  # Append page text to collection
            return "\n\n".join(pages_text)  # Join pages with double newline separator
        except Exception as error:  # Catch any PDF parsing errors
            self._logger.error("PDF load error for %s: %s", file_path, error)  # Log the error
            raise DocumentLoadError(  # Wrap in domain exception
                f"Cannot parse PDF {file_path}: {error}"  # Include original error
            ) from error  # Chain the original exception

    def _load_json_file(self, file_path: Path) -> str:
        """Load and extract text content from a JSON file.

        Expects JSON with a 'content' or 'text' field, or converts
        the entire JSON structure to a formatted string.

        Args:
            file_path: Path to the JSON file to read.

        Returns:
            Extracted text content from the JSON file.

        Raises:
            DocumentLoadError: If the JSON cannot be parsed.
        """
        try:  # Attempt to read and parse the JSON file
            raw_text = file_path.read_text(encoding="utf-8")  # Read the file as string
            data = json.loads(raw_text)  # Parse JSON string into Python object
            if isinstance(data, dict):  # If JSON root is an object
                # Try common text field names first
                for field in ("content", "text", "body"):  # Check common field names
                    if field in data:  # If field exists in the JSON object
                        return str(data[field])  # Return the text field value
            return json.dumps(data, indent=2)  # Fall back to formatted JSON string
        except (json.JSONDecodeError, UnicodeDecodeError) as error:  # Catch parse errors
            self._logger.error("JSON load error for %s: %s", file_path, error)  # Log error
            raise DocumentLoadError(  # Wrap in domain exception
                f"Cannot parse JSON {file_path}: {error}"  # Include original error
            ) from error  # Chain the original exception

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of the content for deduplication.

        Args:
            content: The text content to hash.

        Returns:
            Hexadecimal string of the SHA-256 hash.
        """
        content_bytes = content.encode("utf-8")  # Encode string to bytes for hashing
        hash_object = hashlib.sha256(content_bytes)  # Create SHA-256 hash object
        return hash_object.hexdigest()  # Return the hex digest string

    def _generate_document_id(self) -> str:
        """Generate a unique identifier for a document.

        Returns:
            UUID4 string for uniquely identifying this document.
        """
        return str(uuid.uuid4())  # Generate and return a random UUID as string

    def _build_metadata(self, file_path: Path) -> dict:
        """Build metadata dictionary from file properties.

        Args:
            file_path: Path to the source file.

        Returns:
            Dictionary containing file metadata.
        """
        stat = file_path.stat()  # Get file system statistics
        return {  # Build and return metadata dictionary
            "filename": file_path.name,  # Just the filename without directory
            "extension": file_path.suffix.lower(),  # Lowercase file extension
            "size_bytes": stat.st_size,  # File size in bytes
            "source_directory": str(file_path.parent),  # Parent directory path
        }
