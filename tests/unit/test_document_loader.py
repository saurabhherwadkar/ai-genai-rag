# =============================================================================
# RAG Pipeline - Document Loader Unit Tests
# =============================================================================
# Tests for the DocumentLoader class covering all loading scenarios.

import json  # Import json for creating test JSON files
from pathlib import Path  # Import Path for file operations

import pytest  # Import pytest for test decorators and assertions

from rag_pipeline.ingestion.document_loader import DocumentLoader  # Class under test
from rag_pipeline.utils.exceptions import DocumentLoadError  # Expected exception


class TestDocumentLoader:
    """Unit tests for the DocumentLoader class."""

    @pytest.fixture  # Fixture for creating a DocumentLoader instance
    def loader(self) -> DocumentLoader:
        """Create a DocumentLoader with standard test extensions."""
        return DocumentLoader(supported_extensions=[".txt", ".md", ".json"])

    def test_load_text_file_returns_document(
        self, loader: DocumentLoader, tmp_text_file: Path
    ) -> None:
        """Test that loading a text file returns a valid Document object."""
        document = loader.load_file(tmp_text_file)  # Load the test file
        assert document.content is not None  # Content should not be None
        assert len(document.content) > 0  # Content should not be empty
        assert document.source_path == str(tmp_text_file)  # Source path should match
        assert document.document_id is not None  # Should have generated an ID
        assert document.content_hash != ""  # Should have computed a hash

    def test_load_file_generates_unique_ids(
        self, loader: DocumentLoader, tmp_path: Path
    ) -> None:
        """Test that each loaded document receives a unique ID."""
        file1 = tmp_path / "doc1.txt"  # Create first test file
        file2 = tmp_path / "doc2.txt"  # Create second test file
        file1.write_text("Content one", encoding="utf-8")  # Write content
        file2.write_text("Content two", encoding="utf-8")  # Write content
        doc1 = loader.load_file(file1)  # Load first document
        doc2 = loader.load_file(file2)  # Load second document
        assert doc1.document_id != doc2.document_id  # IDs should be unique

    def test_load_file_computes_content_hash(
        self, loader: DocumentLoader, tmp_path: Path
    ) -> None:
        """Test that content hash is correctly computed for deduplication."""
        file1 = tmp_path / "same1.txt"  # First file with same content
        file2 = tmp_path / "same2.txt"  # Second file with same content
        file1.write_text("Identical content", encoding="utf-8")  # Same content
        file2.write_text("Identical content", encoding="utf-8")  # Same content
        doc1 = loader.load_file(file1)  # Load first
        doc2 = loader.load_file(file2)  # Load second
        assert doc1.content_hash == doc2.content_hash  # Hashes should match

    def test_load_file_raises_for_missing_file(self, loader: DocumentLoader) -> None:
        """Test that loading a nonexistent file raises DocumentLoadError."""
        fake_path = Path("/nonexistent/file.txt")  # Path that does not exist
        with pytest.raises(DocumentLoadError):  # Expect DocumentLoadError
            loader.load_file(fake_path)  # Attempt to load

    def test_load_file_raises_for_unsupported_extension(
        self, loader: DocumentLoader, tmp_path: Path
    ) -> None:
        """Test that unsupported file extensions raise DocumentLoadError."""
        file = tmp_path / "test.xyz"  # Create file with unsupported extension
        file.write_text("content", encoding="utf-8")  # Write some content
        with pytest.raises(DocumentLoadError):  # Expect DocumentLoadError
            loader.load_file(file)  # Attempt to load

    def test_load_json_file_extracts_content_field(
        self, loader: DocumentLoader, tmp_path: Path
    ) -> None:
        """Test that JSON files with 'content' field extract correctly."""
        data = {"content": "Extracted text content", "title": "Test"}  # JSON with content
        file = tmp_path / "test.json"  # Create JSON file
        file.write_text(json.dumps(data), encoding="utf-8")  # Write JSON
        document = loader.load_file(file)  # Load the JSON file
        assert document.content == "Extracted text content"  # Should extract content field

    def test_load_json_file_extracts_text_field(
        self, loader: DocumentLoader, tmp_path: Path
    ) -> None:
        """Test that JSON files with 'text' field extract correctly."""
        data = {"text": "Text field value", "metadata": {}}  # JSON with text field
        file = tmp_path / "test.json"  # Create JSON file
        file.write_text(json.dumps(data), encoding="utf-8")  # Write JSON
        document = loader.load_file(file)  # Load the JSON file
        assert document.content == "Text field value"  # Should extract text field

    def test_load_directory_loads_all_matching_files(
        self, loader: DocumentLoader, tmp_path: Path
    ) -> None:
        """Test that directory loading finds all files with matching extensions."""
        (tmp_path / "doc1.txt").write_text("Doc 1", encoding="utf-8")  # Create file 1
        (tmp_path / "doc2.txt").write_text("Doc 2", encoding="utf-8")  # Create file 2
        (tmp_path / "doc3.md").write_text("Doc 3", encoding="utf-8")  # Create file 3
        (tmp_path / "ignore.xyz").write_text("Ignored", encoding="utf-8")  # Unsupported
        documents = loader.load_directory(tmp_path, [".txt", ".md"])  # Load directory
        assert len(documents) == 3  # Should load 3 matching files

    def test_load_directory_raises_for_missing_directory(
        self, loader: DocumentLoader
    ) -> None:
        """Test that loading from nonexistent directory raises DocumentLoadError."""
        fake_dir = Path("/nonexistent/directory")  # Path that does not exist
        with pytest.raises(DocumentLoadError):  # Expect DocumentLoadError
            loader.load_directory(fake_dir, [".txt"])  # Attempt to load

    def test_load_file_builds_metadata(
        self, loader: DocumentLoader, tmp_text_file: Path
    ) -> None:
        """Test that loaded documents have correct metadata fields."""
        document = loader.load_file(tmp_text_file)  # Load the test file
        assert "filename" in document.metadata  # Should have filename
        assert "extension" in document.metadata  # Should have extension
        assert "size_bytes" in document.metadata  # Should have file size
        assert document.metadata["extension"] == ".txt"  # Should be .txt
