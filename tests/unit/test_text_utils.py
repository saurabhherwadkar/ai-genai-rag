# =============================================================================
# RAG Pipeline - Text Utils Unit Tests
# =============================================================================
# Tests for the TextUtils utility class.


from rag_pipeline.utils.text_utils import TextUtils  # Class under test


class TestTextUtils:
    """Unit tests for the TextUtils class."""

    def test_normalize_whitespace_collapses_spaces(self) -> None:
        """Test that multiple spaces are collapsed to single space."""
        result = TextUtils.normalize_whitespace("Hello   World   Test")  # Multiple spaces
        assert result == "Hello World Test"  # Should be single spaces

    def test_normalize_whitespace_collapses_mixed(self) -> None:
        """Test that mixed whitespace types are collapsed."""
        result = TextUtils.normalize_whitespace("Hello\t\n  World")  # Mixed whitespace
        assert result == "Hello World"  # Should be single space

    def test_normalize_whitespace_strips_edges(self) -> None:
        """Test that leading and trailing whitespace is removed."""
        result = TextUtils.normalize_whitespace("  Hello  ")  # Padded
        assert result == "Hello"  # Should be stripped

    def test_remove_control_characters_removes_bell(self) -> None:
        """Test that control characters like bell are removed."""
        result = TextUtils.remove_control_characters("Hello\x07World")  # Bell char
        assert "\x07" not in result  # Bell should be removed

    def test_remove_control_characters_preserves_newlines(self) -> None:
        """Test that standard whitespace characters are preserved."""
        result = TextUtils.remove_control_characters("Hello\nWorld\tTest")  # Newline, tab
        assert "\n" in result  # Newline should be preserved
        assert "\t" in result  # Tab should be preserved

    def test_normalize_unicode_consistency(self) -> None:
        """Test that Unicode normalization produces consistent output."""
        # e with acute accent: composed vs decomposed forms
        text1 = "é"  # e-acute as single character (NFC)
        text2 = "é"  # e + combining acute accent (NFD)
        norm1 = TextUtils.normalize_unicode(text1)  # Normalize first
        norm2 = TextUtils.normalize_unicode(text2)  # Normalize second
        assert norm1 == norm2  # Both should produce same NFC form

    def test_truncate_text_within_limit(self) -> None:
        """Test that text within limit is returned unchanged."""
        result = TextUtils.truncate_text("Short text", max_length=50)  # Within limit
        assert result == "Short text"  # Should be unchanged

    def test_truncate_text_exceeding_limit(self) -> None:
        """Test that text exceeding limit is truncated with suffix."""
        result = TextUtils.truncate_text("This is a long text", max_length=10)  # Exceeds
        assert len(result) == 10  # Should be exactly max_length
        assert result.endswith("...")  # Should end with default suffix

    def test_truncate_text_custom_suffix(self) -> None:
        """Test truncation with a custom suffix."""
        result = TextUtils.truncate_text("Long text here", max_length=10, suffix="~")
        assert result.endswith("~")  # Should use custom suffix
        assert len(result) == 10  # Should be exactly max_length

    def test_count_tokens_approximate_normal_text(self) -> None:
        """Test approximate token count for normal text."""
        result = TextUtils.count_tokens_approximate("Hello World Test")  # 3 words
        assert result == 3  # Should count 3 tokens

    def test_count_tokens_approximate_empty(self) -> None:
        """Test token count for empty text."""
        result = TextUtils.count_tokens_approximate("")  # Empty string
        assert result == 0  # Should be zero

    def test_count_tokens_approximate_whitespace_only(self) -> None:
        """Test token count for whitespace-only text."""
        result = TextUtils.count_tokens_approximate("   \t\n   ")  # Whitespace only
        assert result == 0  # Should be zero

    def test_extract_sentences_basic(self) -> None:
        """Test sentence extraction from text with standard punctuation."""
        text = "First sentence. Second sentence. Third sentence."  # Three sentences
        result = TextUtils.extract_sentences(text)  # Extract sentences
        assert len(result) == 3  # Should find 3 sentences

    def test_extract_sentences_mixed_punctuation(self) -> None:
        """Test sentence extraction with mixed end punctuation."""
        text = "Question? Exclamation! Statement."  # Mixed punctuation
        result = TextUtils.extract_sentences(text)  # Extract sentences
        assert len(result) == 3  # Should find 3 sentences

    def test_extract_sentences_single_sentence(self) -> None:
        """Test extraction from text with only one sentence."""
        text = "Just one sentence here."  # Single sentence
        result = TextUtils.extract_sentences(text)  # Extract
        assert len(result) == 1  # Should find 1 sentence
