# =============================================================================
# RAG Pipeline - Text Utility Functions
# =============================================================================
# Provides reusable text processing helpers used across the pipeline.

import re  # Import regex module for pattern-based text manipulation
import unicodedata  # Import unicodedata for Unicode normalization


class TextUtils:
    """Collection of static text processing utility methods.

    All methods are stateless and operate purely on input strings,
    making them easy to test and reuse across pipeline components.
    """

    @staticmethod  # No instance state needed for this method
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple whitespace characters into single spaces.

        Args:
            text: Input string that may contain excessive whitespace.

        Returns:
            String with all runs of whitespace replaced by a single space.
        """
        cleaned = re.sub(r"\s+", " ", text)  # Replace all whitespace runs with single space
        return cleaned.strip()  # Remove leading and trailing whitespace

    @staticmethod  # No instance state needed for this method
    def remove_control_characters(text: str) -> str:
        """Remove non-printable control characters from text.

        Preserves standard whitespace (space, tab, newline) while
        removing potentially harmful control characters.

        Args:
            text: Input string that may contain control characters.

        Returns:
            String with control characters removed.
        """
        allowed_categories = {"Cc"}  # Unicode category for control characters
        result_chars = []  # Initialize list to build cleaned string
        for char in text:  # Iterate through each character in the input
            category = unicodedata.category(char)  # Get Unicode category of character
            if category not in allowed_categories:  # Keep non-control characters
                result_chars.append(char)  # Add character to result
            elif char in ("\n", "\r", "\t", " "):  # Preserve standard whitespace
                result_chars.append(char)  # Add whitespace character to result
        return "".join(result_chars)  # Join characters back into a string

    @staticmethod  # No instance state needed for this method
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode text to NFC (Canonical Decomposition + Composition).

        Ensures consistent representation of characters that can be
        encoded in multiple ways (e.g., accented characters).

        Args:
            text: Input string with potentially inconsistent Unicode encoding.

        Returns:
            NFC-normalized string.
        """
        return unicodedata.normalize("NFC", text)  # Apply NFC normalization

    @staticmethod  # No instance state needed for this method
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to a maximum length with a suffix indicator.

        Args:
            text: Input string to potentially truncate.
            max_length: Maximum allowed length of the output string.
            suffix: String to append when truncation occurs.

        Returns:
            Original string if within limit, otherwise truncated with suffix.
        """
        if len(text) <= max_length:  # Check if text is already within limit
            return text  # Return original text unchanged
        truncation_point = max_length - len(suffix)  # Calculate where to cut
        return text[:truncation_point] + suffix  # Return truncated text with suffix

    @staticmethod  # No instance state needed for this method
    def count_tokens_approximate(text: str) -> int:
        """Provide an approximate token count based on whitespace splitting.

        This is a rough estimate; actual tokenization depends on the model.
        Useful for quick size checks without loading a tokenizer.

        Args:
            text: Input string to estimate token count for.

        Returns:
            Approximate number of tokens (words) in the text.
        """
        if not text or not text.strip():  # Check for empty or whitespace-only input
            return 0  # Return zero for empty text
        return len(text.split())  # Split on whitespace and count resulting tokens

    @staticmethod  # No instance state needed for this method
    def extract_sentences(text: str) -> list[str]:
        """Split text into sentences using basic punctuation rules.

        Uses a simple regex-based approach that handles common sentence
        endings (period, exclamation, question mark).

        Args:
            text: Input text to split into sentences.

        Returns:
            List of sentence strings.
        """
        pattern = r"(?<=[.!?])\s+"  # Match whitespace after sentence-ending punctuation
        sentences = re.split(pattern, text)  # Split text at sentence boundaries
        return [s.strip() for s in sentences if s.strip()]  # Filter empty strings
