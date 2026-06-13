# =============================================================================
# RAG Pipeline - Input Sanitizer
# =============================================================================
# Provides input validation and sanitization to prevent injection attacks.

import logging  # Import logging for security event tracking
import re  # Import regex for pattern matching on suspicious inputs
from pathlib import Path  # Import Path for safe file path handling

from rag_pipeline.utils.exceptions import SecurityViolationError  # Import security exception

logger = logging.getLogger(__name__)  # Create module-level logger instance


class InputSanitizer:
    """Sanitizes all user inputs to prevent injection and traversal attacks.

    Provides methods to validate and clean user-supplied queries and file paths
    before they are processed by the pipeline. Implements defense-in-depth by
    applying multiple validation layers.
    """

    # Patterns that may indicate prompt injection attempts
    INJECTION_PATTERNS = [  # List of regex patterns to detect prompt injection
        r"ignore\s+(previous|above|all)\s+instructions",  # Common injection prefix
        r"disregard\s+(previous|above|all)",  # Another injection prefix variant
        r"system\s*:\s*you\s+are",  # Attempt to override system prompt
        r"<\s*script\s*>",  # HTML script tag injection attempt
        r"javascript\s*:",  # JavaScript protocol injection
    ]

    def __init__(self, max_input_length: int, logger_instance: logging.Logger = logger) -> None:
        """Initialize the input sanitizer with configuration parameters.

        Args:
            max_input_length: Maximum allowed character count for any input.
            logger_instance: Logger to use for security event logging.
        """
        self._max_input_length = max_input_length  # Store maximum input length setting
        self._logger = logger_instance  # Store logger reference for security events
        # Compile injection patterns for efficient repeated matching
        self._compiled_patterns = [  # Pre-compile all regex patterns for performance
            re.compile(pattern, re.IGNORECASE)  # Case-insensitive compilation
            for pattern in self.INJECTION_PATTERNS  # Iterate through pattern list
        ]

    def sanitize_query(self, raw_input: str) -> str:
        """Sanitize a user query by applying all validation and cleaning steps.

        Args:
            raw_input: The raw user input string to sanitize.

        Returns:
            Cleaned and validated query string safe for pipeline processing.

        Raises:
            SecurityViolationError: If input fails security validation.
        """
        self._validate_not_empty(raw_input)  # Ensure input is not empty or whitespace
        cleaned = self._strip_control_characters(raw_input)  # Remove control characters
        cleaned = self._enforce_length_limit(cleaned)  # Truncate if exceeds max length
        self._check_prompt_injection(cleaned)  # Check for prompt injection patterns
        self._logger.debug("Query sanitized successfully, length=%d", len(cleaned))  # Log success
        return cleaned  # Return the sanitized query string

    def sanitize_file_path(self, file_path: str) -> Path:
        """Validate and sanitize a file path to prevent path traversal.

        Args:
            file_path: The user-supplied file path string to validate.

        Returns:
            Resolved Path object that has been validated as safe.

        Raises:
            SecurityViolationError: If path traversal is detected.
        """
        self._validate_not_empty(file_path)  # Ensure path string is not empty
        self._check_path_traversal(file_path)  # Check for directory traversal patterns
        resolved_path = Path(file_path).resolve()  # Resolve to absolute path (follows symlinks)
        self._logger.debug("File path sanitized: %s", resolved_path)  # Log the resolved path
        return resolved_path  # Return the validated and resolved path

    def _validate_not_empty(self, text: str) -> None:
        """Validate that input is not empty or whitespace-only.

        Args:
            text: The input string to validate.

        Raises:
            SecurityViolationError: If input is empty or whitespace-only.
        """
        if not text or not text.strip():  # Check for empty or whitespace-only input
            self._logger.warning("Empty input received")  # Log the empty input event
            raise SecurityViolationError("Input must not be empty or whitespace-only")

    def _strip_control_characters(self, text: str) -> str:
        """Remove non-printable control characters while preserving whitespace.

        Args:
            text: Input string potentially containing control characters.

        Returns:
            String with control characters removed.
        """
        # Keep printable chars and standard whitespace (newline, tab, space)
        cleaned = "".join(  # Join filtered characters back into string
            char for char in text  # Iterate through each character
            if char.isprintable() or char in ("\n", "\r", "\t")  # Keep printable + whitespace
        )
        return cleaned  # Return the cleaned string

    def _enforce_length_limit(self, text: str) -> str:
        """Truncate input if it exceeds the configured maximum length.

        Args:
            text: Input string to check against length limit.

        Returns:
            Original string or truncated version if over limit.
        """
        if len(text) > self._max_input_length:  # Check if text exceeds maximum
            self._logger.warning(  # Log the truncation event
                "Input truncated from %d to %d characters",  # Warning message format
                len(text),  # Original length
                self._max_input_length,  # Maximum allowed length
            )
            return text[: self._max_input_length]  # Truncate to maximum length
        return text  # Return original text if within limit

    def _check_path_traversal(self, path: str) -> None:
        """Detect path traversal attempts in file path strings.

        Args:
            path: File path string to check for traversal patterns.

        Raises:
            SecurityViolationError: If path traversal pattern is detected.
        """
        # Check for directory traversal indicators
        dangerous_patterns = ["..", "~", "%2e%2e", "%252e%252e"]  # Common traversal encodings
        path_lower = path.lower()  # Normalize to lowercase for comparison
        for pattern in dangerous_patterns:  # Check each dangerous pattern
            if pattern in path_lower:  # If pattern found in the path
                self._logger.error(  # Log the security violation
                    "Path traversal attempt detected: %s", path  # Include the offending path
                )
                raise SecurityViolationError(  # Raise security exception
                    f"Path traversal detected in: {path}"  # Include path in error message
                )

    def _check_prompt_injection(self, text: str) -> None:
        """Check text for common prompt injection patterns.

        Args:
            text: Input text to scan for injection patterns.

        Raises:
            SecurityViolationError: If a prompt injection pattern is detected.
        """
        for pattern in self._compiled_patterns:  # Check each compiled regex pattern
            if pattern.search(text):  # If pattern matches anywhere in the text
                self._logger.error(  # Log the security violation
                    "Prompt injection pattern detected in input"  # Don't log the actual text
                )
                raise SecurityViolationError(  # Raise security exception
                    "Input contains potentially malicious content"  # Generic message for user
                )
