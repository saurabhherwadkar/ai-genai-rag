# =============================================================================
# RAG Pipeline - Secrets Manager
# =============================================================================
# Manages secure access to sensitive configuration values from environment.

import logging  # Import logging for secret access tracking
import os  # Import os for environment variable access
from pathlib import Path  # Import Path for .env file path handling

from dotenv import load_dotenv  # Import dotenv to load .env files

from rag_pipeline.utils.exceptions import ConfigurationError  # Import config exception

logger = logging.getLogger(__name__)  # Create module-level logger instance


class SecretsManager:
    """Manages secure loading and access to sensitive configuration values.

    Loads secrets exclusively from environment variables or .env files.
    Never logs, prints, or exposes actual secret values.
    """

    MASK_CHAR = "*"  # Character used to mask secrets in log output
    VISIBLE_CHARS = 4  # Number of characters to show when masking a secret

    def __init__(self, env_file_path: Path | None = None) -> None:
        """Initialize the secrets manager, optionally loading a .env file.

        Args:
            env_file_path: Optional path to a .env file to load secrets from.
        """
        self._env_file_path = env_file_path  # Store the .env file path reference
        if env_file_path and env_file_path.exists():  # Check if .env file exists
            load_dotenv(env_file_path)  # Load the .env file into environment
            logger.info("Secrets loaded from env file")  # Log without revealing path

    def get_secret(self, key: str) -> str | None:
        """Retrieve a secret value from environment variables.

        Args:
            key: The environment variable name to look up.

        Returns:
            The secret value if found, or None if not set.
        """
        value = os.environ.get(key)  # Attempt to get the environment variable
        if value is not None:  # If the secret was found in environment
            logger.debug(  # Log access without revealing the value
                "Secret accessed: %s = %s",  # Format with masked value
                key,  # The key name (safe to log)
                self._mask_secret(value),  # Masked version of the value
            )
        return value  # Return the actual value (or None if not found)

    def validate_required_secrets(self, required_keys: list[str]) -> None:
        """Validate that all required secrets are present in the environment.

        Args:
            required_keys: List of environment variable names that must be set.

        Raises:
            ConfigurationError: If any required secret is missing.
        """
        missing_keys = []  # Initialize list to track missing secrets
        for key in required_keys:  # Check each required key
            if os.environ.get(key) is None:  # If key is not in environment
                missing_keys.append(key)  # Add to missing list
        if missing_keys:  # If any required secrets are missing
            logger.error(  # Log the missing keys (names only, never values)
                "Missing required secrets: %s",  # Error message format
                ", ".join(missing_keys),  # Join missing key names
            )
            raise ConfigurationError(  # Raise configuration error
                f"Missing required environment variables: {', '.join(missing_keys)}"
            )

    def _mask_secret(self, value: str) -> str:
        """Mask a secret value for safe logging, showing only last few chars.

        Args:
            value: The secret value to mask.

        Returns:
            Masked string showing only the last few characters.
        """
        if len(value) <= self.VISIBLE_CHARS:  # If value is very short
            return self.MASK_CHAR * len(value)  # Mask the entire value
        # Show only the last VISIBLE_CHARS characters, mask the rest
        masked_length = len(value) - self.VISIBLE_CHARS  # Calculate mask length
        return (self.MASK_CHAR * masked_length) + value[-self.VISIBLE_CHARS:]  # Build masked string
