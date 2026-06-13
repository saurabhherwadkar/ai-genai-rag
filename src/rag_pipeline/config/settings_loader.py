# =============================================================================
# RAG Pipeline - Settings Loader
# =============================================================================
# Loads hierarchical configuration from YAML files and environment variables.

import copy  # Import copy for deep merging of configuration dictionaries
import logging  # Import logging for configuration load status messages
import os  # Import os for environment variable access
from pathlib import Path  # Import Path for cross-platform file path handling
from typing import Any  # Import type hints for method signatures

import yaml  # Import PyYAML for parsing YAML configuration files
from dotenv import load_dotenv  # Import dotenv to load .env files into environment

logger = logging.getLogger(__name__)  # Create module-level logger instance


class SettingsLoader:
    """Loads hierarchical configuration from YAML files and environment variables.

    Priority order (highest priority wins):
    1. Environment variables (prefixed with RAG_)
    2. Environment-specific YAML (e.g., settings.dev.yaml)
    3. Base YAML (settings.yaml)

    This allows deployment-specific overrides without modifying code.
    """

    ENV_PREFIX = "RAG_"  # Prefix for environment variable overrides

    def __init__(self, config_dir: Path) -> None:
        """Initialize the settings loader with the configuration directory path.

        Args:
            config_dir: Path to the directory containing YAML config files.
        """
        self._config_dir = config_dir  # Store the configuration directory path
        self._settings: dict = {}  # Initialize empty settings dictionary
        self._load_env_file()  # Load .env file if it exists in project root

    def _load_env_file(self) -> None:
        """Load environment variables from .env file if it exists."""
        env_file = self._config_dir.parent / ".env"  # Look for .env in project root
        if env_file.exists():  # Check if the .env file is present
            load_dotenv(env_file)  # Parse and load variables into os.environ
            logger.debug("Loaded environment variables from %s", env_file)  # Log success

    def load(self) -> dict:
        """Load and merge all configuration sources into a single dictionary.

        Returns:
            Merged configuration dictionary with all overrides applied.
        """
        base_config = self._load_yaml_file("settings.yaml")  # Load base configuration
        environment = self._determine_environment(base_config)  # Determine active environment
        env_config = self._load_yaml_file(f"settings.{environment}.yaml")  # Load env overrides
        merged_config = self._merge_configs(base_config, env_config)  # Merge env over base
        final_config = self._apply_environment_overrides(merged_config)  # Apply env var overrides
        self._settings = final_config  # Store the final merged configuration
        logger.info("Configuration loaded for environment: %s", environment)  # Log completion
        return self._settings  # Return the complete configuration dictionary

    def _determine_environment(self, config: dict) -> str:
        """Determine the active environment from env var or base config.

        Args:
            config: The base configuration dictionary.

        Returns:
            Environment name string (e.g., 'development', 'production').
        """
        env_var = os.environ.get(f"{self.ENV_PREFIX}ENVIRONMENT")  # Check env var first
        if env_var:  # If environment variable is set, use it
            return env_var.lower()  # Normalize to lowercase
        app_config = config.get("application", {})  # Get application section from config
        return app_config.get("environment", "development")  # Default to development

    def _load_yaml_file(self, filename: str) -> dict:
        """Load a single YAML file from the configuration directory.

        Args:
            filename: Name of the YAML file to load.

        Returns:
            Parsed dictionary from the YAML file, or empty dict if not found.
        """
        file_path = self._config_dir / filename  # Construct full path to YAML file
        if not file_path.exists():  # Check if the file exists on disk
            logger.debug("Config file not found: %s", file_path)  # Log missing file
            return {}  # Return empty dict for missing optional config files
        try:  # Attempt to read and parse the YAML file
            with open(file_path, encoding="utf-8") as yaml_file:  # Open with UTF-8 encoding
                content = yaml.safe_load(yaml_file)  # Parse YAML safely (no code execution)
                return content if content is not None else {}  # Handle empty YAML files
        except yaml.YAMLError as error:  # Catch YAML parsing errors
            logger.error("Failed to parse YAML file %s: %s", file_path, error)  # Log the error
            raise  # Re-raise to signal configuration failure

    def _merge_configs(self, base: dict, override: dict) -> dict:
        """Deep merge two configuration dictionaries, with override taking priority.

        Args:
            base: The base configuration dictionary.
            override: The override dictionary whose values take priority.

        Returns:
            New dictionary with base values overridden where specified.
        """
        merged = copy.deepcopy(base)  # Create a deep copy to avoid mutating base
        for key, value in override.items():  # Iterate through override entries
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # If both values are dicts, recursively merge them
                merged[key] = self._merge_configs(merged[key], value)
            else:  # Otherwise, override value replaces base value entirely
                merged[key] = copy.deepcopy(value)  # Deep copy to prevent shared references
        return merged  # Return the merged configuration dictionary

    def _apply_environment_overrides(self, config: dict) -> dict:
        """Apply environment variable overrides to the configuration.

        Environment variables prefixed with RAG_ override nested config values.
        For example, RAG_LOG_LEVEL overrides the log level setting.

        Args:
            config: The merged configuration dictionary.

        Returns:
            Configuration with environment variable overrides applied.
        """
        log_level = os.environ.get(f"{self.ENV_PREFIX}LOG_LEVEL")  # Check for log level override
        if log_level:  # If log level env var is set
            config.setdefault("logging", {})["level"] = log_level  # Apply log level override
        persist_dir = os.environ.get(f"{self.ENV_PREFIX}PERSIST_DIRECTORY")  # Check persist dir
        if persist_dir:  # If persistence directory env var is set
            config.setdefault("vectorstore", {})["persist_directory"] = persist_dir  # Apply it
        return config  # Return config with environment overrides applied

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a configuration value using dot-notation path.

        Args:
            dotted_key: Dot-separated path (e.g., 'ingestion.chunking.chunk_size').
            default: Value to return if the key path is not found.

        Returns:
            The configuration value at the specified path, or the default.
        """
        keys = dotted_key.split(".")  # Split the dotted path into individual keys
        value: Any = self._settings  # Start from the root of the settings dictionary
        for key in keys:  # Navigate through each level of the path
            if isinstance(value, dict) and key in value:  # Check if key exists at this level
                value = value[key]  # Move deeper into the nested dictionary
            else:  # Key not found at this level
                return default  # Return the default value
        return value  # Return the found value

    def get_section(self, section_name: str) -> dict | None:
        """Retrieve an entire configuration section by its top-level key.

        Args:
            section_name: The top-level key name (e.g., 'ingestion', 'retrieval').

        Returns:
            The configuration section dictionary, or None if not found.
        """
        return self._settings.get(section_name)  # Return the section or None
