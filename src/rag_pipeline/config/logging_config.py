# =============================================================================
# RAG Pipeline - Logging Configurator
# =============================================================================
# Initializes Python logging from a YAML configuration file.

import logging  # Import logging module for logger initialization
import logging.config  # Import logging.config for dictConfig-based setup
from pathlib import Path  # Import Path for cross-platform file path handling

import yaml  # Import PyYAML for parsing the logging configuration file


class LoggingConfigurator:
    """Initializes the logging system from a YAML configuration file.

    Reads the logging.yaml file and applies it via Python's dictConfig.
    Also ensures the log output directory exists before handlers try to write.
    """

    def __init__(self, config_path: Path) -> None:
        """Initialize the logging configurator with the path to logging.yaml.

        Args:
            config_path: Full path to the logging YAML configuration file.
        """
        self._config_path = config_path  # Store the path to the logging config file

    def configure(self) -> None:
        """Load the logging YAML and apply it to Python's logging system."""
        config = self._load_config()  # Load and parse the YAML configuration
        self._ensure_log_directory(config)  # Create log directories if needed
        self._apply_config(config)  # Apply the configuration to Python logging

    def _load_config(self) -> dict:
        """Load the logging configuration from the YAML file.

        Returns:
            Parsed logging configuration dictionary.

        Raises:
            FileNotFoundError: If the logging config file does not exist.
        """
        if not self._config_path.exists():  # Check if config file exists
            self._apply_default_config()  # Fall back to basic configuration
            return {}  # Return empty dict to signal defaults were used
        with open(self._config_path, encoding="utf-8") as config_file:  # Open the YAML file
            config = yaml.safe_load(config_file)  # Parse the YAML content safely
        return config if config is not None else {}  # Handle empty file case

    def _ensure_log_directory(self, config: dict) -> None:
        """Create log output directories if they do not exist.

        Inspects handler configurations for file paths and creates
        their parent directories to prevent FileNotFoundError at runtime.

        Args:
            config: The parsed logging configuration dictionary.
        """
        handlers = config.get("handlers", {})  # Get all configured handlers
        for handler_config in handlers.values():  # Iterate through each handler
            filename = handler_config.get("filename")  # Check for filename setting
            if filename:  # If this handler writes to a file
                log_dir = Path(filename).parent  # Get the directory portion of the path
                log_dir.mkdir(parents=True, exist_ok=True)  # Create directory tree if needed

    def _apply_config(self, config: dict) -> None:
        """Apply the parsed configuration to Python's logging system.

        Args:
            config: The parsed logging configuration dictionary.
        """
        if not config:  # If config is empty (file not found or empty)
            return  # Default config was already applied
        logging.config.dictConfig(config)  # Apply the full configuration via dictConfig

    def _apply_default_config(self) -> None:
        """Apply a minimal default logging configuration as fallback.

        Used when the logging.yaml file is not found.
        """
        logging.basicConfig(  # Configure basic logging to console
            level=logging.INFO,  # Set default level to INFO
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",  # Simple format
            datefmt="%Y-%m-%d %H:%M:%S",  # Standard date format
        )
        logging.getLogger(__name__).warning(  # Warn about missing config file
            "Logging config not found at %s, using defaults",  # Warning message text
            self._config_path,  # Include the expected path in the warning
        )
