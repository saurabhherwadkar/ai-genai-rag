# =============================================================================
# RAG Pipeline - Settings Loader Unit Tests
# =============================================================================
# Tests for the SettingsLoader class covering configuration loading.

from pathlib import Path  # Import Path for file operations

from rag_pipeline.config.settings_loader import SettingsLoader  # Class under test


class TestSettingsLoader:
    """Unit tests for the SettingsLoader class."""

    def test_load_returns_dictionary(self, tmp_config_dir: Path) -> None:
        """Test that load() returns a dictionary of settings."""
        loader = SettingsLoader(tmp_config_dir)  # Create loader with test config
        settings = loader.load()  # Load settings
        assert isinstance(settings, dict)  # Should return a dictionary
        assert len(settings) > 0  # Should not be empty

    def test_load_reads_application_section(self, tmp_config_dir: Path) -> None:
        """Test that application settings are loaded correctly."""
        loader = SettingsLoader(tmp_config_dir)  # Create loader
        settings = loader.load()  # Load settings
        assert "application" in settings  # Should have application section
        assert settings["application"]["name"] == "Test App"  # Should match YAML

    def test_get_retrieves_nested_values(self, tmp_config_dir: Path) -> None:
        """Test that get() navigates dot-notation paths correctly."""
        loader = SettingsLoader(tmp_config_dir)  # Create loader
        loader.load()  # Load settings first
        value = loader.get("application.name")  # Get nested value
        assert value == "Test App"  # Should return the nested value

    def test_get_returns_default_for_missing_key(self, tmp_config_dir: Path) -> None:
        """Test that get() returns default for non-existent paths."""
        loader = SettingsLoader(tmp_config_dir)  # Create loader
        loader.load()  # Load settings
        value = loader.get("nonexistent.key", default="fallback")  # Missing key
        assert value == "fallback"  # Should return the default

    def test_get_section_returns_section_dict(self, tmp_config_dir: Path) -> None:
        """Test that get_section returns the complete section dictionary."""
        loader = SettingsLoader(tmp_config_dir)  # Create loader
        loader.load()  # Load settings
        section = loader.get_section("application")  # Get entire section
        assert isinstance(section, dict)  # Should return a dictionary
        assert "name" in section  # Should have the name key

    def test_get_section_returns_none_for_missing(self, tmp_config_dir: Path) -> None:
        """Test that get_section returns None for non-existent sections."""
        loader = SettingsLoader(tmp_config_dir)  # Create loader
        loader.load()  # Load settings
        section = loader.get_section("nonexistent")  # Missing section
        assert section is None  # Should return None

    def test_merge_configs_override_takes_priority(self, tmp_path: Path) -> None:
        """Test that environment-specific config overrides base config."""
        config_dir = tmp_path / "config"  # Create config directory
        config_dir.mkdir()  # Create the directory
        base = "application:\n  name: Base\n  version: '1.0'\n"  # Base config
        override = "application:\n  name: Override\n"  # Override config
        (config_dir / "settings.yaml").write_text(base, encoding="utf-8")  # Write base
        (config_dir / "settings.development.yaml").write_text(  # Write dev override
            override, encoding="utf-8"
        )
        loader = SettingsLoader(config_dir)  # Create loader
        settings = loader.load()  # Load with merge
        assert settings["application"]["name"] == "Override"  # Override wins
        assert settings["application"]["version"] == "1.0"  # Base preserved

    def test_load_handles_missing_override_file(self, tmp_path: Path) -> None:
        """Test that missing environment override file doesn't cause errors."""
        config_dir = tmp_path / "config"  # Create config directory
        config_dir.mkdir()  # Create the directory
        base = "application:\n  name: OnlyBase\n  environment: production\n"  # Base only
        (config_dir / "settings.yaml").write_text(base, encoding="utf-8")  # Write base
        loader = SettingsLoader(config_dir)  # Create loader
        settings = loader.load()  # Load (no override file exists)
        assert settings["application"]["name"] == "OnlyBase"  # Should work with base only

    def test_load_handles_empty_yaml_file(self, tmp_path: Path) -> None:
        """Test that an empty YAML file doesn't cause errors."""
        config_dir = tmp_path / "config"  # Create config directory
        config_dir.mkdir()  # Create the directory
        (config_dir / "settings.yaml").write_text("", encoding="utf-8")  # Empty file
        loader = SettingsLoader(config_dir)  # Create loader
        settings = loader.load()  # Load empty config
        assert isinstance(settings, dict)  # Should return empty dict
