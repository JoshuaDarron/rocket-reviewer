"""Tests for configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import AGENT_ROUTING, BOT_USERNAMES, MODELS, load_config
from src.errors import ConfigurationError


class TestConstants:
    """Tests for project-wide constants."""

    def test_all_models_defined(self) -> None:
        assert "claude-reviewer" in MODELS
        assert "gpt-reviewer" in MODELS
        assert "gemini-reviewer" in MODELS
        assert "aggregator" in MODELS

    def test_agent_routing_maps_all_bots(self) -> None:
        assert len(AGENT_ROUTING) == 3
        assert "claude-reviewer[bot]" in AGENT_ROUTING
        assert "gpt-reviewer[bot]" in AGENT_ROUTING
        assert "gemini-reviewer[bot]" in AGENT_ROUTING

    def test_bot_usernames_matches_routing_keys(self) -> None:
        assert set(AGENT_ROUTING.keys()) == BOT_USERNAMES

    def test_routing_values_match_model_keys(self) -> None:
        for node_id in AGENT_ROUTING.values():
            assert node_id in MODELS


class TestLoadConfig:
    """Tests for load_config()."""

    def test_load_defaults(self, tmp_path: Path) -> None:
        """Loading with no YAML and no env vars gives defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config(tmp_path)
        assert config.review_context == "full"
        assert config.target_branch == "main"
        assert config.approval_threshold == "high"
        assert config.max_chunk_lines == 500

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """YAML values override defaults."""
        yaml_content = (
            "target_branch: develop\n" "review_context: diff\n" "max_chunk_lines: 300\n"
        )
        config_file = tmp_path / ".rocketride-review.yml"
        config_file.write_text(yaml_content)

        with patch.dict(os.environ, {}, clear=True):
            config = load_config(tmp_path)
        assert config.target_branch == "develop"
        assert config.review_context == "diff"
        assert config.max_chunk_lines == 300

    def test_env_vars_override_yaml(self, tmp_path: Path) -> None:
        """Environment variables take precedence over YAML."""
        yaml_content = "target_branch: develop\n"
        config_file = tmp_path / ".rocketride-review.yml"
        config_file.write_text(yaml_content)

        env = {"INPUT_TARGET_BRANCH": "staging"}
        with patch.dict(os.environ, env, clear=True):
            config = load_config(tmp_path)
        assert config.target_branch == "staging"

    def test_env_review_context(self, tmp_path: Path) -> None:
        """INPUT_REVIEW_CONTEXT env var is applied."""
        env = {"INPUT_REVIEW_CONTEXT": "diff"}
        with patch.dict(os.environ, env, clear=True):
            config = load_config(tmp_path)
        assert config.review_context == "diff"

    def test_invalid_yaml_raises_configuration_error(self, tmp_path: Path) -> None:
        """Malformed YAML raises ConfigurationError."""
        config_file = tmp_path / ".rocketride-review.yml"
        config_file.write_text("{{{{invalid yaml::::")

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ConfigurationError, match="Failed to parse"),
        ):
            load_config(tmp_path)

    def test_yaml_not_mapping_raises_configuration_error(self, tmp_path: Path) -> None:
        """YAML that parses to a list raises ConfigurationError."""
        config_file = tmp_path / ".rocketride-review.yml"
        config_file.write_text("- item1\n- item2\n")

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ConfigurationError, match="YAML mapping"),
        ):
            load_config(tmp_path)

    def test_invalid_value_raises_configuration_error(self, tmp_path: Path) -> None:
        """Invalid config values raise ConfigurationError."""
        yaml_content = "max_chunk_lines: 5\n"  # below ge=50
        config_file = tmp_path / ".rocketride-review.yml"
        config_file.write_text(yaml_content)

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ConfigurationError, match="Invalid configuration"),
        ):
            load_config(tmp_path)

    def test_custom_config_path(self, tmp_path: Path) -> None:
        """Custom config path via INPUT_CONFIG_PATH is honored."""
        custom_file = tmp_path / "custom-review.yml"
        custom_file.write_text("target_branch: custom\n")

        env = {"INPUT_CONFIG_PATH": "custom-review.yml"}
        with patch.dict(os.environ, env, clear=True):
            config = load_config(tmp_path)
        assert config.target_branch == "custom"

    def test_missing_yaml_file_uses_defaults(self, tmp_path: Path) -> None:
        """Missing YAML file silently uses defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_config(tmp_path)
        assert config.target_branch == "main"

    def test_yaml_with_extra_patterns(self, tmp_path: Path) -> None:
        """YAML can set ignore_patterns_extra."""
        yaml_content = 'ignore_patterns_extra: ["*.sql", "migrations/**"]\n'
        config_file = tmp_path / ".rocketride-review.yml"
        config_file.write_text(yaml_content)

        with patch.dict(os.environ, {}, clear=True):
            config = load_config(tmp_path)
        assert "*.sql" in config.ignore_patterns_extra
        assert "migrations/**" in config.ignore_patterns_extra
