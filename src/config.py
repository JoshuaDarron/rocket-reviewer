"""Configuration loading and project-wide constants.

Loads configuration from environment variables (action inputs) and an
optional .rocketride-review.yml file from the repository root. All model
names, bot usernames, and agent routing are defined here as constants.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.errors import ConfigurationError
from src.models import ReviewConfig

logger = logging.getLogger(__name__)

# Model identifiers (not user-configurable in v1)
MODELS: dict[str, str] = {
    "claude-reviewer": "claude-sonnet-4-6",
    "gpt-reviewer": "openai-5-2",
    "gemini-reviewer": "gemini-3-pro",
    "aggregator": "claude-sonnet-4-6",
}

# Pipeline file paths
FULL_REVIEW_PIPELINE_FILE: str = "full-review.pipe.json"

# Maps agent node IDs to per-agent conversation reply pipeline filenames
CONVERSATION_PIPELINE_FILES: dict[str, str] = {
    "claude-reviewer": "conversation-reply-claude.pipe.json",
    "gpt-reviewer": "conversation-reply-openai.pipe.json",
    "gemini-reviewer": "conversation-reply-gemini.pipe.json",
}

# Maps response lane names to reviewer node IDs
LANE_TO_REVIEWER: dict[str, str] = {
    "claude": "claude-reviewer",
    "openai": "gpt-reviewer",
    "gemini": "gemini-reviewer",
}

# Maps GitHub App bot username to RocketRide pipeline node ID
AGENT_ROUTING: dict[str, str] = {
    "claude-reviewer[bot]": "claude-reviewer",
    "gpt-reviewer[bot]": "gpt-reviewer",
    "gemini-reviewer[bot]": "gemini-reviewer",
}

# All bot usernames (for loop prevention)
BOT_USERNAMES: set[str] = set(AGENT_ROUTING.keys())

# Default file ignore patterns
DEFAULT_IGNORE_PATTERNS: list[str] = [
    "*.lock",
    "*.min.js",
    "*.min.css",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "*.generated.*",
    "*.g.dart",
    "*.pb.go",
    "dist/**",
    "build/**",
    "vendor/**",
    "node_modules/**",
    "*.svg",
    "*.png",
    "*.jpg",
    "*.gif",
    "*.ico",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
]

# RocketRide server binary settings
ENGINE_VERSION: str = "v3.1.0"
ENGINE_DOWNLOAD_URL: str = (
    "https://github.com/rocketride-org/rocketride-server/releases/download/"
    f"server-{ENGINE_VERSION}-prerelease/"
    f"rocketride-server-{ENGINE_VERSION}-linux-x64.tar.gz"
)
ENGINE_BINARY_DIR: str = "/tmp/rocketride-server"

# Engine health check settings
ENGINE_HEALTH_CHECK_INTERVAL: float = 2.0
ENGINE_HEALTH_CHECK_TIMEOUT: float = 30.0
ENGINE_PORT: int = 5565

# Retry settings
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE: float = 1.0

# Maps LLM provider names (as used in pipeline JSON) to env var names for API keys
LLM_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "llm_anthropic": "INPUT_ANTHROPIC_API_KEY",
    "llm_openai": "INPUT_OPENAI_API_KEY",
    "llm_gemini": "INPUT_GOOGLE_API_KEY",
}

# Agent credentials: maps agent name to env var names for authentication
AGENT_CREDENTIALS: list[dict[str, str]] = [
    {
        "name": "claude-reviewer",
        "app_id_env": "INPUT_CLAUDE_APP_ID",
        "key_env": "INPUT_CLAUDE_APP_PRIVATE_KEY",
        "api_key_env": "INPUT_ANTHROPIC_API_KEY",
        "api_key_target": "ANTHROPIC_API_KEY",
    },
    {
        "name": "gpt-reviewer",
        "app_id_env": "INPUT_GPT_APP_ID",
        "key_env": "INPUT_GPT_APP_PRIVATE_KEY",
        "api_key_env": "INPUT_OPENAI_API_KEY",
        "api_key_target": "OPENAI_API_KEY",
    },
    {
        "name": "gemini-reviewer",
        "app_id_env": "INPUT_GEMINI_APP_ID",
        "key_env": "INPUT_GEMINI_APP_PRIVATE_KEY",
        "api_key_env": "INPUT_GOOGLE_API_KEY",
        "api_key_target": "GOOGLE_API_KEY",
    },
]


def load_config(repo_root: Path | None = None) -> ReviewConfig:
    """Load configuration from environment variables and optional YAML file.

    Environment variables (from GitHub Action inputs) take the form
    ``INPUT_<NAME>``. If a ``.rocketride-review.yml`` file exists at the
    repository root, its values are merged over the defaults. Environment
    variable overrides are applied last.

    Args:
        repo_root: Repository root directory. Defaults to the current
            working directory.

    Returns:
        A validated ReviewConfig instance.

    Raises:
        ConfigurationError: If the YAML file is malformed or the
            resulting configuration fails validation.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    config_data: dict[str, Any] = {}

    # Load from YAML file if it exists
    config_path_str = os.environ.get("INPUT_CONFIG_PATH", ".rocketride-review.yml")
    config_path = repo_root / config_path_str
    if config_path.is_file():
        try:
            raw = config_path.read_text(encoding="utf-8")
            parsed = yaml.safe_load(raw)
            if parsed is not None:
                if not isinstance(parsed, dict):
                    type_name = type(parsed).__name__
                    msg = f"Config file must be a YAML mapping, got {type_name}"
                    raise ConfigurationError(msg)
                config_data.update(parsed)
        except yaml.YAMLError as e:
            msg = f"Failed to parse config file {config_path}: {e}"
            raise ConfigurationError(msg) from e

    # Override with environment variables
    env_mappings: dict[str, str] = {
        "INPUT_REVIEW_CONTEXT": "review_context",
        "INPUT_TARGET_BRANCH": "target_branch",
    }
    for env_key, config_key in env_mappings.items():
        value = os.environ.get(env_key)
        if value is not None:
            config_data[config_key] = value

    try:
        return ReviewConfig(**config_data)
    except ValidationError as e:
        msg = f"Invalid configuration: {e}"
        raise ConfigurationError(msg) from e
