"""Shared fixtures for integration tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.models import AgentReview, ReviewComment, ReviewConfig, Severity


@pytest.fixture()
def default_config() -> ReviewConfig:
    """Default review configuration."""
    return ReviewConfig()


@pytest.fixture()
def simple_diff() -> str:
    """A small, realistic diff for integration testing."""
    return (
        "diff --git a/src/app.py b/src/app.py\n"
        "index abc1234..def5678 100644\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,3 +1,5 @@\n"
        " import os\n"
        "+import sys\n"
        "+import json\n"
        " \n"
    )


@pytest.fixture()
def filtered_diff() -> str:
    """A diff where all files match default ignore patterns."""
    return (
        "diff --git a/package-lock.json b/package-lock.json\n"
        "--- a/package-lock.json\n"
        "+++ b/package-lock.json\n"
        "@@ -1,1 +1,1 @@\n"
        "+updated\n"
    )


@pytest.fixture()
def oversized_diff() -> str:
    """A diff that exceeds the default max_total_lines threshold."""
    lines = [
        "diff --git a/big.py b/big.py\n",
        "--- a/big.py\n",
        "+++ b/big.py\n",
        "@@ -1,1 +1,6000 @@\n",
    ]
    for i in range(5500):
        lines.append(f"+line {i}\n")
    return "".join(lines)


@pytest.fixture()
def mock_github_client() -> AsyncMock:
    """A fully mocked GitHubClient for integration tests."""
    client = AsyncMock()
    client.get_pr_diff = AsyncMock(return_value="")
    client.get_file_content = AsyncMock(return_value="file content")
    client.post_review_comment = AsyncMock()
    client.submit_review = AsyncMock()
    client.post_issue_comment = AsyncMock()
    client.post_reply_comment = AsyncMock()
    client.get_review_comments = AsyncMock(return_value=[])
    client.get_comment_thread = AsyncMock(return_value=[])
    return client


@pytest.fixture()
def clean_reviews() -> list[AgentReview]:
    """Three agent reviews with no blocking issues."""
    return [
        AgentReview(
            reviewer="claude-reviewer",
            comments=[
                ReviewComment(
                    file="src/app.py",
                    line=2,
                    severity=Severity.LOW,
                    body="Consider renaming.",
                ),
            ],
        ),
        AgentReview(reviewer="gpt-reviewer", comments=[]),
        AgentReview(reviewer="gemini-reviewer", comments=[]),
    ]


@pytest.fixture()
def mock_engine() -> AsyncMock:
    """A mock EngineManager async context manager."""
    engine = AsyncMock()
    engine.__aenter__ = AsyncMock(return_value=engine)
    engine.__aexit__ = AsyncMock(return_value=None)
    return engine


@pytest.fixture()
def mock_pipeline_runner() -> AsyncMock:
    """A mock PipelineRunner."""
    runner = AsyncMock()
    runner.run_full_review = AsyncMock(return_value=([], []))
    runner.run_conversation_reply = AsyncMock(return_value="Reply text.")
    return runner
