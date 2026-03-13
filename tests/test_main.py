"""Tests for gating logic and orchestration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.main import _extract_changed_files, run, should_run
from src.models import ReviewConfig


@pytest.fixture()
def default_config() -> ReviewConfig:
    return ReviewConfig()


@pytest.fixture()
def pr_opened_event() -> dict[str, object]:
    return {
        "action": "opened",
        "pull_request": {
            "number": 42,
            "base": {"ref": "main"},
            "head": {"sha": "abc123"},
            "user": {"login": "developer"},
        },
        "repository": {"full_name": "owner/repo"},
    }


@pytest.fixture()
def pr_sync_event() -> dict[str, object]:
    return {
        "action": "synchronize",
        "pull_request": {
            "number": 42,
            "base": {"ref": "main"},
            "head": {"sha": "def456"},
            "user": {"login": "developer"},
        },
        "repository": {"full_name": "owner/repo"},
    }


class TestShouldRun:
    """Tests for gating logic."""

    def test_pr_opened_on_main(
        self, pr_opened_event: dict[str, object], default_config: ReviewConfig
    ) -> None:
        assert should_run(pr_opened_event, "pull_request", default_config)

    def test_pr_synchronize_on_main(
        self, pr_sync_event: dict[str, object], default_config: ReviewConfig
    ) -> None:
        assert should_run(pr_sync_event, "pull_request", default_config)

    def test_wrong_event_type(
        self, pr_opened_event: dict[str, object], default_config: ReviewConfig
    ) -> None:
        assert not should_run(pr_opened_event, "issue_comment", default_config)

    def test_wrong_action(self, default_config: ReviewConfig) -> None:
        event = {
            "action": "closed",
            "pull_request": {"base": {"ref": "main"}},
        }
        assert not should_run(event, "pull_request", default_config)

    def test_wrong_branch(self, default_config: ReviewConfig) -> None:
        event = {
            "action": "opened",
            "pull_request": {"base": {"ref": "develop"}},
        }
        assert not should_run(event, "pull_request", default_config)

    def test_custom_target_branch(self, pr_opened_event: dict[str, object]) -> None:
        config = ReviewConfig(target_branch="develop")
        assert not should_run(pr_opened_event, "pull_request", config)

    def test_matches_custom_target_branch(self) -> None:
        event = {
            "action": "opened",
            "pull_request": {"base": {"ref": "develop"}},
        }
        config = ReviewConfig(target_branch="develop")
        assert should_run(event, "pull_request", config)

    def test_push_event_rejected(self, default_config: ReviewConfig) -> None:
        event = {"ref": "refs/heads/main"}
        assert not should_run(event, "push", default_config)

    def test_missing_pull_request_payload(self, default_config: ReviewConfig) -> None:
        event = {"action": "opened"}
        assert not should_run(event, "pull_request", default_config)


class TestExtractChangedFiles:
    """Tests for _extract_changed_files()."""

    def test_extract_from_diff(self, mock_pr_diff: str) -> None:
        files = _extract_changed_files(mock_pr_diff)
        assert "src/utils.py" in files
        assert "src/main.py" in files
        assert len(files) == 2

    def test_empty_diff(self) -> None:
        assert _extract_changed_files("") == []

    def test_no_plus_lines(self) -> None:
        diff = "--- a/file.py\nsome content\n"
        assert _extract_changed_files(diff) == []


class TestRunOrchestration:
    """Integration-style tests for the run() function."""

    @pytest.mark.asyncio()
    async def test_missing_event_path_exits_cleanly(self) -> None:
        env = {"GITHUB_EVENT_PATH": "", "GITHUB_EVENT_NAME": ""}
        with patch.dict(os.environ, env, clear=True):
            await run()  # Should not raise

    @pytest.mark.asyncio()
    async def test_wrong_event_type_exits_cleanly(self, tmp_path: Path) -> None:
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({"action": "created"}))

        env = {
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_EVENT_NAME": "issue_comment",
            "GITHUB_WORKSPACE": str(tmp_path),
        }
        with patch.dict(os.environ, env, clear=True):
            await run()  # Should exit cleanly without error

    @pytest.mark.asyncio()
    async def test_all_files_filtered_posts_summary(
        self, tmp_path: Path, pr_opened_event: dict[str, object]
    ) -> None:
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps(pr_opened_event))

        env = {
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_WORKSPACE": str(tmp_path),
            "INPUT_CLAUDE_APP_ID": "12345",
            "INPUT_CLAUDE_APP_PRIVATE_KEY": "fake-key",
            "INPUT_ANTHROPIC_API_KEY": "fake-api-key",
        }

        mock_client = AsyncMock()
        # Return a diff where all files match ignore patterns
        mock_client.get_pr_diff = AsyncMock(
            return_value=(
                "diff --git a/package-lock.json b/package-lock.json\n"
                "--- a/package-lock.json\n"
                "+++ b/package-lock.json\n"
                "@@ -1,1 +1,1 @@\n"
                "+updated\n"
            )
        )

        with (
            patch.dict(os.environ, env, clear=True),
            patch("src.main.GitHubClient", return_value=mock_client),
        ):
            await run()

        mock_client.post_issue_comment.assert_called_once()
        call_args = mock_client.post_issue_comment.call_args
        assert "ignore patterns" in call_args.args[0]

    @pytest.mark.asyncio()
    async def test_engine_failure_exits_cleanly(
        self, tmp_path: Path, pr_opened_event: dict[str, object]
    ) -> None:
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps(pr_opened_event))

        env = {
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_WORKSPACE": str(tmp_path),
            "INPUT_CLAUDE_APP_ID": "12345",
            "INPUT_CLAUDE_APP_PRIVATE_KEY": "fake-key",
            "INPUT_ANTHROPIC_API_KEY": "fake-api-key",
        }

        mock_client = AsyncMock()
        mock_client.get_pr_diff = AsyncMock(
            return_value=(
                "diff --git a/src/app.py b/src/app.py\n"
                "--- a/src/app.py\n"
                "+++ b/src/app.py\n"
                "@@ -1,1 +1,2 @@\n"
                "+new code\n"
            )
        )
        mock_client.get_file_content = AsyncMock(return_value="file content")

        from src.errors import EngineError

        with (
            patch.dict(os.environ, env, clear=True),
            patch("src.main.GitHubClient", return_value=mock_client),
            patch(
                "src.main.EngineManager",
                side_effect=EngineError("Docker not available"),
            ),
        ):
            await run()  # Should not raise

        # Should have tried to post error summary
        mock_client.post_issue_comment.assert_called()

    @pytest.mark.asyncio()
    async def test_oversized_pr_posts_summary(
        self, tmp_path: Path, pr_opened_event: dict[str, object]
    ) -> None:
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps(pr_opened_event))

        # Generate a diff that exceeds max_total_lines (default 5000)
        lines = ["diff --git a/big.py b/big.py\n", "--- a/big.py\n", "+++ b/big.py\n"]
        lines.append("@@ -1,1 +1,6000 @@\n")
        for i in range(5500):
            lines.append(f"+line {i}\n")
        big_diff = "".join(lines)

        env = {
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_WORKSPACE": str(tmp_path),
            "INPUT_CLAUDE_APP_ID": "12345",
            "INPUT_CLAUDE_APP_PRIVATE_KEY": "fake-key",
            "INPUT_ANTHROPIC_API_KEY": "fake-api-key",
        }

        mock_client = AsyncMock()
        mock_client.get_pr_diff = AsyncMock(return_value=big_diff)

        with (
            patch.dict(os.environ, env, clear=True),
            patch("src.main.GitHubClient", return_value=mock_client),
        ):
            await run()

        mock_client.post_issue_comment.assert_called_once()
        call_args = mock_client.post_issue_comment.call_args
        assert "too large" in call_args.args[0]
