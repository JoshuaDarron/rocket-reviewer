"""Tests for gating logic, orchestration, conversation, and multi-agent approval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import BOT_USERNAMES
from src.main import (
    _build_agent_failure_message,
    _determine_cross_agent_statuses,
    _extract_changed_files,
    _format_thread_context,
    _handle_conversation_reply,
    _identify_target_agent,
    _initialize_agents,
    run,
    should_run,
)
from src.models import AgentReview, ReviewComment, ReviewConfig, Severity


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


@pytest.fixture()
def review_comment_event() -> dict[str, object]:
    """A pull_request_review_comment event from a developer replying to a bot."""
    return {
        "action": "created",
        "comment": {
            "id": 200,
            "user": {"login": "developer"},
            "body": "Can you explain why this is an issue?",
            "in_reply_to_id": 100,
            "path": "src/utils.py",
            "line": 11,
        },
        "pull_request": {
            "number": 42,
        },
        "repository": {"full_name": "owner/repo"},
    }


@pytest.fixture()
def review_comment_event_from_bot() -> dict[str, object]:
    """A pull_request_review_comment event from a bot (should be skipped)."""
    return {
        "action": "created",
        "comment": {
            "id": 201,
            "user": {"login": "claude-reviewer[bot]"},
            "body": "Here is my explanation.",
            "in_reply_to_id": 100,
        },
        "pull_request": {
            "number": 42,
        },
        "repository": {"full_name": "owner/repo"},
    }


@pytest.fixture()
def review_comment_event_not_reply() -> dict[str, object]:
    """A pull_request_review_comment event that is not a reply (no in_reply_to_id)."""
    return {
        "action": "created",
        "comment": {
            "id": 202,
            "user": {"login": "developer"},
            "body": "New top-level comment.",
        },
        "pull_request": {
            "number": 42,
        },
        "repository": {"full_name": "owner/repo"},
    }


class TestShouldRun:
    """Tests for gating logic."""

    def test_pr_opened_on_main(
        self, pr_opened_event: dict[str, object], default_config: ReviewConfig
    ) -> None:
        assert (
            should_run(pr_opened_event, "pull_request", default_config) == "full_review"
        )

    def test_pr_synchronize_on_main(
        self, pr_sync_event: dict[str, object], default_config: ReviewConfig
    ) -> None:
        assert (
            should_run(pr_sync_event, "pull_request", default_config) == "full_review"
        )

    def test_wrong_event_type(
        self, pr_opened_event: dict[str, object], default_config: ReviewConfig
    ) -> None:
        assert should_run(pr_opened_event, "issue_comment", default_config) is None

    def test_wrong_action(self, default_config: ReviewConfig) -> None:
        event = {
            "action": "closed",
            "pull_request": {"base": {"ref": "main"}},
        }
        assert should_run(event, "pull_request", default_config) is None

    def test_wrong_branch(self, default_config: ReviewConfig) -> None:
        event = {
            "action": "opened",
            "pull_request": {"base": {"ref": "develop"}},
        }
        assert should_run(event, "pull_request", default_config) is None

    def test_custom_target_branch(self, pr_opened_event: dict[str, object]) -> None:
        config = ReviewConfig(target_branch="develop")
        assert should_run(pr_opened_event, "pull_request", config) is None

    def test_matches_custom_target_branch(self) -> None:
        event = {
            "action": "opened",
            "pull_request": {"base": {"ref": "develop"}},
        }
        config = ReviewConfig(target_branch="develop")
        assert should_run(event, "pull_request", config) == "full_review"

    def test_push_event_rejected(self, default_config: ReviewConfig) -> None:
        event = {"ref": "refs/heads/main"}
        assert should_run(event, "push", default_config) is None

    def test_missing_pull_request_payload(self, default_config: ReviewConfig) -> None:
        event = {"action": "opened"}
        assert should_run(event, "pull_request", default_config) is None


class TestConversationGating:
    """Tests for conversation reply gating logic."""

    def test_developer_replying_to_bot_triggers_conversation(
        self,
        review_comment_event: dict[str, object],
        default_config: ReviewConfig,
    ) -> None:
        result = should_run(
            review_comment_event, "pull_request_review_comment", default_config
        )
        assert result == "conversation"

    def test_bot_comment_skipped_loop_prevention(
        self,
        review_comment_event_from_bot: dict[str, object],
        default_config: ReviewConfig,
    ) -> None:
        result = should_run(
            review_comment_event_from_bot,
            "pull_request_review_comment",
            default_config,
        )
        assert result is None

    def test_all_bot_usernames_blocked(self, default_config: ReviewConfig) -> None:
        """All known bot usernames should be blocked by loop prevention."""
        for bot_username in BOT_USERNAMES:
            event = {
                "action": "created",
                "comment": {
                    "id": 999,
                    "user": {"login": bot_username},
                    "body": "Some reply",
                    "in_reply_to_id": 100,
                },
            }
            result = should_run(event, "pull_request_review_comment", default_config)
            assert result is None, f"Bot {bot_username} was not blocked"

    def test_not_a_reply_skipped(
        self,
        review_comment_event_not_reply: dict[str, object],
        default_config: ReviewConfig,
    ) -> None:
        result = should_run(
            review_comment_event_not_reply,
            "pull_request_review_comment",
            default_config,
        )
        assert result is None

    def test_wrong_action_skipped(self, default_config: ReviewConfig) -> None:
        event = {
            "action": "edited",
            "comment": {
                "id": 200,
                "user": {"login": "developer"},
                "body": "Edited text",
                "in_reply_to_id": 100,
            },
        }
        result = should_run(event, "pull_request_review_comment", default_config)
        assert result is None

    def test_missing_comment_payload_skipped(
        self, default_config: ReviewConfig
    ) -> None:
        event = {"action": "created"}
        result = should_run(event, "pull_request_review_comment", default_config)
        assert result is None

    def test_missing_user_info_skipped(self, default_config: ReviewConfig) -> None:
        event = {
            "action": "created",
            "comment": {
                "id": 200,
                "body": "Some reply",
                "in_reply_to_id": 100,
            },
        }
        result = should_run(event, "pull_request_review_comment", default_config)
        assert result is None


class TestIdentifyTargetAgent:
    """Tests for _identify_target_agent()."""

    def test_valid_reply_event(self, review_comment_event: dict[str, object]) -> None:
        result = _identify_target_agent(review_comment_event)
        assert result is not None
        body, in_reply_to_id, comment_id = result
        assert body == "Can you explain why this is an issue?"
        assert in_reply_to_id == 100
        assert comment_id == 200

    def test_no_in_reply_to_id(
        self, review_comment_event_not_reply: dict[str, object]
    ) -> None:
        result = _identify_target_agent(review_comment_event_not_reply)
        assert result is None

    def test_missing_comment(self) -> None:
        result = _identify_target_agent({"action": "created"})
        assert result is None

    def test_empty_body(self) -> None:
        event = {
            "comment": {
                "id": 200,
                "user": {"login": "developer"},
                "body": "",
                "in_reply_to_id": 100,
            }
        }
        result = _identify_target_agent(event)
        assert result is None


class TestFormatThreadContext:
    """Tests for _format_thread_context()."""

    def test_single_comment(self) -> None:
        thread = [{"user": "claude-reviewer[bot]", "body": "Consider refactoring."}]
        result = _format_thread_context(thread)
        assert "claude-reviewer[bot]" in result
        assert "Consider refactoring." in result

    def test_multi_comment_thread(self) -> None:
        thread = [
            {"user": "claude-reviewer[bot]", "body": "Issue here."},
            {"user": "developer", "body": "Can you explain?"},
        ]
        result = _format_thread_context(thread)
        assert "claude-reviewer[bot]" in result
        assert "developer" in result
        assert "---" in result

    def test_empty_thread(self) -> None:
        result = _format_thread_context([])
        assert result == ""


class TestHandleConversationReply:
    """Tests for _handle_conversation_reply()."""

    @pytest.mark.asyncio()
    async def test_successful_conversation_reply(self) -> None:
        """Full conversation reply flow with mocked dependencies."""
        event = {
            "action": "created",
            "comment": {
                "id": 200,
                "user": {"login": "developer"},
                "body": "Why is this a problem?",
                "in_reply_to_id": 100,
            },
        }

        mock_client = AsyncMock()
        mock_client.get_review_comments = AsyncMock(
            return_value=[
                {
                    "id": 100,
                    "user": "claude-reviewer[bot]",
                    "body": "Potential null reference.",
                    "path": "src/utils.py",
                    "line": 11,
                    "in_reply_to_id": None,
                },
                {
                    "id": 200,
                    "user": "developer",
                    "body": "Why is this a problem?",
                    "path": "src/utils.py",
                    "line": 11,
                    "in_reply_to_id": 100,
                },
            ]
        )
        mock_client.get_comment_thread = AsyncMock(
            return_value=[
                {
                    "id": 100,
                    "user": "claude-reviewer[bot]",
                    "body": "Potential null reference.",
                    "path": "src/utils.py",
                    "line": 11,
                    "in_reply_to_id": None,
                },
                {
                    "id": 200,
                    "user": "developer",
                    "body": "Why is this a problem?",
                    "path": "src/utils.py",
                    "line": 11,
                    "in_reply_to_id": 100,
                },
            ]
        )
        mock_client.get_file_content = AsyncMock(return_value="def foo():\n    pass\n")
        mock_client.post_reply_comment = AsyncMock()

        mock_engine = AsyncMock()
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=None)

        mock_runner = AsyncMock()
        mock_runner.run_conversation_reply = AsyncMock(
            return_value="The variable could be None if the API call fails."
        )

        with (
            patch(
                "src.main._initialize_agents",
                return_value=({"claude-reviewer": mock_client}, []),
            ),
            patch("src.main._initialize_single_agent", return_value=mock_client),
            patch("src.main.EngineManager", return_value=mock_engine),
            patch("src.main.PipelineRunner", return_value=mock_runner),
        ):
            await _handle_conversation_reply(event, "owner/repo", 42)

        mock_client.post_reply_comment.assert_called_once()
        call_args = mock_client.post_reply_comment.call_args
        assert call_args.args[0] == 100  # in_reply_to_id
        assert "None" in call_args.args[1]  # reply text

    @pytest.mark.asyncio()
    async def test_agent_not_available_raises(self) -> None:
        """If no agents can be initialized, raise ConfigurationError."""
        from src.errors import ConfigurationError

        event = {
            "action": "created",
            "comment": {
                "id": 200,
                "user": {"login": "developer"},
                "body": "Why?",
                "in_reply_to_id": 100,
            },
        }

        with (
            patch(
                "src.main._initialize_agents",
                return_value=(
                    {},
                    ["claude-reviewer", "gpt-reviewer", "gemini-reviewer"],
                ),
            ),
            pytest.raises(ConfigurationError),
        ):
            await _handle_conversation_reply(event, "owner/repo", 42)

    @pytest.mark.asyncio()
    async def test_parent_not_from_bot_skips(self) -> None:
        """If parent comment is not from a bot, skip."""
        event = {
            "action": "created",
            "comment": {
                "id": 200,
                "user": {"login": "developer"},
                "body": "Agree with you.",
                "in_reply_to_id": 100,
            },
        }

        mock_client = AsyncMock()
        mock_client.get_review_comments = AsyncMock(
            return_value=[
                {
                    "id": 100,
                    "user": "another-developer",
                    "body": "This looks wrong.",
                    "path": "src/utils.py",
                    "line": 11,
                    "in_reply_to_id": None,
                },
            ]
        )
        mock_client.get_comment_thread = AsyncMock(
            return_value=[
                {
                    "id": 100,
                    "user": "another-developer",
                    "body": "This looks wrong.",
                    "path": "src/utils.py",
                    "line": 11,
                    "in_reply_to_id": None,
                },
            ]
        )

        with patch(
            "src.main._initialize_agents",
            return_value=({"claude-reviewer": mock_client}, []),
        ):
            # Should not raise, should just skip
            await _handle_conversation_reply(event, "owner/repo", 42)

        # No reply should be posted
        mock_client.post_reply_comment.assert_not_called()

    @pytest.mark.asyncio()
    async def test_pipeline_error_propagates(self) -> None:
        """PipelineError during conversation reply propagates up."""
        from src.errors import PipelineError

        event = {
            "action": "created",
            "comment": {
                "id": 200,
                "user": {"login": "developer"},
                "body": "Why?",
                "in_reply_to_id": 100,
            },
        }

        mock_client = AsyncMock()
        mock_client.get_review_comments = AsyncMock(
            return_value=[
                {
                    "id": 100,
                    "user": "claude-reviewer[bot]",
                    "body": "Issue.",
                    "path": "src/main.py",
                    "line": 5,
                    "in_reply_to_id": None,
                },
            ]
        )
        mock_client.get_comment_thread = AsyncMock(
            return_value=[
                {
                    "id": 100,
                    "user": "claude-reviewer[bot]",
                    "body": "Issue.",
                    "path": "src/main.py",
                    "line": 5,
                    "in_reply_to_id": None,
                },
            ]
        )
        mock_client.get_file_content = AsyncMock(return_value="code")

        mock_engine = AsyncMock()
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=None)

        mock_runner = AsyncMock()
        mock_runner.run_conversation_reply = AsyncMock(
            side_effect=PipelineError("Pipeline crashed")
        )

        with (
            patch(
                "src.main._initialize_agents",
                return_value=({"claude-reviewer": mock_client}, []),
            ),
            patch("src.main._initialize_single_agent", return_value=mock_client),
            patch("src.main.EngineManager", return_value=mock_engine),
            patch("src.main.PipelineRunner", return_value=mock_runner),
            pytest.raises(PipelineError, match="Pipeline crashed"),
        ):
            await _handle_conversation_reply(event, "owner/repo", 42)


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


class TestCrossAgentApproval:
    """Tests for _determine_cross_agent_statuses()."""

    def test_all_clean_all_approve(
        self, mock_all_clean_reviews: list[AgentReview]
    ) -> None:
        statuses = _determine_cross_agent_statuses(mock_all_clean_reviews)
        assert all(s == "APPROVE" for s in statuses.values())

    def test_one_critical_mixed_statuses(
        self, mock_mixed_severity_reviews: list[AgentReview]
    ) -> None:
        statuses = _determine_cross_agent_statuses(mock_mixed_severity_reviews)
        assert statuses["claude-reviewer"] == "REQUEST_CHANGES"
        assert statuses["gpt-reviewer"] == "COMMENT"
        assert statuses["gemini-reviewer"] == "COMMENT"

    def test_all_blocking_all_request_changes(self) -> None:
        reviews = [
            AgentReview(
                reviewer="claude-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py", line=1, severity=Severity.CRITICAL, body="Bug"
                    )
                ],
            ),
            AgentReview(
                reviewer="gpt-reviewer",
                comments=[
                    ReviewComment(
                        file="b.py", line=2, severity=Severity.HIGH, body="Issue"
                    )
                ],
            ),
        ]
        statuses = _determine_cross_agent_statuses(reviews)
        assert statuses["claude-reviewer"] == "REQUEST_CHANGES"
        assert statuses["gpt-reviewer"] == "REQUEST_CHANGES"

    def test_critical_only_threshold(self) -> None:
        """With threshold='critical', high is not blocking."""
        reviews = [
            AgentReview(
                reviewer="claude-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py", line=1, severity=Severity.HIGH, body="Issue"
                    )
                ],
            ),
            AgentReview(
                reviewer="gpt-reviewer",
                comments=[],
            ),
        ]
        statuses = _determine_cross_agent_statuses(reviews, "critical")
        assert statuses["claude-reviewer"] == "APPROVE"
        assert statuses["gpt-reviewer"] == "APPROVE"

    def test_empty_reviews_all_approve(self) -> None:
        reviews = [
            AgentReview(reviewer="claude-reviewer", comments=[]),
            AgentReview(reviewer="gpt-reviewer", comments=[]),
            AgentReview(reviewer="gemini-reviewer", comments=[]),
        ]
        statuses = _determine_cross_agent_statuses(reviews)
        assert all(s == "APPROVE" for s in statuses.values())


class TestBuildAgentFailureMessage:
    """Tests for _build_agent_failure_message()."""

    def test_single_failure(self) -> None:
        msg = _build_agent_failure_message(["gpt-reviewer"])
        assert "gpt-reviewer" in msg
        assert "unavailable" in msg

    def test_multiple_failures(self) -> None:
        msg = _build_agent_failure_message(["gpt-reviewer", "gemini-reviewer"])
        assert "gpt-reviewer" in msg
        assert "gemini-reviewer" in msg


class TestInitializeAgents:
    """Tests for _initialize_agents()."""

    def test_missing_credentials_adds_to_failures(self) -> None:
        env = {}
        with patch.dict(os.environ, env, clear=True):
            clients, failures = _initialize_agents("owner/repo", 42)

        assert len(clients) == 0
        assert len(failures) == 3

    def test_auth_failure_adds_to_failures(self) -> None:
        from src.errors import ConfigurationError

        env = {
            "INPUT_CLAUDE_APP_ID": "12345",
            "INPUT_CLAUDE_APP_PRIVATE_KEY": "fake-key",
            "INPUT_ANTHROPIC_API_KEY": "fake-api-key",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch(
                "src.main.GitHubClient",
                side_effect=ConfigurationError("Auth failed"),
            ),
        ):
            clients, failures = _initialize_agents("owner/repo", 42)

        assert len(clients) == 0
        assert "claude-reviewer" in failures
        # GPT and Gemini also failed (missing creds)
        assert len(failures) == 3

    def test_partial_success(self) -> None:
        env = {
            "INPUT_CLAUDE_APP_ID": "12345",
            "INPUT_CLAUDE_APP_PRIVATE_KEY": "fake-key",
            "INPUT_ANTHROPIC_API_KEY": "fake-api-key",
            "INPUT_GPT_APP_ID": "67890",
            "INPUT_GPT_APP_PRIVATE_KEY": "fake-key-2",
            "INPUT_OPENAI_API_KEY": "fake-openai-key",
        }
        mock_client = MagicMock()
        with (
            patch.dict(os.environ, env, clear=True),
            patch("src.main.GitHubClient", return_value=mock_client),
        ):
            clients, failures = _initialize_agents("owner/repo", 42)

        assert len(clients) == 2
        assert "claude-reviewer" in clients
        assert "gpt-reviewer" in clients
        assert len(failures) == 1
        assert "gemini-reviewer" in failures


class TestRunOrchestration:
    """Integration-style tests for the run() function."""

    @pytest.mark.asyncio()
    async def test_missing_event_path_exits_with_error(self) -> None:
        env = {"GITHUB_EVENT_PATH": "", "GITHUB_EVENT_NAME": ""}
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(SystemExit, match="1"),
        ):
            await run()

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
            patch(
                "src.main._initialize_agents",
                return_value=({"claude-reviewer": mock_client}, []),
            ),
        ):
            await run()

        mock_client.post_issue_comment.assert_called_once()
        call_args = mock_client.post_issue_comment.call_args
        assert "ignore patterns" in call_args.args[0]

    @pytest.mark.asyncio()
    async def test_engine_failure_exits_with_error(
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
            patch(
                "src.main._initialize_agents",
                return_value=({"claude-reviewer": mock_client}, []),
            ),
            patch(
                "src.main.EngineManager",
                side_effect=EngineError("Docker not available"),
            ),
            pytest.raises(SystemExit, match="1"),
        ):
            await run()

        # Summary comment is posted before the error propagates
        mock_client.post_issue_comment.assert_called()
        call_args = mock_client.post_issue_comment.call_args
        assert "engine" in call_args.args[0].lower()

    @pytest.mark.asyncio()
    async def test_oversized_pr_posts_summary(
        self, tmp_path: Path, pr_opened_event: dict[str, object]
    ) -> None:
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps(pr_opened_event))

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
            patch(
                "src.main._initialize_agents",
                return_value=({"claude-reviewer": mock_client}, []),
            ),
        ):
            await run()

        mock_client.post_issue_comment.assert_called_once()
        call_args = mock_client.post_issue_comment.call_args
        assert "too large" in call_args.args[0]

    @pytest.mark.asyncio()
    async def test_conversation_event_routes_correctly(self, tmp_path: Path) -> None:
        """A pull_request_review_comment event routes to conversation handler."""
        event = {
            "action": "created",
            "comment": {
                "id": 200,
                "user": {"login": "developer"},
                "body": "Why?",
                "in_reply_to_id": 100,
            },
            "pull_request": {"number": 42},
            "repository": {"full_name": "owner/repo"},
        }
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps(event))

        env = {
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_EVENT_NAME": "pull_request_review_comment",
            "GITHUB_WORKSPACE": str(tmp_path),
        }

        with (
            patch.dict(os.environ, env, clear=True),
            patch("src.main._handle_conversation_reply") as mock_handler,
        ):
            mock_handler.return_value = None
            await run()

        mock_handler.assert_called_once()


class TestAgentFailureIsolation:
    """Tests for agent failure isolation during orchestration."""

    @pytest.mark.asyncio()
    async def test_one_auth_failure_others_proceed(
        self, tmp_path: Path, pr_opened_event: dict[str, object]
    ) -> None:
        """One agent auth failure doesn't prevent others from reviewing."""
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps(pr_opened_event))

        diff = (
            "diff --git a/src/app.py b/src/app.py\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1,1 +1,2 @@\n"
            "+new code\n"
        )

        env = {
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_WORKSPACE": str(tmp_path),
        }

        mock_claude_client = AsyncMock()
        mock_claude_client.get_pr_diff = AsyncMock(return_value=diff)
        mock_claude_client.get_file_content = AsyncMock(return_value="content")

        mock_gpt_client = AsyncMock()

        # claude and gpt init OK, gemini failed
        agent_clients = {
            "claude-reviewer": mock_claude_client,
            "gpt-reviewer": mock_gpt_client,
        }
        agent_failures = ["gemini-reviewer"]

        reviews = [
            AgentReview(reviewer="claude-reviewer", comments=[]),
            AgentReview(reviewer="gpt-reviewer", comments=[]),
        ]

        mock_engine = AsyncMock()
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.dict(os.environ, env, clear=True),
            patch(
                "src.main._initialize_agents",
                return_value=(agent_clients, agent_failures),
            ),
            patch("src.main.EngineManager", return_value=mock_engine),
            patch("src.main.PipelineRunner") as mock_runner_cls,
            patch("src.main.deduplicate_reviews", return_value=reviews),
            patch("src.main.post_agent_review") as mock_post,
        ):
            mock_runner = AsyncMock()
            mock_runner.run_full_review = AsyncMock(return_value=(reviews, []))
            mock_runner_cls.return_value = mock_runner

            await run()

        # Two reviews posted (claude + gpt)
        assert mock_post.call_count == 2
        # Failure message posted about gemini
        mock_claude_client.post_issue_comment.assert_called()
        failure_call = mock_claude_client.post_issue_comment.call_args
        assert "gemini-reviewer" in failure_call.args[0]

    @pytest.mark.asyncio()
    async def test_posting_failure_continues(
        self, tmp_path: Path, pr_opened_event: dict[str, object]
    ) -> None:
        """If posting one agent's review fails, others still post."""
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps(pr_opened_event))

        diff = (
            "diff --git a/src/app.py b/src/app.py\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1,1 +1,2 @@\n"
            "+new code\n"
        )

        env = {
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_WORKSPACE": str(tmp_path),
        }

        mock_claude_client = AsyncMock()
        mock_claude_client.get_pr_diff = AsyncMock(return_value=diff)
        mock_claude_client.get_file_content = AsyncMock(return_value="content")

        mock_gpt_client = AsyncMock()

        agent_clients = {
            "claude-reviewer": mock_claude_client,
            "gpt-reviewer": mock_gpt_client,
        }

        reviews = [
            AgentReview(reviewer="claude-reviewer", comments=[]),
            AgentReview(reviewer="gpt-reviewer", comments=[]),
        ]

        mock_engine = AsyncMock()
        mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
        mock_engine.__aexit__ = AsyncMock(return_value=None)

        call_count = 0

        async def mock_post_side_effect(**kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("GitHub API error")

        with (
            patch.dict(os.environ, env, clear=True),
            patch(
                "src.main._initialize_agents",
                return_value=(agent_clients, []),
            ),
            patch("src.main.EngineManager", return_value=mock_engine),
            patch("src.main.PipelineRunner") as mock_runner_cls,
            patch("src.main.deduplicate_reviews", return_value=reviews),
            patch(
                "src.main.post_agent_review",
                side_effect=mock_post_side_effect,
            ) as mock_post,
        ):
            mock_runner = AsyncMock()
            mock_runner.run_full_review = AsyncMock(return_value=(reviews, []))
            mock_runner_cls.return_value = mock_runner

            await run()  # Should not raise

        # Both were attempted
        assert mock_post.call_count == 2

    @pytest.mark.asyncio()
    async def test_no_agents_initialized_exits_with_error(
        self, tmp_path: Path, pr_opened_event: dict[str, object]
    ) -> None:
        """If no agents could be initialized, exit with error code."""
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps(pr_opened_event))

        env = {
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_WORKSPACE": str(tmp_path),
        }

        with (
            patch.dict(os.environ, env, clear=True),
            patch(
                "src.main._initialize_agents",
                return_value=(
                    {},
                    ["claude-reviewer", "gpt-reviewer", "gemini-reviewer"],
                ),
            ),
            pytest.raises(SystemExit, match="1"),
        ):
            await run()
