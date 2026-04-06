"""Integration tests for the conversation reply flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.config import BOT_USERNAMES
from src.main import should_run
from src.models import ReviewConfig

pytestmark = pytest.mark.integration


@pytest.fixture()
def conversation_event() -> dict:
    """A realistic pull_request_review_comment event payload."""
    return {
        "action": "created",
        "comment": {
            "id": 200,
            "body": "Can you explain this in more detail?",
            "in_reply_to_id": 100,
            "user": {"login": "developer"},
        },
        "pull_request": {
            "number": 42,
        },
        "repository": {"full_name": "owner/repo"},
    }


@pytest.fixture()
def bot_conversation_event() -> dict:
    """A review comment event authored by a bot (should be rejected)."""
    return {
        "action": "created",
        "comment": {
            "id": 201,
            "body": "Follow-up from bot.",
            "in_reply_to_id": 100,
            "user": {"login": "claude-reviewer[bot]"},
        },
        "pull_request": {
            "number": 42,
        },
        "repository": {"full_name": "owner/repo"},
    }


class TestConversationReplyHappyPath:
    """Agent identification -> context -> pipeline -> reply."""

    async def test_conversation_reply_happy_path(
        self,
        conversation_event: dict,
        mock_github_client: AsyncMock,
        mock_engine: AsyncMock,
        mock_pipeline_runner: AsyncMock,
    ) -> None:
        """Full conversation flow: identify agent, build context, get reply, post."""
        from src.main import _handle_conversation_reply

        # The lookup client finds the parent comment authored by claude bot
        mock_github_client.get_review_comments = AsyncMock(
            return_value=[
                {
                    "id": 100,
                    "user": "claude-reviewer[bot]",
                    "path": "src/app.py",
                    "body": "Consider refactoring this.",
                },
            ]
        )
        mock_github_client.get_comment_thread = AsyncMock(
            return_value=[
                {
                    "user": "claude-reviewer[bot]",
                    "body": "Consider refactoring this.",
                    "path": "src/app.py",
                },
                {
                    "user": "developer",
                    "body": "Can you explain this in more detail?",
                },
            ]
        )
        mock_github_client.get_file_content = AsyncMock(
            return_value="def foo():\n    pass\n"
        )
        mock_pipeline_runner.run_conversation_reply = AsyncMock(
            return_value="Here is a more detailed explanation..."
        )

        agent_clients = {
            "claude-reviewer": mock_github_client,
            "gpt-reviewer": mock_github_client,
            "gemini-reviewer": mock_github_client,
        }

        with (
            patch(
                "src.main._initialize_agents",
                return_value=(agent_clients, []),
            ),
            patch("src.main.EngineManager", return_value=mock_engine),
            patch("src.main.PipelineRunner", return_value=mock_pipeline_runner),
        ):
            await _handle_conversation_reply(conversation_event, "owner/repo", 42)

        # Pipeline was called with the correct agent
        mock_pipeline_runner.run_conversation_reply.assert_awaited_once()
        call_kwargs = mock_pipeline_runner.run_conversation_reply.call_args
        assert call_kwargs[1]["agent_node_id"] == "claude-reviewer"

        # Reply was posted
        mock_github_client.post_reply_comment.assert_awaited_once()


class TestConversationBotLoopPrevention:
    """Bot comment is rejected at gating — no pipeline run."""

    def test_bot_comment_rejected_at_gating(
        self,
        bot_conversation_event: dict,
        default_config: ReviewConfig,
    ) -> None:
        """Comments from bot usernames are rejected by should_run gating."""
        result = should_run(
            bot_conversation_event,
            "pull_request_review_comment",
            default_config,
        )
        assert result is None

    def test_all_bot_usernames_rejected(
        self,
        default_config: ReviewConfig,
    ) -> None:
        """Every known bot username is correctly rejected."""
        for bot_name in BOT_USERNAMES:
            event = {
                "action": "created",
                "comment": {
                    "id": 300,
                    "body": "Some text",
                    "in_reply_to_id": 100,
                    "user": {"login": bot_name},
                },
            }
            result = should_run(
                event,
                "pull_request_review_comment",
                default_config,
            )
            assert result is None, f"Bot {bot_name} was not rejected"
