"""Integration tests for the full review orchestration flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.aggregator import deduplicate_reviews
from src.errors import EngineError
from src.models import AgentReview, ReviewComment, ReviewConfig, Severity

pytestmark = pytest.mark.integration


@pytest.fixture()
def pr_event() -> dict:
    """A realistic pull_request opened event payload."""
    return {
        "action": "opened",
        "pull_request": {
            "number": 42,
            "base": {"ref": "main"},
        },
        "repository": {"full_name": "owner/repo"},
    }


@pytest.fixture()
def three_agent_reviews() -> list[AgentReview]:
    """Three successful agent reviews with mixed findings."""
    return [
        AgentReview(
            reviewer="claude-reviewer",
            comments=[
                ReviewComment(
                    file="src/app.py",
                    line=10,
                    severity=Severity.HIGH,
                    body="Potential null reference.",
                ),
            ],
        ),
        AgentReview(
            reviewer="gpt-reviewer",
            comments=[
                ReviewComment(
                    file="src/app.py",
                    line=10,
                    severity=Severity.HIGH,
                    body="Potential null reference here.",
                ),
            ],
        ),
        AgentReview(
            reviewer="gemini-reviewer",
            comments=[
                ReviewComment(
                    file="src/utils.py",
                    line=5,
                    severity=Severity.LOW,
                    body="Consider adding a docstring.",
                ),
            ],
        ),
    ]


class TestFullReviewHappyPath:
    """End-to-end: event -> gating -> diff -> filter -> pipeline -> dedup -> post."""

    async def test_full_review_happy_path(
        self,
        pr_event: dict,
        default_config: ReviewConfig,
        simple_diff: str,
        three_agent_reviews: list[AgentReview],
        mock_github_client: AsyncMock,
        mock_engine: AsyncMock,
        mock_pipeline_runner: AsyncMock,
    ) -> None:
        """Full review runs end-to-end: diff fetched, pipeline runs, reviews posted."""
        from src.main import _handle_full_review

        mock_github_client.get_pr_diff = AsyncMock(return_value=simple_diff)
        mock_github_client.get_file_content = AsyncMock(return_value="file content")
        mock_pipeline_runner.run_full_review = AsyncMock(
            return_value=(three_agent_reviews, [])
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
            await _handle_full_review(pr_event, default_config)

        # Pipeline was called
        mock_pipeline_runner.run_full_review.assert_awaited_once()

        # Reviews were posted (submit_review called for each agent)
        assert mock_github_client.submit_review.await_count >= 1


class TestFullReviewOneAgentFails:
    """One agent returns malformed data; other two succeed."""

    async def test_one_agent_fails_others_succeed(
        self,
        pr_event: dict,
        default_config: ReviewConfig,
        simple_diff: str,
        mock_github_client: AsyncMock,
        mock_engine: AsyncMock,
        mock_pipeline_runner: AsyncMock,
    ) -> None:
        """When one agent fails, the other two reviews are still posted."""
        from src.main import _handle_full_review

        good_reviews = [
            AgentReview(
                reviewer="claude-reviewer",
                comments=[
                    ReviewComment(
                        file="src/app.py",
                        line=2,
                        severity=Severity.LOW,
                        body="Minor suggestion.",
                    ),
                ],
            ),
            AgentReview(
                reviewer="gemini-reviewer",
                comments=[],
            ),
        ]
        mock_github_client.get_pr_diff = AsyncMock(return_value=simple_diff)
        mock_github_client.get_file_content = AsyncMock(return_value="content")
        mock_pipeline_runner.run_full_review = AsyncMock(
            return_value=(good_reviews, ["gpt-reviewer"])
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
            await _handle_full_review(pr_event, default_config)

        # Summary comment posted about failed agent
        mock_github_client.post_issue_comment.assert_awaited()
        call_args = mock_github_client.post_issue_comment.call_args
        assert "gpt-reviewer" in call_args[0][0]


class TestFullReviewAllFilesFiltered:
    """All changed files match ignore patterns — no pipeline run."""

    async def test_all_files_filtered(
        self,
        pr_event: dict,
        default_config: ReviewConfig,
        filtered_diff: str,
        mock_github_client: AsyncMock,
        mock_engine: AsyncMock,
        mock_pipeline_runner: AsyncMock,
    ) -> None:
        """Filtered files produce summary comment, no pipeline run."""
        from src.main import _handle_full_review

        mock_github_client.get_pr_diff = AsyncMock(return_value=filtered_diff)

        agent_clients = {"claude-reviewer": mock_github_client}

        with (
            patch(
                "src.main._initialize_agents",
                return_value=(agent_clients, []),
            ),
            patch("src.main.EngineManager", return_value=mock_engine),
            patch("src.main.PipelineRunner", return_value=mock_pipeline_runner),
        ):
            await _handle_full_review(pr_event, default_config)

        # Pipeline was NOT called
        mock_pipeline_runner.run_full_review.assert_not_awaited()

        # Summary comment about filtering was posted
        mock_github_client.post_issue_comment.assert_awaited()
        call_args = mock_github_client.post_issue_comment.call_args
        assert "ignore patterns" in call_args[0][0]


class TestFullReviewOversizedPR:
    """PR exceeds size limits — skip with comment."""

    async def test_oversized_pr_skipped(
        self,
        pr_event: dict,
        default_config: ReviewConfig,
        oversized_diff: str,
        mock_github_client: AsyncMock,
        mock_engine: AsyncMock,
        mock_pipeline_runner: AsyncMock,
    ) -> None:
        """Oversized PR gets summary comment, no pipeline run."""
        from src.main import _handle_full_review

        mock_github_client.get_pr_diff = AsyncMock(return_value=oversized_diff)
        mock_github_client.get_file_content = AsyncMock(return_value="content")

        agent_clients = {"claude-reviewer": mock_github_client}

        with (
            patch(
                "src.main._initialize_agents",
                return_value=(agent_clients, []),
            ),
            patch("src.main.EngineManager", return_value=mock_engine),
            patch("src.main.PipelineRunner", return_value=mock_pipeline_runner),
        ):
            await _handle_full_review(pr_event, default_config)

        # Pipeline was NOT called
        mock_pipeline_runner.run_full_review.assert_not_awaited()

        # Summary comment about size limits was posted
        mock_github_client.post_issue_comment.assert_awaited()
        call_args = mock_github_client.post_issue_comment.call_args
        assert "too large" in call_args[0][0]


class TestFullReviewEngineFailure:
    """Engine fails to start — error comment posted."""

    async def test_engine_failure_posts_comment(
        self,
        pr_event: dict,
        default_config: ReviewConfig,
        simple_diff: str,
        mock_github_client: AsyncMock,
    ) -> None:
        """Engine failure posts error comment and re-raises."""
        from src.main import _handle_full_review

        mock_github_client.get_pr_diff = AsyncMock(return_value=simple_diff)
        mock_github_client.get_file_content = AsyncMock(return_value="content")

        agent_clients = {"claude-reviewer": mock_github_client}

        failing_engine = AsyncMock()
        failing_engine.__aenter__ = AsyncMock(
            side_effect=EngineError("Engine failed to start")
        )
        failing_engine.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "src.main._initialize_agents",
                return_value=(agent_clients, []),
            ),
            patch("src.main.EngineManager", return_value=failing_engine),
            pytest.raises(EngineError),
        ):
            await _handle_full_review(pr_event, default_config)

        # Error comment was posted
        mock_github_client.post_issue_comment.assert_awaited()
        call_args = mock_github_client.post_issue_comment.call_args
        assert "engine" in call_args[0][0].lower()


class TestDeduplicationIntegration:
    """Verify dedup works correctly in the full review context."""

    def test_cross_agent_duplicates_removed(
        self,
        three_agent_reviews: list[AgentReview],
    ) -> None:
        """Near-duplicate comments across agents are deduplicated."""
        result = deduplicate_reviews(three_agent_reviews)
        total = sum(len(r.comments) for r in result)
        # claude and gpt have near-duplicate comments on same file/line
        # Only one should survive + gemini's unique comment
        assert total == 2
