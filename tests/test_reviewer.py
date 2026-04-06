"""Tests for review posting logic."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.errors import CommentPostingError, ReviewSubmissionError
from src.models import AgentReview, ReviewComment, Severity
from src.reviewer import (
    _build_review_summary,
    _determine_review_status,
    _format_comment_body,
    post_agent_review,
)


class TestFormatCommentBody:
    """Tests for severity badge formatting."""

    def test_critical_badge(self) -> None:
        result = _format_comment_body(Severity.CRITICAL, "Security issue")
        assert "Critical" in result
        assert "Security issue" in result

    def test_nitpick_badge(self) -> None:
        result = _format_comment_body(Severity.NITPICK, "Minor style")
        assert "Nitpick" in result
        assert "Minor style" in result


class TestDetermineReviewStatus:
    """Tests for review status logic."""

    def test_empty_comments_approve(self) -> None:
        review = AgentReview(reviewer="claude-reviewer", comments=[])
        assert _determine_review_status(review) == "APPROVE"

    def test_critical_triggers_request_changes(self) -> None:
        review = AgentReview(
            reviewer="claude-reviewer",
            comments=[
                ReviewComment(
                    file="a.py", line=1, severity=Severity.CRITICAL, body="Bug"
                )
            ],
        )
        assert _determine_review_status(review) == "REQUEST_CHANGES"

    def test_high_triggers_request_changes_default_threshold(self) -> None:
        review = AgentReview(
            reviewer="claude-reviewer",
            comments=[
                ReviewComment(file="a.py", line=1, severity=Severity.HIGH, body="Issue")
            ],
        )
        assert _determine_review_status(review) == "REQUEST_CHANGES"

    def test_high_with_critical_threshold_is_comment(self) -> None:
        """With approval_threshold='critical', high is not blocking."""
        review = AgentReview(
            reviewer="claude-reviewer",
            comments=[
                ReviewComment(file="a.py", line=1, severity=Severity.HIGH, body="Issue")
            ],
        )
        status = _determine_review_status(review, approval_threshold="critical")
        assert status == "COMMENT"

    def test_medium_only_is_comment(self) -> None:
        review = AgentReview(
            reviewer="claude-reviewer",
            comments=[
                ReviewComment(
                    file="a.py", line=1, severity=Severity.MEDIUM, body="Suggestion"
                )
            ],
        )
        assert _determine_review_status(review) == "COMMENT"

    def test_low_and_nitpick_is_comment(self) -> None:
        review = AgentReview(
            reviewer="claude-reviewer",
            comments=[
                ReviewComment(
                    file="a.py", line=1, severity=Severity.LOW, body="Consider"
                ),
                ReviewComment(
                    file="b.py", line=2, severity=Severity.NITPICK, body="Style"
                ),
            ],
        )
        assert _determine_review_status(review) == "COMMENT"


class TestBuildReviewSummary:
    """Tests for review summary markdown generation."""

    def test_no_comments_summary(self) -> None:
        review = AgentReview(reviewer="claude-reviewer", comments=[])
        summary = _build_review_summary(review, [])
        assert "No issues found" in summary
        assert "claude-reviewer" in summary

    def test_with_comments_shows_counts(self) -> None:
        review = AgentReview(
            reviewer="claude-reviewer",
            comments=[
                ReviewComment(
                    file="a.py", line=1, severity=Severity.HIGH, body="Issue"
                ),
                ReviewComment(
                    file="b.py", line=2, severity=Severity.MEDIUM, body="Other"
                ),
            ],
        )
        summary = _build_review_summary(review, [])
        assert "2 comment(s)" in summary
        assert "high: 1" in summary
        assert "medium: 1" in summary

    def test_failed_comments_noted(self) -> None:
        review = AgentReview(reviewer="claude-reviewer", comments=[])
        failed = [
            ReviewComment(file="a.py", line=1, severity=Severity.HIGH, body="Issue")
        ]
        summary = _build_review_summary(review, failed)
        assert "1 comment(s) could not be posted" in summary


class TestPostAgentReview:
    """Tests for the full posting flow."""

    @pytest.mark.asyncio()
    async def test_post_clean_review(
        self, mock_agent_response_clean: AgentReview
    ) -> None:
        mock_client = AsyncMock()

        await post_agent_review(mock_agent_response_clean, mock_client)

        assert mock_client.post_review_comment.call_count == 2
        mock_client.submit_review.assert_called_once()
        call_kwargs = mock_client.submit_review.call_args
        assert call_kwargs.kwargs["status"] == "COMMENT"

    @pytest.mark.asyncio()
    async def test_post_review_with_issues(
        self, mock_agent_response_issues: AgentReview
    ) -> None:
        mock_client = AsyncMock()

        await post_agent_review(mock_agent_response_issues, mock_client)

        assert mock_client.post_review_comment.call_count == 2
        call_kwargs = mock_client.submit_review.call_args
        assert call_kwargs.kwargs["status"] == "REQUEST_CHANGES"

    @pytest.mark.asyncio()
    async def test_comment_failure_continues(
        self, mock_agent_response_issues: AgentReview
    ) -> None:
        mock_client = AsyncMock()
        mock_client.post_review_comment.side_effect = CommentPostingError("fail")

        await post_agent_review(mock_agent_response_issues, mock_client)

        # Both comments attempted, both failed
        assert mock_client.post_review_comment.call_count == 2
        # Review still submitted
        mock_client.submit_review.assert_called_once()
        # Summary should mention failed comments
        summary = mock_client.submit_review.call_args.kwargs["body"]
        assert "could not be posted" in summary

    @pytest.mark.asyncio()
    async def test_empty_review_approves(self) -> None:
        review = AgentReview(reviewer="claude-reviewer", comments=[])
        mock_client = AsyncMock()

        await post_agent_review(review, mock_client)

        mock_client.post_review_comment.assert_not_called()
        call_kwargs = mock_client.submit_review.call_args
        assert call_kwargs.kwargs["status"] == "APPROVE"

    @pytest.mark.asyncio()
    async def test_explicit_review_status_overrides_computed(self) -> None:
        """When review_status is provided, it overrides the computed status."""
        review = AgentReview(
            reviewer="claude-reviewer",
            comments=[
                ReviewComment(file="a.py", line=1, severity=Severity.HIGH, body="Issue")
            ],
        )
        mock_client = AsyncMock()

        # HIGH would normally trigger REQUEST_CHANGES, but we override to COMMENT
        await post_agent_review(review, mock_client, review_status="COMMENT")

        call_kwargs = mock_client.submit_review.call_args
        assert call_kwargs.kwargs["status"] == "COMMENT"

    @pytest.mark.asyncio()
    async def test_review_status_approve_overrides(self) -> None:
        """Empty review with explicit COMMENT status uses COMMENT, not APPROVE."""
        review = AgentReview(reviewer="gpt-reviewer", comments=[])
        mock_client = AsyncMock()

        await post_agent_review(review, mock_client, review_status="COMMENT")

        call_kwargs = mock_client.submit_review.call_args
        assert call_kwargs.kwargs["status"] == "COMMENT"

    @pytest.mark.asyncio()
    async def test_submit_review_failure_caught(self) -> None:
        """ReviewSubmissionError from submit_review is caught, not bare Exception."""
        review = AgentReview(reviewer="claude-reviewer", comments=[])
        mock_client = AsyncMock()
        mock_client.submit_review = AsyncMock(
            side_effect=ReviewSubmissionError("forbidden")
        )

        # Should not raise — error is caught and logged
        await post_agent_review(review, mock_client)
