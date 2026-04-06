"""Tests for Pydantic models and schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    AgentReview,
    CommentStatus,
    ReviewComment,
    ReviewConfig,
    Severity,
)


class TestReviewComment:
    """Tests for the ReviewComment model."""

    def test_valid_comment(self) -> None:
        comment = ReviewComment(
            file="src/main.py",
            line=42,
            severity=Severity.HIGH,
            status=CommentStatus.ADD,
            body="This could cause a null pointer exception.",
        )
        assert comment.file == "src/main.py"
        assert comment.line == 42
        assert comment.severity == Severity.HIGH

    def test_default_status_is_add(self) -> None:
        comment = ReviewComment(
            file="test.py", line=1, severity=Severity.LOW, body="Nitpick."
        )
        assert comment.status == CommentStatus.ADD

    def test_invalid_line_zero(self) -> None:
        with pytest.raises(ValidationError):
            ReviewComment(
                file="test.py", line=0, severity=Severity.LOW, body="Bad line."
            )

    def test_empty_body_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReviewComment(file="test.py", line=1, severity=Severity.LOW, body="")

    def test_invalid_severity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReviewComment(
                file="test.py",
                line=1,
                severity="unknown",  # type: ignore[arg-type]
                body="Bad severity.",
            )

    def test_empty_file_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReviewComment(file="", line=1, severity=Severity.LOW, body="Issue.")

    def test_absolute_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReviewComment(
                file="/etc/passwd", line=1, severity=Severity.LOW, body="Issue."
            )

    def test_path_traversal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReviewComment(
                file="../../../etc/passwd", line=1, severity=Severity.LOW, body="Issue."
            )

    def test_relative_path_with_dotdot_in_dir(self) -> None:
        with pytest.raises(ValidationError):
            ReviewComment(
                file="src/../../secret.py", line=1, severity=Severity.LOW, body="Issue."
            )

    def test_valid_nested_path(self) -> None:
        comment = ReviewComment(
            file="src/utils/helpers.py",
            line=10,
            severity=Severity.LOW,
            body="Suggestion.",
        )
        assert comment.file == "src/utils/helpers.py"

    def test_body_max_length(self) -> None:
        with pytest.raises(ValidationError):
            ReviewComment(
                file="test.py",
                line=1,
                severity=Severity.LOW,
                body="x" * 65537,
            )

    def test_body_at_max_length(self) -> None:
        comment = ReviewComment(
            file="test.py",
            line=1,
            severity=Severity.LOW,
            body="x" * 65536,
        )
        assert len(comment.body) == 65536


class TestAgentReview:
    """Tests for the AgentReview model."""

    def test_valid_review(self, mock_agent_response_clean: AgentReview) -> None:
        assert mock_agent_response_clean.reviewer == "claude-reviewer"
        assert len(mock_agent_response_clean.comments) == 2

    def test_empty_comments_allowed(self) -> None:
        review = AgentReview(reviewer="gpt-reviewer", comments=[])
        assert review.comments == []

    def test_default_comments_is_empty_list(self) -> None:
        review = AgentReview(reviewer="gemini-reviewer")
        assert review.comments == []

    def test_malformed_response_rejected(
        self, mock_agent_response_malformed: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError):
            AgentReview(**mock_agent_response_malformed)


class TestReviewConfig:
    """Tests for the ReviewConfig model."""

    def test_defaults(self, mock_config_default: ReviewConfig) -> None:
        assert mock_config_default.review_context == "full"
        assert mock_config_default.target_branch == "main"
        assert mock_config_default.approval_threshold == "high"
        assert mock_config_default.max_chunk_lines == 500
        assert mock_config_default.max_files == 50

    def test_custom_config(self, mock_config_custom: ReviewConfig) -> None:
        assert mock_config_custom.review_context == "diff"
        assert mock_config_custom.target_branch == "develop"
        assert "migrations/**" in mock_config_custom.ignore_patterns_extra

    def test_invalid_review_context(self) -> None:
        with pytest.raises(ValidationError):
            ReviewConfig(review_context="invalid")  # type: ignore[arg-type]

    def test_chunk_lines_minimum(self) -> None:
        with pytest.raises(ValidationError):
            ReviewConfig(max_chunk_lines=10)
