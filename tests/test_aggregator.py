"""Tests for deduplication logic."""

from __future__ import annotations

import pytest

from src.aggregator import (
    LINE_WINDOW,
    SIMILARITY_THRESHOLD,
    _is_duplicate,
    deduplicate_reviews,
)
from src.config import DEDUP_LINE_WINDOW, DEDUP_SIMILARITY_THRESHOLD
from src.models import AgentReview, ReviewComment, Severity


class TestIsDuplicate:
    """Tests for the pairwise duplicate check."""

    def test_exact_duplicate(self) -> None:
        a = ReviewComment(file="a.py", line=10, severity=Severity.HIGH, body="Bug here")
        b = ReviewComment(file="a.py", line=10, severity=Severity.HIGH, body="Bug here")
        assert _is_duplicate(a, b)

    def test_near_duplicate_within_line_window(self) -> None:
        a = ReviewComment(file="a.py", line=10, severity=Severity.HIGH, body="Bug here")
        b = ReviewComment(
            file="a.py", line=10 + LINE_WINDOW, severity=Severity.HIGH, body="Bug here"
        )
        assert _is_duplicate(a, b)

    def test_beyond_line_window_not_duplicate(self) -> None:
        a = ReviewComment(file="a.py", line=10, severity=Severity.HIGH, body="Bug here")
        b = ReviewComment(
            file="a.py",
            line=10 + LINE_WINDOW + 1,
            severity=Severity.HIGH,
            body="Bug here",
        )
        assert not _is_duplicate(a, b)

    def test_different_file_not_duplicate(self) -> None:
        a = ReviewComment(file="a.py", line=10, severity=Severity.HIGH, body="Bug here")
        b = ReviewComment(file="b.py", line=10, severity=Severity.HIGH, body="Bug here")
        assert not _is_duplicate(a, b)

    def test_same_line_low_similarity_not_duplicate(self) -> None:
        a = ReviewComment(
            file="a.py",
            line=10,
            severity=Severity.HIGH,
            body="This variable should be renamed for clarity.",
        )
        b = ReviewComment(
            file="a.py",
            line=10,
            severity=Severity.MEDIUM,
            body="SQL injection vulnerability detected in query builder.",
        )
        assert not _is_duplicate(a, b)

    def test_similar_text_above_threshold(self) -> None:
        a = ReviewComment(
            file="a.py",
            line=10,
            severity=Severity.MEDIUM,
            body="Consider using a more descriptive variable name here.",
        )
        b = ReviewComment(
            file="a.py",
            line=11,
            severity=Severity.MEDIUM,
            body="Consider using a more descriptive variable name.",
        )
        assert _is_duplicate(a, b)


class TestDeduplicateReviews:
    """Tests for the full deduplication pipeline."""

    def test_exact_duplicates_removed(self) -> None:
        """Exact same comment across two agents — one copy removed."""
        comment = ReviewComment(
            file="a.py", line=10, severity=Severity.HIGH, body="Bug here"
        )
        reviews = [
            AgentReview(reviewer="claude-reviewer", comments=[comment]),
            AgentReview(reviewer="gpt-reviewer", comments=[comment.model_copy()]),
        ]
        result = deduplicate_reviews(reviews)
        total_comments = sum(len(r.comments) for r in result)
        assert total_comments == 1

    def test_near_duplicates_within_window_removed(self) -> None:
        reviews = [
            AgentReview(
                reviewer="claude-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py",
                        line=10,
                        severity=Severity.MEDIUM,
                        body="Consider renaming this variable for clarity.",
                    )
                ],
            ),
            AgentReview(
                reviewer="gpt-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py",
                        line=12,
                        severity=Severity.MEDIUM,
                        body="Consider renaming this variable for clarity please.",
                    )
                ],
            ),
        ]
        result = deduplicate_reviews(reviews)
        total_comments = sum(len(r.comments) for r in result)
        assert total_comments == 1

    def test_non_duplicates_preserved_different_file(self) -> None:
        reviews = [
            AgentReview(
                reviewer="claude-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py", line=10, severity=Severity.HIGH, body="Bug here"
                    )
                ],
            ),
            AgentReview(
                reviewer="gpt-reviewer",
                comments=[
                    ReviewComment(
                        file="b.py", line=10, severity=Severity.HIGH, body="Bug here"
                    )
                ],
            ),
        ]
        result = deduplicate_reviews(reviews)
        total_comments = sum(len(r.comments) for r in result)
        assert total_comments == 2

    def test_non_duplicates_preserved_beyond_window(self) -> None:
        reviews = [
            AgentReview(
                reviewer="claude-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py", line=10, severity=Severity.HIGH, body="Bug here"
                    )
                ],
            ),
            AgentReview(
                reviewer="gpt-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py", line=20, severity=Severity.HIGH, body="Bug here"
                    )
                ],
            ),
        ]
        result = deduplicate_reviews(reviews)
        total_comments = sum(len(r.comments) for r in result)
        assert total_comments == 2

    def test_keeps_longer_version(self) -> None:
        short_body = "Consider renaming this variable for clarity."
        long_body = "Consider renaming this variable for clarity and readability."
        reviews = [
            AgentReview(
                reviewer="claude-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py", line=10, severity=Severity.HIGH, body=short_body
                    )
                ],
            ),
            AgentReview(
                reviewer="gpt-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py", line=10, severity=Severity.HIGH, body=long_body
                    )
                ],
            ),
        ]
        result = deduplicate_reviews(reviews)
        # The longer version from gpt-reviewer should be kept
        total_comments = sum(len(r.comments) for r in result)
        assert total_comments == 1
        assert result[1].comments[0].body == long_body

    def test_single_review_passthrough(self) -> None:
        review = AgentReview(
            reviewer="claude-reviewer",
            comments=[
                ReviewComment(file="a.py", line=10, severity=Severity.HIGH, body="Bug")
            ],
        )
        result = deduplicate_reviews([review])
        assert len(result) == 1
        assert len(result[0].comments) == 1

    def test_empty_reviews_handled(self) -> None:
        result = deduplicate_reviews([])
        assert result == []

    def test_three_agents_same_comment_two_removed(self) -> None:
        """Three agents flag the same issue — two copies removed."""
        body = "This line has a security vulnerability."
        reviews = [
            AgentReview(
                reviewer="claude-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py", line=10, severity=Severity.CRITICAL, body=body
                    )
                ],
            ),
            AgentReview(
                reviewer="gpt-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py", line=10, severity=Severity.CRITICAL, body=body
                    )
                ],
            ),
            AgentReview(
                reviewer="gemini-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py", line=10, severity=Severity.CRITICAL, body=body
                    )
                ],
            ),
        ]
        result = deduplicate_reviews(reviews)
        total_comments = sum(len(r.comments) for r in result)
        assert total_comments == 1

    def test_all_unique_no_removal(self) -> None:
        reviews = [
            AgentReview(
                reviewer="claude-reviewer",
                comments=[
                    ReviewComment(
                        file="a.py", line=10, severity=Severity.HIGH, body="Issue A"
                    )
                ],
            ),
            AgentReview(
                reviewer="gpt-reviewer",
                comments=[
                    ReviewComment(
                        file="b.py",
                        line=20,
                        severity=Severity.MEDIUM,
                        body="Completely different issue B",
                    )
                ],
            ),
            AgentReview(
                reviewer="gemini-reviewer",
                comments=[
                    ReviewComment(
                        file="c.py",
                        line=30,
                        severity=Severity.LOW,
                        body="Yet another unrelated suggestion C",
                    )
                ],
            ),
        ]
        result = deduplicate_reviews(reviews)
        total_comments = sum(len(r.comments) for r in result)
        assert total_comments == 3

    def test_reviews_with_empty_comments_unchanged(self) -> None:
        reviews = [
            AgentReview(reviewer="claude-reviewer", comments=[]),
            AgentReview(reviewer="gpt-reviewer", comments=[]),
        ]
        result = deduplicate_reviews(reviews)
        assert len(result) == 2
        assert all(len(r.comments) == 0 for r in result)


class TestDedupConfigConstants:
    """Tests that aggregator constants come from config."""

    def test_line_window_matches_config(self) -> None:
        assert LINE_WINDOW == DEDUP_LINE_WINDOW

    def test_similarity_threshold_matches_config(self) -> None:
        assert SIMILARITY_THRESHOLD == DEDUP_SIMILARITY_THRESHOLD

    def test_custom_line_window_changes_behavior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Monkeypatching config changes dedup behavior."""
        # With a window of 0, same-file comments 2 lines apart are NOT duplicates
        monkeypatch.setattr("src.aggregator.LINE_WINDOW", 0)
        a = ReviewComment(file="a.py", line=10, severity=Severity.HIGH, body="Bug here")
        b = ReviewComment(file="a.py", line=12, severity=Severity.HIGH, body="Bug here")
        assert not _is_duplicate(a, b)
