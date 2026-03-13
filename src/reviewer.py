"""Review posting logic per GitHub App identity.

Takes agent reviews and posts each agent's comments under its own
GitHub App identity. Handles review status submission based on comment
severities.
"""

from __future__ import annotations

import logging

from src.errors import CommentPostingError
from src.github_client import GitHubClient
from src.models import AgentReview, ReviewComment, Severity

logger = logging.getLogger(__name__)

_SEVERITY_BADGES: dict[Severity, str] = {
    Severity.CRITICAL: "\U0001f6d1 **Critical**",
    Severity.HIGH: "\U0001f7e0 **High**",
    Severity.MEDIUM: "\U0001f7e1 **Medium**",
    Severity.LOW: "\U0001f535 **Low**",
    Severity.NITPICK: "\u2728 **Nitpick**",
}


def _format_comment_body(severity: Severity, body: str) -> str:
    """Prepend a severity badge to the comment body.

    Args:
        severity: Comment severity level.
        body: Original comment text.

    Returns:
        Formatted comment with severity badge prefix.
    """
    badge = _SEVERITY_BADGES.get(severity, severity.value)
    return f"{badge}: {body}"


def _determine_review_status(
    review: AgentReview,
    approval_threshold: str = "high",
) -> str:
    """Determine the review status based on comment severities.

    Args:
        review: The agent's review.
        approval_threshold: The minimum severity that blocks approval.
            Either ``"critical"`` or ``"high"``.

    Returns:
        One of ``"APPROVE"``, ``"REQUEST_CHANGES"``, or ``"COMMENT"``.
    """
    if not review.comments:
        return "APPROVE"

    blocking_severities: set[Severity] = {Severity.CRITICAL}
    if approval_threshold == "high":
        blocking_severities.add(Severity.HIGH)

    has_blocking = any(c.severity in blocking_severities for c in review.comments)
    if has_blocking:
        return "REQUEST_CHANGES"

    return "COMMENT"


def _build_review_summary(
    review: AgentReview,
    failed_comments: list[ReviewComment],
) -> str:
    """Build a markdown summary of the review.

    Args:
        review: The agent's review.
        failed_comments: Comments that could not be posted.

    Returns:
        Formatted markdown summary string.
    """
    severity_counts: dict[str, int] = {}
    for comment in review.comments:
        label = comment.severity.value
        severity_counts[label] = severity_counts.get(label, 0) + 1

    lines = [f"## Review by {review.reviewer}\n"]

    if not review.comments:
        lines.append("No issues found. Looks good!")
    else:
        lines.append(f"**{len(review.comments)} comment(s):**\n")
        for severity_name in ["critical", "high", "medium", "low", "nitpick"]:
            count = severity_counts.get(severity_name, 0)
            if count > 0:
                lines.append(f"- {severity_name}: {count}")

    if failed_comments:
        n = len(failed_comments)
        lines.append(f"\n\u26a0\ufe0f {n} comment(s) could not be posted inline.")

    return "\n".join(lines)


async def post_agent_review(
    review: AgentReview,
    github_client: GitHubClient,
    approval_threshold: str = "high",
) -> None:
    """Post a single agent's review comments and submit review status.

    Each comment is posted individually. If a comment fails to post, it
    is logged and skipped — remaining comments are still posted. A
    summary review is submitted at the end.

    Args:
        review: The agent's review to post.
        github_client: Authenticated GitHub client for this agent's app.
        approval_threshold: Severity threshold for blocking approval.
    """
    failed_comments: list[ReviewComment] = []

    for comment in review.comments:
        formatted_body = _format_comment_body(comment.severity, comment.body)
        try:
            await github_client.post_review_comment(
                body=formatted_body,
                path=comment.file,
                line=comment.line,
            )
        except CommentPostingError:
            logger.warning(
                "Failed to post comment on %s:%d", comment.file, comment.line
            )
            failed_comments.append(comment)

    status = _determine_review_status(review, approval_threshold)
    summary = _build_review_summary(review, failed_comments)

    try:
        await github_client.submit_review(status=status, body=summary)
    except Exception:
        logger.exception("Failed to submit review for %s", review.reviewer)
