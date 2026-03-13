"""GitHub API wrapper for PR data, diffs, comments, and reviews.

Handles authentication via GitHub App installation tokens and provides
methods for fetching PR metadata, posting inline review comments, and
submitting review statuses.
"""

from __future__ import annotations

import logging

import httpx
from github import Auth, GithubIntegration

from src.errors import (
    CommentPostingError,
    ConfigurationError,
    DiffRetrievalError,
    ReviewSubmissionError,
)

logger = logging.getLogger(__name__)


class GitHubClient:
    """Wrapper around PyGithub for PR review operations.

    Args:
        app_id: GitHub App ID.
        private_key: GitHub App private key (PEM format).
        repo_name: Full repository name (e.g., ``owner/repo``).
        pr_number: Pull request number.
    """

    def __init__(
        self,
        app_id: int,
        private_key: str,
        repo_name: str,
        pr_number: int,
    ) -> None:
        try:
            auth = Auth.AppAuth(app_id, private_key)
            gi = GithubIntegration(auth=auth)
            installation = gi.get_installations()[0]
            self._gh = installation.get_github_for_installation()
        except Exception as e:
            msg = f"Failed to authenticate GitHub App {app_id}: {e}"
            raise ConfigurationError(msg) from e

        self._repo_name = repo_name
        self._pr_number = pr_number
        self._repo = self._gh.get_repo(repo_name)
        self._pr = self._repo.get_pull(pr_number)
        self._token = self._gh.requester.auth.token

    async def get_pr_diff(self) -> str:
        """Fetch the unified diff for a pull request.

        Returns:
            The raw unified diff string.

        Raises:
            DiffRetrievalError: If the diff cannot be fetched.
        """
        url = f"https://api.github.com/repos/{self._repo_name}/pulls/{self._pr_number}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github.v3.diff",
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30.0)
                response.raise_for_status()
                return response.text
        except httpx.HTTPError as e:
            msg = f"Failed to fetch diff for PR #{self._pr_number}: {e}"
            raise DiffRetrievalError(msg) from e

    async def get_pr_metadata(self) -> dict[str, object]:
        """Fetch PR metadata.

        Returns:
            Dict with ``target_branch``, ``author``, ``changed_files``,
            and ``head_sha``.
        """
        return {
            "target_branch": self._pr.base.ref,
            "author": self._pr.user.login,
            "changed_files": self._pr.changed_files,
            "head_sha": self._pr.head.sha,
        }

    async def get_file_content(self, path: str) -> str:
        """Fetch full content of a file at the PR's head ref.

        Args:
            path: Relative file path within the repository.

        Returns:
            The decoded file content as a string.
        """
        contents = self._repo.get_contents(path, ref=self._pr.head.sha)
        return contents.decoded_content.decode("utf-8")

    async def post_review_comment(
        self,
        body: str,
        path: str,
        line: int,
    ) -> None:
        """Post an inline review comment on a PR.

        Args:
            body: Comment body text.
            path: File path relative to the repository root.
            line: Line number in the diff.

        Raises:
            CommentPostingError: If the comment cannot be posted.
        """
        try:
            self._pr.create_review_comment(
                body=body,
                commit=self._repo.get_commit(self._pr.head.sha),
                path=path,
                line=line,
            )
        except Exception as e:
            msg = f"Failed to post comment on {path}:{line}: {e}"
            raise CommentPostingError(msg) from e

    async def submit_review(self, status: str, body: str) -> None:
        """Submit a review with the given status.

        Args:
            status: Review event — ``APPROVE``, ``REQUEST_CHANGES``,
                or ``COMMENT``.
            body: Review body/summary.

        Raises:
            ReviewSubmissionError: If the review cannot be submitted.
        """
        try:
            self._pr.create_review(body=body, event=status)
        except Exception as e:
            msg = f"Failed to submit review with status {status}: {e}"
            raise ReviewSubmissionError(msg) from e

    async def post_issue_comment(self, body: str) -> None:
        """Post a general comment on the PR (for summaries/errors).

        This is best-effort — failures are logged but not raised.

        Args:
            body: Comment body text.
        """
        try:
            self._pr.create_issue_comment(body)
        except Exception:
            logger.exception("Failed to post issue comment on PR #%d", self._pr_number)
