"""GitHub API wrapper for PR data, diffs, comments, and reviews.

Handles authentication via GitHub App installation tokens and provides
methods for fetching PR metadata, posting inline review comments, and
submitting review statuses.
"""

from __future__ import annotations

import logging
from typing import Any

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
        auth = self._gh.requester.auth
        if auth is None:
            msg = f"GitHub App {app_id} authentication returned no auth token"
            raise ConfigurationError(msg)
        self._token: str = auth.token

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

    async def get_pr_metadata(self) -> dict[str, Any]:
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
        if isinstance(contents, list):
            msg = f"Expected a single file at {path}, got a directory listing"
            raise DiffRetrievalError(msg)
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

    async def get_review_comments(self) -> list[dict[str, Any]]:
        """Fetch all review comments on the pull request.

        Returns:
            List of comment dicts with ``id``, ``user``, ``body``,
            ``path``, ``line``, and ``in_reply_to_id`` fields.
        """
        comments: list[dict[str, Any]] = []
        for comment in self._pr.get_review_comments():
            comments.append(
                {
                    "id": comment.id,
                    "user": comment.user.login,
                    "body": comment.body,
                    "path": getattr(comment, "path", ""),
                    "line": getattr(comment, "line", 0),
                    "in_reply_to_id": getattr(comment, "in_reply_to_id", None),
                }
            )
        return comments

    async def get_comment_thread(self, comment_id: int) -> list[dict[str, Any]]:
        """Fetch the comment thread for a given review comment.

        Retrieves the parent comment and all replies in the same thread.

        Args:
            comment_id: The ID of a comment in the thread.

        Returns:
            Ordered list of comment dicts forming the thread.
        """
        all_comments = await self.get_review_comments()

        # Find the root comment (the one without in_reply_to_id, or the
        # target of in_reply_to_id chains)
        comment_map: dict[int, dict[str, Any]] = {
            int(c["id"]): c for c in all_comments
        }

        # Find the root of the thread containing comment_id
        target = comment_map.get(comment_id)
        if target is None:
            return []

        # Walk up to find the root
        root_id = comment_id
        visited: set[int] = set()
        while True:
            current = comment_map.get(root_id)
            if current is None:
                break
            parent_id = current.get("in_reply_to_id")
            if parent_id is None or int(parent_id) not in comment_map:
                break
            if root_id in visited:
                break
            visited.add(root_id)
            root_id = int(parent_id)

        # Collect all comments in this thread (root + replies to root)
        thread: list[dict[str, Any]] = []
        if root_id in comment_map:
            thread.append(comment_map[root_id])
        for c in all_comments:
            reply_to = c.get("in_reply_to_id")
            c_id = int(c["id"])
            if reply_to is not None and int(reply_to) == root_id and c_id != root_id:
                thread.append(c)

        return thread

    async def post_reply_comment(self, comment_id: int, body: str) -> None:
        """Post a reply to an existing review comment thread.

        Args:
            comment_id: The ID of the comment to reply to.
            body: Reply body text.

        Raises:
            CommentPostingError: If the reply cannot be posted.
        """
        try:
            self._pr.create_review_comment_reply(comment_id, body)
        except Exception as e:
            msg = f"Failed to post reply to comment {comment_id}: {e}"
            raise CommentPostingError(msg) from e

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
