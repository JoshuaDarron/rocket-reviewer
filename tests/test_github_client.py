"""Tests for GitHub API interaction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from github import GithubException

from src.errors import (
    CommentPostingError,
    ConfigurationError,
    DiffRetrievalError,
    ReviewSubmissionError,
)
from src.github_client import GitHubClient


@pytest.fixture()
def mock_github_setup() -> MagicMock:
    """Patch PyGithub auth and return mock objects."""
    with (
        patch("src.github_client.Auth.AppAuth") as mock_auth_cls,
        patch("src.github_client.GithubIntegration") as mock_gi_cls,
    ):
        mock_auth = MagicMock()
        mock_auth_cls.return_value = mock_auth

        mock_installation = MagicMock()
        mock_gh = MagicMock()
        mock_gh.requester.auth.token = "fake-token"

        mock_repo = MagicMock()
        mock_pr = MagicMock()
        mock_pr.base.ref = "main"
        mock_pr.user.login = "testuser"
        mock_pr.changed_files = 3
        mock_pr.head.sha = "abc123"

        mock_repo.get_pull.return_value = mock_pr
        mock_gh.get_repo.return_value = mock_repo
        mock_installation.get_github_for_installation.return_value = mock_gh

        mock_gi = MagicMock()
        mock_gi.get_installations.return_value = [mock_installation]
        mock_gi_cls.return_value = mock_gi

        yield MagicMock(
            gh=mock_gh,
            repo=mock_repo,
            pr=mock_pr,
            auth_cls=mock_auth_cls,
            gi_cls=mock_gi_cls,
        )


class TestGitHubClientInit:
    """Tests for GitHubClient initialization."""

    def test_successful_init(self, mock_github_setup: MagicMock) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )
        assert client._pr_number == 42

    def test_empty_installations_raises_configuration_error(self) -> None:
        with (
            patch("src.github_client.Auth.AppAuth"),
            patch("src.github_client.GithubIntegration") as mock_gi_cls,
        ):
            mock_gi = MagicMock()
            mock_gi.get_installations.return_value = []
            mock_gi_cls.return_value = mock_gi

            with pytest.raises(ConfigurationError, match="has no installations"):
                GitHubClient(
                    app_id=12345,
                    private_key="fake-key",
                    repo_name="owner/repo",
                    pr_number=42,
                )

    def test_auth_failure_raises_configuration_error(self) -> None:
        with (
            patch(
                "src.github_client.Auth.AppAuth",
                side_effect=GithubException(401, "bad key", None),
            ),
            pytest.raises(ConfigurationError, match="Failed to authenticate"),
        ):
            GitHubClient(
                app_id=12345,
                private_key="bad-key",
                repo_name="owner/repo",
                pr_number=42,
            )


class TestGetPrDiff:
    """Tests for get_pr_diff()."""

    @pytest.mark.asyncio()
    async def test_fetch_diff_success(self, mock_github_setup: MagicMock) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )

        mock_response = MagicMock()
        mock_response.text = "diff --git a/file.py b/file.py\n+new line"
        mock_response.raise_for_status = MagicMock()

        mock_async_client = AsyncMock()
        mock_async_client.get = AsyncMock(return_value=mock_response)

        with patch("src.github_client.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_async_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            diff = await client.get_pr_diff()
            assert "diff --git" in diff

    @pytest.mark.asyncio()
    async def test_fetch_diff_failure(self, mock_github_setup: MagicMock) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )

        mock_async_client = AsyncMock()
        mock_async_client.get = AsyncMock(
            side_effect=httpx.HTTPError("connection failed")
        )

        with patch("src.github_client.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_async_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(DiffRetrievalError, match="Failed to fetch diff"):
                await client.get_pr_diff()


class TestGetPrMetadata:
    """Tests for get_pr_metadata()."""

    @pytest.mark.asyncio()
    async def test_metadata_fields(self, mock_github_setup: MagicMock) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )
        metadata = await client.get_pr_metadata()
        assert metadata["target_branch"] == "main"
        assert metadata["author"] == "testuser"
        assert metadata["changed_files"] == 3
        assert metadata["head_sha"] == "abc123"


class TestGetFileContent:
    """Tests for get_file_content()."""

    @pytest.mark.asyncio()
    async def test_fetch_file(self, mock_github_setup: MagicMock) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )
        mock_contents = MagicMock()
        mock_contents.decoded_content = b"file content here"
        mock_github_setup.repo.get_contents.return_value = mock_contents

        content = await client.get_file_content("src/main.py")
        assert content == "file content here"


class TestPostReviewComment:
    """Tests for post_review_comment()."""

    @pytest.mark.asyncio()
    async def test_post_comment_success(self, mock_github_setup: MagicMock) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )
        await client.post_review_comment(body="issue here", path="src/main.py", line=10)
        mock_github_setup.pr.create_review_comment.assert_called_once()

    @pytest.mark.asyncio()
    async def test_post_comment_failure(self, mock_github_setup: MagicMock) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )
        mock_github_setup.pr.create_review_comment.side_effect = GithubException(
            422, "API error", None
        )

        with pytest.raises(CommentPostingError, match="Failed to post comment"):
            await client.post_review_comment(
                body="issue here", path="src/main.py", line=10
            )


class TestSubmitReview:
    """Tests for submit_review()."""

    @pytest.mark.asyncio()
    async def test_submit_approve(self, mock_github_setup: MagicMock) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )
        await client.submit_review(status="APPROVE", body="LGTM")
        mock_github_setup.pr.create_review.assert_called_once_with(
            body="LGTM", event="APPROVE"
        )

    @pytest.mark.asyncio()
    async def test_submit_request_changes(self, mock_github_setup: MagicMock) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )
        await client.submit_review(status="REQUEST_CHANGES", body="Fix issues")
        mock_github_setup.pr.create_review.assert_called_once_with(
            body="Fix issues", event="REQUEST_CHANGES"
        )

    @pytest.mark.asyncio()
    async def test_submit_failure(self, mock_github_setup: MagicMock) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )
        mock_github_setup.pr.create_review.side_effect = GithubException(
            403, "forbidden", None
        )

        with pytest.raises(ReviewSubmissionError, match="Failed to submit review"):
            await client.submit_review(status="APPROVE", body="LGTM")


class TestPostIssueComment:
    """Tests for post_issue_comment()."""

    @pytest.mark.asyncio()
    async def test_post_issue_comment_success(
        self, mock_github_setup: MagicMock
    ) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )
        await client.post_issue_comment("Summary comment")
        mock_github_setup.pr.create_issue_comment.assert_called_once_with(
            "Summary comment"
        )

    @pytest.mark.asyncio()
    async def test_post_issue_comment_failure_is_silent(
        self, mock_github_setup: MagicMock
    ) -> None:
        client = GitHubClient(
            app_id=12345,
            private_key="fake-key",
            repo_name="owner/repo",
            pr_number=42,
        )
        mock_github_setup.pr.create_issue_comment.side_effect = GithubException(
            500, "oops", None
        )
        # Should not raise
        await client.post_issue_comment("Summary comment")
