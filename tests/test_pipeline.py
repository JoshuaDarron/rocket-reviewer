"""Tests for pipeline execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.errors import AgentError, PipelineError
from src.models import AgentReview
from src.pipeline import PipelineRunner


@pytest.fixture()
def pipeline_dir(tmp_path: Path) -> Path:
    """Create a temporary pipeline directory with a valid pipeline file."""
    pipeline_file = tmp_path / "full_review.json"
    pipeline_file.write_text(
        '{"name": "test", "nodes": [], "edges": []}',
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def valid_agent_response() -> dict[str, object]:
    """A valid single-agent response dict."""
    return {
        "reviewer": "claude-reviewer",
        "comments": [
            {
                "file": "src/main.py",
                "line": 10,
                "severity": "medium",
                "body": "Consider error handling here.",
            }
        ],
    }


class TestPipelineRunner:
    """Tests for pipeline loading and execution."""

    @pytest.mark.asyncio()
    async def test_pipeline_file_missing_raises_error(self, tmp_path: Path) -> None:
        runner = PipelineRunner(pipeline_dir=tmp_path)
        with pytest.raises(PipelineError, match="Pipeline file not found"):
            await runner.run_full_review(diff="some diff")

    @pytest.mark.asyncio()
    async def test_successful_execution(
        self,
        pipeline_dir: Path,
        valid_agent_response: dict[str, object],
    ) -> None:
        runner = PipelineRunner(pipeline_dir=pipeline_dir)

        mock_client = AsyncMock()
        mock_client.use = AsyncMock(return_value="token-123")
        mock_client.send = AsyncMock(return_value=valid_agent_response)
        mock_client.terminate = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("src.pipeline.RocketRideClient", return_value=mock_client):
            reviews = await runner.run_full_review(diff="diff content")

        assert len(reviews) == 1
        assert isinstance(reviews[0], AgentReview)
        assert reviews[0].reviewer == "claude-reviewer"
        assert len(reviews[0].comments) == 1

    @pytest.mark.asyncio()
    async def test_malformed_response_raises_agent_error(
        self, pipeline_dir: Path
    ) -> None:
        runner = PipelineRunner(pipeline_dir=pipeline_dir)

        malformed = {"reviewer": 12345, "comments": "not a list"}

        mock_client = AsyncMock()
        mock_client.use = AsyncMock(return_value="token-123")
        mock_client.send = AsyncMock(return_value=malformed)
        mock_client.terminate = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("src.pipeline.RocketRideClient", return_value=mock_client),
            pytest.raises(AgentError),
        ):
            await runner.run_full_review(diff="diff content")

    @pytest.mark.asyncio()
    async def test_sdk_error_raises_pipeline_error(self, pipeline_dir: Path) -> None:
        runner = PipelineRunner(pipeline_dir=pipeline_dir)

        mock_client = AsyncMock()
        mock_client.use = AsyncMock(side_effect=TimeoutError("SDK timeout"))
        mock_client.terminate = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("src.pipeline.RocketRideClient", return_value=mock_client),
            pytest.raises(PipelineError, match="Pipeline execution failed"),
        ):
            await runner.run_full_review(diff="diff content")

    @pytest.mark.asyncio()
    async def test_token_always_terminated(
        self,
        pipeline_dir: Path,
        valid_agent_response: dict[str, object],
    ) -> None:
        """Token is terminated even on successful runs."""
        runner = PipelineRunner(pipeline_dir=pipeline_dir)

        mock_client = AsyncMock()
        mock_client.use = AsyncMock(return_value="token-123")
        mock_client.send = AsyncMock(return_value=valid_agent_response)
        mock_client.terminate = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("src.pipeline.RocketRideClient", return_value=mock_client):
            await runner.run_full_review(diff="diff content")

        # Verify terminate was called (in the finally block)
        mock_client.terminate.assert_called_once_with("token-123")

    @pytest.mark.asyncio()
    async def test_list_response_multiple_agents(self, pipeline_dir: Path) -> None:
        """Pipeline response as a list produces multiple AgentReview objects."""
        runner = PipelineRunner(pipeline_dir=pipeline_dir)

        responses = [
            {"reviewer": "claude-reviewer", "comments": []},
            {"reviewer": "gpt-reviewer", "comments": []},
        ]

        mock_client = AsyncMock()
        mock_client.use = AsyncMock(return_value="token-123")
        mock_client.send = AsyncMock(return_value=responses)
        mock_client.terminate = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("src.pipeline.RocketRideClient", return_value=mock_client):
            reviews = await runner.run_full_review(diff="diff content")

        assert len(reviews) == 2

    @pytest.mark.asyncio()
    async def test_unexpected_response_type_raises_pipeline_error(
        self, pipeline_dir: Path
    ) -> None:
        runner = PipelineRunner(pipeline_dir=pipeline_dir)

        mock_client = AsyncMock()
        mock_client.use = AsyncMock(return_value="token-123")
        mock_client.send = AsyncMock(return_value="just a string")
        mock_client.terminate = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("src.pipeline.RocketRideClient", return_value=mock_client),
            pytest.raises(PipelineError, match="Unexpected pipeline response type"),
        ):
            await runner.run_full_review(diff="diff content")

    @pytest.mark.asyncio()
    async def test_conversation_reply_not_implemented(self) -> None:
        runner = PipelineRunner()
        with pytest.raises(NotImplementedError, match="Phase 3"):
            await runner.run_conversation_reply("claude-reviewer", "thread context")
