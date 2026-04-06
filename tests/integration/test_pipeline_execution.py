"""Integration tests for pipeline execution mechanics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from rocketride.types.task import TASK_STATE

from src.config import LLM_PROVIDER_API_KEY_ENV
from src.errors import PipelineError
from src.pipeline import PipelineRunner

pytestmark = pytest.mark.integration


@pytest.fixture()
def pipeline_dir() -> Path:
    """Path to the real pipelines directory."""
    return Path(__file__).resolve().parent.parent.parent / "pipelines"


class TestPipelinePollCompletesAfterRetries:
    """Pipeline polling returns non-complete states before succeeding."""

    async def test_poll_completes_after_retries(self) -> None:
        """Multiple non-complete poll responses followed by success."""
        runner = PipelineRunner()

        completed_value = TASK_STATE.COMPLETED.value
        pending_status = {"state": 0, "name": "test-pipeline"}
        completed_status = {
            "state": completed_value,
            "name": "test-pipeline",
            "claude": {
                "reviewer": "claude-reviewer",
                "comments": [],
            },
        }

        mock_client = AsyncMock()
        mock_client.get_task_status = AsyncMock(
            side_effect=[pending_status, pending_status, completed_status]
        )

        with patch("src.pipeline.PIPELINE_POLL_INTERVAL", 0):
            result = await runner._poll_for_result(mock_client, "test-token")

        assert result["state"] == completed_value
        assert mock_client.get_task_status.await_count == 3


class TestPipelinePollErrorDetection:
    """Pipeline polling detects errors in the status response."""

    async def test_poll_error_raises_pipeline_error(self) -> None:
        """When the status contains errors, PipelineError is raised."""
        runner = PipelineRunner()

        error_status = {
            "state": 0,
            "errors": ["Agent timeout: claude model unavailable"],
        }

        mock_client = AsyncMock()
        mock_client.get_task_status = AsyncMock(return_value=error_status)

        with pytest.raises(PipelineError, match="Pipeline failed"):
            await runner._poll_for_result(mock_client, "test-token")

    async def test_poll_cancelled_raises_pipeline_error(self) -> None:
        """When the pipeline is cancelled, PipelineError is raised."""
        runner = PipelineRunner()

        cancelled_status = {"state": TASK_STATE.CANCELLED.value}

        mock_client = AsyncMock()
        mock_client.get_task_status = AsyncMock(return_value=cancelled_status)

        with pytest.raises(PipelineError, match="cancelled"):
            await runner._poll_for_result(mock_client, "test-token")


class TestPipelineApiKeyInjection:
    """Load real pipeline JSON and verify API key injection."""

    def test_full_review_pipeline_key_injection(
        self,
        pipeline_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """API keys are injected into the real full-review pipeline JSON."""
        pipeline_path = pipeline_dir / "full-review.pipe.json"
        if not pipeline_path.is_file():
            pytest.skip("Pipeline file not found")

        pipeline_data = json.loads(pipeline_path.read_text(encoding="utf-8"))

        # Set fake API keys in the environment
        for env_var in LLM_PROVIDER_API_KEY_ENV.values():
            monkeypatch.setenv(env_var, "test-api-key-12345")

        PipelineRunner._inject_api_keys(pipeline_data)

        # Verify keys were injected into all LLM components
        for component in pipeline_data.get("components", []):
            provider = component.get("provider", "")
            if provider not in LLM_PROVIDER_API_KEY_ENV:
                continue

            config = component.get("config", {})
            profile = config.get("profile", "")
            profile_config = config.get(profile, {})

            if "apikey" in profile_config:
                assert (
                    profile_config["apikey"] == "test-api-key-12345"
                ), f"API key not injected for provider {provider}"

    def test_missing_api_key_raises_pipeline_error(
        self,
        pipeline_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing API key env var raises PipelineError."""
        pipeline_path = pipeline_dir / "full-review.pipe.json"
        if not pipeline_path.is_file():
            pytest.skip("Pipeline file not found")

        pipeline_data = json.loads(pipeline_path.read_text(encoding="utf-8"))

        # Ensure API key env vars are NOT set
        for env_var in LLM_PROVIDER_API_KEY_ENV.values():
            monkeypatch.delenv(env_var, raising=False)

        # Check if any component has REPLACE_ME placeholder
        has_placeholder = False
        for component in pipeline_data.get("components", []):
            provider = component.get("provider", "")
            if provider not in LLM_PROVIDER_API_KEY_ENV:
                continue
            config = component.get("config", {})
            profile = config.get("profile", "")
            profile_config = config.get(profile, {})
            if profile_config.get("apikey") == "REPLACE_ME":
                has_placeholder = True
                break

        if not has_placeholder:
            pytest.skip("No REPLACE_ME placeholders found in pipeline")

        with pytest.raises(PipelineError, match="required"):
            PipelineRunner._inject_api_keys(pipeline_data)


class TestPipelineResponseParsing:
    """Verify response parsing handles various formats correctly."""

    def test_named_lane_response_parsed(self) -> None:
        """Named-lane response format is parsed into AgentReview objects."""
        runner = PipelineRunner()
        response = {
            "claude": {
                "comments": [
                    {
                        "file": "src/app.py",
                        "line": 10,
                        "severity": "high",
                        "body": "Issue found.",
                    }
                ],
            },
            "openai": {
                "comments": [],
            },
            "gemini": {
                "comments": [
                    {
                        "file": "src/utils.py",
                        "line": 5,
                        "severity": "low",
                        "body": "Suggestion.",
                    }
                ],
            },
        }

        reviews, failures = runner._parse_response(response)
        assert len(reviews) == 3
        assert len(failures) == 0
        assert reviews[0].reviewer == "claude-reviewer"
        assert reviews[1].reviewer == "gpt-reviewer"
        assert reviews[2].reviewer == "gemini-reviewer"

    def test_malformed_lane_skipped(self) -> None:
        """Malformed lane data is skipped, other lanes succeed."""
        runner = PipelineRunner()
        response = {
            "claude": {
                "comments": [
                    {
                        "file": "src/app.py",
                        "line": 10,
                        "severity": "high",
                        "body": "Issue.",
                    }
                ],
            },
            "openai": "not a dict",  # malformed
            "gemini": {
                "comments": [],
            },
        }

        reviews, failures = runner._parse_response(response)
        assert len(reviews) == 2
        assert "gpt-reviewer" in failures
