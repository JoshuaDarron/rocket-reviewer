"""Pipeline execution for full review and conversation modes.

Loads pipeline JSON, starts pipelines via the RocketRide SDK, sends
diff data, and receives structured agent responses.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError
from rocketride import RocketRideClient

from src.config import ENGINE_PORT
from src.errors import AgentError, PipelineError
from src.models import AgentReview

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Executes RocketRide pipelines and collects agent responses.

    Args:
        pipeline_dir: Directory containing pipeline JSON files.
            Defaults to ``pipelines/`` relative to the project root.
    """

    def __init__(self, pipeline_dir: Path | None = None) -> None:
        if pipeline_dir is None:
            pipeline_dir = Path(__file__).resolve().parent.parent / "pipelines"
        self._pipeline_dir = pipeline_dir

    async def run_full_review(
        self,
        diff: str,
        file_context: dict[str, str] | None = None,
        review_mode: str = "full",
    ) -> list[AgentReview]:
        """Run the full review pipeline.

        Args:
            diff: Unified diff of the pull request.
            file_context: Optional mapping of file paths to content.
            review_mode: Either ``"full"`` or ``"diff"``.

        Returns:
            List of validated AgentReview objects (one per agent node).

        Raises:
            PipelineError: If the pipeline file is missing or execution
                fails.
            AgentError: If an individual agent produces invalid output.
        """
        pipeline_path = self._pipeline_dir / "full_review.json"
        if not pipeline_path.is_file():
            msg = f"Pipeline file not found: {pipeline_path}"
            raise PipelineError(msg)

        pipeline_def = pipeline_path.read_text(encoding="utf-8")

        input_data = {
            "diff": diff,
            "review_mode": review_mode,
        }
        if file_context:
            input_data["file_context"] = file_context

        token = None
        try:
            async with RocketRideClient(f"http://localhost:{ENGINE_PORT}") as client:
                token = await client.use(json.loads(pipeline_def))
                response = await client.send(token, input_data)
        except PipelineError:
            raise
        except Exception as e:
            msg = f"Pipeline execution failed: {e}"
            raise PipelineError(msg) from e
        finally:
            if token is not None:
                try:
                    async with RocketRideClient(
                        f"http://localhost:{ENGINE_PORT}"
                    ) as client:
                        await client.terminate(token)
                except Exception:
                    logger.warning("Failed to terminate pipeline token")

        return self._parse_response(response)

    def _parse_response(self, response: object) -> list[AgentReview]:
        """Parse and validate pipeline response into AgentReview objects.

        Args:
            response: Raw response from the RocketRide SDK.

        Returns:
            List of validated AgentReview objects.

        Raises:
            AgentError: If a node's response fails validation.
            PipelineError: If the response structure is unexpected.
        """
        # The response may be a single dict (Phase 1) or a list of dicts
        if isinstance(response, dict):
            results = [response]
        elif isinstance(response, list):
            results = response
        else:
            msg = f"Unexpected pipeline response type: {type(response).__name__}"
            raise PipelineError(msg)

        reviews: list[AgentReview] = []
        for result in results:
            if not isinstance(result, dict):
                msg = f"Expected dict in pipeline results, got {type(result).__name__}"
                raise PipelineError(msg)

            reviewer_name = result.get("reviewer", "unknown")
            try:
                review = AgentReview(**result)
            except (ValidationError, TypeError) as e:
                msg = f"Invalid response from agent {reviewer_name}: {e}"
                raise AgentError(msg, agent_name=str(reviewer_name)) from e

            reviews.append(review)

        return reviews

    async def run_conversation_reply(
        self,
        agent_node_id: str,
        thread_context: str,
    ) -> dict[str, object]:
        """Run the single-agent conversation reply pipeline.

        Not implemented in Phase 1.
        """
        raise NotImplementedError("Conversation reply is deferred to Phase 3")
