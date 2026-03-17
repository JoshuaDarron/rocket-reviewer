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
from src.errors import PipelineError
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
    ) -> tuple[list[AgentReview], list[str]]:
        """Run the full review pipeline.

        Args:
            diff: Unified diff of the pull request.
            file_context: Optional mapping of file paths to content.
            review_mode: Either ``"full"`` or ``"diff"``.

        Returns:
            A tuple of (valid AgentReview objects, names of failed agents).

        Raises:
            PipelineError: If the pipeline file is missing or execution
                fails.
        """
        pipeline_path = self._pipeline_dir / "full_review.json"
        if not pipeline_path.is_file():
            msg = f"Pipeline file not found: {pipeline_path}"
            raise PipelineError(msg)

        pipeline_def = pipeline_path.read_text(encoding="utf-8")

        input_data: dict[str, object] = {
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

    def _parse_response(self, response: object) -> tuple[list[AgentReview], list[str]]:
        """Parse and validate pipeline response into AgentReview objects.

        Fault-tolerant: malformed agent responses are logged and skipped
        rather than raising. The agent name is added to the failed list.

        Args:
            response: Raw response from the RocketRide SDK.

        Returns:
            A tuple of (valid AgentReview objects, names of failed agents).

        Raises:
            PipelineError: If the top-level response structure is
                unexpected (not a dict or list).
        """
        if isinstance(response, dict):
            results = [response]
        elif isinstance(response, list):
            results = response
        else:
            msg = f"Unexpected pipeline response type: {type(response).__name__}"
            raise PipelineError(msg)

        reviews: list[AgentReview] = []
        failed_agents: list[str] = []

        for result in results:
            if not isinstance(result, dict):
                logger.warning(
                    "Expected dict in pipeline results, got %s — skipping",
                    type(result).__name__,
                )
                failed_agents.append("unknown")
                continue

            reviewer_name = str(result.get("reviewer", "unknown"))
            try:
                review = AgentReview(**result)
            except (ValidationError, TypeError) as e:
                logger.warning(
                    "Invalid response from agent %s: %s — skipping",
                    reviewer_name,
                    e,
                )
                failed_agents.append(reviewer_name)
                continue

            reviews.append(review)

        return reviews, failed_agents

    async def run_conversation_reply(
        self,
        agent_node_id: str,
        thread_context: str,
        file_context: str = "",
    ) -> str:
        """Run the single-agent conversation reply pipeline.

        Loads the ``conversation_reply.json`` pipeline, sends thread
        context and optional file context, and returns the agent's reply.

        Args:
            agent_node_id: The RocketRide node ID of the responding agent
                (e.g., ``"claude-reviewer"``).
            thread_context: Formatted conversation thread text.
            file_context: Optional surrounding code context.

        Returns:
            The reply text from the agent.

        Raises:
            PipelineError: If the pipeline file is missing or execution
                fails.
        """
        pipeline_path = self._pipeline_dir / "conversation_reply.json"
        if not pipeline_path.is_file():
            msg = f"Pipeline file not found: {pipeline_path}"
            raise PipelineError(msg)

        pipeline_def = pipeline_path.read_text(encoding="utf-8")

        input_data: dict[str, str] = {
            "thread_context": thread_context,
            "agent_node_id": agent_node_id,
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
            msg = f"Conversation reply pipeline failed: {e}"
            raise PipelineError(msg) from e
        finally:
            if token is not None:
                try:
                    async with RocketRideClient(
                        f"http://localhost:{ENGINE_PORT}"
                    ) as client:
                        await client.terminate(token)
                except Exception:
                    logger.warning("Failed to terminate conversation pipeline token")

        return self._extract_reply(response)

    def _extract_reply(self, response: object) -> str:
        """Extract the reply text from a conversation pipeline response.

        Args:
            response: Raw response from the RocketRide SDK.

        Returns:
            The reply string.

        Raises:
            PipelineError: If the response structure is unexpected or
                the reply field is missing.
        """
        if not isinstance(response, dict):
            msg = (
                f"Unexpected conversation response type: " f"{type(response).__name__}"
            )
            raise PipelineError(msg)

        reply = response.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            msg = "Conversation response missing 'reply' field or reply is empty"
            raise PipelineError(msg)

        return reply.strip()
