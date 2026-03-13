"""Entry point: event detection, gating, and orchestration.

Reads the GitHub event payload, checks trigger conditions (target branch,
event type, comment author), and orchestrates the review pipeline. The
top-level handler catches all exceptions, logs errors, posts a summary
comment if possible, and always exits with code 0.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from src.config import load_config
from src.engine import EngineManager
from src.filters import get_effective_patterns, should_ignore
from src.github_client import GitHubClient
from src.models import ReviewConfig
from src.pipeline import PipelineRunner
from src.reviewer import post_agent_review

logger = logging.getLogger(__name__)


def should_run(event: dict[str, object], event_name: str, config: ReviewConfig) -> bool:
    """Check whether the review should proceed based on the event.

    Args:
        event: Parsed GitHub event payload.
        event_name: GitHub event name (e.g., ``pull_request``).
        config: Loaded review configuration.

    Returns:
        True if the review should run.
    """
    # Phase 1: only pull_request events
    if event_name != "pull_request":
        logger.info("Skipping: event type is '%s', not 'pull_request'", event_name)
        return False

    action = event.get("action")
    if action not in ("opened", "synchronize"):
        logger.info("Skipping: PR action is '%s'", action)
        return False

    pr = event.get("pull_request", {})
    if not isinstance(pr, dict):
        logger.info("Skipping: missing pull_request payload")
        return False

    base = pr.get("base", {})
    if not isinstance(base, dict):
        logger.info("Skipping: missing base branch info")
        return False

    target_branch = base.get("ref", "")
    if target_branch != config.target_branch:
        logger.info(
            "Skipping: target branch '%s' != configured '%s'",
            target_branch,
            config.target_branch,
        )
        return False

    return True


def _extract_changed_files(diff: str) -> list[str]:
    """Parse changed file paths from a unified diff.

    Args:
        diff: Raw unified diff string.

    Returns:
        List of file paths that were changed.
    """
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:])
    return files


async def _post_summary_comment(client: GitHubClient, message: str) -> None:
    """Post a summary comment on the PR. Best-effort."""
    try:
        await client.post_issue_comment(message)
    except Exception:
        logger.exception("Failed to post summary comment")


async def run() -> None:
    """Execute the review pipeline based on the GitHub event context.

    This is the top-level entry point. It catches all exceptions, logs
    errors, posts a summary comment on the PR if possible, and always
    exits with code 0 so that a failed review never blocks CI.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    client: GitHubClient | None = None

    try:
        # Load event payload
        event_path = os.environ.get("GITHUB_EVENT_PATH", "")
        event_name = os.environ.get("GITHUB_EVENT_NAME", "")

        if not event_path or not Path(event_path).is_file():
            logger.error("GITHUB_EVENT_PATH not set or file missing")
            return

        event = json.loads(Path(event_path).read_text(encoding="utf-8"))

        # Load config
        repo_root = Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd()))
        config = load_config(repo_root)

        # Gating
        if not should_run(event, event_name, config):
            return

        # Extract PR info from event
        pr_data = event.get("pull_request", {})
        repo_name = event.get("repository", {}).get("full_name", "")
        pr_number = pr_data.get("number", 0)

        if not repo_name or not pr_number:
            logger.error("Could not determine repo name or PR number from event")
            return

        # Initialize GitHub client for Claude reviewer
        app_id = int(os.environ.get("INPUT_CLAUDE_APP_ID", "0"))
        app_private_key = os.environ.get("INPUT_CLAUDE_APP_PRIVATE_KEY", "")
        anthropic_api_key = os.environ.get("INPUT_ANTHROPIC_API_KEY", "")

        if not app_id or not app_private_key:
            logger.error("Claude GitHub App credentials not configured")
            return

        if not anthropic_api_key:
            logger.error("Anthropic API key not configured")
            return

        client = GitHubClient(
            app_id=app_id,
            private_key=app_private_key,
            repo_name=repo_name,
            pr_number=pr_number,
        )

        # Fetch diff
        diff = await client.get_pr_diff()

        # Filter files
        changed_files = _extract_changed_files(diff)
        patterns = get_effective_patterns(
            extra=config.ignore_patterns_extra,
            override=config.ignore_patterns_override,
        )
        reviewed_files = [f for f in changed_files if not should_ignore(f, patterns)]

        if not reviewed_files:
            logger.info("All changed files are filtered out — skipping review")
            await _post_summary_comment(
                client,
                "All changed files match ignore patterns. No review performed.",
            )
            return

        # Check oversized PR
        total_lines = diff.count("\n")
        too_many_files = len(reviewed_files) > config.max_files
        too_many_lines = total_lines > config.max_total_lines
        if too_many_files or too_many_lines:
            msg = (
                f"PR is too large for automated review "
                f"({len(reviewed_files)} files, ~{total_lines} lines). "
                f"Limits: {config.max_files} files, {config.max_total_lines} lines."
            )
            logger.info(msg)
            await _post_summary_comment(client, msg)
            return

        # Fetch file context if in full mode
        file_context: dict[str, str] | None = None
        if config.review_context == "full":
            file_context = {}
            for file_path in reviewed_files:
                try:
                    content = await client.get_file_content(file_path)
                    file_context[file_path] = content
                except Exception:
                    logger.warning("Could not fetch content for %s", file_path)

        # Set Anthropic API key for pipeline
        os.environ.setdefault("ANTHROPIC_API_KEY", anthropic_api_key)

        # Start engine and run pipeline
        async with EngineManager() as _engine:
            runner = PipelineRunner()
            reviews = await runner.run_full_review(
                diff=diff,
                file_context=file_context,
                review_mode=config.review_context,
            )

        # Post reviews
        for review in reviews:
            await post_agent_review(
                review=review,
                github_client=client,
                approval_threshold=config.approval_threshold,
            )

        logger.info("Review complete")

    except Exception:
        logger.exception("Review failed with unexpected error")
        if client is not None:
            await _post_summary_comment(
                client,
                "\u26a0\ufe0f RocketRide Reviewer encountered an unexpected error. "
                "See workflow logs for details.",
            )


if __name__ == "__main__":
    asyncio.run(run())
