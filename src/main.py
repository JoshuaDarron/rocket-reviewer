"""Entry point: event detection, gating, and orchestration.

Reads the GitHub event payload, checks trigger conditions (target branch,
event type, comment author), and orchestrates the multi-agent review
pipeline or single-agent conversation reply. On failure, exits with
code 1 so the GitHub check accurately reflects the error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from src.aggregator import deduplicate_reviews
from src.config import AGENT_CREDENTIALS, AGENT_ROUTING, BOT_USERNAMES, load_config
from src.engine import EngineManager
from src.errors import ConfigurationError, EngineError
from src.filters import get_effective_patterns, should_ignore
from src.github_client import GitHubClient
from src.models import AgentReview, ReviewConfig, Severity
from src.pipeline import PipelineRunner
from src.reviewer import post_agent_review

logger = logging.getLogger(__name__)


def should_run(
    event: dict[str, Any], event_name: str, config: ReviewConfig
) -> str | None:
    """Check whether and how the action should proceed based on the event.

    Args:
        event: Parsed GitHub event payload.
        event_name: GitHub event name (e.g., ``pull_request``).
        config: Loaded review configuration.

    Returns:
        ``"full_review"`` for PR open/sync events, ``"conversation"``
        for review comment reply events, or ``None`` to skip.
    """
    if event_name == "pull_request":
        return _check_pull_request_event(event, config)

    if event_name == "pull_request_review_comment":
        return _check_review_comment_event(event)

    logger.info("Skipping: event type is '%s'", event_name)
    return None


def _check_pull_request_event(
    event: dict[str, Any], config: ReviewConfig
) -> str | None:
    """Check whether a pull_request event should trigger a full review.

    Args:
        event: Parsed GitHub event payload.
        config: Loaded review configuration.

    Returns:
        ``"full_review"`` if the event qualifies, otherwise ``None``.
    """
    action = event.get("action")
    if action not in ("opened", "synchronize"):
        logger.info("Skipping: PR action is '%s'", action)
        return None

    pr = event.get("pull_request", {})
    if not isinstance(pr, dict):
        logger.info("Skipping: missing pull_request payload")
        return None

    base = pr.get("base", {})
    if not isinstance(base, dict):
        logger.info("Skipping: missing base branch info")
        return None

    target_branch = base.get("ref", "")
    if target_branch != config.target_branch:
        logger.info(
            "Skipping: target branch '%s' != configured '%s'",
            target_branch,
            config.target_branch,
        )
        return None

    return "full_review"


def _check_review_comment_event(event: dict[str, Any]) -> str | None:
    """Check whether a pull_request_review_comment event should trigger a reply.

    Validates that the comment is a reply to a bot's review comment and
    that the commenter is not a bot (loop prevention).

    Args:
        event: Parsed GitHub event payload.

    Returns:
        ``"conversation"`` if the event qualifies, otherwise ``None``.
    """
    action = event.get("action")
    if action != "created":
        logger.info("Skipping: review comment action is '%s', not 'created'", action)
        return None

    comment = event.get("comment", {})
    if not isinstance(comment, dict):
        logger.info("Skipping: missing comment payload")
        return None

    # Loop prevention: ignore comments from bots
    commenter = comment.get("user", {})
    if not isinstance(commenter, dict):
        logger.info("Skipping: missing comment user info")
        return None

    commenter_login = commenter.get("login", "")
    if not commenter_login:
        logger.info("Skipping: comment has no user login")
        return None

    if commenter_login in BOT_USERNAMES:
        logger.info(
            "Skipping: comment author '%s' is a bot (loop prevention)",
            commenter_login,
        )
        return None

    # Must be a reply to an existing comment (has in_reply_to_id)
    in_reply_to_id = comment.get("in_reply_to_id")
    if in_reply_to_id is None:
        logger.info("Skipping: comment is not a reply (no in_reply_to_id)")
        return None

    return "conversation"


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


def _initialize_agents(
    repo_name: str,
    pr_number: int,
) -> tuple[dict[str, GitHubClient], list[str]]:
    """Initialize GitHub clients for all configured agents.

    For each agent in ``AGENT_CREDENTIALS``, reads environment variables
    for app ID, private key, and API key. On auth failure, logs a warning
    and adds the agent to the failures list.

    Args:
        repo_name: Full repository name (e.g., ``owner/repo``).
        pr_number: Pull request number.

    Returns:
        A tuple of (mapping of agent name to GitHubClient, list of
        agent names that failed to initialize).
    """
    clients: dict[str, GitHubClient] = {}
    failures: list[str] = []

    for cred in AGENT_CREDENTIALS:
        name = cred["name"]
        app_id_str = os.environ.get(cred["app_id_env"], "")
        private_key = os.environ.get(cred["key_env"], "")
        api_key = os.environ.get(cred["api_key_env"], "")

        if not app_id_str or not private_key:
            logger.warning("Credentials not configured for %s — skipping agent", name)
            failures.append(name)
            continue

        if not api_key:
            logger.warning("API key not configured for %s — skipping agent", name)
            failures.append(name)
            continue

        try:
            app_id = int(app_id_str)
            client = GitHubClient(
                app_id=app_id,
                private_key=private_key,
                repo_name=repo_name,
                pr_number=pr_number,
            )
        except (ConfigurationError, ValueError) as e:
            logger.warning("Failed to initialize %s: %s", name, e)
            failures.append(name)
            continue

        # Set the LLM API key in the environment for the pipeline
        os.environ.setdefault(cred["api_key_target"], api_key)
        clients[name] = client

    return clients, failures


def _initialize_single_agent(
    agent_name: str,
    repo_name: str,
    pr_number: int,
) -> GitHubClient | None:
    """Initialize a single agent's GitHub client.

    Args:
        agent_name: Agent name (e.g., ``"claude-reviewer"``).
        repo_name: Full repository name.
        pr_number: Pull request number.

    Returns:
        A GitHubClient for the agent, or ``None`` if initialization failed.
    """
    cred = None
    for c in AGENT_CREDENTIALS:
        if c["name"] == agent_name:
            cred = c
            break

    if cred is None:
        logger.warning("No credentials configuration for agent %s", agent_name)
        return None

    app_id_str = os.environ.get(cred["app_id_env"], "")
    private_key = os.environ.get(cred["key_env"], "")
    api_key = os.environ.get(cred["api_key_env"], "")

    if not app_id_str or not private_key:
        logger.warning("Credentials not configured for %s", agent_name)
        return None

    if not api_key:
        logger.warning("API key not configured for %s", agent_name)
        return None

    try:
        app_id = int(app_id_str)
        client = GitHubClient(
            app_id=app_id,
            private_key=private_key,
            repo_name=repo_name,
            pr_number=pr_number,
        )
    except (ConfigurationError, ValueError) as e:
        logger.warning("Failed to initialize %s: %s", agent_name, e)
        return None

    os.environ.setdefault(cred["api_key_target"], api_key)
    return client


def _determine_cross_agent_statuses(
    reviews: list[AgentReview],
    approval_threshold: str = "high",
) -> dict[str, str]:
    """Compute review statuses based on findings across all agents.

    If no agent found critical/high issues, all agents approve. If any
    agent found blocking issues, that agent requests changes while
    others post as comment.

    Args:
        reviews: List of all agent reviews.
        approval_threshold: Severity at or above which approval is
            blocked (``"critical"`` or ``"high"``).

    Returns:
        Mapping of reviewer name to review status string.
    """
    blocking_severities: set[Severity] = {Severity.CRITICAL}
    if approval_threshold == "high":
        blocking_severities.add(Severity.HIGH)

    # Determine which agents found blocking issues
    flagging_agents: set[str] = set()
    for review in reviews:
        for comment in review.comments:
            if comment.severity in blocking_severities:
                flagging_agents.add(review.reviewer)
                break

    statuses: dict[str, str] = {}
    if not flagging_agents:
        # No blocking issues anywhere — all approve
        for review in reviews:
            statuses[review.reviewer] = "APPROVE"
    else:
        # Some agents found issues
        for review in reviews:
            if review.reviewer in flagging_agents:
                statuses[review.reviewer] = "REQUEST_CHANGES"
            else:
                statuses[review.reviewer] = "COMMENT"

    return statuses


def _build_agent_failure_message(failed_agents: list[str]) -> str:
    """Build a human-readable message about agents that failed.

    Args:
        failed_agents: List of agent names that were unavailable.

    Returns:
        Formatted markdown string for posting as a PR comment.
    """
    agent_list = ", ".join(failed_agents)
    return (
        f"The following reviewer(s) were unavailable for this review: "
        f"{agent_list}. See workflow logs for details."
    )


async def _post_summary_comment(client: GitHubClient, message: str) -> None:
    """Post a summary comment on the PR. Best-effort."""
    try:
        await client.post_issue_comment(message)
    except Exception:
        logger.exception("Failed to post summary comment")


def _identify_target_agent(event: dict[str, Any]) -> tuple[str, int, int] | None:
    """Identify which agent should respond to a review comment reply.

    Looks at the ``in_reply_to_id`` on the comment to determine which
    bot authored the parent comment. Since we cannot fetch the parent
    comment author from the event payload alone, we return the
    ``in_reply_to_id`` so the caller can look it up via the API.

    Args:
        event: Parsed GitHub event payload for a
            ``pull_request_review_comment`` event.

    Returns:
        A tuple of (comment_body, in_reply_to_id, comment_id) or
        ``None`` if the event is malformed.
    """
    comment = event.get("comment", {})
    if not isinstance(comment, dict):
        return None

    in_reply_to_id = comment.get("in_reply_to_id")
    if in_reply_to_id is None:
        return None

    comment_body = comment.get("body", "")
    comment_id = comment.get("id", 0)

    if not comment_body or not comment_id:
        return None

    return str(comment_body), int(in_reply_to_id), int(comment_id)


def _format_thread_context(thread: list[dict[str, Any]]) -> str:
    """Format a comment thread into a readable conversation string.

    Args:
        thread: Ordered list of comment dicts from the GitHub API.

    Returns:
        Formatted conversation string.
    """
    parts: list[str] = []
    for comment in thread:
        user = comment.get("user", "unknown")
        body = comment.get("body", "")
        parts.append(f"**{user}**: {body}")
    return "\n\n---\n\n".join(parts)


async def _handle_conversation_reply(
    event: dict[str, Any],
    repo_name: str,
    pr_number: int,
) -> None:
    """Handle a conversation reply: detect agent, build context, run pipeline, post.

    Args:
        event: Parsed GitHub event payload.
        repo_name: Full repository name.
        pr_number: Pull request number.
    """
    info = _identify_target_agent(event)
    if info is None:
        logger.info("Could not extract reply info from event — skipping")
        return

    comment_body, in_reply_to_id, comment_id = info

    # We need any available agent client to look up the thread.
    # First, try to figure out the target agent by checking all review comments.
    # We'll initialize a "lookup" client from the first available agent.
    all_clients, _ = _initialize_agents(repo_name, pr_number)
    if not all_clients:
        msg = "No agents could be initialized — cannot handle conversation"
        logger.error(msg)
        raise ConfigurationError(msg)

    lookup_client = next(iter(all_clients.values()))

    # Fetch all review comments to find parent comment author
    review_comments = await lookup_client.get_review_comments()

    # Find the parent comment to identify which bot authored it
    parent_author: str | None = None
    parent_path: str = ""
    for rc in review_comments:
        if int(rc["id"]) == in_reply_to_id:
            parent_author = str(rc["user"])
            parent_path = str(rc.get("path", ""))
            break

    if parent_author is None:
        # The parent might be a thread root — walk up through the thread
        thread = await lookup_client.get_comment_thread(in_reply_to_id)
        if thread:
            root = thread[0]
            parent_author = str(root.get("user", ""))
            parent_path = str(root.get("path", ""))

    if parent_author is None or parent_author not in AGENT_ROUTING:
        logger.info(
            "Parent comment author '%s' is not a known bot — skipping",
            parent_author,
        )
        return

    agent_node_id = AGENT_ROUTING[parent_author]

    # Initialize the specific agent's client for posting the reply
    agent_client = all_clients.get(agent_node_id)
    if agent_client is None:
        agent_client = _initialize_single_agent(agent_node_id, repo_name, pr_number)

    if agent_client is None:
        logger.warning(
            "Agent %s could not be initialized — cannot reply", agent_node_id
        )
        return

    # Fetch thread context
    thread = await lookup_client.get_comment_thread(in_reply_to_id)
    thread_context = _format_thread_context(thread)

    # Fetch file context if available
    file_context = ""
    if parent_path:
        try:
            file_context = await lookup_client.get_file_content(parent_path)
        except Exception:
            logger.warning("Could not fetch file content for %s", parent_path)

    # Run conversation pipeline
    async with EngineManager() as _engine:
        runner = PipelineRunner()
        reply_text = await runner.run_conversation_reply(
            agent_node_id=agent_node_id,
            thread_context=thread_context,
            file_context=file_context,
        )

    # Post the reply under the correct agent identity
    await agent_client.post_reply_comment(in_reply_to_id, reply_text)
    logger.info("Posted conversation reply from %s", agent_node_id)


async def _handle_full_review(
    event: dict[str, Any],
    config: ReviewConfig,
) -> None:
    """Handle a full multi-agent review.

    Args:
        event: Parsed GitHub event payload.
        config: Loaded review configuration.
    """
    pr_data = event.get("pull_request", {})
    repo_name = event.get("repository", {}).get("full_name", "")
    pr_number = pr_data.get("number", 0)
    if not repo_name or not pr_number:
        msg = "Could not determine repo name or PR number from event"
        logger.error(msg)
        raise RuntimeError(msg)

    # Initialize all agents
    agent_clients, agent_failures = _initialize_agents(repo_name, pr_number)

    if not agent_clients:
        msg = "No agents could be initialized — aborting review"
        logger.error(msg)
        raise ConfigurationError(msg)

    # Use first available client as primary for diff fetching
    primary_client = next(iter(agent_clients.values()))

    # Fetch diff
    diff = await primary_client.get_pr_diff()

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
            primary_client,
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
        await _post_summary_comment(primary_client, msg)
        return

    # Fetch file context if in full mode
    file_context: dict[str, str] | None = None
    if config.review_context == "full":
        file_context = {}
        for file_path in reviewed_files:
            try:
                content = await primary_client.get_file_content(file_path)
                file_context[file_path] = content
            except Exception:
                logger.warning("Could not fetch content for %s", file_path)

    # Start engine and run pipeline
    try:
        async with EngineManager() as _engine:
            runner = PipelineRunner()
            reviews, pipeline_failures = await runner.run_full_review(
                diff=diff,
                file_context=file_context,
                review_mode=config.review_context,
            )
    except EngineError as e:
        logger.error("Engine failure: %s", e)
        await _post_summary_comment(
            primary_client,
            "RocketRide engine could not start. Review was not performed. "
            "See workflow logs for details.",
        )
        raise
    except Exception as e:
        logger.error("Pipeline failure: %s", e)
        await _post_summary_comment(
            primary_client,
            "RocketRide Reviewer encountered an error during review. "
            "See workflow logs for details.",
        )
        raise

    # Merge pipeline failures into agent failures
    agent_failures.extend(pipeline_failures)

    # Deduplicate
    reviews = deduplicate_reviews(reviews)

    # Compute cross-agent review statuses
    statuses = _determine_cross_agent_statuses(reviews, config.approval_threshold)

    # Post each review under its own app identity
    for review in reviews:
        client = agent_clients.get(review.reviewer)
        if client is None:
            logger.warning(
                "No client for %s — skipping review posting", review.reviewer
            )
            continue

        try:
            await post_agent_review(
                review=review,
                github_client=client,
                approval_threshold=config.approval_threshold,
                review_status=statuses.get(review.reviewer),
            )
        except Exception:
            logger.exception("Failed to post review for %s", review.reviewer)

    # Report agent failures
    if agent_failures:
        failure_msg = _build_agent_failure_message(agent_failures)
        await _post_summary_comment(primary_client, failure_msg)

    logger.info("Review complete")


async def run() -> None:
    """Execute the review action based on the GitHub event.

    This is the top-level entry point. On failure, logs the error and
    exits with code 1 so the GitHub check accurately reflects the result.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        # Load event payload
        event_path = os.environ.get("GITHUB_EVENT_PATH", "")
        event_name = os.environ.get("GITHUB_EVENT_NAME", "")

        if not event_path or not Path(event_path).is_file():
            logger.error("GITHUB_EVENT_PATH not set or file missing")
            sys.exit(1)

        event = json.loads(Path(event_path).read_text(encoding="utf-8"))

        # Load config
        repo_root = Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd()))
        config = load_config(repo_root)

        # Gating
        mode = should_run(event, event_name, config)
        if mode is None:
            return

        if mode == "full_review":
            await _handle_full_review(event, config)
        elif mode == "conversation":
            repo_name = event.get("repository", {}).get("full_name", "")
            pr_number = event.get("pull_request", {}).get("number", 0)
            # For review comment events, PR number may be in the payload differently
            if not pr_number:
                issue = event.get("issue", {})
                if isinstance(issue, dict):
                    pr_number = issue.get("number", 0)

            # Also try the pull_request key at top level
            if not pr_number:
                pr_data = event.get("pull_request", {})
                if isinstance(pr_data, dict):
                    pr_number = pr_data.get("number", 0)

            if not repo_name or not pr_number:
                logger.error(
                    "Could not determine repo name or PR number for conversation"
                )
                sys.exit(1)

            await _handle_conversation_reply(event, repo_name, int(pr_number))
        else:
            logger.error("Unknown mode: %s", mode)
            sys.exit(1)

    except Exception:
        logger.exception("Review failed with unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run())
