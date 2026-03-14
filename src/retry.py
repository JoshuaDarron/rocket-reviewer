"""Retry utility with exponential backoff for transient errors.

Provides a generic async retry wrapper used across the codebase for
HTTP calls and other operations that may experience transient failures.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default exception types considered retryable.
DEFAULT_RETRYABLE: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
)


async def with_retry(
    fn: Callable[[], Any],
    max_retries: int = 3,
    backoff_base: float = 1.0,
    retryable: tuple[type[Exception], ...] = DEFAULT_RETRYABLE,
) -> Any:
    """Execute *fn* with exponential backoff on retryable errors.

    The callable *fn* is invoked up to ``max_retries + 1`` times. On each
    retryable failure the function waits ``backoff_base * 2^attempt``
    seconds before the next attempt. Non-retryable exceptions propagate
    immediately.

    Args:
        fn: An async callable (no arguments) to execute.
        max_retries: Maximum number of retry attempts after the first call.
        backoff_base: Base wait time in seconds for exponential backoff.
        retryable: Tuple of exception types that should trigger a retry.

    Returns:
        The return value of *fn* on success.

    Raises:
        Exception: The last retryable exception if all attempts fail, or
            any non-retryable exception immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except retryable as exc:
            if attempt == max_retries:
                logger.error(
                    "All %d retry attempts exhausted. Last error: %s",
                    max_retries + 1,
                    exc,
                )
                raise
            wait = backoff_base * (2**attempt)
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs.",
                attempt + 1,
                max_retries + 1,
                exc,
                wait,
            )
            await asyncio.sleep(wait)

    # Unreachable, but satisfies type checkers.
    msg = "Retry loop exited unexpectedly"
    raise RuntimeError(msg)
