"""Tests for the async retry utility."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.retry import with_retry


class TestWithRetry:
    """Tests for with_retry exponential backoff logic."""

    @pytest.mark.asyncio
    async def test_succeeds_first_attempt(self) -> None:
        """Function that succeeds on the first call returns immediately."""
        fn = AsyncMock(return_value="ok")
        result = await with_retry(fn)
        assert result == "ok"
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self) -> None:
        """Retries on TimeoutException and succeeds on second attempt."""
        fn = AsyncMock(side_effect=[httpx.TimeoutException("timeout"), "recovered"])
        with patch("src.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await with_retry(fn, backoff_base=0.01)
        assert result == "recovered"
        assert fn.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_network_error(self) -> None:
        """Retries on NetworkError."""
        fn = AsyncMock(side_effect=[httpx.NetworkError("network"), "ok"])
        with patch("src.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await with_retry(fn, backoff_base=0.01)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_exhausts_retries_then_raises(self) -> None:
        """After max_retries failures, the last exception is raised."""
        fn = AsyncMock(side_effect=httpx.TimeoutException("always fails"))
        with (
            patch("src.retry.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(httpx.TimeoutException, match="always fails"),
        ):
            await with_retry(fn, max_retries=2, backoff_base=0.01)
        # 1 initial + 2 retries = 3 total calls
        assert fn.call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception_propagates_immediately(self) -> None:
        """A non-retryable exception is not retried."""
        fn = AsyncMock(side_effect=ValueError("not retryable"))
        with pytest.raises(ValueError, match="not retryable"):
            await with_retry(fn)
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_custom_retryable_exceptions(self) -> None:
        """Custom retryable tuple is respected."""
        fn = AsyncMock(side_effect=[ValueError("retry me"), "ok"])
        with patch("src.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await with_retry(
                fn,
                retryable=(ValueError,),
                backoff_base=0.01,
            )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self) -> None:
        """Sleep durations follow exponential backoff."""
        fn = AsyncMock(
            side_effect=[
                httpx.TimeoutException("1"),
                httpx.TimeoutException("2"),
                httpx.TimeoutException("3"),
                "ok",
            ]
        )
        sleep_mock = AsyncMock()
        with patch("src.retry.asyncio.sleep", sleep_mock):
            result = await with_retry(fn, max_retries=3, backoff_base=1.0)
        assert result == "ok"
        # Backoff: 1*2^0=1, 1*2^1=2, 1*2^2=4
        assert sleep_mock.call_count == 3
        assert sleep_mock.call_args_list[0][0][0] == pytest.approx(1.0)
        assert sleep_mock.call_args_list[1][0][0] == pytest.approx(2.0)
        assert sleep_mock.call_args_list[2][0][0] == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_zero_retries(self) -> None:
        """max_retries=0 means only one attempt."""
        fn = AsyncMock(side_effect=httpx.TimeoutException("fail"))
        with (
            patch("src.retry.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(httpx.TimeoutException),
        ):
            await with_retry(fn, max_retries=0)
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_return_value_preserved(self) -> None:
        """Complex return values are passed through unchanged."""
        expected = {"key": [1, 2, 3], "nested": {"a": True}}
        fn = AsyncMock(return_value=expected)
        result = await with_retry(fn)
        assert result == expected
