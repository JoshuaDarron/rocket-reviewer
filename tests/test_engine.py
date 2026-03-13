"""Tests for RocketRide engine lifecycle."""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.engine import EngineManager
from src.errors import EngineError


class TestEngineStart:
    """Tests for engine container startup."""

    @pytest.mark.asyncio()
    async def test_start_success(self) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "container123abc\n"

        with patch("src.engine.subprocess.run", return_value=mock_result) as mock_run:
            engine = EngineManager()
            await engine.start()
            assert engine._container_id == "container123abc"
            mock_run.assert_called_once()

    @pytest.mark.asyncio()
    async def test_start_failure_raises_engine_error(self) -> None:
        with patch(
            "src.engine.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "docker", stderr="no docker"),
        ):
            engine = EngineManager()
            with pytest.raises(EngineError, match="Failed to start"):
                await engine.start()


class TestEngineHealthCheck:
    """Tests for engine health polling."""

    @pytest.mark.asyncio()
    async def test_healthy_on_first_poll(self) -> None:
        engine = EngineManager()

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("src.engine.asyncio.sleep") as mock_sleep,
            patch("src.engine.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            await engine.wait_for_healthy()
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio()
    async def test_timeout_raises_engine_error(self) -> None:
        engine = EngineManager()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with (
            patch("src.engine.asyncio.sleep", new_callable=AsyncMock),
            patch("src.engine.time.monotonic") as mock_time,
            patch("src.engine.httpx.AsyncClient") as mock_client_cls,
        ):
            # Simulate time passing beyond timeout
            mock_time.side_effect = [0.0, 31.0]

            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(EngineError, match="did not become healthy"):
                await engine.wait_for_healthy()


class TestEngineStop:
    """Tests for engine container teardown."""

    @pytest.mark.asyncio()
    async def test_stop_calls_docker_stop_and_rm(self) -> None:
        engine = EngineManager()
        engine._container_id = "container123"

        with patch("src.engine.subprocess.run") as mock_run:
            await engine.stop()
            assert mock_run.call_count == 2
            calls = mock_run.call_args_list
            assert calls[0].args[0] == ["docker", "stop", "container123"]
            assert calls[1].args[0] == ["docker", "rm", "container123"]
            assert engine._container_id is None

    @pytest.mark.asyncio()
    async def test_stop_no_container_is_noop(self) -> None:
        engine = EngineManager()
        engine._container_id = None

        with patch("src.engine.subprocess.run") as mock_run:
            await engine.stop()
            mock_run.assert_not_called()

    @pytest.mark.asyncio()
    async def test_stop_logs_failure_but_does_not_raise(self) -> None:
        engine = EngineManager()
        engine._container_id = "container123"

        with patch(
            "src.engine.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "docker"),
        ):
            # Should not raise
            await engine.stop()


class TestEngineContextManager:
    """Tests for async context manager lifecycle."""

    @pytest.mark.asyncio()
    async def test_context_manager_starts_and_stops(self) -> None:
        with (
            patch.object(EngineManager, "start") as mock_start,
            patch.object(EngineManager, "wait_for_healthy") as mock_health,
            patch.object(EngineManager, "stop") as mock_stop,
        ):
            async with EngineManager() as engine:
                assert isinstance(engine, EngineManager)

            mock_start.assert_called_once()
            mock_health.assert_called_once()
            mock_stop.assert_called_once()

    @pytest.mark.asyncio()
    async def test_context_manager_stops_on_exception(self) -> None:
        with (
            patch.object(EngineManager, "start"),
            patch.object(EngineManager, "wait_for_healthy"),
            patch.object(EngineManager, "stop") as mock_stop,
        ):
            with pytest.raises(RuntimeError, match="test error"):
                async with EngineManager():
                    raise RuntimeError("test error")

            mock_stop.assert_called_once()
