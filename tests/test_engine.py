"""Tests for RocketRide engine lifecycle."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.engine import EngineManager
from src.errors import EngineError


class TestEngineDownloadAndExtract:
    """Tests for server binary download and extraction."""

    @pytest.mark.asyncio()
    async def test_skips_download_when_binary_exists(self, tmp_path: Path) -> None:
        engine = EngineManager()
        engine._binary_dir = tmp_path
        # Create a dummy file so the directory is non-empty
        (tmp_path / "rocketride-server").touch()

        result = await engine._download_and_extract()
        assert result == tmp_path

    @pytest.mark.asyncio()
    async def test_downloads_and_extracts_tarball(self, tmp_path: Path) -> None:
        engine = EngineManager()
        binary_dir = tmp_path / "server"
        engine._binary_dir = binary_dir

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-tarball-content"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        mock_tar = MagicMock()

        with (
            patch("src.engine.httpx.AsyncClient") as mock_client_cls,
            patch("src.engine.tarfile.open", return_value=mock_tar) as mock_tar_open,
        ):
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await engine._download_and_extract()
            assert result == binary_dir
            assert binary_dir.exists()
            mock_client.get.assert_called_once()
            mock_tar_open.assert_called_once()
            mock_tar.__enter__.return_value.extractall.assert_called_once()

    @pytest.mark.asyncio()
    async def test_download_failure_raises_engine_error(self, tmp_path: Path) -> None:
        engine = EngineManager()
        engine._binary_dir = tmp_path / "server"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "404", request=MagicMock(), response=MagicMock()
            )
        )

        with patch("src.engine.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(EngineError, match="Failed to download"):
                await engine._download_and_extract()

    @pytest.mark.asyncio()
    async def test_extraction_failure_raises_engine_error(self, tmp_path: Path) -> None:
        engine = EngineManager()
        binary_dir = tmp_path / "server"
        engine._binary_dir = binary_dir

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"bad-data"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("src.engine.httpx.AsyncClient") as mock_client_cls,
            patch("src.engine.tarfile.open", side_effect=OSError("corrupt")),
        ):
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(EngineError, match="Failed to extract"):
                await engine._download_and_extract()


class TestEngineFindBinary:
    """Tests for locating the server binary."""

    def test_finds_binary(self, tmp_path: Path) -> None:
        engine = EngineManager()
        engine._binary_dir = tmp_path
        binary = tmp_path / "rocketride-server"
        binary.touch()

        result = engine._find_binary()
        assert result == binary

    def test_finds_binary_in_subdirectory(self, tmp_path: Path) -> None:
        engine = EngineManager()
        engine._binary_dir = tmp_path
        subdir = tmp_path / "bin"
        subdir.mkdir()
        binary = subdir / "rocketride-server"
        binary.touch()

        result = engine._find_binary()
        assert result == binary

    def test_raises_when_no_binary_found(self, tmp_path: Path) -> None:
        engine = EngineManager()
        engine._binary_dir = tmp_path

        with pytest.raises(EngineError, match="binary not found"):
            engine._find_binary()


class TestEngineStart:
    """Tests for engine server startup."""

    @pytest.mark.asyncio()
    async def test_start_success(self, tmp_path: Path) -> None:
        engine = EngineManager()
        engine._binary_dir = tmp_path
        binary = tmp_path / "rocketride-server"
        binary.touch()

        mock_process = MagicMock()
        mock_process.pid = 12345

        with (
            patch.object(engine, "_download_and_extract", new_callable=AsyncMock),
            patch(
                "src.engine.subprocess.Popen", return_value=mock_process
            ) as mock_popen,
        ):
            await engine.start()
            assert engine._process is mock_process
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args
            assert str(binary) in call_args.args[0]

    @pytest.mark.asyncio()
    async def test_start_failure_raises_engine_error(self, tmp_path: Path) -> None:
        engine = EngineManager()
        engine._binary_dir = tmp_path
        binary = tmp_path / "rocketride-server"
        binary.touch()

        with (
            patch.object(engine, "_download_and_extract", new_callable=AsyncMock),
            patch(
                "src.engine.subprocess.Popen",
                side_effect=OSError("permission denied"),
            ),
            pytest.raises(EngineError, match="Failed to start"),
        ):
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
    """Tests for engine server teardown."""

    @pytest.mark.asyncio()
    async def test_stop_terminates_process(self) -> None:
        engine = EngineManager()
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.terminate = MagicMock()
        mock_process.wait = MagicMock()
        engine._process = mock_process

        await engine.stop()
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        assert engine._process is None

    @pytest.mark.asyncio()
    async def test_stop_kills_after_timeout(self) -> None:
        engine = EngineManager()
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.terminate = MagicMock()
        mock_process.wait = MagicMock(
            side_effect=[subprocess.TimeoutExpired("cmd", 5), None]
        )
        mock_process.kill = MagicMock()
        engine._process = mock_process

        await engine.stop()
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        assert engine._process is None

    @pytest.mark.asyncio()
    async def test_stop_no_process_is_noop(self) -> None:
        engine = EngineManager()
        engine._process = None
        # Should not raise
        await engine.stop()

    @pytest.mark.asyncio()
    async def test_stop_logs_failure_but_does_not_raise(self) -> None:
        engine = EngineManager()
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.terminate = MagicMock(side_effect=OSError("no such process"))
        engine._process = mock_process

        # Should not raise
        await engine.stop()
        assert engine._process is None


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
