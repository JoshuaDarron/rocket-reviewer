"""RocketRide engine lifecycle management.

Handles downloading the server binary, launching it as a subprocess,
health check polling, and teardown. Exposed as an async context manager.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from types import TracebackType

import httpx

from src.config import (
    ENGINE_BINARY_DIR,
    ENGINE_DOWNLOAD_URL,
    ENGINE_HEALTH_CHECK_INTERVAL,
    ENGINE_HEALTH_CHECK_TIMEOUT,
    ENGINE_PORT,
)
from src.errors import EngineError

logger = logging.getLogger(__name__)

_TERMINATE_GRACE_PERIOD: float = 5.0


class EngineManager:
    """Manages the RocketRide engine server binary lifecycle.

    Downloads the server binary on first use, launches it as a subprocess,
    and cleans up on exit. Use as an async context manager:

        async with EngineManager() as engine:
            # engine is healthy and ready
            ...
    """

    def __init__(self, port: int = ENGINE_PORT) -> None:
        self._port = port
        self._process: subprocess.Popen[str] | None = None
        self._binary_dir = Path(ENGINE_BINARY_DIR)

    async def _download_and_extract(self) -> Path:
        """Download and extract the server binary tarball.

        Skips the download if the binary directory already exists and
        contains files. Returns the path to the extracted directory.

        Raises:
            EngineError: If the download or extraction fails.
        """
        if self._binary_dir.exists() and any(self._binary_dir.iterdir()):
            logger.info("Server binary already exists at %s", self._binary_dir)
            return self._binary_dir

        logger.info("Downloading RocketRide server from %s", ENGINE_DOWNLOAD_URL)
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(ENGINE_DOWNLOAD_URL, timeout=120.0)
                response.raise_for_status()
        except httpx.HTTPError as e:
            msg = f"Failed to download server binary: {e}"
            raise EngineError(msg) from e

        try:
            self._binary_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = Path(tmp.name)

            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=self._binary_dir)

            tmp_path.unlink()
        except (tarfile.TarError, OSError) as e:
            msg = f"Failed to extract server binary: {e}"
            raise EngineError(msg) from e

        logger.info("Extracted server binary to %s", self._binary_dir)
        return self._binary_dir

    def _find_binary(self) -> Path:
        """Locate the server binary in the extracted directory.

        Raises:
            EngineError: If no executable binary is found.
        """
        for pattern in ("engine", "engine.exe", "rocketride-server*"):
            for candidate in self._binary_dir.rglob(pattern):
                if candidate.is_file():
                    return candidate

        msg = f"Server binary not found in {self._binary_dir}"
        raise EngineError(msg)

    async def start(self) -> None:
        """Download the binary (if needed) and launch the server process.

        Raises:
            EngineError: If the download, extraction, or process launch fails.
        """
        await self._download_and_extract()
        binary = self._find_binary()

        try:
            self._process = subprocess.Popen(
                [str(binary), "--port", str(self._port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logger.info("Started engine server (PID %d)", self._process.pid)
        except OSError as e:
            msg = f"Failed to start engine server: {e}"
            raise EngineError(msg) from e

    async def wait_for_healthy(self) -> None:
        """Poll the engine health endpoint until ready or timeout.

        Polls ``http://localhost:<port>/health`` every
        ``ENGINE_HEALTH_CHECK_INTERVAL`` seconds for up to
        ``ENGINE_HEALTH_CHECK_TIMEOUT`` seconds.

        Raises:
            EngineError: If the engine does not respond within the
                timeout window.
        """
        url = f"http://localhost:{self._port}/health"
        deadline = time.monotonic() + ENGINE_HEALTH_CHECK_TIMEOUT

        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                try:
                    response = await client.get(url, timeout=5.0)
                    if response.status_code == 200:
                        logger.info("Engine is healthy")
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(ENGINE_HEALTH_CHECK_INTERVAL)

        msg = f"Engine did not become healthy within {ENGINE_HEALTH_CHECK_TIMEOUT}s"
        raise EngineError(msg)

    async def stop(self) -> None:
        """Terminate the server process.

        Sends SIGTERM first and waits briefly. If the process is still
        alive after the grace period, sends SIGKILL. Errors are logged
        but not raised.
        """
        if self._process is None:
            return

        pid = self._process.pid
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=_TERMINATE_GRACE_PERIOD)
                logger.info("Engine server (PID %d) terminated gracefully", pid)
            except subprocess.TimeoutExpired:
                logger.warning("Engine server (PID %d) did not terminate, killing", pid)
                self._process.kill()
                self._process.wait(timeout=5.0)
                logger.info("Engine server (PID %d) killed", pid)
        except OSError:
            logger.warning("Failed to stop engine server (PID %d)", pid)

        self._process = None

    async def __aenter__(self) -> EngineManager:
        """Start the engine and wait for it to become healthy."""
        await self.start()
        await self.wait_for_healthy()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the engine, even if an exception occurred."""
        await self.stop()
