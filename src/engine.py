"""RocketRide engine lifecycle management.

Handles starting the Docker container, health check polling, SDK
connection, and teardown. Exposed as an async context manager.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from types import TracebackType

import httpx

from src.config import (
    ENGINE_HEALTH_CHECK_INTERVAL,
    ENGINE_HEALTH_CHECK_TIMEOUT,
    ENGINE_PORT,
)
from src.errors import EngineError

logger = logging.getLogger(__name__)

ENGINE_IMAGE = "rocketride/engine:v1.0.1"


class EngineManager:
    """Manages the RocketRide engine Docker container lifecycle.

    Use as an async context manager to ensure proper startup and cleanup:

        async with EngineManager() as engine:
            # engine is healthy and ready
            ...
    """

    def __init__(self, image: str = ENGINE_IMAGE, port: int = ENGINE_PORT) -> None:
        self._image = image
        self._port = port
        self._container_id: str | None = None

    async def start(self) -> None:
        """Start the RocketRide engine Docker container.

        Raises:
            EngineError: If Docker fails to start the container.
        """
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "-p",
                    f"{self._port}:{self._port}",
                    self._image,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self._container_id = result.stdout.strip()
            logger.info("Started engine container: %s", self._container_id[:12])
        except subprocess.CalledProcessError as e:
            msg = f"Failed to start engine container: {e.stderr}"
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
        """Stop and remove the Docker container.

        Best-effort — errors are logged but not raised.
        """
        if self._container_id is None:
            return

        for cmd in (
            ["docker", "stop", self._container_id],
            ["docker", "rm", self._container_id],
        ):
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError:
                logger.warning("Failed to run: %s", " ".join(cmd))

        logger.info("Stopped engine container: %s", self._container_id[:12])
        self._container_id = None

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
