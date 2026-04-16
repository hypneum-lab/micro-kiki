"""mlx-lm server for Mac Studio with adapter switching.

Architecture: MLX-LM serves Qwen3.5-35B-A3B with LoRA adapters.
Hot-swap is NOT cleanly supported by mlx-lm — each adapter switch
triggers a subprocess restart with ~200ms penalty. The server keeps
track of the current adapter to avoid unnecessary restarts.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLXServerConfig:
    """Immutable configuration for the MLX-LM server."""

    model_path: str = "models/qwen3.5-35b-a3b"
    adapter_dir: str = "outputs/stacks"
    port: int = 8200
    max_active_adapters: int = 4

    @classmethod
    def from_json(cls, path: str | Path) -> MLXServerConfig:
        """Load config from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            model_path=data.get("model_path", cls.model_path),
            adapter_dir=data.get("adapter_dir", cls.adapter_dir),
            port=data.get("port", cls.port),
            max_active_adapters=data.get("max_active_adapters", cls.max_active_adapters),
        )


@dataclass
class MLXServer:
    """Manages an mlx_lm.server subprocess with LoRA adapter switching.

    Adapter switching restarts the subprocess because mlx-lm does not
    support runtime adapter hot-swap. Expect ~200ms restart penalty per
    switch. The server tracks the active adapter to skip no-op switches.
    """

    config: MLXServerConfig = field(default_factory=MLXServerConfig)
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _active_adapter: str | None = field(default=None, init=False, repr=False)
    _client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=30.0), init=False, repr=False)

    @property
    def base_url(self) -> str:
        """Base URL for the MLX-LM server."""
        return f"http://127.0.0.1:{self.config.port}"

    def _build_command(self, adapter: str | None = None) -> list[str]:
        """Build the mlx_lm.server launch command."""
        cmd = [
            "python", "-m", "mlx_lm.server",
            "--model", self.config.model_path,
            "--port", str(self.config.port),
        ]
        if adapter:
            adapter_path = str(Path(self.config.adapter_dir) / adapter)
            cmd.extend(["--adapter-path", adapter_path])
        return cmd

    def start(self, adapter: str | None = None) -> None:
        """Launch the mlx_lm.server subprocess.

        If a process is already running, it is stopped first.

        Args:
            adapter: Optional stack_id for initial LoRA adapter.
        """
        if self._process is not None:
            self.stop()

        cmd = self._build_command(adapter)
        logger.info("Starting MLX server: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._active_adapter = adapter
        logger.info(
            "MLX server started (pid=%s, adapter=%s)",
            self._process.pid,
            adapter or "none",
        )

    def stop(self) -> None:
        """Terminate the running mlx_lm.server subprocess."""
        if self._process is None:
            return

        logger.info("Stopping MLX server (pid=%s)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("MLX server did not stop gracefully, killing")
            self._process.kill()
            self._process.wait()
        self._process = None
        self._active_adapter = None

    def switch_adapter(self, stack_id: str) -> bool:
        """Switch the active LoRA adapter by restarting the server.

        mlx-lm does not support hot-swapping adapters at runtime.
        Each switch incurs a ~200ms restart penalty while the subprocess
        is terminated and relaunched with the new --adapter-path.

        Args:
            stack_id: Identifier for the adapter directory under adapter_dir.

        Returns:
            True if the adapter was switched (restart occurred),
            False if already using the requested adapter.
        """
        if stack_id == self._active_adapter:
            logger.debug("Adapter %s already active, skipping switch", stack_id)
            return False

        logger.info("Switching adapter: %s -> %s", self._active_adapter, stack_id)
        self.stop()
        self.start(adapter=stack_id)
        return True

    def health(self) -> dict[str, Any]:
        """Check server health via GET /health.

        Returns:
            Health response as a dict, or error dict on failure.
        """
        try:
            resp = self._client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.error("Health check failed: %s", exc)
            return {"status": "error", "detail": str(exc)}

    def generate(self, prompt: str, adapter: str | None = None) -> str:
        """Generate a response via POST /v1/chat/completions.

        If an adapter is specified and differs from the active one,
        the server is restarted with the new adapter first.

        Args:
            prompt: User message to send.
            adapter: Optional stack_id; triggers switch if different.

        Returns:
            Generated text content from the model.
        """
        if adapter is not None:
            self.switch_adapter(adapter)

        payload = {
            "model": self.config.model_path,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            resp = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            logger.error("Generate failed: %s", exc)
            raise

    @property
    def active_adapter(self) -> str | None:
        """Currently loaded adapter stack_id, or None."""
        return self._active_adapter

    @property
    def is_running(self) -> bool:
        """Whether the server subprocess is alive."""
        return self._process is not None and self._process.poll() is None
