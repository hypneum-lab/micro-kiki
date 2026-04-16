"""Tests for MLX server with adapter switching."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.serving.mlx_server import MLXServer, MLXServerConfig


# ---------------------------------------------------------------------------
# MLXServerConfig
# ---------------------------------------------------------------------------


class TestMLXServerConfig:
    """Config dataclass tests."""

    def test_defaults(self) -> None:
        cfg = MLXServerConfig()
        assert cfg.model_path == "models/qwen3.5-35b-a3b"
        assert cfg.adapter_dir == "outputs/stacks"
        assert cfg.port == 8200
        assert cfg.max_active_adapters == 4

    def test_frozen(self) -> None:
        cfg = MLXServerConfig()
        with pytest.raises(AttributeError):
            cfg.port = 9999  # type: ignore[misc]

    def test_from_json(self, tmp_path: Path) -> None:
        config_file = tmp_path / "mlx-server.json"
        config_file.write_text(json.dumps({
            "model_path": "models/custom",
            "adapter_dir": "adapters/",
            "port": 9000,
            "max_active_adapters": 2,
        }))
        cfg = MLXServerConfig.from_json(config_file)
        assert cfg.model_path == "models/custom"
        assert cfg.port == 9000
        assert cfg.max_active_adapters == 2

    def test_from_json_partial(self, tmp_path: Path) -> None:
        config_file = tmp_path / "mlx-server.json"
        config_file.write_text(json.dumps({"port": 7777}))
        cfg = MLXServerConfig.from_json(config_file)
        assert cfg.port == 7777
        assert cfg.model_path == "models/qwen3.5-35b-a3b"

    def test_from_project_config(self) -> None:
        """Load the actual project config file."""
        config_path = Path(__file__).resolve().parent.parent / "configs" / "mlx-server.json"
        if config_path.exists():
            cfg = MLXServerConfig.from_json(config_path)
            assert cfg.port == 8200


# ---------------------------------------------------------------------------
# MLXServer — start / stop
# ---------------------------------------------------------------------------


class TestMLXServerLifecycle:
    """Server start/stop with mocked subprocess."""

    @patch("src.serving.mlx_server.subprocess.Popen")
    def test_start(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        server = MLXServer()
        server.start()

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "python" in cmd[0]
        assert "-m" in cmd
        assert "mlx_lm.server" in cmd
        assert "--port" in cmd
        assert "8200" in cmd

    @patch("src.serving.mlx_server.subprocess.Popen")
    def test_start_with_adapter(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        server = MLXServer()
        server.start(adapter="stack-03-python")

        cmd = mock_popen.call_args[0][0]
        assert "--adapter-path" in cmd
        assert "outputs/stacks/stack-03-python" in cmd[-1]

    @patch("src.serving.mlx_server.subprocess.Popen")
    def test_stop(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        server = MLXServer()
        server.start()
        server.stop()

        mock_proc.terminate.assert_called_once()
        assert server._process is None

    @patch("src.serving.mlx_server.subprocess.Popen")
    def test_stop_when_not_running(self, mock_popen: MagicMock) -> None:
        server = MLXServer()
        server.stop()  # should not raise


# ---------------------------------------------------------------------------
# MLXServer — adapter switching
# ---------------------------------------------------------------------------


class TestAdapterSwitch:
    """Adapter switch triggers restart."""

    @patch("src.serving.mlx_server.subprocess.Popen")
    def test_switch_triggers_restart(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        server = MLXServer()
        server.start()
        switched = server.switch_adapter("stack-05-cpp")

        assert switched is True
        assert server.active_adapter == "stack-05-cpp"
        # start() called twice: initial + switch
        assert mock_popen.call_count == 2

    @patch("src.serving.mlx_server.subprocess.Popen")
    def test_switch_same_adapter_noop(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        server = MLXServer()
        server.start(adapter="stack-05-cpp")
        switched = server.switch_adapter("stack-05-cpp")

        assert switched is False
        # Only the initial start, no restart
        assert mock_popen.call_count == 1

    @patch("src.serving.mlx_server.subprocess.Popen")
    def test_switch_different_adapter(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        server = MLXServer()
        server.start(adapter="stack-01-chat-fr")
        switched = server.switch_adapter("stack-03-python")

        assert switched is True
        assert server.active_adapter == "stack-03-python"


# ---------------------------------------------------------------------------
# MLXServer — health
# ---------------------------------------------------------------------------


class TestHealth:
    """Health endpoint tests."""

    @patch("src.serving.mlx_server.subprocess.Popen")
    def test_health_success(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        server = MLXServer()
        server.start()

        with patch.object(server._client, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"status": "ok"}
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = server.health()
            assert result == {"status": "ok"}
            mock_get.assert_called_once_with("http://127.0.0.1:8200/health")

    def test_health_error(self) -> None:
        server = MLXServer()
        # No server running, client will fail
        result = server.health()
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# MLXServer — generate
# ---------------------------------------------------------------------------


class TestGenerate:
    """Generate endpoint tests."""

    @patch("src.serving.mlx_server.subprocess.Popen")
    def test_generate_success(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        server = MLXServer()
        server.start()

        with patch.object(server._client, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Hello world"}}],
            }
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = server.generate("Say hello")
            assert result == "Hello world"

    @patch("src.serving.mlx_server.subprocess.Popen")
    def test_generate_with_adapter_switch(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        server = MLXServer()
        server.start(adapter="stack-01-chat-fr")

        with patch.object(server._client, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "def foo(): pass"}}],
            }
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = server.generate("Write a function", adapter="stack-03-python")
            assert result == "def foo(): pass"
            assert server.active_adapter == "stack-03-python"
            # start (initial) + restart (switch) = 2
            assert mock_popen.call_count == 2


# ---------------------------------------------------------------------------
# MLXServer — properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Property accessors."""

    def test_base_url(self) -> None:
        server = MLXServer()
        assert server.base_url == "http://127.0.0.1:8200"

    def test_base_url_custom_port(self) -> None:
        cfg = MLXServerConfig(port=9999)
        server = MLXServer(config=cfg)
        assert server.base_url == "http://127.0.0.1:9999"

    @patch("src.serving.mlx_server.subprocess.Popen")
    def test_is_running(self, mock_popen: MagicMock) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None  # still running
        mock_popen.return_value = mock_proc

        server = MLXServer()
        assert server.is_running is False

        server.start()
        assert server.is_running is True

    def test_active_adapter_initially_none(self) -> None:
        server = MLXServer()
        assert server.active_adapter is None
