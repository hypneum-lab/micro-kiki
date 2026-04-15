"""mlx-lm server for Mac Studio with adapter switching."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MLX_SERVER_CONFIG = {
    "model": "models/qwen3.5-4b-diffattn/",
    "adapter_path": "outputs/stacks/",
    "port": 8100,
    "max_active_adapters": 4,
}


def get_mlx_command(adapter: str | None = None) -> list[str]:
    cmd = ["python", "-m", "mlx_lm.server", "--model", MLX_SERVER_CONFIG["model"],
           "--port", str(MLX_SERVER_CONFIG["port"])]
    if adapter:
        cmd.extend(["--adapter-path", f"outputs/stacks/{adapter}"])
    return cmd
