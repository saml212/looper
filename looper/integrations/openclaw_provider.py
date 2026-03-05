"""Generate OpenClaw provider config for local MLX model servers.

Produces the models.providers config block that points OpenClaw at a local
MLX inference server, using the OpenAI-compatible API format.
"""

import json
from pathlib import Path


def generate_provider_config(
    port: int,
    model_name: str,
    adapter_path: Path | None = None,
    context_window: int = 32768,
    max_tokens: int = 2048,
) -> dict:
    """Generate an OpenClaw provider config dict for a local MLX server.

    Returns a dict in the format expected by openclaw.json models.providers.<name>.
    """
    config: dict = {
        "baseUrl": f"http://127.0.0.1:{port}/v1",
        "apiKey": "not-needed",
        "api": "openai-completions",
        "models": [
            {
                "id": model_name,
                "name": f"Looper MLX ({model_name})",
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": context_window,
                "maxTokens": max_tokens,
            }
        ],
    }

    # Note: adapter_path is tracked by the experiment, not in OpenClaw config
    # (OpenClaw rejects unrecognized keys)

    return config


def write_provider_config(
    openclaw_config_path: Path,
    provider_config: dict,
    provider_name: str = "looper-mlx",
) -> None:
    """Write or merge a provider config into an OpenClaw config file.

    Uses the models.providers.<name> structure with mode: "merge" so
    hosted models stay available as fallbacks.
    """
    openclaw_config_path.parent.mkdir(parents=True, exist_ok=True)

    if openclaw_config_path.exists():
        data = json.loads(openclaw_config_path.read_text())
    else:
        data = {}

    # Ensure models.mode is "merge" and providers exists
    if "models" not in data:
        data["models"] = {}
    data["models"]["mode"] = "merge"
    if "providers" not in data["models"]:
        data["models"]["providers"] = {}

    # Set or replace the provider
    data["models"]["providers"][provider_name] = provider_config

    openclaw_config_path.write_text(json.dumps(data, indent=2))


def set_default_model(
    openclaw_config_path: Path,
    provider_name: str,
    model_name: str,
) -> None:
    """Set the default model in OpenClaw config to use our provider."""
    if openclaw_config_path.exists():
        data = json.loads(openclaw_config_path.read_text())
    else:
        data = {}

    if "agents" not in data:
        data["agents"] = {}
    if "defaults" not in data["agents"]:
        data["agents"]["defaults"] = {}
    if "model" not in data["agents"]["defaults"]:
        data["agents"]["defaults"]["model"] = {}

    data["agents"]["defaults"]["model"]["primary"] = f"{provider_name}/{model_name}"

    openclaw_config_path.write_text(json.dumps(data, indent=2))


def restore_default_model(
    openclaw_config_path: Path,
    original_model: str = "anthropic/claude-sonnet-4-6",
) -> None:
    """Restore the default model to the original value."""
    if not openclaw_config_path.exists():
        return
    data = json.loads(openclaw_config_path.read_text())
    try:
        data["agents"]["defaults"]["model"]["primary"] = original_model
    except KeyError:
        pass
    openclaw_config_path.write_text(json.dumps(data, indent=2))
