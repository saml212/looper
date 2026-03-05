"""Tests for OpenClaw provider config generator."""

import json
from pathlib import Path

import pytest

from looper.integrations.openclaw_provider import (
    generate_provider_config,
    write_provider_config,
    set_default_model,
    restore_default_model,
)


class TestGenerateProviderConfig:
    def test_basic_config(self):
        config = generate_provider_config(port=8080, model_name="qwen2.5-coder:7b")
        assert config["baseUrl"] == "http://127.0.0.1:8080/v1"
        assert config["api"] == "openai-completions"
        assert config["models"][0]["id"] == "qwen2.5-coder:7b"

    def test_custom_port(self):
        config = generate_provider_config(port=9999, model_name="test-model")
        assert "9999" in config["baseUrl"]

    def test_with_adapter_path(self, tmp_path):
        adapter = tmp_path / "adapters" / "lora_v1"
        config = generate_provider_config(
            port=8080, model_name="test", adapter_path=adapter,
        )
        # adapter_path is not stored in config (OpenClaw rejects unknown keys)
        assert "metadata" not in config

    def test_without_adapter_path(self):
        config = generate_provider_config(port=8080, model_name="test")
        assert "metadata" not in config

    def test_api_key_placeholder(self):
        config = generate_provider_config(port=8080, model_name="test")
        assert config["apiKey"] == "not-needed"

    def test_model_entry_structure(self):
        config = generate_provider_config(
            port=8080, model_name="test", context_window=8192, max_tokens=4096,
        )
        model = config["models"][0]
        assert model["contextWindow"] == 8192
        assert model["maxTokens"] == 4096
        assert model["reasoning"] is False
        assert model["cost"]["input"] == 0


class TestWriteProviderConfig:
    def test_creates_new_config(self, tmp_path):
        config_path = tmp_path / "openclaw.json"
        provider = generate_provider_config(port=8080, model_name="test")
        write_provider_config(config_path, provider)

        data = json.loads(config_path.read_text())
        assert data["models"]["mode"] == "merge"
        assert "looper-mlx" in data["models"]["providers"]
        assert data["models"]["providers"]["looper-mlx"]["baseUrl"] == "http://127.0.0.1:8080/v1"

    def test_merges_into_existing_config(self, tmp_path):
        config_path = tmp_path / "openclaw.json"
        existing = {
            "agents": {"defaults": {"model": {"primary": "anthropic/claude-sonnet-4-6"}}},
            "other_setting": True,
        }
        config_path.write_text(json.dumps(existing))

        provider = generate_provider_config(port=8080, model_name="test")
        write_provider_config(config_path, provider)

        data = json.loads(config_path.read_text())
        assert data["other_setting"] is True
        assert "looper-mlx" in data["models"]["providers"]
        # Existing config preserved
        assert data["agents"]["defaults"]["model"]["primary"] == "anthropic/claude-sonnet-4-6"

    def test_replaces_existing_looper_provider(self, tmp_path):
        config_path = tmp_path / "openclaw.json"
        existing = {
            "models": {
                "mode": "merge",
                "providers": {
                    "looper-mlx": {"baseUrl": "http://127.0.0.1:8080/v1"},
                },
            },
        }
        config_path.write_text(json.dumps(existing))

        provider = generate_provider_config(port=9090, model_name="new-model")
        write_provider_config(config_path, provider)

        data = json.loads(config_path.read_text())
        assert "9090" in data["models"]["providers"]["looper-mlx"]["baseUrl"]

    def test_creates_parent_directory(self, tmp_path):
        config_path = tmp_path / "subdir" / "openclaw.json"
        provider = generate_provider_config(port=8080, model_name="test")
        write_provider_config(config_path, provider)
        assert config_path.exists()


class TestSetDefaultModel:
    def test_sets_model(self, tmp_path):
        config_path = tmp_path / "openclaw.json"
        config_path.write_text(json.dumps({}))
        set_default_model(config_path, "looper-mlx", "test-model")

        data = json.loads(config_path.read_text())
        assert data["agents"]["defaults"]["model"]["primary"] == "looper-mlx/test-model"

    def test_restore_model(self, tmp_path):
        config_path = tmp_path / "openclaw.json"
        config_path.write_text(json.dumps({
            "agents": {"defaults": {"model": {"primary": "looper-mlx/test"}}}
        }))
        restore_default_model(config_path, "anthropic/claude-sonnet-4-6")

        data = json.loads(config_path.read_text())
        assert data["agents"]["defaults"]["model"]["primary"] == "anthropic/claude-sonnet-4-6"
