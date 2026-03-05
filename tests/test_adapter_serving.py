"""Tests for looper.serving.adapter_to_ollama — LoRA adapter serving via Ollama."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from looper.serving.adapter_to_ollama import (
    cleanup_ollama_model,
    create_ollama_model,
    fuse_adapter,
    serve_adapter,
)


# ---------------------------------------------------------------------------
# fuse_adapter
# ---------------------------------------------------------------------------


class TestFuseAdapter:
    """Tests for fuse_adapter."""

    @patch("looper.serving.adapter_to_ollama.subprocess.run")
    def test_calls_mlx_lm_fuse_with_correct_command(self, mock_run, tmp_path):
        """fuse_adapter invokes mlx_lm fuse with the right flags."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        output_dir = tmp_path / "fused"

        fuse_adapter(
            hf_model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=adapter_dir,
            output_path=output_dir,
            export_gguf=True,
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "mlx_lm"
        assert cmd[1] == "fuse"
        assert "--model" in cmd
        assert "Qwen/Qwen2.5-Coder-7B-Instruct" in cmd
        assert "--adapter-path" in cmd
        assert str(adapter_dir) in cmd
        assert "--save-path" in cmd
        assert str(output_dir) in cmd
        assert "--export-gguf" in cmd

    @patch("looper.serving.adapter_to_ollama.subprocess.run")
    def test_returns_gguf_path_when_export_gguf_true(self, mock_run, tmp_path):
        """When export_gguf=True, returns the path to the GGUF file."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        output_dir = tmp_path / "fused"

        result = fuse_adapter(
            hf_model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=adapter_dir,
            output_path=output_dir,
            export_gguf=True,
        )

        assert result == output_dir / "ggml-model-Q4_K_M.gguf"

    @patch("looper.serving.adapter_to_ollama.subprocess.run")
    def test_returns_output_path_when_export_gguf_false(self, mock_run, tmp_path):
        """When export_gguf=False, returns output_path directly."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        output_dir = tmp_path / "fused"

        result = fuse_adapter(
            hf_model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=adapter_dir,
            output_path=output_dir,
            export_gguf=False,
        )

        cmd = mock_run.call_args[0][0]
        assert "--export-gguf" not in cmd
        assert result == output_dir


# ---------------------------------------------------------------------------
# create_ollama_model
# ---------------------------------------------------------------------------


class TestCreateOllamaModel:
    """Tests for create_ollama_model."""

    @patch("looper.serving.adapter_to_ollama.subprocess.run")
    def test_creates_modelfile_with_correct_from_line(self, mock_run, tmp_path):
        """The generated Modelfile contains a FROM line pointing to the GGUF."""
        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        create_ollama_model(
            model_name="looper-test",
            gguf_path=gguf_path,
            system_prompt="You are helpful.",
        )

        # The Modelfile should be written next to the GGUF
        modelfile = gguf_path.parent / "Modelfile"
        content = modelfile.read_text()
        assert f"FROM {gguf_path}" in content
        assert 'SYSTEM "You are helpful."' in content

    @patch("looper.serving.adapter_to_ollama.subprocess.run")
    def test_calls_ollama_create_with_correct_args(self, mock_run, tmp_path):
        """create_ollama_model runs `ollama create` pointing at the Modelfile."""
        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        create_ollama_model(
            model_name="looper-test",
            gguf_path=gguf_path,
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ollama"
        assert cmd[1] == "create"
        assert "looper-test" in cmd
        assert "-f" in cmd
        modelfile_path = str(gguf_path.parent / "Modelfile")
        assert modelfile_path in cmd


# ---------------------------------------------------------------------------
# serve_adapter
# ---------------------------------------------------------------------------


class TestServeAdapter:
    """Tests for serve_adapter."""

    @patch("looper.serving.adapter_to_ollama.create_ollama_model")
    @patch("looper.serving.adapter_to_ollama.fuse_adapter")
    def test_calls_fuse_then_create_in_sequence(
        self, mock_fuse, mock_create, tmp_path
    ):
        """serve_adapter calls fuse_adapter then create_ollama_model."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        gguf_path = tmp_path / "fused" / "ggml-model-Q4_K_M.gguf"
        mock_fuse.return_value = gguf_path

        serve_adapter(
            hf_model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=adapter_dir,
            ollama_model_name="looper-adapted",
        )

        mock_fuse.assert_called_once()
        mock_create.assert_called_once_with(
            model_name="looper-adapted",
            gguf_path=gguf_path,
            system_prompt="",
        )

    @patch("looper.serving.adapter_to_ollama.create_ollama_model")
    @patch("looper.serving.adapter_to_ollama.fuse_adapter")
    def test_returns_model_name(self, mock_fuse, mock_create, tmp_path):
        """serve_adapter returns the Ollama model name."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        mock_fuse.return_value = tmp_path / "fused" / "ggml-model-Q4_K_M.gguf"

        result = serve_adapter(
            hf_model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path=adapter_dir,
            ollama_model_name="my-model",
        )

        assert result == "my-model"


# ---------------------------------------------------------------------------
# cleanup_ollama_model
# ---------------------------------------------------------------------------


class TestCleanupOllamaModel:
    """Tests for cleanup_ollama_model."""

    @patch("looper.serving.adapter_to_ollama.subprocess.run")
    def test_calls_ollama_rm(self, mock_run):
        """cleanup_ollama_model runs `ollama rm <model_name>`."""
        cleanup_ollama_model("looper-adapted")

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["ollama", "rm", "looper-adapted"]
