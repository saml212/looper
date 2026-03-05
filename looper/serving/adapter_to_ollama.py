"""Serve a trained LoRA adapter through Ollama.

Pipeline: fuse adapter into base model -> export GGUF -> create Ollama model.
"""

import subprocess
from pathlib import Path


def fuse_adapter(
    hf_model_name: str,
    adapter_path: Path,
    output_path: Path,
    export_gguf: bool = True,
) -> Path:
    """Fuse LoRA adapter into base model and optionally export as GGUF.

    Uses the ``mlx_lm fuse`` command.

    Args:
        hf_model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-Coder-7B-Instruct")
        adapter_path: Path to the adapter directory (contains adapters.safetensors)
        output_path: Where to save the fused model
        export_gguf: Whether to export as GGUF (default True)

    Returns:
        Path to the GGUF file (output_path / "ggml-model-Q4_K_M.gguf")
        or output_path if export_gguf is False.
    """
    cmd = [
        "mlx_lm",
        "fuse",
        "--model",
        hf_model_name,
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(output_path),
    ]
    if export_gguf:
        cmd.append("--export-gguf")

    subprocess.run(cmd, check=True)

    if export_gguf:
        return output_path / "ggml-model-Q4_K_M.gguf"
    return output_path


def create_ollama_model(
    model_name: str,
    gguf_path: Path,
    system_prompt: str = "",
) -> None:
    """Create an Ollama model from a GGUF file.

    Writes a Modelfile next to the GGUF and runs ``ollama create``.

    Args:
        model_name: Name for the Ollama model (e.g., "looper-adapted")
        gguf_path: Path to the GGUF file
        system_prompt: Optional system prompt to bake in
    """
    modelfile_path = gguf_path.parent / "Modelfile"
    lines = [f"FROM {gguf_path}"]
    if system_prompt:
        lines.append(f'SYSTEM "{system_prompt}"')
    modelfile_path.write_text("\n".join(lines) + "\n")

    subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        check=True,
    )


def serve_adapter(
    hf_model_name: str,
    adapter_path: Path,
    ollama_model_name: str = "looper-adapted",
    output_path: Path | None = None,
) -> str:
    """Full pipeline: fuse adapter, export GGUF, create Ollama model.

    Returns the Ollama model name that can be used with the agent runner.
    """
    if output_path is None:
        output_path = adapter_path.parent / "fused"

    gguf_path = fuse_adapter(
        hf_model_name=hf_model_name,
        adapter_path=adapter_path,
        output_path=output_path,
        export_gguf=True,
    )

    create_ollama_model(
        model_name=ollama_model_name,
        gguf_path=gguf_path,
        system_prompt="",
    )

    return ollama_model_name


def cleanup_ollama_model(model_name: str) -> None:
    """Remove a model from Ollama."""
    subprocess.run(["ollama", "rm", model_name], check=True)
