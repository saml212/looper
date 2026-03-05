"""Chat clients for Ollama and MLX inference."""

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Lazily loaded MLX model for adapter-based inference
_mlx_model = None
_mlx_tokenizer = None


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    content: str
    total_tokens: int
    model: str


def chat(
    messages: list[ChatMessage],
    model: str = "qwen2.5-coder:7b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ChatResponse:
    """Send a chat request to Ollama and return the response.

    Uses the /api/chat endpoint with stream=False.
    Raises httpx.HTTPStatusError on non-2xx responses.
    Raises httpx.ConnectError (or other httpx errors) on network failures.
    """
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    last_exc = None
    for attempt in range(3):
        try:
            response = httpx.post(
                f"{base_url}/api/chat", json=payload, timeout=600.0
            )
            response.raise_for_status()
            data = response.json()
            return ChatResponse(
                content=data["message"]["content"],
                total_tokens=data.get("eval_count", 0)
                + data.get("prompt_eval_count", 0),
                model=data["model"],
            )
        except httpx.ReadTimeout as exc:
            last_exc = exc
            logger.warning(f"Ollama timeout (attempt {attempt + 1}/3), retrying...")
            time.sleep(2)
    raise last_exc


def openai_chat(
    messages: list[ChatMessage],
    model: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    base_url: str = "http://localhost:8080",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
) -> ChatResponse:
    """Send a chat request to an OpenAI-compatible API (e.g., mlx_lm.server).

    Uses the /v1/chat/completions endpoint.
    """
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_exc = None
    for attempt in range(3):
        try:
            response = httpx.post(
                f"{base_url}/v1/chat/completions", json=payload, timeout=600.0
            )
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return ChatResponse(
                content=choice["message"]["content"],
                total_tokens=usage.get("total_tokens", 0),
                model=data.get("model", model),
            )
        except httpx.ReadTimeout as exc:
            last_exc = exc
            logger.warning(
                f"OpenAI API timeout (attempt {attempt + 1}/3), retrying..."
            )
            time.sleep(2)
    raise last_exc


def load_mlx_model(model_path: str, adapter_path: str | None = None) -> None:
    """Load an MLX model (and optional adapter) for mlx_chat inference.

    Call this once before using mlx_chat. The model stays in memory.
    """
    global _mlx_model, _mlx_tokenizer  # noqa: PLW0603
    from mlx_lm import load

    if adapter_path:
        _mlx_model, _mlx_tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        _mlx_model, _mlx_tokenizer = load(model_path)


def mlx_chat(
    messages: list[ChatMessage],
    model: str = "",
    max_tokens: int = 4096,
    **kwargs,
) -> ChatResponse:
    """Generate a chat response using a locally loaded MLX model.

    Requires load_mlx_model() to have been called first.
    """
    from mlx_lm import generate

    if _mlx_model is None or _mlx_tokenizer is None:
        raise RuntimeError("Call load_mlx_model() before using mlx_chat()")

    # Format messages using the tokenizer's chat template
    formatted = _mlx_tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=False,
        add_generation_prompt=True,
    )

    response_text = generate(
        _mlx_model,
        _mlx_tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
    )

    return ChatResponse(
        content=response_text,
        total_tokens=len(formatted.split()) + len(response_text.split()),
        model=model or "mlx-local",
    )
