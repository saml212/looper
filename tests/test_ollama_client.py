"""Tests for the Ollama chat client."""

import httpx
import pytest

from looper.agent.ollama_client import ChatMessage, ChatResponse, chat


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


def test_chat_message_construction():
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_chat_response_construction():
    resp = ChatResponse(content="hi", total_tokens=42, model="qwen2.5-coder:7b")
    assert resp.content == "hi"
    assert resp.total_tokens == 42
    assert resp.model == "qwen2.5-coder:7b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_OLLAMA_RESPONSE = {
    "model": "qwen2.5-coder:7b",
    "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?",
    },
    "total_duration": 1_000_000_000,
    "eval_count": 80,
    "prompt_eval_count": 20,
}


def _make_fake_post(response_json=None, status_code=200):
    """Return a fake httpx.post that captures the request and returns a canned response."""
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        resp = httpx.Response(
            status_code=status_code,
            json=response_json or FAKE_OLLAMA_RESPONSE,
            request=httpx.Request("POST", url),
        )
        return resp

    return fake_post, captured


# ---------------------------------------------------------------------------
# chat() — request formatting
# ---------------------------------------------------------------------------


def test_chat_formats_request_body(monkeypatch):
    fake_post, captured = _make_fake_post()
    monkeypatch.setattr(httpx, "post", fake_post)

    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Say hi"),
    ]
    chat(messages, model="test-model", temperature=0.5, max_tokens=512)

    body = captured["json"]
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say hi"},
    ]
    assert body["options"]["temperature"] == 0.5
    assert body["options"]["num_predict"] == 512


def test_chat_sends_to_correct_url(monkeypatch):
    fake_post, captured = _make_fake_post()
    monkeypatch.setattr(httpx, "post", fake_post)

    chat([ChatMessage(role="user", content="hi")], base_url="http://myhost:1234")
    assert captured["url"] == "http://myhost:1234/api/chat"


# ---------------------------------------------------------------------------
# chat() — response parsing
# ---------------------------------------------------------------------------


def test_chat_parses_response(monkeypatch):
    fake_post, _ = _make_fake_post()
    monkeypatch.setattr(httpx, "post", fake_post)

    result = chat([ChatMessage(role="user", content="hi")])

    assert isinstance(result, ChatResponse)
    assert result.content == "Hello! How can I help you?"
    assert result.model == "qwen2.5-coder:7b"


def test_chat_total_tokens_is_sum_of_eval_and_prompt_eval(monkeypatch):
    fake_post, _ = _make_fake_post()
    monkeypatch.setattr(httpx, "post", fake_post)

    result = chat([ChatMessage(role="user", content="hi")])
    # eval_count=80 + prompt_eval_count=20 = 100
    assert result.total_tokens == 100


# ---------------------------------------------------------------------------
# chat() — error handling
# ---------------------------------------------------------------------------


def test_chat_raises_on_http_error(monkeypatch):
    fake_post, _ = _make_fake_post(
        response_json={"error": "model not found"}, status_code=404
    )
    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        chat([ChatMessage(role="user", content="hi")])


def test_chat_raises_on_network_error(monkeypatch):
    def exploding_post(url, json=None, timeout=None):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "post", exploding_post)

    with pytest.raises(httpx.ConnectError):
        chat([ChatMessage(role="user", content="hi")])
