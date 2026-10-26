from types import SimpleNamespace

import ollama
import pytest

from src import generation


def install_ollama_responses(monkeypatch, *, digest="full-model-digest"):
    monkeypatch.setattr(
        ollama,
        "list",
        lambda: SimpleNamespace(
            models=[SimpleNamespace(model="llama3.2:3b", digest=digest)]
        ),
    )
    monkeypatch.setattr(
        ollama,
        "chat",
        lambda **kwargs: SimpleNamespace(
            message=SimpleNamespace(content="Paris is the capital of France."),
            done_reason="stop",
        ),
    )


def test_generate_answer_sends_separate_prompts_and_disables_streaming(monkeypatch):
    install_ollama_responses(monkeypatch)
    calls = []
    def fake_chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content="answer"), done_reason="stop"
        )

    monkeypatch.setattr(ollama, "chat", fake_chat)

    generation.generate_answer("system text", "user text")

    assert calls[0]["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert calls[0]["stream"] is False


def test_generate_answer_uses_defaults_and_merges_overrides_without_mutation(monkeypatch):
    install_ollama_responses(monkeypatch)
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content="answer"), done_reason="stop"
        )

    monkeypatch.setattr(ollama, "chat", fake_chat)
    caller_settings = {"temperature": 0.4, "top_p": 0.8}

    generation.generate_answer("system", "user", generation_settings=caller_settings)

    assert captured["options"] == {
        "temperature": 0.4,
        "seed": 42,
        "num_predict": 256,
        "num_ctx": 8192,
        "top_p": 0.8,
    }
    assert caller_settings == {"temperature": 0.4, "top_p": 0.8}


def test_generate_answer_returns_text_metadata_and_nonnegative_latency(monkeypatch):
    install_ollama_responses(monkeypatch)

    result = generation.generate_answer("system", "user")

    assert result["generated_answer"] == "Paris is the capital of France."
    assert result["model"] == "llama3.2:3b"
    assert result["model_digest"] == "full-model-digest"
    assert result["generation_settings"] == {
        "temperature": 0,
        "seed": 42,
        "num_predict": 256,
        "num_ctx": 8192,
    }
    assert result["latency_seconds"] >= 0
    assert result["done_reason"] == "stop"


def test_generate_answer_raises_clear_error_when_service_is_unavailable(monkeypatch):
    def unavailable():
        raise ConnectionError("connection refused")

    monkeypatch.setattr(ollama, "list", unavailable)

    with pytest.raises(RuntimeError, match="Ollama service is unavailable"):
        generation.generate_answer("system", "user")


def test_generate_answer_raises_clear_error_when_model_is_missing(monkeypatch):
    monkeypatch.setattr(ollama, "list", lambda: SimpleNamespace(models=[]))

    with pytest.raises(ValueError, match="Ollama model 'llama3.2:3b' is not installed"):
        generation.generate_answer("system", "user")
