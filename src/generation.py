"""Generation helpers for the RAG reliability benchmark."""

import json
import time

import ollama


DEFAULT_GENERATION_SETTINGS = {
    "temperature": 0,
    "seed": 42,
    "num_predict": 256,
    "num_ctx": 8192,
}


def _field(value, name, default=None):
    """Read a field from either an Ollama response model or a mapping."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def generate_answer(
    system_prompt,
    user_prompt,
    model="llama3.2:3b",
    generation_settings=None,
    response_format=None,
):
    """Generate text from local Ollama, optionally requesting structured JSON."""
    try:
        installed_models = _field(ollama.list(), "models", [])
    except Exception as exc:
        raise RuntimeError(
            f"Ollama service is unavailable while looking up model '{model}': {exc}"
        ) from exc

    model_entry = next(
        (
            entry
            for entry in installed_models
            if _field(entry, "model") == model or _field(entry, "name") == model
        ),
        None,
    )
    if model_entry is None:
        raise ValueError(f"Ollama model '{model}' is not installed")

    model_digest = _field(model_entry, "digest")
    options = DEFAULT_GENERATION_SETTINGS.copy()
    if generation_settings is not None:
        options.update(generation_settings)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        start = time.perf_counter()
        chat_options = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if response_format is not None:
            chat_options["format"] = response_format
        response = ollama.chat(
            **chat_options,
        )
        latency_seconds = time.perf_counter() - start
    except Exception as exc:
        raise RuntimeError(
            f"Ollama service is unavailable while generating with model '{model}': {exc}"
        ) from exc

    message = _field(response, "message", {})
    generated_answer = _field(message, "content")
    if generated_answer is None or (generated_answer == "" and response_format is None):
        raise RuntimeError(f"Ollama returned an empty answer for model '{model}'")

    return {
        "generated_answer": generated_answer,
        "model": model,
        "model_digest": model_digest,
        "generation_settings": options,
        "latency_seconds": latency_seconds,
        "done_reason": _field(response, "done_reason"),
    }


def main():
    result = generate_answer(
        "Answer questions clearly and concisely.",
        "What is the capital of France? Answer in one sentence.",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
