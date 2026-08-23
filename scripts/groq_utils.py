"""Shared Groq chat-completion helpers for gpt-oss reasoning models."""

from __future__ import annotations

LLM_MODEL = "openai/gpt-oss-20b"
MAX_RETRY_TOKENS = 4096


def _parts_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                pieces.append(str(item.get("text") or item.get("content") or ""))
            else:
                pieces.append(str(getattr(item, "text", "") or ""))
        return "\n".join(part for part in pieces if part).strip()
    return str(content).strip()


def extract_choice_text(choice) -> tuple[str, str | None]:
    """Return (content, finish_reason) from a Groq/OpenAI chat choice."""
    finish_reason = getattr(choice, "finish_reason", None)
    message = getattr(choice, "message", None)
    if message is None:
        return "", finish_reason
    text = _parts_to_text(getattr(message, "content", None))
    return text, finish_reason


def _create_completion(client, messages: list[dict], max_tokens: int, temperature: float):
    attempts = [
        {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "low",
        },
        {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": "low",
        },
        {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return client.chat.completions.create(**kwargs)
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            message = str(exc).lower()
            retryable = any(
                token in message
                for token in (
                    "reasoning_effort",
                    "max_completion_tokens",
                    "unexpected keyword",
                    "unknown parameter",
                )
            )
            if retryable:
                last_error = exc
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("Groq completion failed with no response.")


def complete_chat(
    client,
    messages: list[dict],
    *,
    max_tokens: int = 1500,
    temperature: float = 0.1,
    continue_on_length: bool = False,
) -> tuple[str, str | None]:
    """
    Complete a chat request and return (text, finish_reason).

    gpt-oss spends max_tokens on reasoning + answer. If the budget is too
    small, Groq returns empty content with finish_reason=length. This helper
    retries with a larger budget and can continue a truncated answer.
    """
    response = _create_completion(client, messages, max_tokens, temperature)
    choice = response.choices[0]
    text, finish_reason = extract_choice_text(choice)

    if not text:
        bigger = min(max(max_tokens * 2, 2048), MAX_RETRY_TOKENS)
        if bigger > max_tokens:
            response = _create_completion(client, messages, bigger, temperature)
            choice = response.choices[0]
            text, finish_reason = extract_choice_text(choice)
            max_tokens = bigger

    if continue_on_length and text and finish_reason == "length":
        continuation = messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    "Continue from where you stopped. "
                    "Do not repeat what you already wrote. Return only the continuation."
                ),
            },
        ]
        extra_response = _create_completion(
            client, continuation, max_tokens, temperature
        )
        extra_text, extra_reason = extract_choice_text(extra_response.choices[0])
        if extra_text:
            text = f"{text}\n{extra_text}".strip()
            finish_reason = extra_reason

    return text, finish_reason
