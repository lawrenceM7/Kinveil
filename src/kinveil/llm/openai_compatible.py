from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import LLMRequest, LLMResponse
from .config import ChatModelConfig


class ChatConnectionError(RuntimeError):
    pass


class OpenAICompatibleLanguageModel:
    """Small provider-neutral client for OpenAI-compatible chat endpoints."""

    def __init__(self, config: ChatModelConfig, api_key: str) -> None:
        if not api_key.strip():
            raise ValueError("API key cannot be empty")
        self.config = config
        self._api_key = api_key.strip()

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "stream": False,
            "max_tokens": self.config.maximum_output_tokens,
            "temperature": self.config.temperature,
        }
        if request.tools:
            payload["tools"] = list(request.tools)
        http_request = urllib.request.Request(
            _chat_completions_url(self.config.base_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.config.timeout_seconds
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ChatConnectionError(f"chat service returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ChatConnectionError("could not reach the chat service") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ChatConnectionError("chat service returned an invalid response") from exc
        try:
            message = body["choices"][0]["message"]
            text = message.get("content") or ""
            tool_calls = tuple(message.get("tool_calls") or ())
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatConnectionError("chat service response contained no message") from exc
        return LLMResponse(str(text).strip(), tool_calls)


def _chat_completions_url(base_url: str) -> str:
    root = base_url.strip().rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return f"{root}/chat/completions"
