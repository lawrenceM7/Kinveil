from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMRequest:
    messages: Sequence[ChatMessage]
    tools: Sequence[dict[str, Any]] = field(default_factory=tuple)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: Sequence[dict[str, Any]] = field(default_factory=tuple)


class LanguageModel(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...
