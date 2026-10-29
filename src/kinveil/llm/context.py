from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence

from .base import ChatMessage, LLMRequest


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class ConservativeTokenCounter:
    """Safe provider-neutral fallback; exact model tokenizers can replace it."""

    def count(self, text: str) -> int:
        return max(1, len(text.encode("utf-8")))


class ContextOverflowError(ValueError):
    pass


@dataclass(frozen=True)
class ContextLimits:
    maximum_context_tokens: int = 8192
    reserved_output_tokens: int = 1024
    maximum_memory_tokens: int = 800

    def __post_init__(self) -> None:
        if self.maximum_context_tokens <= 0:
            raise ValueError("maximum_context_tokens must be positive")
        if not 0 < self.reserved_output_tokens < self.maximum_context_tokens:
            raise ValueError("reserved_output_tokens must leave room for input")
        if self.maximum_memory_tokens < 0:
            raise ValueError("maximum_memory_tokens cannot be negative")
        if self.maximum_memory_tokens > self.maximum_input_tokens:
            raise ValueError("maximum_memory_tokens cannot exceed the input budget")

    @property
    def maximum_input_tokens(self) -> int:
        return self.maximum_context_tokens - self.reserved_output_tokens


@dataclass(frozen=True)
class BoundedLLMRequest:
    request: LLMRequest
    estimated_input_tokens: int
    maximum_input_tokens: int
    dropped_history_messages: int
    memory_included: bool


class BoundedContextBuilder:
    """Build an LLM request that cannot exceed the configured input allowance."""

    def __init__(
        self,
        limits: ContextLimits | None = None,
        *,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.limits = limits or ContextLimits()
        self.token_counter = token_counter or ConservativeTokenCounter()

    def build(
        self,
        *,
        system_messages: Sequence[ChatMessage],
        history: Sequence[ChatMessage],
        current_user: ChatMessage,
        memory_context: str = "",
        tools: Sequence[dict] = (),
    ) -> BoundedLLMRequest:
        if current_user.role != "user":
            raise ValueError("current_user must have the user role")
        tool_cost = self.token_counter.count(
            json.dumps(tuple(tools), ensure_ascii=False, sort_keys=True)
        ) if tools else 0
        mandatory = [*system_messages, current_user]
        mandatory_cost = sum(self._message_cost(message) for message in mandatory) + tool_cost
        if mandatory_cost > self.limits.maximum_input_tokens:
            raise ContextOverflowError(
                "system prompt, current user turn, and tools exceed the input budget"
            )

        memory_message = None
        memory_cost = 0
        if memory_context:
            memory_message = ChatMessage("system", memory_context)
            memory_cost = self._message_cost(memory_message)
            if self.token_counter.count(memory_context) > self.limits.maximum_memory_tokens:
                raise ContextOverflowError("memory context exceeds its dedicated budget")
            if mandatory_cost + memory_cost > self.limits.maximum_input_tokens:
                memory_message = None
                memory_cost = 0

        remaining = self.limits.maximum_input_tokens - mandatory_cost - memory_cost
        selected_reversed: list[ChatMessage] = []
        for message in reversed(history):
            cost = self._message_cost(message)
            if cost > remaining:
                break
            selected_reversed.append(message)
            remaining -= cost
        selected_history = list(reversed(selected_reversed))
        messages = [*system_messages]
        if memory_message is not None:
            messages.append(memory_message)
        messages.extend(selected_history)
        messages.append(current_user)
        estimated = sum(self._message_cost(message) for message in messages) + tool_cost
        return BoundedLLMRequest(
            request=LLMRequest(messages=tuple(messages), tools=tuple(tools)),
            estimated_input_tokens=estimated,
            maximum_input_tokens=self.limits.maximum_input_tokens,
            dropped_history_messages=len(history) - len(selected_history),
            memory_included=memory_message is not None,
        )

    def _message_cost(self, message: ChatMessage) -> int:
        return 8 + self.token_counter.count(message.role) + self.token_counter.count(
            message.content
        )
