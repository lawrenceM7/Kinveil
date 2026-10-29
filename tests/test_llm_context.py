import pytest

from kinveil.llm import (
    BoundedContextBuilder,
    ChatMessage,
    ContextLimits,
    ContextOverflowError,
)


def test_context_builder_reserves_output_and_keeps_recent_history() -> None:
    builder = BoundedContextBuilder(
        ContextLimits(
            maximum_context_tokens=360,
            reserved_output_tokens=100,
            maximum_memory_tokens=100,
        )
    )
    history = tuple(
        ChatMessage("user" if index % 2 == 0 else "assistant", f"历史消息{index}" * 5)
        for index in range(8)
    )

    result = builder.build(
        system_messages=(ChatMessage("system", "你是 Kinveil"),),
        history=history,
        current_user=ChatMessage("user", "现在的问题"),
        memory_context="<memory_context>偏好低延迟</memory_context>",
    )

    assert result.estimated_input_tokens <= result.maximum_input_tokens
    assert result.maximum_input_tokens == 260
    assert result.memory_included
    assert result.dropped_history_messages > 0
    assert result.request.messages[-1].content == "现在的问题"
    kept_history = result.request.messages[2:-1]
    assert kept_history[-1].content == history[-1].content


def test_context_builder_rejects_oversized_mandatory_content() -> None:
    builder = BoundedContextBuilder(
        ContextLimits(
            maximum_context_tokens=100,
            reserved_output_tokens=20,
            maximum_memory_tokens=20,
        )
    )

    with pytest.raises(ContextOverflowError):
        builder.build(
            system_messages=(ChatMessage("system", "x" * 80),),
            history=(),
            current_user=ChatMessage("user", "question"),
        )


def test_context_builder_rejects_memory_above_its_own_slice() -> None:
    builder = BoundedContextBuilder(
        ContextLimits(
            maximum_context_tokens=300,
            reserved_output_tokens=50,
            maximum_memory_tokens=20,
        )
    )

    with pytest.raises(ContextOverflowError, match="memory context"):
        builder.build(
            system_messages=(),
            history=(),
            current_user=ChatMessage("user", "question"),
            memory_context="m" * 30,
        )
