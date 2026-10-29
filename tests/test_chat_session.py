from __future__ import annotations

from kinveil.audio.voice import UserTurn
from kinveil.llm import BoundedContextBuilder, ChatSession, ContextLimits, LLMResponse
from kinveil.memory import MemoryContext


class RecordingModel:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse("收到")


class RecordingMemories:
    def __init__(self) -> None:
        self.calls = []

    def context_for(self, requester_id, query):
        self.calls.append((requester_id, query))
        return MemoryContext((), "只属于这个人的记忆", 10, 100, False)


def _turn(*, person_id="person-a", identity_status="confirmed", text="你好"):
    return UserTurn(
        person_id=person_id,
        label="Alice" if person_id else "Unknown speaker",
        text=text,
        started_at=1.0,
        ended_at=2.0,
        identity_status=identity_status,
        segment_count=1,
        language="zh",
        audio_duration=1.0,
        audio_rms=0.02,
        lip_score=0.03,
        attribution_reasons=("test",),
    )


def _session():
    model = RecordingModel()
    memories = RecordingMemories()
    context = BoundedContextBuilder(
        ContextLimits(
            maximum_context_tokens=1000,
            reserved_output_tokens=200,
            maximum_memory_tokens=200,
        )
    )
    return ChatSession(model, context, memories), model, memories


def test_confirmed_person_receives_own_memory_and_history() -> None:
    session, model, memories = _session()

    session.respond(_turn(text="第一句"))
    session.respond(_turn(text="第二句"))

    assert memories.calls == [("person-a", "第一句"), ("person-a", "第二句")]
    assert [message.content for message in model.requests[1].messages][-3:] == [
        "第一句",
        "收到",
        "第二句",
    ]


def test_unknown_speakers_receive_no_memory_or_retained_history() -> None:
    session, model, memories = _session()

    session.respond(_turn(person_id=None, identity_status="unknown", text="陌生人一"))
    session.respond(_turn(person_id=None, identity_status="unknown", text="陌生人二"))

    assert memories.calls == []
    assert "陌生人一" not in [message.content for message in model.requests[1].messages]
