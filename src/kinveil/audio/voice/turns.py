from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock

from kinveil.audio.stt import Transcript

from .active_speaker import ActiveSpeakerResult


class TurnState(str, Enum):
    LISTENING = "listening"
    SPEAKING = "speaking"
    WAITING_CONTINUATION = "waiting_continuation"
    FINALIZING = "finalizing"


@dataclass(frozen=True)
class UserTurn:
    person_id: str | None
    label: str
    text: str
    started_at: float
    ended_at: float
    identity_status: str
    segment_count: int
    language: str | None
    audio_duration: float
    audio_rms: float | None
    lip_score: float
    attribution_reasons: tuple[str, ...]


@dataclass(frozen=True)
class _AttributedSegment:
    result: ActiveSpeakerResult
    transcript: Transcript
    started_at: float
    ended_at: float


class UserTurnStateMachine:
    """Combine attributed ASR segments into final user turns."""

    def __init__(
        self,
        *,
        normal_pause_seconds: float = 0.65,
        incomplete_pause_seconds: float = 1.0,
        maximum_join_gap_seconds: float = 1.1,
    ) -> None:
        if not 0 < normal_pause_seconds <= incomplete_pause_seconds:
            raise ValueError("turn pause durations must be positive and ordered")
        self.normal_pause_seconds = normal_pause_seconds
        self.incomplete_pause_seconds = incomplete_pause_seconds
        self.maximum_join_gap_seconds = maximum_join_gap_seconds
        self.state = TurnState.LISTENING
        self._segments: list[_AttributedSegment] = []
        self._speech_active = False
        self._lock = Lock()

    def observe_speech(self, active: bool) -> None:
        with self._lock:
            self._speech_active = active
            if active:
                self.state = TurnState.SPEAKING
            elif self._segments:
                self.state = TurnState.WAITING_CONTINUATION
            else:
                self.state = TurnState.LISTENING

    def ingest(
        self,
        result: ActiveSpeakerResult,
        transcript: Transcript,
        *,
        started_at: float,
        ended_at: float,
    ) -> tuple[UserTurn, ...]:
        if not transcript.text.strip():
            return ()
        incoming = _AttributedSegment(result, transcript, started_at, ended_at)
        ready: list[UserTurn] = []
        with self._lock:
            if self._segments and not self._can_join(incoming):
                ready.append(self._finalize_locked())
            self._segments.append(incoming)
            self.state = (
                TurnState.SPEAKING if self._speech_active else TurnState.WAITING_CONTINUATION
            )
        return tuple(ready)

    def poll(self, now: float) -> tuple[UserTurn, ...]:
        with self._lock:
            if not self._segments or self._speech_active:
                return ()
            text = _join_text(tuple(item.transcript.text for item in self._segments))
            pause = (
                self.incomplete_pause_seconds
                if _looks_incomplete(text)
                else self.normal_pause_seconds
            )
            if now - self._segments[-1].ended_at < pause:
                self.state = TurnState.WAITING_CONTINUATION
                return ()
            return (self._finalize_locked(),)

    def flush(self) -> tuple[UserTurn, ...]:
        with self._lock:
            return (self._finalize_locked(),) if self._segments else ()

    def _can_join(self, incoming: _AttributedSegment) -> bool:
        previous = self._segments[-1]
        gap = incoming.started_at - previous.ended_at
        if gap > self.maximum_join_gap_seconds:
            return False
        previous_id = previous.result.person_id if previous.result.accepted else None
        incoming_id = incoming.result.person_id if incoming.result.accepted else None
        return previous_id is None or incoming_id is None or previous_id == incoming_id

    def _finalize_locked(self) -> UserTurn:
        self.state = TurnState.FINALIZING
        segments = tuple(self._segments)
        self._segments.clear()
        known = [
            item.result
            for item in segments
            if item.result.accepted and item.result.person_id is not None
        ]
        person_ids = {item.person_id for item in known}
        if len(person_ids) == 1:
            person_id = known[-1].person_id
            label = known[-1].label
            identity_status = "confirmed" if len(known) == len(segments) else "partial"
        else:
            person_id = None
            label = "Unknown speaker"
            identity_status = "unknown" if not person_ids else "conflict"
        durations = [
            item.transcript.audio_duration
            for item in segments
            if item.transcript.audio_duration is not None
        ]
        rms_values = [
            item.transcript.audio_rms
            for item in segments
            if item.transcript.audio_rms is not None
        ]
        turn = UserTurn(
            person_id=person_id,
            label=label,
            text=_join_text(tuple(item.transcript.text for item in segments)),
            started_at=segments[0].started_at,
            ended_at=segments[-1].ended_at,
            identity_status=identity_status,
            segment_count=len(segments),
            language=next(
                (
                    item.transcript.language
                    for item in reversed(segments)
                    if item.transcript.language
                ),
                None,
            ),
            audio_duration=float(sum(durations)),
            audio_rms=(float(sum(rms_values) / len(rms_values)) if rms_values else None),
            lip_score=max(item.result.lip_score for item in segments),
            attribution_reasons=tuple(dict.fromkeys(item.result.reason for item in segments)),
        )
        self.state = TurnState.SPEAKING if self._speech_active else TurnState.LISTENING
        return turn


def _looks_incomplete(text: str) -> bool:
    normalized = text.strip().rstrip("，,、 ")
    if not normalized:
        return True
    continuations = (
        "然后",
        "但是",
        "不过",
        "因为",
        "所以",
        "如果",
        "虽然",
        "还有",
        "或者",
        "以及",
        "关于",
        "我想",
        "我想让你",
        "帮我",
        "那个",
        "就是",
    )
    return normalized.endswith(continuations)


def _join_text(parts: tuple[str, ...]) -> str:
    combined = ""
    for part in parts:
        current = part.strip()
        if not current:
            continue
        overlap = 0
        limit = min(32, len(combined), len(current))
        for size in range(limit, 1, -1):
            if combined[-size:] == current[:size]:
                overlap = size
                break
        current = current[overlap:].lstrip()
        if not current:
            continue
        separator = (
            " "
            if combined and combined[-1].isascii() and current[0].isascii()
            else ""
        )
        combined += separator + current
    return combined
