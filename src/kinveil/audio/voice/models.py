from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceMatch:
    person_id: str | None
    similarity: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class SpeakerTrackSnapshot:
    track_id: str
    person_id: str | None
    label: str
    identity_confidence: float
    persistent: bool
    observations: int
    last_heard: float
