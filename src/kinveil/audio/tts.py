from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SynthesizedSpeech:
    audio: bytes
    media_type: str
    sample_rate: int


class TextToSpeech(Protocol):
    def synthesize(self, text: str, voice: str | None = None) -> SynthesizedSpeech: ...
