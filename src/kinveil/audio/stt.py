from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None
    confidence: float | None = None


class SpeechToText(Protocol):
    def transcribe(self, audio: Path) -> Transcript: ...
