from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmotionSignal:
    label: str
    confidence: float
    source: str


class EmotionDetector(Protocol):
    """Produces fallible conversational signals, never medical conclusions."""

    def detect(self, text: str) -> EmotionSignal: ...
