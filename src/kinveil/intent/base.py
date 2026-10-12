from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Intent:
    name: str
    confidence: float
    slots: dict[str, Any] = field(default_factory=dict)


class IntentRecognizer(Protocol):
    def classify(self, text: str) -> Intent: ...
