from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, Sequence


class Visibility(str, Enum):
    PRIVATE = "private"
    SHARED = "shared"
    GLOBAL = "global"


@dataclass(frozen=True)
class MemoryRecord:
    owner_id: str
    content: str
    importance: float
    visibility: Visibility = Visibility.PRIVATE
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryStore(Protocol):
    def add(self, record: MemoryRecord) -> str: ...
    def recall(self, owner_id: str, query: str, limit: int = 10) -> Sequence[MemoryRecord]: ...
