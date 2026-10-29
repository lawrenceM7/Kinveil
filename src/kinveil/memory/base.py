from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, Sequence


class Visibility(str, Enum):
    PRIVATE = "private"
    SHARED = "shared"
    GLOBAL = "global"


class MemoryKind(str, Enum):
    OBSERVATION = "observation"
    FACT = "fact"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    INSTRUCTION = "instruction"
    REFLECTION = "reflection"


class RetentionClass(str, Enum):
    SESSION = "session"
    SHORT_TERM = "short_term"
    STANDARD = "standard"
    DURABLE = "durable"
    PINNED = "pinned"


class Sensitivity(str, Enum):
    ORDINARY = "ordinary"
    SENSITIVE = "sensitive"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class PolicyAction(str, Enum):
    STORE = "store"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True)
class MemoryRecord:
    owner_id: str
    content: str
    importance: float
    visibility: Visibility = Visibility.PRIVATE
    kind: MemoryKind = MemoryKind.FACT
    sensitivity: Sensitivity = Sensitivity.ORDINARY
    confidence: float = 1.0
    retention: RetentionClass = RetentionClass.STANDARD
    audience_ids: tuple[str, ...] = ()
    source_turn_id: str | None = None
    supersedes_id: str | None = None
    topic_key: str | None = None
    derived_from_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    expires_at: datetime | None = None
    last_observed_at: datetime | None = None
    last_accessed_at: datetime | None = None
    access_count: int = 0
    reinforcement_count: int = 1
    id: str | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE


@dataclass(frozen=True)
class MemoryCandidate:
    """A model proposal. It is not trusted or persistent until policy accepts it."""

    content: str
    owner_id: str | None
    identity_status: str
    confidence: float
    importance: float
    visibility: Visibility = Visibility.PRIVATE
    kind: MemoryKind = MemoryKind.FACT
    sensitivity: Sensitivity = Sensitivity.ORDINARY
    retention: RetentionClass = RetentionClass.STANDARD
    audience_ids: tuple[str, ...] = ()
    source_turn_id: str | None = None
    requested_by_user: bool = False
    confirmed_by_user: bool = False
    supersedes_id: str | None = None
    topic_key: str | None = None
    derived_from_ids: tuple[str, ...] = ()
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    reason: str


@dataclass(frozen=True)
class MemorySubmission:
    decision: PolicyDecision
    memory_id: str | None = None
    review_id: str | None = None


@dataclass(frozen=True)
class PendingMemoryReview:
    id: str
    candidate: MemoryCandidate
    reason: str
    created_at: datetime


@dataclass(frozen=True)
class MemoryContext:
    records: tuple[MemoryRecord, ...]
    text: str
    estimated_tokens: int
    max_tokens: int
    truncated: bool


class MemoryEmbeddingBackend(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed(self, text: str) -> Sequence[float]: ...


class MemoryStore(Protocol):
    def add(self, record: MemoryRecord) -> str: ...

    def recall(
        self, owner_id: str, query: str, limit: int = 10
    ) -> Sequence[MemoryRecord]: ...

    def search(
        self,
        requester_id: str,
        query: str,
        *,
        limit: int = 10,
        owner_id: str | None = None,
    ) -> Sequence[MemoryRecord]: ...
