from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class ActionRequest:
    actor_id: str
    capability: str
    reason: str
    arguments: dict[str, Any] = field(default_factory=dict)


class PermissionManager(Protocol):
    def evaluate(self, request: ActionRequest) -> Decision: ...
