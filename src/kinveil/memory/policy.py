from __future__ import annotations

from .base import MemoryCandidate, PolicyAction, PolicyDecision, Sensitivity, Visibility


class MemoryPolicy:
    """Deterministic firewall between model proposals and persistent memory."""

    def __init__(
        self,
        *,
        auto_store_confidence: float = 0.78,
        review_confidence: float = 0.5,
        minimum_importance: float = 0.25,
    ) -> None:
        self.auto_store_confidence = auto_store_confidence
        self.review_confidence = review_confidence
        self.minimum_importance = minimum_importance

    def evaluate(
        self, candidate: MemoryCandidate, *, memory_consent: bool
    ) -> PolicyDecision:
        content = candidate.content.strip()
        if not content:
            return PolicyDecision(PolicyAction.REJECT, "empty memory")
        if candidate.owner_id is None:
            return PolicyDecision(PolicyAction.REJECT, "no stable person identity")
        if candidate.identity_status in {"unknown", "conflict"}:
            return PolicyDecision(
                PolicyAction.REJECT,
                f"identity is {candidate.identity_status}",
            )
        if candidate.visibility is Visibility.SHARED and not candidate.audience_ids:
            return PolicyDecision(PolicyAction.REVIEW, "shared memory has no audience")
        if candidate.visibility is Visibility.PRIVATE and candidate.audience_ids:
            return PolicyDecision(
                PolicyAction.REJECT,
                "private memory cannot name another audience",
            )
        if candidate.visibility is not Visibility.PRIVATE and not candidate.confirmed_by_user:
            return PolicyDecision(PolicyAction.REVIEW, "sharing memory needs confirmation")
        if not memory_consent:
            return PolicyDecision(PolicyAction.REVIEW, "memory consent is required")
        if candidate.identity_status != "confirmed":
            return PolicyDecision(PolicyAction.REVIEW, "speaker identity needs confirmation")
        if candidate.sensitivity is Sensitivity.SENSITIVE and not candidate.confirmed_by_user:
            return PolicyDecision(PolicyAction.REVIEW, "sensitive memory needs confirmation")
        if candidate.requested_by_user:
            return PolicyDecision(PolicyAction.STORE, "user explicitly asked to remember")
        if candidate.importance < self.minimum_importance:
            return PolicyDecision(PolicyAction.REJECT, "memory is too transient")
        if candidate.confidence < self.review_confidence:
            return PolicyDecision(PolicyAction.REJECT, "proposal confidence is too low")
        if candidate.confidence < self.auto_store_confidence:
            return PolicyDecision(PolicyAction.REVIEW, "proposal needs confirmation")
        return PolicyDecision(PolicyAction.STORE, "safe automatic memory")
