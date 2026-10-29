from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from kinveil.audio.voice.turns import UserTurn

from .base import (
    MemoryCandidate,
    MemoryContext,
    MemoryKind,
    MemoryRecord,
    MemorySubmission,
    PolicyAction,
    PolicyDecision,
    RetentionClass,
)
from .policy import MemoryPolicy
from .store import SQLiteMemoryStore


class MemoryExtractor(Protocol):
    """Provider-neutral boundary for an LLM or small memory-curation model."""

    def extract(
        self, turn: UserTurn, assistant_text: str | None = None
    ) -> Sequence[MemoryCandidate]: ...


class NullMemoryExtractor:
    def extract(
        self, turn: UserTurn, assistant_text: str | None = None
    ) -> Sequence[MemoryCandidate]:
        return ()


class MemoryReflector(Protocol):
    def reflect(
        self, owner_id: str, evidence: Sequence[MemoryRecord]
    ) -> Sequence[MemoryCandidate]: ...


class MemoryService:
    """Orchestrates untrusted extraction, deterministic policy, review, and storage."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        policy: MemoryPolicy | None = None,
        extractor: MemoryExtractor | None = None,
        maximum_context_tokens: int = 800,
    ) -> None:
        if maximum_context_tokens <= 0:
            raise ValueError("maximum_context_tokens must be positive")
        self.store = store
        self.policy = policy or MemoryPolicy()
        self.extractor = extractor or NullMemoryExtractor()
        self.maximum_context_tokens = maximum_context_tokens

    def context_for(
        self,
        requester_id: str,
        query: str,
        *,
        max_tokens: int | None = None,
        owner_id: str | None = None,
    ) -> MemoryContext:
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        requested = max_tokens if max_tokens is not None else self.maximum_context_tokens
        budget = min(requested, self.maximum_context_tokens)
        return self.store.build_context(
            requester_id,
            query,
            max_tokens=budget,
            owner_id=owner_id,
        )

    def process_turn(
        self, turn: UserTurn, assistant_text: str | None = None
    ) -> tuple[MemorySubmission, ...]:
        """Extract candidates after a final UserTurn; safe to run off the response path."""
        return tuple(
            self.submit(candidate)
            for candidate in self.extractor.extract(turn, assistant_text)
        )

    def submit(self, candidate: MemoryCandidate) -> MemorySubmission:
        if candidate.requested_by_user and candidate.retention is RetentionClass.STANDARD:
            candidate = replace(candidate, retention=RetentionClass.PINNED)
        if candidate.owner_id and candidate.topic_key and candidate.supersedes_id is None:
            existing = self.store.find_active_by_topic(candidate.owner_id, candidate.topic_key)
            if existing is not None and existing.content.strip() != candidate.content.strip():
                if candidate.confirmed_by_user or candidate.requested_by_user:
                    candidate = replace(candidate, supersedes_id=existing.id)
                else:
                    decision = PolicyDecision(
                        PolicyAction.REVIEW,
                        "candidate conflicts with an active memory",
                    )
                    review_id = self.store.queue_review(candidate, decision.reason)
                    return MemorySubmission(decision, review_id=review_id)
        consent = bool(candidate.owner_id and self.store.has_consent(candidate.owner_id))
        decision = self.policy.evaluate(candidate, memory_consent=consent)
        if decision.action is PolicyAction.REJECT:
            return MemorySubmission(decision)
        if decision.action is PolicyAction.REVIEW:
            review_id = self.store.queue_review(candidate, decision.reason)
            return MemorySubmission(decision, review_id=review_id)
        memory_id = self.store.add(_record_from_candidate(candidate))
        return MemorySubmission(decision, memory_id=memory_id)

    def approve_review(self, review_id: str, *, confirmed_owner_id: str) -> str:
        review = next(
            (item for item in self.store.pending_reviews() if item.id == review_id),
            None,
        )
        if review is None:
            raise KeyError(f"unknown memory review: {review_id}")
        if review.candidate.owner_id != confirmed_owner_id:
            raise PermissionError("review owner does not match the confirmed person")
        if not self.store.has_consent(confirmed_owner_id):
            raise PermissionError("memory storage consent is required")
        confirmed = replace(
            review.candidate,
            identity_status="confirmed",
            confirmed_by_user=True,
        )
        if confirmed.owner_id and confirmed.topic_key and confirmed.supersedes_id is None:
            existing = self.store.find_active_by_topic(
                confirmed.owner_id, confirmed.topic_key
            )
            if existing is not None and existing.content.strip() != confirmed.content.strip():
                confirmed = replace(confirmed, supersedes_id=existing.id)
        memory_id = self.store.add(_record_from_candidate(confirmed))
        self.store.reject_review(review_id)
        return memory_id

    def reject_review(self, review_id: str) -> bool:
        return self.store.reject_review(review_id)


class BackgroundMemoryWorker:
    """Runs extraction and writes in order without blocking the response path."""

    def __init__(self, service: MemoryService) -> None:
        self.service = service
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kinveil-memory")

    def submit_turn(
        self, turn: UserTurn, assistant_text: str | None = None
    ) -> Future[tuple[MemorySubmission, ...]]:
        return self._executor.submit(self.service.process_turn, turn, assistant_text)

    def submit_maintenance(self) -> Future[dict[str, int]]:
        return self._executor.submit(self.service.store.maintain)

    def submit_reflection(
        self, engine: ReflectionEngine, owner_id: str
    ) -> Future[tuple[MemorySubmission, ...]]:
        return self._executor.submit(engine.run_if_due, owner_id)

    def close(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> BackgroundMemoryWorker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class ReflectionEngine:
    """Periodically consolidates recent records into evidence-backed reflections."""

    def __init__(
        self,
        service: MemoryService,
        reflector: MemoryReflector,
        *,
        importance_threshold: float = 4.0,
        maximum_evidence: int = 100,
    ) -> None:
        self.service = service
        self.reflector = reflector
        self.importance_threshold = importance_threshold
        self.maximum_evidence = maximum_evidence

    def run_if_due(self, owner_id: str) -> tuple[MemorySubmission, ...]:
        evidence = self.service.store.reflection_inputs(
            owner_id, limit=self.maximum_evidence
        )
        if sum(item.importance for item in evidence) < self.importance_threshold:
            return ()
        evidence_ids = tuple(item.id for item in evidence if item.id)
        submissions = []
        for proposal in self.reflector.reflect(owner_id, evidence):
            candidate = replace(
                proposal,
                owner_id=owner_id,
                identity_status="confirmed",
                kind=MemoryKind.REFLECTION,
                derived_from_ids=proposal.derived_from_ids or evidence_ids,
            )
            submissions.append(self.service.submit(candidate))
        if any(item.memory_id is not None for item in submissions):
            self.service.store.archive_consolidated(evidence_ids)
        self.service.store.mark_reflected(owner_id)
        return tuple(submissions)


def _record_from_candidate(candidate: MemoryCandidate) -> MemoryRecord:
    if candidate.owner_id is None:
        raise ValueError("memory candidate has no owner")
    return MemoryRecord(
        owner_id=candidate.owner_id,
        content=candidate.content,
        importance=candidate.importance,
        visibility=candidate.visibility,
        kind=candidate.kind,
        sensitivity=candidate.sensitivity,
        confidence=candidate.confidence,
        retention=candidate.retention,
        audience_ids=candidate.audience_ids,
        source_turn_id=candidate.source_turn_id,
        supersedes_id=candidate.supersedes_id,
        topic_key=candidate.topic_key,
        derived_from_ids=candidate.derived_from_ids,
        expires_at=candidate.expires_at,
        metadata=candidate.metadata,
    )
