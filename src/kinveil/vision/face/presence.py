from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

import numpy as np

from .backend import FaceBackend, normalize
from .models import DetectedFace, FaceBox, Person, TrackSnapshot
from .recognizer import FaceRecognizer
from .store import FaceStore


@dataclass(frozen=True)
class PresencePermissions:
    camera_allowed: bool
    transient_processing_allowed: bool


@dataclass
class _Track:
    id: str
    visitor_number: int
    box: FaceBox
    last_seen: float
    person_id: str | None = None
    label: str = ""
    identity_confidence: float = 0.0
    observations: int = 0
    embeddings: list[np.ndarray] = field(default_factory=list)
    pose_labels: list[str] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        return normalize(np.mean(np.vstack(self.embeddings), axis=0))


class PresenceEngine:
    """Resolves sampled camera frames without persisting unknown visitors."""

    def __init__(
        self,
        backend: FaceBackend,
        store: FaceStore,
        permissions: PresencePermissions,
        *,
        identity_threshold: float = 0.48,
        identity_margin: float = 0.06,
        track_threshold: float = 0.72,
        track_ttl_seconds: float = 8.0,
        max_transient_samples: int = 12,
        max_persistent_samples: int = 24,
        max_samples_per_pose: int = 8,
        template_update_threshold: float = 0.60,
        min_detection_score: float = 0.70,
        min_face_size: int = 80,
    ) -> None:
        if not permissions.camera_allowed:
            raise PermissionError("camera permission is required")
        if not permissions.transient_processing_allowed:
            raise PermissionError("transient face processing permission is required")
        self.backend = backend
        self.store = store
        self.recognizer = FaceRecognizer(store, identity_threshold, identity_margin)
        self.track_threshold = track_threshold
        self.track_ttl_seconds = track_ttl_seconds
        self.max_transient_samples = max_transient_samples
        self.max_persistent_samples = max_persistent_samples
        self.max_samples_per_pose = max_samples_per_pose
        self.template_update_threshold = template_update_threshold
        self.min_detection_score = min_detection_score
        self.min_face_size = min_face_size
        self._tracks: dict[str, _Track] = {}
        self._next_visitor_number = 1

    def process_frame(
        self, frame: np.ndarray, *, observed_at: float | None = None
    ) -> tuple[TrackSnapshot, ...]:
        now = monotonic() if observed_at is None else observed_at
        self._expire_tracks(now)
        faces = tuple(
            face
            for face in self.backend.detect_and_embed(frame)
            if face.detection_score >= self.min_detection_score
            and min(face.box.width, face.box.height) >= self.min_face_size
        )
        assigned: set[str] = set()
        snapshots: list[TrackSnapshot] = []
        for face in faces:
            track = self._resolve_face(face, now, assigned)
            assigned.add(track.id)
            snapshots.append(self._snapshot(track))
        return tuple(snapshots)

    def active_tracks(self) -> tuple[TrackSnapshot, ...]:
        return tuple(self._snapshot(track) for track in self._tracks.values())

    def persist_track(
        self,
        track_id: str,
        *,
        biometric_consent: bool,
        display_name: str | None = None,
    ) -> Person:
        if not biometric_consent:
            raise ValueError("explicit biometric consent is required")
        track = self._require_track(track_id)
        if track.person_id:
            person = self.store.get_person(track.person_id)
            if display_name:
                person = self.store.set_display_name(person.id, display_name)
            return person
        if not track.embeddings:
            raise ValueError("track has no usable face samples")
        person = self.store.create_person(display_name, consent=True)
        for embedding, pose_label, quality_score in zip(
            track.embeddings,
            track.pose_labels,
            track.quality_scores,
            strict=True,
        ):
            self.store.maintain_embedding_pool(
                person.id,
                embedding,
                pose_label=pose_label,
                quality_score=quality_score,
                max_samples=self.max_persistent_samples,
                max_samples_per_pose=self.max_samples_per_pose,
            )
        track.person_id = person.id
        track.label = person.name
        track.identity_confidence = 1.0
        return self.store.get_person(person.id)

    def confirm_display_name(
        self, person_id: str, display_name: str, *, confirmed: bool
    ) -> Person:
        if not confirmed:
            raise ValueError("a conversational name must be confirmed before saving")
        person = self.store.set_display_name(person_id, display_name)
        for track in self._tracks.values():
            if track.person_id == person_id:
                track.label = person.name
        return person

    def forget_track(self, track_id: str) -> None:
        self._tracks.pop(track_id, None)

    def _resolve_face(
        self, face: DetectedFace, now: float, assigned: set[str]
    ) -> _Track:
        embedding = normalize(face.embedding)
        match = self.recognizer.match(embedding)
        if match.accepted and match.person_id:
            track = next(
                (
                    item
                    for item in self._tracks.values()
                    if item.person_id == match.person_id and item.id not in assigned
                ),
                None,
            )
            if track is None:
                track = self._new_track(face.box, now)
            track.person_id = match.person_id
            track.label = match.name or track.label
            track.identity_confidence = match.similarity
            if (
                match.similarity >= self.template_update_threshold
                and face.detection_score >= 0.85
            ):
                self.store.maintain_embedding_pool(
                    match.person_id,
                    embedding,
                    pose_label=face.pose_label,
                    quality_score=face.detection_score,
                    max_samples=self.max_persistent_samples,
                    max_samples_per_pose=self.max_samples_per_pose,
                )
        else:
            track = self._closest_transient_track(embedding, assigned)
            if track is None:
                track = self._new_track(face.box, now)
            track.identity_confidence = max(0.0, match.similarity)
        track.box = face.box
        track.last_seen = now
        track.observations += 1
        self._add_sample(track, embedding, face.pose_label, face.detection_score)
        return track

    def _closest_transient_track(
        self, embedding: np.ndarray, assigned: set[str]
    ) -> _Track | None:
        candidates = [
            (float(track.centroid @ embedding), track)
            for track in self._tracks.values()
            if track.person_id is None and track.id not in assigned and track.embeddings
        ]
        if not candidates:
            return None
        score, track = max(candidates, key=lambda item: item[0])
        return track if score >= self.track_threshold else None

    def _new_track(self, box: FaceBox, now: float) -> _Track:
        number = self._next_visitor_number
        self._next_visitor_number += 1
        track = _Track(
            id=str(uuid4()),
            visitor_number=number,
            box=box,
            last_seen=now,
            label=f"Visitor {_visitor_label(number)}",
        )
        self._tracks[track.id] = track
        return track

    def _add_sample(
        self,
        track: _Track,
        embedding: np.ndarray,
        pose_label: str,
        quality_score: float,
    ) -> None:
        if track.embeddings and float(track.centroid @ embedding) > 0.995:
            return
        track.embeddings.append(embedding)
        track.pose_labels.append(pose_label)
        track.quality_scores.append(quality_score)
        if len(track.embeddings) > self.max_transient_samples:
            track.embeddings.pop(0)
            track.pose_labels.pop(0)
            track.quality_scores.pop(0)

    def _expire_tracks(self, now: float) -> None:
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if now - track.last_seen > self.track_ttl_seconds
        ]
        for track_id in stale:
            del self._tracks[track_id]

    def _require_track(self, track_id: str) -> _Track:
        try:
            return self._tracks[track_id]
        except KeyError as exc:
            raise KeyError(f"unknown or expired track: {track_id}") from exc

    @staticmethod
    def _snapshot(track: _Track) -> TrackSnapshot:
        return TrackSnapshot(
            track_id=track.id,
            person_id=track.person_id,
            label=track.label,
            box=track.box,
            identity_confidence=track.identity_confidence,
            persistent=track.person_id is not None,
            last_seen=track.last_seen,
            observations=track.observations,
        )


def _visitor_label(number: int) -> str:
    result = ""
    value = number
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result
