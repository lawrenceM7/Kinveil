from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from .backend import VoiceEmbeddingBackend, normalize_voice_embedding
from .models import SpeakerTrackSnapshot
from .recognizer import VoiceRecognizer
from .store import VoiceStore


@dataclass
class _SpeakerTrack:
    id: str
    label: str
    person_id: str | None = None
    confidence: float = 0.0
    observations: int = 0
    last_heard: float = 0.0
    embeddings: list[np.ndarray] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        return normalize_voice_embedding(np.mean(np.vstack(self.embeddings), axis=0))


class VoiceIdentityEngine:
    """Resolve already-segmented, single-speaker speech into session tracks.

    Voice activity detection and speaker diarization belong upstream. This class
    never stores the supplied audio samples; only normalized embeddings can be
    retained, and persistence requires explicit voice-biometric consent.
    """

    def __init__(
        self,
        backend: VoiceEmbeddingBackend,
        store: VoiceStore,
        *,
        recognition_threshold: float = 0.65,
        ambiguity_margin: float = 0.08,
        track_threshold: float = 0.78,
        track_ttl_seconds: float = 45.0,
        max_transient_samples: int = 12,
        max_persistent_samples: int = 24,
        update_quality_threshold: float = 0.70,
    ) -> None:
        self.backend = backend
        self.store = store
        self.recognizer = VoiceRecognizer(
            store,
            threshold=recognition_threshold,
            margin=ambiguity_margin,
        )
        self.track_threshold = track_threshold
        self.track_ttl_seconds = track_ttl_seconds
        self.max_transient_samples = max_transient_samples
        self.max_persistent_samples = max_persistent_samples
        self.update_quality_threshold = update_quality_threshold
        self._tracks: dict[str, _SpeakerTrack] = {}
        self._next_speaker_number = 0

    def process_audio(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        quality_score: float = 0.0,
        observed_at: float | None = None,
        allow_persistent_update: bool = True,
    ) -> SpeakerTrackSnapshot:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            raise ValueError("audio samples must be finite and non-empty")
        embedding = self.backend.embed(audio, sample_rate)
        return self.process_embedding(
            embedding,
            quality_score=quality_score,
            observed_at=observed_at,
            allow_persistent_update=allow_persistent_update,
        )

    def process_embedding(
        self,
        embedding: np.ndarray,
        *,
        quality_score: float = 0.0,
        observed_at: float | None = None,
        allow_persistent_update: bool = True,
    ) -> SpeakerTrackSnapshot:
        now = time.monotonic() if observed_at is None else observed_at
        self.expire_tracks(now)
        vector = normalize_voice_embedding(embedding)
        match = self.recognizer.match(vector)

        if match.accepted and match.person_id is not None:
            track = self._known_track(match.person_id)
            if track is None:
                track = self._new_track(
                    now,
                    person_id=match.person_id,
                    label=self.store.person_label(match.person_id),
                )
            track.confidence = match.similarity
            if allow_persistent_update and quality_score >= self.update_quality_threshold:
                self.store.maintain_embedding_pool(
                    match.person_id,
                    vector,
                    quality_score=quality_score,
                    max_samples=self.max_persistent_samples,
                )
        else:
            track = self._nearest_transient_track(vector)
            if track is None:
                track = self._new_track(now)
            track.confidence = 0.0

        self._observe(track, vector, quality_score, now)
        return self._snapshot(track)

    def persist_track(
        self,
        track_id: str,
        person_id: str,
        *,
        voice_consent: bool,
        minimum_quality: float | None = None,
        minimum_samples: int = 1,
    ) -> SpeakerTrackSnapshot:
        if not voice_consent:
            raise ValueError("explicit voice biometric consent is required")
        track = self._tracks.get(track_id)
        if track is None:
            raise KeyError(f"unknown speaker track: {track_id}")
        if track.person_id is not None and track.person_id != person_id:
            raise ValueError("speaker track is already bound to another person")
        if minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")

        selected = [
            (vector, quality)
            for vector, quality in zip(track.embeddings, track.qualities, strict=True)
            if minimum_quality is None or quality >= minimum_quality
        ]
        if len(selected) < minimum_samples:
            raise ValueError("not enough high-quality voice samples to persist")

        self.store.grant_consent(person_id, consent=True)
        for vector, quality in selected:
            self.store.maintain_embedding_pool(
                person_id,
                vector,
                quality_score=quality,
                max_samples=self.max_persistent_samples,
            )
        track.person_id = person_id
        track.label = self.store.person_label(person_id)
        track.confidence = 1.0
        return self._snapshot(track)

    def expire_tracks(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if current - track.last_heard > self.track_ttl_seconds
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def tracks(self) -> tuple[SpeakerTrackSnapshot, ...]:
        return tuple(self._snapshot(track) for track in self._tracks.values())

    def _known_track(self, person_id: str) -> _SpeakerTrack | None:
        return next(
            (track for track in self._tracks.values() if track.person_id == person_id),
            None,
        )

    def _nearest_transient_track(self, vector: np.ndarray) -> _SpeakerTrack | None:
        compatible = [
            track
            for track in self._tracks.values()
            if track.person_id is None and track.embeddings[0].shape == vector.shape
        ]
        if not compatible:
            return None
        score, track = max(
            ((float(candidate.centroid @ vector), candidate) for candidate in compatible),
            key=lambda result: result[0],
        )
        return track if score >= self.track_threshold else None

    def _new_track(
        self,
        now: float,
        *,
        person_id: str | None = None,
        label: str | None = None,
    ) -> _SpeakerTrack:
        if label is None:
            label = self._speaker_label(self._next_speaker_number)
            self._next_speaker_number += 1
        track = _SpeakerTrack(str(uuid.uuid4()), label, person_id=person_id, last_heard=now)
        self._tracks[track.id] = track
        return track

    def _observe(
        self,
        track: _SpeakerTrack,
        vector: np.ndarray,
        quality_score: float,
        now: float,
    ) -> None:
        track.observations += 1
        track.last_heard = now
        track.embeddings.append(vector.copy())
        track.qualities.append(float(quality_score))
        if len(track.embeddings) > self.max_transient_samples:
            del track.embeddings[0]
            del track.qualities[0]

    @staticmethod
    def _snapshot(track: _SpeakerTrack) -> SpeakerTrackSnapshot:
        return SpeakerTrackSnapshot(
            track.id,
            track.person_id,
            track.label,
            track.confidence,
            track.person_id is not None,
            track.observations,
            track.last_heard,
        )

    @staticmethod
    def _speaker_label(index: int) -> str:
        letters = ""
        number = index
        while True:
            number, remainder = divmod(number, 26)
            letters = chr(ord("A") + remainder) + letters
            if number == 0:
                break
            number -= 1
        return f"Speaker {letters}"
