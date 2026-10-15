from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from kinveil.vision.face import TrackSnapshot

from .models import SpeakerTrackSnapshot


@dataclass(frozen=True)
class MouthMotion:
    face_track_id: str
    person_id: str | None
    label: str
    observed_at: float
    score: float
    visible: bool


@dataclass(frozen=True)
class ActiveSpeakerResult:
    person_id: str | None
    label: str
    accepted: bool
    reason: str
    face_track_id: str | None
    voice_track_id: str
    lip_score: float


class MouthMotionAnalyzer:
    """Estimate lip-area motion without retaining camera frames."""

    def __init__(self, *, history_seconds: float = 30.0) -> None:
        self.history_seconds = history_seconds
        self._previous: dict[str, np.ndarray] = {}
        self._history: deque[MouthMotion] = deque()

    def observe(
        self,
        frame: np.ndarray,
        faces: tuple[TrackSnapshot, ...],
        *,
        observed_at: float,
    ) -> tuple[MouthMotion, ...]:
        observations: list[MouthMotion] = []
        active_ids = {face.track_id for face in faces}
        for track_id in set(self._previous) - active_ids:
            del self._previous[track_id]
        for face in faces:
            mouth = self._mouth_patch(frame, face)
            previous = self._previous.get(face.track_id)
            visible = mouth is not None
            score = 0.0
            if mouth is not None:
                if previous is not None:
                    score = float(np.mean(np.abs(mouth.astype(np.float32) - previous))) / 255.0
                self._previous[face.track_id] = mouth
            observation = MouthMotion(
                face.track_id,
                face.person_id,
                face.label,
                observed_at,
                score,
                visible,
            )
            self._history.append(observation)
            observations.append(observation)
        self._expire(observed_at)
        return tuple(observations)

    def history(self) -> tuple[MouthMotion, ...]:
        return tuple(self._history)

    def _expire(self, now: float) -> None:
        cutoff = now - self.history_seconds
        while self._history and self._history[0].observed_at < cutoff:
            self._history.popleft()

    @staticmethod
    def _mouth_patch(frame: np.ndarray, face: TrackSnapshot) -> np.ndarray | None:
        height, width = frame.shape[:2]
        if face.landmarks is None or len(face.landmarks) < 5:
            return None
        left_corner = np.asarray(face.landmarks[3], dtype=np.float32)
        right_corner = np.asarray(face.landmarks[4], dtype=np.float32)
        mouth_width = float(np.linalg.norm(right_corner - left_corner))
        if not np.isfinite(mouth_width) or mouth_width < max(6.0, face.box.width * 0.10):
            return None
        center = (left_corner + right_corner) / 2.0
        left = max(0, int(center[0] - mouth_width * 0.9))
        right = min(width, int(center[0] + mouth_width * 0.9))
        top = max(0, int(center[1] - mouth_width * 0.55))
        bottom = min(height, int(center[1] + mouth_width * 0.65))
        if right - left < 12 or bottom - top < 8:
            return None
        patch = frame[top:bottom, left:right]
        if patch.size == 0:
            return None
        gray = np.mean(patch.astype(np.float32), axis=2) if patch.ndim == 3 else patch
        y_indices = np.linspace(0, gray.shape[0] - 1, 16).astype(int)
        x_indices = np.linspace(0, gray.shape[1] - 1, 32).astype(int)
        return np.asarray(gray[np.ix_(y_indices, x_indices)], dtype=np.uint8)


class AudioVisualSpeakerResolver:
    """Fuse lip motion, face identity, and voice identity conservatively."""

    def __init__(
        self,
        *,
        motion_threshold: float = 0.012,
        simultaneous_motion_threshold: float = 0.010,
        min_observations: int = 3,
        synchronization_margin_seconds: float = 0.2,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.simultaneous_motion_threshold = simultaneous_motion_threshold
        self.min_observations = min_observations
        self.synchronization_margin_seconds = synchronization_margin_seconds

    def resolve(
        self,
        voice: SpeakerTrackSnapshot,
        motions: tuple[MouthMotion, ...],
        *,
        segment_started_at: float,
        segment_ended_at: float,
    ) -> ActiveSpeakerResult:
        grouped: dict[str, list[MouthMotion]] = defaultdict(list)
        for motion in motions:
            if (
                segment_started_at - self.synchronization_margin_seconds
                <= motion.observed_at
                <= segment_ended_at + self.synchronization_margin_seconds
                and motion.visible
            ):
                grouped[motion.face_track_id].append(motion)
        scored = []
        for values in grouped.values():
            if len(values) < self.min_observations:
                continue
            scores = np.asarray([value.score for value in values], dtype=np.float32)
            activity = float(np.mean(np.sort(scores)[-max(1, len(scores) // 2) :]))
            scored.append((activity, values[-1]))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or scored[0][0] < self.motion_threshold:
            return self._unknown(voice, "no synchronized visible mouth", scored)

        if voice.person_id is not None:
            matching = [
                (score, face)
                for score, face in scored
                if score >= self.motion_threshold and face.person_id == voice.person_id
            ]
            if len(matching) == 1:
                lip_score, face = matching[0]
                return ActiveSpeakerResult(
                    face.person_id,
                    face.label,
                    True,
                    "face, voice, and lip motion agree",
                    face.face_track_id,
                    voice.track_id,
                    lip_score,
                )
            if len(matching) > 1:
                return self._unknown(voice, "voice matches multiple visible faces", scored)
        if len(scored) > 1 and scored[1][0] >= self.simultaneous_motion_threshold:
            return self._unknown(voice, "multiple mouths moving during speech", scored)

        lip_score, face = scored[0]
        if voice.person_id is not None and face.person_id is not None:
            if voice.person_id != face.person_id:
                return self._unknown(voice, "face and voice identities conflict", scored)
            return ActiveSpeakerResult(
                face.person_id,
                face.label,
                True,
                "face, voice, and lip motion agree",
                face.face_track_id,
                voice.track_id,
                lip_score,
            )
        if face.person_id is not None and voice.person_id is None:
            return ActiveSpeakerResult(
                face.person_id,
                face.label,
                True,
                "face and lip motion agree; voice remains transient",
                face.face_track_id,
                voice.track_id,
                lip_score,
            )
        if face.person_id is None and voice.person_id is None:
            return ActiveSpeakerResult(
                None,
                face.label,
                True,
                "temporary face and lip motion agree",
                face.face_track_id,
                voice.track_id,
                lip_score,
            )
        return self._unknown(voice, "voice identity lacks matching face identity", scored)

    @staticmethod
    def _unknown(
        voice: SpeakerTrackSnapshot,
        reason: str,
        scored: list[tuple[float, MouthMotion]],
    ) -> ActiveSpeakerResult:
        return ActiveSpeakerResult(
            None,
            "Unknown speaker",
            False,
            reason,
            scored[0][1].face_track_id if scored else None,
            voice.track_id,
            scored[0][0] if scored else 0.0,
        )
