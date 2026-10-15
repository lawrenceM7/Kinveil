from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class FaceBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class DetectedFace:
    box: FaceBox
    embedding: np.ndarray
    detection_score: float
    pose: tuple[float, float, float] | None = None
    landmarks: tuple[tuple[float, float], ...] | None = None

    @property
    def pose_label(self) -> str:
        """Return a coarse pose bucket from InsightFace's pitch/yaw/roll values."""
        if self.pose is None:
            return "unknown"
        pitch, yaw, _roll = self.pose
        if yaw <= -18.0:
            return "left"
        if yaw >= 18.0:
            return "right"
        if pitch <= -12.0:
            return "up"
        if pitch >= 12.0:
            return "down"
        return "front"


@dataclass(frozen=True)
class Person:
    id: str
    display_name: str | None
    local_alias: str
    created_at: datetime
    sample_count: int
    biometric_consented_at: datetime | None

    @property
    def name(self) -> str:
        """Compatibility label for callers written before anonymous identities."""
        return self.display_name or self.local_alias


@dataclass(frozen=True)
class FaceMatch:
    person_id: str | None
    name: str | None
    similarity: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class EnrollmentSummary:
    person: Person
    frames_seen: int
    samples_saved: int
    rejected_frames: int


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: str
    person_id: str | None
    label: str
    box: FaceBox
    identity_confidence: float
    persistent: bool
    last_seen: float
    observations: int
    landmarks: tuple[tuple[float, float], ...] | None = None
