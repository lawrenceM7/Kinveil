from __future__ import annotations

from pathlib import Path

import numpy as np

from .backend import FaceBackend, normalize
from .models import EnrollmentSummary
from .store import FaceStore


class FaceEnrollment:
    def __init__(
        self,
        backend: FaceBackend,
        store: FaceStore,
        *,
        sample_every: int = 8,
        max_samples: int = 24,
        min_face_size: int = 100,
        min_detection_score: float = 0.70,
        min_diversity: float = 0.015,
    ) -> None:
        self.backend = backend
        self.store = store
        self.sample_every = sample_every
        self.max_samples = max_samples
        self.min_face_size = min_face_size
        self.min_detection_score = min_detection_score
        self.min_diversity = min_diversity

    def enroll_video(self, name: str, video: Path, *, consent: bool) -> EnrollmentSummary:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required. Install with: pip install 'kinveil[face]'"
            ) from exc

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video}")

        person = self.store.create_person(name, consent)
        frames_seen = rejected = 0
        accepted: list[np.ndarray] = []
        try:
            while len(accepted) < self.max_samples:
                ok, frame = capture.read()
                if not ok:
                    break
                frames_seen += 1
                if (frames_seen - 1) % self.sample_every:
                    continue
                faces = self.backend.detect_and_embed(frame)
                if len(faces) != 1:
                    rejected += 1
                    continue
                face = faces[0]
                if (
                    face.detection_score < self.min_detection_score
                    or min(face.box.width, face.box.height) < self.min_face_size
                ):
                    rejected += 1
                    continue
                vector = normalize(face.embedding)
                too_similar = accepted and max(
                    float(known @ vector) for known in accepted
                ) > 1.0 - self.min_diversity
                if too_similar:
                    rejected += 1
                    continue
                self.store.add_embedding(
                    person.id,
                    vector,
                    pose_label=face.pose_label,
                    quality_score=face.detection_score,
                )
                accepted.append(vector)
        finally:
            capture.release()

        if not accepted:
            self.store.delete_person(person.id)
            raise ValueError("no usable single-face frames found; enrollment was rolled back")
        return EnrollmentSummary(
            self.store.get_person(person.id),
            frames_seen,
            len(accepted),
            rejected,
        )
