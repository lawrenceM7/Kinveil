from __future__ import annotations

from typing import Protocol

import numpy as np

from .models import DetectedFace, FaceBox


class FaceBackend(Protocol):
    def detect_and_embed(self, frame: np.ndarray) -> tuple[DetectedFace, ...]: ...


class InsightFaceBackend:
    """Lazy adapter around InsightFace so the core remains testable without ML packages."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        detector_size: tuple[int, int] = (640, 640),
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "Face dependencies are missing. Install with: pip install 'kinveil[face]'"
            ) from exc

        self._app = FaceAnalysis(
            name=model_name,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=detector_size)

    def detect_and_embed(self, frame: np.ndarray) -> tuple[DetectedFace, ...]:
        results: list[DetectedFace] = []
        for face in self._app.get(frame):
            vector = np.asarray(face.normed_embedding, dtype=np.float32)
            raw_pose = getattr(face, "pose", None)
            pose = None
            if raw_pose is not None:
                values = np.asarray(raw_pose, dtype=np.float32).reshape(-1)
                if values.size >= 3 and np.all(np.isfinite(values[:3])):
                    pose = tuple(float(value) for value in values[:3])
            results.append(
                DetectedFace(
                    box=FaceBox(*(int(value) for value in face.bbox)),
                    embedding=normalize(vector),
                    detection_score=float(face.det_score),
                    pose=pose,
                )
            )
        return tuple(results)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("face embedding must be finite and non-zero")
    return vector / norm
