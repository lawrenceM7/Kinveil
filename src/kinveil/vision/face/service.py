from __future__ import annotations

from pathlib import Path

from .backend import FaceBackend
from .models import FaceMatch
from .recognizer import FaceRecognizer


def recognize_image(
    path: Path,
    backend: FaceBackend,
    recognizer: FaceRecognizer,
) -> tuple[FaceMatch, ...]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install with: pip install 'kinveil[face]'") from exc
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return tuple(recognizer.match(face.embedding) for face in backend.detect_and_embed(image))
