from __future__ import annotations

from collections.abc import Callable
from time import monotonic

from .models import TrackSnapshot
from .presence import PresenceEngine


class CameraPresenceRunner:
    """Temporary OpenCV shell around the UI-independent presence engine."""

    def __init__(
        self,
        engine: PresenceEngine,
        *,
        camera_index: int = 0,
        sample_interval_seconds: float = 0.5,
        window_title: str = "Kinveil Presence",
        enrollment_handler: Callable[[TrackSnapshot], str | None] | None = None,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample interval must be positive")
        self.engine = engine
        self.camera_index = camera_index
        self.sample_interval_seconds = sample_interval_seconds
        self.window_title = window_title
        self.enrollment_handler = enrollment_handler

    def run(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required. Install with: pip install 'kinveil[face]'"
            ) from exc
        camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            raise RuntimeError(f"cannot open camera {self.camera_index}")
        snapshots: tuple[TrackSnapshot, ...] = ()
        last_sample = float("-inf")
        status_message = ""
        status_until = float("-inf")
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("camera stopped returning frames")
                now = monotonic()
                if now - last_sample >= self.sample_interval_seconds:
                    snapshots = self.engine.process_frame(frame, observed_at=now)
                    last_sample = now
                self._draw(
                    cv2,
                    frame,
                    snapshots,
                    status_message if now < status_until else "",
                )
                cv2.imshow(self.window_title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("e"):
                    unknown = [snapshot for snapshot in snapshots if not snapshot.persistent]
                    if self.enrollment_handler is None:
                        status_message = "Enrollment is unavailable"
                    elif len(unknown) != 1:
                        status_message = "Keep exactly one unknown visitor in view"
                    else:
                        result = self.enrollment_handler(unknown[0])
                        status_message = result or "Enrollment cancelled"
                    status_until = monotonic() + 4.0
        finally:
            camera.release()
            cv2.destroyAllWindows()

    @staticmethod
    def _draw(
        cv2: object,
        frame: object,
        snapshots: tuple[TrackSnapshot, ...],
        status_message: str = "",
    ) -> None:
        for snapshot in snapshots:
            box = snapshot.box
            color = (62, 191, 110) if snapshot.persistent else (70, 170, 255)
            cv2.rectangle(frame, (box.left, box.top), (box.right, box.bottom), color, 2)
            label = snapshot.label
            if snapshot.persistent:
                label += f" {snapshot.identity_confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (box.left, max(20, box.top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        cv2.putText(
            frame,
            "E: save one visitor   Q/Esc: quit",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        if status_message:
            cv2.putText(
                frame,
                status_message,
                (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (90, 220, 255),
                2,
            )
