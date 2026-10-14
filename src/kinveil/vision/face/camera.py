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
        enrollment_observations: int = 4,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample interval must be positive")
        if enrollment_observations <= 0:
            raise ValueError("enrollment observations must be positive")
        self.engine = engine
        self.camera_index = camera_index
        self.sample_interval_seconds = sample_interval_seconds
        self.window_title = window_title
        self.enrollment_handler = enrollment_handler
        self.enrollment_observations = enrollment_observations

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
        pending_enrollment: TrackSnapshot | None = None
        prompted_tracks: set[str] = set()
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("camera stopped returning frames")
                now = monotonic()
                if now - last_sample >= self.sample_interval_seconds:
                    snapshots = self.engine.process_frame(frame, observed_at=now)
                    last_sample = now
                    active_ids = {
                        snapshot.track_id for snapshot in self.engine.active_tracks()
                    }
                    prompted_tracks.intersection_update(active_ids)
                    if pending_enrollment is None and self.enrollment_handler is not None:
                        pending_enrollment = self._next_enrollment_candidate(
                            snapshots,
                            prompted_tracks,
                            self.enrollment_observations,
                        )
                        if pending_enrollment is not None:
                            prompted_tracks.add(pending_enrollment.track_id)
                self._draw(
                    cv2,
                    frame,
                    snapshots,
                    status_message if now < status_until else "",
                    pending_enrollment,
                )
                cv2.imshow(self.window_title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if pending_enrollment is not None and key in (ord("y"), ord("Y")):
                    try:
                        result = self.enrollment_handler(pending_enrollment)
                        status_message = result or "Enrollment cancelled"
                    except (KeyError, ValueError) as exc:
                        status_message = str(exc)
                    pending_enrollment = None
                    status_until = monotonic() + 4.0
                elif pending_enrollment is not None and key in (ord("n"), ord("N")):
                    status_message = (
                        f"{pending_enrollment.label} remains temporary for this session"
                    )
                    pending_enrollment = None
                    status_until = monotonic() + 4.0
        finally:
            camera.release()
            cv2.destroyAllWindows()

    @staticmethod
    def _next_enrollment_candidate(
        snapshots: tuple[TrackSnapshot, ...],
        prompted_tracks: set[str],
        required_observations: int,
    ) -> TrackSnapshot | None:
        return next(
            (
                snapshot
                for snapshot in snapshots
                if not snapshot.persistent
                and snapshot.observations >= required_observations
                and snapshot.track_id not in prompted_tracks
            ),
            None,
        )

    @staticmethod
    def _draw(
        cv2: object,
        frame: object,
        snapshots: tuple[TrackSnapshot, ...],
        status_message: str = "",
        pending_enrollment: TrackSnapshot | None = None,
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
            "Unknown visitors stay temporary until they consent   Q/Esc: quit",
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
        if pending_enrollment is not None:
            cv2.putText(
                frame,
                f"Recognize {pending_enrollment.label} on future visits? Y / N",
                (16, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (80, 230, 255),
                2,
            )
