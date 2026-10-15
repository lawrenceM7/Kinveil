from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

import numpy as np

from kinveil.audio.stt import AudioTranscriber, Transcript
from kinveil.vision.face import PresenceEngine, TrackSnapshot

from .active_speaker import (
    ActiveSpeakerResult,
    AudioVisualSpeakerResolver,
    MouthMotionAnalyzer,
)
from .engine import VoiceIdentityEngine
from .live import SpeechSegment, VoiceSegmenter, _import_sounddevice, _segment_quality
from .models import SpeakerTrackSnapshot
from .overlap import OverlapDetection, OverlapDetector
from .separation import SpeechSeparator


@dataclass(frozen=True)
class _VoiceObservation:
    voice: SpeakerTrackSnapshot
    transcript: Transcript | None = None


@dataclass(frozen=True)
class _VoiceEvent:
    voices: tuple[_VoiceObservation, ...]
    started_at: float
    ended_at: float
    overlap: OverlapDetection

    @property
    def voice(self) -> SpeakerTrackSnapshot | None:
        return self.voices[0].voice if self.voices else None


@dataclass(frozen=True)
class _QueuedSpeechSegment:
    samples: np.ndarray
    started_at: float
    ended_at: float


class AudioVisualSessionRunner:
    """Run camera presence, VAD, voice identity, and lip synchronization together."""

    def __init__(
        self,
        presence: PresenceEngine,
        voice: VoiceIdentityEngine,
        segmenter: VoiceSegmenter,
        overlap_detector: OverlapDetector,
        separator: SpeechSeparator | None = None,
        transcriber: AudioTranscriber | None = None,
        *,
        camera_index: int = 0,
        microphone_device: int | str | None = None,
        face_interval_seconds: float = 0.5,
        window_title: str = "Kinveil Audio-Visual Presence",
        on_transcript: Callable[[ActiveSpeakerResult, Transcript], None] | None = None,
        face_enrollment_handler: Callable[[TrackSnapshot], str | None] | None = None,
        face_enrollment_observations: int = 4,
        auto_voice_enrollment: bool = False,
        voice_enrollment_observations: int = 3,
        show_transcript_overlay: bool = True,
    ) -> None:
        if face_enrollment_observations <= 0:
            raise ValueError("face enrollment observations must be positive")
        if voice_enrollment_observations <= 0:
            raise ValueError("voice enrollment observations must be positive")
        self.presence = presence
        self.voice = voice
        self.segmenter = segmenter
        self.overlap_detector = overlap_detector
        self.separator = separator
        self.transcriber = transcriber
        self.camera_index = camera_index
        self.microphone_device = microphone_device
        self.face_interval_seconds = face_interval_seconds
        self.window_title = window_title
        self.on_transcript = on_transcript
        self.face_enrollment_handler = face_enrollment_handler
        self.face_enrollment_observations = face_enrollment_observations
        self.auto_voice_enrollment = auto_voice_enrollment
        self.voice_enrollment_observations = voice_enrollment_observations
        self.show_transcript_overlay = show_transcript_overlay
        self.motion = MouthMotionAnalyzer()
        self.resolver = AudioVisualSpeakerResolver()
        self._events: queue.Queue[_VoiceEvent | Exception] = queue.Queue()
        self._segments: queue.Queue[_QueuedSpeechSegment] = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._voice_face_agreement: dict[tuple[str, str], int] = {}

    def run(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required; install Kinveil's face extra") from exc
        camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            raise RuntimeError(f"cannot open camera {self.camera_index}")
        capture_thread = threading.Thread(target=self._audio_loop, daemon=True)
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        capture_thread.start()
        processing_thread.start()
        faces: tuple[TrackSnapshot, ...] = ()
        active: tuple[tuple[ActiveSpeakerResult, Transcript | None], ...] = ()
        active_until = 0.0
        last_face_sample = float("-inf")
        pending_enrollment: TrackSnapshot | None = None
        prompted_tracks: set[str] = set()
        status_message = ""
        status_until = float("-inf")
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("camera stopped returning frames")
                now = monotonic()
                if now - last_face_sample >= self.face_interval_seconds:
                    faces = self.presence.process_frame(frame, observed_at=now)
                    last_face_sample = now
                    active_face_ids = {
                        snapshot.track_id for snapshot in self.presence.active_tracks()
                    }
                    prompted_tracks.intersection_update(active_face_ids)
                    if pending_enrollment is None:
                        pending_enrollment = self._next_enrollment_candidate(
                            faces,
                            prompted_tracks,
                        )
                        if pending_enrollment is not None:
                            prompted_tracks.add(pending_enrollment.track_id)
                self.motion.observe(frame, faces, observed_at=now)
                event = self._latest_voice_event()
                if isinstance(event, Exception):
                    raise event
                if event is not None:
                    if event.overlap.detected:
                        if event.voices:
                            active = tuple(
                                (
                                    self.resolver.resolve(
                                        observation.voice,
                                        self.motion.history(),
                                        segment_started_at=event.started_at,
                                        segment_ended_at=event.ended_at,
                                    ),
                                    observation.transcript,
                                )
                                for observation in event.voices
                            )
                        else:
                            active = (
                                (
                                    ActiveSpeakerResult(
                                        None,
                                        "Unknown speaker",
                                        False,
                                        "overlap detected but speech separation is unavailable",
                                        None,
                                        "acoustic-overlap",
                                        0.0,
                                    ),
                                    None,
                                ),
                            )
                    elif event.voices:
                        observation = event.voices[0]
                        result = self.resolver.resolve(
                            observation.voice,
                            self.motion.history(),
                            segment_started_at=event.started_at,
                            segment_ended_at=event.ended_at,
                        )
                        active = (
                            (
                                result,
                                observation.transcript,
                            ),
                        )
                        if self._maybe_persist_voice(result, observation.voice):
                            status_message = f"Saved voice identity for {result.label}"
                            print(status_message)
                            status_until = monotonic() + 4.0
                    self._report_transcripts(active)
                    active_until = now + 4.0
                self._draw(
                    cv2,
                    frame,
                    faces,
                    active if now < active_until else (),
                    pending_enrollment,
                    status_message if now < status_until else "",
                    self.show_transcript_overlay,
                )
                cv2.imshow(self.window_title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if pending_enrollment is not None and key in (ord("y"), ord("Y")):
                    try:
                        result = self.face_enrollment_handler(pending_enrollment)
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
            self._stop.set()
            capture_thread.join(timeout=2.0)
            processing_thread.join(timeout=2.0)
            camera.release()
            cv2.destroyAllWindows()

    def _report_transcripts(
        self,
        active: tuple[tuple[ActiveSpeakerResult, Transcript | None], ...],
    ) -> None:
        if self.on_transcript is None:
            return
        for result, transcript in active:
            if transcript is not None and transcript.text:
                self.on_transcript(result, transcript)

    def _next_enrollment_candidate(
        self,
        faces: tuple[TrackSnapshot, ...],
        prompted_tracks: set[str],
    ) -> TrackSnapshot | None:
        if self.face_enrollment_handler is None:
            return None
        return next(
            (
                face
                for face in faces
                if not face.persistent
                and face.observations >= self.face_enrollment_observations
                and face.track_id not in prompted_tracks
            ),
            None,
        )

    def _audio_loop(self) -> None:
        try:
            sounddevice = _import_sounddevice()
            block_size = int(self.segmenter.sample_rate * 0.1)
            with sounddevice.InputStream(
                channels=1,
                dtype="float32",
                samplerate=self.segmenter.sample_rate,
                device=self.microphone_device,
            ) as stream:
                stream_started_at = monotonic()
                while not self._stop.is_set():
                    samples, _overflowed = stream.read(block_size)
                    for segment in self.segmenter.accept(samples):
                        self._queue_segment(segment, stream_started_at)
        except Exception as exc:
            self._events.put(exc)

    def _queue_segment(self, segment: SpeechSegment, stream_started_at: float) -> None:
        waveform = np.asarray(segment.samples, dtype=np.float32).reshape(-1).copy()
        queued = _QueuedSpeechSegment(
            waveform,
            stream_started_at + segment.start_sample / self.segmenter.sample_rate,
            stream_started_at + segment.end_sample / self.segmenter.sample_rate,
        )
        try:
            self._segments.put_nowait(queued)
        except queue.Full:
            self._segments.get_nowait()
            self._segments.put_nowait(queued)

    def _processing_loop(self) -> None:
        try:
            while not self._stop.is_set() or not self._segments.empty():
                try:
                    segment = self._segments.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._events.put(
                    self.process_audio_segment(
                        segment.samples,
                        started_at=segment.started_at,
                        ended_at=segment.ended_at,
                    )
                )
        except Exception as exc:
            self._events.put(exc)

    def process_audio_segment(
        self,
        segment: np.ndarray,
        *,
        started_at: float | None = None,
        ended_at: float | None = None,
    ) -> _VoiceEvent:
        end = monotonic() if ended_at is None else ended_at
        waveform = np.asarray(segment, dtype=np.float32).reshape(-1)
        duration = len(waveform) / self.segmenter.sample_rate
        start = end - duration if started_at is None else started_at
        overlap = self.overlap_detector.analyze(
            waveform,
            self.segmenter.sample_rate,
        )
        if overlap.detected:
            if self.separator is None:
                return _VoiceEvent((), start, end, overlap)
            separated = self.separator.separate(waveform, self.segmenter.sample_rate)
            observations = tuple(
                self._observe_source(
                    source,
                    separated.sample_rate,
                    observed_at=end,
                    allow_persistent_update=False,
                )
                for source in separated.sources[:2]
                if _segment_quality(source, separated.sample_rate) > 0.0
            )
            return _VoiceEvent(observations, start, end, overlap)
        observation = self._observe_source(
            waveform,
            self.segmenter.sample_rate,
            observed_at=end,
            allow_persistent_update=True,
        )
        return _VoiceEvent((observation,), start, end, overlap)

    def _observe_source(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        observed_at: float,
        allow_persistent_update: bool,
    ) -> _VoiceObservation:
        voice = self.voice.process_audio(
            waveform,
            sample_rate,
            quality_score=_segment_quality(waveform, sample_rate),
            observed_at=observed_at,
            allow_persistent_update=allow_persistent_update,
        )
        transcript = (
            self.transcriber.transcribe_samples(waveform, sample_rate)
            if self.transcriber is not None
            else None
        )
        return _VoiceObservation(voice, transcript)

    def _latest_voice_event(self) -> _VoiceEvent | Exception | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def _maybe_persist_voice(
        self,
        result: ActiveSpeakerResult,
        voice: SpeakerTrackSnapshot,
    ) -> bool:
        if (
            not self.auto_voice_enrollment
            or voice.persistent
            or not result.accepted
            or result.person_id is None
        ):
            return False
        key = (voice.track_id, result.person_id)
        for existing in tuple(self._voice_face_agreement):
            if existing[0] == voice.track_id and existing != key:
                del self._voice_face_agreement[existing]
        count = self._voice_face_agreement.get(key, 0) + 1
        self._voice_face_agreement[key] = count
        if count < self.voice_enrollment_observations:
            return False
        try:
            self.voice.persist_track(
                voice.track_id,
                result.person_id,
                voice_consent=True,
                minimum_quality=0.35,
                minimum_samples=2,
            )
        except ValueError:
            return False
        del self._voice_face_agreement[key]
        return True

    @staticmethod
    def _draw(
        cv2: object,
        frame: np.ndarray,
        faces: tuple[TrackSnapshot, ...],
        active: tuple[tuple[ActiveSpeakerResult, Transcript | None], ...],
        pending_enrollment: TrackSnapshot | None = None,
        status_message: str = "",
        show_transcripts: bool = True,
    ) -> None:
        for face in faces:
            is_speaking = any(result.face_track_id == face.track_id for result, _ in active)
            color = (70, 220, 90) if is_speaking else (70, 170, 255)
            box = face.box
            cv2.rectangle(frame, (box.left, box.top), (box.right, box.bottom), color, 2)
            cv2.putText(
                frame,
                face.label,
                (box.left, max(20, box.top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        label = (
            " | ".join(
                f"{result.label}: {transcript.text}"
                if show_transcripts and transcript and transcript.text
                else result.label
                for result, transcript in active
            )
            or "Listening..."
        )
        reason = "; ".join(result.reason for result, _ in active) or "Q/Esc: quit"
        cv2.putText(frame, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, reason, (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if pending_enrollment is not None:
            cv2.putText(
                frame,
                f"Recognize {pending_enrollment.label} on future visits? Y / N",
                (16, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 230, 255),
                2,
            )
        if status_message:
            cv2.putText(
                frame,
                status_message,
                (16, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (90, 220, 255),
                2,
            )
