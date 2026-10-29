from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .engine import VoiceIdentityEngine
from .models import SpeakerTrackSnapshot
from .sherpa_backend import _import_sherpa_onnx


@dataclass(frozen=True)
class SpeechSegment:
    samples: np.ndarray
    start_sample: int
    end_sample: int


class VoiceSegmenter(Protocol):
    sample_rate: int

    def accept(self, samples: np.ndarray) -> tuple[SpeechSegment, ...]: ...

    @property
    def speech_detected(self) -> bool: ...


class AdaptivePauseSegmenter:
    """Merge nearby VAD chunks using a pause budget that follows phrase length."""

    def __init__(
        self,
        segmenter: VoiceSegmenter,
        *,
        min_pause_seconds: float = 0.4,
        max_pause_seconds: float = 1.0,
        max_segment_seconds: float = 20.0,
        history_seconds: float = 30.0,
    ) -> None:
        if not 0 <= min_pause_seconds <= max_pause_seconds:
            raise ValueError("pause range must be non-negative and ordered")
        self.segmenter = segmenter
        self.sample_rate = segmenter.sample_rate
        self.min_pause_seconds = min_pause_seconds
        self.max_pause_seconds = max_pause_seconds
        self.max_segment_samples = round(max_segment_seconds * self.sample_rate)
        self._history_capacity = round(history_seconds * self.sample_rate)
        self._history = np.empty(0, dtype=np.float32)
        self._history_start = 0
        self._accepted_samples = 0
        self._pending: SpeechSegment | None = None

    def accept(self, samples: np.ndarray) -> tuple[SpeechSegment, ...]:
        waveform = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        self._append_history(waveform)
        ready: list[SpeechSegment] = []
        for segment in self.segmenter.accept(waveform):
            if self._pending is None:
                self._pending = segment
                continue
            gap = max(0, segment.start_sample - self._pending.end_sample)
            combined_size = segment.end_sample - self._pending.start_sample
            pause_samples = round(self._pause_seconds(self._pending) * self.sample_rate)
            if gap <= pause_samples and combined_size <= self.max_segment_samples:
                self._pending = self._combined(self._pending, segment)
            else:
                ready.append(self._pending)
                self._pending = segment
        if self._pending is not None:
            pause_samples = round(self._pause_seconds(self._pending) * self.sample_rate)
            if self._accepted_samples - self._pending.end_sample >= pause_samples:
                ready.append(self._pending)
                self._pending = None
        return tuple(ready)

    @property
    def speech_detected(self) -> bool:
        return bool(getattr(self.segmenter, "speech_detected", False))

    def _pause_seconds(self, segment: SpeechSegment) -> float:
        duration = (segment.end_sample - segment.start_sample) / self.sample_rate
        # The segment already carries 200 ms of trailing audio, so this budget
        # targets the remainder of an approximately 600-1200 ms pause. The
        # underlying VAD may already have waited for part of that interval.
        fraction = min(1.0, max(0.0, duration / 4.0))
        return self.max_pause_seconds - fraction * (
            self.max_pause_seconds - self.min_pause_seconds
        )

    def _combined(self, first: SpeechSegment, second: SpeechSegment) -> SpeechSegment:
        samples = self._history_slice(first.start_sample, second.end_sample)
        if samples is None:
            samples = np.concatenate((first.samples, second.samples))
        return SpeechSegment(samples, first.start_sample, second.end_sample)

    def _append_history(self, waveform: np.ndarray) -> None:
        if waveform.size:
            self._history = np.concatenate((self._history, waveform))
            self._accepted_samples += waveform.size
        overflow = self._history.size - self._history_capacity
        if overflow > 0:
            self._history = self._history[overflow:]
            self._history_start += overflow

    def _history_slice(self, start_sample: int, end_sample: int) -> np.ndarray | None:
        relative_start = start_sample - self._history_start
        relative_end = end_sample - self._history_start
        if relative_start < 0 or relative_end > self._history.size:
            return None
        return self._history[relative_start:relative_end].copy()


class SherpaVadSegmenter:
    """Turn streaming 16 kHz microphone samples into speech-only segments."""

    def __init__(
        self,
        model: Path,
        *,
        sample_rate: int = 16_000,
        buffer_seconds: float = 30.0,
        min_silence_seconds: float = 0.8,
        min_speech_seconds: float = 0.25,
        max_speech_seconds: float = 20.0,
        pre_roll_seconds: float = 0.3,
        post_roll_seconds: float = 0.2,
    ) -> None:
        if not model.is_file():
            raise FileNotFoundError(f"VAD model not found: {model}")
        sherpa_onnx = _import_sherpa_onnx()
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(model)
        config.silero_vad.min_silence_duration = min_silence_seconds
        config.silero_vad.min_speech_duration = min_speech_seconds
        config.silero_vad.max_speech_duration = max_speech_seconds
        config.sample_rate = sample_rate
        self.sample_rate = sample_rate
        self.pre_roll_samples = round(pre_roll_seconds * sample_rate)
        self.post_roll_samples = round(post_roll_seconds * sample_rate)
        self._history_capacity = round(buffer_seconds * sample_rate)
        self._history = np.empty(0, dtype=np.float32)
        self._history_start = 0
        self._accepted_samples = 0
        self._vad = sherpa_onnx.VoiceActivityDetector(
            config,
            buffer_size_in_seconds=buffer_seconds,
        )

    def accept(self, samples: np.ndarray) -> tuple[SpeechSegment, ...]:
        waveform = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        self._append_history(waveform)
        self._vad.accept_waveform(waveform)
        segments: list[SpeechSegment] = []
        while not self._vad.empty():
            front = self._vad.front
            captured = np.asarray(front.samples, dtype=np.float32).copy()
            start_sample = int(front.start)
            end_sample = start_sample + captured.size
            padded_start = max(0, start_sample - self.pre_roll_samples)
            padded_end = min(self._accepted_samples, end_sample + self.post_roll_samples)
            padded = self._history_slice(padded_start, padded_end)
            if padded is not None:
                captured = padded
                start_sample = padded_start
                end_sample = padded_end
            segments.append(
                SpeechSegment(
                    captured,
                    start_sample,
                    end_sample,
                )
            )
            self._vad.pop()
        return tuple(segments)

    @property
    def speech_detected(self) -> bool:
        detected = getattr(self._vad, "detected", False)
        return bool(detected() if callable(detected) else detected)

    def _append_history(self, waveform: np.ndarray) -> None:
        if waveform.size:
            self._history = np.concatenate((self._history, waveform))
            self._accepted_samples += waveform.size
        overflow = self._history.size - self._history_capacity
        if overflow > 0:
            self._history = self._history[overflow:]
            self._history_start += overflow

    def _history_slice(self, start_sample: int, end_sample: int) -> np.ndarray | None:
        relative_start = start_sample - self._history_start
        relative_end = end_sample - self._history_start
        if relative_start < 0 or relative_end > self._history.size:
            return None
        return self._history[relative_start:relative_end].copy()


class MicrophoneVoiceRunner:
    """Read a microphone continuously and resolve each VAD speech segment."""

    def __init__(
        self,
        engine: VoiceIdentityEngine,
        segmenter: VoiceSegmenter,
        *,
        device: int | str | None = None,
        block_seconds: float = 0.1,
        on_result: Callable[[SpeakerTrackSnapshot, float], None] | None = None,
    ) -> None:
        self.engine = engine
        self.segmenter = segmenter
        self.device = device
        self.block_seconds = block_seconds
        self.on_result = on_result

    def run(self) -> None:
        sounddevice = _import_sounddevice()
        block_size = int(self.segmenter.sample_rate * self.block_seconds)
        if block_size <= 0:
            raise ValueError("block_seconds is too small")
        with sounddevice.InputStream(
            channels=1,
            dtype="float32",
            samplerate=self.segmenter.sample_rate,
            device=self.device,
        ) as stream:
            print("Listening locally. Speak naturally; press Ctrl+C to stop.")
            while True:
                samples, overflowed = stream.read(block_size)
                if overflowed:
                    print("Warning: microphone input overflowed; one block was dropped.")
                for segment in self.segmenter.accept(samples):
                    self.process_segment(segment.samples)

    def process_segment(self, samples: np.ndarray) -> SpeakerTrackSnapshot:
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        quality = _segment_quality(waveform, self.segmenter.sample_rate)
        snapshot = self.engine.process_audio(
            waveform,
            self.segmenter.sample_rate,
            quality_score=quality,
        )
        if self.on_result is not None:
            self.on_result(snapshot, len(waveform) / self.segmenter.sample_rate)
        return snapshot


def _segment_quality(samples: np.ndarray, sample_rate: int) -> float:
    if samples.size == 0:
        return 0.0
    duration = samples.size / sample_rate
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    clipping = float(np.mean(np.abs(samples) >= 0.99))
    duration_score = min(1.0, duration / 3.0)
    level_score = min(1.0, rms / 0.08)
    clipping_penalty = min(1.0, clipping * 20.0)
    return max(0.0, min(1.0, 0.55 * duration_score + 0.45 * level_score - clipping_penalty))


def _import_sounddevice():
    try:
        import sounddevice
    except ImportError as exc:
        raise RuntimeError(
            "microphone support is not installed; run: "
            "python -m pip install -e '.[voice]'"
        ) from exc
    return sounddevice
