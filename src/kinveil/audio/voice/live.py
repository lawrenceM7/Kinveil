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
        self._vad = sherpa_onnx.VoiceActivityDetector(
            config,
            buffer_size_in_seconds=buffer_seconds,
        )

    def accept(self, samples: np.ndarray) -> tuple[SpeechSegment, ...]:
        waveform = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        self._vad.accept_waveform(waveform)
        segments: list[SpeechSegment] = []
        while not self._vad.empty():
            front = self._vad.front
            captured = np.asarray(front.samples, dtype=np.float32).copy()
            start_sample = int(front.start)
            segments.append(
                SpeechSegment(
                    captured,
                    start_sample,
                    start_sample + captured.size,
                )
            )
            self._vad.pop()
        return tuple(segments)


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
