from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None
    confidence: float | None = None
    audio_duration: float | None = None
    audio_rms: float | None = None


class SpeechToText(Protocol):
    def transcribe(self, audio: Path) -> Transcript: ...


class AudioTranscriber(Protocol):
    def transcribe_samples(self, samples: np.ndarray, sample_rate: int) -> Transcript: ...


class SherpaOnnxWhisperTranscriber:
    """Session-aware offline Whisper transcription through sherpa-onnx.

    Adjacent chunks share a short audio tail. Repeated words introduced by that
    overlap are removed, and automatic language detection is locked only after
    two consistent, sufficiently long observations.
    """

    def __init__(
        self,
        model_directory: Path,
        *,
        language: str = "",
        num_threads: int = 2,
        overlap_seconds: float = 0.3,
        language_lock_observations: int = 2,
        language_lock_min_seconds: float = 1.0,
    ) -> None:
        self._encoder = _find_model_file(model_directory, "encoder.int8.onnx")
        self._decoder = _find_model_file(model_directory, "decoder.int8.onnx")
        self._tokens = _find_model_file(model_directory, "tokens.txt")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("speech transcription requires Kinveil's voice extra") from exc
        self._sherpa_onnx = sherpa_onnx
        self._num_threads = num_threads
        self._configured_language = language
        self._locked_language = language or None
        self._language_lock_observations = language_lock_observations
        self._language_lock_min_seconds = language_lock_min_seconds
        self._language_votes: dict[str, int] = {}
        self._overlap_seconds = overlap_seconds
        self._audio_tail = np.empty(0, dtype=np.float32)
        self._previous_text = ""
        self._recognizer = self._create_recognizer(language)

    def _create_recognizer(self, language: str):
        return self._sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(self._encoder),
            decoder=str(self._decoder),
            tokens=str(self._tokens),
            language=language,
            task="transcribe",
            num_threads=self._num_threads,
        )

    def transcribe_samples(self, samples: np.ndarray, sample_rate: int) -> Transcript:
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if waveform.size == 0 or not np.all(np.isfinite(waveform)):
            raise ValueError("audio samples must be finite and non-empty")
        original_duration = waveform.size / sample_rate
        contextual = np.concatenate((self._audio_tail, waveform))
        tail_size = max(0, round(self._overlap_seconds * sample_rate))
        self._audio_tail = waveform[-tail_size:].copy() if tail_size else waveform[:0]
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, contextual)
        self._recognizer.decode_stream(stream)
        result = stream.result
        language = getattr(result, "lang", None) or getattr(result, "language", None)
        self._observe_language(language, original_duration)
        text = _remove_repeated_context(self._previous_text, result.text.strip())
        if text:
            self._previous_text = result.text.strip()
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        return Transcript(
            text,
            language=self._locked_language or language,
            audio_duration=original_duration,
            audio_rms=rms,
        )

    def _observe_language(self, language: str | None, duration: float) -> None:
        if self._locked_language or not language or duration < self._language_lock_min_seconds:
            return
        normalized = language.strip().lower()
        if not normalized:
            return
        self._language_votes[normalized] = self._language_votes.get(normalized, 0) + 1
        if self._language_votes[normalized] >= self._language_lock_observations:
            self._locked_language = normalized
            self._recognizer = self._create_recognizer(normalized)


class SherpaOnnxParaformerTranscriber:
    """Chinese/English offline transcription using a Paraformer model."""

    def __init__(self, model_directory: Path, *, num_threads: int = 2) -> None:
        model = _find_model_file(model_directory, "model.int8.onnx")
        tokens = _find_model_file(model_directory, "tokens.txt")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("speech transcription requires Kinveil's voice extra") from exc
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(model),
            tokens=str(tokens),
            num_threads=num_threads,
        )

    def transcribe_samples(self, samples: np.ndarray, sample_rate: int) -> Transcript:
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if waveform.size == 0 or not np.all(np.isfinite(waveform)):
            raise ValueError("audio samples must be finite and non-empty")
        duration = waveform.size / sample_rate
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        normalized = _normalize_asr_audio(waveform, rms)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, normalized)
        self._recognizer.decode_stream(stream)
        return Transcript(
            stream.result.text.strip(),
            language="zh-en",
            audio_duration=duration,
            audio_rms=rms,
        )


def _normalize_asr_audio(
    waveform: np.ndarray,
    rms: float,
    *,
    target_rms: float = 0.05,
    max_gain: float = 3.0,
) -> np.ndarray:
    """Apply bounded gain without clipping quiet microphone input."""
    if rms <= 0.0 or rms >= target_rms:
        return waveform
    peak = float(np.max(np.abs(waveform)))
    gain = min(max_gain, target_rms / rms)
    if peak > 0.0:
        gain = min(gain, 0.95 / peak)
    return np.asarray(waveform * max(1.0, gain), dtype=np.float32)


def _remove_repeated_context(previous: str, current: str) -> str:
    """Remove a repeated prefix caused by overlapping adjacent audio chunks."""
    if not previous or not current:
        return current
    limit = min(len(previous), len(current), 32)
    for size in range(limit, 1, -1):
        if previous[-size:] == current[:size]:
            return current[size:].lstrip()
    return current


def _find_model_file(directory: Path, suffix: str) -> Path:
    matches = tuple(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one *{suffix} file in Whisper model directory: {directory}"
        )
    return matches[0]
