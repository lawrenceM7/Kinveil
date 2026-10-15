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


class SpeechToText(Protocol):
    def transcribe(self, audio: Path) -> Transcript: ...


class AudioTranscriber(Protocol):
    def transcribe_samples(self, samples: np.ndarray, sample_rate: int) -> Transcript: ...


class SherpaOnnxWhisperTranscriber:
    """Local, non-streaming Whisper transcription through sherpa-onnx."""

    def __init__(
        self,
        model_directory: Path,
        *,
        language: str = "",
        num_threads: int = 2,
    ) -> None:
        encoder = _find_model_file(model_directory, "encoder.int8.onnx")
        decoder = _find_model_file(model_directory, "decoder.int8.onnx")
        tokens = _find_model_file(model_directory, "tokens.txt")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("speech transcription requires Kinveil's voice extra") from exc
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(encoder),
            decoder=str(decoder),
            tokens=str(tokens),
            language=language,
            task="transcribe",
            num_threads=num_threads,
        )

    def transcribe_samples(self, samples: np.ndarray, sample_rate: int) -> Transcript:
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if waveform.size == 0 or not np.all(np.isfinite(waveform)):
            raise ValueError("audio samples must be finite and non-empty")
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, waveform)
        self._recognizer.decode_stream(stream)
        result = stream.result
        language = getattr(result, "lang", None) or getattr(result, "language", None)
        return Transcript(result.text.strip(), language=language)


def _find_model_file(directory: Path, suffix: str) -> Path:
    matches = tuple(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one *{suffix} file in Whisper model directory: {directory}"
        )
    return matches[0]
