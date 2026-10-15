from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .backend import normalize_voice_embedding


class _SherpaStream(Protocol):
    def accept_waveform(self, *, sample_rate: int, waveform: np.ndarray) -> None: ...

    def input_finished(self) -> None: ...


class SherpaOnnxVoiceBackend:
    """Local speaker-embedding adapter backed by sherpa-onnx."""

    def __init__(
        self,
        model: Path,
        *,
        num_threads: int = 2,
        provider: str = "cpu",
        debug: bool = False,
    ) -> None:
        if not model.is_file():
            raise FileNotFoundError(f"speaker embedding model not found: {model}")
        sherpa_onnx = _import_sherpa_onnx()
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model),
            num_threads=num_threads,
            provider=provider,
            debug=debug,
        )
        if not config.validate():
            raise ValueError(f"invalid sherpa-onnx speaker model configuration: {model}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        waveform = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        stream: _SherpaStream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=waveform)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            raise ValueError("speech segment is too short for a speaker embedding")
        embedding = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return normalize_voice_embedding(embedding)


def _import_sherpa_onnx():
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "live voice support is not installed; run: "
            "python -m pip install -e '.[voice]'"
        ) from exc
    return sherpa_onnx
