from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class OverlapDetection:
    detected: bool
    overlap_fraction: float
    overlap_duration: float
    max_speakers: int


class OverlapDetector(Protocol):
    def analyze(self, samples: np.ndarray, sample_rate: int) -> OverlapDetection: ...


class PyannoteOverlapDetector:
    """Detect two-speaker acoustic overlap with Pyannote Segmentation 3.0."""

    def __init__(
        self,
        model: Path,
        *,
        min_overlap_seconds: float = 0.15,
        min_overlap_fraction: float = 0.05,
        num_threads: int = 1,
    ) -> None:
        if not model.is_file():
            raise FileNotFoundError(f"overlap detection model not found: {model}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "overlap detection requires onnxruntime; install Kinveil's voice extra"
            ) from exc
        options = ort.SessionOptions()
        options.inter_op_num_threads = num_threads
        options.intra_op_num_threads = num_threads
        self._session = ort.InferenceSession(
            str(model),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        metadata = self._session.get_modelmeta().custom_metadata_map
        self.sample_rate = int(metadata["sample_rate"])
        self.frame_shift = int(metadata["receptive_field_shift"])
        self.num_classes = int(metadata["num_classes"])
        self.num_speakers = int(metadata["num_speakers"])
        self.powerset_max_classes = int(metadata["powerset_max_classes"])
        if self.num_classes != 7 or self.num_speakers != 3 or self.powerset_max_classes != 2:
            raise ValueError("unsupported Pyannote powerset model metadata")
        self.min_overlap_seconds = min_overlap_seconds
        self.min_overlap_fraction = min_overlap_fraction

    def analyze(self, samples: np.ndarray, sample_rate: int) -> OverlapDetection:
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"overlap model expects {self.sample_rate} Hz audio, got {sample_rate} Hz"
            )
        waveform = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if waveform.size < self.sample_rate // 2:
            return OverlapDetection(False, 0.0, 0.0, 0)
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        (logits,) = self._session.run(
            [output_name],
            {input_name: waveform.reshape(1, 1, -1)},
        )
        return self.classify_logits(
            logits[0],
            frame_seconds=self.frame_shift / self.sample_rate,
            min_overlap_seconds=self.min_overlap_seconds,
            min_overlap_fraction=self.min_overlap_fraction,
        )

    @staticmethod
    def classify_logits(
        logits: np.ndarray,
        *,
        frame_seconds: float,
        min_overlap_seconds: float = 0.15,
        min_overlap_fraction: float = 0.05,
    ) -> OverlapDetection:
        values = np.asarray(logits)
        if values.ndim != 2 or values.shape[1] != 7:
            raise ValueError("expected Pyannote logits with shape (frames, 7)")
        classes = np.argmax(values, axis=1)
        speech = classes != 0
        overlap = classes >= 4
        speech_frames = int(np.count_nonzero(speech))
        overlap_frames = int(np.count_nonzero(overlap))
        fraction = overlap_frames / max(1, speech_frames)
        duration = overlap_frames * frame_seconds
        detected = (
            duration >= min_overlap_seconds and fraction >= min_overlap_fraction
        )
        max_speakers = 2 if overlap_frames else (1 if speech_frames else 0)
        return OverlapDetection(detected, fraction, duration, max_speakers)
