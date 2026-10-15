from __future__ import annotations

from typing import Protocol

import numpy as np


class VoiceEmbeddingBackend(Protocol):
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...


def normalize_voice_embedding(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("voice embedding must be finite and non-zero")
    return vector / norm
