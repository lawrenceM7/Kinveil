from __future__ import annotations

from collections import defaultdict

import numpy as np

from .backend import normalize_voice_embedding
from .models import VoiceMatch
from .store import VoiceStore


class VoiceRecognizer:
    def __init__(self, store: VoiceStore, threshold: float = 0.65, margin: float = 0.08) -> None:
        self.store = store
        self.threshold = threshold
        self.margin = margin

    def match(self, embedding: np.ndarray) -> VoiceMatch:
        query = normalize_voice_embedding(embedding)
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for person_id, vector in self.store.embeddings():
            if vector.shape == query.shape:
                grouped[person_id].append(vector)
        if not grouped:
            return VoiceMatch(None, 0.0, False, "no enrolled speakers")
        scores = sorted(
            (
                (float(np.max(np.vstack(vectors) @ query)), person_id)
                for person_id, vectors in grouped.items()
            ),
            reverse=True,
        )
        best_score, person_id = scores[0]
        if best_score < self.threshold:
            return VoiceMatch(None, best_score, False, "below similarity threshold")
        if len(scores) > 1 and best_score - scores[1][0] < self.margin:
            return VoiceMatch(None, best_score, False, "ambiguous speaker")
        return VoiceMatch(person_id, best_score, True, "matched")
