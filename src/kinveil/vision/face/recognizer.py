from __future__ import annotations

from collections import defaultdict

import numpy as np

from .backend import normalize
from .models import FaceMatch
from .store import FaceStore


class FaceRecognizer:
    def __init__(self, store: FaceStore, threshold: float = 0.48, margin: float = 0.06) -> None:
        self.store = store
        self.threshold = threshold
        self.margin = margin

    def match(self, embedding: np.ndarray) -> FaceMatch:
        query = normalize(embedding)
        grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        for person_id, name, vector in self.store.embeddings():
            if vector.shape == query.shape:
                grouped[(person_id, name)].append(vector)
        if not grouped:
            return FaceMatch(None, None, 0.0, False, "no enrolled identities")

        scores = sorted(
            (
                (float(np.max(np.vstack(vectors) @ query)), person_id, name)
                for (person_id, name), vectors in grouped.items()
            ),
            reverse=True,
        )
        best_score, person_id, name = scores[0]
        if best_score < self.threshold:
            return FaceMatch(None, None, best_score, False, "below similarity threshold")
        if len(scores) > 1 and best_score - scores[1][0] < self.margin:
            return FaceMatch(None, None, best_score, False, "ambiguous identity")
        return FaceMatch(person_id, name, best_score, True, "matched")
