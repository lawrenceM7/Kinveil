from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .backend import normalize_voice_embedding


class VoiceStore:
    """Persistent consent and bounded voice embeddings; raw audio is never stored."""

    def __init__(self, database: Path) -> None:
        self.database = database
        database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database))
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS voice_consents (
                    person_id TEXT PRIMARY KEY REFERENCES people(id) ON DELETE CASCADE,
                    consented_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS voice_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    dimension INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    quality_score REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_voice_embeddings_person
                    ON voice_embeddings(person_id);
                """
            )

    def grant_consent(self, person_id: str, *, consent: bool) -> None:
        if not consent:
            raise ValueError("explicit voice biometric consent is required")
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO voice_consents(person_id, consented_at)
                       VALUES (?, ?)
                       ON CONFLICT(person_id) DO UPDATE SET consented_at = excluded.consented_at""",
                    (person_id, datetime.now(UTC).isoformat()),
                )
        except sqlite3.IntegrityError as exc:
            raise KeyError(f"unknown person: {person_id}") from exc

    def has_consent(self, person_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM voice_consents WHERE person_id = ?", (person_id,)
            ).fetchone()
        return row is not None

    def maintain_embedding_pool(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        quality_score: float,
        max_samples: int = 24,
        duplicate_similarity: float = 0.995,
    ) -> bool:
        if not self.has_consent(person_id):
            raise PermissionError("voice biometric consent is required")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        vector = normalize_voice_embedding(embedding).astype("<f4", copy=False)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, dimension, embedding, quality_score
                   FROM voice_embeddings WHERE person_id = ? ORDER BY id""",
                (person_id,),
            ).fetchall()
            compatible = [
                (row_id, np.frombuffer(blob, dtype="<f4", count=dimension), quality)
                for row_id, dimension, blob, quality in rows
                if dimension == vector.size
            ]
            nearest = max(
                (
                    (float(known @ vector), row_id, quality)
                    for row_id, known, quality in compatible
                ),
                default=None,
            )
            if nearest is not None and nearest[0] >= duplicate_similarity:
                _, row_id, existing_quality = nearest
                if quality_score <= existing_quality + 0.01:
                    return False
                connection.execute(
                    """UPDATE voice_embeddings
                       SET embedding = ?, quality_score = ?, created_at = ? WHERE id = ?""",
                    (
                        vector.tobytes(),
                        float(quality_score),
                        datetime.now(UTC).isoformat(),
                        row_id,
                    ),
                )
                return True

            connection.execute(
                """INSERT INTO voice_embeddings(
                       person_id, dimension, embedding, quality_score, created_at
                   ) VALUES (?, ?, ?, ?, ?)""",
                (
                    person_id,
                    vector.size,
                    vector.tobytes(),
                    float(quality_score),
                    datetime.now(UTC).isoformat(),
                ),
            )
            count = connection.execute(
                "SELECT COUNT(*) FROM voice_embeddings WHERE person_id = ?", (person_id,)
            ).fetchone()[0]
            if count > max_samples:
                connection.execute(
                    """DELETE FROM voice_embeddings WHERE id = (
                           SELECT id FROM voice_embeddings WHERE person_id = ?
                           ORDER BY quality_score ASC, created_at ASC LIMIT 1
                       )""",
                    (person_id,),
                )
        return True

    def embeddings(self) -> tuple[tuple[str, np.ndarray], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT e.person_id, e.dimension, e.embedding
                   FROM voice_embeddings e
                   JOIN voice_consents c ON c.person_id = e.person_id"""
            ).fetchall()
        return tuple(
            (person_id, np.frombuffer(blob, dtype="<f4", count=dimension).copy())
            for person_id, dimension, blob in rows
        )

    def sample_count(self, person_id: str) -> int:
        with self._connect() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM voice_embeddings WHERE person_id = ?", (person_id,)
            ).fetchone()[0]

    def person_label(self, person_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT COALESCE(display_name, local_alias)
                   FROM people WHERE id = ?""",
                (person_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown person: {person_id}")
        return str(row[0])

    def revoke_consent(self, person_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM voice_consents WHERE person_id = ?", (person_id,)
            )
            connection.execute(
                "DELETE FROM voice_embeddings WHERE person_id = ?", (person_id,)
            )
        return cursor.rowcount > 0
