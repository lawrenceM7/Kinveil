from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .backend import normalize
from .models import Person


@dataclass(frozen=True)
class _StoredTemplate:
    id: int
    pose_label: str
    embedding: np.ndarray
    quality_score: float
    created_at: str


class FaceStore:
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
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT,
                    local_alias TEXT,
                    created_at TEXT NOT NULL,
                    consented_at TEXT NOT NULL,
                    biometric_consented_at TEXT
                );
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    dimension INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    pose_label TEXT NOT NULL DEFAULT 'unknown',
                    quality_score REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_face_embeddings_person
                    ON face_embeddings(person_id);
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(people)").fetchall()
            }
            if "display_name" not in columns:
                connection.execute("ALTER TABLE people ADD COLUMN display_name TEXT")
                connection.execute("UPDATE people SET display_name = name")
            if "local_alias" not in columns:
                connection.execute("ALTER TABLE people ADD COLUMN local_alias TEXT")
                connection.execute(
                    """UPDATE people SET local_alias =
                       'Person ' || UPPER(SUBSTR(REPLACE(id, '-', ''), 1, 4))"""
                )
            if "biometric_consented_at" not in columns:
                connection.execute(
                    "ALTER TABLE people ADD COLUMN biometric_consented_at TEXT"
                )
                connection.execute(
                    "UPDATE people SET biometric_consented_at = consented_at"
                )
            embedding_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(face_embeddings)").fetchall()
            }
            if "pose_label" not in embedding_columns:
                connection.execute(
                    "ALTER TABLE face_embeddings ADD COLUMN pose_label TEXT "
                    "NOT NULL DEFAULT 'unknown'"
                )
            if "quality_score" not in embedding_columns:
                connection.execute(
                    "ALTER TABLE face_embeddings ADD COLUMN quality_score REAL "
                    "NOT NULL DEFAULT 0.0"
                )

    def create_person(self, name: str | None = None, consent: bool = False) -> Person:
        clean_name = name.strip() if name else None
        if name is not None and not clean_name:
            raise ValueError("person name cannot be empty")
        if not consent:
            raise ValueError("explicit consent is required for face enrollment")
        now = datetime.now(UTC)
        person_id = str(uuid.uuid4())
        local_alias = f"Person {person_id.replace('-', '')[:4].upper()}"
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO people(
                           id, name, display_name, local_alias, created_at,
                           consented_at, biometric_consented_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        person_id,
                        local_alias,
                        clean_name,
                        local_alias,
                        now.isoformat(),
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("could not create person identity") from exc
        return Person(person_id, clean_name, local_alias, now, 0, now)

    def create_anonymous_person(self, *, consent: bool) -> Person:
        return self.create_person(None, consent)

    def add_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        pose_label: str = "unknown",
        quality_score: float = 0.0,
    ) -> None:
        vector = normalize(embedding).astype("<f4", copy=False)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO face_embeddings(
                       person_id, dimension, embedding, pose_label, quality_score, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    person_id,
                    vector.size,
                    vector.tobytes(),
                    pose_label,
                    float(quality_score),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def maintain_embedding_pool(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        pose_label: str = "unknown",
        quality_score: float = 0.0,
        max_samples: int = 24,
        max_samples_per_pose: int = 8,
        duplicate_similarity: float = 0.995,
    ) -> bool:
        """Add a useful template and prune redundant samples to fixed limits."""
        if max_samples <= 0 or max_samples_per_pose <= 0:
            raise ValueError("template limits must be positive")
        vector = normalize(embedding).astype("<f4", copy=False)
        with self._connect() as connection:
            templates = self._templates(connection, person_id)
            same_pose = [item for item in templates if item.pose_label == pose_label]
            nearest = self._nearest(vector, same_pose)
            if nearest is not None and nearest[0] >= duplicate_similarity:
                existing = nearest[1]
                if quality_score <= existing.quality_score + 0.01:
                    return False
                connection.execute(
                    """UPDATE face_embeddings
                       SET dimension = ?, embedding = ?, quality_score = ?, created_at = ?
                       WHERE id = ?""",
                    (
                        vector.size,
                        vector.tobytes(),
                        float(quality_score),
                        datetime.now(UTC).isoformat(),
                        existing.id,
                    ),
                )
                return True

            connection.execute(
                """INSERT INTO face_embeddings(
                       person_id, dimension, embedding, pose_label, quality_score, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    person_id,
                    vector.size,
                    vector.tobytes(),
                    pose_label,
                    float(quality_score),
                    datetime.now(UTC).isoformat(),
                ),
            )
            templates = self._templates(connection, person_id)
            self._prune_pose(connection, templates, pose_label, max_samples_per_pose)
            templates = self._templates(connection, person_id)
            while len(templates) > max_samples:
                counts: dict[str, int] = {}
                for item in templates:
                    counts[item.pose_label] = counts.get(item.pose_label, 0) + 1
                largest_pose = max(counts, key=lambda key: counts[key])
                candidates = [item for item in templates if item.pose_label == largest_pose]
                connection.execute(
                    "DELETE FROM face_embeddings WHERE id = ?",
                    (self._least_useful(candidates).id,),
                )
                templates = self._templates(connection, person_id)
        return True

    @staticmethod
    def _templates(
        connection: sqlite3.Connection, person_id: str
    ) -> list[_StoredTemplate]:
        rows = connection.execute(
            """SELECT id, pose_label, dimension, embedding, quality_score, created_at
               FROM face_embeddings WHERE person_id = ? ORDER BY id""",
            (person_id,),
        ).fetchall()
        return [
            _StoredTemplate(
                row_id,
                pose_label,
                np.frombuffer(blob, dtype="<f4", count=dimension).copy(),
                quality_score,
                created_at,
            )
            for row_id, pose_label, dimension, blob, quality_score, created_at in rows
        ]

    @staticmethod
    def _nearest(
        vector: np.ndarray, templates: list[_StoredTemplate]
    ) -> tuple[float, _StoredTemplate] | None:
        compatible = [item for item in templates if item.embedding.shape == vector.shape]
        if not compatible:
            return None
        return max(
            ((float(item.embedding @ vector), item) for item in compatible),
            key=lambda result: result[0],
        )

    def _prune_pose(
        self,
        connection: sqlite3.Connection,
        templates: list[_StoredTemplate],
        pose_label: str,
        limit: int,
    ) -> None:
        candidates = [item for item in templates if item.pose_label == pose_label]
        while len(candidates) > limit:
            removed = self._least_useful(candidates)
            connection.execute("DELETE FROM face_embeddings WHERE id = ?", (removed.id,))
            candidates.remove(removed)

    @staticmethod
    def _least_useful(templates: list[_StoredTemplate]) -> _StoredTemplate:
        if len(templates) == 1:
            return templates[0]

        def utility(item: _StoredTemplate) -> tuple[float, float, str]:
            similarities = [
                float(item.embedding @ other.embedding)
                for other in templates
                if other.id != item.id and other.embedding.shape == item.embedding.shape
            ]
            redundancy = max(similarities, default=0.0)
            score = (1.0 - redundancy) * 0.7 + item.quality_score * 0.3
            return score, item.quality_score, item.created_at

        return min(templates, key=utility)

    def embeddings(self) -> tuple[tuple[str, str, np.ndarray], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT p.id, COALESCE(p.display_name, p.local_alias),
                          e.dimension, e.embedding
                   FROM face_embeddings e JOIN people p ON p.id = e.person_id
                   WHERE p.biometric_consented_at IS NOT NULL"""
            ).fetchall()
        return tuple(
            (person_id, name, np.frombuffer(blob, dtype="<f4", count=dimension).copy())
            for person_id, name, dimension, blob in rows
        )

    def pose_counts(self, person_id: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT pose_label, COUNT(*) FROM face_embeddings
                   WHERE person_id = ? GROUP BY pose_label""",
                (person_id,),
            ).fetchall()
        return {pose_label: count for pose_label, count in rows}

    def list_people(self) -> tuple[Person, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT p.id, p.display_name, p.local_alias, p.created_at,
                          COUNT(e.id), p.biometric_consented_at
                   FROM people p LEFT JOIN face_embeddings e ON e.person_id = p.id
                   GROUP BY p.id
                   ORDER BY COALESCE(p.display_name, p.local_alias) COLLATE NOCASE"""
            ).fetchall()
        return tuple(
            Person(
                person_id,
                display_name,
                local_alias,
                datetime.fromisoformat(created_at),
                sample_count,
                datetime.fromisoformat(consented_at) if consented_at else None,
            )
            for (
                person_id,
                display_name,
                local_alias,
                created_at,
                sample_count,
                consented_at,
            ) in rows
        )

    def get_person(self, person_id: str) -> Person:
        for person in self.list_people():
            if person.id == person_id:
                return person
        raise KeyError(f"unknown person: {person_id}")

    def delete_person(self, person_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM people WHERE id = ?", (person_id,))
        return cursor.rowcount > 0

    def set_display_name(self, person_id: str, display_name: str) -> Person:
        clean_name = display_name.strip()
        if not clean_name:
            raise ValueError("display name cannot be empty")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE people SET display_name = ? WHERE id = ?",
                (clean_name, person_id),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"unknown person: {person_id}")
        return self.get_person(person_id)

    def revoke_biometric_consent(self, person_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE people SET biometric_consented_at = NULL WHERE id = ?",
                (person_id,),
            )
            connection.execute(
                "DELETE FROM face_embeddings WHERE person_id = ?",
                (person_id,),
            )
        return cursor.rowcount > 0
