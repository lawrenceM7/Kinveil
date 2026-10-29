from __future__ import annotations

import hashlib
import html
import json
import math
import re
import sqlite3
import unicodedata
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from .base import (
    MemoryCandidate,
    MemoryContext,
    MemoryEmbeddingBackend,
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    PendingMemoryReview,
    RetentionClass,
    Sensitivity,
    Visibility,
)


class SQLiteMemoryStore:
    """Local identity-scoped memories with consent and an on-disk review queue."""

    def __init__(
        self,
        database: Path,
        *,
        embedding_backend: MemoryEmbeddingBackend | None = None,
    ) -> None:
        self.database = database
        self.embedding_backend = embedding_backend
        database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database))
        connection.row_factory = sqlite3.Row
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
                CREATE TABLE IF NOT EXISTS memory_consents (
                    person_id TEXT PRIMARY KEY,
                    granted_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    sensitivity TEXT NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    retention TEXT NOT NULL DEFAULT 'standard',
                    source_turn_id TEXT,
                    topic_key TEXT,
                    supersedes_id TEXT REFERENCES memories(id),
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_observed_at TEXT,
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    reinforcement_count INTEGER NOT NULL DEFAULT 1,
                    expires_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_active_duplicate
                    ON memories(owner_id, content_hash)
                    WHERE status = 'active';
                CREATE INDEX IF NOT EXISTS idx_memories_owner_status
                    ON memories(owner_id, status);
                CREATE TABLE IF NOT EXISTS memory_audience (
                    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    person_id TEXT NOT NULL,
                    PRIMARY KEY(memory_id, person_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_audience_person
                    ON memory_audience(person_id);
                CREATE TABLE IF NOT EXISTS memory_reviews (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT,
                    candidate_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_reviews_owner
                    ON memory_reviews(owner_id, created_at);
                CREATE TABLE IF NOT EXISTS memory_edges (
                    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    source_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    relation TEXT NOT NULL,
                    PRIMARY KEY(memory_id, source_memory_id, relation)
                );
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    model_id TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY(memory_id, model_id)
                );
                CREATE TABLE IF NOT EXISTS memory_reflection_state (
                    owner_id TEXT PRIMARY KEY,
                    last_reflected_at TEXT NOT NULL
                );
                """
            )
            self._migrate_memories(connection)
            connection.execute(
                """CREATE INDEX IF NOT EXISTS idx_memories_owner_topic
                   ON memories(owner_id, topic_key, status)"""
            )

    @staticmethod
    def _migrate_memories(connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(memories)").fetchall()
        }
        additions = {
            "retention": "TEXT NOT NULL DEFAULT 'standard'",
            "topic_key": "TEXT",
            "last_observed_at": "TEXT",
            "last_accessed_at": "TEXT",
            "access_count": "INTEGER NOT NULL DEFAULT 0",
            "reinforcement_count": "INTEGER NOT NULL DEFAULT 1",
        }
        for name, declaration in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE memories ADD COLUMN {name} {declaration}")
        connection.execute(
            "UPDATE memories SET last_observed_at = updated_at WHERE last_observed_at IS NULL"
        )

    def grant_consent(self, person_id: str, *, consent: bool) -> None:
        clean_id = person_id.strip()
        if not clean_id:
            raise ValueError("person_id cannot be empty")
        if not consent:
            raise ValueError("explicit memory consent is required")
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO memory_consents(person_id, granted_at) VALUES (?, ?)
                   ON CONFLICT(person_id) DO UPDATE SET granted_at = excluded.granted_at""",
                (clean_id, datetime.now(UTC).isoformat()),
            )

    def has_consent(self, person_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM memory_consents WHERE person_id = ?", (person_id,)
            ).fetchone()
        return row is not None

    def revoke_consent(self, person_id: str, *, delete_memories: bool) -> bool:
        """Revoke future writes and explicitly retain or delete existing memories."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memory_consents WHERE person_id = ?", (person_id,)
            )
            if delete_memories:
                connection.execute("DELETE FROM memories WHERE owner_id = ?", (person_id,))
                connection.execute("DELETE FROM memory_reviews WHERE owner_id = ?", (person_id,))
        return cursor.rowcount > 0

    def add(self, record: MemoryRecord) -> str:
        if not self.has_consent(record.owner_id):
            raise PermissionError("memory storage consent is required")
        content = record.content.strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        if not 0 <= record.importance <= 1 or not 0 <= record.confidence <= 1:
            raise ValueError("memory importance and confidence must be between 0 and 1")
        audience = tuple(
            dict.fromkeys(item.strip() for item in record.audience_ids if item.strip())
        )
        if record.visibility is Visibility.PRIVATE and audience:
            raise ValueError("private memory cannot have an audience")
        if record.visibility is Visibility.SHARED and not audience:
            raise ValueError("shared memory requires an audience")

        normalized = _normalize(content)
        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        now = datetime.now(UTC)
        memory_id = record.id or str(uuid.uuid4())
        created_at = _as_utc(record.created_at).isoformat()
        updated_at = _as_utc(record.updated_at or now).isoformat()
        last_observed_at = _as_utc(record.last_observed_at or now).isoformat()
        selected_expiry = record.expires_at or _default_expiry(record.retention, now)
        expires_at = _as_utc(selected_expiry).isoformat() if selected_expiry else None

        with self._connect() as connection:
            duplicate = connection.execute(
                """SELECT id, importance, confidence, reinforcement_count, retention
                   FROM memories
                   WHERE owner_id = ? AND content_hash = ? AND status = 'active'""",
                (record.owner_id, content_hash),
            ).fetchone()
            should_replace_audience = duplicate is None
            if duplicate is not None:
                memory_id = str(duplicate["id"])
                duplicate_retention = RetentionClass(str(duplicate["retention"]))
                duplicate_expiry = _default_expiry(duplicate_retention, now)
                connection.execute(
                    """UPDATE memories SET importance = ?, confidence = ?, updated_at = ?,
                              last_observed_at = ?, reinforcement_count = ?,
                              expires_at = ?, metadata_json = ? WHERE id = ?""",
                    (
                        max(float(duplicate["importance"]), record.importance),
                        max(float(duplicate["confidence"]), record.confidence),
                        updated_at,
                        last_observed_at,
                        int(duplicate["reinforcement_count"]) + 1,
                        (
                            _as_utc(duplicate_expiry).isoformat()
                            if duplicate_expiry is not None
                            else None
                        ),
                        json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                        memory_id,
                    ),
                )
            else:
                if record.supersedes_id is not None:
                    previous = connection.execute(
                        "SELECT owner_id, status FROM memories WHERE id = ?",
                        (record.supersedes_id,),
                    ).fetchone()
                    if previous is None or previous["owner_id"] != record.owner_id:
                        raise ValueError("superseded memory must belong to the same owner")
                connection.execute(
                    """INSERT INTO memories(
                           id, owner_id, content, normalized_content, content_hash, kind,
                           visibility, sensitivity, importance, confidence, retention,
                           source_turn_id, topic_key, supersedes_id, status, metadata_json,
                           created_at, updated_at, last_observed_at, last_accessed_at,
                           access_count, reinforcement_count, expires_at
                       ) VALUES (
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                       )""",
                    (
                        memory_id,
                        record.owner_id,
                        content,
                        normalized,
                        content_hash,
                        record.kind.value,
                        record.visibility.value,
                        record.sensitivity.value,
                        record.importance,
                        record.confidence,
                        record.retention.value,
                        record.source_turn_id,
                        record.topic_key,
                        record.supersedes_id,
                        MemoryStatus.ACTIVE.value,
                        json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                        created_at,
                        updated_at,
                        last_observed_at,
                        (
                            _as_utc(record.last_accessed_at).isoformat()
                            if record.last_accessed_at
                            else None
                        ),
                        record.access_count,
                        record.reinforcement_count,
                        expires_at,
                    ),
                )
                if record.supersedes_id is not None:
                    connection.execute(
                        "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                        (MemoryStatus.SUPERSEDED.value, updated_at, record.supersedes_id),
                    )

            if should_replace_audience:
                connection.executemany(
                    "INSERT INTO memory_audience(memory_id, person_id) VALUES (?, ?)",
                    ((memory_id, person_id) for person_id in audience),
                )
                self._store_embedding(connection, memory_id, content)
            self._store_edges(connection, memory_id, record)
        return memory_id

    def _store_edges(
        self, connection: sqlite3.Connection, memory_id: str, record: MemoryRecord
    ) -> None:
        for source_id in dict.fromkeys(record.derived_from_ids):
            source = connection.execute(
                "SELECT owner_id FROM memories WHERE id = ?", (source_id,)
            ).fetchone()
            if source is None or source["owner_id"] != record.owner_id:
                raise ValueError("reflection evidence must belong to the same owner")
            connection.execute(
                """INSERT OR IGNORE INTO memory_edges(
                       memory_id, source_memory_id, relation
                   ) VALUES (?, ?, 'derived_from')""",
                (memory_id, source_id),
            )

    def _store_embedding(
        self, connection: sqlite3.Connection, memory_id: str, content: str
    ) -> None:
        if self.embedding_backend is None:
            return
        vector = _normalized_vector(self.embedding_backend.embed(content))
        connection.execute(
            """INSERT OR REPLACE INTO memory_embeddings(
                   memory_id, model_id, dimension, embedding
               ) VALUES (?, ?, ?, ?)""",
            (
                memory_id,
                self.embedding_backend.model_id,
                int(vector.size),
                vector.astype("<f4", copy=False).tobytes(),
            ),
        )

    def find_active_by_topic(self, owner_id: str, topic_key: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT m.*, GROUP_CONCAT(a.person_id, char(31)) AS audience,
                          (SELECT GROUP_CONCAT(e.source_memory_id, char(31))
                           FROM memory_edges e WHERE e.memory_id = m.id) AS evidence
                   FROM memories m
                   LEFT JOIN memory_audience a ON a.memory_id = m.id
                   WHERE m.owner_id = ? AND m.topic_key = ? AND m.status = 'active'
                   GROUP BY m.id ORDER BY m.updated_at DESC LIMIT 1""",
                (owner_id, topic_key),
            ).fetchone()
        if row is None:
            return None
        audience = tuple(filter(None, (row["audience"] or "").split(chr(31))))
        return self._row_to_record(row, audience)

    def recall(self, owner_id: str, query: str, limit: int = 10) -> Sequence[MemoryRecord]:
        return self.search(owner_id, query, limit=limit, owner_id=owner_id)

    def search(
        self,
        requester_id: str,
        query: str,
        *,
        limit: int = 10,
        owner_id: str | None = None,
    ) -> Sequence[MemoryRecord]:
        if limit <= 0:
            return ()
        current_time = datetime.now(UTC)
        now = current_time.isoformat()
        clauses = ["m.status = 'active'", "(m.expires_at IS NULL OR m.expires_at > ?)"]
        parameters: list[object] = [now]
        if owner_id is not None:
            clauses.append("m.owner_id = ?")
            parameters.append(owner_id)
        sql = f"""SELECT m.*,
                    GROUP_CONCAT(a.person_id, char(31)) AS audience,
                    (SELECT GROUP_CONCAT(e.source_memory_id, char(31))
                     FROM memory_edges e WHERE e.memory_id = m.id) AS evidence
                  FROM memories m
                  LEFT JOIN memory_audience a ON a.memory_id = m.id
                  WHERE {' AND '.join(clauses)}
                  GROUP BY m.id"""
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
            stored_embeddings = self._load_embeddings(connection, rows)

        query_vector = None
        if self.embedding_backend is not None and query.strip():
            query_vector = _normalized_vector(self.embedding_backend.embed(query))

        accessible: list[tuple[float, MemoryRecord]] = []
        query_terms = _search_terms(query)
        for row in rows:
            audience = tuple(filter(None, (row["audience"] or "").split(chr(31))))
            visibility = Visibility(row["visibility"])
            if visibility is Visibility.PRIVATE and row["owner_id"] != requester_id:
                continue
            if (
                visibility is Visibility.SHARED
                and row["owner_id"] != requester_id
                and requester_id not in audience
            ):
                continue
            record = self._row_to_record(row, audience)
            semantic = _cosine(query_vector, stored_embeddings.get(record.id or ""))
            accessible.append(
                (_relevance(record, query_terms, semantic, current_time), record)
            )
        accessible.sort(
            key=lambda item: (item[0], item[1].updated_at or item[1].created_at),
            reverse=True,
        )
        return tuple(record for _, record in accessible[:limit])

    def _load_embeddings(
        self, connection: sqlite3.Connection, rows: Sequence[sqlite3.Row]
    ) -> dict[str, np.ndarray]:
        if self.embedding_backend is None or not rows:
            return {}
        ids = [str(row["id"]) for row in rows]
        placeholders = ",".join("?" for _ in ids)
        stored = connection.execute(
            f"""SELECT memory_id, dimension, embedding FROM memory_embeddings
                WHERE model_id = ? AND memory_id IN ({placeholders})""",
            [self.embedding_backend.model_id, *ids],
        ).fetchall()
        return {
            str(row["memory_id"]): np.frombuffer(
                row["embedding"], dtype="<f4", count=int(row["dimension"])
            )
            for row in stored
        }

    def build_context(
        self,
        requester_id: str,
        query: str,
        *,
        max_tokens: int = 800,
        candidate_limit: int = 50,
        owner_id: str | None = None,
    ) -> MemoryContext:
        """Pack ranked memories into a hard, conservative prompt token budget."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        candidates = self.search(
            requester_id,
            query,
            limit=max(candidate_limit, 1),
            owner_id=owner_id,
        )
        selected: list[MemoryRecord] = []
        lines: list[str] = []
        opening = "<memory_context data-only=\"true\">\n"
        closing = "\n</memory_context>"
        wrapper_cost = estimate_tokens(opening + closing)
        used = wrapper_cost
        for record in candidates:
            safe_content = html.escape(record.content, quote=False)
            line = f"- [{record.kind.value}] {safe_content}"
            cost = estimate_tokens(line + "\n")
            if used + cost > max_tokens:
                continue
            selected.append(record)
            lines.append(line)
            used += cost
        self.mark_accessed(tuple(item.id for item in selected if item.id))
        text = opening + "\n".join(lines) + closing if lines else ""
        return MemoryContext(
            records=tuple(selected),
            text=text,
            estimated_tokens=used if lines else 0,
            max_tokens=max_tokens,
            truncated=len(selected) < len(candidates),
        )

    def mark_accessed(self, memory_ids: tuple[str, ...]) -> None:
        if not memory_ids:
            return
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.executemany(
                """UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1
                   WHERE id = ? AND status = 'active'""",
                ((now, memory_id) for memory_id in memory_ids),
            )

    def maintain(
        self,
        *,
        now: datetime | None = None,
        review_max_age: timedelta = timedelta(days=7),
        purge_expired_after: timedelta = timedelta(days=30),
    ) -> dict[str, int]:
        """Expire stale memories, prune old reviews, and later purge expired payloads."""
        selected_now = _as_utc(now or datetime.now(UTC))
        review_cutoff = (selected_now - review_max_age).isoformat()
        purge_cutoff = (selected_now - purge_expired_after).isoformat()
        with self._connect() as connection:
            expired = connection.execute(
                """UPDATE memories SET status = 'expired', updated_at = ?
                   WHERE status IN ('active', 'archived')
                         AND expires_at IS NOT NULL AND expires_at <= ?""",
                (selected_now.isoformat(), selected_now.isoformat()),
            ).rowcount
            reviews = connection.execute(
                "DELETE FROM memory_reviews WHERE created_at <= ?", (review_cutoff,)
            ).rowcount
            purged = connection.execute(
                """DELETE FROM memories
                   WHERE status = 'expired' AND updated_at <= ?""",
                (purge_cutoff,),
            ).rowcount
        return {"expired": expired, "reviews": reviews, "purged": purged}

    def close_session_memories(self, owner_id: str | None = None) -> int:
        clauses = ["retention = 'session'", "status = 'active'"]
        parameters: list[object] = [datetime.now(UTC).isoformat()]
        if owner_id is not None:
            clauses.append("owner_id = ?")
            parameters.append(owner_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""UPDATE memories SET status = 'expired', updated_at = ?
                    WHERE {' AND '.join(clauses)}""",
                parameters,
            )
        return cursor.rowcount

    def archive_consolidated(self, memory_ids: tuple[str, ...]) -> int:
        if not memory_ids:
            return 0
        now = datetime.now(UTC).isoformat()
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""UPDATE memories SET status = 'archived', updated_at = ?
                    WHERE id IN ({placeholders}) AND status = 'active'
                      AND retention IN ('session', 'short_term')
                      AND kind IN ('observation', 'event')""",
                [now, *memory_ids],
            )
        return cursor.rowcount

    def reflection_inputs(
        self, owner_id: str, *, limit: int = 100
    ) -> tuple[MemoryRecord, ...]:
        with self._connect() as connection:
            state = connection.execute(
                "SELECT last_reflected_at FROM memory_reflection_state WHERE owner_id = ?",
                (owner_id,),
            ).fetchone()
            since = state["last_reflected_at"] if state is not None else "1970-01-01T00:00:00+00:00"
            rows = connection.execute(
                """SELECT m.*, GROUP_CONCAT(a.person_id, char(31)) AS audience,
                          (SELECT GROUP_CONCAT(e.source_memory_id, char(31))
                           FROM memory_edges e WHERE e.memory_id = m.id) AS evidence
                   FROM memories m
                   LEFT JOIN memory_audience a ON a.memory_id = m.id
                   WHERE m.owner_id = ? AND m.status = 'active'
                         AND m.kind != 'reflection' AND m.updated_at > ?
                   GROUP BY m.id ORDER BY m.updated_at DESC LIMIT ?""",
                (owner_id, since, limit),
            ).fetchall()
        return tuple(
            self._row_to_record(
                row,
                tuple(filter(None, (row["audience"] or "").split(chr(31)))),
            )
            for row in rows
        )

    def mark_reflected(self, owner_id: str, *, at: datetime | None = None) -> None:
        timestamp = _as_utc(at or datetime.now(UTC)).isoformat()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO memory_reflection_state(owner_id, last_reflected_at)
                   VALUES (?, ?)
                   ON CONFLICT(owner_id) DO UPDATE
                   SET last_reflected_at = excluded.last_reflected_at""",
                (owner_id, timestamp),
            )

    def queue_review(self, candidate: MemoryCandidate, reason: str) -> str:
        review_id = str(uuid.uuid4())
        payload = _candidate_payload(candidate)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO memory_reviews(
                       id, owner_id, candidate_json, reason, created_at
                   ) VALUES (?, ?, ?, ?, ?)""",
                (
                    review_id,
                    candidate.owner_id,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    reason,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return review_id

    def pending_reviews(self, owner_id: str | None = None) -> tuple[PendingMemoryReview, ...]:
        with self._connect() as connection:
            if owner_id is None:
                rows = connection.execute(
                    "SELECT * FROM memory_reviews ORDER BY created_at"
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM memory_reviews WHERE owner_id = ?
                       ORDER BY created_at""",
                    (owner_id,),
                ).fetchall()
        return tuple(
            PendingMemoryReview(
                id=str(row["id"]),
                candidate=_candidate_from_payload(json.loads(row["candidate_json"])),
                reason=str(row["reason"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        )

    def pop_review(self, review_id: str) -> PendingMemoryReview:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_reviews WHERE id = ?", (review_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown memory review: {review_id}")
            connection.execute("DELETE FROM memory_reviews WHERE id = ?", (review_id,))
        return PendingMemoryReview(
            id=str(row["id"]),
            candidate=_candidate_from_payload(json.loads(row["candidate_json"])),
            reason=str(row["reason"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def reject_review(self, review_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memory_reviews WHERE id = ?", (review_id,)
            )
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_record(row: sqlite3.Row, audience: tuple[str, ...]) -> MemoryRecord:
        return MemoryRecord(
            id=str(row["id"]),
            owner_id=str(row["owner_id"]),
            content=str(row["content"]),
            kind=MemoryKind(row["kind"]),
            visibility=Visibility(row["visibility"]),
            sensitivity=Sensitivity(row["sensitivity"]),
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            retention=RetentionClass(row["retention"]),
            audience_ids=audience,
            source_turn_id=row["source_turn_id"],
            supersedes_id=row["supersedes_id"],
            topic_key=row["topic_key"],
            derived_from_ids=tuple(
                filter(None, (row["evidence"] or "").split(chr(31)))
            ),
            status=MemoryStatus(row["status"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=(datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None),
            last_observed_at=(
                datetime.fromisoformat(row["last_observed_at"])
                if row["last_observed_at"]
                else None
            ),
            last_accessed_at=(
                datetime.fromisoformat(row["last_accessed_at"])
                if row["last_accessed_at"]
                else None
            ),
            access_count=int(row["access_count"]),
            reinforcement_count=int(row["reinforcement_count"]),
        )


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _search_terms(text: str) -> set[str]:
    normalized = _normalize(text)
    words = set(re.findall(r"[a-z0-9_]+", normalized))
    chinese = "".join(re.findall(r"[\u3400-\u9fff]", normalized))
    words.update(chinese)
    words.update(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
    return {item for item in words if item}


def _relevance(
    record: MemoryRecord,
    query_terms: set[str],
    semantic: float | None,
    now: datetime,
) -> float:
    if query_terms:
        memory_terms = _search_terms(record.content)
        lexical = len(query_terms & memory_terms) / len(query_terms)
    else:
        lexical = 0.0
    if semantic is None:
        relevance = lexical
    else:
        relevance = max(0.0, semantic) * 0.8 + lexical * 0.2
    observed = record.last_observed_at or record.updated_at or record.created_at
    age_days = max(0.0, (now - _as_utc(observed)).total_seconds() / 86400)
    half_life = _half_life_days(record.retention)
    recency = 1.0 if math.isinf(half_life) else 2 ** (-age_days / half_life)
    reinforcement = min(1.0, math.log2(record.reinforcement_count + 1) / 5)
    return (
        relevance * 0.45
        + record.importance * 0.2
        + recency * 0.15
        + record.confidence * 0.1
        + reinforcement * 0.1
    )


def estimate_tokens(text: str) -> int:
    """Tokenizer-free upper estimate using UTF-8 bytes as the conservative ceiling."""
    return max(1, len(text.encode("utf-8")))


def _half_life_days(retention: RetentionClass) -> float:
    return {
        RetentionClass.SESSION: 0.25,
        RetentionClass.SHORT_TERM: 7.0,
        RetentionClass.STANDARD: 45.0,
        RetentionClass.DURABLE: 365.0,
        RetentionClass.PINNED: math.inf,
    }[retention]


def _default_expiry(retention: RetentionClass, now: datetime) -> datetime | None:
    duration = {
        RetentionClass.SESSION: timedelta(days=1),
        RetentionClass.SHORT_TERM: timedelta(days=30),
        RetentionClass.STANDARD: timedelta(days=365),
        RetentionClass.DURABLE: timedelta(days=3650),
        RetentionClass.PINNED: None,
    }[retention]
    return now + duration if duration is not None else None


def _normalized_vector(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or norm <= 0:
        raise ValueError("memory embedding must be a non-zero vector")
    return vector / norm


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(left @ right)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _candidate_payload(candidate: MemoryCandidate) -> dict[str, object]:
    return {
        "content": candidate.content,
        "owner_id": candidate.owner_id,
        "identity_status": candidate.identity_status,
        "confidence": candidate.confidence,
        "importance": candidate.importance,
        "visibility": candidate.visibility.value,
        "kind": candidate.kind.value,
        "sensitivity": candidate.sensitivity.value,
        "retention": candidate.retention.value,
        "audience_ids": list(candidate.audience_ids),
        "source_turn_id": candidate.source_turn_id,
        "requested_by_user": candidate.requested_by_user,
        "confirmed_by_user": candidate.confirmed_by_user,
        "supersedes_id": candidate.supersedes_id,
        "topic_key": candidate.topic_key,
        "derived_from_ids": list(candidate.derived_from_ids),
        "expires_at": candidate.expires_at.isoformat() if candidate.expires_at else None,
        "metadata": candidate.metadata,
    }


def _candidate_from_payload(payload: dict[str, object]) -> MemoryCandidate:
    return MemoryCandidate(
        content=str(payload["content"]),
        owner_id=(str(payload["owner_id"]) if payload.get("owner_id") is not None else None),
        identity_status=str(payload["identity_status"]),
        confidence=float(payload["confidence"]),
        importance=float(payload["importance"]),
        visibility=Visibility(str(payload["visibility"])),
        kind=MemoryKind(str(payload["kind"])),
        sensitivity=Sensitivity(str(payload["sensitivity"])),
        retention=RetentionClass(str(payload.get("retention", "standard"))),
        audience_ids=tuple(str(item) for item in payload.get("audience_ids", [])),
        source_turn_id=(
            str(payload["source_turn_id"])
            if payload.get("source_turn_id") is not None
            else None
        ),
        requested_by_user=bool(payload.get("requested_by_user", False)),
        confirmed_by_user=bool(payload.get("confirmed_by_user", False)),
        supersedes_id=(
            str(payload["supersedes_id"])
            if payload.get("supersedes_id") is not None
            else None
        ),
        topic_key=(str(payload["topic_key"]) if payload.get("topic_key") else None),
        derived_from_ids=tuple(str(item) for item in payload.get("derived_from_ids", [])),
        expires_at=(
            datetime.fromisoformat(str(payload["expires_at"]))
            if payload.get("expires_at")
            else None
        ),
        metadata=dict(payload.get("metadata", {})),
    )
