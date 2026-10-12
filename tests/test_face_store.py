from pathlib import Path

import numpy as np
import pytest

from kinveil.vision.face import FaceRecognizer, FaceStore


def test_enrollment_requires_consent(tmp_path: Path) -> None:
    store = FaceStore(tmp_path / "faces.db")
    with pytest.raises(ValueError, match="consent"):
        store.create_person("Ada", consent=False)


def test_store_round_trip_and_delete(tmp_path: Path) -> None:
    store = FaceStore(tmp_path / "faces.db")
    person = store.create_person("Ada", consent=True)
    store.add_embedding(person.id, np.array([3.0, 4.0], dtype=np.float32))

    people = store.list_people()
    assert people[0].name == "Ada"
    assert people[0].sample_count == 1
    assert np.allclose(store.embeddings()[0][2], [0.6, 0.8])
    assert store.delete_person(person.id)
    assert store.embeddings() == ()


def test_template_pool_is_bounded_and_keeps_pose_buckets(tmp_path: Path) -> None:
    store = FaceStore(tmp_path / "faces.db")
    person = store.create_person("Ada", consent=True)

    for index in range(12):
        angle = index * 0.04
        store.maintain_embedding_pool(
            person.id,
            np.array([1.0, angle, angle * angle]),
            pose_label="front",
            quality_score=0.9,
            max_samples=10,
            max_samples_per_pose=4,
            duplicate_similarity=0.99999,
        )
    for index in range(3):
        store.maintain_embedding_pool(
            person.id,
            np.array([0.7, 0.7, index * 0.05]),
            pose_label="left",
            quality_score=0.9,
            max_samples=10,
            max_samples_per_pose=4,
            duplicate_similarity=0.99999,
        )

    counts = store.pose_counts(person.id)
    assert counts["front"] <= 4
    assert counts["left"] == 3
    assert sum(counts.values()) <= 10


def test_revoked_consent_stays_revoked_after_reopening_store(tmp_path: Path) -> None:
    database = tmp_path / "faces.db"
    store = FaceStore(database)
    person = store.create_person("Ada", consent=True)
    store.add_embedding(person.id, np.array([1.0, 0.0]))
    store.revoke_biometric_consent(person.id)

    reopened = FaceStore(database)
    assert reopened.get_person(person.id).biometric_consented_at is None
    assert reopened.embeddings() == ()


def test_recognizer_accepts_clear_match_and_rejects_unknown(tmp_path: Path) -> None:
    store = FaceStore(tmp_path / "faces.db")
    ada = store.create_person("Ada", consent=True)
    grace = store.create_person("Grace", consent=True)
    store.add_embedding(ada.id, np.array([1.0, 0.0]))
    store.add_embedding(grace.id, np.array([0.0, 1.0]))
    recognizer = FaceRecognizer(store, threshold=0.75, margin=0.10)

    match = recognizer.match(np.array([0.99, 0.01]))
    assert match.accepted and match.name == "Ada"

    unknown = recognizer.match(np.array([1.0, 1.0]))
    assert not unknown.accepted


def test_recognizer_rejects_ambiguous_match(tmp_path: Path) -> None:
    store = FaceStore(tmp_path / "faces.db")
    first = store.create_person("First", consent=True)
    second = store.create_person("Second", consent=True)
    store.add_embedding(first.id, np.array([1.0, 0.0]))
    store.add_embedding(second.id, np.array([0.99, 0.1]))

    result = FaceRecognizer(store, threshold=0.5, margin=0.1).match(np.array([1.0, 0.05]))
    assert not result.accepted
    assert result.reason == "ambiguous identity"
