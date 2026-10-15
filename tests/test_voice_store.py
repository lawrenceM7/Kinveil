import numpy as np
import pytest

from kinveil.audio.voice import VoiceStore
from kinveil.vision.face import FaceStore


def _person_and_store(tmp_path):
    database = tmp_path / "identities.db"
    faces = FaceStore(database)
    person = faces.create_anonymous_person(consent=True)
    return person, VoiceStore(database)


def test_voice_storage_requires_separate_consent(tmp_path):
    person, store = _person_and_store(tmp_path)

    with pytest.raises(PermissionError):
        store.maintain_embedding_pool(
            person.id,
            np.array([1.0, 0.0]),
            quality_score=0.9,
        )

    store.grant_consent(person.id, consent=True)
    store.maintain_embedding_pool(
        person.id,
        np.array([2.0, 0.0]),
        quality_score=0.9,
    )

    stored = store.embeddings()
    assert len(stored) == 1
    assert stored[0][0] == person.id
    np.testing.assert_allclose(stored[0][1], [1.0, 0.0])


def test_voice_pool_is_bounded_and_survives_restart(tmp_path):
    person, store = _person_and_store(tmp_path)
    store.grant_consent(person.id, consent=True)

    for index in range(8):
        angle = index * 0.2
        store.maintain_embedding_pool(
            person.id,
            np.array([np.cos(angle), np.sin(angle)]),
            quality_score=index / 10,
            max_samples=3,
            duplicate_similarity=1.1,
        )

    assert store.sample_count(person.id) == 3
    reopened = VoiceStore(store.database)
    assert reopened.has_consent(person.id)
    assert reopened.sample_count(person.id) == 3


def test_revoking_voice_consent_deletes_only_voice_templates(tmp_path):
    person, store = _person_and_store(tmp_path)
    store.grant_consent(person.id, consent=True)
    store.maintain_embedding_pool(
        person.id,
        np.array([1.0, 0.0]),
        quality_score=0.9,
    )

    assert store.revoke_consent(person.id)
    assert not store.has_consent(person.id)
    assert store.sample_count(person.id) == 0
    assert FaceStore(store.database).get_person(person.id).id == person.id


def test_voice_consent_cannot_target_unknown_person(tmp_path):
    database = tmp_path / "identities.db"
    FaceStore(database)
    store = VoiceStore(database)

    with pytest.raises(KeyError):
        store.grant_consent("missing", consent=True)
