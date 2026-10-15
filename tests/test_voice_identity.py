import numpy as np
import pytest

from kinveil.audio.voice import VoiceIdentityEngine, VoiceStore
from kinveil.vision.face import FaceStore


class MeanBackend:
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.array([samples.mean(), samples.std() + 0.01], dtype=np.float32)


def _engine(tmp_path):
    database = tmp_path / "identities.db"
    faces = FaceStore(database)
    voices = VoiceStore(database)
    return faces, voices, VoiceIdentityEngine(MeanBackend(), voices)


def test_natural_speech_builds_stable_transient_speaker(tmp_path):
    _, _, engine = _engine(tmp_path)

    first = engine.process_embedding(np.array([1.0, 0.0]), observed_at=1.0)
    second = engine.process_embedding(np.array([0.99, 0.04]), observed_at=2.0)

    assert first.track_id == second.track_id
    assert second.label == "Speaker A"
    assert second.person_id is None
    assert not second.persistent
    assert second.observations == 2


def test_consented_track_is_recognized_after_restart(tmp_path):
    faces, voices, engine = _engine(tmp_path)
    person = faces.create_person("Alice", consent=True)
    transient = engine.process_embedding(np.array([1.0, 0.0]), quality_score=0.9, observed_at=1.0)

    saved = engine.persist_track(transient.track_id, person.id, voice_consent=True)

    assert saved.person_id == person.id
    assert saved.label == "Alice"
    assert voices.sample_count(person.id) == 1

    restarted = VoiceIdentityEngine(MeanBackend(), VoiceStore(voices.database))
    recognized = restarted.process_embedding(
        np.array([0.99, 0.01]), quality_score=0.9, observed_at=1.0
    )
    assert recognized.person_id == person.id
    assert recognized.label == "Alice"
    assert recognized.persistent


def test_persistence_requires_voice_consent(tmp_path):
    faces, _, engine = _engine(tmp_path)
    person = faces.create_anonymous_person(consent=True)
    track = engine.process_embedding(np.array([1.0, 0.0]), observed_at=1.0)

    with pytest.raises(ValueError, match="voice biometric consent"):
        engine.persist_track(track.track_id, person.id, voice_consent=False)


def test_automatic_persistence_filters_low_quality_voice_samples(tmp_path):
    faces, voices, engine = _engine(tmp_path)
    person = faces.create_person("Future User", consent=True)
    track = engine.process_embedding(
        np.array([1.0, 0.0]), quality_score=0.2, observed_at=1.0
    )
    engine.process_embedding(
        np.array([0.8, 0.6]), quality_score=0.6, observed_at=2.0
    )
    track = engine.process_embedding(
        np.array([0.6, 0.8]), quality_score=0.7, observed_at=3.0
    )

    engine.persist_track(
        track.track_id,
        person.id,
        voice_consent=True,
        minimum_quality=0.35,
        minimum_samples=2,
    )

    assert voices.sample_count(person.id) == 2


def test_engine_never_passes_audio_to_store(tmp_path):
    _, voices, engine = _engine(tmp_path)
    snapshot = engine.process_audio(
        np.array([0.3, 0.4, 0.2], dtype=np.float32),
        16_000,
        observed_at=1.0,
    )

    assert snapshot.label == "Speaker A"
    assert voices.embeddings() == ()


def test_expired_speaker_gets_a_new_session_track(tmp_path):
    _, _, engine = _engine(tmp_path)
    first = engine.process_embedding(np.array([1.0, 0.0]), observed_at=1.0)
    second = engine.process_embedding(np.array([1.0, 0.0]), observed_at=50.0)

    assert first.track_id != second.track_id
    assert second.label == "Speaker B"


def test_separated_audio_cannot_refresh_persistent_template_pool(tmp_path):
    faces, voices, engine = _engine(tmp_path)
    person = faces.create_person("Alice", consent=True)
    track = engine.process_embedding(np.array([1.0, 0.0]), quality_score=0.9, observed_at=1.0)
    engine.persist_track(track.track_id, person.id, voice_consent=True)

    engine.process_embedding(
        np.array([0.99, 0.01]),
        quality_score=1.0,
        observed_at=2.0,
        allow_persistent_update=False,
    )

    assert voices.sample_count(person.id) == 1
