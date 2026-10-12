from pathlib import Path

import numpy as np
import pytest

from kinveil.vision.face import (
    DetectedFace,
    FaceBox,
    FaceStore,
    PresenceEngine,
    PresencePermissions,
)


class FakeBackend:
    def __init__(self, batches: list[tuple[DetectedFace, ...]]) -> None:
        self.batches = batches

    def detect_and_embed(self, frame: np.ndarray) -> tuple[DetectedFace, ...]:
        return self.batches.pop(0)


def face(vector: tuple[float, float], left: int = 0) -> DetectedFace:
    return DetectedFace(
        FaceBox(left, 0, left + 120, 120),
        np.asarray(vector, dtype=np.float32),
        0.99,
    )


def posed_face(
    vector: tuple[float, float], pose: tuple[float, float, float]
) -> DetectedFace:
    return DetectedFace(
        FaceBox(0, 0, 120, 120),
        np.asarray(vector, dtype=np.float32),
        0.99,
        pose,
    )


def permissions() -> PresencePermissions:
    return PresencePermissions(
        camera_allowed=True,
        transient_processing_allowed=True,
    )


def test_presence_requires_both_camera_permissions(tmp_path: Path) -> None:
    backend = FakeBackend([])
    store = FaceStore(tmp_path / "faces.db")
    with pytest.raises(PermissionError, match="camera"):
        PresenceEngine(backend, store, PresencePermissions(False, True))
    with pytest.raises(PermissionError, match="transient"):
        PresenceEngine(backend, store, PresencePermissions(True, False))


def test_unknown_people_stay_ephemeral_and_keep_labels(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            (face((1.0, 0.0)), face((0.0, 1.0), 200)),
            (face((0.01, 1.0), 205), face((1.0, 0.01), 5)),
        ]
    )
    store = FaceStore(tmp_path / "faces.db")
    engine = PresenceEngine(backend, store, permissions())

    first = engine.process_frame(np.zeros((1, 1, 3)), observed_at=1.0)
    second = engine.process_frame(np.zeros((1, 1, 3)), observed_at=2.0)

    assert [item.label for item in first] == ["Visitor A", "Visitor B"]
    assert [item.label for item in second] == ["Visitor B", "Visitor A"]
    assert store.list_people() == ()


def test_track_requires_consent_before_becoming_persistent(tmp_path: Path) -> None:
    backend = FakeBackend([(face((1.0, 0.0)),)])
    store = FaceStore(tmp_path / "faces.db")
    engine = PresenceEngine(backend, store, permissions())
    track = engine.process_frame(np.zeros((1, 1, 3)), observed_at=1.0)[0]

    with pytest.raises(ValueError, match="consent"):
        engine.persist_track(track.track_id, biometric_consent=False)
    person = engine.persist_track(track.track_id, biometric_consent=True)

    assert person.display_name is None
    assert person.local_alias.startswith("Person ")
    assert store.get_person(person.id).sample_count == 1


def test_persistent_person_is_recognized_and_can_be_named(tmp_path: Path) -> None:
    store = FaceStore(tmp_path / "faces.db")
    person = store.create_anonymous_person(consent=True)
    store.add_embedding(person.id, np.array([1.0, 0.0]))
    backend = FakeBackend([(face((0.999, 0.01)),)])
    engine = PresenceEngine(backend, store, permissions())

    snapshot = engine.process_frame(np.zeros((1, 1, 3)), observed_at=1.0)[0]
    assert snapshot.person_id == person.id
    assert snapshot.persistent

    with pytest.raises(ValueError, match="confirmed"):
        engine.confirm_display_name(person.id, "Alice", confirmed=False)
    named = engine.confirm_display_name(person.id, "Alice", confirmed=True)
    assert named.display_name == "Alice"
    assert engine.active_tracks()[0].label == "Alice"


def test_persistent_person_continuously_collects_bounded_pose_templates(
    tmp_path: Path,
) -> None:
    store = FaceStore(tmp_path / "faces.db")
    person = store.create_anonymous_person(consent=True)
    store.add_embedding(person.id, np.array([1.0, 0.0]), pose_label="front")
    backend = FakeBackend(
        [
            (posed_face((0.999, 0.01), (0.0, -25.0, 0.0)),),
            (posed_face((0.998, 0.02), (0.0, -27.0, 0.0)),),
        ]
    )
    engine = PresenceEngine(
        backend,
        store,
        permissions(),
        max_persistent_samples=3,
        max_samples_per_pose=2,
    )

    engine.process_frame(np.zeros((1, 1, 3)), observed_at=1.0)
    engine.process_frame(np.zeros((1, 1, 3)), observed_at=2.0)

    counts = store.pose_counts(person.id)
    assert counts["front"] == 1
    assert 1 <= counts["left"] <= 2
    assert sum(counts.values()) <= 3


def test_revoking_biometrics_removes_templates(tmp_path: Path) -> None:
    store = FaceStore(tmp_path / "faces.db")
    person = store.create_person("Alice", consent=True)
    store.add_embedding(person.id, np.array([1.0, 0.0]))

    assert store.revoke_biometric_consent(person.id)
    assert store.embeddings() == ()
    assert store.get_person(person.id).biometric_consented_at is None
