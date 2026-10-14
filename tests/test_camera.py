from kinveil.vision.face import CameraPresenceRunner, FaceBox, TrackSnapshot


def snapshot(
    track_id: str,
    *,
    persistent: bool = False,
    observations: int = 1,
) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=track_id,
        person_id="person-id" if persistent else None,
        label="Known" if persistent else "Visitor A",
        box=FaceBox(0, 0, 100, 100),
        identity_confidence=1.0 if persistent else 0.0,
        persistent=persistent,
        last_seen=1.0,
        observations=observations,
    )


def test_enrollment_candidate_must_be_stable_unknown_and_not_already_prompted() -> None:
    known = snapshot("known", persistent=True, observations=10)
    early = snapshot("early", observations=3)
    declined = snapshot("declined", observations=5)
    ready = snapshot("ready", observations=4)

    selected = CameraPresenceRunner._next_enrollment_candidate(
        (known, early, declined, ready),
        {"declined"},
        required_observations=4,
    )

    assert selected == ready
