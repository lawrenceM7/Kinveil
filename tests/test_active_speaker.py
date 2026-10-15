import numpy as np

from kinveil.audio.voice import AudioVisualSpeakerResolver, MouthMotionAnalyzer
from kinveil.audio.voice.active_speaker import MouthMotion
from kinveil.audio.voice.models import SpeakerTrackSnapshot
from kinveil.vision.face import FaceBox, TrackSnapshot


def voice(person_id: str | None = None) -> SpeakerTrackSnapshot:
    return SpeakerTrackSnapshot(
        "voice-1",
        person_id,
        "Alice" if person_id else "Speaker A",
        0.9 if person_id else 0.0,
        person_id is not None,
        3,
        2.0,
    )


def face(person_id: str | None = "person-1", track_id: str = "face-1") -> TrackSnapshot:
    return TrackSnapshot(
        track_id,
        person_id,
        "Alice" if person_id else "Visitor A",
        FaceBox(20, 20, 140, 140),
        0.9 if person_id else 0.0,
        person_id is not None,
        2.0,
        4,
        ((55.0, 55.0), (105.0, 55.0), (80.0, 80.0), (60.0, 105.0), (100.0, 105.0)),
    )


def motions(
    person_id: str | None = "person-1",
    track_id: str = "face-1",
    score: float = 0.03,
) -> tuple[MouthMotion, ...]:
    return tuple(
        MouthMotion(
            track_id,
            person_id,
            "Alice" if person_id else "Visitor A",
            timestamp,
            score,
            True,
        )
        for timestamp in (1.1, 1.4, 1.7, 1.9)
    )


def test_known_face_and_transient_voice_resolve_without_identity_confirmation():
    result = AudioVisualSpeakerResolver().resolve(
        voice(),
        motions(),
        segment_started_at=1.0,
        segment_ended_at=2.0,
    )

    assert result.accepted
    assert result.person_id == "person-1"
    assert result.label == "Alice"
    assert "voice remains transient" in result.reason


def test_matching_face_and_voice_are_accepted():
    result = AudioVisualSpeakerResolver().resolve(
        voice("person-1"),
        motions(),
        segment_started_at=1.0,
        segment_ended_at=2.0,
    )

    assert result.accepted
    assert result.reason == "face, voice, and lip motion agree"


def test_face_voice_conflict_is_unknown():
    result = AudioVisualSpeakerResolver().resolve(
        voice("different-person"),
        motions(),
        segment_started_at=1.0,
        segment_ended_at=2.0,
    )

    assert not result.accepted
    assert result.label == "Unknown speaker"
    assert "conflict" in result.reason


def test_two_moving_mouths_are_unknown():
    result = AudioVisualSpeakerResolver().resolve(
        voice(),
        motions() + motions("person-2", "face-2", 0.02),
        segment_started_at=1.0,
        segment_ended_at=2.0,
    )

    assert not result.accepted
    assert result.reason == "multiple mouths moving during speech"


def test_known_separated_voice_selects_its_matching_moving_face():
    result = AudioVisualSpeakerResolver().resolve(
        voice("person-1"),
        motions("person-1", "face-1", 0.03) + motions("person-2", "face-2", 0.02),
        segment_started_at=1.0,
        segment_ended_at=2.0,
    )

    assert result.accepted
    assert result.person_id == "person-1"
    assert result.face_track_id == "face-1"


def test_no_visible_mouth_is_unknown():
    hidden = tuple(
        MouthMotion(item.face_track_id, item.person_id, item.label, item.observed_at, 0.0, False)
        for item in motions()
    )
    result = AudioVisualSpeakerResolver().resolve(
        voice(),
        hidden,
        segment_started_at=1.0,
        segment_ended_at=2.0,
    )

    assert not result.accepted
    assert result.reason == "no synchronized visible mouth"


def test_short_utterance_accepts_mouth_motion_just_outside_audio_boundary():
    near_boundary = tuple(
        MouthMotion("face-1", "person-1", "Alice", timestamp, 0.03, True)
        for timestamp in (0.85, 0.95, 1.05)
    )

    result = AudioVisualSpeakerResolver().resolve(
        voice(),
        near_boundary,
        segment_started_at=1.0,
        segment_ended_at=1.1,
    )

    assert result.accepted
    assert result.label == "Alice"


def test_mouth_motion_analyzer_uses_only_face_mouth_region():
    analyzer = MouthMotionAnalyzer()
    first_frame = np.zeros((180, 180, 3), dtype=np.uint8)
    second_frame = first_frame.copy()
    second_frame[90:125, 45:115] = 255
    tracked_face = face()

    first = analyzer.observe(first_frame, (tracked_face,), observed_at=1.0)[0]
    second = analyzer.observe(second_frame, (tracked_face,), observed_at=1.1)[0]

    assert first.score == 0.0
    assert second.score > 0.1
