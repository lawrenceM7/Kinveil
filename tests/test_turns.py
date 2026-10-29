from kinveil.audio import Transcript
from kinveil.audio.voice import ActiveSpeakerResult, TurnState, UserTurnStateMachine


def result(person_id="person-1", label="Alice", *, accepted=True):
    return ActiveSpeakerResult(
        person_id if accepted else None,
        label if accepted else "Unknown speaker",
        accepted,
        "face and lip motion agree" if accepted else "no synchronized visible mouth",
        "face-1" if accepted else None,
        "voice-1",
        0.03 if accepted else 0.0,
    )


def transcript(text):
    return Transcript(text, "zh-en", audio_duration=1.0, audio_rms=0.02)


def test_normal_complete_turn_finalizes_after_short_pause():
    turns = UserTurnStateMachine(normal_pause_seconds=0.65)
    turns.ingest(result(), transcript("帮我查一下天气"), started_at=1.0, ended_at=2.0)

    assert turns.poll(2.64) == ()
    (turn,) = turns.poll(2.66)

    assert turn.text == "帮我查一下天气"
    assert turn.person_id == "person-1"
    assert turn.identity_status == "confirmed"
    assert turns.state == TurnState.LISTENING


def test_incomplete_phrase_waits_longer_for_continuation():
    turns = UserTurnStateMachine(normal_pause_seconds=0.65, incomplete_pause_seconds=1.0)
    turns.ingest(result(), transcript("我想让你"), started_at=1.0, ended_at=2.0)

    assert turns.poll(2.8) == ()
    turns.ingest(result(), transcript("帮我查一下天气"), started_at=2.85, ended_at=3.8)
    (turn,) = turns.poll(4.45)

    assert turn.text == "我想让你帮我查一下天气"
    assert turn.segment_count == 2


def test_active_speech_prevents_early_finalization():
    turns = UserTurnStateMachine()
    turns.ingest(result(), transcript("第一段"), started_at=1.0, ended_at=2.0)
    turns.observe_speech(True)

    assert turns.poll(5.0) == ()
    assert turns.state == TurnState.SPEAKING


def test_different_speaker_finalizes_previous_turn_immediately():
    turns = UserTurnStateMachine()
    turns.ingest(result(), transcript("第一句话"), started_at=1.0, ended_at=2.0)

    (first,) = turns.ingest(
        result("person-2", "Bob"),
        transcript("第二句话"),
        started_at=2.2,
        ended_at=3.0,
    )

    assert first.label == "Alice"
    assert first.text == "第一句话"


def test_unknown_segment_makes_identity_partial_without_losing_known_person():
    turns = UserTurnStateMachine()
    turns.ingest(result(), transcript("这是一句"), started_at=1.0, ended_at=2.0)
    turns.ingest(
        result(accepted=False),
        transcript("完整的话"),
        started_at=2.2,
        ended_at=3.0,
    )

    (turn,) = turns.flush()

    assert turn.person_id == "person-1"
    assert turn.identity_status == "partial"
    assert turn.text == "这是一句完整的话"


def test_repeated_overlap_is_removed_when_segments_are_joined():
    turns = UserTurnStateMachine()
    turns.ingest(result(), transcript("我们讨论一下"), started_at=1.0, ended_at=2.0)
    turns.ingest(result(), transcript("讨论一下这个问题"), started_at=2.2, ended_at=3.0)

    (turn,) = turns.flush()

    assert turn.text == "我们讨论一下这个问题"
