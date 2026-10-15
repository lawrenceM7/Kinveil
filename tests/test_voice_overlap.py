import numpy as np

from kinveil.audio.stt import Transcript
from kinveil.audio.voice import AudioVisualSessionRunner, SpeechSegment
from kinveil.audio.voice.active_speaker import ActiveSpeakerResult
from kinveil.audio.voice.models import SpeakerTrackSnapshot
from kinveil.audio.voice.overlap import OverlapDetection, PyannoteOverlapDetector
from kinveil.audio.voice.separation import SeparatedSources
from kinveil.vision.face import FaceBox, TrackSnapshot


def logits(classes: list[int]) -> np.ndarray:
    values = np.full((len(classes), 7), -10.0, dtype=np.float32)
    values[np.arange(len(classes)), classes] = 10.0
    return values


def test_pyannote_powerset_detects_sustained_two_speaker_frames():
    result = PyannoteOverlapDetector.classify_logits(
        logits([1] * 40 + [4] * 12 + [2] * 40),
        frame_seconds=0.017,
    )

    assert result.detected
    assert result.max_speakers == 2
    assert result.overlap_duration > 0.15


def test_brief_overlap_does_not_reject_whole_segment():
    result = PyannoteOverlapDetector.classify_logits(
        logits([1] * 100 + [4] * 2),
        frame_seconds=0.017,
    )

    assert not result.detected
    assert result.max_speakers == 2


def test_silence_and_single_speaker_are_not_overlap():
    silence = PyannoteOverlapDetector.classify_logits(
        logits([0] * 50),
        frame_seconds=0.017,
    )
    single = PyannoteOverlapDetector.classify_logits(
        logits([0] * 5 + [3] * 45),
        frame_seconds=0.017,
    )

    assert not silence.detected
    assert silence.max_speakers == 0
    assert not single.detected
    assert single.max_speakers == 1


class AlwaysOverlap:
    def analyze(self, samples: np.ndarray, sample_rate: int) -> OverlapDetection:
        del samples, sample_rate
        return OverlapDetection(True, 0.3, 0.4, 2)


class NeverCalledVoice:
    def process_audio(self, *args, **kwargs):
        raise AssertionError("overlapping speech must not reach voice identity")


class Segmenter:
    sample_rate = 16_000


class TwoSourceSeparator:
    def separate(self, samples: np.ndarray, sample_rate: int) -> SeparatedSources:
        del samples
        return SeparatedSources(
            (
                np.full(sample_rate, 0.2, dtype=np.float32),
                np.full(sample_rate, -0.2, dtype=np.float32),
            ),
            sample_rate,
        )


class RecordingVoice:
    def __init__(self):
        self.calls = []

    def process_audio(self, samples, sample_rate, **kwargs):
        from kinveil.audio.voice.models import SpeakerTrackSnapshot

        self.calls.append((samples, sample_rate, kwargs))
        number = len(self.calls)
        return SpeakerTrackSnapshot(
            f"voice-{number}",
            f"person-{number}",
            f"Person {number}",
            0.9,
            True,
            1,
            kwargs["observed_at"],
        )


def test_overlapping_audio_never_updates_voice_identity():
    runner = AudioVisualSessionRunner(
        presence=object(),
        voice=NeverCalledVoice(),
        segmenter=Segmenter(),
        overlap_detector=AlwaysOverlap(),
    )

    event = runner.process_audio_segment(
        np.zeros(16_000, dtype=np.float32),
        ended_at=2.0,
    )

    assert event.voice is None
    assert event.overlap.detected


def test_overlapping_audio_is_separated_without_updating_templates():
    voice = RecordingVoice()
    runner = AudioVisualSessionRunner(
        presence=object(),
        voice=voice,
        segmenter=Segmenter(),
        overlap_detector=AlwaysOverlap(),
        separator=TwoSourceSeparator(),
    )

    event = runner.process_audio_segment(
        np.zeros(16_000, dtype=np.float32),
        ended_at=2.0,
    )

    assert len(event.voices) == 2
    assert [item.voice.label for item in event.voices] == ["Person 1", "Person 2"]
    assert all(not call[2]["allow_persistent_update"] for call in voice.calls)


def test_debug_transcript_callback_receives_resolved_speaker_and_text():
    reported = []
    runner = AudioVisualSessionRunner(
        presence=object(),
        voice=object(),
        segmenter=Segmenter(),
        overlap_detector=AlwaysOverlap(),
        on_transcript=lambda result, transcript: reported.append((result.label, transcript.text)),
    )
    result = ActiveSpeakerResult("person-1", "Alice", True, "matched", "face-1", "voice-1", 0.1)

    runner._report_transcripts(((result, Transcript("hello")),))

    assert reported == [("Alice", "hello")]


def test_combined_runner_offers_stable_unknown_face_enrollment():
    runner = AudioVisualSessionRunner(
        presence=object(),
        voice=object(),
        segmenter=Segmenter(),
        overlap_detector=AlwaysOverlap(),
        face_enrollment_handler=lambda _track: "saved",
    )
    visitor = TrackSnapshot(
        "face-1",
        None,
        "Visitor A",
        FaceBox(0, 0, 100, 100),
        0.0,
        False,
        1.0,
        4,
        None,
    )

    candidate = runner._next_enrollment_candidate((visitor,), set())

    assert candidate == visitor


def test_audio_queue_preserves_vad_sample_timestamps():
    runner = AudioVisualSessionRunner(
        presence=object(),
        voice=object(),
        segmenter=Segmenter(),
        overlap_detector=AlwaysOverlap(),
    )
    segment = SpeechSegment(np.ones(8_000, dtype=np.float32), 16_000, 24_000)

    runner._queue_segment(segment, stream_started_at=10.0)

    queued = runner._segments.get_nowait()
    assert queued.started_at == 11.0
    assert queued.ended_at == 11.5


def test_generic_face_voice_agreement_auto_enrolls_after_three_observations():
    class PersistingVoice:
        def __init__(self):
            self.calls = []

        def persist_track(self, track_id, person_id, **kwargs):
            self.calls.append((track_id, person_id, kwargs))

    voice_engine = PersistingVoice()
    runner = AudioVisualSessionRunner(
        presence=object(),
        voice=voice_engine,
        segmenter=Segmenter(),
        overlap_detector=AlwaysOverlap(),
        auto_voice_enrollment=True,
    )
    voice = SpeakerTrackSnapshot("voice-any", None, "Speaker A", 0.0, False, 3, 1.0)
    result = ActiveSpeakerResult(
        "person-any",
        "Any future user",
        True,
        "face and lip motion agree; voice remains transient",
        "face-any",
        "voice-any",
        0.02,
    )

    assert not runner._maybe_persist_voice(result, voice)
    assert not runner._maybe_persist_voice(result, voice)
    assert runner._maybe_persist_voice(result, voice)

    assert voice_engine.calls == [
        (
            "voice-any",
            "person-any",
            {
                "voice_consent": True,
                "minimum_quality": 0.35,
                "minimum_samples": 2,
            },
        )
    ]
