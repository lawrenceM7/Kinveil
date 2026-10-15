import numpy as np

from kinveil.audio.voice import (
    AdaptivePauseSegmenter,
    MicrophoneVoiceRunner,
    SherpaVadSegmenter,
    SpeechSegment,
    VoiceIdentityEngine,
    VoiceStore,
)
from kinveil.audio.voice import live as live_module
from kinveil.cli import build_parser
from kinveil.vision.face import FaceStore


class SampleBackend:
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate
        return np.array([float(samples.mean()), 0.1], dtype=np.float32)


class FakeSegmenter:
    sample_rate = 16_000

    def accept(self, samples: np.ndarray) -> tuple[np.ndarray, ...]:
        return (samples,)


class ScheduledSegmenter:
    sample_rate = 10

    def __init__(self, schedule):
        self.schedule = iter(schedule)

    def accept(self, samples):
        del samples
        return next(self.schedule, ())


def test_live_vad_accepts_short_natural_phrases(monkeypatch, tmp_path):
    class SileroConfig:
        model = ""
        min_silence_duration = 0.0
        min_speech_duration = 0.0
        max_speech_duration = 0.0

    class VadConfig:
        def __init__(self):
            self.silero_vad = SileroConfig()
            self.sample_rate = 0

    class Detector:
        def __init__(self, config, *, buffer_size_in_seconds):
            self.config = config
            self.buffer_size_in_seconds = buffer_size_in_seconds

    class FakeSherpa:
        VadModelConfig = VadConfig
        VoiceActivityDetector = Detector

    monkeypatch.setattr(live_module, "_import_sherpa_onnx", lambda: FakeSherpa)
    model = tmp_path / "vad.onnx"
    model.write_bytes(b"model")

    segmenter = SherpaVadSegmenter(model)

    assert segmenter._vad.config.silero_vad.min_speech_duration == 0.25
    assert segmenter.pre_roll_samples == 4_800
    assert segmenter.post_roll_samples == 3_200


def test_live_vad_adds_audio_before_and_after_detected_speech(monkeypatch, tmp_path):
    class SileroConfig:
        model = ""
        min_silence_duration = 0.0
        min_speech_duration = 0.0
        max_speech_duration = 0.0

    class VadConfig:
        def __init__(self):
            self.silero_vad = SileroConfig()
            self.sample_rate = 0

    class Front:
        start = 4
        samples = np.array([4.0, 5.0], dtype=np.float32)

    class Detector:
        def __init__(self, config, *, buffer_size_in_seconds):
            del config, buffer_size_in_seconds
            self.front = Front()
            self.ready = False

        def accept_waveform(self, waveform):
            del waveform
            self.ready = True

        def empty(self):
            return not self.ready

        def pop(self):
            self.ready = False

    class FakeSherpa:
        VadModelConfig = VadConfig
        VoiceActivityDetector = Detector

    monkeypatch.setattr(live_module, "_import_sherpa_onnx", lambda: FakeSherpa)
    model = tmp_path / "vad.onnx"
    model.write_bytes(b"model")
    segmenter = SherpaVadSegmenter(
        model,
        sample_rate=10,
        pre_roll_seconds=0.3,
        post_roll_seconds=0.2,
    )

    (segment,) = segmenter.accept(np.arange(10, dtype=np.float32))

    assert segment.start_sample == 1
    assert segment.end_sample == 8
    assert segment.samples.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_microphone_runner_processes_segment_without_storing_audio(tmp_path):
    database = tmp_path / "identities.db"
    FaceStore(database)
    store = VoiceStore(database)
    engine = VoiceIdentityEngine(SampleBackend(), store)
    results = []
    runner = MicrophoneVoiceRunner(
        engine,
        FakeSegmenter(),
        on_result=lambda snapshot, duration: results.append((snapshot, duration)),
    )
    speech = np.full(32_000, 0.2, dtype=np.float32)

    snapshot = runner.process_segment(speech)

    assert snapshot.label == "Speaker A"
    assert results[0][0] == snapshot
    assert results[0][1] == 2.0
    assert store.embeddings() == ()


def test_adaptive_pause_merges_short_fragments_with_their_continuation():
    first = SpeechSegment(np.ones(5, dtype=np.float32), 0, 5)
    second = SpeechSegment(np.ones(5, dtype=np.float32), 11, 16)
    segmenter = AdaptivePauseSegmenter(
        ScheduledSegmenter(((first,), (second,), (), ())),
        min_pause_seconds=0.6,
        max_pause_seconds=1.2,
    )

    assert segmenter.accept(np.ones(10, dtype=np.float32)) == ()
    assert segmenter.accept(np.ones(10, dtype=np.float32)) == ()
    (combined,) = segmenter.accept(np.ones(10, dtype=np.float32))

    assert combined.start_sample == 0
    assert combined.end_sample == 16
    assert combined.samples.size == 16


def test_adaptive_pause_releases_a_complete_segment_after_timeout():
    speech = SpeechSegment(np.ones(20, dtype=np.float32), 0, 20)
    segmenter = AdaptivePauseSegmenter(
        ScheduledSegmenter(((speech,), (), ())),
        min_pause_seconds=0.6,
        max_pause_seconds=1.2,
    )

    assert segmenter.accept(np.ones(20, dtype=np.float32)) == ()
    (released,) = segmenter.accept(np.zeros(10, dtype=np.float32))

    assert released == speech


def test_voice_watch_command_requires_local_model_paths():
    args = build_parser().parse_args(
        [
            "voices",
            "watch",
            "--embedding-model",
            "speaker.onnx",
            "--vad-model",
            "vad.onnx",
        ]
    )

    assert args.command == "voices"
    assert args.voice_command == "watch"
    assert args.embedding_model.name == "speaker.onnx"


def test_voice_model_setup_requires_explicit_license_flag():
    args = build_parser().parse_args(["voices", "setup-models"])

    assert args.voice_command == "setup-models"
    assert not args.accept_model_licenses


def test_voice_list_command_is_available():
    args = build_parser().parse_args(["voices", "list"])

    assert args.command == "voices"
    assert args.voice_command == "list"


def test_combined_session_watch_command():
    args = build_parser().parse_args(
        [
            "session",
            "watch",
            "--camera",
            "1",
            "--device",
            "2",
            "--whisper-model-dir",
            "whisper",
        ]
    )

    assert args.command == "session"
    assert args.camera == 1
    assert args.device == "2"
    assert args.whisper_model_dir.name == "whisper"


def test_start_is_the_combined_production_command():
    args = build_parser().parse_args(["start", "--camera", "1", "--device", "2"])

    assert args.command == "start"
    assert args.camera == 1
    assert args.device == "2"
    assert not args.debug


def test_start_debug_enables_console_transcription_mode():
    args = build_parser().parse_args(["start", "--debug", "--whisper-model-dir", "whisper"])

    assert args.command == "start"
    assert args.debug
    assert args.whisper_model_dir.name == "whisper"


def test_start_accepts_default_chinese_paraformer_directory():
    args = build_parser().parse_args(
        ["start", "--debug", "--paraformer-model-dir", "paraformer"]
    )

    assert args.paraformer_model_dir.name == "paraformer"


def test_separation_setup_requires_explicit_license_flag():
    args = build_parser().parse_args(["voices", "setup-separation-model"])

    assert args.voice_command == "setup-separation-model"
    assert not args.accept_model_license
