import numpy as np

from kinveil.audio.voice import (
    MicrophoneVoiceRunner,
    SherpaVadSegmenter,
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


def test_separation_setup_requires_explicit_license_flag():
    args = build_parser().parse_args(["voices", "setup-separation-model"])

    assert args.voice_command == "setup-separation-model"
    assert not args.accept_model_license
