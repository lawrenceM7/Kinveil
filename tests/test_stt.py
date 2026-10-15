import sys
from types import SimpleNamespace

import numpy as np
import pytest

from kinveil.audio.stt import (
    SherpaOnnxParaformerTranscriber,
    _find_model_file,
    _normalize_asr_audio,
    _remove_repeated_context,
)


def test_whisper_model_file_lookup_accepts_one_matching_file(tmp_path):
    expected = tmp_path / "small-encoder.int8.onnx"
    expected.write_bytes(b"model")

    assert _find_model_file(tmp_path, "encoder.int8.onnx") == expected


def test_whisper_model_file_lookup_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="encoder"):
        _find_model_file(tmp_path, "encoder.int8.onnx")


def test_transcript_input_contract_uses_float_audio():
    samples = np.zeros(16_000, dtype=np.float32)
    assert samples.dtype == np.float32


def test_repeated_text_from_audio_overlap_is_removed():
    assert _remove_repeated_context("我们可以聊一下", "聊一下你知道我是谁吗") == "你知道我是谁吗"


def test_distinct_adjacent_text_is_preserved():
    assert _remove_repeated_context("你好", "你知道我是谁吗") == "你知道我是谁吗"


def test_quiet_asr_audio_receives_bounded_gain_without_clipping():
    samples = np.array([0.01, -0.02, 0.01], dtype=np.float32)
    normalized = _normalize_asr_audio(samples, rms=0.01)

    assert np.max(np.abs(normalized)) <= 0.95
    assert np.allclose(normalized, samples * 3.0)


def test_paraformer_transcriber_uses_chinese_model_and_reports_input_metrics(
    monkeypatch, tmp_path
):
    (tmp_path / "model.int8.onnx").write_bytes(b"model")
    (tmp_path / "tokens.txt").write_text("token 0\n")
    accepted = []

    class Stream:
        result = SimpleNamespace(text="讨论一下")

        def accept_waveform(self, sample_rate, samples):
            accepted.append((sample_rate, samples.copy()))

    class Recognizer:
        @classmethod
        def from_paraformer(cls, **kwargs):
            assert kwargs["paraformer"].endswith("model.int8.onnx")
            return cls()

        def create_stream(self):
            return Stream()

        def decode_stream(self, stream):
            del stream

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=Recognizer),
    )
    transcriber = SherpaOnnxParaformerTranscriber(tmp_path)

    result = transcriber.transcribe_samples(np.full(16_000, 0.01, np.float32), 16_000)

    assert result.text == "讨论一下"
    assert result.language == "zh-en"
    assert result.audio_duration == 1.0
    assert result.audio_rms == pytest.approx(0.01)
    assert np.max(accepted[0][1]) == pytest.approx(0.03)
