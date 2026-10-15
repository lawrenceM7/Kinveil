import numpy as np
import pytest

from kinveil.audio.stt import _find_model_file


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
