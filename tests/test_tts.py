import io
import wave

import numpy as np
import pytest

from kinveil.audio.tts import _encode_pcm16_wav, _parse_voice_id
from kinveil.cli import build_parser


def test_generated_samples_are_encoded_as_mono_pcm16_wav():
    data = _encode_pcm16_wav(np.array([-1.0, 0.0, 1.0], dtype=np.float32), 24_000)

    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24_000
        assert wav.getnframes() == 3


def test_kokoro_voice_id_must_be_numeric():
    assert _parse_voice_id("48") == 48
    with pytest.raises(ValueError, match="numeric"):
        _parse_voice_id("xiaoyi")


def test_tts_synthesize_command_exposes_model_voice_and_speed():
    args = build_parser().parse_args(
        [
            "tts",
            "synthesize",
            "你好 Kinveil",
            "--output",
            "reply.wav",
            "--voice-id",
            "48",
            "--speed",
            "1.1",
        ]
    )

    assert args.command == "tts"
    assert args.tts_command == "synthesize"
    assert args.voice_id == 48
    assert args.speed == 1.1
