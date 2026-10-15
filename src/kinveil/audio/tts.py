from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SynthesizedSpeech:
    audio: bytes
    media_type: str
    sample_rate: int


class TextToSpeech(Protocol):
    def synthesize(self, text: str, voice: str | None = None) -> SynthesizedSpeech: ...


class SherpaOnnxKokoroTTS:
    """Local Chinese/English Kokoro synthesis through sherpa-onnx."""

    def __init__(
        self,
        model_directory: Path,
        *,
        default_voice_id: int = 48,
        speed: float = 1.0,
        num_threads: int = 2,
    ) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        model = _one_of(
            model_directory / "model.int8.onnx",
            model_directory / "model.onnx",
        )
        voices = _required(model_directory / "voices.bin")
        tokens = _required(model_directory / "tokens.txt")
        data_dir = _required_directory(model_directory / "espeak-ng-data")
        lexicons = [
            path
            for path in (
                model_directory / "lexicon-us-en.txt",
                model_directory / "lexicon-gb-en.txt",
                model_directory / "lexicon-zh.txt",
            )
            if path.is_file()
        ]
        if not lexicons:
            raise FileNotFoundError(f"Kokoro lexicon files not found in {model_directory}")
        rule_fsts = [
            path
            for path in (
                model_directory / "phone-zh.fst",
                model_directory / "date-zh.fst",
                model_directory / "number-zh.fst",
            )
            if path.is_file()
        ]
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("text-to-speech requires Kinveil's voice extra") from exc

        kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(model),
            voices=str(voices),
            tokens=str(tokens),
            lexicon=",".join(str(path) for path in lexicons),
            data_dir=str(data_dir),
        )
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=kokoro,
            num_threads=num_threads,
            provider="cpu",
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            rule_fsts=",".join(str(path) for path in rule_fsts),
        )
        self._tts = sherpa_onnx.OfflineTts(config)
        self.default_voice_id = default_voice_id
        self.speed = speed

    @property
    def num_voices(self) -> int:
        return int(self._tts.num_speakers)

    def synthesize(self, text: str, voice: str | None = None) -> SynthesizedSpeech:
        content = text.strip()
        if not content:
            raise ValueError("text must not be empty")
        voice_id = self.default_voice_id if voice is None else _parse_voice_id(voice)
        if not 0 <= voice_id < self.num_voices:
            raise ValueError(
                f"voice ID must be between 0 and {self.num_voices - 1}, got {voice_id}"
            )
        generated = self._tts.generate(content, sid=voice_id, speed=self.speed)
        sample_rate = int(generated.sample_rate)
        return SynthesizedSpeech(
            _encode_pcm16_wav(generated.samples, sample_rate),
            "audio/wav",
            sample_rate,
        )


def _parse_voice_id(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"voice must be a numeric Kokoro speaker ID, got {value!r}") from exc


def _encode_pcm16_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise ValueError("generated samples must be finite and non-empty")
    pcm = np.rint(np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def _required(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required Kokoro model file not found: {path}")
    return path


def _required_directory(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"required Kokoro model directory not found: {path}")
    return path


def _one_of(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"none of the required Kokoro model files exist: {paths}")
