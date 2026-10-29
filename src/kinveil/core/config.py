from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    face_model: str = "buffalo_l"
    face_threshold: float = 0.48
    face_margin: float = 0.06
    detector_size: tuple[int, int] = (640, 640)
    memory_context_tokens: int = 800
    llm_context_tokens: int = 8192
    llm_reserved_output_tokens: int = 1024

    @classmethod
    def default(cls) -> Settings:
        override = os.environ.get("KINVEIL_DATA_DIR")
        data_dir = Path(override) if override else Path(user_data_dir("Kinveil", "Kinveil"))
        memory_context_tokens = int(os.environ.get("KINVEIL_MEMORY_CONTEXT_TOKENS", "800"))
        llm_context_tokens = int(os.environ.get("KINVEIL_LLM_CONTEXT_TOKENS", "8192"))
        llm_reserved_output_tokens = int(
            os.environ.get("KINVEIL_LLM_RESERVED_OUTPUT_TOKENS", "1024")
        )
        return cls(
            data_dir=data_dir,
            memory_context_tokens=memory_context_tokens,
            llm_context_tokens=llm_context_tokens,
            llm_reserved_output_tokens=llm_reserved_output_tokens,
        )

    @property
    def face_database(self) -> Path:
        return self.data_dir / "faces.sqlite3"

    @property
    def memory_database(self) -> Path:
        return self.data_dir / "memories.sqlite3"

    @property
    def voice_model_dir(self) -> Path:
        return self.data_dir / "models" / "voice"

    @property
    def voice_embedding_model(self) -> Path:
        return self.voice_model_dir / "3dspeaker-campplus-zh-cn-16k-common.onnx"

    @property
    def voice_vad_model(self) -> Path:
        return self.voice_model_dir / "silero-vad.onnx"

    @property
    def voice_overlap_model(self) -> Path:
        return self.voice_model_dir / "pyannote-segmentation-3.0.onnx"

    @property
    def voice_separation_model_dir(self) -> Path:
        return self.voice_model_dir / "sepformer-wham"

    @property
    def voice_transcription_model_dir(self) -> Path:
        return self.voice_model_dir / "sherpa-onnx-whisper-small"

    @property
    def voice_paraformer_model_dir(self) -> Path:
        return self.voice_model_dir / "sherpa-onnx-paraformer-zh-2024-03-09"

    @property
    def tts_model_dir(self) -> Path:
        return self.data_dir / "models" / "tts" / "kokoro-multi-lang-v1_0"
