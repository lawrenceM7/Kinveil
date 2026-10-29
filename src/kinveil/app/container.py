from __future__ import annotations

from dataclasses import dataclass

from kinveil.audio.voice import VoiceStore
from kinveil.core import Settings
from kinveil.llm import BoundedContextBuilder, ContextLimits
from kinveil.memory import MemoryService, SQLiteMemoryStore
from kinveil.vision.face import FaceStore


@dataclass
class AppContainer:
    settings: Settings
    faces: FaceStore
    voices: VoiceStore
    memories: MemoryService
    llm_context: BoundedContextBuilder

    @classmethod
    def build(cls, settings: Settings | None = None) -> AppContainer:
        selected = settings or Settings.default()
        faces = FaceStore(selected.face_database)
        memory_store = SQLiteMemoryStore(selected.memory_database)
        return cls(
            selected,
            faces,
            VoiceStore(selected.face_database),
            MemoryService(
                memory_store,
                maximum_context_tokens=selected.memory_context_tokens,
            ),
            BoundedContextBuilder(
                ContextLimits(
                    maximum_context_tokens=selected.llm_context_tokens,
                    reserved_output_tokens=selected.llm_reserved_output_tokens,
                    maximum_memory_tokens=selected.memory_context_tokens,
                )
            ),
        )
