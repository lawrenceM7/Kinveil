from __future__ import annotations

from dataclasses import dataclass

from kinveil.audio.voice import VoiceStore
from kinveil.core import Settings
from kinveil.vision.face import FaceStore


@dataclass
class AppContainer:
    settings: Settings
    faces: FaceStore
    voices: VoiceStore

    @classmethod
    def build(cls, settings: Settings | None = None) -> AppContainer:
        selected = settings or Settings.default()
        faces = FaceStore(selected.face_database)
        return cls(selected, faces, VoiceStore(selected.face_database))
