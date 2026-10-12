from __future__ import annotations

from dataclasses import dataclass

from kinveil.core import Settings
from kinveil.vision.face import FaceStore


@dataclass
class AppContainer:
    settings: Settings
    faces: FaceStore

    @classmethod
    def build(cls, settings: Settings | None = None) -> "AppContainer":
        selected = settings or Settings.default()
        return cls(selected, FaceStore(selected.face_database))
