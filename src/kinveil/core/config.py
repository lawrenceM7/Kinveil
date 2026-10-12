from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from platformdirs import user_data_dir


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    face_model: str = "buffalo_l"
    face_threshold: float = 0.48
    face_margin: float = 0.06
    detector_size: tuple[int, int] = (640, 640)

    @classmethod
    def default(cls) -> "Settings":
        override = os.environ.get("KINVEIL_DATA_DIR")
        return cls(data_dir=Path(override) if override else Path(user_data_dir("Kinveil", "Kinveil")))

    @property
    def face_database(self) -> Path:
        return self.data_dir / "faces.sqlite3"
