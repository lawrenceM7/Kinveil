from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

SEPFORMER_WHAM_REVISION = "3f70635a1a640713e5af867b35de7929834bdeda"


@dataclass(frozen=True)
class SeparatedSources:
    sources: tuple[np.ndarray, ...]
    sample_rate: int


class SpeechSeparator(Protocol):
    def separate(self, samples: np.ndarray, sample_rate: int) -> SeparatedSources: ...


class SpeechBrainSepFormerSeparator:
    """Two-speaker separation backed by a local SpeechBrain SepFormer model."""

    def __init__(self, model_directory: Path) -> None:
        hyperparams = model_directory / "hyperparams.yaml"
        if not hyperparams.is_file():
            raise FileNotFoundError(f"SepFormer model not found: {hyperparams}")
        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError as exc:
            raise RuntimeError(
                "speech separation requires Kinveil's separation extra: "
                "python -m pip install -e '.[separation]'"
            ) from exc
        self._model = SepformerSeparation.from_hparams(
            source=str(model_directory),
            savedir=str(model_directory),
            run_opts={"device": "cpu"},
        )
        self.sample_rate = int(self._model.hparams.sample_rate)

    def separate(self, samples: np.ndarray, sample_rate: int) -> SeparatedSources:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if waveform.size == 0 or not np.all(np.isfinite(waveform)):
            raise ValueError("audio samples must be finite and non-empty")
        try:
            import torch
            import torchaudio.functional as audio_functional
        except ImportError as exc:
            raise RuntimeError("speech separation requires Kinveil's separation extra") from exc

        batch = torch.from_numpy(waveform).unsqueeze(0)
        if sample_rate != self.sample_rate:
            batch = audio_functional.resample(batch, sample_rate, self.sample_rate)
        with torch.inference_mode():
            separated = self._model.separate_batch(batch).detach().cpu()[0]
        sources = tuple(
            np.ascontiguousarray(separated[:, index].numpy(), dtype=np.float32)
            for index in range(separated.shape[-1])
        )
        return SeparatedSources(sources, self.sample_rate)


def install_sepformer_model(
    directory: Path,
    *,
    accept_model_license: bool,
    source: str = "speechbrain/sepformer-wham",
    revision: str = SEPFORMER_WHAM_REVISION,
) -> Path:
    """Download only the inference files SpeechBrain declares for SepFormer."""
    if not accept_model_license:
        raise ValueError("the Apache-2.0 SepFormer model license must be accepted")
    try:
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.utils.fetching import LocalStrategy
    except ImportError as exc:
        raise RuntimeError(
            "install Kinveil's separation extra first: python -m pip install -e '.[separation]'"
        ) from exc
    directory.mkdir(parents=True, exist_ok=True)
    SepformerSeparation.from_hparams(
        source=source,
        savedir=str(directory),
        run_opts={"device": "cpu"},
        revision=revision,
        local_strategy=LocalStrategy.COPY,
    )
    return directory
