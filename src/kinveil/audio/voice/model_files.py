from __future__ import annotations

import hashlib
import os
import tarfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class VoiceModelSpec:
    filename: str
    url: str
    sha256: str
    license_name: str
    source_url: str


@dataclass(frozen=True)
class ArchivedVoiceModelSpec:
    filename: str
    url: str
    sha256: str
    member: str
    installed_filename: str
    installed_sha256: str
    license_name: str
    source_url: str


DEFAULT_VOICE_MODELS = (
    VoiceModelSpec(
        filename="3dspeaker-campplus-zh-cn-16k-common.onnx",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-recongition-models/"
            "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
        ),
        sha256="f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11",
        license_name="Apache-2.0 (3D-Speaker)",
        source_url="https://github.com/modelscope/3D-Speaker",
    ),
    VoiceModelSpec(
        filename="silero-vad.onnx",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "asr-models/silero_vad.onnx"
        ),
        sha256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        license_name="MIT (Silero VAD)",
        source_url="https://github.com/snakers4/silero-vad",
    ),
)

DEFAULT_OVERLAP_MODEL = ArchivedVoiceModelSpec(
    filename="sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-segmentation-models/"
        "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    ),
    sha256="24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    member="sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    installed_filename="pyannote-segmentation-3.0.onnx",
    installed_sha256="220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    license_name="MIT (Pyannote Segmentation 3.0)",
    source_url="https://huggingface.co/pyannote/segmentation-3.0",
)

DEFAULT_TRANSCRIPTION_MODELS = (
    VoiceModelSpec(
        filename="small-encoder.int8.onnx",
        url=(
            "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small/"
            "resolve/8f3c18b358db4d1f2fc1eae49d75cd20989e4309/"
            "small-encoder.int8.onnx"
        ),
        sha256="4cbe7b22fa9026b843b60a68640c747de05bafb1a11b57edc0e66c232d9f33a9",
        license_name="MIT (OpenAI Whisper Small)",
        source_url="https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small",
    ),
    VoiceModelSpec(
        filename="small-decoder.int8.onnx",
        url=(
            "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small/"
            "resolve/8f3c18b358db4d1f2fc1eae49d75cd20989e4309/"
            "small-decoder.int8.onnx"
        ),
        sha256="acad50b5c782696e91b55914cc5ab4f756f1532f76e22aa6fc615f39fb69a8ee",
        license_name="MIT (OpenAI Whisper Small)",
        source_url="https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small",
    ),
    VoiceModelSpec(
        filename="small-tokens.txt",
        url=(
            "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small/"
            "resolve/8f3c18b358db4d1f2fc1eae49d75cd20989e4309/"
            "small-tokens.txt"
        ),
        sha256="b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126",
        license_name="MIT (OpenAI Whisper Small)",
        source_url="https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small",
    ),
)

DEFAULT_PARAFORMER_MODELS = (
    VoiceModelSpec(
        filename="model.int8.onnx",
        url=(
            "https://huggingface.co/csukuangfj/"
            "sherpa-onnx-paraformer-zh-2024-03-09/resolve/"
            "906992d326ebf0c5171cde675aa0902be9e5bc6c/model.int8.onnx"
        ),
        sha256="90bc03034ae1bef9575f8cc798cd1519c8be8aa9e8b458a033e32017ff4d584c",
        license_name="Apache-2.0 (FunASR Paraformer)",
        source_url=(
            "https://huggingface.co/csukuangfj/"
            "sherpa-onnx-paraformer-zh-2024-03-09"
        ),
    ),
    VoiceModelSpec(
        filename="tokens.txt",
        url=(
            "https://huggingface.co/csukuangfj/"
            "sherpa-onnx-paraformer-zh-2024-03-09/resolve/"
            "906992d326ebf0c5171cde675aa0902be9e5bc6c/tokens.txt"
        ),
        sha256="6c0e3b35cece259829e6cb5b8d90d13db88f61ea3a2953d11898e4b2bfd7a2e2",
        license_name="Apache-2.0 (FunASR Paraformer)",
        source_url=(
            "https://huggingface.co/csukuangfj/"
            "sherpa-onnx-paraformer-zh-2024-03-09"
        ),
    ),
)


def install_default_voice_models(
    directory: Path,
    *,
    accept_model_licenses: bool,
    specs: Iterable[VoiceModelSpec] = DEFAULT_VOICE_MODELS,
    archive_spec: ArchivedVoiceModelSpec | None = DEFAULT_OVERLAP_MODEL,
    downloader: Callable[[VoiceModelSpec, Path], None] | None = None,
) -> tuple[Path, ...]:
    if not accept_model_licenses:
        raise ValueError("model licenses must be explicitly accepted")
    directory.mkdir(parents=True, exist_ok=True)
    selected = tuple(specs)
    installed: list[Path] = []
    for spec in selected:
        destination = directory / spec.filename
        if destination.is_file() and _sha256(destination) == spec.sha256:
            installed.append(destination)
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        temporary.unlink(missing_ok=True)
        try:
            (downloader or _download)(spec, temporary)
            actual = _sha256(temporary)
            if actual != spec.sha256:
                raise ValueError(
                    f"hash mismatch for {spec.filename}: expected {spec.sha256}, got {actual}"
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        installed.append(destination)
    if archive_spec is not None:
        installed.append(
            _install_archived_model(
                directory,
                archive_spec,
                downloader=downloader,
            )
        )
    return tuple(installed)


def install_whisper_model(
    directory: Path,
    *,
    accept_model_license: bool,
    specs: Iterable[VoiceModelSpec] = DEFAULT_TRANSCRIPTION_MODELS,
    downloader: Callable[[VoiceModelSpec, Path], None] | None = None,
) -> Path:
    """Install the hash-pinned Whisper Small INT8 runtime files."""
    install_default_voice_models(
        directory,
        accept_model_licenses=accept_model_license,
        specs=specs,
        archive_spec=None,
        downloader=downloader,
    )
    return directory


def install_paraformer_model(
    directory: Path,
    *,
    accept_model_license: bool,
    specs: Iterable[VoiceModelSpec] = DEFAULT_PARAFORMER_MODELS,
    downloader: Callable[[VoiceModelSpec, Path], None] | None = None,
) -> Path:
    """Install the hash-pinned Chinese/English Paraformer INT8 runtime files."""
    install_default_voice_models(
        directory,
        accept_model_licenses=accept_model_license,
        specs=specs,
        archive_spec=None,
        downloader=downloader,
    )
    return directory


def voice_model_license_summary(
    specs: Iterable[VoiceModelSpec] = DEFAULT_VOICE_MODELS,
    archive_spec: ArchivedVoiceModelSpec | None = DEFAULT_OVERLAP_MODEL,
    transcription_specs: Iterable[VoiceModelSpec] = DEFAULT_PARAFORMER_MODELS,
) -> str:
    entries = [
        f"{spec.filename}: {spec.license_name} ({spec.source_url})" for spec in specs
    ]
    if archive_spec is not None:
        entries.append(
            f"{archive_spec.installed_filename}: {archive_spec.license_name} "
            f"({archive_spec.source_url})"
        )
    transcription = tuple(transcription_specs)
    if transcription:
        first = transcription[0]
        entries.append(f"Paraformer Chinese/English INT8: {first.license_name} ({first.source_url})")
    return "; ".join(entries)


def _install_archived_model(
    directory: Path,
    spec: ArchivedVoiceModelSpec,
    *,
    downloader: Callable[[VoiceModelSpec, Path], None] | None,
) -> Path:
    destination = directory / spec.installed_filename
    if destination.is_file() and _sha256(destination) == spec.installed_sha256:
        return destination
    archive = directory / f"{spec.filename}.download"
    temporary = destination.with_suffix(destination.suffix + ".download")
    archive.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    download_spec = VoiceModelSpec(
        spec.filename,
        spec.url,
        spec.sha256,
        spec.license_name,
        spec.source_url,
    )
    try:
        (downloader or _download)(download_spec, archive)
        actual_archive_hash = _sha256(archive)
        if actual_archive_hash != spec.sha256:
            raise ValueError(
                f"hash mismatch for {spec.filename}: expected {spec.sha256}, "
                f"got {actual_archive_hash}"
            )
        with tarfile.open(archive, "r:bz2") as bundle:
            member = bundle.getmember(spec.member)
            extracted = bundle.extractfile(member)
            if extracted is None or not member.isfile():
                raise ValueError(f"model archive member is not a file: {spec.member}")
            with temporary.open("wb") as output:
                while chunk := extracted.read(1024 * 1024):
                    output.write(chunk)
        actual_model_hash = _sha256(temporary)
        if actual_model_hash != spec.installed_sha256:
            raise ValueError(
                f"hash mismatch for {spec.installed_filename}: "
                f"expected {spec.installed_sha256}, got {actual_model_hash}"
            )
        os.replace(temporary, destination)
    finally:
        archive.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return destination


def _download(spec: VoiceModelSpec, destination: Path) -> None:
    request = Request(spec.url, headers={"User-Agent": "Kinveil/0.1"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
