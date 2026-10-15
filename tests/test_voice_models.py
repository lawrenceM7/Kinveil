import hashlib
import io
import tarfile

import pytest

from kinveil.audio.voice.model_files import (
    ArchivedVoiceModelSpec,
    VoiceModelSpec,
    install_default_voice_models,
    install_whisper_model,
)


def _spec(content: bytes) -> VoiceModelSpec:
    return VoiceModelSpec(
        filename="test.onnx",
        url="https://example.invalid/test.onnx",
        sha256=hashlib.sha256(content).hexdigest(),
        license_name="Test",
        source_url="https://example.invalid",
    )


def test_model_install_requires_explicit_license_acceptance(tmp_path):
    with pytest.raises(ValueError, match="licenses"):
        install_default_voice_models(
            tmp_path,
            accept_model_licenses=False,
            specs=(),
            archive_spec=None,
        )


def test_model_install_verifies_hash_and_reuses_valid_file(tmp_path):
    content = b"verified model"
    calls = 0

    def download(spec: VoiceModelSpec, destination):
        nonlocal calls
        del spec
        calls += 1
        destination.write_bytes(content)

    first = install_default_voice_models(
        tmp_path,
        accept_model_licenses=True,
        specs=(_spec(content),),
        archive_spec=None,
        downloader=download,
    )
    second = install_default_voice_models(
        tmp_path,
        accept_model_licenses=True,
        specs=(_spec(content),),
        archive_spec=None,
        downloader=download,
    )

    assert first == second
    assert first[0].read_bytes() == content
    assert calls == 1


def test_model_install_rejects_changed_download(tmp_path):
    def download(spec: VoiceModelSpec, destination):
        del spec
        destination.write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="hash mismatch"):
        install_default_voice_models(
            tmp_path,
            accept_model_licenses=True,
            specs=(_spec(b"expected"),),
            archive_spec=None,
            downloader=download,
        )

    assert not (tmp_path / "test.onnx").exists()
    assert not (tmp_path / "test.onnx.download").exists()


def test_archived_overlap_model_is_hash_checked_and_safely_extracted(tmp_path):
    model = b"overlap model"
    source = tmp_path / "source.tar.bz2"
    member_name = "bundle/model.onnx"
    with tarfile.open(source, "w:bz2") as archive:
        member = tarfile.TarInfo(member_name)
        member.size = len(model)
        archive.addfile(member, io.BytesIO(model))
    archive_bytes = source.read_bytes()
    spec = ArchivedVoiceModelSpec(
        filename="bundle.tar.bz2",
        url="https://example.invalid/bundle.tar.bz2",
        sha256=hashlib.sha256(archive_bytes).hexdigest(),
        member=member_name,
        installed_filename="overlap.onnx",
        installed_sha256=hashlib.sha256(model).hexdigest(),
        license_name="MIT",
        source_url="https://example.invalid",
    )

    installed = install_default_voice_models(
        tmp_path / "models",
        accept_model_licenses=True,
        specs=(),
        archive_spec=spec,
        downloader=lambda _spec, destination: destination.write_bytes(archive_bytes),
    )

    assert installed[0].name == "overlap.onnx"
    assert installed[0].read_bytes() == model


def test_whisper_install_downloads_only_hash_checked_runtime_files(tmp_path):
    encoder = b"encoder"
    decoder = b"decoder"
    tokens = b"tokens"
    files = {
        "small-encoder.int8.onnx": encoder,
        "small-decoder.int8.onnx": decoder,
        "small-tokens.txt": tokens,
    }
    specs = tuple(
        VoiceModelSpec(
            filename=name,
            url=f"https://example.invalid/{name}",
            sha256=hashlib.sha256(content).hexdigest(),
            license_name="MIT",
            source_url="https://example.invalid",
        )
        for name, content in files.items()
    )

    installed = install_whisper_model(
        tmp_path / "models" / "whisper",
        accept_model_license=True,
        specs=specs,
        downloader=lambda spec, destination: destination.write_bytes(files[spec.filename]),
    )

    assert (installed / "small-encoder.int8.onnx").read_bytes() == encoder
    assert (installed / "small-decoder.int8.onnx").read_bytes() == decoder
    assert (installed / "small-tokens.txt").read_bytes() == tokens


def test_whisper_install_requires_explicit_license_acceptance(tmp_path):
    with pytest.raises(ValueError, match="license"):
        install_whisper_model(tmp_path, accept_model_license=False)
