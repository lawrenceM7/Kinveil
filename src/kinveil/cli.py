from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinveil.app import AppContainer
from kinveil.audio import SherpaOnnxKokoroTTS, SherpaOnnxWhisperTranscriber, Transcript
from kinveil.audio.voice import (
    ActiveSpeakerResult,
    AudioVisualSessionRunner,
    MicrophoneVoiceRunner,
    PyannoteOverlapDetector,
    SherpaOnnxVoiceBackend,
    SherpaVadSegmenter,
    SpeakerTrackSnapshot,
    SpeechBrainSepFormerSeparator,
    VoiceIdentityEngine,
    install_default_voice_models,
    install_sepformer_model,
    install_whisper_model,
    voice_model_license_summary,
)
from kinveil.vision.face import (
    CameraPresenceRunner,
    FaceEnrollment,
    FaceRecognizer,
    InsightFaceBackend,
    Person,
    PresenceEngine,
    PresencePermissions,
    TrackSnapshot,
)
from kinveil.vision.face.service import recognize_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kinveil", description="Local-first personal AI")
    parser.add_argument("--data-dir", type=Path, help="override Kinveil's local data directory")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="start Kinveil with camera and microphone")
    _add_session_arguments(start, include_debug=True)

    tts = commands.add_parser("tts", help="synthesize speech locally")
    tts_commands = tts.add_subparsers(dest="tts_command", required=True)
    synthesize = tts_commands.add_parser("synthesize", help="convert text to a local WAV file")
    synthesize.add_argument("text", help="text to speak")
    synthesize.add_argument("--output", type=Path, required=True, help="output WAV path")
    synthesize.add_argument("--model-dir", type=Path, help="Kokoro model directory")
    synthesize.add_argument("--voice-id", type=int, default=48)
    synthesize.add_argument("--speed", type=float, default=1.0)

    faces = commands.add_parser("faces", help="manage local face identities")
    face_commands = faces.add_subparsers(dest="face_command", required=True)

    enroll = face_commands.add_parser("enroll-video", help="enroll one consenting person")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--video", required=True, type=Path)
    enroll.add_argument("--consent", action="store_true", help="confirm the subject consented")

    face_commands.add_parser("list", help="list enrolled people")
    recognize = face_commands.add_parser("recognize-image", help="identify faces in one image")
    recognize.add_argument("image", type=Path)
    delete = face_commands.add_parser("delete", help="delete a person and all face templates")
    delete.add_argument("person_id")
    name = face_commands.add_parser("name", help="confirm a display name for a person")
    name.add_argument("person_id")
    name.add_argument("display_name")
    name.add_argument("--confirmed", action="store_true")
    revoke = face_commands.add_parser(
        "revoke-biometrics", help="delete face templates and revoke biometric consent"
    )
    revoke.add_argument("person_id")
    watch = face_commands.add_parser("watch", help="continuously notice people through the camera")
    watch.add_argument("--camera", type=int, default=0)
    watch.add_argument("--interval", type=float, default=0.5)
    watch.add_argument(
        "--allow-camera-processing",
        action="store_true",
        help="allow camera access and transient in-memory face processing",
    )
    watch.add_argument("--allow-camera", action="store_true")
    watch.add_argument("--allow-transient-processing", action="store_true")

    voices = commands.add_parser("voices", help="recognize voices locally")
    voice_commands = voices.add_subparsers(dest="voice_command", required=True)
    voice_watch = voice_commands.add_parser(
        "watch", help="listen for natural speech and track speakers"
    )
    voice_watch.add_argument("--embedding-model", type=Path)
    voice_watch.add_argument("--vad-model", type=Path)
    voice_watch.add_argument("--device", help="microphone device index or name")
    voice_watch.add_argument(
        "--person-id",
        help="existing person to offer voice enrollment for after stable speech",
    )
    voice_watch.add_argument(
        "--allow-microphone-processing",
        action="store_true",
        help="allow local microphone capture and transient voice processing",
    )
    voice_revoke = voice_commands.add_parser(
        "revoke", help="delete one person's stored voice templates"
    )
    voice_revoke.add_argument("person_id")
    voice_commands.add_parser("list", help="list voice consent and template counts")
    voice_setup = voice_commands.add_parser(
        "setup-models", help="download pinned voice and debug transcription models"
    )
    voice_setup.add_argument(
        "--accept-model-licenses",
        action="store_true",
        help="accept the listed third-party model licenses",
    )
    separation_setup = voice_commands.add_parser(
        "setup-separation-model",
        help="download the optional two-speaker SepFormer model",
    )
    separation_setup.add_argument(
        "--accept-model-license",
        action="store_true",
        help="accept the Apache-2.0 SpeechBrain SepFormer model license",
    )

    session = commands.add_parser(
        "session", help="run combined face, lip-motion, and voice presence"
    )
    session_commands = session.add_subparsers(dest="session_command", required=True)
    session_watch = session_commands.add_parser(
        "watch", help="identify the active speaker with camera and microphone"
    )
    _add_session_arguments(session_watch, include_debug=False)
    return parser


def _add_session_arguments(parser: argparse.ArgumentParser, *, include_debug: bool) -> None:
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", help="microphone device index or name")
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument("--vad-model", type=Path)
    parser.add_argument("--overlap-model", type=Path)
    parser.add_argument(
        "--separation-model",
        type=Path,
        help="local SpeechBrain SepFormer model directory",
    )
    parser.add_argument(
        "--whisper-model-dir",
        type=Path,
        help="local sherpa-onnx Whisper directory",
    )
    parser.add_argument(
        "--transcription-language",
        default="",
        help="Whisper language code; omit it for automatic detection",
    )
    parser.add_argument(
        "--allow-local-sensing",
        action="store_true",
        help=(
            "allow camera/microphone processing and bounded voice-template enrollment "
            "for recognized people"
        ),
    )
    if include_debug:
        parser.add_argument(
            "--debug",
            action="store_true",
            help="transcribe recognized speech and print it to the console",
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from kinveil.core import Settings

    settings = Settings.default()
    if args.data_dir:
        settings = Settings(data_dir=args.data_dir)
    container = AppContainer.build(settings)

    if args.command == "tts":
        model_directory = args.model_dir or settings.tts_model_dir
        if not model_directory.is_dir():
            raise SystemExit(
                "Kokoro model is missing; extract kokoro-multi-lang-v1_0 into "
                f"{model_directory} or pass --model-dir"
            )
        engine = SherpaOnnxKokoroTTS(
            model_directory,
            default_voice_id=args.voice_id,
            speed=args.speed,
        )
        speech = engine.synthesize(args.text)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(speech.audio)
        print(f"Saved {args.output} ({speech.sample_rate} Hz, voice {args.voice_id})")
        return 0

    if args.command in {"start", "session"}:
        sensing_allowed = args.allow_local_sensing or _confirm(
            "Allow Kinveil to use the camera and microphone for local face, voice, "
            "and lip-motion analysis, and to keep a bounded voice-template pool for "
            "people already approved for recognition? Raw media is not recorded"
        )
        if not sensing_allowed:
            raise SystemExit("camera and microphone processing were not granted")
        embedding_model = args.embedding_model or settings.voice_embedding_model
        vad_model = args.vad_model or settings.voice_vad_model
        overlap_model = args.overlap_model or settings.voice_overlap_model
        if not embedding_model.is_file() or not vad_model.is_file() or not overlap_model.is_file():
            raise SystemExit(
                "voice models are missing; run: kinveil voices setup-models --accept-model-licenses"
            )
        face_backend = InsightFaceBackend(settings.face_model, settings.detector_size)
        presence = PresenceEngine(
            face_backend,
            container.faces,
            PresencePermissions(True, True),
            identity_threshold=settings.face_threshold,
            identity_margin=settings.face_margin,
        )
        voice_backend = SherpaOnnxVoiceBackend(embedding_model)
        voice = VoiceIdentityEngine(voice_backend, container.voices)
        segmenter = SherpaVadSegmenter(vad_model)
        overlap_detector = PyannoteOverlapDetector(overlap_model)
        separation_model = args.separation_model or settings.voice_separation_model_dir
        separator = (
            SpeechBrainSepFormerSeparator(separation_model)
            if (separation_model / "hyperparams.yaml").is_file()
            else None
        )
        debug = bool(getattr(args, "debug", False))
        whisper_model_dir = args.whisper_model_dir or (
            settings.voice_transcription_model_dir if debug else None
        )
        if debug and whisper_model_dir is not None and not whisper_model_dir.is_dir():
            raise SystemExit(
                "debug transcription model is missing; run: kinveil voices setup-models "
                "--accept-model-licenses, or pass --whisper-model-dir"
            )
        transcriber = (
            SherpaOnnxWhisperTranscriber(
                whisper_model_dir,
                language=args.transcription_language,
            )
            if whisper_model_dir
            else None
        )
        device = int(args.device) if args.device and args.device.isdigit() else args.device
        try:
            AudioVisualSessionRunner(
                presence,
                voice,
                segmenter,
                overlap_detector,
                separator=separator,
                transcriber=transcriber,
                camera_index=args.camera,
                microphone_device=device,
                on_transcript=(
                    _print_debug_transcript
                    if debug
                    else None
                ),
                face_enrollment_handler=lambda track: _enroll_camera_visitor(presence, track),
                auto_voice_enrollment=True,
                show_transcript_overlay=not debug,
            ).run()
        except KeyboardInterrupt:
            print("\nStopped audio-visual session.")
        return 0

    if args.command == "voices":
        if args.voice_command == "list":
            identities = [
                {
                    "person_id": person.id,
                    "name": person.name,
                    "voice_consented": container.voices.has_consent(person.id),
                    "voice_templates": container.voices.sample_count(person.id),
                }
                for person in container.faces.list_people()
            ]
            print(json.dumps(identities, ensure_ascii=False, indent=2))
            return 0
        if args.voice_command == "setup-separation-model":
            if not args.accept_model_license:
                raise SystemExit(
                    "SpeechBrain sepformer-wham is Apache-2.0 licensed; review "
                    "https://huggingface.co/speechbrain/sepformer-wham and rerun "
                    "with --accept-model-license"
                )
            installed = install_sepformer_model(
                settings.voice_separation_model_dir,
                accept_model_license=True,
            )
            print(f"Installed {installed}")
            return 0
        if args.voice_command == "setup-models":
            if not args.accept_model_licenses:
                print(voice_model_license_summary())
                raise SystemExit("review the licenses, then rerun with --accept-model-licenses")
            for model in install_default_voice_models(
                settings.voice_model_dir,
                accept_model_licenses=True,
            ):
                print(f"Installed {model}")
            transcription_model = install_whisper_model(
                settings.voice_transcription_model_dir,
                accept_model_license=True,
            )
            print(f"Installed {transcription_model}")
            return 0
        if args.voice_command == "revoke":
            if not container.voices.revoke_consent(args.person_id):
                raise SystemExit(f"voice enrollment not found: {args.person_id}")
            return 0
        microphone_allowed = args.allow_microphone_processing or _confirm(
            "Allow Kinveil to use the microphone and analyze speech locally in memory? "
            "Raw audio is not recorded"
        )
        if not microphone_allowed:
            raise SystemExit("microphone access and transient voice processing were not granted")
        enrollment_person = None
        if args.person_id:
            try:
                enrollment_person = container.faces.get_person(args.person_id)
            except KeyError as exc:
                raise SystemExit(str(exc)) from exc
        embedding_model = args.embedding_model or settings.voice_embedding_model
        vad_model = args.vad_model or settings.voice_vad_model
        if not embedding_model.is_file() or not vad_model.is_file():
            raise SystemExit(
                "voice models are missing; run: kinveil voices setup-models --accept-model-licenses"
            )
        backend = SherpaOnnxVoiceBackend(embedding_model)
        engine = VoiceIdentityEngine(backend, container.voices)
        segmenter = SherpaVadSegmenter(vad_model)
        prompted_tracks: set[str] = set()

        def report(snapshot: SpeakerTrackSnapshot, duration: float) -> None:
            confidence = (
                f" match={snapshot.identity_confidence:.2f}"
                if snapshot.persistent
                else " temporary"
            )
            print(
                f"{snapshot.label}: {duration:.1f}s; "
                f"observations={snapshot.observations};{confidence}"
            )
            if (
                enrollment_person is None
                or snapshot.persistent
                or snapshot.observations < 3
                or snapshot.track_id in prompted_tracks
            ):
                return
            prompted_tracks.add(snapshot.track_id)
            if _confirm(
                f"Does {enrollment_person.name} consent to Kinveil storing voice "
                "templates on this device for future recognition?"
            ):
                saved = engine.persist_track(
                    snapshot.track_id,
                    enrollment_person.id,
                    voice_consent=True,
                )
                print(f"Saved voice identity for {saved.label}.")
            else:
                print(f"{snapshot.label} remains temporary for this session.")

        device = int(args.device) if args.device and args.device.isdigit() else args.device
        try:
            MicrophoneVoiceRunner(
                engine,
                segmenter,
                device=device,
                on_result=report,
            ).run()
        except KeyboardInterrupt:
            print("\nStopped listening.")
        return 0

    if args.face_command == "list":
        people = [
            _person_json(person, container.faces.pose_counts(person.id))
            for person in container.faces.list_people()
        ]
        print(json.dumps(people, ensure_ascii=False, indent=2))
        return 0
    if args.face_command == "delete":
        if not container.faces.delete_person(args.person_id):
            raise SystemExit(f"person not found: {args.person_id}")
        return 0
    if args.face_command == "name":
        if not args.confirmed:
            raise SystemExit("refusing to save an unconfirmed conversational name")
        person = container.faces.set_display_name(args.person_id, args.display_name)
        print(
            json.dumps(
                _person_json(person, container.faces.pose_counts(person.id)),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.face_command == "revoke-biometrics":
        if not container.faces.revoke_biometric_consent(args.person_id):
            raise SystemExit(f"person not found: {args.person_id}")
        return 0

    if args.face_command == "watch":
        preapproved = args.allow_camera_processing or (
            args.allow_camera and args.allow_transient_processing
        )
        camera_processing_allowed = preapproved or _confirm(
            "Allow Kinveil to open the camera and analyze faces transiently in memory? "
            "Frames are not recorded"
        )
        if not camera_processing_allowed:
            raise SystemExit("camera access and transient face processing were not granted")
        backend = InsightFaceBackend(settings.face_model, settings.detector_size)
        engine = PresenceEngine(
            backend,
            container.faces,
            PresencePermissions(
                camera_allowed=True,
                transient_processing_allowed=True,
            ),
            identity_threshold=settings.face_threshold,
            identity_margin=settings.face_margin,
        )
        CameraPresenceRunner(
            engine,
            camera_index=args.camera,
            sample_interval_seconds=args.interval,
            enrollment_handler=lambda track: _enroll_camera_visitor(engine, track),
        ).run()
        return 0
    backend = InsightFaceBackend(settings.face_model, settings.detector_size)
    if args.face_command == "enroll-video":
        result = FaceEnrollment(backend, container.faces).enroll_video(
            args.name, args.video, consent=args.consent
        )
        print(
            json.dumps(
                {
                    "person": _person_json(
                        result.person, container.faces.pose_counts(result.person.id)
                    ),
                    "frames_seen": result.frames_seen,
                    "samples_saved": result.samples_saved,
                    "rejected_frames": result.rejected_frames,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    recognizer = FaceRecognizer(container.faces, settings.face_threshold, settings.face_margin)
    matches = recognize_image(args.image, backend, recognizer)
    print(json.dumps([match.__dict__ for match in matches], ensure_ascii=False, indent=2))
    return 0


def _print_debug_transcript(result: ActiveSpeakerResult, transcript: Transcript) -> None:
    detail = f"lip={result.lip_score:.3f}; {result.reason}"
    print(f"[{result.label}] {transcript.text} ({detail})")


def _person_json(person: Person, pose_counts: dict[str, int] | None = None) -> dict[str, object]:
    return {
        "id": person.id,
        "display_name": person.display_name,
        "local_alias": person.local_alias,
        "label": person.name,
        "created_at": person.created_at.isoformat(),
        "sample_count": person.sample_count,
        "pose_counts": pose_counts or {},
        "biometric_consented_at": (
            person.biometric_consented_at.isoformat() if person.biometric_consented_at else None
        ),
    }


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def _enroll_camera_visitor(engine: PresenceEngine, track: TrackSnapshot) -> str | None:
    person = engine.persist_track(
        track.track_id,
        biometric_consent=True,
    )
    counts = engine.store.pose_counts(person.id)
    summary = ", ".join(f"{pose}:{count}" for pose, count in sorted(counts.items()))
    print(f"Saved {person.name} ({person.id}); templates: {summary}")
    return f"Saved {person.name}; keep moving slowly to improve recognition"


if __name__ == "__main__":
    raise SystemExit(main())
