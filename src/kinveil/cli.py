from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinveil.app import AppContainer
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
    watch = face_commands.add_parser(
        "watch", help="continuously notice people through the camera"
    )
    watch.add_argument("--camera", type=int, default=0)
    watch.add_argument("--interval", type=float, default=0.5)
    watch.add_argument(
        "--allow-camera-processing",
        action="store_true",
        help="allow camera access and transient in-memory face processing",
    )
    watch.add_argument("--allow-camera", action="store_true")
    watch.add_argument("--allow-transient-processing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from kinveil.core import Settings

    settings = Settings.default()
    if args.data_dir:
        settings = Settings(data_dir=args.data_dir)
    container = AppContainer.build(settings)

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
        print(json.dumps({
            "person": _person_json(
                result.person, container.faces.pose_counts(result.person.id)
            ), "frames_seen": result.frames_seen,
            "samples_saved": result.samples_saved, "rejected_frames": result.rejected_frames,
        }, ensure_ascii=False, indent=2))
        return 0
    recognizer = FaceRecognizer(container.faces, settings.face_threshold, settings.face_margin)
    matches = recognize_image(args.image, backend, recognizer)
    print(json.dumps([match.__dict__ for match in matches], ensure_ascii=False, indent=2))
    return 0


def _person_json(
    person: Person, pose_counts: dict[str, int] | None = None
) -> dict[str, object]:
    return {
        "id": person.id,
        "display_name": person.display_name,
        "local_alias": person.local_alias,
        "label": person.name,
        "created_at": person.created_at.isoformat(),
        "sample_count": person.sample_count,
        "pose_counts": pose_counts or {},
        "biometric_consented_at": (
            person.biometric_consented_at.isoformat()
            if person.biometric_consented_at
            else None
        ),
    }


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def _enroll_camera_visitor(
    engine: PresenceEngine, track: TrackSnapshot
) -> str | None:
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
