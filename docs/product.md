# Kinveil product specification

## Product statement

Kinveil is a local-first ambient AI that can distinguish the people sharing a
computer, keep their memories separate, and gradually learn how each person
wants to be addressed. It should feel like meeting someone, not configuring an
account.

The first useful loop is:

1. Ask permission to use the camera.
2. Notice the people currently present.
3. Give unknown people temporary labels such as `Visitor A`.
4. Recognize a returning, consented person without requiring a name first.
5. Learn a display name during conversation and confirm it before saving.
6. Attach memories only when identity confidence is sufficient.

## Product principles

### Anonymous first

A face and a name are different facts. Kinveil creates a stable random person ID
before it knows a display name. Naming or renaming a person never changes their
identity or moves their memories.

### Uncertainty is a first-class state

Kinveil may answer `unknown` or `ambiguous`. It must not guess merely to make
the interface look confident. An uncertain utterance stays in the current
session or a review queue instead of being written into somebody's memory.

### Permission is progressive

Camera access, biometric processing/storage, and memory storage remain distinct
purposes. The combined local session can explain camera/microphone processing
and bounded voice-template storage in one startup permission, while an unknown
visitor still requires approval before becoming a persistent identity.

### Local by default

Camera frames are processed in memory. The face subsystem stores embeddings,
consent metadata, and model versions—not raw frames or recordings. The voice
identity subsystem likewise stores only bounded embeddings after explicit
voice-biometric consent, never raw audio. Any future cloud feature must be
separately disclosed and optional.

## Identity lifecycle

```text
camera face
    |
    v
session track (Track-1)
    |
    +-- matches consented identity --> Person ID --> optional display name
    |
    +-- no match --> Visitor A (memory only)
                         |
                         +-- explicit biometric consent
                                  |
                                  v
                         anonymous Person ID
                                  |
                                  +-- confirmed conversational name
                                           |
                                           v
                                      named profile
```

### Session track

A track exists only while Kinveil is running. It holds a small, bounded set of
face embeddings in memory so that an unnamed visitor remains `Visitor A` while
moving around. It is discarded after expiry or application shutdown.

### Persistent person

A persistent person has a random UUID and explicit biometric consent. Their
`display_name` may be empty. The UI uses a non-biometric local alias such as
`Person A7F2` until a name is confirmed.

### Naming

An LLM may propose a name from conversation, but cannot silently commit it. The
application asks for confirmation, then updates only `display_name`. Names need
not be unique; the UUID remains authoritative.

## Presence pipeline

The camera may provide 30 frames per second, but Kinveil does not run full face
recognition on every frame.

- UI preview and lightweight tracking: camera frame rate.
- Face detection and identity resolution: configurable, initially 2 Hz.
- New identity or active speaker: temporarily increase sampling.
- Template storage: only novel, high-quality samples and only with consent.

For every sampled frame Kinveil detects all faces, filters weak detections,
extracts normalized embeddings, resolves consented identities, associates
unmatched faces with in-memory tracks, and expires stale tracks.

## Multiple people and speech

Presence and speaker identity are separate signals. A later audio-visual speaker
resolver combines voice activity detection, consented voice embeddings, visible
people, acoustic overlap detection, and lip-motion correlation. Conflicting
evidence produces an unknown speaker, not a forced attribution. With the
optional separator, at most two overlapping sources are processed independently.

Natural speech can create temporary in-memory tracks such as `Speaker A`
without advance enrollment. When bounded voice-template storage was allowed at
startup, three consistent observations of a unique persistent face, synchronized
mouth motion, and the same voice track can bind high-quality samples to that
stable person ID. Face presence by itself is never sufficient.

## Memory firewall

Memories are addressed by stable person UUID, never by a display name or current
track label.

| Identity confidence | Allowed memory behavior |
| --- | --- |
| High | Write ordinary memories to the confirmed person |
| Medium | Hold in a review queue or current session |
| Low/ambiguous | Do not attach to a person |

Sensitive memories require a higher threshold and may require confirmation even
when face identity is confident.

## Consent model

The desktop UI will expose distinct states:

- `camera_allowed`: the application may open the selected camera.
- `transient_processing_allowed`: frames may be analyzed in memory.
- `biometric_storage_allowed`: templates may be persisted for this person.
- `memory_storage_allowed`: conversational memories may be persisted.

Revoking biometric consent deletes stored face templates. Revoking memory
consent is handled separately so the user can make an explicit retention choice.

## Desktop experience

On first launch Kinveil explains the camera indicator and requests access. The
main window always shows whether the camera is active and provides an immediate
pause control. Unknown visitors remain ephemeral. When a visitor wants Kinveil
to recognize them later, the UI asks for biometric consent and guides them
through front, left, right, up, and down views without recording a video file.

The experience may resemble Face ID enrollment, but Kinveil must not claim Face
ID-level security: ordinary RGB webcams generally lack dedicated depth and
infrared anti-spoofing hardware.

## First implementation milestone

This repository milestone provides nullable names, stable anonymous person IDs,
explicit consent metadata, temporary multi-face and speaker tracks, recognition
of persistent identities, consented promotion of tracks, later naming, an
always-on camera preview command, and a provider-neutral voice identity core.

It does not yet provide liveness detection, encrypted biometric storage,
production-grade object tracking, separation of overlapping speakers into
independent audio streams, or a polished desktop UI.
