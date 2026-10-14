# Kinveil

> One AI. Many relationships. Private by design.

Kinveil is a local-first, face-aware AI assistant for shared spaces. It recognizes enrolled people, keeps identity-scoped memories separate, and provides modular speech, language-model, permission, and tool runtimes.

The project is in its first development stage. The application architecture is scaffolded and the face identity module is the first implemented vertical slice.

## Why face embeddings instead of training a new CNN

Kinveil does **not** mix enrollment videos with a public face dataset and retrain a classifier for every person. It uses a pretrained face-recognition network to produce embeddings, then stores several high-quality, varied embeddings for each enrolled person.

This design:

- enrolls a new person without retraining;
- works with a short consented video;
- avoids downloading a large biometric dataset;
- supports incremental deletion and re-enrollment;
- separates model training from private identity data;
- can reject unknown or ambiguous faces.

The current production backend adapter targets InsightFace through ONNX Runtime. The core pipeline is backend-independent and tested with deterministic fake embeddings.

## Architecture

```text
Camera / microphone
        |
        v
Perception
  face identity ---- emotion signals
        |                  |
        +--------+---------+
                 v
         Session identity
                 |
                 v
          Memory firewall
                 |
                 v
     Intent -> LLM -> permissions -> tools
                 |
                 v
            Text to speech
```

```text
src/kinveil/
├── app/           application composition
├── audio/         speech-to-text and text-to-speech contracts
├── emotion/       multimodal emotion-signal contracts
├── intent/        intent classification contracts
├── llm/           provider-neutral LLM contracts
├── memory/        identity-scoped memory contracts
├── permissions/   action authorization contracts
├── tools/         typed tool registry
└── vision/face/   enrollment, storage, and recognition
```

## Development setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

Install the optional face backend:

```bash
python -m pip install -e '.[face]'
```

InsightFace downloads model weights on first use. Review its model licensing before commercial distribution.

## Face enrollment

Record a short video while slowly turning left, right, up, and down. Good lighting and a single face in frame produce the best enrollment.

```bash
kinveil faces enroll-video \
  --name Alice \
  --video ./alice-enrollment.mp4 \
  --consent
```

List enrolled identities:

```bash
kinveil faces list
```

Recognize faces from one image:

```bash
kinveil faces recognize-image ./photo.jpg
```

Delete an identity and all associated embeddings:

```bash
kinveil faces delete PERSON_ID
```

By default the identity database is stored under the platform data directory. Override it with `KINVEIL_DATA_DIR`.

## Privacy and security status

- Enrollment requires an explicit `--consent` acknowledgement.
- Enrollment videos are read in place and are not copied into Kinveil storage.
- The store contains biometric embeddings and must be treated as sensitive data.
- Database encryption and liveness detection are not implemented yet.
- Face identity is for personalization, not authentication or access control.
- Unknown and ambiguous matches are rejected rather than guessed.

See [docs/face-recognition.md](docs/face-recognition.md) for the design and limitations.
See [docs/product.md](docs/product.md) for the anonymous-first identity,
consent, presence, and memory behavior.

## Continuous presence prototype

Kinveil can keep a camera preview open and periodically resolve every visible
face. Unknown visitors receive temporary in-memory labels and are not enrolled:

    kinveil faces watch

Kinveil uses one startup prompt that clearly covers both camera access and
transient in-memory face analysis before initializing the model or camera.
Press q or Escape to stop. Automated development sessions can acknowledge this
prompt with `--allow-camera-processing`; the two older allow flags remain
compatible when supplied together.

An unknown visitor is tracked automatically in memory. After the same visitor
has been observed in several samples, the camera window asks whether Kinveil may
recognize that visitor on future visits. Press `y` to grant persistent biometric
storage or `n` to keep the visitor temporary for that session; no enrollment
hotkey or terminal input is required. A consented visitor initially receives an
anonymous local alias and can be named later. After enrollment, Kinveil
continuously refreshes only sufficiently confident, high-quality templates. It
stores at most 24 templates per person and at most 8 per coarse pose bucket
(`front`, `left`, `right`, `up`, `down`, or `unknown`), pruning redundant
samples instead of growing the database without limit. Raw camera frames are
not stored.

A persistent identity may initially have no name. Once a conversationally
discovered name has been confirmed, save it independently of the face identity:

    kinveil faces name PERSON_ID Alice --confirmed

Biometric consent and all stored face templates can be revoked without silently
deleting unrelated memories:

    kinveil faces revoke-biometrics PERSON_ID

## Roadmap

1. Face enrollment and recognition.
2. Live camera tracking and explicit active-person sessions.
3. Local speech recognition and synthesis.
4. Identity-scoped memories with private/shared visibility.
5. LLM and intent routing.
6. Permission-gated skills and tools.
7. Original point-cloud face UI.

## License

No license has been selected yet. Do not assume redistribution rights until a license file is added.
