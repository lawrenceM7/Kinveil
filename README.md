# Kinveil

> One AI. Many relationships. Private by design.

Kinveil is a local-first, face- and voice-aware AI assistant for shared spaces. It recognizes enrolled people, keeps identity-scoped memories separate, and provides modular speech, language-model, permission, and tool runtimes.

See [MODELS.md](MODELS.md) for the complete model manifest, download methods, disk usage, storage locations, and license notes.

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
├── audio/         speech, voice identity, and text-to-speech contracts
├── emotion/       multimodal emotion-signal contracts
├── intent/        intent classification contracts
├── llm/           provider-neutral LLM contracts
├── memory/        identity-scoped memory contracts
├── permissions/   action authorization contracts
├── tools/         typed tool registry
└── vision/face/   enrollment, storage, and recognition
```

See [docs/memory.md](docs/memory.md) for the identity-scoped background memory,
consent, review, and retrieval design.

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

## Voice identity foundation

Kinveil can build temporary `Speaker A`, `Speaker B`, ... tracks from naturally
spoken, single-speaker audio segments. Nothing has to be recorded in advance.
Before consent, those tracks and embeddings live only in memory. In the combined
session, one startup permission covers local sensing and bounded voice-template
storage for people who were already approved for persistent recognition. After
three consistent face, lip-motion, and voice observations, the generic runtime
can bind the voice track to that person's stable ID. Future speech can then
resolve to the same person after a restart, and high-quality embeddings can
refresh a bounded pool of at most 24 templates.

The voice templates share the local identity database with face templates, but
voice consent and deletion are independent. Raw audio is never written by the
voice identity layer.

Install live voice dependencies and the pinned local models:

```bash
python -m pip install -e '.[voice]'
kinveil voices setup-models --accept-model-licenses
```

The setup command prints the third-party model licenses unless the acceptance
flag is present. It downloads an Apache-2.0 3D-Speaker CAM++ embedding model,
the MIT-licensed Silero VAD and Pyannote Segmentation models, and the MIT-licensed
Paraformer Chinese/English INT8 files used by debug transcription. Every archive
and installed model file is checked against a fixed SHA-256 hash.

Listen without persisting unknown speakers:

```bash
kinveil voices watch
```

Inspect which people currently have persistent voice templates:

```bash
kinveil voices list
```

To offer enrollment for one existing person after their voice has appeared in
three speech segments, pass their stable ID:

```bash
kinveil voices watch --person-id PERSON_ID
```

Kinveil then asks for voice-biometric consent once. Install the optional
two-speaker separation backend and its Apache-2.0 model separately:

```bash
python -m pip install -e '.[separation]'
kinveil voices setup-separation-model --accept-model-license
```

See [docs/voice-identity.md](docs/voice-identity.md).

Run the combined camera and microphone session:

```bash
kinveil start
```

`kinveil start` is the primary application entry point. It opens the camera and
microphone together and runs face presence, voice identity, lip synchronization,
and acoustic overlap detection in one session. The older
`kinveil session watch` command remains available as a compatibility alias.

To additionally transcribe recognized speech and print speaker-attributed text
to the terminal, use:

```bash
kinveil start --debug
```

Debug lines include the resolved identity, transcript, lip-motion score, and
decision reason. The regular
`kinveil voices setup-models --accept-model-licenses` command installs the
default Paraformer model at
`models/voice/sherpa-onnx-paraformer-zh-2024-03-09`; pass
`--paraformer-model-dir PATH` to use another directory. Existing Whisper models
remain available through `--whisper-model-dir PATH`. Normal `kinveil start`
does not load an ASR model or print transcripts.

This session attributes speech only when visible mouth motion overlaps the
detected speech interval. A known voice must agree with the recognized face.
Conflicting identities, no visible mouth, or insufficient motion produce
`Unknown speaker`. When multiple mouths move, a separated persistent voice is
accepted only if exactly one moving face has the same stable identity. A known
face with a new transient voice can be identified for the current utterance.
When startup voice-template storage was allowed, three consistent multimodal
observations automatically enroll at least two high-quality voice embeddings;
the rule is identical for every enrolled person and contains no name-specific
behavior.

Before extracting a voice embedding, the combined session runs acoustic
overlapped-speech detection. Pyannote's seven-class powerset output distinguishes
silence, one active speaker, and two active speakers per time frame. At least
150 ms of overlap and 5% of speech frames must agree before SepFormer separates
the segment into at most two sources. Each source is recognized independently
and checked against synchronized face identity and mouth motion. Separated audio
is never allowed to refresh a persistent voice-template pool, preventing model
artifacts from contaminating long-term identity data. Without the optional
separator, overlap safely falls back to `Unknown speaker`.

Optional local speech-to-text can run after identity attribution. Download and
extract a sherpa-onnx multilingual Whisper model, then pass its directory:

```bash
kinveil session watch --whisper-model-dir /path/to/sherpa-onnx-whisper-small
```

The int8 Whisper Small encoder, decoder, and token file use roughly 375 MB.
Transcripts remain in memory and appear as `Person: text`; this layer does not
write raw audio or transcripts to disk.

Kinveil also includes a reserved local text-to-speech backend for the
Chinese/English Kokoro model through sherpa-onnx. Once a model directory is
available, generate a WAV file with:

```bash
kinveil tts synthesize "你好，我是 Kinveil。" \
  --model-dir /path/to/kokoro-multi-lang-v1_0 \
  --voice-id 48 \
  --output ./reply.wav
```

See [docs/audio-models.md](docs/audio-models.md) for model sizes, storage
tradeoffs, and the planned real-time playback path.

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
3. Live voice identity, local speech recognition, and synthesis.
4. Identity-scoped memories with private/shared visibility.
5. LLM and intent routing.
6. Permission-gated skills and tools.
7. Original point-cloud face UI.

## License

No license has been selected yet. Do not assume redistribution rights until a license file is added.
