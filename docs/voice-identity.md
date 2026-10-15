# Voice identity design

## Goal

Voice identity should work during ordinary conversation rather than requiring a
special enrollment recording. An upstream audio pipeline detects speech and
separates overlapping speakers. Kinveil converts each clean, single-speaker
segment into an embedding and uses it to maintain a temporary speaker track.

```text
microphone
   -> voice activity detection
   -> speaker diarization / one-speaker segment
   -> speaker embedding
   -> known person or temporary Speaker A/B
```

## Consent and persistence

An unknown speaker starts as a bounded in-memory track. Its samples disappear
when the track expires or the program stops. A track may be bound to an existing
person only after voice-biometric storage is allowed. The combined session asks
once at startup for camera/microphone processing and bounded voice-template
storage; face enrollment by itself does not silently grant voice consent.

After consent, normalized voice embeddings are stored in the same local SQLite
identity database as the person's face identity. Voice consent has its own row
and can be revoked without deleting the person or their face templates. Deleting
the person deletes both kinds of template.

The persistent pool is capped at 24 voice embeddings per person. Near-duplicate
samples are ignored unless the new sample is better quality; when full, weaker
samples are pruned. Raw audio is never stored by this layer.

## Recognition behavior

Recognition uses cosine similarity against consented templates. It rejects a
segment when the best score is below threshold or too close to another person's
score. A high-confidence known speaker can contribute a new high-quality sample
to the bounded pool, allowing natural, gradual improvement across sessions.

Voice identity is a personalization signal, not authentication. Illness,
microphone changes, noise, replayed speech, and voice cloning can all affect or
spoof it. Sensitive memory writes should require stronger multimodal evidence or
confirmation.

## Live prototype

Install the optional dependencies and reviewed, hash-pinned models:

```bash
python -m pip install -e '.[voice]'
kinveil voices setup-models --accept-model-licenses
```

The default speaker model is 3D-Speaker CAM++ for Chinese/common 16 kHz speech
(Apache-2.0 project license). Voice activity detection uses Silero VAD (MIT).
Pyannote Segmentation 3.0 (MIT) detects overlapping speech. All models run
locally, and the setup command verifies fixed archive and model SHA-256 hashes
before making them available to Kinveil.

Start transient recognition with `kinveil voices watch`. Add
`--person-id PERSON_ID` when the intended enrolling person has already been
identified. After three consistent speech segments, Kinveil asks for separate
voice-biometric consent and only then saves embeddings.

The voice-only prototype handles turn-taking—one active speaker in each VAD
segment. The combined audio-visual session can additionally use an optional
SpeechBrain SepFormer model to separate at most two overlapping speakers.

## Audio-visual active speaker session

`kinveil session watch` runs camera presence, microphone VAD, voice identity,
and mouth-region motion analysis on one monotonic timeline. A single visible
person is not accepted merely because they are alone: their mouth must move
during the detected speech interval.

The resolver follows conservative rules:

- acoustic two-speaker overlap is separated into at most two source streams;
- each separated stream is embedded and matched independently;
- separated streams may be recognized but never update stored voice templates;
- matching persistent face and voice identities are accepted;
- a persistent face plus a transient voice may identify the current utterance,
  and can enroll it after three consistent multimodal observations when startup
  voice-template storage was allowed;
- conflicting persistent face and voice IDs are rejected;
- a known separated voice can select the one moving face with the same identity;
- transient voices with multiple moving mouths remain unknown;
- an invisible mouth or insufficient synchronized motion is rejected.

Rejected cases display `Unknown speaker`. The enrollment rule is generic: it
uses stable person IDs, never display names, and requires a unique persistent
face, synchronized mouth motion, a consistent transient voice track, and at
least two samples above the quality threshold. Raw audio is never stored.

Silero VAD keeps pauses shorter than 800 ms inside one utterance and accepts
speech activity from 250 ms upward. Its original sample offsets are preserved
on the monotonic camera timeline instead of estimating timestamps after model
inference, which makes lip synchronization less sensitive to processing delay.

The acoustic overlap gate interprets Pyannote's powerset output frame by frame.
A segment is separated after at least 150 ms of detected overlap comprising at
least 5% of its speech frames. If SepFormer is unavailable or separation leaves
an ambiguous identity, the result is `Unknown speaker`.

Install separation independently because SpeechBrain, PyTorch, and torchaudio
are substantially larger than the normal voice runtime:

```bash
python -m pip install -e '.[separation]'
kinveil voices setup-separation-model --accept-model-license
```

The model is stored under Kinveil's platform data directory in
`models/voice/sepformer-wham`. It is Apache-2.0 licensed, runs locally after the
one-time download, expects 8 kHz audio internally, and produces two candidate
source streams.

## Local speech-to-text

Kinveil can optionally transcribe every clean or separated source using a local
sherpa-onnx Whisper model. Pass an extracted multilingual model directory to
`kinveil session watch --whisper-model-dir PATH`. Each transcript is attached to
the corresponding resolved speaker for display, so the UI can show who said
what. The current layer keeps transcripts in memory only; it does not yet write
conversation memories.
