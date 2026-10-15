# Local audio models and disk budget

Kinveil keeps optional audio capabilities separate so users do not have to
download every model. The sizes below are approximate extracted sizes and may
change slightly across filesystem formats.

| Capability | Default model | Runtime files used | Approximate size |
| --- | --- | ---: | ---: |
| Voice activity | Silero VAD | `silero-vad.onnx` | 0.6 MB |
| Voice identity | 3D-Speaker CAM++ | one ONNX file | 28.3 MB |
| Overlap detection | Pyannote Segmentation 3.0 | one ONNX file | 5.7 MB |
| Two-speaker separation | SpeechBrain SepFormer WHAM | inference checkpoints | 113 MB |
| Speech-to-text | Paraformer Chinese/English INT8 | model and tokens | 217 MB |
| Multilingual STT fallback | Whisper Small multilingual INT8 | encoder, decoder, tokens | 375 MB |
| Text-to-speech | Kokoro multi-lang v1.0 | model, voices, lexicons | about 350 MB |

The default setup downloads Paraformer's INT8 model and token file instead of
Whisper. Both files are pinned to one upstream revision and checked against
fixed SHA-256 hashes. Existing Whisper installations remain supported through
`--whisper-model-dir` for languages outside Chinese and English.

SepFormer additionally needs SpeechBrain, PyTorch, and torchaudio. Their
installed footprint depends on platform and what is already present; allow
roughly another 0.5–1.5 GB. The SpeechBrain downloader can also leave a second
copy in the Hugging Face cache.

## Text-to-speech reserve

The `SherpaOnnxKokoroTTS` adapter accepts a local Kokoro Chinese/English model
directory and produces mono PCM16 WAV bytes. No generated audio is persisted by
the adapter itself. The CLI can explicitly save one result:

```bash
kinveil tts synthesize "你好，我是 Kinveil。" \
  --model-dir /path/to/kokoro-multi-lang-v1_0 \
  --voice-id 48 \
  --output ./reply.wav
```

Kokoro v1.0 contains 53 speaker styles. IDs 45–48 are Chinese female styles
and IDs 49–52 are Chinese male styles. English and mixed Chinese-English input
are supported. The default Kinveil voice ID is 48.

The future conversation output path should be:

```text
LLM response text
  -> sentence queue
  -> Kokoro synthesis worker
  -> short PCM chunks
  -> speaker playback
```

Playback must run outside microphone capture and should signal an
`assistant-is-speaking` state. The input pipeline can then suppress echo or temporarily pause
speaker attribution, preventing Kinveil from recognizing its own synthesized
voice as a nearby person.

## Primary runtime command

`kinveil start` is the combined production entry point for camera and microphone
processing. `kinveil start --debug` additionally loads the local Paraformer model
and prints each accepted or rejected speaker attribution with its transcript to
the console. Debug transcription remains local and is not persisted.

The live Silero VAD accepts speech activity from 250 ms upward so ordinary short
phrases are still passed to the recognizer, and keeps pauses shorter than 800 ms inside
one utterance to reduce truncated sentences. This only controls audio
segmentation; it does not relax face, lip-motion, or voice-identity agreement
rules.
