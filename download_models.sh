#!/usr/bin/env bash

set -euo pipefail

# Download the models used by Kinveil's default camera + microphone session.
# Existing model files are skipped. This script intentionally does not verify
# checksums or load the models after downloading them.

readonly INSIGHTFACE_MODEL_DIR="${HOME}/.insightface/models/buffalo_l"
readonly INSIGHTFACE_MODEL_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

if [[ -n "${KINVEIL_DATA_DIR:-}" ]]; then
  KINVEIL_MODEL_ROOT="${KINVEIL_DATA_DIR}/models"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  KINVEIL_MODEL_ROOT="${HOME}/Library/Application Support/Kinveil/models"
else
  KINVEIL_MODEL_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/Kinveil/models"
fi

readonly KINVEIL_MODEL_ROOT
readonly VOICE_MODEL_DIR="${KINVEIL_MODEL_ROOT}/voice"
readonly WHISPER_MODEL_DIR="${VOICE_MODEL_DIR}/sherpa-onnx-whisper-small"
readonly TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kinveil-models.XXXXXX")"

cleanup() {
  rm -r -- "${TEMP_DIR}"
}
trap cleanup EXIT

download_file() {
  local url="$1"
  local destination="$2"
  local label="$3"

  if [[ -f "${destination}" ]]; then
    echo "[skip] ${label}: ${destination}"
    return
  fi

  mkdir -p "$(dirname "${destination}")"
  echo "[download] ${label}"
  curl -L --fail --show-error --progress-bar \
    --output "${destination}.download" \
    "${url}"
  mv "${destination}.download" "${destination}"
  echo "[ready] ${destination}"
}

download_insightface() {
  local required_files=(
    "1k3d68.onnx"
    "2d106det.onnx"
    "det_10g.onnx"
    "genderage.onnx"
    "w600k_r50.onnx"
  )
  local filename
  local complete=true

  for filename in "${required_files[@]}"; do
    if [[ ! -f "${INSIGHTFACE_MODEL_DIR}/${filename}" ]]; then
      complete=false
      break
    fi
  done

  if [[ "${complete}" == true ]]; then
    echo "[skip] InsightFace buffalo_l: ${INSIGHTFACE_MODEL_DIR}"
    return
  fi

  echo "[download] InsightFace buffalo_l"
  curl -L --fail --show-error --progress-bar \
    --output "${TEMP_DIR}/buffalo_l.zip" \
    "${INSIGHTFACE_MODEL_URL}"
  mkdir -p "${INSIGHTFACE_MODEL_DIR}"
  unzip -o -q "${TEMP_DIR}/buffalo_l.zip" -d "${INSIGHTFACE_MODEL_DIR}"
  echo "[ready] ${INSIGHTFACE_MODEL_DIR}"
}

download_pyannote() {
  local destination="${VOICE_MODEL_DIR}/pyannote-segmentation-3.0.onnx"
  local archive="${TEMP_DIR}/pyannote-segmentation.tar.bz2"
  local extracted="${TEMP_DIR}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"

  if [[ -f "${destination}" ]]; then
    echo "[skip] Pyannote Segmentation 3.0: ${destination}"
    return
  fi

  echo "[download] Pyannote Segmentation 3.0"
  curl -L --fail --show-error --progress-bar \
    --output "${archive}" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  tar -xjf "${archive}" -C "${TEMP_DIR}"
  mkdir -p "${VOICE_MODEL_DIR}"
  cp "${extracted}" "${destination}"
  echo "[ready] ${destination}"
}

echo "Kinveil model directory: ${KINVEIL_MODEL_ROOT}"
echo

download_insightface

download_file \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx" \
  "${VOICE_MODEL_DIR}/3dspeaker-campplus-zh-cn-16k-common.onnx" \
  "3D-Speaker CAM++"

download_file \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" \
  "${VOICE_MODEL_DIR}/silero-vad.onnx" \
  "Silero VAD"

download_pyannote

readonly WHISPER_REVISION="8f3c18b358db4d1f2fc1eae49d75cd20989e4309"
readonly WHISPER_BASE_URL="https://huggingface.co/csukuangfj/sherpa-onnx-whisper-small/resolve/${WHISPER_REVISION}"

download_file \
  "${WHISPER_BASE_URL}/small-encoder.int8.onnx" \
  "${WHISPER_MODEL_DIR}/small-encoder.int8.onnx" \
  "Whisper Small encoder"

download_file \
  "${WHISPER_BASE_URL}/small-decoder.int8.onnx" \
  "${WHISPER_MODEL_DIR}/small-decoder.int8.onnx" \
  "Whisper Small decoder"

download_file \
  "${WHISPER_BASE_URL}/small-tokens.txt" \
  "${WHISPER_MODEL_DIR}/small-tokens.txt" \
  "Whisper Small tokens"

echo
echo "All default visual and voice models are present."
echo "Run Kinveil with: kinveil start --debug"
