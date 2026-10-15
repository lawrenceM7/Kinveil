#!/usr/bin/env bash

set -euo pipefail

# Download the models used by Kinveil's default camera + microphone session.
# Existing model files are skipped. Default transcription files are pinned to
# one upstream revision and verified before installation.

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
readonly PARAFORMER_MODEL_DIR="${VOICE_MODEL_DIR}/sherpa-onnx-paraformer-zh-2024-03-09"
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

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

download_verified_file() {
  local url="$1"
  local destination="$2"
  local expected_sha256="$3"
  local label="$4"

  if [[ -f "${destination}" ]] && [[ "$(sha256_file "${destination}")" == "${expected_sha256}" ]]; then
    echo "[skip] ${label}: ${destination}"
    return
  fi

  mkdir -p "$(dirname "${destination}")"
  echo "[download] ${label}"
  curl -L --fail --show-error --progress-bar \
    --output "${destination}.download" \
    "${url}"
  local actual_sha256
  actual_sha256="$(sha256_file "${destination}.download")"
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    rm -f -- "${destination}.download"
    echo "[error] ${label}: SHA-256 mismatch" >&2
    return 1
  fi
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

readonly PARAFORMER_REVISION="906992d326ebf0c5171cde675aa0902be9e5bc6c"
readonly PARAFORMER_BASE_URL="https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09/resolve/${PARAFORMER_REVISION}"

download_verified_file \
  "${PARAFORMER_BASE_URL}/model.int8.onnx" \
  "${PARAFORMER_MODEL_DIR}/model.int8.onnx" \
  "90bc03034ae1bef9575f8cc798cd1519c8be8aa9e8b458a033e32017ff4d584c" \
  "Paraformer Chinese/English INT8 model"

download_verified_file \
  "${PARAFORMER_BASE_URL}/tokens.txt" \
  "${PARAFORMER_MODEL_DIR}/tokens.txt" \
  "6c0e3b35cece259829e6cb5b8d90d13db88f61ea3a2953d11898e4b2bfd7a2e2" \
  "Paraformer Chinese/English tokens"

echo
echo "All default visual and voice models are present."
echo "Run Kinveil with: kinveil start --debug"
