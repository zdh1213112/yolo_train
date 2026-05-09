#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

: "${YOLO_MODEL:=yolo26n-obb.pt}"
: "${YOLO_MODEL_URL:=https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.pt}"
: "${YOLO_MODEL_URL_MIRROR:=}"

MODEL_TARGET="${YOLO_MODEL#./}"
if [[ "${MODEL_TARGET}" != */* ]]; then
  MODEL_TARGET="models/${MODEL_TARGET}"
fi

DOWNLOAD_URL="${YOLO_MODEL_URL}"
if [[ -n "${YOLO_MODEL_URL_MIRROR}" ]]; then
  DOWNLOAD_URL="${YOLO_MODEL_URL_MIRROR}"
fi

MODEL_PATH="${ROOT_DIR}/${MODEL_TARGET}"
MODEL_DIR="$(dirname "${MODEL_PATH}")"

mkdir -p "${MODEL_DIR}"

if command -v curl >/dev/null 2>&1; then
  curl --location --fail --continue-at - --output "${MODEL_PATH}" "${DOWNLOAD_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget --continue --output-document="${MODEL_PATH}" "${DOWNLOAD_URL}"
else
  echo "Neither curl nor wget is available." >&2
  exit 1
fi

if [[ ! -s "${MODEL_PATH}" ]]; then
  echo "Downloaded file is empty: ${MODEL_PATH}" >&2
  exit 1
fi

echo "Model downloaded to: ${MODEL_PATH}"
echo "Set YOLO_MODEL=${MODEL_TARGET} in .env if you want training to always use this local file."
