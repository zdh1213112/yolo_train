#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

: "${YOLO_EXPORT_FORMAT:=onnx}"
: "${YOLO_EXPORT_IMGSZ:=1024}"
: "${YOLO_EXPORT_DEVICE:=cpu}"
: "${YOLO_WEIGHTS:=}"

WEIGHTS_PATH=""
LATEST_WEIGHTS="$(find "${ROOT_DIR}/outputs/train" -maxdepth 3 -type f -name 'best.pt' | sort | tail -n 1 || true)"
if [[ -n "${LATEST_WEIGHTS}" && -f "${LATEST_WEIGHTS}" ]]; then
  WEIGHTS_PATH="${LATEST_WEIGHTS}"
fi

if [[ -z "${WEIGHTS_PATH}" ]]; then
  if [[ -n "${YOLO_WEIGHTS}" ]]; then
    CANDIDATE_PATH="${ROOT_DIR}/${YOLO_WEIGHTS#./}"
    if [[ -f "${CANDIDATE_PATH}" ]]; then
      WEIGHTS_PATH="${CANDIDATE_PATH}"
    fi
  fi
fi

if [[ -z "${WEIGHTS_PATH}" ]]; then
  echo "No export weights found. Train a model first or set YOLO_WEIGHTS in .env." >&2
  exit 1
fi

yolo mode=export \
  model="${WEIGHTS_PATH}" \
  format="${YOLO_EXPORT_FORMAT}" \
  imgsz="${YOLO_EXPORT_IMGSZ}" \
  device="${YOLO_EXPORT_DEVICE}"
