#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash ./scripts/use_local_model.sh /path/to/yolo26n-obb.pt" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_PATH="$1"
TARGET_DIR="${ROOT_DIR}/models"
TARGET_PATH="${TARGET_DIR}/$(basename "${SOURCE_PATH}")"

if [[ ! -f "${SOURCE_PATH}" ]]; then
  echo "Model file not found: ${SOURCE_PATH}" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"
cp -f "${SOURCE_PATH}" "${TARGET_PATH}"

echo "Model copied to: ${TARGET_PATH}"
echo "Set YOLO_MODEL=models/$(basename "${SOURCE_PATH}") in .env before training."
