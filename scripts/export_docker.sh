#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${YOLO_ENV_FILE:-${ROOT_DIR}/.env}"
ULTRALYTICS_CONFIG_DIR="${ROOT_DIR}/.cache/ultralytics"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

cd "${ROOT_DIR}"
mkdir -p "${ULTRALYTICS_CONFIG_DIR}"

if ! docker image inspect yolo-easy-trainer:latest >/dev/null 2>&1; then
  if [[ -f "${ENV_FILE}" ]]; then
    docker compose --env-file "${ENV_FILE}" build trainer
  else
    docker compose build trainer
  fi
fi

if [[ -f "${ENV_FILE}" ]]; then
  docker compose --env-file "${ENV_FILE}" run --rm trainer ./scripts/export.sh
else
  docker compose run --rm trainer ./scripts/export.sh
fi
