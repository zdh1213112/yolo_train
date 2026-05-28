#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${YOLO_ENV_FILE:-${ROOT_DIR}/.env}"
CONTAINER_ROOT="/app"
CONTAINER_ENV_FILE="${CONTAINER_ROOT}/.env"
ULTRALYTICS_CONFIG_DIR="${ROOT_DIR}/.cache/ultralytics"
CONTAINER_ULTRALYTICS_CONFIG_DIR="${CONTAINER_ROOT}/.cache/ultralytics"

if [[ "${ENV_FILE}" == "${ROOT_DIR}"/* ]]; then
  CONTAINER_ENV_FILE="${CONTAINER_ROOT}/${ENV_FILE#${ROOT_DIR}/}"
fi

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

DOCKER_RUN_ARGS=(--rm)

if [[ -n "${DOCKER_GPUS:-}" ]]; then
  DOCKER_RUN_ARGS+=(--gpus "${DOCKER_GPUS}")
fi

if [[ -n "${DOCKER_SHM_SIZE:-}" ]]; then
  DOCKER_RUN_ARGS+=(--shm-size "${DOCKER_SHM_SIZE}")
fi

if [[ -f "${ENV_FILE}" ]]; then
  DOCKER_RUN_ARGS+=(--env-file "${ENV_FILE}")
fi

DOCKER_RUN_ARGS+=(
  -e "YOLO_ENV_FILE=${CONTAINER_ENV_FILE}"
  -e "YOLO_CONFIG_DIR=${CONTAINER_ULTRALYTICS_CONFIG_DIR}"
  -v "${ROOT_DIR}:/app"
  -v "${ULTRALYTICS_CONFIG_DIR}:${CONTAINER_ULTRALYTICS_CONFIG_DIR}"
  -w /app
)

if [[ -n "${DOCKER_GPUS:-}" ]]; then
  docker run "${DOCKER_RUN_ARGS[@]}" \
    yolo-easy-trainer:latest \
    ./scripts/train.sh
else
  docker run "${DOCKER_RUN_ARGS[@]}" \
    yolo-easy-trainer:latest \
    ./scripts/train.sh
fi
