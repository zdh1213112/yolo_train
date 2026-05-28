#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
EXAMPLE_ENV_FILE="${ROOT_DIR}/.env.example"
MODEL_PATH="${ROOT_DIR}/models/yolo26n-obb.pt"
ULTRALYTICS_CACHE_DIR="${ROOT_DIR}/.cache/ultralytics"
REQUIRED_IMAGES=(
  "yolo-easy-trainer:latest"
  "yolo-easy-api:latest"
)

log() {
  printf '[setup_offline] %s\n' "$*"
}

fail() {
  printf '[setup_offline] %s\n' "$*" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required command: $1"
  fi
}

ensure_env_file() {
  if [[ -f "${ENV_FILE}" ]]; then
    return
  fi

  if [[ -f "${EXAMPLE_ENV_FILE}" ]]; then
    cp "${EXAMPLE_ENV_FILE}" "${ENV_FILE}"
    log "Created .env from .env.example"
    return
  fi

  fail "Missing .env and .env.example"
}

ensure_images() {
  local image
  for image in "${REQUIRED_IMAGES[@]}"; do
    if ! docker image inspect "${image}" >/dev/null 2>&1; then
      fail "Missing Docker image: ${image}. Run ./scripts/load_offline.sh first."
    fi
  done
}

ensure_model() {
  if [[ ! -f "${MODEL_PATH}" ]]; then
    fail "Missing required model file: ${MODEL_PATH}"
  fi
}

check_gpu_runtime() {
  if [[ ! -f "${ENV_FILE}" ]]; then
    return
  fi

  set -a
  source "${ENV_FILE}"
  set +a

  if [[ "${DOCKER_GPUS:-}" == "" ]]; then
    return
  fi

  if [[ "${YOLO_DEVICE:-cpu}" == "cpu" ]]; then
    log "YOLO_DEVICE=cpu, skipping GPU runtime check."
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "GPU mode is configured but nvidia-smi is not available on this machine."
  fi

  if ! nvidia-smi >/dev/null 2>&1; then
    fail "GPU mode is configured but host nvidia-smi failed."
  fi

  log "Host NVIDIA driver looks available."
}

main() {
  require_command docker
  ensure_env_file
  ensure_images
  ensure_model
  mkdir -p "${ULTRALYTICS_CACHE_DIR}"
  check_gpu_runtime

  log "Offline environment is ready."
  log "Next step:"
  log "  ./scripts/train_docker.sh"
}

main "$@"
