#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist/offline"
IMAGE_ARCHIVE="${DIST_DIR}/yolo_easy_images.tar"
DEFAULT_IMAGE_TAGS=(
  "yolo-easy-trainer:latest"
  "yolo-easy-api:latest"
)

log() {
  printf '[package_offline] %s\n' "$*"
}

fail() {
  printf '[package_offline] %s\n' "$*" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required command: $1"
  fi
}

ensure_env_file() {
  if [[ -f "${ROOT_DIR}/.env" ]]; then
    return
  fi

  if [[ -f "${ROOT_DIR}/.env.example" ]]; then
    cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
    log "Created .env from .env.example for packaging."
    return
  fi

  fail "Neither .env nor .env.example exists."
}

ensure_local_model() {
  local model_path="${ROOT_DIR}/models/yolo26n-obb.pt"
  if [[ ! -f "${model_path}" ]]; then
    fail "Required local model is missing: ${model_path}"
  fi
}

ensure_images() {
  local missing=0
  local image
  for image in "${DEFAULT_IMAGE_TAGS[@]}"; do
    if ! docker image inspect "${image}" >/dev/null 2>&1; then
      missing=1
      break
    fi
  done

  if [[ "${missing}" -eq 1 ]]; then
    log "Building trainer/api images because at least one target image is missing."
    docker compose build trainer api
  fi
}

write_checksum_file() {
  (
    cd "${DIST_DIR}"
    sha256sum "$(basename "${IMAGE_ARCHIVE}")" > SHA256SUMS
  )
}

main() {
  require_command docker
  require_command sha256sum

  cd "${ROOT_DIR}"
  ensure_env_file
  ensure_local_model
  ensure_images

  mkdir -p "${DIST_DIR}"

  log "Saving Docker images to ${IMAGE_ARCHIVE}"
  docker save -o "${IMAGE_ARCHIVE}" "${DEFAULT_IMAGE_TAGS[@]}"

  write_checksum_file

  log "Offline package ready:"
  log "  ${IMAGE_ARCHIVE}"
  log "  ${DIST_DIR}/SHA256SUMS"
}

main "$@"
