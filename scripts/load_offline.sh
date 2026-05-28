#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ARCHIVE="${ROOT_DIR}/yolo_easy_images.tar"
ARCHIVE_PATH="${1:-${YOLO_OFFLINE_IMAGE_ARCHIVE:-${DEFAULT_ARCHIVE}}}"
REQUIRED_IMAGES=(
  "yolo-easy-trainer:latest"
  "yolo-easy-api:latest"
)

log() {
  printf '[load_offline] %s\n' "$*"
}

fail() {
  printf '[load_offline] %s\n' "$*" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required command: $1"
  fi
}

ensure_images_loaded() {
  local image
  for image in "${REQUIRED_IMAGES[@]}"; do
    if ! docker image inspect "${image}" >/dev/null 2>&1; then
      fail "Required image is not available after load: ${image}"
    fi
  done
}

main() {
  require_command docker

  if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    fail "Image archive not found: ${ARCHIVE_PATH}"
  fi

  log "Loading Docker images from ${ARCHIVE_PATH}"
  docker load -i "${ARCHIVE_PATH}"
  ensure_images_loaded
  log "Offline images are ready."
}

main "$@"
