#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

: "${WEBUI_HOST:=0.0.0.0}"
: "${WEBUI_PORT:=7860}"

cd "${ROOT_DIR}"
python web/train_ui.py --host "${WEBUI_HOST}" --port "${WEBUI_PORT}"
