#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

: "${API_HOST:=0.0.0.0}"
if [[ -z "${API_PORT:-}" ]]; then
  API_PORT="${PORT:-8000}"
fi

cd "${ROOT_DIR}"
python -m uvicorn api.main:app --host "${API_HOST}" --port "${API_PORT}"
