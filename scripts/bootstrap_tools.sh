#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install -r "${ROOT_DIR}/requirements-tools.txt"

echo "Tool environment ready: ${VENV_DIR}"
echo "Launch collection with: ./scripts/collect_dataset.sh"
echo "Launch auto-annotation with: ./scripts/annotate_dataset.sh"
