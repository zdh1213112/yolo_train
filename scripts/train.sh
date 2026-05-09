#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${YOLO_ENV_FILE:-${ROOT_DIR}/.env}"

resolve_runtime_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${ROOT_DIR}/${value#./}"
  fi
}

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

: "${YOLO_TASK:=obb}"
: "${YOLO_MODEL:=yolo26n-obb.pt}"
: "${YOLO_DATA:=data/dataset.yaml}"
: "${YOLO_PROJECT:=outputs/train}"
: "${YOLO_RUN_NAME:=obb_demo}"
: "${YOLO_EPOCHS:=100}"
: "${YOLO_IMGSZ:=1024}"
: "${YOLO_BATCH:=8}"
: "${YOLO_WORKERS:=4}"
: "${YOLO_DEVICE:=0}"

DATA_PATH="$(resolve_runtime_path "${YOLO_DATA}")"
PROJECT_PATH="$(resolve_runtime_path "${YOLO_PROJECT}")"
MODEL_TARGET="${YOLO_MODEL#./}"
MODEL_PATH="$(resolve_runtime_path "${MODEL_TARGET}")"
MODEL_VALUE="${YOLO_MODEL}"
TRAIN_DATA_PATH="${DATA_PATH}"

if [[ -f "${MODEL_PATH}" ]]; then
  MODEL_VALUE="${MODEL_PATH}"
elif [[ "${MODEL_TARGET}" != */* && -f "${ROOT_DIR}/models/${MODEL_TARGET}" ]]; then
  MODEL_VALUE="${ROOT_DIR}/models/${MODEL_TARGET}"
fi

mkdir -p "${PROJECT_PATH}"

if [[ -f "${DATA_PATH}" ]]; then
  DATA_DIR="$(dirname "${DATA_PATH}")"
  DATA_BASENAME="$(basename "${DATA_PATH}")"
  TEMP_DATA_PATH="${PROJECT_PATH}/resolved-${DATA_BASENAME}"
  export DATA_DIR DATA_PATH TEMP_DATA_PATH
  python3 - <<'PY'
import os
from pathlib import Path

data_path = Path(os.environ["DATA_PATH"])
data_dir = Path(os.environ["DATA_DIR"])
temp_data_path = Path(os.environ["TEMP_DATA_PATH"])

lines = data_path.read_text(encoding="utf-8").splitlines()
resolved = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith("path:"):
        raw = stripped.split(":", 1)[1].strip()
        base = Path(raw)
        if not base.is_absolute():
            base = (data_dir / base).resolve()
        resolved.append(f"path: {base}")
    else:
        resolved.append(line)

temp_data_path.write_text("\n".join(resolved) + "\n", encoding="utf-8")
PY
  TRAIN_DATA_PATH="${TEMP_DATA_PATH}"
fi

if [[ "${YOLO_DEVICE}" != "cpu" ]]; then
  export YOLO_DEVICE
  python3 - <<'PY'
import os
import sys

import torch

device = os.environ["YOLO_DEVICE"].strip()

def is_cpu_device(value: str) -> bool:
    return value.lower() == "cpu"

def is_cuda_device(value: str) -> bool:
    if is_cpu_device(value):
        return False
    if value.isdigit():
        return True
    return all(part.strip().isdigit() for part in value.split(",") if part.strip())

if is_cuda_device(device) and not torch.cuda.is_available():
    print("CUDA is not available inside the training environment.", file=sys.stderr)
    print(f"Requested YOLO_DEVICE={device}", file=sys.stderr)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}", file=sys.stderr)
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}", file=sys.stderr)
    print(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        file=sys.stderr,
    )
    print("", file=sys.stderr)
    print("Use one of the following fixes:", file=sys.stderr)
    print("1. If you only need to run training now, set YOLO_DEVICE=cpu in .env.", file=sys.stderr)
    print("2. If you expect GPU training, fix the host NVIDIA driver/runtime first.", file=sys.stderr)
    print("   The host should pass `nvidia-smi` before starting Docker training.", file=sys.stderr)
    sys.exit(1)
PY
fi

yolo task="${YOLO_TASK}" mode=train \
  model="${MODEL_VALUE}" \
  data="${TRAIN_DATA_PATH}" \
  epochs="${YOLO_EPOCHS}" \
  imgsz="${YOLO_IMGSZ}" \
  batch="${YOLO_BATCH}" \
  workers="${YOLO_WORKERS}" \
  device="${YOLO_DEVICE}" \
  project="${PROJECT_PATH}" \
  name="${YOLO_RUN_NAME}"
