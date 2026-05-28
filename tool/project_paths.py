from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REALSENSE_DATASET_DIR = DATA_DIR / "realsense_dataset"
REALSENSE_COLOR_DIR = REALSENSE_DATASET_DIR / "color"
REALSENSE_DEPTH_DIR = REALSENSE_DATASET_DIR / "depth"
REALSENSE_BG_DIR = REALSENSE_DATASET_DIR / "background"
OBB_DATASET_DIR = DATA_DIR / "obb_dataset"
PROJECT_DATASET_YAML = DATA_DIR / "dataset.yaml"


def display_path(path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return str(target.relative_to(ROOT_DIR))
    except ValueError:
        return str(target)
