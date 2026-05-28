from __future__ import annotations

import io
import os
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_LOCK = Lock()
MODEL: YOLO | None = None


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def find_latest_weights() -> Path | None:
    matches = sorted((ROOT_DIR / "outputs" / "train").glob("*/weights/best.pt"))
    return matches[-1] if matches else None


def get_weights_path() -> Path:
    latest = find_latest_weights()
    if latest is not None:
        return latest
    configured = os.getenv("YOLO_WEIGHTS", "").strip()
    if configured:
        candidate = resolve_path(configured)
        if candidate.exists():
            return candidate
    return resolve_path("outputs/train/obb_demo/weights/best.pt")


def load_model() -> YOLO:
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            weights_path = get_weights_path()
            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights not found: {weights_path}")
            MODEL = YOLO(str(weights_path))
        return MODEL


app = FastAPI(title="YOLO Easy Deploy API", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    weights_path = get_weights_path()
    return {
        "status": "ok",
        "weights_path": str(weights_path),
        "weights_exists": weights_path.exists(),
        "model_loaded": MODEL is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(default=float(os.getenv("API_CONF", "0.25")), ge=0.0, le=1.0),
    iou: float = Query(default=float(os.getenv("API_IOU", "0.45")), ge=0.0, le=1.0),
    imgsz: int = Query(default=int(os.getenv("API_IMGSZ", "1024")), ge=32),
) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        model = load_model()
        results = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    if not results:
        return {"filename": file.filename, "predictions": [], "count": 0}

    result = results[0]
    predictions = result.summary(normalize=False)
    return {
        "filename": file.filename,
        "image_size": {"width": image.width, "height": image.height},
        "count": len(predictions),
        "predictions": predictions,
        "speed_ms": result.speed,
    }
