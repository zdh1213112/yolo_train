from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import traceback
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parents[1]
UPLOADS_DIR = ROOT_DIR / "data" / "web_uploads"
JOBS_DIR = ROOT_DIR / "outputs" / "webui_jobs"
DEFAULT_ENV_FILES = [ROOT_DIR / ".env", ROOT_DIR / ".env.example"]
TRAIN_LOCK = Lock()


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_")
    return cleaned.lower() or "train"


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def load_base_env() -> dict[str, str]:
    for candidate in DEFAULT_ENV_FILES:
        values = parse_env_file(candidate)
        if values:
            return values
    return {}


def write_env_file(path: Path, values: dict[str, str]) -> None:
    ordered_keys = list(values.keys())
    lines = [f"{key}={values[key]}" for key in ordered_keys]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_class_names(raw: str) -> list[str]:
    parts = [item.strip() for item in raw.split(",")]
    names = [item for item in parts if item]
    return names or ["object"]


def append_log(log_lines: deque[str], message: str) -> str:
    for line in message.rstrip().splitlines():
        log_lines.append(line)
    return "\n".join(log_lines)


def find_single_dataset_yaml(root: Path) -> Path | None:
    candidates = sorted(root.rglob("dataset.yaml"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def looks_like_yolo_dataset(root: Path) -> bool:
    required = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    return all(path.is_dir() for path in required)


def describe_dataset_root(root: Path) -> str:
    expected = {
        "images/train": (root / "images" / "train").is_dir(),
        "images/val": (root / "images" / "val").is_dir(),
        "labels/train": (root / "labels" / "train").is_dir(),
        "labels/val": (root / "labels" / "val").is_dir(),
        "dataset.yaml": (root / "dataset.yaml").is_file(),
    }
    available = [name for name, exists in expected.items() if exists]
    missing = [name for name, exists in expected.items() if not exists]
    return (
        f"目录 {root} 中已检测到: {', '.join(available) if available else '无'}; "
        f"缺少: {', '.join(missing) if missing else '无'}"
    )


def generate_dataset_yaml(root: Path, class_names: list[str]) -> Path:
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for index, name in enumerate(class_names):
        lines.append(f"  {index}: {name}")

    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset_yaml


def resolve_dataset_yaml(extract_dir: Path, class_names: list[str]) -> Path:
    dataset_yaml = find_single_dataset_yaml(extract_dir)
    if dataset_yaml is not None:
        return dataset_yaml

    candidate_roots = [extract_dir]
    candidate_roots.extend(path for path in extract_dir.iterdir() if path.is_dir())
    matching_roots = [path for path in candidate_roots if looks_like_yolo_dataset(path)]

    if len(matching_roots) == 1:
        return generate_dataset_yaml(matching_roots[0], class_names)

    if len(matching_roots) > 1:
        raise ValueError("压缩包里检测到多个可训练数据集目录，请只保留一个。")

    inspected = [
        path for path in candidate_roots
        if any((path / part).exists() for part in ("images", "labels", "dataset.yaml"))
    ]
    if inspected:
        details = "\n".join(describe_dataset_root(path) for path in inspected)
        raise ValueError(
            "没有找到可直接训练的数据集结构。\n"
            f"{details}\n"
            "请上传包含 dataset.yaml 的 zip，或保证 zip 内有完整的 "
            "images/train、images/val、labels/train、labels/val。"
        )

    raise ValueError(
        "没有找到 dataset.yaml，也没有检测到标准 YOLO 目录结构。"
        "请上传包含 dataset.yaml 的 zip，或保证 zip 内有 images/ 和 labels/ 目录。"
    )


def prepare_dataset(upload_path: str, run_slug: str, class_names: list[str]) -> tuple[Path, Path]:
    source = Path(upload_path)
    if source.suffix.lower() != ".zip":
        raise ValueError("目前只支持上传 .zip 数据集。")

    upload_dir = UPLOADS_DIR / run_slug
    extract_dir = upload_dir / "extracted"
    upload_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    target_zip = upload_dir / "dataset.zip"
    shutil.copy2(source, target_zip)

    with zipfile.ZipFile(target_zip) as archive:
        archive.extractall(extract_dir)

    dataset_yaml = resolve_dataset_yaml(extract_dir, class_names)
    return dataset_yaml, upload_dir


def find_latest_run_dir(project_dir: Path, run_name: str, started_at: float) -> Path | None:
    candidates = [
        path
        for path in project_dir.glob(f"{run_name}*")
        if path.is_dir() and path.stat().st_mtime >= started_at - 5
    ]
    if not candidates:
        candidates = [path for path in project_dir.glob(f"{run_name}*") if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def collect_artifacts(run_dir: Path | None) -> list[str]:
    if run_dir is None:
        return []

    artifact_candidates = [
        run_dir / "weights" / "best.pt",
        run_dir / "args.yaml",
        run_dir / "results.csv",
        run_dir / "results.png",
        run_dir / "confusion_matrix.png",
    ]
    return [str(path) for path in artifact_candidates if path.exists()]


def build_env_values(
    dataset_yaml: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> dict[str, str]:
    values = load_base_env()
    values.update(
        {
            "YOLO_DATA": os.path.relpath(dataset_yaml, ROOT_DIR),
            "YOLO_RUN_NAME": run_name,
            "YOLO_EPOCHS": str(epochs),
            "YOLO_IMGSZ": str(imgsz),
            "YOLO_BATCH": str(batch),
            "YOLO_WORKERS": str(workers),
            "YOLO_DEVICE": device.strip() or "cpu",
        }
    )
    return values


def start_training(
    dataset_zip: str | None,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    class_names: str,
):
    if dataset_zip is None:
        yield "未开始", "请先上传数据集 zip。", "", []
        return

    if not TRAIN_LOCK.acquire(blocking=False):
        yield "忙碌中", "当前已有一个训练任务在运行，请等待它结束。", "", []
        return

    log_lines: deque[str] = deque(maxlen=400)
    run_slug = slugify(run_name or f"train-{datetime.now():%Y%m%d-%H%M%S}")
    job_dir = JOBS_DIR / run_slug
    project_dir = ROOT_DIR / load_base_env().get("YOLO_PROJECT", "outputs/train")

    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)

        logs = append_log(log_lines, f"[prepare] job={run_slug}")
        yield "准备数据集", logs, "", []

        dataset_yaml, upload_dir = prepare_dataset(dataset_zip, run_slug, parse_class_names(class_names))
        logs = append_log(log_lines, f"[prepare] dataset_yaml={dataset_yaml}")
        logs = append_log(log_lines, f"[prepare] upload_dir={upload_dir}")
        yield "写入训练配置", logs, "", []

        env_values = build_env_values(dataset_yaml, run_slug, epochs, imgsz, batch, workers, device)
        env_path = job_dir / "job.env"
        write_env_file(env_path, env_values)
        logs = append_log(log_lines, f"[prepare] env_file={env_path}")

        process = subprocess.Popen(
            ["bash", "scripts/train_docker.sh"],
            cwd=ROOT_DIR,
            env={**os.environ, "YOLO_ENV_FILE": str(env_path)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        started_at = datetime.now().timestamp()
        yield "训练中", logs, "", []

        assert process.stdout is not None
        for line in process.stdout:
            logs = append_log(log_lines, line)
            yield "训练中", logs, "", []

        return_code = process.wait()
        if return_code != 0:
            logs = append_log(log_lines, f"[error] training exited with code {return_code}")
            yield "训练失败", logs, "", []
            return

        run_dir = find_latest_run_dir(project_dir, run_slug, started_at)
        artifact_files = collect_artifacts(run_dir)
        logs = append_log(log_lines, f"[done] run_dir={run_dir or '未识别'}")
        yield "训练完成", logs, str(run_dir or ""), artifact_files
    except Exception:
        error_text = traceback.format_exc()
        logs = append_log(log_lines, error_text)
        yield "训练失败", logs, "", []
    finally:
        TRAIN_LOCK.release()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="YOLO Easy Deploy Trainer") as app:
        gr.Markdown(
            """
            # YOLO Easy Deploy Web 训练界面

            上传一个 `.zip` 数据集后即可直接启动现有 Docker 训练流程。

            推荐 zip 结构：
            - 包含 `dataset.yaml`
            - 或者至少包含 `images/train`、`images/val`、`labels/train`、`labels/val`
            """
        )

        with gr.Row():
            dataset_zip = gr.File(label="数据集 Zip", file_types=[".zip"], type="filepath")
            class_names = gr.Textbox(
                label="类别名",
                value="object",
                info="逗号分隔。只有当 zip 里没有 dataset.yaml 时才会用它自动生成配置。",
            )

        with gr.Row():
            run_name = gr.Textbox(label="训练任务名", value="web_train")
            device = gr.Textbox(label="训练设备", value="0", info="填 cpu 或 GPU 编号，例如 0、0,1")

        with gr.Row():
            epochs = gr.Number(label="Epochs", value=100, precision=0)
            imgsz = gr.Number(label="Image Size", value=640, precision=0)
            batch = gr.Number(label="Batch", value=8, precision=0)
            workers = gr.Number(label="Workers", value=4, precision=0)

        start_button = gr.Button("开始训练", variant="primary")
        status = gr.Textbox(label="状态", interactive=False)
        run_dir = gr.Textbox(label="输出目录", interactive=False)
        logs = gr.Textbox(label="训练日志", lines=24, interactive=False)
        artifacts = gr.File(label="训练产物", file_count="multiple", interactive=False)

        start_button.click(
            fn=start_training,
            inputs=[dataset_zip, run_name, epochs, imgsz, batch, workers, device, class_names],
            outputs=[status, logs, run_dir, artifacts],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO Easy Deploy Web trainer")
    parser.add_argument("--host", default=os.getenv("WEBUI_HOST", "0.0.0.0"))
    default_port = int(os.getenv("WEBUI_PORT") or os.getenv("PORT", "7860"))
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()

    app = build_ui()
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
