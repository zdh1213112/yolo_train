# YOLO Easy Deploy

一个独立的、最小可用的 YOLO 项目模板，步骤完成：

1. 准备数据
2. 一键训练
3. 一键导出
4. 启动 HTTP 推理服务

## 离线开箱即用

如果代码本身通过 `git clone` 获取，当前项目支持只分发预构建镜像，避免别的用户自己 `docker compose build`。

分发方在自己的机器上执行：

```bash
./scripts/package_offline.sh
```

脚本会生成：

- `dist/offline/yolo_easy_images.tar`
- `dist/offline/SHA256SUMS`

建议只把这个镜像包发给对方：

- `yolo_easy_images.tar`：已经构建好的 `trainer` / `api` 镜像

接收方拿到包后只需要：

1. `git clone` 项目代码并进入项目目录
2. 执行 `bash ./scripts/load_offline.sh /path/to/yolo_easy_images.tar`
3. 执行 `bash ./scripts/setup_offline.sh`

之后直接运行：

```bash
./scripts/train_docker.sh
```

如果只启动推理 API：

```bash
./scripts/serve_docker.sh
```

离线分发的现实边界：

- 项目代码和 Docker 镜像可以直接打包
- GPU 驱动和 Docker GPU 运行时不能替对方打包
- 如果 `.env` 里启用了 GPU，对方机器仍然必须先通过 `nvidia-smi`

默认仓库内已经提供：

- `.env.example`
- `models/yolo26n-obb.pt`
- `scripts/load_offline.sh`
- `scripts/setup_offline.sh`



## 目录结构

```text
yolo_easy_deploy/
├── api/                  # FastAPI 推理服务
├── data/                 # 数据集和 dataset.yaml
├── models/               # 可选：手动放预训练模型或导出的模型
├── outputs/              # 训练和导出结果
├── scripts/              # 一键脚本
├── .env.example          # 参数模板
├── docker-compose.yml    # 容器启动方式
├── Dockerfile            # 独立镜像
└── requirements.txt      # 本地依赖
```

## 数据采集与标注链路

项目内已经集成了两步工具链：

1. `tool/collect_dataset_qt.py`
使用 RealSense 相机采集原始彩色图和深度图。
默认输出到：
`data/realsense_dataset/color/`
`data/realsense_dataset/depth/`

2. `tool/final_auto_obb_qt2.py`
使用模板匹配 + MobileSAM 自动生成 OBB 数据集。
默认读取：
`data/realsense_dataset/color/`
默认输出到：
`data/obb_dataset/`

标注完成后，工具会自动刷新项目训练使用的 [`data/dataset.yaml`](/home/zdh/yolo_one/yolo_easy_deploy/data/dataset.yaml)，所以后续可以直接接上训练脚本，不需要再手工改路径。

如果你要使用这两步工具，先安装额外依赖：

```bash
./scripts/bootstrap_tools.sh
```

这里安装的是本地 Python 虚拟环境依赖，不是 Docker 镜像依赖。原因是这两步工具需要：

- 本地打开 PyQt 图形界面
- 直接访问宿主机 RealSense 相机
- 本地交互式框选和路径选择

所以：

- `./scripts/collect_dataset.sh`
- `./scripts/annotate_dataset.sh`

应该在本机运行。

而下面这些脚本可以继续走 Docker：

- `./scripts/train_docker.sh`
- `./scripts/export_docker.sh`
- `./scripts/serve_docker.sh`

然后按这个顺序使用：

```bash
./scripts/collect_dataset.sh
./scripts/annotate_dataset.sh
./scripts/train_docker.sh
```

## Docker 用法

纯 `docker compose build` 不依赖 `.env`。这个项目只有在 `docker compose run` 或 `docker compose up` 时，才会从 `.env` 读取训练和 API 参数。

默认构建：

```bash
docker compose build
```

国内网络推荐直接这样构建：

```bash
BASE_IMAGE=docker.m.daocloud.io/python:3.11-slim docker compose build
```

如果拉取 `python:3.11-slim` 时访问 `auth.docker.io` 超时，说明宿主机到 Docker Hub 的网络不稳定，上面这条命令通常更稳。

如果基础镜像已经能拉下来，但 `apt-get update` 或 `pip install` 很慢，可以把 Debian 和 PyPI 也切到国内源：

！！！国内用户请使用此命令：！！！！构建时间较长，请稍等！

```bash
BASE_IMAGE=docker.m.daocloud.io/python:3.11-slim \
APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
docker compose build
```

如果你是 50 系 NVIDIA 显卡，建议同时固定官方 PyTorch CUDA 12.8 wheel，避免 `pip` 自动解析到兼容性更差的组合：

```bash
BASE_IMAGE=docker.m.daocloud.io/python:3.11-slim \
APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
TORCH_VERSION=2.10.0 \
TORCHVISION_VERSION=0.25.0 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
docker compose build
```

如果你希望后续每次都直接执行 `docker compose build`，可以先准备 `.env`，把 `BASE_IMAGE` 固定写进去：

```bash
cp .env.example .env
```

```text
BASE_IMAGE=docker.m.daocloud.io/python:3.11-slim
TORCH_VERSION=2.10.0
TORCHVISION_VERSION=0.25.0
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
```

如果还是超时，问题通常在宿主机访问 Docker Hub 的网络、IPv6、DNS 或代理配置，不是这个项目的 Dockerfile 本身有问题。

如果你后面要运行容器，而不只是构建镜像，再准备 `.env`：

```bash
cp .env.example .env
```

容器内训练：

```bash
./scripts/train_docker.sh
```

这个脚本会先构建 `trainer` 镜像，再直接用 `docker run` 启动训练容器；这样可以绕开部分 `docker compose run` 版本不支持 `--gpus` 的问题。

如果你希望容器使用 GPU，除了 `YOLO_DEVICE=0` 之外，还需要 Docker 在运行容器时把 GPU 暴露进去。推荐在 `.env` 里加：

```text
YOLO_DEVICE=0
DOCKER_GPUS=all
```

如果训练时出现 `unable to allocate shared memory(shm)`，可以继续在 `.env` 里加大共享内存：

```text
DOCKER_SHM_SIZE=8g
```

如果容器里仍然报 `torch.cuda.is_available(): False`，先不要急着改 YOLO 参数，按这个顺序排查：

1. 先在宿主机执行 `nvidia-smi`
2. 再执行 `docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi`

如果第 1 步就失败，说明宿主机 NVIDIA 驱动本身没工作，容器里也不可能正常使用 CUDA。

如果第 1 步正常，但第 2 步失败，说明问题不在 YOLO，而在 Docker 没拿到宿主机 GPU。

如果宿主机 `nvidia-smi` 正常，但 Docker 里的 `--gpus all` 仍然失败，可以在宿主机执行：

```bash
bash ./scripts/install_nvidia_container_toolkit.sh
```

执行后用这条命令验证：

```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

如果你只是想先把流程跑通，可以临时改成 CPU：

```text
YOLO_DEVICE=cpu
```

如果容器里下载预训练权重很慢，也建议先把权重文件放到 `models/`，并在 `.env` 中设置：

```text
YOLO_MODEL=models/yolo26n-obb.pt
```

为了尽量避免训练启动时再额外联网，这个项目默认还做了两件事：

- `YOLO_AMP=False`
- `YOLO_PLOTS=False`

这样可以避免 Ultralytics 在 AMP 自检阶段额外下载 `yolo26n.pt`，也可以减少首次训练时为了绘图去准备字体资源。

另外，Docker 脚本会把 Ultralytics 缓存目录持久化到项目内的：

```text
.cache/ultralytics/
```

所以同一台机器第一次运行后，后续通常不会重复创建 Ultralytics settings 或重复下载同样的缓存文件。

如果你修改了 `Dockerfile`、想让首次运行也直接带上预置字体，需要重新构建镜像：

```bash
docker compose build trainer api
```

如果你后面确认自己的环境稳定，也想重新打开混合精度或训练图表，可以在 `.env` 里改回：

```text
YOLO_AMP=True
YOLO_PLOTS=True
```

如果你已经把权重下载到了本机任意目录，也可以直接导入项目：

```bash
bash ./scripts/use_local_model.sh /path/to/yolo26n-obb.pt
```

容器内导出：

```bash
./scripts/export_docker.sh
```

容器方式启动 API：

```bash
./scripts/serve_docker.sh
```

## Web 训练界面

如果你希望非技术用户像打开网页一样点按钮开始训练，可以直接使用项目自带的 Gradio 页面：

```bash
./scripts/web_train.sh
```

默认访问地址：

```text
http://127.0.0.1:7860
```

如果运行平台会注入 `PORT`（有些 PaaS 或在线 Docker 平台会这样做，例如要求 `7890`），`./scripts/web_train.sh` 会优先监听 `PORT`，否则再回退到 `WEBUI_PORT` 或默认 `7860`。

页面支持：

- 上传标准 YOLO 数据集 zip
- 填写 `epochs`、`imgsz`、`batch`、`device`
- 自动调用现有 `./scripts/train_docker.sh`
- 页面内实时查看训练日志

推荐 zip 结构：

- 直接包含 `dataset.yaml`
- 或者包含 `images/train`、`images/val`、`labels/train`、`labels/val`

如果 zip 里没有 `dataset.yaml`，页面会根据你填写的类别名自动生成一个。

## 不使用docker，本地快速开始

### 1. 复制环境变量

```bash
cp .env.example .env
```

### 2. 本地安装

```bash
./scripts/bootstrap.sh
```

如果你还要使用 RealSense 采集和自动标注工具，改用：

```bash
./scripts/bootstrap_tools.sh
```

### 3. 准备预训练权重

如果 `models/yolo26n-obb.pt` 还不存在，先用浏览器、下载工具或你自己的网络方式把 `yolo26n-obb.pt` 下载到本机任意位置，然后导入项目。

原始下载地址：

```text
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.pt
```

拿到文件后执行：

```bash
bash ./scripts/use_local_model.sh /path/to/yolo26n-obb.pt
```

这会把权重文件复制到项目里的 `models/` 目录。只要是有效的 `.pt` 文件，就算你的本地文件名不是 `yolo26n-obb.pt` 也可以导入。

### 4. 开始训练

```bash
./scripts/train.sh
```

训练结果通常会输出到：

```text
outputs/train/<run-name>/
```

其中最常用的权重文件是：

```text
outputs/train/<run-name>/weights/best.pt
```

如果训练目录自动变成了 `obb_demo-2`、`obb_demo-3` 这类名字，也不用手改权重路径。导出脚本和 API 会优先选择 `outputs/train` 下最新一次训练的 `best.pt`。

只有当你想强制覆盖“最新权重”逻辑时，才需要在 `.env` 里填写 `YOLO_WEIGHTS`。

### 5. 导出 ONNX

```bash
./scripts/export.sh
```

### 6. 启动推理 API

```bash
./scripts/serve.sh
```

服务默认地址：

```text
http://127.0.0.1:8000
```

如果运行平台会注入 `PORT`，`./scripts/serve.sh` 会优先监听 `PORT`，否则再回退到 `API_PORT` 或默认 `8000`。

健康检查：

```bash
curl http://127.0.0.1:8000/healthz
```

推理示例：

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/obb_dataset/images/val/00002.png"
```

容器方式启动 API：

```bash
./scripts/serve_docker.sh
```

## 离线分发详细说明

### 发送给别人之前

先在你自己的机器上确认本地模型和镜像都准备好了：

```bash
./scripts/package_offline.sh
```

如果 `trainer` 或 `api` 镜像不存在，这个脚本会先自动执行：

```bash
docker compose build trainer api
```

然后自动导出：

```bash
docker save -o dist/offline/yolo_easy_images.tar \
  yolo-easy-trainer:latest \
  yolo-easy-api:latest
```

### 对方拿到镜像包之后

假设对方已经安装好了 Docker。

先获取项目代码：

```bash
git clone <your-repo-url>
cd <your-repo-dir>
```

导入镜像：

```bash
bash ./scripts/load_offline.sh /path/to/yolo_easy_images.tar
```

初始化并检查本地环境：

```bash
bash ./scripts/setup_offline.sh
```

开始训练：

```bash
./scripts/train_docker.sh
```

如果只需要导出：

```bash
./scripts/export_docker.sh
```

如果只需要启动 API：

```bash
./scripts/serve_docker.sh
```

### setup_offline.sh 会做什么

- 没有 `.env` 时自动从 `.env.example` 复制
- 检查 `yolo-easy-trainer:latest` 和 `yolo-easy-api:latest` 是否已经导入
- 检查 `models/yolo26n-obb.pt` 是否存在
- 创建 `.cache/ultralytics/`
- 如果 `.env` 里配置的是 GPU 训练，额外检查宿主机 `nvidia-smi`

### 离线包适合什么场景

- 内网环境
- U 盘分发
- 微信或网盘直接传文件
- 对方不方便自己 `docker compose build`

### 常见构建失败排查

如果报错出现在拉取基础镜像阶段，例如：

```text
failed to fetch anonymous token
load metadata for docker.io/library/python:3.11-slim
auth.docker.io
```

这通常不是项目代码问题，而是宿主机访问 Docker Hub 失败。当前项目的基础镜像只是：

```Dockerfile
FROM python:3.11-slim
```

可以按这个顺序排查：

1. 先直接尝试国内镜像源构建：

```bash
BASE_IMAGE=docker.m.daocloud.io/python:3.11-slim docker compose build
```

2. 如果基础镜像能拉下来，但卡在 `apt-get update` 或 `pip install`，再切 Debian 和 PyPI 国内源：

```bash
BASE_IMAGE=docker.m.daocloud.io/python:3.11-slim \
APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
docker compose build
```

3. 如果这样能成功，说明项目本身没问题，问题在宿主机到 Docker Hub、Debian 源或 PyPI 的网络链路。
4. 如果还是失败，重点检查宿主机的 Docker DNS、代理、IPv6 连通性，或者是否能直接 `docker pull python:3.11-slim`。
5. 如果报错已经进入项目代码复制或应用启动阶段，再回来看 Dockerfile、依赖版本或项目本身。

## 你需要改的地方

1. 把自己的数据集放到 `data/obb_dataset/`
2. 按自己的类别修改 `data/dataset.yaml`
3. 按自己的机器修改 `.env` 里的 `YOLO_DEVICE`、`YOLO_BATCH`、`YOLO_IMGSZ`

## 默认设计

- 默认任务：`obb`
- 默认模型：`yolo26n-obb.pt`
- 默认导出格式：`onnx`
- 默认服务接口：`/predict`

如果以后你切换到普通检测任务，也只需要改：

- `.env` 里的 `YOLO_TASK`
- `data/dataset.yaml`
- `YOLO_MODEL`
