ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG APT_MIRROR=
ARG PIP_INDEX_URL=
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org/debian|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources && \
        sed -i "s|http://security.debian.org/debian-security|${APT_MIRROR}-security|g" /etc/apt/sources.list.d/debian.sources; \
    fi

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN if [ -n "${PIP_INDEX_URL}" ]; then \
        pip config set global.index-url "${PIP_INDEX_URL}"; \
    fi && \
    pip install --upgrade pip && \
    pip install \
        --index-url "${TORCH_INDEX_URL}" \
        --extra-index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" && \
    pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8000 7860 7890

CMD ["bash", "./scripts/serve.sh"]
