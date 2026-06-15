FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    ninja-build \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    htop \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ARG UID
ARG GID
ARG USER_NAME

RUN groupadd -g ${GID} ${USER_NAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER_NAME}

# /opt/venv を有効化（パスを通す）
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# uv sync 時にプロジェクト内（.venv）ではなく /opt/venv を使うように指定
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"

RUN python3.12 -m venv /opt/venv && \
    chown -R ${USER_NAME}:${GID} /opt/venv

USER ${USER_NAME}
WORKDIR /homes/${USER_NAME}/working/mdmethods

COPY --chown=${USER_NAME}:${GID} pyproject.toml uv.lock ./
RUN uv sync
RUN uv pip install "causal-conv1d==1.5.3.post1" "mamba-ssm==2.2.6.post3" --no-build-isolation --no-binary causal-conv1d,mamba-ssm --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 --reinstall
