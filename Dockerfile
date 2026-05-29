FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    ASR_MODEL_DIR=/cache/models/parakeet

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

# torch first from cu126 index
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.5.1 torchaudio==2.5.1

COPY pyproject.toml ./
COPY src/ ./src/

# constraint file to keep torch pinned during package resolution
RUN printf "torch==2.5.1\ntorchaudio==2.5.1\n" > /tmp/constraints.txt \
    && pip install -c /tmp/constraints.txt .

# build-time smoke test
RUN python3 -c "import torch; import nemo.collections.asr; from stt_streaming import server; print('OK')"

# bake model weights
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/parakeet-tdt-0.6b-v3', local_dir='/cache/models/parakeet')"

EXPOSE 8117

HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=120s \
    CMD curl -fsS http://localhost:8117/healthz | grep -q '"status":"ok"'

CMD ["stt-streaming"]
