FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    ASR_MODEL_DIR=/cache/models/parakeet

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
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
    torch==2.6.0 torchaudio==2.6.0

COPY pyproject.toml ./
COPY src/ ./src/

# constraint file to keep torch pinned during package resolution
RUN printf "torch==2.6.0\ntorchaudio==2.6.0\nnumpy>=1.26,<2.0\n" > /tmp/constraints.txt \
    && pip install --prefer-binary -c /tmp/constraints.txt .

# build-time smoke test
RUN python3 -c "import torch; import nemo.collections.asr; from stt_streaming import server; print('OK')"

# Model is downloaded on first start to /cache/models (mount a volume there
# in production to persist across restarts). Baking added 3+ GB and pushed
# the unpacked image past the disk budget on our shared 97 GB GPU box.
VOLUME ["/cache/models", "/cache/hf"]

EXPOSE 8117

# Longer start-period covers first-boot model download (~3 GB).
HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=300s \
    CMD curl -fsS http://localhost:8117/healthz | grep -q '"status":"ok"'

CMD ["stt-streaming"]
