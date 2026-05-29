# stt-streaming

Streaming speech-to-text pod for the Vocence ops fleet. Wraps **NVIDIA Parakeet TDT 0.6B v3** behind a WebSocket that accepts raw 16 kHz mono PCM frames and emits interim (`partial`) and committed (`final`) transcripts in real time.

Single-GPU container (RTX 4090 / L4 / A10 / A100 class). Targets >=30 concurrent streams per 4090 with TTFP p95 <= 400 ms idle.

See [`stt-streaming-implementation.md`](../stt-streaming-implementation.md) for the full spec — protocol, env vars, performance targets, definition of done.

## Quick start

```bash
# 1. Build
docker build -t vocence/stt-streaming:dev .

# 2. Run on a CUDA-capable GPU
docker run --rm --gpus all -p 8114:8114 \
  -e ASR_API_KEY=test_key_local_only \
  vocence/stt-streaming:dev

# 3. Smoke test
python scripts/smoke_test.py \
  --url ws://localhost:8114/v1/stream \
  --api-key test_key_local_only \
  --wav tests/fixtures/librispeech_short.wav
```

Expected: at least one `partial`, one `final` with the known transcript, clean close (1000).

## Endpoints

| Path             | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| `GET /healthz`   | Liveness + readiness (`status`: `ok` / `warming` / `degraded` / `error`)|
| `GET /metrics`   | Prometheus plaintext (`asr_*` counters, see spec §4.2)                  |
| `WS  /v1/stream` | Streaming session — see spec §5 for the message protocol                |

All endpoints require `X-API-Key: $ASR_API_KEY`.

## Configuration

Key env vars (full table in spec §7):

- `ASR_API_KEY` (**required**) — shared secret
- `ASR_MODEL` — HuggingFace model id, default `nvidia/parakeet-tdt-0.6b-v3`
- `ASR_PORT` — default `8114`
- `ASR_MAX_CONCURRENT` — default `32`

## Model attribution

Parakeet TDT 0.6B v3 (c) NVIDIA, licensed CC-BY-4.0. Surfaced via `/healthz` `model_attribution`.
