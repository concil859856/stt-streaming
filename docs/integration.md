# Vocence backend integration

This pod exposes **two transcription paths** sharing one Parakeet TDT 0.6B v3
model on a single GPU. The dashboard backend picks the right one based on the
call site.

| Path | When | Protocol | Latency |
|---|---|---|---|
| `POST /v1/transcribe` | Studio file upload, batch jobs | HTTP multipart | full-audio (≈100ms / 5s of audio) |
| `WS /v1/stream` | Voice agents (live conversation) | WebSocket + raw PCM16LE | streaming partials (~200-400ms TTFP) |

Auth is the same on both: header `X-API-Key: <DUBBING_API_KEY>` (set at pod
deploy time via the ops UI; the dispatcher injects the pod's own key
automatically).

---

## 1. Batch endpoint — `POST /v1/transcribe`

For the Studio `/stt` page (user uploads a file → gets one transcript back).
Drop-in replacement for the existing batch `asr-streaming` pod's `/transcribe`.

### Request

```
POST /v1/transcribe
Host: <pod_url>          # e.g. http://94.101.98.58:8117
X-API-Key: <key>
Content-Type: multipart/form-data

audio    = <file>        # WAV / MP3 / M4A / OGG / FLAC / WebM — anything
                         #   soundfile or ffmpeg can decode. Max 100 MB.
language = "en"          # Optional; echoed back in the response. The model
                         #   auto-detects, this field is metadata only.
```

The audio is auto-resampled to 16 kHz mono inside the pod — the client
doesn't need to preprocess.

### Response

```json
{
  "text":       "Hello world, this is the transcript.",
  "language":   "en",
  "audio_ms":   5239,
  "latency_ms": 103,
  "model":      "nvidia/parakeet-tdt-0.6b-v3"
}
```

### Error responses

| HTTP | Body | When |
|---|---|---|
| `400` | `{"error":"empty audio"}` | Zero-byte upload |
| `400` | `{"error":"audio decode failed: ..."}` | Unrecognised audio container |
| `401` | `{"error":"unauthorized"}` | Missing or wrong `X-API-Key` |
| `413` | `{"error":"audio too large (max 100 MB)"}` | File over 100 MB |
| `500` | `{"error":"internal","message":"..."}` | Model inference failure |

### Backend integration in `studio_tts_service.py`

The existing `transcribe_audio()` function POSTs to
`pod.url + "/transcribe"` with a JSON body containing base64 audio. To use
this new pod, change the URL suffix to `/v1/transcribe` and the body to
multipart form-data:

```python
async with aiohttp.ClientSession() as session:
    form = aiohttp.FormData()
    form.add_field("audio", audio_bytes, filename="audio.wav",
                   content_type="audio/wav")
    if language:
        form.add_field("language", language)
    async with session.post(
        f"{pod.url}/v1/transcribe",
        headers={"X-API-Key": pod.api_key} if pod.api_key else {},
        data=form,
        timeout=aiohttp.ClientTimeout(total=180),
    ) as resp:
        data = await resp.json()
        return {"text": data["text"], "language": data.get("language")}
```

To pick the right pod, register this pod under ops service name
`stt_streaming` (separate slot from the existing `stt` batch pod) and have
`transcribe_audio()` prefer `gpu_pool.pick_pod("stt_streaming")` with a
fallback to `gpu_pool.pick_pod("stt")` so existing batch deployments keep
working during the migration.

---

## 2. Streaming endpoint — `WS /v1/stream`

For voicechat. Spec details live in §5 of `stt-streaming-implementation.md`,
not duplicated here. Summary:

1. Open `WS /v1/stream` with `X-API-Key` header.
2. Send one `{"type":"start", "language":"en", "sample_rate":16000, "encoding":"pcm_s16le"}` JSON frame.
3. Wait for `{"type":"ready", ...}`.
4. Stream raw PCM16LE binary frames (320 samples / 20 ms recommended).
5. Receive `{"type":"partial", "text":"..."}` JSON frames continuously and
   `{"type":"final", "text":"..."}` when the model commits an utterance.
6. Either send `{"type":"close"}` or just close the WS.

---

## 3. Health and metrics (both shared)

| Endpoint | Description |
|---|---|
| `GET /healthz` | Returns `{"status":"ok", "service":"stt-streaming", ...}` once model is loaded |
| `GET /metrics` | Prometheus plaintext with `asr_requests_total{status="ok"}`, latency sums, etc. |

Both require `X-API-Key`. The ops dashboard health poller and metrics
scraper handle these automatically — no changes needed on that side.

---

## 4. Notes for migration

- This pod and the existing batch `asr-streaming` pod can run on the same
  GPU simultaneously (Parakeet ~2.4 GB VRAM idle, Whisper-based batch
  ~2 GB). Total well under a single 4090's 24 GB.
- Different ops service slots (`stt_streaming` vs `stt`) means the
  dispatcher won't mix them. Decide per-feature which to route to.
- The streaming pod's batch endpoint (`/v1/transcribe`) gives the same
  Parakeet output as the streaming one, just delivered as one blob.
  Quality should match — only latency profile differs.
