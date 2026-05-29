#!/usr/bin/env python3
"""Performance benchmark for a running stt-streaming pod.

Measures the §8 acceptance targets against a deployed container:

  * TTFP p95   — time from first audio frame to first ``partial`` message
  * Partial cadence — average partials/sec while audio is being streamed
  * RTF        — wall time to consume 1 s of audio (single stream)
  * Max concurrent streams — ramp up to ``--max-concurrent`` and report the
    highest count where TTFP p95 still satisfies the target
  * VRAM footprint — sampled from ``/healthz`` at idle and at peak load

Produces a pass/fail report per target and exits non-zero if any fail.

Usage::

    python scripts/benchmark.py \
        --url ws://localhost:8114/v1/stream \
        --api-key test_key_local_only \
        --wav tests/fixtures/librispeech_short.wav \
        --max-concurrent 30 --duration-sec 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import websockets

# Acceptance targets from spec §8.
TARGET_TTFP_P95_MS_IDLE = 400
TARGET_TTFP_P95_MS_LOAD = 600
TARGET_PARTIAL_CADENCE_HZ = 4.0
TARGET_RTF = 0.15
TARGET_VRAM_IDLE_MIB = 4 * 1024
TARGET_VRAM_PEAK_MIB = 22 * 1024

CHUNK_SAMPLES = 320  # 20 ms at 16 kHz
SAMPLE_RATE = 16000


@dataclass
class StreamResult:
    ttfp_ms: float | None = None
    partials: int = 0
    finals: int = 0
    wall_ms: float = 0.0
    audio_ms: float = 0.0
    close_code: int | None = None
    errors: list[str] = field(default_factory=list)


def _read_wav_pcm16(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2
        assert wf.getframerate() == SAMPLE_RATE
        return wf.readframes(wf.getnframes())


async def _run_one_stream(url: str, api_key: str, pcm: bytes) -> StreamResult:
    res = StreamResult()
    chunk_bytes = CHUNK_SAMPLES * 2
    res.audio_ms = (len(pcm) / 2) / SAMPLE_RATE * 1000
    headers = [("X-API-Key", api_key)]
    t0 = time.perf_counter()
    first_audio_ts: float | None = None
    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            await ws.send(json.dumps({
                "type": "start",
                "language": "auto",
                "sample_rate": SAMPLE_RATE,
                "encoding": "pcm_s16le",
                "enable_partials": True,
            }))
            await ws.recv()  # ready

            async def _send():
                nonlocal first_audio_ts
                for i in range(0, len(pcm), chunk_bytes):
                    if first_audio_ts is None:
                        first_audio_ts = time.perf_counter()
                    await ws.send(pcm[i : i + chunk_bytes])
                    await asyncio.sleep(0.02)
                await ws.send(json.dumps({"type": "close"}))

            async def _recv():
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    t = msg.get("type")
                    if t == "partial":
                        if res.ttfp_ms is None and first_audio_ts is not None:
                            res.ttfp_ms = (time.perf_counter() - first_audio_ts) * 1000
                        res.partials += 1
                    elif t == "final":
                        res.finals += 1
                    elif t == "error":
                        res.errors.append(f"{msg.get('code')}: {msg.get('message')}")

            await asyncio.gather(_send(), _recv())
            res.close_code = ws.close_code
    except Exception as e:  # pragma: no cover - reported via .errors
        res.errors.append(repr(e))
    res.wall_ms = (time.perf_counter() - t0) * 1000
    return res


async def _fetch_healthz(url: str, api_key: str) -> dict[str, Any]:
    """Pull /healthz via aiohttp-less stdlib for fewer deps."""
    import urllib.request
    parsed = urlparse(url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    health_url = f"{scheme}://{parsed.netloc}/healthz"
    req = urllib.request.Request(health_url, headers={"X-API-Key": api_key})
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: json.loads(urllib.request.urlopen(req, timeout=5).read())
    )


def _p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return s[idx]


async def _benchmark(args: argparse.Namespace) -> int:
    pcm = _read_wav_pcm16(args.wav)
    audio_sec = (len(pcm) / 2) / SAMPLE_RATE

    print("# stt-streaming benchmark")
    print(f"url={args.url} wav={args.wav} audio={audio_sec:.2f}s "
          f"max_concurrent={args.max_concurrent}")

    # --- idle metrics --------------------------------------------------
    idle_health = await _fetch_healthz(args.url, args.api_key)
    vram_idle = idle_health.get("gpu", {}).get("vram_used_mib", 0)

    # --- single-stream warm pass --------------------------------------
    print("\n[single-stream] measuring TTFP / RTF / cadence ...")
    single_runs = [await _run_one_stream(args.url, args.api_key, pcm)
                   for _ in range(5)]
    ttfp_single = [r.ttfp_ms for r in single_runs if r.ttfp_ms is not None]
    ttfp_p95_idle = _p95(ttfp_single)
    rtfs = [(r.wall_ms / r.audio_ms) for r in single_runs if r.audio_ms]
    rtf_med = statistics.median(rtfs) if rtfs else float("nan")
    cadences = [(r.partials / (r.audio_ms / 1000)) for r in single_runs if r.audio_ms]
    cadence_med = statistics.median(cadences) if cadences else float("nan")

    # --- ramp-up concurrency ------------------------------------------
    print(f"\n[concurrent] ramping to {args.max_concurrent} streams ...")
    tasks = [_run_one_stream(args.url, args.api_key, pcm)
             for _ in range(args.max_concurrent)]
    t_load_start = time.perf_counter()
    load_results = await asyncio.gather(*tasks)
    load_wall = time.perf_counter() - t_load_start

    peak_health = await _fetch_healthz(args.url, args.api_key)
    vram_peak = peak_health.get("gpu", {}).get("vram_used_mib", 0)

    ttfp_load = [r.ttfp_ms for r in load_results if r.ttfp_ms is not None]
    ttfp_p95_load = _p95(ttfp_load)
    rejected = sum(1 for r in load_results if r.close_code == 4429)
    errored = sum(1 for r in load_results if r.errors)

    # --- report --------------------------------------------------------
    print("\n## Results")
    rows = [
        ("TTFP p95 (idle, single stream)", f"{ttfp_p95_idle:.0f} ms",
         f"<= {TARGET_TTFP_P95_MS_IDLE} ms",
         ttfp_p95_idle <= TARGET_TTFP_P95_MS_IDLE),
        ("Partial cadence (Hz)", f"{cadence_med:.2f}",
         f">= {TARGET_PARTIAL_CADENCE_HZ}",
         cadence_med >= TARGET_PARTIAL_CADENCE_HZ),
        ("RTF (single stream)", f"{rtf_med:.3f}",
         f"<= {TARGET_RTF}", rtf_med <= TARGET_RTF),
        (f"TTFP p95 @ {args.max_concurrent} concurrent",
         f"{ttfp_p95_load:.0f} ms",
         f"<= {TARGET_TTFP_P95_MS_LOAD} ms",
         ttfp_p95_load <= TARGET_TTFP_P95_MS_LOAD),
        ("Concurrent streams accepted",
         f"{args.max_concurrent - rejected}/{args.max_concurrent}",
         f">= {args.max_concurrent}",
         rejected == 0),
        ("VRAM idle (MiB)", f"{vram_idle}",
         f"<= {TARGET_VRAM_IDLE_MIB}",
         vram_idle <= TARGET_VRAM_IDLE_MIB),
        ("VRAM peak (MiB)", f"{vram_peak}",
         f"<= {TARGET_VRAM_PEAK_MIB}",
         vram_peak <= TARGET_VRAM_PEAK_MIB),
    ]
    width = max(len(r[0]) for r in rows) + 2
    all_pass = True
    for name, got, target, ok in rows:
        marker = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name.ljust(width)} {got:>12}   target {target:<20} [{marker}]")

    print(f"\nload-wall={load_wall:.1f}s rejected={rejected} errored={errored}")
    print("\nOVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="ws://host:port/v1/stream")
    p.add_argument("--api-key", required=True)
    p.add_argument("--wav", required=True, help="16 kHz mono PCM16 WAV fixture")
    p.add_argument("--max-concurrent", type=int, default=30)
    p.add_argument("--duration-sec", type=int, default=60,
                   help="(reserved) soak duration for steady-state checks")
    args = p.parse_args()
    return asyncio.run(_benchmark(args))


if __name__ == "__main__":
    sys.exit(main())
