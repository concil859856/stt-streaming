from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from . import __version__

_STARTED_AT_MONO = time.monotonic()
_STARTED_AT_ISO = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

_MODEL_ATTRIBUTION = "Parakeet TDT 0.6B v3 (c) NVIDIA, CC-BY-4.0"


@dataclass
class _State:
    status: str = "warming"
    model: str = "nvidia/parakeet-tdt-0.6b-v3"
    in_flight: int = 0
    max_concurrent: int = 32


_state = _State()


def set_status(status: str) -> None:
    _state.status = status


def set_model(model: str) -> None:
    _state.model = model


def set_max_concurrent(n: int) -> None:
    _state.max_concurrent = n


def set_in_flight(n: int) -> None:
    _state.in_flight = n


def _gpu_info() -> Optional[Dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total = int(props.total_memory)
        used = int(torch.cuda.memory_allocated(idx))
        util_pct = int(round(used * 100 / total)) if total else 0
        return {
            "name": props.name,
            "vram_used_mib": used // (1024 * 1024),
            "vram_total_mib": total // (1024 * 1024),
            "utilization_pct": util_pct,
        }
    except Exception:
        return None


def get_health_state() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": _state.status,
        "service": "stt-streaming",
        "model": _state.model,
        "model_attribution": _MODEL_ATTRIBUTION,
        "version": __version__,
        "uptime_seconds": int(time.monotonic() - _STARTED_AT_MONO),
        "in_flight": _state.in_flight,
        "max_concurrent_streams": _state.max_concurrent,
        "started_at": _STARTED_AT_ISO,
    }
    gpu = _gpu_info()
    if gpu is not None:
        out["gpu"] = gpu
    return out
