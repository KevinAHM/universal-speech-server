"""Dependency-free host telemetry for the authenticated resources endpoint."""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Mapping

from .config import ServerConfig

MIB = 1024 * 1024


def _windows_memory() -> tuple[dict[str, int] | None, int | None]:
    if os.name != "nt":
        return None, None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None, None
    total = int(status.total_physical)
    free = int(status.available_physical)

    process_bytes = None
    try:
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p
        get_process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_process_memory.restype = ctypes.c_int
        ok = get_process_memory(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        if ok:
            process_bytes = int(counters.working_set_size)
    except (AttributeError, OSError):
        pass
    return {"totalBytes": total, "usedBytes": total - free, "freeBytes": free}, process_bytes


def _posix_memory() -> tuple[dict[str, int] | None, int | None]:
    if os.name == "nt":
        return None, None
    ram = None
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                name, value = line.split(":", 1)
                values[name] = int(value.strip().split()[0]) * 1024
        total = values["MemTotal"]
        free = values.get("MemAvailable", values.get("MemFree", 0))
        ram = {"totalBytes": total, "usedBytes": total - free, "freeBytes": free}
    except (OSError, KeyError, ValueError):
        pass

    process_bytes = None
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        process_bytes = resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError, IndexError):
        pass
    return ram, process_bytes


def _selected_nvidia_identity(environ: Mapping[str, str]) -> str | None:
    """Return the first physical CUDA device explicitly exposed to the server."""
    visible = environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible.split(",", 1)[0].strip()
    requested = environ.get("SPEECH_SERVER_GPU_DEVICE", "").strip()
    return requested.split(",", 1)[0].strip() if requested else None


def _nvidia_gpus(
    environ: Mapping[str, str] = os.environ,
) -> tuple[list[dict[str, Any]], str | None]:
    selected_identity = _selected_nvidia_identity(environ)
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    if selected_identity is not None:
        command.insert(1, f"--id={selected_identity}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
    except FileNotFoundError:
        return [], "nvidia-smi-not-found"
    except subprocess.TimeoutExpired:
        return [], "nvidia-smi-timeout"
    except OSError:
        return [], "nvidia-smi-unavailable"
    if result.returncode != 0:
        return [], "nvidia-smi-failed"

    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 5)]
        if len(parts) != 6:
            continue
        try:
            index, uuid, name, total, used, free = parts
            gpus.append(
                {
                    "index": int(index),
                    "uuid": uuid,
                    "name": name,
                    "totalBytes": int(total) * MIB,
                    "usedBytes": int(used) * MIB,
                    "freeBytes": int(free) * MIB,
                    "selected": selected_identity is not None,
                }
            )
        except ValueError:
            continue
    return gpus, None if gpus else "nvidia-smi-malformed"


def sample_resources(pool, aligner=None) -> dict[str, Any]:
    ram, process_ram = _windows_memory() if os.name == "nt" else _posix_memory()
    gpus, gpu_error = _nvidia_gpus()
    selected_gpu = next((gpu for gpu in gpus if gpu.get("selected")), None)
    components = pool.model_residency()
    loaded_model_ids = [component["id"] for component in components]
    upscaler = pool.upscaler_residency()
    if upscaler is not None:
        components.append(upscaler)
    aligner_loaded = bool(aligner and aligner.loaded)
    if aligner_loaded and aligner.spec is not None:
        components.append(
            {
                "kind": "aligner",
                "id": aligner.spec.id,
                "loaded": True,
                "busy": aligner.busy,
                "evictable": False,
                "sticky": True,
                "resources": aligner.spec.resource_requirements(),
            }
        )
    return {
        "sampledAt": datetime.now(timezone.utc).isoformat(),
        "ram": ram,
        "processRamBytes": process_ram,
        "gpus": gpus,
        "selectedGpuIndex": selected_gpu.get("index") if selected_gpu else None,
        "gpuTelemetry": "nvidia-smi" if gpus else "unavailable",
        "gpuTelemetryError": gpu_error,
        "loadedModelIds": loaded_model_ids,
        "upscalerLoaded": upscaler is not None,
        "alignerLoaded": aligner_loaded,
        "components": components,
    }


def _estimate_bytes(spec, kind: str) -> int | None:
    value = spec.resource_requirements().get(kind, {}).get("estimatedBytes")
    return (
        int(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


def _sum_estimates(specs: list, kind: str) -> int | None:
    total = 0
    for spec in specs:
        value = _estimate_bytes(spec, kind)
        if value is None:
            return None
        total += value
    return total


def build_stack_load_plan(
    cfg: ServerConfig,
    pool,
    aligner,
    sampled: dict[str, Any],
    *,
    model_ids: list[str],
    upscale: bool,
    alignment: bool,
) -> dict[str, Any]:
    """Project the steady-state resource fit for an exact component stack."""
    desired_ids = list(dict.fromkeys(model_ids))
    if not desired_ids:
        raise ValueError("at least one model is required")
    for model_id in desired_ids:
        if model_id not in cfg.models:
            raise ValueError(f"unknown model {model_id!r}")
    if upscale and cfg.upscaler is None:
        raise ValueError("upscaler is unavailable")
    if alignment and cfg.aligner is None:
        raise ValueError("alignment is unavailable")

    replacement = pool.stack_replacement_plan(desired_ids)
    load_specs = []
    reuse = []
    load = []
    evict = []
    busy = []

    for model_id in replacement["loaded"]:
        reuse.append(
            {"kind": "model", "id": model_id, "task": cfg.models[model_id].task}
        )
    for model_id in replacement["load"]:
        load_specs.append(cfg.models[model_id])
        load.append(
            {"kind": "model", "id": model_id, "task": cfg.models[model_id].task}
        )

    evicted_specs = []
    for victim_id in replacement.get("evict", []):
        victim = cfg.models[victim_id]
        evicted_specs.append(victim)
        evict.append({"kind": "model", "id": victim_id, "task": victim.task})
    for busy_id in replacement.get("busy", []):
        busy.append(
            {"kind": "model", "id": busy_id, "task": cfg.models[busy_id].task}
        )

    if upscale:
        if pool.upscaler_loaded():
            reuse.append({"kind": "upscaler", "id": cfg.upscaler.id})
        else:
            load_specs.append(cfg.upscaler)
            load.append({"kind": "upscaler", "id": cfg.upscaler.id})

    if alignment:
        if aligner and aligner.loaded:
            reuse.append({"kind": "aligner", "id": cfg.aligner.id})
        else:
            load_specs.append(cfg.aligner)
            load.append({"kind": "aligner", "id": cfg.aligner.id})

    ram_free = (sampled.get("ram") or {}).get("freeBytes")
    gpus = [
        gpu for gpu in sampled.get("gpus", [])
        if isinstance(gpu, dict) and isinstance(gpu.get("freeBytes"), (int, float))
    ]
    gpu = next((value for value in gpus if value.get("selected")), None)
    if gpu is None:
        gpu = min(gpus, key=lambda value: value.get("index", math.inf), default=None)
    available = {
        "ram": ram_free if isinstance(ram_free, (int, float)) else None,
        "vram": gpu.get("freeBytes") if gpu else None,
    }
    requirements = {}
    ratios = []
    unknown_constraint = False
    for kind in ("ram", "vram"):
        additional = _sum_estimates(load_specs, kind)
        reclaimable = _sum_estimates(evicted_specs, kind)
        projected_free = (
            int(available[kind]) + int(reclaimable)
            if available[kind] is not None and reclaimable is not None
            else None
        )
        ratio = (
            projected_free / additional
            if additional not in (None, 0) and projected_free is not None
            else None
        )
        if ratio is not None:
            ratios.append(ratio)
        if additional is None or (additional > 0 and projected_free is None):
            unknown_constraint = True
        requirements[kind] = {
            "additionalBytes": additional,
            "reclaimableBytes": reclaimable,
            "projectedFreeBytes": projected_free,
            "ratio": ratio,
        }

    if not replacement.get("capacity", True):
        fit_status, fit_ratio = "insufficient", None
    elif busy:
        fit_status, fit_ratio = "busy", None
    elif not load_specs:
        fit_status, fit_ratio = "comfortable", None
    elif unknown_constraint or not ratios:
        fit_status, fit_ratio = "unknown", None
    else:
        fit_ratio = min(ratios)
        fit_status = (
            "comfortable" if fit_ratio >= 1.5
            else "tight" if fit_ratio >= 1.0
            else "insufficient"
        )
    return {
        "desiredModels": [
            {"id": model_id, "task": cfg.models[model_id].task}
            for model_id in desired_ids
        ],
        "upscale": upscale,
        "alignment": alignment,
        "residentLimit": cfg.resident_limit,
        "residentCapacitySatisfied": replacement.get("capacity", True),
        "reuse": reuse,
        "load": load,
        "evict": evict,
        "busy": busy,
        "requirements": requirements,
        "fit": {"status": fit_status, "ratio": fit_ratio, "estimated": True},
        "gpuIndex": gpu.get("index") if gpu else None,
        "sampledAt": sampled.get("sampledAt", ""),
    }


def build_load_plan(
    cfg: ServerConfig,
    pool,
    aligner,
    sampled: dict[str, Any],
    *,
    model_id: str,
    upscale: bool,
    adaptive_batching: bool,
) -> dict[str, Any]:
    model = cfg.models[model_id]
    if adaptive_batching and (
        cfg.aligner is None or model.segmentation is None
    ):
        raise ValueError("adaptive batching is unavailable for this model")
    plan = build_stack_load_plan(
        cfg,
        pool,
        aligner,
        sampled,
        model_ids=[model_id],
        upscale=upscale,
        alignment=adaptive_batching,
    )
    for field in ("reuse", "load", "evict", "busy"):
        plan[field] = [
            {key: value for key, value in action.items() if key != "task"}
            for action in plan[field]
        ]
    plan.update(
        {
            "modelId": model_id,
            "adaptiveBatching": adaptive_batching,
        }
    )
    return plan
