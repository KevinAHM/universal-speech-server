"""Interactive process-wide GPU selection for the startup scripts."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, TextIO


DEFAULT_RUNTIME_MANIFEST = (
    Path(__file__).resolve().parent.parent / "runtime" / "crispasr" / "installed.json"
)


@dataclass(frozen=True)
class GPUDevice:
    index: str
    name: str


def enumerate_nvidia_devices(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[GPUDevice]:
    try:
        result = runner(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    devices: list[GPUDevice] = []
    for line in result.stdout.splitlines():
        index, separator, name = line.partition(",")
        if not separator or not index.strip().isdigit() or not name.strip():
            continue
        devices.append(GPUDevice(index=index.strip(), name=name.strip()))
    return devices


class _VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(_VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def _vulkan_library_name() -> str:
    if sys.platform == "win32":
        return "vulkan-1.dll"
    return ctypes.util.find_library("vulkan") or (
        "libvulkan.1.dylib" if sys.platform == "darwin" else "libvulkan.so.1"
    )


def enumerate_vulkan_devices(loader=None) -> list[GPUDevice]:
    """Return Vulkan physical devices in the loader's native index order."""
    instance = ctypes.c_void_p()
    try:
        if loader is None:
            loader_type = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
            loader = loader_type(_vulkan_library_name())
        loader.vkCreateInstance.argtypes = [
            ctypes.POINTER(_VkInstanceCreateInfo),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        loader.vkCreateInstance.restype = ctypes.c_int32
        loader.vkEnumeratePhysicalDevices.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        loader.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
        loader.vkGetPhysicalDeviceProperties.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        loader.vkGetPhysicalDeviceProperties.restype = None
        loader.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        loader.vkDestroyInstance.restype = None

        application = _VkApplicationInfo(
            sType=0,
            pApplicationName=b"Universal Speech Server",
            applicationVersion=1,
            pEngineName=b"CrispASR",
            engineVersion=1,
            apiVersion=1 << 22,
        )
        create_info = _VkInstanceCreateInfo(
            sType=1,
            pApplicationInfo=ctypes.pointer(application),
        )
        if (
            loader.vkCreateInstance(
                ctypes.byref(create_info), None, ctypes.byref(instance)
            )
            != 0
        ):
            return []
        count = ctypes.c_uint32()
        if loader.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None) != 0:
            return []
        if count.value == 0:
            return []
        handles = (ctypes.c_void_p * count.value)()
        if (
            loader.vkEnumeratePhysicalDevices(
                instance, ctypes.byref(count), handles
            )
            != 0
        ):
            return []
        devices = []
        for index, handle in enumerate(handles[: count.value]):
            # VkPhysicalDeviceProperties begins with five uint32 fields followed
            # by deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]. A generous opaque
            # buffer avoids coupling this probe to the large trailing limits struct.
            properties = ctypes.create_string_buffer(4096)
            loader.vkGetPhysicalDeviceProperties(handle, properties)
            raw_name = properties.raw[20 : 20 + 256].split(b"\0", 1)[0]
            name = raw_name.decode("utf-8", errors="replace").strip()
            devices.append(
                GPUDevice(index=str(index), name=name or f"Vulkan GPU {index}")
            )
        return devices
    except (AttributeError, OSError, TypeError, ValueError):
        return []
    finally:
        if instance.value and loader is not None:
            try:
                loader.vkDestroyInstance(instance, None)
            except (AttributeError, OSError, TypeError, ValueError):
                pass


def backend_for_target(target: str) -> str | None:
    normalized = target.strip().lower()
    if "cuda" in normalized or normalized == "nvidia":
        return "cuda"
    if "vulkan" in normalized or normalized in {"amd", "intel", "gpu", "other"}:
        return "vulkan"
    return None


def _manifest_target(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return ""
    target = value.get("target") if isinstance(value, dict) else None
    return target if isinstance(target, str) else ""


def _library_backend(path: str) -> str | None:
    try:
        directory = Path(path).expanduser().resolve().parent
        names = {item.name.lower() for item in directory.iterdir() if item.is_file()}
    except OSError:
        return None
    if any(name.startswith(("ggml-cuda", "libggml-cuda")) for name in names):
        return "cuda"
    if any(name.startswith(("ggml-vulkan", "libggml-vulkan")) for name in names):
        return "vulkan"
    return None


def choose_device(
    devices: list[GPUDevice],
    *,
    backend: str,
    requested: str = "",
    interactive: bool | None = None,
    input_fn: Callable[[], str] = input,
    output: TextIO = sys.stderr,
) -> str | None:
    if requested:
        if devices and requested not in {device.index for device in devices}:
            available = ", ".join(device.index for device in devices)
            raise ValueError(
                f"GPU device {requested!r} is unavailable; "
                f"available indices: {available}"
            )
        print(f"Using requested {backend.upper()} GPU device {requested}.", file=output)
        return requested
    if not devices:
        print(
            f"Could not enumerate {backend.upper()} GPUs; "
            "leaving device selection to CrispASR.",
            file=output,
        )
        return None
    if len(devices) == 1:
        device = devices[0]
        print(
            f"Using the only detected {backend.upper()} GPU: "
            f"[{device.index}] {device.name}",
            file=output,
        )
        return device.index
    print(f"Detected {len(devices)} {backend.upper()} GPUs:", file=output)
    for device in devices:
        print(f"  [{device.index}] {device.name}", file=output)
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        first = devices[0]
        print(
            f"No interactive terminal is available; "
            f"using [{first.index}] {first.name}.",
            file=output,
        )
        return first.index
    valid = {device.index for device in devices}
    default = devices[0].index
    while True:
        print(f"Select GPU [{default}]: ", end="", file=output, flush=True)
        try:
            choice = input_fn().strip() or default
        except EOFError:
            choice = default
        if choice in valid:
            return choice
        print(f"Enter one of: {', '.join(sorted(valid))}", file=output)


def startup_assignment(
    *,
    target: str,
    manifest: Path,
    environ: Mapping[str, str] = os.environ,
    interactive: bool | None = None,
    input_fn: Callable[[], str] = input,
    output: TextIO = sys.stderr,
) -> str:
    explicit_library = environ.get("SPEECH_SERVER_LIB", "").strip()
    backend = _library_backend(explicit_library) if explicit_library else None
    if backend is None:
        actual_target = _manifest_target(manifest) or target
        backend = backend_for_target(actual_target)
    if backend is None:
        return ""
    variable = (
        "CUDA_VISIBLE_DEVICES" if backend == "cuda" else "GGML_VK_VISIBLE_DEVICES"
    )
    existing = environ.get(variable, "").strip()
    if existing:
        print(f"Using existing {variable}={existing}.", file=output)
        return f"{variable}={existing}"
    devices = (
        enumerate_nvidia_devices() if backend == "cuda" else enumerate_vulkan_devices()
    )
    selected = choose_device(
        devices,
        backend=backend,
        requested=environ.get("SPEECH_SERVER_GPU_DEVICE", "").strip(),
        interactive=interactive,
        input_fn=input_fn,
        output=output,
    )
    return f"{variable}={selected}" if selected is not None else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="auto")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest or Path(
        os.getenv("SPEECH_SERVER_RUNTIME_MANIFEST", str(DEFAULT_RUNTIME_MANIFEST))
    )
    try:
        assignment = startup_assignment(target=args.target, manifest=manifest)
    except ValueError as exc:
        parser.error(str(exc))
    if assignment:
        print(assignment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
