"""Sample allocated GPU memory; resolve CUDA ordinals in a temporary process."""

import argparse
import csv
import ctypes
import os
import re
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def cuda_device_uuid():
    """Use the driver v2 UUID, which identifies a MIG instance, not its parent."""
    cuda = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    value = (ctypes.c_ubyte * 16)()
    for function, arguments in (
        (cuda.cuInit, (0,)),
        (cuda.cuDeviceGet, (ctypes.byref(device), 0)),
        (cuda.cuDeviceGetUuid_v2, (ctypes.byref(value), device)),
    ):
        result = function(*arguments)
        if result != 0:
            raise RuntimeError(f"{function.__name__} failed with CUDA error {result}")
    return str(uuid.UUID(bytes=bytes(value)))


def device_uuid(visible):
    """Resolve CUDA's device zero, which need not be nvidia-smi's GPU zero."""
    if "," in visible:
        raise ValueError("This monitor expects exactly one allocated GPU")
    if visible.startswith(("GPU-", "MIG-")):
        return visible
    if not visible.isdecimal():
        raise ValueError(f"Unsupported CUDA_VISIBLE_DEVICES value: {visible}")

    # Exit the resolver before sampling so its CUDA context consumes no memory.
    allocated_uuid = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), "--device-uuid"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": visible},
        text=True,
    ).strip()
    listing = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    for listed_uuid in re.findall(r"UUID: ((?:GPU|MIG)-[^)]+)", listing):
        if listed_uuid.split("-", 1)[1] == allocated_uuid:
            return listed_uuid
    raise ValueError(f"Allocated CUDA UUID is absent from nvidia-smi: {allocated_uuid}")


def device_location(visible):
    visible = device_uuid(visible)
    if not visible.startswith("MIG-"):
        return visible, None
    listing = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    parent = None
    for line in listing.splitlines():
        gpu = re.search(r"^GPU \d+:.*\(UUID: (GPU-[^)]+)\)", line)
        if gpu:
            parent = gpu.group(1)
        mig = re.search(r"Device\s+(\d+): \(UUID: (MIG-[^)]+)\)", line)
        if mig and mig.group(2) == visible:
            return parent, int(mig.group(1))
    raise ValueError(f"Allocated MIG device is absent from nvidia-smi: {visible}")


def sample_memory(parent, mig_index):
    root = ET.fromstring(
        subprocess.check_output(["nvidia-smi", "-q", "-x", "-i", parent])
    )
    gpu = root.find("gpu")
    if mig_index is None:
        device = gpu
    else:
        device = next(
            entry
            for entry in gpu.findall(".//mig_device")
            if int(entry.findtext("index")) == mig_index
        )
    return (
        device.findtext("fb_memory_usage/used"),
        device.findtext("fb_memory_usage/total"),
        gpu.findtext("utilization/gpu_util") if mig_index is None else "N/A",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--device-uuid", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, default=0, help="0 samples continuously")
    args = parser.parse_args()
    if args.device_uuid:
        print(cuda_device_uuid())
        sys.exit(0)
    if args.output is None:
        parser.error("output is required")
    visible = device_uuid(os.environ["CUDA_VISIBLE_DEVICES"])
    parent, mig_index = device_location(visible)
    with args.output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp",
                "index",
                "uuid",
                "memory.used [MiB]",
                "memory.total [MiB]",
                "utilization.gpu [%]",
            ]
        )
        count = 0
        while True:
            used, total, utilization = sample_memory(parent, mig_index)
            writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    mig_index,
                    visible,
                    used,
                    total,
                    utilization,
                ]
            )
            file.flush()
            count += 1
            if count == args.samples:
                break
            time.sleep(5)
