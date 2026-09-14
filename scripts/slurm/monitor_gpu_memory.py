"""Sample memory for one allocated GPU or MIG instance without a CUDA context."""

import argparse
import csv
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def device_location(visible):
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
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=0, help="0 samples continuously")
    args = parser.parse_args()
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    if "," in visible:
        raise ValueError("This monitor expects exactly one allocated GPU")
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
