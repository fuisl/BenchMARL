"""GPU monitoring must follow CUDA allocation identities, including MIG slices."""

import sys
import uuid

import pytest

from scripts.slurm import monitor_gpu_memory


GPU_UUID = "GPU-5e3f679d-0465-059b-dc09-ab31760ca7dd"
MIG_UUID = "MIG-b41c5bd0-f3d2-564b-859a-e6f2e060b5ee"
LISTING = f"""GPU 0: NVIDIA A100 (UUID: GPU-d845bc89-502b-9954-b14b-3e51094e9c3a)
GPU 1: NVIDIA A100 (UUID: {GPU_UUID})
  MIG 3g.20gb Device 0: (UUID: {MIG_UUID})
  MIG 2g.10gb Device 1: (UUID: MIG-e8a45f6e-545a-5aab-ae71-db8d5b336de4)
"""
MEMORY_XML = b"""<nvidia_smi_log><gpu>
  <fb_memory_usage><used>5000 MiB</used><total>40960 MiB</total></fb_memory_usage>
  <utilization><gpu_util>75 %</gpu_util></utilization>
  <mig_devices><mig_device><index>0</index>
    <fb_memory_usage><used>1200 MiB</used><total>20096 MiB</total></fb_memory_usage>
  </mig_device></mig_devices>
</gpu></nvidia_smi_log>"""


@pytest.mark.parametrize(
    "visible,uuid,index", [("0", MIG_UUID, 0), ("1", GPU_UUID, None)]
)
def test_cuda_ordinal_resolves_allocated_memory(monkeypatch, visible, uuid, index):
    def check_output(command, **kwargs):
        if command[0] == sys.executable:
            assert command[-1] == "--device-uuid"
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == visible
            return uuid.split("-", 1)[1] + "\n"
        if command == ["nvidia-smi", "-L"]:
            return LISTING
        assert command == ["nvidia-smi", "-q", "-x", "-i", GPU_UUID]
        return MEMORY_XML

    monkeypatch.setattr(monitor_gpu_memory.subprocess, "check_output", check_output)
    assert monitor_gpu_memory.device_uuid(visible) == uuid
    parent, mig_index = monitor_gpu_memory.device_location(visible)
    assert (parent, mig_index) == (GPU_UUID, index)
    expected = (
        ("1200 MiB", "20096 MiB", "N/A")
        if index is not None
        else ("5000 MiB", "40960 MiB", "75 %")
    )
    assert monitor_gpu_memory.sample_memory(parent, mig_index) == expected


def test_explicit_mig_uuid_avoids_cuda(monkeypatch):
    def check_output(command, **kwargs):
        assert command == ["nvidia-smi", "-L"]
        return LISTING

    monkeypatch.setattr(monitor_gpu_memory.subprocess, "check_output", check_output)
    assert monitor_gpu_memory.device_location(MIG_UUID) == (GPU_UUID, 0)


def test_unknown_cuda_uuid_does_not_fall_back_to_numeric_gpu(monkeypatch):
    def check_output(command, **kwargs):
        return "unknown-uuid\n" if command[0] == sys.executable else LISTING

    monkeypatch.setattr(monitor_gpu_memory.subprocess, "check_output", check_output)
    with pytest.raises(ValueError, match="Allocated CUDA UUID is absent"):
        monitor_gpu_memory.device_location("0")


def test_multiple_devices_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        monitor_gpu_memory.device_location("0,1")


def test_driver_v2_reports_mig_uuid(monkeypatch):
    class Driver:
        def cuInit(self, flags):
            assert flags == 0
            return 0

        def cuDeviceGet(self, device, ordinal):
            assert ordinal == 0
            device._obj.value = 0
            return 0

        def cuDeviceGetUuid_v2(self, value, device):
            assert device.value == 0
            value._obj[:] = uuid.UUID(MIG_UUID.removeprefix("MIG-")).bytes
            return 0

        def cuDeviceGetUuid(self, value, device):
            pytest.fail("The older UUID query returns the parent GPU for this MIG")

    monkeypatch.setattr(monitor_gpu_memory.ctypes, "CDLL", lambda library: Driver())
    assert monitor_gpu_memory.cuda_device_uuid() == MIG_UUID.removeprefix("MIG-")
