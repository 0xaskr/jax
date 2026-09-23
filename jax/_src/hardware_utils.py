# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 文件职责：探测本机可用硬件，供 JAX 的后端初始化与平台选择使用。
# 它通过 PCI 设备表识别 Google TPU 的版本与芯片数量，并检查 NVIDIA/AMD GPU、
# 透明大页开关以及 /dev/shm 的大小。这些探测只读取 /sys、/dev 等路径与 KFD
# 拓扑信息，不加载任何加速器运行时，因此可以在导入期安全调用。

import enum
import logging
import os
import pathlib
import glob
import re

_GOOGLE_PCI_VENDOR_ID = '0x1ae0'

_NVIDIA_GPU_DEVICES = [
    '/dev/nvidia0',
    '/dev/nvidiactl',  # Docker/Kubernetes
    '/dev/dxg',  # WSL2
]


class TpuVersion(enum.IntEnum):
  # TPU v2、v3
  v2 = 0
  v3 = 1
  # 无公开名称（plc）
  plc = 2
  # TPU v4
  v4 = 3
  # TPU v5p
  v5p = 4
  # TPU v5e
  v5e = 5
  # TPU v6e
  v6e = 6
  # TPU7x
  tpu7x = 7
  # TPU8i
  tpu8i = 8
  # TPU8t
  tpu8t = 9


_TPU_PCI_DEVICE_IDS = {
    '0x0027': TpuVersion.v3,
    '0x0056': TpuVersion.plc,
    '0x005e': TpuVersion.v4,
    '0x0062': TpuVersion.v5p,
    '0x0063': TpuVersion.v5e,
    '0x006f': TpuVersion.v6e,
    '0x0076': TpuVersion.tpu7x,
    '0x0083': TpuVersion.tpu8i,
    '0x007c': TpuVersion.tpu8t,
}

def num_available_tpu_chips_and_device_id():
  """返回通过 PCI 挂载的 TPU 芯片数量与设备 id。"""
  num_chips = 0
  tpu_version = None
  for vendor_path in glob.glob('/sys/bus/pci/devices/*/vendor'):
    vendor_id = pathlib.Path(vendor_path).read_text().strip()
    if vendor_id != _GOOGLE_PCI_VENDOR_ID:
      continue

    device_path = os.path.join(os.path.dirname(vendor_path), 'device')
    device_id = pathlib.Path(device_path).read_text().strip()
    if device_id in _TPU_PCI_DEVICE_IDS:
      tpu_version = _TPU_PCI_DEVICE_IDS[device_id]
      num_chips += 1

  return num_chips, tpu_version


def has_visible_nvidia_gpu() -> bool:
  """若设备上存在可见的 NVIDIA GPU 则返回 True，否则返回 False。"""

  return any(os.path.exists(d) for d in _NVIDIA_GPU_DEVICES)


def transparent_hugepages_enabled() -> bool:
  # 有关透明大页的更多信息，参见
  # https://docs.kernel.org/admin-guide/mm/transhuge.html
  path = pathlib.Path('/sys/kernel/mm/transparent_hugepage/enabled')
  return path.exists() and path.read_text().strip() == '[always] madvise never'


logger = logging.getLogger(__name__)


def num_available_amd_gpus(stop_at: int | None = None) -> int:
  """统计通过 KFD 内核驱动可用的 AMD GPU 数量。

  本函数通过检查 KFD 内核驱动实体是否存在，作为判断 AMD GPU 是否可用
  的代理手段。在 WSL 环境中若 /dev/dxg 存在，该检查会为初始化门控把
  结果硬编码为 1 个 GPU。这一方案在性能、可靠性与简洁性之间取得了很好
  的折中。这类实体存在并不保证 GPU 可以通过 HIP 与 PJRT 使用，然而如果
  不额外启动一个设置可能相当复杂的进程来运行真正的 HIP 代码，我们也无
  法做得更好。而且我们不想现在就在当前进程内初始化 HIP，因为这样做
  可能破坏后续 PJRT 启动时 rocprofiler-sdk 的正常初始化。

  Args:
    stop_at: 若提供，则在找到这么多个 GPU 后停止计数。
             这样在只检查阈值时可以提前退出。

  Returns:
    检测到的 AMD GPU 数量（若提供了 stop_at，则上限为该值）。
  """
  try:
    if os.path.exists("/dev/dxg"):
      return 1

    kfd_nodes_path = "/sys/class/kfd/kfd/topology/nodes/"
    if not os.path.exists(kfd_nodes_path):
      return 0

    gpu_count = 0
    # 该正则在 "simd_count ##" 这类字符串中匹配并提取出数字 ##
    r_simd_count = re.compile(r"\bsimd_count\s+(\d+)\b", re.MULTILINE)
    # 参照 KFD 的实现，我们以非零的 simd_count 作为 GPU 的特征，
    # 见下面的链接：
    # https://github.com/torvalds/linux/blob/ea1013c1539270e372fc99854bc6e4d94eaeff66/drivers/gpu/drm/amd/amdkfd/kfd_topology.c#L941

    for node in os.listdir(kfd_nodes_path):
      node_props_path = os.path.join(kfd_nodes_path, node, "properties")

      if not os.path.exists(node_props_path):
        continue

      try:
        file_size = os.path.getsize(node_props_path)
        # 16KB 已远超合理上限
        if file_size <= 0 or file_size > 16 * 1024:
          continue

        with open(node_props_path, encoding="ascii") as f:
          match = r_simd_count.search(f.read())
          if match:
            simd_count = int(match.group(1))
            if simd_count > 0:
              gpu_count += 1
              if stop_at is not None and gpu_count >= stop_at:
                return gpu_count
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug(
          "Failed to read KFD node file '%s': %s", node_props_path, e
        )
        continue

  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Failed to count AMD GPUs: %s", e)
    return -1
  return gpu_count


def get_shm_size_in_mb():
  """获取 /dev/shm 的大小，单位为 MB。

  Returns:
    若 /dev/shm 存在则返回以 MB 为单位的大小，不存在则返回 None，出错返回 0。
  """
  try:
    shm_path = "/dev/shm"
    if not os.path.exists(shm_path):
      return 0

    stat = os.statvfs(shm_path)
    # 以字节为单位的总大小
    shm_size_bytes = stat.f_blocks * stat.f_frsize
    shm_size_mb = shm_size_bytes / (1024 * 1024)

    return shm_size_mb

  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.debug("Failed to check /dev/shm size: %s", e)
    return 0
