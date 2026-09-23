# Copyright 2025 The JAX Authors.
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

# 文件职责：描述 TPU 硬件规格并提供查询接口，是 JAX 中了解“当前或目标 TPU 芯片
# 长什么样”的单一事实来源。
# 它把 `device_kind` 映射为 `ChipVersion` 枚举，给出每个 TensorCore 的 lane/sublane
# 布局、MXU 列宽与数量、VMEM/CMEM/SMEM/HBM 容量和 bf16/int8/fp8/int4 峰值算力，
# 并区分 Megacore 与 split 两种多核使用模式；`Tiling`/`infer_tiling` 还据此推断
# 内存布局的分块因子。`get_tpu_info()` 查当前设备，`get_tpu_info_for_chip()` 查指定芯片。

"""对外暴露 TPU 硬件信息。"""

import dataclasses
import enum
from typing import cast
from collections.abc import Callable

from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import mesh as mesh_lib
from jax._src import util as jax_util
from jax._src.interpreters import pxla

import numpy as np

class ChipVersionBase:
  pass


class ChipVersion(ChipVersionBase, enum.Enum):
  """TPU 芯片版本。

  下表汇总了各 TPU 版本之间的差异：

  +------+--------------------------+-----------+---------------+
  | 版本 | 每芯片物理 TensorCore 数 | Lite 芯片 | Megacore 支持 |
  +======+==========================+===========+===============+
  | v2   | 2                        | 否        | 否            |
  +------+--------------------------+-----------+---------------+
  | v3   | 2                        | 否        | 否            |
  +------+--------------------------+-----------+---------------+
  | v4i  | 1                        | 是        | 否            |
  +------+--------------------------+-----------+---------------+
  | v4   | 2                        | 否        | 是            |
  +------+--------------------------+-----------+---------------+
  | v5e  | 1                        | 是        | 否            |
  +------+--------------------------+-----------+---------------+
  | v5p  | 2                        | 否        | 是            |
  +------+--------------------------+-----------+---------------+
  | v6e  | 1                        | 是        | 否            |
  +------+--------------------------+-----------+---------------+
  | 7    | 2                        | 否        | 否            |
  +------+--------------------------+-----------+---------------+
  | 7x   | 2                        | 否        | 否            |
  +------+--------------------------+-----------+---------------+
  | 8i   | 2                        | 否        | 否            |
  +------+--------------------------+-----------+---------------+
  | 8t   | 1                        | 否        | 否            |
  +------+--------------------------+-----------+---------------+
  """

  TPU_V2 = "v2"
  TPU_V3 = "v3"
  TPU_V4I = "v4i"
  TPU_V4 = "v4"
  TPU_V5E = "v5e"
  TPU_V5P = "v5p"
  TPU_V6E = "v6e"
  TPU_7 = "7"
  TPU_7X = "7x"
  TPU_8I = "8i"
  TPU_8T = "8t"

  def __str__(self) -> str:
    return self.value

  @property
  def num_physical_tensor_cores_per_chip(self) -> int:
    match self:
      case (
          ChipVersion.TPU_V2
          | ChipVersion.TPU_V3
          | ChipVersion.TPU_V4
          | ChipVersion.TPU_V5P
          | ChipVersion.TPU_7
          | ChipVersion.TPU_7X
          | ChipVersion.TPU_8I
      ):
        return 2
      case (
          ChipVersion.TPU_V4I
          | ChipVersion.TPU_V5E
          | ChipVersion.TPU_V6E
          | ChipVersion.TPU_8T
      ):
        return 1

  @property
  def supports_megacore(self) -> bool:
    match self:
      case ChipVersion.TPU_V4 | ChipVersion.TPU_V5P:
        return True
      case _:
        return False

  @property
  def is_lite(self) -> bool:
    match self:
      case ChipVersion.TPU_V4I | ChipVersion.TPU_V5E | ChipVersion.TPU_V6E:
        return True
      case _:
        return False


def chip_version_from_device_kind(device_kind: str) -> ChipVersion | None:
  match device_kind:
    case "TPU v2":
      return ChipVersion.TPU_V2
    case "TPU v3":
      return ChipVersion.TPU_V3
    case "TPU v4":
      return ChipVersion.TPU_V4
    case "TPU v4 lite":
      return ChipVersion.TPU_V4I
    case "TPU v5e" | "TPU v5 lite":
      return ChipVersion.TPU_V5E
    case "TPU v5" | "TPU v5p":
      return ChipVersion.TPU_V5P
    case "TPU v6e" | "TPU v6 lite":
      return ChipVersion.TPU_V6E
    case "TPU7":
      return ChipVersion.TPU_7
    case "TPU7x":
      return ChipVersion.TPU_7X
    case "TPU8i":
      return ChipVersion.TPU_8I
    case "TPU8t":
      return ChipVersion.TPU_8T
    case _:
      return None


@dataclasses.dataclass(frozen=True, kw_only=True)
class SparseCoreInfo:
  """SparseCore 特有的信息。"""

  num_cores: int
  num_subcores: int
  num_lanes: int
  vmem_capacity_bytes: int
  dma_granule_size_bytes: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class TpuInfo:
  """TPU 硬件信息。

  注意：这里的所有信息都是按 TensorCore 计的，需要乘以 `num_cores`
  才能得到整块芯片的总量。
  """

  chip_version: ChipVersionBase
  generation: int
  num_cores: int
  num_lanes: int
  num_sublanes: int
  mxu_column_size: int
  # 每个核心可用的 MXU 数量。
  num_mxus: int
  # 每个 MXU 可用的、形状为 (num_sublanes, mxu_column_size) 的
  # 32 位累加器缓冲区数量。
  num_accumulators: int
  vmem_capacity_bytes: int
  cmem_capacity_bytes: int
  smem_capacity_bytes: int
  hbm_capacity_bytes: int
  mem_bw_bytes_per_second: int
  bf16_ops_per_second: int
  int8_ops_per_second: int
  fp8_ops_per_second: int
  int4_ops_per_second: int

  sparse_core: SparseCoreInfo | None = None

  @property
  def is_lite(self) -> bool:
    return cast(ChipVersion, self.chip_version).is_lite

  @property
  def is_split_chip(self) -> bool:
    """若芯片是多核芯片但以单核模式使用，则返回 True。

    某些 TPU 代次（例如 v4、v5p）每块芯片上有多个 TensorCore。
    这些芯片可以工作在两种模式下：
    1. `Megacore` 模式，即把多个核心合并成单个逻辑
    设备（若支持）。
    2. `split` 模式，即把每个核心视为独立的逻辑
    设备。

    若芯片处于 `split` 模式（情况 2），该属性返回 True。
    """
    return self.num_cores == 1 and (
        cast(ChipVersion, self.chip_version).num_physical_tensor_cores_per_chip
        > 1
    )

  @property
  def is_megacore(self) -> bool:
    """若芯片被配置为 Megacore 模式，则返回 True。

    Megacore 模式意味着两个物理 TensorCore 被合并成单个
    逻辑设备。
    """
    return self.num_cores > 1

  def is_matmul_supported(
      self,
      lhs_dtype: dtypes.DTypeLike,
      rhs_dtype: dtypes.DTypeLike,
  ) -> bool:
    """返回该芯片是否原生支持给定输入数据类型上的 matmul（无需类型转换）。"""
    lhs_dtype = dtypes.dtype(lhs_dtype)
    rhs_dtype = dtypes.dtype(rhs_dtype)

    F32 = np.float32
    BF16 = dtypes.bfloat16
    S8 = np.int8
    U8 = np.uint8
    F8E4M3B11FNUZ = dtypes.float8_e4m3b11fnuz
    F8E4M3FN = dtypes.float8_e4m3fn
    F8E5M2 = dtypes.float8_e5m2
    S4 = dtypes.int4
    U4 = dtypes.uint4
    match self.chip_version:
      case ChipVersion.TPU_V2 | ChipVersion.TPU_V3:
        return lhs_dtype == rhs_dtype == F32
      case ChipVersion.TPU_V4I | ChipVersion.TPU_V4:
        return lhs_dtype in (F32, BF16) and rhs_dtype in (F32, BF16, S8)
      case ChipVersion.TPU_V5E | ChipVersion.TPU_V5P | ChipVersion.TPU_V6E:
        return (
            (
                lhs_dtype in (F32, BF16, F8E5M2, F8E4M3B11FNUZ)
                and rhs_dtype in (F32, BF16, F8E5M2, F8E4M3B11FNUZ)
            )
            or (lhs_dtype in (U8, S8) and rhs_dtype in (U8, S8))
            or (lhs_dtype in (U4, S4) and rhs_dtype in (U4, S4))
        )
      case ChipVersion.TPU_7 | ChipVersion.TPU_7X:
        return (
            lhs_dtype in (F32, BF16)
            and rhs_dtype in (F32, BF16, F8E5M2, F8E4M3FN)
        ) or (
            lhs_dtype in (F8E5M2, F8E4M3FN) and rhs_dtype in (F8E5M2, F8E4M3FN)
        )
      case ChipVersion.TPU_8I:
        return (
            lhs_dtype in (F32, BF16)
            and rhs_dtype in (F32, BF16, F8E5M2, F8E4M3FN, U4, S4)
        ) or (
            lhs_dtype in (F8E5M2, F8E4M3FN)
            and rhs_dtype in (F8E5M2, F8E4M3FN, U4, S4)
        )
      case ChipVersion.TPU_8T:
        # TODO: b/543848756 - 补充其余的数据类型。
        return (lhs_dtype in (F32, BF16) and rhs_dtype in (F32, BF16, S4)) or (
            lhs_dtype in (F32, BF16, F8E5M2, F8E4M3FN)
            and rhs_dtype in (F8E5M2, F8E4M3FN, S4)
        )
      case _:
        return False

  def get_sublane_tiling(self, dtype: dtypes.DType) -> int:
    """返回给定 itemsize 的 sublane 分块（tiling）。

    注意这是一个启发式规则，取决于 XLA flag 的设置。
    """
    bitwidth = dtypes.itemsize_bits(dtype)
    if self.generation < 7:
      # 注意：在 TPU7x 之前，XLA 默认不启用大 2nd minor 分块，但可以通过设置
      # flag `xla_tpu_enable_large_2nd_minor_layout_for_x16` 来启用。
      if bitwidth == 16 or bitwidth == 32:
        return self.num_sublanes
      else:
        # 其他类型则启用大 2nd minor 分块。
        return self.num_sublanes * (32 // bitwidth)
    # 从 TPU7x 开始，XLA 默认允许大 2nd minor 分块。
    if self.generation == 7 or self.generation == 8:
      return self.num_sublanes * (32 // bitwidth)
    raise NotImplementedError("TPU generation is not supported")


def is_tpu_device() -> bool:
  return chip_version_from_device_kind(get_device_kind()) is not None


registry: dict[str, Callable[[], TpuInfo]] = {}


def _get_tpu_info_impl(chip_version: ChipVersion, num_cores: int) -> TpuInfo:
  """返回给定芯片版本与核心数下的 TPU 硬件信息。

  注意：这里的所有信息都是*按 TensorCore 计*的，需要乘以 `num_cores`
  才能得到整块芯片的总量。

  Args:
    chip_version: TPU 芯片版本。
    num_cores: 该配置下每块芯片的 TensorCore 数量。它受
      TPU 版本以及是否启用 Megacore 的影响。
  """
  # 所有 TensorCore 共用的参数
  NUM_LANES = 128
  NUM_SUBLANES = 8
  MXU_COLUMN_SIZE_GEN_LT_6 = 128
  MXU_COLUMN_SIZE_GEN_GE_6 = 256
  tensor_cores_per_chip = chip_version.num_physical_tensor_cores_per_chip
  match chip_version:
    case ChipVersion.TPU_V2:
      return TpuInfo(
          chip_version=chip_version,
          generation=2,
          num_cores=num_cores,
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          num_mxus=1,
          num_accumulators=0,  # 不可用
          vmem_capacity_bytes=16 * 1024 * 1024,  # 每个核心 16 MiB
          cmem_capacity_bytes=0,
          smem_capacity_bytes=16 * 1024,  # 每个核心 16 KiB
          hbm_capacity_bytes=int(16_000_000_000 // tensor_cores_per_chip),
          mem_bw_bytes_per_second=int(7.16e11 // tensor_cores_per_chip),
          bf16_ops_per_second=int(4.6e13 // tensor_cores_per_chip),
          int8_ops_per_second=0,  # 不可用
          fp8_ops_per_second=0,  # 不可用
          int4_ops_per_second=0,  # 不可用
      )
    case ChipVersion.TPU_V3:
      return TpuInfo(
          chip_version=chip_version,
          generation=3,
          num_cores=num_cores,
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          num_mxus=2,
          num_accumulators=0,  # 不可用
          vmem_capacity_bytes=16 * 1024 * 1024,  # 每个核心 16 MiB
          cmem_capacity_bytes=0,
          smem_capacity_bytes=16 * 1024,  # 每个核心 16 KiB
          hbm_capacity_bytes=34_400_000_000 // tensor_cores_per_chip,
          mem_bw_bytes_per_second=int(8.25e11 // tensor_cores_per_chip),
          bf16_ops_per_second=int(1.40e14 // tensor_cores_per_chip),
          int8_ops_per_second=0,  # 不可用
          fp8_ops_per_second=0,  # 不可用
          int4_ops_per_second=0,  # 不可用
      )
    case ChipVersion.TPU_V4I:
      return TpuInfo(
          chip_version=chip_version,
          generation=4,
          num_cores=num_cores,
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          num_mxus=4,
          num_accumulators=0,  # 不可用
          vmem_capacity_bytes=16 * 1024 * 1024,  # 每个核心 16 MiB
          cmem_capacity_bytes=134_000_000,
          smem_capacity_bytes=1024 * 1024,  # 每个核心 1 MiB
          hbm_capacity_bytes=8_590_000_000,
          mem_bw_bytes_per_second=int(6.14e11),
          bf16_ops_per_second=int(1.37e14),
          int8_ops_per_second=0,  # 不可用
          fp8_ops_per_second=0,  # 不可用
          int4_ops_per_second=0,  # 不可用
      )
    case ChipVersion.TPU_V4:
      return TpuInfo(
          chip_version=chip_version,
          generation=4,
          num_cores=num_cores,
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          num_mxus=4,
          num_accumulators=0,  # 不可用
          vmem_capacity_bytes=16 * 1024 * 1024,  # 每个核心 16 MiB
          cmem_capacity_bytes=134_000_000 // tensor_cores_per_chip,
          smem_capacity_bytes=1024 * 1024,  # 每个核心 1 MiB
          hbm_capacity_bytes=34_400_000_000 // tensor_cores_per_chip,
          mem_bw_bytes_per_second=int(1.23e12 // tensor_cores_per_chip),
          bf16_ops_per_second=int(2.75e14 // tensor_cores_per_chip),
          int8_ops_per_second=0,  # 不可用
          fp8_ops_per_second=0,  # 不可用
          int4_ops_per_second=0,  # 不可用
      )
    case ChipVersion.TPU_V5E:
      return TpuInfo(
          chip_version=chip_version,
          generation=5,
          num_cores=num_cores,
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          num_mxus=4,
          num_accumulators=0,  # 不可用
          vmem_capacity_bytes=128 * 1024 * 1024,  # 每个核心 128 MiB
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 每个核心 1 MiB
          hbm_capacity_bytes=17_200_000_000,
          mem_bw_bytes_per_second=int(8.20e11),
          bf16_ops_per_second=int(1.97e14),
          int8_ops_per_second=int(3.94e14),
          fp8_ops_per_second=0,  # 不可用
          int4_ops_per_second=int(7.88e14),
      )
    case ChipVersion.TPU_V5P:
      return TpuInfo(
          chip_version=chip_version,
          generation=5,
          num_cores=num_cores,
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_LT_6,
          num_mxus=4,
          num_accumulators=0,  # 不可用
          vmem_capacity_bytes=64 * 1024 * 1024,  # 每个核心 64 MiB
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 每个核心 1 MiB
          hbm_capacity_bytes=103_000_000_000 // tensor_cores_per_chip,
          mem_bw_bytes_per_second=int(2.46e12 // tensor_cores_per_chip),
          bf16_ops_per_second=int(4.59e14 // tensor_cores_per_chip),
          int8_ops_per_second=int(9.18e14 // tensor_cores_per_chip),
          fp8_ops_per_second=0,  # 不可用
          int4_ops_per_second=int(1.84e15 // tensor_cores_per_chip),
          sparse_core=SparseCoreInfo(
              num_cores=4,
              num_subcores=16,
              num_lanes=8,
              vmem_capacity_bytes=512 * 1024,  # 每个向量 subcore 512 KiB
              dma_granule_size_bytes=32,
          ),
      )
    case ChipVersion.TPU_V6E:
      return TpuInfo(
          chip_version=chip_version,
          generation=6,
          num_cores=num_cores,
          num_lanes=NUM_LANES,
          num_sublanes=NUM_SUBLANES,
          mxu_column_size=MXU_COLUMN_SIZE_GEN_GE_6,
          num_mxus=2,
          num_accumulators=0,  # 不可用
          vmem_capacity_bytes=128 * 1024 * 1024,  # 每个核心 128 MiB
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 每个核心 1 MiB
          hbm_capacity_bytes=34_400_000_000,
          mem_bw_bytes_per_second=int(1.64e12),
          bf16_ops_per_second=int(9.20e14),
          int8_ops_per_second=int(1.84e15),
          fp8_ops_per_second=int(9.20e14),
          int4_ops_per_second=int(3.68e15),
          sparse_core=SparseCoreInfo(
              num_cores=2,
              num_subcores=16,
              num_lanes=8,
              vmem_capacity_bytes=256 * 1024,  # 每个向量 subcore 256 KiB
              dma_granule_size_bytes=32,
          ),
      )
    case ChipVersion.TPU_7 | ChipVersion.TPU_7X:
      return TpuInfo(
          chip_version=chip_version,
          generation=7,
          num_cores=num_cores,
          num_lanes=128,
          num_sublanes=8,
          mxu_column_size=256,
          num_mxus=2,
          num_accumulators=128,
          vmem_capacity_bytes=64 * 1024 * 1024,  # 每个核心 64 MiB
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 每个核心 1 MiB
          hbm_capacity_bytes=206_000_000_000 // tensor_cores_per_chip,
          mem_bw_bytes_per_second=int(7.40e12 // tensor_cores_per_chip),
          bf16_ops_per_second=int(2.31e15 // tensor_cores_per_chip),
          int8_ops_per_second=0,  # 不可用
          fp8_ops_per_second=int(4.60e15 // tensor_cores_per_chip),
          int4_ops_per_second=0,  # 不可用
          sparse_core=SparseCoreInfo(
              num_cores=2,
              num_subcores=16,
              num_lanes=16,
              vmem_capacity_bytes=512 * 1024,  # 每个向量 subcore 512 KiB
              dma_granule_size_bytes=32,
          ),
      )
    case ChipVersion.TPU_8I:
      return TpuInfo(
          chip_version=chip_version,
          generation=8,
          num_cores=num_cores,
          num_lanes=128,
          num_sublanes=8,
          mxu_column_size=256,
          num_mxus=2,
          num_accumulators=256,
          vmem_capacity_bytes=192 * 1024 * 1024,  # 每个核心 192 MiB
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 每个核心 1 MiB
          hbm_capacity_bytes=309_000_000_000 // tensor_cores_per_chip,
          mem_bw_bytes_per_second=int(8.60e12 // tensor_cores_per_chip),
          bf16_ops_per_second=int(1.101e15 // tensor_cores_per_chip),
          int8_ops_per_second=0,  # 不可用
          fp8_ops_per_second=int(8.808e15 // tensor_cores_per_chip),
          int4_ops_per_second=0,  # 不可用
          sparse_core=SparseCoreInfo(
              num_cores=1,
              num_subcores=4,
              num_lanes=16,
              vmem_capacity_bytes=512 * 1024,  # 每个向量 subcore 512 KiB
              dma_granule_size_bytes=64,
          ),
      )
    case ChipVersion.TPU_8T:
      return TpuInfo(
          chip_version=chip_version,
          generation=8,
          num_cores=num_cores,
          num_lanes=128,
          num_sublanes=16,
          mxu_column_size=256,
          num_mxus=2,
          num_accumulators=256,  #  待确认
          vmem_capacity_bytes=128 * 1024 * 1024,  # 每个核心 128 MiB
          cmem_capacity_bytes=0,
          smem_capacity_bytes=1024 * 1024,  # 每个核心 1 MiB
          hbm_capacity_bytes=231_000_000_000 // tensor_cores_per_chip,
          mem_bw_bytes_per_second=int(6.4e12 // tensor_cores_per_chip),
          bf16_ops_per_second=int(0.9961e15 // tensor_cores_per_chip),
          int8_ops_per_second=int(0.9961e15 // tensor_cores_per_chip),
          fp8_ops_per_second=int(5.9769e15 // tensor_cores_per_chip),
          int4_ops_per_second=int(11.9538e15 // tensor_cores_per_chip),
          sparse_core=SparseCoreInfo(
              num_cores=2,
              num_subcores=16,
              num_lanes=16,
              vmem_capacity_bytes=256 * 1024,  # 每个向量 subcore 256 KiB
              dma_granule_size_bytes=64,
          ),
      )
    case _:
      raise ValueError(f"Unsupported TPU chip version: {chip_version}")


@jax_util.cache(trace_context_in_key=True)
def get_tpu_info() -> TpuInfo:
  """返回当前设备的 TPU 硬件信息。

  注意：这里的所有信息都是*按 TensorCore 计*的，需要乘以 `num_cores`
  才能得到整块芯片的总量。
  """
  device_kind = get_device_kind()
  chip_version = chip_version_from_device_kind(device_kind)
  if chip_version is None:
    if device_kind in registry:
      return registry[device_kind]()
    raise ValueError(
        f"Unsupported TPU device kind: {device_kind}. If you are not running "
        "on a TPU device, you need to wrap your code in a "
        "`jax.sharding.use_abstract_mesh` context manager whose `AbstractMesh` "
        "argument specifies the exact TPU version you intend to target."
    )
  return _get_tpu_info_impl(chip_version, get_num_device_cores())


@jax_util.cache(trace_context_in_key=True)
def get_tpu_info_for_chip(
    chip_version: ChipVersion, num_tensor_cores_per_logical_device: int
) -> TpuInfo:
  """返回给定 TPU 芯片版本的 TPU 硬件信息。

  注意：这里的所有信息都是*按 TensorCore 计*的，需要乘以
  `num_tensor_cores_per_logical_device` 才能得到整块芯片的总量。

  Args:
    chip_version: TPU 芯片版本。
    num_tensor_cores_per_logical_device: 请求的配置下每个逻辑设备上的
      TensorCore 数量。对单核芯片（TPU_V4I、TPU_V5E、TPU_V6E），
      该值应为 1。对支持 Megacore 的双核芯片（TPU_V4、TPU_V5P），
      该值可以是 2（Megacore 模式）或 1（split 模式）。对不支持
      Megacore 的双核芯片（TPU_V2、TPU_V3、TPU_7X），该值必须
      为 1。
  """
  if (
      chip_version.is_lite
      or chip_version
      in {
          ChipVersion.TPU_V2,
          ChipVersion.TPU_V3,
          ChipVersion.TPU_7,
          ChipVersion.TPU_7X,
          ChipVersion.TPU_8I,
          ChipVersion.TPU_8T,
      }
  ) and num_tensor_cores_per_logical_device != 1:
    raise ValueError(
        "Lite chips, single core chips, and dual-core chips that do not support"
        " Megacore must have num_tensor_cores_per_logical_device=1, but got"
        f" {num_tensor_cores_per_logical_device}."
    )

  return _get_tpu_info_impl(chip_version, num_tensor_cores_per_logical_device)


# TODO(sharadmv): 泛化 Tiling 以覆盖各种选项
# （compact 2nd minor、large 2nd minor、常规分块）
class Tiling(enum.Enum):
  COMPACT = enum.auto()
  SPARSE_CORE = enum.auto()

  @property
  def shape(self) -> tuple[int, ...]:
    # TODO(slebedev): 使用 ``get_tpu_info()`` 而不是硬编码这些值。
    match self:
      case Tiling.COMPACT:
        return (8, 128)
      case Tiling.SPARSE_CORE:
        return (8,)


def _get_tiling_factor(src: int, max_tiling: int, packing: int) -> int:
  # 这大致对应 infer-memref-layout 中的 ``getTilingFactor``。
  tpu_generation = get_tpu_info().generation
  tiling = (1 + int(tpu_generation < 4)) * packing
  while tiling < min(src, max_tiling):
    tiling *= 2
  return tiling


def infer_tiling(
    ty: jax_core.AbstractValue, tiling: Tiling | None = None
) -> tuple[int | None, ...]:
  """为给定的形状与类型计算分块（tiling）。

  对于 n 维形状，返回最后 ``len(tiling.shape)`` 个维度的分块，
  前导维度返回 1。例如：
  - 2D 分块：(256, 256) -> (8, 128)，(2, 3, 128, 128) -> (1, 1, 8, 128)。
  - 1D 分块：(16,) -> (8,)，(2, 3, 8) -> (1, 1, 8)。

  类型不要求带有 dtype，因此对这类类型，由于分块未知，
  我们对所有维度都返回 None。
  """
  assert hasattr(ty, "shape")
  shape = ty.shape
  if not hasattr(ty, "dtype"):
    return (None,) * len(shape)
  if ty.dtype == dtypes.dtype("int4"):
    packing = 8
  else:
    packing = 4 // ty.dtype.itemsize

  if tiling is None:
    tiling = Tiling.COMPACT
  tiling_rank = len(tiling.shape)
  if len(shape) == 1 and tiling == Tiling.COMPACT:
    _, lane_count = tiling.shape
    tpu_generation = get_tpu_info().generation
    return ((1 + int(tpu_generation < 4)) * packing * lane_count,)
  if len(shape) < tiling_rank:
    raise ValueError(
        f"Shape must have at least {tiling_rank} dimensions: {shape=}"
    )

  leading_dims, final_dims = shape[:-tiling_rank], shape[-tiling_rank:]
  match tiling:
    case Tiling.COMPACT:
      second_minor, _ = final_dims
      factor = _get_tiling_factor(second_minor, tiling.shape[0], packing)
      return (*(1,) * len(leading_dims), factor, tiling.shape[1])
    case Tiling.SPARSE_CORE:
      [tile_size] = tiling.shape
      return (*(1,) * len(leading_dims), tile_size * packing)


def get_device_kind() -> str:
  if abstract_device := mesh_lib.get_abstract_mesh().abstract_device:
    return abstract_device.device_kind
  return pxla.get_default_device().device_kind


def get_num_device_cores() -> int:
  if abstract_device := mesh_lib.get_abstract_mesh().abstract_device:
    return abstract_device.num_cores
  return pxla.get_default_device().num_cores
