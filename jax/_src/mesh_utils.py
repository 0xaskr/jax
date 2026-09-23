# Copyright 2021 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# 文件职责：构建设备网格（device mesh）的工具，供 `jax.sharding.Mesh` 使用。
# 依据物理拓扑（TPU 托盘内的环形连接、N 维环面网络，或 ICI/DCN 的混合网络）
# 与逻辑网格形状重排设备顺序，目标是让集合通信获得最大带宽。
# 对外暴露 `create_device_mesh()` 与 `create_hybrid_device_mesh()` 两个入口；
# 内部按设备类型选用预设的设备顺序（如 TPU v2/v3、v4i/v8i、v5e、v5p、7x），
# 并实现 N 维环面上的逻辑轴到物理轴分配算法、按需拆分物理轴，以及用于生成
# 连续子网格的转置技巧。
"""构建设备网格的工具。"""

from __future__ import annotations

import collections
from collections.abc import Callable, Generator, MutableMapping, Sequence
import itertools
import math
from typing import Any

from jax._src import xla_bridge as xb
import numpy as np

_TPU_V2 = 'TPU v2'
_TPU_V3 = 'TPU v3'
_TPU_V4 = 'TPU v4'
_TPU_V4_LITE = "TPU v4 lite"
_TPU_V5_LITE = "TPU v5 lite"
_TPU_V5 = "TPU v5"
_TPU_V5E = "TPU v5e"
_TPU_V5P = "TPU v5p"
_TPU_V6_LITE = "TPU v6 lite"
_TPU_7X = "TPU7x"
_TPU_7 = "TPU7"
_TPU_8I = "TPU8i"

# 将物理拓扑映射到网格形状，再映射到 jekbradbury 那个著名的连续网格技巧
# 所使用的转置。
#
# 该技巧只对特定的拓扑和网格形状有效。所列形状可以追加大小为 1 的平凡维度，
# 这些形状同样受支持。
_TRANSPOSE_TRICKS: dict[
    tuple[int, ...], dict[tuple[int, ...], tuple[int, ...]]
] = {
    (2, 2, 1): {
        (2, 2): (0, 1, 2),
    },
    (2, 2, 4): {
        (4, 4): (0, 1, 2),
    },
    (4, 4, 4): {
        (16, 4): (0, 2, 1),
    },
    (4, 8, 8): {
        (64, 4): (0, 2, 1),
        (4, 64): (0, 2, 1),
    },
    (8, 8, 8): {
        (64, 8): (0, 2, 1),
    },
    (8, 16, 16): {
        (256, 8): (0, 2, 1),
        (8, 256): (0, 2, 1),
    },
}

# 托盘（tray）中核心 ID 的物理顺序，该顺序构成一个环
_TRAY_RING_ORDER = (0, 1, 2, 3, 6, 7, 4, 5)
_TRAY_2x2_RING_ORDER = (0, 1, 3, 2)
_TRAY_4x4_RING_ORDER = (0, 1, 2, 3, 7, 6, 5, 9, 10, 11, 15, 14, 13, 12, 8, 4)
_V5E_TRAY_RING_ORDER = (0, 1, 2, 3, 7, 6, 5, 4)
_V5E_TRAY_IOTA_ORDER = (0, 4, 2, 6, 1, 5, 3, 7)
_V5P_2x2x2_ORDER = (0, 1, 3, 2, 6, 7, 5, 4)
_7X_TRAY_2x2x2_RING_ORDER = (0, 1, 2, 3, 6, 7, 4, 5)


def _tpu_v2_v3_create_device_mesh(
    mesh_shape: Sequence[int],
    devices: Sequence[Any],
    **unused_kwargs,
) -> np.ndarray:
  if len(devices) == 8:
    device_mesh = np.asarray(devices)
    device_mesh = device_mesh[np.array(_TRAY_RING_ORDER)]
    device_mesh = device_mesh.reshape(mesh_shape)
    return device_mesh
  elif mesh_shape[-1] == 8:
    device_mesh = np.asarray(devices).reshape(mesh_shape)
    perm = np.array(_TRAY_RING_ORDER)
    device_mesh = device_mesh[..., perm]
    return device_mesh
  else:
    # TODO(skye): 在这里实现二维 mesh_shape 的逻辑：
    # https://github.com/tensorflow/lingvo/blob/0df40cf604dfcd14e28f7087d73687a0bd2fe5c6/lingvo/core/gshard_utils.py#L187
    # （可能会取代上面 mesh_shape[-1] == 8 的分支）
    return np.asarray(devices).reshape(mesh_shape)


# TODO(b/303712469): 为这些处理函数补充单元测试。
# 在 v4i 上创建物理环 0->1->3->2。
def _v4i_v8i_create_device_mesh(
    mesh_shape: Sequence[int], devices: Sequence[Any], **unused_kwargs
) -> np.ndarray | None:
  if len(devices) == 4:
    device_mesh = np.asarray(devices)
    device_mesh = device_mesh[np.array(_TRAY_2x2_RING_ORDER)]
    device_mesh = device_mesh.reshape(mesh_shape)
    return device_mesh
  return None


def _v5e_create_device_mesh(
    mesh_shape: Sequence[int], devices: Sequence[Any], **unused_kwargs
) -> np.ndarray | None:
  """为选定的拓扑创建旋转式 pincer 设备分配。

  Args:
    mesh_shape: 模型使用的逻辑网格形状。
    devices: TPU 设备。
    **unused_kwargs: ...

  Returns:
    None，或者重排后按 `mesh_shape` 变形的设备数组。
  """
  max_x, max_y, max_z = max(getattr(d, "coords", (0, 0, 0)) for d in devices)
  bound_x, bound_y, bound_z = max_x + 1, max_y + 1, max_z + 1
  # 只有在传入的设备按顺序排列时，我们的环形重排才有意义，而实际情况未必
  # 如此。reversed() 把 z 为主序改为 x 为主序。
  sequential_devices = sorted(
      devices,
      key=lambda d: tuple(reversed(getattr(d, "coords", (0, 0, 0)))))

  if bound_x == bound_y == 2 and bound_z == 1 and len(devices) == 4:
    device_mesh = np.asarray(sequential_devices)
    device_mesh = device_mesh[np.array(_TRAY_2x2_RING_ORDER)]
    device_mesh = device_mesh.reshape(mesh_shape)
    return device_mesh

  if len(devices) == 8:
    device_mesh = np.asarray(sequential_devices)
    if bound_x == bound_y == bound_z == 2:  # v5e 2x2x2
      order = _V5E_TRAY_IOTA_ORDER
    else:
      order = _V5E_TRAY_RING_ORDER
    device_mesh = device_mesh[np.array(order)]
    device_mesh = device_mesh.reshape(mesh_shape)
    return device_mesh

  if bound_x == bound_y == 4 and bound_z == 1 and len(devices) == 16:  # v5e4x4
    # 仅当整个网格是一个副本组时才使用环形顺序。
    if max(mesh_shape) == len(devices):
      device_mesh = np.asarray(sequential_devices)
      device_mesh = device_mesh[np.array(_TRAY_4x4_RING_ORDER)]
      device_mesh = device_mesh.reshape(mesh_shape)
      return device_mesh

  return None


def _v5p_create_device_mesh(
    mesh_shape: Sequence[int], devices: Sequence[Any], **unused_kwargs
) -> np.ndarray | None:
  """为选定的拓扑创建设备分配。

  Args:
    mesh_shape: 模型使用的逻辑网格形状。
    devices: TPU 设备。
    **unused_kwargs: ...

  Returns:
    None，或者重排后按 `mesh_shape` 变形的设备数组。
  """
  max_x, max_y, max_z = max(getattr(d, "coords", (0, 0, 0)) for d in devices)
  bound_x, bound_y, bound_z = max_x + 1, max_y + 1, max_z + 1
  # 只有在传入的设备按顺序排列时，我们的环形重排才有意义，而实际情况未必
  # 如此。reversed() 把 z 为主序改为 x 为主序。
  sequential_devices = sorted(
      devices,
      key=lambda d: tuple(reversed(getattr(d, "coords", (0, 0, 0)))))

  if bound_x == bound_y == bound_z == 2 and len(devices) == 8:
    device_mesh = np.asarray(sequential_devices)
    device_mesh = device_mesh[np.array(_V5P_2x2x2_ORDER)]
    device_mesh = device_mesh.reshape(mesh_shape)
    return device_mesh
  return None

def _7x_create_device_mesh(
    mesh_shape: Sequence[int], devices: Sequence[Any], **unused_kwargs
) -> np.ndarray | None:
  """为小规模 7x 拓扑创建设备分配。

  该设备分配会通过划分设备环来尽量减少相邻设备之间的跳数，并由于核心轴具有
  更高的带宽而优先分配核心轴。

  Args:
    mesh_shape: 模型使用的逻辑网格形状。
    devices: TPU 设备。
    **unused_kwargs: ...

  Returns:
    None，或者重排后按 `mesh_shape` 变形的设备数组。
  """
  if len(devices) % 8 != 0 or len(devices) > 32:
    return None

  physical_mesh_shape = _get_physical_tpu_mesh(devices).shape
  # 对于 x 和 y 轴，我们最多只支持 2x2，因为我们可以沿这些轴构成一个环，
  # 再在 z 轴上用其他彼此独立的环重复。
  if physical_mesh_shape[0] > 2 or physical_mesh_shape[1] > 2:
    return None

  indices = []
  for i in range(0, len(devices), 8):
    new_indices = [x + i for x in _7X_TRAY_2x2x2_RING_ORDER]
    indices.extend(new_indices)

  device_mesh = np.asarray(devices)
  device_mesh = device_mesh[np.array(indices)]
  device_mesh = device_mesh.reshape(mesh_shape)
  return device_mesh


# 注册为特定设备类型创建设备网格的函数。其优先级高于 create_device_mesh()
# 中更通用的逻辑。处理函数可以返回 None；此时将回退到默认逻辑。
device_kind_handler_dict: dict[
    str,
    Callable[..., np.ndarray | None],
] = {
    _TPU_V2: _tpu_v2_v3_create_device_mesh,
    _TPU_V3: _tpu_v2_v3_create_device_mesh,
    _TPU_V4_LITE: _v4i_v8i_create_device_mesh,
    _TPU_V5_LITE: _v5e_create_device_mesh,
    _TPU_V5E: _v5e_create_device_mesh,
    _TPU_V5P: _v5p_create_device_mesh,
    _TPU_V5: _v5p_create_device_mesh,
    _TPU_V6_LITE: _v5e_create_device_mesh,
    _TPU_7X: _7x_create_device_mesh,
    _TPU_7: _7x_create_device_mesh,
    _TPU_8I: _v4i_v8i_create_device_mesh,
}


def _create_device_mesh_for_nd_torus(
    physical_mesh: np.ndarray,
    mesh_shape: Sequence[int],
    *,
    allow_split_physical_axes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  """把逻辑并行轴分配到 N 维环面网络的物理轴上。

  给定大小由 `mesh_shape` 描述的逻辑并行轴，以及由 `physical_mesh` 表示的
  N 维环面网络中的设备，把每个逻辑轴映射到一个或多个物理轴。倾向于把对性能
  更敏感的逻辑轴映射到更多的物理轴上，以最大化其可用带宽。在可能的情况下，
  也倾向于把逻辑轴分配到多个大小相同的物理轴（例如一个二维正方形），而不是
  多个大小不同的物理轴。

  如果 allow_split_physical_axes = False（默认），本函数会直接报错，而不是把
  一个物理轴拆分给多个逻辑轴（那样会降低总可用带宽）。

  我们用一个具体例子来解释这些概念与考量。

  作为示例，假设逻辑网格为 [data, model]，分别对应数据并行和模型并行。再假设
  数据并行对性能的敏感程度低于模型并行。考虑一个形状为 4x4x16 的三维 TPU
  pod slice，它由形状 (4, 4, 16) 的物理网格表示。

  TPU pod slice 借助回绕链路在所有轴上具有相同带宽，但大小为 4x4 的二维平面
  可能比非正方形平面或一维子组拥有更快的 XLA 集合通信实现。如果 mesh_shape
  为 [16, 16]，我们可能希望把对性能更敏感的 `model` 轴映射到 4x4 的 XY 平面上。

  Args:
    physical_mesh: 形状为 N 维环面物理拓扑的设备 np.ndarray。
    mesh_shape: 逻辑网格的形状（各个逻辑并行轴的大小），其轴按网络强度递增
      的顺序排列。
    allow_split_physical_axes: 若为 True，我们会在必要时拆分物理轴以匹配所需
      的网格形状。

  Returns:
    一个形状为逻辑网格（mesh_shape）的设备 np.ndarray，其中每个逻辑并行轴都
      映射到一个或多个物理网格轴。
    轴分配矩阵，即一个二维数组，把 (physical_axis, logical_axis) 映射到所分配的
      大小，并满足不变式 np.prod(assignment, axis=1) = physical_mesh_shape 与
      np.prod(assignment, axis=0) = mesh_shape。
  """
  # 尚未分配给逻辑轴的剩余物理轴。
  assignable_physical_mesh = list(physical_mesh.shape)
  # 把每个逻辑轴映射到物理轴的一个子集。
  assignment: list[tuple[int, ...]] = [() for _ in mesh_shape]

  # 构建尊重物理轴带宽的优先级映射。值越小 = 优先级越高。目前已知只有
  # TPU v7/v7x 具有非对称带宽：在形状为 (x, y, z, core) 的四维网格上，core
  # 轴带宽最高，因此获得最高优先级（rank 0）。其他所有设备类型对每个物理轴
  # 都一视同仁（priority_map 为 None -> 不做改动）。
  priority_map: dict[int, int] | None = None
  if (
      len(physical_mesh.shape) == 4
      and getattr(physical_mesh.flat[0], 'device_kind', None) in (_TPU_7X, _TPU_7)
  ):
    # 四维网格 (x, y, z, core)：core（轴 3）优先，然后是 x、y、z。
    priority_map = {0: 1, 1: 2, 2: 3, 3: 0}

  # 从网络强度最高到最低依次分配逻辑轴。
  # 假定 `mesh_shape` 按网络强度从低到高排列，因此先将其反转。
  for logical_axis_index, logical_axis_size in reversed(
      list(enumerate(mesh_shape))
  ):
    # 为获得更高带宽，优先映射到更多的物理轴。
    for num_axes in range(len(physical_mesh.shape), 0, -1):
      # 尝试分配到任意大小为 num_axes 的子集。生成所有候选。
      candidates = list(
          itertools.combinations(enumerate(assignable_physical_mesh), num_axes)
      )

      # 若提供了优先级映射则对候选排序，使包含更高优先级物理轴的候选先被
      # 尝试。这确保网络强度高的逻辑轴能分配到高带宽的物理轴。
      if priority_map is not None:
        def _candidate_priority(candidate):
          # rank 越小 = 优先级越高。先按候选中所有轴里最好（最小）的优先级
          # 排序，再以优先级之和作为次序判定。
          indices = tuple(c[0] for c in candidate)
          best_priority = min(priority_map[i] for i in indices)  # type: ignore
          total_priority = sum(priority_map[i] for i in indices)  # type: ignore
          return (best_priority, total_priority)

        candidates.sort(key=_candidate_priority)

      for elem in candidates:
        c_indices, c_axes = zip(*elem)
        # TODO(zhangqiaorjc): 由于 XLA 的限制，二维集合通信目前只为正方的
        # 二维平面实现。把物理轴映射到两个逻辑轴时，对非正方的二维平面
        # 可能更慢，例如把 32 映射为 4x8 或单个轴。如果 XLA 的二维集合
        # 通信很快支持非正方平面，我们就可以继续普遍地优先映射到二维
        # 平面；否则，我们应当同等对待非正方二维平面和一维子网格。
        if np.prod(c_axes) == logical_axis_size:
          assignment[logical_axis_index] = c_indices
          # 把已分配的物理轴清零。
          assignable_physical_mesh = [
              0 if i in c_indices else v
              for i, v in enumerate(assignable_physical_mesh)
          ]
          break
      if assignment[logical_axis_index]:
        # 上面已经从一个候选中找到了分配方案。
        break
    else:
      # 如果 num_axes 的 for 循环没有 break，即所有候选都不适用，就会带着
      # 这个 while-else 结构走到这里。
      if logical_axis_size > 1:
        if not allow_split_physical_axes:
          # 尽管该功能现已实现，但仍有下游任务依赖这里抛出
          # NotImplementedError。
          raise NotImplementedError(
              'Failed to find assignment for logical_axis_index'
              f' {logical_axis_index} of size {logical_axis_size} with'
              f' remaining assignable mesh {assignable_physical_mesh}. The size'
              ' of each axis in your logical mesh must be equal to the product'
              ' of some subset of the physical mesh axis sizes. E.g. logical'
              ' mesh (4, 16) is compatible with physical mesh 4x4x4 since 4=4'
              ' and 16=4x4. If you want to split physical axes, set '
              ' allow_split_physical_axes to True.'
          )
        else:
          # 我们会继续尝试寻找分配方案，即便这意味着拆分物理轴，而这需要
          # 更精细的实现。
          return _create_device_mesh_for_nd_torus_splitting_axes(
              physical_mesh, mesh_shape
          )

  # 展平分配结果，例如 [(), (2,), (0, 1)] -> (2, 0, 1)。
  transpose: list[int] = []
  assignment_array = np.ones(
      [len(physical_mesh.shape), len(mesh_shape)], dtype=np.int64
  )
  for i, x in enumerate(assignment):
    for y in x:
      physical_mesh_axis = int(y)
      assignment_array[physical_mesh_axis, i] = physical_mesh.shape[
          physical_mesh_axis
      ]
      transpose.append(physical_mesh_axis)
  return (
      physical_mesh.transpose(transpose).reshape(mesh_shape),
      assignment_array,
  )


def _create_device_mesh_for_nd_torus_splitting_axes(
    physical_mesh: np.ndarray,
    mesh_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
  """把逻辑并行轴分配到 N 维环面网络的物理轴上。

  该实现允许创建需要拆分物理轴的网格，因此只要设备数量匹配，就可以生成任意
  形状的逻辑网格，例如：

  - 从 4x4 创建 2x2x4；

  - 从 8x8 创建 2x2x16；

  Args:
    physical_mesh: 形状为 N 维环面物理拓扑的设备 np.ndarray。
    mesh_shape: 逻辑网格的形状（各个逻辑并行轴的大小），其轴按网络强度递增
      的顺序排列。

  Returns:
    一个形状为逻辑网格（mesh_shape）的设备 np.ndarray，其中每个逻辑并行轴都
      映射到一个或多个物理网格轴。
    轴分配矩阵，即一个二维数组，把 (physical_axis, logical_axis) 映射到所分配的
      大小，并满足不变式 np.prod(assignment, axis=1) = physical_mesh_shape 与
      np.prod(assignment, axis=0) = mesh_shape。
  """
  if np.prod(physical_mesh.shape) != np.prod(mesh_shape):
    raise ValueError(
        'The number of devices in physical mesh'
        f' {physical_mesh.shape} does not match the number of devices'
        f' in logical mesh {mesh_shape}.'
    )

  physical_mesh_shape = physical_mesh.shape
  logical_mesh_shape = tuple(mesh_shape)

  # （部分）分配映射，表示为二维数组 [p_axis, l_axis] -> size。
  assignment = np.ones(
      [len(physical_mesh_shape), len(logical_mesh_shape)], dtype=np.int64
  )

  # 从网络强度最高到最低依次处理逻辑轴。
  # 假定 `mesh_shape` 按网络强度从低到高排列，因此将其反转。
  for logical_axis, logical_axis_size in reversed(
      list(enumerate(logical_mesh_shape))
  ):
    # 遍历该逻辑轴所有可能的分配方案，包括会拆分多个物理轴的方案。
    best_logical_axis_assignment: np.ndarray | None = None
    for logical_axis_assignment in _enumerate_feasible_logical_axis_assignments(
        physical_mesh_shape, assignment, logical_axis_size
    ):
      # TODO(rosun): 不要使用启发式规则，而是用能反映底层硬件特性的合适评分
      # 函数来替代。
      if (
          best_logical_axis_assignment is None
          or _prefer_first_logical_axis_assignment(
              logical_axis_assignment,
              best_logical_axis_assignment,
              physical_mesh_shape=physical_mesh_shape,
              assignment=assignment,
          )
      ):
        best_logical_axis_assignment = logical_axis_assignment
    assignment[:, logical_axis] = best_logical_axis_assignment  # pyrefly: ignore[unsupported-operation]  # numpy 2.2

  # 读出分配结果。
  logical_mesh = _generate_logical_mesh(
      physical_mesh, logical_mesh_shape, assignment
  )

  return logical_mesh, assignment


def _get_prime_factors(x: int) -> list[int]:
  """返回给定数字的有序质因数列表。"""
  assert x > 0
  factors = []
  p = 2
  while p * p <= x:
    while x % p == 0:
      factors.append(p)
      x //= p
    p += 1
  if x > 1:
    factors.append(x)
  return factors


def _enumerate_feasible_logical_axis_assignments(
    physical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
    logical_axis_size: int,
) -> Generator[np.ndarray]:
  """为单个逻辑轴生成可行的分配方案。

  对于形状为 [x_1, ..., x_n] 的物理网格，以及各个物理轴上此前已有分配大小的
  乘积 [y_1, ..., y_n]，本函数以一维数组 [z_1, ..., z_n] 的形式生成该轴所有
  可能的分配方案，使其满足：

  - prod(z_1, ..., z_n) = logical_axis_size

  - x_i % (z_i * y_i) = 0

  Args:
    physical_mesh_shape: 物理网格形状。
    assignment: 已有的分配矩阵。
    logical_axis_size: 待分配的逻辑轴大小。

  Yields:
    该逻辑轴的所有合法分配方案。每个方案表示为长度为 len(physical_mesh_shape)
    的整数数组。
  """
  logical_axis_factors: MutableMapping[int, int] = collections.defaultdict(int)
  for factor in _get_prime_factors(logical_axis_size):
    logical_axis_factors[factor] += 1

  available_physical_mesh_shape = np.array(physical_mesh_shape) // np.prod(
      assignment, axis=-1
  )

  # 为实现高效枚举，我们先用质因数给物理轴建立索引。既然已知逻辑轴大小的
  # 质因数分解，我们只需为每个质因数挑选正确的个数即可完成枚举。
  physical_axes_by_factor: MutableMapping[int, list[int]] = (
      collections.defaultdict(list)
  )
  for physical_axis, physical_axis_size in enumerate(
      available_physical_mesh_shape
  ):
    for factor in _get_prime_factors(physical_axis_size):
      if factor not in logical_axis_factors:
        continue
      physical_axes_by_factor[factor].append(physical_axis)

  factors = []
  assignments_by_factor = []
  for factor, multiplicity in logical_axis_factors.items():
    factors.append(factor)
    assignments_by_factor.append(
        set(
            itertools.combinations(
                physical_axes_by_factor[factor], multiplicity
            )
        )
    )

  for axis_assignment in itertools.product(*assignments_by_factor):
    result = np.ones([len(physical_mesh_shape)], dtype=np.int64)
    for factor_index, per_factor_assignment in enumerate(axis_assignment):
      for physical_axis in per_factor_assignment:
        result[physical_axis] *= factors[factor_index]
    yield result


def _prefer_first_logical_axis_assignment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    physical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
) -> bool:
  """若第一个轴分配方案优于第二个，则返回 True。

  目前这只是一些非常简单的启发式规则。不过，我们完全可以在这里引入例如基于
  对底层硬件更精确建模的价值函数。

  TODO(rosun): 使用网络容量的一个代理指标来选择划分方案。

  Args:
    x: 逻辑轴分配方案，形状为 [len(physical_mesh_shape)] 的数组。
    y: 逻辑轴分配方案，形状为 [len(physical_mesh_shape)] 的数组。
    physical_mesh_shape: 物理网格形状。
    assignment: 分配矩阵。

  Returns:
    若 x 优于 y 则返回 True。
  """
  # 优先占满完整的物理轴。我对此没有很好的理由，只是它与既有行为兼容。
  #
  # 例如，在 4 x 4 x 8 上，[4, 4, -] 优于 [4, -, 4]，后者又优于 [2, 2, 4]。
  x_whole_axis_size = np.prod(
      [s for i, s in enumerate(x) if s == physical_mesh_shape[i]]
  )
  y_whole_axis_size = np.prod(
      [s for i, s in enumerate(y) if s == physical_mesh_shape[i]]
  )

  if x_whole_axis_size != y_whole_axis_size:
    return x_whole_axis_size > y_whole_axis_size

  # 优先占满更多完整的物理轴，以获得更好的带宽。
  #
  # 这与既有逻辑一致，即 2 x 2 优于 4。
  x_num_whole_axes = len(
      [1 for i, s in enumerate(x) if s == physical_mesh_shape[i] and s > 1]
  )
  y_num_whole_axes = len(
      [1 for i, s in enumerate(y) if s == physical_mesh_shape[i] and s > 1]
  )

  if x_num_whole_axes != y_num_whole_axes:
    return x_num_whole_axes > y_num_whole_axes

  # 优先选择尚未被网络强度更高的逻辑轴占用的物理轴。例如对于 4 x 4 x 4，
  # 假设此前的分配为 1 x 2 x 4，现在要放置一个大小为 2 的新逻辑轴，我们会
  # 选择 [2, 1, 1] 而不是 [1, 2, 1]，因为后者会占用已被更高强度轴占用的
  # 带宽。
  assigned_physical_mesh_shape = np.prod(assignment, axis=-1)

  x_non_overlapping_axis_size = np.prod(
      [s for i, s in enumerate(x) if assigned_physical_mesh_shape[i] > 1]
  )
  y_non_overlapping_axis_size = np.prod(
      [s for i, s in enumerate(y) if assigned_physical_mesh_shape[i] > 1]
  )

  if x_non_overlapping_axis_size != y_non_overlapping_axis_size:
    return x_non_overlapping_axis_size > y_non_overlapping_axis_size

  # 否则按逆字典序排序，以与既有行为保持一致。
  return tuple(x) > tuple(y)


def _generate_logical_mesh(
    physical_mesh: np.ndarray,
    logical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
) -> np.ndarray:
  """根据分配映射计算逻辑网格。

  Args:
    physical_mesh: 物理设备网格。
    logical_mesh_shape: 逻辑网格形状。
    assignment: 形状为 [physical_dims, logical_dims] 的二维分配矩阵。

  Returns:
    由物理网格变形得到的逻辑网格。
  """
  physical_indices = np.broadcast_to(
      np.expand_dims(
          np.arange(len(physical_mesh.shape), dtype=np.int64), axis=-1
      ),
      assignment.shape,
  ).reshape([-1])

  logical_indices = np.broadcast_to(
      np.expand_dims(
          np.arange(len(logical_mesh_shape), dtype=np.int64), axis=0
      ),
      assignment.shape,
  ).reshape([-1])

  # 逻辑网格的轴按 (physical_axis, logical_axis) 排序。
  #
  # 注意我们为每个 physical_axis 对 logical_axis 排序，使强度更高的逻辑轴被
  # 复制到更靠内（次要）的维度上。
  #
  # 例如，若某个维度大小为 12 = 3x4，其中 3 强度更高、4 更低，我们希望变形
  # 后得到 12 = 4x3。可以想象在一维情形下，这会让强度更高的轴之间有更多
  # 连接。
  logical_mesh = np.reshape(physical_mesh, assignment.reshape([-1]))

  # 接着按 l_axis 分组，因为这是输出所期望的形式。
  _, _, transpose_axes = zip(
      *sorted(
          zip(logical_indices, physical_indices, range(len(logical_indices)))
      )
  )
  logical_mesh = np.transpose(logical_mesh, transpose_axes)

  # 通过变形把大小为 1 的平凡维度加回来。
  logical_mesh = np.reshape(logical_mesh, logical_mesh_shape)

  return logical_mesh


def _get_physical_tpu_mesh(jax_devices: Sequence[Any]) -> np.ndarray:
  r"""把 TPU slice 中的设备重排为物理网格。

  Args:
    jax_devices: TPU slice 中 JAX 设备的列表，按进程切分的 z、y、x、core 顺序
      排列，例如来自 jax.devices()。这些设备的坐标应构成一个无空洞的长方体；
      例如坐标可以是 {(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)}（一个 1x2x2
      的长方体）；若只传入其中 3 个设备，该长方体中就会出现一个“空洞”，这会
      导致错误。如我们的例子所示，长方体不要求包含点 (0, 0, 0)。

  Returns:
    形状为 [global_x, global_y, global_z] 的 JAX 设备 np.ndarray。在 v2 和 v3
      上，global_z 改为 cores_per_chip（即 2）。
  """
  device_kind = jax_devices[0].device_kind
  device_coords = [d.coords for d in jax_devices]
  coord_size = len(device_coords[0])
  # 逐位置的最大与最小坐标：
  max_coords = tuple(
      max(dc[i] for dc in device_coords) for i in range(coord_size)
  )
  min_coords = tuple(
      min(dc[i] for dc in device_coords) for i in range(coord_size)
  )
  dims = tuple(h - l + 1 for (h, l) in zip(max_coords, min_coords))

  max_cores_per_chip = max(d.core_on_chip for d in jax_devices)
  min_cores_per_chip = min(d.core_on_chip for d in jax_devices)
  cores_per_chip = max_cores_per_chip - min_cores_per_chip + 1

  assert len(dims) == 3, dims
  assert (
      len(jax_devices) == np.prod(dims) * cores_per_chip
  ), f'{jax_devices=} {dims=} {cores_per_chip=}'

  if device_kind in (_TPU_V2, _TPU_V3):
    out = np.empty(dims[:2] + (cores_per_chip,), dtype=object)
    for d in jax_devices:
      coords = d.coords
      assert coords[2] == 0, d
      out[
          coords[0] - min_coords[0],
          coords[1] - min_coords[1],
          d.core_on_chip - min_cores_per_chip,
      ] = d
  elif (device_kind in (_TPU_7X, _TPU_7, _TPU_8I) or
        (device_kind in (_TPU_V5P,_TPU_V5) and cores_per_chip == 2)):
    out = np.empty(dims + (cores_per_chip,), dtype=object)
    for d in jax_devices:
      coords = d.coords
      out[
          coords[0] - min_coords[0],
          coords[1] - min_coords[1],
          coords[2] - min_coords[2],
          d.core_on_chip - min_cores_per_chip,
      ] = d
  else:
    out = np.empty(dims, dtype=object)
    for d in jax_devices:
      coords = d.coords
      if d.core_on_chip != 0:
        raise AssertionError(
            'Creating meshes for TPU >v3 requires one device per chip'
            f' ("megacore" mode). Got device id {d.core_on_chip} for a device'
            f' of kind {device_kind}: {d}.'
        )
      out[
          coords[0] - min_coords[0],
          coords[1] - min_coords[1],
          coords[2] - min_coords[2],
      ] = d

  # 检查我们构造的网格中不存在“空洞”。
  if (out == None).any():
    raise AssertionError(
        'Constructed mesh contains a "hole"; probable cause: coordinates '
        f'of jax_devices are not a contiguous cuboid: {jax_devices}'
    )
  return out


# jekbradbury 那个用于创建连续子网格的著名技巧（在可用的情况下）
def _transpose_trick(
    physical_mesh: np.ndarray, mesh_shape: Sequence[int]
) -> np.ndarray:
  mesh_shape = tuple(mesh_shape)
  topology = physical_mesh.shape
  if topology not in _TRANSPOSE_TRICKS:
    raise ValueError(
        'create_device_mesh cannot create contiguous submeshes for '
        f'physical mesh topology {topology}'
    )

  mesh_shape_no_trivial_dims: tuple[int, ...] = ()
  for dim_size in mesh_shape:
    if dim_size != 1:
      mesh_shape_no_trivial_dims += (dim_size,)

  if mesh_shape_no_trivial_dims not in _TRANSPOSE_TRICKS[topology]:
    raise ValueError(
        'create_device_mesh cannot create contiguous submeshes for '
        f'mesh_shape {mesh_shape} and physical mesh topology {topology}. '
        f'Available mesh_shapes: {list(_TRANSPOSE_TRICKS[topology].keys())}'
    )

  return physical_mesh.transpose(
      *_TRANSPOSE_TRICKS[topology][mesh_shape_no_trivial_dims]
  )

def _canonicalize_axis_sizes(axis_sizes: Sequence[int]
                             ) -> tuple[int, ...] | None:
  new_sizes = []
  for s in axis_sizes:
    try:
      new_sizes.append(int(s))
    except:
      return None
  return tuple(new_sizes)

def create_device_mesh(
    mesh_shape: Sequence[int],
    devices: Sequence[Any] | None = None,
    *,
    contiguous_submeshes: bool = False,
    allow_split_physical_axes: bool = False,
) -> np.ndarray:
  """为 jax.sharding.Mesh 创建一个高性能的设备网格。

  Args:
    mesh_shape: 逻辑网格的形状，按网络强度递增的顺序排列，例如
      [replica, data, mdl]，其中 mdl 的网络通信需求最大。
    devices: 可选，用于构建网格的设备。默认为 jax.devices()。
    contiguous_submeshes: 若为 True，本函数会尝试创建一个让每个进程的本地设备
      构成连续子网格的网格。若无法生成合适的网格，将抛出 ValueError。在引入
      jax.Array 之前，为了保证本地数组不是参差不齐的，有时需要该设置；如果
      使用 jax.Array，最好把它保持为 False。
    allow_split_physical_axes: 若为 True，我们会在必要时拆分物理轴，以生成所需
      的设备网格。

  Raises:
    ValueError: 若设备数量不等于 `mesh_shape` 的乘积。

  Returns:
    一个以 mesh_shape 为形状的 JAX 设备 np.ndarray，可传入 jax.sharding.Mesh
    并获得良好的集合通信性能。
  """
  if devices is None:
    devices = xb.devices()

  new_mesh_shape = _canonicalize_axis_sizes(mesh_shape)
  if new_mesh_shape is None:
    raise ValueError(
        f'`mesh_shape` passed to `create_device_mesh` should be a sequence of'
        f' ints. Got {mesh_shape}')
  del mesh_shape

  if math.prod(new_mesh_shape) != len(devices):
    raise ValueError(
        f'Number of devices {len(devices)} must equal the product '
        f'of mesh_shape {new_mesh_shape}'
    )
  last_device = devices[-1]

  handler = device_kind_handler_dict.get(last_device.device_kind, None)
  if handler is not None:
    result = handler(
        new_mesh_shape, devices, contiguous_submeshes=contiguous_submeshes
    )
    if result is not None:
      return result

  if last_device.platform == 'tpu':
    physical_mesh = _get_physical_tpu_mesh(devices)
    if contiguous_submeshes:
      physical_mesh = _transpose_trick(physical_mesh, new_mesh_shape)

    # 对于 TPU v7/v7x，core 轴（最后一个轴）的带宽高于 x/y/z 轴。
    # _create_device_mesh_for_nd_torus 会从设备类型识别出这一点，并在无需
    # 调用方提供优先级的情况下，优先把网络强度高的逻辑轴映射到 core 轴。
    device_mesh, _ = _create_device_mesh_for_nd_torus(
        physical_mesh,
        new_mesh_shape,
        allow_split_physical_axes=allow_split_physical_axes,
    )
    return device_mesh
  elif last_device.platform == 'gpu':
    # 默认的 jax.devices() 顺序不保证高性能，因为它基于进程顺序，而不是 XLA
    # 分配的拓扑感知全局编号方案。如果设备列表来自单个 slice，这在现代系统
    # 上没有影响，但当传入多 slice 的设备列表时，这里的排序可以避免一个
    # 隐患。
    return np.asarray(sorted(devices, key=lambda d: d.id)).reshape(new_mesh_shape)
  else:
    device_mesh = np.asarray(devices).reshape(new_mesh_shape)
    return device_mesh


def create_hybrid_device_mesh(
    mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int],
    devices: Sequence[Any] | None = None,
    *,
    process_is_granule: bool = False,
    should_sort_granules_by_key: bool = True,
    allow_split_physical_axes: bool = False,
) -> np.ndarray:
  """为混合（例如 ICI 与 DCN）并行创建可用的设备网格。

  Args:
    mesh_shape: 更快/内层网络的逻辑网格形状，按网络强度递增的顺序排列，例如
      [replica, data, mdl]，其中 mdl 的网络通信需求最大。
    dcn_mesh_shape: 更慢/外层网络的逻辑网格形状，顺序与 mesh_shape 相同。
    devices: 可选，用于构建网格的设备。默认为 jax.devices()。
    process_is_granule: 若为 True，本函数会把进程视为更慢/外层网络的单位。
      否则它会查找设备上的 slice_index 属性，并以 slice 为单位。启用该选项
      是为了给那些不设置 slice_index 的平台提供回退方案。
    should_sort_granules_by_key: 是否按 granule 键（取决于 process_is_granule，
      为 slice 索引或进程索引）对设备 granule 排序。
    allow_split_physical_axes: 若为 True，我们会在必要时拆分物理轴，以生成所需
      的设备网格。

  Raises:
    ValueError: 若 `devices` 所属的 slice 数量不等于 `dcn_mesh_shape` 的乘积，
      或者任一单个 slice 中的设备数量不等于 `mesh_shape` 的乘积。

  Returns:
    一个以 mesh_shape * dcn_mesh_shape 为形状的 JAX 设备 np.ndarray，可传入
    jax.sharding.Mesh 用于混合并行。
  """
  if devices is None:
    devices = xb.devices()
  attr = 'process_index' if process_is_granule else 'slice_index'
  if not hasattr(devices[0], attr):
    raise ValueError(
        f'Device {devices[0]} does not have attribute {attr}. See'
        ' `process_is_granule` option.'
    )
  granule_dict = collections.defaultdict(list)
  for dev in devices:
    granule_dict[getattr(dev, attr)].append(dev)
  granules = (
      [granule_dict[key] for key in sorted(granule_dict.keys())]
      if should_sort_granules_by_key
      else granule_dict.values()
  )
  if np.prod(dcn_mesh_shape) != len(granules):
    raise ValueError(
        f'Number of slices {len(granules)} must equal the product of '
        f'dcn_mesh_shape {dcn_mesh_shape}'
    )
  per_granule_meshes = [
      create_device_mesh(
          mesh_shape,
          granule,
          allow_split_physical_axes=allow_split_physical_axes,
      )
      for granule in granules
  ]
  # TODO(jekbradbury): 处理非均匀的 DCN 拓扑
  granule_mesh = np.arange(len(granules)).reshape(dcn_mesh_shape)
  blocks = np.vectorize(lambda i: per_granule_meshes[i], otypes=[object])(
      granule_mesh
  )
  device_mesh = np.block(blocks.tolist())
  return device_mesh
