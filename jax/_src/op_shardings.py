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
# 文件职责：把 XLA 的 `HloSharding` / `OpSharding` 描述转换成主机侧可用的分片索引。
# 供 JAX 中需要查询分片方式的代码使用，例如判断分片是否等价、是否只是复制。
# 核心逻辑会解析 tile assignment 的维度与子分组类型（复制、非规约），得出每个维度
# 的分片数；再据此求出每台设备对应的多维切片索引，供上层按设备重建数组分片。
"""分片工具"""

from __future__ import annotations

from collections.abc import Sequence
import itertools

import numpy as np

from jax._src.lib import xla_client as xc


def get_num_ways_dim_sharded(hlo_sharding: xc.HloSharding
                             ) -> tuple[list[int], int]:
  assert not hlo_sharding.is_manual()
  if hlo_sharding.is_replicated():
    return [], 1
  if hlo_sharding.is_unreduced():
    return [], 1
  partitions = hlo_sharding.tile_assignment_dimensions()
  subgroup_types = hlo_sharding.subgroup_types()

  if subgroup_types == [xc.OpSharding.Type.REPLICATED]:
    return list(partitions[:-1]), partitions[-1]
  elif subgroup_types == [xc.OpSharding.Type.UNREDUCED]:
    return list(partitions[:-1]), 1
  elif set(subgroup_types) == {xc.OpSharding.Type.REPLICATED,
                               xc.OpSharding.Type.UNREDUCED}:
    replicated_loc = subgroup_types.index(xc.OpSharding.Type.REPLICATED)
    return list(partitions[:-2]), partitions[-2:][replicated_loc]
  elif hlo_sharding.replicate_on_last_tile_dim():
    return list(partitions[:-1]), partitions[-1]
  else:
    if subgroup_types:
      raise NotImplementedError(f"Unhandled OpSharding type: {hlo_sharding}. "
                                "Please open a bug report!")
    return list(partitions), 1


def is_hlo_sharding_replicated(hc: xc.HloSharding) -> bool:
  return True if hc.num_devices() == 1 else hc.is_replicated()


def are_hlo_shardings_equal(hc1: xc.HloSharding, hc2: xc.HloSharding) -> bool:
  if hc1 is hc2:
    return True
  if is_hlo_sharding_replicated(hc1) and is_hlo_sharding_replicated(hc2):
    return True
  return hc1 == hc2


_Index = int | slice | tuple[int | slice, ...]


def op_sharding_to_numpy_indices(
    hlo_sharding: xc.HloSharding, shape: Sequence[int],
    num_devices: int) -> np.ndarray:
  indices = np.empty(num_devices, dtype=np.object_)

  # 当 hlo_sharding 为 REPLICATED 时，必须把 num_devices 作为参数传入。
  # 不能使用 `jax.device_count()`，因为可以创建出设备数少于
  # `jax.device_count()` 的 opsharding。
  if is_hlo_sharding_replicated(hlo_sharding):
    indices.fill((slice(None),) * len(shape))
    return indices

  assert num_devices == hlo_sharding.num_devices()

  partitions, num_replicas = get_num_ways_dim_sharded(hlo_sharding)
  assert len(partitions) == len(shape), (len(partitions), len(shape))

  axis_indices: list[Sequence[_Index]] = []
  for dim, n_shards in zip(shape, partitions):
    if n_shards == 1:
      axis_indices.append([slice(None)])
    elif n_shards > 1:
      shard_size, ragged = divmod(dim, n_shards)
      assert not ragged, (dim, n_shards)
      axis_indices.append([slice(i * shard_size, (i + 1) * shard_size)
                           for i in range(n_shards)])
    else:
      raise AssertionError('Unrecognized number of shards. Please file a bug!')

  device_it = iter(hlo_sharding.tile_assignment_devices())

  for idxs in itertools.product(*axis_indices):
    for _ in range(num_replicas):
      indices[next(device_it)] = idxs
  return indices


def op_sharding_to_indices(
    op_sharding: xc.HloSharding, shape: Sequence[int],
    num_devices: int) -> tuple[tuple[slice, ...], ...]:
  indices = op_sharding_to_numpy_indices(op_sharding, shape, num_devices)
  return tuple(indices.flat)
