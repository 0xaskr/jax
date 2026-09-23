# Copyright 2021 The JAX Authors.
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

# 文件职责：定义 JAX 中描述数组如何在设备间分布的分片（sharding）抽象。
# `Sharding` 是面向用户的基类，规定 `device_set`、`is_fully_replicated`、
# `is_fully_addressable`、`num_devices`、`memory_kind` 等必须由子类实现的接口，
# 并给出 `addressable_devices_indices_map`、`devices_indices_map`、`shard_shape`、
# `is_equivalent_to` 等默认实现；模块还提供把分片转换为 XLA HLO / sdy 表示、
# 比较两个分片是否等价，以及计算并缓存“设备到索引映射”和分片形状的辅助函数。

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools

from jax._src.util import safe_zip, use_cpp_class, cache
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.op_shardings import (
    are_hlo_shardings_equal, get_num_ways_dim_sharded,
    is_hlo_sharding_replicated, op_sharding_to_indices)

Shape = tuple[int, ...]
Device = xc.Device
Index = tuple[slice, ...]
XLADeviceAssignment = Sequence[Device]

class IndivisibleError(ValueError):
  pass

@cache(max_size=4096, trace_context_in_key=False)
def _addressable_devices_indices_map(
    sharding: Sharding, global_shape: Shape) -> Mapping[Device, Index | None]:
  global_map = sharding.devices_indices_map(global_shape)
  if sharding.is_fully_addressable:
    return global_map
  return {d: global_map[d]
          for d in sharding._internal_device_list.addressable_device_list}

@cache(max_size=4096, trace_context_in_key=False)
def common_devices_indices_map(
    s: Sharding, global_shape: Shape) -> Mapping[Device, Index]:
  s.shard_shape(global_shape)  # 抛出信息友好的错误
  hlo_sharding = s._to_xla_hlo_sharding(len(global_shape))
  if (xc.OpSharding.Type.UNREDUCED in hlo_sharding.subgroup_types() or
      hlo_sharding.is_unreduced()):
    raise NotImplementedError(
        "device_indices_map doesn't work with unreduced. Please file a bug at"
        ' https://github.com/jax-ml/jax/issues')
  indices = op_sharding_to_indices(hlo_sharding, global_shape,
                                   len(s._device_assignment))
  return dict(safe_zip(s._device_assignment, indices))


@cache(max_size=4096, trace_context_in_key=False)
def _common_shard_shape(self, global_shape: Shape) -> Shape:
  hlo_sharding = self._to_xla_hlo_sharding(len(global_shape))
  if is_hlo_sharding_replicated(hlo_sharding):
    return global_shape
  if hlo_sharding.is_unreduced():
    return global_shape
  partitions, _ = get_num_ways_dim_sharded(hlo_sharding)
  assert len(partitions) == len(global_shape), (len(partitions), len(global_shape))
  out = []
  for dim, (s, p) in enumerate(safe_zip(global_shape, partitions)):
    quotient, remainder = divmod(s, p)
    if remainder != 0:
      raise IndivisibleError(
          f"{self} implies that array axis {dim} is partitioned "
          f"{p} times, but the dimension size is {s} "
          f"(full shape: {global_shape}, "
          f"per-dimension tiling factors: {tuple(partitions)} should evenly "
          "divide the shape)")
    out.append(quotient)
  return tuple(out)

def common_is_equivalent_to(s1: Sharding, s2: Sharding, ndim: int,
                            check_devices: bool = True) -> bool:
  hlo_s_eq = are_hlo_shardings_equal(
      s1._to_xla_hlo_sharding(ndim), s2._to_xla_hlo_sharding(ndim))
  mem_eq = s1.memory_kind == s2.memory_kind
  if check_devices:
    return (hlo_s_eq and mem_eq and
            s1._internal_device_list == s2._internal_device_list)
  else:
    return hlo_s_eq and mem_eq


@use_cpp_class(xc.Sharding)
class Sharding:
  """描述 :class:`jax.Array` 如何布局到各个设备上。
  """

  # 以下为抽象方法，应由子类实现。
  @property
  def device_set(self) -> set[Device]:
    """该 :class:`Sharding` 所跨越的设备集合。

    在多控制器 JAX 中，设备集合是全局的，即包含
    来自其他进程的不可寻址设备。
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @property
  def is_fully_replicated(self) -> bool:
    """该分片是否完全复制？

    如果每个设备都拥有整个数据的完整副本，
    则该分片是完全复制的。
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @property
  def is_fully_addressable(self) -> bool:
    """该分片是否完全可寻址？

    如果当前进程能够寻址 :class:`Sharding` 中列出的所有设备，
    则该分片就是完全可寻址的。``is_fully_addressable``
    等价于多进程 JAX 中的 "is_local"。
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @property
  def num_devices(self) -> int:
    """该分片包含的设备数量。"""
    raise NotImplementedError('Subclasses should implement this method.')

  @property
  def memory_kind(self) -> str | None:
    """返回该分片的内存种类。"""
    raise NotImplementedError('Subclasses should implement this method.')

  def with_memory_kind(self, kind: str) -> Sharding:
    """返回具有指定内存种类的新 `Sharding` 实例。"""
    raise NotImplementedError('Subclasses should implement this method')

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    raise NotImplementedError('Subclasses should implement this method.')

  @property
  def _internal_device_list(self) -> xc.DeviceList:
    raise NotImplementedError('Subclasses should implement this method.')

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    raise NotImplementedError('Subclasses should implement this method.')

  def _to_sdy_sharding(self, num_dimensions: int,
                       modify_wrt_axis_types: bool = False):
    raise NotImplementedError('Subclasses should implement this method.')

  #############################################################################
  # 以下为所有子类都会继承的默认实现。

  @property
  def _is_concrete(self) -> bool:
    return True

  @functools.cached_property
  def addressable_devices(self) -> set[Device]:
    """该 :class:`Sharding` 中可被当前进程寻址的
       设备集合。
    """
    # 为单控制器运行时添加快速路径。
    if xb.process_count() == 1:
      return self.device_set
    return {d for d in self.device_set
            if d.process_index == d.client.process_index()}

  def addressable_devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Index | None]:
    """从可寻址设备到各设备所含数组数据切片的映射。

    ``addressable_devices_indices_map`` 包含
    ``device_indices_map`` 中适用于可寻址设备的那部分。
    """
    return _addressable_devices_indices_map(self, global_shape)

  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    """返回从设备到各设备所含数组切片的映射。

    该映射包含所有全局设备，即包含
    来自其他进程的不可寻址设备。
    """
    return common_devices_indices_map(self, global_shape)

  @property
  def has_addressable_devices(self) -> bool:
    return len(self._internal_device_list.addressable_device_list) > 0

  @functools.cached_property
  def _addressable_device_assignment(self) -> XLADeviceAssignment:
    if self.is_fully_addressable:
      return self._device_assignment
    return tuple(self._internal_device_list.addressable_device_list)

  def shard_shape(self, global_shape: Shape) -> Shape:
    """返回每个设备上数据的形状。

    该函数返回的分片形状由
    ``global_shape`` 与分片自身的属性计算得到。
    """
    return _common_shard_shape(self, global_shape)

  def is_equivalent_to(self: Sharding, other: Sharding, ndim: int) -> bool:
    """若两个分片等价则返回 ``True``。

    如果两个分片把相同的逻辑数组分片放置在
    相同的设备上，则它们是等价的。
    """
    return common_is_equivalent_to(self, other, ndim)
