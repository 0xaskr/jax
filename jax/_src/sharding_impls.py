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

# 文件职责：实现 `jax.sharding` 的底层分片类型与 mesh 上下文，是 GSPMD 自动分片
# 与显式分片模式的公共基础设施。
# 主要内容：具体分片类 `SingleDeviceSharding`、`GSPMDSharding`（XLA HloSharding
# 的 JAX 包装）、并行计算的轴上下文 `SPMDAxisContext`/`ShardingContext`、
# OpSharding 与 PartitionSpec 的互转工具（含 process-uniform 判定与 local→global
# 形状推导），以及 `make_mesh`/`set_mesh`/`get_mesh` 等 mesh 上下文接口。

from __future__ import annotations

import collections
import contextlib
from collections.abc import Mapping, Sequence
import dataclasses
import functools
import math
import itertools as it
from typing import Any, cast

from jax._src import config
from jax._src import core
from jax._src import sharding as jsharding
from jax._src import tree_util
from jax._src import util
from jax._src import source_info_util
from jax._src import xla_bridge as xb
from jax._src import mesh_utils
from jax._src.mesh import (
    Mesh, AbstractMesh, AxisType, empty_abstract_mesh, empty_concrete_mesh,
    get_abstract_mesh, get_concrete_mesh)
from jax._src.lib import _jax
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir.dialects import sdy
from jax._src.named_sharding import (  # noqa: F401
    SdyArray, SdyDim, UnspecifiedValue, flatten_spec, NamedSharding,
    _check_unique_resources, UNSPECIFIED,
    ArrayMapping, ArrayMappingOrAutoOrUnspecified, get_array_mapping,
    array_mapping_to_axis_resources, named_sharding_to_xla_hlo_sharding)
from jax._src.op_shardings import (
    are_hlo_shardings_equal, get_num_ways_dim_sharded,
    is_hlo_sharding_replicated)
from jax._src.partition_spec import PartitionSpec
from jax._src.util import use_cpp_class, use_cpp_method
import numpy as np


config_ext = _jax.config

Shape = tuple[int, ...]
Device = xc.Device
Index = tuple[slice, ...]
XLADeviceAssignment = tuple[Device, ...]
# TODO(yashkatariya): 弃用满 3 个月后移除此项。
XLACompatibleSharding = jsharding.Sharding


def hashed_index(x) -> int:
  assert all(v.step is None for v in x if isinstance(v, slice))
  return hash(tuple((v.start, v.stop) if isinstance(v, slice) else v for v in x))


@util.cache(max_size=4096, trace_context_in_key=False)
def device_replica_id_map(sharding, global_shape: Shape) -> Mapping[Device, int]:
  try:
    device_indices_map_fn = sharding.devices_indices_map
  except AttributeError:
    raise ValueError(
        f'Cannot calculate replica ids from sharding: {sharding}. Please '
        'create a device to index mapping for your sharding from which replica '
        'ids will be calculated.') from None

  index_to_replica: dict[int, int] = collections.Counter()
  out = {}
  for device, index in device_indices_map_fn(global_shape).items():
    h_index = hashed_index(index)
    replica_id = index_to_replica[h_index]
    index_to_replica[h_index] += 1
    out[device] = replica_id
  return out


@dataclasses.dataclass(frozen=True, slots=True)
class SdyArrayList:
  shardings: tuple[SdyArray, ...]

  def build(self, cache: dict[SdyArray, sdy.TensorShardingAttr]
            ) -> sdy.TensorShardingPerValueAttr:
    return sdy.TensorShardingPerValueAttr.get(
        [sharding.build(cache) for sharding in self.shardings])


replicated_hlo_sharding = xc.HloSharding.replicate()


def _unpickle_single_device_sharding(device, memory_kind):
  return SingleDeviceSharding(device, memory_kind=memory_kind)


@use_cpp_class(xc.SingleDeviceSharding)
class SingleDeviceSharding(jsharding.Sharding):
  """一种把数据放置在单个设备上的 :class:`Sharding`。

  Args:
    device: 单个 :py:class:`Device`。

  Examples:

    >>> single_device_sharding = jax.sharding.SingleDeviceSharding(
    ...     jax.devices()[0])
  """

  _device: Device
  _memory_kind: str | None

  @use_cpp_method()
  def __init__(self, device: Device, *, memory_kind: str | None = None):
    self._device = device
    self._memory_kind = memory_kind

  def __reduce__(self):
    return (_unpickle_single_device_sharding, (self._device, self._memory_kind))

  def __repr__(self):
    mem = '' if self._memory_kind is None else f', memory_kind={self._memory_kind}'
    return f"SingleDeviceSharding(device={self._device!r}{mem})"

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self._device, self.memory_kind))
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, SingleDeviceSharding):
      return False
    if self is other:
      return True
    return (self._device == other._device and
            self.memory_kind == other.memory_kind)

  @property
  def num_devices(self) -> int:
    return len(self.device_set)

  @property
  def device_set(self) -> set[Device]:
    return {self._device}

  @property
  def memory_kind(self) -> str | None:
    return self._memory_kind

  def with_memory_kind(self, kind: str) -> SingleDeviceSharding:
    return SingleDeviceSharding(self._device, memory_kind=kind)

  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    return {self._device: (slice(None),) * len(global_shape)}

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return (self._device,)

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    return replicated_hlo_sharding

  def _to_sdy_sharding(self, num_dimensions: int,
                       modify_wrt_axis_types: bool = False) -> SdyArray:
    sdy_dim_sharding = (SdyDim(axes=(), is_open=False),) * num_dimensions
    return SdyArray(mesh_shape=None, dim_shardings=sdy_dim_sharding)

  @property
  def is_fully_replicated(self) -> bool:
    return True

  @property
  def is_fully_addressable(self) -> bool:
    return xb.process_index(self._device.client) == self._device.process_index

  def check_compatible_aval(self, aval_shape: Shape) -> None:
    return

SingleDeviceSharding.__module__ = 'jax.sharding'


def make_single_device_sharding(device, *, memory_kind=None):
  return SingleDeviceSharding(device, memory_kind=memory_kind)


def _unpickle_gspmd_sharding(devices, op_sharding, memory_kind):
  return GSPMDSharding(devices, op_sharding, memory_kind=memory_kind)

@use_cpp_class(xc.GSPMDSharding)
class GSPMDSharding(jsharding.Sharding):
  _devices: xc.DeviceList
  _hlo_sharding: xc.HloSharding
  _memory_kind: str | None
  _internal_device_list: xc.DeviceList

  @use_cpp_method()
  def __init__(self, devices: Sequence[Device] | xc.DeviceList,
               op_sharding: xc.OpSharding | xc.HloSharding,
               *, memory_kind: str | None = None):
    self._devices = (devices if isinstance(devices, xc.DeviceList) else
                     xc.DeviceList(tuple(devices)))
    self._hlo_sharding = (xc.HloSharding.from_proto(op_sharding)
                          if isinstance(op_sharding, xc.OpSharding) else
                          op_sharding)
    # 将 HloShardingV3 转换为 V2，因为对于 XLA 返回的分片，
    # JAX 期望的是平铺(tiled)分片。
    self._hlo_sharding = xc.HloSharding.v3_to_v2_sharding(self._hlo_sharding)
    self._memory_kind = memory_kind

  def __reduce__(self):
    return (_unpickle_gspmd_sharding,
            (self._devices, self._hlo_sharding.to_proto(), self._memory_kind))

  @functools.cached_property
  def _hlo_sharding_hash(self):
    if self.is_fully_replicated:
      return hash(replicated_hlo_sharding)
    return hash(self._hlo_sharding)

  def __eq__(self, other):
    if not isinstance(other, GSPMDSharding):
      return False
    if self is other:
      return True
    return (are_hlo_shardings_equal(self._hlo_sharding, other._hlo_sharding)
            and self.memory_kind == other.memory_kind
            and self._internal_device_list == other._internal_device_list)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self._internal_device_list, self._hlo_sharding_hash,
                        self.memory_kind))
    return self._hash

  def __repr__(self):
    mem = '' if self._memory_kind is None else f', memory_kind={self._memory_kind}'
    return f'GSPMDSharding({self._hlo_sharding!r}{mem})'

  def check_compatible_aval(self, aval_shape: Shape) -> None:
    num_ways_dim_sharded, _ = get_num_ways_dim_sharded(self._hlo_sharding)
    if len(aval_shape) < len(num_ways_dim_sharded):
      raise ValueError(
          f"Sharding {self} is only valid for values of rank at least "
          f"{len(num_ways_dim_sharded)}, but was applied to a value of rank "
          f"{len(aval_shape)}")

  @property
  def num_devices(self) -> int:
    return len(self._internal_device_list)

  @functools.cached_property
  def device_set(self) -> set[Device]:
    return set(self._devices)

  @property
  def memory_kind(self) -> str | None:
    return self._memory_kind

  def with_memory_kind(self, kind: str) -> GSPMDSharding:
    return GSPMDSharding(self._devices, self._hlo_sharding, memory_kind=kind)

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return tuple(self._devices)

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    return self._hlo_sharding

  def _to_sdy_sharding(self, num_dimensions: int,
                       modify_wrt_axis_types: bool = False) -> SdyArray:
    if self._hlo_sharding.tuple_elements():
      raise TypeError(
          f'Cannot convert GSPMDSharding {self._hlo_sharding} into SdyArray.')
    elif self._hlo_sharding.is_replicated():
      empty_mesh = AbstractMesh((), ())
      return NamedSharding(empty_mesh, PartitionSpec()
                           )._to_sdy_sharding(num_dimensions)
    elif self._hlo_sharding.is_tiled():
      if not self._hlo_sharding.is_tile_assignment_iota():
        raise TypeError(
            f'Cannot convert GSPMDSharding {self._hlo_sharding} into SdyArray.')
      axis_sizes = tuple(self._hlo_sharding.get_axis_sizes())
      axis_names = tuple(f'_axis_{i}' for i in range(len(axis_sizes)))
      mesh = AbstractMesh(axis_sizes, axis_names)
      return _gspmd_to_named_sharding_via_mesh(
          self, mesh)._to_sdy_sharding(num_dimensions)
    else:
      raise TypeError(
          f'Cannot convert GSPMDSharding {self._hlo_sharding} into SdyArray.')

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    return is_hlo_sharding_replicated(self._hlo_sharding)

  @functools.cached_property
  def is_fully_addressable(self) -> bool:
    return self._internal_device_list.is_fully_addressable

  @classmethod
  def get_replicated(cls, device_assignment, *, memory_kind: str | None = None):
    return cls(device_assignment, replicated_hlo_sharding,
               memory_kind=memory_kind)


MeshAxisName = Any


def prepare_axis_resources(axis_resources, arg_name,
                           allow_unconstrained_dims=False):
  entries, treedef = tree_util.tree_flatten(
      axis_resources, is_leaf=lambda x: x is None)
  what = f"{arg_name} leaf specifications"

  new_entries: list[Any] = []
  for entry in entries:
    if isinstance(entry, UnspecifiedValue) or entry is None:
      new_entries.append(entry)
    elif isinstance(entry, jsharding.Sharding):
      if isinstance(entry, NamedSharding) and entry.mesh.empty:
        raise ValueError(f'One of {what} got an empty NamedSharding: {entry} '
                         'which is not allowed.')
      if (not allow_unconstrained_dims and isinstance(entry, NamedSharding) and
          PartitionSpec.UNCONSTRAINED in entry.spec.partitions):
        raise ValueError(
            f'Unconstrained dims are not allowed when passed to {arg_name}:'
            f' {entry}')
      new_entries.append(entry)
    else:
      if not isinstance(entry, PartitionSpec):
        raise TypeError(f"{what} are expected to be "
                        f"PartitionSpec instances or None, but got {entry}")
      if (not allow_unconstrained_dims and
          PartitionSpec.UNCONSTRAINED in entry.partitions):
        raise ValueError(
            f'Unconstrained dims are not allowed when passed to {arg_name}:'
            f' {entry}')
      _check_unique_resources(entry, arg_name)
      new_entries.append(entry)
  return tree_util.tree_unflatten(treedef, new_entries)


# 轴环境

@dataclasses.dataclass(frozen=True, slots=True)
class SPMDAxisContext:
  """使用 GSPMD 分区器的并行计算的硬件轴上下文。

  其中包含稍后用于执行该计算的 mesh，
  以及一组当前以 MANUAL 分片模式
  降级的 mesh 轴。
  """
  mesh: Mesh
  manual_axes: frozenset[MeshAxisName] = frozenset()


@dataclasses.dataclass(frozen=True, slots=True)
class ShardingContext:
  """使用分片接口的并行计算的
  硬件轴上下文。

  该上下文同样使用 GSPMD 分区器。
  """
  num_devices: int
  device_assignment: tuple[xc.Device, ...] | None = None
  abstract_mesh: AbstractMesh | None = None

  def __post_init__(self):
    if self.device_assignment is not None:
      assert isinstance(self.device_assignment, tuple)
      assert self.num_devices == len(self.device_assignment)


# -------------------- XLA OpSharding 转 PartitionSpec --------------------
# 注意，OpSharding 的表达能力比 PartitionSpec 更强，因此并不总能
# 在两者之间转换；但下面的代码至少应能处理
# 所有可转换的情况。

def strides_for_sizes(sizes):
  """返回从主到次(major-to-minor)的尺寸所对应的步长数组。"""
  return np.cumprod(sizes[::-1])[::-1] // np.asarray(sizes)

def unflatten_array(named_sizes, assignment):
  """根据设备分配还原轴名称的顺序。

  本函数能够转换为轴顺序的设备分配
  形如::

    np.arange(np.prod(named_sizes.values())).transpose(...).flatten()

  其中 ``...`` 表示某个转置。由 partition spec 生成的
  所有 OpSharding 分配都满足这一形式。

  Arguments:
    named_sizes: 从轴名称映射到其大小的字典。
    assignment: 0 到所有命名大小之积之间的整数
      的一个排列。

  Returns:
    与给定 assignment 对应的、从主到次的轴名称列表。
  """
  named_sizes = {name: size for name, size in named_sizes.items() if size != 1}
  sizes = np.fromiter(named_sizes.values(), dtype=np.int64)
  strides = strides_for_sizes(sizes)
  dims = explode_superdims(sizes, unflatten_superdims(assignment))
  dim_to_name = {(size, stride): name for size, stride, name in zip(sizes, strides, named_sizes)}
  return [dim_to_name[d] for d in dims]

def unflatten_superdims(assignment):
  """反扁平化一组维度大小及其步长，它们生成 assignment。

  如果本函数对给定的 ``assignment`` 成功，则应满足
  以下性质::

    dims_with_strides = unflatten_superdims(assignment)
    base_array = np.arange(map(fst, sorted(dims_with_strides, key=snd, reverse=True)))
    assignment == base_array.transpose(argsort(dims_with_strides, key=snd, reverse=True)).flatten()

  也就是说，返回的维度列出了基准数组的所有大小（步长
  表示它们的初始顺序）。列表中维度的顺序对应于
  作用在基准数组上、从而生成该 assignment 的排列。
  """
  def check(cond):
    if cond: return
    raise NotImplementedError("Failed to convert OpSharding into a ShardingSpec. "
                              "Please open a bug report!")
  flat_assignment = np.asarray(assignment, dtype=np.int64)
  check(flat_assignment[0] == 0)
  dims = []
  while flat_assignment.size > 1:
    stride = flat_assignment[1]
    for i in range(len(flat_assignment)):
      if flat_assignment[i] != i * stride: break
    else:
      # 该循环结束后，i 应指向“序列之后的元素”，因此如果整个
      # 数组是一个步长序列，就需要将它自增一次。
      i += 1
    size = i
    dims.append((size, stride))
    assert size > 1  # 确保有进展
    flat_assignment = flat_assignment[::size]
  return dims

def explode_superdims(sizes, dims):
  """拆分超维(superdim)以匹配已知形状。

  反扁平化过程可能错误地生成过少且过大的维度。
  例如 ``unflatten_superdims(np.arange(n))`` 总是返回 ``[(n, 1)]``。
  本函数接收这样一组连续的超维，并将其拆分为更小的
  维度，使得::

    set(map(fst, explode_superdims(sizes, dims))) == set(sizes)
  """
  strides_to_sizes = {stride: size for size, stride in zip(sizes, strides_for_sizes(sizes))}
  dims = list(reversed(dims))
  final_dims = []
  for size, stride in dims:
    target_size = strides_to_sizes[stride]
    new_dims = []
    while size > target_size:
      assert target_size > 1  # 确保有进展
      assert size % target_size == 0
      new_dims.append((target_size, stride))
      size //= target_size
      stride *= target_size
      target_size = strides_to_sizes[stride]
    assert size == target_size
    new_dims.append((size, stride))
    final_dims += reversed(new_dims)
  return final_dims

def parse_flatten_op_sharding(
    hlo_sharding: xc.OpSharding | xc.HloSharding,
    mesh: Mesh | AbstractMesh) -> Sequence[PartitionSpec]:
  if isinstance(hlo_sharding, xc.OpSharding):
    hlo_sharding = xc.HloSharding.from_proto(hlo_sharding)
  if hlo_sharding.tuple_elements():
    out: list[PartitionSpec] = []
    for s in hlo_sharding.tuple_elements():
      out.extend(parse_flatten_op_sharding(s, mesh))
    return out
  elif hlo_sharding.is_replicated():
    return [PartitionSpec()]
  elif hlo_sharding.is_maximal() and mesh.size == 1:
    return [PartitionSpec()]
  elif hlo_sharding.is_tiled():
    mesh_shape = mesh.shape
    mesh_axis_order = unflatten_array(
        mesh.shape, hlo_sharding.tile_assignment_devices()
    )
    mesh_axis = iter(mesh_axis_order)
    shape = hlo_sharding.tile_assignment_dimensions()
    partitions = []
    for dim_size in shape:
      dim_partitions = []
      while dim_size > 1:
        axis = next(mesh_axis)
        axis_size = mesh_shape[axis]
        quotient, remainder = divmod(dim_size, axis_size)
        if remainder != 0:
          raise jsharding.IndivisibleError(
              f'{shape=} is incompatible with {mesh_shape=}: '
              f'{dim_size=} is not divisible by {axis_size=}.')
        dim_size = quotient
        dim_partitions.append(axis)
      partitions.append(tuple(dim_partitions))
    if len(hlo_sharding.subgroup_types()) > 1:
      raise NotImplementedError(
          'Unhandled HloSharding type. Please open a bug report!'
      )
    if hlo_sharding.replicate_on_last_tile_dim():
      partitions = partitions[:-1]
    while partitions and partitions[-1] == ():
      partitions.pop()
    return [PartitionSpec(*partitions)]
  else:
    raise AssertionError("Unhandled OpSharding type. Please open a bug report!")


def _slice_as_tuple(s: slice):
  assert s.step is None
  return (s.start, s.stop)


class NonUniformShardingError(ValueError):
  """当分片在各进程间不一致时抛出。"""


@util.cache(max_size=4096, trace_context_in_key=False)
def get_process_index_and_count(
    tensor_sharding: jsharding.Sharding, dim: int, ndims: int) -> tuple[int, int]:
  """获取给定维度的当前进程索引与唯一进程数量。

  本函数便于把进程级数据映射到各个设备。
  每个进程都可以用自己的索引来获取与该索引对应的数据。
  如果进程级数据在多个维度上被分片，可以用本函数构造
  各个分片轴上索引的笛卡尔积。
  需要加载相同数据的进程会得到相同的索引。
  对于每进程数据不按网格分布的分片，其不同分片的数量
  会使得：在保持本地进程数据呈“立方体”形状的前提下，
  仍可以构造出目标形状。

  例如，对于 4 个主机且分片分布如下：

  1234
  2143

  维度 0（行）：所有进程都需要访问所有行，因此返回 (0, 1)
  维度 1（列）：
     进程 1 和 2 返回 2 个中的索引 0（需要第 0、1 列），
     进程 3 和 4 返回 2 个中的索引 1（需要第 2、3 列）。

  另一方面，对于如下分片：

  1212
  3434

  维度 0（行）：进程 1 和 2 返回 (0, 2)，进程 3 和 4 返回 (1, 2)
  维度 1（列）：进程 1 和 3 返回 (0, 2)，进程 2 和 4 返回 (1, 2)

  Note: 本函数要求分片在维度
  `dim` 上是进程一致的(process uniform)：
   每个进程在该维度上拥有相同数量的可寻址索引，并且
   各进程之间的所有索引集合要么互不相交，要么完全相同。

  分片要达到进程一致，其可寻址分片不必构成
  连续的子张量，甚至不必构成稀疏网格；对于交错的高维
  张量，分片可能只在部分维度上进程一致，
  而在其他维度上并不一致。

  例如：
    1111 and 12 and 1212 and 1212
    2222     21     2121     1212

  这些分片在两个维度上都是一致的。然而

    1122
    2121
    1121
    1222

  它在维度 0 上是一致的（两个主机都访问所有行），但
  在维度 1 上不一致（主机 1 访问第 0、1、3 列），
  而主机 2 访问 (0, 1, 2, 3)。

  Returns:
    给定维度的 (index, num_distinct_shards) 元组。
    可以保证在所有进程上，`index` 都会覆盖 0 到
    `num_distinct_shards - 1`。

  Raises:
    NonUniformShardingError: 如果分片在维度
    `dim` 上不是进程一致的。
  """
  # TODO(sandler, yashkatariya): 考虑将此函数公开。

  if (tensor_sharding.is_fully_addressable or
      tensor_sharding.is_fully_replicated):
    return (0, 1)
  # 获取设备到索引的映射，我们并不关心这里具体的全局形状，
  # 只是要用 (num_devices, num_devices, ...) 得到分片
  # 在张量上的分布。这是一个与任何拥有 num_devices 个设备的
  # mesh 都兼容的通用形状。
  device_map = tensor_sharding.devices_indices_map(
      (tensor_sharding.num_devices,) * ndims)

  # 获取所有设备在 'dim' 维度上的切片。
  global_slice = {k: v[dim] for k, v in device_map.items()}

  # 保存从 process_index 到该进程切片集合的映射。
  process_to_slice = collections.defaultdict(set)
  # 保存所有进程的全局切片集合。
  all_slices = set()

  # 计算每个进程的切片集合以及全局切片集合。
  for d, v in global_slice.items():
    key = (v.start, v.stop)
    process_to_slice[d.process_index].add(key)
    all_slices.add(key)

  # 获取当前进程的切片集合，我们将用它来计算
  # 当前进程的索引。
  current_pid = next(iter(tensor_sharding.addressable_devices)).process_index
  addressable_slices = frozenset(process_to_slice[current_pid])

  # 校验所有进程拥有相同数量的切片。
  slices_per_process = len(addressable_slices)
  if any(len(x) != slices_per_process for x in process_to_slice.values()):
    raise NonUniformShardingError(
        f'{tensor_sharding=} is non-uniform on {dim=} as some processes have '
        'different number of slices.'
    )
  unique_processes = list({frozenset(x) for x in process_to_slice.values()})

  # 去掉重复的进程后，所有唯一的切片应恰好
  # 覆盖该维度一次。如果不满足，就说明
  # 分片不是一致的。
  if sum(len(h) for h in unique_processes) != len(all_slices):
    raise NonUniformShardingError(
        f'{tensor_sharding=} is non-uniform on {dim=}'
    )
  return (unique_processes.index(addressable_slices), len(unique_processes))


def local_to_global_shape(
    sharding: jsharding.Sharding, local_shape: Shape) -> tuple[int | None, ...]:
  """在可能时根据每进程数据计算全局形状。

  返回的形状在该维度上给出全局张量的大小，若无法计算
  则为 None。当分片沿该维度不一致时会出现后者，
  例如不同主机需要不同的形状，
  或者不同进程的数据存在部分重叠。

  如果至多只有一个维度被分片，形状总是可计算的。
  一般来说，对大多数实用 mesh（包括感知拓扑的 mesh，
  例如 mesh_utils.create_device_mesh 返回的 mesh）都可计算全局形状。

  一些示例：假设 mesh 为 {'a': 2, 'b': 2, 'c': 2}，每个主机 2 个设备，
  共 4 个主机。对于不同的 spec 我们得到：
  - P():
      global_shape = local_shape

  - P(('a', 'b', 'c'), None):
      global_shape =  (4 * local_shape[0], local_shape[1])
      Note: 每设备形状为 (local_shape[0] / 2, local_shape[1])

  - P(('a', 'b'), None)
      global_shape =  (4 * local_shape[0], local_shape[1])
      # NB: 与上面相同的全局形状，因为沿 'c' 维度的分片
      # 恰好发生在进程内部，因此不影响全局形状。
      # 底层差异体现在每 *设备* 形状上，此处
      # 该形状为 (local_shape[0], local_shape[1])。

  - P(None, ('a', 'c'))
      global_shape = (local_shape[0], 2 * local_shape[1])
      # 每设备形状为 (local_shape[0], local_shape[1] / 2)
  - P(('a', 'c'), 'b'):
      global_shape = (2 * local_shape[0], 2 * local_shape[1])
      # 每设备形状为 (local_shape[0] / 2, local_shape[1])
  - 如果 Mesh 中的设备被随机置换：对于任何
  分片了多于 1 个轴的 partition spec：例如 P('a', ('b', 'c'))：
      global_shape = (None, None)

  Args:
    local_shape: 张量的全局形状。

  Returns:
    global_shape，其中非一致维度上为 None。
  """

  global_shape : list[int | None] = [None] * len(local_shape)
  for i, local_dim in enumerate(local_shape):
    try:
      _, shard_count = get_process_index_and_count(
          sharding, i, ndims=len(local_shape))
      global_shape[i] = local_dim * shard_count
    except NonUniformShardingError:
      global_shape[i] = None
      continue

  return tuple(global_shape)


def num_addressable_indices(
    tensor_sharding: jsharding.Sharding, dim: int, global_shape: Shape) -> int:
  """返回本主机对给定维度可访问的索引数量。

  每个主机都可以拥有多个设备，
  这些设备跨越的数据切片可能并不连续。
  本函数计算其任一可寻址设备在维度 `dim` 上
  所持有的唯一索引总数。

  大多数情况下，可寻址索引构成稀疏网格（某些情况下
  构成子立方体），因此每个主机在每个维度上持有的
  索引数量相同。然而，也可以设计出使得可寻址分片
  构成复杂模式的 mesh。此时返回值是至少被一个
  设备可寻址的索引数量。

  例如，假设分片如下所示：（数字表示
  主机索引）

    1221
    1221
    0000

  那么在主机 1 和 2 上，dim 0（行）和 dim=1（列）的大小都是 2，
  而在主机 0 上，dim 0 的大小为 1，dim 1 的大小为 4。

  Args:
    tensor_sharding: 张量的分片。
    dim: 沿其计算可寻址索引数量的维度。
    global_shape: 张量的全局形状。

  Returns:
    本主机在维度 `dim` 上持有的索引数量。
  """
  # TODO(sandler, yashkatariya): 考虑将此函数公开。
  addressables = tensor_sharding.addressable_devices_indices_map(global_shape)
  addressables = cast(Mapping[jsharding.Device, Index], addressables)
  num_unique_slices = len({
      _slice_as_tuple(addressable[dim]) for addressable in addressables.values()
  })
  shard_size = tensor_sharding.shard_shape(global_shape)[dim]
  return shard_size * num_unique_slices


def physical_hlo_sharding(aval, hlo_sharding: xc.HloSharding) -> xc.HloSharding:
  elt_aval = core.physical_element_aval(aval.dtype)
  new_op_sharding = hlo_sharding.to_proto().clone()
  partitions, num_replicas = get_num_ways_dim_sharded(hlo_sharding)
  suffix = [] if num_replicas == 1 else [num_replicas]
  tad = partitions + [1] * elt_aval.ndim + suffix
  new_op_sharding.tile_assignment_dimensions = tad
  return xc.HloSharding.from_proto(new_op_sharding)


def make_key_array_phys_sharding(aval, sharding):
  if sharding.num_devices == 1:
    return sharding
  elif isinstance(sharding, NamedSharding):
    elt_aval = core.physical_element_aval(aval.dtype)
    trailing_spec = [None] * elt_aval.ndim
    out_partitions = (*sharding.spec.partitions, *trailing_spec)
    return sharding.update(spec=sharding.spec.update(partitions=out_partitions))
  else:
    hlos = sharding._to_xla_hlo_sharding(aval.ndim)
    return GSPMDSharding(
        sharding._internal_device_list, physical_hlo_sharding(aval, hlos))


def physical_sharding(aval, sharding: jsharding.Sharding) -> jsharding.Sharding:
  return make_key_array_phys_sharding(aval, sharding)


def get_logical_gspmd_sharding(logical_shape, dtype, phys_sharding):
  elt_aval = core.physical_element_aval(dtype)
  phys_hlo_sharding = phys_sharding._to_xla_hlo_sharding(
      len(logical_shape) + elt_aval.ndim)
  partitions, num_replicas = get_num_ways_dim_sharded(phys_hlo_sharding)
  suffix = [] if num_replicas == 1 else [num_replicas]
  # 通过裁掉被复制的尾部维度来构造逻辑分片。
  logical_op_sharding = phys_hlo_sharding.to_proto().clone()
  tad = partitions[:-elt_aval.ndim] + suffix
  logical_op_sharding.tile_assignment_dimensions = tad
  return GSPMDSharding(phys_sharding._internal_device_list,
                       xc.HloSharding.from_proto(logical_op_sharding))

def check_replicated_trailing_dims(sharding: jsharding.Sharding,
                                   logical_shape, dtype):
  if isinstance(sharding, NamedSharding) and sharding.mesh._any_axis_manual:
    return
  phys_shape = core.physical_shape(logical_shape, dtype)
  hlo_s = sharding._to_xla_hlo_sharding(len(phys_shape))
  partitions, _ = get_num_ways_dim_sharded(hlo_s)
  num_trailing_dims = len(phys_shape) - len(logical_shape)
  if not all(i == 1 for i in partitions[-num_trailing_dims:]):
    raise AssertionError(
        "The trailing dims of extended dtypes should be replicated. Got"
        f" sharding: {sharding}, partitions: {partitions}, "
        f"num_trailing_dims: {num_trailing_dims}")

def logical_sharding(logical_shape, dtype, phys_sharding) -> jsharding.Sharding:
  # 尾部维度应始终保持复制状态。
  # TODO(yashkatariya): 也许可以移除这个检查，或把它放到 pxla 层？
  check_replicated_trailing_dims(phys_sharding, logical_shape, dtype)

  if isinstance(phys_sharding, NamedSharding):
    elt_aval = core.physical_element_aval(dtype)
    phys_shape = core.physical_shape(logical_shape, dtype)
    if len(phys_sharding.spec) < len(phys_shape):
      phys_spec = (*phys_sharding.spec,
                   *[None] * (len(phys_shape) - len(phys_sharding.spec)))
    else:
      phys_spec = phys_sharding.spec
    return phys_sharding.update(spec=phys_spec[:-elt_aval.ndim])
  elif phys_sharding.num_devices == 1:
    return phys_sharding
  else:
    return get_logical_gspmd_sharding(logical_shape, dtype, phys_sharding)


@util.cache()
def cached_named_sharding(
    mesh: Mesh | AbstractMesh, pspec: PartitionSpec,
    memory_kind: str | None = None) -> NamedSharding:
  return NamedSharding(mesh, pspec, memory_kind=memory_kind)


def _gspmd_to_named_sharding_via_mesh(
    out_s: GSPMDSharding, mesh: Mesh | AbstractMesh
) -> NamedSharding:
  spec = parse_flatten_op_sharding(out_s._hlo_sharding, mesh)[0]
  return cached_named_sharding(mesh, spec, out_s.memory_kind)


@util.cache()
def canonicalize_sharding(sharding: NamedSharding | PartitionSpec | None,
                          api_name: str, check_mesh_consistency: bool = True
                          ) -> NamedSharding | None:
  if sharding is None:
    return None
  if isinstance(sharding, NamedSharding) and sharding.mesh.empty:
    return None
  if not isinstance(sharding, (NamedSharding, PartitionSpec)):
    raise TypeError(
        f"`out_sharding` argument of {api_name} only supports instances of"
        f" `NamedSharding` or `PartitionSpec`. Got {sharding} of type:"
        f" {type(sharding)}")

  cur_mesh = get_abstract_mesh()
  if isinstance(sharding, PartitionSpec):
    if cur_mesh.empty:
      raise ValueError(
          'Using PartitionSpec when you are not under a mesh context is not'
          ' allowed. Please pass a NamedSharding instance or enter into a mesh'
          f' context via `jax.set_mesh`. Got {sharding}')
    sharding = NamedSharding(cur_mesh, sharding)
  else:
    # 存在设置了多个 mesh 的情况。由于已有的用例，
    # 全自动模式允许这种做法。
    # TODO(yashkatariya): 一旦我们禁止使用不同的 mesh，就移除这段逻辑，
    # 并修复现有的用例。
    if (sharding.mesh.abstract_mesh.are_all_axes_auto and
        cur_mesh.are_all_axes_auto):
      check_mesh_consistency = False
    if (check_mesh_consistency and not cur_mesh.empty and
        sharding.mesh.abstract_mesh != cur_mesh):
      raise ValueError(
          f'Context mesh {cur_mesh} should match the mesh of sharding'
          f' {sharding.mesh.abstract_mesh} passed to {api_name}.'
          ' This error occurs at source: '
          f' {source_info_util.summarize(source_info_util.current())}')
    # TODO(yashkatariya): 也许可以对 jnp.zeros 之类的 API 在顶层
    # 允许具体 mesh，即 `core.trace_state_clean()`？
    if isinstance(sharding.mesh, Mesh):
      sharding = NamedSharding(sharding.mesh.abstract_mesh, sharding.spec)

  for s in it.chain(flatten_spec(sharding.spec), sharding.spec.unreduced,
                    sharding.spec.reduced):
    if s is None:
      continue
    if sharding.mesh._name_to_type[s] in {
        AxisType.Auto, AxisType.Manual}:
      raise ValueError(
          f'PartitionSpec passed to {api_name} cannot contain axis'
          ' names that are of type Auto or Manual. Got PartitionSpec:'
          f' {sharding.spec} with axis name: {s} of type:'
          f' {sharding.mesh._name_to_type[s]}. This error occurs at source: '
          f' {source_info_util.summarize(source_info_util.current())}')
  return sharding


def make_mesh(axis_sizes: Sequence[int], axis_names: Sequence[str],
              axis_types: tuple[AxisType, ...] | None = None,
              *, devices: Sequence[xc.Device] | None = None) -> Mesh:
  """按指定的形状和轴名称创建高效的 mesh。

  本函数尝试自动计算从一组逻辑轴到物理 mesh 的
  良好映射。例如，在拥有 8 个设备的 TPU v3 上：

  >>> mesh = jax.make_mesh((8,), ('x'))  # doctest: +SKIP
  >>> [d.id for d in mesh.devices.flat]  # doctest: +SKIP
  [0, 1, 2, 3, 6, 7, 4, 5]

  上面的顺序考虑了 TPU v3 的物理拓扑。
  它把设备排成一个环，从而在 TPU v3 上
  获得高效的 all-reduce。

  现在看另一个例子，使用 TPU v3 的 16 个设备：

  >>> mesh = jax.make_mesh((2, 8), ('x', 'y'))  # doctest: +SKIP
  >>> [d.id for d in mesh.devices.flat]  # doctest: +SKIP
  [0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11, 14, 15, 12, 13]
  >>> mesh = jax.make_mesh((4, 4), ('x', 'y'))  # doctest: +SKIP
  >>> [d.id for d in mesh.devices.flat]  # doctest: +SKIP
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

  可以看到，逻辑轴（`axis_sizes`）会影响
  设备的顺序。

  如果你想使用 `jax.experimental.mesh_utils.create_device_mesh`
  提供的额外参数，例如 `contiguous_submeshes` 和
  `allow_split_physical_axes`，可以使用该函数。

  Args:
    axis_sizes: mesh 的形状。例如 axis_shape=(4, 2)
    axis_names: mesh 各轴的名称。例如 axis_names=('x', 'y')
    axis_types: 可选的 :class:`jax.sharding.AxisType` 元组，
      与 ``axis_names`` 一一对应。更多信息
      参见 `Explicit Sharding`_。
    devices: 可选的关键字参数，用于指定
      你希望用来创建 mesh 的设备。

  Returns:
    一个 :class:`jax.sharding.Mesh` 对象。

  .. _Explicit Sharding:  https://docs.jax.dev/en/latest/parallel.html
  """
  if devices is None:
    devices = xb.devices()
  new_axis_sizes = mesh_utils._canonicalize_axis_sizes(axis_sizes)
  if new_axis_sizes is None:
    raise ValueError(
        '`axis_sizes` passed to `make_mesh` should be a sequence of ints.'
        f' Got {axis_sizes}')
  del axis_sizes

  axis_size = math.prod(new_axis_sizes)
  if axis_size > len(devices):
    raise ValueError(
        f'Number of devices {len(devices)} must be >= the product '
        f'of mesh_shape {new_axis_sizes}')
  elif axis_size < len(devices):
    devices = devices[:axis_size]
  if devices[0].device_kind in (mesh_utils._TPU_V5_LITE, mesh_utils._TPU_V5E):
    allow_split_physical_axes = True
  else:
    allow_split_physical_axes = False
  mesh_devices = mesh_utils.create_device_mesh(
      new_axis_sizes, devices,
      allow_split_physical_axes=allow_split_physical_axes)
  if (hasattr(mesh_devices.flat[0], 'slice_index') and
      len({d.slice_index for d in mesh_devices.flat}) > 1):
    raise ValueError(
        '`jax.make_mesh` does not support multi-slice topologies. Please use'
        ' jax.experimental.mesh_utils.create_hybrid_device_mesh')
  if axis_types is None:
    axis_types = (AxisType.Explicit,) * len(mesh_devices.shape)
  return Mesh(mesh_devices, axis_names, axis_types=axis_types)


class set_mesh:
  """在线程本地上下文中设置具体 mesh。

  ``jax.set_mesh`` 具有双重行为。你既可以把它当作全局设置器，
  也可以当作上下文管理器使用。

  当通过 ``jax.set_mesh`` 使某个 mesh 处于上下文中时，
  你可以向所有接受 sharding 参数的 API 传入原始的 PartitionSpec。
  启用显式分片模式也需要使用 ``jax.set_mesh``：
  https://docs.jax.dev/en/latest/parallel.html

  例如::

    mesh = jax.make_mesh((2,), ('x',))
    jax.set_mesh(mesh)  # 把该 API 当作全局设置器使用

    with jax.set_mesh(mesh):  # 把该 API 当作上下文管理器使用
      ...

  Note: ``jax.set_mesh`` 只能在 ``jax.jit`` 之外使用。
  """
  __slots__ = ["prev_abstract_mesh", "prev_mesh"]

  def __init__(self, mesh: Mesh | None):
    if mesh is not None and not isinstance(mesh, Mesh):
      raise ValueError(
          f"Expected mesh of type `jax.sharding.Mesh`. Got {type(mesh)}")
    if not core.trace_state_clean():
      raise ValueError('`set_mesh` can only be used outside of `jax.jit`.')
    if mesh is not None and mesh._any_axis_manual:
      raise ValueError(
          f'mesh {mesh} contains manual axes which is not allowed when using'
          ' `jax.set_mesh`. Please use `jax.shard_map` to enter into `Manual`'
          ' mode instead.')

    abs_mesh = empty_abstract_mesh if mesh is None else mesh.abstract_mesh
    conc_mesh = empty_concrete_mesh if mesh is None else mesh
    self.prev_abstract_mesh = config.abstract_mesh_context_manager.swap_local(
        abs_mesh)
    self.prev_mesh = config.device_context.swap_local(conc_mesh)

  def __enter__(self):
    pass

  def __exit__(self, exc_type, exc_value, traceback):
    config.abstract_mesh_context_manager.set_local(self.prev_abstract_mesh)
    config.device_context.set_local(self.prev_mesh)


def get_mesh() -> Mesh:
  if not core.trace_state_clean():
    raise ValueError(
        '`get_mesh` can only be used outside of `jax.jit`. Maybe you want'
        ' `jax.sharding.get_abstract_mesh()`?')
  return get_concrete_mesh()


@contextlib.contextmanager
def _internal_use_concrete_mesh(mesh: Mesh):
  assert isinstance(mesh, Mesh)
  prev_val = config.device_context.swap_local(mesh)
  try:
    yield
  finally:
    config.device_context.set_local(prev_val)
