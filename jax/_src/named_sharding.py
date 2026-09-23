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

# 文件职责：用命名轴（mesh 轴名 + `PartitionSpec`）表达数组分片方式。
# 中心是 `NamedSharding`，它把设备网格 `Mesh` 与 `PartitionSpec` 组合起来，
# 并给出设备集合、地址可达设备、全复制/复制轴等属性，是 `jax.jit` 的
# `in_shardings` / `out_shardings` 与 `jax.device_put` 等接口常用的分片类型。
# 模块还负责下游表示转换：`NamedSharding` 转换为 XLA 的 `HloSharding`
# （含手动/非规约轴与逻辑设备顺序）以及 shardy 的 `SdyArray`。
# 校验函数检查 `PartitionSpec` 中的轴在 mesh 中唯一存在且类型合法。

from __future__ import annotations

from collections.abc import Sequence
import collections
import dataclasses
import functools
from typing import Any

from jax._src.util import use_cpp_class, cache, use_cpp_method, unzip3
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir.dialects import sdy
from jax._src import mesh as mesh_lib
from jax._src.mesh import AxisType
from jax._src.partition_spec import PartitionSpec, UnreducedKind
from jax._src import sharding as jsharding
import numpy as np
from jax._src.lib import ifrt_version, jaxlib_extension_version

Shape = tuple[int, ...]
Device = xc.Device
Index = tuple[slice, ...]
XLADeviceAssignment = Sequence[Device]


class UnspecifiedValue:
  def __repr__(self):
    return "UnspecifiedValue"
UNSPECIFIED = UnspecifiedValue()


MeshAxisName = Any

"""
ArrayMapping 规定了 ndarray 应该如何映射到 mesh 轴上。

注意：当该映射不是单射时（即多个 mesh 轴映射到同一个位置轴），
条目的顺序至关重要。此时映射条目的顺序决定了 mesh 轴上从主到次
的顺序，重复维度上的数据块会按这个顺序被分配。

例如，考虑映射 {'x': 1, 'y': 1} 以及形状为 {'x': 2, 'y': 3} 的 mesh。
值的第二个维度会被切成 6 块，并按把 'y' 当作变化最快（次要）维度的
方式分配到 mesh 上。在这种情况下，这意味着一维的数据块列表可以不加
改动地分配给展平后的 mesh 设备列表。如果映射是 {'y': 1, 'x': 1}，
则必须先对 mesh 设备组成的 ndarray 做转置，再进行展平和分配。
"""
ArrayMapping = collections.OrderedDict[MeshAxisName, int]
ArrayMappingOrAutoOrUnspecified = ArrayMapping | UnspecifiedValue


def _unpickle_named_sharding(mesh, spec, memory_kind, logical_device_ids):
  return NamedSharding(mesh, spec, memory_kind=memory_kind,
                       _logical_device_ids=logical_device_ids)


@use_cpp_class(xc.NamedSharding)
class NamedSharding(jsharding.Sharding):
  r"""`NamedSharding` 用命名轴来表达分片。

  `NamedSharding` 由一对对象组成：设备组成的 `Mesh`，以及描述如何在该
  mesh 上对一个数组做分片的 `PartitionSpec`。

  `Mesh` 是 JAX 设备组成的多维 NumPy 数组，其中 mesh 的每个轴都有一个
  名字，例如 ``'x'`` 或 ``'y'``。

  `PartitionSpec` 是一个元组，其元素可以是 ``None``、一个 mesh 轴，
  或者一个由 mesh 轴组成的元组。每个元素描述输入的某个维度如何被划分到
  零个或多个 mesh 维度上。例如，``PartitionSpec('x', 'y')`` 表示数据的
  第一个维度沿 mesh 的 ``x`` 轴分片，第二个维度沿 mesh 的 ``y`` 轴分片。

  `Distributed arrays and automatic parallelization`_
  与 `Explicit Sharding`_ 教程给出了更多细节和图示，
  解释 `Mesh` 与 `PartitionSpec` 的用法。

  Args:
    mesh: 一个 :class:`jax.sharding.Mesh` 对象。
    spec: 一个 :class:`jax.sharding.PartitionSpec` 对象。
    memory_kind: 表示该分片的内存类型的字符串。

  Examples:

    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
    >>> spec = P('x', 'y')
    >>> named_sharding = jax.sharding.NamedSharding(mesh, spec)

  .. _Distributed arrays and automatic parallelization: https://docs.jax.dev/en/latest/parallel.html
  .. _Explicit Sharding:  https://docs.jax.dev/en/latest/parallel.html
  """

  mesh: mesh_lib.Mesh | mesh_lib.AbstractMesh
  spec: PartitionSpec
  _memory_kind: str | None
  _logical_device_ids: tuple[int, ...] | None

  @use_cpp_method()
  def __init__(
      self, mesh: mesh_lib.Mesh | mesh_lib.AbstractMesh, spec: PartitionSpec, *,
      memory_kind: str | None = None, _logical_device_ids=None):
    self.mesh = mesh
    self.spec = spec
    self._memory_kind = memory_kind
    self._logical_device_ids = _logical_device_ids
    check_pspec(self.mesh, self.spec)

  def __repr__(self):
    mem = '' if self.memory_kind is None else f', memory_kind={self.memory_kind}'
    ldi = ('' if self._logical_device_ids is None else
           f', logical_device_ids={self._logical_device_ids}')
    mesh_repr = f"{str(self.mesh)}"
    return f'NamedSharding(mesh={mesh_repr}, spec={self.spec}{mem}{ldi})'

  def __reduce__(self):
    return (_unpickle_named_sharding,
            (self.mesh, self.spec, self.memory_kind, self._logical_device_ids))

  @property
  def memory_kind(self) -> str | None:
    return self._memory_kind

  @use_cpp_method()
  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(
          (self.mesh, self.memory_kind, self.spec, self._logical_device_ids))
    return self._hash

  @use_cpp_method()
  def __eq__(self, other):
    if not isinstance(other, NamedSharding):
      return False
    if self is other:
      return True
    if (self.spec != other.spec
        or self.memory_kind != other.memory_kind
        or self._logical_device_ids != other._logical_device_ids):
      return False
    return self.mesh is other.mesh or self.mesh == other.mesh

  def check_compatible_aval(self, aval_shape: Shape) -> None:
    if len(aval_shape) < len(self.spec):
      extra_msg = (' For scalars the PartitionSpec should be P()'
                   if len(aval_shape) == 0 else '')
      raise ValueError(
          f"Sharding {self} is only valid for values of rank at least "
          f"{len(self.spec)}, but was applied to a value of rank "
          f"{len(aval_shape)}.{extra_msg}")

  @property
  def num_devices(self) -> int:
    return self.mesh.size

  @property
  def device_set(self) -> set[Device]:
    if isinstance(self.mesh, mesh_lib.AbstractMesh):
      raise ValueError(
          'device_set is not implemented for `jax.sharding.AbstractMesh`.')
    return self.mesh._flat_devices_set

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    if isinstance(self.mesh, mesh_lib.AbstractMesh):
      raise ValueError('_device_assignment is not implemented for'
                       ' `jax.sharding.AbstractMesh`.')
    return self.mesh._flat_devices_tuple

  @property
  def is_fully_addressable(self) -> bool:
    if isinstance(self.mesh, mesh_lib.AbstractMesh):
      raise ValueError('is_fully_addressable is not implemented for '
                       '`jax.sharding.AbstractMesh`.')
    # 如果 addressable_device_list 为空则返回 False。
    return self._internal_device_list.is_fully_addressable

  @property
  def _is_concrete(self) -> bool:
    if isinstance(self.mesh, mesh_lib.AbstractMesh):
      return False
    return True

  @property
  def addressable_devices(self) -> set[Device]:
    if isinstance(self.mesh, mesh_lib.AbstractMesh):
      raise ValueError('addressable_devices is not implemented for '
                       '`jax.sharding.AbstractMesh`.')
    # 重写 addressable devices，因为多个 NamedSharding 对象
    # 很可能共用同一个 mesh。
    return self.mesh._local_devices_set

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    if self.mesh.size == 1:
      return True
    if self.spec.unreduced:
      return False
    array_mapping = get_array_mapping(self.spec)
    mesh_shape = self.mesh.shape
    num_partitions = 1
    for name in array_mapping:
      num_partitions *= mesh_shape[name]
    return num_partitions == 1

  @functools.cached_property
  def replicated_axes(self) -> frozenset[MeshAxisName]:
    return get_replicated_axes(self.spec, self.mesh)

  def with_memory_kind(self, kind: str) -> NamedSharding:
    return self.update(memory_kind=kind)

  def update(self, **kwargs) -> NamedSharding:
    spec = kwargs.pop("spec", self.spec)
    if not isinstance(spec, PartitionSpec):
      spec = PartitionSpec(*spec)
    return NamedSharding(
        mesh=kwargs.pop("mesh", self.mesh),
        spec=spec,
        memory_kind=kwargs.pop("memory_kind", self.memory_kind),
        _logical_device_ids=kwargs.pop("_logical_device_ids",
                                       self._logical_device_ids))

  def is_equivalent_to(self, other, ndim: int) -> bool:
    if (isinstance(self.mesh, mesh_lib.AbstractMesh) and
        isinstance(other.mesh, mesh_lib.AbstractMesh)):
      return jsharding.common_is_equivalent_to(self, other, ndim,
                                               check_devices=False)
    return jsharding.common_is_equivalent_to(self, other, ndim)

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    return named_sharding_to_xla_hlo_sharding(
        self.mesh.abstract_mesh, self.spec, self._logical_device_ids,
        num_dimensions)

  def _to_sdy_sharding(self, num_dimensions: int,
                       modify_wrt_axis_types: bool = False) -> SdyArray:
    """降级为 shardy 对 NamedSharding 的表示。

    当 modify_wrt_axis_types=True 时，`Explicit` 类型的 mesh 轴如果在
    PartitionSpec 中未被使用，就会在 `SdyArray` 中被标记为
    `replicated_axes`。这意味着 shardy 不能再用这些轴去分片任何开放的维度。
    """
    return named_sharding_to_sdy_sharding(self, num_dimensions,
                                          modify_wrt_axis_types)

NamedSharding.__module__ = 'jax.sharding'


def get_replicated_axes(spec, mesh):
  flat_spec = frozenset(
      s for s in flatten_spec(spec)
      if s is not None and s is not PartitionSpec.UNCONSTRAINED)
  return frozenset(mesh.axis_names) - (flat_spec | spec.unreduced | spec.reduced)

def flatten_spec(spec):
  out = []
  for s in (spec.partitions if isinstance(spec, PartitionSpec) else spec):
    if isinstance(s, tuple):
      out.extend(s)
    else:
      out.append(s)
  return out

def get_array_mapping(axis_resources):
  if isinstance(axis_resources, UnspecifiedValue):
    return axis_resources
  d = collections.OrderedDict()
  for i, axes in enumerate(axis_resources.partitions):
    if axes is None or axes is PartitionSpec.UNCONSTRAINED:
      continue
    axes = axes if isinstance(axes, tuple) else (axes,)
    for axis in axes:
      d[axis] = i
  return d

@dataclasses.dataclass(frozen=True, slots=True)
class SdyDim:
  axes: tuple[str, ...]
  is_open: bool

  replace = dataclasses.replace

  def build(self) -> sdy.DimensionShardingAttr:
    return sdy.DimensionShardingAttr.get(
        [sdy.AxisRefAttr.get(axis) for axis in self.axes],
        is_closed=not self.is_open)

  def __repr__(self):
    return f'SdyDim({self._custom_repr()})'

  def _custom_repr(self):
    axes_repr = ', '.join(f"'{a}'" for a in self.axes)
    open_repr = ''
    if self.is_open:
      open_repr = ', ?' if self.axes else '?'
    return f'{{{axes_repr}{open_repr}}}'


def _get_axes(axes, mesh_shape):
  if not axes:
    return ()
  assert mesh_shape is not None
  # 按 mesh 轴名排序，使顺序确定，避免在 McJAX 中挂起
  return tuple(n for n, _ in mesh_shape if n in axes)


@dataclasses.dataclass(kw_only=True, frozen=True, slots=True)
class SdyArray:
  mesh_shape: tuple[tuple[str, int], ...] | None
  dim_shardings: tuple[SdyDim, ...]
  logical_device_ids: tuple[int, ...] | None = None
  replicated_axes: frozenset[str] = frozenset()
  unreduced_axes: frozenset[str] = frozenset()
  reduction_op: Any = None

  replace = dataclasses.replace

  def build(self, attr_cache: dict[SdyArray, sdy.TensorShardingAttr]
            ) -> sdy.TensorShardingAttr:
    attr = attr_cache.get(self, None)
    if attr is not None:
      return attr

    if self.mesh_shape is None:
      mesh_attr = sdy.MeshAttr.get([])
    else:
      ldi = ([] if self.logical_device_ids is None else
             list(self.logical_device_ids))
      mesh_attr = sdy.MeshAttr.get(
          [sdy.MeshAxisAttr.get(name, size) for name, size in self.mesh_shape],
          ldi)

    replicated_axes = _get_axes(self.replicated_axes, self.mesh_shape)
    unreduced_axes = _get_axes(self.unreduced_axes, self.mesh_shape)
    attr = sdy.TensorShardingAttr.get(
        mesh_attr,
        [dim_sharding.build() for dim_sharding in self.dim_shardings],
        replicated_axes=[sdy.AxisRefAttr.get(axis) for axis in replicated_axes],
        unreduced_axes=[sdy.AxisRefAttr.get(axis) for axis in unreduced_axes],
        reduction_op=self.reduction_op)  # type: ignore
    attr_cache[self] = attr
    return attr

  def __repr__(self):
    dim_sharding_repr = ', '.join(
        d._custom_repr() for d in self.dim_shardings)
    device_id_repr = (f', device_ids={self.logical_device_ids}'
                      if self.logical_device_ids is not None else '')
    rar = (f', replicated_axes={self.replicated_axes}'
           if self.replicated_axes else '')
    ur = (f', unreduced_axes={self.unreduced_axes}'
          if self.unreduced_axes else '')
    red_op = (f', reduction_op={self.reduction_op}'
              if self.reduction_op is not None else '')
    return f"SdyArray([{dim_sharding_repr}]{device_id_repr}{rar}{ur}{red_op})"


def remove_size_one_mesh_axis_from_spec(spec, mesh) -> PartitionSpec:
  new_spec: list[Any] = []
  for s in spec.partitions:
    if s is None or s is PartitionSpec.UNCONSTRAINED:
      new_spec.append(s)
    elif isinstance(s, tuple):
      new_spec.append(tuple(i for i in s if mesh.shape[i] != 1))
    else:
      new_spec.append(None if mesh.shape[s] == 1 else s)
  unreduced = frozenset(u for u in spec.unreduced if mesh.shape[u] != 1)
  reduced = frozenset(r for r in spec.reduced if mesh.shape[r] != 1)
  u_kind = spec.unreduced_kind if unreduced else None
  return PartitionSpec(*new_spec, unreduced=unreduced, reduced=reduced,
                       unreduced_kind=u_kind)


def get_non_one_sized_mesh_spec(mesh, spec):
  assert isinstance(mesh, mesh_lib.AbstractMesh)
  spec = remove_size_one_mesh_axis_from_spec(spec, mesh)
  axis_sizes, axis_names, axis_types = unzip3(
      [(s, n, t) for s, n, t in zip(mesh.axis_sizes, mesh.axis_names, mesh.axis_types)
      if s != 1])
  mesh = mesh.update(axis_sizes, axis_names, axis_types)
  return mesh, spec

@cache(max_size=4096, trace_context_in_key=False)
def named_sharding_to_xla_hlo_sharding(
    abs_mesh, spec, logical_device_ids, num_dimensions: int) -> xc.HloSharding:
  abs_mesh, spec = get_non_one_sized_mesh_spec(abs_mesh, spec)
  mesh_shape = abs_mesh.shape
  array_mapping = get_array_mapping(spec)
  mesh_axis_pos = {name: i for i, name in enumerate(abs_mesh.axis_names)}

  special_axes = {}
  if (manual_axes := frozenset(abs_mesh.manual_axes)):
    axis_names = abs_mesh.axis_names
    for manual_axis in manual_axes:
      special_axes[axis_names.index(manual_axis)] = xc.OpSharding.Type.MANUAL

  if (unreduced_axes := spec.unreduced):
    axis_names = abs_mesh.axis_names
    for u in unreduced_axes:
      special_axes[axis_names.index(u)] = xc.OpSharding.Type.UNREDUCED

  replicated_mesh_axes = []
  for i, (axis_name, axis_val) in enumerate(mesh_shape.items()):
    if axis_name not in array_mapping:
      replicated_mesh_axes.append((i, axis_val))

  if len(replicated_mesh_axes) == len(mesh_shape) and not special_axes:
    return xc.HloSharding.replicate()

  mesh_permutation = []
  new_mesh_shape = [1] * num_dimensions
  for name, pos in sorted(array_mapping.items(), key=lambda x: x[1]):
    new_mesh_shape[pos] *= mesh_shape[name]
    mesh_permutation.append(mesh_axis_pos[name])

  last_tile_dims = []
  if replicated_mesh_axes:
    axes_by_type: dict[Any, list[int]] = collections.defaultdict(list)
    size_by_type = collections.defaultdict(lambda: 1)
    assert {x[0] for x in replicated_mesh_axes}.issuperset(set(special_axes.keys()))
    for i, size in replicated_mesh_axes:
      ty = special_axes.get(i, xc.OpSharding.Type.REPLICATED)
      axes_by_type[ty].append(i)
      size_by_type[ty] *= size
    for ty, axes in sorted(axes_by_type.items(), key=lambda x: x[0].value):
      last_tile_dims.append(ty)
      new_mesh_shape.append(size_by_type[ty])
      mesh_permutation.extend(axes)

  # `HloSharding.iota_tile` 各参数的含义说明。
  # 这是 HloShardingV2 格式：
  #   * dims：每个维度被分片成多少份。
  #       复制/手动维度会被追加到末尾
  #   * reshape_dims：就是 mesh 的形状。
  #   * transpose_perm：PartitionSpec 中的 mesh 轴相对于 mesh.axis_names
  #       顺序的出现次序。
  #   * subgroup_types：OpSharding 类型的列表。类型可以是 REPLICATED 和 MANUAL。
  # 来看一个例子：
  #   考虑 input_shape=(8, 4, 2, 2)，mesh={'a': 2, 'b': 2, 'c': 2, 'd': 2}
  #   以及 partition_spec=P(None, ('d', 'b'), 'c')。
  #   传给 iota_tile 的参数将是：
  #     dims = [1, 4, 2, 1, 2]  # 'a' 是复制的，因此 `2` 位于末尾。
  #     reshape_dims = [2, 2, 2, 2]
  #     transpose_perm = [3, 1, 2, 0]  # 'a' 是复制的，因此 0 位于末尾
  #     subgroup_types = [xc.OpSharding.Type.REPLICATED]
  dims = new_mesh_shape
  reshape_dims = abs_mesh.axis_sizes
  if logical_device_ids is None:
    hlo_s = xc.HloSharding.iota_tile(
        dims=dims, reshape_dims=reshape_dims, transpose_perm=mesh_permutation,
        subgroup_types=last_tile_dims)
  else:
    hlo_s = xc.HloSharding.subgroup_with_device_ordering(
        np.asarray(logical_device_ids)
        .reshape(dims).reshape(reshape_dims).transpose(mesh_permutation)
        .reshape(dims), subgroup_types=last_tile_dims)

  if jaxlib_extension_version >= 478 and ifrt_version >= 61:
    reduction_op = uk_map.get(spec.unreduced_kind, None)
    if reduction_op is not None:
      hlo_s.set_reduction_op(reduction_op.value)  # type: ignore[attr-defined]
  return hlo_s

uk_map = {UnreducedKind.sum: sdy.ReductionOp.SUM,
          UnreducedKind.max: sdy.ReductionOp.MAX,
          UnreducedKind.min: sdy.ReductionOp.MIN}

@cache(max_size=4096, trace_context_in_key=False)
def named_sharding_to_sdy_sharding(self, num_dimensions: int,
                                   modify_wrt_axis_types: bool) -> SdyArray:
  dim_shardings = [SdyDim(axes=(), is_open=False)] * num_dimensions
  for i, dim_spec in enumerate(self.spec.partitions):
    if dim_spec is PartitionSpec.UNCONSTRAINED:
      dim_shardings[i] = SdyDim(axes=(), is_open=True)
    elif dim_spec is None:
      # 已经是空的封闭分片。
      pass
    else:
      dim_spec = dim_spec if isinstance(dim_spec, tuple) else (dim_spec,)
      dim_shardings[i] = SdyDim(axes=dim_spec, is_open=False)

  explicit_replicated_axes = frozenset()
  if modify_wrt_axis_types and self.mesh._any_axis_auto:
    dim_shardings = [d.replace(is_open=True) for d in dim_shardings]
    explicit_replicated_axes = frozenset(
        r for r in self.replicated_axes
        if self.mesh._name_to_type[r] == mesh_lib.AxisType.Explicit)

  reduction_op = uk_map.get(self.spec.unreduced_kind, None)
  return SdyArray(mesh_shape=self.mesh.shape_tuple,
                  dim_shardings=tuple(dim_shardings),
                  logical_device_ids=self._logical_device_ids,
                  replicated_axes=explicit_replicated_axes,
                  unreduced_axes=self.spec.unreduced,
                  reduction_op=reduction_op)


def array_mapping_to_axis_resources(array_mapping: ArrayMapping):
  if not array_mapping:
    return PartitionSpec()
  max_index = -1
  reverse_map = collections.defaultdict(list)
  for axis, index in array_mapping.items():
    reverse_map[index].append(axis)
    if index > max_index:
      max_index = index
  partitions: list[MeshAxisName | None] = []
  for i in range(max_index + 1):
    axis = reverse_map[i]
    if axis:
      partitions.append(axis[0] if len(axis) == 1 else tuple(axis))
    else:
      partitions.append(None)
  return PartitionSpec(*partitions)


@cache(max_size=128, trace_context_in_key=False)
def check_pspec(mesh, spec, _manual_axes=frozenset()):
  _check_unique_resources(spec, "NamedSharding spec", mesh)
  _check_mesh_resource_axis(mesh, spec)
  _check_mesh_unreduced(mesh, spec)

class DuplicateSpecError(Exception):
  def __init__(self, message, mesh, pspec):
    super().__init__(message)
    self.message = message
    self.mesh = mesh
    self.pspec = pspec

  def __str__(self):
    return f"{self.message}"

def _check_unique_resources(pspec: PartitionSpec, arg_name: str, mesh=None
                            ) -> None:
  resource_counts: dict[MeshAxisName, int] = {}
  duplicate = False
  for d in pspec.partitions:
    if d is PartitionSpec.UNCONSTRAINED or d is None:
      continue
    d = d if isinstance(d, tuple) else (d,)
    for resource in d:
      count = resource_counts.get(resource, 0)
      if count > 0:
        duplicate = True
      resource_counts[resource] = count + 1
  if duplicate:
    multiple_uses = [r for r, c in resource_counts.items() if c > 1]
    raise DuplicateSpecError(
        message=(
            f'A single {arg_name} specification can map every mesh axis to at'
            f' most one positional dimension, but {pspec} has duplicate entries'
            f' for {mesh_lib.show_axes(multiple_uses)}'),
        mesh=mesh, pspec=pspec)

def _check_mesh_resource_axis(mesh, pspec):
  for p in pspec.partitions:
    if p is PartitionSpec.UNCONSTRAINED or p is None:
      continue
    p = p if isinstance(p, tuple) else (p,)
    for r in p:
      if r not in mesh.axis_names:
        raise ValueError(
            f"Resource axis: {r} of {pspec} "
            f"is not found in mesh: {tuple(mesh.shape.keys())}.")
  if (AxisType.Auto not in mesh.axis_types and
      PartitionSpec.UNCONSTRAINED in pspec.partitions):
    raise ValueError(
        f'{pspec} cannot contain'
        ' `P.UNCONSTRAINED` when no mesh axis_types are `Auto`. Got mesh'
        f' axis_types: {mesh.axis_types}')

def _check_mesh_unreduced(mesh, pspec):
  for u in pspec.unreduced:
    if u not in mesh.axis_names:
      raise ValueError(
          f'Unreduced axes {u} is not found in {mesh.axis_names=}. '
          f'Got {pspec=}')
    if mesh._name_to_type[u] in (AxisType.Auto, AxisType.Manual):
      raise ValueError(
          'Unreduced axes can only refer to mesh axes that is of type'
          f' `Explicit`. Got unreduced axes: {pspec.unreduced} and'
          f' mesh: {mesh}')

  for u in pspec.reduced:
    if u not in mesh.axis_names:
      raise ValueError(
          f'Reduced axes {u} is not found in {mesh.axis_names=}. '
          f'Got {pspec=}')
    if mesh._name_to_type[u] in (AxisType.Auto, AxisType.Manual):
      raise ValueError(
          'Reduced axes can only refer to mesh axes that is of type'
          f' `Explicit`. Got reduced axes: {pspec.reduced} and'
          f' mesh: {mesh}')
