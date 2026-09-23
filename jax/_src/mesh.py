# Copyright 2018 The JAX Authors.
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
# 文件职责：定义 `Mesh` 与 `AbstractMesh`，即 JAX 中逻辑网格的抽象。
# `Mesh` 用一张多维设备数组加轴名描述可用硬件资源，是 `NamedSharding`、
# `shard_map`、`jax.jit` 等接口进行分片与并行计算的基础；模块同时提供
# `AxisType`（Auto/Explicit/Manual）区分各轴的自动、显式与手动分片类型，
# 以及 `AbstractMesh`（只保留轴名与轴大小、不含具体设备）来避免设备变化
# 引发的追踪与降级缓存失效。作用域内的网格通过线程局部的资源栈维护。
"""Mesh 与 AbstractMesh 的定义"""

from __future__ import annotations

import collections
from collections.abc import Hashable, Sequence
import contextlib
import dataclasses
import enum
import functools
import math
import threading
from typing import Any, NamedTuple
import warnings

import numpy as np

from jax._src import config as jax_config
from jax._src import xla_bridge as xb
from jax._src.util import (safe_zip, cache, tuple_delete, weak_value_interner,
                           immutable)
from jax._src.lib import _jax
from jax._src.lib import xla_client as xc

zip, unsafe_zip = safe_zip, zip
config_ext = _jax.config

MeshAxisName = Any
ResourceAxisName = Hashable


def show_axes(axes):
  return ", ".join(sorted(f"`{a}`" for a in axes))


class ResourceEnv(NamedTuple):
  physical_mesh: Mesh

  def with_mesh(self, mesh: Mesh):
    overlap = set(mesh.axis_names) & (self.resource_axes - set(self.physical_mesh.axis_names))
    if overlap:
      raise ValueError(f"Cannot update the mesh of the current resource "
                       f"environment. The new mesh shadows already defined axes "
                       f"{show_axes(overlap)}")
    return self._replace(physical_mesh=mesh)

  @property
  def physical_resource_axes(self) -> set[ResourceAxisName]:
    return set(self.physical_mesh.axis_names)

  @property
  def resource_axes(self) -> set[ResourceAxisName]:
    return self.physical_resource_axes

  @property
  def shape(self):
    return self.physical_mesh.shape

  @property
  def local_shape(self):
    return self.physical_mesh.local_mesh.shape

  def __repr__(self):
    mesh_repr = ", ".join(
        f"'{k}': {v}" for k, v in self.physical_mesh.shape.items())
    return f"ResourceEnv(mesh=Mesh({mesh_repr}))"


@cache(max_size=128, trace_context_in_key=False)
def _get_local_mesh(global_mesh: Mesh, process_index: int) -> Mesh:
  if global_mesh.empty:
    return global_mesh
  is_local_device = np.vectorize(
      lambda d: d.process_index == process_index, otypes=[bool])(global_mesh.devices)
  subcube_indices = []
  # 我们取每个维度上不会跳过任何本地设备的最小切片。
  for axis in range(global_mesh.devices.ndim):
    other_axes = tuple_delete(tuple(range(global_mesh.devices.ndim)), axis)
    # 注意：这里会多次在多个轴上重复归约，所以肯定还有优化空间，
    #       但我希望短期内它不会成为瓶颈。
    local_slices = is_local_device.any(other_axes, keepdims=False)
    nonzero_indices = np.flatnonzero(local_slices)
    start, end = int(np.min(nonzero_indices)), int(np.max(nonzero_indices))
    subcube_indices.append(slice(start, end + 1))
  subcube_indices_tuple = tuple(subcube_indices)
  # 只有当本地设备构成完整设备数组的一个子立方体时，最终才会所有条件都为真。
  # 因为我们在取切片时倾向于取由这些设备张成的"外壳（hull）"，一旦本地设备
  # 不构成子立方体，该外壳就会包含非本地设备。
  if not is_local_device[subcube_indices_tuple].all():
    raise ValueError(
        "When passing host local inputs to pjit, devices connected to a single"
        " host must form a contiguous subcube of the global device mesh"
    )
  return Mesh(global_mesh.devices[subcube_indices_tuple], global_mesh.axis_names)


class AxisType(enum.Enum):
  Auto = enum.auto()
  Explicit = enum.auto()
  Manual = enum.auto()

  def __repr__(self):
    return self.name

def _normalize_axis_types(axis_names, axis_types, name, default_axis_type):
  axis_types = ((default_axis_type,) * len(axis_names)
                if axis_types is None else axis_types)
  if not isinstance(axis_types, tuple):
    axis_types = (axis_types,)

  if not all(isinstance(a, AxisType) for a in axis_types):
    raise TypeError(
        f"axis_types passed to {name} must be of type `jax.sharding.AxisType`."
        f" Got {axis_types} of type {tuple(type(a) for a in axis_types)}")
  if len(axis_names) != len(axis_types):
    raise ValueError(
        "Number of axis names should match the number of axis_types. Got"
        f" axis_names={axis_names} and axis_types={axis_types}")
  return axis_types

def all_axis_types_match(axis_types, ty: AxisType) -> bool:
  if not axis_types:
    return False
  return all(t == ty for t in axis_types)

def any_axis_types_match(axis_types, ty: AxisType) -> bool:
  if not axis_types:
    return False
  return any(t == ty for t in axis_types)


class BaseMesh:
  axis_names: tuple[MeshAxisName, ...]
  shape_tuple: tuple[tuple[str, int], ...]
  axis_types: tuple[AxisType, ...]

  @functools.cached_property
  def are_all_axes_manual(self) -> bool:
    return all_axis_types_match(self.axis_types, AxisType.Manual)

  @functools.cached_property
  def are_all_axes_auto(self) -> bool:
    return all_axis_types_match(self.axis_types, AxisType.Auto)

  @functools.cached_property
  def are_all_axes_explicit(self) -> bool:
    return all_axis_types_match(self.axis_types, AxisType.Explicit)

  @functools.cached_property
  def _are_all_axes_auto_or_manual(self) -> bool:
    if not self.axis_types:
      return False
    return all(t == AxisType.Auto or t == AxisType.Manual
               for t in self.axis_types)

  @functools.cached_property
  def _are_all_axes_explicit_or_manual(self) -> bool:
    if not self.axis_types:
      return False
    return all(t == AxisType.Explicit or t == AxisType.Manual
               for t in self.axis_types)

  @functools.cached_property
  def _any_axis_manual(self) -> bool:
    return any_axis_types_match(self.axis_types, AxisType.Manual)

  @functools.cached_property
  def _any_axis_auto(self) -> bool:
    return any_axis_types_match(self.axis_types, AxisType.Auto)

  @functools.cached_property
  def _any_axis_explicit(self) -> bool:
    return any_axis_types_match(self.axis_types, AxisType.Explicit)

  @functools.cached_property
  def _any_axis_auto_or_manual(self) -> bool:
    if not self.axis_types:
      return False
    return any(t == AxisType.Auto or t == AxisType.Manual
               for t in self.axis_types)

  @functools.cached_property
  def auto_axes(self):
    return tuple(n for n, t in safe_zip(self.axis_names, self.axis_types)
                 if t == AxisType.Auto)

  @functools.cached_property
  def explicit_axes(self):
    return tuple(n for n, t in safe_zip(self.axis_names, self.axis_types)
                 if t == AxisType.Explicit)

  @functools.cached_property
  def manual_axes(self):
    return tuple(n for n, t in safe_zip(self.axis_names, self.axis_types)
                 if t == AxisType.Manual)

  @functools.cached_property
  def _name_to_type(self):
    return dict(safe_zip(self.axis_names, self.axis_types))


@immutable
class Mesh(BaseMesh, contextlib.ContextDecorator):
  """声明在该管理器作用域内可用的硬件资源。

  参见 `Distributed arrays and automatic parallelization`_ 与
  `Explicit Sharding`_ 教程。

  Args:
    devices: 一个 NumPy ndarray 对象，其中包含 JAX 设备对象（例如通过
      :py:func:`jax.devices` 获得）。
    axis_names: 要分配给 ``devices`` 参数各维度的资源轴名称序列。
      其长度应与 ``devices`` 的秩（rank）一致。
    axis_types: 与 ``axis_names`` 对应的 :class:`jax.sharding.AxisType` 条目组成的
      可选元组。更多信息参见 `Explicit Sharding`_。

  Examples:

    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P, NamedSharding
    >>> import numpy as np
    ...
    >>> # 声明一个带有轴 `x` 和 `y` 的二维网格。
    >>> devices = np.array(jax.devices()).reshape(4, 2)
    >>> mesh = Mesh(devices, ('x', 'y'))
    >>> inp = np.arange(16).reshape(8, 2)
    >>> arr = jax.device_put(inp, NamedSharding(mesh, P('x', 'y')))
    >>> out = jax.jit(lambda x: x * 2)(arr)
    >>> assert out.sharding == NamedSharding(mesh, P('x', 'y'))

  .. _Distributed arrays and automatic parallelization: https://docs.jax.dev/en/latest/parallel.html
  .. _Explicit Sharding:  https://docs.jax.dev/en/latest/parallel.html
  """

  devices: np.ndarray
  axis_names: tuple[MeshAxisName, ...]
  size: int

  @staticmethod
  @weak_value_interner
  def _create(flat_devices_tuple, device_shape, axis_names, axis_types, size):
    devices = np.array(flat_devices_tuple).reshape(device_shape)
    devices.flags.writeable = False
    obj = object.__new__(Mesh)
    object.__setattr__(obj, 'devices', devices)
    object.__setattr__(obj, 'axis_names', axis_names)
    object.__setattr__(obj, 'axis_types', axis_types)
    object.__setattr__(obj, 'size', size)
    return obj

  def __new__(cls, devices: np.ndarray | Sequence[xc.Device],
              axis_names: str | Sequence[MeshAxisName],
              axis_types: tuple[AxisType, ...] | None = None):
    if not isinstance(devices, np.ndarray):
      devices = np.array(devices)
    if isinstance(axis_names, str):
      axis_names = (axis_names,)
    axis_names = tuple(axis_names)
    if any(i is None for i in axis_names):
      raise ValueError(f"Mesh axis names cannot be None. Got: {axis_names}")
    if devices.ndim != len(axis_names):
      raise ValueError(
          "Mesh requires the ndim of its first argument (`devices`) to equal "
          "the length of its second argument (`axis_names`), but got "
          f"devices.ndim == {devices.ndim} and "
          f"len(axis_names) == {len(axis_names)}.")

    devices_flat = tuple(devices.flat)
    axis_types = _normalize_axis_types(axis_names, axis_types, 'Mesh',
                                       AxisType.Auto)
    empty = not axis_names and devices_flat[0] is None
    size = 0 if empty else math.prod(devices.shape)
    return cls._create(devices_flat, devices.shape, axis_names,
                       axis_types, size)

  # 没有 __eq__ 或 __hash__：被驻留（intern）的类使用对象身份比较。

  @property
  def is_scalar(self):
    return self.size == 1 and not self.axis_names

  def __getnewargs_ex__(self):
    return (self.devices, self.axis_names, self.axis_types), {}

  def __enter__(self):
    if jax_config.disallow_mesh_context_manager.value:
      raise RuntimeError("Mesh context manager is disabled.")
    warnings.warn(
        "`with mesh:` context manager has been deprecated. Please use `with"
        " jax.set_mesh(mesh):` instead.",
        category=DeprecationWarning, stacklevel=2)
    new_env = thread_resources.stack[-1].with_mesh(self)
    thread_resources.stack.append(new_env)
    thread_resources.env = new_env
    jax_config.mesh_context_manager.set_local(
        tuple(t.physical_mesh for t in thread_resources.stack
              if not t.physical_mesh.empty))
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    thread_resources.stack.pop()
    thread_resources.env = thread_resources.stack[-1]
    jax_config.mesh_context_manager.set_local(
        tuple(t.physical_mesh for t in thread_resources.stack
              if not t.physical_mesh.empty))
    return False

  def update(self, devices=None, axis_names=None, axis_types=None):
    if devices is None:
      devices = self.devices
    if axis_names is None:
      axis_names = self.axis_names
    if axis_types is None:
      axis_types = self.axis_types
    return Mesh(devices, axis_names, axis_types)

  @functools.cached_property
  def shape(self):
    return collections.OrderedDict(
        (name, size)
        for name, size in safe_zip(self.axis_names, self.devices.shape))

  @functools.cached_property
  def shape_tuple(self):  # pyrefly: ignore[bad-override]
    return tuple(
        (name, size)
        for name, size in safe_zip(self.axis_names, self.devices.shape))

  @property
  def axis_sizes(self) -> tuple[int, ...]:
    return self.devices.shape

  @property
  def empty(self):
    return self.size == 0

  @functools.cached_property
  def is_multi_process(self):
    return self.devices.size != len(self.local_devices)

  @property
  def local_mesh(self):
    return self._local_mesh(xb.process_index())

  def _local_mesh(self, process_index):
    return _get_local_mesh(self, process_index)

  @functools.cached_property
  def device_ids(self):
    assert not self.empty
    return np.vectorize(lambda d: d.id, otypes=[int])(self.devices)

  @functools.cached_property
  def _local_devices_set(self):
    return set(self.local_devices)

  @functools.cached_property
  def _flat_devices_tuple(self):
    return tuple(self.devices.flat)

  @functools.cached_property
  def _internal_device_list(self):
    return xc.DeviceList(self._flat_devices_tuple)

  @functools.cached_property
  def _flat_devices_set(self):
    return set(self.devices.flat)

  def __str__(self):
    if self.empty:
      return "Mesh()"
    mesh_str = ", ".join(f"'{k}': {v}" for k, v in self.shape.items())
    atr = f", axis_types={self.axis_types}"
    return f"Mesh({mesh_str}{atr})"

  @functools.cached_property
  def _repr(self):
    if self.empty:
      return "Mesh(axis_sizes=(), axis_names=())"
    atr = f", axis_types={self.axis_types}"
    return (f"Mesh(axis_sizes={self.device_ids.shape}, "
            f"axis_names={self.axis_names!r}{atr})")

  def __repr__(self):
    return self._repr

  @functools.cached_property
  def local_devices(self):
    return [d for d in self.devices.flat
            if d.process_index == d.client.process_index()]

  @functools.cached_property
  def abstract_mesh(self):
    if len(self.axis_names) == 0:
      return empty_abstract_mesh
    return AbstractMesh(
        self.axis_sizes, self.axis_names, axis_types=self.axis_types,
        abstract_device=abstract_device_from(self.devices.flat[0]))


EMPTY_ENV = ResourceEnv(Mesh(np.empty((), dtype=object), ()))

class _ThreadResourcesLocalState(threading.local):

  def __init__(self):
    self.stack = [EMPTY_ENV]
    self.env = self.stack[-1]

thread_resources = _ThreadResourcesLocalState()


@dataclasses.dataclass(frozen=True, slots=True)
class AbstractDevice:
  device_kind: str
  num_cores: int | None
  platform: str

  def __repr__(self):
    return (f"AbstractDevice({self._repr()})")

  def _repr(self):
    return (f"device_kind={self.device_kind}, num_cores={self.num_cores}, "
            f"platform={self.platform}")


def abstract_device_from(d) -> AbstractDevice | None:
  if d is None:
    return None
  if d.platform == 'tpu':
    num_cores = getattr(d, 'num_cores', None)
  elif d.platform == 'gpu':
    num_cores = getattr(d, 'core_count', None)
  else:
    num_cores = None
  return AbstractDevice(device_kind=d.device_kind, num_cores=num_cores,
                        platform=d.platform)


@immutable
class AbstractMesh(BaseMesh):
  """AbstractMesh 只包含轴名与轴大小。

  与 `jax.sharding.Mesh` 相比，它不包含具体设备。当网格形状与轴名保持不变、
  但设备发生变化时，应把它用作 with_sharding_constraint 的 sharding 入参以及
  shard_map 的 mesh 入参，以避免追踪与降级（lowering）的缓存未命中。
  更多细节参见 https://github.com/jax-ml/jax/pull/23022 的描述。

  Args:
    axis_sizes: 一个整数元组，指定每个资源轴的大小。
    axis_names: 要分配给 ``devices`` 参数各维度的资源轴名称元组。
      其长度应与 ``devices`` 的秩（rank）一致。
    axis_types: 与 ``axis_names`` 对应的 :class:`jax.sharding.AxisType` 条目组成的
      可选元组。更多信息参见 `Explicit Sharding`_。

  .. _Explicit Sharding:  https://docs.jax.dev/en/latest/parallel.html
  """
  axis_sizes: Any
  abstract_device: Any
  size: Any

  @staticmethod
  @weak_value_interner
  def _create(axis_sizes, axis_names, axis_types, abstract_device):
    obj = object.__new__(AbstractMesh)
    object.__setattr__(obj, 'axis_sizes', axis_sizes)
    object.__setattr__(obj, 'axis_names', axis_names)
    object.__setattr__(obj, 'axis_types', axis_types)
    object.__setattr__(obj, 'abstract_device', abstract_device)
    object.__setattr__(obj, 'size', math.prod(axis_sizes) if axis_sizes else 0)
    return obj

  def __new__(cls, axis_sizes: tuple[int, ...], axis_names: tuple[str, ...],
               axis_types: AxisType | tuple[AxisType, ...] | None = None,
               *, abstract_device=None):
    axis_types = _normalize_axis_types(axis_names, axis_types, 'AbstractMesh',
                                       AxisType.Explicit)
    return AbstractMesh._create(axis_sizes, axis_names, axis_types, abstract_device)

  # 没有 __eq__ 或 __hash__：被驻留（intern）的类使用对象身份比较。

  def __getnewargs_ex__(self):
    return ((self.axis_sizes, self.axis_names, self.axis_types),
            {'abstract_device': self.abstract_device})

  def __repr__(self):
    mesh_repr = (", ".join(f"'{n}': {v}" for n, v in self.shape_tuple)
                 if self.shape_tuple else "()")
    atr = f", axis_types={self.axis_types}"
    ad = ("" if self.abstract_device is None else
          f", {self.abstract_device._repr()}")
    return f"AbstractMesh({mesh_repr}{atr}{ad})"

  def update(self, axis_sizes=None, axis_names=None, axis_types=None, **kwargs):
    if axis_sizes is None:
      axis_sizes = self.axis_sizes
    if axis_names is None:
      axis_names = self.axis_names
    if axis_types is None:
      axis_types = self.axis_types
    if 'abstract_device' not in kwargs:
      kwargs['abstract_device'] = self.abstract_device
    return AbstractMesh(axis_sizes, axis_names, axis_types, **kwargs)

  @functools.cached_property
  def shape(self):
    return collections.OrderedDict(self.shape_tuple)

  @functools.cached_property
  def shape_tuple(self):  # pyrefly: ignore[bad-override]
    return tuple(
        (name, size)
        for name, size in safe_zip(self.axis_names, self.axis_sizes))

  @property
  def _internal_device_list(self):
    return None

  @property
  def empty(self):
    return self.size == 0

  @property
  def abstract_mesh(self):
    return self

  def update_axis_types(self, name_to_type: dict[MeshAxisName, AxisType]):
    new_axis_types = tuple(name_to_type[n] if n in name_to_type else a
                           for n, a in zip(self.axis_names, self.axis_types))
    return self.update(axis_types=new_axis_types)

  @property
  def devices(self):
    _raise_value_error("devices")

  @property
  def device_ids(self):
    _raise_value_error("device_ids")

  @property
  def is_multi_process(self):
    _raise_value_error("is_multi_process")

  @property
  def local_devices(self):
    _raise_value_error("local_devices")

  @property
  def local_mesh(self):
    _raise_value_error("local_mesh")

  def __enter__(self):
    _raise_value_error("__enter__")

  def __exit__(self, exc_type, exc_value, traceback):
    _raise_value_error("__exit__")


# 之所以加这层间接调用，是因为如果某个 property 无条件抛异常，pytype 就无法把
# 它识别为 property。等这个问题修好后即可移除。
def _raise_value_error(name):
  raise ValueError(f"AbstractMesh does not implement {name}")

empty_abstract_mesh = AbstractMesh((), ())
empty_concrete_mesh = Mesh(np.empty((), dtype=object), ())

class use_abstract_mesh:
  """在线程局部上下文中设置一个抽象网格。

  ``jax.sharding.use_abstract_mesh`` 可以作为上下文管理器使用。

  例如::

    abstract_device = jax.sharding.AbstractDevice(
        device_kind='TPU v6 lite', num_cores=1, platform='tpu')
    abstract_mesh = jax.sharding.AbstractMesh((2,), ('x',), (AxisType.Explicit,),
                                               abstract_device=abstract_device)

    @jax.jit
    def f(x):
      return x * 2

    with jax.sharding.use_abstract_mesh(abstract_mesh):
      # 注意：`f` 会针对 TPU 平台被追踪并降级。
      f.trace(inp).lower()
      # 注意：`f` 会针对 TPU 被追踪，但降级到 CPU。
      f.trace(inp).lower(lowering_platforms=('cpu',))

  Note: 在上面的例子中，只有在所有网格轴都是 Explicit 时，在顶层设置抽象网格
        才会生效。这属于临时限制，直到我们修复底层问题为止。
  """
  __slots__ = ['mesh', 'prev']

  def __init__(self, mesh: AbstractMesh):
    if not isinstance(mesh, AbstractMesh):
      raise ValueError(
          "Expected mesh of type `jax.sharding.AbstractMesh`. Got type:"
          f" {type(mesh)}")
    self.mesh = mesh

  def __enter__(self):
    self.prev = jax_config.abstract_mesh_context_manager.swap_local(self.mesh)
    if (self.prev is not config_ext.unset and
        not self.prev.empty and not self.mesh.empty and
        self.prev.size != self.mesh.size):
      jax_config.abstract_mesh_context_manager.set_local(self.prev)
      raise ValueError(
          "use_abstract_mesh cannot change the size of the mesh. Got new mesh:"
          f" {self.mesh} with size={self.mesh.size} and prev mesh:"
          f" {self.prev} with size={self.prev.size}")

  def __exit__(self, exc_type, exc_value, traceback):
    jax_config.abstract_mesh_context_manager.set_local(self.prev)


def get_abstract_mesh() -> AbstractMesh:
  val = jax_config.abstract_mesh_context_manager.value
  return empty_abstract_mesh if val is None else val

def get_concrete_mesh() -> Mesh:
  val = jax_config.device_context.value
  return empty_concrete_mesh if val is None else val
