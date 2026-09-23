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

# 文件职责：实现 `jax.Array` 的数组表示与分片基础设施。
# 核心类 `ArrayImpl`（经 `use_cpp_class` 映射到 C++ 的 `xc.ArrayImpl`）把全局
# 形状、`Sharding` 分片方案与每个可寻址设备的本地缓冲区统一封装为惰性、可能跨
# 进程的数组，并提供 `Shard` 视图、主机拷贝、删除检测、DLPack 等接口。
# 同文件还提供 `make_array_from_callback` 等构造入口，并向 pxla 注册分片参数
# 与全局结果处理器，是 `jax.device_put` 与分布式分片机制的关键实现。

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import enum
import functools
from functools import partial
import math
import operator as op
from typing import Any, cast

from jax._src import api
from jax._src import basearray
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import errors
from jax._src import literals
from jax._src import profiler
from jax._src import util
from jax._src import xla_bridge
from jax._src.op_shardings import are_hlo_shardings_equal
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.layout import AutoLayoutSingleton, Format, Layout
from jax._src.lib import _jax
from jax._src.lib import xla_client as xc
from jax._src.mesh import (empty_concrete_mesh, empty_abstract_mesh,
                           use_abstract_mesh)
from jax._src.sharding import Sharding
from jax._src.tree_util import broadcast_prefix, tree_flatten, tree_unflatten
from jax._src.sharding_impls import (
    make_single_device_sharding, NamedSharding,
    device_replica_id_map, hashed_index, num_addressable_indices,
    local_to_global_shape, _internal_use_concrete_mesh)  # pyformat: disable
from jax._src.typing import ArrayLike, DLDeviceType, DTypeLike, ExtendedDType
from jax._src.util import safe_zip, unzip3, use_cpp_class, use_cpp_method, cache
import numpy as np

zip, unsafe_zip = safe_zip, zip

Shape = tuple[int, ...]
Device = xc.Device
Index = tuple[slice, ...]
PRNGKeyArray = Any  # TODO(jakevdp): 修复循环依赖并改为导入它。

def _get_device(a: ArrayImpl) -> Device:
  devices = a.sharding._internal_device_list
  if len(devices) != 1:
    raise ValueError(
        "When making an array from single-device arrays the input arrays must "
        f"have one shard each. An argument array had {len(devices)} shard(s).")
  return devices[0]


class Shard:
  """`Array` 的单个数据分片。

  Attributes:
    device : 该分片所在的设备。
    index : 该分片在全局数组中的索引。
    replica_id : 整数 id，表示该分片属于全局数组的哪个副本。对于完全分片数据
      （即只有 1 个副本时）始终为 0。
    data : 该分片的数据。若 ``device`` 非本地，则为 None。
  """

  def __init__(self, device: Device, sharding: Sharding, global_shape: Shape,
               data: None | ArrayImpl | PRNGKeyArray = None):
    self._device = device
    self._sharding = sharding
    self._global_shape = global_shape
    self._data = data

  def __repr__(self):
    try:
      return (f'Shard(device={self.device!r}, index={self.index}, '
              f'replica_id={self.replica_id}, data={self.data})')
    except ValueError:
      return f'Shard(device={self.device!r}, data={self.data})'

  @functools.cached_property
  def index(self) -> Index:
    try:
      device_indices_map_fn = self._sharding.devices_indices_map
    except AttributeError:
      raise ValueError('Cannot calculate indices from sharding: '
                       f'{self._sharding}. Please create a device to index '
                       'mapping for your sharding.') from None
    index = device_indices_map_fn(self._global_shape)[self.device]
    assert index is not None
    return index

  @functools.cached_property
  def replica_id(self) -> int:
    return device_replica_id_map(self._sharding, self._global_shape)[self.device]

  @property
  def device(self):
    return self._device

  @property
  def data(self):
    return self._data


def _reconstruct_array(fun, args, arr_state, aval_state):
  """从序列化状态重建设备数组的方法。"""
  np_value = fun(*args)
  np_value.__setstate__(arr_state)
  jnp_value = api.device_put(np_value)
  jnp_value.aval = jnp_value.aval.update(**aval_state)
  return jnp_value


@cache(max_size=4096, trace_context_in_key=False)
def _cached_index_calc(s, shape):
  map_ = s.addressable_devices_indices_map(shape)
  seen_h_indices = set()
  l = []
  for array_index, index in enumerate(map_.values()):
    h_index = hashed_index(index)
    if h_index not in seen_h_indices:
      seen_h_indices.add(h_index)
      l.append((array_index, index))
  return l


@cache(max_size=4096, trace_context_in_key=False)
def _process_has_full_value_in_mcjax(s, shape):
  # 单主机时作为快速路径直接返回 False。
  if xla_bridge.process_count() == 1:
    return False

  num_unique_indices = len(
      {hashed_index(v) for v in s.devices_indices_map(shape).values()})
  num_addressable_unique_indices = len(
      {hashed_index(v) for v in s.addressable_devices_indices_map(shape).values()})
  return num_unique_indices == num_addressable_unique_indices


def _validate_shape_and_dtype_for_per_device_arrays(
    arrays: Sequence[ArrayImpl | np.ndarray | literals.TypedNdArray],
    sharding: Sharding,
    aval: core.ShapedArray,
    expected_shape: Shape,
):
  """校验每设备数组合法且相互一致。"""
  expected_dtype = aval.dtype
  for db in arrays:
    if db.dtype != expected_dtype:
      raise ValueError(
          "Input buffers to `Array` must have matching dtypes. "
          f"Got {db.dtype}, expected {expected_dtype} for buffer: {db}"
      )
    if db.shape != expected_shape:
      raise ValueError(
          f"Expected shard shape {expected_shape} doesn't match the single "
          f"device array shape {db.shape}. Shape of Array is "
          f"{aval.str_short()} with sharding {sharding}"
      )


@use_cpp_class(xc.ArrayImpl)
class ArrayImpl(basearray.Array):
  aval: core.ShapedArray
  _sharding: Sharding
  _arrays: list[ArrayImpl]
  _committed: bool
  _skip_checks: bool
  _npy_value: np.ndarray | None

  @use_cpp_method()
  def __init__(self, aval: core.ShapedArray, sharding: Sharding,
               arrays: Sequence[ArrayImpl],
               committed: bool, _skip_checks: bool = False):
    # NOTE: 构造函数的具体实现已移到 C++ 中。

    self.aval = aval
    self._sharding = sharding
    self._committed = committed
    self._npy_value = None
    arrays = [a._arrays[0] for a in arrays]

    # 若启用了 skip_checks 就不要重排，因为这里假定输入缓冲区
    # 已经排布正确。这通常发生在 `Array` 作为 JAX 变换
    # （如 pjit 等）的输出被创建时。
    if not _skip_checks or config.enable_checks.value:
      arrays = self._check_and_rearrange(arrays, self._sharding, self.aval)
    self._arrays = arrays

  def _check_and_rearrange(self, arrays, sharding, aval):
    device_id_to_buffer = {_get_device(db).id: db for db in arrays}

    addressable_dev = sharding.addressable_devices
    if len(arrays) != len(addressable_dev):
      raise ValueError(
          f"Expected {len(addressable_dev)} per-device arrays "
          "(this is how many devices are addressable by the sharding), but "
          f"got {len(arrays)}")

    array_device_ids = set(device_id_to_buffer.keys())
    addressable_device_ids = {d.id for d in addressable_dev}
    if len(array_device_ids) != len(arrays):
      buffer_device_ids = [_get_device(db).id for db in arrays]
      raise ValueError(
          "When making an array from single-device arrays, the input arrays"
          " must be from distinct devices, but got device IDs"
          f" {buffer_device_ids}")

    # 计算对称差，因为 sharding 与 _arrays 的
    # 设备 id 本应一致。
    diff = array_device_ids ^ addressable_device_ids
    if diff:
      dev_in_sharding_not_in_arrays = addressable_device_ids - array_device_ids
      dev_in_arrays_not_in_sharding = array_device_ids - addressable_device_ids
      err_msg = (
          "Addressable devices and per-device arrays devices do not match.")
      if dev_in_sharding_not_in_arrays:
        err_msg += (f" Sharding contains devices {dev_in_sharding_not_in_arrays} "
                    "that are not present in per-device arrays.")
      if dev_in_arrays_not_in_sharding:
        err_msg += (f" Per-device arrays contain devices {dev_in_arrays_not_in_sharding} "
                    "that are not present in the sharding.")
      raise ValueError(err_msg)

    _validate_shape_and_dtype_for_per_device_arrays(
        arrays,
        sharding=sharding,
        aval=aval,
        expected_shape=sharding.shard_shape(aval.shape),
    )

    # 根据设备分配重排数组。
    addressable_da = sharding._addressable_device_assignment
    return [device_id_to_buffer[device.id] for device in addressable_da]

  @property
  def shape(self) -> Shape:
    return self.aval.shape

  @property
  def dtype(self):
    return self.aval.dtype

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def size(self):
    return math.prod(self.shape)

  @property
  def sharding(self):
    return self._sharding

  @property
  def device(self):
    self._check_if_deleted()
    if len(self.sharding.device_set) == 1:
      return list(self.sharding.device_set)[0]
    return self.sharding

  @property
  def weak_type(self):
    return self.aval.weak_type

  @property
  def committed(self) -> bool:
    return self._committed

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError as err:
      raise TypeError("len() of unsized object") from err  # 与 numpy 的报错一致

  def __bool__(self):
    core.check_bool_conversion(self)
    return bool(self._value)

  def __float__(self):
    core.check_scalar_conversion(self)
    return self._value.__float__()

  def __int__(self):
    core.check_scalar_conversion(self)
    return self._value.__int__()

  def __complex__(self):
    core.check_scalar_conversion(self)
    return self._value.__complex__()

  def __hex__(self):
    core.check_integer_conversion(self)
    return hex(self._value)

  def __oct__(self):
    core.check_integer_conversion(self)
    return oct(self._value)

  def __index__(self):
    core.check_integer_conversion(self)
    return op.index(self._value)

  def tobytes(self, order="C"):
    return self._value.tobytes(order)

  def tolist(self):
    return self._value.tolist()

  def __format__(self, format_spec):
    if isinstance(self.sharding, NamedSharding) and self.sharding.spec.unreduced:
      return repr(self)
    elif (self.is_fully_addressable or self.is_fully_replicated and
          self.sharding.has_addressable_devices):
      # 模拟 https://github.com/numpy/numpy/pull/9883 的行为
      return format(self._value if self.ndim else self._value[()], format_spec)
    else:
      return repr(self)

  def __getitem__(self, idx, /):
    from jax._src.numpy import indexing  # pyrefly: ignore[missing-import]
    self._check_if_deleted()

    return indexing.rewriting_take(self, idx)

  def __iter__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")  # 与 numpy 的报错一致
    else:
      assert self.is_fully_replicated or self.is_fully_addressable
      if self.sharding.num_devices == 1 or self.is_fully_replicated:
        return (sl for chunk in self._chunk_iter(100) for sl in chunk._unstack())  # pyrefly: ignore[missing-attribute]
      else:
        # TODO(yashkatariya): 在支持非均匀分区后，不要绕到主机，
        # 而是直接使用此处的 `_chunk_iter` 路径。
        return (api.device_put(self._value[i]) for i in range(self.shape[0]))

  @property
  def is_fully_replicated(self) -> bool:
    return self.sharding.is_fully_replicated

  def __repr__(self):
    prefix = 'Array('
    if self.aval is not None and self.aval.weak_type:
      dtype_str = f'dtype={self.dtype.name}, weak_type=True'
    else:
      dtype_str = f'dtype={self.dtype.name}'

    if isinstance(self.sharding, NamedSharding) and self.sharding.spec.unreduced:
      return f"Array(shape={self.shape}, {dtype_str}, sharding={self.sharding})"
    elif self.is_fully_addressable or self.is_fully_replicated:
      line_width = np.get_printoptions()["linewidth"]
      if self.size == 0:
        s = f"[], shape={self.shape}"
      elif not self.sharding.has_addressable_devices or self.nbytes >= 1 << 20:
        s = f"shape={self.shape}"
      else:
        s = np.array2string(self._value, prefix=prefix, suffix=',',
                            separator=', ', max_line_width=line_width)
      last_line_len = len(s) - s.rfind('\n') + 1
      sep = ' '
      if last_line_len + len(dtype_str) + 2 > line_width:
        sep = ' ' * len(prefix)
      return f"{prefix}{s},{sep}{dtype_str})"
    else:
      return f"{prefix}shape={self.shape}, {dtype_str})"

  def __str__(self):
    if isinstance(self.sharding, NamedSharding) and self.sharding.spec.unreduced:
      return repr(self)
    elif (self.is_fully_addressable or self.is_fully_replicated and
          self.sharding.has_addressable_devices) and self.nbytes < 1 << 20:
      return str(self._value)  # 不会打印 Array(...)
    else:
      return repr(self)

  @property
  def is_fully_addressable(self) -> bool:
    """该 `Array` 是否完全可寻址？

    如果当前进程能够寻址 :class:`Sharding` 中命名的所有设备，那么该
    jax.Array 就是完全可寻址的。``is_fully_addressable`` 等价于多进程
    JAX 中的 "is_local"。

    注意，完全复制并不等于完全可寻址，也就是说，完全复制的 jax.Array
    可以跨多个主机，因而是不完全可寻址的。
    """
    return self.sharding.is_fully_addressable

  def __array__(self, dtype: np.dtype | None = None,
                context: None = None, copy: bool | None = None):
    del context  # 未使用
    # 从 numpy 2.0 起 np.asarray 支持 copy 参数
    kwds = {} if copy is None else {'copy': copy}
    return np.asarray(self._value, dtype=dtype, **kwds)  # pyrefly: ignore[no-matching-overload]

  def __dlpack__(self, *, stream: int | Any | None = None,
                 max_version: tuple[int, int] | None = None,
                 dl_device: tuple[DLDeviceType, int] | None = None,
                 copy: bool | None = None):
    from jax._src.dlpack import to_dlpack  # pyrefly: ignore[missing-import]

    device_set = self.sharding.device_set
    if len(device_set) > 1:
      raise BufferError(
        "to_dlpack can only pack a dlpack tensor from an array on a singular "
        f"device, but an array with a Sharding over {len(device_set)} devices "
        "was provided."
      )
    device, = device_set
    return to_dlpack(self, stream=stream,
                     max_version=max_version,
                     src_device=device,
                     dl_device=dl_device,
                     copy=copy)

  def __dlpack_device__(self) -> tuple[enum.Enum, int]:
    if len(self._arrays) != 1:
      raise BufferError("__dlpack__ only supported for unsharded arrays.")

    from jax._src.dlpack import DLDeviceType  # pyrefly: ignore[missing-import]

    if self.platform() == "cpu":
      return DLDeviceType.kDLCPU, 0

    elif self.platform() == "gpu":
      platform_version = _get_device(self).client.platform_version
      if "cuda" in platform_version:
        if self.sharding.memory_kind == "pinned_host":
          dl_device_type = DLDeviceType.kDLCUDAHost
        else:
          dl_device_type = DLDeviceType.kDLCUDA
      elif "rocm" in platform_version:
        if self.sharding.memory_kind == "pinned_host":
          dl_device_type = DLDeviceType.kDLROCMHost
        else:
          dl_device_type = DLDeviceType.kDLROCM
      elif "oneapi" in platform_version:
        dl_device_type = DLDeviceType.kDLOneAPI
      else:
        raise BufferError("Unknown GPU platform for __dlpack__: "
                         f"{platform_version}")

      local_hardware_id = _get_device(self).local_hardware_id
      if local_hardware_id is None:
        raise BufferError("Couldn't get local_hardware_id for __dlpack__")

      return dl_device_type, local_hardware_id

    elif self.platform() == "tpu":
      if self.sharding.memory_kind == "pinned_host":
        dl_device_type = DLDeviceType.kDLTPUHost
      else:
        raise BufferError(
            "__dlpack__ device only supported for TPU pinned host memory"
        )

      local_hardware_id = _get_device(self).local_hardware_id
      if local_hardware_id is None:
        raise BufferError("Couldn't get local_hardware_id for __dlpack__")

      return dl_device_type, local_hardware_id

    else:
      raise BufferError(
          "__dlpack__ device only supported for CPU, GPU and TPU pinned host,"
          f" got platform: {self.platform()}"
      )

  def __reduce__(self):
    fun, args, arr_state = self._value.__reduce__()
    aval_state = {'weak_type': self.aval.weak_type}
    return (_reconstruct_array, (fun, args, arr_state, aval_state))

  @use_cpp_method()
  def unsafe_buffer_pointer(self):
    if len(self._arrays) != 1:
      raise ValueError("unsafe_buffer_pointer() is supported only for unsharded"
                       " arrays.")
    return self._arrays[0].unsafe_buffer_pointer()

  @property
  @use_cpp_method()
  def __cuda_array_interface__(self):
    if len(self._arrays) != 1:
      raise ValueError("__cuda_array_interface__() is supported only for "
                       "unsharded arrays.")
    return self._arrays[0].__cuda_array_interface__  # bind-properties

  @use_cpp_method()
  def on_device_size_in_bytes(self):
    """返回该数组在设备上的全局总字节大小。"""
    arr = self._arrays[0]
    per_shard_size = arr.on_device_size_in_bytes()
    return per_shard_size * self.sharding.num_devices

  def devices(self) -> set[Device]:
    self._check_if_deleted()
    return self.sharding.device_set

  @property
  def device_buffer(self):
    raise AttributeError(
      "arr.device_buffer has been deprecated. Use arr.addressable_data(0)")

  @property
  def device_buffers(self):
    raise AttributeError(
      "arr.device_buffers has been deprecated. Use [x.data for x in arr.addressable_shards]")

  def addressable_data(self, index: int) -> ArrayImpl:
    self._check_if_deleted()
    if self.is_fully_replicated:
      return self._fully_replicated_shard()  # pyrefly: ignore[missing-attribute]
    return self._arrays[index]

  @property
  def addressable_shards(self) -> Sequence[Shard]:
    self._check_if_deleted()
    val = self.__dict__.get("addressable_shards", None)
    if val is not None:
      return val
    out = []
    for a in self._arrays:
      out.append(Shard(_get_device(a), self.sharding, self.shape, a))
    if len(out) != 1:
      # 当 len(out) == 1 时，out 只是 [Shard(self)]，会构成循环引用。
      self.__dict__["addressable_shards"] = out
    return out

  @property
  def format(self):
    # TODO(yashkatariya): 从这里移除“已删除”检查。
    if self.is_deleted():
      return Format(None, self.sharding)
    try:
      return Format(Layout.from_pjrt_layout(self._pjrt_layout),  # pyrefly: ignore[missing-attribute]
                    self.sharding)
    except _jax.JaxRuntimeError as e:
      msg, *_ = e.args
      if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
        return Format(None, self.sharding)
      else:
        raise

  @property
  def global_shards(self) -> Sequence[Shard]:
    """返回该 `Array` 跨所有设备的全部 `Shard` 列表。

    结果包含当前进程无法寻址的分片。如果某个 `Shard` 不可寻址，
    那么它的 `data` 将为 `None`。
    """
    self._check_if_deleted()
    if self.is_fully_addressable:
      return self.addressable_shards

    out = []
    device_id_to_buffer = {_get_device(a).id: a for a in self._arrays}
    for global_d in self.sharding.device_set:
      if device_id_to_buffer.get(global_d.id, None) is not None:
        array = device_id_to_buffer[global_d.id]
      else:
        array = None
      out.append(Shard(global_d, self.sharding, self.shape, array))
    return out

  @use_cpp_method()
  def delete(self):
    if self._arrays is None:
      return
    for buf in self._arrays:
      buf.delete()
    self._arrays = None  # pyrefly: ignore[bad-assignment]
    self._npy_value = None

  @use_cpp_method()
  def is_deleted(self):
    if self._arrays is None:
      return True
    # 当创建了 `Array` 的视图且原 Array 被删除时会走这条路径。
    # 此时该视图所表示的缓冲区也会一并被删除。
    return any(buf.is_deleted() for buf in self._arrays)

  def _check_if_deleted(self):
    if self.is_deleted():
      raise RuntimeError(
          f"Array has been deleted with shape={self.aval.str_short()}.")

  @use_cpp_method()
  def block_until_ready(self):
    self._check_if_deleted()
    for db in self._arrays:
      db.block_until_ready()
    return self

  @use_cpp_method()
  def _single_device_array_to_np_array_did_copy(self) -> tuple[np.ndarray, bool]:
    ...

  @use_cpp_method()
  def _copy_single_device_array_to_host_async(self):
    self._arrays[0].copy_to_host_async()

  @profiler.annotate_function
  def copy_to_host_async(self):
    self._check_if_deleted()
    if self._npy_value is None:
      if self.is_fully_replicated and self.sharding.has_addressable_devices:
        self._copy_single_device_array_to_host_async()
        return
      for i, _ in _cached_index_calc(self.sharding, self.shape):
        self._arrays[i]._copy_single_device_array_to_host_async()

  @property
  @functools.partial(profiler.annotate_function, name="np.asarray(jax.Array)")
  def _value(self) -> np.ndarray:
    self._check_if_deleted()

    if self._npy_value is None:
      # addressable_device_list 可能为空。若为空，我们会在下面报错
      if self.is_fully_replicated and self.sharding.has_addressable_devices:
        npy_value, did_copy = self._single_device_array_to_np_array_did_copy()
        npy_value.flags.writeable = False
        if did_copy:
          self._npy_value = npy_value
        return npy_value

      # TODO(yashkatariya): 把 `_process_has_full_value_in_mcjax` 与
      # is_fully_addressable 合并。
      # 当 addressable_device_list 为空时，is_fully_addressable 返回 False。
      if (not self.is_fully_addressable and
          not _process_has_full_value_in_mcjax(self.sharding, self.shape)):
        raise RuntimeError(
            "Fetching value for `jax.Array` that spans non-addressable"
            " (non process local) devices is not possible. You can use"
            " `jax.experimental.multihost_utils.process_allgather` to print the"
            " global array or use `.addressable_shards` method of jax.Array to"
            " inspect the addressable (process local) shards."
        )

      for i, _ in _cached_index_calc(self.sharding, self.shape):
        self._arrays[i]._copy_single_device_array_to_host_async()

      npy_value = np.empty(self.shape, self.dtype)
      for i, ind in _cached_index_calc(self.sharding, self.shape):
        npy_value[ind], _ = self._arrays[i]._single_device_array_to_np_array_did_copy()
      self._npy_value = npy_value
      self._npy_value.flags.writeable = False
    return self._npy_value


def _get_shape_from_index(slc: Index, shape: Shape) -> Shape:
  return tuple(
      (s.stop or dim) - (s.start or 0)
      for s, dim in safe_zip(slc, shape)
      if isinstance(s, slice)  # 若元素是 int，则该维度被规约
  )


def _get_and_check_dtype(
    arrays: Sequence[basearray.Array | np.ndarray | literals.TypedNdArray],
    dtype: DTypeLike | ExtendedDType | None,
    fname: str,
):
  if dtype is None:
    if arrays:
      dtype = arrays[0].dtype
    else:
      raise ValueError(
          "If the Array has no addressable shards, `dtype` must be provided "
          f"via the `dtype` argument to `jax.{fname}`.")
  else:
    dtype = dtypes.check_and_canonicalize_user_dtype(dtype, fname)
    if arrays and arrays[0].dtype != dtype:
      raise ValueError(
          f"If `dtype` is provided to `jax.{fname}`, it must match the dtype "
          f"of the addressable shards. Got dtype={dtype} and shard "
          f"dtype={arrays[0].dtype}`.")
  return dtype


# 显式设为不可哈希。
setattr(ArrayImpl, "__hash__", None)
setattr(ArrayImpl, "__array_priority__", 100)

# TODO(yashkatariya): 从回调输入类型中移除 None。

def make_array_from_callback(
    shape: Shape, sharding: Sharding | Format,
    data_callback: Callable[[Index | None], ArrayLike],
    dtype: DTypeLike | None = None) -> ArrayImpl:
  # pyformat: disable
  """通过 ``data_callback`` 获取的数据返回一个 ``jax.Array``。

  ``data_callback`` 用于获取返回的 ``jax.Array`` 每个可寻址分片的数据。该函数
  必须返回具体数组，这意味着 ``make_array_from_callback`` 与 :func:`jit` 或
  :func:`vmap` 等 JAX 变换的兼容性有限。

  Args:
    shape : ``jax.Array`` 的形状。
    sharding: 一个 ``Sharding`` 实例，用于描述该 ``jax.Array``
      如何在各个设备上布局。
    data_callback : 以全局数组值中的索引作为输入、返回全局数组值
      对应数据的回调。返回的数据可以是任意类数组对象，
      例如 ``numpy.ndarray``。
    dtype: 输出 ``jax.Array`` 的数据类型。若未提供，则使用第一个
      可寻址分片的数据类型。若不存在可寻址分片，
      则必须提供 ``dtype`` 参数。

  Returns:
    通过 ``data_callback`` 获取的数据构造出的
    ``jax.Array``。

  Examples:

    >>> import math
    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> import numpy as np
    ...
    >>> input_shape = (8, 8)
    >>> global_input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    >>> global_mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
    >>> inp_sharding = jax.sharding.NamedSharding(global_mesh, P('x', 'y'))
    ...
    >>> def cb(index):
    ...  return global_input_data[index]
    ...
    >>> arr = jax.make_array_from_callback(input_shape, inp_sharding, cb)
    >>> arr.addressable_data(0).shape
    (4, 2)
  """
  # pyformat: enable
  dll = sharding.layout if isinstance(sharding, Format) else None
  if isinstance(dll, AutoLayoutSingleton):
    raise TypeError(
        "`Layout.AUTO` cannot be used in place of a device-local"
        f" layout when calling `jax.make_array_from_callback`. Got {sharding}")
  processed_sharding = sharding.sharding if isinstance(sharding, Format) else sharding
  if not isinstance(processed_sharding, Sharding):
    raise TypeError(
        f"sharding should be an instance of `jax.sharding`. Got {processed_sharding} of"
        f" type {type(processed_sharding)}")
  sharding = processed_sharding

  def get_data(
      index: Index | None,
  ) -> ArrayImpl | literals.TypedNdArray | np.ndarray:
    # 也许可以在这里按索引做缓存，这样就能统一下面完全复制与
    # 非完全复制两种情况，并在部分复制的情形下更快。
    assert index is not None
    r = data_callback(index)
    if isinstance(r, core.Tracer):
      raise errors.UnexpectedTracerError(
          "jax.make_array_from_callback cannot be called within a traced"
          " context."
      )
    # 值可能是 Python 标量，把它解析成带数据类型的形式。
    r = dtypes.canonicalize_value(r)
    if isinstance(r, (literals.TypedInt, literals.TypedFloat,
                      literals.TypedComplex)):
      r = literals.TypedNdArray(np.asarray(r, dtype=r.dtype))
    elif isinstance(r, bool):
      r = literals.TypedNdArray(np.asarray(r, dtype=np.bool_))
    return r

  if sharding.is_fully_replicated:
    devices = list(sharding._internal_device_list.addressable_device_list)
    # 只计算一次数据。
    per_device_values = [get_data((slice(None),) * len(shape))] * len(devices)
  else:
    device_to_index_map = sharding.addressable_devices_indices_map(shape)
    devices = list(device_to_index_map.keys())
    per_device_values = [
        get_data(device_to_index_map[device]) for device in devices
    ]

  dtype = _get_and_check_dtype(
      per_device_values, dtype, "make_array_from_callback")
  expected_shape = sharding.shard_shape(shape)
  aval = core.update_aval_with_sharding(
      core.ShapedArray(shape, dtype), sharding)

  _validate_shape_and_dtype_for_per_device_arrays(
      per_device_values,
      expected_shape=expected_shape,
      aval=aval,
      sharding=sharding,
  )
  first_value = None
  if per_device_values:
    first_value = per_device_values[0]
    if (isinstance(first_value, ArrayImpl)
        and first_value._committed
        and sharding.is_fully_replicated
        and first_value.is_fully_replicated
        and first_value.sharding._device_assignment == tuple(devices)
        and first_value.format.layout == dll):
      return first_value

  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    # TODO(yashkatariya): 这里也能用 batched_device_put 吗？
    arrays = api.device_put(per_device_values, devices)
    return aval.dtype._rules.make_sharded_array(
        aval, sharding, arrays, committed=True
    )

  if dll is not None:
    devices = [Format(dll, make_single_device_sharding(d)) for d in devices]
    # pxla.batched_device_put 不支持 Layout……只能走慢路径
    arrays = api.device_put(per_device_values, devices)
    return ArrayImpl(aval, sharding, arrays, committed=True)

  if isinstance(first_value, ArrayImpl) and len(first_value.devices()) > 1:
    # 回调的输出已经是分片数组，把它移动到目标设备。
    per_device_values = api.device_put(per_device_values, devices)

  return pxla.batched_device_put(aval, sharding, per_device_values, devices)


def make_array_from_process_local_data(
    sharding,  # PyTree[jax.sharding.Sharding]
    local_data,  # PyTree[np.ndarray]
    global_shape=None):  # PyTree[Shape]
  # pyformat: disable
  """使用进程内可用的数据创建一个分布式张量。

  该函数是 `make_array_from_callback` 的一个常见特例。
  它假定数据已经存在于当前进程内，
  并自动处理索引的换算工作。

  最常见的情形是分片沿 batch 维度切分，每个主机只加载
  与之对应的那个子批次。该函数也支持更一般的情形，
  例如多主机与多轴混合的复制与分片，但此时你需要自行
  正确计算进程本地数据的大小与内容，
  才能满足分片方案所施加的约束条件。

  特别地，如果任意两个主机互为副本，
  那么这两个主机上的 host_local_data 也必须完全相同。

  global_shape 是可选的。若未提供，它会根据 local_data 与
  sharding 推断出来，前提是对于均匀分片，每个主机只表示
  自己的数据。若分片是非均匀的（见下方说明），
  则会抛出异常。

  显式设置 global_shape 可以获得更精细的控制，并且同样适用于
  非均匀分片。global_shape 的每一维要么与 host_local_data 匹配，
  要么与推断出的 sharding 全局形状匹配（这等价于将其设为
  None，但表达更明确）。

  例如，若维度 `i` 被完全分片，则该大小为
  `per_device_shape[i] * jax.local_device_count()`。每个设备都会映射到
  `local_data` 数组中对应的本地切片。例如，若给定进程处理
  切片 (8, 12) 与 (24, 28)，那么这些切片将被映射到
  `local_data` 的 (0, 4) 与 (4, 8)。

  对于 global_shape 与 local_shape 匹配的每一维，每个设备都会到
  local_data 中查找对应的切片。例如若
  global_shape == local_data.shape，则假定本地数据就是要分片到
  设备上的实际目标数组。

  如果 global_shape 与 local_data.shape 相同，
  那么所有主机上的数据也必须完全一致。

  Examples:
    >>> from jax.sharding import PartitionSpec as P
    >>> mesh_rows = 2
    >>> mesh_cols =  jax.device_count() // 2
    ...
    >>> mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(mesh_rows, mesh_cols), ('x', 'y'))

    >>> sharding = jax.sharding.NamedSharding(mesh, P(('x', 'y'),))
    >>> rows_per_device = 2
    >>> feature_length = 32
    >>> per_device_shape = (rows_per_device, feature_length)
    >>> per_host_shape = (rows_per_device * len(mesh.local_devices), feature_length)
    >>> per_host_generator = lambda : np.arange(np.prod(per_host_shape)).reshape(per_host_shape)
    >>> per_host_data = per_host_generator()  # replace with your own per-host data pipeline that outputs numpy arrays
    >>> global_shape = (rows_per_device * len(sharding.device_set), ) + per_device_shape[1:]
    >>> output_global_array = jax.make_array_from_process_local_data(sharding, per_host_data, global_shape)
    ...
    >>> assert output_global_array.addressable_data(0).shape == per_device_shape
    >>> assert output_global_array.shape == global_shape

  NB: 虽然大多数分片是均匀的，但也可以设计出奇特的分片网格：
  在某些维度上每个进程的设备呈非网格状排列，
  或者索引之间存在非平凡的重叠。
  这种分片在这些维度上被称为“非均匀”。
  此时，这些方向上的全局形状必须与本地形状一致，
  因为不存在以不重叠方式表示所有所需进程内数据的
  有意义做法。例如对于 global_shape 4x4，
  若分片形如::

      0123
      2103
      4675
      4567

  共有 4 个进程，分别包含设备 (0,1)、(2, 3)、(4, 5)、(6, 7)。
  那么每个主机的数据形如::

      xx..    ..xx     ....    ....
      .xx.    x..x     ....    ....
      ....    ....     x..x    .xx.
      ....    ....     xx..    ..xx

  该分片在行方向上是均匀的（每个主机需要第 1-2 行或第 3-4 行），
  在列方向上是非均匀的（各主机需要的列集合彼此重叠但不相同）。
  因此所有主机的本地数据形状都必须是 2x4 或 4x4，
  尽管每个主机本来都可能放进 2x2 的形状。
  此时用户必须显式提供 global_shape，并且对于 local_shape=(2, 4)，
  可能合法的全局形状是 (2, 4) 与 (4, 4)。

  另一方面，对于如下分片::

      0213   x.x.  .x.x.  ....  ....
      0213   x.x.  .x.x.  ....  ....
      4657   ....  ....   .x.x  x.x.
      4657   ....  ....   .x.x  x.x.

  对于 local_shape=(2, 2)，该函数可以接受 2x2、2x4、4x2 与
  4x4 的全局形状。此时把 global_shape 设为 None 等价于
  将其设为 (4, 4)。

  Args:
    sharding: 全局数组的分片方式。
    local_data: 主机上的数据，将被放置到本地设备上。
      每一维要么与 global_shape 匹配，
      要么与 num_addressable_indices(dim) 匹配。
    global_shape: 全局数组的目标形状。若为 None，则根据
      local_data 与 sharding 推断。

  Returns:
    一个分片为 sharding、形状为 global_shape 的张量。
  """
  # pyformat: enable
  local_data_flat, treedef = tree_flatten(local_data)
  sharding_flat = broadcast_prefix(sharding, local_data)
  sharding_flat = map(
      partial(api.pspec_to_sharding, 'make_array_from_process_local_data'),
      sharding_flat)
  global_shape_flat = broadcast_prefix(
      global_shape, local_data,
      is_leaf=lambda x: x is None or isinstance(x, tuple))
  if xla_bridge.process_count() == 1:
    # 安全检查：所提供的数据是否与期望的 global_shape 匹配
    for s, d in zip(global_shape_flat, local_data_flat):
      if s is not None and s != d.shape:
        raise ValueError(
            "When calling `make_array_from_process_local_data` on a single"
            " process, global_shape should be None or equal to"
            f" local_data.shape.Got global_shape={s} and"
            f" local_data.shape={d.shape}."
        )
    return api.device_put(local_data, sharding)

  out = [_array_from_process_local_data(data, s, shape)
         for data, s, shape in zip(local_data_flat, sharding_flat, global_shape_flat)]
  return tree_unflatten(treedef, out)

def _array_from_process_local_data(
    local_data: np.ndarray, sharding: Sharding,
    global_shape: Shape | None = None) -> ArrayImpl:
  # TODO(sandler): 考虑支持部分指定的 global_shape，或
  # 在 api 中公开 local_to_global_shape。
  local_shape = local_data.shape
  if global_shape is None:
    global_shape = local_to_global_shape(sharding, local_shape)  # pyrefly: ignore[bad-assignment]
    assert global_shape is not None
    if None in global_shape:
      raise ValueError(
          "Unable to compute global_shape due to non-uniform sharding."
          f" Specify global shape directly. Partially computed {global_shape=}."
      )
  elif None in global_shape:
    raise ValueError(f"{global_shape=} has Nones. This is not supported.")
  full_dim = []
  for i, (data_dim, global_dim) in enumerate(
      zip(local_data.shape, global_shape)
  ):
    full_dim.append(data_dim == global_dim)
    if data_dim != global_dim:
      process_slice = num_addressable_indices(sharding, i, global_shape)
      if process_slice != data_dim:
        raise ValueError(
            "Invalid host data, each dimension should match either global or "
            f"process shape. In dimension {i}, the process data has {data_dim} "
            f"elements. Process addresses {process_slice} elements and "
            f"{global_shape=}."
        )
  addressable_shards = sharding.addressable_devices_indices_map(global_shape)
  shard = next(iter(addressable_shards.values()))
  assert shard is not None
  shard_shape = _get_shape_from_index(shard, global_shape)
  slices_for_each_dim: list[list[int]] = [[] for _ in global_shape]
  for shard_index in addressable_shards.values():
    assert shard_index is not None
    for i, slc in enumerate(shard_index):
      slices_for_each_dim[i].append(slc.start or 0)
  for i in range(len(global_shape)):
    slices_for_each_dim[i] = sorted(set(slices_for_each_dim[i]))

  @functools.lru_cache(maxsize=4096)
  def local_slice(i, start):
    # 查找该切片在本维度切片列表中的位置。
    # 这决定了它在 host_local_data 中的切片。
    start = slices_for_each_dim[i].index(start or 0) * shard_shape[i]
    end = start + shard_shape[i]
    return slice(start, end)

  def cb(index: Index | None) -> ArrayLike:
    assert index is not None
    data_slice = (
        slc if full_dim[i] else local_slice(i, slc.start)
        for i, slc in enumerate(index)
    )
    return local_data[tuple(data_slice)]

  return make_array_from_callback(global_shape, sharding, cb)


def make_array_from_single_device_arrays(
    shape: Shape, sharding: Sharding, arrays: Sequence[basearray.Array], *,
    dtype: DTypeLike | None = None,
) -> ArrayImpl:
  r"""由一组 ``jax.Array``（每个位于单个设备上）返回一个 ``jax.Array``。
      输入 ``sharding`` 的网格中每个设备都必须在 ``arrays`` 中有对应的数组。

  Args:
    shape : 输出 ``jax.Array`` 的形状。它表达的信息已包含在
      ``sharding`` 与 ``arrays`` 中，此处用作双重检查。
    sharding: 全局 ``Sharding`` 实例，描述输出 jax.Array 如何在各设备上布局。
    arrays: 每个都可在单设备上寻址的 ``jax.Array`` 组成的 `list` 或 `tuple`。``len(arrays)``
      必须等于 ``len(sharding.addressable_devices)``，且各数组的形状必须相同。在多进程代码中，
      每个进程会用各自数据对应的不同 ``arrays`` 参数调用。
      这些数组通常通过 ``jax.device_put`` 创建。
    dtype: 输出 ``jax.Array`` 的数据类型。若未提供，则使用 ``arrays`` 中第一个数组的
      数据类型。若 ``arrays`` 为空，则必须提供 ``dtype`` 参数。

  Returns:
    一个全局 ``jax.Array``，按 ``sharding`` 分片，形状等于 ``shape``，
      且每个设备上的内容与 ``arrays`` 对应。

  Examples:

    >>> import math
    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> import numpy as np
    ...
    >>> mesh_rows = 2
    >>> mesh_cols =  jax.device_count() // 2
    ...
    >>> global_shape = (8, 8)
    >>> mesh = Mesh(np.array(jax.devices()).reshape(mesh_rows, mesh_cols), ('x', 'y'))
    >>> sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    >>> inp_data = np.arange(math.prod(global_shape)).reshape(global_shape)
    ...
    >>> arrays = [
    ...    jax.device_put(inp_data[index], d)
    ...        for d, index in sharding.addressable_devices_indices_map(global_shape).items()]
    ...
    >>> arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
    >>> assert arr.shape == (8,8) # arr.shape is (8,8) regardless of jax.device_count()

  如果你有一个本地数组并想把它转换为全局 jax.Array，
  请使用 ``jax.make_array_from_process_local_data``。
  """
  if isinstance(arrays, Sequence):
    dtype = _get_and_check_dtype(
        arrays, dtype, "make_array_from_single_device_arrays")

  # 所有输入数组都应是已提交（committed）的。
  # 在单控制器系统上检查这一点开销很大。
  aval = core.update_aval_with_sharding(
      core.ShapedArray(shape, dtype, weak_type=False), sharding)
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    return aval.dtype._rules.make_sharded_array(aval, sharding, arrays,
                                                committed=True)
  arrays = list(arrays) if isinstance(arrays, tuple) else arrays
  # TODO(phawkins): 理想情况下 cast() 应该被检查。
  try:
    return ArrayImpl(aval, sharding, cast(Sequence[ArrayImpl], arrays),
                     committed=True)
  except TypeError:
    if not isinstance(arrays, list):
      raise TypeError("jax.make_array_from_single_device_arrays `arrays` "
                      "argument must be a list or tuple, but got "
                      f"{type(arrays)}.")
    if any(isinstance(arr, core.Tracer) for arr in arrays):
      raise ValueError(
          "jax.make_array_from_single_device_arrays requires a list of concrete"
          f" arrays as input, but got types {set(map(type, arrays))}")
    raise

dtypes.register_canonicalize_value_handler(ArrayImpl, None)

def _get_aval_array(self):
  return core.update_aval_with_sharding(self.aval, self.sharding)
core.pytype_aval_mappings[ArrayImpl] = _get_aval_array


def _array_mlir_constant_handler(val, aval):
  try:
    return mlir.ir_constant(val._value)
  except RuntimeError as e:
    # TODO(yashkatariya): 理想情况下应该捕获 ArrayImpl 中 `_value`
    # 函数抛出的自定义异常，而不是检查错误字符串。
    if 'Fetching value for `jax.Array` that spans non-addressable' in str(e):
      raise RuntimeError(
          "Closing over jax.Array that spans non-addressable (non process"
          " local) devices is not allowed. Please pass such arrays as arguments"
          f" to the function. Got jax.Array: {val.aval.str_short()}") from e
    raise

mlir.register_constant_handler(ArrayImpl, _array_mlir_constant_handler)

if config.use_simplified_jaxpr_constants.value:
  core.literalable_types.add(ArrayImpl)

# NOTE(skye): 我们可以重构为直接从输入 ShardingSpec 生成 _multi_slice 参数，
# 而不是从索引生成。但这需要重复 spec_to_indices 的排序逻辑，
# 而那部分逻辑比我们这里要支持的索引逻辑更微妙、也更易变动。
def as_slice_indices(arr: Any, idx: Index) -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
  """返回 start_indices、limit_indices、removed_dims"""
  start_indices = [0] * arr.ndim
  limit_indices = list(arr.shape)
  removed_dims: list[int] = []

  tuple_idx = idx if isinstance(idx, tuple) else (idx,)
  for dim, sub_idx in enumerate(tuple_idx):
    if isinstance(sub_idx, int):
      start_indices[dim] = sub_idx
      limit_indices[dim] = sub_idx + 1
      removed_dims.append(dim)
    elif sub_idx == slice(None):
      continue
    else:
      assert isinstance(sub_idx, slice), sub_idx
      assert isinstance(sub_idx.start, int), sub_idx
      assert isinstance(sub_idx.stop, int), sub_idx
      start_indices[dim] = sub_idx.start
      limit_indices[dim] = sub_idx.stop

  return tuple(start_indices), tuple(limit_indices), tuple(removed_dims)


def shard_device_array(x, devices, indices, sharding):
  start_indices, limit_indices, removed_dims = unzip3(
      as_slice_indices(x, idx) for idx in indices)
  if sharding.is_fully_replicated:
    shards = [x] * len(devices)
  else:
    # TODO(yashkatariya): 也许应该在 InputsHandler.__call__
    # 中调用该处理器时设置它？
    with (_internal_use_concrete_mesh(empty_concrete_mesh),
          use_abstract_mesh(empty_abstract_mesh)):
      shards = x._multi_slice(start_indices, limit_indices, removed_dims)
  aval = core.shaped_abstractify(x)
  return pxla.batched_device_put(aval, sharding, shards, devices)


def shard_sharded_device_array_slow_path(x, devices, indices, sharding):
  candidates = defaultdict(list)
  bufs = [buf.data for buf in x.addressable_shards]
  arr_indices = tuple(x.sharding.devices_indices_map(x.shape).values())
  for buf, idx in safe_zip(bufs, arr_indices):
    candidates[hashed_index(idx)].append(buf)

  bufs = []
  for idx, device in safe_zip(indices, devices):
    # 查找所有包含逻辑数组正确切片的缓冲区。
    candidates_list = candidates[hashed_index(idx)]
    if not candidates_list:
      return pxla.shard_args([sharding], [None],
                             [xc.ArrayCopySemantics.REUSE_INPUT], [x._value],
                             canonicalize=False)[0]
    # 尝试找一个已经在正确设备上的候选缓冲区，
    # 否则就复制其中一个。
    for buf in candidates_list:
      if buf.devices() == {device}:
        bufs.append(buf)
        break
    else:
      bufs.append(candidates_list[-1])
  return pxla.batched_device_put(x.aval, sharding, bufs, devices)


@cache(max_size=4096, trace_context_in_key=False)
def _fallback_check_via_indices(src_sharding, dst_sharding, shape):
  src_indices = src_sharding.addressable_devices_indices_map(shape).values()
  dst_indices = dst_sharding.addressable_devices_indices_map(shape).values()
  return tuple(src_indices) == tuple(dst_indices)

@cache(max_size=4096, trace_context_in_key=False)
def _sharding_indices_and_eq(src_sharding, dst_sharding, ndim):
  hlos_eq = are_hlo_shardings_equal(src_sharding._to_xla_hlo_sharding(ndim),
                                    dst_sharding._to_xla_hlo_sharding(ndim))
  len_eq = (len(src_sharding._internal_device_list.addressable_device_list) ==
            len(dst_sharding._internal_device_list.addressable_device_list))
  return hlos_eq and len_eq


def _array_shard_arg(xs, shardings, layouts, copy_semantics):
  util.test_event("_array_shard_arg")
  results = []
  batch_xs, batch_devs, batch_shardings, batch_indices = [], [], [], []
  batch_cs = []

  for i, (x, sharding, layout, cs) in enumerate(
      safe_zip(xs, shardings, layouts, copy_semantics)):
    x._check_if_deleted()
    try:
      same_sharding = _sharding_indices_and_eq(x.sharding, sharding, len(x.shape))
    except NotImplementedError:
      same_sharding = _fallback_check_via_indices(x.sharding, sharding, x.shape)
    same_layout = True if layout is None else x.format.layout == layout

    if not x.is_fully_addressable:
      if same_sharding and same_layout:
        results.append(x)
      else:
        raise NotImplementedError(
            "Cannot reshard an input that is not fully addressable")
    else:
      devices = sharding._internal_device_list.addressable_device_list
      if same_sharding and same_layout:
        # 先加入占位结果，稍后再填充。
        results.append(None)
        # 累积传给 `batched_copy_array_to_devices_with_sharding` 的参数。
        batch_xs.append(x)
        batch_devs.append(devices)
        batch_shardings.append(sharding)
        batch_indices.append(i)
        batch_cs.append(cs)
      # 重新分片从这里开始：
      elif not same_layout:
        results.append(api.device_put(x, Format(layout, sharding)))
      else:
        indices = sharding.addressable_devices_indices_map(x.shape).values()
        if x.sharding.num_devices == 1:
          results.append(shard_device_array(x, devices, indices, sharding))
        else:
          results.append(
              shard_sharded_device_array_slow_path(x, devices, indices, sharding))

  util.test_event("batched_copy_array")
  copy_outs = xc.batched_copy_array_to_devices_with_sharding(
      batch_xs, batch_devs, batch_shardings, batch_cs)
  for i, copy_out in safe_zip(batch_indices, copy_outs):
    assert results[i] is None
    results[i] = copy_out
  return results
pxla.shard_arg_handlers[ArrayImpl] = _array_shard_arg


def _array_global_result_handler(global_aval, out_sharding, committed):
  if global_aval.dtype == dtypes.float0:
    def handler(xs):
      return literals.TypedNdArray(np.zeros(global_aval.shape, dtypes.float0),
                                   aval=global_aval)
    phys_aval = core.physical_aval(global_aval)
    return xc.array_result_handler(phys_aval, out_sharding, committed=committed,
                                   _skip_checks=True).wrap(handler)
  if dtypes.issubdtype(global_aval.dtype, dtypes.extended):
    return global_aval.dtype._rules.global_sharded_result_handler(
        global_aval, out_sharding, committed)
  return xc.array_result_handler(
      global_aval, out_sharding, committed=committed, _skip_checks=True
  )
pxla.global_result_handlers[core.ShapedArray] = _array_global_result_handler

# Token 处理器

def _token_shard_arg(xs, shardings, layouts, copy_semantics):
  results = []
  for x, sharding, layout in safe_zip(xs, shardings, layouts):
    assert layout is None
    x.block_until_ready()
    x = np.array([], dtype=bool)
    aval = core.typeof(x)
    devices = sharding._addressable_device_assignment
    results.append(pxla.batched_device_put(
        aval, sharding, [x] * len(devices), devices))
  return results
pxla.shard_arg_handlers[core.Token] = _token_shard_arg


def _token_global_result_handler(global_aval, out_sharding, committed):
  array_handler = _array_global_result_handler(
      core.get_token_aval(), out_sharding, committed)
  def wrapper(array):
    return core.Token(array)
  return array_handler.wrap(wrapper)
pxla.global_result_handlers[core.AbstractToken] = _token_global_result_handler
