# Copyright 2020 The JAX Authors.
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

# 文件职责：实现 JAX 数组与 DLPack 张量之间的互转（`to_dlpack` / `from_dlpack`）。
# DLPack 是跨框架零拷贝共享张量内存的开放标准，本模块据此与 PyTorch、
# TensorFlow、CuPy 等外部框架交换设备缓冲区，并尽可能避免数据复制。
# 关键概念：DLPack 版本协商（`max_version`）、`(DLDeviceType, local_hardware_id)`
# 形式的设备描述、外部流的同步，以及 `copy` 语义（True 必须复制 / False 绝不
# 复制 / None 按需复制）；设备平台映射见 `_DL_DEVICE_TO_PLATFORM`。

from __future__ import annotations

from typing import Any

from jax._src import array
from jax._src import dtypes
from jax._src import xla_bridge
from jax._src.api import device_put
from jax._src.lax.lax import _array_copy
from jax._src.lib import _jax
from jax._src.lib import xla_client
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy import scalar_types as jnp_types
from jax._src.sharding import Sharding
from jax._src.typing import Array, DLDeviceType, DTypeLike

import numpy as np

DLPACK_VERSION = (0, 8)
MIN_DLPACK_VERSION = (0, 5)

# dlpack 支持的一组数据类型。
# 注意：在该集合中查找时一定要用“类型”，而不是 dtype 实例，
# 因为二者的哈希值不同。
# 例如，
# hash(jnp.float32) != hash(jnp.dtype(jnp.float32))
# hash(jnp.float32) == hash(jnp.dtype(jnp.float32).type)

# TODO(vanderplas): 移除这个集合
SUPPORTED_DTYPES: frozenset[DTypeLike] = frozenset({
    jnp_types.int8, jnp_types.int16, jnp_types.int32, jnp_types.int64,
    jnp_types.uint8, jnp_types.uint16, jnp_types.uint32, jnp_types.uint64,
    jnp_types.float16, jnp_types.bfloat16, jnp_types.float32, jnp_types.float64,
    jnp_types.complex64, jnp_types.complex128, jnp_types.bool_})

SUPPORTED_DTYPES_SET: frozenset[np.dtype] = frozenset({np.dtype(dt) for dt in SUPPORTED_DTYPES})


def is_supported_dtype(dtype: DTypeLike) -> bool:
  """检查 `jax.dlpack` 是否支持该 dtype。"""
  if dtype is None:
    # NumPy 会静默地把它转换为 float64，这可能出乎意料。
    raise TypeError(f"Expected a string or dtype-like object; got {dtype=}")
  return np.dtype(dtype) in SUPPORTED_DTYPES_SET


def _to_dlpack(x: Array, stream: int | Any | None,
               src_device: _jax.Device | None = None,
               device: _jax.Device | None = None,
               copy: bool | None = None):

  if src_device is None:
    src_device, = x.devices()
  if device and (src_device is None or device != src_device):
    if copy is not None and not copy:
      raise ValueError(
        f"Specified {device=} which requires a copy since the source device "
        f"is {repr(src_device)}, however copy=False. Set copy=True or "
        "copy=None to perform the requested operation."
      )
    else:
      arr = device_put(x, device)
  else:
    arr = _array_copy(x) if copy else x
  return _jax.buffer_to_dlpack_managed_tensor(
    arr.addressable_data(0), stream=stream
  )


_DL_DEVICE_TO_PLATFORM = {
    DLDeviceType.kDLCPU: "cpu",
    DLDeviceType.kDLCUDA: "cuda",
    DLDeviceType.kDLCUDAHost: "cuda",
    DLDeviceType.kDLROCM: "rocm",
    DLDeviceType.kDLROCMHost: "rocm",
    DLDeviceType.kDLTPUHost: "tpu",
    DLDeviceType.kDLOneAPI: "oneapi",
}


def to_dlpack(x: Array, stream: int | Any | None = None,
              src_device: _jax.Device | None = None,
              dl_device: tuple[DLDeviceType, int] | None = None,
              max_version: tuple[int, int] | None = None,
              copy : bool | None = None):
  """返回一个封装了 :class:`~jax.Array` ``x`` 的 DLPack 张量。

  Args:
    x: 一个 :class:`~jax.Array`，位于 CPU 或 GPU 上。
    stream: 可选的、依赖平台的流，需要在其上等待直到缓冲区就绪。它对应
      https://dmlc.github.io/dlpack/latest/python_spec.html 中记载的
      ``__dlpack__`` 的 `stream` 参数。
    src_device: 一个 CPU 或 GPU 的 :class:`~jax.Device`。
    dl_device: DLPack 格式的 ``(dl_device_type, local_hardware_id)`` 元组，
      例如由 ``__dlpack_device__`` 产生的那种形式。
    max_version: 消费者（即 ``__dlpack__`` 的调用方）所支持的
      最高 DLPack 版本，形式为 ``(major, minor)`` 二元组。
      本函数并不保证返回的 capsule 版本为
      ``max_version``。
    copy: 布尔值，表示是否复制输入。若 ``copy=True``，
      则函数必须始终复制。若 ``copy=False``，则函数
      绝不能复制，并在确有必要复制时抛出错误。
      若 ``copy=None``，则函数应尽可能避免复制，
      但在需要时可以复制。

  Returns:
    一个 DLPack PyCapsule 对象。

  Note:
    虽然 JAX 数组始终不可变，``DLPackManagedTensor`` 缓冲区
    却无法被标记为不可变，并且 JAX 之外的进程有可能就地
    修改它们。如果由 JAX 数组派生出的 DLPack 缓冲区被修改，
    那么在使用对应的 JAX 数组时可能导致未定义行为。当 JAX
    最终支持 ``DLManagedTensorVersioned``（DLPack 1.0）之后，
    就可以把缓冲区指定为只读。
  """
  if not isinstance(x, array.ArrayImpl):
    raise TypeError("Argument to to_dlpack must be a jax.Array, "
                    f"got {type(x)}")

  device = None
  dl_device_type, local_hardware_id = dl_device if dl_device else (None, None)
  if dl_device_type:
    try:
      dl_device_platform = _DL_DEVICE_TO_PLATFORM[dl_device_type]
      backend = xla_bridge.get_backend(dl_device_platform)
      device = backend.device_from_local_hardware_id(local_hardware_id)
    except KeyError:
      # https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
      # 建议使用 BufferError。
      raise BufferError(
          "The device specification passed to to_dlpack contains an"
          f" unsupported device type (DLDeviceType: {dl_device_type})"
      ) from None

  # 随着新版本逐渐被采用，我们可以通过 max_version 参数保留一些
  # 用于兼容的旧代码路径。
  # TODO(micky774): 当 XLA 支持 DLManagedTensorVersioned（DLPack 1.0）之后，
  # 弃用默认使用 DLPackManagedTensor 的做法，并把当前的 _to_dlpack 改作
  # (0,5) <= max_version < (1,0) 的旧版路径。
  if max_version is None or max_version >= DLPACK_VERSION:
    # 最新版本
    return _to_dlpack(
      x, stream=stream,
      src_device=src_device,
      device=device,
      copy=copy
    )
  elif max_version >= MIN_DLPACK_VERSION:
    # 支持的最旧版本
    return _to_dlpack(
      x, stream=stream,
      src_device=src_device,
      device=device,
      copy=copy
    )
  else:
    raise BufferError(
      f"JAX does not support any version below {MIN_DLPACK_VERSION} but "
      f"version ({max_version}) was requested."
    )

def _check_device(device, dlpack_device, copy):
  if device and dlpack_device != device:
    if copy is not None and not copy:
      raise ValueError(
        f"Specified {device=} which requires a copy since the source device "
        f"is {repr(dlpack_device)}, however copy=False. Set copy=True or "
        "copy=None to perform the requested operation."
      )

def _place_array(_arr, device, dlpack_device, copy):
  if device and dlpack_device != device:
    return device_put(_arr, device)
  if copy:
    return jnp.array(_arr, copy=True)
  return _arr

def _is_tensorflow_tensor(external_array):
  t = type(external_array)
  return (
      t.__qualname__ == "EagerTensor"
      and t.__module__.endswith("tensorflow.python.framework.ops")
  )

def from_dlpack(external_array,
                device: _jax.Device | Sharding | None = None,
                copy: bool | None = None):
  """返回 DLPack 张量对应的 :class:`~jax.Array` 表示。

  如果没有请求设备迁移或复制，返回的 :class:`~jax.Array` 会与
  ``external_array`` 共享内存。

  Args:
    external_array: 一个拥有 ``__dlpack__`` 和 ``__dlpack_device__`` 方法的
      数组对象。
    device: （可选的）:py:class:`Device`，表示返回的数组
      应放置在哪台设备上。若给出，则结果会被提交到该设备。
      若未指定，则结果数组会被解包到它原本来自的那台
      设备上。把 ``device`` 设为与 ``external_array`` 来源
      不同的设备将需要复制，这意味着 ``copy`` 必须设为
      ``True`` 或 ``None``。
    copy: （可选的）布尔值，控制是否执行复制。若 ``copy=True``，
      则始终执行复制，即使解包到同一设备上也是如此。
      若 ``copy=False``，则绝不执行复制，并在必要时抛出错误。
      当 ``copy=None`` 时，如果设备迁移需要，则可以执行
      复制。

  Returns:
    一个 jax.Array

  Note:
    虽然 JAX 数组始终不可变，dlpack 缓冲区却无法被
    标记为不可变，并且 JAX 之外的进程有可能就地修改
    它们。如果由 dlpack 缓冲区构造出 jax 数组，而该
    缓冲区后来被就地修改，那么在使用对应的 JAX 数组时
    可能导致未定义行为。
  """
  if isinstance(device, Sharding):
    device_set = device.device_set
    if len(device_set) > 1:
      raise ValueError(
        "from_dlpack can only unpack a dlpack tensor onto a singular device, but "
        f"a Sharding with {len(device_set)} devices was provided."
      )
    device, = device_set
  if not hasattr(external_array, "__dlpack__") or not hasattr(external_array, "__dlpack_device__"):
    raise TypeError(
        "The array passed to from_dlpack must have __dlpack__ and __dlpack_device__ methods."
    )

  dl_device_type, device_id = external_array.__dlpack_device__()
  try:
    dl_device_platform = _DL_DEVICE_TO_PLATFORM[dl_device_type]
  except KeyError:
    raise TypeError(
        "Array passed to from_dlpack is on unsupported device type "
        f"(DLDeviceType: {dl_device_type}, array: {external_array}"
    ) from None

  backend = xla_bridge.get_backend(dl_device_platform)
  dlpack_device = backend.device_from_local_hardware_id(device_id)
  _check_device(device, dlpack_device, copy)
  if _is_tensorflow_tensor(external_array):
    # TensorFlow 不支持 stream=。
    stream = None
  elif dl_device_type in (
      DLDeviceType.kDLCUDAHost,
      DLDeviceType.kDLROCMHost,
      DLDeviceType.kDLTPUHost,
  ):
    # 某些生产者（例如使用 is_pinned() 的 torch.Tensor）会让固定内存张量
    # 走它们的 CPU __dlpack__，而后者拒绝非 None 的 stream 参数。
    stream = None
  else:
    try:
      stream = dlpack_device.get_stream_for_external_ready_events()
    except _jax.JaxRuntimeError as err:
      if "UNIMPLEMENTED" in str(err):
        stream = None
      else:
        raise
  dlpack = external_array.__dlpack__(stream=stream)

  try:
    arr = _jax.dlpack_managed_tensor_to_buffer(
      dlpack, dlpack_device, stream, copy, int(dl_device_type))
  except xla_client.XlaRuntimeError as e:
    se = str(e)
    if "is not aligned to" in se:
      i = se.index("is not aligned to")
      raise ValueError(
        "Specified input which requires a copy since the source data "
        f"buffer {se[i:]} However copy=False. Set copy=True or "
        "copy=None to perform the requested operation."
      )
    else:
      raise
  # TODO(phawkins): 当我们准备好在非 x64 模式下支持 x64 数组时，
  # 就修改语义，不在此处做规范化。
  arr = jnp.asarray(arr, dtype=dtypes.canonicalize_dtype(arr.dtype))
  if copy:
    # 复制已由 dlpack_managed_tensor_to_buffer 处理。
    copy = None
  return _place_array(arr, device, dlpack_device, copy)
