# Copyright 2022 The JAX Authors.
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

# 文件职责：`jax.Array` 的基类与类型标注。
# 定义 Array 抽象基类（dtype/ndim/size/shape/sharding/committed/device 等属性，
# addressable_data、copy_to_host_async 等接口），运行时由 C++ 的 xc.Array 替换实现；
# 另定义 StaticScalar 与 ArrayLike 类型别名。具体数组实现见 `jax._src.array`。

# 注意：本文件的类型标注定义在 basearray.pyi 中

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jax._src import deprecations
from jax._src.lib import xla_client as xc
from jax._src.util import use_cpp_class
import numpy as np


# TODO(jakevdp): 修复循环导入并定义这些类型。
Device = Any
Shard = Any
Sharding = Any

# Array 是标准 JAX 数组、以及由 jax.lax 和 jax.numpy 中核心函数产生的
# 追踪器的类型标注；它并不打算涵盖未来那些非标准数组类型，
# 例如 KeyArray 和 BInt。


class Array:
  """JAX 的数组基类

  ``jax.Array`` 是用于对 JAX 数组和追踪器做实例检查与类型标注的公开接口。
  它的主要用途是实例检查和类型标注；例如::

    x = jnp.arange(5)
    isinstance(x, jax.Array)  # 在被追踪函数内部和外部都返回 True。

    def f(x: Array) -> Array:  # 类型标注对已追踪和未追踪类型都有效。
      return x

  不应直接使用 ``jax.Array`` 来创建数组；而应使用 :mod:`jax.numpy` 提供的
  数组创建例程，例如 :func:`jax.numpy.array`、:func:`jax.numpy.zeros`、
  :func:`jax.numpy.ones`、:func:`jax.numpy.full`、
  :func:`jax.numpy.arange` 等。
  """
  # 为了静态类型分析，这些定义在配套的 basearray.pyi 文件中有对应的镜像定义。

  __slots__ = ['__weakref__']
  __hash__ = None

  # TODO(jakevdp): 弃用期结束后把 __numpy_dtype__ 设为 None。
  @property
  def __numpy_dtype__(self) -> np.dtype:
    # __numpy_dtype__ 协议在 NumPy v2.4.0 中加入。
    deprecations.warn(
      'jax-array-numpy-dtype',
      (
        "Implicit conversion of an array to a dtype is deprecated;"
        " rather than dtype=arr use dtype=arr.dtype. In the future"
        " this will result in an error."
      ),
      stacklevel=2,
      error_class=TypeError)
    return self.dtype

  @property
  def dtype(self) -> np.dtype:
    """数组的数据类型（:class:`numpy.dtype`）。"""
    raise NotImplementedError

  @property
  def ndim(self) -> int:
    """数组的维数。"""
    raise NotImplementedError

  @property
  def size(self) -> int:
    """数组中元素的总个数。"""
    raise NotImplementedError

  @property
  def shape(self) -> tuple[int, ...]:
    """数组的形状。"""
    raise NotImplementedError

  # 以下是在 ArrayImpl 上定义的分片相关方法与属性的文档：
  def addressable_data(self, index: int) -> Array:
    """返回特定索引处的可寻址数据所组成的数组。"""
    raise NotImplementedError

  @property
  def addressable_shards(self) -> Sequence[Shard]:
    """可寻址分片的列表。"""
    raise NotImplementedError

  @property
  def global_shards(self) -> Sequence[Shard]:
    """全局分片的列表。"""
    raise NotImplementedError

  @property
  def is_fully_addressable(self) -> bool:
    """这个 Array 是否完全可寻址？

    如果当前进程能够寻址到 :class:`Sharding` 中指定的所有设备，那么该
    jax.Array 就是完全可寻址的。在多进程 JAX 中，
    ``is_fully_addressable`` 等价于 “is_local”。

    注意，完全复制并不等于完全可寻址；也就是说，一个完全复制的
    jax.Array 可能横跨多台主机，并且不是完全可寻址的。
    """
    raise NotImplementedError

  @property
  def is_fully_replicated(self) -> bool:
    """这个 Array 是否完全复制？"""
    raise NotImplementedError

  @property
  def sharding(self) -> Sharding:
    """该数组的分片方式。"""
    raise NotImplementedError

  @property
  def committed(self) -> bool:
    """该数组是否已提交（committed）。

    当数组通过 JAX API 被显式放到某（些）设备上时，它就是已提交的。
    例如 ``jax.device_put(np.arange(8), jax.devices()[0])`` 被提交到设备 0，
    而 ``jax.device_put(np.arange(8))`` 未提交，会被放到默认设备上。

    涉及某些已提交输入的计算会在这些已提交的设备上进行，结果也会提交到
    同一（些）设备上。对已提交到不同设备上的参数调用运算则会报错。

    Examples:
      >>> a = jax.device_put(np.arange(8), jax.devices()[0])
      >>> b = jax.device_put(np.arange(8), jax.devices()[1])
      >>> a + b  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
        ...
      ValueError: Received incompatible devices for jitted computation.
    """
    raise NotImplementedError

  @property
  def device(self) -> Device | Sharding:
    """与 Array API 兼容的 device 属性。

    对于单设备数组，它返回一个 Device；对于分片数组，它返回一个 Sharding。
    """
    raise NotImplementedError

  def copy_to_host_async(self):
    """把 ``Array`` 异步复制到主机。

    对于位于加速器（例如 GPU 或 TPU）上的数组，JAX 可能会在主机上缓存
    该数组的值。通常，当用户请求读取设备上数组的值时，这会在幕后自动
    发生；但如果要一直等到用户请求时才发起设备到主机的复制，JAX 就必须
    在等待复制完成期间阻塞调用方。

    ``copy_to_host_async`` 请求 JAX 填充它在主机上维护的数组缓存，但不
    等待复制完成。这可以加快将来在主机上访问该数组内容的速度。
    """
    raise NotImplementedError


Array = use_cpp_class(xc.Array)(Array)
Array.__module__ = "jax"


# StaticScalar 是所有可转换为 JAX 数组、并且可以被标记为静态参数的
# 标量类型的联合。
StaticScalar = (
  np.bool_ | np.number  # NumPy 标量类型
  | bool | int | float | complex  # Python 标量类型
)
"""与 JAX 兼容的静态标量的类型标注。"""

# ArrayLike 是所有可隐式转换为标准 JAX 数组的对象的联合
# （即不包括未来那些非标准数组类型，例如 KeyArray 和 BInt）。
# 它与 np.typing.ArrayLike 的不同之处在于：它既不接受任意序列，
# 也不接受字符串数据。
ArrayLike = (
  Array  # JAX 数组类型
  | np.ndarray  # NumPy 数组类型
  | StaticScalar  # 合法的标量
)
"""JAX 类数组对象的类型标注。"""
