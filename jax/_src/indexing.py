# Copyright 2026 The JAX Authors.
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

# 文件职责：为 JAX 提供支持动态起始索引的切片抽象。
# `Slice` 是一个 pytree，用起始索引、大小和步长描述一次切片，其中起始与大小
# 既可以是静态已知的，也可以是运行期才确定的数组；`dslice`（别名 `ds`）把它
# 包装成类似内置 `slice` 的构造接口，于是 `x[jax.ds(i, size)]` 能在 `jit`
# 内部使用动态下标。该模块是 `jax.numpy` 动态切片语义的底层支撑。

from __future__ import annotations

import dataclasses
from typing import overload

from jax._src import core
from jax._src import tree_util
from jax._src.typing import Array


@tree_util.register_pytree_node_class
@dataclasses.dataclass(slots=True)
class Slice:
  """带有起始索引和大小的切片。

  起始索引和大小既可以是静态的，即在追踪
  与编译期已知，也可以是动态的。
  """

  start: int | Array
  size: int | Array
  stride: int = 1

  def __post_init__(self):
    if self.stride < 0:
      raise ValueError("`stride` must be >= 0.")

  @property
  def is_dynamic_start(self):
    return not core.is_dim(self.start)

  @property
  def is_dynamic_size(self):
    return not core.is_dim(self.size)

  def tree_flatten(self):
    # 若 `start` 静态已知，就把它当作静态信息处理
    xs = ()
    data = ()
    xs += (self.start,) if self.is_dynamic_start else (None,)
    data += (None,) if self.is_dynamic_start else (self.start,)
    xs += (self.size,) if self.is_dynamic_size else (None,)
    data += (None,) if self.is_dynamic_size else (self.size,)
    data += (self.stride,)
    return xs, data

  @classmethod
  def tree_unflatten(cls, aux_data, children) -> Slice:
    start, size = (
        a if a is not None else b for a, b in zip(children, aux_data[:2])
    )
    return cls(start, size, aux_data[2])

  @classmethod
  def from_slice(cls, slc: slice, size: int) -> Slice:
    start, step, size = core.canonicalize_slice(slc, size)
    if step < 1:
      raise ValueError(f"slice must have a step >= 1 (found: {step})")
    return cls(start, size, step)


class _NotSpecified:
  pass


@overload
def dslice(
    start: None,
    size: _NotSpecified,
    stride: int | None = ...,
) -> slice:
  ...


@overload
def dslice(
    start: int | Array | None,
    size: int | Array | _NotSpecified = ...,
    stride: int | None = ...,
) -> Slice:
  ...


def dslice(
    start: int | Array | None,
    size: int | Array | _NotSpecified = _NotSpecified(),
    stride: int | None = None,
) -> slice | Slice:
  """由起始索引和大小构造一个 ``Slice``。

  ``dslice`` 的语义与内置的 ``slice`` 类型一致：

  * ``dslice(None)`` 即 ``:``
  * ``dslice(j)`` 即 ``:j``
  * ``dslice(i, j)`` 即 ``i:i+j``
  * ``dslice(i, j, stride)`` 即 ``i:i+j:stride``

  Examples:

    >>> x = jax.numpy.arange(10)
    >>> i = 4
    >>> x[i: i + 2]  # standard indexing requires i to be static
    Array([4, 5], dtype=int32)
    >>> x[jax.ds(i, 2)]  # equivalent which allows i to be dynamic
    Array([4, 5], dtype=int32)

    下面是一个使用动态起始索引进行切片的明确示例：

    >>> @jax.jit(static_argnames='size')
    ... def f(x, i, size):  # example of when `
    ...   return x[jax.ds(i, size)]
    ...
    >>> f(x, i, 2)
    Array([4, 5], dtype=int32)
  """
  if start is None:
    if isinstance(size, _NotSpecified):
      return slice(None, None, stride)
    start = 0
  if stride is None:
    stride = 1
  if not isinstance(stride, int):
    raise ValueError("Non-static stride in `dslice`")
  if isinstance(size, _NotSpecified):
    start, size = 0, start
  return Slice(start, size, stride)


ds = dslice  # 便捷别名。
