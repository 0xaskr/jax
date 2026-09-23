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

"""
`jax._src.typing`：JAX 类型注解
---------------------------------------

本子模块仍在开发中；当我们最终确定这里的内容后，它会在 `jax.typing` 处导出。
在此之前，这里的内容应视为不稳定，可能随时变更而不另行通知。

要查看促成这些工具开发的提案，请见
https://github.com/jax-ml/jax/pull/11859/。
"""

# 文件职责：定义 JAX 在静态类型检查语境下使用的公共类型注解。
# 这里给出 `SupportsDType` / `SupportsShape` / `SupportsSize` / `SupportsNdim`
# 等结构化协议，以及 `DTypeLike`、`Shape`、`Index`、`ArrayLike` 等类型别名，
# 供类型检查器与用户标注 JAX 数组、dtype、形状和索引相关的接口。
# 本模块内容尚未稳定，最终会以 `jax.typing` 的形式对外导出。

from __future__ import annotations

from collections.abc import Sequence
import enum
import typing
from types import EllipsisType
from typing import Any, Protocol

from jax._src.basearray import (
    ArrayLike as ArrayLike,
    Array as Array,
    StaticScalar as StaticScalar,
)
import numpy as np

DType = np.dtype

# TODO(jakevdp, froystig): 把 ExtendedDType 改为协议
ExtendedDType = Any


@typing.runtime_checkable
class SupportsDType(Protocol):
  @property
  def dtype(self, /) -> DType: ...

class SupportsShape(Protocol):
  @property
  def shape(self, /) -> tuple[int, ...]: ...

class SupportsSize(Protocol):
  @property
  def size(self, /) -> int: ...

class SupportsNdim(Protocol):
  @property
  def ndim(self, /) -> int: ...

# `DTypeLike` 用于标注 `np.dtype` 的输入，这些输入会返回
# 一个合法的 JAX 数据类型。它与 `numpy.typing.DTypeLike` 不同，
# 因为 JAX 不支持 object 或结构化数据类型。
# 与 `np.typing.DTypeLike` 不同，我们排除了 `None`，当允许 `None` 时
# 要求显式标注。
# TODO(jakevdp): 考虑是否把 ExtendedDtype 加入联合类型。
DTypeLike = (
  str            # 例如 'float32'、'int32'
  | type[Any]    # 例如 np.float32、np.int32、float、int
  | np.dtype     # 例如 np.dtype('float32')、np.dtype('int32')
  | SupportsDType  # 例如 jnp.float32、jnp.int32
)

# 形状是维度大小的元组，维度大小通常是整数。我们允许
# 各模块扩展维度大小的集合以包含其他类型，例如
# `export.DimExpr` 中的符号维度。
DimSize = int | Any  # 可扩展的
Shape = Sequence[DimSize]

class DuckTypedArray(Protocol):
  @property
  def dtype(self) -> DType: ...
  @property
  def shape(self) -> Shape: ...

# `Array` 是标准 JAX 数组以及由 `jax.lax`、`jax.numpy` 中核心函数
# 产生的追踪器的类型注解；它不打算涵盖未来出现的非标准数组类型，
# 如 `KeyArray` 和 `BInt`。它在上面被导入。

# `ArrayLike` 是所有可被隐式转换为标准 JAX 数组的对象的联合类型
# （即不包括未来出现的非标准数组类型，如 `KeyArray` 和 `BInt`）。
# 它与 `np.typing.ArrayLike` 不同，因为它既不接受任意序列，
# 也不接受字符串数据。

# 我们为已废弃的参数使用一个类，以避免使用 Any/object 类型，
# 因为那会在静态分析中引入复杂情况与错误
class DeprecatedArg:
  def __repr__(self):
    return "Deprecated"

# dlpack.h 枚举的镜像
class DLDeviceType(enum.IntEnum):
  kDLCPU = 1
  kDLCUDA = 2
  kDLCUDAHost = 3
  kDLROCM = 10
  kDLROCMHost = 11
  kDLTPUHost = 20
  kDLOneAPI = 14

AnyInt = int | np.integer
StaticIndex = AnyInt | slice | EllipsisType
Index = StaticIndex | None | Sequence[AnyInt] | Array | np.ndarray
