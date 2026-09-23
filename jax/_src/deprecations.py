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

# 文件职责：为 JAX 内部提供统一的弃用（deprecation）登记与触发机制。
# 一方面用 `deprecation_getattr` 生成模块级 `__getattr__`，让被弃用的模块属性在
# 首次访问时发出 DeprecationWarning 或抛出 AttributeError；另一方面用
# `register`/`accelerate` 登记非导入类弃用（例如函数参数的弃用），并让已加速的
# 弃用直接报错而非仅发出警告。各 `jax._src` 子模块在更名、迁移对外接口时复用这些机制。

from dataclasses import dataclass
import functools
from types import ModuleType
import warnings

# 模块级 `__getattr__` 工厂：当使用了已弃用的名称时发出警告。
#
# 用法示例：
# from jax._src.interpreters.pxla import (
#   Mesh as _deprecated_Mesh,
# )
#
# _deprecations = {
#   # 添加于 2023 年 2 月 8 日：
#   "Mesh": (
#     "jax.interpreters.pxla.Mesh is deprecated. Use jax.sharding.Mesh.",
#     _deprecated_Mesh,
#   ),
# }
#
# from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
# __getattr__ = _deprecation_getattr(__name__, _deprecations)
# del _deprecation_getattr

# 注意：诸如 Pyrefly 之类的类型检查器并不知道这些已弃用的名称。
# 如果希望某个已弃用的名称能被类型检查器识别，
# 可加入：
# import typing
# if typing.TYPE_CHECKING:
#   from jax._src.interpreters.pxla import (
#     Mesh as Mesh,
#   )
# del typing
def deprecation_getattr(module, deprecations):
  @functools.cache
  def getattr(name):
    if name in deprecations:
      message, fn = deprecations[name]
      if fn is None:  # 这个弃用是否已加速？
        raise AttributeError(message)
      warnings.warn(message, DeprecationWarning, stacklevel=2)
      return fn
    raise AttributeError(f"module {module!r} has no attribute {name!r}")

  return getattr


def accelerate_getattr_deprecation(module: ModuleType, *names: str) -> None:
  """加速某个模块级属性的弃用。

  在访问该属性时抛出 AttributeError，而不是发出 DeprecationWarning。
  用于 Google 内部代码以实现更快的弃用。
  """
  for name in names:
    message, _ = module._deprecations[name]
    module._deprecations[name] = (message, None)


def is_accelerated_attribute(module: ModuleType, name: str) -> bool:
  """如果给定名称已加速则返回 true。

  如果 name 不是 module 中已弃用的属性，则报错。
  """
  return module._deprecations[name][1] is None

# 下面是一套独立的机制，用于登记并加速那些并非导入的弃用
# （例如函数参数的弃用）。
# 它把全局唯一的字符串 ID 映射到 DeprecationState，
# 由后者记录该弃用是否已加速。
# 其意图是：未加速的弃用会发出警告，
# 而已加速的弃用则会报错。

@dataclass(slots=True)
class DeprecationState:
  accelerated: bool = False

_registered_deprecations: dict[str, DeprecationState] = {}


def register(deprecation_id: str) -> None:
  _registered_deprecations[deprecation_id] = DeprecationState()


def unregister(deprecation_id: str) -> None:
  if deprecation_id not in _registered_deprecations:
    raise ValueError(f"{deprecation_id=!r} not registered.")
  _registered_deprecations.pop(deprecation_id)


def accelerate(deprecation_id: str) -> None:
  if deprecation_id not in _registered_deprecations:
    raise ValueError(f"{deprecation_id=!r} not registered.")
  _registered_deprecations[deprecation_id].accelerated = True


def is_accelerated(deprecation_id: str) -> bool:
  if deprecation_id not in _registered_deprecations:
    raise ValueError(f"{deprecation_id=!r} not registered.")
  return _registered_deprecations[deprecation_id].accelerated


def warn(deprecation_id: str, message: str, stacklevel: int, *,
         error_class: type[Exception] = ValueError) -> None:
  """就某个弃用发出警告；若该弃用已加速则报错。"""
  if is_accelerated(deprecation_id):
    assert issubclass(error_class, Exception)
    raise error_class(message)
  else:
    warnings.warn(message, category=DeprecationWarning,
                  stacklevel=stacklevel + 1)


# 登记若干弃用：在此处登记是为了确保在调用 `accelerate` 和 `is_acelerated`
# 时它们总是已经注册。
register('jax-array-numpy-dtype')
register('jax-nn-one-hot-float-input')
register('jax-numpy-astype-complex-to-real')
register('jax-array-positional-args')
register('jax-pallas-call-mgpu')
register('jax-pallas-mgpu-shapes-types')
register('jax-numpy-cross-2d-input')
register('jax-pallas-mgpu-load-idx')
register('jax-pallas-triton')
