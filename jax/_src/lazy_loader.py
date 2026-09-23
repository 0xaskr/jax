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

# 文件职责：为 JAX 的包提供惰性加载子模块的通用工具。
# `attach` 在包的 `__init__.py` 中被调用，用按需导入的 `__getattr__`、`__dir__`
# 与 `__all__` 替换包中原有的同名定义，使子模块只在首次访问时才真正被导入。
# 导入完成后会把结果写回模块的全局名字，后续访问不再走 `__getattr__`。
# 使用方包括 `jax.scipy`、`jax._src.lib.mlir.dialects` 等希望加快导入速度的包。

"""惰性加载器类。"""

from collections.abc import Callable, Sequence
import importlib
import sys
from typing import Any


def attach(package_name: str, submodules: Sequence[str]) -> tuple[
    Callable[[str], Any],
    Callable[[], list[str]],
    list[str],
]:
  """惰性加载某个包的子模块。

  Returns:
    一个元组，包含 ``__getattr__``、``__dir__`` 函数和 ``__all__`` ——
    ``__all__`` 是可用全局名字的列表，可用来替换包中
    对应的定义。

  Raises:
    RuntimeError: 若无法确定调用者的 ``__name__``。
  """
  owner_name = sys._getframe(1).f_globals.get("__name__")
  if owner_name is None:
    raise RuntimeError("Cannot determine the ``__name__`` of the caller.")

  __all__ = list(submodules)

  def __getattr__(name: str) -> Any:
    if name in submodules:
      value = importlib.import_module(f"{package_name}.{name}")
      # 更新模块级全局名字，避免为这个 ``name``
      # 再次调用 ``__getattr__``。
      assert owner_name is not None  # pyrefly#40
      setattr(sys.modules[owner_name], name, value)
      return value
    raise AttributeError(f"module '{package_name}' has no attribute '{name}'")

  def __dir__() -> list[str]:
    return __all__

  return __getattr__, __dir__, __all__
