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

from typing import Any
from jax._src import core

# 文件职责：`jax.ref` 公开 API 的实现侧外壳，提供可变数组引用（ref）的构造入口。
# `new_ref` 把调用转发给 `jax._src.core.new_ref`，并补齐内存空间、变更语义与
# 固定缓冲区等选项，用于在 JAX 中表达就地更新的可变缓冲区。
# 该模块只做参数转发与文档承载，真正的引用语义、效果（effect）追踪与
# 重物化规则都在核心层实现。

def new_ref(
    init_val: Any, *, memory_space: Any = None, kind: str | None = None,
    pin: bool = False
) -> core.Ref:
  """创建一个以 ``init_val`` 为初始值的可变数组引用。

  更多讨论参见 `Ref guide`_。

  Args:
    init_val: 一个 :class:`jax.Array`，表示该缓冲区的初始状态。
    memory_space: 该引用的可选内存空间属性。
    kind: 一个可选字符串，指明在重物化下的变更语义。目前仅支持
      ``'no_grad_no_remat'`` 或 ``None``。
    pin: 是否在 HLO 中把该引用降级为固定（pinned）缓冲区。

  Returns:
    一个 :class:`jax.ref.Ref`，其中包含指向可变缓冲区的引用。

  .. _Ref guide: https://docs.jax.dev/en/latest/array_refs.html
  """
  return core.new_ref(init_val, memory_space=memory_space, kind=kind, pin=pin)
