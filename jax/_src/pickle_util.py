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

# 文件职责：为 JAX 提供 pickle 序列化的轻量封装，主要服务于主机回调的序列化。
# 优先使用 `cloudpickle`，以便序列化 lambda、闭包等在标准 `pickle` 下无法处理的函数；
# 若未安装 `cloudpickle`，调用 `dumps`/`loads` 会抛出 `ModuleNotFoundError`。
# `dumps` 中自定义的 `Pickler` 修补了 dataclass 内部单例对象的序列化缺陷，
# 并为这两个入口加上 `profiler.annotate_function`，便于在性能剖析中分别计时。

import dataclasses
import functools
import io
from typing import Any

try:
  import cloudpickle
except ImportError:
  cloudpickle = None

from jax._src import profiler


@functools.partial(profiler.annotate_function, name='pickle_util.dumps')
def dumps(obj: Any) -> bytes:
  """参见 `pickle.dumps`。用于在 jaxlib 中序列化主机回调。"""
  if cloudpickle is None:
    raise ModuleNotFoundError('No module named "cloudpickle"')

  class Pickler(cloudpickle.CloudPickler):
    """定制 cloudpickle 的行为。"""

    # 复制一份，避免影响其他用户对 cloudpickle 的使用。
    dispatch_table = cloudpickle.CloudPickler.dispatch_table.copy()  # pyrefly: ignore[missing-attribute]

    # 修复 dataclass 内部单例对象的序列化问题。
    # 缺陷：https://github.com/cloudpipe/cloudpickle/issues/386
    dispatch_table[dataclasses._FIELD_BASE] = lambda x: f'{x.name}'  # pyrefly: ignore[missing-attribute]
    dispatch_table[dataclasses._MISSING_TYPE] = lambda _: 'MISSING'
    dispatch_table[dataclasses._HAS_DEFAULT_FACTORY_CLASS] = (  # pyrefly: ignore[missing-attribute]
        lambda _: '_HAS_DEFAULT_FACTORY'
    )
    if hasattr(dataclasses, '_KW_ONLY_TYPE'):
      dispatch_table[dataclasses._KW_ONLY_TYPE] = (
          lambda _: '_KW_ONLY_TYPE'
      )  # 在 Python 3.10 中加入。

  with io.BytesIO() as file:
    Pickler(file).dump(obj)
    return file.getvalue()


@functools.partial(profiler.annotate_function, name='pickle_util.loads')
def loads(data: bytes) -> Any:
  """参见 `pickle.loads`。"""
  if cloudpickle is None:
    raise ModuleNotFoundError('No module named "cloudpickle"')

  return cloudpickle.loads(data)
