# Copyright 2024 The JAX Authors.
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

from jax._src import config
from jax._src.lib import _jax

# 文件职责：承载 XLA 元数据（metadata）的不可变值对象与增删工具，
# 供 `jax._src.xla_metadata` 提供的 `xla_metadata` 上下文管理器使用。
# `XlaMetadata` 把元数据字典包装成可哈希的值（缓存命中的键之一），
# `update_metadata` 在当前元数据之上叠加更新，`current_xla_metadata` 读取
# 当前配置环境中的元数据；这些元数据最终随计算一起传给 XLA 作为编译提示。

config_ext = _jax.config


class XlaMetadata:
  __slots__ = ['val', 'hash']

  val: dict[str, Any]

  def __init__(self, val):
    self.val = val
    self.hash = hash(tuple(sorted(self.val.items())))

  def __hash__(self):
    return self.hash

  def __eq__(self, other):
    return other is not None and self.val == other.val


def filter_nones(d: dict) -> dict:
  return {k: v for k, v in d.items() if v is not None}


def update_metadata(a, b: dict[str, Any]):
  if not b:
    return a
  if a is None or a is config_ext.unset:
    val = {}
  else:
    val = a.val.copy()
  val.update(b)
  return XlaMetadata(filter_nones(val))


def current_xla_metadata() -> dict[str, Any] | None:
  metadata = config.xla_metadata_context_manager.value
  return None if metadata is None else metadata.val
