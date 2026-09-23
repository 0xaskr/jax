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

from __future__ import annotations

import abc
import pathlib

from jax._src import util

# 文件职责：定义 JAX 编译缓存（compilation cache）的抽象接口。
# `CacheInterface` 声明了按字符串键读写编译产物的最小组约：`get` 取回缓存项、
# `put` 写入缓存项，并约定实现方以 `_path` 暴露缓存所在路径。
# 具体的持久化/淘汰策略由后端实现类提供，JAX 的编译流程只依赖这个接口，
# 从而可以在磁盘缓存、内存缓存等不同实现之间切换。

class CacheInterface(util.StrictABC):
  _path: pathlib.Path

  @abc.abstractmethod
  def get(self, key: str):
    pass

  @abc.abstractmethod
  def put(self, key: str, value: bytes):
    pass
