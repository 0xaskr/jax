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

# 文件职责：定义不可变且可哈希的映射类型 `FrozenDict`，供 JAX 内部承载“冻结”的配置与元数据。
# 它继承 `collections.abc.Mapping`，构造时把传入映射的内容拷贝进内部普通字典，此后不再提供任何修改接口。
# 哈希值由键值对构成的 `frozenset` 计算，因此要求所有值自身可哈希；正因如此，`FrozenDict` 可充当缓存键、
# 放入集合，或用作需要可静态比较的参数（如 `AxisEnv` 的轴尺寸、Pallas 调用的 metadata、FFI 的哈希化关键字参数）。
# 相等性仅在对方也是 `FrozenDict` 且内部内容一致时成立，从而保证其作为键的语义稳定。

from collections.abc import Iterator, Mapping
from typing import Any


class FrozenDict[K, V](Mapping[K, V]):

  def __init__(self, d: Mapping[K, V]):
    self._d = dict(d.items())

  def __repr__(self) -> str:
    return f"FrozenDict({self._d!r})"

  def __str__(self) -> str:
    return f"FrozenDict({self._d})"

  def __getitem__(self, key: K) -> V:
    return self._d[key]

  def __hash__(self) -> int:
    # 这里假定所有值都是可哈希的。
    return hash(frozenset(self._d.items()))

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, FrozenDict):
      return False
    return self._d == other._d

  def __iter__(self) -> Iterator[K]:
    return iter(self._d)

  def __len__(self) -> int:
    return len(self._d)
