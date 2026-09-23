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
# See the License for the

# 文件职责：定义 `HashableArray`，一个把 NumPy 数组包装成可哈希、可比较对象的轻量包装器。
# NumPy 数组本身不可哈希，且 `==` 返回的是逐元素布尔数组，因此无法直接用作
# jaxpr 参数或缓存键；本模块保存一份只读副本，并以形状、数据类型和原始字节定义哈希，
# 以形状、数据类型和逐元素相等定义判等。它被 `lax.py`（原语参数）、`ffi.py`（FFI 调用
# 绑定参数）、`experimental/key_reuse` 以及 MLIR 解释器的属性处理器用来支持缓存与序列化。

import numpy as np


class HashableArray:
  __slots__ = ["val"]
  val: np.ndarray

  def __init__(self, val):
    self.val = np.array(val, copy=True)
    self.val.setflags(write=False)

  def __repr__(self):
    return f"HashableArray({self.val!r})"

  def __str__(self):
    return f"HashableArray({self.val})"

  def __hash__(self):
    return hash((self.val.shape, self.val.dtype, self.val.tobytes()))

  def __eq__(self, other):
    if not isinstance(other, HashableArray):
      return False
    if self.val.shape != other.val.shape or self.val.dtype != other.val.dtype:
      return False
    return np.array_equal(self.val, other.val)
