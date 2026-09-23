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

# 文件职责：实现 `EArray`，一种可承载扩展数据类型（extended dtype）的数组类型。
# 它在 `basearray.Array` 之上包一层逻辑抽象值（aval），把 `shape`、`dtype`
# 等属性转发给 aval，把设备、缓冲区等物理属性转发给底层数据。
# 同时注册 `EArray` 的分片参数处理器、pytype 到 aval 的映射以及 pytree 节点，
# 使扩展数据类型能像普通数组一样参与分派、分片与 pytree 扁平化。

from __future__ import annotations

import math

from jax._src import basearray
from jax._src import core
from jax._src import dtypes
from jax._src import tree_util
from jax._src import sharding_impls
from jax._src.interpreters import pxla
from jax._src.util import safe_zip, safe_map

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

# `EArray` 是一种可以承载扩展数据类型的 `Array`。
class EArray(basearray.Array):
  __slots__ = ['_aval', '_data']
  __hash__ = None
  __array_priority__ = 100

  def __init__(self, aval, data):
    self._aval = aval
    self._data = data

  @property
  def aval(self):
    return self._aval

  def block_until_ready(self):
    _ = self._data.block_until_ready()
    return self

  def copy_to_host_async(self):
    self._data.copy_to_host_async()

  def copy(self):
    return EArray(self.aval, self._data.copy())

  def __repr__(self):
    return 'E' + repr(self._data)

  def __iter__(self):
    if self.ndim == 0: raise TypeError('iteration over a 0-d array')
    raise NotImplementedError

  # 转发给 aval
  shape = property(lambda self: self.aval.shape)
  dtype = property(lambda self: self.aval.dtype)

  # 由形状和数据类型计算得到
  ndim = property(lambda self: len(self.aval.shape))
  size = property(lambda self: math.prod(self.aval.shape))
  itemsize = property(lambda self: self.aval.dtype.itemsize)
  def __len__(self):
    if self.ndim == 0: raise TypeError('len() of unsized object')
    return self.shape[0]

  # 转发给 self._data
  devices = property(lambda self: self._data.devices)  # pyrefly: ignore[bad-override]
  _committed = property(lambda self: self._data._committed)
  is_fully_addressable = property(lambda self: self._data.is_fully_addressable)
  is_fully_replicated = property(lambda self: self._data.is_fully_replicated)
  delete = property(lambda self: self._data.delete)  # pyrefly: ignore[bad-override]
  is_deleted = property(lambda self: self._data.is_deleted)  # pyrefly: ignore[bad-override]
  on_device_size_in_bytes = property(lambda self: self._data.on_device_size_in_bytes)  # pyrefly: ignore[bad-override]
  unsafe_buffer_pointer = property(lambda self: self._data.unsafe_buffer_pointer)  # pyrefly: ignore[bad-override]

  # 交由扩展数据类型规则处理
  @property
  def sharding(self):
    phys_sharding = self._data.sharding
    return sharding_impls.logical_sharding(self.shape, self.dtype, phys_sharding)

  @property
  def committed(self):
    return self._data.committed

  @property
  def device(self):
    if len(self.sharding.device_set) == 1:
      return self._data.device
    return self.sharding

  # TODO(mattjj): 以下尚未实现，还需要 `ArrayImpl` 中的更多方法

  def addressable_data(self, index: int) -> EArray:
    raise NotImplementedError

  @property
  def addressable_shards(self):
    raise NotImplementedError

  @property
  def global_shards(self):
    raise NotImplementedError

# TODO(mattjj): _set_array_base_attributes

def _earray_shard_arg_handler(xs, shardings, layouts, copy_semantics):
  arrs = [x._data for x in xs]
  phys_shardings = [sharding_impls.physical_sharding(x.aval, sharding)
                    for x, sharding in zip(xs, shardings)]
  # TODO(yashkatariya): `layouts` 应当被转换为物理布局。
  return pxla.shard_args(phys_shardings, layouts, copy_semantics, arrs)
pxla.shard_arg_handlers[EArray] = _earray_shard_arg_handler

core.pytype_aval_mappings[EArray] = lambda x: x.aval
dtypes.register_canonicalize_value_handler(EArray, None)
tree_util.dispatch_registry.register_node(
    EArray, lambda x: ((x._data,), x.aval), lambda a, xs: EArray(a, xs[0]))
