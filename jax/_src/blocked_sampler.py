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
# 文件职责：为“块不变（block-invariant）”的随机数生成提供底层工具。
# 目标是无论相邻块如何划分，只要全局 key 与块索引相同，同一块内生成的随机样本就完全一致，
# 这是分片（sharding）与流水线并行下随机数可复现的关键。
# `blocked_fold_in` 依据全局 key、块大小与 tile 大小算出该块所需的 key 网格；
# `sample_block` 再按同一套 tile 划分逐块采样并沿各轴拼接；
# tile 是生成样本的最小单位，应选为所有需保持不变性的块大小的公约数。
from collections.abc import Sequence
from typing import Any, Protocol

from jax._src import numpy as jnp
from jax._src import random
from jax._src.typing import Array, ArrayLike

NdKeyList = Any
Shape = random.Shape

class SampleFn(Protocol):
  def __call__(self, key: ArrayLike, *args, shape: Shape,
               **kwargs) -> Array:
    ...


def _compute_tile_index(block_index: Sequence[ArrayLike],
                        block_size_in_tiles: Shape,
                        total_size_in_tiles: Shape,
                        tile_index_in_block: Sequence[ArrayLike]) -> ArrayLike:
  ndims = len(block_index)
  dim_size: ArrayLike = 1
  total_idx: ArrayLike = 0
  for i in range(ndims-1, -1, -1):
    dim_idx = tile_index_in_block[i] + block_index[i] * block_size_in_tiles[i]
    total_idx += dim_idx * dim_size
    dim_size *= total_size_in_tiles[i]
  return total_idx


def blocked_fold_in(
  global_key: ArrayLike,
  total_size: Shape,
  block_size: Shape,
  tile_size: Shape,
  block_index: Sequence[ArrayLike],
  ) -> NdKeyList:
  """计算用于块不变采样的 key 网格。

  假设我们想构造一个 16x512 的随机数数组，使用 16x128 和 16x256 两种块大小。
  我们可以选取 tile 大小为 8x128（它同时整除 16x128 与 16x256），
  并把整个数组按 tile 划分为：
  ---------------------------------
  | 8x128 | 8x128 | 8x128 | 8x128 |
  ---------------------------------
  | 8x128 | 8x128 | 8x128 | 8x128 |
  ---------------------------------

  我们为每个 tile 生成一个 key：
    tile_key = fold_in(global_key, tile_idx)

  其中 tile_idx 是每个元素按行优先展开后的索引：
  -----------------
  | 0 | 1 | 2 | 3 |
  -----------------
  | 4 | 5 | 6 | 7 |
  -----------------

  随后我们计算并返回采样组成当前块（由 `block_index` 指定）的那些 tile 所需的 key。
  对于 16x256 的块大小，每个块需要 4 个（2x2）tile key：
  ---------------
  | 0, 1 | 2, 3 |
  | 4, 5 | 6, 7 |
  ---------------
  因此我们为每个块返回 2x2 的 key 网格（共 2 个块）。

  对于 16x128 的块大小，每个块需要 2 个（2x1）tile key：
  -----------------
  | 0 | 1 | 2 | 3 |
  | 4 | 5 | 6 | 7 |
  -----------------
  因此我们为每个块返回 2x1 的 key 网格（共 4 个块）。

  Args:
    global_key: 在所有块之间共享的全局 key。
    total_size: 正在生成的数组的形状。
    block_size: 单个块的形状。
    tile_size: `tile` 的形状，tile 是生成样本的最小单位。
      应将其选为所有需要保持不变性的块大小的公约数。
    block_index: 指明为哪个块生成 key 的索引。

  Returns:
    采样由 `block_index` 指定的块所对应的 tile 所需的 key，
    是一个 N 维嵌套列表。
  """
  block_size_in_tiles = tuple(
      _shape // _element for _shape, _element in zip(block_size, tile_size)
  )

  # 向上取整，确保每个 tile 都被编号。
  total_size_in_tiles = tuple(
      (_shape + _element - 1) // _element
        for _shape, _element in zip(total_size, tile_size)
  )

  def _keygen_loop(axis, prefix):
    if axis == len(block_size_in_tiles):
      subtile_key = random.fold_in(
          global_key, _compute_tile_index(
              block_index, block_size_in_tiles, total_size_in_tiles, prefix))
      return subtile_key
    else:
      keys = []
      for i in range(block_size_in_tiles[axis]):
        keys.append(_keygen_loop(axis+1, prefix+(i,)))
      return keys
  return _keygen_loop(0, ())


def sample_block(
    sampler_fn: SampleFn,
    keys: NdKeyList,
    block_size: Shape,
    tile_size: Shape,
    *args,
    **kwargs
  ) -> Array:
  """为单个块抽取随机样本。

  本函数旨在与 `blocked_fold_in` 配合使用：
  ```
  key_list = blocked_fold_in(global_key, total_size, block_size, tile_size,
                             block_index)
  samples = sample_block(jax.random.uniform, key_list, block_size, tile_size)
  ```

  Args:
    sampler_fn: 随机采样函数，例如 jax.random.uniform。
    keys: 由 `blocked_fold_in` 生成的 key 网格。
    block_size: 单个块的形状。
    tile_size: `tile` 的形状，tile 是生成样本的最小单位。
      应将其选为所有需要保持不变性的块大小的公约数。
    args: 传给 sampler_fn 的可变位置参数。
    kwargs: 传给 sampler_fn 的关键字参数。

  Returns:
    使用 sampler_fn 抽取的随机样本数组。
  """
  size_in_tiles = tuple(
      _shape // _element for _shape, _element in zip(block_size, tile_size))
  def _nested_index(arr: Array, idx: Sequence[int]) -> Array:
    if len(idx) == 1:
      return arr[idx[0]]
    return _nested_index(arr[idx[0]], idx[1:])

  def _sample_loop(axis: int, prefix: tuple[int, ...]) -> Array:
    if axis == len(size_in_tiles):
      return sampler_fn(_nested_index(keys, prefix), *args,
                        shape=tile_size, **kwargs)
    else:
      samples = []
      for i in range(size_in_tiles[axis]):
        samples.append(_sample_loop(axis+1, prefix+(i,)))
      return jnp.concatenate(samples, axis=axis)
  return _sample_loop(0, ())
