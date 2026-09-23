# Copyright 2018 The JAX Authors.
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
# 文件职责：提供把数组组成的 pytree 展平成一维数组、并能还原回原结构的工具。
# 对外主要暴露 `ravel_pytree` 与 `unravel_pytree`：按叶子顺序拼接所有数组叶子，
# 并返回一个可调用对象，用于把相同长度的一维向量还原成结构相同的 pytree。
# 当各叶子 dtype 不一致时会先做类型提升，并在还原时逐叶子转回原 dtype，
# 因此展平结果对 dtype 是多态的；各 dtype 一致时则跳过转换以避免额外开销。

from collections.abc import Iterable
import numpy as np
from typing import Any, TypeAlias
from collections.abc import Callable

from jax._src.lax import lax
from jax._src import dtypes
from jax._src.tree_util import tree_flatten, tree_unflatten, PyTreeDef, Leaf
from jax._src.util import safe_zip as zip, unzip2, HashablePartial
from jax._src.typing import Array

Sizes: TypeAlias = tuple[int, ...]
Shapes: TypeAlias = tuple[tuple[int, ...], ...]


def ravel_pytree(pytree: Any) -> tuple[Array, Callable[[Array], Any]]:
  """把由数组组成的 pytree 展平（ravel）成一维数组。

  Args:
    pytree: 要展平的、由数组与标量组成的 pytree。

  Returns:
    一个二元组。第一个元素是表示展平并拼接后的叶子值的一维数组，其 dtype 由各
    叶子值的 dtype 提升决定。
    第二个元素是可调用对象，用于把相同长度的一维向量还原成与输入 ``pytree``
    结构相同的 pytree。
    如果输入 pytree 为空（即没有叶子），则按约定在输出的第一个分量返回一个
    dtype 为 float32 的一维空数组。

  关于 dtype 提升的细节，见
  https://docs.jax.dev/en/latest/type_promotion.html.

  """
  leaves, treedef = tree_flatten(pytree)
  flat, unravel_list = _ravel_list(leaves)
  return flat, HashablePartial(unravel_pytree, treedef, unravel_list)


def unravel_pytree(
  treedef: PyTreeDef,
  unravel_list: Callable[[Array], Iterable[Leaf]],
  flat: Array,
) -> Any:
  return tree_unflatten(treedef, unravel_list(flat))


def _ravel_list(lst: list[Any], /) -> tuple[Array, Callable[[Array], list[Any]]]:
  if not lst:
    return lax.full([0], 0, "float32"), lambda _: []
  from_dtypes = tuple(dtypes.dtype(l) for l in lst)
  to_dtype = dtypes.result_type(*from_dtypes)
  sizes, shapes = unzip2((np.size(x), np.shape(x)) for x in lst)

  if all(dt == to_dtype for dt in from_dtypes):
    # 跳过任何 dtype 转换，从而得到 dtype 多态的 `unravel`。
    # 参见 https://github.com/jax-ml/jax/issues/7809。
    del from_dtypes, to_dtype
    ravel = lambda e: lax.reshape(e, (np.size(e),))
    raveled = lax.concatenate([ravel(e) for e in lst], dimension=0)
    return raveled, HashablePartial(_unravel_list_single_dtype, sizes, shapes)

  # 当存在多个不同的输入 dtype 时，我们执行类型转换，
  # 并生成一个针对特定 dtype 的 unravel 函数。
  ravel = lambda e: lax.convert_element_type(e, to_dtype).ravel()
  raveled = lax.concatenate([ravel(e) for e in lst], dimension=0)
  unrav = HashablePartial(_unravel_list, sizes, shapes, from_dtypes, to_dtype)
  return raveled, unrav


def _unravel_list_single_dtype(sizes: Sizes, shapes: Shapes, arr: Array) -> list[Array]:
  chunks = lax.split(arr, sizes)
  return [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]


def _unravel_list(
  sizes: Sizes,
  shapes: Shapes,
  from_dtypes: tuple[np.dtype, ...],
  to_dtype: np.dtype,
  arr: Array,
) -> list[Array]:
  arr_dtype = dtypes.dtype(arr)
  if arr_dtype != to_dtype:
    raise TypeError(
      f"unravel function given array of dtype {arr_dtype}, "
      f"but expected dtype {to_dtype}"
    )
  chunks = lax.split(arr, sizes)
  return [
    lax._convert_element_type(
      chunk.reshape(shape), dtype, warn_on_complex_to_real_cast=False
    )
    for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)
  ]
