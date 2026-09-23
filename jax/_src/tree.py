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
# 文件职责：面向用户的 `jax.tree` 命名空间，提供 pytree（嵌套数据结构）的
# 遍历与操作 API，是对 `jax.tree_util` 中底层实现的轻量公开包装。
# 关键概念：pytree 由叶子与 `PyTreeDef`（树结构）描述；可选回调 `is_leaf`
# 控制展平时哪些对象被整体当作叶子；`*_with_path` 变体额外返回每个叶子的
# 键路径（`KeyPath`）；`transpose` 在（外层, 内层）结构间转置，`broadcast` 把
# 前缀树广播到完整结构；末尾的 `static` 把 dataclass 字段标记为 pytree 静态属性。
from __future__ import annotations

from collections.abc import Callable, Iterable
import dataclasses
from typing import Any, TypeVar, TYPE_CHECKING

from jax._src import tree_util

T = TypeVar("T")


def all(tree: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> bool:
  """对树的叶子调用 all()。

  Args:
    tree: 要对其求值的 pytree
    is_leaf : 一个可选指定的函数，会在每个展平步骤被调用。它应返回一个布尔值，
      指示展平是应遍历当前对象，还是应立即停止并把整个子树视为一个叶子。

  Returns:
    result: 布尔值 True 或 False

  Examples:
    >>> import jax
    >>> jax.tree.all([True, {'a': True, 'b': (True, True)}])
    True
    >>> jax.tree.all([False, (True, False)])
    False

  See Also:
    - :func:`jax.tree.reduce`
    - :func:`jax.tree.leaves`
  """
  return tree_util.tree_all(tree, is_leaf=is_leaf)


def flatten(tree: Any,
            is_leaf: Callable[[Any], bool] | None = None
            ) -> tuple[list[tree_util.Leaf], tree_util.PyTreeDef]:
  """展平一个 pytree。

  展平顺序（即输出列表中元素的顺序）是确定性的，
  对应于从左到右的深度优先树遍历。

  Args:
    tree: 要展平的 pytree。
    is_leaf: 一个可选指定的函数，会在每个展平步骤被调用。它应返回一个布尔值：
      true 表示停止遍历并把整个子树视为一个叶子，false 表示展平应继续
      遍历当前对象。

  Returns:
    一个二元组，第一个元素是叶子值列表，第二个元素是表示展平后树结构的
    treedef。

  Examples:
    >>> import jax
    >>> vals, treedef = jax.tree.flatten([1, (2, 3), [4, 5]])
    >>> vals
    [1, 2, 3, 4, 5]
    >>> treedef
    PyTreeDef([*, (*, *), [*, *]])

  See Also:
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.structure`
    - :func:`jax.tree.unflatten`
  """
  return tree_util.tree_flatten(tree, is_leaf)


def leaves(tree: Any,
           is_leaf: Callable[[Any], bool] | None = None
           ) -> list[tree_util.Leaf]:
  """获取一个 pytree 的叶子。

  Args:
    tree: 要获取叶子的 pytree
    is_leaf : 一个可选指定的函数，会在每个展平步骤被调用。它应返回一个布尔值，
      指示展平是应遍历当前对象，还是应立即停止并把整个子树视为一个叶子。

  Returns:
    leaves: 树叶子组成的列表。

  Examples:
    >>> import jax
    >>> jax.tree.leaves([1, (2, 3), [4, 5]])
    [1, 2, 3, 4, 5]

  See Also:
    - :func:`jax.tree.flatten`
    - :func:`jax.tree.structure`
    - :func:`jax.tree.unflatten`
  """
  return tree_util.tree_leaves(tree, is_leaf)


def map(f: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], bool] | None = None) -> Any:
  """把一个多输入函数映射到各 pytree 参数上，产生一个新的 pytree。

  Args:
    f: 接受 ``1 + len(rest)`` 个参数的函数，会被应用到各 pytree
      对应的叶子上。
    tree: 要被映射的 pytree，它的每个叶子提供传给 ``f`` 的第一个
      位置参数。
    rest: 一组 pytree，其中每个都与 ``tree`` 具有相同结构，或以 ``tree``
      为前缀。
    is_leaf: 一个可选指定的函数，会在每个展平步骤被调用。它应返回一个布尔值，
      指示展平是应遍历当前对象，还是应立即停止并把整个子树视为一个叶子。

  Returns:
    一个新的 pytree，结构与 ``tree`` 相同，但每个叶子上的值由 ``f(x, *xs)``
    给出，其中 ``x`` 是 ``tree`` 中对应叶子上的值，``xs`` 是 ``rest``
    中对应节点上的值组成的元组。

  Examples:

    >>> import jax
    >>> jax.tree.map(lambda x: x + 1, {"x": 7, "y": 42})
    {'x': 8, 'y': 43}

    如果传入多个输入，树的结构取自第一个输入；后续输入只需以 ``tree``
    为前缀：

    >>> jax.tree.map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]

  See Also:
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.reduce`
  """
  return tree_util.tree_map(f, tree, *rest, is_leaf=is_leaf)


def reduce(function: Callable[[T, Any], T],
           tree: Any,
           initializer: T | tree_util.Unspecified = tree_util.Unspecified(),
           is_leaf: Callable[[Any], bool] | None = None) -> T:
  """对树的叶子调用 reduce()。

  Args:
    function: 归约函数
    tree: 要归约的 pytree
    initializer: 可选的初始值
    is_leaf : 一个可选指定的函数，会在每个展平步骤被调用。它应返回一个布尔值，
      指示展平是应遍历当前对象，还是应立即停止并把整个子树视为一个叶子。

  Returns:
    result: 归约后的值。

  Examples:
    >>> import jax
    >>> import operator
    >>> jax.tree.reduce(operator.add, [1, (2, 3), [4, 5, 6]])
    21

  Notes:
    **提示**：可以先通过 :func:`jax.tree.map` 把想要排除的叶子映射为
    ``None``，这样它们之后就不会被计为叶子，从而被排除在归约之外。

  See Also:
    - :func:`jax.tree.reduce_associative`
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.map`
  """
  return tree_util.tree_reduce(function, tree, initializer, is_leaf=is_leaf)


def reduce_associative(
    operation: Callable[[T, T], T],
    tree: Any,
    *,
    identity: T | tree_util.Unspecified = tree_util.Unspecified(),
    is_leaf: Callable[[Any], bool] | None = None,
) -> T:
  """使用满足结合律的二元运算对一个 pytree 执行归约。

  本函数利用该运算满足结合律这一性质，以并行方式（对数深度）
  完成归约。

  Args:
    operation: 满足结合律的二元运算
    tree: 要归约的 pytree
    identity: 该结合性二元运算的单位元。
      它仅在树为空时使用，其他情况下是可选的。
    is_leaf: 一个可选指定的函数，会在每个展平步骤被调用。它应返回一个布尔值，
      指示展平是应遍历当前对象，还是应立即停止并把整个子树视为一个叶子。

  Returns:
    result: 归约后的值

  Examples:
    >>> import jax
    >>> import operator
    >>> jax.tree.reduce_associative(operator.add, [1, (2, 3), [4, 5, 6]])
    21

  Notes:
    **提示**：可以先通过 :func:`jax.tree.map` 把想要排除的叶子映射为
    ``None``，这样它们之后就不会被计为叶子，从而被排除在归约之外。

  See Also:
    - :func:`jax.tree.reduce`
  """
  return tree_util.tree_reduce_associative(
      operation,
      tree,
      identity=identity,
      is_leaf=is_leaf,
  )


def structure(tree: Any,
              is_leaf: None | (Callable[[Any], bool]) = None) -> tree_util.PyTreeDef:
  """获取一个 pytree 的 treedef。

  Args:
    tree: 要获取叶子的 pytree
    is_leaf : 一个可选指定的函数，会在每个展平步骤被调用。它应返回一个布尔值，
      指示展平是应遍历当前对象，还是应立即停止并把整个子树视为一个叶子。

  Returns:
    pytreedef: 表示该树结构的 PyTreeDef。

  Examples:
    >>> import jax
    >>> jax.tree.structure([1, (2, 3), [4, 5]])
    PyTreeDef([*, (*, *), [*, *]])

  See Also:
    - :func:`jax.tree.flatten`
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.unflatten`
  """
  return tree_util.tree_structure(tree, is_leaf)


def transpose(outer_treedef: tree_util.PyTreeDef,
              inner_treedef: tree_util.PyTreeDef | None,
              pytree_to_transpose: Any) -> Any:
  """把树结构为 (outer, inner) 的树变换为结构为 (inner, outer) 的树。

  Args:
    outer_treedef: 表示外层树的 PyTreeDef。
    inner_treedef: 表示内层树的 PyTreeDef。
      若为 None，则会根据 outer_treedef 和 pytree_to_transpose
      的结构推断得到。
    pytree_to_transpose: 要转置的 pytree。

  Returns:
    transposed_pytree: 转置后的 pytree。

  Examples:
    >>> import jax
    >>> tree = [(1, 2, 3), (4, 5, 6)]
    >>> inner_structure = jax.tree.structure(('*', '*', '*'))
    >>> outer_structure = jax.tree.structure(['*', '*'])
    >>> jax.tree.transpose(outer_structure, inner_structure, tree)
    ([1, 4], [2, 5], [3, 6])

    推断内层结构：

    >>> jax.tree.transpose(outer_structure, None, tree)
    ([1, 4], [2, 5], [3, 6])
  """
  return tree_util.tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose)


def unflatten(treedef: tree_util.PyTreeDef,
              leaves: Iterable[tree_util.Leaf]) -> Any:
  """根据 treedef 和叶子重建一个 pytree。

  :func:`tree_flatten` 的逆操作。

  Args:
    treedef: 要用于重建的 treedef
    leaves: 用于重建的叶子的可迭代对象。该可迭代对象必须
      与 treedef 的叶子相匹配。

  Returns:
    重建得到的 pytree，其中 ``leaves`` 被放置在 ``treedef`` 所描述的
    结构之中。

  Examples:
    >>> import jax
    >>> vals, treedef = jax.tree.flatten([1, (2, 3), [4, 5]])
    >>> newvals = [100, 200, 300, 400, 500]
    >>> jax.tree.unflatten(treedef, newvals)
    [100, (200, 300), [400, 500]]

  See Also:
    - :func:`jax.tree.flatten`
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.structure`
  """
  return tree_util.tree_unflatten(treedef, leaves)


def flatten_with_path(
    tree: Any, is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> tuple[list[tuple[tree_util.KeyPath, Any]], tree_util.PyTreeDef]:
  """类似 ``tree_flatten`` 那样展平 pytree，但还会返回每个叶子的键路径。

  Args:
    tree: 要展平的 pytree。如果它包含自定义类型，建议
      用 ``register_pytree_with_keys`` 注册。

  Returns:
    一个二元组，其第一个元素是键-叶子对的列表，每一项
    包含一个叶子及其键路径。第二个元素是表示展平后树结构的
    treedef。

  Examples:
    >>> import jax
    >>> path_vals, treedef = jax.tree.flatten_with_path([1, {'x': 3}])
    >>> path_vals
    [((SequenceKey(idx=0),), 1), ((SequenceKey(idx=1), DictKey(key='x')), 3)]
    >>> treedef
    PyTreeDef([*, {'x': *}])

  See Also:
    - :func:`jax.tree.flatten`
    - :func:`jax.tree.map_with_path`
    - :func:`jax.tree_util.register_pytree_with_keys`
  """
  return tree_util.tree_flatten_with_path(tree, is_leaf, is_leaf_takes_path)


def leaves_with_path(
    tree: Any, is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> list[tuple[tree_util.KeyPath, Any]]:
  """类似 ``tree_leaves`` 那样获取 pytree 的叶子，并返回每个叶子的键路径。

  Args:
    tree: 一个 pytree。如果它包含自定义类型，建议
      用 ``register_pytree_with_keys`` 注册。

  Returns:
    键-叶子对的列表，其中每一项包含一个叶子及其键路径。

  Examples:
    >>> import jax
    >>> jax.tree.leaves_with_path([1, {'x': 3}])
    [((SequenceKey(idx=0),), 1), ((SequenceKey(idx=1), DictKey(key='x')), 3)]

  See Also:
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.flatten_with_path`
    - :func:`jax.tree_util.register_pytree_with_keys`
  """
  return tree_util.tree_leaves_with_path(tree, is_leaf, is_leaf_takes_path)


def map_with_path(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> Any:
  """把一个多输入函数映射到 pytree 的键路径和参数上，产生一个新的 pytree。

  这是 ``tree_map`` 一个更强大的替代版本，它还能把每个叶子的键路径
  也作为输入参数。

  Args:
    f: 接受 ``2 + len(rest)`` 个参数的函数，即键路径以及各 pytree 中
      对应的叶子。
    tree: 要被映射的 pytree，每个叶子的键路径作为传给 ``f`` 的第一个
      位置参数，叶子本身作为第二个参数。
    *rest: 一组 pytree，其中每个都与 ``tree`` 具有相同结构，或以 ``tree``
      为前缀。

  Returns:
    一个新的 pytree，结构与 ``tree`` 相同，但每个叶子上的值由
    ``f(kp, x, *xs)`` 给出，其中 ``kp`` 是 ``tree`` 中对应叶子的键路径，
    ``x`` 是叶子值，``xs`` 是 ``rest`` 中对应节点上的值组成的元组。

  Examples:
    >>> import jax
    >>> jax.tree.map_with_path(lambda path, x: x + path[0].idx, [1, 2, 3])
    [1, 3, 5]

  See Also:
    - :func:`jax.tree.map`
    - :func:`jax.tree.flatten_with_path`
    - :func:`jax.tree.leaves_with_path`
    - :func:`jax.tree_util.register_pytree_with_keys`
  """
  return tree_util.tree_map_with_path(
      f, tree, *rest, is_leaf=is_leaf, is_leaf_takes_path=is_leaf_takes_path
  )


def broadcast(prefix_tree: Any, full_tree: Any,
              is_leaf: Callable[[Any], bool] | None = None
              ) -> Any:
  """把一个树前缀广播到给定树的完整结构中。

    Args:
      prefix_tree: 一个 pytree，它是 full_tree 的树前缀。
      full_tree: 一个 pytree，其结构用于把前缀的叶子广播进去。
      is_leaf: 一个可选指定的函数，会在每个展平步骤被调用。它应返回一个
        布尔值：true 表示停止遍历并把整个子树视为一个叶子，false
        表示展平应继续遍历当前对象。

    Returns:
      一个与 full_tree 结构匹配的 pytree，其中 prefix_tree 的叶子已被
      广播到每个对应子树的叶子中。

    Examples:
      >>> import jax
      >>> prefix = (1, 2, 3)
      >>> full = (0, {'a': 0, 'b': 0}, (0, 0))
      >>> jax.tree.broadcast(prefix, full)
      (1, {'a': 2, 'b': 2}, (3, 3))

    See Also:
      - :func:`jax.tree.leaves`
      - :func:`jax.tree.structure`
  """
  return tree_util.tree_broadcast(prefix_tree, full_tree, is_leaf=is_leaf)


# dataclasses.field 会被静态类型检查器特殊处理
# （参见 https://peps.python.org/pep-0681/）。为了让 static()
# 能用于内置 dataclass，类型检查器需要识别出
# 它与 `dataclasses.field` 完全相同。由于不存在
# 通用的注册机制，我们按下面的方式定义它：
if TYPE_CHECKING:
  static = dataclasses.field
else:
  def static(**kwargs):
    """用于声明静态 pytree 属性的便捷包装器。

    参数与 :func:`dataclasses.field` 相同，但 :func:`static`
    会自动把 `metadata` 填充为 `static = True`，
    这正是 :func:`jax.tree_util.register_dataclass` 所使用的方式。

    Example:

      >>> import jax
      >>> from dataclasses import dataclass
      ...
      >>> @jax.tree_util.register_dataclass
      ... @dataclass
      ... class MyOp:
      ...   x: jax.Array
      ...   y: jax.Array
      ...   op: str = jax.tree.static(default="add")  # static string field
      ...
      >>> m = MyOp(x=jnp.ones(3), y=jnp.arange(3))
      >>> m
      MyOp(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

      >>> leaves, treedef = jax.tree.flatten(m)
      >>> leaves
      [Array([1., 1., 1.], dtype=float32), Array([0, 1, 2], dtype=int32)]

      >>> treedef
      PyTreeDef(CustomNode(MyOp[('add',)], [*, *]))

      >>> jax.tree.unflatten(treedef, leaves)
      MyOp(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

    See also:
      - :func:`jax.tree_util.register_dataclass`
    """
    metadata = {"static": True, **(kwargs.pop('metadata', {}) or {})}
    return dataclasses.field(metadata=metadata, **kwargs)
