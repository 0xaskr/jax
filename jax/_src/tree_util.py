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
from __future__ import annotations

import collections
from collections.abc import Callable, Hashable, Iterable, Sequence
import dataclasses
import difflib
import functools
import operator as op
import textwrap
from typing import Any, TypeVar

from jax._src import traceback_util
from jax._src.lib import pytree
from jax._src.util import safe_zip, set_module
from jax._src.util import unzip2

# 文件职责：实现 JAX 的 pytree 抽象层，把嵌套的 Python 容器（list、tuple、dict、
# 自定义类等）视为可整体展平/重建的树，是 jit、vmap、grad 等变换处理非数组参数的
# 统一接口。对外经 jax.tree_util 暴露 flatten/unflatten、map/reduce、transpose、
# broadcast 等操作，以及带键路径（KeyPath）的遍历与注册 API。
# 内部维护多个 PyTreeRegistry（默认、None 视作叶子的、供 C++ 快速分派的、用于追踪的），
# 并提供结构不匹配时的诊断信息生成（_prefix_error、_equality_errors）。

export = set_module('jax.tree_util')

traceback_util.register_exclusion(__file__)

T = TypeVar("T")
Typ = TypeVar("Typ", bound=type[Any])
H = TypeVar("H", bound=Hashable)

Leaf = Any
PyTree = Any
PyTreeDef = pytree.PyTreeDef

default_registry = pytree.default_registry()
# 设置 __module__ 与 __name__，使该注册表可以按引用被 pickle 序列化。
default_registry.__module__ = __name__
default_registry.__name__ = "default_registry"  # pyrefly: ignore[missing-attribute]

# 默认注册表的一份副本，其中 None 被当作叶子。
none_leaf_registry = pytree.PyTreeRegistry(
    enable_none=False, enable_tuple=True, enable_namedtuple=True,
    enable_list=True, enable_dict=True)
none_leaf_registry.__module__ = __name__
none_leaf_registry.__name__ = "none_leaf_registry"  # pyrefly: ignore[missing-attribute]

# 一个特殊的内部 pytree 注册表，它包含 `default_registry` 中的全部内容，
# 并额外包含我们想让快速分派路径（“C++ 分派”）学会如何展平与重建的、
# 由 Python 定义的类型。一个关键例子是 PRNG 密钥数组，它目前是一个由
# Python 定义的类（位于 `jax._src.prng`）。这些对象在系统的所有位置
# （例如 Jaxpr 中）本应都是叶子节点，但我们希望在快速分派边界上对它们
# 进行拆包与重新打包。若不在这个注册表里注册这类类型，快速分派路径就不
# 知道该如何把它们当作参数来处理，而会始终报告“缓存未命中”并走慢路径分派。
dispatch_registry = pytree.PyTreeRegistry(
    enable_none=True, enable_tuple=True, enable_namedtuple=True,
    enable_list=True, enable_dict=True)
dispatch_registry.__module__ = __name__
dispatch_registry.__name__ = "dispatch_registry"  # pyrefly: ignore[missing-attribute]

tracing_registry = pytree.PyTreeRegistry()
tracing_registry.__module__ = __name__
tracing_registry.__name__ = "tracing_registry"  # pyrefly: ignore[missing-attribute]


_all_registries = (
    default_registry,
    none_leaf_registry,
    dispatch_registry,
    tracing_registry,
)


@export
def tree_flatten(tree: Any,
                 is_leaf: Callable[[Any], bool] | None = None
                 ) -> tuple[list[Leaf], PyTreeDef]:
  """`jax.tree.flatten` 的别名。"""
  return default_registry.flatten(tree, is_leaf)


@export
def tree_unflatten(treedef: PyTreeDef, leaves: Iterable[Leaf]) -> Any:
  """`jax.tree.unflatten` 的别名。"""
  return treedef.unflatten(leaves)


@export
def tree_leaves(tree: Any,
                is_leaf: Callable[[Any], bool] | None = None
                ) -> list[Leaf]:
  """`jax.tree.leaves` 的别名。"""
  return default_registry.flatten(tree, is_leaf)[0]


@export
def tree_leaves_checked(treedef_expected: PyTreeDef, tree: Any) -> list[Leaf]:
  flat_vals, treedef_actual = tracing_registry.flatten(tree)
  assert treedef_actual == treedef_expected
  return flat_vals


@export
def tree_structure(tree: Any,
                   is_leaf: None | (Callable[[Any],
                                              bool]) = None) -> PyTreeDef:
  """`jax.tree.structure` 的别名。"""
  return default_registry.flatten(tree, is_leaf)[1]

# TODO: 彻底去掉这套树注册表机制（改用 FlatTree）
def treedef_tuple_tracing_registry(treedefs: Iterable[PyTreeDef]) -> PyTreeDef:
  return pytree.treedef_tuple(tracing_registry, list(treedefs))

@export
def treedef_tuple(treedefs: Iterable[PyTreeDef]) -> PyTreeDef:
  """由子 treedef 的可迭代对象构造出一个元组 treedef。

  Args:
    treedefs: PyTree 结构的可迭代对象

  Returns:
    一个表示这些结构所组成的元组的单一 treedef

  Examples:
    >>> import jax
    >>> x = [1, 2, 3]
    >>> y = {'a': 4, 'b': 5}
    >>> x_tree = jax.tree.structure(x)
    >>> y_tree = jax.tree.structure(y)
    >>> xy_tree = jax.tree_util.treedef_tuple([x_tree, y_tree])
    >>> xy_tree == jax.tree.structure((x, y))
    True

  See Also:
    - :func:`jax.tree_util.treedef_children`
  """
  return pytree.treedef_tuple(default_registry, list(treedefs))


@export
def treedef_children(treedef: PyTreeDef) -> list[PyTreeDef]:
  """返回直接子节点的 treedef 列表

  Args:
    treedef: 单个 PyTreeDef

  Returns:
    一个 PyTreeDef 列表，表示 treedef 的各子节点。

  Examples:
    >>> import jax
    >>> x = [(1, 2), 3, {'a': 4}]
    >>> treedef = jax.tree.structure(x)
    >>> jax.tree_util.treedef_children(treedef)
    [PyTreeDef((*, *)), PyTreeDef(*), PyTreeDef({'a': *})]
    >>> _ == [jax.tree.structure(vals) for vals in x]
    True

  See Also:
    - :func:`jax.tree_util.treedef_tuple`
  """
  return treedef.children()


@export
def treedef_is_leaf(treedef: PyTreeDef) -> bool:
  """若该 treedef 表示一个叶子则返回 True。

  Args:
    treedef: 待检查的树

  Returns:
    若 treedef 是一个叶子（即只有一个节点）则为 True；否则为 False。

  Examples:
    >>> import jax
    >>> tree1 = jax.tree.structure(1)
    >>> jax.tree_util.treedef_is_leaf(tree1)
    True
    >>> tree2 = jax.tree.structure([1, 2])
    >>> jax.tree_util.treedef_is_leaf(tree2)
    False
  """
  return treedef.num_nodes == 1


# treedef_is_strict_leaf 不对外导出。
def treedef_is_strict_leaf(treedef: PyTreeDef) -> bool:
  return treedef.num_nodes == 1 and treedef.num_leaves == 1


@export
def all_leaves(iterable: Iterable[Any],
               is_leaf: Callable[[Any], bool] | None = None) -> bool:
  """检验给定可迭代对象中的所有元素是否都是叶子。

  该函数在高级场景中很有用：例如某个库允许对所展平的叶子序列做任意 map 操作，
  它可能想检查结果是否仍然是叶子所构成的一维序列。

  Args:
    iterable: 叶子的可迭代对象。

  Returns:
    一个布尔值，表示输入中的所有元素是否都是叶子。

  Examples:
    >>> import jax
    >>> tree = {"a": [1, 2, 3]}
    >>> assert all_leaves(jax.tree_util.tree_leaves(tree))
    >>> assert not all_leaves([tree])
  """
  if is_leaf is None:
    return pytree.all_leaves(default_registry, iterable)
  else:
    items = list(iterable)
    leaves = tree_leaves(items, is_leaf)
    return len(leaves) == len(items) and all(
        item is leaf for item, leaf in zip(items, leaves, strict=True)
    )


@export
def is_tree_node(typ: type) -> bool:
  """若该类型是已注册的 PyTree 节点类型则返回 True。

  Args:
    typ: 要检查的类型。

  Returns:
    若该类型是已注册的 PyTree 节点类型（内置或自定义）或 namedtuple
    类型，则为 True。
  """
  return default_registry.is_node(typ)


_Children = TypeVar("_Children", bound=Iterable[Any])
_AuxData = TypeVar("_AuxData", bound=Hashable)
KeyEntry = TypeVar("KeyEntry", bound=Any)
KeyLeafPair = tuple[KeyEntry, Any]
KeyLeafPairs = Iterable[KeyLeafPair]
KeyPath = tuple[KeyEntry, ...]


@export
def register_pytree_node(
    nodetype: type[T],
    flatten_func: Callable[[T], tuple[_Children, _AuxData]],
    unflatten_func: Callable[[_AuxData, _Children], T],
    flatten_with_keys_func: (
        Callable[[T], tuple[KeyLeafPairs, _AuxData]] | None
    ) = None,
) -> None:
  """扩充被视为 pytree 内部节点的类型集合。

  参见 :ref:`使用示例 <pytrees>`。

  Args:
    nodetype: 要注册为 pytree 的 Python 类型。
    flatten_func: 展平时使用的函数，接受一个 ``nodetype`` 类型的值并返回
      一个二元组，其中 (1) 是一个可迭代对象，给出需要被递归展平的子节点，
      (2) 是一些可哈希的辅助数据，会被存入 treedef 并传给 ``unflatten_func``。
    unflatten_func: 接受两个参数的函数：由 ``flatten_func`` 返回并存入
      treedef 的辅助数据，以及已重建的子节点。该函数应返回一个
      ``nodetype`` 的实例。

  See Also:
    - :func:`~jax.tree_util.register_static`：用于注册静态 pytree 的更简单 API。
    - :func:`~jax.tree_util.register_dataclass`：用于注册 dataclass 的更简单 API。
    - :func:`~jax.tree_util.register_pytree_with_keys`
    - :func:`~jax.tree_util.register_pytree_node_class`
    - :func:`~jax.tree_util.register_pytree_with_keys_class`

  Examples:
    首先定义一个自定义类型：

    >>> class MyContainer:
    ...   def __init__(self, size):
    ...     self.x = jnp.zeros(size)
    ...     self.y = jnp.ones(size)
    ...     self.size = size

    如果直接在 JIT 编译的函数中使用它，就会报错，因为 JAX 尚不知道
    如何处理这个类型：

    >>> m = MyContainer(size=5)
    >>> def f(m):
    ...   return m.x + m.y + jnp.arange(m.size)
    >>> jax.jit(f)(m)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    TypeError: Cannot interpret value of type <class 'jax.tree_util.MyContainer'> as an abstract array; it does not have a dtype attribute

    为了让 JAX 能识别我们的对象，必须把它注册为一个 pytree：

    >>> def flatten_func(obj):
    ...   children = (obj.x, obj.y)  # children must contain arrays & pytrees
    ...   aux_data = (obj.size,)  # aux_data must contain static, hashable data.
    ...   return (children, aux_data)
    ...
    >>> def unflatten_func(aux_data, children):
    ...   # Here we avoid `__init__` because it has extra logic we don't require:
    ...   obj = object.__new__(MyContainer)
    ...   obj.x, obj.y = children
    ...   obj.size, = aux_data
    ...   return obj
    ...
    >>> jax.tree_util.register_pytree_node(MyContainer, flatten_func, unflatten_func)

    这样定义之后，就可以在 JIT 编译的函数中使用该类型的实例了。

    >>> jax.jit(f)(m)
    Array([1., 2., 3., 4., 5.], dtype=float32)
  """
  for registry in _all_registries:
    registry.register_node(
        nodetype, flatten_func, unflatten_func, flatten_with_keys_func
    )
  _registry[nodetype] = _RegistryEntry(flatten_func, unflatten_func)


@export
def register_pytree_node_class(cls: Typ) -> Typ:
  """扩充被视为 pytree 内部节点的类型集合。

  本函数是 ``register_pytree_node`` 的薄包装，提供面向类的接口。

  Args:
    cls: 要注册为 pytree 的类型

  Returns:
    输入类 ``cls`` 在加入 JAX 的 pytree 注册表后被原样返回。借助该返回值，
    ``register_pytree_node_class`` 可以用作装饰器。

  See Also:
    - :func:`~jax.tree_util.register_static`：用于注册静态 pytree 的更简单 API。
    - :func:`~jax.tree_util.register_dataclass`：用于注册 dataclass 的更简单 API。
    - :func:`~jax.tree_util.register_pytree_node`
    - :func:`~jax.tree_util.register_pytree_with_keys`
    - :func:`~jax.tree_util.register_pytree_with_keys_class`

  Examples:
    这里定义一个与 :func:`jax.jit` 及其他 JAX 变换兼容的自定义容器：

    >>> import jax
    >>> @jax.tree_util.register_pytree_node_class
    ... class MyContainer:
    ...   def __init__(self, x, y):
    ...     self.x = x
    ...     self.y = y
    ...   def tree_flatten(self):
    ...     return ((self.x, self.y), None)
    ...   @classmethod
    ...   def tree_unflatten(cls, aux_data, children):
    ...     return cls(*children)
    ...
    >>> m = MyContainer(jnp.zeros(4), jnp.arange(4))
    >>> def f(m):
    ...   return m.x + 2 * m.y
    >>> jax.jit(f)(m)
    Array([0., 2., 4., 6.], dtype=float32)
  """
  register_pytree_node(
    cls,
    op.methodcaller("tree_flatten"),
    cls.tree_unflatten  # pyrefly: ignore[missing-attribute]
  )
  return cls


@export
def tree_map(f: Callable[..., Any],
             tree: Any,
             *rest: Any,
             is_leaf: Callable[[Any], bool] | None = None) -> Any:
  """`jax.tree.map` 的别名。"""
  leaves, treedef = tree_flatten(tree, is_leaf)
  try:
    all_leaves = [leaves] + [treedef.flatten_up_to(r2 := r) for r in rest]
  except Exception as e:
    err = next(_prefix_error((), tree, r2, is_leaf), None)  # type: ignore
    raise (err('tree_map tree') if err is not None else e) from None
  return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


@export
def tree_transpose(outer_treedef: PyTreeDef, inner_treedef: PyTreeDef | None,
                   pytree_to_transpose: Any) -> Any:
  """`jax.tree.transpose` 的别名。"""
  flat, treedef = tree_flatten(pytree_to_transpose)
  if inner_treedef is None:
    inner_treedef = tree_structure(outer_treedef.flatten_up_to(pytree_to_transpose)[0])
  inner_size = inner_treedef.num_leaves
  outer_size = outer_treedef.num_leaves
  if treedef.num_leaves != (inner_size * outer_size):
    expected_treedef = outer_treedef.compose(inner_treedef)
    raise TypeError(f"Mismatch\n{treedef}\n != \n{expected_treedef}")
  iter_flat = iter(flat)
  lol = [
      [next(iter_flat) for _ in range(inner_size)] for __ in range(outer_size)
  ]
  transposed_lol = zip(*lol)
  subtrees = map(functools.partial(tree_unflatten, outer_treedef), transposed_lol)
  return tree_unflatten(inner_treedef, subtrees)


# TODO(mattjj): 当 C++ 侧的注册表可查询性足够强、足以表达 _replace_nones 时，
# 就移除 Python 侧的注册表。那也许意味着等到我们有了 flatten_one 函数之后。
_RegistryEntry = collections.namedtuple("_RegistryEntry", ["to_iter", "from_iter"])
_registry: dict[type[Any], _RegistryEntry] = {
    tuple: _RegistryEntry(lambda xs: (xs, None), lambda _, xs: tuple(xs)),
    list: _RegistryEntry(lambda xs: (xs, None), lambda _, xs: list(xs)),
    dict: _RegistryEntry(lambda xs: unzip2(sorted(xs.items()))[::-1],
                         lambda keys, xs: dict(zip(keys, xs))),
    type(None): _RegistryEntry(lambda z: ((), None), lambda _, xs: None),
}


class Unspecified:
  pass


@export
def tree_reduce(function: Callable[[T, Any], T],
                tree: Any,
                initializer: T | Unspecified = Unspecified(),
                is_leaf: Callable[[Any], bool] | None = None) -> T:
  """`jax.tree.reduce` 的别名。"""
  if isinstance(initializer, Unspecified):
    return functools.reduce(function, tree_leaves(tree, is_leaf=is_leaf))
  else:
    return functools.reduce(function, tree_leaves(tree, is_leaf=is_leaf), initializer)


def _parallel_reduce(
    sequence: list[T],
    operation: Callable[[T, T], T],
    identity: T | Unspecified = Unspecified(),
) -> T:
  length = len(sequence)
  if length == 0:
    if isinstance(identity, Unspecified):
      raise TypeError("Must specify identity for parallel reduction of empty sequence.")
    return identity
  elif length == 1:
    return sequence[0]
  else:
    index = length // 2
    a = _parallel_reduce(sequence[:index], operation, identity)
    b = _parallel_reduce(sequence[index:], operation, identity)
    return operation(a, b)


@export
def tree_reduce_associative(
    operation: Callable[[T, T], T],
    tree: Any,
    *,
    identity: T | Unspecified = Unspecified(),
    is_leaf: Callable[[Any], bool] | None = None,
) -> T:
  """`jax.tree.reduce_associative` 的别名。"""
  sequence = tree_leaves(tree, is_leaf=is_leaf)
  return _parallel_reduce(sequence, operation, identity)


@export
def tree_all(tree: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> bool:
  """`jax.tree.all` 的别名。"""
  return all(tree_leaves(tree, is_leaf=is_leaf))


class _HashableCallableShim:
  """把 __call__、__hash__ 与 __eq__ 委托给另一个对象的对象。"""

  def __init__(self, fun):
    self.fun = fun

  def __call__(self, *args, **kw):
    return self.fun(*args, **kw)

  def __hash__(self):
    return hash(self.fun)

  def __eq__(self, other):
    if isinstance(other, _HashableCallableShim):
      return self.fun == other.fun
    return self.fun == other

  def __repr__(self):
    return f'_HashableCallableShim({self.fun!r})'


@export
class Partial(functools.partial):
  """`functools.partial` 的一个可用于 pytree 的版本。

  当你需要以与 JAX 变换兼容的方式进行偏函数求值时使用它，例如
  ``Partial(func, *args, **kwargs)``。

  （你需要显式选择启用这种行为，因为我们不想让 `functools.partial` 的语义
  与普通函数闭包不同。）

  例如，下面是与 ``functools.partial`` 用法类似的一个 ``Partial`` 基本示例：

  >>> import jax.numpy as jnp
  >>> add_one = Partial(jnp.add, 1)
  >>> add_one(2)
  Array(3, dtype=int32, weak_type=True)

  pytree 兼容意味着得到的偏函数可以作为参数传入经过变换的 JAX 函数，
  而标准的 ``functools.partial`` 函数做不到这一点：

  >>> from jax import jit
  >>> @jit
  ... def call_func(f, *args):
  ...   return f(*args)
  ...
  >>> call_func(add_one, 2)
  Array(3, dtype=int32, weak_type=True)

  向 ``Partial`` 传入零个参数实际上是把原函数包装起来，使它在 JAX 变换后的
  函数中成为合法参数：

  >>> call_func(Partial(jnp.add), 1, 2)
  Array(3, dtype=int32, weak_type=True)

  若我们直接把 ``jnp.add`` 传给 ``call_func``，则会引发 ``TypeError``。

  注意：如果 ``Partial`` 的结果被用在需要追踪值的上下文中，那么当它被传给
  这个已被部分求值的函数时，所有已绑定的参数都会被追踪：

  >>> print_zero = Partial(print, 0)
  >>> print_zero()
  0
  >>> call_func(print_zero)  # doctest:+ELLIPSIS
  JitTracer(~int32[])
  """

  def __new__(klass, func, *args, **kw):
    # 在 Python 3.10+ 中，如果 func 本身就是 functools.partial 的实例，
    # functools.partial.__new__ 会把该 Partial 实例的参数与 func 的参数合并。
    # 我们把 func 装进一个（目前）没有 `func` 属性的类里来破除这一优化，
    # 因为我们关心的正是哪些参数被视为 pytree 的一部分。
    if isinstance(func, functools.partial):
      original_func = func
      func = _HashableCallableShim(original_func)
      out = super().__new__(klass, func, *args, **kw)
      func.func = original_func.func  # pyrefly: ignore[missing-attribute]
      func.args = original_func.args  # pyrefly: ignore[missing-attribute]
      func.keywords = original_func.keywords  # pyrefly: ignore[missing-attribute]
      return out
    else:
      return super().__new__(klass, func, *args, **kw)


register_pytree_node(
    Partial,
    lambda partial_: ((partial_.args, partial_.keywords), partial_.func),
    lambda func, xs: Partial(func, *xs[0], **xs[1]),
)


@export
def tree_broadcast(prefix_tree: Any, full_tree: Any,
                   is_leaf: Callable[[Any], bool] | None = None
                  ) -> Any:
  """`jax.tree.broadcast` 的别名。"""
  broadcast_leaves = broadcast_prefix(prefix_tree, full_tree, is_leaf=is_leaf)
  return tree_structure(full_tree).unflatten(broadcast_leaves)


# broadcast_prefix 不对外导出
def broadcast_prefix(prefix_tree: Any, full_tree: Any,
                     is_leaf: Callable[[Any], bool] | None = None
                     ) -> list[Any]:
  """把树前缀的叶子广播为给定完整树的全部叶子。

    Args:
      prefix_tree: 一个 pytree，它是 full_tree 的树前缀。
      full_tree: 一个 pytree，其结构用于承载被广播的前缀叶子。
      is_leaf: 一个可选指定的函数，会在 prefix_tree 的每个展平步骤上被调用。
        它应返回一个布尔值：为真时停止遍历并把整棵子树当作一个叶子，
        为假时表示展平应继续遍历当前对象。

    Returns:
      一个叶子列表，其数量与完整树所期望的数量一致；其中每个前缀树的
      叶子都被复制，以匹配其对应子树的数量。
  """
  result = []
  num_leaves = lambda t: tree_structure(t).num_leaves
  add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))
  try:
    tree_map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
  except ValueError:
      e, *_ = prefix_errors(prefix_tree, full_tree)
      raise e('broadcast_prefix prefix_tree') from None
  return result


# broadcast_flattened_prefix_with_treedef 不对外导出
def broadcast_flattened_prefix_with_treedef(
    prefix_leaves: list[Any],
    prefix_treedef: PyTreeDef,
    full_treedef: PyTreeDef,
) -> list[Any]:
  """把树前缀的叶子广播为给定完整 treedef 的全部叶子。

    Args:
      prefix_leaves: 某个 pytree 的叶子，该 pytree 是
        full_treedef 的树前缀。
      prefix_treedef: 某个 pytree 的 PyTreeDef，该 pytree 是
        full_treedef 的树前缀。
      full_treedef: 一个 PyTreeDef，其结构用于承载被广播的前缀叶子。

    Returns:
      一个叶子列表，其数量与完整树所期望的数量一致；其中前缀树的每个
      叶子都被复制，以匹配其对应子树的数量。
  """
  # 注意：目前 `broadcast_flattened_prefix_with_treedef` 只被
  # `api_util.flatten_axes` 调用，而后者会用自身的异常与错误信息替换
  # 这里抛出的任何异常。在这个函数被更多地方使用之前，
  # 它所抛出的错误信息大概应该先改进一下。
  #
  # TODO(jburnim): 把 `broadcast_prefix` 与这个函数合并？
  # prefix_leaves, prefix_treedef = tree_flatten(prefix_tree, is_leaf)
  ret = []

  # TODO(jburnim): 这个遍历应该用 C++ 实现吗？
  def _broadcast(broadcast_fn, leaf_start, leaf_end, prefix_treedef, treedef):
    if treedef_is_strict_leaf(prefix_treedef):
      # 我们在前缀中遇到了一个叶子，于是为树对应部分中的每个叶子
      # 重复该前缀叶子。
      assert (leaf_end - leaf_start) == 1
      ret.extend(prefix_leaves[leaf_start:leaf_end] * treedef.num_leaves)
      return

    if treedef_is_strict_leaf(treedef):
      raise ValueError('`prefix_treedef` is not a prefix of `full_treedef`')

    prefix_node_data = prefix_treedef.node_data()
    node_data = treedef.node_data()
    if prefix_node_data != node_data:
      raise ValueError(f'expected {node_data}, got {prefix_node_data}')

    prefix_i = leaf_start
    for prefix_child, tree_child in zip(
        prefix_treedef.children(), treedef.children(), strict=True):
      broadcast_fn(broadcast_fn, prefix_i, prefix_i + prefix_child.num_leaves,
                   prefix_child, tree_child,
      )
      prefix_i += prefix_child.num_leaves

  # 把 _broadcast 作为参数传入，以免它成为自身闭包中的自由变量，
  # 那会形成引用环。
  _broadcast(_broadcast, 0, len(prefix_leaves), prefix_treedef, full_treedef)
  return ret


@export
def flatten_one_level(tree: Any) -> tuple[Iterable[Any], Hashable]:
  """把给定的 pytree 节点展平一层。

  Args:
    tree: 一个合法的 pytree 节点，可以是内置的，也可以是通过
      :func:`register_pytree_node` 或相关函数注册的。

  Returns:
    一个二元组，包含被展平的 pytree 子节点及其可哈希的元数据。

  Raises:
    ValueError: 如果给定的 pytree 既不是内置容器，也不是通过
    ``register_pytree_node`` 或 ``register_pytree_with_keys`` 注册的容器。

  Examples:
    >>> import jax
    >>> from jax._src.tree_util import flatten_one_level
    >>> flattened, meta = flatten_one_level({'a': [1, 2], 'b': {'c': 3}})
    >>> flattened
    ([1, 2], {'c': 3})
    >>> meta
    ('a', 'b')
  """
  out = default_registry.flatten_one_level(tree)
  if out is None:
    raise ValueError(f"can't tree-flatten type: {type(tree)}")
  else:
    return out


@export
def flatten_one_level_with_keys(
    tree: Any,
) -> tuple[Iterable[KeyLeafPair], Hashable]:
  """把给定的 pytree 节点展平一层，并带上键。"""
  out = default_registry.flatten_one_level_with_keys(tree)
  if out is None:
    raise ValueError(f"can't tree-flatten type: {type(tree)}")
  else:
    return out


# prefix_errors 不对外导出
def prefix_errors(prefix_tree: Any, full_tree: Any,
                  is_leaf: Callable[[Any], bool] | None = None,
                  ) -> list[Callable[[str], ValueError]]:
  return list(_prefix_error((), prefix_tree, full_tree, is_leaf))


# equality_errors 不对外导出
def equality_errors(
    tree1: Any, tree2: Any, is_leaf: Callable[[Any], bool] | None = None,
) -> Iterable[tuple[KeyPath, str, str, str]]:
  """用于描述两个 pytree 之间结构差异的辅助函数。

  Args:
    tree1, tree2: 已知结构不同的 pytree。

  Usage:

    raise Exception(
        "Value 1 and value 2 must have the same pytree structure, but they have "
        "the following structural differences:\n" +
        ("\n".join(
           f"   - {keystr(path)} is a {thing1} in value 1 and a {thing2} in "
           f" value 2, so {explanation}.\n"
           for path, thing1, thing2, explanation
           in equality_errors(val1, val2))))
  """
  yield from _equality_errors((), tree1, tree2, is_leaf)

def equality_errors_pytreedef(
    tree1: PyTreeDef,
    tree2: PyTreeDef) -> Iterable[tuple[KeyPath, str, str, str]]:
  """与 `equality_errors` 类似，但作用于 PyTreeDef。"""
  # TODO(mattjj): 让 equality_errors 不再打印类型名，从而避免元类
  leaf = type("LeafMeta", (type,), dict(__repr__=lambda _: "pytree leaf")
              )("Leaf", (), {})()
  return equality_errors(tree_unflatten(tree1, [leaf] * tree1.num_leaves),
                         tree_unflatten(tree2, [leaf] * tree2.num_leaves))

# TODO(mattjj): 也许与 _prefix_error 共用一部分逻辑？
def _equality_errors(path, t1, t2, is_leaf):
  # 如果二者都是叶子，这就不算结构相等性错误。
  if (treedef_is_strict_leaf(tree_structure(t1, is_leaf=is_leaf)) and
      treedef_is_strict_leaf(tree_structure(t2, is_leaf=is_leaf))): return

  # 两棵树可能因为它们类型不同而不一致：
  if type(t1) != type(t2):
    yield path, str(type(t1)), str(type(t2)), 'their Python types differ'
    return  # 不再查找更多错误

  # 或者它们可能因为根节点的子节点数量或键不同而不一致
  #（对 list/tuple 做特殊处理）：
  if isinstance(t1, (list, tuple)):
    assert type(t1) == type(t2)
    if len(t1) != len(t2):
      yield (path,
             f'{type(t1).__name__} of length {len(t1)}',
             f'{type(t2).__name__} of length {len(t2)}',
             'the lengths do not match')
      return  # 不再查找更多错误
  t1_children, t1_meta = flatten_one_level(t1)
  t2_children, t2_meta = flatten_one_level(t2)
  t1_children = tuple(t1_children)
  t2_children = tuple(t2_children)
  t1_keys, t2_keys = _child_keys(t1), _child_keys(t2)
  try:
    diff = ' '.join(repr(k.key) for k in
                    set(t1_keys).symmetric_difference(set(t2_keys)))
  except:
    diff = ''
  if len(t1_children) != len(t2_children):
    yield (path,
           f'{type(t1)} with {len(t1_children)} child'
           f'{"ren" if len(t1_children) > 1 else ""}',
           f'{type(t2)} with {len(t2_children)} child'
           f'{"ren" if len(t2_children) > 1 else ""}',
           'the numbers of children do not match' +
           (diff and f', with the symmetric difference of key sets: {{{diff}}}')
           )
    return  # 不再查找更多错误

  # 或者它们可能因为根节点的 pytree 元数据不同而不一致：
  if t1_meta != t2_meta:
    yield (path,
           f'{type(t1)} with pytree metadata {t1_meta}',
           f'{type(t2)} with pytree metadata {t2_meta}',
           'the pytree node metadata does not match')
    return  # 不再查找更多错误

  # 如果根节点类型与子节点数量都一致，那么不匹配一定出现在某棵子树中，
  # 于是递归下去：
  assert t1_keys == t2_keys, \
      f"equal pytree nodes gave different tree keys: {t1_keys} and {t2_keys}"
  for k, c1, c2 in zip(t1_keys, t1_children, t2_children):
    yield from _equality_errors((*path, k), c1, c2, is_leaf)


SequenceKey: Any = pytree.SequenceKey
DictKey: Any = pytree.DictKey
GetAttrKey: Any = pytree.GetAttrKey
FlattenedIndexKey: Any = pytree.FlattenedIndexKey


@export
def keystr(keys: KeyPath, *, simple: bool = False, separator: str = '') -> str:
  """用于把键的元组美观地打印出来的辅助函数。

  Args:
    keys: 一个由 ``KeyEntry`` 组成的元组，或任何可转换为字符串的类。
    simple: 若为 True，则对键使用简化后的字符串表示。键的简化表示会比默认
      表示更紧凑，但在某些情况下有歧义（例如 "0" 可能指列表中的第一项，
      也可能指整数 0 或字符串 "0" 对应的字典键）。
    separator: 用于连接各键字符串表示的连接符。

  Returns:
    一个把所有键的字符串表示连接起来的字符串。

  Examples:
    >>> import jax
    >>> params = {'foo': {'bar': {'baz': 1, 'bat': [2, 3]}}}
    >>> for path, _ in jax.tree_util.tree_leaves_with_path(params):
    ...   print(jax.tree_util.keystr(path))
    ['foo']['bar']['bat'][0]
    ['foo']['bar']['bat'][1]
    ['foo']['bar']['baz']
    >>> for path, _ in jax.tree_util.tree_leaves_with_path(params):
    ...   print(jax.tree_util.keystr(path, simple=True, separator='/'))
    foo/bar/bat/0
    foo/bar/bat/1
    foo/bar/baz
  """
  str_fn = _simple_entrystr if simple else str
  return separator.join(map(str_fn, keys))


def _simple_entrystr(key: KeyEntry) -> str:
  match key:
    case (
        SequenceKey(idx=key)
        | DictKey(key=key)
        | GetAttrKey(name=key)
        | FlattenedIndexKey(key=key)
    ):
      return str(key)
    case _:
      return str(key)


@export
def register_pytree_with_keys(
    nodetype: type[T],
    flatten_with_keys: Callable[[T], tuple[Iterable[KeyLeafPair], _AuxData]],
    unflatten_func: Callable[[_AuxData, Iterable[Any]], T],
    flatten_func: None | (Callable[[T], tuple[Iterable[Any], _AuxData]]) = None,
):
  """扩充被视为 pytree 内部节点的类型集合。

  这是 ``register_pytree_node`` 的一个更强的替代方案，允许你在展平与树映射时
  访问每个 pytree 叶子的键路径。

  Args:
    nodetype: 要当作 pytree 内部节点的 Python 类型。
    flatten_with_keys: 展平时使用的函数，接受一个 ``nodetype`` 类型的值并返回
      一个二元组，其中 (1) 是一个可迭代对象，给出每个键路径及其子节点组成的
      元组，(2) 是一些可哈希的辅助数据，会被存入 treedef 并传给
      ``unflatten_func``。
    unflatten_func: 接受两个参数的函数：由 ``flatten_func`` 返回并存入
      treedef 的辅助数据，以及已重建的子节点。该函数应返回一个
      ``nodetype`` 的实例。
    flatten_func: 一个可选的函数，与 ``flatten_with_keys`` 类似，但只返回
      子节点与辅助数据。它返回子节点的顺序必须与 ``flatten_with_keys`` 相同，
      返回的辅助数据也必须相同。该参数是可选的，只在调用 ``tree_map``、
      ``tree_flatten`` 这类不带键的函数时用于加速遍历。

  Examples:
    首先定义一个自定义类型：

    >>> class MyContainer:
    ...   def __init__(self, size):
    ...     self.x = jnp.zeros(size)
    ...     self.y = jnp.ones(size)
    ...     self.size = size

    现在用一个能感知键的展平函数来注册它：

    >>> from jax.tree_util import register_pytree_with_keys_class, GetAttrKey
    >>> def flatten_with_keys(obj):
    ...   children = [(GetAttrKey('x'), obj.x),
    ...               (GetAttrKey('y'), obj.y)]  # children must contain arrays & pytrees
    ...   aux_data = (obj.size,)  # aux_data must contain static, hashable data.
    ...   return children, aux_data
    ...
    >>> def unflatten(aux_data, children):
    ...   # Here we avoid `__init__` because it has extra logic we don't require:
    ...   obj = object.__new__(MyContainer)
    ...   obj.x, obj.y = children
    ...   obj.size, = aux_data
    ...   return obj
    ...
    >>> jax.tree_util.register_pytree_node(MyContainer, flatten_with_keys, unflatten)

    这样它就可以与 :func:`~jax.tree_util.tree_flatten_with_path` 这类函数
    一起使用了：

    >>> m = MyContainer(4)
    >>> leaves, treedef = jax.tree_util.tree_flatten_with_path(m)
  """
  if not flatten_func:
    def flatten_func_impl(tree):
      key_children, treedef = flatten_with_keys(tree)
      return [c for _, c in key_children], treedef
    flatten_func = flatten_func_impl

  register_pytree_node(
      nodetype, flatten_func, unflatten_func, flatten_with_keys
  )


@export
def register_pytree_with_keys_class(cls: Typ) -> Typ:
  """扩充被视为 pytree 内部节点的类型集合。

  本函数与 ``register_pytree_node_class`` 类似，但要求类中定义好了如何带键
  展平。

  它是 ``register_pytree_with_keys`` 的薄包装，并提供面向类的接口：

  Args:
    cls: 要注册为 pytree 的类型

  Returns:
    输入类 ``cls`` 在加入 JAX 的 pytree 注册表后被原样返回。借助该返回值，
    ``register_pytree_node_class`` 可以用作装饰器。

  See also:
    - :func:`~jax.tree_util.register_static`：用于注册静态 pytree 的更简单 API。
    - :func:`~jax.tree_util.register_dataclass`：用于注册 dataclass 的更简单 API。
    - :func:`~jax.tree_util.register_pytree_node`
    - :func:`~jax.tree_util.register_pytree_with_keys`
    - :func:`~jax.tree_util.register_pytree_node_class`

  Examples:
    >>> from jax.tree_util import register_pytree_with_keys_class, GetAttrKey
    >>> @register_pytree_with_keys_class
    ... class Special:
    ...   def __init__(self, x, y):
    ...     self.x = x
    ...     self.y = y
    ...   def tree_flatten_with_keys(self):
    ...     return (((GetAttrKey('x'), self.x), (GetAttrKey('y'), self.y)), None)
    ...   @classmethod
    ...   def tree_unflatten(cls, aux_data, children):
    ...     return cls(*children)
  """
  flatten_func = (
      op.methodcaller("tree_flatten") if hasattr(cls, "tree_flatten") else None
  )
  register_pytree_with_keys(
      cls, op.methodcaller("tree_flatten_with_keys"),
      cls.tree_unflatten,  # pyrefly: ignore[missing-attribute]
      flatten_func
  )
  return cls


@export
def register_dataclass(
    nodetype: Typ,
    data_fields: Sequence[str] | None = None,
    meta_fields: Sequence[str] | None = None,
    drop_fields: Sequence[str] = (),
) -> Typ:
  """扩充被视为 pytree 内部节点的类型集合。

  它与 ``register_pytree_with_keys_class`` 的区别在于：C++ 侧的注册表会使用
  优化过的 C++ dataclass 内建实现，而不是这些参数函数。

  关于注册 pytree 的更多信息，参见 :ref:`pytrees-custom-pytree-nodes`。

  Args:
    nodetype: 要当作 pytree 内部节点的 Python 类型。这里假定它具备
      :obj:`~dataclasses.dataclass` 的语义：即类属性代表对象的全部状态，
      并且可以作为关键字参数传给类构造函数来创建对象的一份副本。
      所有已定义的属性都应列在 ``meta_fields`` 或 ``data_fields`` 中。
    meta_fields: 元数据字段名：当该 pytree 被传给 :func:`jax.jit` 时，这些属性
      会被视为 :term:`static` 静态值。只有当 ``nodetype`` 是 dataclass 时，
      ``meta_fields`` 才可以省略；此时可通过 :func:`dataclasses.field` 把各个
      字段标记为静态（见下面的示例）。元数据字段*必须*是静态、可哈希、
      不可变的对象，因为这些对象会被用来生成 JIT 缓存键。特别地，元数据字段
      不能包含 :class:`jax.Array` 或 :class:`numpy.ndarray` 对象。
    data_fields: 数据字段名：当该 pytree 被传给 :func:`jax.jit` 时，这些属性会被
      视为非静态值。只有当 ``nodetype`` 是 dataclass 时，``data_fields`` 才可以
      省略；此时除非通过 :func:`dataclasses.field` 标记（见下面的示例）或出现在
      drop_fields 中，字段都默认视为数据字段。数据字段*必须*是与 JAX 兼容的对象，
      例如数组（:class:`jax.Array` 或 :class:`numpy.ndarray`）、标量，或以数组
      或标量为叶子的 pytree。注意 ``None`` 是合法的数据字段，因为 JAX 会把它
      识别为空 pytree。
    drop_fields: 仅当 ``nodetype`` 是 dataclass 时才起作用。指定一个
      ``dataclasses.fields(nodetype)`` 中字段名的序列，这些字段将被排除在
      pytree 注册之外。

  Returns:
    输入类 ``nodetype`` 在加入 JAX 的 pytree 注册表后被原样返回，因此
    :func:`register_dataclass` 可以用作装饰器。

  Examples:
    在 JAX v0.4.35 及更早版本中，必须指定 ``data_fields`` 与 ``meta_fields``
    才能使用这个装饰器：

    >>> import jax
    >>> from dataclasses import dataclass
    >>> from functools import partial
    ...
    >>> @partial(jax.tree_util.register_dataclass,
    ...          data_fields=['x', 'y'],
    ...          meta_fields=['op'])
    ... @dataclass
    ... class MyStruct:
    ...   x: jax.Array
    ...   y: jax.Array
    ...   op: str
    ...
    >>> m = MyStruct(x=jnp.ones(3), y=jnp.arange(3), op='add')
    >>> m
    MyStruct(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

    从 JAX v0.4.36 开始，对于 :func:`~dataclasses.dataclass` 输入，
    ``data_fields`` 与 ``meta_fields`` 参数变为可选：字段默认归入
    ``data_fields``，除非用 :func:`dataclasses.field` 的 `static` 元数据
    标记为静态。

    >>> import jax
    >>> from dataclasses import dataclass, field
    ...
    >>> @jax.tree_util.register_dataclass
    ... @dataclass
    ... class MyStruct:
    ...   x: jax.Array  # defaults to non-static data field
    ...   y: jax.Array  # defaults to non-static data field
    ...   op: str = field(metadata=dict(static=True))  # marked as static meta field.
    ...
    >>> m = MyStruct(x=jnp.ones(3), y=jnp.arange(3), op='add')
    >>> m
    MyStruct(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

    该类注册之后，就可以与 :mod:`jax.tree` 和 :mod:`jax.tree_util` 中的函数
    一起使用了：

    >>> leaves, treedef = jax.tree.flatten(m)
    >>> leaves
    [Array([1., 1., 1.], dtype=float32), Array([0, 1, 2], dtype=int32)]
    >>> treedef
    PyTreeDef(CustomNode(MyStruct[('add',)], [*, *]))
    >>> jax.tree.unflatten(treedef, leaves)
    MyStruct(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

    特别地，这一注册使得 ``m`` 能够无缝地穿过用 :func:`jax.jit` 及其他 JAX
    变换包装的代码：其中 ``data_fields`` 被当作动态参数，而 ``meta_fields``
    被当作静态参数：

    >>> @jax.jit
    ... def compiled_func(m):
    ...   if m.op == 'add':
    ...     return m.x + m.y
    ...   else:
    ...     raise ValueError(f"{m.op=}")
    ...
    >>> compiled_func(m)
    Array([1., 2., 3.], dtype=float32)
  """
  if data_fields is None or meta_fields is None:
    if (data_fields is None) != (meta_fields is None):
      raise TypeError("register_dataclass: data_fields and meta_fields must both be specified"
                      f" when either is specified. Got {data_fields=} {meta_fields=}.")
    if not dataclasses.is_dataclass(nodetype):
      raise TypeError("register_dataclass: data_fields and meta_fields are required when"
                      f" nodetype is not a dataclass. Got {nodetype=}.")
    data_fields = [
        f.name
        for f in dataclasses.fields(nodetype)
        if not f.metadata.get("static", False) and f.name not in drop_fields
    ]
    meta_fields = [
        f.name
        for f in dataclasses.fields(nodetype)
        if f.metadata.get("static", False) and f.name not in drop_fields
    ]

  assert meta_fields is not None
  assert data_fields is not None

  # 在当前作用域内把输入存为不可变元组，因为后续求值会闭包捕获它们。
  # 这样可以避免调用方传入的列表之后被修改而带来的、可能令人困惑的行为。
  meta_fields = tuple(meta_fields)
  data_fields = tuple(data_fields)

  if dataclasses.is_dataclass(nodetype):
    init_fields = {f.name for f in dataclasses.fields(nodetype) if f.init}
    init_fields.difference_update(drop_fields)
    if {*meta_fields, *data_fields} != init_fields:
      msg = (
          "data_fields and meta_fields must include all dataclass fields with"
          " ``init=True`` and only them."
      )
      if missing := init_fields - {*meta_fields, *data_fields}:
        msg += (
            f" Missing fields: {missing}. Add them to drop_fields to suppress"
            " this error."
        )
      if unexpected := {*meta_fields, *data_fields} - init_fields:
        msg += f" Unexpected fields: {unexpected}."
      raise ValueError(msg)

  if overlap := set(data_fields) & set(meta_fields):
    raise ValueError(
        "data_fields and meta_fields must not overlap. Overlapping fields:"
        f" {overlap}."
    )

  def unflatten_func(meta, data):
    meta_args = tuple(zip(meta_fields, meta))
    data_args = tuple(zip(data_fields, data))
    kwargs = dict(meta_args + data_args)
    return nodetype(**kwargs)

  def flatten_func(x):
    meta = tuple(getattr(x, name) for name in meta_fields)
    data = tuple(getattr(x, name) for name in data_fields)
    return data, meta

  for registry in _all_registries:
    registry.register_dataclass_node(nodetype, list(data_fields), list(meta_fields))
  _registry[nodetype] = _RegistryEntry(flatten_func, unflatten_func)
  return nodetype


register_pytree_with_keys(
    collections.OrderedDict,
    lambda x: (tuple((DictKey(k), x[k]) for k in x.keys()), tuple(x.keys())),
    lambda keys, values: collections.OrderedDict(safe_zip(keys, values)),
)

def _flatten_defaultdict_with_keys(d):
  keys = tuple(sorted(d))
  return tuple((DictKey(k), d[k]) for k in keys), (d.default_factory, keys)

register_pytree_with_keys(
    collections.defaultdict,
    _flatten_defaultdict_with_keys,
    lambda s, values: collections.defaultdict(s[0], safe_zip(s[1], values)),
)


@export
def register_static(cls: type[H]) -> type[H]:
  """把 `cls` 注册为一个不含叶子的 pytree。

  这类实例会被 :func:`jax.jit`、:func:`jax.pmap` 等视为静态值。这可以替代
  用 ``jit`` 的 ``static_argnums`` 与 ``static_argnames`` 关键字参数、
  ``pmap`` 的 ``static_broadcasted_argnums`` 等把参数标记为静态的做法。

  Args:
    cls: 要注册为静态的类型。必须是可哈希的，定义见
      https://docs.python.org/3/glossary.html#term-hashable。

  Returns:
    输入类 ``cls`` 在加入 JAX 的 pytree 注册表后被原样返回。这使得
    ``register_static`` 可以用作装饰器。

  Examples:
    >>> import jax
    >>> @jax.tree_util.register_static
    ... class StaticStr(str):
    ...   pass

    现在就可以在 :func:`jax.jit` 编译的函数中直接使用这个静态字符串，
    而无需用 ``static_argnums`` 把该变量标记为静态：

    >>> @jax.jit
    ... def f(x, y, s):
    ...   return x + y if s == 'add' else x - y
    ...
    >>> f(1, 2, StaticStr('add'))
    Array(3, dtype=int32, weak_type=True)
  """
  flatten = lambda obj: ((), obj)
  unflatten = lambda obj, empty_iter_children: obj
  register_pytree_with_keys(cls, flatten, unflatten)
  return cls


@export
def tree_flatten_with_path(
    tree: Any, is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> tuple[list[tuple[KeyPath, Any]], PyTreeDef]:
  """`jax.tree.flatten_with_path` 的别名。"""
  is_leaf_with_kp: Callable[[Any, Any], bool] | None = is_leaf
  if not is_leaf_takes_path and is_leaf is not None:
    is_leaf_with_kp = lambda _, x: is_leaf(x)
  return default_registry.flatten_with_path(tree, is_leaf_with_kp)


@export
def tree_leaves_with_path(
    tree: Any, is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> list[tuple[KeyPath, Any]]:
  """`jax.tree.leaves_with_path` 的别名。"""
  return tree_flatten_with_path(tree, is_leaf, is_leaf_takes_path)[0]
generate_key_paths = tree_leaves_with_path


@export
def tree_map_with_path(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[..., bool] | None = None,
    is_leaf_takes_path: bool = False,
) -> Any:
  """`jax.tree.map_with_path` 的别名。"""
  keypath_leaves, treedef = tree_flatten_with_path(
      tree, is_leaf, is_leaf_takes_path
  )
  keypath_leaves = list(zip(*keypath_leaves))
  try:
    all_keypath_leaves = keypath_leaves + [treedef.flatten_up_to(r2 := r) for r in rest]
  except Exception as e:
    err = next(_prefix_error((), tree, r2, is_leaf), None)  # type: ignore
    raise (err('tree_map_with_path tree') if err is not None else e) from None
  return treedef.unflatten(f(*xs) for xs in zip(*all_keypath_leaves))


def _child_keys(pytree: Any) -> KeyPath:
  assert not treedef_is_strict_leaf(tree_structure(pytree))
  return tuple(k for k, _ in flatten_one_level_with_keys(pytree)[0])


def _prefix_error(
    key_path: KeyPath,
    prefix_tree: Any,
    full_tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Iterable[Callable[[str], ValueError]]:
  # 叶子是任何树的合法前缀：
  if treedef_is_strict_leaf(tree_structure(prefix_tree, is_leaf=is_leaf)):
    return

  # 两棵子树可能因为根节点类型不同而不一致：
  if type(prefix_tree) != type(full_tree):
    yield lambda name: ValueError(
      "pytree structure error: different types at key path\n"
      f"    {name}{keystr(key_path)}\n"
      f"At that key path, the prefix pytree {name} has a subtree of type\n"
      f"    {type(prefix_tree)}\n"
      f"but at the same key path the full pytree has a subtree of different type\n"
      f"    {type(full_tree)}.")
    return  # 不再在这棵子树中查找更多错误

  # 或者它们可能因为根节点的子节点数量或键不同而不一致。此时 prefix_tree
  # 与 full_tree 类型相同，且 prefix_tree 不是叶子，因此二者都可以展平一层：
  prefix_tree_children, prefix_tree_meta = flatten_one_level(prefix_tree)
  full_tree_children, full_tree_meta = flatten_one_level(full_tree)
  prefix_tree_children = tuple(prefix_tree_children)
  full_tree_children = tuple(full_tree_children)
  prefix_tree_keys = _child_keys(prefix_tree)
  full_tree_keys = _child_keys(full_tree)
  # 我们首先检查特殊类型（list 与 tuple；如果它们也是 pytree，这里其实还可以
  # 检查字符串和集合，基本上就是 Sequence），这样就能报告长度不一致而不是
  # 整数键不一致：
  if isinstance(prefix_tree, (list, tuple)):
    if len(prefix_tree) != len(full_tree):
      ty = type(prefix_tree)
      yield lambda name: ValueError(
          f"pytree structure error: different lengths of {ty.__name__} at key path\n"
          f"    {name}{keystr(key_path)}\n"
          f"At that key path, the prefix pytree {name} has a subtree of type "
          f"{ty.__name__} of length {len(prefix_tree)}, but the full pytree "
          f"has a subtree of the same type but of length {len(full_tree)}.")
      return  # 不再在这棵子树中查找更多错误
  else:
    # 接下来处理检查子键的一般情况。
    try:
      diff = set(prefix_tree_keys).symmetric_difference(set(full_tree_keys))
    except:
      diff = None
    if len(prefix_tree_children) != len(full_tree_children):
      yield lambda name: ValueError(
        "pytree structure error: different numbers of pytree children at key path\n"
        f"    {name}{keystr(key_path)}\n"
        f"At that key path, the prefix pytree {name} has a subtree of type\n"
        f"    {type(prefix_tree)}\n"
        f"with {len(prefix_tree_children)} child keys\n"
        f"    {' '.join(str(k) for k in prefix_tree_keys)}\n"
        f"but at the same key path the full pytree has a subtree of the same "
        f"type but with {len(full_tree_children)} child keys\n"
        f"    {' '.join(str(k) for k in full_tree_keys)}\n"
        + ("" if diff is None else
           f"so the symmetric difference on key sets is\n"
           f"    {' '.join(str(k) for k in diff)}"))
      return  # 不再在这棵子树中查找更多错误

  # 或者它们可能因为根节点的 pytree 元数据不同而不一致：
  if prefix_tree_meta != full_tree_meta:
    prefix_tree_meta_str = str(prefix_tree_meta)
    full_tree_meta_str = str(full_tree_meta)
    metadata_diff = textwrap.indent(
        "\n".join(
            difflib.ndiff(prefix_tree_meta_str.splitlines(),
                          full_tree_meta_str.splitlines())),
        prefix="    ")
    yield lambda name: ValueError(
      "pytree structure error: different pytree metadata at key path\n"
      f"    {name}{keystr(key_path)}\n"
      f"At that key path, the prefix pytree {name} has a subtree of type\n"
      f"    {type(prefix_tree)}\n"
      f"with metadata\n"
      f"    {prefix_tree_meta_str}\n"
      f"but at the same key path the full pytree has a subtree of the same "
      f"type but with metadata\n"
      f"    {full_tree_meta_str}\n"
      f"so the diff in the metadata at these pytree nodes is\n"
      f"{metadata_diff}")
    return  # 不再在这棵子树中查找更多错误

  # 如果根节点类型与子节点数量都一致，那么错误一定出现在某棵子树中，
  # 于是递归下去：
  assert prefix_tree_keys == full_tree_keys, \
    ("equal pytree nodes gave differing prefix_tree_keys: "
     f"{prefix_tree_keys} and {full_tree_keys}")
  for k, t1, t2 in zip(prefix_tree_keys, prefix_tree_children, full_tree_children):
    yield from _prefix_error((*key_path, k), t1, t2)

def _ensure_inbounds(allow_invalid: bool, num_args: int, argnums: Sequence[int]
                     ) -> tuple[int, ...]:
  """确保 argnum 在界内。同时解析负数 argnum。"""
  result = []
  for i in argnums:
    if i >= num_args and allow_invalid: continue
    if not -num_args <= i < num_args:
      raise ValueError(
          "Positional argument indices, e.g. for `static_argnums`, must have "
          "value greater than or equal to -len(args) and less than len(args), "
          f"but got value {i} for len(args) == {num_args}.")
    result.append(i % num_args)  # 解析负值
  return tuple(result)
