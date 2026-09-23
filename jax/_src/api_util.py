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

# 文件职责：为 `jax.jit` 等 API 提供参数规格解析与追踪调试信息的基础工具。
# 它负责校验并补全 `static_argnums`/`static_argnames`/`donate_argnums`/`donate_argnames`，
# 计算捐赠(donation)向量，判断静态实参是否可哈希，并在给定示例参数与函数签名时
# 构造 `core.DebugInfo`；同时提供把 pytree 参数展平为扁平实参的 `linear_util` 变换，
# 以及 pmap 风格轴规格(`out_axes`/`axis_resources`)与前缀匹配错误报告工具。

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import inspect
import operator
from functools import partial, lru_cache
import re
from typing import Any, NoReturn

from jax._src import core
from jax._src import config
from jax._src import dtypes
from jax._src.state.types import AbstractRef
from jax._src.tree_util import (
    PyTreeDef, tree_flatten, tree_unflatten, treedef_children,
    broadcast_prefix, prefix_errors, none_leaf_registry,
    broadcast_flattened_prefix_with_treedef, treedef_is_leaf, tree_structure,
    tracing_registry)
from jax._src import linear_util as lu
from jax._src.util import (safe_map, HashableFunction, Unhashable, safe_zip,
                           weakref_lru_cache)
from jax._src import traceback_util

traceback_util.register_exclusion(__file__)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

def _ensure_index(x: Any) -> int | tuple[int, ...]:
  """确保 x 是索引或索引元组。"""
  x = core.concrete_or_error(None, x, "expected a static index or sequence of indices.")
  try:
    return operator.index(x)
  except TypeError:
    return tuple(map(operator.index, x))

def _ensure_index_tuple(x: Any) -> tuple[int, ...]:
  """把 x 转换为索引元组。"""
  x = core.concrete_or_error(None, x, "expected a static index or sequence of indices.")
  try:
    return (operator.index(x),)
  except TypeError:
    return tuple(map(operator.index, x))

def _ensure_str(x: str) -> str:
  if not isinstance(x, str):
    raise TypeError(f"argument is not a string: {x}")
  return x

def _ensure_str_tuple(x: str | Iterable[str]) -> tuple[str, ...]:
  """把 x 转换为字符串元组。"""
  if isinstance(x, str):
    return (x,)
  else:
    return tuple(map(_ensure_str, x))

@lu.transformation_with_aux2
def flatten_fun(f: Callable, store: lu.Store,
                in_tree: PyTreeDef, *args_flat):
  py_args, py_kwargs = tree_unflatten(in_tree, args_flat)
  ans = f(*py_args, **py_kwargs)
  ans, out_tree = tree_flatten(ans)
  store.store(out_tree)
  return ans

@lu.transformation_with_aux2
def flatten_fun_nokwargs(f: Callable, store: lu.Store,
                         in_tree: PyTreeDef, *args_flat):
  py_args = tree_unflatten(in_tree, args_flat)
  ans = f(*py_args)
  ans, out_tree = tree_flatten(ans)
  store.store(out_tree)
  return ans

class _HashableWithStrictTypeEquality:
  """在把静态参数作为 jit 键进行比较时所用的装箱对象。

  要求使用 `is` 做精确类型相等判断，并使用值相等判断。"""
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return hash(self.val)

  def __eq__(self, other):
    return type(self.val) is type(other.val) and self.val == other.val

_POSITIONAL_ARGUMENTS = (
  inspect.Parameter.POSITIONAL_ONLY,
  inspect.Parameter.POSITIONAL_OR_KEYWORD
)

def _validate_argnums(sig: inspect.Signature, argnums: tuple[int, ...], argnums_name: str) -> None:
  """
  校验 argnums 对给定函数而言是否合理。

  对于接受可变数量位置参数（`f(..., *args)`）的函数，所有正数 argnums 都被视为有效。
  """
  n_pos_args = 0
  for param in sig.parameters.values():
    if param.kind in _POSITIONAL_ARGUMENTS:
      n_pos_args += 1

    elif param.kind is inspect.Parameter.VAR_POSITIONAL:
      # 位置参数的数量可以是任意的
      return

  if argnums and (-min(argnums) > n_pos_args or max(argnums) >= n_pos_args):
    raise ValueError(f"Jitted function has {argnums_name}={argnums}, "
                     f"but only accepts {n_pos_args} positional arguments.")

_INVALID_KEYWORD_ARGUMENTS = (
  inspect.Parameter.POSITIONAL_ONLY,
  inspect.Parameter.VAR_POSITIONAL
)


_KEYWORD_ARGUMENTS = (
  inspect.Parameter.POSITIONAL_OR_KEYWORD,
  inspect.Parameter.KEYWORD_ONLY,
)
def _validate_argnames(
    sig: inspect.Signature, argnames: tuple[str, ...], argnames_name: str
) -> None:
  """
  校验 argnames 对给定函数而言是否合理。

  对于接受可变数量关键字参数（`f(..., **kwargs)`）的函数，除被标记为仅位置（`f(pos_only, /, ...)`）
  的参数名之外，所有 argnames 都被视为有效。
  """
  var_kwargs = False
  valid_kwargs: set[str] = set()
  invalid_kwargs: set[str] = set()
  for param_name, param in sig.parameters.items():
    if param.kind in _KEYWORD_ARGUMENTS:
      valid_kwargs.add(param_name)

    elif param.kind is inspect.Parameter.VAR_KEYWORD:
      var_kwargs = True

    elif param.kind in _INVALID_KEYWORD_ARGUMENTS:
      invalid_kwargs.add(param_name)

  # 检查是否有 kwargs 因仅位置而无效
  if invalid_argnames := (invalid_kwargs & set(argnames)):
    raise ValueError(f"Jitted function has invalid argnames {invalid_argnames} "
                     f"in {argnames_name}. These are positional-only")

  # 接受任意 kwargs
  if var_kwargs:
    return

  # 检查所有 argnames 都存在于该函数上
  if invalid_argnames := (set(argnames) - valid_kwargs):
    raise ValueError(f"Jitted function has invalid argnames {invalid_argnames} "
                     f"in {argnames_name}. Function does not take these args.")


def argnums_partial(f: lu.WrappedFun, dyn_argnums: int | Sequence[int],
                    args: Sequence, require_static_args_hashable=True):
  dyn_argnums = _ensure_index_tuple(dyn_argnums)
  dyn_argnums = _ensure_inbounds(False, len(args), dyn_argnums)
  fixed_args: list
  if require_static_args_hashable:
    fixed_args = []
    for i, arg in enumerate(args):
      if i in dyn_argnums: continue
      if not is_hashable(arg):
        raise ValueError(
            "Non-hashable static arguments are not supported, as this can lead "
            f"to unexpected cache-misses. Static argument (index {i}) of type "
            f"{type(arg)} for function {f.__name__} is non-hashable.")
      fixed_args.append(_HashableWithStrictTypeEquality(arg))
  else:
    fixed_args = [Unhashable(arg) for i, arg in enumerate(args)
                  if i not in dyn_argnums]
  dyn_args = tuple(args[i] for i in dyn_argnums)
  return _argnums_partial(f, dyn_argnums, tuple(fixed_args)), dyn_args

def argnums_partial2(f: Callable, dyn_argnums: int | Sequence[int],
                     args: Sequence, kwargs: dict):
  # 类似 argnums_partial，但作用于可调用对象而不是 WrappedFun
  dyn_argnums = _ensure_index_tuple(dyn_argnums)
  dyn_argnums = _ensure_inbounds(False, len(args), dyn_argnums)
  static_args = list(args)
  dyn_args = []
  for i in dyn_argnums:
    x = static_args[i]
    dyn_args.append(x)
    static_args[i] = None

  def f_wrapped(*dyn_args_):
    args_ = list(static_args)
    for i, x in zip(dyn_argnums, dyn_args_):
      args_[i] = x
    return f(*args_, **kwargs)

  return f_wrapped, tuple(dyn_args)

def prepend_static_args(f, static_args):
  return _prepend_static_args(f, tuple(Unhashable(arg) for arg in static_args))


@lu.transformation2
def _prepend_static_args(f, static_args, *args, **kwargs):
  static_args = tuple(arg.val for arg in static_args)
  all_args = static_args + args
  return f(*all_args, **kwargs)


def _ensure_inbounds(allow_invalid: bool, num_args: int, argnums: Sequence[int]
                     ) -> tuple[int, ...]:
  """确保 argnum 在边界内。同时解析负的 argnums。"""
  result = []
  for i in argnums:
    if i >= num_args and allow_invalid: continue
    if not -num_args <= i < num_args:
      raise ValueError(
          "Positional argument indices, e.g. for `static_argnums`, must have "
          "value greater than or equal to -len(args) and less than len(args), "
          f"but got value {i} for len(args) == {num_args}.")
    result.append(i % num_args)  # 解析负索引
  return tuple(result)


@lu.transformation2
def _argnums_partial(_fun: Callable,
                     _dyn_argnums: Sequence[int],
                     _fixed_args: Sequence, *dyn_args, **kwargs):
  sentinel = object()
  args = [sentinel] * (len(_fixed_args) + len(dyn_args))
  for i, arg in zip(_dyn_argnums, dyn_args):
    args[i] = arg
  fixed_args_ = iter(_fixed_args)
  args = [next(fixed_args_).val if x is sentinel else x for x in args]
  assert next(fixed_args_, sentinel) is sentinel
  return _fun(*args, **kwargs)


@lru_cache(maxsize=4096)
def donation_vector(donate_argnums, donate_argnames, in_tree,
                    kws: bool = True) -> tuple[bool, ...]:
  """返回一个元组，为 args 与 kwargs 中的每个叶子给出一个布尔值。

  如果用户只指定了 donate_argnums 却以 kwargs 调用该函数，或者反过来，会怎样？此时在
  `resolve_argnums` 中会利用函数签名计算出与之对应的另一项（分别是 donate_argnames 或
  donate_argnums），因此调用本函数时 donate_argnums 与 donate_argnames 都可用。这使得
  JAX 在只指定 donate_argnums 时也能捐赠 kwargs，反之亦然。

  当 donate_argnums 与 donate_argnames 都被指定时，只有被指定到的 args 和
  kwargs 会被捐赠。
  """
  res: list[bool] = []
  if kws:
    args_tree, kwargs_tree = treedef_children(in_tree)
  else:
    args_tree, kwargs_tree = in_tree, None
  for i, arg in enumerate(args_tree.children()):
    donate = bool(i in donate_argnums)
    res.extend((donate,) * arg.num_leaves)
  if kwargs_tree is not None:
    for key, val in zip(kwargs_tree.node_data()[1], kwargs_tree.children()):  # pyrefly: ignore[unsupported-operation]
      donate = key in donate_argnames
      res.extend((donate,) * val.num_leaves)
  return tuple(res)

def rebase_donate_argnums(donate_argnums, static_argnums) -> tuple[int, ...]:
  """平移 donate 以计入 static。

  >>> rebase_donate_argnums((3, 4), (0, 1))
  (1, 2)

  Args:
    donate_argnums: 一个整数可迭代对象。
    static_argnums: 一个整数可迭代对象。

  Returns:
    一个由去重且排序后的整数值组成的元组，基于 donate_argnums，其中每个
    元素都做了偏移以计入 static_argnums。
  """
  if not (static_argnums or donate_argnums):
    return tuple(sorted(donate_argnums))

  static_argnums = sorted(set(static_argnums))
  donate_argnums = sorted(set(donate_argnums))
  i = j = o = 0
  out = []
  while j < len(donate_argnums):
    if i < len(static_argnums) and static_argnums[i] == donate_argnums[j]:
      raise ValueError(f"`static_argnums` {static_argnums} and "
                       f"`donate_argnums` {donate_argnums} cannot intersect.")

    if i < len(static_argnums) and static_argnums[i] < donate_argnums[j]:
      o += 1
      i += 1
    else:
      out.append(donate_argnums[j] - o)
      j += 1
  return tuple(out)


def is_hashable(arg):
  try:
    hash(arg)
    return True
  except TypeError:
    return False


class WrapHashably:
  val: Any
  hash: int
  hashable: bool

  def __init__(self, val):
    self.val = val
    try:
      self.hash = hash(val)
      self.hashable = True
    except:
      self.hash = id(val)
      self.hashable = False
  def __hash__(self):
    return self.hash
  def __eq__(self, other):
    if isinstance(other, WrapHashably):
      if self.hashable and other.hashable:
        return self.val == other.val
      else:
        return self.val is other.val
    return False

# 这种缓存有助于在使用 static_argnums 时也避免重复追踪。
# 参见 api_benchmark.py:bench_remat_eager_retracing_overheads_static_argnums。
# 在该基准测试中，加入这一缓存会带来约 10 倍的差异（若让被追踪的函数更大，
# 这一差异可以变得任意大）。
def dyn_args_fun(fun: Callable, static_argnums: frozenset[int],
                 static_args: tuple[WrapHashably, ...], nargs: int):
  if any(isinstance(x.val, core.Tracer) for x in static_args):
    return _dyn_args_fun_uncached(fun, static_argnums, static_args, nargs)
  return _dyn_args_fun_cached(fun, static_argnums, static_args, nargs)

def _dyn_args_fun_uncached(fun: Callable, static_argnums: frozenset[int],
                           static_args: tuple[WrapHashably, ...], nargs: int):
  def new_fun(*dyn_args, **kwargs):
    static_args_, dyn_args_ = iter(static_args), iter(dyn_args)
    full_args = [next(static_args_).val if i in static_argnums
                 else next(dyn_args_) for i in range(nargs)]
    return fun(*full_args, **kwargs)
  new_fun.__name__ = getattr(fun, '__name__', '<unnamed function>')
  return new_fun

_dyn_args_fun_cached = weakref_lru_cache(_dyn_args_fun_uncached)


SENTINEL = object()


def flatten_axes(name, treedef, axis_tree, *, kws=False, tupled_args=False):
  # 给定轴规格树 axis_tree（一棵叶子为整数与 None 的 pytree，即把 None 也当作叶子），
  # 它是给定 treedef 的树前缀，构造一棵结构相同、完整的轴规格树并返回展平后的结果
  axis_tree_leaves, axis_treedef = none_leaf_registry.flatten(axis_tree)
  try:
    axes = broadcast_flattened_prefix_with_treedef(
        axis_tree_leaves, axis_treedef, treedef)
  except ValueError:
    if kws:
      # 如果树中包含关键字参数，我们只把错误消息调整成针对位置参数的形式
      treedef, _ = treedef_children(treedef)
      axis_tree, _ = axis_tree
    hint = ""
    if tupled_args:
      hint += (f" Note that {name} that are non-trivial pytrees should always be "
               f"wrapped in a tuple representing the argument list.")
      if len(treedef.children()) == 1:
        try:
          flatten_axes(name, treedef, (axis_tree,))
        except ValueError:
          pass  # 问题不在这里。
        else:
          hint += (f" In particular, you're passing in a single argument which "
                   f"means that {name} might need to be wrapped in "
                   f"a singleton tuple.")
    dummy_tree = tree_unflatten(treedef, [PytreeLeaf()] * treedef.num_leaves)
    errors = prefix_errors(axis_tree, dummy_tree)
    if errors:
      details = "\n  ".join(e(name).args[0] for e in errors)
      prefix_err_msg = (
          f"\n  Mismatch details ({len(errors)} found):\n  {details}")
      raise ValueError(
          f"{name} specification must be a tree prefix of the "
          f"corresponding value; {hint}{prefix_err_msg}") from None
    # 到这里说明没能找到树前缀错误。
    assert False, "unreachable code"
  assert len(axes) == treedef.num_leaves
  return axes

class PytreeLeaf:
  def __repr__(self):
    return "pytree leaf"


def flatten_axis_resources(what, tree, shardings, tupled_args):
  try:
    return tuple(flatten_axes(what, tree, shardings, tupled_args=tupled_args))
  except ValueError:
    pass  # 在下面抛出树前缀错误

  # 树的叶子总是合法前缀，因此如果这里假设的前缀错误确实发生，axis_resources
  # 就一定不是叶子。
  assert not treedef_is_leaf(tree_structure(shardings))

  # 直接检查类型而不是用 isinstance，因为要处理 namedtuple。
  if tupled_args and (type(shardings) is not tuple or
                      len(shardings) != len(tree.children())):
    # 我们知道 axis_resources 本应是与会话参数元组对应的元组，但它虽然是非叶 pytree，
    # 要么不是元组，要么长度不对。
    msg = (f"{what} specification must be a tree prefix of the positional "
           f"arguments tuple. In particular, {what} must either be a Sharding, "
           "a PartitionSpec, or a tuple of length equal to the number of "
           "positional arguments.")
    # 如果 `tree` 表示参数元组，那么 `axis_resources` 就必须是元组。
    # TODO(mattjj,apaszke): 禁用隐式列表转换，删除下面的 'or list'
    if type(shardings) is not tuple:
      msg += f" But {what} is not a tuple: got {type(shardings)} instead."
    elif len(shardings) != len(tree.children()):
      msg += (f" But {what} is the wrong length: got a tuple or list of length "
              f"{len(shardings)} for an args tuple of length "
              f"{len(tree.children())}.")

    # 作为额外提示，检查一下用户是不是只是忘了把 shardings 包进单元素元组。
    if len(tree.children()) == 1:
      try: flatten_axes(what, tree, (shardings,))
      except ValueError: pass  # 问题不在这里。
      else:
        msg += (f" Given the corresponding argument being "
                f"passed, it looks like {what} might need to be wrapped in "
                f"a singleton tuple.")

    raise ValueError(msg)

  axis_tree = shardings

  # 因为这里只有 `tree` 这个 treedef 而非完整的 pytree，我们构造一棵虚拟树来比较。
  # 是否要修改调用方？
  dummy_tree = tree_unflatten(tree, [PytreeLeaf()] * tree.num_leaves)
  errors = prefix_errors(axis_tree, dummy_tree)
  if errors:
    details = "\n".join(e(what).args[0] for e in errors)
    raise ValueError(
        f"Mismatch details ({len(errors)} found):\n{details}"
    )

  # 到这里说明没能找到树前缀错误。
  assert False, "Please open a bug report!"  # 这里本应不可达。


def flat_out_axes(
    f: lu.WrappedFun, out_spec: Any
) -> tuple[lu.WrappedFun, Callable]:
  leaves, treedef = tree_flatten(out_spec)
  f, out_axes = _flat_out_axes(f, tuple(leaves), treedef)
  return f, HashableFunction(out_axes, closure=(tuple(leaves), treedef))

@lu.transformation_with_aux2
def _flat_out_axes(_fun, _store, _leaves, _treedef, *args, **kwargs):
  ans = _fun(*args, **kwargs)
  spec = tree_unflatten(_treedef, _leaves)
  try:
    spec_flat = tuple(broadcast_prefix(spec, ans, is_leaf=lambda x: x is None))
  except ValueError:
    e, *_ = prefix_errors(spec, ans)
    # TODO(mattjj): 目前是硬编码用于 pmap 的；后续工作中要推广到 vmap
    msg, = e('pmap out_axes').args
    msg += ("\n\nThe full pytree is the output of the pmapped function. Ensure "
            "that the `out_axes` argument to `pmap` is a pytree prefix of the "
            "pmapped function's output.")
    raise ValueError(msg) from None
  _store.store(spec_flat)
  return ans

def check_callable(fun):
  # 在 Python 3.10+ 中，唯一阻碍我们支持 staticmethod 的原因是
  # 无法对它们取弱引用，而 C++ JIT 需要弱引用。
  if isinstance(fun, staticmethod):
    raise TypeError(f"staticmethod arguments are not supported, got {fun}")
  if not callable(fun):
    raise TypeError(f"Expected a callable value, got {fun}")
  if inspect.isgeneratorfunction(fun):
    raise TypeError(f"Expected a function, got a generator function: {fun}")

_POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD

def infer_argnums_and_argnames(
    sig: inspect.Signature,
    argnums: int | Iterable[int] | None,
    argnames: str | Iterable[str] | None,
  ) -> tuple[tuple[int, ...], tuple[str, ...]]:
  """用 inspect 为函数推断缺失的 argnums 与 argnames。"""
  if argnums is None and argnames is None:
    return (), ()

  if argnums is not None and argnames is not None:
    argnums = _ensure_index_tuple(argnums)
    argnames = _ensure_str_tuple(argnames)
    return argnums, argnames

  parameters = sig.parameters
  if argnums is None:
    assert argnames is not None
    argnames = _ensure_str_tuple(argnames)
    argnums = tuple(
        i for i, (k, param) in enumerate(parameters.items())
        if param.kind == _POSITIONAL_OR_KEYWORD and k in argnames
    )
  else:
    argnums = _ensure_index_tuple(argnums)
    argnames = tuple(
        k for i, (k, param) in enumerate(parameters.items())
        if param.kind == _POSITIONAL_OR_KEYWORD and i in argnums
    )

  return argnums, argnames


def resolve_argnums(
    fun: Callable,
    signature: inspect.Signature | None,
    donate_argnums: int | Sequence[int] | None,
    donate_argnames: str | Iterable[str] | None,
    static_argnums: int | Sequence[int] | None,
    static_argnames: str | Iterable[str] | None,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
  """校验并补全 jit 的 argnum/argname 规格。

  * 补齐任何缺失的部分（例如由名称给出编号，或反过来），
  * 依据函数签名校验参数名/编号，
  * 校验被捐赠的参数与静态参数没有交集。
  """
  if signature is None:
    # 有些内置函数不支持签名。
    # 参见：https://github.com/python/cpython/issues/73485
    # 这种情况下不做任何校验
    static_argnums = () if static_argnums is None else _ensure_index_tuple(
        static_argnums)
    static_argnames = () if static_argnames is None else _ensure_str_tuple(
        static_argnames)
    donate_argnums = () if donate_argnums is None else _ensure_index_tuple(
        donate_argnums)
    if donate_argnames is not None:
      raise ValueError(f"Getting the signature of function {fun} failed. "
                       "Pass donate_argnums instead of donate_argnames.")
    assert donate_argnames is None
    donate_argnames = ()
  else:
    # 按 docstring 推断 argnums 与 argnames
    # 如果 nums 为 None 而 names 不为 None，则从 names 推断 nums，反之亦然。
    static_argnums, static_argnames = infer_argnums_and_argnames(
        signature, static_argnums, static_argnames)
    donate_argnums, donate_argnames = infer_argnums_and_argnames(
        signature, donate_argnums, donate_argnames)

    # 校验
    _validate_argnums(signature, static_argnums, "static_argnums")
    _validate_argnames(signature, static_argnames, "static_argnames")
    _validate_argnums(signature, donate_argnums, "donate_argnums")
    _validate_argnames(signature, donate_argnames, "donate_argnames")

  # 补偿静态 argnums 吸收掉的参数
  _assert_no_intersection(static_argnames, donate_argnames)
  return donate_argnums, donate_argnames, static_argnums, static_argnames


def _assert_no_intersection(static_argnames, donate_argnames):
  out = set(static_argnames).intersection(set(donate_argnames))
  if out:
    raise ValueError(
        "static_argnames and donate_argnames cannot intersect. Argument names "
        f"{out} appear in both static_argnames and donate_argnames")


def resolve_kwargs(fun: Callable, args, kwargs) -> tuple[Any, ...]:
  """按照函数签名把输入参数解析为位置参数。

  如果调用方传入了任何仅关键字参数，本函数会抛出 TypeError。
  """
  if isinstance(fun, partial):
    # functools.partial 应具有不透明签名。
    fun = lambda *args, **kwargs: None
  ba = inspect.signature(fun).bind(*args, **kwargs)
  ba.apply_defaults()
  if ba.kwargs:
    passed_kwargs = [k for k in ba.kwargs if k in kwargs]
    if passed_kwargs:
      raise TypeError(
          "The following keyword arguments could not be resolved to positions: "
          f"{', '.join(passed_kwargs)}"
      )
  return ba.args


def _dtype(x):
  try:
    return dtypes.result_type(x)
  except ValueError:
    return dtypes.result_type(getattr(x, 'dtype'))


# 这个装饰器存在的目的是让 JAX 中的 API 更易于被猴子补丁替换。
# 默认情况下它什么都不做，但可以被猴子补丁改成做其他事情。
def api_hook(fun, tag: str):
  return fun


def debug_info(
    traced_for: str,
    fun: Callable,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
    result_paths_thunk: Callable[[], tuple[str, ...]] | core.InitialResultPaths = core.initial_result_paths,
    # TODO(necula): 检查我们是否真的需要这个，例如为了加快追踪速度？
    sourceinfo: str | None = None,
    signature: inspect.Signature | None = None,
) -> core.DebugInfo:
  """根据示例参数 args 和 kwargs 为函数构造 core.DebugInfo。

  `args` 和 `kwargs` 是示例位置参数与关键字参数，与 `inspect.Signature` 一起
  用来得到参数的名称。出于追踪目的被视为静态的参数也应包含在内，并用
  `static_argnums` 和 `static_argnames` 指定。

  参见 linear_util.DebugInfo 的文档字符串。
  """
  res = getattr(fun, "__fun_debug_info__", None)
  if res is not None:
    return res
  if sourceinfo is None:
    sourceinfo = fun_sourceinfo(fun)
  if signature is None:
    signature = fun_signature(fun)
  arg_names = _non_static_arg_names(signature, args, kwargs, static_argnums,
                                    static_argnames)
  return core.DebugInfo(traced_for, sourceinfo, arg_names, result_paths_thunk)


def fun_signature(fun: Callable) -> inspect.Signature | None:
  try:
    return inspect.signature(fun)
  except (ValueError, TypeError):
    return None

def save_wrapped_fun_debug_info(wrapper: Callable,
                                dbg: core.DebugInfo) -> None:
  setattr(wrapper, "__fun_debug_info__", dbg)

_fun_name_re = re.compile(r"(?:<built-in function (\S+)>)")

# TODO(mattjj): 把这个函数改成该模块内部使用
def fun_sourceinfo(fun: Callable) -> str:
  # 参见 DebugInfo.fun_src_info
  while isinstance(fun, partial):
    fun = fun.func
  fun = inspect.unwrap(fun)
  try:
    filename = fun.__code__.co_filename
    lineno = fun.__code__.co_firstlineno
    return f"{fun.__name__} at {filename}:{lineno}"
  except AttributeError:
    try:
      fun_str = str(fun)
    except:
      return "<unknown>"
    # 按照约定，函数名中不含空格；另外我们还想避免生成形如
    # "<object Foo at 0x1234>" 的 fun_sourceinfo，因为它会让降级变得不确定。
    if m := _fun_name_re.match(fun_str):
      return m.group(1)
    return "<unknown>"

def _non_static_arg_names(fn_signature: inspect.Signature | None,
                          args: Sequence[Any], kwargs: Mapping[str, Any],
                          static_argnums: Sequence[int],
                          static_argnames: Sequence[str],
                          ) -> tuple[str, ...]:
  """返回非静态参数的名称。

  如果给定了 `fn_signature`，我们就从中取得顶层参数的名称。在其他情况下，
  包括 `args` 与 `kwargs` 和签名不匹配时，我们使用 `args[0]`、`args[1]` 之类的名称。
  """
  # 使用与 jit 相同的参数解析方式：先是位置参数，然后是按键排序的 kwargs。
  static = object()
  static_argnums_ = _ensure_inbounds(True, len(args), static_argnums)
  static_argnames_ = set(static_argnames)
  args_ = [static if i in static_argnums_ else x for i, x in enumerate(args)]
  kwargs_ = {k: static if k in static_argnames_ else x for k, x in kwargs.items()}
  ordered_args: Sequence[tuple[str, Any]] | None = None
  if fn_signature is not None:
    try:
      ba = fn_signature.bind(*args_, **kwargs_)
    except (ValueError, TypeError):
      pass
    else:
      # 是否有 **kwargs
      kwargs_name = next((name for name, p in fn_signature.parameters.items()
                          if p.kind == inspect.Parameter.VAR_KEYWORD), None)
      # 位置参数是那些既没有按关键字传入、也没有通过 **kwargs 传入的参数。
      positional = [(name, x) for name, x in ba.arguments.items()
                    if name not in kwargs and name != kwargs_name]
      # 关键字参数按实际 kwarg 的关键字排序后传入
      sorted_kwargs = sorted(((name, x) for name, x in kwargs_.items()),
                              key=lambda name_x: name_x[0])
      sorted_kwargs = [(name if name in ba.arguments else f"{kwargs_name}['{name}']",
                        x)
                       for name, x in sorted_kwargs]
      ordered_args = positional + sorted_kwargs

  if ordered_args is None:
    positional = [("args", args_)]
    keyword = sorted([(f"kwargs['{name}']", x) for name, x in kwargs_.items() if x is not static],
                     key=lambda name_x: name_x[0])
    ordered_args = positional + keyword

  return tuple(f'{name}{lu._clean_keystr_arg_names(path)}'
               for name, x in ordered_args
               for path, l in tracing_registry.flatten_with_path(x)[0]
               if l is not static)

# TODO(mattjj): 让这个函数更快
def check_no_aliased_ref_args(dbg_fn: Callable[[], core.DebugInfo],
                              maybe_avals, args) -> None:
  assert config.mutable_array_checks.value
  refs: dict[int, int] = {}
  for i, (a, x) in enumerate(zip(maybe_avals, args)):
    if (isinstance(a, AbstractRef) and
        (dup_idx := refs.setdefault(id(core.get_referent(x)), i)) != i):
      dbg = dbg_fn()
      raise ValueError(
        "only one reference to a mutable array may be passed as an argument "
        f"to a function, but when tracing {dbg.func_src_info} for {dbg.traced_for} "
        f"the mutable array reference of type {a.str_short()} appeared at both "
        f"{dbg.arg_names[dup_idx] if dbg.arg_names is not None else 'unknown'} "
        f"and {dbg.arg_names[i] if dbg.arg_names is not None else 'unknown'}."
        if dbg else
        f"at both flat index {dup_idx} and flat index {i}") from None

def _check_no_aliased_closed_over_refs(dbg: core.DebugInfo, consts, args) -> None:
  assert config.mutable_array_checks.value
  refs: set[int] = {id(core.get_referent(c)) for c in consts
                    if isinstance(core.typeof(c), AbstractRef)}
  for i, x in enumerate(args):
    if id(core.get_referent(x)) in refs:
      a = core.shaped_abstractify(x)
      raise ValueError(
          f"when tracing {dbg.func_src_info} for {dbg.traced_for}, a mutable "
          f"array reference of type {a.str_short()} was both closed over and "
          f"passed as the argument "
          f"{dbg.safe_arg_names(len(args))[i]}" if dbg else "at flat index {i}")

def check_no_transformed_refs_args(dbg_fn: Callable[[], core.DebugInfo],
                                   args_flat) -> None:
  from jax._src.state.types import TransformedRef
  for i, arg in enumerate(args_flat):
    if isinstance(arg, TransformedRef):
      dbg = dbg_fn()
      raise TypeError(
        f"When tracing {dbg.func_src_info} for {dbg.traced_for}, "
        "TransformedRefs are not allowed, but got a TransformedRef with name "
        f"{dbg.arg_names[i] if dbg.arg_names is not None else 'unknown'}."
        if dbg else
        "TransformedRefs are not allowed in this context.") from None

class InternalFloatingPointError(Exception):
  name: str
  ty: str

  def __init__(self, name: str, ty: str):
    self.name = name
    self.ty = ty

def maybe_recursive_nan_check(
    e: Exception, fun: Callable, args, kwargs
) -> NoReturn:
  print("Invalid nan value encountered in the output of a jax.jit "
        "function. Calling the de-optimized version.")
  try:
    _ = fun(*args, **kwargs)
  except (FloatingPointError, ZeroDivisionError) as e2:
    raise e2 from None
  else:
    _raise_no_nan_in_deoptimized(e)


def _raise_no_nan_in_deoptimized(e) -> NoReturn:
  msg = (f"{str(e)}. Because "
        "jax_config.debug_nans.value and/or config.jax_debug_infs is set, the "
        "de-optimized function (i.e., the function as if the `jit` "
        "decorator were removed) was called in an attempt to get a more "
        "precise error message. However, the de-optimized function did not "
        "produce invalid values during its execution. This behavior can "
        "result from `jit` optimizations causing the invalid value to be "
        "produced. It may also arise from having nan/inf literals as "
        "inputs or outputs, like `jax.jit(lambda ...: jax.numpy.nan)(...)`. "
        "\n\n"
        "It may be possible to avoid the invalid value by removing the "
        "`jit` decorator, at the cost of losing optimizations. "
        "\n\n"
        "If you see this error, consider opening a bug report at "
        "https://github.com/jax-ml/jax.")
  raise FloatingPointError(msg) from None
