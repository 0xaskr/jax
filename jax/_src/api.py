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
# 文件职责：`jax/_src/api.py` 是 JAX 面向用户的变换与工具层，负责把内部变换包装成
# `jax.jit` / `jax.grad` / `jax.vmap` / `jax.jvp` / `jax.jacfwd` / `jax.jacrev` /
# `jax.hessian` / `jax.linearize` 等公开 API，并处理 Python 容器（pytree）形式的参数与
# 输出。它同时承担这些 API 的选项校验、缓存与追踪边界管理（`api_boundary` 统一包装
# 并复现报错），并注册 NaN/Inf 调试钩子；它位于用户层与 `pjit`、`ad`、`batching`、
# `partial_eval`、`pxla` 等解释器之间，是 JAX 变换语义对外的统一入口。

"""JAX 面向用户的变换与工具。

这里的变换大多是对内部变换的包装，提供便于控制行为的选项，并处理 Python 容器形式的
参数与输出。所处理的 Python 容器是 pytree（见 tree_util.py），其中包含嵌套的
tuple/list/dict，其叶子为数组。
"""
from __future__ import annotations

import gc
import sys
import os
import atexit
import collections
from collections.abc import Callable, Hashable, Iterable, Sequence
import dataclasses
import enum
from functools import partial
import inspect
from typing import Any, Literal, overload, cast, TYPE_CHECKING
import weakref

import numpy as np
from contextlib import contextmanager

from jax._src import api_util
from jax._src import linear_util as lu
from jax._src import flattree as ft
from jax._src.tree_util import (
    tree_map, tree_flatten, tree_unflatten, tree_structure, tree_transpose,
    tree_leaves, Partial, PyTreeDef, keystr, generate_key_paths,
    tree_flatten_with_path, equality_errors_pytreedef, register_pytree_node,
    register_dataclass, treedef_is_strict_leaf, broadcast_prefix)
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import array
from jax._src import basearray
from jax._src import distributed
from jax._src import dtypes
from jax._src.dtypes import canonicalize_value
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import pjit
from jax._src import xla_bridge as xb
from jax._src.core import eval_jaxpr, shaped_abstractify, ShapedArray, typeof
from jax._src.api_util import (
  flatten_fun_nokwargs,
  flatten_axes, _ensure_index, check_callable, debug_info, argnums_partial2)
from jax._src.lib import jax_jit
from jax._src.lib import _jax
from jax._src.lib import xla_client as xc
from jax._src.sharding import Sharding
from jax._src.mesh import get_concrete_mesh, get_abstract_mesh, Mesh
from jax._src.sharding_impls import PartitionSpec as P, NamedSharding
from jax._src.layout import Format
from jax._src.traceback_util import api_boundary
from jax._src import tree_util
from jax._src.util import unzip2, safe_map, safe_zip, wraps
from jax._src import util
from jax._src.ad_util import a2tz

from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import pxla

config_ext = _jax.config


traceback_util.register_exclusion(__file__)

_dtype = dtypes.dtype

AxisName = Hashable

Device = xc.Device

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

ShapeDtypeStruct = core.ShapeDtypeStruct

class Inline(enum.Enum):
  """枚举，用于指定嵌套 jit 函数的内联行为。

  Values:
    JAX_EARLY: 在 JAX 追踪期间内联到外层 JAXPR 中。
    JAX_LATE: 在 JAX 到 MLIR 的降级期间内联。
    XLA_EARLY: 保留在 MLIR 中，并标记为由 XLA 编译器尽早内联。
    XLA_LATE: 保留在 MLIR 中，并标记为由 XLA 编译器推迟内联。
    AUTO: 由 JAX 与 XLA 自动决定的内联策略。
  """
  JAX_EARLY = "jax_early"
  JAX_LATE = "jax_late"
  XLA_EARLY = "xla_early"
  XLA_LATE = "xla_late"
  AUTO = "auto"

@api_boundary
def _nan_check_posthook(fun, args, kwargs, output):
  """由 C++ 版 jit/pmap 调用的钩子函数，用于执行 NaN 检查。"""
  buffers = []
  for leaf in tree_leaves(output):
    if hasattr(leaf, "addressable_shards"):
      buffers.extend([shard.data for shard in leaf.addressable_shards])

  try:
    dispatch.check_special(pjit.jit_p.name, buffers)
  except api_util.InternalFloatingPointError as e:
    assert config.debug_nans.value or config.debug_infs.value
    if hasattr(fun, '_fun'):
      f = fun._fun
      if getattr(f, '_apply_primitive', False):
        raise FloatingPointError(f"invalid value ({e.ty}) encountered in {f.__qualname__}") from None
      # 这种情况下只有 compiled_fun 会抛出异常
      api_util.maybe_recursive_nan_check(e, f, args, kwargs)
      raise AssertionError("Unreachable") from e
    else:
      # TODO(emilyaf): 不应该需要这个回退。
      raise

_post_hook_state = config_ext.Config[Callable | None](
    "post_hook", None, include_in_jit_key=False
)
jax_jit.set_post_hook_state(_post_hook_state)

def _update_debug_special_global(_):
  if config._read("jax_debug_nans") or config._read("jax_debug_infs"):
    _post_hook_state.set_global(_nan_check_posthook)
  else:
    _post_hook_state.set_global(None)

def _update_debug_special_thread_local(_):
  if (config.debug_nans.get_local() == True or
      config.debug_infs.get_local() == True):
    _post_hook_state.set_local(_nan_check_posthook)
  else:
    _post_hook_state.set_local(None)

config.debug_nans._add_hooks(_update_debug_special_global,
                             _update_debug_special_thread_local)
config.debug_infs._add_hooks(_update_debug_special_global,
                             _update_debug_special_thread_local)


float0 = dtypes.float0

class NotSpecified:
  """供 jax.jit 使用的哨兵值"""
  def __repr__(self):
    return "<not-specified>"

@overload
def jit(
  fun: Callable, /, *,
  in_shardings: Any = ...,
  out_shardings: Any = ...,
  static_argnums: int | Sequence[int] | None = ...,
  static_argnames: str | Iterable[str] | None = ...,
  donate_argnums: int | Sequence[int] | None = ...,
  donate_argnames: str | Iterable[str] | None = ...,
  keep_unused: bool = ...,
  device: xc.Device | None = ...,
  backend: str | None = ...,
  inline: bool | Inline = ...,
  compiler_options: dict[str, Any] | None = ...,
) -> pjit.JitWrapped:
  ...

@overload
def jit(
  *,
  in_shardings: Any = ...,
  out_shardings: Any = ...,
  static_argnums: int | Sequence[int] | None = ...,
  static_argnames: str | Iterable[str] | None = ...,
  donate_argnums: int | Sequence[int] | None = ...,
  donate_argnames: str | Iterable[str] | None = ...,
  keep_unused: bool = ...,
  device: xc.Device | None = ...,
  backend: str | None = ...,
  inline: bool | Inline = ...,
  compiler_options: dict[str, Any] | None = ...,
) -> Callable[[Callable], pjit.JitWrapped]:
  ...

def jit(
  fun: Callable | NotSpecified = NotSpecified(), /, *,
  in_shardings: Any = sharding_impls.UNSPECIFIED,
  out_shardings: Any = sharding_impls.UNSPECIFIED,
  static_argnums: int | Sequence[int] | None = None,
  static_argnames: str | Iterable[str] | None = None,
  donate_argnums: int | Sequence[int] | None = None,
  donate_argnames: str | Iterable[str] | None = None,
  keep_unused: bool = False,
  device: xc.Device | None = None,
  backend: str | None = None,
  inline: bool | Inline = False,
  compiler_options: dict[str, Any] | None = None,
) -> pjit.JitWrapped | Callable[[Callable], pjit.JitWrapped]:
  """为 ``fun`` 设置基于 XLA 的即时编译。

  Args:
    fun: 要 jit 的函数。``fun`` 应当是纯函数。
      ``fun`` 的参数与返回值应当是数组、标量，或它们构成的（嵌套）标准 Python 容器
      （tuple/list/dict）。由 ``static_argnums`` 指定的位置参数可以是任意可哈希类型。
      静态参数会作为编译缓存键的一部分，因此必须定义哈希与相等运算符。JAX 会持有
      ``fun`` 的弱引用以用作编译缓存键，所以对象 ``fun`` 必须是可弱引用的。从 JAX
      v0.8.1 起，若省略 ``fun``，返回值将是一个部分求值的函数，以支持装饰器工厂
      写法（见下方 Examples）。
    in_shardings: 可选，一个 :py:class:`Sharding`，或叶子为 :py:class:`Sharding` 的
      pytree，其结构是 ``fun`` 位置参数元组的树前缀。若提供，传给 ``fun`` 的位置参数
      其分片必须与 ``in_shardings`` 兼容，否则会抛出错误，且编译后的计算具有与
      ``in_shardings`` 对应的输入分片。若不提供，编译后计算的输入分片由参数的分片
      推断得到。
    out_shardings: 可选，一个 :py:class:`Sharding`，或叶子为 :py:class:`Sharding` 的
      pytree，其结构是 ``fun`` 输出的树前缀。若提供，其效果等同于对 ``fun`` 的输出
      应用 :py:func:`jax.lax.with_sharding_constraint`。
    static_argnums: 可选，一个 int 或 int 的集合，指定哪些位置参数被视为静态
      （追踪期与编译期常量）。

      静态参数应当可哈希，即同时实现了 ``__hash__`` 与 ``__eq__``，并且不可变。除此
      之外，它们可以是任意 Python 对象。用不同的值调用这些常量会使 jit 后的函数重新
      编译。不属于类数组对象或其容器的参数必须标记为静态。

      若 ``static_argnums`` 与 ``static_argnames`` 都未提供，则没有参数被视为静态。
      若未提供 ``static_argnums`` 但提供了 ``static_argnames``，或反之，JAX 会用
      :code:`inspect.signature(fun)` 找出与 ``static_argnames`` 对应的位置参数
      （或反之）。若 ``static_argnums`` 与 ``static_argnames`` 都提供，则不会使用
      ``inspect.signature``，只有 ``static_argnums`` 或 ``static_argnames`` 中实际
      列出的参数才会被视为静态。
    static_argnames: 可选，一个字符串或字符串的集合，指定哪些具名参数被视为静态
      （编译期常量）。详见 ``static_argnums`` 的说明。若未提供但设置了
      ``static_argnums``，则默认通过调用 ``inspect.signature(fun)`` 找出对应的具名
      参数。
    donate_argnums: 可选，int 的集合，指定哪些位置参数的缓冲区可被计算覆写并在调用方
      标记为已删除。若计算开始后你不再需要这些参数缓冲区，那么捐赠它们是安全的。
      在某些情况下，XLA 可以利用捐赠的缓冲区来减少执行计算所需的内存量，例如回收
      某个输入缓冲区来存放结果。你不应再使用已捐赠给计算的缓冲区；若这样做 JAX 会
      抛出错误。默认不捐赠任何参数缓冲区。

      若 ``donate_argnums`` 与 ``donate_argnames`` 都未提供，则不捐赠任何参数。
      若未提供 ``donate_argnums`` 但提供了 ``donate_argnames``，或反之，JAX 会用
      :code:`inspect.signature(fun)` 找出与 ``donate_argnames`` 对应的位置参数
      （或反之）。若 ``donate_argnums`` 与 ``donate_argnames`` 都提供，则不会使用
      ``inspect.signature``，只有 ``donate_argnums`` 或 ``donate_argnames`` 中实际
      列出的参数才会被捐赠。

      关于缓冲区捐赠的更多细节见
      `FAQ <https://docs.jax.dev/en/latest/faq.html#buffer-donation>`_。
    donate_argnames: 可选，一个字符串或字符串的集合，指定哪些具名参数被捐赠给计算。
      详见 ``donate_argnums`` 的说明。若未提供但设置了 ``donate_argnums``，则默认通过
      调用 ``inspect.signature(fun)`` 找出对应的具名参数。
    keep_unused: 可选布尔值。若为 `False`（默认值），JAX 判定为 `fun` 未使用的参数
      *可能* 会从生成的编译后 XLA 可执行文件中被丢弃。这类参数既不会被传输到设备，
      也不会提供给底层可执行文件。若为 `True`，未使用的参数不会被剪除。
    device: 这是实验性特性，API 很可能会变化。
      可选，jit 后函数将在其上运行的设备。（可用设备可通过 :py:func:`jax.devices`
      获取。）默认值继承自 XLA 的 DeviceAssignment 逻辑，通常等价于使用
      ``jax.devices()[0]``。
    backend: 这是实验性特性，API 很可能会变化。
      可选，表示 XLA 后端的字符串：``cpu``、``gpu`` 或 ``tpu``。
    inline: 可选，布尔值或 :class:`jax.Inline` 实例，指定嵌套 jit 函数的内联策略。
      可传布尔值（``True`` 表示 ``jax.Inline.JAX_EARLY``，``False`` 表示
      ``jax.Inline.AUTO``），也可传 :class:`jax.Inline` 枚举成员。默认为 ``False``
      （即 ``jax.Inline.AUTO``）。

  Returns:
    经过包装的 ``fun`` 版本，已设置好即时编译。

  Examples:
    在下面的例子中，``selu`` 可被 XLA 编译成一个融合 kernel：

    >>> import jax
    >>>
    >>> @jax.jit
    ... def selu(x, alpha=1.67, lmbda=1.05):
    ...   return lmbda * jax.numpy.where(x > 0, x, alpha * jax.numpy.exp(x) - alpha)
    >>>
    >>> key = jax.random.key(0)
    >>> x = jax.random.normal(key, (10,))
    >>> print(selu(x))  # doctest: +SKIP
    [-0.54485  0.27744 -0.29255 -0.91421 -0.62452 -0.24748
    -0.85743 -0.78232  0.76827  0.59566 ]

    从 JAX v0.8.1 起，:func:`jit` 支持用装饰器工厂写法来指定可选关键字参数：

    >>> @jax.jit(static_argnames=['n'])
    ... def g(x, n):
    ...   for i in range(n):
    ...     x = x ** 2
    ...   return x
    >>>
    >>> g(jnp.arange(4), 3)
    Array([   0,    1,  256, 6561], dtype=int32)

    为兼容较旧的 JAX 版本，一种常见写法是使用 :func:`functools.partial`：

    >>> from functools import partial
    >>>
    >>> @partial(jax.jit, static_argnames=['n'])
    ... def g(x, n):
    ...   for i in range(n):
    ...     x = x ** 2
    ...   return x
    >>>
    >>> g(jnp.arange(4), 3)
    Array([   0,    1,  256, 6561], dtype=int32)
  """
  kwds = dict(
      in_shardings=in_shardings, out_shardings=out_shardings,
      static_argnums=static_argnums, static_argnames=static_argnames,
      donate_argnums=donate_argnums, donate_argnames=donate_argnames,
      keep_unused=keep_unused, device=device, backend=backend, inline=inline,
      compiler_options=compiler_options, use_resource_env=False)
  if isinstance(fun, NotSpecified):
    return lambda fun: pjit.make_jit(fun, **kwds)
  else:
    return pjit.make_jit(fun, **kwds)

if not TYPE_CHECKING:
  # TODO(slebedev): 这里本应当是一个装饰器，但那样似乎会让
  # pytype 忽略这些重载
  jit = api_boundary(jit, repro_api_name="jax.jit")


@contextmanager
def disable_jit(disable: bool = True):
  """上下文管理器，在其动态上下文中禁用 :py:func:`jit` 行为。

  调试时，有一个能在动态上下文中到处禁用 :py:func:`jit` 的机制很有用。注意这不仅会禁用
  用户对 :func:`jit` 的显式使用，还会移除 JAX 库使用的任何隐式 JIT 编译：这包括
  传给 :func:`~jax.lax.scan` 与 :func:`~jax.lax.while_loop` 等高阶原语的 `body` 与
  `cond` 函数的隐式 JIT 计算、:mod:`jax.numpy` 函数实现中使用的 JIT，以及任何在 API
  实现内部使用 :func:`jit` 的情形。但请注意，即使在 `disable_jit` 下，单个原语运算
  仍会像普通的逐运算即时执行那样由 XLA 编译。

  与 jit 函数的参数存在数据依赖的值会被追踪并抽象化。例如，抽象值可以是
  :py:class:`ShapedArray` 实例，它表示所有具有给定形状与数据类型的可能数组的集合，
  而不表示某个具有具体值的具体数组。如果你在 jit 函数中使用良性的带副作用操作
  （例如打印），就可能见到这类抽象值：

  >>> import jax
  >>>
  >>> @jax.jit
  ... def f(x):
  ...   y = x * 2
  ...   print("Value of y is", y)
  ...   return y + 3
  ...
  >>> print(f(jax.numpy.array([1, 2, 3])))
  Value of y is JitTracer(int32[3])
  [5 7 9]

  这里 ``y`` 已被 :py:func:`jit` 抽象为 :py:class:`ShapedArray`，它表示一个形状与类型
  固定但取值任意的数组。``y`` 的取值同样被追踪。如果我们想在调试时看到具体值并同时
  避开追踪器，可以使用 :py:func:`disable_jit` 上下文管理器：

  >>> import jax
  >>>
  >>> with jax.disable_jit():
  ...   print(f(jax.numpy.array([1, 2, 3])))
  ...
  Value of y is [2 4 6]
  [5 7 9]
  """
  with config.disable_jit(disable):
    yield


@partial(api_boundary, repro_api_name="jax.grad")
def grad(fun: Callable, argnums: int | Sequence[int] = 0,
         has_aux: bool = False, holomorphic: bool = False,
         allow_int: bool = False,
         reduce_axes: Sequence[AxisName] = ()) -> Callable:
  """创建一个求 ``fun`` 梯度的函数。

  Args:
    fun: 要被微分的函数。由 ``argnums`` 指定位置上的参数应当是数组、标量，或标准
      Python 容器。由 ``argnums`` 指定位置上的参数数组必须是非精确（即浮点或复数）
      类型。它应当返回标量（包括形状为 ``()`` 的数组，但不包括形状为 ``(1,)`` 等
      的数组）。
    argnums: 可选，int 或 int 序列。指定对哪些位置参数求导（默认 0）。
    has_aux: 可选，bool。表示 ``fun`` 是否返回一个二元组，其第一个元素被视为要微分
      的数学函数的输出，第二个元素是辅助数据。默认 False。
    holomorphic: 可选，bool。表示 ``fun`` 是否被保证为全纯。若为 True，输入与输出
      必须是复数。默认 False。
    allow_int: 可选，bool。是否允许对整数值的输入求导。整数输入的梯度将具有平凡的
      向量空间数据类型（float0）。默认 False。

  Returns:
    一个与 ``fun`` 参数相同的函数，用于求 ``fun`` 的梯度。若 ``argnums`` 是整数，
    梯度与该整数所指定位置参数的形状与类型相同。若 argnums 是整数元组，梯度是一个
    值的元组，其形状与类型与对应的参数相同。若 ``has_aux`` 为 True，则返回
    （梯度, 辅助数据）二元组。

  例如：

  >>> import jax
  >>>
  >>> grad_tanh = jax.grad(jax.numpy.tanh)
  >>> print(grad_tanh(0.2))
  0.961043
  """
  if reduce_axes:
    raise NotImplementedError("reduce_axes argument to grad is deprecated")
  del reduce_axes
  value_and_grad_f = value_and_grad(fun, argnums, has_aux=has_aux,
                                    holomorphic=holomorphic,
                                    allow_int=allow_int)

  docstr = ("Gradient of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "gradient, which has the same shape as the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def grad_f(*args, **kwargs):
    _, g = value_and_grad_f(*args, **kwargs)
    return g

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def grad_f_aux(*args, **kwargs):
    (_, aux), g = value_and_grad_f(*args, **kwargs)
    return g, aux

  return grad_f_aux if has_aux else grad_f


@partial(api_boundary, repro_api_name="jax.value_and_grad")
def value_and_grad(fun: Callable, argnums: int | Sequence[int] = 0,
                   has_aux: bool = False, holomorphic: bool = False,
                   allow_int: bool = False, reduce_axes: Sequence[AxisName] = ()
  ) -> Callable[..., tuple[Any, Any]]:
  """创建一个同时求 ``fun`` 及其梯度的函数。

  Args:
    fun: 要被微分的函数。由 ``argnums`` 指定位置上的参数应当是数组、标量，或标准
      Python 容器。它应当返回标量（包括形状为 ``()`` 的数组，但不包括形状为 ``(1,)``
      等的数组）。
    argnums: 可选，int 或 int 序列。指定对哪些位置参数求导（默认 0）。
    has_aux: 可选，bool。表示 ``fun`` 是否返回一个二元组，其第一个元素被视为要微分
      的数学函数的输出，第二个元素是辅助数据。默认 False。
    holomorphic: 可选，bool。表示 ``fun`` 是否被保证为全纯。若为 True，输入与输出
      必须是复数。默认 False。
    allow_int: 可选，bool。是否允许对整数值的输入求导。整数输入的梯度将具有平凡的
      向量空间数据类型（float0）。默认 False。

  Returns:
    一个与 ``fun`` 参数相同的函数，它同时求 ``fun`` 与 ``fun`` 的梯度并以二元组
    （两元素元组）返回。若 ``argnums`` 是整数，梯度与该整数所指定位置参数的形状与
    类型相同。若 argnums 是整数序列，梯度是一个值的元组，其形状与类型与对应的参数
    相同。若 ``has_aux`` 为 True，则返回 ((值, 辅助数据), 梯度) 元组。
  """
  from jax._src.lax import lax as lax_internal  # pyrefly: ignore[missing-import]

  if reduce_axes:
    raise NotImplementedError("reduce_axes argument to grad is deprecated")
  del reduce_axes

  docstr = ("Value and gradient of {fun} with respect to positional "
            "argument(s) {argnums}. Takes the same arguments as {fun} but "
            "returns a two-element tuple where the first element is the value "
            "of {fun} and the second element is the gradient, which has the "
            "same shape as the arguments at positions {argnums}.")

  check_callable(fun)
  argnums = core.concrete_or_error(_ensure_index, argnums)

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def value_and_grad_f(*args, **kwargs):
    max_argnum = argnums if isinstance(argnums, int) else max(argnums)
    if max_argnum >= len(args):
      raise TypeError(f"differentiating with respect to {argnums=} requires at least "
                      f"{max_argnum + 1} positional arguments to be passed by the caller, "
                      f"but got only {len(args)} positional arguments.")
    f_partial, dyn_args = argnums_partial2(fun, argnums, args, kwargs)
    for leaf in tree_leaves(dyn_args):
      _check_input_dtype_grad(holomorphic, allow_int, leaf)
    ans, vjp_py, *maybe_aux = vjp(f_partial, *dyn_args, has_aux=has_aux)
    _check_scalar(ans)
    tree_map(partial(_check_output_dtype_grad, holomorphic), ans)
    g = vjp_py(lax_internal._one_vjp(ans))
    g = g[0] if isinstance(argnums, int) else g
    ans_aux = (ans, *maybe_aux) if has_aux else ans
    return ans_aux, g

  return value_and_grad_f

def _check_scalar(x):
  msg = "Gradient only defined for scalar-output functions. Output {}.".format
  try:
    aval = core.typeof(x)
  except TypeError as e:
    raise TypeError(msg(f"was {x}")) from e
  else:
    if isinstance(aval, ShapedArray):
      if aval.shape != ():
        if aval.size == 1:
          idx = " or output[0]" if aval.ndim == 1 else ""
          extract = f"extract the scalar with output.reshape(()){idx}, "
        else:
          extract = ""
        hint = (f" To get the gradient, {extract}reduce the output to a "
                "scalar with output.sum(), or use jax.jacobian to "
                "differentiate a function with a non-scalar output.")
        raise TypeError(msg(f"had shape: {aval.shape}") + hint)
    else:
      raise TypeError(msg(f"had abstract value {aval}"))

def _check_input_dtype_revderiv(name, holomorphic, allow_int, x):
  dispatch.check_arg(x)
  aval = core.typeof(x)
  if _is_ref_aval(aval):
    raise TypeError(
        f"{name} cannot differentiate with respect to a Ref-typed argument, "
        "because the gradient for a ref must itself be accumulated into a "
        "ref. Instead, use `jax.vjp` and bind a gradient ref using the "
        "returned VJP function's `with_refs` method. (Ref arguments not "
        f"selected by {name}'s argnums are fine: they are treated as "
        "plumbing, and are not differentiated.)")
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError(f"{name} with holomorphic=True requires inputs with complex dtype, "
                      f"but got {aval.dtype.name}.")
  if isinstance(aval, ShapedArray):
    if (dtypes.issubdtype(aval.dtype, dtypes.extended) or
        dtypes.issubdtype(aval.dtype, np.integer) or
        dtypes.issubdtype(aval.dtype, np.bool_)):
      if not allow_int:
        raise TypeError(f"{name} requires real- or complex-valued inputs (input dtype "
                        f"that is a sub-dtype of np.inexact), but got {aval.dtype.name}. "
                        "If you want to use Boolean- or integer-valued inputs, use vjp "
                        "or set allow_int to True.")
    elif not dtypes.issubdtype(aval.dtype, np.inexact):
      raise TypeError(f"{name} requires numerical-valued inputs (input dtype that is a "
                      f"sub-dtype of np.bool_ or np.number), but got {aval.dtype.name}.")
_check_input_dtype_grad = partial(_check_input_dtype_revderiv, "grad")

def _check_output_dtype_revderiv(name, holomorphic, x):
  aval = core.typeof(x)
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    raise TypeError(
        f"{name} with output element type {aval.dtype.name}")
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError(f"{name} with holomorphic=True requires outputs with complex dtype, "
                      f"but got {aval.dtype.name}.")
  elif dtypes.issubdtype(aval.dtype, np.complexfloating):
    raise TypeError(f"{name} requires real-valued outputs (output dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For holomorphic differentiation, pass holomorphic=True. "
                    "For differentiation of non-holomorphic functions involving complex "
                    "outputs, use jax.vjp directly.")
  elif not dtypes.issubdtype(aval.dtype, np.floating):
    raise TypeError(f"{name} requires real-valued outputs (output dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For differentiation of functions with integer outputs, use "
                    "jax.vjp directly.")
_check_output_dtype_grad = partial(_check_output_dtype_revderiv, "grad")

@partial(api_boundary, repro_api_name="jax.fwd_and_bwd")
def fwd_and_bwd(
    fun: Callable, argnums: int | Sequence[int], has_aux: bool = False,
    jitted: bool = True,
) -> tuple[Callable, Callable]:
  """创建与给定函数 ``fun`` 的前向和反向传播相对应的函数 ``fwd`` 与 ``bwd``。前向函数
  ``fwd(*args)`` 在功能上很像 ``y, fun_vjp = jax.vjp(fun, *args)``，但允许在多次迭代
  中复用反向函数 ``bwd``，这在前向与反向最终没有落在同一个 jit 函数中时有助于避免
  重新编译：

  >>> import jax
  >>>
  >>> x = W = cot_out = jax.numpy.ones((4,4))
  >>>
  >>> def f(x, W):
  ...     return x @ W
  ...
  >>> f_jitted = jax.jit(f)
  >>> for i in range(3):
  ...     y, f_vjp = jax.vjp(f_jitted, x, W)
  ...     cot_x, cot_W = f_vjp(cot_out)           # not jitted
  ...     cot_x, cot_W = jax.jit(f_vjp)(cot_out)  # recompiles on every iteration
  ...
  >>> fwd, bwd = jax.fwd_and_bwd(f, argnums=(0,1))
  >>> for i in range(3):
  ...     y, residuals = fwd(x, W)
  ...     cot_x, cot_W = bwd(residuals, cot_out)  # jitted, compiles once
  ...

  Args:
    fun: 要生成其前向与反向的函数。
    argnums: 整数或整数序列。指定对哪些位置参数求导。
    has_aux: 可选，bool。表示 ``fun`` 是否返回一个二元组，其第一个元素被视为要微分
     的数学函数的输出，第二个元素是辅助数据。默认 False。
    jitted: 可选，bool。表示是否返回前向与反向的 ``jax.jit``。注意只对反向而不对前向
      做 jit 会导致反向在每次调用时重新编译，因此我们默认对两者都做 jit。

  Returns:
    两个函数，``fwd`` 与 ``bwd``。

    若 ``has_aux`` 为 ``False``，``fwd(*primals)`` 返回元组
    ``(primals_out, residuals)``，其中 ``primals_out`` 即 ``fun(*primals)``。
    若 ``has_aux`` 为 ``True``，则返回 ``(primals_out, residuals, aux)`` 元组，
    其中 ``aux`` 是 ``fun`` 返回的辅助数据。

    ``bwd`` 是一个函数，它接收 ``residuals`` 以及与 ``primals_out`` 形状相同的余切
    向量，返回一个余切向量元组，其个数与形状与 ``argnums`` 指定的 ``primals`` 相同，
    表示 ``fun`` 在 ``primals`` 处求值的向量-雅可比乘积。
  """
  check_callable(fun)
  argnums = _ensure_index(argnums)

  def fwd(*args, **kwargs):
    f_partial, dyn_args = argnums_partial2(fun, argnums, args, {})
    return vjp(f_partial, *dyn_args, has_aux=has_aux)
  def bwd(f_vjp, outgrad):
    g = f_vjp(outgrad)
    g = g[0] if isinstance(argnums, int) else g
    return g
  if jitted:
    fwd = jit(fwd)
    bwd = jit(bwd)
  return fwd, bwd


@partial(api_boundary, repro_api_name="jax.jacfwd")
def jacfwd(fun: Callable, argnums: int | Sequence[int] = 0,
           has_aux: bool = False, holomorphic: bool = False) -> Callable:
  """使用前向模式 AD 逐列求值的 ``fun`` 的 Jacobian（雅可比矩阵）。

  Args:
    fun: 需要计算其 Jacobian 的函数。
    argnums: 可选，整数或整数序列。
      指定对哪些位置参数求导（默认为 ``0``）。
    has_aux: 可选，bool。指示 ``fun`` 是否返回一个二元组，
      其中第一个元素被视为待求导的数学函数的输出，
      第二个元素是辅助数据。默认为 False。
    holomorphic: 可选，bool。
      指示 ``fun`` 是否保证为全纯。默认为 False。

  Returns:
    一个与 ``fun`` 具有相同参数的函数，它使用前向模式自动微分计算
    ``fun`` 的 Jacobian。如果 ``has_aux`` 为 True，
    则返回 (jacobian, auxiliary_data) 二元组。

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.jacfwd(f)(jnp.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  check_callable(fun)
  argnums = _ensure_index(argnums)

  docstr = ("Jacobian of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "jacobian of the output with respect to the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  def jacfun(*args, **kwargs):
    f_partial, dyn_args = argnums_partial2(fun, argnums, args, kwargs)
    tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
    pushfwd: Callable = partial(_jvp, f_partial, dyn_args, has_aux=has_aux)
    if has_aux:
      y, jac, aux = vmap(pushfwd, out_axes=(None, -1, None))(_std_basis(dyn_args))
    else:
      y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
      aux = None
    tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
    if not has_aux:
      return jac_tree
    else:
      return jac_tree, aux

  return jacfun

def _check_input_dtype_jacfwd(holomorphic: bool, x: Any) -> None:
  dispatch.check_arg(x)
  aval = core.typeof(x)
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    raise TypeError(
        f"jacfwd with input element type {aval.dtype.name}")
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError("jacfwd with holomorphic=True requires inputs with complex "
                      f"dtype, but got {aval.dtype.name}.")
  elif not dtypes.issubdtype(aval.dtype, np.floating):
    raise TypeError("jacfwd requires real-valued inputs (input dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For holomorphic differentiation, pass holomorphic=True. "
                    "For differentiation of non-holomorphic functions involving "
                    "complex inputs or integer inputs, use jax.jvp directly.")

def _check_output_dtype_jacfwd(holomorphic, x):
  aval = core.typeof(x)
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError("jacfwd with holomorphic=True requires outputs with complex dtype, "
                      f"but got {aval.dtype.name}.")

@partial(api_boundary, repro_api_name="jax.jacrev")
def jacrev(fun: Callable, argnums: int | Sequence[int] = 0,
           has_aux: bool = False, holomorphic: bool = False,
           allow_int: bool = False) -> Callable:
  """使用反向模式 AD 逐行求值的 ``fun`` 的 Jacobian（雅可比矩阵）。

  Args:
    fun: 需要计算其 Jacobian 的函数。
    argnums: 可选，整数或整数序列。
      指定对哪些位置参数求导（默认为 ``0``）。
    has_aux: 可选，bool。指示 ``fun`` 是否返回一个二元组，
      其中第一个元素被视为待求导的数学函数的输出，
      第二个元素是辅助数据。默认为 False。
    holomorphic: 可选，bool。
      指示 ``fun`` 是否保证为全纯。默认为 False。
    allow_int: 可选，bool。是否允许对整数值输入求导。
      整数输入的梯度具有平凡的向量空间数据类型
      （float0）。默认为 False。

  Returns:
    一个与 ``fun`` 具有相同参数的函数，它使用反向模式自动微分计算
    ``fun`` 的 Jacobian。如果 ``has_aux`` 为 True，
    则返回 (jacobian, auxiliary_data) 二元组。

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.jacrev(f)(jnp.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  check_callable(fun)

  docstr = ("Jacobian of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "jacobian of the output with respect to the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  def jacfun(*args, **kwargs):
    f_partial, dyn_args = argnums_partial2(fun, argnums, args, kwargs)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
    y, pullback, *maybe_aux = vjp(f_partial, *dyn_args, has_aux=has_aux)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac_tree = tree_map(partial(_jacrev_unravel, y), example_args, jac)
    jac_tree = tree_transpose(tree_structure(example_args), tree_structure(y), jac_tree)
    return (jac_tree, *maybe_aux) if has_aux else jac_tree

  return jacfun


def jacobian(fun: Callable, argnums: int | Sequence[int] = 0,
             has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> Callable:
  """是 :func:`jax.jacrev` 的别名。"""
  return jacrev(fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic, allow_int=allow_int)


_check_input_dtype_jacrev = partial(_check_input_dtype_revderiv, "jacrev")
_check_output_dtype_jacrev = partial(_check_output_dtype_revderiv, "jacrev")


@partial(api_boundary, repro_api_name="jax.hessian")
def hessian(fun: Callable, argnums: int | Sequence[int] = 0,
            has_aux: bool = False, holomorphic: bool = False) -> Callable:
  """以稠密数组形式给出的 ``fun`` 的 Hessian（黑塞矩阵）。

  Args:
    fun: 需要计算其 Hessian 的函数。其位于 ``argnums``
      指定位置的参数应为数组、标量，或由它们构成的标准 Python 容器。
      它应返回数组、标量，或由它们构成的标准 Python 容器。
    argnums: 可选，整数或整数序列。
      指定对哪些位置参数求导（默认为 ``0``）。
    has_aux: 可选，bool。
      指示 ``fun`` 是否返回一个二元组，
      其中第一个元素被视为待求导的数学函数的输出，
      第二个元素是辅助数据。默认为 False。
    holomorphic: 可选，bool。
      指示 ``fun`` 是否保证为全纯。默认为 False。

  Returns:
    一个与 ``fun`` 具有相同参数的函数，
    它计算 ``fun`` 的 Hessian。

  >>> import jax
  >>>
  >>> g = lambda x: x[0]**3 - 2*x[0]*x[1] - x[1]**6
  >>> print(jax.hessian(g)(jax.numpy.array([1., 2.])))
  [[   6.   -2.]
   [  -2. -480.]]

  :py:func:`hessian` 是通常 Hessian 定义的一种推广，
  它支持以嵌套 Python 容器（即 pytree）作为输入和输出。
  ``jax.hessian(fun)(x)`` 的树结构由 ``fun(x)``
  的结构与 ``x`` 的结构的两份副本的树乘积构成。
  两个树结构的树乘积的构成方式是：
  把第一个树的每个叶子替换为第二个树的一份副本。例如：

  >>> import jax.numpy as jnp
  >>> f = lambda dct: {"c": jnp.power(dct["a"], dct["b"])}
  >>> print(jax.hessian(f)({"a": jnp.arange(2.) + 1., "b": jnp.arange(2.) + 2.}))
  {'c': {'a': {'a': Array([[[ 2.,  0.], [ 0.,  0.]],
                           [[ 0.,  0.], [ 0., 12.]]], dtype=float32),
               'b': Array([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                           [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32)},
         'b': {'a': Array([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                           [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32),
               'b': Array([[[0.      , 0.      ], [0.      , 0.      ]],
                           [[0.      , 0.      ], [0.      , 3.843624]]], dtype=float32)}}}

  因此，``jax.hessian(fun)(x)`` 树结构中的每个叶子对应于
  ``fun(x)`` 的一个叶子与 ``x`` 的一对叶子。
  对于 ``jax.hessian(fun)(x)`` 中的每个叶子，如果 ``fun(x)``
  中对应的数组叶子形状为 ``(out_1, out_2, ...)``，而 ``x``
  中对应的数组叶子形状分别为 ``(in_1_1, in_1_2, ...)`` 和
  ``(in_2_1, in_2_2, ...)``，那么该 Hessian 叶子的形状为
  ``(out_1, out_2, ..., in_1_1, in_1_2, ..., in_2_1, in_2_2, ...)``。
  换句话说，Python 树结构表示 Hessian 的分块结构，
  其中分块由输入和输出 pytree 决定。

  特别地，当函数输入 ``x`` 和输出 ``fun(x)`` 各为单个数组时
  （不涉及任何 pytree），得到的就是一个数组，如上面的 ``g`` 例子所示。
  如果 ``fun(x)`` 的形状为 ``(out1, out2, ...)``，``x`` 的形状为
  ``(in1, in2, ...)``，那么 ``jax.hessian(fun)(x)`` 的形状为
  ``(out1, out2, ..., in1, in2, ..., in1, in2, ...)``。要把 pytree
  展平为一维向量，可以考虑使用 :py:func:`jax.flatten_util.flatten_pytree`。
  """
  return jacfwd(jacrev(fun, argnums, has_aux=has_aux, holomorphic=holomorphic),
                argnums, has_aux=has_aux, holomorphic=holomorphic)

def _insert_pvary(basis, leaf):
  if not config._check_vma.value or not config.auto_pcast.value:
    return basis
  return core.pvary(basis, tuple(core.typeof(leaf).mat.varying))

def _std_basis(pytree):
  import jax.numpy as jnp  # pyrefly: ignore[missing-import]
  leaves, _ = tree_flatten(pytree)
  ndim = sum(map(np.size, leaves))
  dtype = dtypes.result_type(*leaves)
  flat_basis = jnp.eye(ndim, dtype=dtype)
  axis = 1
  arr_s = [None] * flat_basis.ndim
  specs = [P(*arr_s[:axis], *core.typeof(l).sharding.spec, *arr_s[axis+1:])
           for l in leaves]
  out_pytree = _unravel_array_into_pytree(pytree, axis, None, flat_basis, specs)
  out_pytree = tree_map(_insert_pvary, out_pytree, pytree)
  return out_pytree

def _jacfwd_unravel(input_pytree, output_pytree_leaf, arr):
  axis = -1 % arr.ndim
  arr_s = core.typeof(arr).sharding.spec
  specs = tree_map(
      lambda l: P(*arr_s[:axis], *[None] * len(np.shape(l)), *arr_s[axis+1:]),
      input_pytree)
  return _unravel_array_into_pytree(
    input_pytree, axis, output_pytree_leaf, arr, specs)

def _jacrev_unravel(output_pytree, input_pytree_leaf, arr):
  specs = tree_map(
      lambda l: P(*[None] * len(np.shape(l)), *core.typeof(arr).sharding.spec[1:]),
      output_pytree)
  return _unravel_array_into_pytree(
    output_pytree, 0, input_pytree_leaf, arr, specs)

def _possible_downcast(x, example, spec):
  from jax._src.lax import lax as lax_internal  # pyrefly: ignore[missing-import]
  if (dtypes.issubdtype(x.dtype, np.complexfloating) and
      not dtypes.issubdtype(_dtype(example), np.complexfloating)):
    x = x.real
  dtype = _dtype(example)
  weak_type = dtypes.is_weakly_typed(example)
  sharding = NamedSharding(core.typeof(example).sharding.mesh, spec)
  return lax_internal._convert_element_type(
      x, dtype, weak_type, sharding=sharding)

def _unravel_array_into_pytree(pytree, axis, example, arr, specs):
  """把数组拆解（unravel）为具有给定结构的 PyTree。
  Args:
      pytree: 提供结构的 pytree。
      axis: 参数 axis 取值为 -1、
        0 或 1。它控制结果的形状。
      example: 如果指定，则把各分量转换为匹配的 dtype/weak_type，
        否则在 example 为 None 时使用 pytree 叶子的类型。
      arr: 待拆解的数组。
  """
  leaves, treedef = tree_flatten(pytree)
  specs, _ = tree_flatten(specs)
  shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis+1:] for l in leaves]
  parts = _split(arr, np.cumsum(map(np.size, leaves[:-1])), axis)
  reshaped_parts = [
      _possible_downcast(np.reshape(x, shape),
                         leaf if example is None else example,
                         spec=spec)
      for x, shape, leaf, spec in zip(parts, shapes, leaves, specs)]
  return tree_unflatten(treedef, reshaped_parts)

def _split(x, indices, axis):
  if isinstance(x, np.ndarray):
    return np.split(x, indices, axis)
  else:
    return x._split(indices, axis)


@partial(api_boundary, repro_api_name="jax.vmap")
def vmap[F: Callable](
    fun: F,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    sum_match: bool = False
    ) -> F:
  """向量化映射。创建一个把 ``fun`` 映射到参数轴上的函数。

  Args:
    fun: 要在额外轴上映射的函数。
    in_axes: 一个整数、None，或值的序列，
      指定要映射哪些输入数组轴。

      如果 ``fun`` 的每个位置参数都是数组，
      那么 ``in_axes`` 可以是整数、None，或由整数和 None
      组成的元组，其长度等于 ``fun`` 的位置参数个数。
      整数或 ``None`` 表示为所有参数映射哪个数组轴
      （``None`` 表示不映射任何轴），
      而元组表示为每个对应的位置参数映射哪个轴。
      轴整数必须在每个数组的 ``[-ndim, ndim)``
      范围内，其中 ``ndim`` 是相应输入数组的维度（轴）数。

      如果 ``fun`` 的位置参数是容器（pytree）类型，
      ``in_axes`` 必须是长度等于 ``fun`` 位置参数个数的序列，
      并且对每个参数，``in_axes`` 中对应的元素可以是具有匹配
      pytree 结构的容器，用来指定其容器元素的映射方式。
      换句话说，``in_axes`` 必须是由传给 ``fun``
      的位置参数元组构成的容器树前缀。更多细节见以下链接：
      https://docs.jax.dev/en/latest/pytrees.html#applying-optional-parameters-to-pytrees

      必须显式提供 ``axis_size``，
      或者至少有一个位置参数的 ``in_axes`` 不为 None。
      所有被映射的位置参数，其映射输入轴的大小必须全部相等。

      以关键字形式传入的参数总是沿其首轴
      （即轴索引 0）映射。

      示例见下文。

    out_axes: 一个整数、None，或由它们构成的（嵌套）
      标准 Python 容器（tuple/list/dict），
      指示映射轴应出现在输出中的位置。
      所有带映射轴的输出都必须有非
      None 的 ``out_axes`` 指定（但请参见下面的
      ``sum_match``）。对于每个输出数组，轴整数必须在
      ``[-ndim, ndim)`` 范围内，其中 ``ndim`` 是被
      :func:`vmap` 处理的函数所返回数组的维度（轴）数，
      它比 ``fun`` 返回的相应数组的维度（轴）数多一。
    axis_name: 可选，一个可哈希的 Python 对象，
      用于标识被映射的轴，以便应用并行集合通信。
    axis_size: 可选，一个整数，指示要映射的轴的大小。
      如果未提供，则根据参数推断映射轴的大小。
    sum_match: 可选，一个布尔值（默认为 ``False``），改变
      ``out_axes`` 指定为 ``None`` 的输出的处理方式。
      默认情况下，如果这类输出沿映射轴发生变化，
      就会报错。当 ``sum_match=True`` 时，
      这类输出改为沿映射轴求和；不沿映射轴变化的输
      出照常原样返回。这在自动微分场景中很有用，
      因为沿映射轴求和正是沿该轴广播的转置。
      例如：

      >>> from jax import vmap
      >>> import jax.numpy as jnp
      >>> vmap(lambda x: 2. * x, out_axes=None, sum_match=True)(jnp.arange(3.))
      Array(6., dtype=float32)

  Returns:
    ``fun`` 的批处理/向量化版本，其参数与 ``fun`` 的参数对应，
    但在 ``in_axes`` 指示的位置上带有额外的数组轴；
    其返回值与 ``fun`` 的返回值对应，
    但在 ``out_axes`` 指示的位置上带有额外的数组轴。

  例如，我们可以用向量点积实现矩阵-
  矩阵乘积：

  >>> import jax.numpy as jnp
  >>>
  >>> vv = lambda x, y: jnp.vdot(x, y)  #  ([a], [a]) -> []
  >>> mv = vmap(vv, (0, None), 0)      #  ([b,a], [a]) -> [b]      (b is the mapped axis)
  >>> mm = vmap(mv, (None, 1), 1)      #  ([b,a], [a,c]) -> [b,c]  (c is the mapped axis)

  这里我们用 ``[a,b]`` 表示形状为
  (a,b) 的数组。下面是一些变体：

  >>> mv1 = vmap(vv, (0, 0), 0)   #  ([b,a], [b,a]) -> [b]        (b is the mapped axis)
  >>> mv2 = vmap(vv, (0, 1), 0)   #  ([b,a], [a,b]) -> [b]        (b is the mapped axis)
  >>> mm2 = vmap(mv2, (1, 1), 0)  #  ([b,c,a], [a,c,b]) -> [c,b]  (c is the mapped axis)

  下面是一个在 ``in_axes`` 中使用容器类型的例子，
  用来指定要映射容器元素的哪些轴：

  >>> A, B, C, D = 2, 3, 4, 5
  >>> x = jnp.ones((A, B))
  >>> y = jnp.ones((B, C))
  >>> z = jnp.ones((C, D))
  >>> def foo(tree_arg):
  ...   x, (y, z) = tree_arg
  ...   return jnp.dot(x, jnp.dot(y, z))
  >>> tree = (x, (y, z))
  >>> print(foo(tree))
  [[12. 12. 12. 12. 12.]
   [12. 12. 12. 12. 12.]]
  >>> from jax import vmap
  >>> K = 6  # batch size
  >>> x = jnp.ones((K, A, B))  # batch axis in different locations
  >>> y = jnp.ones((B, K, C))
  >>> z = jnp.ones((C, D, K))
  >>> tree = (x, (y, z))
  >>> vfoo = vmap(foo, in_axes=((0, (1, 2)),))
  >>> print(vfoo(tree).shape)
  (6, 2, 5)

  下面是另一个在 ``in_axes`` 中使用容器类型的例子，
  这次用的是字典，用来指定要映射的容器元素：

  >>> dct = {'a': 0., 'b': jnp.arange(5.)}
  >>> x = 1.
  >>> def foo(dct, x):
  ...  return dct['a'] + dct['b'] + x
  >>> out = vmap(foo, in_axes=({'a': None, 'b': 0}, None))(dct, x)
  >>> print(out)
  [1. 2. 3. 4. 5.]

  向量化函数的结果可以是已映射的或未映射的。例如，
  下面的函数返回一个二元组，其中第一个元素已映射，
  第二个元素未映射。只有对未映射的结果，我们才能把
  ``out_axes`` 指定为 ``None``（以使其保持未映射）。

  >>> print(vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=(0, None))(jnp.arange(2.), 4.))
  (Array([4., 5.], dtype=float32), 8.0)

  如果为未映射的结果指定了 ``out_axes``，
  该结果会沿映射轴广播：

  >>> print(vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=0)(jnp.arange(2.), 4.))
  (Array([4., 5.], dtype=float32), Array([8., 8.], dtype=float32, weak_type=True))

  如果为已映射的结果指定了 ``out_axes``，
  该结果会相应转置。

  最后，这里是一个把 ``axis_name`` 与集合通信一起使用的例子：

  >>> xs = jnp.arange(3. * 4.).reshape(3, 4)
  >>> print(vmap(lambda x: lax.psum(x, 'i'), axis_name='i')(xs))
  [[12. 15. 18. 21.]
   [12. 15. 18. 21.]
   [12. 15. 18. 21.]]

  涉及集合通信的更多示例请参见 :py:func:`jax.pmap` 的文档字符串。
  """
  check_callable(fun)
  docstr = ("Vectorized version of {fun}. Takes similar arguments as {fun} "
            "but with additional array axes over which {fun} is mapped.")
  if fun.__doc__:
    docstr += "\n\nOriginal documentation:\n\n"
    docstr += fun.__doc__

  axis_name = core.no_axis_name if axis_name is None else axis_name
  if spmd_axis_name is not None and not isinstance(spmd_axis_name, tuple):
    spmd_axis_name = (spmd_axis_name,)

  if isinstance(in_axes, list):
    # 要成为位置参数元组的树前缀，in_axes 绝不能是列表：如果 in_axes
    # 不是叶子，它必须是树的元组。然而，在此类情况下用户期望元组和
    # 列表基本上可以互换使用，因此我们在这里把列表规范化为元组，
    # 而不是抛出错误。
    # https://github.com/jax-ml/jax/issues/2367
    in_axes = tuple(in_axes)

  from jax._src import hijax  # pyrefly: ignore[missing-module-attribute]
  if not (in_axes is None or type(in_axes) in {int, tuple, *batching.spec_types}
          or isinstance(in_axes, hijax.MappingSpec)):
    raise TypeError("vmap in_axes must be an int, None, or a tuple of entries corresponding "
                    f"to the positional arguments passed to the function, but got {in_axes}.")
  if not all(type(l) in {int, *batching.spec_types} or isinstance(l, hijax.MappingSpec)
             for l in tree_leaves(in_axes)):
    raise TypeError("vmap in_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {in_axes}.")
  if not all(type(l) in {int, *batching.spec_types} or isinstance(l, hijax.MappingSpec)
             for l in tree_leaves(out_axes)):
    raise TypeError("vmap out_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {out_axes}.")

  @wraps(fun, docstr=docstr)
  @api_boundary
  def vmap_f(*args, **kwargs):
    nonlocal spmd_axis_name
    if isinstance(in_axes, tuple) and len(in_axes) != len(args):
      msg = ("vmap in_axes must be an int, None, or a tuple of entries "
             "corresponding to the positional arguments passed to the "
             f"function, but got {len(in_axes)=}, {len(args)=}.")
      if kwargs:
        msg += (" Note that the function was called with keyword arguments "
                f"{list(kwargs)}, which the entries of a tuple in_axes never "
                "correspond to: keyword arguments are always mapped along "
                "their leading axis (axis 0). If that is the intended "
                "mapping, make in_axes correspond to the positional "
                "arguments only; otherwise pass the arguments positionally, "
                "or bind unmapped keyword arguments with functools.partial "
                "before applying vmap.")
      raise ValueError(msg)

    args_flat, in_tree  = tree_flatten((args, kwargs), is_leaf=batching.is_vmappable)
    dbg = debug_info("vmap", fun, args, kwargs)
    api_util.check_no_transformed_refs_args(lambda: dbg, args_flat)
    f = lu.wrap_init(fun, debug_info=dbg)
    flat_fun, out_tree = batching.flatten_fun_for_vmap(f, in_tree)
    in_axes_flat = flatten_axes("vmap in_axes", in_tree, (in_axes, 0), kws=True)

    if config.mutable_array_checks.value:
      avals = [None if d is None or batching.is_vmappable(x) else core.typeof(x)
               for x, d in zip(args_flat, in_axes_flat)]
      api_util.check_no_aliased_ref_args(lambda: dbg, avals, args_flat)

    axis_size_ = _mapped_axis_size(
        fun, in_tree, args_flat, in_axes_flat, "vmap", axis_size=axis_size)
    explicit_mesh_axis = _mapped_axis_spec(args_flat, in_axes_flat)
    _check_ema_unmapped_args(explicit_mesh_axis, args_flat, in_axes_flat)
    if spmd_axis_name is not None and explicit_mesh_axis is not None:
      if config.remove_size_one_mesh_axis_from_type.value:
        mesh = get_abstract_mesh()
        spmd_axis_name = tuple(i for i in spmd_axis_name if mesh.shape[i] != 1)
      if spmd_axis_name == explicit_mesh_axis:
        spmd_axis_name = None
      else:
        raise ValueError(
            "Only one of spmd_axis_name or arrays sharded on `Explicit` mesh"
            f" axis type is allowed. Got {spmd_axis_name=} and"
            f" arrays sharded on {explicit_mesh_axis=}")
      assert spmd_axis_name is None
    try:
      axis_data = batching.AxisData(axis_name, axis_size_, spmd_axis_name,
                                    explicit_mesh_axis)
      out_flat, inferred_out_axes = batching.batch(
          flat_fun, axis_data, in_axes_flat,
          lambda: flatten_axes("vmap out_axes", out_tree(), out_axes),
          sum_match=sum_match
      ).call_wrapped(*args_flat)
    except batching.SpecMatchError as e:
      out_axes_flat = flatten_axes("vmap out_axes", out_tree(), out_axes)
      out_axes_full = tree_unflatten(out_tree(), out_axes_flat)
      pairs, _ = tree_flatten_with_path(out_axes_full, is_leaf=lambda x: x is None)

      path, _ = pairs[e.leaf_idx]
      raise ValueError(f'at vmap out_axes{keystr(path)}, got axis spec {e.dst} '
                       f'but output was batched on axis {e.src}') from None
    if any(d is batching.infer for d in tree_leaves(out_axes)):
      return (tree_unflatten(out_tree(), out_flat),
              tree_unflatten(out_tree(), inferred_out_axes))
    else:
      return tree_unflatten(out_tree(), out_flat)

  return cast(F, vmap_f)

def _mapped_axis_spec(args_flat, in_axes_flat):
  def _get_spec(arg, i):
    try:
      # 像 BCOO 数组这样的鸭子类型数组可以传给 vmap。
      return shaped_abstractify(arg).sharding.spec[i]
    except (IndexError, TypeError, AttributeError):
      return None

  out_spec = None
  non_none_count = 0
  for arg, i in zip(args_flat, in_axes_flat):
    if i is not None:
      spec = _get_spec(arg, i)
      if non_none_count != 0 and out_spec != spec:
        raise ValueError(
            "Mapped away dimension of inputs passed to vmap should be sharded"
            f" the same. Got inconsistent axis specs: {out_spec} vs {spec}")
      out_spec = spec
      non_none_count += 1
  if out_spec is not None and not isinstance(out_spec, tuple):
    out_spec = (out_spec,)
  return out_spec

def _check_ema_unmapped_args(ema, args_flat, in_axes_flat):
  if ema is None:
    return
  for a, i in zip(args_flat, in_axes_flat):
    if i is None:
      aval = core.typeof(a)
      spec = set(sharding_impls.flatten_spec(aval.sharding.spec))
      if any(e in spec for e in ema):
        raise ValueError(
            "Unmapped values passed to vmap cannot be sharded along the mesh"
            f" axis you are vmapping over. Got type: {aval.str_short(True)},"
            f" in_axes: {i} and vmapped mesh axis: {ema}")

def _mapped_axis_size(fn, tree, vals, dims, name, axis_size=None):
  if not vals:
    if axis_size is not None:
      return axis_size
    args, kwargs = tree_unflatten(tree, vals)
    raise ValueError(
        f"{name} wrapped function must be passed at least one argument "
        "containing an array or axis_size must be specified, got empty "
        f"*args={args} and **kwargs={kwargs}"
    )

  def _get_axis_size(name: str, x, axis: int) -> core.AxisSize | None:
    shape: tuple[core.AxisSize, ...] = ()
    try:
      shape = np.shape(x)
      return shape[axis]
    except (IndexError, TypeError) as e:
      if not core.valid_jaxtype(x) or not isinstance(axis, int):
        return None  # 对自定义的、可被 vmap 的类型抑制该检查。
      if core.typeof(x).is_high:
        raise ValueError(
            f"{name} was requested to map a value of non-array type "
            f"{core.typeof(x)} along axis {axis}, but non-array types can't "
            "be mapped along an integer axis. Instead pass a mapping spec (a "
            "MappingSpec instance) as this argument's in_axes entry, and "
            "pass axis_size explicitly.") from None
      min_rank = axis + 1 if axis >= 0 else -axis
      # TODO(mattjj): 这里的错误信息可以更好
      raise ValueError(
          f"{name} was requested to map its argument along axis {axis}, "
          f"which implies that its rank should be at least {min_rank}, "
          f"but is only {len(shape)} (its shape is {shape})") from e

  all_mapped_sizes = [
    None if d is None else _get_axis_size(name, x, d)
    for x, d in zip(vals, dims)
  ]
  all_sizes = [s for s in all_mapped_sizes if s is not None]
  if axis_size is not None:
    all_sizes.append(axis_size)
  sizes = core.dedup_referents(all_sizes)
  if len(sizes) == 1:
    sz, = sizes
    return sz
  if not sizes:
    raise ValueError(f"{name} must have at least one non-None value in in_axes "
                     "or axis_size must be specified")

  def _get_argument_type(x):
    try:
      return shaped_abstractify(x).str_short()
    except TypeError: # 兜底捕获无法被解释为数据类型的用户指定对象
      return "unknown"
  msg = [f"{name} got inconsistent sizes for array axes to be mapped:\n"]
  args, kwargs = tree_unflatten(tree, vals)
  try:
    ba = inspect.signature(fn).bind(*args, **kwargs)
    signature_parameters: list[str] | None = list(ba.signature.parameters.keys())
  except (TypeError, ValueError):
    signature_parameters = None

  def arg_name(key_path):
    if signature_parameters is None:
      return f"args{keystr(key_path)}"
    # args 是元组，因此 key_path[0].idx 就是 args 中的索引。
    i = key_path[0].idx
    # 在使用星号参数（*args）时可能出现这种情况
    if i >= len(signature_parameters):
      return f"args{keystr(key_path)}"
    res = f"argument {signature_parameters[i]}"
    if len(key_path) > 1:
      res += keystr(key_path[1:])
    return res

  args_paths = [
    f"{arg_name(p)} of type {_get_argument_type(x)}"
    for (p, x) in generate_key_paths(args)
  ]
  kwargs_paths = [
    f"kwargs{keystr(p)} of type {_get_argument_type(x)}"
    for p, x in generate_key_paths(kwargs)
  ]
  key_paths = [*args_paths, *kwargs_paths]
  size_counts = collections.Counter(s for s in all_mapped_sizes if s is not None)
  (sz, ct), *other_counts = counts = size_counts.most_common()

  def _all_sizes_index(sz):
    for i, isz in enumerate(all_mapped_sizes):
      if core.definitely_equal(isz, sz): return i
    assert False, (sz, all_mapped_sizes)

  ex, *examples = (key_paths[_all_sizes_index(sz)] for sz, _ in counts)
  ax, *axs = (dims[_all_sizes_index(sz)] for sz, _ in counts)

  if axis_size is not None:
    msg.append(f"  * the `axis_size` argument was {axis_size};\n")
  if ct == 1:
    msg.append(f"  * one axis had size {sz}: axis {ax} of {ex};\n")
  else:
    msg.append(f"  * most axes ({ct} of them) had size {sz}, e.g. axis {ax} of {ex};\n")
  for ex, ax, (sz, ct) in zip(examples, axs, other_counts):
    if ct == 1:
      msg.append(f"  * one axis had size {sz}: axis {ax} of {ex};\n")
    else:
      msg.append(f"  * some axes ({ct} of them) had size {sz}, e.g. axis {ax} of {ex};\n")
  raise ValueError(''.join(msg)[:-2])  # 去掉最后的分号和换行


@partial(api_boundary, repro_api_name="jax.jvp")
def jvp(
    fun: Callable, primals, tangents, has_aux: bool = False
  ) -> tuple[Any, ...]:
  """计算 ``fun`` 的（前向模式）Jacobian-向量乘积。

  Args:
    fun: 需要求导的函数。其参数应为数组、标量，
      或由数组或标量构成的标准 Python 容器。它应返回数组、
      标量，或由数组或标量构成的标准 Python 容器。
    primals: 用于计算 ``fun`` 的 Jacobian 的原始值。
      应为参数的元组或列表，其长度应等于
      ``fun`` 的位置参数个数。
    tangents: 用于计算 Jacobian-
      向量乘积的切向量。应为切向量的元组或列表，
      其树结构和数组形状与 ``primals`` 相同。
    has_aux: 可选，bool。
     指示 ``fun`` 是否返回一个二元组，
     其中第一个元素被视为待求导的数学函数的输出，
     第二个元素是辅助数据。默认为 False。

  Returns:
    如果 ``has_aux`` 为 ``False``，
    返回 ``(primals_out, tangents_out)`` 二元组，其中
    ``primals_out`` 是 ``fun(*primals)``，``tangents_out`` 是
    ``function`` 在 ``primals`` 处用 ``tangents`` 求得的 Jacobian-
    向量乘积。``tangents_out`` 的值具有与 ``primals_out``
    相同的 Python 树结构和形状。如果 ``has_aux`` 为 ``True``，
    返回 ``(primals_out, tangents_out, aux)``
    三元组，其中 ``aux`` 是 ``fun`` 返回的辅助数据。

  例如：

  >>> import jax
  >>>
  >>> primals, tangents = jax.jvp(jax.numpy.sin, (0.1,), (0.2,))
  >>> print(primals)
  0.09983342
  >>> print(tangents)
  0.19900084
  """
  check_callable(fun)
  if (not isinstance(primals, (tuple, list)) or
      not isinstance(tangents, (tuple, list))):
    raise TypeError("primal and tangent arguments to jax.jvp must be tuples or lists; "
                    f"found {type(primals).__name__} and {type(tangents).__name__}.")
  return _jvp(fun, primals, tangents, has_aux=has_aux)

def _jvp(fun: Callable, primals, tangents, has_aux=False):
  ps_ft = ft.flatten(primals)
  ts_ft = ft.flatten(tangents)
  if ps_ft.tree != ts_ft.tree:
    raise TypeError("primal and tangent arguments to jax.jvp must have the same tree "
                    f"structure; primals have tree structure {ps_ft.tree} whereas tangents have "
                    f"tree structure {ts_ft.tree}.")
  for p, t in zip(ps_ft, ts_ft):
    if not isinstance(core.typeof(p), ShapedArray): continue
    if core.primal_dtype_to_tangent_dtype(_dtype(p)) != _dtype(t):
      raise TypeError("primal and tangent arguments to jax.jvp do not match; "
                      "dtypes must be equal, or in case of int/bool primal dtype "
                      "the tangent dtype must be float0."
                      f"Got primal dtype {_dtype(p)} and so expected tangent dtype "
                      f"{core.primal_dtype_to_tangent_dtype(_dtype(p))}, but got "
                      f"tangent dtype {_dtype(t)} instead.")
    if np.shape(p) != np.shape(t):
      raise ValueError("jvp called with different primal and tangent shapes;"
                       f"Got primal shape {np.shape(p)} and tangent shape as {np.shape(t)}")

  out_primals, out_tangents, *aux = ad.jvp(fun, ps_ft, ts_ft, has_aux=has_aux)
  return out_primals.unflatten(), out_tangents.unflatten(), *aux

@overload
def linearize(fun: Callable, *primals, has_aux: Literal[False] = False,
              in_nzs: Any = None) -> tuple[Any, Callable]:
  ...

@overload
def linearize(fun: Callable, *primals, has_aux: Literal[True],
              in_nzs: Any = None) -> tuple[Any, Callable, Any]:
  ...

@partial(api_boundary, repro_api_name="jax.linearize")
def linearize(fun: Callable, *primals, has_aux: bool = False,
              in_nzs: Any = None
              ) -> tuple[Any, Callable] | tuple[Any, Callable, Any]:
  """使用 :py:func:`jvp` 与部分求值产生 ``fun`` 的线性近似。

  Args:
    fun: 需要被微分的函数。其参数应该是数组、标量，
      或数组、标量的标准 Python 容器。它应该返回
      数组、标量，或数组、标量的标准 Python 容器。
    primals: 求 ``fun`` 的 Jacobian 时所对应的原始值。
      应该是由数组、标量，或它们的标准 Python
      容器构成的元组。元组长度等于 ``fun`` 的
      位置参数个数。
    has_aux: 可选，bool。指示 ``fun`` 是否返回一个二元组，其中第一个
      元素被视为待线性化的数学函数的输出，
      第二个元素是辅助数据。默认为 False。
    in_nzs: 可选，一个由 bool 构成的 tuple-tree（参见 :func:`jax.vjp` 上的
      ``in_nzs``），默认为 ``None``，表示全为 True。它声明哪些原始输入
      具有（可能）非零的切向量；被标记为 False 的输入的切向量在
      线性化过程中按符号零处理。由此得到的逐输出非零
      模式可以通过返回的线性化函数的 ``out_nzs``
      属性获取（不过在 pytree 展平后不会被保留）。

  Returns:
    如果 ``has_aux`` 为 ``False``，返回一个二元组，其中第一个元素是
    ``f(*primals)`` 的值，第二个元素是一个函数，它计算 ``fun`` 在 ``primals``
    处求值的（前向模式）Jacobian-向量乘积，而无需
    重新做线性化的工作。如果 ``has_aux`` 为 ``True``，返回
    ``(primals_out, lin_fn, aux)`` 元组，其中 ``aux`` 是 ``fun``
    返回的辅助数据。

  就所计算的值而言，:py:func:`linearize` 的行为很像柯里化的
  :py:func:`jvp`，下面这两个代码块计算的是相同的值::

    y, out_tangent = jax.jvp(f, (x,), (in_tangent,))

    y, f_jvp = jax.linearize(f, x)
    out_tangent = f_jvp(in_tangent)

  但区别在于，:py:func:`linearize` 使用部分求值，
  因此函数 ``f`` 在调用 ``f_jvp`` 时不会被重新线性化。
  一般来说，这意味着内存占用随计算规模增长，
  与反向模式十分相似。（确实，:py:func:`linearize`
  的签名与 :py:func:`vjp` 很接近！）

  当你想要多次应用 ``f_jvp`` 时，这个函数特别有用，也就是说，
  在同一个线性化点上针对许多不同的输入切向量求 pushforward。
  此外，如果所有输入切向量都同时已知，用 :py:func:`vmap` 做向量化
  可能更高效，例如::

    pushfwd = partial(jvp, f, (x,))
    y, out_tangents = vmap(pushfwd, out_axes=(None, 0))((in_tangents,))

  像这样把 :py:func:`vmap` 与 :py:func:`jvp` 结合使用，
  我们就避免了 :py:func:`linearize` 和 :py:func:`vjp` 都要承担的、
  随计算深度增长的存储线性化的内存开销。

  下面是一个更完整的使用 :py:func:`linearize` 的例子：

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x): return 3. * jnp.sin(x) + jnp.cos(x / 2.)
  ...
  >>> jax.jvp(f, (2.,), (3.,))
  (Array(3.2681944, dtype=float32, weak_type=True), Array(-5.007528, dtype=float32, weak_type=True))
  >>> y, f_jvp = jax.linearize(f, 2.)
  >>> print(y)
  3.2681944
  >>> print(f_jvp(3.))
  -5.007528
  >>> print(f_jvp(4.))
  -6.676704
  """
  check_callable(fun)
  primals_ft = ft.flatten(primals)
  in_nzs_flat = None if in_nzs is None else tuptree_flags(
      in_nzs, primals_ft.tree, 'in_nzs', 'the in_nzs argument to jax.linearize')
  out_primals_ft, out_zeros, jaxpr, consts, structured_residuals, *maybe_aux = \
      ad.linearize(fun, primals_ft, has_aux=has_aux, in_nzs=in_nzs_flat)
  in_avals = primals_ft.map(core.typeof)
  out_avals = out_primals_ft.map(core.typeof)
  lifted_jvp = Partial(
      partial(_lift_linearized, jaxpr, in_avals, out_avals, out_zeros),
      consts, structured_residuals)
  lifted_jvp.out_nzs = tuple(not z for z in out_zeros)  # pyrefly: ignore[missing-attribute]
  return out_primals_ft.unflatten(), lifted_jvp, *maybe_aux


def _lift_linearized(jaxpr, in_avals, out_avals, out_zeros, consts,
                     structured_residuals, *tangents):
  tangents_ft = ft.flatten(tangents)
  if tangents_ft.tree != in_avals.tree:
    raise TypeError(f"expected {in_avals.tree}, got {tangents_ft.tree}")

  tangent_avals = tangents_ft.map(core.typeof)
  for primal_aval, tangent_aval in zip(in_avals, tangent_avals):
    expected_tangent_aval  = primal_aval.to_tangent_aval()
    if not core.typecompat(expected_tangent_aval, tangent_aval):
      extra_msg = ''
      if (isinstance(primal_aval, core.ShapedArray) and
          isinstance(tangent_aval, core.ShapedArray) and
          primal_aval.mat != tangent_aval.mat):
        # TODO(yashkatariya): 调整报错信息。
        pvary_applications = []
        if left := tangent_aval.mat.varying - primal_aval.mat.varying:
          pvary_applications.append(
              f"applying `jax.lax.pcast(..., {tuple(left)}, to='varying')` to"
              " the primal value passed to `jax.linearize`")
        if left := primal_aval.mat.varying - tangent_aval.mat.varying:
          pvary_applications.append(
              f"applying `jax.lax.pcast(..., {tuple(left)}, to='varying')` to"
              " the tangent value passed to the callable `f_jvp` returned by"
              " `jax.linearize`")
        extra_msg = " \nThis might be fixed by:\n" + "\n".join(
            f"  * {d};" for d in pvary_applications)
      raise ValueError(
          "linearized function called on tangent values inconsistent with "
          "the original primal values:\n"
          f"Got tangent aval {tangent_aval} for primal aval {primal_aval} "
          f"but expected {expected_tangent_aval}.{extra_msg}")
  sres_flat = tree_leaves(structured_residuals)
  tangents_out = eval_jaxpr(jaxpr, consts, *tangents_ft, *sres_flat)
  tangents_out_ = iter(tangents_out)
  full_out = [a2tz(aval).instantiate() if known else next(tangents_out_)
              for aval, known in zip(out_avals, out_zeros)]
  assert next(tangents_out_, None) is None
  return out_avals.update(full_out).unflatten()

# TODO(mattjj): 参见 custom_derivatives.py 中的类似函数
def _temporary_dtype_exception(a, a_) -> bool:
  if isinstance(a, core.ShapedArray) and isinstance(a_, core.ShapedArray):
    return a.shape == a_.shape and a_.dtype == float0
  return False

@overload
def vjp[T](fun: Callable[..., T],
           *primals: Any,
           has_aux: Literal[False] = False,
           reduce_axes: Sequence[AxisName] = (),
           saveable_args: Any = True,
           in_nzs: Any = None) -> tuple[T, Callable]:
  ...

@overload
def vjp[T, U](fun: Callable[..., tuple[T, U]], *primals: Any,
              has_aux: Literal[True],
              reduce_axes: Sequence[AxisName] = (),
              saveable_args: Any = True,
              in_nzs: Any = None) -> tuple[T, Callable, U]:
  ...

@partial(api_boundary, repro_api_name="jax.vjp")
def vjp(
    fun: Callable, *primals, has_aux: bool = False, reduce_axes=(),
    saveable_args: Any = True, in_nzs: Any = None,
  ) -> tuple[Any, Callable] | tuple[Any, Callable, Any]:
  """计算 ``fun`` 的（反向模式）向量-Jacobian 乘积。

  :py:func:`grad` 就是作为 :py:func:`vjp` 的一个特例实现的。

  Args:
    fun: 需要被微分的函数。其参数应该是数组、标量，
      或数组、标量的标准 Python 容器。它应该返回
      数组、标量，或数组、标量的标准 Python 容器。
    primals: 一个原始值序列，求 ``fun`` 的 Jacobian 时以它们为求值点。
      ``primals`` 的个数应该等于 ``fun`` 的位置参数个数。
      每个原始值应该是一个数组、标量，或它们的 pytree（标准 Python 容器）。
    has_aux: 可选，bool。指示 ``fun`` 是否返回一个二元组，其中
     第一个元素被视为待微分的数学函数的输出，
     第二个元素是辅助数据。默认为 False。
    saveable_args: 可选，一个由 bool 构成的 tuple-tree（即叶子为 bool
      的嵌套元组），或者等价地说，是 ``primals`` 的一个叶子为 bool 的
      pytree 前缀，默认为单个 bool ``True``。它指示
      每个原始参数（或参数的子 pytree，或叶子）是否可以
      为反向传播保存。它必须在 pytree 节点类型这一层面上
      构成 ``primals`` 的树前缀：元组与参数容器
      只按子节点个数匹配，因此例如一个元组项可以对应
      一个 dict 参数。在 False 项生效的位置，原本会
      原样保存为残差的参数值会被替换为
      ``vjpfun`` 的 ``args_res`` 属性中的 ``NotSaveable``
      哨兵值，调用方必须把它们恢复（例如通过给
      ``vjpfun.args_res`` 赋值）才能应用
      ``vjpfun``。只有原样保存的参数值会受影响；
      由参数计算出的残差照常保存。
    in_nzs: 可选，一个与 ``saveable_args`` 类似的由 bool 构成的 tuple-tree，默认为
      ``None``，表示全为 True。它声明哪些原始输入具有（可能）
      非零的切向量。在 False 项生效的位置，该输入的切向量
      在线性化过程中按符号零处理，这可以使更多
      输出的切向量成为符号零；由此得到的逐输出非零
      模式可以通过 ``vjpfun`` 的 ``out_nzs`` 属性获取，并且
      ``vjpfun`` 对任何被标记为 False 的输入返回零余切向量。

  Returns:
    如果 ``has_aux`` 为 ``False``，返回 ``(primals_out, vjpfun)`` 二元组，其中
    ``primals_out`` 是 ``fun(*primals)``。如果 ``has_aux`` 为 ``True``，返回
    ``(primals_out, vjpfun, aux)`` 元组，其中 ``aux`` 是 ``fun``
    返回的辅助数据。

    ``vjpfun`` 是一个函数，它把与 ``primals_out`` 形状相同的余切向量
    映射到与 ``primals`` 个数和形状都相同的余切向量元组，
    表示 ``fun`` 在 ``primals`` 处求值得到的
    向量-Jacobian 乘积。

  >>> import jax
  >>>
  >>> def f(x, y):
  ...   return jax.numpy.sin(x), jax.numpy.cos(y)
  ...
  >>> primals, f_vjp = jax.vjp(f, 0.5, 1.0)
  >>> xbar, ybar = f_vjp((-0.7, 0.3))
  >>> print(xbar)
  -0.61430776
  >>> print(ybar)
  -0.2524413
  """
  if reduce_axes:
    raise NotImplementedError("reduce_axes argument to vjp is deprecated")
  del reduce_axes
  check_callable(fun)
  canon = lambda x: x if isinstance(x, core.Tracer) else canonicalize_value(x)
  primals_ft = ft.flatten(primals).map(canon)
  primals_ft.map(dispatch.check_arg)
  saveable = _saveable_args_flags(saveable_args, primals_ft.tree)
  in_nzs_flat = None if in_nzs is None else tuptree_flags(
      in_nzs, primals_ft.tree, 'in_nzs', 'the in_nzs argument to jax.vjp')
  out_primals_ft, out_zeros, jaxpr, residuals, structured_residuals, *maybe_aux = \
      ad.linearize(fun, primals_ft, is_vjp=True, has_aux=has_aux,
                   in_nzs=in_nzs_flat)

  id_map = {id(x): i for i, x in enumerate(primals_ft)}
  used, opaque_residuals = set(), []
  spec = [used.add(id(r)) or RSpec(id_map[id(r)], True) if id(r) in id_map else
          RSpec(opaque_residuals.append(r) or (len(opaque_residuals) - 1), False)
          for r in residuals]
  keep = lambda x, s: ((x if s else NotSaveable()) if id(x) in used else NotNeeded())
  args_res = tuptree_map(keep, primals_ft.tree, primals_ft, saveable)
  out_primal_avals = list(out_primals_ft.map(typeof))
  f_vjp = VJP(partial(_vjp3_callable, spec, out_zeros, jaxpr, out_primal_avals),
              primals_ft.tree, out_primals_ft.tree, list(args_res),
              opaque_residuals, structured_residuals)
  return out_primals_ft.unflatten(), f_vjp, *maybe_aux

def _vjp3_callable(spec, out_zeros, jaxpr, out_primal_avals, in_tree, out_tree,
                   args_res, opaque_res, structured_res, want_logs,
                   *maybe_ct_refs):
  explicit_refs = bool(maybe_ct_refs)
  if explicit_refs:
    maybe_ct_refs_flat, in_tree_ = tree_flatten(maybe_ct_refs)
    if in_tree != in_tree_:
      _vjp_with_refs_tree_error(jaxpr, in_tree, in_tree_)
  else:
    maybe_ct_refs_flat = [GradValue()] * in_tree.num_leaves
  args_res_ = tree_leaves(
      args_res, is_leaf=lambda x: isinstance(x, (NotNeeded, NotSaveable)))
  if not_restored := [i.idx for i in spec if i.primal and
                      isinstance(args_res_[i.idx], NotSaveable)]:
    _vjp_not_saveable_error(jaxpr, in_tree, not_restored)
  residuals = [args_res_[i.idx] if i.primal else opaque_res[i.idx] for i in spec]
  arg_invars = jaxpr.invars[len(spec):]  # 跳过残差输入变量
  maybe_accums = [_vjp_accum(jaxpr, in_tree, explicit_refs, idx, v, x)
                  for idx, (v, x) in enumerate(unsafe_zip(arg_invars, maybe_ct_refs_flat))]
  return Partial(partial(_vjp3_bwd, in_tree, out_tree, out_zeros, jaxpr,
                         out_primal_avals, want_logs), residuals, structured_res,
                 maybe_accums)

def _vjp_accum(jaxpr, in_tree, explicit_refs, idx, v, x):
  if isinstance(x, ad.GradAccum):
    return check_accum(v.aval.to_ct_aval(), x)
  elif _is_ref(x):
    expected_aval = _ref_aval(v.aval).to_ct_aval()
    given_aval = _ref_aval(typeof(x))
    if (not core.typecompat(expected_aval, given_aval) and
        not _temporary_dtype_exception(given_aval, expected_aval)):
      raise ValueError(
          "unexpected JAX type (e.g. shape/dtype) for gradient ref passed to "
          f"the VJP function's `with_refs` method for "
          f"{_vjp_arg_name(jaxpr, in_tree, idx)}: the given ref has type "
          f"{typeof(x).str_short()}, but accumulating this argument's "
          f"gradient requires a ref of type Ref{{{expected_aval.str_short()}}}")
    return ad.RefAccum(expected_aval, x)
  elif isinstance(x, DontWant):
    return ad.NullAccum(v.aval.to_ct_aval())
  elif _is_ref_aval(v.aval):
    if explicit_refs:
      raise ValueError(
          f"the gradient for {_vjp_arg_name(jaxpr, in_tree, idx)}, which is "
          "Ref-typed, can't be returned as a value. In the arguments to the "
          "VJP function's `with_refs` method, pass a `Ref` for it, to "
          "accumulate the gradient into the ref in-place, or pass "
          "`jax.ad.DontWant()` to skip computing this argument's gradient.")
    else:
      raise ValueError(
          f"{_vjp_arg_name(jaxpr, in_tree, idx)} is Ref-typed, so its "
          "gradient must be accumulated into a ref, but no gradient ref was "
          "provided. Bind one using the VJP function's `with_refs` method "
          "before applying it, as in `f_vjp.with_refs(grad_ref)(ct)`; the "
          "gradient will be accumulated into `grad_ref` in-place via "
          "addition. Or, to skip computing this argument's gradient, pass "
          "`jax.ad.DontWant()` in place of a gradient ref.")
  else:
    return ad.ValAccum(v.aval.to_ct_aval())

def _vjp_arg_name(jaxpr, in_tree, idx):
  try:
    dummy_args = tree_unflatten(in_tree, list(range(in_tree.num_leaves)))
    path, _ = list(generate_key_paths(dummy_args))[idx]
    position = f"args{keystr(path)}"
  except Exception:  # 反展平（unflatten）自定义 pytree 节点时可能会拒绝哑叶子
    position = f"flat argument index {idx}"
  return (f"the argument at position {position} of the "
          f"differentiated function {jaxpr.debug_info.func_src_info}")

def _vjp_with_refs_tree_error(jaxpr, in_tree, refs_tree):
  msg = f"""unexpected tree structure in the arguments to a VJP function's \
`with_refs` method.

The arguments to `with_refs` must match the pytree structure of the primal
arguments of the differentiated function {jaxpr.debug_info.func_src_info},
with one entry for each array argument. Each entry must be a `Ref` (to
accumulate this argument's gradient into the ref in-place), a
`jax.ad.GradValue()` (to have this argument's gradient returned as a value,
the default behavior), or a `jax.ad.DontWant()` (to skip computing this
argument's gradient). Note that `None` is an empty pytree, so it can't be
used as a placeholder entry.

But the tree structures differ:
"""
  msg += '\n'.join(f"  * args{keystr(path)} was a {thing1} in the primal "
                   f"arguments, but a {thing2} in the `with_refs` arguments, "
                   f"so {explanation}."
                   for path, thing1, thing2, explanation
                   in equality_errors_pytreedef(in_tree, refs_tree))
  raise ValueError(msg)

def _vjp_not_saveable_error(jaxpr, in_tree, idxs):
  msg = """the VJP function was applied before restoring its not-saveable residuals.

Because `saveable_args` was passed to `jax.vjp`, some argument values that
would have been saved for the backward pass were instead replaced with
`NotSaveable()` sentinels. Before the VJP function can be applied, these
values must be restored, e.g. by assigning to the VJP function's `args_res`
attribute. The values not yet restored correspond to:
"""
  msg += '\n'.join(f"  * {_vjp_arg_name(jaxpr, in_tree, idx)};" for idx in idxs)
  raise ValueError(msg)

def check_accum(aval, acc):
  if not core.typecompat(acc.aval, aval):
    raise ValueError(f"Accumulator aval mismatch: expected {aval}, got {acc.aval}")
  return acc

def _vjp3_bwd(in_tree, out_tree, out_zeros, jaxpr, out_primal_avals, want_logs,
              residuals, structured_res, maybe_accums, out_ct):
  cts_flat, out_tree_ = tree_flatten(out_ct, is_leaf=lambda x: isinstance(x, ad.Zero))
  if out_tree != out_tree_:
    _vjp_ct_tree_error(jaxpr, out_tree, out_tree_)
  _vjp_check_ct_avals(cts_flat, out_primal_avals)
  cts_flat = [ct for ct, k in zip(cts_flat, out_zeros) if not k]
  primals_in = [*maybe_accums, *tree_leaves(structured_res)]
  logs = ad.backward_pass3(jaxpr, True, residuals, primals_in, cts_flat)
  arg_cts = [x.freeze() if isinstance(x, ad.ValAccum) else
             DidntWant() if isinstance(x, ad.NullAccum) else GradRef()
             for x in maybe_accums]
  arg_cts = map(ad.instantiate_zeros, arg_cts)
  arg_cts = tree_unflatten(in_tree, arg_cts)
  return (arg_cts, logs) if want_logs else arg_cts


@dataclasses.dataclass(frozen=True, slots=True)
class RSpec:
  idx: int
  primal: bool

def tuptree_map(f, treedef, *args):
  return treedef.walk(lambda xs, _: tuple(xs), lambda xs: f(*xs), zip(*args))

def tuptree_flags(prefix, treedef, name: str, full_name: str) -> list[bool]:
  """把 flags 前缀扩展为 `treedef` 的逐叶子 flags。

  前缀可以是 bool、`treedef` 的叶子为 bool 的 pytree 前缀，或者
  一个 tuple-tree：仅由 bool 和元组构成，在 pytree 节点类型这一层面上
  构成 `treedef` 的树前缀，其中元组与容器只按
  子节点个数匹配。"""
  if isinstance(prefix, bool):
    return [prefix] * treedef.num_leaves
  try:
    dummy = treedef.unflatten(list(range(treedef.num_leaves)))
    flags = broadcast_prefix(prefix, dummy)
  except ValueError:
    pass
  else:
    if all(isinstance(f, bool) for f in flags):
      return list(flags)
  ret: list[bool] = []
  _tuptree_flags_rec(prefix, treedef, name, full_name, (), ret)
  return ret

def _saveable_args_flags(saveable_args, treedef) -> list[bool]:
  return tuptree_flags(saveable_args, treedef, 'saveable_args',
                       'the saveable_args argument to jax.vjp')

def _tuptree_flags_rec(prefix, td, name, full_name, path, ret):
  if isinstance(prefix, bool):
    ret.extend([prefix] * td.num_leaves)
    return
  where = name + ''.join(f'[{i}]' for i in path)
  if not isinstance(prefix, tuple):
    raise ValueError(
        f"{full_name} must be a pytree prefix with bool leaves or a "
        f"tuple-tree of bools "
        f"(made of bools and tuples only), but {where} is {prefix!r} of type "
        f"{type(prefix).__name__}")
  if treedef_is_strict_leaf(td):
    raise ValueError(
        f"{full_name} must form a tree prefix of "
        f"the corresponding values (up to pytree node types), but {where} is "
        "a tuple while the corresponding part of the values is a leaf; use "
        "a single bool there instead")
  td_children = td.children()
  if len(prefix) != len(td_children):
    raise ValueError(
        f"{full_name} must form a tree prefix of "
        "the corresponding values (up to pytree node types, so containers "
        f"need only match in their number of children), but {where} has "
        f"{len(prefix)} children while the corresponding container has "
        f"{len(td_children)}")
  for i, (p, td_) in enumerate(zip(prefix, td_children)):
    _tuptree_flags_rec(p, td_, name, full_name, (*path, i), ret)

def _is_ref(x):
  from jax._src.state.types import AbstractRef
  try:
    return isinstance(typeof(x), AbstractRef)
  except:
    return False

def _is_ref_aval(a):
  from jax._src.state.types import AbstractRef
  return isinstance(a, AbstractRef)

def _ref_aval(a):
  from jax._src.state.types import AbstractRef
  return a.inner_aval if isinstance(a, AbstractRef) else a


_vjp_too_many_args = """
The function returned by `jax.vjp` applied to {} was called with {} arguments,
but functions returned by `jax.vjp` must be called with a single argument
corresponding to the single value returned by the function being differentiated
(even if that returned value is a tuple or other container).

For example, if we have:

  def f(x):
    return (x, x)
  _, f_vjp = jax.vjp(f, 1.0)

the function `f` returns a single tuple as output, and so we call `f_vjp` with a
single tuple as its argument:

  x_bar, = f_vjp((2.0, 2.0))

If we instead call `f_vjp(2.0, 2.0)`, with the values 'splatted out' as
arguments rather than in a tuple, this error can arise.
""".format


def _vjp_ct_tree_error(jaxpr, out_tree, ct_tree):
  msg = f"""unexpected tree structure.

The argument to a VJP function returned by `jax.vjp` must match the pytree
structure of the differentiated function {jaxpr.debug_info.func_src_info}.

But the tree structures differ:
"""
  msg += '\n'.join(f"  * out{keystr(path)} was a {thing1} in the original "
                   f"output, but a {thing2} here, so {explanation}."
                   for path, thing1, thing2, explanation
                   in equality_errors_pytreedef(out_tree, ct_tree))
  raise ValueError(msg)


def _vjp_check_ct_avals(cts, primal_avals):
  # TODO(mattjj): 改进这个报错，一开始就带着键做展平
  for ct, aval in zip(cts, primal_avals):
    if isinstance(ct, ad.Zero): continue
    ct_aval = typeof(ct)
    ct_aval_expected = aval.to_ct_aval()
    if (not core.typecompat(ct_aval, ct_aval_expected) and
        not _temporary_dtype_exception(ct_aval, ct_aval_expected)):
      raise ValueError(
          "unexpected JAX type (e.g. shape/dtype) for argument to VJP function: "
          f"got {ct_aval.str_short()}, but expected {ct_aval_expected.str_short()} "
          "because the corresponding output of the differentiated function had JAX type "
          f"{aval.str_short()}")


@register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class NotNeeded:
  pass

@register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class NotSaveable:
  pass

@dataclasses.dataclass(frozen=True, slots=True)
class GradValue:
  pass

@register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class GradRef:
  pass

@dataclasses.dataclass(frozen=True, slots=True)
class DontWant:
  pass

@register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class DidntWant:
  pass


@dataclasses.dataclass(slots=True, weakref_slot=True)
class VJP:
  fun: Callable  # partial(_vjp3_callable, ...)
  in_tree: PyTreeDef
  out_tree: PyTreeDef
  args_res: list[Any]
  opaque_residuals: list[Any]
  structured_residuals: list[Any]
  want_logs: bool = False
  jaxpr = property(lambda self: self.fun.args[2])
  out_nzs = property(lambda self: tuple(not z for z in self.fun.args[1]))

  def __call__(self, out_ct, *extra_args):
    if extra_args:
      name, *_ = self.jaxpr.debug_info.func_src_info.split(' ')
      raise TypeError(_vjp_too_many_args(name, len(extra_args) + 1))
    return self.fun(self.in_tree, self.out_tree, self.args_res,
                    self.opaque_residuals, self.structured_residuals,
                    self.want_logs)(out_ct)

  # 类似 __call__，但返回一对 (arg_cts, logs)，其中 logs 是一个字典，
  # 它按反向执行顺序、以覆盖语义合并各 transpose/vjp_bwd 规则记录的字典。
  # 普通的 __call__ 会丢弃这些 logs。
  with_logs = property(lambda self: self.replace(want_logs=True))

  def with_refs(self, *maybe_ct_refs):
    return self.fun(self.in_tree, self.out_tree, self.args_res,
                    self.opaque_residuals, self.structured_residuals,
                    self.want_logs, *maybe_ct_refs)

  replace = dataclasses.replace

  # 只有在残差不会被改写时，才可以安全地把它们放进缓存键中。注意！
  __hash__ = object.__hash__
  __eq__ = object.__eq__

register_pytree_node(
    VJP,
    lambda vjp: ((vjp.args_res, vjp.opaque_residuals, vjp.structured_residuals),
                 (vjp.fun, vjp.in_tree, vjp.out_tree, vjp.want_logs)),
    lambda meta, args_res: VJP(*meta[:3], *args_res, meta[3]))  # type: ignore


@partial(api_boundary, repro_api_name="jax.linear_transpose")
def linear_transpose(fun: Callable, *primals, reduce_axes=()) -> Callable:
  """转置一个承诺为线性的函数。

  对于线性函数，这个变换等价于 :py:func:`vjp`，但
  避免了计算前向传播的开销。

  转置后的函数的输出总是具有与 ``primals`` 完全相同的数据类型，
  即使某些值被截断（例如从复数到
  float，或从 float64 到 float32）。为避免截断，请让
  ``primals`` 中的数据类型与转置函数期望输出的完整
  范围相匹配。不支持整数数据类型。

  Args:
    fun: 需要被转置的线性函数。
    *primals: 一个位置参数元组，由数组、标量或它们的
      （嵌套）标准 Python 容器（元组、列表、dict、namedtuple，即
      pytree）构成，用于求值 ``fun(*primals)`` 的形状/数据类型。
      这些参数可以是真实的标量/ndarray，但
      并非必须如此：只会访问它们的 ``shape`` 与 ``dtype`` 属性。
      参见下面的例子。（注意，鸭子类型的对象不能是
      namedtuple，因为那类对象会被当作标准 Python 容器处理。）

  Returns:
    一个可调用对象，它计算 ``fun`` 的转置。传入这个函数的有效输入
    必须与 ``fun(*primals)`` 的结果具有相同的
    形状/数据类型/结构。输出将是一个元组，具有与 ``primals``
    相同的形状/数据类型/结构。

  >>> import jax
  >>>
  >>> f = lambda x, y: 0.5 * x - 0.5 * y
  >>> scalar = jax.ShapeDtypeStruct(shape=(), dtype=np.dtype(np.float32))
  >>> f_transpose = jax.linear_transpose(f, scalar, scalar)
  >>> f_transpose(1.0)
  (Array(0.5, dtype=float32), Array(-0.5, dtype=float32))
  """
  if reduce_axes:
    raise NotImplementedError("reduce_axes argument to transpose is deprecated")
  del reduce_axes
  primals_flat, in_tree = tree_flatten(primals)
  flat_fun, out_tree = flatten_fun_nokwargs(
      lu.wrap_init(fun,
                   debug_info=debug_info("linear_transpose", fun, primals, {})),
      in_tree)
  in_avals = [shaped_abstractify(x) for x in primals_flat]
  in_dtypes = [a.dtype for a in in_avals if not a.is_high]

  in_pvals = map(pe.PartialVal.unknown, in_avals)
  jaxpr, out_pvals, const = pe.trace_to_jaxpr_nounits(flat_fun, in_pvals,
                                                      instantiate=True)
  jaxpr, _ = pe.dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars), True)
  out_avals, _ = unzip2(out_pvals)
  out_dtypes = [a.dtype for a in out_avals if not a.is_high]
  if not (all(dtypes.issubdtype(d, np.inexact) for d in in_dtypes + out_dtypes)
          or all(dtypes.issubdtype(d, np.integer)
                 for d in in_dtypes + out_dtypes)):
    raise TypeError("linear_transpose only supports [float or complex] -> "
                    "[float or complex], and integer -> integer functions, "
                    f"but got {in_dtypes} -> {out_dtypes}.")

  @api_boundary
  def transposed_fun(const, out_cotangent):
    out_cts, out_tree2 = tree_flatten(out_cotangent)
    if out_tree() != out_tree2:
      raise TypeError("cotangent tree does not match function output, "
                      f"expected {out_tree()} but got {out_tree2}")
    if not all(map(core.typecheck, out_avals, out_cts)):
      raise TypeError("cotangent type does not match function output, "
                      f"expected {out_avals} but got {out_cts}")
    dummies = [ad.UndefinedPrimal(a.to_ct_aval()) for a in in_avals]
    in_cts = ad.backward_pass(jaxpr, True, const, dummies, out_cts)
    in_cts = map(ad.instantiate_zeros, in_cts)
    return tree_unflatten(in_tree, in_cts)

  # 确保 transposed_fun 是一个 PyTree
  return Partial(transposed_fun, const)


@overload
def make_jaxpr(
    fun: Callable,
    static_argnums: int | Sequence[int] = (),
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: Literal[False] = ...,
) -> Callable[..., core.Jaxpr]:
  ...

@overload
def make_jaxpr(
    fun: Callable,
    static_argnums: int | Sequence[int] = (),
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: Literal[True] = ...,
) -> Callable[..., tuple[core.Jaxpr, Any]]:
  ...

@partial(api_boundary, repro_api_name="jax.make_japr")
def make_jaxpr(
    fun: Callable,
    static_argnums: int | Sequence[int] = (),
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: bool = False,
) -> Callable[..., core.Jaxpr | tuple[core.Jaxpr, Any]]:
  """创建一个函数，它在给定示例参数时返回 ``fun`` 的 jaxpr。

  Args:
    fun: 要计算其 ``jaxpr`` 的函数。它的位置参数以及它的返回值，
      都应当是数组、标量，或由这些类型构成的标准 Python 容器
      （tuple/list/dict）。
    static_argnums: 参见 :py:func:`jax.jit` 的文档字符串。
    axis_env: 可选，一个由数对构成的序列。
      每个数对的第一个元素是一个轴的名字，
      第二个元素是一个正整数，表示以该名字命名的映射轴的大小。
      当降级涉及并行通信集合操作的函数时，这个参数很有用，
      它指定了 :py:func:`jax.pmap` 的各次应用所会建立起来的
      轴名字/大小环境。
    return_shape: 可选布尔值，默认为 ``False``。
      若为 ``True``，则被包装的函数返回一个数对，
      第一个元素是 ``fun`` 的 ``Jaxpr`` 表示，
      第二个元素是一个 pytree，其结构与 ``fun`` 的输出相同，
      它的叶子是带有 ``shape`` 和 ``dtype`` 属性的对象，
      表示输出中各叶子对应的类型。

  Returns:
    一个经过包装的 ``fun``：当它作用于示例参数时，会返回 ``fun``
    在这些参数上的 ``Jaxpr`` 表示。如果参数 ``return_shape``
    为 ``True``，那么返回的函数改为返回一个数对，
    其中第一个元素是 ``fun`` 的 ``Jaxpr`` 表示，
    第二个元素是一个 pytree，表示 ``fun`` 输出的结构、
    形状、数据类型和具名形状。

  ``jaxpr`` 是 JAX 用来表示程序追踪的中间表示。``jaxpr`` 语言
  基于带 let 绑定的简单类型一阶 lambda 演算。:py:func:`make_jaxpr`
  把一个函数改造成返回其 ``jaxpr`` 的形式，我们可以借此检查
  JAX 内部究竟在做什么。返回的 ``jaxpr`` 是 ``fun`` 的追踪，
  它被抽象到 :py:class:`ShapedArray` 层级。
  在 JAX 内部还存在其他抽象层级。

  这里不详细描述 ``jaxpr`` 语言的语义，
  而是给出几个例子。

  >>> import jax
  >>>
  >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
  >>> print(f(3.0))
  -0.83602
  >>> jax.make_jaxpr(f)(3.0)
  { lambda ; a:f32[]. let b:f32[] = cos a; c:f32[] = sin b in (c,) }
  >>> jax.make_jaxpr(jax.grad(f))(3.0)
  { lambda ; a:f32[]. let
      b:f32[] = cos a
      c:f32[] = sin a
      _:f32[] = sin b
      d:f32[] = cos b
      e:f32[] = mul 1.0:f32[] d
      f:f32[] = neg e
      g:f32[] = mul f c
    in (g,) }
  """
  try:
    hash(fun)
    weakref.ref(fun)
  except TypeError:
    fun = partial(fun)

  @wraps(fun)
  @api_boundary
  def make_jaxpr_f(*args, **kwargs):
    with core.extend_axis_env_nd(axis_env or []):
      traced = jit(fun, static_argnums=static_argnums).trace(*args, **kwargs)
    # `jit` 会把常量中的追踪器转换为参数，但 `make_jaxpr` 的调用者
    # 期望常量不被转换。
    jaxpr = (traced.jaxpr.with_consts(traced._consts) if traced._consts
             else traced.jaxpr)
    if return_shape:
      return jaxpr, traced.out_info
    return jaxpr

  make_jaxpr_f.__module__ = "jax"
  if hasattr(fun, "__qualname__"):
    make_jaxpr_f.__qualname__ = f"make_jaxpr({fun.__qualname__})"
  if hasattr(fun, "__name__"):
    make_jaxpr_f.__name__ = f"make_jaxpr({fun.__name__})"
  return make_jaxpr_f

def _infer_src_sharding(src, x, x_aval) -> Sharding | None:
  if src is not None:
    return src
  if isinstance(x, array.ArrayImpl):
    return x.sharding
  if isinstance(x, core.Tracer):
    val = x.to_concrete_value()
    if val is not None and isinstance(val, array.ArrayImpl):
      return val.sharding
  if (isinstance(x_aval, core.ShapedArray) and
      x_aval.sharding.mesh.are_all_axes_explicit):
    return x_aval.sharding.update(
        memory_kind=core.mem_space_to_kind(x_aval.memory_space))
  return None


@util.cache(max_size=2048, trace_context_in_key=False)
def _check_string_compatible_sharding(s):
  """检查目标设备是否与字符串数组兼容。"""
  if isinstance(s, xc.Device) and s.device_kind == "cpu":
    return
  if (isinstance(s, Sharding)
      and s._internal_device_list[0].device_kind == "cpu"):
    return
  raise TypeError(
      "String arrays can only be sharded to CPU devices. Received"
      f" unsupported device or sharding: {s}")


@util.cache(max_size=2048, trace_context_in_key=False)
def _check_sharding(aval, s):
  if (s is not None and
      not isinstance(s, (xc.Device, Sharding, Format, core.MemorySpace))):
    raise ValueError(
        "`jax.device_put` only accepts `None`, `jax.sharding.Sharding`,"
        " `jax.Device`, `Format`, `jax.memory.Space` or a pytree of these"
        f" values. Received invalid value: {s}")
  if isinstance(aval, core.ShapedArray) and aval.dtype == dtypes.string_dtype:
    _check_string_compatible_sharding(s)

  if isinstance(s, Sharding):
    if isinstance(aval, core.AbstractToken):
      aval = core.get_token_aval()
    pjit.pjit_check_aval_sharding(
        (s,), (aval,), ("",), "device_put args", allow_uneven_sharding=False
    )
    s.shard_shape(aval.shape)  # 若形状不兼容应当抛出错误

def pspec_to_sharding(name, val):
  if isinstance(val, P):
    mesh = get_concrete_mesh()
    if mesh.empty:
      raise ValueError(
          "Please set a mesh via `jax.set_mesh` if a PartitionSpec is"
          f" passed to {name}")
    return NamedSharding(mesh, val)
  return val


def device_put(
    x,
    device: None | xc.Device | Sharding | P | Format | Any = None,
    *, src: None | xc.Device | Sharding | P | Format | Any = None,
    donate: bool | Any = False, may_alias: bool | None | Any = None):
  """把 ``x`` 传输到 ``device``。

  Args:
    x: 一个数组、标量，或由它们构成的（嵌套）标准 Python 容器。
    device: （可选）:py:class:`Device`、:py:class:`Sharding`，或标准
      Python 容器中的（嵌套）:py:class:`Sharding`（必须是 ``x`` 的
      树前缀），表示 ``x`` 应该被传输到的设备。如果给出该参数，
      那么结果会被提交到这些设备上。
    src: （可选）:py:class:`Device`、:py:class:`Sharding`，或标准 Python
      容器中的（嵌套）:py:class:`Sharding`（必须是 ``x`` 的树前缀），
      表示 ``x`` 当前所属的设备。
    donate: bool，或标准 Python 容器中的（嵌套）bool（必须是 ``x`` 的
      树前缀）。若为 True，则调用方可以覆写 ``x`` 并将其标记为已删除。
      这只是尽力而为：JAX 在可能的情况下会捐赠，否则不会。
      若发生了捐赠，输入缓冲区（在将来）总会被删除。
    may_alias: bool、None，或标准 Python 容器中的（嵌套）bool
      （必须是 ``x`` 的树前缀）。若为 False，``x`` 会被复制；
      若为 True，``x`` 是否被别名化取决于运行时的实现。

  Returns:
    ``x`` 的一份副本，它位于 ``device`` 上。

  如果 ``device`` 参数为 ``None``，那么当操作数已经位于某个设备上时，
  这个操作的行为类似于恒等函数；否则它会把数据传输到默认设备，
  且不把结果提交到该设备上。

  该函数始终是异步的，也就是说它会立即返回，
  不会阻塞调用它的 Python 线程直到传输完成。
  """
  with config.explicit_device_put_scope():
    x_flat, treedef = tree_flatten(x)
    x_avals = [shaped_abstractify(x) for x in x_flat]
    if (device is None or
        isinstance(device, (xc.Device, Sharding, core.MemorySpace))):
      device_flat = [device] * len(x_flat)
    else:
      device_flat = flatten_axes("device_put device", treedef, device)

    if (src is None or
        isinstance(src, (xc.Device, Sharding, core.MemorySpace))):
      src_flat = list(map(partial(_infer_src_sharding, src), x_flat, x_avals))
    else:
      src_flat = flatten_axes("device_put source", treedef, src)
      src_flat = list(map(_infer_src_sharding, src_flat, x_flat, x_avals))

    device_flat = map(partial(pspec_to_sharding, 'device_put'), device_flat)
    src_flat = map(partial(pspec_to_sharding, 'device_put'), src_flat)

    if isinstance(donate, bool):
      donate_flat = [donate] * len(x_flat)
    else:
      donate_flat = flatten_axes("device_put donate", treedef, donate)

    if isinstance(may_alias, bool):
      may_alias_flat = [may_alias] * len(x_flat)
    else:
      may_alias_flat = flatten_axes("device_put may_alias", treedef, may_alias)

    copy_semantics = []
    for m, d in zip(may_alias_flat, donate_flat):
      if m and d:
        raise ValueError('may_alias and donate cannot be True at the same time.')
      if m is None:
        m = not d
      if m and not d:
        copy_semantics.append(dispatch.ArrayCopySemantics.REUSE_INPUT)
      elif not m and d:
        copy_semantics.append(dispatch.ArrayCopySemantics.DONATE_INPUT)
      else:
        assert not m and not d
        copy_semantics.append(dispatch.ArrayCopySemantics.ALWAYS_COPY)

    dst_avals = []
    for x_aval, d in zip(x_avals, device_flat):
      if x_aval.is_high:
        raise NotImplementedError(
            "jax.device_put does not yet support values of hijax type "
            f"{x_aval}. Instead, shard the value's components (e.g. before "
            "constructing it), or produce the value with the desired "
            "sharding under jit or shard_map.")
      aval = dispatch.update_dp_aval(x_aval, d)
      dst_avals.append(aval)
      _check_sharding(aval, d)
    if core.trace_state_clean():
      out_flat = dispatch._batched_device_put_impl(
          *x_flat, devices=device_flat, srcs=src_flat,
          copy_semantics=copy_semantics, dst_avals=dst_avals)
    else:
      out_flat = dispatch.device_put_p.bind(
          *x_flat, devices=tuple(device_flat), srcs=tuple(src_flat),
          copy_semantics=tuple(copy_semantics))
    return tree_unflatten(treedef, out_flat)


def device_put_sharded(shards: Sequence[Any], devices: Sequence[xc.Device]):  # noqa: F811
  """把数组分片传输到指定设备并组成 Array。

  Args:
    shards: 一个由数组、标量或它们的（嵌套）标准 Python 容器构成的
      序列，表示要堆叠在一起构成输出的各个分片。``shards`` 的长度
      必须等于 ``devices`` 的长度。
    devices: 一个由 :py:class:`Device` 实例构成的序列，表示 ``shards``
      中对应的分片将被传输到的那些设备。

  该函数始终是异步的，也就是说它会立即返回。

  Returns:
    一个 Array 或它的（嵌套）Python 容器，表示把 ``shards`` 的各元素
    堆叠在一起的结果，其中每个分片都由 ``devices`` 中对应条目
    所指定的物理设备内存支撑。

  Examples:
    为 ``shards`` 传入一个数组列表，会得到一个分片数组，
    其中包含把各输入堆叠起来的结果：

    >>> import jax
    >>> devices = jax.local_devices()
    >>> x = [jax.numpy.ones(5) for device in devices]
    >>> y = jax.device_put_sharded(x, devices)  # doctest: +SKIP
    >>> np.allclose(y, jax.numpy.stack(x))  # doctest: +SKIP
    True

    为 ``shards`` 传入一个列表，其中元素是叶子为数组的嵌套容器对象，
    这对应于在每个叶子上分别堆叠分片。这要求列表中的所有条目
    都具有相同的树结构：

    >>> x = [(i, jax.numpy.arange(i, i + 4)) for i in range(len(devices))]
    >>> y = jax.device_put_sharded(x, devices)  # doctest: +SKIP
    >>> type(y)  # doctest: +SKIP
    <class 'tuple'>
    >>> y0 = jax.device_put_sharded([a for a, b in x], devices)  # doctest: +SKIP
    >>> y1 = jax.device_put_sharded([b for a, b in x], devices)  # doctest: +SKIP
    >>> np.allclose(y[0], y0)  # doctest: +SKIP
    True
    >>> np.allclose(y[1], y1)  # doctest: +SKIP
    True

  See Also:
    - device_put
    - device_put_replicated
  """
  # TODO(jakevdp): 为 devices 提供一个默认值，
  # 该默认值同时考虑本地设备和 pod
  if not isinstance(shards, Sequence):
    raise TypeError("device_put_sharded `shards` input must be a sequence; "
                     f"got {type(shards)}")
  if len(shards) != len(devices):
    raise ValueError(f"len(shards) = {len(shards)} must equal "
                     f"len(devices) = {len(devices)}.")

  def _device_put_sharded(*xs):
    avals = [core.typeof(x) for x in xs]
    if not all(a1 == a2 for a1, a2 in zip(avals[:-1], avals[1:])):
      a1, a2 = next((a1, a2) for a1, a2 in zip(avals[:-1], avals[1:])
                    if a1 != a2)
      raise ValueError("the shards passed to device_put_sharded must have "
                       f"consistent shape and dtype, but got {a1} and {a2}.")
    stacked_aval = avals[0].update(shape=(len(devices),) + avals[0].shape)
    mesh = Mesh(np.array(devices), ("_device_put_sharded",))
    sharding = NamedSharding(mesh, P("_device_put_sharded"))
    if dtypes.issubdtype(stacked_aval.dtype, dtypes.extended):
      return stacked_aval.dtype._rules.device_put_sharded(xs, stacked_aval, sharding, devices)
    ys = []
    for x in xs:
      if not isinstance(x, (np.ndarray, basearray.Array)):
        x = np.asarray(x)
      ys.append(x[None])
    return pxla.batched_device_put(stacked_aval, sharding, ys, list(devices))


  with config.explicit_device_put_scope():
    return tree_map(_device_put_sharded, *shards)


def device_put_replicated(x: Any, devices: Sequence[xc.Device]):  # noqa: F811
  """把数组传输到每一个指定设备并组成 Array。

  Args:
    x: 一个数组、标量或它们的（嵌套）标准 Python 容器，
      表示要复制多份以构成输出的数组。
    devices: 一个由 :py:class:`Device` 实例构成的序列，
      表示 ``x`` 将被传输到的那些设备。

  该函数始终是异步的，也就是说它会立即返回。

  Returns:
    一个 Array 或它的（嵌套）Python 容器，表示把 ``x`` 的值沿一个
    大小为 ``len(devices)`` 的新前导轴广播后的结果，其中沿该新前导轴的
    每个切片都由 ``devices`` 中对应条目所指定的设备上的
    内存支撑。

  Examples:
    传入一个数组：

    >>> import jax
    >>> devices = jax.local_devices()
    >>> x = jax.numpy.array([1., 2., 3.])
    >>> y = jax.device_put_replicated(x, devices)  # doctest: +SKIP
    >>> np.allclose(y, jax.numpy.stack([x for _ in devices]))  # doctest: +SKIP
    True

  See Also:
    - device_put
    - device_put_sharded
  """
  if not isinstance(devices, Sequence) or not devices:
    raise ValueError("`devices` argument to `device_put_replicated must be "
                     "a non-empty sequence.")
  def _device_put_replicated(x):
    aval = core.unmapped_aval(len(devices), 0, core.typeof(x))
    assert isinstance(aval, ShapedArray)
    if isinstance(x, (np.ndarray, basearray.Array)):
      buf = device_put(x[None], devices[0])
    else:
      buf = device_put(x, devices[0])[None]
    mesh = Mesh(np.array(devices), ("_device_put_replicated",))
    sharding = NamedSharding(mesh, P("_device_put_replicated"))
    if dtypes.issubdtype(aval.dtype, dtypes.extended):
      return aval.dtype._rules.device_put_replicated(buf, aval, sharding, devices)
    return pxla.batched_device_put(aval, sharding, [buf] * len(devices), devices)

  with config.explicit_device_put_scope():
    return tree_map(_device_put_replicated, x)


# TODO(mattjj): 考虑修订
def _device_get(x):
  if isinstance(x, core.Tracer):
    return x

  # 扩展数据类型通过它们各自的 device_get 规则进行分派。
  if isinstance(x, basearray.Array) and dtypes.issubdtype(x.dtype, dtypes.extended):
    bufs, tree = tree_util.dispatch_registry.flatten(x)
    return tree.unflatten(device_get(bufs))

  # 其他类型通过它们的 __array__ 方法进行分派。
  try:
    toarray = x.__array__
  except AttributeError:
    return x
  else:
    return toarray()

def device_get(x: Any):
  """把 ``x`` 传输到主机。

  如果 ``x`` 是一个 pytree，那么各个缓冲区会被并行复制。

  Args:
    x: 一个数组、标量、Array 或它们的（嵌套）标准 Python 容器，
      表示要传输到主机的数组。

  Returns:
    一个数组或它的（嵌套）Python 容器，
    表示 ``x`` 的值。

  Examples:
    传入一个 Array：

    >>> import jax
    >>> x = jax.numpy.array([1., 2., 3.])
    >>> jax.device_get(x)
    array([1., 2., 3.], dtype=float32)

    传入一个标量（不会有任何效果）：

    >>> jax.device_get(1)
    1

  See Also:
    - device_put
    - device_put_sharded
    - device_put_replicated
  """
  with config.explicit_device_get_scope():
    for y in tree_leaves(x):
      try:
        y.copy_to_host_async()
      except AttributeError:
        pass
    return tree_map(_device_get, x)


@partial(api_boundary, repro_api_name="jax.eval_shape")
def eval_shape(fun: Callable, *args, **kwargs):
  """在不执行任何 FLOP 的情况下计算 ``fun`` 的形状/数据类型。

  这个工具函数可用于进行形状推断。它的输入/输出行为
  由下式定义::

    def eval_shape(fun, *args, **kwargs):
      out = fun(*args, **kwargs)
      return jax.tree_util.tree_map(jax.ShapeDtypeStruct.like, out)

  但它并不直接应用可能开销很大的 ``fun``，而是使用 JAX 的
  抽象解释机制来求值形状，
  完全不执行任何 FLOP。

  使用 :py:func:`eval_shape` 还能捕获形状错误，它会抛出与求值
  ``fun(*args, **kwargs)`` 相同的形状错误。

  Args:
    fun: 需要求值其输出形状的函数。
    *args: 一个位置参数元组，其中的元素是数组、标量，或这些类型的
      （嵌套）标准 Python 容器（元组、列表、字典、具名元组，即 pytree）。
      由于只会访问 ``shape`` 和 ``dtype`` 属性，因此可以使用
      :class:`jax.ShapeDtypeStruct` 或另一个鸭子类型化为 ndarray 的
      容器（不过请注意，鸭子类型化的对象不能是具名元组，
      因为具名元组会被当作标准 Python 容器处理）。
    **kwargs: 一个关键字参数字典，其中的值是数组、标量，或这些类型的
      （嵌套）标准 Python 容器（pytree）。与 ``args`` 中一样，数组值
      只需按鸭子类型化方式具有 ``shape`` 和 ``dtype`` 属性即可。

  Returns:
    out: 一个嵌套的 PyTree，其叶子是 :class:`jax.ShapeDtypeStruct` 对象。

  例如：

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> f = lambda A, x: jnp.tanh(jnp.dot(A, x))
  >>> A = jax.ShapeDtypeStruct((2000, 3000), jnp.float32)
  >>> x = jax.ShapeDtypeStruct((3000, 1000), jnp.float32)
  >>> out = jax.eval_shape(f, A, x)  # no FLOPs performed
  >>> print(out.shape)
  (2000, 1000)
  >>> print(out.dtype)
  float32

  通过 :func:`eval_shape` 传入的所有参数都会被当作动态参数；
  静态参数可以通过闭包引入，例如使用 :func:`functools.partial`：

  >>> import jax
  >>> from jax import lax
  >>> from functools import partial
  >>> import jax.numpy as jnp
  >>>
  >>> x = jax.ShapeDtypeStruct((1, 1, 28, 28), jnp.float32)
  >>> kernel = jax.ShapeDtypeStruct((32, 1, 3, 3), jnp.float32)
  >>>
  >>> conv_same = partial(lax.conv_general_dilated, window_strides=(1, 1), padding="SAME")
  >>> out = jax.eval_shape(conv_same, x, kernel)
  >>> print(out.shape)
  (1, 32, 28, 28)
  >>> print(out.dtype)
  float32
  """
  if type(fun) is _jax.PjitFunction:
    return fun.trace(*args, **kwargs).out_info  # pyrefly: ignore[missing-attribute]
  try: hash(fun)
  except TypeError: fun = partial(fun)
  return jit(fun).trace(*args, **kwargs).out_info


@partial(api_boundary, repro_api_name="jax.named_call")
def named_call[F: Callable](
    fun: F,
    *,
    name: str | None = None,
) -> F:
  """在暂存 JAX 计算时给函数加上用户指定的名字。

  在为即时编译到 XLA（或 TensorFlow 等其他后端）而暂存计算时，
  JAX 会运行你的 Python 程序，但默认情况下它不会保留
  任何函数名或与这些函数关联的其他元数据。
  这会让调试程序的已暂存（和/或已编译）表示变得复杂，
  因为对每个正在执行的操作而言，
  可用的上下文信息都非常有限。

  `named_call` 让 JAX 把给定的函数暂存为一个具有特定名字的子计算。
  当暂存出的程序用 XLA 编译时，这些具名子计算会被保留，
  并出现在 TensorBoard 的 TensorFlow Profiler 等调试工具中。
  在使用 :func:`experimental.jax2tf.convert` 把 JAX 程序暂存到
  TensorFlow 时，名字同样会被保留。

  Args:
    fun: 要被包装的函数。它可以是任何 Callable。
    name: 可选。用于为名字作用域内创建的所有子计算命名的前缀。
      如果未指定，则使用 ``fun.__name__``。

  Returns:
    一个被包装在 ``named_scope`` 中的 ``fun`` 版本。
  """
  if name is None:
    name = fun.__name__

  return source_info_util.extend_name_stack(name)(fun)


def named_scope(
    name: str,
  ) -> source_info_util.ExtendNameStackContextManager:
  """一个上下文管理器，把用户指定的名字加入 JAX 的名字栈。

  在为即时编译到 XLA（或 TensorFlow 等其他后端）而暂存计算时，
  JAX 默认不会保留它所遇到的 Python 函数的名字
  （或其他源码元数据）。
  这会让调试程序的已暂存（和/或已编译）表示变得复杂，
  因为对每个正在执行的操作而言，
  可用的上下文信息都非常有限。

  ``named_scope`` 让 JAX 在暂存给定函数时，给底层操作加上额外的注解。
  JAX 内部会在一个名字栈中记录这些注解。
  当暂存出的程序用 XLA 编译时，这些注解会被保留，
  并出现在 TensorBoard 的 TensorFlow Profiler 等调试工具中。
  在使用 :func:`experimental.jax2tf.convert` 把 JAX 程序暂存到
  TensorFlow 时，名字同样会被保留。


  Args:
    name: 用于为在名字作用域内创建的所有操作
      命名的前缀。
  Yields:
    产出 ``None``，但会进入一个上下文，在该上下文中 `name`
    会被追加到当前活动的名字栈上。

  Examples:
    ``named_scope`` 可以在已编译函数内部用作上下文管理器：

    >>> import jax
    >>>
    >>> @jax.jit
    ... def layer(w, x):
    ...   with jax.named_scope("dot_product"):
    ...     logits = w.dot(x)
    ...   with jax.named_scope("activation"):
    ...     return jax.nn.relu(logits)

    它也可以用作装饰器：

    >>> @jax.jit
    ... @jax.named_scope("layer")
    ... def layer(w, x):
    ...   logits = w.dot(x)
    ...   return jax.nn.relu(logits)
  """
  if not isinstance(name, str):
    raise TypeError("named_scope name argument must be a string.")
  return source_info_util.extend_name_stack(name)

def effects_barrier():
  """等待已有的函数完成它们的副作用。"""
  dispatch.runtime_tokens.block_until_ready()

def block_until_ready(x):
  """
  尝试在 pytree 的叶子上调用 ``block_until_ready`` 方法。

  Args:
    x: 一个 pytree，通常它的叶子中至少有一些是 JAX 数组实例。

  Returns:
    一个与输入具有相同结构和值的 pytree，
    其中所有 JAX 数组叶子的值都已就绪。
  """
  def try_to_block(x):
    try:
      return x.block_until_ready()
    except AttributeError:
      return x

  arrays = []
  for leaf in tree_leaves(x):
    if isinstance(leaf, array.ArrayImpl):
      arrays.append(leaf)
    else:
      try_to_block(leaf)

  if not arrays:
    # 如果 tree_leaves(x) 为空，或者所有叶子都不是 jax.Array，
    # 那么 `arrays` 会是空的。
    pass
  elif len(arrays) == 1:
    # 单个数组的快速路径。
    try_to_block(arrays[0])
  else:
    # 为多个数组优化的路径。
    xc.batched_block_until_ready(arrays)

  return x

def copy_to_host_async(x):
  """
  尝试在 pytree 的叶子上调用 ``copy_to_host_async`` 方法。

  对每个叶子，该方法都会尝试在叶子上调用 ``copy_to_host_async`` 方法。
  如果该叶子不是 JAX 数组，或者该叶子没有 ``copy_to_host_async`` 方法，
  那么该方法不会对这个叶子做任何事。

  Args:
    x: 一个 pytree，通常它的叶子中至少有一些是 JAX 数组实例。

  Returns:
    一个与输入具有相同结构和值的 pytree，
    其中所有 JAX 数组叶子值的主机副本都已被启动复制。
  """
  for leaf in tree_leaves(x):
    try:
      copy_fn = leaf.copy_to_host_async
    except AttributeError:
      pass
    else:
      copy_fn()

  return x


def clear_backends(_crash=False):
  """
  清除所有后端客户端，以便之后可以创建新的后端客户端。
  """
  clients = []
  if config.debug_leaked_clients_on_clear_backends.value:
    try:
      if xb.backends_are_initialized():
        clients = [weakref.ref(c) for c in xb._backends.values()]
    except Exception:
      pass

  effects_barrier()
  xb._clear_backends()
  util.clear_all_caches()
  pjit._cpp_pjit_cache_fun_only.clear()
  pjit._cpp_pjit_cache_explicit_attributes.clear()
  _jax.PjitFunctionCache.clear_all()

  if clients:
    # 垃圾回收几次，因为存在一些假循环，它们似乎是由测试期间
    # 抛出的异常中所捕获的堆栈回溯造成的。
    # TODO(parkers): 想办法把它变成一次 gc.collect() 调用。
    for _ in range(4):
      gc.collect()
    for r in clients:
      if r() is not None:
        if _crash:
          print("A jax.Client was leaked", file=sys.stderr)
          os._exit(-1)
        else:
          raise RuntimeError("A jax.Client was leaked")

@atexit.register
def clean_up():
  if xb._default_backend is not None:
    clear_backends(_crash=True)
  clear_caches()

  # 如果分布式系统存在就关闭它。否则这是一个空操作。
  distributed.shutdown()


def live_arrays(platform=None):
  """返回后端中 `platform` 上的所有存活数组。

  若 platform 为 None，则指默认后端。
  """
  return xb.get_backend(platform).live_arrays()

def clear_caches():
  """清空所有编译缓存与暂存缓存。

  这不会清空持久化缓存；若要在基准测试等场景中禁用它，
  请把 jax_enable_compilation_cache 配置项设为 False。
  """
  # 清空所有 lu.cache、util.cache 与 util.weakref_lru_cache 实例
  # （用于暂存阶段以及 Python 分派的已编译可执行文件缓存）。
  util.clear_all_caches()
  # 清空 pjit 的所有 C++ 已编译可执行文件缓存
  pjit._cpp_pjit_cache_fun_only.clear()
  pjit._cpp_pjit_cache_explicit_attributes.clear()
  _jax.PjitFunctionCache.clear_all()
