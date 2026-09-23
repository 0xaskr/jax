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

# 文件职责：实现 `jax.checkpoint` / `jax.remat` 的梯度检查点（重物化）机制。
# 反向模式自动微分默认保存前向过程中的全部线性化点，本模块允许改为只保存少量
# 输入/中间值、其余在反向传播时重物化，以计算换内存。核心包括：可选的保存策略
# 集合 `checkpoint_policies`（按名称、点积、offload 等选择残差）、面向用户的
# `checkpoint` / `remat` / `checkpoint_name` 装饰器与 API，以及 `remat_p` 原语
# 在 JVP、部分求值、转置、batching、DCE、MLIR 降级等各变换下的规则实现。

from __future__ import annotations

from collections.abc import Callable, Sequence, Iterable
from dataclasses import dataclass
from functools import partial
import logging
from typing import Any
import types

import numpy as np

from jax._src import ad_util
from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import api_util
from jax._src import custom_derivatives
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters.remat import remat_transform
from jax._src.hijax import VJPHiPrimitive, call_hi_primitive_p, Static
from jax._src.lax import lax as lax_internal
from jax._src.lax import convolution as lax_convolution
from jax._src.lib.mlir.dialects import hlo
from jax._src.state import discharge
from jax._src.state.types import AbstractRef
from jax._src.traceback_util import api_boundary
from jax._src import flattree as ft
from jax._src.tree_util import (
    PyTreeDef, tree_flatten, tree_unflatten, tree_structure, broadcast_prefix,
    tree_map, tree_leaves, tree_leaves_checked, Partial, tracing_registry)
from jax._src.util import (unzip2, wraps, split_list, partition_list, safe_map,
                           safe_zip, merge_lists, subs_list, weakref_lru_cache)
from jax._src.core import typeof

source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)

map = safe_map
zip = safe_zip
def identity(x): return x

logger = logging.getLogger(__name__)

### Policies

def everything_saveable(*_, **__) -> bool:
  """默认策略，等同于完全没有使用 ``jax.checkpoint``。

  这是不使用 jax.remat 时实际生效的策略。"""
  return True

def nothing_saveable(*_, **__) -> bool:
  """重物化一切，等同于完全没有使用自定义策略。

  这是使用 jax.remat 但未显式给出策略时实际生效的策略。"""
  return False

@dataclass(frozen=True)
class DotsSaveable:
  only_if_no_batch_dims: bool
  def __call__(self, prim, *args, **params):
    if self.only_if_no_batch_dims:
      if prim is lax_internal.dot_general_p:
        (_, _), (lhs_b, rhs_b) = params['dimension_numbers']
        if not lhs_b and not rhs_b:
          return True
      if prim.name == "scaled_matmul_wrapper":  # 避免导入 cudnn
        return args[0].shape[0] == 1  # 仅当批维大小为 1 时才保存该 dot
      return False
    else:
      ps = {lax_internal.dot_general_p, lax_convolution.conv_general_dilated_p}
      return prim in ps or prim.name == "scaled_matmul_wrapper"

checkpoint_dots = dots_saveable = DotsSaveable(False)
dots_with_no_batch_dims_saveable = DotsSaveable(True)

@dataclass(frozen=True)
class OffloadDotWithNoBatchDims:
  offload_src: str
  offload_dst: str

  def __call__(self, prim, *_, **params) -> Any:
    if prim is lax_internal.dot_general_p:
      (_, _), (lhs_b, rhs_b) = params['dimension_numbers']
      if not lhs_b and not rhs_b:
        return pe.Offloadable(src=self.offload_src, dst=self.offload_dst)
    return pe.Recompute

def offload_dot_with_no_batch_dims(offload_src, offload_dst):
  """与 ``dots_with_no_batch_dims_saveable`` 相同，但改为卸载到 CPU 内存
  而不是重算。

  这对 transformer 是一个有用的启发式策略。"""
  return OffloadDotWithNoBatchDims(offload_src, offload_dst)


name_p = core.Primitive('name')


# TODO TODO 这个策略大概根本行不通，把这项改动单独拆出去
def save_anything_except_these_names(*names_not_to_save):
  """保存任何值（不限于已命名值），但排除给定的名字。"""
  return lambda *_, **__: True

def save_any_names_but_these(*names_not_to_save):
  """只保存已命名的值，即 `checkpoint_name` 的任何输出，但排除给定的
  名字。"""
  return SaveAnyNamesButThese(frozenset(names_not_to_save))

@dataclass(frozen=True)
class SaveOnlyTheseNames:
  saveable_names: frozenset[str]
  def __call__(self, prim, *_, **params):
    if prim is name_p:
      return params['name'] in self.saveable_names
    return False  # 除非在允许列表中，否则不可保存

@dataclass(frozen=True)
class SaveAnyNamesButThese:
  names: frozenset[str]
  def __call__(self, prim, *_, **params):
    if prim is name_p:
      return params['name'] not in self.names
    return False  # 只允许保存已命名的值

def save_only_these_names(*names_which_can_be_saved):
  """只保存已命名的值，且仅限给定的那些名字。"""
  return SaveOnlyTheseNames(frozenset(names_which_can_be_saved))

@dataclass(frozen=True)
class SaveAndOffloadOnlyTheseNames:
  names_which_can_be_saved: frozenset[str]
  names_which_can_be_offloaded: frozenset[str]
  offload_src: str
  offload_dst: str

  def __call__(self, prim, *_, **params) -> Any:
    if prim is name_p and params['name'] in self.names_which_can_be_saved:
      return pe.Saveable
    if prim is name_p and params['name'] in self.names_which_can_be_offloaded:
      return pe.Offloadable(src=self.offload_src, dst=self.offload_dst)
    return pe.Recompute  # 除非在允许列表中，否则不可保存

def save_and_offload_only_these_names(
    *, names_which_can_be_saved, names_which_can_be_offloaded,
    offload_src, offload_dst):
  """与 ``save_only_these_names`` 相同，但改为卸载到 CPU 内存而不是
  重算。"""
  names_which_can_be_saved = frozenset(names_which_can_be_saved)
  names_which_can_be_offloaded = frozenset(names_which_can_be_offloaded)
  intersection = names_which_can_be_saved & names_which_can_be_offloaded
  if intersection:
    raise ValueError(
        "The names should be exclusive and should not intersect in"
        " `names_which_can_be_saved` and `names_which_can_be_offloaded`. Got"
        f" names_which_can_be_saved={set(names_which_can_be_saved)},"
        f" names_which_can_be_offloaded={set(names_which_can_be_offloaded)} and"
        f" the intersection={set(intersection)}")
  return SaveAndOffloadOnlyTheseNames(
      names_which_can_be_saved, names_which_can_be_offloaded,
      offload_src, offload_dst)


def save_from_both_policies(policy_1, policy_2):
  """给定两个策略的逻辑或。

  当且仅当某个残差按其中任一策略可保存时，它才可保存。"""
  def policy(prim, *args, **params):
    out1 = policy_1(prim, *args, **params)
    out2 = policy_2(prim, *args, **params)
    if not (isinstance(out1, bool) and isinstance(out2, bool)):
      raise ValueError(
          "The return value of the policies should be a boolean. Got:"
          f" {out1} and {out2}. Please write a custom policy function directly,"
          " rather than using this helper function.")
    return out1 or out2
  return policy


# 若有新增策略，请同步更新 docs/gradient-checkpointing.md 以保证文档一致。
checkpoint_policies = types.SimpleNamespace(
    SaveOnlyTheseNames=SaveOnlyTheseNames,
    SaveAnyNamesButThese=SaveAnyNamesButThese,
    SaveAndOffloadOnlyTheseNames=SaveAndOffloadOnlyTheseNames,
    everything_saveable=everything_saveable,
    nothing_saveable=nothing_saveable,
    dots_saveable=dots_saveable,
    checkpoint_dots=dots_saveable,
    dots_with_no_batch_dims_saveable=dots_with_no_batch_dims_saveable,
    checkpoint_dots_with_no_batch_dims=dots_with_no_batch_dims_saveable,
    offload_dot_with_no_batch_dims=offload_dot_with_no_batch_dims,
    save_anything_except_these_names=save_anything_except_these_names,
    save_any_names_but_these=save_any_names_but_these,
    save_only_these_names=save_only_these_names,
    save_from_both_policies=save_from_both_policies,
    save_and_offload_only_these_names=save_and_offload_only_these_names)


### Main API

@partial(api_boundary, repro_api_name="jax.checkpoint")
def checkpoint(fun: Callable, *, prevent_cse: bool | Sequence[bool] = True,
               policy: Callable[..., bool] | None = None,
               static_argnums: int | tuple[int, ...] = (),
               static_argnames: str | Iterable[str] = ()) -> Callable:
  """让 ``fun`` 在被微分时重新计算内部的线性化点。

  :func:`jax.checkpoint` 装饰器（别名为 :func:`jax.remat`）提供了一种在自动
  微分的场景下权衡计算时间与内存开销的方式，尤其适用于 :func:`jax.grad` 和
  :func:`jax.vjp` 这类反向模式自动微分，也适用于 :func:`jax.linearize`。

  在反向模式微分一个函数时，默认情况下所有线性化点（例如逐元素非线性原语
  运算的输入）都会在前向求值时被保存下来，以便在反向过程中复用。这种求值策略
  可能导致很高的内存开销，甚至在访存比 FLOPs 昂贵得多的硬件加速器上导致性能
  变差。

  另一种求值策略是让其中一些线性化点被重新计算（即重物化）而不是被保存。
  这种做法可以以增加计算量为代价降低内存占用。

  本函数装饰器会生成 ``fun`` 的一个新版本，它遵循重物化策略而不是默认的
  “保存一切”策略。也就是说，它返回的 ``fun`` 新版本在被微分时不会保存任何
  中间线性化点，而是根据函数保存下来的输入重新计算这些线性化点。

  参见下面的示例。

  Args:
    fun: 需要把自动微分求值策略从默认的“保存全部中间线性化点”改为“重新计算
      它们”的函数。其参数与返回值应为数组、标量，或它们的（嵌套）标准 Python
      容器（tuple/list/dict）。
    prevent_cse: 可选的、仅限关键字的布尔参数，表示是否要阻止由微分生成的 HLO
      中的公共子表达式消除（CSE）优化。这种 CSE 阻止是有代价的，因为它可能
      妨碍其他优化，并且在某些后端上（尤其是 GPU）会带来很高的开销。默认值为
      True，因为否则在 :func:`~jax.jit` 或 :func:`~jax.pmap` 下，CSE 会使这个
      装饰器失去意义。
      但在某些场景中，例如在 :func:`~jax.lax.scan` 内部使用时，这种 CSE 阻止
      机制并无必要，此时可以把 ``prevent_cse`` 设为 False。
      ``prevent_cse`` 也可以是一个与参数结构匹配的 pytree 前缀，其叶子为 bool
      —— 等价地也可以是由 bool 和 tuple 构成的元组树前缀（tuple 按其子元素
      个数与参数容器匹配）—— 用来选择固定哪些参数的已保存值。只要有任一阻止
      生效，由策略保存的计算残差就总会被固定。
      在 ``jax_remat3`` 下，除非启用 ``jax_remat_barrier_no_cotangents``
      升级标志，否则余切也同样会被固定。
    static_argnums: 可选的 int 或 int 序列，仅限关键字的参数，表示要把哪些参数
      值视为静态值以便追踪与缓存。把参数指定为静态可以避免追踪时出现
      ConcretizationTypeError，但代价是更多的重复追踪开销。参见下面的示例。
    policy: 可选的、可调用的仅限关键字参数。它应当是
      ``jax.checkpoint_policies`` 的某个属性。该可调用对象接受一阶原语应用的
      类型级描述作为输入，返回一个布尔值，表示相应的输出值能否作为残差被保存
      （还是必须在（余）切向量计算中按需重新计算）。

  Returns:
    一个（可调用的）函数，其输入/输出行为与 ``fun`` 相同，但在使用例如
    :func:`jax.grad`、:func:`jax.vjp` 或 :func:`jax.linearize` 微分时，会重新
    计算而不是保存中间的线性化点，从而可能以额外的计算量换取内存的节省。

  下面是一个简单的例子：

  >>> import jax
  >>> import jax.numpy as jnp

  >>> @jax.checkpoint
  ... def g(x):
  ...   y = jnp.sin(x)
  ...   z = jnp.sin(y)
  ...   return z
  ...
  >>> jax.value_and_grad(g)(2.0)
  (Array(0.78907233, dtype=float32, weak_type=True), Array(-0.2556391, dtype=float32, weak_type=True))

  这里，无论是否加上 :func:`jax.checkpoint` 装饰器，得到的值都相同。没有该
  装饰器时，``jnp.cos(2.0)`` 和 ``jnp.cos(jnp.sin(2.0))`` 会在前向过程中被
  计算并保存下来供反向过程使用，因为它们在反向过程中需要且只依赖原始输入。
  使用 :func:`jax.checkpoint` 时，前向过程只会计算原始输出，并且只有原始输入
  （``2.0``）会被保存下来供反向过程使用。届时，``jnp.sin(2.0)`` 以及
  ``jnp.cos(2.0)`` 和 ``jnp.cos(jnp.sin(2.0))`` 都会被重新计算。

  虽然 :func:`jax.checkpoint` 控制从前向过程保存哪些值供反向过程使用，但求值
  一个函数或其 VJP 所需的总内存还取决于该函数的许多其他内部细节，包括使用了
  哪些数值原语、它们如何组合、在何处使用了 jit 和 scan 这类控制流原语，以及
  其他因素。

  :func:`jax.checkpoint` 装饰器可以递归套用来表达精巧的自动微分重物化策略。
  例如：

  >>> def recursive_checkpoint(funs):
  ...   if len(funs) == 1:
  ...     return funs[0]
  ...   elif len(funs) == 2:
  ...     f1, f2 = funs
  ...     return lambda x: f1(f2(x))
  ...   else:
  ...     f1 = recursive_checkpoint(funs[:len(funs)//2])
  ...     f2 = recursive_checkpoint(funs[len(funs)//2:])
  ...     return lambda x: f1(jax.checkpoint(f2)(x))
  ...

  如果 ``fun`` 包含依赖参数值的 Python 控制流，可能就需要使用
  ``static_argnums`` 参数。例如，考虑一个布尔标志参数::

    from functools import partial

    @partial(jax.checkpoint, static_argnums=(1,))
    def foo(x, is_training):
      if is_training:
        ...
      else:
        ...

  这里使用 ``static_argnums`` 可以让 ``if`` 语句的条件依赖于 ``is_training``
  的值。使用 ``static_argnums`` 的代价是会引入跨调用的重复追踪开销：在这个
  例子中，每次用新的 ``is_training`` 值调用 ``foo`` 时都会重新追踪它。在某些
  情况下还需要用到 ``jax.ensure_compile_time_eval``::

    @partial(jax.checkpoint, static_argnums=(1,))
    def foo(x, y):
      with jax.ensure_compile_time_eval():
        y_pos = y > 0
      if y_pos:
        ...
      else:
        ...

  除了使用 ``static_argnums``（以及 ``jax.ensure_compile_time_eval``），
  另一种也许更简单的做法是在 :func:`jax.checkpoint` 装饰的函数外部计算某些值，
  然后通过闭包捕获它们。
  """
  if isinstance(static_argnums, int):
    static_argnums = static_argnums,
  if isinstance(prevent_cse, Sequence):
    prevent_cse = tuple(prevent_cse)
  if not isinstance(prevent_cse, (tuple, bool)):
    raise TypeError("prevent_cse must be a bool or tuple of bools, got "
                    f"{type(prevent_cse)=}")

  if config.remat3.value:
    policy = None if policy is nothing_saveable else policy
    return remat3(fun, policy=policy, static_argnums=static_argnums,
                  static_argnames=static_argnames, prevent_cse=prevent_cse)

  @wraps(fun)
  @api_boundary
  def fun_remat(*args, **kwargs):
    debug = api_util.debug_info(
        "checkpoint / remat", fun,
        args, kwargs, static_argnums=static_argnums)
    fun_, args = _remat_static_argnums(fun, static_argnums, args)
    args_flat, in_tree = tracing_registry.flatten((args, kwargs))
    api_util.check_no_transformed_refs_args(lambda: debug, args_flat)
    in_avals = [core.shaped_abstractify(x) for x in args_flat]
    jaxpr, consts, out_tree = _trace_to_jaxpr(fun_, in_tree, tuple(in_avals), debug)
    if isinstance(prevent_cse, tuple):
      cse_args = (tuple(args), kwargs) if kwargs else tuple(args)
      cse = (False,) * len(consts) + tuple(broadcast_prefix(prevent_cse, cse_args))
    else:
      cse = prevent_cse
    out_flat = remat_p.bind(
        *consts, *args_flat, jaxpr=jaxpr, prevent_cse=cse, differentiated=False,
        policy=policy)
    return tree_unflatten(out_tree, out_flat)
  return fun_remat


def remat(fun: Callable, *, prevent_cse: bool = True,
          policy: Callable[..., bool] | None = None,
          static_argnums: int | tuple[int, ...] = ()) -> Callable:
  """:func:`jax.checkpoint` 的别名。"""
  return checkpoint(fun, prevent_cse=prevent_cse, policy=policy,
                    static_argnums=static_argnums)

# 这个函数与 api_util.argnums_partial 类似，区别在于错误信息是针对 jax.remat
# 定制的（因而更可操作），哈希/缓存行为略有不同，并且本函数还接受布尔值的
# static_argnums。也许两者可以去重合并。
def _remat_static_argnums(fun, static_argnums, args):
  if type(static_argnums) is int:
    static_argnums = (static_argnums,)
  elif not (type(static_argnums) is tuple and
            all(type(d) is int for d in static_argnums)):
    raise TypeError("the `static_argnums` argument to `jax.checkpoint` / "
                    "`jax.remat` must be an int, tuple of ints or, bool, but "
                    f"got value {static_argnums}")

  if not all(-len(args) <= d < len(args) for d in static_argnums):
    raise ValueError("the `static_argnums` argument to `jax.checkpoint` / "
                     "`jax.remat` can only take integer values greater than or "
                     "equal to `-len(args)` and less than `len(args)`, but got "
                     f"{static_argnums}, while `len(args)` = {len(args)}")

  if not static_argnums:
    return fun, args
  nargs = len(args)
  static_argnums_ = frozenset(d % len(args) for d in static_argnums)
  dyn_args, static_args = [], []
  for i, x in enumerate(args):
    if i in static_argnums_: static_args.append(WrapHashably(x))
    else: dyn_args.append(x)
  new_fun = _dyn_args_fun(fun, static_argnums_, tuple(static_args), nargs)
  return new_fun, dyn_args

WrapHashably = api_util.WrapHashably
_dyn_args_fun = api_util.dyn_args_fun

# 这个辅助函数与 control_flow/common.py 中的那些类似，但带有 remat 专有的
# 错误信息。
@weakref_lru_cache
def _trace_to_jaxpr(fun: Callable,
                    in_tree: PyTreeDef,
                    in_avals: Sequence[core.AbstractValue],
                    debug: core.DebugInfo
                    ) -> tuple[core.Jaxpr, Sequence[Any], PyTreeDef]:
  in_avals_flat_tree = ft.treedef_args_to_ft(in_tree, in_avals)
  try:
    closed_jaxpr, out_avals = pe.trace_to_jaxpr(fun, in_avals_flat_tree, debug)
  except core.ConcretizationTypeError as e:
    msg, = e.args
    if 'for checkpoint' in msg:
      msg += "\n\n" + (
          "Consider using the `static_argnums` parameter for `jax.remat` or "
          "`jax.checkpoint`. See the `jax.checkpoint` docstring and its example "
          "involving `static_argnums`:\n"
          "https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html"
          "\n")
      e.args = msg,
    raise
  return pe.convert_constvars_jaxpr(closed_jaxpr), closed_jaxpr.consts, out_avals.tree


### Utilities

def saved_residuals(f: Callable,
                    *args, **kwargs) -> list[tuple[core.AbstractValue, str]]:
  in_leaves, in_tree = tree_flatten((args, kwargs))

  def f_(*args):
    args, kwargs = tree_unflatten(in_tree, args)
    return f(*args, **kwargs)

  debug_info = api_util.debug_info("saved_residuals", f, args, kwargs)
  out = api.make_jaxpr(lambda *args: api.vjp(f_, *args),
                       return_shape=True)(*in_leaves)
  assert isinstance(out, tuple)
  jaxpr_, out_shape_ = out
  jaxpr = jaxpr_
  out_shape = out_shape_[1]
  num_res = tree_structure(out_shape).num_leaves
  jaxpr = jaxpr.replace(
      outvars=jaxpr.outvars[len(jaxpr.outvars) - num_res:],
      debug_info=debug_info._replace(result_paths=None))
  assert len(jaxpr.invars) == len(in_leaves)
  return _saved_residuals(jaxpr, debug_info.arg_names or ("unknown",) * len(jaxpr.invars))

def _saved_residuals(jaxpr: core.Jaxpr,
                     arg_names: Sequence[str]) -> list[tuple[core.AbstractValue, str]]:
  res_lits = [x for x in jaxpr.outvars if     isinstance(x, core.Literal)]
  res_vars = {x for x in jaxpr.outvars if not isinstance(x, core.Literal)}

  # 不要把 reduce_precision_p 算作生产者，而是透过它继续向前看
  subst = {e.outvars[0]: e.invars[0] for e in jaxpr.eqns
           if e.primitive is lax_internal.reduce_precision_p}
  res_vars = {subst.get(v, v) for v in res_vars}

  results = []

  for x in res_lits:
    results.append((x.aval, 'from a literal'))

  for v in jaxpr.constvars:
    if v in res_vars:
      results.append((v.aval, 'from a constant'))

  for i, v in enumerate(jaxpr.invars):
    if v in res_vars:
      if arg_names[i]:
        src = f'from the argument {arg_names[i]}'
      else:
        src = 'from the argument at flattened index {i}'
      results.append((v.aval, src))

  def get_name(eqn) -> str | None:
    if eqn.primitive is name_p:
      return eqn.params['name']
    elif (eqn.primitive is call_hi_primitive_p
          and isinstance(p := eqn.params['_prim'], CheckpointName)):
      return p.name

  # TODO(mattjj): 其实我们想把这种情况标记为有问题，也就是 name_p 的
  # 输入还有别的消费者
  # named_vars = {v: e for e in jaxpr.eqns if e.primitive is name_p
  #               for v in e.invars}

  for eqn in jaxpr.eqns:
    for v in eqn.outvars:
      if v in res_vars:
        src = source_info_util.summarize(eqn.source_info)
        if name := get_name(eqn):
          results.append((v.aval, f"named '{name}' from {src}"))
        elif eqn.primitive.name == 'jit':
          results.append((v.aval,
                          f"output of jitted function '{eqn.params['name']}' "
                          f"from {src}"))
        else:
          results.append((v.aval, f'output of {eqn.primitive.name} from {src}'))

  assert len(results) == len(jaxpr.outvars)
  return results

def print_saved_residuals(f, *args, **kwargs):
  for aval, src in saved_residuals(f, *args, **kwargs):
    print(f'{aval.str_short(short_dtypes=True)} {src}')


### Implementation

remat_p = core.Primitive('remat2')
remat_p.multiple_results = True

def _remat_bind(*args, jaxpr, prevent_cse, differentiated, policy):
  assert isinstance(prevent_cse, bool) or len(prevent_cse) == len(args)
  return core.Primitive.bind(remat_p, *args, jaxpr=jaxpr, prevent_cse=prevent_cse,
                             differentiated=differentiated, policy=policy)
remat_p.bind = _remat_bind

@remat_p.def_impl
def remat_impl(*args, jaxpr, prevent_cse, differentiated, policy):
  del prevent_cse, differentiated, policy  # 未使用。
  return core.eval_jaxpr(jaxpr, (), *args)

@remat_p.def_effectful_abstract_eval
def remat_abstract_eval(*args, jaxpr, prevent_cse, differentiated, policy):
  del args, prevent_cse, differentiated, policy  # 未使用。
  return [v.aval for v in jaxpr.outvars], core.positional_effects(jaxpr)

def remat_jvp(primals, tangents, jaxpr, prevent_cse, differentiated, policy):
  assert not jaxpr.constvars
  in_nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  jaxpr_jvp_, out_nz = ad.jvp_jaxpr(jaxpr, in_nonzeros, False)
  nonzero_tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  jaxpr_jvp = pe.convert_constvars_jaxpr(jaxpr_jvp_)
  if isinstance(prevent_cse, tuple):
    prevent_cse += (True,) * len(nonzero_tangents)
  outs = remat_p.bind(
      *jaxpr_jvp_.consts, *primals, *nonzero_tangents, jaxpr=jaxpr_jvp,
      prevent_cse=prevent_cse, differentiated=differentiated, policy=policy)
  out_primals, out_tangents_ = split_list(outs, [len(jaxpr.outvars)])
  out_tangents_ = iter(out_tangents_)
  out_tangents = [next(out_tangents_) if nz else ad_util.p2tz(p)
                  for p, nz in zip(out_primals, out_nz)]
  return out_primals, out_tangents
ad.primitive_jvps[remat_p] = remat_jvp

def remat_partial_eval(trace: pe.JaxprTrace, *tracers: core.Tracer,
                       jaxpr: core.Jaxpr, prevent_cse, **params):
  assert not jaxpr.constvars
  disallowed_effects = effects.remat_allowed_effects.filter_not_in(jaxpr.effects)
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in AD of `checkpoint`/`remat`: {disallowed_effects}')
  policy = params['policy'] or nothing_saveable
  in_unknowns = [not t.is_known() for t in tracers]
  jaxpr_known, jaxpr_staged, out_unknowns, out_inst, num_res = \
      pe.partial_eval_jaxpr_custom(
          jaxpr, in_unknowns, [True] * len(in_unknowns), False, False, policy)

  # 对 jaxpr_staged 做 DCE，只保留那些已实例化且未知的输出
  _, out_inst_unknown = partition_list(out_inst, out_unknowns)
  jaxpr_unknown, in_used_staged = pe.dce_jaxpr(jaxpr_staged, out_inst_unknown)
  used_res, in_used_staged = split_list(in_used_staged, [num_res])

  # 对 jaxpr_known 做 DCE，保留所有已知输出但丢弃被消除的 res
  out_used_known = [True] * (len(out_unknowns) - sum(out_unknowns)) + used_res
  jaxpr_known, in_used_known = pe.dce_jaxpr(jaxpr_known, out_used_known)
  num_res = sum(used_res)

  # 为了避免因 XLA 的过量精度导致前向和反向过程出现精度不一致，在任何残差的
  # 生产者上插入显式的 x = reduce_precision(x, **finfo(x.dtype)) 调用。
  # 参见 https://github.com/jax-ml/jax/pull/22244。
  jaxpr_known_ = _insert_reduce_precision(jaxpr_known, num_res)

  # 计算已知输出与残差（提升到 remat 原语之外）
  _, in_consts_ = unzip2(t.pval for t in tracers if t.pval.is_known())
  _, in_consts = partition_list(in_used_known, in_consts_)
  out_consts = core.eval_jaxpr(jaxpr_known_, (), *in_consts)
  out_knowns, residuals = split_list(out_consts, [len(out_consts)-num_res])

  # 为未知输出建立调用 remat 的配方（recipe）
  res_tracers = map(trace.new_instantiated_const, residuals)
  _, tracers_staged = partition_list(in_used_staged, tracers)
  in_jaxpr_tracers = res_tracers + map(trace.instantiate_const, tracers_staged)  # pyrefly: ignore[bad-argument-type]
  out_jaxpr_tracers = [pe.JaxprTracer(trace, pe.PartialVal.unknown(x.aval), None)
                       for x in jaxpr_unknown.outvars]
  if isinstance(prevent_cse, tuple):
    _, prevent_cse_ = partition_list(in_used_staged, prevent_cse)
    prevent_cse = (True,) * len(res_tracers) + tuple(prevent_cse_)
  new_params = dict(params, jaxpr=jaxpr_unknown, differentiated=True,
                    prevent_cse=prevent_cse)
  recipe = pe.new_eqn_recipe(trace, in_jaxpr_tracers, out_jaxpr_tracers, remat_p,
                             new_params, core.positional_effects(jaxpr_unknown),
                             source_info_util.current())

  # 记录所保存残差的信息
  log_level = logging.WARNING if config.log_checkpoint_residuals.value else logging.DEBUG
  if logger.isEnabledFor(log_level):
    try:
      _, staged_unk = partition_list(in_used_staged, in_unknowns)
      res_invars, _ = partition_list(staged_unk, jaxpr_unknown.invars[num_res:])
      res_outvars = jaxpr_known.outvars[len(jaxpr_known.outvars) - num_res:]
      body_res = _saved_residuals(jaxpr_known.replace(outvars=res_outvars),
                                  ("",) * len(jaxpr_known.invars))
      logger.log(log_level,
                'remat-decorated function ' +
                'saving inputs with shapes:\n' * bool(res_invars) +
                '  %s\n' * len(res_invars) +
                'and ' * bool(res_invars) * bool(body_res) +
                'saving these intermediates:\n' * bool(body_res) +
                '  %s from %s\n' * len(body_res),
                *[v.aval.str_short() for v in res_invars],
                *[elt for (a, s) in body_res for elt in [a.str_short(), s]])
    except:
      pass  # 失败时干脆不记录任何日志

  for t in out_jaxpr_tracers: t.recipe = recipe

  # 把已知输出与未知输出 zip 到一起
  return merge_lists(out_unknowns, out_knowns, out_jaxpr_tracers)
pe.custom_partial_eval_rules[remat_p] = remat_partial_eval

@weakref_lru_cache
def _insert_reduce_precision(jaxpr: core.Jaxpr, num_res: int) -> core.Jaxpr:
  res_vars = jaxpr.outvars[len(jaxpr.outvars) - num_res:]
  used_vars = {x for e in jaxpr.eqns for x in e.invars if isinstance(x, core.Var)}
  invars, constvars, eqns = jaxpr.invars[:], jaxpr.constvars[:], jaxpr.eqns[:]
  for v in res_vars:
    if (not isinstance(v.aval, core.ShapedArray) or
        not dtypes.issubdtype(v.aval.dtype, np.inexact)):
      continue
    if v not in used_vars:
      continue
    assert isinstance(v, core.Var)
    newvar = core.Var(v.aval)
    finfo = dtypes.finfo(v.aval.dtype)
    params = dict(exponent_bits=finfo.nexp, mantissa_bits=finfo.nmant)
    if v in constvars or v in invars:
      lst = constvars if v in constvars else invars
      new_eqn = core.new_jaxpr_eqn(
          [newvar], [v], lax_internal.reduce_precision_p, params, set())
      lst[lst.index(v)] = newvar
      eqns.insert(0, new_eqn)
    else:
      (eqn_idx, eqn), = ((i, e) for i, e in enumerate(eqns) if v in e.outvars)
      if (eqn.primitive == lax_internal.reduce_precision_p and
          eqn.params == params):
        continue
      replace_eqn = eqn.replace(outvars=[v_ if v_ != v else newvar
                                         for v_ in eqn.outvars])
      new_eqn = core.new_jaxpr_eqn(
          [newvar], [v], lax_internal.reduce_precision_p, params, set(),
          eqn.source_info, eqn.ctx)
      eqns[eqn_idx] = replace_eqn
      eqns.insert(eqn_idx+1, new_eqn)
  new_jaxpr = jaxpr.replace(invars=invars, constvars=constvars, eqns=eqns)
  config.enable_checks.value and core.check_jaxpr(new_jaxpr)
  return new_jaxpr

def remat_partial_eval_custom_params_updater(*args):
  unks_in, inst_in, *_, params_known, params_staged = args
  prevent_cse = params_known['prevent_cse']
  assert prevent_cse == params_staged['prevent_cse']
  if isinstance(prevent_cse, tuple):
    prevent_cse_known, _ = partition_list(unks_in, prevent_cse)
    _, prevent_cse_staged = partition_list(inst_in, prevent_cse)
    params_known = dict(params_known, prevent_cse=tuple(prevent_cse_known))
    params_staged = dict(params_staged, prevent_cse=tuple(prevent_cse_staged))
  return params_known, dict(params_staged, differentiated=True)
pe.partial_eval_jaxpr_custom_rules[remat_p] = \
    partial(pe.call_partial_eval_custom_rule, 'jaxpr',
            remat_partial_eval_custom_params_updater)

def remat_transpose(out_cts, *args, jaxpr, prevent_cse, **params):
  # TODO(mattjj): 避免与 UndefinedPrimals 来回转换
  args_ = [ad.UndefinedPrimal(x.aval) if isinstance(x, ad.GradAccum) else x
           for x in args]

  assert not jaxpr.constvars
  in_linear = [ad.is_undefined_primal(x) for x in args_]
  out_zeros = [type(ct) is ad_util.Zero for ct in out_cts]
  transposed_jaxpr_, in_zeros, out_tree = transpose_jaxpr(
      jaxpr, in_linear, out_zeros)
  transposed_jaxpr, consts = transposed_jaxpr_, transposed_jaxpr_.consts
  transposed_jaxpr = pe.convert_constvars_jaxpr(transposed_jaxpr)
  flat_args, _ = tree_flatten((args_, out_cts))
  if isinstance(prevent_cse, tuple):
    prevent_cse_, _ = partition_list(in_linear, prevent_cse)
    prevent_cse = tuple(prevent_cse_) + (True,) * (len(out_zeros) - sum(out_zeros))
  outs = remat_p.bind(*consts, *flat_args, jaxpr=transposed_jaxpr,
                      prevent_cse=prevent_cse, **params)
  in_cts_nz, logs = tree_unflatten(out_tree, outs)
  in_cts_nz_, in_zeros_ = iter(in_cts_nz), iter(in_zeros)
  for x in args:
    if isinstance(x, ad.GradAccum) and not next(in_zeros_):
      x.accum(next(in_cts_nz_))
  return logs
ad.fancy_transposes[remat_p] = remat_transpose

# TODO(mattjj): 把它移到 ad.py
def transpose_jaxpr(jaxpr: core.Jaxpr, in_linear: bool | Sequence[bool],
                    out_zeros: bool | Sequence[bool],
                    ) -> tuple[core.Jaxpr, list[bool], PyTreeDef]:
  if isinstance(in_linear, bool):
    in_linear = (in_linear,) * len(jaxpr.in_avals)
  if isinstance(out_zeros, bool):
    out_zeros = (out_zeros,) * len(jaxpr.out_avals)
  return _transpose_jaxpr(jaxpr, tuple(in_linear), tuple(out_zeros))

@weakref_lru_cache
def _transpose_jaxpr(jaxpr: core.Jaxpr,
                     in_lin: Sequence[bool],
                     out_zeros: Sequence[bool]):
  in_avals = ([a for a,  lin in zip(jaxpr.in_avals,  in_lin   ) if not lin] +
              [a.to_ct_aval() for a, zero in zip(jaxpr.out_avals, out_zeros)
               if not zero])
  cell = lambda: None

  def transposed(*args_flat):
    ins_flat, out_cts_flat = split_list(args_flat, [len(in_lin) - sum(in_lin)])

    # 通过部分求值来求值非线性部分，从而得到一个线性 jaxpr。
    # TODO(mattjj): 改成不需要禁用检查
    with config.mutable_array_checks(False):
      jaxpr_rematted, lin_jaxpr, out_uk, res_avals = \
          pe.partial_eval_jaxpr_nounits(jaxpr, in_lin, False)
    with source_info_util.extend_name_stack('rematted_computation'):
      consts = core.jaxpr_as_fun(jaxpr_rematted)(*ins_flat)

    # 转置该线性 jaxpr（它只有线性输入）。
    out_cts_iter = iter(out_cts_flat)
    out_cts = [ad_util.Zero(aval.to_ct_aval()) if zero else next(out_cts_iter)
               for aval, zero in zip(jaxpr.out_avals, out_zeros)]
    assert next(out_cts_iter, None) is None
    dummy_args = [ad.UndefinedPrimal(aval.to_ct_aval())
                  for aval in lin_jaxpr.in_avals[len(consts):]]
    in_cts, logs = ad.backward_pass(lin_jaxpr, False, lin_jaxpr.consts,
                                    [*consts, *dummy_args], out_cts,
                                    return_logs=True)
    in_cts = in_cts[len(consts):]

    # 找出所得余切中的符号零，并返回非零项，同时让反向过程的日志叶子作为
    # 额外输出一并带出。
    in_zeros = cell.in_cts_zero = [type(ct) is ad_util.Zero for ct in in_cts]  # pyrefly: ignore[missing-attribute]
    in_cts_nz, _ = partition_list(in_zeros, in_cts)
    outs, cell.out_tree = tree_flatten((in_cts_nz, logs))  # pyrefly: ignore[missing-attribute]
    return outs

  dbg = jaxpr.debug_info.with_unknown_names()
  in_avals_flat_tree = ft.flatten((tuple(in_avals), {}))
  transposed_closed_jaxpr, _ = pe.trace_to_jaxpr(
      transposed, in_avals_flat_tree, dbg)
  return transposed_closed_jaxpr, cell.in_cts_zero, cell.out_tree  # pyrefly: ignore[missing-attribute]

def remat_vmap(axis_data, args, dims, *, jaxpr, **params):
  assert not jaxpr.constvars
  jaxpr_batched_, out_batched = batching.batch_jaxpr_axes(
      jaxpr, axis_data, dims,
      [batching.zero_if_mapped] * len(jaxpr.outvars))
  jaxpr_batched, consts = jaxpr_batched_, jaxpr_batched_.consts
  if consts:
    jaxpr_batched = pe.convert_constvars_jaxpr(jaxpr_batched)
  out_dims = [0 if b else None for b in out_batched]
  return remat_p.bind(*consts, *args, jaxpr=jaxpr_batched, **params), out_dims
batching.fancy_primitive_batchers[remat_p] = remat_vmap

# TODO(mattjj,sharadmv): 与 pe.dce_jaxpr_call_rule 去重
def remat_dce(used_outputs: list[bool], eqn: core.JaxprEqn
              ) -> tuple[list[bool], core.JaxprEqn | None]:
  if not any(used_outputs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None
  new_jaxpr, used_inputs = pe.dce_jaxpr(eqn.params['jaxpr'], used_outputs)
  prevent_cse = eqn.params['prevent_cse']
  if isinstance(prevent_cse, tuple):
    prevent_cse = tuple(p for p, u in zip(prevent_cse, used_inputs) if u)
  new_params = dict(eqn.params, jaxpr=new_jaxpr, prevent_cse=prevent_cse)
  if (not any(used_inputs) and not any(used_outputs) and
      _has_effects(new_jaxpr.effects)):
    return used_inputs, None
  else:
    new_invars = [v for v, used in zip(eqn.invars, used_inputs) if used]
    new_eqn = pe.new_jaxpr_eqn(
        new_invars,
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, core.eqn_effects(new_jaxpr, new_invars),
        eqn.source_info, eqn.ctx)
    return used_inputs, new_eqn
pe.dce_rules[remat_p] = remat_dce

def _has_effects(effects) -> bool:
  not_really_effects = (core.NamedAxisEffect, core.InternalMutableArrayEffect)
  return any(not isinstance(e, not_really_effects) for e in effects)


def _remat_lowering(
    ctx: mlir.LoweringRuleContext,
    *args,
    jaxpr: core.Jaxpr,
    prevent_cse: bool,
    differentiated: bool,
    policy,
):
  if isinstance(prevent_cse, bool):
    prevent_cse = (prevent_cse,) * len(ctx.avals_in)  # pyrefly: ignore[bad-assignment]
  assert isinstance(prevent_cse, tuple)
  if differentiated and any(prevent_cse):
    _, barrier_avals = partition_list(prevent_cse, ctx.avals_in)
    other_args, barrier_args = partition_list(prevent_cse, args)
    flat_barrier_args, _ = mlir.ir_tree_registry.flatten(barrier_args)
    barrier_op = hlo.OptimizationBarrierOp(flat_barrier_args)
    _, barrier_treedef = mlir.ir_tree_registry.flatten(
        [mlir._aval_to_ir_types(ctx.module_context, a) for a in barrier_avals])
    res = [mlir.lower_with_sharding_in_types(ctx, op, aval)
           for op, aval in zip(barrier_op.results, barrier_avals)]
    barrier_results = barrier_treedef.unflatten(res)
    args = merge_lists(prevent_cse, other_args, barrier_results)
  outs, tokens_out = mlir.jaxpr_subcomp(
      ctx.module_context, jaxpr, ctx.name_stack.extend('checkpoint'),
      ctx.tokens_in, (), *args, dim_var_values=ctx.dim_var_values,
      const_lowering=ctx.const_lowering, outer_traceback=ctx.traceback)
  ctx.set_tokens_out(tokens_out)
  return outs

mlir.register_lowering(remat_p, _remat_lowering)


def _remat_is_high(*_, jaxpr, **__) -> bool:
  return jaxpr.is_high
remat_p.is_high = _remat_is_high


def _remat_to_lojax(*hi_args, jaxpr, **kwds):
  closed_lo_jaxpr = pe.lower_jaxpr2(jaxpr)
  lo_args = [lo_val for aval, x in zip(jaxpr.in_avals, hi_args)
             for lo_val in aval.lower_val(x)]
  lo_jaxpr = pe.convert_constvars_jaxpr(closed_lo_jaxpr)
  lo_args = (*closed_lo_jaxpr.consts, *lo_args)
  return remat_p.bind(*lo_args, jaxpr=lo_jaxpr, **kwds)
remat_p.to_lojax = _remat_to_lojax


def checkpoint_name(x, name):
  """在 :func:`jax.checkpoint` 内部用名字来标识一个值。

  本函数在运行时的行为相当于恒等函数（原样返回 ``x``），但在 JAX 的追踪中给
  该值附加了一个字符串名字。特定的检查点策略（参见
  :ref:`checkpoint-policies`）可以针对这些名字，来控制前向过程中保存哪些中间
  值、反向过程中重新计算哪些中间值。

  Args:
    x: 要被命名的数组或数组 pytree。
    name: 要与值 ``x`` 关联的字符串名字。

  Returns:
    输入 ``x``，保持不变。

  See Also:
    - :func:`jax.checkpoint`（别名 :func:`jax.remat`）：启用检查点的装饰器。
    - :mod:`jax.checkpoint_policies`：一个命名空间，其中包含使用
      ``checkpoint_name`` 标记的名字来决定行为的各种策略。

  Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jax.ad_checkpoint import checkpoint_name

    >>> # Define a function where we explicitly name an intermediate value
    >>> def f(x):
    ...   y = jnp.sin(x)
    ...   z = checkpoint_name(y, "my_intermediate")
    ...   return jnp.cos(z)

    >>> # Use a policy that saves only the named value
    >>> policy = jax.checkpoint_policies.save_only_these_names("my_intermediate")
    >>> f_checkpointed = jax.checkpoint(f, policy=policy)

    更多示例请参见 `remat 示例笔记本
    <https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html>`_。
  """
  if config.remat3.value:
    return tree_map(lambda x: checkpoint_name3(name, x), x)
  return tree_map(partial(name_p.bind, name=name), x)

name_p.def_impl(lambda x, *, name: x)
name_p.def_abstract_eval(lambda x, *, name: x)

def name_jvp(primals, tangents, *, name):
  (x,), (xdot,) = primals, tangents
  return name_p.bind(x, name=name), xdot  # 不给切向量命名
ad.primitive_jvps[name_p] = name_jvp

mlir.register_lowering(name_p, lambda ctx, x, *, name: [x])

def name_batcher(args, dims, *, name):
  (x,), (d,) = args, dims
  return name_p.bind(x, name=name), d
batching.primitive_batchers[name_p] = name_batcher


@discharge.register_discharge_rule(remat_p)
def _remat_state_discharge_rule(
    ctx, *args, jaxpr, **params):
  discharged_jaxpr = discharge.discharge_state(jaxpr)
  if discharged_jaxpr.consts:
    raise NotImplementedError
  out_vals_ref_vals = remat_p.bind(
      *args, jaxpr=discharged_jaxpr, **params
  )
  out_vals, ref_vals = split_list(out_vals_ref_vals, [len(jaxpr.outvars)])
  ref_vals_ = iter(ref_vals)
  new_invals = [next(ref_vals_) if isinstance(a, AbstractRef) else None
                for a in ctx.in_avals]
  assert next(ref_vals_, None) is None
  return new_invals, out_vals


# -------------------- Remat 3 --------------------

# TODO
#  [ ] 零值传播（需要单独的规则集，也许还要改进 jax.vjp）

def checkpoint_name3(name, x):
  return CheckpointName(name, typeof(x))(x)

def remat3(f=None, /, policy=None, static_argnums=(), static_argnames=(),
           prevent_cse=True):
  """重物化装饰器（新实现，``jax_remat3``）。

  关于与 :func:`jax.custom_vjp` 交互的说明：在微分经过重物化的代码时，出现在
  *另一个 custom_vjp 的 fwd 规则内部* 的 custom_vjp 应用会被作为一个不透明的
  单元整体重物化。对于“fwd 规则调用自身被 custom_vjp 装饰的函数来计算原始
  输出”这一惯用写法而言，这正是预期的语义。在任意阶微分下，值与梯度都不受
  影响。唯一可观察到的后果只出现在高阶自动微分下：再次微分时会运行内层应用
  自己的 fwd 规则（在一阶时它从不运行，因为外层的 bwd 规则已经完成了求导），
  而其中的值（例如用 ``checkpoint_name`` 标记的中间值）无法在那里被检查点的
  ``policy`` 标记为可保存；它们总会被重新计算。
  """
  kwargs = dict(policy=policy, static_argnums=static_argnums,
                static_argnames=static_argnames, prevent_cse=prevent_cse)
  if f is None: return lambda g: _remat3(g, **kwargs)
  return _remat3(f, **kwargs)

def _remat3(f, *, policy, static_argnums, static_argnames, prevent_cse=True):
  if not isinstance(prevent_cse, bool) and (static_argnums or static_argnames):
    raise NotImplementedError(
        "non-bool prevent_cse together with static_argnums/static_argnames")
  @wraps(f)
  def decorator(*args, **kwargs):
    if static_argnums or static_argnames:
      # 与经典 remat（以及 custom_vjp3）一样，对于不可哈希的静态值，改为通过
      # 闭包捕获它们，而不是把它们穿过会对它们做哈希的追踪机制。
      args_ = api_util.resolve_kwargs(f, args, kwargs)
      argnums_ = (static_argnums,) if type(static_argnums) is int else static_argnums
      argnums = frozenset(i % len(args_) for i in _static_argnums(
          f, argnums_, static_argnames))
      if not all(api_util.is_hashable(args_[i]) for i in argnums):
        which_static = [i in argnums for i in range(len(args_))]
        dyn_args, static_args = partition_list(which_static, args_)
        f2 = _dyn_args_fun(f, argnums, tuple(map(WrapHashably, static_args)),
                           len(args_))
        return _remat3(f2, policy=policy, static_argnums=(),
                       static_argnames=())(*dyn_args)
    args_ft = ft.flatten_static_argnums_argnames(
        args, kwargs, static_argnums, static_argnames)
    avals_ft = args_ft.map(typeof)
    dbg = api_util.debug_info(
        'remat3', f, args, kwargs, static_argnums=static_argnums,
        static_argnames=static_argnames)
    jaxpr_, out_avals_ft = pe.trace_to_jaxpr(f, avals_ft, dbg)
    jaxpr, consts = pe.separate_consts(jaxpr_)
    if isinstance(prevent_cse, bool):
      prevent_cse_ = prevent_cse
    else:
      cse_args = (args, kwargs) if kwargs else args
      prevent_cse_ = (False,) * len(consts) + tuple(api.tuptree_flags(
          prevent_cse, tree_structure(cse_args), 'prevent_cse',
          'the prevent_cse argument to jax.checkpoint'))
    out_flat = RematTraced(jaxpr, policy, prevent_cse_)(*consts, *args_ft)
    return out_avals_ft.update(out_flat).unflatten()
  return decorator

def _static_argnums(f, argnums, argnames) -> frozenset[int]:
  argnums = set(argnums)
  if argnames:
    sig = api_util.fun_signature(f)
    assert sig is not None
    argnums |= set(api_util.infer_argnums_and_argnames(sig, None, argnames)[0])
  return frozenset(argnums)

def dce(traced, policy):
  in_fwd = pe._jaxpr_forwarding(traced.jaxpr)
  jaxpr = pe.prune_jaxpr_outputs(traced.jaxpr, [f is None for f in in_fwd])
  for v in jaxpr.outvars:
    if isinstance(v.aval, AbstractRef):
      raise ValueError(
          "the rematted computation's closure contains a mutable array "
          f"reference of type {v.aval.str_short()} that is not one of the "
          "rematted function's inputs, but such refs cannot be saved")
  # dce_jaxpr 会保留附加的常量（constvars 从不被裁剪）。
  jaxpr, used = pe.dce_jaxpr(jaxpr, True)
  keep = [u or i in {*in_fwd} for i, u in enumerate(used)]
  kept_idx = {i: p for p, i in enumerate(i for i, k in enumerate(keep) if k)}
  in_fwd = tuple(kept_idx[f] if f is not None else None for f in in_fwd)
  take = tuple(kept_idx[i] for i, u in enumerate(used) if u)
  keep_res, keep_primals = split_list(keep, [traced._num_consts])
  res = [r for r, u in zip(traced._consts, keep_res) if u]
  return keep_primals, Partial(
      partial(_dced, jaxpr, in_fwd, take, traced.out_tree, policy), res)

@source_info_util.extend_name_stack('rematted_computation')
def _dced(jaxpr, in_fwd, take, out_tree, policy, res, *args):
  ins = [*res, *args]
  outs = RematTraced(jaxpr, policy)(*[ins[i] for i in take])
  return tree_unflatten(out_tree, subs_list(in_fwd, ins, outs))


class RematTraced(VJPHiPrimitive):
  jaxpr: core.Jaxpr
  policy: Any
  prevent_cse: bool | tuple[bool, ...]

  def __init__(self, jaxpr, policy, prevent_cse=True):
    assert (isinstance(prevent_cse, bool) or
            len(prevent_cse) == len(jaxpr.in_avals))
    self.in_avals = tuple(jaxpr.in_avals)
    self.out_aval = jaxpr.out_avals
    self.params = dict(jaxpr=jaxpr, policy=policy, prevent_cse=prevent_cse)
    self.effects = frozenset(core.positional_effects(jaxpr))
    super().__init__()

  def _check_differentiable(self):
    disallowed = effects.remat_allowed_effects.filter_not_in(self.jaxpr.effects)
    if disallowed:
      raise NotImplementedError(
          'Effects not supported in partial-eval of `checkpoint`/`remat`: '
          f'{disallowed}')

  @source_info_util.extend_name_stack('checkpoint')
  def expand(self, *args):
    return core.eval_jaxpr_p.bind(*args, call_jaxpr=self.jaxpr)

  def vjp_fwd(self, nzs_in, *primals):
    # TODO eval_jaxpr_p 的追踪耗时
    self._check_differentiable()
    traced = core.jaxpr_as_fun(self.jaxpr)
    primals_out, fwd2 = remat_transform(self.policy, traced, *primals,
                                        custom_vjp_rules=True)
    in_nzs = tuple(tree_leaves(nzs_in))
    out_nzs_cell = []
    def make_vjp(*xs):
      _, f_vjp = api.vjp(fwd2, *xs, in_nzs=in_nzs)
      out_nzs_cell.append(f_vjp.out_nzs)  # pyrefly: ignore[missing-attribute]
      return f_vjp
    with config.mutable_array_checks(False):
      traced_vjp = api.jit(make_vjp).trace(*primals)
    used, rem = dce(traced_vjp, self.policy)
    primals_ = [x for x, u in zip(tree_leaves(primals), used) if u]
    if isinstance(self.prevent_cse, bool):
      prevent_cse = self.prevent_cse
    else:
      prevent_cse = tuple(f for f, u in zip(self.prevent_cse, used) if u)
    out_nzs, = out_nzs_cell
    return primals_out, (primals_, Static(prevent_cse), rem), list(out_nzs)

  def vjp_bwd(self, primals_rem, outgrad, *arg_accums):
    primals, prevent_cse, rem = primals_rem
    prevent_cse = prevent_cse.val
    outgrad = tree_map(ad_util.instantiate, outgrad,
                       is_leaf=lambda x: isinstance(x, ad_util.Zero))
    if prevent_cse is not False:
      which = ([True] * len(primals) if prevent_cse is True else
               list(prevent_cse))
      unpinned, pinned = partition_list(which, primals)
      res, = rem.args
      if config.remat_barrier_no_cotangents.value:
        if pinned or res:
          pinned, res = lax_internal.optimization_barrier((pinned, res))
      else:
        pinned, res, outgrad = lax_internal.optimization_barrier(
            (pinned, res, outgrad))
      rem = Partial(rem.func, res)
      primals = merge_lists(which, unpinned, pinned)
    bwd = rem(*primals)
    _, logs = bwd.with_logs.with_refs(*arg_accums)(outgrad)
    return logs

  def jvp(self, primals, tangents):
    traced = core.jaxpr_as_fun(self.jaxpr)
    tangents = tuple(map(ad_util.instantiate, tangents))  # TODO 不要实例化
    return api.jvp(traced, primals, tangents)

  def lin(self, nzs_in, *primals):
    self._check_differentiable()
    traced = core.jaxpr_as_fun(self.jaxpr)
    primals_out, fwd2 = remat_transform(self.policy, traced, *primals,
                                        custom_vjp_rules=True)
    in_nzs = tuple(tree_leaves(nzs_in))
    out_nzs_cell = []
    def make_lin(*xs):
      _, f_jvp = api.linearize(fwd2, *xs, in_nzs=in_nzs)
      out_nzs_cell.append(f_jvp.out_nzs)  # pyrefly: ignore[missing-attribute]
      return f_jvp
    with config.mutable_array_checks(False):
      traced_lin = api.jit(make_lin).trace(*primals)
    used, rem = dce(traced_lin, self.policy)
    primals_ = [x for x, u in zip(tree_leaves(primals), used) if u]
    out_nzs, = out_nzs_cell
    return primals_out, (primals_, rem, tuple(out_nzs)), list(out_nzs)

  def linearized(self, primals_rem, *tangents):  # pyrefly: ignore[bad-param-name-override]
    primals, rem, out_nzs = primals_rem
    lin = rem(*lax_internal.optimization_barrier(primals))
    tangents = map(ad_util.instantiate, tangents)  # TODO
    outs = lin(*tangents)
    return [o if nz else ad_util.Zero(typeof(o))
            for o, nz in zip(outs, out_nzs)]

  def batch(self, axis_data, args, dims):
    jaxpr_batched, out_batched = batching.batch_jaxpr_axes(
        self.jaxpr, axis_data, dims,
        [batching.zero_if_mapped] * len(self.jaxpr.outvars))
    out_dims = [0 if b else None for b in out_batched]
    return RematTraced(jaxpr_batched, self.policy,
                       self.prevent_cse)(*args), out_dims

  def remat(self, trace, *args):  # pyrefly: ignore[bad-param-name-override]
    traced = core.jaxpr_as_fun(self.jaxpr)
    out, rem_ = remat_transform(self.policy, traced, *args,
                                custom_vjp_rules=trace.custom_vjp_rules)
    (jaxpr, in_tree, out_tree), (res,) = rem_.func.args, rem_.args
    def rem(*args_):
      args_flat = tree_leaves_checked(in_tree, args_)
      out_flat = RematTraced(jaxpr, self.policy)(*res, *args_flat)
      return tree_unflatten(out_tree, out_flat)
    return out, rem

  def dce(self, used_outs):
    used_outs_flat = tree_leaves_checked(self.out_tree, used_outs)
    if not any(used_outs_flat):
      return False, False, None
    new_jaxpr, used_ins = pe.dce_jaxpr(self.jaxpr, used_outs_flat)
    if all(used_ins) and all(used_outs_flat):
      return True, True, self
    if isinstance(self.prevent_cse, bool):
      prevent_cse = self.prevent_cse
    else:
      prevent_cse = tuple(f for f, u in zip(self.prevent_cse, used_ins) if u)
    return (tuple(used_ins), tuple(used_outs_flat),
            RematTraced(new_jaxpr, self.policy, prevent_cse))

class CheckpointName(VJPHiPrimitive):
  name: str

  def __init__(self, name, aval):
    self.in_avals = aval,
    self.out_aval = aval
    self.params = dict(name=name)
    super().__init__()

  def expand(self, x):  # pyrefly: ignore[bad-override]
    return x

  def remat(self, trace, x):  # pyrefly: ignore[bad-override]
    policy = trace.policy
    x = CheckpointName(self.name, self.in_avals[0])(x)
    if isinstance(policy, (SaveOnlyTheseNames, SaveAnyNamesButThese)):
      saveable = (self.name not in policy.names if isinstance(policy, SaveAnyNamesButThese)
                  else self.name in policy.saveable_names)
      rem = partial(primal_left_tangent_right, x) if saveable else lambda x: x
      return x, rem
    elif isinstance(policy, SaveAndOffloadOnlyTheseNames):
      if self.name in policy.names_which_can_be_saved:
        return x, partial(primal_left_tangent_right, x)
      elif self.name in policy.names_which_can_be_offloaded:
        x_host = api.device_put(x, core.mem_kind_to_space(policy.offload_dst),
                                may_alias=False)
        src_space = core.mem_kind_to_space(policy.offload_src)
        def rem(x_rem):
          x_dev = api.device_put(x_host, src_space, may_alias=False)
          return primal_left_tangent_right(x_dev, x_rem)
        return x, rem
      else:
        return x, lambda x: x  # 完全重物化
    elif policy is everything_saveable:
      return x, partial(primal_left_tangent_right, x)
    else:
      return x, lambda x: x  # 完全重物化

  def jvp(self, primals, tangents):
    (x,), (xdot,) = primals, tangents
    return CheckpointName(self.name, self.in_avals[0])(x), xdot

  def vjp_fwd(self, _nzs_in, x):  # type: ignore
    return CheckpointName(self.name, self.in_avals[0])(x), None

  def vjp_bwd_retval(self, _, g):
    return g,

  def lin(self, nzs_in, x):  # type: ignore
    return CheckpointName(self.name, self.in_avals[0])(x), None

  def linearized(self, _, g):  # type: ignore
    return g

  def batch_dim_rule(self, axis_data, dims, /):
    return dims[0]

class PrimalLeftTangentRight(VJPHiPrimitive):
  def __init__(self, aval_x, aval__x):
    self.in_avals = aval_x, aval__x
    self.out_aval = aval_x
    self.params = {}
    super().__init__()

  def expand(self, x, _x):  # pyrefly: ignore[bad-override]
    return x

  def lin(self, nzs_in, x, _x):  # type: ignore
    return x, None

  def linearized(self, _, xdot, _xdot):  # type: ignore
    return _xdot

  def vjp_fwd(self, nzs_in, x, _x):  # type: ignore
    return x, None

  def vjp_bwd_retval(self, _, g):
    return None, g

  def jvp(self, primals, tangents):
    assert False

  def batch(self, axis_data, args, dims):
    assert False

def primal_left_tangent_right(x, _x):
  return PrimalLeftTangentRight(typeof(x), typeof(_x))(x, _x)


def custom_remat(f, f_fwd, f_rem, f_bwd, *, static_argnums=(),
                 static_argnames=()):
  """用自定义的重物化行为包装 ``f``，用于反向模式自动微分。

  与 :func:`jax.checkpoint` 的策略按名字选择可保存的值不同，被
  ``custom_remat`` 包装的函数带有自己的重物化规则，这些规则可以依赖当前的
  检查点策略。需要 ``jax_remat3`` 实现。

  Args:
    f: 要包装的函数，在普通求值时被调用（或被追踪）。
    f_fwd: 重物化微分下的前向规则，签名为
      ``f_fwd(policy, *args) -> (out, res)``。它接收当前的检查点策略以及
      ``f`` 的参数，返回原始输出与需要保存的残差（残差可以为 ``None``，
      表示什么都不保存）。
    f_rem: 重物化规则，签名为 ``f_rem(res, *args) -> (out, res2)``。在反向
      过程中它接收 ``f_fwd`` 保存的残差以及 ``f`` 的参数，重新计算原始输出
      以及 ``f_bwd`` 所需的残差。
    f_bwd: 反向规则，签名为 ``f_bwd(res2, out_ct) -> args_ct``，返回一个
      余切元组，其中每个 ``f`` 的参数对应一项。
    static_argnums: 同 :func:`jax.jit`。
    static_argnames: 同 :func:`jax.jit`。

  Returns:
    包装后的 ``f``，调用行为相同，但在重物化下进行反向模式微分（例如在
    :func:`jax.checkpoint` 下）时会应用给定的规则。前向模式微分则回退为直接
    微分 ``f``。
  """
  # TODO 仅支持反向模式……改用 hijax 而不是 custom_vjp
  helper = custom_derivatives.custom_vjp(lambda _, *args: f(*args))
  helper.defvjp(f_rem, lambda res, g: (None, *f_bwd(res, g)))
  def call(*args, **kwargs):
    args_ft = ft.flatten_static_argnums_argnames(
        args, kwargs, static_argnums, static_argnames)
    avals_ft = args_ft.map(typeof)
    dbg = api_util.debug_info(
        'custom_remat', f, args, kwargs, static_argnums=static_argnums,
        static_argnames=static_argnames)
    jaxpr_, out_avals_ft = pe.trace_to_jaxpr(f, avals_ft, dbg)
    jaxpr, consts = pe.separate_consts(jaxpr_)
    out_flat = CustomRemat(jaxpr, f_fwd, helper, args_ft.tree, out_avals_ft.tree)(*consts, *args_ft)
    return out_avals_ft.update(out_flat).unflatten()
  return call

class CustomRemat(VJPHiPrimitive):
  jaxpr: core.Jaxpr
  f1: Callable
  f2_fbwd: Callable

  def __init__(self, jaxpr, f1, f2_fbwd, in_tree, out_tree):
    self.in_avals = tuple(jaxpr.in_avals)
    self.out_aval = jaxpr.out_avals
    self.params = dict(jaxpr=jaxpr, f1=f1, f2_fbwd=f2_fbwd, _in_tree=in_tree,
                       _out_tree=out_tree)
    super().__init__()

  def expand(self, *args):
    return core.jaxpr_as_fun(self.jaxpr)(*args)

  def remat(self, trace, *args_flat):  # type: ignore
    args, kwargs = tree_unflatten(self._in_tree, args_flat)  # type: ignore
    out_primal, res = self.f1(trace.policy, *args, **kwargs)
    out_primal_flat = tree_leaves_checked(self._out_tree, out_primal)  # type: ignore
    def rem_flat(*args_flat):
      args, kwargs = tree_unflatten(self._in_tree, args_flat)  # type: ignore
      out_primal = self.f2_fbwd(res, *args, **kwargs)
      return tree_leaves_checked(self._out_tree, out_primal)  # type: ignore
    return out_primal_flat, rem_flat

  def jvp(self, primals, tangents):
    traced = core.jaxpr_as_fun(self.jaxpr)
    tangents = tuple(map(ad_util.instantiate, tangents))  # TODO
    return api.jvp(traced, primals, tangents)

  def lin(self, nzs_in, *primals):
    raise NotImplementedError  # TODO(mattjj)

  def linearized(self, res, *tangents):  # pyrefly: ignore[bad-param-name-override]
    raise NotImplementedError  # TODO(mattjj)

  def vjp_fwd(self, in_nzs, *args_flat):  # type: ignore
    raise NotImplementedError  # TODO(mattjj)

  def vjp_bwd(self, res, ybar):  # type: ignore
    raise NotImplementedError  # TODO(mattjj)
