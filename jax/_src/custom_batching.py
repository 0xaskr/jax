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

# 文件职责：实现 `jax.custom_batching` 公开 API，让用户为函数自定义 `vmap` 批处理行为。
# 核心是 `custom_vmap` 装饰器：被装饰函数照常执行，但批量调用时会改走用户通过
# `def_vmap` 注册的规则；该规则接收轴大小、输入是否带批维的 pytree 以及已搬运到前排的
# 批维参数，并返回输出及其批处理说明。模块用原语 `custom_vmap_call` 把被追踪函数与规则
# 一同绑定，并注册实现、抽象值求值、批处理与 `jvp` 规则（不支持反向模式自动微分），
# 另提供基于循环的 `sequential_vmap` 特例，用于原生不支持批维的函数。

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import functools
import operator

from jax._src import api
from jax._src import core
from jax._src import flattree as ft
from jax._src import custom_api_util
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src import api_util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (tree_flatten, tree_map, tree_structure,
                                tree_unflatten, treedef_tuple)


source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


@custom_api_util.register_custom_decorator_type
class custom_vmap:
  """自定义可被 JAX 变换的函数的 vmap 行为。

  该装饰器用于自定义 JAX 函数在 :func:`jax.vmap` 变换下的行为。被
  ``custom_vmap`` 装饰的函数大多（注意事项见下文）与底层函数行为一致，
  唯一的区别出现在用 :py:func:`jax.vmap` 进行批量处理时。批量处理时，
  会使用通过 :py:func:`~jax.custom_batching.custom_vmap.def_vmap` 定义的规则。

  例如：

    >>> @jax.custom_batching.custom_vmap
    ... def f(x, y):
    ...   return x + y
    ...
    >>> @f.def_vmap
    ... def f_vmap_rule(axis_size, in_batched, xs, ys):
    ...   assert all(in_batched)
    ...   assert xs.shape[0] == axis_size
    ...   assert ys.shape[0] == axis_size
    ...   out_batched = True
    ...   return xs * ys, out_batched
    ...
    >>> xs = jnp.arange(3)
    >>> ys = jnp.arange(1, 4)
    >>> jax.vmap(f)(xs, ys)  # prints xs * ys instead of xs + ys
    Array([0, 2, 6], dtype=int32)

  值得注意的是，``custom_vmap`` 函数不支持反向模式自动微分。若要同时自定义
  vmap 与反向模式自动微分，请把 ``custom_vmap`` 与
  :py:class:`jax.custom_vjp` 结合使用。例如：

    >>> @jax.custom_vjp
    ... @jax.custom_batching.custom_vmap
    ... def f(x, y):
    ...   return jnp.sin(x) * y
    ...
    >>> @f.def_vmap
    ... def f_vmap_rule(axis_size, in_batched, xs, ys):
    ...   return jnp.cos(xs) * ys, True
    ...
    >>> def f_fwd(x, y):
    ...   return f(x, y), (jnp.cos(x), jnp.sin(x), y)
    ...
    >>> def f_bwd(res, g):
    ...   cos_x, sin_x, y = res
    ...   return (cos_x * g * y, sin_x * g)
    ...
    >>> f.defvjp(f_fwd, f_bwd)
    >>> jax.vmap(f)(jnp.zeros(3), jnp.ones(3))
    Array([1., 1., 1.], dtype=float32)
    >>> jax.grad(f)(jnp.zeros(()), jnp.ones(()))
    Array(1., dtype=float32)

  注意 :py:class:`jax.custom_vjp` 必须位于外层，包裹被 ``custom_vmap``
  装饰的函数。
  """

  fun: Callable[..., Any]
  vmap_rule: Callable[..., tuple[Any, Any]] | None

  def __init__(self, fun: Callable[..., Any]):
    functools.update_wrapper(self, fun)
    self.fun = fun
    self.vmap_rule = None

  __getattr__ = custom_api_util.forward_attr

  def def_vmap(
      self,
      vmap_rule: Callable[..., tuple[Any, Any]],
  ) -> Callable[..., tuple[Any, Any]]:
    """为该 custom_vmap 函数定义 vmap 规则。

    Args:
      vmap_rule: 实现 vmap 规则的函数。该函数应接受以下参数：(1) 整数
        ``axis_size`` 作为第一个参数，(2) 一个布尔值 pytree，其结构与函数
        输入的结构相同，用于指明每个参数是否被批量处理，(3) 被批量处理的
        参数。它应返回一个元组，包含批量处理后的输出，以及一个结构与输出
        相同的布尔值 pytree，用于指明每个输出元素是否被批量处理。示例见
        :py:func:`jax.custom_batching.custom_vmap` 的文档。

    Returns:
      该方法原样透传规则，返回未做改动的 ``vmap_rule``。
    """
    self.vmap_rule = vmap_rule
    return vmap_rule

  @traceback_util.api_boundary
  def __call__(self, *args, **kwargs):
    debug_fun = api_util.debug_info("custom_vmap fun", self.fun,
                                    args, kwargs)
    try:
      args = api_util.resolve_kwargs(self.fun, args, kwargs)
    except TypeError as e:
      raise TypeError(
          "The input arguments to the custom_vmap-decorated function "
          f"{debug_fun.func_name} could not be resolved to positional-only "
          f"arguments. Binding failed with the error:\n{e}"
      ) from e

    if not self.vmap_rule:
      raise AttributeError(
          f"No batching rule defined for custom_vmap function {debug_fun.func_name} "
          "using def_vmap.")
    args_flat, in_tree = tree_flatten(args)
    in_avals = [core.typeof(x) for x in args_flat]
    jaxpr, out_avals = pe.trace_to_jaxpr(
        self.fun, ft.pack((ft.treedef_args_to_ft(in_tree, in_avals), {})),
        debug_fun)
    closed_call, consts = pe.separate_consts(jaxpr)
    out_tree = tree_structure(out_avals.unflatten())
    in_tree = treedef_tuple((tree_structure(consts), in_tree))
    assert self.vmap_rule is not None
    debug_rule = api_util.debug_info("custom_vmap rule", self.vmap_rule,
                                     (0, args, args), {})
    out_flat = custom_vmap_p.bind(*consts, *args_flat,
                                  call=closed_call,
                                  rule=ClosedRule(self.vmap_rule,
                                                  debug_rule),
                                  in_tree=in_tree,
                                  out_tree=out_tree)
    return tree_unflatten(out_tree, out_flat)


### utils

# 定义一个类而不是定义一个闭包捕获 `rule` 的函数，这样我们可以覆写 __str__
class ClosedRule:
  def __init__(self, rule: Callable, debug: core.DebugInfo):
    functools.update_wrapper(self, rule)
    self.rule = rule
    self.debug = debug

  def __call__(self, axis_size, all_in_batched, *all_args):
    _, args = all_args
    consts_batched, in_batched = all_in_batched
    assert not any(tree_util.tree_leaves(consts_batched)), consts_batched
    return call_rule(self.rule, axis_size, in_batched, args)

  def __str__(self):
    return str(self.rule)

def ensure_list(xs):
  return xs if type(xs) is list else list(xs)

def rule_name(rule):
  return getattr(rule, '__name__', '<unnamed rule>')

def call_rule(rule, axis_size, in_batched, args):
  return rule(axis_size, ensure_list(in_batched), *args)

def check_vmap_rule_trees(rule, original_out_tree, out_tree, out_batched_tree):
  if out_tree != out_batched_tree:
    raise ValueError(
        'structure of output value and output batching specification returned '
        f'by custom vmap rule ({rule_name(rule)}) do not match.\n'
        f'Output values: {out_tree}\n'
        f'Batching spec: {out_batched_tree}')
  if out_tree != original_out_tree:
    raise ValueError(
        f'structure of output returned by custom vmap rule ({rule_name(rule)}) '
        'does not match that of original custom-vmapped function.\n'
        f'Original output: {original_out_tree}\n'
        f'Rule output: {out_tree}')

# 类似 batching.bdim_at_front，但在未映射时不进行广播
def maybe_bdim_at_front(x, bdim):
  if bdim is None:
    return x
  else:
    return util.moveaxis(x, bdim, 0)

# 类似 batching.batch，但 (a) 未柯里化，(b) 返回推断出的输出轴，
# 而不是接受并匹配给定的输出轴规格。假定 `f` 已按 pytree 展平
def vmap_unrestricted(f: lu.WrappedFun, *args, in_axes, axis_name, axis_size):
  axis_data = batching.AxisData(axis_name, axis_size, None, None)
  tag = core.TraceTag()
  f, out_axes = batching.batch_subtrace(f, tag, axis_data, in_axes)
  outs = f.call_wrapped(*args)
  return outs, out_axes()


### custom_vmap_p rules


def custom_vmap_impl(*args, call, rule, in_tree, out_tree):
  del rule, in_tree, out_tree
  return core.jaxpr_as_fun(call)(*args)


def custom_vmap_batching(args_flat, dims, *, call, rule, in_tree, out_tree):
  del call
  axis_size, = {x.shape[d] for x, d in zip(args_flat, dims) if d is not None}
  args_flat = map(maybe_bdim_at_front, args_flat, dims)
  flat_in_batched = [d is not None for d in dims]

  args = tree_unflatten(in_tree, args_flat)
  in_batched = tree_unflatten(in_tree, flat_in_batched)
  out, out_batched = call_rule(rule, axis_size, in_batched, args)
  flat_outs, tree1 = tree_flatten(out)
  flat_out_batched, tree2 = tree_flatten(out_batched)
  check_vmap_rule_trees(rule, out_tree, tree1, tree2)
  flat_out_dims = [0 if b else None for b in flat_out_batched]
  return flat_outs, flat_out_dims


def custom_vmap_abstract_eval(*in_avals, call, **_):
  del in_avals
  return call.out_avals, core.positional_effects(call)


def custom_vmap_jvp(primals, tangents, *,
                    call: core.Jaxpr,
                    rule: ClosedRule,
                    in_tree: tree_util.PyTreeDef, out_tree: tree_util.PyTreeDef):
  def jvp_of_rule_rule(axis_size: int, in_batched, primals, tangents):
    in_batched_ps, in_batched_ts = in_batched

    mutually_batched = tree_map(operator.and_, in_batched_ps, in_batched_ts)
    extra_batched_ps = tree_map(lambda pb, tb: 0 if pb and not tb else None,
                                in_batched_ps, in_batched_ts)
    extra_batched_ts = tree_map(lambda pb, tb: 0 if tb and not pb else None,
                                in_batched_ps, in_batched_ts)

    out_mutually_batched = lu.Store()
    flat_ps_ts, tree_ps_ts = tree_flatten((primals, tangents))
    flat_extra_batched_ps_ts, tree_ps_ts2 = tree_flatten(
        (extra_batched_ps, extra_batched_ts),
        is_leaf=lambda x: x is None)

    # TODO(frostig): 断言这些也相等：
    #   treedef_tuple((in_tree, in_tree))
    # 待 https://github.com/jax-ml/jax/issues/9066 修复之后
    assert tree_ps_ts == tree_ps_ts2
    del tree_ps_ts2

    def to_jvp(*primals):
      out, out_batched = call_rule(rule, axis_size, mutually_batched, primals)
      check_vmap_rule_trees(
          rule, out_tree, tree_structure(out), tree_structure(out_batched))
      out_mutually_batched.store(out_batched)
      return out

    api_util.save_wrapped_fun_debug_info(to_jvp, call.debug_info)
    def to_vmap_over_extra_batched_dims(primals, tangents):
      return api.jvp(to_jvp, primals, tangents)

    to_vmap_over_extra_batched_dims_flat, out_tree2 = api_util.flatten_fun_nokwargs(
        lu.wrap_init(to_vmap_over_extra_batched_dims,
                     # TODO(necula): 修复 debug_info 的调用约定
                     debug_info=call.debug_info),
        tree_ps_ts)

    flat_out_ps_ts, flat_out_axes = vmap_unrestricted(
        to_vmap_over_extra_batched_dims_flat, *flat_ps_ts,
        in_axes=flat_extra_batched_ps_ts,
        axis_name=core.no_axis_name, axis_size=axis_size)

    n, ragged = divmod(len(flat_out_ps_ts), 2)
    assert not ragged
    flat_out_ps, flat_out_ts = flat_out_ps_ts[:n], flat_out_ps_ts[n:]
    flat_out_axes_p, flat_out_axes_t = flat_out_axes[:n], flat_out_axes[n:]
    flat_out_ps = map(maybe_bdim_at_front, flat_out_ps, flat_out_axes_p)
    flat_out_extra_batched_ps = [d is not None for d in flat_out_axes_p]
    flat_out_ts = map(maybe_bdim_at_front, flat_out_ts, flat_out_axes_t)
    flat_out_extra_batched_ts = [d is not None for d in flat_out_axes_t]

    out_ps, out_ts = tree_unflatten(
        out_tree2(), [*flat_out_ps, *flat_out_ts])
    out_extra_batched_ps, out_extra_batched_ts = tree_unflatten(
        out_tree2(), [*flat_out_extra_batched_ps, *flat_out_extra_batched_ts])

    out_batched_ps = tree_map(
        operator.or_, out_mutually_batched.val, out_extra_batched_ps)
    out_batched_ts = tree_map(
        operator.or_, out_mutually_batched.val, out_extra_batched_ts)

    return (out_ps, out_ts), (out_batched_ps, out_batched_ts)

  tangents = map(ad.instantiate_zeros, tangents)
  jvp_call, _ = ad.jvp_jaxpr(call, [True] * len(primals), True)
  jvp_in_tree = treedef_tuple((in_tree, in_tree))
  jvp_out_tree = treedef_tuple((out_tree, out_tree))
  outs = custom_vmap_p.bind(
      *primals, *tangents,
      call=jvp_call, rule=jvp_of_rule_rule,
      in_tree=jvp_in_tree, out_tree=jvp_out_tree)
  assert len(outs) % 2 == 0, len(outs)
  out_primals, out_tangents = util.split_list(outs, [len(outs) // 2])
  return out_primals, out_tangents


custom_vmap_p = core.Primitive('custom_vmap_call')
custom_vmap_p.multiple_results = True
custom_vmap_p.def_impl(custom_vmap_impl)
custom_vmap_p.def_effectful_abstract_eval(custom_vmap_abstract_eval)
batching.primitive_batchers[custom_vmap_p] = custom_vmap_batching
ad.primitive_jvps[custom_vmap_p] = custom_vmap_jvp
mlir.register_lowering(custom_vmap_p, mlir.lower_fun(
    custom_vmap_impl, multiple_results=True))
custom_vmap_p.to_lojax = custom_vmap_impl


# -- custom vmap 的应用


def tree_split(mask, tree):
  lhs = tree_map(lambda l, x: x if l else None, mask, tree)
  rhs = tree_map(lambda l, x: None if l else x, mask, tree)
  return lhs, rhs

def tree_merge(mask, lhs_tree, rhs_tree):
  return tree_map(lambda l, x_l, x_r: x_l if l else x_r,
                  mask, lhs_tree, rhs_tree)

def sequential_vmap(f):
  """``custom_vmap`` 的一个使用循环的特例。

  用 ``sequential_vmap`` 装饰的函数在被批量处理时，会在循环中被顺序调用。
  这对于原生不支持批维的函数很有用。

  例如：

    >>> @jax.custom_batching.sequential_vmap
    ... def f(x):
    ...   jax.debug.print("{}", x)
    ...   return x + 1
    ...
    >>> jax.vmap(f)(jnp.arange(3))
    0
    1
    2
    Array([1, 2, 3], dtype=int32)

  其中打印语句表明这个 :py:func:`~jax.vmap` 是用循环生成的。

  更多细节见 :py:class:`~jax.custom_batching.custom_vmap` 的文档。
  """
  from jax._src.lax import control_flow  # pyrefly: ignore[missing-import]

  f = custom_vmap(f)

  @f.def_vmap
  def rule(axis_size, in_batched, *args):
    del axis_size

    def to_map(mapped_args):
      args = tree_merge(in_batched, mapped_args, bcast_args)
      return f(*args)

    mapped_args, bcast_args = tree_split(in_batched, list(args))
    out = control_flow.map(to_map, mapped_args)
    out_batched = tree_map(lambda _: True, out)
    return out, out_batched

  return f
