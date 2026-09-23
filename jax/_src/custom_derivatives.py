# Copyright 2020 The JAX Authors.
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

# 文件职责：实现 JAX 的自定义导数机制（custom_jvp、custom_vjp、
# custom_gradient），让用户用自己提供的 JVP/VJP 规则替换自动微分。
# 这里定义了这些装饰器类及其原语（custom_jvp_call、custom_vjp_call），
# 以及它们在抽象求值、MLIR 降级、转置、部分求值/死代码消除、
# 打印等解释器中的规则；此外还提供 closure_convert、linear_call 等辅助 API。

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
from functools import update_wrapper, reduce, partial, wraps
from typing import Any

from jax._src import config
from jax._src import core
from jax._src import custom_api_util
from jax._src import dtypes
from jax._src import effects
from jax._src import flattree as ft
from jax._src import linear_util as lu
from jax._src import traceback_util
from jax._src.ad_util import (
    stop_gradient_p, SymbolicZero, Zero, zeros_like_aval, p2tz)
from jax._src.api_util import (
  argnums_partial, resolve_kwargs,
  prepend_static_args, debug_info, fun_signature,
  infer_argnums_and_argnames, fun_sourceinfo)
from jax._src.errors import UnexpectedTracerError
from jax._src.state.types import AbstractRef
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (
    tree_flatten, tree_unflatten, tree_map, treedef_is_leaf, treedef_tuple,
    register_pytree_node_class, tree_leaves, tree_flatten_with_path,
    tree_leaves_with_path, keystr, treedef_children, tree_structure, PyTreeDef)
from jax._src.util import (cache, safe_zip, safe_map, split_list, unzip2,
                           weakref_lru_cache)


traceback_util.register_exclusion(__file__)

map = safe_map
zip = safe_zip


### 工具函数

def _sum_tangents(_, x, *xs):
  return reduce(ad.add_tangents, xs, x)

def _zeros_like_pytree(x):
  return tree_map(p2tz, x)

_stop_gradient = partial(
    tree_map,
    lambda x: stop_gradient_p.bind(x) if isinstance(x, core.Tracer) else x,
)


# 与 api_util.py 中的同名函数类似，但这里还会抓取输出的 aval 以便做错误检查
@lu.transformation_with_aux2
def _flatten_fun_nokwargs(f: Callable,
                          store: lu.Store, in_tree: PyTreeDef,
                          *args_flat):
  py_args = tree_unflatten(in_tree, args_flat)
  ans = f(*py_args)
  ans_flat, ans_tree = tree_flatten(ans)
  ans_avals = [core.typeof(x) for x in ans_flat]
  store.store((ans_tree, ans_avals, ()))
  return ans_flat


### JVP

@custom_api_util.register_custom_decorator_type
class custom_jvp[ReturnValue]:
  """注册一个可被 JAX 变换的函数，以便为它定义自定义 JVP 规则。

  该类用作函数装饰器。其实例是可调用对象，行为与被装饰的底层函数相似，
  区别在于：当施加微分变换（如 :py:func:`jax.jvp` 或 :py:func:`jax.grad`）时，
  会改用用户提供的自定义 JVP 规则函数，而不是追踪进底层函数的实现体
  并对其执行自动微分。

  定义自定义 JVP 规则有两个实例方法可用：
  :py:func:`~jax.custom_jvp.defjvp` 为函数的所有输入定义*单个*自定义 JVP 规则；
  为方便起见还有 :py:func:`~jax.custom_jvp.defjvps`，它包装了
  :py:func:`~jax.custom_jvp.defjvp`，并允许你为函数关于各个参数的偏导数
  分别给出定义。

  例如::

    @jax.custom_jvp
    def f(x, y):
      return jnp.sin(x) * y

    @f.defjvp
    def f_jvp(primals, tangents):
      x, y = primals
      x_dot, y_dot = tangents
      primal_out = f(x, y)
      tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
      return primal_out, tangent_out

  更详细的介绍参见教程 tutorial_。

  .. _tutorial: https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
  """
  fun: Callable[..., ReturnValue]
  nondiff_argnums: Sequence[int]
  nondiff_argnames: Sequence[str]
  jvp: Callable[..., tuple[ReturnValue, ReturnValue]] | None = None
  symbolic_zeros: bool = False

  def __new__(cls, fun=None, nondiff_argnums=(), nondiff_argnames=()):
    if fun is not None and config.custom_jvp3.value:
      from jax._src.hijax import custom_jvp3  # pyrefly: ignore[missing-import]
      return custom_jvp3(fun, nondiff_argnums, nondiff_argnames)
    else:
      return super().__new__(cls)

  def __init__(self,
               fun: Callable[..., ReturnValue],
               nondiff_argnums: Sequence[int] = (),
               nondiff_argnames: Sequence[str] = (),
               ):
    update_wrapper(self, fun)
    self.fun = fun

    nondiff_argnums_: set[int] = set()
    if nondiff_argnames:
      sig = fun_signature(self.fun)
      assert sig is not None
      inferred_nondiff_argnums, _ = infer_argnums_and_argnames(
          sig, None, nondiff_argnames
      )
      nondiff_argnums_.update(inferred_nondiff_argnums)

    if nondiff_argnums:
      nondiff_argnums_.update(nondiff_argnums)

    self.nondiff_argnums = tuple(sorted(nondiff_argnums_))

  __getattr__ = custom_api_util.forward_attr

  def defjvp(self,
             jvp: Callable[..., tuple[ReturnValue, ReturnValue]],
             symbolic_zeros: bool = False,
             ) -> Callable[..., tuple[ReturnValue, ReturnValue]]:
    """为本实例所表示的函数定义一条自定义 JVP 规则。

    Args:
      jvp: 表示自定义 JVP 规则的 Python 可调用对象。当没有 ``nondiff_argnums``
        时，``jvp`` 函数应接受两个参数：第一个是原始输入（primal）的元组，
        第二个是切向量输入（tangent）的元组。两个元组的长度都等于
        :class:`~jax.custom_jvp` 函数的参数个数。``jvp`` 函数应输出一个二元组，
        其中第一个元素是原始输出，第二个元素是切向量输出。输入和输出元组的
        元素可以是数组，也可以是它们任意嵌套的 tuple/list/dict。
      symbolic_zeros: 布尔值，表示是否在切向量参数中传入代表静态符号零的对象，
        以与未被扰动的值相对应；否则只传入标准 JAX 类型（例如类数组对象）。
        将该选项设为 ``True`` 可让 JVP 规则检测某些输入是否不参与微分，
        代价是必须对这类对象做特殊处理（例如它们无法传入 jax.numpy 函数）。
        默认为 ``False``。

    Returns:
      返回 ``jvp``，以便 ``defjvp`` 可用作装饰器。

    Examples:

      >>> @jax.custom_jvp
      ... def f(x, y):
      ...   return jnp.sin(x) * y
      ...
      >>> @f.defjvp
      ... def f_jvp(primals, tangents):
      ...   x, y = primals
      ...   x_dot, y_dot = tangents
      ...   primal_out = f(x, y)
      ...   tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
      ...   return primal_out, tangent_out

      >>> x = jnp.float32(1.0)
      >>> y = jnp.float32(2.0)
      >>> with jnp.printoptions(precision=2):
      ...   print(jax.value_and_grad(f)(x, y))
      (Array(1.68, dtype=float32), Array(1.08, dtype=float32))
    """
    self.jvp = jvp
    self.symbolic_zeros = symbolic_zeros
    return jvp

  def defjvps(self, *jvps: Callable[..., ReturnValue] | None) -> None:
    """用于为每个参数分别定义 JVP 的便捷包装器。

    该便捷包装器不能与 ``nondiff_argnums`` 一起使用。

    Args:
      *jvps: 一个函数序列，为 :class:`~jax.custom_jvp` 函数的每个位置参数各
        提供一个函数。每个函数接受的参数依次是：对应原始输入的切向量值、
        原始输出以及各个原始输入。参见下面的例子。

    Returns:
      None。

    Examples:

      >>> @jax.custom_jvp
      ... def f(x, y):
      ...   return jnp.sin(x) * y
      ...
      >>> f.defjvps(lambda x_dot, primal_out, x, y: jnp.cos(x) * x_dot * y,
      ...           lambda y_dot, primal_out, x, y: jnp.sin(x) * y_dot)

      >>> x = jnp.float32(1.0)
      >>> y = jnp.float32(2.0)
      >>> with jnp.printoptions(precision=2):
      ...   print(jax.value_and_grad(f)(x, y))
      (Array(1.68, dtype=float32), Array(1.08, dtype=float32))
    """
    if self.nondiff_argnums:
      raise TypeError("Can't use ``defjvps`` with ``nondiff_argnums``.")

    def jvp(primals, tangents):
      primal_out = self(*primals)
      zeros = _zeros_like_pytree(primal_out)
      all_tangents_out = [jvp(t, primal_out, *primals) if jvp else zeros
                          for t, jvp in zip(tangents, jvps)]
      tangent_out = tree_map(_sum_tangents, primal_out, *all_tangents_out)
      return primal_out, tangent_out

    self.defjvp(jvp)

  @partial(traceback_util.api_boundary,
           repro_api_name="jax.custom_jvp.__call__")
  def __call__(self, *args: Any, **kwargs: Any) -> ReturnValue:
    debug = debug_info("custom_jvp fun", self.fun, args, kwargs,
                       static_argnums=self.nondiff_argnums)
    primal_name = debug.func_name
    if not self.jvp:
      msg = f"No JVP defined for custom_jvp function {primal_name} using defjvp."
      raise AttributeError(msg)

    try:
      args = resolve_kwargs(self.fun, args, kwargs)
    except TypeError as e:
      raise TypeError(
          "The input arguments to the custom_jvp-decorated function "
          f"{primal_name} could not be resolved to positional-only arguments. "
          f"Binding failed with the error:\n{e}"
      ) from e

    if self.nondiff_argnums:
      args = tuple(_stop_gradient(x) if i in self.nondiff_argnums else x
                   for i, x in enumerate(args))
      diff_argnums = [i for i in range(len(args)) if i not in self.nondiff_argnums]
      f_, dyn_args = argnums_partial(lu.wrap_init(self.fun, debug_info=debug),
                                     diff_argnums, args,
                                     require_static_args_hashable=False)
      static_args = [args[i] for i in self.nondiff_argnums]
      diff_args = [args[i] for i, a in enumerate(args) if i not in self.nondiff_argnums]
      debug_jvp = debug_info("custom_jvp jvp", self.jvp,
                             (*static_args, diff_args, diff_args),
                             {},
                             static_argnums=tuple(range(len(static_args))))
      jvp = prepend_static_args(lu.wrap_init(self.jvp,
                                             debug_info=debug_jvp), static_args)
    else:
      f_, dyn_args = lu.wrap_init(self.fun, debug_info=debug), args
      debug_jvp = debug_info("custom_jvp jvp", self.jvp,
                             (args, args),
                             {})
      jvp = lu.wrap_init(self.jvp, debug_info=debug_jvp)
    args_flat, in_tree = tree_flatten(dyn_args)
    flat_fun, out_type1 = _flatten_fun_nokwargs(f_, in_tree)
    flat_jvp, out_type2 = _flatten_jvp(jvp, primal_name, debug_jvp.func_name,
                                       in_tree, out_type1)
    out_flat = custom_jvp_call_p.bind(*args_flat, subfuns=(flat_fun, flat_jvp),
                                      symbolic_zeros=self.symbolic_zeros)
    _, (out_tree, _, _) = lu.merge_linear_aux(out_type1, out_type2)
    return tree_unflatten(out_tree, out_flat)

@partial(lu.transformation_with_aux2, use_eq_store=True)
def _flatten_jvp(f, store, primal_name, jvp_name, in_tree, maybe_out_type, *args):
  primals_in, tangents_in = split_list(args, [len(args) // 2])
  py_primals = tree_unflatten(in_tree, primals_in)
  py_tangents = tree_unflatten(in_tree, tangents_in)
  pair_out = f(py_primals, py_tangents)
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    msg = (f"Custom JVP rule {jvp_name} for function {primal_name} "
           "must produce a pair (list or tuple of length two) representing "
           f"primal and tangent outputs, but got {pair_out}.")
    raise TypeError(msg)
  py_primals_out, py_tangents_out = pair_out
  primals_out, out_tree = tree_flatten(py_primals_out)
  tangents_out, out_tree2 = tree_flatten(py_tangents_out)
  primal_avals = [core.typeof(x) for x in primals_out]
  if out_tree != out_tree2:
    msg = (f"Custom JVP rule {jvp_name} for function {primal_name} must "
           "produce primal and tangent outputs with equal container (pytree) "
           f"structures, but got {out_tree} and {out_tree2} respectively.")
    raise TypeError(msg)
  # 如果原始函数已经运行过，则检查 out_tree 是否一致。
  try: out_type_ = maybe_out_type()
  except lu.StoreException: out_type_ = None
  if out_type_ is not None:
    out_tree_, primal_avals_, () = out_type_
    ty_tree  = tree_unflatten(out_tree , [a.str_short() for a in primal_avals])
    ty_tree_ = tree_unflatten(out_tree_, [a.str_short() for a in primal_avals_])
    if out_tree_ != out_tree:
      m = (f"Custom JVP rule {jvp_name} for function {primal_name} must "
           "produce a pair (list or tuple of length two) "
           "where the first element represents the primal output "
           "(equal in value to the output of the custom_jvp-decorated function "
           f"{primal_name}, "
           "and in particular of the same container/pytree structure), but "
           "instead the JVP rule output's first element had container/pytree "
           "structure:\n"
           f"""    {str(ty_tree ).replace("'", "")}\n"""
           f"while the custom_jvp-decorated function {primal_name} had output "
           "container/pytree structure:\n"
           f"""    {str(ty_tree_).replace("'", "")}.""")
      raise TypeError(m)
    if not all(map(core.typematch, primal_avals, primal_avals_)):
      m = (f"Custom JVP rule {jvp_name} for function {primal_name} must "
           "produce a pair (list or tuple of length two) "
           "where the first element represents the primal output "
           "(equal in value to the output of the custom_jvp-decorated function "
           f"{primal_name}, "
           "and in particular with leaves of the same shape/dtype), but "
           "instead the JVP rule output's first element had shapes/dtypes of:\n"
           f"""    {str(ty_tree ).replace("'", "")}\n"""
           f"while the custom_jvp-decorated function {primal_name} had output "
           "shapes/dtypes of:\n"
           f"""    {str(ty_tree_).replace("'", "")}""")
      raise TypeError(m)
  primal_avals_out = [core.typeof(x).strip_weak_type() for x in primals_out]
  expected_tangent_avals_out = [
    core.typeof(x).strip_weak_type().to_tangent_aval()
    for x in primals_out]
  tangent_avals_out = [core.typeof(t).strip_weak_type()
                       if type(t) is not SymbolicZero else t.aval.strip_weak_type()
                       for t in tangents_out]
  if not all(map(core.typematch, expected_tangent_avals_out, tangent_avals_out)):
    if len(expected_tangent_avals_out) == 1:
      (av_p,), (av_et,), (av_t,) = primal_avals_out, expected_tangent_avals_out, tangent_avals_out
      msg = ("Custom JVP rule must produce primal and tangent outputs with "
             "corresponding shapes and dtypes. Expected {} (tangent type of {}) but got {}.")
      raise TypeError(msg.format(av_et.str_short(), av_p.str_short(), av_t.str_short()))
    else:
      msg = ("Custom JVP rule must produce primal and tangent outputs with "
             "corresponding shapes and dtypes, but got:\n{}")
      disagreements = (
          f"  primal {av_p.str_short()} with tangent {av_t.str_short()}, expecting tangent {av_et}"
          for av_p, av_et, av_t in zip(primal_avals_out, expected_tangent_avals_out, tangent_avals_out)
          if av_et != av_t)
      raise TypeError(msg.format('\n'.join(disagreements)))
  store.store((out_tree, primal_avals, ()))
  return primals_out + tangents_out

class CustomJVPCallPrimitive(core.Primitive):
  multiple_results = True
  skip_canonicalization = True

  def bind_with_trace(self, trace, args, avals, params, /):
    params = dict(params)
    fun, jvp = params.pop('subfuns')
    return trace.process_custom_jvp_call(self, fun, jvp, args, **params)

  def impl(self, fun, _, *args):
    raise NotImplementedError

  def is_high(self, *_, call_jaxpr, **__):
    return call_jaxpr.is_high

  def to_lojax(self, *hi_args, call_jaxpr: core.Jaxpr, **params):
    return pe._lower_and_eval(pe.eval_jaxpr_p, call_jaxpr, hi_args)

  def get_bind_params(self, params):
    new_params = dict(params)
    call_jaxpr: core.Jaxpr = new_params.pop('call_jaxpr')
    num_consts: int = new_params.pop('num_consts')
    jvp_jaxpr_fun = new_params.pop('jvp_jaxpr_fun')
    fun = lu.wrap_init(core.jaxpr_as_fun(call_jaxpr),
                       debug_info=call_jaxpr.debug_info)
    jvp = lift_jvp(num_consts, jvp_jaxpr_fun)
    new_params['subfuns'] = (fun, jvp)
    return new_params

def lift_jvp(num_consts: int, jvp_jaxpr_fun: lu.WrappedFun) -> lu.WrappedFun:
  def jvp(*xs):
    n, ragged = divmod(len(xs), 2)
    assert not ragged
    primals, tangents = xs[num_consts:n], xs[n+num_consts:]
    zeros = [type(t) is SymbolicZero for t in tangents]
    jvp_jaxpr, jvp_consts, out_zeros = jvp_jaxpr_fun.call_wrapped(*zeros)
    nonzero_tangents = [t for t in tangents if type(t) is not SymbolicZero]
    out = core.eval_jaxpr(jvp_jaxpr, jvp_consts, *primals, *nonzero_tangents)
    out_primals, nz_out_tangents = split_list(out, [len(out_zeros)])
    nz_out_tangents_ = iter(nz_out_tangents)
    out_tangents = [SymbolicZero(core.typeof(p).to_tangent_aval())
                    if z else next(nz_out_tangents_)
                    for p, z in zip(out_primals, out_zeros)]
    assert next(nz_out_tangents_, None) is None
    return [*out_primals, *out_tangents]
  return lu.wrap_init(jvp, debug_info=jvp_jaxpr_fun.debug_info)

custom_jvp_call_p = CustomJVPCallPrimitive('custom_jvp_call')

def _custom_jvp_call_typecheck(_, *in_avals, call_jaxpr, jvp_jaxpr_fun,
                               num_consts, symbolic_zeros):
  # TODO(mattjj): 这里还可以做更多检查...
  del in_avals, jvp_jaxpr_fun, num_consts
  disallowed_effects = effects.custom_derivatives_allowed_effects.filter_not_in(call_jaxpr.effects)
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `custom_jvp`: {disallowed_effects}')
  return call_jaxpr.out_avals, core.positional_effects(call_jaxpr)
core.custom_typechecks[custom_jvp_call_p] = _custom_jvp_call_typecheck

def _custom_jvp_vjp_call_lowering(ctx: mlir.LoweringRuleContext, *args,
                                  call_jaxpr: core.Jaxpr, **_):
  consts = mlir.ir_consts(
      call_jaxpr.consts, [v.aval for v in call_jaxpr.constvars])
  out, tokens = mlir.jaxpr_subcomp(ctx.module_context, call_jaxpr,
                                   ctx.name_stack, ctx.tokens_in, consts,
                                   *args, dim_var_values=ctx.dim_var_values,
                                   const_lowering=ctx.const_lowering,
                                   outer_traceback=ctx.traceback)
  ctx.set_tokens_out(tokens)
  return out
mlir.register_lowering(custom_jvp_call_p, _custom_jvp_vjp_call_lowering)

def _custom_jvp_call_transpose_fancy(params, jaxpr, args, ct, _):
  del params
  return ad.backward_pass3(jaxpr, False, jaxpr.consts, args, ct)
ad.fancy_transposes[custom_jvp_call_p] = _custom_jvp_call_transpose_fancy

@weakref_lru_cache
def _cached_closed_call_dce_instantiate(jaxpr_: core.Jaxpr,
                                        used_outputs: tuple[bool, ...]
                                        ) -> tuple[core.Jaxpr, list[bool]]:
  # dce_jaxpr 与 replace 都会保留挂载的常量。
  return pe.dce_jaxpr(
      jaxpr_.replace(debug_info=jaxpr_.debug_info.with_unknown_names()),
      used_outputs, True)

def _custom_jvp_call_dce(
    used_outs: Sequence[bool], eqn: core.JaxprEqn
) -> tuple[list[bool], core.JaxprEqn | None]:
  if not any(used_outs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None

  call_jaxpr = eqn.params["call_jaxpr"]
  jvp_jaxpr_fun = eqn.params["jvp_jaxpr_fun"]
  # 必须设置 instantiate=True，因为经 DCE 后的原始函数未使用的某些输入
  # 仍可能在 JVP 规则中被用到。
  dce_call_jaxpr, used_ins = _cached_closed_call_dce_instantiate(
      call_jaxpr, tuple(used_outs))
  assert all(used_ins)

  @pe._memoize
  def dce_jvp_jaxpr_thunk(*in_zeros):
    jvp_jaxpr, consts, out_zeros = jvp_jaxpr_fun.call_wrapped(*in_zeros)
    sz = eqn.params["symbolic_zeros"]
    nz_used_outs = [u for u, z in zip(used_outs, out_zeros) if not z] if sz else used_outs
    dce_jvp_jaxpr, _ = pe.dce_jaxpr(jvp_jaxpr, [*used_outs, *nz_used_outs], True)
    dce_out_zeros = [v for used, v in zip(used_outs, out_zeros) if used]
    return dce_jvp_jaxpr, consts, dce_out_zeros

  outvars = [v for used, v in zip(used_outs, eqn.outvars) if used]
  new_params = dict(
      eqn.params,
      call_jaxpr=dce_call_jaxpr,
      jvp_jaxpr_fun=lu.wrap_init(dce_jvp_jaxpr_thunk,
                                 debug_info=jvp_jaxpr_fun.debug_info)
  )
  new_eqn = pe.new_jaxpr_eqn(
      eqn.invars, outvars, eqn.primitive, new_params,
      core.eqn_effects(dce_call_jaxpr, eqn.invars),
      eqn.source_info, eqn.ctx)
  return used_ins, new_eqn
pe.dce_rules[custom_jvp_call_p] = _custom_jvp_call_dce


def _custom_jvp_call_pp_rule(eqn: core.JaxprEqn,
                             context: core.JaxprPpContext,
                             settings: core.JaxprPpSettings) -> core.pp.Doc:
  params = dict(eqn.params)
  if not params["num_consts"]:
    params.pop("num_consts")
  params["jvp"] = params.pop("jvp_jaxpr_fun").debug_info.func_name
  names = sorted(params)
  params["name"] = params["call_jaxpr"].debug_info.func_name
  return core._pp_eqn(eqn.replace(params=params), context, settings,
                      params=["name"] + names)


core.pp_eqn_rules[custom_jvp_call_p] = _custom_jvp_call_pp_rule

### VJP

@custom_api_util.register_custom_decorator_type
class custom_vjp[ReturnValue]:
  """注册一个可被 JAX 变换的函数，以便为它定义自定义 VJP 规则。

  该类用作函数装饰器。其实例是可调用对象，行为与被装饰的底层函数相似，
  区别在于：当施加反向模式微分变换（如 :py:func:`jax.grad`）时，
  会改用用户提供的自定义 VJP 规则函数，而不是追踪进底层函数的实现体
  并对其执行自动微分。该类只有一个实例方法
  :py:func:`~jax.custom_vjp.defvjp`，可用于定义自定义 VJP 规则。

  该装饰器会禁止使用正向模式自动微分。

  例如::

    @jax.custom_vjp
    def f(x, y):
      return jnp.sin(x) * y

    def f_fwd(x, y):
      return f(x, y), (jnp.cos(x), jnp.sin(x), y)

    def f_bwd(res, g):
      cos_x, sin_x, y = res
      return (cos_x * g * y, sin_x * g)

    f.defvjp(f_fwd, f_bwd)

  更详细的介绍参见教程 tutorial_。

  .. _tutorial: https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
  """

  def __new__(cls, fun=None, nondiff_argnums=(), nondiff_argnames=()):
    if fun is not None and config.custom_vjp3.value:
      from jax._src.hijax import custom_vjp3  # pyrefly: ignore[missing-import]
      return custom_vjp3(fun, nondiff_argnums, nondiff_argnames)
    else:
      return super().__new__(cls)

  def __init__(self,
               fun: Callable[..., ReturnValue],
               nondiff_argnums: Sequence[int] = (),
               nondiff_argnames: Sequence[str] = ()):
    update_wrapper(self, fun)
    self.fun = fun

    nondiff_argnums_: set[int] = set()
    if nondiff_argnames:
      sig = fun_signature(self.fun)
      assert sig is not None
      inferred_nondiff_argnums, _ = infer_argnums_and_argnames(
          sig, None, nondiff_argnames
      )
      nondiff_argnums_.update(inferred_nondiff_argnums)

    if nondiff_argnums:
      nondiff_argnums_.update(nondiff_argnums)

    self.nondiff_argnums = tuple(sorted(nondiff_argnums_))
    self.fwd: Callable[..., tuple[ReturnValue, Any]] | None = None
    self.bwd: Callable[..., tuple[Any, ...]] | None = None
    self.symbolic_zeros = False
    self.optimize_remat = False
    self.with_logs = False

  __getattr__ = custom_api_util.forward_attr

  def defvjp(self,
             fwd: Callable[..., tuple[ReturnValue, Any]],
             bwd: Callable[..., tuple[Any, ...]],
             symbolic_zeros: bool = False,
             optimize_remat: bool = False,
             ) -> None:
    """为本实例所表示的函数定义一条自定义 VJP 规则。

    Args:
      fwd: 表示自定义 VJP 规则前向传播的 Python 可调用对象。当没有
        ``nondiff_argnums`` 时，``fwd`` 函数与底层原始函数具有相同的输入签名。
        它应输出一个二元组，其中第一个元素表示原始输出，第二个元素表示前向传播中
        需要保存、供 ``bwd`` 函数在反向传播时使用的任意“残差”值。输入参数以及
        输出二元组的元素可以是数组，也可以是它们任意嵌套的 tuple/list/dict。
      bwd: 表示自定义 VJP 规则反向传播的 Python 可调用对象。当没有
        ``nondiff_argnums`` 时，``bwd`` 函数接受两个参数：第一个是 ``fwd``
        在前向传播中产生的“残差”值，第二个是与原始函数输出结构相同的输出余切。
        ``bwd`` 的输出必须是一个元组，其长度等于原始函数的参数个数；元组元素
        可以是数组，也可以是它们任意嵌套的 tuple/list/dict，以便与原始输入
        参数的结构相匹配。
      symbolic_zeros: 布尔值，决定是否向 ``fwd`` 和 ``bwd`` 规则指示符号零。
        启用该选项可让自定义导数规则检测某些输入以及某些输出余切是否不参与微分。
        若为 ``True``：

        * ``fwd`` 必须改为接受一个对象（类型为
          ``jax.custom_derivatives.CustomVJPPrimal``）来代替构成原始函数某个
          参数的 pytree 中的每个叶值 ``x``；该对象带有两个属性：``value`` 和
          ``perturbed``。``value`` 字段就是原始的 primal 参数，``perturbed``
          是一个布尔值。该 ``perturbed`` 位表示此参数是否参与微分
          （即若为 ``False``，则对应的 Jacobian “列”为零）。

        * ``bwd`` 会在其余切参数中收到代表静态符号零的对象，以与未被扰动的值
          相对应；否则只传入标准 JAX 类型（例如类数组对象）。

        将该选项设为 ``True`` 可让这些规则检测某些输入和输出是否不参与微分，
        代价是需要特殊处理。例如：

        * ``fwd`` 的签名会改变，并且传给它的对象不能由该规则直接输出。

        * 传给 ``bwd`` 规则的对象并非完全是类数组的，无法传给大多数
          ``jax.numpy`` 函数。

        * 原始函数参数中涉及的任何自定义 pytree 节点，其反扁平化函数必须能接受
          作为输入叶值传给 ``fwd`` 规则的双字段记录对象。

        默认为 ``False``。
      optimize_remat: 布尔值，一个实验性开关：当该函数在 :func:`jax.remat` 下
        使用时启用自动优化。当 ``fwd`` 规则是不透明的调用（例如 Pallas kernel
        或自定义调用）时，这一优化最为有用。默认为 ``False``。

    Returns:
      None。

    Examples:

      >>> @jax.custom_vjp
      ... def f(x, y):
      ...   return jnp.sin(x) * y
      ...
      >>> def f_fwd(x, y):
      ...   return f(x, y), (jnp.cos(x), jnp.sin(x), y)
      ...
      >>> def f_bwd(res, g):
      ...   cos_x, sin_x, y = res
      ...   return (cos_x * g * y, sin_x * g)
      ...
      >>> f.defvjp(f_fwd, f_bwd)

      >>> x = jnp.float32(1.0)
      >>> y = jnp.float32(2.0)
      >>> with jnp.printoptions(precision=2):
      ...   print(jax.value_and_grad(f)(x, y))
      (Array(1.68, dtype=float32), Array(1.08, dtype=float32))
    """
    self.fwd = fwd
    self.bwd = bwd
    self.symbolic_zeros = symbolic_zeros
    self.optimize_remat = optimize_remat
    if self.symbolic_zeros and self.optimize_remat:
      raise NotImplementedError(
          "remat optimization for custom_vjp does not support symbolic zeros")

  def defvjp_with_logs(self,
                       fwd: Callable[..., tuple[ReturnValue, Any]],
                       bwd: Callable[..., tuple[tuple[Any, ...], dict | None]],
                       symbolic_zeros: bool = False,
                       optimize_remat: bool = False,
                       ) -> None:
    """类似于 :py:func:`~jax.custom_vjp.defvjp`，但 ``bwd`` 还可以记录日志。

    与 ``defvjp`` 的唯一区别在于 ``bwd`` 的返回约定：它必须返回一个二元组
    ``(in_cts, logs)``，其中 ``in_cts`` 是通常的余切元组（每个原始参数一项），
    ``logs`` 是一个把命名 pytree 记录到反向传播之外的 dict，若什么都不记录则为
    ``None``。要接收这些日志，请通过调用 VJP 函数的 ``with_logs`` 方法：
    ``f_vjp.with_logs(out_ct)`` 返回一个二元组 ``(arg_cts, logs)``。
    日志默认被丢弃：直接调用 ``f_vjp(out_ct)`` 会忽略日志，而在 ``jit`` 下
    记录日志的计算会被死代码消除。
    """
    self.defvjp(fwd, bwd, symbolic_zeros=symbolic_zeros,
                optimize_remat=optimize_remat)
    self.with_logs = True

  @partial(traceback_util.api_boundary,
           repro_api_name="jax.custom_vjp.__call__")
  def __call__(self, *args: Any, **kwargs: Any) -> ReturnValue:
    debug_fun = debug_info("custom_vjp fun", self.fun, args, kwargs,
                           static_argnums=self.nondiff_argnums)
    if not self.fwd or not self.bwd:
      msg = f"No VJP defined for custom_vjp function {debug_fun.func_name} using defvjp."
      raise AttributeError(msg)

    try:
      args = resolve_kwargs(self.fun, args, kwargs)
    except TypeError as e:
      raise TypeError(
          "The input arguments to the custom_vjp-decorated function "
          f"{debug_fun.func_name} could not be resolved to positional-only "
          f"arguments. Binding failed with the error:\n{e}"
      ) from e

    debug_fwd = debug_info("custom_vjp fwd", self.fwd, args, kwargs,
                           static_argnums=self.nondiff_argnums)
    # TODO(necula): 需要弄清如何构造 debug_bwd 的参数
    debug_bwd = debug_info("custom_vjp bwd", self.bwd, args, {})
    if self.optimize_remat:
      fwd = optimize_remat_of_custom_vjp_fwd(
          self.fun, debug_fun, self.fwd, debug_fwd,
          nondiff_argnums=self.nondiff_argnums,
          symbolic_zeros=self.symbolic_zeros)
    else:
      fwd = self.fwd
    if self.nondiff_argnums:
      for i in self.nondiff_argnums: _check_for_tracers(args[i])
      dyn_argnums = [i for i in range(len(args)) if i not in self.nondiff_argnums]
      f_, dyn_args = argnums_partial(
          lu.wrap_init(self.fun, debug_info=debug_fun), dyn_argnums,
          args, require_static_args_hashable=False)
      static_args = [args[i] for i in self.nondiff_argnums]
      fwd_, _ = argnums_partial(lu.wrap_init(fwd, debug_info=debug_fwd),
                                dyn_argnums, args,
                                require_static_args_hashable=False)
      bwd = prepend_static_args(lu.wrap_init(self.bwd, debug_info=debug_bwd),
                                static_args)
    else:
      f_, dyn_args = lu.wrap_init(self.fun, debug_info=debug_fun), args
      fwd_ = lu.wrap_init(fwd, debug_info=debug_fwd)
      bwd = lu.wrap_init(self.bwd, debug_info=debug_bwd)
    args_flat, in_tree = tree_flatten(dyn_args)
    in_avals = tuple(core.typeof(x) for x in args_flat)
    if config.mutable_array_checks.value:
      f_ = _check_primal_refs(f_, self.nondiff_argnums, f_.debug_info)
    flat_fun, out_type = _flatten_fun_nokwargs(f_, in_tree)
    flat_fwd, out_trees = _flatten_fwd(
        fwd_, self.nondiff_argnums, self.symbolic_zeros, debug_fun,
        debug_fwd, in_tree, out_type)
    flat_bwd = _flatten_bwd(bwd, in_tree, in_avals, out_trees, self.fun,
                            self.with_logs)
    out_flat = custom_vjp_call_p.bind(*args_flat, subfuns=(flat_fun, flat_fwd, flat_bwd),
                                      out_trees=out_trees,
                                      symbolic_zeros=self.symbolic_zeros)
    _, (out_tree, _, _) = lu.merge_linear_aux(out_type, out_trees)
    return tree_unflatten(out_tree, out_flat)
@lu.transformation2
def _check_primal_refs(
    f: Callable, nondiff_argnums: Sequence[int], debug: core.DebugInfo, *args):
  _check_for_aliased_refs(f, nondiff_argnums, debug, args)
  out = f(*args)
  _check_for_returned_refs(f, out, 'primal', [], 0)
  return out

def _check_for_aliased_refs(
    f: Callable, nondiff_argnums: Sequence[int], debug: core.DebugInfo, args):
  argnums = [x for i, arg in enumerate(args)
             for x in [i] * tree_structure(arg).num_leaves]
  leaves = tree_leaves(args)
  refs: dict[int, int] = {}
  for i, (argnum, x) in enumerate(zip(argnums, leaves)):
    if argnum in nondiff_argnums: continue
    x = x.value if isinstance(x, CustomVJPPrimal) else x
    if (isinstance((a := core.typeof(x)), AbstractRef) and
        (dup_idx := refs.setdefault(id(core.get_referent(x)), i)) != i):
      arg_names = debug.safe_arg_names(len(leaves))
      raise ValueError(
          "only one reference to a mutable array may be passed as an argument "
          f"to a function, but custom_vjp function {f} got the same mutable "
          f"array reference of type {a.str_short()} at {arg_names[dup_idx]} and"
          f" {arg_names[i]}.")

def _check_for_returned_refs(f, out, kind, args, after_idx):
  args = [x.value if isinstance(x, CustomVJPPrimal) else x for x in args]
  ids = {id(x) for x in args if isinstance(core.typeof(x), AbstractRef)}
  leaves = tree_leaves_with_path(out)
  for i, (path, leaf) in enumerate(leaves):
    if isinstance((a := core.typeof(leaf)), AbstractRef):
      loc = f' at output tree path {keystr(path)}' if path else ''
      if i < after_idx:
        raise ValueError(f"custom_vjp {kind} function {f} returned a mutable "
                         f"array reference of type {a.str_short()}{loc}, "
                         "but mutable array references cannot be returned there.")
      if id(leaf) not in ids:
        raise ValueError(f"custom_vjp {kind} function {f} returned a mutable "
                         f"array reference of type {a.str_short()}{loc} "
                         "that was not an argument.")

@dataclasses.dataclass(slots=True)
class CustomVJPPrimal:
  """设置了 ``symbolic_zeros`` 时 ``custom_vjp`` 前向规则的原始值"""
  value: Any
  perturbed: bool

def custom_vjp_primal_tree_values(tree):
  """从正向规则的参数中剥离扰动信息。

  这是一个辅助函数，供使用 ``custom_vjp`` 装饰函数的 ``defvjp`` 方法中
  ``symbolic_zeros`` 选项的用户使用。

  在 ``symbolic_zeros`` 模式下，自定义正向规则收到的参数，其 pytree
  叶子是带有 ``value`` 属性的记录，该属性承载原始值参数。此函数把这类
  参数树还原为原始形式，即在每个叶子上把这类记录替换为其承载的值。
  """
  def value(leaf):
    if type(leaf) is not CustomVJPPrimal:
      raise TypeError(f"unexpected leaf type {type(leaf)}")
    return leaf.value
  return tree_map(value, tree)

def _check_for_tracers(x):
  for leaf in tree_leaves(x):
    if isinstance(leaf, core.Tracer):
      msg = ("Found a JAX Tracer object passed as an argument to a custom_vjp "
            "function in a position indicated by nondiff_argnums as "
            "non-differentiable. Tracers cannot be passed as non-differentiable "
            "arguments to custom_vjp functions; instead, nondiff_argnums should "
            "only be used for arguments that can't be or contain JAX tracers, "
            "e.g. function-valued arguments. In particular, array-valued "
            "arguments should typically not be indicated as nondiff_argnums.")
      raise UnexpectedTracerError(msg)

@partial(lu.transformation_with_aux2, use_eq_store=True)
def _flatten_fwd(f: Callable, store: lu.EqualStore,
                 nondiff_argnums: Sequence[int],
                 symbolic_zeros: bool,
                 debug_primal: core.DebugInfo,
                 debug_fwd: core.DebugInfo,
                 in_tree: PyTreeDef, maybe_out_type, *args):
  primal_name = debug_primal.func_name if debug_primal else str(f)
  fwd_name = debug_fwd.func_name if debug_fwd else "<unknown>"
  if symbolic_zeros:
    args = tuple(CustomVJPPrimal(x, z) for x, z in zip(args[::2], args[1::2]))
  else:
    args = args[::2]
  py_args = tree_unflatten(in_tree, args)
  if config.mutable_array_checks.value:
    _check_for_aliased_refs(f, nondiff_argnums, debug_primal, py_args)
  pair_out = f(*py_args)
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    msg = (f"Custom VJP fwd rule {fwd_name} for function {primal_name} "
           "must produce a pair (list or tuple of length two) where the first "
           "element represents the primal output (equal to those of the "
           f"custom_vjp-decorated function {primal_name}) and the "
           "second element represents residuals (i.e. values stored from the "
           "forward pass for use on the backward pass), but "
           f"instead of a pair the fwd rule {fwd_name} produced {pair_out}.")
    raise TypeError(msg)
  py_primals_out, res = pair_out
  primals_out, out_tree = tree_flatten(py_primals_out)
  res, res_tree = tree_flatten(res)
  if config.mutable_array_checks.value:
    _check_for_returned_refs(f, pair_out, "fwd", args, out_tree.num_leaves)
  primal_avals = [core.typeof(x) for x in primals_out]
  # 如果原始函数已经运行过，则检查 out_tree 是否一致。
  try: out_type_ = maybe_out_type()
  except lu.StoreException: out_type_ = None
  if out_type_ is not None:
    out_tree_, primal_avals_, () = out_type_
    ty_tree  = tree_unflatten(out_tree , [a.str_short() for a in primal_avals])
    ty_tree_ = tree_unflatten(out_tree_, [a.str_short() for a in primal_avals_])
    if out_tree_ != out_tree:
      m = (f"Custom VJP fwd rule {fwd_name} for function {primal_name} "
           "must produce a pair (list or tuple of length two) where the first "
           "element represents the primal output "
           "(equal to the output of the custom_vjp-decorated function "
           f"{primal_name}) and the "
           "second element represents residuals (i.e. values stored from the "
           "forward pass for use on the backward pass), but "
           "instead the fwd rule output's first element had container/pytree "
           "structure:\n"
           f"""    {str(ty_tree ).replace("'", "")}\n"""
           f"while the custom_vjp-decorated function {primal_name} had output "
           "container/pytree structure:\n"
           f"""    {str(ty_tree_).replace("'", "")}.""")
      raise TypeError(m)
    if not all(map(core.typematch, primal_avals, primal_avals_)):
      m = (f"Custom VJP fwd rule {fwd_name} for function {primal_name} must "
           "produce a pair (list or tuple of length two) "
           "where the first element represents the primal output "
           "(equal to the output of the custom_vjp-decorated function "
           f"{primal_name}) and the second element represents residuals "
           "(i.e. values stored from the forward pass for use on the "
           "backward pass), but "
           "instead the fwd rule output's first element had shapes/dtypes of:\n"
           f"""    {str(ty_tree ).replace("'", "")}\n"""
           f"while the custom_vjp-decorated function {primal_name} had output "
           "shapes/dtypes of:\n"
           f"""    {str(ty_tree_).replace("'", "")}""")
      raise TypeError(m)
  pruned_res, input_forwards = _filter_forwarded_inputs(res, args)  # 剪枝
  store.store((out_tree, res_tree, input_forwards))
  return (*pruned_res, *primals_out)

def _filter_forwarded_inputs(outs, ins):
  idxs: dict[int, int] = {id(x): i for i, x in enumerate(ins)}
  return [o for o in outs if id(o) not in idxs], [idxs.get(id(o)) for o in outs]

@lu.transformation2
def _flatten_bwd(f: Callable,
                 in_tree: PyTreeDef,
                 in_avals: Sequence[core.AbstractValue],  # 输入原始值的抽象值(aval)
                 out_trees: Callable[[], tuple[PyTreeDef, PyTreeDef, list[int | None]]],
                 primal_fun, with_logs: bool, *args):
  out_tree, res_tree, _ = out_trees()
  assert len(args) == res_tree.num_leaves + out_tree.num_leaves
  res, cts_out = split_list(args, [res_tree.num_leaves])
  py_res = tree_unflatten(res_tree, res)
  py_cts_out = tree_unflatten(out_tree, cts_out)
  py_cts_in = f(py_res, py_cts_out)
  if with_logs:
    if not (isinstance(py_cts_in, (list, tuple)) and len(py_cts_in) == 2):
      raise TypeError(
          "Custom VJP bwd rule was registered with defvjp_with_logs and so "
          f"must produce a pair (in_cts, logs), but got {py_cts_in}.")
    py_cts_in, logs = py_cts_in
    if logs is not None and type(logs) is not dict:
      raise TypeError(
          "Custom VJP bwd rule was registered with defvjp_with_logs, and so "
          "the second element of the pair it returns must be None or a dict "
          f"of backward-pass log entries, but got {type(logs).__name__}.")
  else:
    logs = None
  if isinstance(py_cts_in, list) and len(py_cts_in) == len(treedef_children(in_tree)):
    py_cts_in = tuple(py_cts_in)
  # 对于 py_cts_in 中每个 None（表示规则不为其产生切向量的参数），
  # 我们把它替换为一个 pytree，其结构与 in_tree 的对应子树相同，
  # 其叶子是非 pytree 的哨兵值对象；这些哨兵值会在最终返回的
  # 结果中被替换回 None。
  zero = object()  # 非 pytree 哨兵值，用于替换 py_cts_in 中的 None
  dummy = tree_unflatten(in_tree, [object()] * in_tree.num_leaves)
  keypaths, _ = unzip2(tree_flatten_with_path(dummy)[0])
  cts_in_flat = []
  def append(x, d):
    num_leaves = len(tree_flatten(d)[0])
    if x is None and d is not None:
      cts_in_flat.extend([zero] * num_leaves)
    elif x is not None:
      cts_in_flat.extend([x] * num_leaves)
    return x
  try:
    if not isinstance(py_cts_in, tuple):
      raise ValueError
    tree_map(append, py_cts_in, dummy, is_leaf=lambda x: x is None)
  except ValueError:
    _, in_tree2 = tree_flatten(py_cts_in)
    msg = ("Custom VJP bwd rule must produce an output with the same container "
           "(pytree) structure as the args tuple of the primal function, "
           "and in particular must produce a tuple of length equal to the "
           "number of arguments to the primal function, but got bwd output "
           "structure {} for primal input structure {}.")
    raise TypeError(msg.format(in_tree2, in_tree)) from None
  results: list[Any] = []
  for kp, a, ct in zip(keypaths, in_avals, cts_in_flat):
    if ct is zero or getattr(a.to_ct_aval(), 'dtype') == dtypes.float0:
      results.append(Zero(a.to_ct_aval()))
    elif type(ct) is SymbolicZero:
      if not core.typecompat(a.to_ct_aval(), a_ := ct.aval):
        msg = ("Custom VJP bwd rule produced a SymbolicZero with a shape/dtype "
               "that does not match the corresponding input tangent shape/dtype: "
               f"at output{keystr(kp)} the SymbolicZero had shape/dtype "
               f"{a_.str_short()} while the "
               f"corresponding input had shape/dtype {a.str_short()}. "
               "Consider just returning a None here instead of a SymbolicZero "
               "object.")
        raise ValueError(msg)
      results.append(Zero(ct.aval))
    else:
      if (not config.disable_bwd_checks.value and
          not core.typecompat(a.to_ct_aval(), a_ := core.typeof(ct))
          and not _ref_typecompat(a.to_ct_aval(), a_)
          and not _temporary_dtype_exception(a.to_ct_aval(), a_)):
        primal_info = fun_sourceinfo(primal_fun)
        msg = (f"Custom VJP bwd rule attached to {primal_info} must produce an "
               "output with the same "
               "type as the args tuple of the primal function, but at "
               f"output{keystr(kp)} the bwd rule produced an output of "
               f"type {a_.str_short()} corresponding "
               f"to an input of type {a.str_short()}"
               f"{core.aval_mismatch_extra(a, a_)}")
        raise ValueError(msg)
      results.append(ct)
  return results, logs

def _ref_typecompat(a, a_):
  return (isinstance(a, AbstractRef) and
          core.typecompat(a.to_ct_aval().inner_aval, a_))

# TODO(mattjj): 移除切向量兼容性检查中的这两个例外
def _temporary_dtype_exception(a, a_) -> bool:
  if isinstance(a, core.ShapedArray) and isinstance(a_, core.ShapedArray):
    return (a.shape == a_.shape and
            core.typematch(a, a_, no_dtype_check=True) and
            (dtypes.issubdtype(a_.dtype, dtypes.extended) or
             dtypes.issubdtype(a.dtype, dtypes.np.inexact)))
  return False


class CustomVJPCallPrimitive(core.Primitive):
  multiple_results = True
  skip_canonicalization = True

  def bind_with_trace(self, trace, args, avals, params, /):
    params = dict(params)
    fun, fwd, bwd = params.pop('subfuns')
    return trace.process_custom_vjp_call(self, fun, fwd, bwd, args, **params)

  def impl(self, fun, fwd, bwd, *args):
    raise NotImplementedError

  def is_high(self, *_, call_jaxpr, **__):
    return call_jaxpr.is_high

  def to_lojax(self, *hi_args, call_jaxpr: core.Jaxpr, **params):
    return pe._lower_and_eval(pe.eval_jaxpr_p, call_jaxpr, hi_args)

  def get_bind_params(self, params):
    new_params = dict(params)
    call_jaxpr: core.Jaxpr = new_params.pop('call_jaxpr')
    num_consts: int = new_params.pop('num_consts')
    fwd_jaxpr_thunk = new_params.pop('fwd_jaxpr_thunk')
    fun = lu.wrap_init(core.jaxpr_as_fun(call_jaxpr),
                       debug_info=call_jaxpr.debug_info)
    fwd = lift_fwd(num_consts, fwd_jaxpr_thunk)
    const_avals, _ = split_list(call_jaxpr.in_avals, [num_consts])
    bwd = _handle_consts_in_bwd(new_params.pop('bwd'), const_avals)
    new_params['subfuns'] = (fun, fwd, bwd)
    return new_params

def lift_fwd(num_consts: int, fwd_jaxpr_thunk: lu.WrappedFun) -> lu.WrappedFun:
  def fwd(*args):
    vals, nonzeros = args[::2], args[1::2]
    assert len(vals) == len(nonzeros)
    _, primals = split_list(vals, [num_consts])
    const_nonzeros, in_nonzeros = split_list(nonzeros, [num_consts])
    if any(const_nonzeros): raise ad.CustomVJPException()
    fwd_jaxpr, fwd_consts = fwd_jaxpr_thunk.call_wrapped(*in_nonzeros)
    return core.eval_jaxpr(fwd_jaxpr, fwd_consts, *primals)
  return lu.wrap_init(fwd, debug_info=fwd_jaxpr_thunk.debug_info)

@lu.transformation2
def _handle_consts_in_bwd(f, const_avals, *args):
  cts, logs = f(*args)
  return [Zero(a) for a in const_avals] + list(cts), logs

custom_vjp_call_p = CustomVJPCallPrimitive('custom_vjp_call')
# TODO(phawkins,mattjj): 让这个原语可缓存。
mlir.register_lowering(custom_vjp_call_p, _custom_jvp_vjp_call_lowering,
                       cacheable=False)

def _custom_vjp_call_typecheck(_, *in_avals, call_jaxpr, **kwargs):
  del in_avals, kwargs
  disallowed_effects = effects.custom_derivatives_allowed_effects.filter_not_in(
      call_jaxpr.effects)
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `custom_vjp`: {disallowed_effects}')
  return call_jaxpr.out_avals, core.positional_effects(call_jaxpr)
core.custom_typechecks[custom_vjp_call_p] = _custom_vjp_call_typecheck

def _custom_vjp_call_dce(
    used_outs: Sequence[bool], eqn: core.JaxprEqn
) -> tuple[list[bool], core.JaxprEqn | None]:
  if not any(used_outs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None
  call_jaxpr: core.Jaxpr = eqn.params["call_jaxpr"]
  fwd_jaxpr_thunk = eqn.params["fwd_jaxpr_thunk"]
  bwd: lu.WrappedFun = eqn.params["bwd"]
  out_trees: Callable[[], tuple[PyTreeDef, PyTreeDef, list[int | None]]] = eqn.params["out_trees"]
  symbolic_zeros: bool = eqn.params["symbolic_zeros"]
  dce_call_jaxpr: core.Jaxpr
  used_ins: Sequence[bool]
  dce_call_jaxpr, used_ins = _cached_closed_call_dce_instantiate(
      call_jaxpr, tuple(used_outs))
  assert all(used_ins)

  @partial(lu.wrap_init, debug_info=fwd_jaxpr_thunk.debug_info)
  @pe._memoize
  def dce_fwd_jaxpr_thunk(*zeros):
    fwd_jaxpr_, fwd_consts_ = fwd_jaxpr_thunk.call_wrapped(*zeros)
    fwd_jaxpr = fwd_jaxpr_.with_consts(fwd_consts_)
    _, res_tree, fwds = out_trees()
    num_res_out = res_tree.num_leaves - sum(f is not None for f in fwds)
    dce_fwd_jaxpr, _ = _cached_closed_call_dce_instantiate(
        fwd_jaxpr, (True,) * num_res_out + tuple(used_outs))
    return dce_fwd_jaxpr, dce_fwd_jaxpr.consts

  def dce_bwd(*args):
    _, res_tree, _ = out_trees()
    res, cts = split_list(args, [res_tree.num_leaves])
    cts_ = iter(cts)
    all_cts = []
    for used, aval in zip(used_outs, call_jaxpr.out_avals):
      if used:
        all_cts.append(next(cts_))
      else:
        ct_aval = aval.to_ct_aval()
        if symbolic_zeros:
          all_cts.append(SymbolicZero(ct_aval))
        else:
          all_cts.append(zeros_like_aval(ct_aval))
    assert next(cts_, None) is None
    return bwd.call_wrapped(*res, *all_cts)

  dce_bwd_wrapped = lu.wrap_init(dce_bwd,
                                 debug_info=bwd.debug_info)
  outvars = [v for used, v in zip(used_outs, eqn.outvars) if used]
  new_params = dict(
      eqn.params,
      call_jaxpr=dce_call_jaxpr,
      fwd_jaxpr_thunk=dce_fwd_jaxpr_thunk,
      bwd=dce_bwd_wrapped,
  )
  new_eqn = pe.new_jaxpr_eqn(
      eqn.invars, outvars, eqn.primitive, new_params,
      core.eqn_effects(dce_call_jaxpr, eqn.invars),
      eqn.source_info, eqn.ctx)
  return list(used_ins), new_eqn
pe.dce_rules[custom_vjp_call_p] = _custom_vjp_call_dce


def _custom_vjp_call_pp_rule(eqn: core.JaxprEqn,
                             context: core.JaxprPpContext,
                             settings: core.JaxprPpSettings) -> core.pp.Doc:
  params = dict(eqn.params)
  if not params["num_consts"]:
    params.pop("num_consts")
  params.pop("out_trees")
  params["fwd"] = params.pop("fwd_jaxpr_thunk").debug_info.func_name
  params["bwd"] = params.pop("bwd").debug_info.func_name
  names = sorted(params)
  params["name"] = params["call_jaxpr"].debug_info.func_name
  return core._pp_eqn(eqn.replace(params=params), context, settings,
                      params=["name"] + names)

core.pp_eqn_rules[custom_vjp_call_p] = _custom_vjp_call_pp_rule

batching.primitive_batchers[ad.custom_lin_p] = ad.raise_custom_vjp_error_on_jvp
# TODO(phawkins,mattjj): 让这个原语可缓存。
mlir.register_lowering(ad.custom_lin_p, ad.raise_custom_vjp_error_on_jvp,
                       cacheable=False)


def custom_gradient(fun=None, *, with_logs: bool = False):
  """用于定义自定义 VJP 规则（即自定义梯度）的便捷函数。

  虽然定义自定义 VJP 规则的规范方式是通过 ``jax.custom_vjp``，但
  ``custom_gradient`` 这个便捷包装器遵循 TensorFlow 的
  ``tf.custom_gradient`` API。区别在于，``custom_gradient`` 可以用作
  单个函数的装饰器，该函数同时返回原始值（表示待微分数学函数的输出）
  和 VJP（梯度）函数。参见
  https://www.tensorflow.org/api_docs/python/tf/custom_gradient。

  若待微分的数学函数具有 Haskell 风格的签名 ``a -> b``，那么 Python
  可调用对象 ``fun`` 的签名应为 ``a -> (b, CT b --o CT a)``，其中用
  ``CT x`` 表示 ``x`` 的切向量类型，用 ``--o`` 箭头表示线性函数。参见
  下面的示例。也就是说，``fun`` 应返回一个 pair，其第一个元素表示待
  微分数学函数的值，第二个元素是在反向模式自动微分的反向传播中调用的
  函数（即“自定义梯度”函数）。

  作为 ``fun`` 输出第二个元素返回的函数，可以闭包捕获求值待微分函数时
  计算出的中间值。也就是说，使用词法闭包在反向模式自动微分的前向传播
  与反向传播之间共享计算。然而，它不能执行依赖于被闭包捕获的中间值或
  其切向量参数取值的 Python 控制流；如果该函数包含这类控制流，就会
  抛出错误。

  Args:
    fun: 一个 Python 可调用对象，同时指定待微分的数学函数及其反向模式
      微分规则。它应返回一个 pair，由输出值和一个表示自定义梯度函数的
      Python 可调用对象组成。
    with_logs: 可选 bool，默认 ``False``。若为 ``True``，自定义梯度函数
      必须返回一个 pair ``(in_cts, logs)`` 而不只是切向量；其中 ``logs``
      是一个从名字到 pytree 的字典，用于从反向传播中记录日志，若为
      ``None`` 则不记录任何内容，与
      :py:meth:`jax.custom_vjp.defvjp_with_logs` 一致。

  Returns:
    一个 Python 可调用对象，它接受与 ``fun`` 相同的参数，并返回由
    ``fun`` 输出 pair 的第一个元素所指定的输出值。

  例如：

  >>> @jax.custom_gradient
  ... def f(x):
  ...   return x ** 2, lambda g: (g * x,)
  ...
  >>> print(f(3.))
  9.0
  >>> print(jax.grad(f)(3.))
  3.0

  下面是双参数函数的示例，此时 VJP 函数必须返回长度为二的元组：

  >>> @jax.custom_gradient
  ... def f(x, y):
  ...   return x * y, lambda g: (g * y, g * x)
  ...
  >>> print(f(3., 4.))
  12.0
  >>> print(jax.grad(f, argnums=(0, 1))(3., 4.))
  (Array(4., dtype=float32, weak_type=True), Array(3., dtype=float32, weak_type=True))

  使用 ``with_logs=True`` 时，VJP 函数返回一个 pair，包含切向量和一个
  反向传播日志字典，该字典通过 :py:func:`jax.vjp` 返回的 VJP 函数的
  ``with_logs`` 方法获得：

  >>> @jax.custom_gradient(with_logs=True)
  ... def f(x):
  ...   return x ** 2, lambda g: ((g * 2 * x,), {'ct_out': g})
  ...
  >>> print(jax.grad(f)(3.))
  6.0
  >>> _, f_vjp = jax.vjp(f, 3.)
  >>> (x_ct,), logs = f_vjp.with_logs(1.)
  >>> print(logs['ct_out'])
  1.0
  """
  if fun is None:
    return lambda f: custom_gradient(f, with_logs=with_logs)

  def wrapped_fun(*args, **kwargs):
    ans, _ = fun(*args, **kwargs)
    return ans

  wrapped_fun.__name__ = getattr(fun, '__name__', '<unnamed>')
  wrapped_fun.__qualname__ = getattr(fun, '__qualname__', '<unnamed>')
  wrapped_fun = custom_vjp(wrapped_fun)

  def fwd(*args, **kwargs):
    ans, rule = fun(*args, **kwargs)
    if with_logs:
      rule = _custom_gradient_logs_rule(rule)
    ans_flat, out_tree = tree_flatten(((ans,), {}))
    debug_fwd = debug_info("custom_gradient fwd", rule, (ans,), {})
    ans_avals = [core.typeof(x).to_ct_aval() for x in ans_flat]
    closed_jaxpr, rule_out = pe.trace_to_jaxpr(
        rule, ft.treedef_args_to_ft(out_tree, ans_avals), debug_fwd)
    jaxpr, consts = pe.separate_consts(closed_jaxpr)
    return ans, Residuals(jaxpr, rule_out.tree, out_tree, consts)

  def bwd(res, cts):
    jaxpr, in_tree, out_tree, consts = res
    cts_flat, out_tree_ = tree_flatten(((cts,), {}))
    if out_tree != out_tree_: raise TypeError(f'{out_tree}\n!=\n{out_tree_}')
    cts_out = core.eval_jaxpr(jaxpr, consts, *cts_flat)
    cts_out = tree_unflatten(in_tree, cts_out)
    if with_logs:
      cts_out, logs = cts_out
      cts_tree, _ = treedef_children(in_tree)
      if treedef_is_leaf(cts_tree):
        cts_out = (cts_out,)
      return cts_out, logs
    if treedef_is_leaf(in_tree):
      cts_out = (cts_out,)
    return cts_out

  if with_logs:
    wrapped_fun.defvjp_with_logs(fwd, bwd)
  else:
    wrapped_fun.defvjp(fwd, bwd)
  return wrapped_fun

def _custom_gradient_logs_rule(rule):
  @wraps(rule)
  def rule_with_logs(*cts):
    out = rule(*cts)
    if not (isinstance(out, (list, tuple)) and len(out) == 2):
      raise TypeError(
          "custom_gradient function used with with_logs=True must return a "
          "VJP function producing a pair (in_cts, logs), but the VJP function "
          f"returned {out}.")
    in_cts, logs = out
    if logs is not None and type(logs) is not dict:
      raise TypeError(
          "custom_gradient function used with with_logs=True must return a "
          "VJP function whose second output is None or a dict of "
          f"backward-pass log entries, but got {type(logs).__name__}.")
    return in_cts, logs
  return rule_with_logs

@register_pytree_node_class
class Residuals:
  def __init__(self, jaxpr, in_tree, out_tree, consts):
    self.jaxpr = jaxpr
    self.in_tree = in_tree
    self.out_tree = out_tree
    self.consts = consts
  def __iter__(self):
    return iter((self.jaxpr, self.in_tree, self.out_tree, self.consts))
  def tree_flatten(self):
    return self.consts, (self.jaxpr, self.in_tree, self.out_tree)
  @classmethod
  def tree_unflatten(cls, aux, consts):
    jaxpr, in_tree, out_tree = aux
    return cls(jaxpr, in_tree, out_tree, consts)


def closure_convert(fun: Callable, *example_args) -> tuple[Callable, list[Any]]:
  """闭包转换工具，用于高阶自定义导数。

  要用 ``jax.custom_vjp(f)`` 这类方式定义自定义导数，目标函数 ``f`` 必须
  把所有参与微分的值都作为形式参数接收。如果 ``f`` 是高阶函数，即它接受
  一个 Python 函数 ``g`` 作为参数，那么存储在 ``g`` 闭包中的值对自定义
  导数规则不可见，涉及这些值的 AD 尝试将会失败。绕过这一点的一种办法是
  做闭包转换，把这些值提取出来，并作为显式形式参数跨越自定义导数边界传递。
  本工具执行该转换。更准确地说，它把特化到 ``example_args`` 中所给参数
  类型的函数 ``fun`` 做闭包转换。

  这里所说的 ``fun`` “闭包中的值”，并不是指定义 ``fun`` 时 Python 直接
  捕获的值（例如 ``fun.__closure__`` 中的 Python 对象，若该属性存在）。
  我们指的是在 ``example_args`` 上执行 ``fun`` 期间遇到、并决定其输出的
  值。例如，这可能包括在 Python 闭包中被传递性捕获的数组，即在 ``fun``
  所调用函数的 Python 闭包、这些函数所调用函数的闭包等之中捕获的数组。

  函数 ``fun`` 必须是纯函数。

  用法示例::

    def minimize(objective_fn, x0):
      converted_fn, aux_args = closure_convert(objective_fn, x0)
      return _minimize(converted_fn, x0, *aux_args)

    @partial(custom_vjp, nondiff_argnums=(0,))
    def _minimize(objective_fn, x0, *args):
      z = objective_fn(x0, *args)
      # ... 求最小化点 x_opt ...
      return x_opt

    def fwd(objective_fn, x0, *args):
      y = _minimize(objective_fn, x0, *args)
      return y, (y, args)

    def rev(objective_fn, res, g):
      y, args = res
      y_bar = g
      # ... 自定义反向模式 AD ...
      return x0_bar, *args_bars

    _minimize.defvjp(fwd, rev)

  Args:
    fun: 要转换的 Python 可调用对象。必须是纯函数。
    example_args: 数组、标量或其（嵌套的）标准 Python 容器
      （元组、列表、字典、namedtuple，即 pytree），用于确定 ``fun``
      各形式参数的类型。``fun`` 按类型特化后的这种形式，就是要被
      闭包转换的函数。

  Returns:
    一个 pair，由 (i) 一个 Python 可调用对象（它接受与 ``fun`` 相同的
    参数，其后跟与从闭包中提升出来的值对应的参数）和 (ii) 一个从闭包
    中提升出来的值组成的列表构成。
  """
  flat_args, in_tree = tree_flatten((example_args, {}))
  in_avals = tuple(map(core.typeof, flat_args))
  debug = debug_info("closure_convert", fun, example_args, {})
  if config.check_tracer_leaks.value:
    return _closure_convert_for_avals.__wrapped__(fun, in_tree, in_avals, debug)
  else:
    return _closure_convert_for_avals(fun, in_tree, in_avals, debug)

def _maybe_perturbed(x: Any) -> bool:
  # 若 x 无法表示被 AD 扰动过的值（即带有非平凡切向量的值），
  # 按启发式判断返回 False，否则返回 True。
  # 动机参见 https://github.com/jax-ml/jax/issues/6415。
  if not isinstance(x, core.Tracer):
    # 若 x 不是 Tracer，它就不可能被扰动。
    return False
  elif isinstance(x, ad.JVPTracer) and isinstance(x.tangent, ad.Zero):
    return _maybe_perturbed(x.primal)
  elif isinstance(x, pe.DynamicJaxprTracer):
    # 若 x 是 DynamicJaxprTracer，说明我们正在暂存输出；微分可能稍后
    # 才发生，但某些类型的切向量总是平凡的。
    vspace = x.aval.to_tangent_aval()
    return not (vspace is core.abstract_token or
                getattr(vspace, 'dtype', None) == dtypes.float0)
  elif not isinstance(x, ad.JVPTracer):
    # 若 x 不是 JVPTracer，则递归检查其内容。
    return any(_maybe_perturbed(attr) for name, attr in x._contents())
  else:
    return True  # 我们无法确定！

@cache()
def _closure_convert_for_avals(fun, in_tree, in_avals,
                               debug_info: core.DebugInfo):
  closed_jaxpr, out_avals = pe.trace_to_jaxpr(
      fun, ft.treedef_args_to_ft(in_tree, in_avals), debug_info)
  jaxpr, consts = pe.separate_consts(closed_jaxpr)
  out_tree = out_avals.tree

  (closure_consts, const_args), merge = partition_list(_maybe_perturbed, consts)
  num_consts = len(const_args)

  def converted_fun(*args_hconsts):
    num_args = len(args_hconsts) - num_consts
    args, const_args = split_list(args_hconsts, [num_args])
    consts = merge(closure_consts, const_args)
    all_args, in_tree2 = tree_flatten((tuple(args), {}))
    if in_tree != in_tree2:
      msg = ("The inputs to the closure produced by closure_convert must have "
             "the same Pytree structure as the example arguments passed when "
             f"closure_convert was called. Expected {in_tree}, but got "
             f"{in_tree2}")
      raise TypeError(msg)
    out_flat = core.eval_jaxpr(jaxpr, consts, *all_args)
    return tree_unflatten(out_tree, out_flat)

  return converted_fun, const_args

def partition_list(choice, lst):
  out = [], []
  which = [out[choice(elt)].append(elt) or choice(elt) for elt in lst]
  def merge(l1, l2):
    i1, i2 = iter(l1), iter(l2)
    return [next(i2 if snd else i1) for snd in which]
  return out, merge


### 自定义转置

def linear_call(fun: Callable,
                fun_transpose: Callable, residual_args,
                linear_args):
  """调用一个线性函数，并为其转置提供自定义实现。

  ``fun`` 和 ``fun_transpose`` 的 `Haskell-like type signatures`_ 为：

  .. code-block:: haskell

    fun           :: r -> a -o b
    fun_transpose :: r -> b -o a

  其中 ``-o`` 箭头表示线性函数，``r`` 是残差输入类型，``a`` 是线性输入类型。

  ``fun`` 和 ``fun_transpose`` 彼此互为转置。具体来说，
  ``linear_call`` 原语的转置是另一个针对 ``fun_transpose`` 的
  ``linear_call``，并把 ``fun`` 作为其自定义转置。

  例如：

  >>> def f(r, x):
  ...   return x / r

  >>> def t(r, t):
  ...   return t / r

  >>> def div_add(x, denom):
  ...   return x + linear_call(f, t, denom, x)

  >>> def transpose(f, x_example):
  ...   def transposed(y):
  ...     x, = jax.linear_transpose(f, x_example)(y)
  ...     return x
  ...   return transposed

  >>> div_add(9., 3.)
  Array(12., dtype=float32, weak_type=True)

  >>> transpose(partial(div_add, denom=3.), 1.)(18.)  # custom
  Array(24., dtype=float32, weak_type=True)

  >>> transpose(lambda x: x + x / 3., 1.)(18.)  # reference
  Array(24., dtype=float32, weak_type=True)

  上面 ``f`` 的定义说明了残差参数的用途：除法对其中一个输入（被除数
  ``x``）是线性的，但对另一个输入（除数 ``r``）不是。

  再举一个例子：

  >>> def custom_id(x):
  ...   def f(_, x): return x
  ...   def t(_, t): return 7.
  ...   return linear_call(f, t, (), x)
  >>> custom_id(1.)
  TypedFloat(1.0, dtype=float32)
  >>> transpose(custom_id, 1.)(1.)
  TypedFloat(7.0, dtype=float32)
  >>> transpose(transpose(custom_id, 1.), 1.)(1.)
  TypedFloat(1.0, dtype=float32)
  >>> transpose(transpose(transpose(custom_id, 1.), 1.), 1.)(1.)
  TypedFloat(7.0, dtype=float32)

  Args:
    fun: 一个 Python 可调用对象，指定一个线性函数。它应接受两个参数：
      一个是“残差”输入（类型 ``r``），即函数对其不一定线性的输入；
      另一个是“线性”输入（类型 ``a``）。它应返回各分量关于线性输入
      为线性的输出（类型 ``b``）。
    fun_transpose: 一个 Python 可调用对象，指定一个结构上线性的函数，
      它是 ``fun`` 关于其线性输入的转置。它的第一个参数是与 ``fun``
      相同的残差输入（``r``）。它的第二个参数类型为 ``b``。最后，
      它的输出类型为 ``a``，且其每个分量关于其第二个参数（``b`` 输入）
      是线性的。
    residual_args: ``fun`` 和 ``fun_transpose`` 对其不一定线性的参数。
      不参与转置。
    linear_args: ``fun`` 和 ``fun_transpose`` 对其均为线性、且两者
      关于它互为转置的参数。

  Returns:
    调用结果，即 ``fun(residual_args, linear_args)``。

  .. _Haskell-like type signatures: https://wiki.haskell.org/Type_signature
  """
  operands_res, res_tree = tree_flatten(residual_args)
  operands_lin, lin_tree = tree_flatten(linear_args)

  res_avals = map(core.typeof, operands_res)
  lin_avals = map(core.typeof, operands_lin)
  f_jaxpr, f_out_avals = pe.trace_to_jaxpr(
      fun,
      ft.pack(((ft.FTPyTree(res_avals, res_tree),
                ft.FTPyTree(lin_avals, lin_tree)), {})),
      debug_info("linear_call fun", fun, (residual_args, linear_args), {}))
  f_jaxpr_closed, f_consts = pe.separate_consts(f_jaxpr)
  out_avals = f_jaxpr_closed.out_avals
  out_tree = f_out_avals.tree

  @pe._memoize
  def transpose_thunk():
    t_jaxpr, t_out_avals = pe.trace_to_jaxpr(
        fun_transpose,
        ft.pack(((ft.FTPyTree(res_avals, res_tree),
                  ft.FTPyTree(list(out_avals), out_tree)), {})),
        # TODO(necula): fun_transpose 接收的是 fun 的残差和输出！
        debug_info("linear_call fun_transpose", fun_transpose,
                   (residual_args, linear_args), {}).with_unknown_names())
    if t_out_avals.tree != lin_tree:
      raise TypeError(
          'transpose output pytree structure must match that of linear inputs, '
          f'got output structure {t_out_avals.tree} '
          f'and input structure {lin_tree}.')
    return pe.separate_consts(t_jaxpr)

  out = linear_call_p.bind(*f_consts, *operands_res, *operands_lin,
                           callee=f_jaxpr_closed,
                           transpose_thunk=transpose_thunk,
                           num_callee_consts=len(f_consts),
                           num_res=len(operands_res))

  return tree_unflatten(out_tree, out)
def _linear_call_impl(*args, callee, transpose_thunk, num_callee_consts,
                      num_res):
  del transpose_thunk, num_callee_consts, num_res
  return core.eval_jaxpr(callee, (), *args)

def _linear_call_jvp_rule(primals, tangents, callee, transpose_thunk,
                          num_callee_consts, num_res):
  consts_and_res, primals = split_list(primals, [num_callee_consts + num_res])
  const_tangents, tangents = split_list(tangents, [num_callee_consts + num_res])
  assert all(type(t) is Zero for t in const_tangents)
  primals_out = linear_call_p.bind(
      *consts_and_res, *primals, callee=callee, transpose_thunk=transpose_thunk,
      num_callee_consts=num_callee_consts, num_res=num_res)
  tangents_out = linear_call_p.bind(
      *consts_and_res, *tangents, callee=callee, transpose_thunk=transpose_thunk,
      num_callee_consts=num_callee_consts, num_res=num_res)
  return primals_out, tangents_out

def _linear_call_transpose_rule(cts, *args, callee, transpose_thunk,
                                num_callee_consts, num_res):
  transpose, t_consts = transpose_thunk()
  f_consts, operands_res, operands_lin = split_list(
      args, [num_callee_consts, num_res])
  _, _, cts_avals = split_list(
      transpose.in_avals, [len(t_consts), num_res])

  assert all(ad.is_undefined_primal(x)     for x in operands_lin)
  assert all(not ad.is_undefined_primal(x) for x in operands_res)

  def new_transpose_thunk():
    return callee, f_consts

  cts = [zeros_like_aval(a) if type(ct) is Zero else ct
         for ct, a in zip(cts, cts_avals)]
  cts_out = linear_call_p.bind(*t_consts, *operands_res, *cts,
                               callee=transpose,
                               transpose_thunk=new_transpose_thunk,
                               num_callee_consts=len(t_consts),
                               num_res=len(operands_res))

  return [None] * (num_callee_consts + num_res) + cts_out

def _linear_call_abstract_eval(*args, **kwargs):
  return kwargs['callee'].out_avals

linear_call_p = core.Primitive('linear_call')
linear_call_p.multiple_results = True
linear_call_p.def_impl(_linear_call_impl)
linear_call_p.def_abstract_eval(_linear_call_abstract_eval)
ad.primitive_jvps[linear_call_p] = _linear_call_jvp_rule
ad.primitive_transposes[linear_call_p] = _linear_call_transpose_rule
mlir.register_lowering(linear_call_p, mlir.lower_fun(
    _linear_call_impl, multiple_results=True))


# 一个可暂存的原语，在求值时失败
unreachable_p: core.Primitive = core.Primitive('unreachable')
unreachable_p.multiple_results = True

def unreachable_impl(*_, out_avals, exc_type, message):
  del out_avals
  raise exc_type(message)

# 求值会抛出异常
unreachable_p.def_impl(unreachable_impl)

# 转换（lowering）会抛出异常
# TODO(frostig,mattjj): 对于一个会出错的函数，我们还没有好的转换办法。
# 由于 MLIR 降级是对具体求值的过近似，我们暂时选择在 MLIR 降级阶段报错。
mlir.register_lowering(unreachable_p, unreachable_impl)

# 抽象求值可以正常进行，以便支持暂存
unreachable_p.def_abstract_eval(lambda *_, out_avals, **__: out_avals)

def unreachable(*args, out_avals=None, exc_type=TypeError,
                message='unreachable'):
  """在具体求值时失败（但允许暂存）。

  该函数允许断言某种求值不可能发生。可以用它来保证求值不会
  “到达”某个点：也就是说它不会被执行，但 JAX 仍然可以
  在不报错的情况下把它暂存出去。

  Args:
    *args: 传给该函数的任意 pytree 参数。
    out_avals: 可选参数，从暂存的角度说明这次函数调用的
     输出类型。若为 ``None``，则这些类型取为与输入
     参数类型相同。
    exc_type: 可选参数，为求值时抛出的 Python 异常提供
      构造函数。
    message: 可选参数，为求值时抛出的 Python 异常提供
      字符串消息。

  """
  if out_avals is None:
    out_avals = tree_map(core.typeof, args)

  args_flat, in_tree = tree_flatten(args)
  out_avals_flat, out_tree = tree_flatten(out_avals)
  out = unreachable_p.bind(*args_flat, out_avals=out_avals_flat,
                           exc_type=exc_type, message=message)
  return tree_unflatten(out_tree, out)


disallow_jvp = partial(
    unreachable,
    exc_type=TypeError,
    message="can't apply forward-mode autodiff (jvp) to a custom_vjp function.")


# TODO(mattjj): 删除这些桩（stub），它们的存在是为了避免破坏内部使用者
custom_jvp_call_jaxpr_p = core.Primitive("custom_jvp_call_jaxpr")

# 下面是一个辅助函数，用于优化 custom_vjp 在 remat 之下使用时的
# 行为。它真正有用的场景，是 custom_vjp 的 `fwd` 函数执行一个黑盒
# kernel 的时候；否则，DCE 会自动完成这项优化。
#
# TODO(dfm): 如果能让它在大多数情况下都是无操作，最终这大概应该成为
# custom_vjp 的默认行为。目前它以 "initial-style" 方式编写，因此不支持
# 即时（eager）模式。当初这么写时，由于能让实现更简单，这是一个合理
# 的折中，但值得重新审视。
def optimize_remat_of_custom_vjp_fwd[ReturnValue](
    fun: Callable[..., ReturnValue],
    debug_fun: core.DebugInfo,
    fwd: Callable[..., tuple[ReturnValue, Any]],
    debug_fwd: core.DebugInfo,
    nondiff_argnums: Sequence[int] = (),
    symbolic_zeros: bool = False,
) -> Callable[..., tuple[ReturnValue, Any]]:
  if symbolic_zeros:
    # TODO(dfm): 支持它大概不会太难。
    raise NotImplementedError(
        "remat optimization for custom_vjp does not support symbolic zeros")

  @wraps(fwd)
  def wrapped_fwd(*args, **kwargs) -> tuple[ReturnValue, Any]:
    # TODO(dfm): 这里开头的逻辑与上面 custom_vjp.__call__ 中的
    # 逻辑重复，最好把它们合并起来。
    # 注意：这里使用 `fun` 而不是 `fwd`，是为了与上面的
    # custom_vjp.__call__ 保持一致。
    args = resolve_kwargs(fun, args, kwargs)
    if nondiff_argnums:
      for i in nondiff_argnums: _check_for_tracers(args[i])
      nondiff_argnums_ = set(nondiff_argnums)
      dyn_argnums = [i for i in range(len(args)) if i not in nondiff_argnums_]
      f_, dyn_args = argnums_partial(lu.wrap_init(fun, debug_info=debug_fun),
                                     dyn_argnums,
                                     args, require_static_args_hashable=False)
      fwd_, _ = argnums_partial(lu.wrap_init(fwd, debug_info=debug_fwd),
                                dyn_argnums, args,
                                require_static_args_hashable=False)
    else:
      f_, dyn_args = lu.wrap_init(fun, debug_info=debug_fun), args
      fwd_ = lu.wrap_init(fwd, debug_info=debug_fwd)
    args_flat, in_tree = tree_flatten(dyn_args)
    flat_fun, out_type = _flatten_fun_nokwargs(f_, in_tree)
    flat_fwd, out_trees = _flatten_fwd(fwd_, nondiff_argnums, False,
                                       debug_fun, debug_fwd, in_tree, out_type)
    flat_fwd = _fix_fwd_args(flat_fwd)

    in_avals = [core.typeof(x) for x in args_flat]
    fwd_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fwd.with_unknown_names(),
                                                     in_avals)
    fwd_jaxpr = pe.convert_constvars_jaxpr(fwd_jaxpr)
    prim_tree, res_tree, fwds = out_trees()
    num_res_out = res_tree.num_leaves - sum(f is not None for f in fwds)

    disallowed_effects = effects.custom_derivatives_allowed_effects.filter_not_in(fwd_jaxpr.effects)
    if disallowed_effects:
      raise NotImplementedError(
          "remat optimization for custom_vjp does not support forward "
          f"functions with these side effects: {disallowed_effects}")

    @pe._memoize
    def fun_jaxpr_thunk():
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
      return jaxpr, consts

    out_flat = remat_opt_p.bind(*consts, *args_flat, num_consts=len(consts),
                                num_res=num_res_out, fwd_jaxpr=fwd_jaxpr,
                                fun_jaxpr_thunk=fun_jaxpr_thunk)
    res, out_flat = split_list(out_flat, [num_res_out])
    res_ = iter(res)
    res = [next(res_) if f is None else args_flat[f] for f in fwds]
    assert next(res_, None) is None
    out_tree = treedef_tuple((prim_tree, res_tree))
    return tree_unflatten(out_tree, (*out_flat, *res))

  return wrapped_fwd

@lu.transformation2
def _fix_fwd_args(f, *args):
  args = [(x, True) for x in args]
  args = [x for pair in args for x in pair]
  return f(*args)

def _remat_opt_impl(
    *args,
    num_consts: int,
    num_res: int,
    fwd_jaxpr: core.Jaxpr,
    fun_jaxpr_thunk: Callable[[], tuple[core.Jaxpr, Sequence[Any]]],
):
  del num_consts, num_res, fun_jaxpr_thunk  # 未使用
  return core.jaxpr_as_fun(fwd_jaxpr)(*args)

def _remat_opt_abstract_eval(*args, fwd_jaxpr: core.Jaxpr, **_):
  del args
  return fwd_jaxpr.out_avals, core.positional_effects(fwd_jaxpr)

def _remat_opt_vmap(
    axis_data, args, in_dims,
    *,
    num_consts: int,
    num_res: int,
    fwd_jaxpr: core.Jaxpr,
    fun_jaxpr_thunk: Callable[[], tuple[core.Jaxpr, Sequence[Any]]],
):
  args = [batching.moveaxis(x, d, 0) if d is not None and d != 0
          else x for x, d in zip(args, in_dims)]
  in_batched = [d is not None for d in in_dims]
  batched_fwd_jaxpr, out_batched = batching.batch_jaxpr(
      fwd_jaxpr, axis_data, in_batched, False)
  extra_consts = batched_fwd_jaxpr.consts
  batched_fwd_jaxpr = pe.convert_constvars_jaxpr(batched_fwd_jaxpr)
  out_dims = [0 if b else None for b in out_batched]

  _, prim_batched = split_list(in_batched, [num_consts])

  @pe._memoize
  def batched_fun_jaxpr_thunk():
    fun_jaxpr_, fun_consts_ = fun_jaxpr_thunk()
    fun_jaxpr = fun_jaxpr_.with_consts(fun_consts_)
    batched_fun_jaxpr, out_batched = batching.batch_jaxpr(
        fun_jaxpr, axis_data, prim_batched, False)
    return batched_fun_jaxpr, batched_fun_jaxpr.consts

  batched_outs = remat_opt_p.bind(*extra_consts, *args,
                                  num_consts=num_consts + len(extra_consts),
                                  num_res=num_res,
                                  fwd_jaxpr=batched_fwd_jaxpr,
                                  fun_jaxpr_thunk=batched_fun_jaxpr_thunk)

  return batched_outs, out_dims

def _remat_opt_jvp(
    primals,
    tangents,
    *,
    num_consts: int,
    num_res: int,
    fwd_jaxpr: core.Jaxpr,
    fun_jaxpr_thunk: Callable[[], tuple[core.Jaxpr, Sequence[Any]]],
):
  consts, primals = split_list(primals, [num_consts])
  consts_dot, tangents = split_list(tangents, [num_consts])
  # 切向量必须被实例化，以防之后被死代码消除（DCE）。
  tangents = map(ad.instantiate_zeros, tangents)
  consts_nz = [not isinstance(t, Zero) for t in consts_dot]
  consts_dot = [c for nz, c in zip(consts_nz, consts_dot) if nz]
  in_nz = consts_nz + [True] * len(tangents)
  fwd_jaxpr_jvp_, out_nz = ad.jvp_jaxpr(fwd_jaxpr, in_nz, True)
  num_out = len(out_nz) - num_res
  fwd_jaxpr_jvp_ = ad.rearrange_binders(
      fwd_jaxpr_jvp_, [num_consts, len(primals)],
      [len(consts_dot), len(tangents)], [num_res, num_out], [num_res, num_out])
  fwd_jaxpr_jvp = pe.convert_constvars_jaxpr(fwd_jaxpr_jvp_)

  # @pe._memoize
  def fun_jvp_jaxpr_thunk():
    fun_jaxpr_, fun_consts_ = fun_jaxpr_thunk()
    fun_jaxpr = fun_jaxpr_.with_consts(fun_consts_)
    in_nz = [True] * len(primals)
    fun_jvp_jaxpr, _ = ad.jvp_jaxpr(fun_jaxpr, in_nz, True)
    return fun_jvp_jaxpr, fun_jvp_jaxpr.consts

  new_num_consts = len(fwd_jaxpr_jvp_.consts) + num_consts + len(consts_dot)
  outs = remat_opt_p.bind(*fwd_jaxpr_jvp_.consts, *consts, *consts_dot,
                          *primals, *tangents, num_consts=new_num_consts,
                          num_res=2 * num_res, fwd_jaxpr=fwd_jaxpr_jvp,
                          fun_jaxpr_thunk=fun_jvp_jaxpr_thunk)
  res, res_dot, outs, outs_dot = split_list(outs, [num_res, num_res, num_out])
  return (*res, *outs), (*res_dot, *outs_dot)

def _remat_opt_transpose(
    cts, *args,
    num_consts: int,
    num_res: int,
    fwd_jaxpr: core.Jaxpr,
    fun_jaxpr_thunk: Callable[[], tuple[core.Jaxpr, Sequence[Any]]],
):
  # TODO(dfm): 将来如有需要，实现它应该不会太难。
  raise NotImplementedError(
      "remat optimization for custom_vjp does not support higher-order AD")

def _remat_opt_dce(used_outs: list[bool], eqn: core.JaxprEqn):
  if not any(used_outs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None
  used_res, used_prims = split_list(used_outs, [eqn.params["num_res"]])
  outvars = [v for used, v in zip(used_outs, eqn.outvars) if used]
  if any(used_res):
    # 如果任何一个残差被使用，此时我们仍然需要运行 fwd，但之后
    # 可能还会再次进行死代码消除，因此必须实例化所有的输入原始值。
    instantiate = [False] * eqn.params["num_consts"]
    instantiate += [True] * (len(eqn.invars) - eqn.params["num_consts"])
    new_jaxpr, used_ins = pe.dce_jaxpr(eqn.params["fwd_jaxpr"], used_outs,
                                       instantiate=instantiate)
    assert not new_jaxpr.constvars
    closed_jaxpr = new_jaxpr
    invars = [v for used, v in zip(used_ins, eqn.invars) if used]
    new_params = dict(eqn.params)
    new_num_consts = sum(split_list(used_ins, [eqn.params["num_consts"]])[0])
    new_params["num_consts"] = new_num_consts
    new_params["fwd_jaxpr"] = closed_jaxpr
    new_params["num_res"] = sum(used_res)
    new_eqn = pe.new_jaxpr_eqn(
        invars, outvars, remat_opt_p, new_params,
        core.eqn_effects(closed_jaxpr, invars),
        eqn.source_info, eqn.ctx)
    return used_ins, new_eqn
  else:
    # 如果没有任何残差被使用，我们就改为运行原始值计算。此时我们放弃
    # 这一自定义 DCE 行为；但由于原始值计算可能与 fwd 拥有不同的常量，
    # 因此我们用一个 `closed_call` 原语构造新的 `JaxprEqn`。
    fun_jaxpr, consts = eqn.params["fun_jaxpr_thunk"]()
    closed_jaxpr, _, used_ins = pe.dce_jaxpr_consts(
        fun_jaxpr.with_consts(consts), used_prims)
    _, invars = split_list(eqn.invars, [eqn.params["num_consts"]])
    invars = [v for used, v in zip(used_ins, invars) if used]
    new_eqn = pe.new_jaxpr_eqn(
        invars, outvars, core.eval_jaxpr_p, dict(call_jaxpr=closed_jaxpr),
        core.eqn_effects(closed_jaxpr, invars), eqn.source_info, eqn.ctx)
    used_ins = [False] * eqn.params["num_consts"] + used_ins
    return used_ins, new_eqn

def _remat_opt_to_lojax(*hi_args, fwd_jaxpr: core.Jaxpr, num_consts, **params):
  return pe._lower_and_eval(pe.eval_jaxpr_p, fwd_jaxpr, hi_args)

remat_opt_p = core.Primitive("remat_opt")
remat_opt_p.multiple_results = True
remat_opt_p.is_high = lambda *_, fwd_jaxpr, **__: fwd_jaxpr.is_high
remat_opt_p.to_lojax = _remat_opt_to_lojax
remat_opt_p.def_impl(_remat_opt_impl)
remat_opt_p.def_effectful_abstract_eval(_remat_opt_abstract_eval)
mlir.register_lowering(remat_opt_p, mlir.lower_fun(
    _remat_opt_impl, multiple_results=True))


batching.fancy_primitive_batchers[remat_opt_p] = _remat_opt_vmap
ad.primitive_jvps[remat_opt_p] = _remat_opt_jvp
ad.primitive_transposes[remat_opt_p] = _remat_opt_transpose
pe.dce_rules[remat_opt_p] = _remat_opt_dce
