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

from collections import namedtuple
from collections.abc import Callable, Sequence
import contextlib
from dataclasses import dataclass
from functools import partial
import itertools as it
import logging
import operator as op
from typing import Any, NamedTuple
from weakref import ReferenceType, WeakValueDictionary, finalize, ref

import numpy as np

from jax._src import ad_util
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import profiler
from jax._src import source_info_util
from jax._src import tree_util
from jax._src.core import (
    Trace, Tracer, TraceTag, Jaxpr, Literal, AbstractValue,
    new_jaxpr_eqn, Var, DropVar, Atom, JaxprEqn, Primitive,
    get_referent, JaxprEqnContext, typeof)
from jax._src.lib import _jax
from jax._src.source_info_util import SourceInfo
from jax._src.state.types import AbstractRef, ReadEffect
from jax._src import flattree as ft
from jax._src.tree_util import PyTreeDef
from jax._src.util import (unzip2, safe_zip, safe_map, toposort, split_list,
                           merge_lists, partition_list, OrderedSet,
                           weakref_lru_cache, multi_weakref_lru_cache,
                           foreach, test_event)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip
def identity(x): return x

TracerId = int
AvalId = int
ConstId = int

AttrKind = Any
PyTree = Any
logger = logging.getLogger(__name__)

TracebackScope = _jax.TracebackScope

class PartialVal(tuple):
  """部分值：要么是已知值，要么是未知（抽象）值。

  表示为 `(aval_opt, const)` 二元组，属于以下两种之一：
  * `(None, <Constant>)` 表示已知值，其中该常量满足
    `core.valid_jaxtype(const)`；
  * `(<AbstractValue>, None)` 表示未知值，其特征由
    一个抽象值刻画。
  """
  def __new__(cls, xs: tuple[AbstractValue | None, core.Value]):
    pv, const = xs
    if config.enable_checks.value:
      # 类型检查
      assert isinstance(pv, (AbstractValue, type(None))), xs
      assert (const is None or core.valid_jaxtype(const)), const
      # 不变量检查
      assert (pv is None) ^ (const is None)
    return tuple.__new__(cls, xs)

  @classmethod
  def known(cls, const: core.Value) -> PartialVal:
    return PartialVal((None, const))

  @classmethod
  def unknown(cls, aval: AbstractValue) -> PartialVal:
    return PartialVal((aval, None))

  def is_known(self) -> bool:
    return self[0] is None

  def get_known(self) -> core.Value | None:
    """获取已知值，若已知则返回值，否则返回 None。"""
    return self[1] if self[0] is None else None

  def get_aval(self) -> AbstractValue:
    """直接获取 AbstractValue（若未知），或从常量获取（若已知）。"""
    known = self.get_known()
    if known is not None:
      return typeof(known)
    else:
      return self[0]

@dataclass(frozen=True, slots=True)
class EffectHandle:
  parents : list[Tracer]
  recipe : JaxprEqnRecipe

class JaxprTrace(Trace):

  def __init__(self, parent_trace:Trace, name_stack: source_info_util.NameStack, tag:TraceTag):
    super().__init__()
    self.name_stack = name_stack
    self.tag = tag
    self.parent_trace = parent_trace
    self.requires_low = False
    self.effect_handles : list[EffectHandle] = []
    self.counter = it.count()

  def to_jaxpr_tracer(self, x):
    if isinstance(x, JaxprTracer) and x._trace.tag is self.tag:
      if x._trace is self:
        return x
      else:
        return JaxprTracer(self, x.pval, FreeVar(x))
    else:
      return self.new_const(x)

  def stage_value(self, val):
    return self.to_jaxpr_tracer(val)

  def new_const(self, val) -> JaxprTracer:
    return JaxprTracer(self, PartialVal.known(val), None)

  def new_instantiated_literal(self, val) -> JaxprTracer:
    aval = typeof(val)
    return JaxprTracer(self, PartialVal.unknown(aval), Literal(val, aval))

  def new_instantiated_const(self, val) -> JaxprTracer:
    aval = typeof(val)
    return JaxprTracer(self, PartialVal.unknown(aval), ConstVar(val))

  def new_arg(self, pval: PartialVal) -> JaxprTracer:
    const = pval.get_known()
    # XXX: 在修改这个常量参数剪枝之前请三思！
    # 这对 partial_eval_jaxpr 有着极其重要的影响。
    # 最重要的是，这保证了未知 jaxpr 绝不会使用
    # 已知输入（如果它需要这些输入，它们会作为残差传递过去）。
    if const is None:
      aval = pval.get_aval()
      return JaxprTracer(self, PartialVal.unknown(aval), LambdaBinding())
    else:
      return self.new_const(const)

  def instantiate_const(self, tracer: JaxprTracer) -> JaxprTracer:
    const = tracer.pval.get_known()
    if const is None:
      return tracer
    else:
      if core.is_literalable(const, True):
        return self.new_instantiated_literal(const)
      else:
        return self.new_instantiated_const(const)

  def process_primitive(self, primitive, tracers, params, /):
    with core.set_current_trace(self.parent_trace):
      if primitive in custom_partial_eval_rules:
        tracers = map(self.to_jaxpr_tracer, tracers)
        return custom_partial_eval_rules[primitive](self, *tracers, **params)
      else:
        return self.default_process_primitive(primitive, tracers, params)

  def default_process_primitive(self, primitive, tracers, params):
    # 默认情况下，如果所有输入追踪器都是已知的，就绑定该原语
    # 并认为所有输出都是已知的。否则，把该应用暂存到
    # jaxpr 中，并认为所有输出都是未知的。
    tracers = map(self.to_jaxpr_tracer, tracers)
    consts = [t.pval.get_known() for t in tracers]
    if all(c is not None for c in consts):
      return primitive.bind_with_trace(self.parent_trace, consts,
                                       tuple(t.aval for t in tracers), params)
    tracers = map(self.instantiate_const, tracers)
    avals = [t.aval for t in tracers]
    out_aval, effs = primitive.abstract_eval(*avals, **params)
    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    if primitive.multiple_results:
      out_tracers = [JaxprTracer(self, PartialVal.unknown(aval), None)
                     for aval in out_aval]
      eqn = new_eqn_recipe(self, tracers, out_tracers, primitive, params, effs,
                           source)
      if effects.partial_eval_kept_effects.filter_in(effs):
        self.effect_handles.append(EffectHandle(tracers, eqn))  # pyrefly: ignore[bad-argument-type]  # pyrefly#2385
      for t in out_tracers: t.recipe = eqn
      return out_tracers
    else:
      out_tracer = JaxprTracer(self, PartialVal.unknown(out_aval), None)
      eqn = new_eqn_recipe(self, tracers, [out_tracer], primitive,
                           params, effs, source)
      if effects.partial_eval_kept_effects.filter_in(effs):
        self.effect_handles.append(EffectHandle(tracers, eqn))  # pyrefly: ignore[bad-argument-type]  # pyrefly#2385
      out_tracer.recipe = eqn
      return out_tracer

  def _current_truncated_name_stack(self):
    return source_info_util.current_name_stack()[len(self.name_stack):]

  def process_custom_jvp_call(self, prim, fun, jvp, tracers, /, *, symbolic_zeros):
    tracers = map(self.to_jaxpr_tracer, tracers)
    if all(t.is_known() for t in tracers):
      with core.set_current_trace(self.parent_trace):
        vals = [t.pval[1] for t in tracers]
        return prim.bind(*vals, subfuns=(fun, jvp), symbolic_zeros=symbolic_zeros)
    # 我们假定非平凡的部分求值只是为了构建线性
    # 函数，因此不需要保留自定义 JVP 规则。
    del jvp, symbolic_zeros
    with core.set_current_trace(self):
      return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, prim, f, fwd, bwd, tracers, /, *, out_trees, symbolic_zeros):
    tracers = map(self.to_jaxpr_tracer, tracers)
    if all(t.is_known() for t in tracers):
      vals = [t.pval[1] for t in tracers]
      with core.set_current_trace(self.parent_trace):
        return prim.bind(*vals, subfuns=(f, fwd, bwd), out_trees=out_trees,
                         symbolic_zeros=symbolic_zeros)

    tracers = map(self.instantiate_const, tracers)
    in_knowns = (False,) * len(tracers)
    in_avals = tuple(t.aval for t in tracers)
    f_ = trace_to_subjaxpr_nounits2(f, self.tag, f.debug_info, True)
    f_, aux = partial_eval_wrapper_nounits(f_, in_knowns, in_avals)
    params = dict(subfuns=(f_, fwd, bwd), out_trees=out_trees,
                  symbolic_zeros=symbolic_zeros)
    res = prim.bind_with_trace(self.parent_trace, (), (), params)
    out_knowns, out_avals, jaxpr, env = aux()
    assert not any(out_knowns)
    res_tracers = map(self.instantiate_const, map(self.new_const, res))
    env_tracers = map(self.to_jaxpr_tracer, env)
    out_tracers = [JaxprTracer(self, PartialVal.unknown(a), None)
                   for a in out_avals]
    closed_jaxpr = convert_constvars_jaxpr(jaxpr)

    @partial(lu.wrap_init, debug_info=fwd.debug_info)
    @_memoize
    def fwd_jaxpr_thunk(*zeros):
      fwd_ = _interleave_fun(fwd.with_unknown_names(), zeros)
      fwd_jaxpr, _, consts = trace_to_jaxpr_dynamic(fwd_, in_avals)
      return fwd_jaxpr, consts

    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    params = dict(
        call_jaxpr=closed_jaxpr,
        fwd_jaxpr_thunk=fwd_jaxpr_thunk,
        num_consts=len(res) + len(env),
        bwd=bwd,
        out_trees=out_trees,
        symbolic_zeros=symbolic_zeros
    )
    eqn = new_eqn_recipe(self, (*res_tracers, *env_tracers, *tracers),
                         out_tracers, prim, params,
                         core.positional_effects(closed_jaxpr), source)
    for t in out_tracers: t.recipe = eqn
    return out_tracers

def partition_pvals(
    pvals: list[PartialVal]
  ) -> tuple[list[bool], list[AbstractValue], list[Any]]:
  knowns = [pval.is_known()  for pval in pvals                       ]
  avals  = [pval.get_aval()  for pval in pvals if not pval.is_known()]
  consts = [pval.get_known() for pval in pvals if     pval.is_known()]
  return knowns, avals, consts

@lu.transformation_with_aux2
def partial_eval_wrapper_nounits(
    f: Callable,
    store: lu.Store,
    in_knowns: Sequence[bool],
    in_avals: Sequence[AbstractValue],
    *in_consts: Any):
  in_avals_, in_consts_ = iter(in_avals), iter(in_consts)
  in_pvals = [PartialVal.known(next(in_consts_)) if known else
              PartialVal.unknown(next(in_avals_)) for known in in_knowns]
  sentinel = object()
  assert next(in_avals_, sentinel) is next(in_consts_, sentinel) is sentinel
  jaxpr, (*maybe_fwds, out_pvals, res, env) = f(in_pvals)
  out_knowns, out_avals, out_consts = partition_pvals(out_pvals)
  store.store((*maybe_fwds, out_knowns, out_avals, jaxpr, env))
  return (*out_consts, *res)

custom_partial_eval_rules: dict[Primitive, Callable] = {}


def abstract_eval_fun(fun: Callable, *avals,
                      debug_info: core.DebugInfo, **params):
  _, avals_out = trace_to_jaxpr(
      partial(fun, **params), ft.flatten_args(*avals), debug_info)
  assert all(isinstance(aval, AbstractValue) for aval in avals_out)
  return list(avals_out)


class JaxprTracer(Tracer[JaxprTrace]):
  __slots__ = ['pval', 'recipe']

  _trace: JaxprTrace

  def __init__(self, trace: JaxprTrace, pval: PartialVal,
               recipe: JaxprTracerRecipe | None):
    assert isinstance(pval, PartialVal)
    super().__init__(trace, pval.get_aval())
    self.pval = pval
    self.recipe = recipe

  def __repr__(self):
    return f'Traced<{self.aval}:{self._trace}>'

  @property
  def parents(self) -> Sequence[JaxprTracer]:
    if isinstance(self.recipe, JaxprEqnRecipe):
      # TODO broadcast_in_dim 可能会创建一个新的追踪器……
      return self.recipe.in_tracers
    else:
      return []

  def full_lower(self):
    known = self.pval.get_known()
    if known is not None:
      return core.full_lower(known)
    else:
      return self

  def is_known(self):
    return self.pval.is_known()

  def get_referent(self):
    if self.pval.is_known():
      return get_referent(self.pval.get_known())
    elif isinstance(self.recipe, (FreeVar, ConstVar, Literal)):
      return get_referent(self.recipe.val)
    else:
      return self


@profiler.annotate_function
def trace_to_jaxpr_nounits(
    fun: lu.WrappedFun, pvals: Sequence[PartialVal],
    instantiate: bool | Sequence[bool] = False,
  ) -> tuple[Jaxpr, list[PartialVal], list[core.Value]]:
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = JaxprTrace(parent_trace, current_name_stack, TraceTag())
    with core.ensure_no_leaks(trace):
      fun = trace_to_subjaxpr_nounits(fun, trace, instantiate, fun.debug_info)
      with core.set_current_trace(trace):
        jaxpr, (out_pvals, consts, env) = fun.call_wrapped(pvals)
        assert not env
      del trace, fun
      return jaxpr, out_pvals, consts

# TODO(mattjj): 多余的包装器……？
@lu.transformation2
def trace_to_subjaxpr_nounits(
    f: Callable,
    trace: JaxprTrace,
    instantiate: Sequence[bool] | bool,
    debug_info: core.DebugInfo,
    in_pvals: Sequence[PartialVal]):
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  out_tracers, jaxpr, out_consts, env = _trace_to_subjaxpr_nounits(
      f, trace, instantiate, in_pvals, debug_info)
  out_pvals = [t.pval for t in out_tracers]
  del out_tracers
  return jaxpr, (out_pvals, out_consts, env)

@lu.transformation2
def trace_to_subjaxpr_nounits2(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal]):
  assert isinstance(tag, TraceTag)
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = JaxprTrace(parent_trace, current_name_stack, tag)
    out_tracers, jaxpr, out_consts, env = _trace_to_subjaxpr_nounits(
        f, trace, instantiate, in_pvals, debug_info)
    out_pvals = [t.pval for t in out_tracers]
    del out_tracers
  return jaxpr, (out_pvals, out_consts, env)

def _trace_to_subjaxpr_nounits_no_lu_2(f: Callable, trace: JaxprTrace,
                               instantiate: Sequence[bool] | bool,
                               in_pvals: Sequence[PartialVal],
                               debug_info: core.DebugInfo):
  in_knowns  = [pval.is_known()     for pval in in_pvals]
  in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
  in_tracers = [trace.new_arg(pval) for pval in in_pvals if not pval.is_known()]
  in_args = merge_lists(in_knowns, in_tracers, in_consts)
  with core.set_current_trace(trace):
    ans = f(*in_args)
  assert isinstance(ans, ft.FlatTree), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  assert all(isinstance(x, Tracer) or core.valid_jaxtype(x) for x in ans), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  if isinstance(instantiate, bool):
    instantiate = [instantiate] * len(ans)
  out_tracers = ans.map(trace.to_jaxpr_tracer)
  out_tracers = out_tracers.map2(
      instantiate, lambda t, inst: trace.instantiate_const(t) if inst else t)
  out_tracers_ = [t for t in out_tracers if not t.is_known()]
  jaxpr, out_consts, env = tracers_to_jaxpr(
      in_tracers, out_tracers_, trace.effect_handles,
      debug_info.with_unknown_names())
  return out_tracers, jaxpr, out_consts, env

def _trace_to_subjaxpr_nounits(f: Callable, trace: JaxprTrace,
                               instantiate: Sequence[bool] | bool,
                               in_pvals: Sequence[PartialVal],
                               debug_info: core.DebugInfo):
  in_knowns  = [pval.is_known()     for pval in in_pvals]
  in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
  in_tracers = [trace.new_arg(pval) for pval in in_pvals if not pval.is_known()]
  in_args = merge_lists(in_knowns, in_tracers, in_consts)
  with core.set_current_trace(trace):
    ans = f(*in_args)
  assert isinstance(ans, (list, tuple)), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  assert all(core.valid_jaxtype(x) for x in ans), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  if isinstance(instantiate, bool):
    instantiate = [instantiate] * len(ans)
  out_tracers = map(trace.to_jaxpr_tracer, ans)
  out_tracers = [trace.instantiate_const(t) if inst else t
                 for inst, t in zip(instantiate, out_tracers)]
  out_tracers_ = [t for t in out_tracers if not t.is_known()]
  jaxpr, out_consts, env = tracers_to_jaxpr(
      in_tracers, out_tracers_, trace.effect_handles,
      debug_info.with_unknown_names())
  return out_tracers, jaxpr, out_consts, env

# 下面这个变体实现了一项优化：同时作为输入的残差
# 在辅助数据中指明，而不是作为输出传递出去。
# TODO(mattjj): 更新所有调用方以使用这个版本，并删除另一个版本。
@lu.transformation2
def trace_to_subjaxpr_nounits_fwd(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal]):
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = JaxprTrace(parent_trace, current_name_stack, tag)
    with core.set_current_trace(trace):
      out_tracers, jaxpr, out_consts, env = _trace_to_subjaxpr_nounits(
          f, trace, instantiate, in_pvals, debug_info)
    out_pvals = [t.pval for t in out_tracers]

    # 哪些 out_consts（即残差）只是被转发的输入？检查对象 id。
    in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
    id_map = {id(c): i for i, c in enumerate(in_consts)}
    fwds: list[int | None] = [id_map.get(id(c)) for c in out_consts]
    pruned_consts = [c for c, fwd in zip(out_consts, fwds) if fwd is None]

    del out_tracers
  return jaxpr, (fwds, out_pvals, pruned_consts, env)

# 下面这个变体实现了两项优化：
#  1. 同时也是原始输入的残差在辅助数据中标出，而不是
#     作为输出传递；
#  2. 同时也是原始输出的残差在辅助数据中标出，而不是
#     作为冗余输出传递。
def trace_to_subjaxpr_nounits_fwd2(
    f: Callable,
    tag: TraceTag,
    debug_info: core.DebugInfo,
    instantiate: bool | Sequence[bool],
    in_pvals: Sequence[PartialVal]):
  assert all(isinstance(pv, PartialVal) for pv in in_pvals), in_pvals
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = JaxprTrace(parent_trace, current_name_stack, tag)
    out_tracers, jaxpr, consts, env = _trace_to_subjaxpr_nounits_no_lu_2(
        f, trace, instantiate, in_pvals, debug_info)
    out_pvals = out_tracers.map(lambda t: t.pval)

  # 哪些 consts（即残差）只是被转发的输入？检查对象 id。
  in_consts  = [pval.get_known()    for pval in  in_pvals if    pval.is_known()]
  id_map = {id(c): i for i, c in enumerate(in_consts)}
  input_fwds: list[int | None] = [id_map.get(id(c)) for c in consts]

  # 哪些 consts（即残差）已经是原始输出？检查对象 id。
  out_consts = [pval.get_known()    for pval in out_pvals if    pval.is_known()]
  id_map = {id(c): i for i, c in enumerate(out_consts)}
  output_fwds: list[int | None] = [id_map.get(id(c)) for c in consts]

  pruned_consts = [c for c, f1, f2 in zip(consts, input_fwds, output_fwds)
                   if f1 is None and f2 is None]

  del out_tracers
  return jaxpr, (input_fwds, output_fwds, out_pvals, pruned_consts, env)


FreeVar = namedtuple('FreeVar', ['val'])
ConstVar = namedtuple('ConstVar', ['val'])
LambdaBinding = namedtuple('LambdaBinding', [])
class JaxprEqnRecipe(NamedTuple):
  eqn_id: Any
  in_tracers: Sequence[JaxprTracer]
  out_tracer_refs: Sequence[ref[JaxprTracer]]
  out_avals: Sequence[core.AbstractValue]
  primitive: Primitive
  params: dict[str, Any]
  effects: core.Effects
  source_info: source_info_util.SourceInfo
  ctx: JaxprEqnContext

JaxprTracerRecipe = (
    JaxprEqnRecipe | LambdaBinding | FreeVar | ConstVar | Literal
)

def new_eqn_recipe(trace: JaxprTrace,
                   in_tracers: Sequence[JaxprTracer],
                   out_tracers: Sequence[JaxprTracer],
                   primitive: Primitive,
                   params: dict[str, Any],
                   effects: core.Effects,
                   source_info: source_info_util.SourceInfo,
                   ctx: JaxprEqnContext | None = None) -> JaxprEqnRecipe:
  out_avals = [t.aval for t in out_tracers]
  ctx = ctx or core.current_jaxpr_eqn_context()
  return JaxprEqnRecipe(next(trace.counter), tuple(in_tracers), map(ref, out_tracers),
                        out_avals, primitive, params, effects, source_info,
                        ctx)

def tracers_to_jaxpr(
  in_tracers: Sequence[JaxprTracer],
  out_tracers: Sequence[JaxprTracer],
  effect_handles: Sequence[Any],
  debug_info: core.DebugInfo,
  ) -> tuple[Jaxpr, tuple[Any, ...], tuple[Any, ...]]:
  """根据输入和输出的追踪器构造 Jaxpr。

  Params:
    in_tracers: 为函数输入创建的追踪器
    out_tracers: 函数输出的追踪器。
    debug_info: 函数的调试信息。

  Returns: 一个三元组，包含一个 `Jaxpr`、一个对应于返回的 Jaxpr 中
    `constvars` 的常量值列表，以及一个环境值列表。
    环境值的变量已被前置到该
    Jaxpr 的 `invars` 中。
  """
  gensym = core.gensym()

  t_to_var: dict[TracerId, Var] = {}
  consts: dict[Var, Any] = {}
  env: dict[Var, JaxprTracer] = {}
  constid_to_var: dict[ConstId, Var] = {}  # 用于去重

  def get_atom(t: JaxprTracer) -> Atom:
    return t.recipe if type(t.recipe) is Literal else t_to_var[id(t)]

  def newvar(t: JaxprTracer | None) -> Var:
    assert t is not None
    var = gensym(t.aval)
    var_ = t_to_var.setdefault(id(t), var)
    assert var is var_
    return var

  processed_eqn_ids = set()
  eqns: list[core.JaxprEqn] = []
  is_high = False

  reachable = toposort
  tracers = reachable((*in_tracers, *out_tracers, *effect_handles))
  def sort_key(t):
    r = t.recipe
    return r.eqn_id if isinstance(r, JaxprEqnRecipe) else -1
  tracers = sorted(tracers, key=sort_key)

  for t in tracers:
    r = t.recipe
    if isinstance(r, JaxprEqnRecipe):
      # TODO broadcast_in_dim 可能创建一个新追踪器，它不在父节点中
      if r.eqn_id not in processed_eqn_ids:
        in_atoms = map(get_atom, r.in_tracers)
        outvars = [DropVar(a) if rf() is None else newvar(rf())
                   for a, rf in zip(r.out_avals, r.out_tracer_refs)]
        eqns.append(new_jaxpr_eqn(in_atoms, outvars, r.primitive, r.params,
                                  r.effects, r.source_info, r.ctx))
        in_avals = [x.aval for x in in_atoms]
        is_high |= r.primitive.is_high(*in_avals, **r.params)
        processed_eqn_ids.add(r.eqn_id)
    elif isinstance(r, LambdaBinding):
      if not any(t is in_tracer for in_tracer in in_tracers):
        raise core.escaped_tracer_error(t, f"Tracer not in input tracers: {t}")
      newvar(t)
    elif isinstance(r, ConstVar):
      var = constid_to_var.get(id(r.val))
      if var is None:
        var = constid_to_var[id(r.val)] = newvar(t)
        consts[var] = r.val
      t_to_var[id(t)] = var
    elif isinstance(r, FreeVar):
      env[newvar(t)] = r.val
    elif isinstance(r, Literal):
      pass
    elif r is None:
      assert False
    else:
      raise TypeError(r)

  env_vars, env_vals = unzip2(env.items())
  invars = [*env_vars, *map(get_atom, in_tracers)]
  const_vars, const_vals = unzip2(consts.items())
  outvars = map(get_atom, out_tracers)
  jaxpr_effects = make_jaxpr_effects(const_vars, invars, outvars, eqns)
  is_high |= any(x.aval.is_high for x in it.chain(const_vars, invars, outvars))
  jaxpr = Jaxpr(const_vars, invars,  # pyrefly: ignore[bad-argument-type]
                outvars, eqns, jaxpr_effects, debug_info, is_high)
  config.enable_checks.value and core.check_jaxpr(jaxpr)
  # del getvar  # 显然需要这样来避免循环引用的闭包！
  return jaxpr, const_vals, env_vals

@weakref_lru_cache
def move_envvars(jaxpr: Jaxpr, which: tuple[bool, ...]) -> Jaxpr:
  """把由 `which` 选中的开头若干 invars 移到未被选中的 invars 之后。"""
  assert not jaxpr.consts
  keep, env = partition_list(which, jaxpr.invars[:len(which)])
  return jaxpr.replace(invars=[*keep, *env, *jaxpr.invars[len(which):]])

def separate_consts(jaxpr: Jaxpr) -> tuple[Jaxpr, list[Any]]:
  """分离出常量并将它们显式返回。"""
  return convert_constvars_jaxpr(jaxpr), jaxpr.consts

def convert_constvars_jaxpr(jaxpr: Jaxpr) -> Jaxpr:
  """分离出常量，把常量输入暴露为开头的 invars。"""
  return _detach_consts(jaxpr) if jaxpr.consts else jaxpr

@weakref_lru_cache
def _detach_consts(jaxpr: Jaxpr) -> Jaxpr:
  return jaxpr.with_consts(())


def partial_eval_jaxpr_nounits(
    jaxpr: Jaxpr, unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool],
  ) -> tuple[Jaxpr, Jaxpr, list[bool], list[AbstractValue]]:
  """按数据依赖把一个 jaxpr 一分为二，拆成“已知”和“未知”两部分。

  也就是说，给定一个 jaxpr 和一个布尔序列，指明哪些 jaxpr
  输入（即 invars）被视为未知，则产出两个 jaxpr、一个
  布尔列表，表示原 jaxpr 的哪些输出是未知的（即
  对某个未知输入存在数据依赖），以及一个抽象值列表，
  这些抽象值代表残差（是第一个 jaxpr 输出的一部分，也是第二个
  jaxpr 的输入）。这两个 jaxpr 来自对原 jaxpr 的
  一阶原语应用所做的划分：依据某个应用的全部输入
  是否都已知（此时该应用被表示在
  “已知”jaxpr 中，其结果也被视为已知），还是其任一输入
  未知（此时该应用被表示在
  “未知”jaxpr 中，其结果也被视为未知）。高阶原语
  会被递归地一分为二。

  `instantiate` 参数可用于确保某些输出被提升到
  “未知”jaxpr 中。

  例如，给定如下输入 jaxpr：

    { lambda ; a:f32[] b:f32[]. let
        c:f32[] = cos a
        d:f32[] = sin a
        e:f32[] = neg d
        f:f32[] = mul e b
      in (c, f) }

  那么以 `unknowns=[False, True]` 和
  `instantiate=False` 应用该函数，会产出如下三元组：

    # jaxpr_known
    { lambda ; a:f32[]. let
       b:f32[] = cos a
       c:f32[] = sin a
       d:f32[] = neg c
     in (b, d) }

    # jaxpr_unknown
    { lambda ; a:f32[] b:f32[]. let c:f32[] = mul b a in (c,) }

    # out_unknowns
    [False, True]

  特别要注意，第一个输出（jaxpr_known）包含所有
  不与未知输入存在数据依赖的
  原语应用。还要注意输入与输出类型：第一个
  jaxpr 产出的输入类型表示原 jaxpr 中已知输入的类型，
  而第二个 jaxpr 产出的输出类型表示原 jaxpr 中
  未知输出的类型。

  在上例中，jaxpr_known 名为 `d` 的输出是一个_残差_
  输出，它对应 jaxpr_unknown 中名为 `a` 的输入。一般来说，
  jaxpr_known 会产生额外的输出（位于其输出列表末尾），
  这些输出对应原 jaxpr 的中间值，它们必须
  被传给 jaxpr_unknown（作为开头的输入）。
  """
  instantiate = tuple(instantiate) if isinstance(instantiate, list) else instantiate
  return _partial_eval_jaxpr_nounits(jaxpr, tuple(unknowns), instantiate, False)[:-1]

def partial_eval_jaxpr_nounits_fwd(
    jaxpr: Jaxpr, unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool],
    fwd: bool | Sequence[bool] = True,
) -> tuple[Jaxpr, Jaxpr, list[bool], list[AbstractValue], list[int | None]]:
  instantiate = tuple(instantiate) if isinstance(instantiate, list) else instantiate
  fwd = tuple(fwd) if isinstance(fwd, list) else fwd
  return _partial_eval_jaxpr_nounits(jaxpr, tuple(unknowns), instantiate, fwd)

@weakref_lru_cache
def _partial_eval_jaxpr_nounits(
    jaxpr: Jaxpr, in_unknowns: Sequence[bool],
    instantiate: bool | Sequence[bool], fwd: bool | Sequence[bool]):
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr), debug_info=jaxpr.debug_info)

  cell = []
  def fun(*known_vals_in):
    known_vals_in_ = iter(known_vals_in)
    unknown_avals = (a for a, uk in zip(jaxpr.in_avals, in_unknowns) if uk)
    in_pvals = [PartialVal.unknown(next(unknown_avals)) if uk
                else PartialVal.known(next(known_vals_in_)) for uk in in_unknowns]
    assert next(known_vals_in_, None) is next(unknown_avals, None) is None
    jaxpr_unknown_, (fwds, out_pvals, residuals, ()) = trace_to_subjaxpr_nounits_fwd(
        f, TraceTag(), jaxpr.debug_info, instantiate).call_wrapped(in_pvals)
    jaxpr_unknown = convert_constvars_jaxpr(jaxpr_unknown_)
    out_unknowns = [not pval.is_known() for pval in out_pvals]
    if type(fwd) is bool and not fwd:
      residuals_ = iter(residuals)
      residuals = [next(residuals_) if f is None else known_vals_in[f]
                   for f in fwds]
      assert next(residuals_, None) is None
      fwds = [None] * len(fwds)
    else:
      if type(fwd) is tuple:
        fwd_ = [f for f, uk in zip(fwd, in_unknowns) if not uk]
        residuals_, residuals = iter(residuals), []
        fwds = [residuals.append(next(residuals_)) if f is None else
                residuals.append(known_vals_in[f]) if not fwd_[f] else
                f for f in fwds]
      fwds, residuals = _include_consts_in_fwds(jaxpr.consts, fwds, residuals)
    res_avals = [core.typeof(r) for r in residuals]
    cell.append((out_unknowns, jaxpr_unknown, res_avals, fwds))
    known_vals_out = [pval.get_known() for pval in out_pvals if pval.is_known()]
    return [*known_vals_out, *residuals]

  known_avals = [a for a, uk in zip(jaxpr.in_avals, in_unknowns) if not uk]
  closed_jaxpr_known, _ = trace_to_jaxpr(
      fun, ft.flatten_args(*known_avals), f.debug_info.with_unknown_names())
  (out_unknowns, jaxpr_unknown, res_avals, fwds), = cell

  if config.enable_checks.value:
    core.check_jaxpr(closed_jaxpr_known)
    core.check_jaxpr(jaxpr_unknown)

  closed_jaxpr_unknown = jaxpr_unknown
  return closed_jaxpr_known, closed_jaxpr_unknown, out_unknowns, res_avals, fwds

def _include_consts_in_fwds(consts, fwds, residuals):
  if all(f is None for f in fwds):
    return fwds, residuals
  dummys = [object() for _ in range(max(f for f in fwds if f is not None) + 1)]
  residuals_ = iter(residuals)
  residuals = [next(residuals_) if f is None else dummys[f] for f in fwds]
  assert next(residuals_, None) is None
  idxs = {id(x): i for i, x in enumerate((*consts, *dummys))}
  fwds = [idxs.get(id(r)) for r in residuals]
  residuals = [r for r in residuals if id(r) not in idxs]
  return fwds, residuals


def partial_eval_jaxpr_custom(
    jaxpr: Jaxpr,
    in_unknowns: Sequence[bool],
    in_inst: bool | Sequence[bool],
    ensure_out_unknowns: bool | Sequence[bool],
    ensure_out_inst: bool | Sequence[bool],
    saveable: Callable[..., RematCases_],
  ) -> tuple[Jaxpr, Jaxpr, list[bool], list[bool], int]:
  *outs, num_res_ref = partial_eval_jaxpr_stateful(
      jaxpr, in_unknowns, in_inst, ensure_out_unknowns, ensure_out_inst, saveable)
  if num_res_ref:
    raise ValueError("Cannot use `partial_eval_jaxpr_custom` with stateful jaxprs.")
  return *outs,  # pyrefly: ignore[bad-return]

def partial_eval_jaxpr_stateful(
    jaxpr: Jaxpr,
    in_unknowns: Sequence[bool],
    in_inst: bool | Sequence[bool],
    ensure_out_unknowns: bool | Sequence[bool],
    ensure_out_inst: bool | Sequence[bool],
    saveable: Callable[..., RematCases_] | None,
  ) -> tuple[Jaxpr, Jaxpr, list[bool], list[bool], int, int]:
  if type(in_inst) is bool:
    in_inst = (in_inst,) * len(jaxpr.invars)
  if type(ensure_out_unknowns) is bool:
    ensure_out_unknowns = (ensure_out_unknowns,) * len(jaxpr.outvars)
  if type(ensure_out_inst) is bool:
    ensure_out_inst = (ensure_out_inst,) * len(jaxpr.outvars)
  if saveable is None:
    saveable = everything_saveable
  jaxpr_known, jaxpr_staged, out_unknowns, out_inst, num_res, num_res_ref = \
      _partial_eval_jaxpr_custom_cached(

          jaxpr, tuple(in_unknowns), tuple(in_inst), tuple(ensure_out_unknowns),

          tuple(ensure_out_inst), saveable)
  return jaxpr_known, jaxpr_staged, out_unknowns, out_inst, num_res, num_res_ref

everything_saveable = lambda *_, **__: True

@weakref_lru_cache
def _partial_eval_jaxpr_custom_cached(
    jaxpr: Jaxpr,
    in_unknowns: tuple[bool, ...],
    in_inst: tuple[bool, ...],
    ensure_out_unknowns: tuple[bool, ...],
    ensure_out_inst: tuple[bool, ...],
    saveable: Callable[..., RematCases_],
  ) -> tuple[Jaxpr, Jaxpr, list[bool], list[bool], int, int]:
  env: dict[Var, tuple[bool, bool]] = {}
  residuals: OrderedSet[Var] = OrderedSet()
  residual_refs: OrderedSet[Var] = OrderedSet()

  def read(x: Atom) -> tuple[bool, bool]:
    if type(x) is Var:
      return env[x]
    return (False, True)

  def write(unk: bool, inst: bool, v: Var) -> None:
    assert (unk, inst) != (True, False)
    env[v] = (unk, inst)

  def ensure_instantiated(inst: bool, x: Atom) -> Atom:
    if type(x) is Var and not inst:
      residuals.add(x)
    return x

  def has_effects(effects) -> bool:
    not_really_effects = (core.NamedAxisEffect, core.InternalMutableArrayEffect)
    return any(not isinstance(e, not_really_effects) for e in effects)

  known_eqns, staged_eqns = [], []
  foreach(write, in_unknowns, in_inst, jaxpr.invars)
  foreach(partial(write, False, True), jaxpr.constvars)
  for eqn in jaxpr.eqns:
    unks_in, inst_in = unzip2(map(read, eqn.invars))
    rule = partial_eval_jaxpr_custom_rules.get(eqn.primitive)
    if rule:
      eqn1, eqn2, unks_out, inst_out, res = rule(saveable, unks_in, inst_in, eqn)
      eqn1 and known_eqns.append(eqn1); eqn2 and staged_eqns.append(eqn2)
      for r in res:
        if isinstance(r.aval, AbstractRef):
          residual_refs.add(r)
        else:
          residuals.add(r)
      foreach(write, unks_out, inst_out, eqn.outvars)
    elif any(unks_in):
      inputs = map(ensure_instantiated, inst_in, eqn.invars)
      staged_eqns.append(eqn.replace(invars=inputs))
      foreach(partial(write, True, True), eqn.outvars)
    else:
      known_eqns.append(eqn)
      # 如果它是有副作用的原语，我们总是执行它并避免将它暂存。
      policy = ensure_enum(saveable(
          eqn.primitive, *[x.aval for x in eqn.invars], **eqn.params))
      if has_effects(eqn.effects) or isinstance(policy, SaveableType):
        foreach(partial(write, False, False), eqn.outvars)
      elif isinstance(policy, Offloadable):
        # TODO(slebedev): 这是一个真实的（类型检查）报错，需要修改 BUILD 才能解决。
        from jax._src.dispatch import device_put_p, ArrayCopySemantics  # pyrefly: ignore[missing-import]
        resvars = [Var(v.aval.update(memory_space=core.mem_kind_to_space(policy.dst)))
                   for v in eqn.outvars]
        offload_eqn = core.JaxprEqn(
            eqn.outvars, resvars, device_put_p,
            dict(
                devices=(core.mem_kind_to_space(policy.dst),) * len(eqn.outvars),
                srcs=(None,),
                copy_semantics=(ArrayCopySemantics.ALWAYS_COPY,),
            ),
            set(), source_info_util.new_source_info(), core.current_jaxpr_eqn_context())
        known_eqns.append(offload_eqn)
        # resvars 是已知的，并且在反向 jaxpr 中可用。
        foreach(partial(write, False, True), resvars)
        assert all(o.aval.memory_space == core.mem_kind_to_space(policy.src)  # pyrefly: ignore[missing-attribute]
                   for o in eqn.outvars)
        residuals.update(resvars)
        reload_eqn = core.JaxprEqn(
            resvars, eqn.outvars, device_put_p,
            dict(
              devices=(core.mem_kind_to_space(policy.src),) * len(resvars),
              srcs=(None,),
              copy_semantics=(ArrayCopySemantics.ALWAYS_COPY,)
            ),
            set(), source_info_util.new_source_info(), core.current_jaxpr_eqn_context())
        staged_eqns.append(reload_eqn)
        # outvars 是已知的，并且在反向 jaxpr 中可用。
        foreach(partial(write, False, True), eqn.outvars)
      else:
        assert isinstance(policy, RecomputeType)
        inputs = map(ensure_instantiated, inst_in, eqn.invars)
        staged_eqns.append(eqn.replace(invars=inputs))
        foreach(partial(write, False, True), eqn.outvars)
  unzipped = unzip2(map(read, jaxpr.outvars))
  out_unknowns, out_inst = list(unzipped[0]), list(unzipped[1])
  assert all(type(v) is Var for v in residuals), residuals

  for x, inst, ensure_inst in zip(jaxpr.outvars, out_inst, ensure_out_inst):
    if ensure_inst: ensure_instantiated(inst, x)
  out_unknowns = map(op.or_, out_unknowns, ensure_out_unknowns)
  out_inst     = map(op.or_, out_inst,     ensure_out_inst)

  ins_known, _ = partition_list(in_unknowns, jaxpr.invars)
  outs_known, _ = partition_list(out_unknowns, jaxpr.outvars)
  ref_res_is_input = [r in ins_known for r in residual_refs]
  non_input_res_refs, _ = partition_list(ref_res_is_input, list(residual_refs))
  ins_known_and_ref_res = [*ins_known, *non_input_res_refs]
  known_outvars = [*outs_known, *residuals]
  known_effects = make_jaxpr_effects(jaxpr.constvars, ins_known_and_ref_res,
                                     known_outvars, known_eqns)

  # TODO(mattjj,necula): 这里应更新调试信息
  jaxpr_known = jaxpr.replace(
      invars=ins_known_and_ref_res, outvars=known_outvars,
      eqns=known_eqns, effects=known_effects,
      debug_info=jaxpr.debug_info.with_unknown_names())
  config.enable_checks.value and core.check_jaxpr(jaxpr_known)

  _, ins_staged = partition_list(in_inst, jaxpr.invars)
  _, outs_staged = partition_list(out_inst, jaxpr.outvars)
  staged_invars = [*residuals, *non_input_res_refs, *ins_staged]
  staged_effects = make_jaxpr_effects(jaxpr.constvars, staged_invars,
                                      outs_staged, staged_eqns)
  # TODO(mattjj,necula): 这里应更新调试信息
  jaxpr_staged = jaxpr.replace(
      invars=staged_invars, outvars=outs_staged, eqns=staged_eqns,
      effects=staged_effects,
      debug_info=jaxpr.debug_info.with_unknown_names())
  config.enable_checks.value and core.check_jaxpr(jaxpr_staged)

  return (jaxpr_known, jaxpr_staged, out_unknowns, out_inst, len(residuals),
          len(non_input_res_refs))


MemoryKind = str

class RecomputeType: pass
Recompute = RecomputeType()

class SaveableType: pass
Saveable = SaveableType()

class Offloadable(NamedTuple):
  src: MemoryKind
  dst: MemoryKind

RematCases = RecomputeType | SaveableType | Offloadable
RematCases_ = RematCases | bool

def ensure_enum(case: bool | RematCases) -> RematCases:
  if isinstance(case, bool):
    return Saveable if case else Recompute
  if not isinstance(case, (RecomputeType, SaveableType, Offloadable)):
    msg = ("Value returned by a remat policy should be a bool or"
           " `ad_checkpoint.Recompute`, `ad_checkpoint.Saveable` or"
           " `ad_checkpoint.Offloadable(...)`."
           f" Got {case} of type {type(case)}.")
    if isinstance(case, Offloadable):
      msg += ("Did you return `Offloadable` instead of an instantiated"
              " `Offloadable(...)`?")
    raise TypeError(msg)
  return case

# 用于策略驱动的部分求值的原语规则返回一个 5 元组，
# 其中各分量分别表示：
#  * 表示 'known'（已知）一侧的 JaxprEqn（若没有已知分量则为 None），
#  * 表示 'unknown'（未知）一侧的 JaxprEqn（或 None），
#  * 一个布尔列表，指示哪些原始输出是未知的，
#  * 一个布尔列表，指示哪些原始输出在 'unknown' 一侧
#    已被实例化（即可用），
#  * 一个 Var 实例列表，表示要添加的残差（即要作为
#    'known' 一侧 jaxpr 的输出引出，并作为输入绑定变量（input binder）加入
#    'unknown' 一侧的 jaxpr）。
PartialEvalCustomResult = tuple[JaxprEqn | None, JaxprEqn | None,
                                Sequence[bool], Sequence[bool], list[Var]]
PartialEvalCustomRule = Callable[
    [Callable[..., RematCases_], Sequence[bool], Sequence[bool], JaxprEqn],
    PartialEvalCustomResult]
partial_eval_jaxpr_custom_rules: dict[Primitive, PartialEvalCustomRule] = {}


ParamsUpdater = Callable[[Sequence[bool], Sequence[bool], Sequence[bool],
                          Sequence[bool], int, dict, dict],
                         tuple[dict, dict]]
ResAvalUpdater = Callable[[dict[str, Any], AbstractValue], AbstractValue]
def _default_res_aval_updater(
    params: dict[str, Any], aval: AbstractValue) -> AbstractValue:
  return aval


def call_partial_eval_custom_rule(
    jaxpr_param_name: str, params_updater: ParamsUpdater,
    saveable: Callable[..., RematCases_], unks_in: list[bool], inst_in: list[bool],
    eqn: JaxprEqn, *, res_aval: ResAvalUpdater = _default_res_aval_updater,
    ctx = contextlib.nullcontext,
  ) -> tuple[JaxprEqn, JaxprEqn, Sequence[bool], Sequence[bool], list[Var]]:
  jaxpr = eqn.params[jaxpr_param_name]
  with ctx(eqn.params):
    jaxpr_known, jaxpr_staged, unks_out, inst_out, num_res = \
        partial_eval_jaxpr_custom(jaxpr, unks_in, inst_in, False, False, saveable)
  ins_known, _ = partition_list(unks_in, eqn.invars)
  out_binders_known, _ = partition_list(unks_out, eqn.outvars)
  _, ins_staged = partition_list(inst_in, eqn.invars)
  _, out_binders_staged = partition_list(inst_out, eqn.outvars)
  params_known = {**eqn.params, jaxpr_param_name: jaxpr_known}
  params_staged = {**eqn.params, jaxpr_param_name: jaxpr_staged}
  params_known, params_staged = params_updater(
      unks_in, inst_in, map(op.not_, unks_out), inst_out, num_res, params_known,
      params_staged)
  residuals = [Var(res_aval(params_known, var.aval))
               for var in jaxpr_staged.invars[:num_res]]
  eqn_known = new_jaxpr_eqn(
      ins_known, [*out_binders_known, *residuals], eqn.primitive, params_known,
      core.eqn_effects(jaxpr_known, ins_known), eqn.source_info, eqn.ctx)
  eqn_staged = new_jaxpr_eqn(
      [*residuals, *ins_staged], out_binders_staged, eqn.primitive,
      params_staged, core.eqn_effects(jaxpr_staged, [*residuals, *ins_staged]),
      eqn.source_info, eqn.ctx)
  assert len(eqn_staged.invars) == len(jaxpr_staged.invars)
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is Var and not inst]
  return eqn_known, eqn_staged, unks_out, inst_out, new_inst + residuals

# TODO(mattjj): 与 ParamsUpdater 统一（这个多接收一个 int）
ParamsUpdater2 = Callable[[Sequence[bool], Sequence[bool], Sequence[bool],
                           Sequence[bool], int, int, dict, dict],
                          tuple[dict, dict]]

def closed_call_partial_eval_custom_rule(
    jaxpr_param_name: str, params_updater: ParamsUpdater2,
    saveable: Callable[..., RematCases_], unks_in: list[bool], inst_in: list[bool],
    eqn: JaxprEqn, *, res_aval: ResAvalUpdater = _default_res_aval_updater,
  ) -> tuple[JaxprEqn, JaxprEqn, Sequence[bool], Sequence[bool], list[Var]]:
  # TODO(sharadmv,mattjj): 将这条规则与 call_partial_eval_custom_rule 去重。
  disallow_output_fwds = tuple(isinstance(v, DropVar) for v in eqn.outvars)
  # TODO(mattjj): 这只是为了 pjit……但我们还是把这段代码全删掉吧
  from jax._src.sharding_impls import UNSPECIFIED  # pyrefly: ignore[missing-import]
  in_shardings, in_layouts = eqn.params.get('in_shardings'), eqn.params.get('in_layouts')
  if in_shardings is not None:
    assert in_layouts is not None
    disallow_input_fwds = tuple(s is not UNSPECIFIED or l is not None
                                for s, l in zip(in_shardings, in_layouts))
  else:
    disallow_input_fwds = (False,) * len(unks_in)
  jaxpr_known, jaxpr_staged, unks_out, inst_out, num_res_ref, num_res_val, in_fwd, out_fwd = \
      _closed_jaxpr_partial_eval_custom_cached(
          eqn.params[jaxpr_param_name], (*unks_in,), (*inst_in,),
          disallow_input_fwds, disallow_output_fwds, saveable)
  num_res = num_res_ref + num_res_val
  out_binders_known, _ = partition_list(unks_out, eqn.outvars)
  ins_known, _ = partition_list(unks_in, eqn.invars)
  _, ins_staged = partition_list(inst_in, eqn.invars)
  _, out_binders_staged = partition_list(inst_out, eqn.outvars)
  params_known = {**eqn.params, jaxpr_param_name: jaxpr_known}
  params_staged = {**eqn.params, jaxpr_param_name: jaxpr_staged}
  params_known, params_staged = params_updater(
      unks_in, inst_in, map(op.not_, unks_out), inst_out,
      sum(fin is fout is None for fin, fout in zip(in_fwd, out_fwd)),
      num_res, params_known, params_staged)
  res_val_binders, res_ref_binders = split_list(
      [Var(res_aval(params_known, v))
       for v in jaxpr_staged.in_avals[:num_res]], [num_res_val])
  res_val_binders = [v for v, fin, fout in zip(res_val_binders, in_fwd, out_fwd)
                     if fin is fout is None]
  res_val_binders_ = iter(res_val_binders)
  res_val_vars = [out_binders_known[fout] if fout is not None else
                  ins_known[fin] if fin is not None else
                  next(res_val_binders_) for fin, fout in zip(in_fwd, out_fwd)]
  assert next(res_val_binders_, None) is None
  eqn_known = new_jaxpr_eqn(
      [*ins_known, *res_ref_binders], [*out_binders_known, *res_val_binders],
      eqn.primitive, params_known,
      core.eqn_effects(jaxpr_known, [*ins_known, *res_ref_binders]),
      eqn.source_info, eqn.ctx)
  eqn_staged = new_jaxpr_eqn(
      [*res_val_vars, *res_ref_binders, *ins_staged], out_binders_staged,
      eqn.primitive, params_staged,
      core.eqn_effects(jaxpr_staged, [*res_val_vars, *res_ref_binders, *ins_staged]),
      eqn.source_info, eqn.ctx)
  assert len(eqn_staged.invars) == len(jaxpr_staged.in_avals)
  assert len(ins_known) + len(res_ref_binders) == len(jaxpr_known.invars)
  assert len(ins_staged) + len(res_ref_binders) + len(res_val_vars) == len(jaxpr_staged.invars)
  assert len(out_binders_known) + len(res_val_binders) == len(jaxpr_known.outvars)
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is Var and not inst]
  new_vars = [*new_inst, *res_val_vars, *res_ref_binders]
  return eqn_known, eqn_staged, unks_out, inst_out, new_vars  # pyrefly: ignore[bad-return]

@weakref_lru_cache
def _closed_jaxpr_partial_eval_custom_cached(
    jaxpr: Jaxpr, unks_in: tuple[bool, ...], inst_in: tuple[bool, ...],
    disallowed_input_forwards: tuple[bool, ...],
    disallowed_output_forwards: tuple[bool, ...],
    saveable: Callable
    ) -> tuple[Jaxpr, Jaxpr, Sequence[bool], Sequence[bool],
               int, int, Sequence[int | None], Sequence[int | None]]:
  jaxpr_known_, jaxpr_staged_, unks_out, inst_out, num_res_val, num_res_ref = \
      partial_eval_jaxpr_stateful(jaxpr, unks_in, inst_in,
                                  False, False, saveable)

  num_out_primals = len(jaxpr_known_.outvars) - num_res_val
  out_vars, res_vars = split_list(jaxpr_known_.outvars, [num_out_primals])

  # 计算哪些残差值输出同时也是原始输入。
  disallowed, _ = partition_list(unks_in, disallowed_input_forwards)
  idx_map = {id(v): i for i, (v, b) in enumerate(zip(jaxpr_known_.invars, disallowed))
             if not b}
  in_fwd = [idx_map.get(id(v)) for v in res_vars]

  # 计算哪些残差值输出同时也是*未被丢弃的*原始输出。
  disallowed, _ = partition_list(unks_out, disallowed_output_forwards)
  idx_map = {id(v): i for i, (v, b) in enumerate(zip(out_vars, disallowed))
             if not b}
  out_fwd = [idx_map.get(id(v)) for v in res_vars]

  # 通过移除被转发的输出（forwards）来精简 jaxpr_known_ 的输出。
  keep = [f1 is f2 is None for f1, f2 in zip(in_fwd, out_fwd)]
  jaxpr_known_ = prune_jaxpr_outputs(jaxpr_known_, [True] * num_out_primals + keep)

  return (jaxpr_known_, jaxpr_staged_, unks_out, inst_out, num_res_ref,
          num_res_val, in_fwd, out_fwd)


def _jaxpr_forwarding(jaxpr: Jaxpr) -> list[int | None]:
  # 计算哪些输入只是被转发（forward）到输出。
  fwds: dict[Var, Atom] = dict(zip(jaxpr.invars, jaxpr.invars))
  for eqn in jaxpr.eqns:
    if eqn.primitive in forwarding_rules:
      eqn = eqn.replace(invars=[a if type(a) is Literal else fwds.get(a, a)
                                for a in eqn.invars])
      fwd_idx, _ = forwarding_rules[eqn.primitive](eqn)
      for v_orig, idx in zip(eqn.outvars, fwd_idx):
        if idx is not None:
          fwds[v_orig] = eqn.invars[idx]
  idxs: dict[Var, int] = {v: i for i, v in enumerate(jaxpr.invars)}
  return [None if type(v) is Literal else idxs.get(fwds.get(v))  # pyrefly: ignore[bad-argument-type]
          for v in jaxpr.outvars]


def prune_jaxpr_outputs(jaxpr: Jaxpr, used_outputs: Sequence[bool]) -> Jaxpr:
  return _prune_jaxpr_outputs_cached(jaxpr, tuple(used_outputs))

def _prune_jaxpr_outputs(jaxpr: Jaxpr, used_outputs: tuple[bool, ...]) -> Jaxpr:
  outvars = [v for v, b in zip(jaxpr.outvars, used_outputs) if b]
  dbg = core.DebugInfo(
      jaxpr.debug_info.traced_for, jaxpr.debug_info.func_src_info,
      jaxpr.debug_info.arg_names,
      jaxpr.debug_info.filter_result_paths(used_outputs))
  new_jaxpr = jaxpr.replace(outvars=outvars, debug_info=dbg)
  config.enable_checks.value and core.check_jaxpr(new_jaxpr)
  return new_jaxpr
_prune_jaxpr_outputs_cached = weakref_lru_cache(_prune_jaxpr_outputs)

# 自从 Jaxpr/Jaxpr 合并之后，这与 prune_jaxpr_outputs 相同
# （附带常量由 Jaxpr.replace 保留）。
prune_closed_jaxpr_outputs = prune_jaxpr_outputs

def dedup_jaxpr_outputs(jaxpr: Jaxpr, num_kept_prefix: int,
                        fwdable_prefix: Sequence[bool] | None = None,
                        ) -> tuple[Jaxpr, list[int | None]]:
  """精简重复的输出，供调用方在求值后恢复。"""
  prefix_vars, rest_vars = split_list(jaxpr.outvars, [num_kept_prefix])
  fwdable = ([True] * num_kept_prefix if fwdable_prefix is None
             else fwdable_prefix)
  idx_map = {id(v): i for i, (v, f) in enumerate(zip(prefix_vars, fwdable))
             if f}
  num_kept = num_kept_prefix
  out_fwd: list[int | None] = [None] * num_kept_prefix
  for v in rest_vars:
    if (fwd := idx_map.get(id(v))) is None:
      idx_map[id(v)] = num_kept
      num_kept += 1
    out_fwd.append(fwd)
  jaxpr = prune_jaxpr_outputs(jaxpr, [f is None for f in out_fwd])
  return jaxpr, out_fwd


def dce_jaxpr(jaxpr: Jaxpr, used_outputs: bool | Sequence[bool],
              instantiate: bool | Sequence[bool] = False,
              ) -> tuple[Jaxpr, list[bool]]:
  """对给定的 jaxpr 执行死代码消除。

  Args:
    jaxpr: 要进行 DCE 的 jaxpr。
    used_outputs: 一个布尔列表，指示哪些输出被使用。
    instantiate: 一个布尔值或布尔列表，指示哪些输入应被视为已使用，
      无论它们是否真的在 jaxpr 中被使用。
      若是布尔值，则对所有输入使用相同的值。

  Returns:
    一个 ``(new_jaxpr, used_inputs)`` 元组。
  """
  if type(used_outputs) is bool:
    used_outputs = (used_outputs,) * len(jaxpr.outvars)
  if type(instantiate) is bool:
    instantiate = (instantiate,) * len(jaxpr.invars)

  return _dce_jaxpr(jaxpr, tuple(used_outputs), tuple(instantiate))


def dce_jaxpr_consts(jaxpr: Jaxpr, used_outputs: Sequence[bool],
                     instantiate: bool | Sequence[bool] = False,
                     ) -> tuple[Jaxpr, list[bool], list[bool]]:
  new_jaxpr, used_inputs_ = dce_jaxpr(convert_constvars_jaxpr(jaxpr),
                                      used_outputs, instantiate)
  used_consts, used_inputs = split_list(used_inputs_, [len(jaxpr.constvars)])
  new_jaxpr = new_jaxpr.with_consts(
      [c for c, used in zip(jaxpr.consts, used_consts) if used])
  return new_jaxpr, used_consts, used_inputs


def _default_dce_rule(
    used_outs: list[bool], eqn: JaxprEqn
  ) -> tuple[list[bool], JaxprEqn | None]:
  if not any(used_outs) and not has_effects(eqn):
    return [False] * len(eqn.invars), None
  return [True] * len(eqn.invars), eqn

dce_rules: dict[Primitive, DCERule] = {}

dceable_effects = effects.EffectTypeSet()

dceable_effects.add_type(ReadEffect)
dceable_effects.add_type(core.NamedAxisEffect)
dceable_effects.add_type(core.InternalMutableArrayEffect)

def _free_ref_dce_rule(
    used_outs: list[bool], eqn: JaxprEqn
) -> tuple[list[bool], JaxprEqn | None]:
  # 绝不会对 free_ref 做 DCE。
  del used_outs
  return [True] * len(eqn.invars), eqn
dce_rules[core.free_ref_p] = _free_ref_dce_rule


def has_effects(eqn: JaxprEqn) -> bool:
  effs = {e for e in eqn.effects if not dceable_effects.contains(e)}
  return bool(effs)


@weakref_lru_cache
def _dce_jaxpr(jaxpr: Jaxpr, used_outputs: tuple[bool, ...],
               instantiate: tuple[bool, ...]
               ) -> tuple[Jaxpr, list[bool]]:
  env: dict[Var, bool] = {}

  def read(v: Var) -> bool:
    return env.get(v, False)

  def write(x: Atom, b: bool) -> None:
    if type(x) is Var:
      env[x] = read(x) or b

  new_eqns = []
  foreach(write, jaxpr.outvars, used_outputs)
  for eqn in jaxpr.eqns[::-1]:
    used_outs = map(read, eqn.outvars)
    rule = dce_rules.get(eqn.primitive, _default_dce_rule)
    used_ins, new_eqn = rule(used_outs, eqn)
    if new_eqn is not None:
      new_eqns.append(new_eqn)
    foreach(write, eqn.invars, used_ins)
  used_inputs = map(read, jaxpr.invars)
  used_inputs = map(op.or_, instantiate, used_inputs)

  invars = [v for v, b in zip(jaxpr.invars, used_inputs)   if b]
  outvars = [v for v, b in zip(jaxpr.outvars, used_outputs) if b]
  eqns = new_eqns[::-1]
  jaxpr_effects = make_jaxpr_effects(jaxpr.constvars, invars, outvars, eqns)

  dbg = core.DebugInfo(
      jaxpr.debug_info.traced_for, jaxpr.debug_info.func_src_info,
      jaxpr.debug_info.filter_arg_names(used_inputs),
      jaxpr.debug_info.filter_result_paths(used_outputs))
  new_jaxpr = jaxpr.replace(invars=invars, outvars=outvars, eqns=eqns,
                            effects=jaxpr_effects, debug_info=dbg)
  config.enable_checks.value and core.check_jaxpr(new_jaxpr)

  return new_jaxpr, used_inputs

DCERule = Callable[[list[bool], JaxprEqn],
                   tuple[list[bool], JaxprEqn | None]]


@weakref_lru_cache
def _cached_closed_call_dce(jaxpr_, used_outputs: tuple[bool, ...]
                            ) -> tuple[Jaxpr, list[bool]]:
  # dce_jaxpr 会保留附带常量（constvars 永远不会被精简）。
  return dce_jaxpr(jaxpr_, used_outputs)

def dce_jaxpr_closed_call_rule(used_outputs: list[bool], eqn: JaxprEqn
                               ) -> tuple[list[bool], JaxprEqn | None]:
  # TODO(mattjj): 是否与上面的规则去重？
  if not any(used_outputs) and not has_effects(eqn):
    return [False] * len(eqn.invars), None
  jaxpr_ = eqn.params['call_jaxpr']
  closed_jaxpr, used_inputs = _cached_closed_call_dce(jaxpr_, tuple(used_outputs))
  new_invars = [v for v, used in zip(eqn.invars, used_inputs) if used]
  effects = core.eqn_effects(closed_jaxpr, new_invars)
  new_params = dict(eqn.params, call_jaxpr=closed_jaxpr)
  new_eqn = new_jaxpr_eqn(
      new_invars,
      [v for v, used in zip(eqn.outvars, used_outputs) if used],
      eqn.primitive, new_params, effects, eqn.source_info, eqn.ctx)
  return used_inputs, new_eqn

def close_jaxpr(jaxpr: Jaxpr) -> Jaxpr:
  # 现在 Jaxpr 和 ClosedJaxpr 已合并，每个 jaxpr 都是封闭的：
  # constvars 正是附带常量值的那些输入。
  return jaxpr

def move_binders_to_front(closed_jaxpr: Jaxpr, to_move: Sequence[bool]
                          ) -> Jaxpr:
  return _move_binders_to_front(closed_jaxpr, tuple(to_move))

@weakref_lru_cache
def _move_binders_to_front(jaxpr: Jaxpr, to_move: tuple[bool, ...]
                           ) -> Jaxpr:
  assert len(jaxpr.in_avals) == len(to_move)
  new_invars = _move_to_front(jaxpr.invars, to_move)
  if jaxpr.debug_info.arg_names is None:
    new_arg_names = None
  else:
    new_arg_names = tuple(_move_to_front(jaxpr.debug_info.arg_names, to_move))
  dbg = jaxpr.debug_info._replace(arg_names=new_arg_names)
  return jaxpr.replace(invars=new_invars, debug_info=dbg)

def _move_to_front(lst: Sequence, to_move: Sequence[bool]) -> Sequence:
  return ([elt for elt, move in zip(lst, to_move) if move] +
          [elt for elt, move in zip(lst, to_move) if not move])

def move_binders_to_back(closed_jaxpr: Jaxpr, to_move: Sequence[bool]
                         ) -> Jaxpr:
  assert len(to_move) <= len(closed_jaxpr.invars)
  to_move = [*to_move] + [False] * (len(closed_jaxpr.invars) - len(to_move))
  return move_binders_to_front(closed_jaxpr, map(op.not_, to_move))


class DynamicJaxprTracer(Tracer['DynamicJaxprTrace']):
  __slots__ = ['val', 'parent', '_debug_info']

  _trace: DynamicJaxprTrace

  def __init__(self, trace: DynamicJaxprTrace,
               aval: core.AbstractValue,
               val : Atom,
               line_info: source_info_util.SourceInfo | None = None,
               parent : TracingEqn | None = None):
    # TODO(dougalm): 移除 aval。既然有了 val，它就是多余的。
    Tracer.__init__(self, trace, aval)  # 比 super() 稍快一些
    self._line_info = line_info
    self._debug_info = self._trace.frame.debug_info  # 用于 UnexpectedTracerError
    self.val = val
    self.parent = parent

  def _short_repr(self):
    return f"JitTracer({self.aval})"

  def full_lower(self):
    atom = self.val
    if isinstance(atom, Literal):
      return atom.val
    else:
      maybe_const = self._trace.frame.constvar_to_val.get(atom)
      if maybe_const is None:
        return self
      else:
        return core.full_lower(maybe_const)

  def _contents(self):
    return ()

  def _origin_msg(self):
    invar_pos, progenitor_eqns = self._trace.frame.find_progenitors(self)
    dbg = self._debug_info
    if dbg is None:
      return ""

    origin = ("The error occurred while tracing the function "
              f"{dbg.func_src_info} for {dbg.traced_for}. ")
    if invar_pos:
      try:
        arg_names = [(dbg.arg_names[i] if dbg.arg_names is not None else "unknown")
                     for i in invar_pos]
      except IndexError:
        return ""  # TODO(mattjj): 弄清楚何时不满足 (invar_pos < len(arg_info))
      if len(arg_names) == 1:
        arg_info_str = f"the argument {arg_names[0]}"
      elif len(arg_names) == 2:
        arg_info_str = f"the arguments {arg_names[0]} and {arg_names[1]}"
      else:
        *rest, last = arg_names
        arg_info_str = f"the arguments {', '.join(rest)}, and {last}"
      origin += ("This concrete value was not available in Python because it "
                 f"depends on the value{'s' if len(invar_pos) > 1 else ''} "
                 f"of {arg_info_str}.")
    elif progenitor_eqns:
      msts = ["  operation "
              f"{core.pp_eqn(eqn, core.JaxprPpContext(), core.JaxprPpSettings(print_shapes=True))}\n"
              f"    from line {source_info_util.summarize(eqn.source_info)}"
              for eqn in progenitor_eqns[:5]]  # 最多显示 5 条
      origin += ("This value became a tracer due to JAX operations on these lines:"
                 "\n\n" + "\n\n".join(msts))
      if len(progenitor_eqns) > 5:
        origin += "\n\n(Additional originating lines are not shown.)"
    return "\n" + origin

  def get_const(self):
    return self._trace.get_const(self)

  def get_referent(self):
    frame = self._trace.frame
    atom = self.val
    val = frame.constvar_to_val.get(atom) if isinstance(atom, Var) else None
    return self if val is None else get_referent(val)

core.pytype_aval_mappings[DynamicJaxprTracer] = lambda x: x.aval

def make_jaxpr_effects(constvars, invars, outvars, eqns) -> effects.Effects:
  jaxpr_effects = set()
  input_vars = {*constvars, *invars}
  mut_arrays = set()
  for eqn in eqns:
    if eqn.primitive.ref_allocating:
      outvar, = eqn.outvars
      mut_arrays.add(outvar)
    for eff in eqn.effects:
      if isinstance(eff, effects.JaxprInputEffect):
        if eff.input in mut_arrays:
          continue
        if eff.input not in input_vars:
          # TODO(mattjj): 请求宽恕（ask for forgiveness）
          dbg = type('Fake', (), {'resolve_result_paths': lambda self_: self_,
                                  'assert_arg_names': lambda _, __: None,
                                  'assert_result_paths': lambda _, __: None,
                                  })()
          raise ValueError(
                f"`JaxprInputEffect` {eff} does not have "
                f"a corresponding jaxpr input."
                f"\n Equation: {eqn}\n"
                f"\n Effects: {eqn.effects}\n"
                "\n Jaxpr: "
                f"{core.Jaxpr(constvars, invars, outvars, eqns, set(), dbg)}")
      jaxpr_effects.add(eff)
  return jaxpr_effects


class JaxprStackFrame:
  __slots__ = (
      'gensym', 'constid_to_tracer', 'constvar_to_val', 'tracing_eqns',
      'invars', 'effects', 'debug_info', 'is_high', 'auto_dce')

  gensym: Callable[[AbstractValue], Var]
  constid_to_tracer: WeakValueDictionary[ConstId, DynamicJaxprTracer]
  constvar_to_val: dict[Var, Any]
  tracing_eqns: list[ReferenceType[TracingEqn] | TracingEqn]
  invars: list[Var]
  effects: core.Effects
  debug_info: core.DebugInfo | None
  is_high: bool
  auto_dce: bool

  def __init__(self, debug_info: core.DebugInfo | None, auto_dce: bool):
    self.gensym = core.gensym()
    self.constid_to_tracer = WeakValueDictionary()
    self.constvar_to_val = {}
    self.tracing_eqns = []      # 当我们从 main 弹出栈帧时清空
    self.invars = []
    self.effects = set()
    self.debug_info = debug_info
    self.is_high = False
    self.auto_dce = auto_dce

  def add_eqn(self, eqn: TracingEqn):
    assert isinstance(eqn, TracingEqn)
    r = eqn if (eqn.effects or not self.auto_dce) else ref(eqn)
    self.tracing_eqns.append(r)

  def get_eqns(self):
    eqns = []
    for tracing_eqn in self.tracing_eqns:
      e = tracing_eqn() if isinstance(tracing_eqn, ReferenceType) else tracing_eqn
      if e is None: continue
      if isinstance(e, TracingEqn):
        e = JaxprEqn(
            [t.val for t in e.in_tracers],
            e.outvars, e.primitive, e.params, e.effects, e.source_info, e.ctx)
      eqns.append(e)
    return eqns

  def to_jaxpr(
      self, trace: DynamicJaxprTrace,
      out_tracers: Sequence[Tracer],
      debug_info: core.DebugInfo,
      source_info: SourceInfo,
    ) -> tuple[Jaxpr, list[Any]]:
    # 我们特意在调用 get_eqns() 之前对 constvar_to_val 做快照，
    # 以避免以下情形：
    # * 我们调用 get_eqns()，对各个方程做快照。
    # * 垃圾回收运行，删除了一些 TracingEqn，这可能传递性地
    #   释放常量 Var。
    # * 此时我们的方程中就带有悬空的 Var 引用。
    # 如果先对 Var 做快照，就不会有这个问题。
    constvars, constvals = unzip2(self.constvar_to_val.copy().items())
    eqns = self.get_eqns()
    outvars = [t.val for t in out_tracers]
    effs = make_jaxpr_effects(constvars, self.invars, outvars, eqns)

    all_vars = it.chain(constvars, self.invars, outvars)
    is_high = self.is_high or any(v.aval.is_high for v in all_vars)

    jaxpr = Jaxpr(constvars, self.invars, outvars, eqns, effs, debug_info, is_high)
    return jaxpr, list(constvals)

  def newvar(self, aval):
    return self.gensym(aval)

  def find_progenitors(self, tracer):
    eqns = self.get_eqns()
    var = tracer.val
    if not var or isinstance(var, Literal):
      return None, None
    active_vars = {var}
    for eqn in eqns[::-1]:
      produced = set(eqn.outvars) & active_vars
      if produced:
        active_vars.difference_update(produced)
        active_vars.update({v for v in eqn.invars if type(v) is Var})
    invar_positions = [i for i, v in enumerate(self.invars) if v in active_vars]
    constvars = active_vars & set(self.constvar_to_val.copy())
    const_eqns = [eqn for eqn in eqns if any(
        v in constvars if type(v) is Var else type(v) is Literal
        for v in eqn.invars)]
    return invar_positions, const_eqns


ConstFoldRule = Callable[
    [list[Any | None], Any, list[AbstractValue]],
    list[Any | None] | None,
]
const_fold_rules: dict[Primitive, ConstFoldRule] = {}

ForwardingRule = Callable[
    [JaxprEqn],
    tuple[list[int | None], JaxprEqn | None]
]
forwarding_rules: dict[Primitive, ForwardingRule] = {}


@multi_weakref_lru_cache
def _cached_abstract_eval(primitive: core.Primitive, *avals, **params):
  return primitive.abstract_eval(*avals, **params)


def _verify_params_are_hashable(
    primitive: core.Primitive, params: dict[str, Any]) -> None:
  for k, v in params.items():
    try:
      hash(v)
    except TypeError as e:
      raise TypeError(
        "As of JAX v0.7, parameters to jaxpr equations must have __hash__ and "
        f"__eq__ methods. In a call to primitive {primitive}, the value of "
        f"parameter {k} was not hashable: {v}") from e

# 在追踪期间我们使用 TracingEqn 而非 JaxprEqn，以便基于 Python
# 引用计数实现自动的即时死代码消除（DCE）。DynamicJaxprTracer 指向
# TracingEqn，TracingEqn 又指向 DynamicJaxprTracer，因此不可达的常量
# 可以被释放。

@dataclass(slots=True, weakref_slot=True)
class TracingEqn:
  in_tracers: list[DynamicJaxprTracer]
  outvars: list[Var]
  primitive: Primitive
  params: dict[str, Any]
  effects: core.Effects
  source_info: source_info_util.SourceInfo
  ctx: JaxprEqnContext

  def __init__(self, in_tracers, outvars, primitive, params, effects, source_info, ctx):
    self.in_tracers = in_tracers
    self.outvars = outvars
    self.primitive = primitive
    self.params = params
    self.effects = effects
    self.source_info = source_info
    self.ctx = ctx

  # 允许 TracingEqn 对 JaxpeEqn 做鸭子类型（duck-type），因为某些转发
  # 规则需要同时兼容两者。TODO(dougalm): 等我们修好转发后，
  # 就移除这一点。
  @property
  def invars(self):
    return self.in_tracers

class DynamicJaxprTrace(core.Trace):
  __slots__ = ("frame", "tag", "parent_trace")

  # 注意，只有当 DynamicJaxprTrace 与 LinearizeTrace 关联时才会使用
  # tag；否则 tag 将处于未定义状态。
  tag: core.TraceTag
  frame: JaxprStackFrame
  parent_trace: core.Trace | None
  requires_low: bool

  def __init__(self, debug_info: core.DebugInfo | None,
               parent_trace: core.Trace | None = None,
               lower: bool = False, auto_dce: bool =False):
    super().__init__()
    self.requires_low = lower
    self.frame = JaxprStackFrame(debug_info, auto_dce)
    self.parent_trace = parent_trace

  def invalidate(self):
    # TODO(mattjj): 暴露了已存在的追踪器泄漏；修复它们并重新启用！
    # super().invalidate()

    # 避免循环引用
    self.frame.tracing_eqns = []  # thunk -> eqn -> in_tracers -> trace ->
    # -> frame -> tracing_eqns -> thunk

    # TODO(dougalm): 考虑到基于引用计数的 DCE，我们或许可以移除这些
    self.frame.constid_to_tracer = {}  # pyrefly: ignore[bad-assignment]
    self.frame.constvar_to_val = {}

  def to_jaxpr_tracer(self, x, source_info: SourceInfo):
    if isinstance(x, DynamicJaxprTracer) and x._trace is self:
      return x
    else:
      m = getattr(x, "dimension_as_value", None)
      if m is not None:
        with core.set_current_trace(self):
          x = m()
        return self.to_jaxpr_tracer(x, source_info)
      else:
        return self.new_const(x, source_info)

  def var_to_tracer(self, var, source_info, parent=None):
    return DynamicJaxprTracer(self, var.aval, var, source_info, parent)

  def new_arg(self, aval, source_info: SourceInfo):
    var = self.frame.newvar(aval)
    tracer = DynamicJaxprTracer(self, aval, var, source_info)
    self.frame.invars.append(var)
    return tracer

  def make_eqn(self, in_tracers, out_avals, primitive, params,
               effects, source_info=None, ctx = None):
    source_info = source_info or source_info_util.new_source_info()
    ctx = ctx or core.current_jaxpr_eqn_context()
    outvars = map(self.frame.newvar, out_avals)
    if effects:
      effects = core.resolve_input_effects(effects, [t.val for t in in_tracers])
    if config.enable_checks.value:
      assert all(isinstance(x, DynamicJaxprTracer) for x in in_tracers)
      assert all(isinstance(v,  Var)               for v in outvars)
    eqn = TracingEqn(in_tracers, outvars, primitive, params, effects, source_info, ctx)
    out_tracers = [self.var_to_tracer(v, source_info, eqn) for v in outvars]
    return eqn, out_tracers

  def emit_eqn(self, in_tracers, out_avals, primitive, params, effects, source_info=None, ctx=None):
    eqn, out_tracers = self.make_eqn(in_tracers, out_avals, primitive, params, effects, source_info, ctx)
    self.frame.add_eqn(eqn)
    return out_tracers

  def new_const(self, c, source_info: SourceInfo,
                aval: AbstractValue | None = None):
    # TODO(mattjj): 对于 int 或可哈希的常量，不要依赖 id
    tracer = self.frame.constid_to_tracer.get(id(c))
    if tracer is None:
      if aval is None:
        aval = typeof(c)
      tracer = self._new_const(aval, c, source_info)
    return tracer

  pure = lift = new_const

  def _new_const(self, aval, c, source_info: SourceInfo) -> DynamicJaxprTracer:
    id_c = id(c)
    assert type(c) not in (int, float, complex, np.generic, np.ndarray), (
        f"non-canonical constant of type {type(c).__name__}: {c!r}")
    if core.is_literalable(c, self.frame.auto_dce):
      val = Literal(c, aval)
      return DynamicJaxprTracer(self, aval, val, source_info)
    else:
      var = self.frame.newvar(aval)
      tracer = DynamicJaxprTracer(self, aval, var, source_info)
      self.frame.constid_to_tracer[id_c] = tracer
      self.frame.constvar_to_val[var] = c
      finalize(tracer, self.finalize_const, var, id_c)
      return tracer

  def finalize_const(self, var, constid):
    self.frame.constvar_to_val.pop(var, None)

  def get_const(self, tracer) -> Any:
    atom = tracer.val
    if isinstance(atom, Literal):
      return atom.val
    else:
      return self.frame.constvar_to_val.get(atom)

  def stage_value(self, val):
    if config.eager_constant_folding.value and not isinstance(val, Tracer):
      return core.eval_trace.stage_value(val)
    source_info = source_info_util.current()
    return self.to_jaxpr_tracer(val, source_info=source_info)

  def process_primitive(self, primitive, tracers, params, /):
    self.frame.is_high |= primitive.is_high(*map(typeof, tracers), **params)
    if config.eager_constant_folding.value and not any(isinstance(x, Tracer) for x in tracers):
      avals = tuple(core.typeof(x) for x in tracers)
      return primitive.bind_with_trace(core.eval_trace, tracers, avals, params)
    source_info = source_info_util.current()
    to_jaxpr_tracer = partial(self.to_jaxpr_tracer, source_info=source_info)
    jaxpr_tracers = map(to_jaxpr_tracer, tracers)
    if primitive in custom_staging_rules:
      return custom_staging_rules[primitive](self, source_info, *jaxpr_tracers,
                                             **params)
    return self.default_process_primitive(
        primitive, jaxpr_tracers, params, source_info)

  def default_process_primitive(self, primitive, tracers, params,
                                source_info=None):
    avals = [t.aval for t in tracers]
    # TODO(mattjj): 让 custom_lin 拥有可哈希的 params。
    # TODO(dougalm): 给原语添加一个属性，用来标记那些其 abstract_eval
    # 规则有副作用的原语。
    if (primitive.ref_allocating or
        primitive.name in ("custom_lin", "call_hi_primitive_linearized",
                           "call_hi_primitive")):
      out_avals, effs = primitive.abstract_eval(*avals, **params)
    else:
      try:
        out_avals, effs = _cached_abstract_eval(primitive, *avals, **params)
      except Exception:
        # TODO(phawkins): 在 JAX v0.7 发布 3 个月后移除此段代码。
        _verify_params_are_hashable(primitive, params)
        raise

    if isinstance(out_avals, (tuple, list)) != primitive.multiple_results:
      raise ValueError(f"{primitive}.abstract_eval() method should return "
                       f"a tuple or a list iff {primitive}.multiple_results.")
    out_avals = [out_avals] if not primitive.multiple_results else out_avals
    source_info = source_info or source_info_util.current()

    maybe_consts_out = try_constant_folding(primitive, tracers, params, out_avals)
    if maybe_consts_out is not None:
      eqn = None
      out_tracers = [self.new_const(c, source_info=source_info, aval=aval)
                     for c, aval in zip(maybe_consts_out, out_avals)]
    else:
      eqn, out_tracers = self.make_eqn(tracers, out_avals, primitive, params,
                                       effs, source_info=source_info)
    # 输入到输出的追踪器转发
    no_input_effects = not any(isinstance(e, effects.JaxprInputEffect) for e in effs)
    if eqn is not None and no_input_effects and primitive in forwarding_rules:
      in_fwd, eqn = forwarding_rules[primitive](eqn)
      for out_idx, in_idx in enumerate(in_fwd):
        if in_idx is not None:
          out_tracers[out_idx] = tracers[in_idx]

    if eqn is not None:
      self.frame.add_eqn(eqn)  # pyrefly: ignore[bad-argument-type]
    return out_tracers if primitive.multiple_results else out_tracers.pop()

  def process_custom_jvp_call(self, prim, fun: lu.WrappedFun,
                              jvp: lu.WrappedFun, tracers, /, *,
                              symbolic_zeros: bool):
    if self.requires_low:
      with core.set_current_trace(self):
        return fun.call_wrapped(*tracers)

    if config.eager_constant_folding.value and not any(isinstance(x, Tracer) for x in tracers):
      avals = tuple(core.typeof(x) for x in tracers)
      return prim.bind_with_trace(
        core.eval_trace, tracers, avals,
        dict(subfuns=(fun, jvp), symbolic_zeros=symbolic_zeros))
    source_info = source_info_util.current()
    to_jaxpr_tracer = partial(self.to_jaxpr_tracer, source_info=source_info)
    tracers = map(to_jaxpr_tracer, tracers)
    in_avals = [t.aval for t in tracers]
    in_tangent_avals = [t.to_tangent_aval() for t in in_avals]
    fun_jaxpr, out_avals, consts = trace_to_jaxpr_dynamic(fun, in_avals, lower=self.requires_low)
    self.frame.is_high |= fun_jaxpr.is_high
    closed_fun_jaxpr = convert_constvars_jaxpr(fun_jaxpr)

    @partial(lu.wrap_init, debug_info=jvp.debug_info)
    @_memoize
    def jvp_jaxpr_thunk(*in_zeros):
      for store in jvp.stores: store and store.reset()
      nz_tangent_avals, zero_avals = partition_list(in_zeros, in_tangent_avals)
      jvp_, out_zeros = _jvp_jaxpr_zeros(jvp, in_zeros, tuple(zero_avals))
      in_avals_ = (*in_avals, *nz_tangent_avals)
      jaxpr, _, out_consts = trace_to_jaxpr_dynamic(jvp_.with_unknown_names(),
                                                    in_avals_, lower=self.requires_low)
      return jaxpr, out_consts, out_zeros()

    const_tracers = map(to_jaxpr_tracer, consts)
    return self.emit_eqn(
        [*const_tracers, *tracers], out_avals, prim,
        dict(call_jaxpr=closed_fun_jaxpr,
             jvp_jaxpr_fun=jvp_jaxpr_thunk,
             num_consts=len(consts),
             symbolic_zeros=symbolic_zeros),
        core.positional_effects(closed_fun_jaxpr),
        source_info=source_info)

  def process_custom_vjp_call(self, prim: core.Primitive,
                              fun: lu.WrappedFun,
                              fwd: lu.WrappedFun, bwd: lu.WrappedFun, tracers, /, *,
                              out_trees: Callable[[], tuple[PyTreeDef, PyTreeDef, list[int | None]]],
                              symbolic_zeros: bool):
    if self.requires_low:
      with core.set_current_trace(self):
        return fun.call_wrapped(*tracers)

    if config.eager_constant_folding.value and not any(isinstance(x, Tracer) for x in tracers):
      avals = tuple(core.typeof(x) for x in tracers)
      return prim.bind_with_trace(
        core.eval_trace, tuple(tracers), avals,
        dict(subfuns=(fun, fwd, bwd), out_trees=out_trees,
             symbolic_zeros=symbolic_zeros))
    source_info = source_info_util.current()
    to_jaxpr_tracer = partial(self.to_jaxpr_tracer, source_info=source_info)
    tracers = map(to_jaxpr_tracer, tracers)
    in_avals = [t.aval for t in tracers]
    fun_jaxpr, out_avals, consts = trace_to_jaxpr_dynamic(fun.with_unknown_names(), in_avals, lower=self.requires_low)
    self.frame.is_high |= fun_jaxpr.is_high
    num_consts = len(consts)
    closed_fun_jaxpr = convert_constvars_jaxpr(fun_jaxpr)

    @partial(lu.wrap_init, debug_info=fwd.debug_info)
    @_memoize
    def fwd_jaxpr_from_zeros(*zeros):
      for store in fwd.stores: store and store.reset()
      fwd_ = _interleave_fun(fwd.with_unknown_names(), zeros)
      jaxpr, _, consts = trace_to_jaxpr_dynamic(fwd_, in_avals, lower=self.requires_low)
      return jaxpr, consts

    def out_trees_():
      out_tree, res_tree, input_fwds = out_trees()
      input_fwds = [f if f is None else f + num_consts for f in input_fwds]
      return out_tree, res_tree, input_fwds

    const_tracers = map(to_jaxpr_tracer, consts)
    return self.emit_eqn(
        [*const_tracers, *tracers], out_avals, prim,
        dict(call_jaxpr=closed_fun_jaxpr,
             fwd_jaxpr_thunk=fwd_jaxpr_from_zeros,
             num_consts=num_consts,
             bwd=bwd, out_trees=out_trees_,
             symbolic_zeros=symbolic_zeros),
        core.positional_effects(closed_fun_jaxpr),
        source_info=source_info)

  def to_jaxpr(self, out_tracers: Sequence[Tracer],
               debug_info: core.DebugInfo, source_info: SourceInfo):
    return self.frame.to_jaxpr(self, out_tracers, debug_info, source_info)


custom_staging_rules: dict[Primitive, Callable] = {}

@lu.transformation2
def _interleave_fun(f, every_others, *args, **kwargs):
  args_ = [x for pair in zip(args, every_others) for x in pair]
  return f(*args_, **kwargs)

# TODO: 考虑改名为 "lazy_thunk"
def _memoize(fn):
  cells = {}
  sentinel = object()
  def memoized(*args):
    out = cells.get(args, sentinel)
    if out is sentinel:
      with core.set_current_trace(None):
        out = cells[args] = fn(*args)
    return out
  return memoized

@lu.transformation_with_aux2
def _jvp_jaxpr_zeros(f, store, in_zeros, zero_avals, *primal_tangent_avals):
  in_primals, nz_in_tangents = split_list(primal_tangent_avals, [len(in_zeros)])
  symbolic_zeros = map(ad_util.SymbolicZero, zero_avals)
  tangents = merge_lists(in_zeros, nz_in_tangents, symbolic_zeros)
  out = f(*in_primals, *tangents)
  n, ragged = divmod(len(out), 2)
  assert not ragged
  out_primals, out_tangents = out[:n], out[n:]
  out_zeros = [type(t) is ad_util.SymbolicZero for t in out_tangents]
  out_nz_tangents, _ = partition_list(out_zeros, out_tangents)
  store.store(out_zeros)
  return [*out_primals, *out_nz_tangents]

callsites_with_tracing_cache_miss: set[str] = set()

def explain(keys, fun, in_avals, debug_info, *context, **_):
  func_filename = debug_info.func_filename
  if func_filename and not source_info_util.is_user_filename(func_filename):
   return

  msg: list[str] = []
  p = msg.append

  callsite = source_info_util.summarize(source_info_util.current())
  p(f"TRACING CACHE MISS at {callsite}:")

  src_info = ""
  if func_filename:
    src_info += f" defined at {func_filename}"
  if func_lineno := debug_info.func_lineno:
    src_info += f":{func_lineno}"
  func_name = debug_info.func_name

  # 我们之前究竟是否见过这个函数？
  keys = [key for fun_ref, *key in keys if fun_ref() is fun]
  if not keys:
    p(f"  never seen function:\n    {func_name} id={id(fun)}{src_info}")
    if callsite in callsites_with_tracing_cache_miss:
      p("  but seen another function defined on the same line; maybe the function is\n"
        "  being re-defined repeatedly, preventing caching?")
    else:
      callsites_with_tracing_cache_miss.add(callsite)
    return logger.log(logging.WARNING, "\n".join(msg))

  p(f"  for {func_name}{src_info}")

  key = (config.trace_context(), (in_avals, debug_info, *context), {})
  min_diff = min(diff_tracing_cache_keys(key, k) for k in keys)[-1]
  p('  all previously seen cache keys differ. For the closest previous key:')
  p('  ' + min_diff)
  return logger.log(logging.WARNING, "\n".join(msg))

def diff_tracing_cache_keys(new_key, old_key) -> tuple[int, int, str]:
  """解释两个追踪缓存键之间的差异。
  Returns:
    一个元组 (severity, num_diffs, explanation)，描述两个键之间的差异。
    severity 是一个整数，越小越好。
  """
  new_ctx, (new_tree, new_dbg, *_), () = new_key
  old_ctx, (old_tree, old_dbg, *_), () = old_key
  return (diff_tracing_ctx(new_ctx, old_ctx) or
          diff_trees(new_tree.tree, old_tree.tree) or
          diff_debug(new_dbg, old_dbg) or
          diff_types(new_dbg, new_tree.vals, old_tree.vals) or
          (4, 0, 'cache miss explanation unavailable'))

def diff_tracing_ctx(new_ctx, old_ctx) -> tuple[int, int, str] | None:
  if new_ctx == old_ctx: return None
  diffs: list[str] = []
  msg = "Tracing context doesn't match, e.g. due to config or context manager."
  if len(new_ctx) != len(old_ctx):
    num_diff = abs(len(new_ctx) - len(old_ctx))
    diffs.append("  * number of tracing context values differs: "
                f"now {len(new_ctx)} and before {len(old_ctx)}")
    return 0, num_diff, msg + "\n" + "\n".join(diffs)

  num_diff = sum(map(op.ne, new_ctx, old_ctx))

  context_names = config.trace_context_names()
  if len(context_names) != len(new_ctx):
    diffs.append("  * number of tracing context names differs: "
                  f"context_names {len(context_names)} vs "
                  f"context length {len(new_ctx)}")
    return 0, num_diff, msg + "\n" + "\n".join(diffs)
  for name, new, old in zip(context_names, new_ctx, old_ctx):
    if new != old:
      diffs.append(f"  * {name} differs: now {new} and before {old}")
  return 0, num_diff, msg + "\n" + "\n".join(diffs)

def diff_trees(new_tree, old_tree) -> tuple[int, int, str] | None:
  errs = tree_util.equality_errors_pytreedef(new_tree, old_tree)
  tree_diffs = []
  for path, thing1, thing2, explanation in errs:
    tree_diffs.append(
        f"  * at input path {tree_util.keystr(tuple(path))}, now {thing1} and "
        f"before {thing2}, so {explanation}")
  msg = 'different input pytree:\n' + '\n'.join(tree_diffs)
  if tree_diffs: return 1, len(tree_diffs), msg

def diff_debug(new_dbg, old_dbg) -> tuple[int, int, str] | None:
  msg = "Debug info doesn't match."
  num_diff = sum(map(op.ne, new_dbg, old_dbg))
  if num_diff: return 2, num_diff, msg

def diff_types(dbg, new_leaves, old_leaves) -> tuple[int, int, str] | None:
  if new_leaves == old_leaves: return
  diffs = []
  add_weak_type_hint = False
  for name, new_ty, old_ty in zip(dbg.arg_names, new_leaves, old_leaves):
    if new_ty != old_ty:
      new_str, old_str = new_ty.str_short(True), old_ty.str_short(True)
      if type(new_ty) is type(old_ty) is core.ShapedArray:
        if new_ty.sharding != old_ty.sharding:
          new_str, old_str = new_ty.str_short(True, True), old_ty.str_short(True, True)
        if new_ty.weak_type != old_ty.weak_type:
          add_weak_type_hint = True
          new_str += f'{{weak_type={new_ty.weak_type}}}'
          old_str += f'{{weak_type={old_ty.weak_type}}}'
      diffs.append(f"  * at {name}, now {new_str} and before {old_str}")
  msg = 'different input types:\n' + '\n'.join(diffs)
  if add_weak_type_hint:
    msg += 'https://docs.jax.dev/en/latest/type_promotion.html#weak-types'
  if diffs: return 3, len(diffs), msg


def _lower_debug_info(hi_jaxpr):
  debug_info = hi_jaxpr.debug_info
  if debug_info.arg_names is not None:
    lo_arg_names = tuple(
        name for aval, name in zip(hi_jaxpr.in_avals, debug_info.arg_names)
        for _ in aval.lo_ty())
    debug_info = debug_info._replace(arg_names=lo_arg_names)
  if debug_info.result_paths is not None:
    lo_result_paths = tuple(
    path for aval, path in zip(hi_jaxpr.out_avals, debug_info.result_paths)
        for _ in aval.lo_ty())
    debug_info = debug_info._replace(result_paths=lo_result_paths)
  return debug_info

def trace_to_jaxpr_nocache(
    fun: Callable,
    in_avals: ft.FlatTree,  # (args, kwargs) 对
    debug_info: core.DebugInfo,
    # TODO: 我们干脆为此写一个 `trace_to_jaxpr_ft` 函数吧
    fun_takes_flat_tree_arg=False,
    fun_returns_flat_tree=False,
    requires_low=False,
) -> tuple[Jaxpr, ft.FlatTree]:
  if config.no_tracing.value:
    raise RuntimeError(f"re-tracing function {fun} for "
                       "`jit`, but 'no_tracing' is set")
  test_event("trace_to_jaxpr")
  config.enable_checks.value and debug_info.assert_arg_names(len(in_avals))
  parent_trace = core.trace_ctx.trace
  trace = DynamicJaxprTrace(debug_info, parent_trace=parent_trace,
                            lower=requires_low)
  # 名称栈与回溯作用域被重置，因为 jaxpr 方程上的元数据应当以
  # 外层 jaxpr 为根，而不应包含任何来自调用点的上下文。否则当
  # 我们（例如）进行内联时，一个调用者的元数据会渗透进
  # 另一个调用者的元数据中。
  with (core.ensure_no_leaks(trace), source_info_util.reset_name_stack(),
        TracebackScope()):
    source_info = source_info_util.current()
    if requires_low:
      if debug_info.arg_names is not None:
        debug_info = debug_info._replace(arg_names=tuple(
            name for aval, name in zip(in_avals, debug_info.arg_names)
            for _ in aval.lo_ty()))
      def new_arg(aval):
        lo_tracers = [trace.new_arg(lo_aval, source_info=source_info) for lo_aval in aval.lo_ty()]  # noqa: F821
        return aval.raise_val(*lo_tracers)
      in_tracers = in_avals.map(new_arg)
    else:
      in_tracers = in_avals.map(partial(trace.new_arg, source_info=source_info))

    with core.set_current_trace(trace):
      if fun_takes_flat_tree_arg:
        args_ft, kwargs_ft = in_tracers.unpack()
        assert kwargs_ft.unflatten() == {}  # TODO: 处理 kwargs
        kwargs = {}
        args = args_ft.unpack()
        del args_ft
      else:
        args, kwargs = in_tracers.unflatten()
      ans_pytree = fun(*args, **kwargs)
      if fun_returns_flat_tree:
        # TODO(dougalm): 让结果路径变为可选
        ans = ans_pytree
        debug_info = debug_info.set_result_paths([''] * len(ans))
      else:
        debug_info = debug_info.set_result_paths(ans_pytree)
        ans = ft.flatten(ans_pytree)
      del ans_pytree, args, kwargs

    _check_returned_jaxtypes(debug_info, list(ans))
    ans = ans.map(dtypes.canonicalize_value)
    out_avals = ans.map(typeof)
    if requires_low:
      flat_out_tracers = [trace.to_jaxpr_tracer(x, source_info=source_info)
                          for aval, hi_val in zip(out_avals, ans)
                          for x in aval.lower_val(hi_val)]
      debug_info = debug_info.resolve_result_paths()
      if debug_info.result_paths is not None:
        debug_info = debug_info._replace(result_paths=tuple(
            path for aval, path in zip(out_avals, debug_info.result_paths)
            for _ in aval.lo_ty()))
    else:
      flat_out_tracers = [trace.to_jaxpr_tracer(x, source_info=source_info)
                          for x in ans]

    _check_no_returned_refs(debug_info, list(flat_out_tracers))
    jaxpr, consts = trace.frame.to_jaxpr(trace, list(flat_out_tracers), debug_info,
                                         source_info)
    del trace, fun, in_tracers, flat_out_tracers, ans
  config.enable_checks.value and core.check_jaxpr(jaxpr)
  return jaxpr.with_consts(consts), out_avals


trace_to_jaxpr = weakref_lru_cache(maxsize=None, explain=explain)(
    trace_to_jaxpr_nocache)


# TODO(dougalm): 移除它，改用 `trace_to_jaxpr`
@profiler.annotate_function
def trace_to_jaxpr_dynamic(
    fun: lu.WrappedFun, in_avals: Sequence[AbstractValue],
    *, keep_inputs: list[bool] | None = None, lower: bool = False,
    auto_dce: bool = False) -> tuple[Jaxpr, list[AbstractValue], list[Any]]:
  config.enable_checks.value and fun.debug_info.assert_arg_names(len(in_avals))
  keep_inputs = [True] * len(in_avals) if keep_inputs is None else keep_inputs
  parent_trace = core.trace_ctx.trace
  trace = DynamicJaxprTrace(fun.debug_info, parent_trace=parent_trace,
                            lower=lower, auto_dce=auto_dce)
  # 名称栈与回溯作用域被重置，因为 jaxpr 方程上的元数据应当以
  # 外层 jaxpr 为根，而不应包含任何来自调用点的上下文。否则当
  # 我们（例如）进行内联时，一个调用者的元数据会渗透进
  # 另一个调用者的元数据中。
  with (core.ensure_no_leaks(trace), source_info_util.reset_name_stack(),
        TracebackScope()):
    source_info = source_info_util.current()
    in_tracers = map(partial(trace.new_arg, source_info=source_info), in_avals)
    in_tracers = [t for t, keep in zip(in_tracers, keep_inputs) if keep]
    with core.set_current_trace(trace):
      ans = fun.call_wrapped(*in_tracers)
    _check_returned_jaxtypes(fun.debug_info, ans)
    ans = map(dtypes.canonicalize_value, ans)
    out_tracers = map(partial(trace.to_jaxpr_tracer, source_info=source_info), ans)
    _check_no_returned_refs(fun.debug_info, out_tracers)
    jaxpr, consts = trace.frame.to_jaxpr(trace, out_tracers, fun.debug_info,
                                         source_info)
    del trace, fun, in_tracers, out_tracers, ans
  config.enable_checks.value and core.check_jaxpr(jaxpr)
  return jaxpr, [v.aval for v in jaxpr.outvars], consts

def _check_returned_jaxtypes(dbg, out_tracers):
  for i, x in enumerate(out_tracers):
    try: typeof(x)
    except TypeError:
      if (dbg and len(paths := dbg.resolve_result_paths().result_paths) > i and
          (p := paths[i].removeprefix('result'))):
        extra = f' at output component {p}'
      else:
        extra = ''
      raise TypeError(
      f"function {dbg.func_src_info} traced for {dbg.traced_for} returned a "
      f"value of type {type(x)}{extra}, which is not a valid JAX type") from None

def _check_no_returned_refs(
    dbg: core.DebugInfo,
    out_tracers: Sequence[DynamicJaxprTracer]
) -> None:
  if not config.mutable_array_checks.value: return
  for i, t in enumerate(out_tracers):
    a = t.aval
    if isinstance(a, AbstractRef):
      result_paths = dbg.resolve_result_paths().safe_result_paths(len(out_tracers))
      if list(result_paths) == ["result"]: result_paths = [""]  # TODO(mattjj): 在被调用者中修复
      loc = result_paths[i] and f' at output tree path {result_paths[i]}'
      frame = t._trace.frame
      v = t.val
      eqns = frame.get_eqns()
      # TODO(dougalm): 可以用更高效的方式实现
      eqn = next((e for e in eqns if v in e.outvars), None)
      if eqn:
        assert eqn.primitive in (core.ref_p, core.empty_ref_p)
        origin_info = ('\n\nThe returned mutable array was created on line '
                       f'{source_info_util.summarize(eqn.source_info)}.')
      elif v in frame.invars:
        assert isinstance(v, Var)
        arg_name = dbg.safe_arg_names(len(frame.invars))[frame.invars.index(v)]
        origin_info = ('\n\nThe returned mutable array was passed in as the '
                       f'argument {arg_name}.')
      else:
        origin_info = ''
      raise ValueError(
          f"function {dbg.func_src_info} traced for {dbg.traced_for} returned "
          f"a mutable array reference of type {a.str_short()}{loc}, but "
          f"mutable array references cannot be returned.{origin_info}")

class TracerAsName:
  ref: Any
  def __init__(self, tracer):
    self.ref = core.get_referent(tracer)
  def __eq__(self, other):
    return isinstance(other, TracerAsName) and self.ref is other.ref
  def __hash__(self):
    return id(self.ref)

Const = Any
Val = Any


def inline_jaxpr_into_trace(
    trace: DynamicJaxprTrace, src: SourceInfo, jaxpr: Jaxpr,
    consts: Sequence[Any], *arg_tracers: DynamicJaxprTracer) -> list[Any]:
  # 该函数在概念上等同于直接调用 eval_jaxpr，
  const_tracers = map(partial(trace.new_const, source_info=src), consts)
  env: dict[Var, DynamicJaxprTracer] = dict(
      zip([*jaxpr.constvars, *jaxpr.invars],
          [*const_tracers, *arg_tracers]))

  def inline_atom(src_, x):
    if isinstance(x, Literal):
      return DynamicJaxprTracer(trace, x.aval, x, src_)
    else:
      return env[x]

  for eqn in jaxpr.eqns:
    src_ = (src if not eqn.source_info.name_stack else
            src.replace(name_stack=src.name_stack + eqn.source_info.name_stack))
    in_tracers = map(partial(inline_atom, src_), eqn.invars)
    out_avals = [v.aval for v in eqn.outvars]

    maybe_consts = try_constant_folding(eqn.primitive, in_tracers, eqn.params, out_avals)
    if maybe_consts is not None:
      out_tracers = [trace.new_const(c, source_info=src_, aval=aval)
                     for c, aval in zip(maybe_consts, out_avals)]
    else:
      effs = {e.replace(env[e.input].val)
              if isinstance(e, effects.JaxprInputEffect) else e
              for e in eqn.effects} if eqn.effects else eqn.effects
      out_tracers = trace.emit_eqn(in_tracers, out_avals, eqn.primitive,
                                   eqn.params, effs, src_, eqn.ctx)
    foreach(env.setdefault, eqn.outvars, out_tracers)

  return map(partial(inline_atom, src), jaxpr.outvars)


def try_constant_folding(primitive, tracers, params, out_avals):
  if primitive in const_fold_rules:
    consts_in = [t.get_const() for t in tracers]
    if any(c is not None for c in consts_in):
      return const_fold_rules[primitive](consts_in, params, out_avals)
  return None

@weakref_lru_cache
def lower_jaxpr2(hi_jaxpr) -> Jaxpr:
  in_avals = ft.flatten(([a.lo_ty() for a in hi_jaxpr.in_avals], {}))
  lo_jaxpr, _ = lower_jaxpr(hi_jaxpr, in_avals)
  return lo_jaxpr

@weakref_lru_cache
def lower_jaxpr(hi_jaxpr: Jaxpr, lo_avals) -> tuple[Jaxpr, ft.FlatTree]:
  env: dict[Var, DynamicJaxprTracer | HTLV] = {}  # noqa # type:ignore

  parent_trace = core.trace_ctx.trace
  trace = DynamicJaxprTrace(hi_jaxpr.debug_info.with_unknown_names(),
                            parent_trace=parent_trace, lower=True)

  def read(src, x):
    if isinstance(x, Literal):
      return x.val
    elif (htlv := env.get(x)) is not None:  # noqa
      return htlv
    else:
      assert not x.aval.is_high
      return trace.var_to_tracer(x, src)  # noqa

  def write(v, x):
    if v.aval.is_high:
      env[v] = x  # noqa: F821
    elif isinstance(x, DynamicJaxprTracer) and x._trace is trace:  # noqa: F821
      env[v] = x  # noqa: F821
    else:
      env[v] = lift_lo_const(v, x)  # noqa: F821

  def lift_lo_const(v, c) -> DynamicJaxprTracer:
    tracer = DynamicJaxprTracer(trace, v.aval, v, src)  # noqa: F821
    trace.frame.constid_to_tracer[id(c)] = tracer  # noqa: F821
    trace.frame.constvar_to_val[v] = c  # noqa: F821
    return tracer

  with (core.ensure_no_leaks(trace), source_info_util.reset_name_stack(),
        TracebackScope()):
    src = source_info_util.current()
    invals = outs = maybe_invals = new_invars = xs = t = None

    lo_avals_lol, () = lo_avals.unflatten()
    for v, xs in zip(hi_jaxpr.invars, lo_avals_lol):
      if v.aval.is_high:
        xs = [trace.new_arg(x, source_info=src) for x in xs]
        env[v] = v.aval.raise_val(*xs)  # type: ignore
      else:
        trace.frame.invars.append(v)

    for v, c in zip(hi_jaxpr.constvars, hi_jaxpr.consts):
      if v.aval.is_high:
        env[v] = c  # 视为 HTLV
      else:
        env[v] = lift_lo_const(v, c)

    with core.set_current_trace(trace):
      eqns = trace.frame.tracing_eqns
      for eqn in hi_jaxpr.eqns:
        maybe_invals = [env.get(x) if isinstance(x, Var) else None for x in eqn.invars]
        hi = eqn.primitive.is_high(*[v.aval for v in eqn.invars], **eqn.params)
        if all(x is None for x in maybe_invals) and not hi:
          eqns.append(eqn)  # type: ignore
        elif not hi:
          new_invars = [x if isinstance(x, Literal) else
                        t.val if (t := env.get(x)) is not None else x
                        for x in eqn.invars]
          eqns.append(eqn.replace(invars=new_invars))
        else:
          invals = map(partial(read, eqn.source_info), eqn.invars)
          name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
          with (source_info_util.user_context(eqn.source_info.traceback, name_stack=name_stack),
                eqn.ctx.manager):
            outs = eqn.primitive.to_lojax(*invals, **eqn.params)
          foreach(write, eqn.outvars, outs if eqn.primitive.multiple_results else [outs])

    tracer = partial(trace.to_jaxpr_tracer, source_info=src)
    out_tracers = [dtypes.canonicalize_value(read(src, x)) for x in hi_jaxpr.outvars]
    out_tracers = [v.aval.lower_val2(hi_val).map(tracer)
                  for v, hi_val in zip(hi_jaxpr.outvars, out_tracers)]
    out_tracers = ft.pack(tuple(out_tracers))
    out_avals = out_tracers.map(typeof)
    dbg = _lower_debug_info(hi_jaxpr)
    jaxpr, consts = trace.frame.to_jaxpr(trace, list(out_tracers), dbg, src)
    del trace, env, out_tracers, tracer, read, outs, invals, eqns
    del maybe_invals, new_invars, xs, t

  config.enable_checks.value and core.check_jaxpr(jaxpr)
  assert not any(v.aval.is_high for v in it.chain(jaxpr.constvars, jaxpr.invars))
  return jaxpr.with_consts(consts), out_avals

@weakref_lru_cache
def lower_jaxpr_reference(hi_jaxpr: Jaxpr, lo_avals) -> tuple[Jaxpr, ft.FlatTree]:
  """lower_jaxpr 的参考实现，用于测试与调试。

  可通过 `pe.lower_jaxpr = pe.lower_jaxpr_reference` 把它替换进去；所有调用点
  都会在调用时查找该名字。
  """
  dbg = _lower_debug_info(hi_jaxpr)
  return trace_to_jaxpr(partial(_lower_traceable, hi_jaxpr), lo_avals, dbg,
                        requires_low=True, fun_returns_flat_tree=True)

def _lower_traceable(jaxpr, *lo_args):
  hi_args = [a.raise_val(*xs) for a, xs in zip(jaxpr.in_avals, lo_args)]
  hi_outs = core.jaxpr_as_fun(jaxpr)(*hi_args)
  lo_outs = [a.lower_val2(y) for a, y in zip(jaxpr.out_avals, hi_outs)]
  return ft.pack(tuple(lo_outs))

# 遗留的 hijax 辅助函数
def raise_lo_outs(hi_avals, lo_outs):
  lo_outs_ = iter(lo_outs)
  hi_outs = [t.raise_val(*it.islice(lo_outs_, len(t.lo_ty()))) for t in hi_avals]
  assert next(lo_outs_, None) is None
  return hi_outs


eval_jaxpr_p = core.eval_jaxpr_p

dce_rules[eval_jaxpr_p] = dce_jaxpr_closed_call_rule
dce_jaxpr_call_rule = dce_jaxpr_closed_call_rule  # 供下游用户使用的别名

def _eval_jaxpr_partial_eval(prim, trace, *in_tracers, call_jaxpr, **params):
  in_pvals = [t.pval for t in in_tracers]
  unknown_ins = [not pv.is_known() for pv in in_pvals]
  known_jaxpr, unknown_jaxpr, unknown_outs, res_avals, in_fwd_res = \
      partial_eval_jaxpr_nounits_fwd(call_jaxpr, unknown_ins, instantiate=False)
  consts = [pv.get_known() for pv in in_pvals if pv.is_known()]
  all_known_outs = prim.bind(*consts, call_jaxpr=known_jaxpr, **params)
  known_outs, res = split_list(all_known_outs, [len(all_known_outs) - len(res_avals)])
  res_ = iter(res)
  res = [next(res_) if f is None else [*call_jaxpr.consts, *consts][f]
         for f in in_fwd_res]
  assert next(res_, sentinel := object()) is sentinel
  res_tracers = map(trace.new_instantiated_const, res)
  unk_tracers_in = [t for t in in_tracers if not t.pval.is_known()]
  unk_tracers_out = [JaxprTracer(trace, PartialVal.unknown(aval), None)
                     for aval in unknown_jaxpr.out_avals]
  eqn = new_eqn_recipe(trace, [*res_tracers, *unk_tracers_in], unk_tracers_out,
                       prim, dict(params, call_jaxpr=unknown_jaxpr),
                       core.positional_effects(unknown_jaxpr),
                       source_info_util.current())
  for t in unk_tracers_out: t.recipe = eqn
  if effects.partial_eval_kept_effects.filter_in(unknown_jaxpr.effects):
    trace.effect_handles.append(EffectHandle([*unk_tracers_in, *res_tracers], eqn))  # type: ignore
  return merge_lists(unknown_outs, known_outs, unk_tracers_out)
custom_partial_eval_rules[eval_jaxpr_p] = partial(_eval_jaxpr_partial_eval, eval_jaxpr_p)

partial_eval_jaxpr_custom_rules[eval_jaxpr_p] = \
    partial(closed_call_partial_eval_custom_rule, 'call_jaxpr',
            lambda _, __, ___, ____, _____, ______, x, y: (x, y))

def _lower_and_eval(prim, jaxpr: Jaxpr, args: Sequence[Any], **params):
  lo_jaxpr = lower_jaxpr2(jaxpr)
  lo_args = [lo_val for aval, x in zip(jaxpr.in_avals, args)
             for lo_val in aval.lower_val(x)]
  lo_outs = prim.bind(*lo_args, call_jaxpr=lo_jaxpr, **params)
  lo_outs_ = iter(lo_outs)
  hi_outs = [t.raise_val(*it.islice(lo_outs_, len(t.lo_ty())))
             for t in jaxpr.out_avals]
  assert next(lo_outs_, None) is None
  return hi_outs

def _eval_jaxpr_to_lojax(prim, *hi_args, call_jaxpr: Jaxpr, **params):
  return _lower_and_eval(prim, call_jaxpr, hi_args, **params)
eval_jaxpr_p.to_lojax = partial(_eval_jaxpr_to_lojax, eval_jaxpr_p)
