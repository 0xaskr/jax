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

# 文件职责：实现把 XLA 元数据（如 scheduling_group 等）附加到编译产物的内部机制。
# 本模块为 JAX 内部模块 `jax._src.xla_metadata`，对外由
# `jax.experimental.xla_metadata` 暴露 `set_xla_metadata`、`xla_metadata_call`、
# `xla_metadata_call2` 等接口。核心思路是：元数据先随追踪上下文
# （`XlaMetadataContextManager`）传播，或在降级时由原语 `xla_metadata_value`、
# `xla_metadata_call` 写成被调用计算与算子上的 `mhlo.frontend_attributes`，供 XLA 在后续优化中使用。

from collections.abc import Mapping
import contextlib
from functools import partial, wraps
from typing import Any

from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import flattree as ft
from jax._src import tree_util
from jax._src import xla_metadata_lib
from jax._src.api_util import debug_info
from jax._src.interpreters import ad, batching, mlir, partial_eval as pe
from jax._src.lib import _jax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten
from jax._src.util import (safe_map, safe_zip, weakref_lru_cache, unzip2,
                           split_list, subs_list)

config_ext = _jax.config

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


class _XlaMetadataWrapper:
  """包装器类，让 XlaMetadataContextManager 可以当作装饰器使用。

  当 XlaMetadataContextManager 被用作函数 `f` 的装饰器时，它会返回本类的
  一个实例。该包装器保证调用 `f` 时运行在元数据上下文内。它还通过
  `__getattr__` 转发对 `f` 的属性访问；若 `f` 的某个属性可调用（例如
  已 jit 函数的 `.lower()` 方法），它也会包装该属性，使其被调用时同样
  运行在元数据上下文内。这样被装饰的函数就能与 `jax.jit` 等 JAX 变换
  无缝配合使用。
  """

  def __init__(self, f, ctx):
    self._f = f
    self._ctx = ctx
    wraps(f)(self)

  def __call__(self, *args, **kwargs):
    with self._ctx:
      return self._f(*args, **kwargs)

  def __getattr__(self, name):
    attr = getattr(self._f, name)
    if not callable(attr):
      return attr

    @wraps(attr)
    def wrapper(*args, **kwargs):
      with self._ctx:
        return attr(*args, **kwargs)

    return wrapper


class XlaMetadataContextManager:
  __slots__ = ["prev", "updates"]

  def __init__(self, updates):
    self.updates = updates

  def __enter__(self):
    if not self.updates:
      return

    self.prev = config.xla_metadata_context_manager.get_local()
    config.xla_metadata_context_manager.set_local(
        xla_metadata_lib.update_metadata(self.prev, self.updates)
    )

  def __exit__(self, exc_type, exc_value, traceback):
    if not self.updates:
      return
    config.xla_metadata_context_manager.set_local(self.prev)

  def __call__(self, f):
    return _XlaMetadataWrapper(f, self)


@contextlib.contextmanager
def clear_xla_metadata():
  """内部上下文管理器，用于临时清除环境中的 XLA 元数据。"""
  prev = config.xla_metadata_context_manager.swap_local(None)
  try:
    yield
  finally:
    config.xla_metadata_context_manager.set_local(prev)


def set_xla_metadata(x=None, **kwargs):
  if x is None:
    return XlaMetadataContextManager(kwargs)
  else:
    hashable_metadata = tuple(sorted(kwargs.items()))
    return tree_util.tree_map(
        lambda v: xla_metadata_value_p.bind(
            v, xla_metadata_kvs=hashable_metadata
        ),
        x,
    )


# `xla_metadata_value_p` 是一个恒等原语，用于把 frontend_attributes
# 附加到产生该值的（父/属主）算子上。
xla_metadata_value_p = core.Primitive("xla_metadata_value")
xla_metadata_value_p.def_impl(
    partial(dispatch.apply_primitive, xla_metadata_value_p)
)
xla_metadata_value_p.def_abstract_eval(lambda aval, *, xla_metadata_kvs: aval)
batching.defvectorized(xla_metadata_value_p)
# TODO(nbasile): 实现用元数据标记梯度算子。
ad.deflinear2(xla_metadata_value_p, lambda ct, _, **kwargs: (ct,))


def _xla_metadata_value_lowering_rule(
    ctx: mlir.LoweringRuleContext, val: ir.Value, *, xla_metadata_kvs
):
  xla_metadata = dict(xla_metadata_kvs)
  op_to_attach_metadata = _target_op_to_attach_metadata(val)
  if op_to_attach_metadata is not None:
    _attach_xla_metadata_to_op(xla_metadata, op_to_attach_metadata)
  return [val]


# 若保留 `cacheable=True`，在降级规则中 `val.owner` 会变成被缓存的 `FuncOp`。
# 而 `FuncOp` 的 owner 是 Block，无法给 Block 打标签。
mlir.register_lowering(
    xla_metadata_value_p, _xla_metadata_value_lowering_rule, cacheable=False
)


def _target_op_to_attach_metadata(value_mlir: ir.Value) -> ir.Operation | None:
  op = value_mlir.owner
  if op is None or isinstance(op, ir.Block):
    return None
  return op.operation


def _attach_xla_metadata_to_op(
    xla_metadata: dict[str, Any], op: ir.Operation
) -> None:
  if xla_metadata:
    ctx_attributes, existing_attributes = {}, {}
    for k, v in xla_metadata.items():
      v_str = str(v).lower() if isinstance(v, bool) else str(v)
      ctx_attributes[k] = ir.StringAttr.get(v_str)
    # 与已有的 mhlo.frontend_attributes 合并
    for attr in op.attributes:
      if attr == "mhlo.frontend_attributes":
        for a in ir.DictAttr(op.attributes[attr]):
          existing_attributes[a.name] = a.attr
    op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
        ctx_attributes | existing_attributes
    )


def xla_metadata_call(f=None, /, **meta):
  """包装一个函数，使其降级为带有 XLA 元数据标记的 call 算子。

  这是 :func:`xla_metadata_call2` 的语法糖，元数据以关键字参数传入并使用
  默认选项。被包装的函数会被暂存为一份独立的计算，并通过调用一个 call
  算子来调用，该 call 算子带有以 ``frontend_attributes`` 形式给出的
  元数据。XLA 在将该 call 内联时，会把这些属性传播到 call 内部的各个
  算子上。

  与 ``set_xla_metadata`` 上下文管理器不同，本函数不会扰动追踪上下文，
  因此不会导致已 jit 的函数被重新追踪或重新编译。元数据也会随计算一起
  穿过各种变换：在 ``jax.grad`` 下，由该函数导出的前向计算与反向计算
  都会携带该元数据。若希望让反向计算不带标记，
  或给它打上与其他计算不同的标记，请使用
  :func:`xla_metadata_call2`，并通过它的 ``ad_metadata``
  选项来控制具体做法。

  Args:
    f: 要包装的函数。若未提供，则返回一个装饰器。
    **meta: 要附加的元数据，以关键字参数给出。取值可以是字符串、
      布尔、整数或浮点数；它们会以字符串形式附加（布尔值渲染为
      ``"true"``/``"false"``）。

  Returns:
    已应用元数据的 ``f`` 的包装版本，或者一个装饰器。

  Example:

    >>> import jax, jax.numpy as jnp
    >>> from jax.experimental.xla_metadata import xla_metadata_call
    >>> @xla_metadata_call(tag="my_block")
    ... def f(x):
    ...   return jnp.sin(x) * jnp.cos(x)
  """
  if f is None:
    return lambda g: _xla_metadata_call(g, meta, 'same')
  return _xla_metadata_call(f, meta, 'same')


def xla_metadata_call2(f=None, /, metadata=None, *, ad_metadata='same'):
  """与 :func:`xla_metadata_call` 类似，但元数据以字典给出，并多了若干选项。

  Args:
    f: 要包装的函数。若未提供，则返回一个装饰器。
    metadata: 要作为 ``frontend_attributes`` 附加到 call 算子上的元数据
      字典。取值可以是字符串、布尔、整数或浮点数；这些值都会以字符串
      形式附加到算子上（其中布尔值渲染为
      ``"true"``/``"false"``）。
    ad_metadata: 自动微分在该函数前向计算之外导出的计算所用的元数据，
      这里的计算指的是线性化（切向量）计算，
      以及转置（反向）计算。
      默认值 ``'same'`` 表示也给它们附加 ``metadata``；``'drop'``
      表示不给它们打标记；传入字典则改为附加该字典。在前向模式
      ``jax.jvp`` 下，原始计算与切向量计算被融合暂存，
      因此它们无论如何都会保留 ``metadata``。

  Returns:
    已应用元数据的 ``f`` 的包装版本，或者一个装饰器。

  Example:

    >>> import jax, jax.numpy as jnp
    >>> from jax.experimental.xla_metadata import xla_metadata_call2
    >>> @xla_metadata_call2({"scheduling_group": "l3"},
    ...                     ad_metadata={"scheduling_group": "l3_bwd"})
    ... def layer3(x):
    ...   return jnp.sin(x) * jnp.cos(x)
  """
  if isinstance(f, Mapping) and metadata is None:
    f, metadata = None, f
  if f is not None and not callable(f):
    raise TypeError(f"expected a callable to wrap, got {f!r}")
  if f is None:
    return lambda g: _xla_metadata_call(g, metadata, ad_metadata)
  return _xla_metadata_call(f, metadata, ad_metadata)


def _canonicalize_metadata(meta):
  canonical = {}
  for k, v in meta.items():
    if isinstance(v, bool):
      canonical[k] = str(v).lower()
    elif isinstance(v, (str, int, float)):
      canonical[k] = str(v)
    else:
      raise TypeError(
          "xla_metadata_call metadata values must be str, bool, int, or "
          f"float, got {type(v)} for key {k!r}")
  return tuple(sorted(canonical.items()))

def _canonicalize_ad_metadata(ad_metadata):
  if ad_metadata == 'same':
    return 'same'
  elif ad_metadata == 'drop':
    return ()
  elif isinstance(ad_metadata, Mapping):
    return _canonicalize_metadata(ad_metadata)
  else:
    raise TypeError(
        "ad_metadata must be 'same', 'drop', or a dict of metadata, got "
        f"{ad_metadata!r}")

# TODO(yashkatariya): 想办法与 compute_on_p、fused_p 复用代码
def _xla_metadata_call(fun, metadata, ad_metadata):
  if metadata is not None and not isinstance(metadata, Mapping):
    raise TypeError(f"metadata must be a dict, got {metadata!r}")
  metadata_kvs = _canonicalize_metadata(metadata or {})
  ad_metadata_kvs = _canonicalize_ad_metadata(ad_metadata)
  @wraps(fun)
  def wrapped(*args, **kwargs):
    dbg = debug_info('xla_metadata_call', fun, args, kwargs)
    args_ft = ft.flatten((args, kwargs))
    in_avals = args_ft.map(core.shaped_abstractify)
    jaxpr, out_avals = pe.trace_to_jaxpr(fun, in_avals, dbg)
    if any(isinstance(c, core.Tracer) for c in jaxpr.consts):
      jaxpr, consts = pe.separate_consts(jaxpr)
    else:
      consts = []
    outs_flat = xla_metadata_call_p.bind(*consts, *args_ft.vals, jaxpr=jaxpr,
                                         xla_metadata=metadata_kvs,
                                         ad_metadata=ad_metadata_kvs)
    return tree_unflatten(out_avals.tree, outs_flat)
  return wrapped

xla_metadata_call_p = core.Primitive('xla_metadata_call')
xla_metadata_call_p.multiple_results = True
dispatch.simple_impl(xla_metadata_call_p)


def _xla_metadata_call_abstract_eval(*in_avals, jaxpr, xla_metadata,
                                     ad_metadata):
  return jaxpr.out_avals
xla_metadata_call_p.def_abstract_eval(_xla_metadata_call_abstract_eval)


def _resolve_ad_metadata(xla_metadata, ad_metadata):
  return xla_metadata if ad_metadata == 'same' else ad_metadata


def _xla_metadata_call_lowering(ctx, *args, jaxpr, xla_metadata, ad_metadata):
  const_args_and_avals = core.jaxpr_const_args(jaxpr)
  const_args, const_avals = unzip2(const_args_and_avals)
  in_avals = (*const_avals, *jaxpr.in_avals)
  func_op, output_types, effects = mlir.lower_called_computation(
      "xla_metadata_call", jaxpr, ctx.module_context, len(const_args), in_avals,
      ctx.avals_out, ctx.tokens_in)

  symbol_name = func_op.name.value
  flat_output_types, treedef = mlir.ir_tree_registry.flatten(output_types)
  tokens = [ctx.tokens_in.get(eff) for eff in effects]
  hoisted_const_values, _ = mlir.ir_tree_registry.flatten([
      mlir.ir_constants(c, const_lowering=ctx.const_lowering, aval=aval)
      for c, aval in const_args_and_avals
  ])
  args = (*ctx.dim_var_values, *tokens, *hoisted_const_values, *args)
  flat_args, _ = mlir.ir_tree_registry.flatten(args)
  call = func_dialect.CallOp(
      flat_output_types, ir.FlatSymbolRefAttr.get(symbol_name),
      flat_args)
  if xla_metadata:
    call.operation.attributes['mhlo.frontend_attributes'] = ir.DictAttr.get(
        {k: ir.StringAttr.get(v) for k, v in xla_metadata})
  out_nodes = treedef.unflatten(call.results)
  tokens, out_nodes = split_list(out_nodes, [len(effects)])
  tokens_out = ctx.tokens_in.update_tokens(mlir.TokenSet(dict(zip(effects, tokens))))
  ctx.set_tokens_out(tokens_out)
  return out_nodes
mlir.register_lowering(xla_metadata_call_p, _xla_metadata_call_lowering)


def _xla_metadata_call_batcher(axis_data, vals_in, dims_in, *, jaxpr,
                               xla_metadata, ad_metadata):
  batched_jaxpr, dims_out = batching.batch_jaxpr2(jaxpr, axis_data, dims_in)
  outs = xla_metadata_call_p.bind(*vals_in, jaxpr=batched_jaxpr,
                                  xla_metadata=xla_metadata,
                                  ad_metadata=ad_metadata)
  return outs, dims_out
batching.fancy_primitive_batchers[xla_metadata_call_p] = _xla_metadata_call_batcher


def _xla_metadata_call_jvp(primals, tangents, *, jaxpr, xla_metadata,
                           ad_metadata):
  # jvp 的 jaxpr 融合了原始算子与切向量算子，因此保留原始计算的元数据；
  # ad_metadata 仍然决定后续的线性化/转置行为。
  nzs = [not isinstance(t, ad.Zero) for t in tangents]
  jaxpr_jvp, out_nzs = ad.jvp_jaxpr(jaxpr, nzs, False)
  nz_tangents = [t for t in tangents if not isinstance(t, ad.Zero)]
  outs = xla_metadata_call_p.bind(*primals, *nz_tangents, jaxpr=jaxpr_jvp,
                                  xla_metadata=xla_metadata,
                                  ad_metadata=ad_metadata)
  primals_out, nz_tangents_out = outs[:len(out_nzs)], outs[len(out_nzs):]
  nz_outs = iter(nz_tangents_out)
  tangents_out = [next(nz_outs) if nz else ad.Zero(aval.to_tangent_aval())
                  for aval, nz in zip(jaxpr.out_avals, out_nzs)]
  assert next(nz_outs, None) is None
  return primals_out, tangents_out
ad.primitive_jvps[xla_metadata_call_p] = _xla_metadata_call_jvp


def _xla_metadata_call_lin(is_vjp, nzs, *primals, jaxpr, xla_metadata,
                           ad_metadata):
  primal_jaxpr, out_tree, nzs_out, in_fwd_res, tangent_jaxpr = \
      ad.linearize_jaxpr(jaxpr, nzs, is_vjp=is_vjp)
  _, ures_avals, sres_avals = out_tree.unpack()
  num_residuals_out = len(ures_avals) + len(sres_avals)
  num_primals_out = len(primal_jaxpr.out_avals) - num_residuals_out

  _, in_fwd_ures, in_fwd_sres = split_list(
      pe._jaxpr_forwarding(primal_jaxpr), [num_primals_out, len(ures_avals)])
  assert all(f is None for f in in_fwd_ures)
  in_fwd = [None] * (num_primals_out + len(ures_avals)) + in_fwd_sres
  primal_jaxpr = pe.prune_closed_jaxpr_outputs(
      primal_jaxpr, [f is None for f in in_fwd])
  primal_jaxpr, out_fwd = pe.dedup_jaxpr_outputs(primal_jaxpr, num_primals_out)

  tangent_avals_out = [a.to_tangent_aval() for a in jaxpr.out_avals]
  tangent_metadata = _resolve_ad_metadata(xla_metadata, ad_metadata)

  def _filter_zeros(is_nz_l, l):
    return tuple(x for nz, x in zip(is_nz_l, l) if nz)

  def tangent_fun(residuals, structured_residuals, *tangents):
    tangents_nz = _filter_zeros(nzs, tangents)
    sres_flat = tree_leaves(structured_residuals)
    assert (len(residuals) + len(tangents_nz) + len(sres_flat)
            == len(tangent_jaxpr.invars)), (
        len(residuals), len(tangents_nz), len(sres_flat),
        len(tangent_jaxpr.invars))
    nz_outs = xla_metadata_call_p.bind(*residuals, *tangents_nz, *sres_flat,
                                       jaxpr=tangent_jaxpr,
                                       xla_metadata=tangent_metadata,
                                       ad_metadata='same')
    nz_outs_ = iter(nz_outs)
    outs = [next(nz_outs_) if nz else ad.Zero(a)
            for nz, a in zip(nzs_out, tangent_avals_out)]
    assert next(nz_outs_, None) is None
    return outs

  ans = xla_metadata_call_p.bind(*primals, jaxpr=primal_jaxpr,
                                 xla_metadata=xla_metadata,
                                 ad_metadata=ad_metadata)
  ans = subs_list(out_fwd, ans, ans)
  ans = subs_list(in_fwd, primals, ans)
  primal_ans, residuals_ans = split_list(ans, [len(ans) - num_residuals_out])
  ures, sres_flat = split_list(residuals_ans, [len(ures_avals)])
  ures = subs_list(in_fwd_res, [*jaxpr.consts, *primals], ures)
  sres = sres_avals.update(sres_flat).unflatten()
  return primal_ans, nzs_out, ures, sres, tangent_fun
ad.primitive_linearizations[xla_metadata_call_p] = _xla_metadata_call_lin


@weakref_lru_cache
def _transpose_jaxpr(jaxpr, in_tree, in_avals, specs):
  out_tree = None
  def transposed(*in_flat):
    nonlocal out_tree
    primals_ctrefs, cts_in = tree_unflatten(in_tree, in_flat)
    args = ad.unproject_accums(specs, primals_ctrefs)
    logs = ad.backward_pass3(jaxpr, False, jaxpr.consts, args, cts_in)
    cts_out = [x.freeze() if isinstance(x, ad.ValAccum) else None for x in args]
    outs, out_tree = tree_flatten((cts_out, logs))
    return outs
  dbg = jaxpr.debug_info.with_unknown_names()
  trans_jaxpr, _ = pe.trace_to_jaxpr(
      transposed, ft.flatten_args(*in_avals), dbg)
  return trans_jaxpr, out_tree


def _xla_metadata_call_transpose(cts_in, *args, jaxpr, xla_metadata,
                                 ad_metadata):
  primals_ctrefs, specs = ad.project_accums(args)
  in_flat, in_tree = tree_flatten((primals_ctrefs, cts_in))
  in_avals = [core.typeof(x) for x in in_flat]
  trans_jaxpr, out_tree = _transpose_jaxpr(jaxpr, in_tree, (*in_avals,), specs)

  outs = xla_metadata_call_p.bind(
      *in_flat, jaxpr=trans_jaxpr,
      xla_metadata=_resolve_ad_metadata(xla_metadata, ad_metadata),
      ad_metadata='same')

  cts_out, logs = tree_unflatten(out_tree, outs)
  for x, ct in zip(args, cts_out):
    if isinstance(x, ad.ValAccum):
      x.accum(ct)
  return logs


ad.fancy_transposes[xla_metadata_call_p] = _xla_metadata_call_transpose


def _xla_metadata_call_to_lojax(*hi_args, jaxpr, xla_metadata, ad_metadata):
  lo_args_lol = [a.lower_val(x) for a, x in zip(jaxpr.in_avals, hi_args)]
  lo_args = [x for xs in lo_args_lol for x in xs]
  in_avals = ft.flatten(([[core.typeof(x) for x in xs] for xs in lo_args_lol],
                         {}))
  lo_jaxpr, out_avals = pe.lower_jaxpr(jaxpr, in_avals)
  all_outs = xla_metadata_call_p.bind(*lo_args, jaxpr=lo_jaxpr,
                                      xla_metadata=xla_metadata,
                                      ad_metadata=ad_metadata)
  lo_outs = out_avals.update(all_outs)
  return [a.raise_val2(y) for a, y in zip(jaxpr.out_avals, lo_outs.unpack())]
xla_metadata_call_p.to_lojax = _xla_metadata_call_to_lojax


def _xla_metadata_call_partial_eval_custom_params_updater(
    unks_in,
    inst_in,
    kept_outs_known,
    kept_outs_staged,
    num_res_out,
    num_res_in,
    params_known,
    params_staged,
):
  return params_known, params_staged


pe.partial_eval_jaxpr_custom_rules[xla_metadata_call_p] = partial(
    pe.closed_call_partial_eval_custom_rule,
    'jaxpr',
    _xla_metadata_call_partial_eval_custom_params_updater,
)


def dce_jaxpr_xla_metadata_rule(used_outputs: list[bool], eqn: pe.JaxprEqn
                                ) -> tuple[list[bool], pe.JaxprEqn | None]:
  if not any(used_outputs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None
  dced_jaxpr, used_inputs = pe._cached_closed_call_dce(
      eqn.params['jaxpr'], tuple(used_outputs))
  new_params = dict(eqn.params, jaxpr=dced_jaxpr)
  if not any(used_inputs) and not any(used_outputs) and not dced_jaxpr.effects:
    return used_inputs, None
  else:
    new_invars = [v for v, used in zip(eqn.invars, used_inputs) if used]
    new_effs = core.eqn_effects(dced_jaxpr, new_invars)
    new_eqn = pe.new_jaxpr_eqn(
        new_invars,
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, new_effs, eqn.source_info, eqn.ctx)
    return used_inputs, new_eqn
pe.dce_rules[xla_metadata_call_p] = dce_jaxpr_xla_metadata_rule
