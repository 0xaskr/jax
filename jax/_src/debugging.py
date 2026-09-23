# Copyright 2022 The JAX Authors.
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

# 文件职责：实现 JAX 的调试原语（`jax.debug.*`）及其变换规则。
# 对外提供 `jax.debug.callback`、`jax.debug.print`/`debug_log`，以及
# `visualize_sharding`、`inspect_array_sharding` 等分片可视化工具。
# 内部定义 `DebugEffect`/`OrderedDebugEffect` 两种效果和 `debug_callback`、
# `debug_print`、`inspect_sharding` 原语，并为它们注册批处理、JVP、转置、
# 部分求值、降级（lowering）与 `shard_map` 即时求值规则。

"""JAX 调试原语及相关功能的模块。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
from functools import partial
import importlib.util
import logging
import string
import sys
from typing import Any, overload

import numpy as np

from jax._src import api
from jax._src import callback as cb
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import effects
from jax._src import lax
from jax._src import mesh as mesh_lib
from jax._src import shard_map
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import tree_util
from jax._src import util
from jax._src import xla_bridge
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.numpy import lax_numpy as jnp
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (
    NamedSharding, PartitionSpec as P, parse_flatten_op_sharding)
from jax._src.state import discharge as state_discharge

logger = logging.getLogger(__name__)

class DebugEffect(effects.Effect):
  __str__ = lambda self: "Debug"
debug_effect = DebugEffect()

class OrderedDebugEffect(effects.Effect):
  __str__ = lambda self: "OrderedDebug"

ordered_debug_effect = OrderedDebugEffect()
effects.ordered_effects.add_type(OrderedDebugEffect)
effects.lowerable_effects.add_type(DebugEffect)
effects.lowerable_effects.add_type(OrderedDebugEffect)
effects.control_flow_allowed_effects.add_type(DebugEffect)
effects.control_flow_allowed_effects.add_type(OrderedDebugEffect)
effects.remat_allowed_effects.add_type(DebugEffect)
effects.remat_allowed_effects.add_type(OrderedDebugEffect)
effects.custom_derivatives_allowed_effects.add_type(DebugEffect)
effects.custom_derivatives_allowed_effects.add_type(OrderedDebugEffect)
effects.partial_eval_kept_effects.add_type(DebugEffect)
effects.partial_eval_kept_effects.add_type(OrderedDebugEffect)

# `debug_callback_p` 是把 Python 回调暂存（stage out）出去的主要原语。
debug_callback_p = core.Primitive('debug_callback')
debug_callback_p.multiple_results = True

map, unsafe_map = util.safe_map, map

@debug_callback_p.def_impl
def debug_callback_impl(*args, callback: Callable[..., Any],
                        effect: DebugEffect, partitioned: bool):
  del effect, partitioned
  try:
    cpu_device, *_ = xla_bridge.local_devices(backend="cpu")
  except RuntimeError as e:
    raise RuntimeError(
        "jax.debug.callback failed to find a local CPU device to place the"
        " inputs on. Make sure \"cpu\" is listed in --jax_platforms or the"
        " JAX_PLATFORMS environment variable."
    ) from e
  args = api.device_put(args, cpu_device)
  with (config.default_device(cpu_device),
        sharding_impls._internal_use_concrete_mesh(mesh_lib.empty_concrete_mesh),
        mesh_lib.use_abstract_mesh(mesh_lib.empty_abstract_mesh)):
    try:
      callback(*args)
    except BaseException:
      logger.exception("jax.debug.callback failed")
      raise
  return ()

@debug_callback_p.def_effectful_abstract_eval
def debug_callback_abstract_eval(*flat_avals, callback: Callable[..., Any],
                                 effect: DebugEffect, partitioned: bool):
  del flat_avals, callback, partitioned
  return [], {effect}


def debug_batching_rule(args, dims, *, primitive, **params):
  """把调试回调沿被映射的轴展开（unroll）。"""
  axis_size = next(x.shape[i] for x, i in zip(args, dims)
                   if i is not None)
  # TODO(sharadmv): 改为用循环（rolled loop）实现，而不是展开（unrolled）。
  def get_arg_at_dim(i, dim, arg):
    if dim is None:
      # 广播未被映射的参数
      return arg
    return lax.index_in_dim(arg, i, axis=dim, keepdims=False)
  outs = []
  for i in range(axis_size):
    args_idx = map(partial(get_arg_at_dim, i), dims, args)
    outs.append(primitive.bind(*args_idx, **params))
  outs = [jnp.stack(xs) for xs in zip(*outs)]
  return outs, (0,) * len(outs)


batching.primitive_batchers[debug_callback_p] = partial(
    debug_batching_rule, primitive=debug_callback_p
)

def debug_callback_jvp_rule(primals, tangents, **params):
  return debug_callback_p.bind(*primals, **params), []
ad.primitive_jvps[debug_callback_p] = debug_callback_jvp_rule

def debug_callback_transpose_rule(_, *flat_args, callback: Callable[..., Any],
                                  effect: DebugEffect, partitioned):
  del callback, effect, partitioned
  return [None for _ in flat_args]
ad.primitive_transposes[debug_callback_p] = debug_callback_transpose_rule

def _debug_callback_partial_auto(axis_context, *args, **params):
  partial_auto = list(set(axis_context.mesh.axis_names) - axis_context.manual_axes)
  def f():
    idx = lax.axis_index(*partial_auto)
    return lax.cond(idx == 0,
                    lambda: debug_callback_p.bind(*args, **params),
                    lambda: [])
  return shard_map.shard_map(f, in_specs=(), out_specs=[])()

def debug_callback_lowering(ctx, *args, effect, partitioned, callback, **params):
  axis_context = ctx.module_context.axis_context
  if isinstance(axis_context, sharding_impls.SPMDAxisContext):
    # 我们处在 shard_map 中，可能是部分手动（partial-manual）或全手动（full-manual）分片。
    partial_auto = set(axis_context.mesh.axis_names) - axis_context.manual_axes
    if partial_auto:
      # 如果存在部分手动 / 部分自动的分片，我们会先做聚集（gather），再
      # 有条件地执行回调。
      lower = partial(
          _debug_callback_partial_auto,
          axis_context,
          effect=effect,
          partitioned=partitioned,
          callback=callback,
          **params,
      )
      return mlir.lower_fun(lower)(ctx, *args)
    elif set(axis_context.manual_axes) == set(axis_context.mesh.axis_names):
      # 如果降级时是全手动分片，说明该 JAX 程序具有逐设备（per-device）语义，
      # 因此我们在每台设备上都运行一次回调。
      if config.use_shardy_partitioner.value:
        sharding = cb._get_sdy_array_list_for_callbacks(ctx.avals_out)
      else:
        sharding = xc.OpSharding()
        sharding.type = xc.OpSharding.Type.MANUAL
    else:
      assert False  # 不可达
  elif isinstance(axis_context, sharding_impls.ShardingContext):
    # 如果降级时是全自动分片，说明该 JAX 程序具有整体数组（bulk array）语义，
    # 因此我们用 MAXIMAL 分片来运行回调，
    # 于是它只在完整的逻辑值上执行一次。
    if config.use_shardy_partitioner.value:
      sharding = sharding_impls.SdyArrayList((
          sharding_impls.SdyArray(
              mesh_shape=(), dim_shardings=(), logical_device_ids=(0,)),))
    else:
      sharding = xc.OpSharding()
      sharding.type = xc.OpSharding.Type.MAXIMAL
      sharding.tile_assignment_dimensions = [1]
      sharding.tile_assignment_devices = [0]
  else:
    # 没有进行 SPMD 分区时，不要标注分片。
    sharding = None

  def _callback(*flat_args):
    debug_callback_p.impl(
        *flat_args,
        effect=effect,
        partitioned=partitioned,
        callback=callback,
        **params,
    )
    return ()
  if effects.ordered_effects.contains(effect):
    token = ctx.tokens_in.get(effect)
    result, token, _ = cb.emit_python_callback(
        ctx, _callback, token, list(args), ctx.avals_in, ctx.avals_out,
        has_side_effect=True, returns_token=True, partitioned=partitioned)
    ctx.set_tokens_out(
        ctx.tokens_in.update_tokens(mlir.TokenSet({effect: token})))
  else:
    result, _, _ = cb.emit_python_callback(
        ctx, _callback, None, list(args), ctx.avals_in, ctx.avals_out,
        has_side_effect=True, returns_token=True, partitioned=partitioned,
        sharding=sharding)
  return result
mlir.register_lowering(debug_callback_p, debug_callback_lowering,
                       platform="cpu")
mlir.register_lowering(
    debug_callback_p, debug_callback_lowering, platform="gpu")
# 调试回调在 TPU 上使用 channel ID，因此不能缓存。
mlir.register_lowering(
    debug_callback_p, debug_callback_lowering, platform="tpu",
    cacheable=False)


def _debug_partial_eval_custom(saveable, unks_in, inst_in, eqn, primitive):
  # 带效果的原语默认行为是尽量不把它们暂存（stage out）。
  # 但对调试回调，我们恰恰希望把它暂存出去，
  # 以便向用户提供更多信息。这条规则绕过了 partial_eval 的
  # 常规行为来实现这一点。具体来说，在以下情况下
  # 我们就会暂存该回调：
  # 1) 策略认为 debug_callback 不可保存（saveable）
  # 2) 策略认为 debug_callback 可保存，但所有输入
  #    值都已实例化。
  # 这样做的目的是在回调时提供尽可能多的信息，
  # 同时避免不必要地暂存其他值。
  if any(unks_in):
    # 常见情形（只要存在未知量，就需要把它暂存出去）
    res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
    return None, eqn, [], [], res
  if saveable(primitive, *[v.aval for v in eqn.invars], **eqn.params):
    # 策略告诉我们，可以保存这个调试回调。
    if all(inst_in):
      # 如果所有输入都已实例化，我们也把
      # debug_callback 暂存出去。
      return eqn, eqn, [], [], []
    else:
      # 如果有输入尚未实例化，我们不做额外的暂存，以免
      # 影响计算结果。
      return eqn, None, [], [], []
  # 如果（根据策略）不能保存调试回调，我们就遵从策略，
  # 把它暂存出去。
  return eqn, eqn, [], [], []


pe.partial_eval_jaxpr_custom_rules[debug_callback_p] = partial(
    _debug_partial_eval_custom, primitive=debug_callback_p
)

@state_discharge.register_discharge_rule(debug_callback_p)
def _debug_callback_state_discharge_rule(
    ctx, *args, effect, partitioned, callback, **params
):
  del ctx  # 未使用。
  out = debug_callback_p.bind(
      *args, effect=effect, partitioned=partitioned, callback=callback, **params
  )
  return args, out


def _split_callback_args(args, kwargs):
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  static_args, dyn_args = {}, []
  for i, a in enumerate(flat_args):
    try:
      core.shaped_abstractify(a)
      dyn_args.append(a)
    except (AssertionError, TypeError):
      static_args[i] = a
  return in_tree, dyn_args, static_args


def merge_callback_args(in_tree, dyn_args, static_args):
  static_args_dict = dict(static_args)
  all_args = [None] * (len(static_args) + len(dyn_args))
  di = iter(dyn_args)
  for i in range(len(all_args)):
    if i in static_args_dict:
      all_args[i] = static_args_dict[i]
    else:
      all_args[i] = next(di)
  assert next(di, None) is None
  return tree_util.tree_unflatten(in_tree, all_args)


def _make_flat_callback(in_tree, callback, static_args):
  def _flat_callback(*dyn_args):
    args, kwargs = merge_callback_args(in_tree, dyn_args, static_args)
    callback(*args, **kwargs)
    return ()
  return _flat_callback


debug_print_p = core.Primitive("debug_print")
debug_print_p.multiple_results = True


@debug_print_p.def_impl
def debug_print_impl(
    *args: Any,
    fmt: str,
    ordered,
    partitioned,
    in_tree,
    static_args,
    np_printoptions,
    has_placeholders,
    logging_record,
):
  callback = partial(
      _format_print_callback, fmt, dict(np_printoptions), has_placeholders,
      logging_record,
  )
  callback = _make_flat_callback(in_tree, callback, static_args)
  effect = ordered_debug_effect if ordered else debug_effect
  debug_callback_impl(
      *args, callback=callback, effect=effect, partitioned=partitioned
  )
  return ()


@debug_print_p.def_effectful_abstract_eval
def debug_print_abstract_eval(*avals: Any, fmt: str, ordered, **kwargs):
  del avals, fmt, kwargs  # 未使用。
  effect = ordered_debug_effect if ordered else debug_effect
  return [], {effect}


batching.primitive_batchers[debug_print_p] = partial(
    debug_batching_rule, primitive=debug_print_p
)


def debug_print_jvp_rule(primals, tangents, **params):
  return debug_print_p.bind(*primals, **params), []


ad.primitive_jvps[debug_print_p] = debug_print_jvp_rule


def debug_print_transpose_rule(_, *args, **kwargs):
  del kwargs
  return [None for _ in args]


ad.primitive_transposes[debug_print_p] = debug_print_transpose_rule


def debug_print_lowering_rule(
    ctx,
    *dyn_args,
    fmt,
    ordered,
    partitioned,
    in_tree,
    static_args,
    np_printoptions,
    has_placeholders,
    logging_record,
):
  callback = partial(
      _format_print_callback,
      fmt,
      dict(np_printoptions),
      has_placeholders,
      logging_record,
  )
  callback = _make_flat_callback(in_tree, callback, static_args)
  effect = ordered_debug_effect if ordered else debug_effect
  return debug_callback_lowering(
      ctx, *dyn_args, effect=effect, partitioned=partitioned, callback=callback
  )


mlir.register_lowering(debug_print_p, debug_print_lowering_rule, platform="cpu")
mlir.register_lowering(debug_print_p, debug_print_lowering_rule, platform="gpu")
mlir.register_lowering(
    debug_print_p, debug_print_lowering_rule, platform="tpu", cacheable=False
)

pe.partial_eval_jaxpr_custom_rules[debug_print_p] = partial(
    _debug_partial_eval_custom, primitive=debug_print_p
)


@state_discharge.register_discharge_rule(debug_print_p)
def _debug_print_state_discharge_rule(ctx, *args, **kwargs):
  del ctx  # 未使用。
  out = debug_print_p.bind(*args, **kwargs)
  return args, out


@overload
def debug_callback(
    callback: Callable[..., None],
    *args: Any,
    ordered: bool = False,
    partitioned: bool = False,
    **kwargs: Any,
) -> None:
  ...

@overload
def debug_callback(
    *,
    ordered: bool = False,
    partitioned: bool = False,
) -> Callable[..., None]:
  ...

def debug_callback(
    callback: Callable[..., None] | None = None,
    *args: Any,
    ordered: bool = False,
    partitioned: bool = False,
    **kwargs: Any,
) -> Callable[..., None] | None:
  """调用一个可暂存的 Python 回调（callback）。

  更多说明参见 `External Callbacks`_。

  ``jax.debug.callback`` 让你可以传入一个 Python 函数，
  并能在已暂存的 JAX 程序中调用它。``jax.debug.callback`` 遵循
  JAX 变换既有的*纯*（pure）操作语义，因此
  对副作用一无所知。这意味着在高阶原语与变换的作用下，
  该效果可能被丢弃、复制，
  甚至被重新排序。

  我们之所以希望如此，是因为想让 ``jax.debug.callback``
  保持“无害”（innocuous），即希望这些原语在尽可能少地
  改变 JAX 计算的同时，尽可能多地暴露关于它的信息，
  例如计算的哪些部分被复制或被丢弃。

  ``jax.debug.callback`` 支持两种调用方式：

  1. 两次调用形式（推荐）：
     ``jax.debug.callback(ordered=True)(callback, *args, **kwargs)``
     选项在第一次调用中传入。回调及其参数
     在第二次调用中传入。第二次调用不接受
     任何选项参数。

  2. 单次调用形式：
     ``jax.debug.callback(callback, *args, ordered=True, **kwargs)``
     （软弃用）把 `ordered` 与 `partitioned` 选项与回调的
     ``kwargs`` 混在一起使用已被软弃用。

  Args:
    callback: 一个返回 None 的 Python 可调用对象。
    *args: 传给回调的位置参数。
    ordered: 仅关键字参数，用于指示已暂存的计算是否
      会对该回调与其他 ordered 回调之间的先后顺序
      强制执行排序。
    partitioned: 若为 True，则只打印本地分片；该选项可避免
      对操作数做 all-gather。若为 False，则用逻辑操作数打印；
      该选项需要先对操作数做一次 all-gather。
    **kwargs: 传给回调的关键字参数。

  Returns:
    None

  See Also:
    - :func:`jax.experimental.io_callback`: 为不纯（impure）
      函数设计的回调。
    - :func:`jax.pure_callback`: 为纯函数设计的回调。
    - :func:`jax.debug.print`: 为打印设计的回调。

  .. _External Callbacks:
     https://docs.jax.dev/en/latest/notebooks/external_callbacks.html
  """
  def _debug_callback(
      callback: Callable[..., None], *c_args: Any, **c_kwargs: Any
  ):
    if not callable(callback):
      raise TypeError(
          "first argument to jax.debug.callback must be callable, "
          f"but got an object of type {type(callback)}"
      )
    in_tree, dyn_args, static_args = _split_callback_args(c_args, c_kwargs)

    def _flat_callback(*dyn_args_flat):
      all_args = [None] * (len(static_args) + len(dyn_args_flat))
      di = iter(dyn_args_flat)
      for i in range(len(all_args)):
        if i in static_args:
          all_args[i] = static_args[i]
        else:
          all_args[i] = next(di)
      assert next(di, None) is None
      args_, kwargs_ = tree_util.tree_unflatten(in_tree, all_args)
      callback(*args_, **kwargs_)
      return ()

    effect = ordered_debug_effect if ordered else debug_effect
    debug_callback_p.bind(
        *dyn_args,
        callback=_flat_callback,
        effect=effect,
        partitioned=partitioned,
    )

  if callback is not None:
    _debug_callback(callback, *args, **kwargs)
    return None

  if args or kwargs:
    raise TypeError(
        "debug_callback received unexpected arguments in the two-call form:"
        f" {args=} {kwargs=}"
    )
  return _debug_callback


class _DebugPrintFormatChecker(string.Formatter):

  def format_field(self, value, format_spec):
    del value, format_spec
    return ""  # 不做任何格式化。

  def check_unused_args(self, used_args, args, kwargs):
    unused_args = [arg for i, arg in enumerate(args) if i not in used_args]
    unused_kwargs = [k for k in kwargs if k not in used_args]
    if unused_args:
      raise ValueError(
          f"Unused positional arguments to `jax.debug.print`: {unused_args}")
    if unused_kwargs:
      raise ValueError(
          f"Unused keyword arguments to `jax.debug.print`: {unused_kwargs}. "
          "You may be passing an f-string (i.e, `f\"{x}\"`) into "
          "`jax.debug.print` and instead should pass in a regular string.")

formatter = _DebugPrintFormatChecker()


def _format_print_callback(
    fmt: str, np_printoptions, has_placeholders, logging_record, *args, **kwargs
):
  if has_placeholders:
    with np.printoptions(**np_printoptions):
      msg = fmt.format(*args, **kwargs)
  else:
    assert not kwargs, "Format without placeholders should not have kwargs."
    msg = " ".join((fmt, *(str(a) for a in args)))
  if logging_record:
    logging_record = copy.copy(logging_record)
    logging_record.msg = msg
    logger.handle(logging_record)
  else:
    sys.stdout.write(msg + "\n")


def _make_logging_record(level):
  si = source_info_util.current()
  user_frame = source_info_util.user_frame(si.traceback)

  file_name = "(unknown file)"
  line_no = 0
  if user_frame:
    file_name = user_frame.file_name
    line_no = user_frame.start_line
  args = ()
  return logger.makeRecord(
      logger.name, level, file_name, line_no, "", args, None
  )

@overload
def debug_print(
    fmt: str,
    *args: Any,
    ordered: bool = False,
    partitioned: bool = False,
    skip_format_check: bool = False,
    _use_logging: bool = False,
    **kwargs: Any,
) -> None:
  ...

@overload
def debug_print(
    *,
    ordered: bool = False,
    partitioned: bool = False,
    skip_format_check: bool = False,
    _use_logging: bool = False,
) -> Callable[..., None]:
  ...

def debug_print(
    fmt: str | None = None,
    *args,
    ordered: bool = False,
    partitioned: bool = False,
    skip_format_check: bool = False,
    _use_logging: bool = False,
    **kwargs,
) -> Callable[..., None] | None:
  """打印值，并能在已暂存（staged out）的 JAX 函数中工作。

  该函数*不*支持 f-string，因为格式化被延迟了。
  所以不要写 ``jax.debug.print(f"hello {bar}")``，而应写
  ``jax.debug.print("hello {bar}", bar=bar)``。

  ``jax.debug.print`` 支持两种调用方式：

  1. 两次调用形式（推荐）：
     ``jax.debug.print(ordered=True)("hello {x}", x=42)``
     选项在第一次调用中传入。格式字符串与参数
     在第二次调用中传入。第二次调用不接受
     任何选项参数。

  2. 单次调用形式：
     ``jax.debug.print("hello {x}", x=42, ordered=True)``
     （软弃用）把 `ordered` 与 `partitioned` 选项与 print 的
     ``kwargs`` 混在一起使用已被软弃用。

  Args:
    fmt: 格式字符串，例如 ``"hello {x}"``，用于格式化
      输入参数，用法类似 ``str.format``。参见 Python 文档中的 `string
      formatting <https://docs.python.org/3/library/stdtypes.html#str.format>`_
      以及 `format string syntax
      <https://docs.python.org/3/library/string.html#formatstrings>`_。
    *args: 要被格式化的位置参数列表，如同传给
      ``fmt.format``。
    ordered: 仅关键字参数，用于指示已暂存的计算是否
      会对这个 ``jax.debug.print`` 相对于其他 ordered 的
      ``jax.debug.print`` 调用强制排序。
    partitioned: 若为 True，则只打印本地分片；该选项可避免
      对操作数做 all-gather。若为 False，则用逻辑操作数打印；
      该选项需要先对操作数做一次 all-gather。
    skip_format_check: 若为 True，则不检查格式字符串。这在
      从 Pallas TPU kernel 内部使用该函数时很有用，此时标量
      参数会打印在格式字符串之后。
    **kwargs: 要被格式化的额外关键字参数，如同传给
      ``fmt.format``。
  """
  def _debug_print(fmt: str, *c_args, **c_kwargs):
    if not skip_format_check:
      # 检查我们传给格式化的参数是否正确。
      formatter.format(fmt, *c_args, **c_kwargs)
    has_placeholders = False
    if fmt:
      _, field_name, *_ = next(iter(string.Formatter().parse(fmt)))
      has_placeholders = field_name is not None
    in_tree, dyn_args, static_args = _split_callback_args(c_args, c_kwargs)
    static_args = tuple(static_args.items())
    np_printoptions = tuple(np.get_printoptions().items())

    debug_print_p.bind(
        *dyn_args,
        fmt=fmt,
        ordered=ordered,
        partitioned=partitioned,
        in_tree=in_tree,
        static_args=static_args,
        np_printoptions=np_printoptions,
        has_placeholders=has_placeholders,
        logging_record=(
            _make_logging_record(logging.INFO) if _use_logging else None
        ),
    )

  if fmt is not None:
    _debug_print(fmt, *args, **kwargs)
    return None
  if args or kwargs:
    raise TypeError(
        "debug_print received unexpected arguments in the two-call form:"
        f" {args=} {kwargs=}"
    )
  return _debug_print


debug_log = partial(debug_print, _use_logging=True)

# 分片可视化

inspect_sharding_p = core.Primitive("inspect_sharding")
inspect_sharding_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(inspect_sharding_p)

def _inspect_sharding_impl(value, *, callback):
  callback(value.sharding)
  return []
inspect_sharding_p.def_impl(_inspect_sharding_impl)

def _inspect_sharding_abstract_eval(aval, **_):
  del aval
  # 带效果的抽象求值可避免死代码消除（DCE）
  return [], {debug_effect}
inspect_sharding_p.def_effectful_abstract_eval(_inspect_sharding_abstract_eval)

def _inspect_sharding_batching_rule(args, _, *, callback):
  value, = args
  inspect_sharding_p.bind(value, callback=callback)
  return [], []
batching.primitive_batchers[inspect_sharding_p] = (
    _inspect_sharding_batching_rule)

def _inspect_sharding_jvp_rule(primals, _, **params):
  return inspect_sharding_p.bind(*primals, **params), []
ad.primitive_jvps[inspect_sharding_p] = _inspect_sharding_jvp_rule

_INSPECT_SHARDING_CALL_NAME = "InspectSharding"

def _inspect_sharding_lowering_rule(ctx: mlir.LoweringRuleContext, value, *,
                                    callback):

  mesh = mesh_lib.thread_resources.env.physical_mesh
  axis_context = ctx.module_context.axis_context

  if isinstance(axis_context, sharding_impls.ShardingContext):
    devices = axis_context.device_assignment
    if devices is None:
      raise AssertionError(
          'Please file a bug at https://github.com/jax-ml/jax/issues')
    am = axis_context.abstract_mesh
    if am is not None:
      mesh = mesh_lib.Mesh(np.array(devices).reshape(am.axis_sizes),
                           am.axis_names)
  elif isinstance(axis_context, sharding_impls.SPMDAxisContext):
    mesh = axis_context.mesh
    devices = axis_context.mesh._flat_devices_tuple
  else:
    raise NotImplementedError(type(axis_context))
  assert devices is not None

  # 如果存在非平凡的并行计算，我们需要等到 SPMD 分区器
  # 用 `HloSharding` 回调回来。
  def _hlo_sharding_callback(hlo_sharding: xc.HloSharding):
    if mesh.empty:
      return callback(
          sharding_impls.GSPMDSharding(devices, hlo_sharding))
    pspec = (P() if hlo_sharding.is_manual() else
             parse_flatten_op_sharding(hlo_sharding, mesh)[0])
    return callback(NamedSharding(mesh, pspec))

  if len(devices) == 1:
    # 如果计算中只有一台设备，我们可以直接构造一个
    # 复制的（replicated）HloSharding 并立即调用它。
    _hlo_sharding_callback(sharding_impls.replicated_hlo_sharding)
    return []

  key = xc.encode_inspect_sharding_callback(_hlo_sharding_callback)
  # 我们需要确保 SPMD 分区器运行时 `_hlo_sharding_callback` 仍然存活，
  # 因此把它挂到可执行文件上以保持其存活。
  ctx.module_context.add_keepalive(_hlo_sharding_callback)

  hlo.CustomCallOp([value.type], [value],
                   call_target_name=ir.StringAttr.get(
                     _INSPECT_SHARDING_CALL_NAME),
                   has_side_effect=ir.BoolAttr.get(True),
                   api_version=mlir.i32_attr(1),
                   called_computations=ir.ArrayAttr.get([]),
                   backend_config=ir.StringAttr.get(key),
                   operand_layouts=None,
                   result_layouts=None)
  return []
mlir.register_lowering(inspect_sharding_p, _inspect_sharding_lowering_rule)

def _slice_to_chunk_idx(size: int, slc: slice) -> int:
  if slc.stop == slc.start == None:
    return 0
  slice_size = slc.stop - slc.start
  assert slc.start % slice_size == 0
  assert size % slice_size == 0
  return slc.start // slice_size

def _raise_to_slice(slc: slice | int):
  if isinstance(slc, int):
    return slice(slc, slc + 1)
  return slc

Color = tuple[float, float, float] | str
ColorMap = Callable[[float], tuple[float, float, float, float]]

def _canonicalize_color(color: Color) -> str:
  if isinstance(color, str):
    return color
  r, g, b = (int(a * 255) for a in color)
  return f"#{r:02X}{g:02X}{b:02X}"

def _get_text_color(color: str) -> str:
  r, g, b = map(lambda x: int(x, 16), (color[1:3], color[3:5], color[5:7]))
  if (r * 0.299 + g * 0.587 + b * 0.114) > 186:
    return "#000000"
  return "#ffffff"

def make_color_iter(color_map, num_rows, num_cols):
  num_colors = num_rows * num_cols
  color_values = np.linspace(0, 1, num_colors)
  idx = 0
  for _ in range(num_colors):
    yield color_map(color_values[idx])
    idx = (idx + num_colors // 2 + bool(num_colors % 2 == 0)) % num_colors

def visualize_sharding(shape: Sequence[int], sharding: Sharding, *,
                       use_color: bool = True, scale: float = 1.,
                       min_width: int = 9, max_width: int = 80,
                       color_map: ColorMap | None = None):
  """用 ``rich`` 可视化一个 ``Sharding``。"""
  if not importlib.util.find_spec("rich"):
    raise ValueError("`visualize_sharding` requires `rich` to be installed.")

  # 这些导入放在函数内部，以免影响 JAX 的导入时间。
  import rich.align  # pyrefly: ignore[missing-import]
  import rich.console  # pyrefly: ignore[missing-import]
  import rich.box  # pyrefly: ignore[missing-import]
  import rich.padding  # pyrefly: ignore[missing-import]
  import rich.style  # pyrefly: ignore[missing-import]
  import rich.table  # pyrefly: ignore[missing-import]

  if len(shape) > 2 or len(shape) < 1:
    raise ValueError(
        "`visualize_sharding` only works for shapes with 1 and 2 dimensions.")
  console = rich.console.Console(width=max_width)
  use_color = use_color and console.color_system is not None
  if use_color and not color_map:
    try:
      import matplotlib as mpl  # pyrefly: ignore[missing-import]
      color_map = mpl.colormaps["tab20b"]
    except ModuleNotFoundError:
      use_color = False

  base_height = int(10 * scale)
  aspect_ratio = (shape[1] if len(shape) == 2 else 1) / shape[0]
  base_width = int(base_height * aspect_ratio)
  height_to_width_ratio = 2.5

  # 从第一台设备获取设备类型
  device_kind = next(iter(sharding.device_set)).platform.upper()

  device_indices_map = sharding.devices_indices_map(tuple(shape))
  slices: dict[tuple[int, ...], set[int]] = {}
  heights: dict[tuple[int, ...], float | None] = {}
  widths: dict[tuple[int, ...], float] = {}

  for i, (dev, slcs) in enumerate(device_indices_map.items()):
    assert slcs is not None
    slcs = tuple(map(_raise_to_slice, slcs))
    chunk_idxs = tuple(map(_slice_to_chunk_idx, shape, slcs))
    if slcs is None:
      raise NotImplementedError
    if len(slcs) == 2:
      vert, horiz = slcs
      vert_size  = ((vert.stop  - vert.start ) if vert.stop  is not None
                    else shape[0])
      horiz_size = ((horiz.stop - horiz.start) if horiz.stop is not None
                    else shape[1])
      chunk_height = vert_size / shape[0]
      chunk_width = horiz_size / shape[1]
      heights[chunk_idxs] = chunk_height
      widths[chunk_idxs] = chunk_width
    else:
      # 在一维情形下，我们把高度设为 1。
      horiz, = slcs
      vert = slice(0, 1, None)
      horiz_size = (
          (horiz.stop - horiz.start) if horiz.stop is not None else shape[0])
      chunk_idxs = (0, *chunk_idxs)
      heights[chunk_idxs] = None
      widths[chunk_idxs]  = horiz_size / shape[0]
    slices.setdefault(chunk_idxs, set()).add(dev.id)
  num_rows = max(a[0] for a in slices.keys()) + 1
  if len(list(slices.keys())[0]) == 1:
    num_cols = 1
  else:
    num_cols = max(a[1] for a in slices.keys()) + 1

  color_iter = make_color_iter(color_map, num_rows, num_cols)
  table = rich.table.Table(show_header=False, show_lines=not use_color,
                           padding=0,
                           highlight=not use_color, pad_edge=False,
                           box=rich.box.SQUARE if not use_color else None)
  for i in range(num_rows):
    col = []
    for j in range(num_cols):
      entry = f"{device_kind} "+",".join([str(s) for s in sorted(slices[i, j])])
      width, maybe_height = widths[i, j], heights[i, j]
      width = int(width * base_width * height_to_width_ratio)
      if maybe_height is None:
        height = 1
      else:
        height = int(maybe_height * base_height)
      width = min(max(width, min_width), max_width)
      left_padding, remainder = divmod(width - len(entry) - 2, 2)
      right_padding = left_padding + remainder
      top_padding, remainder = divmod(height - 2, 2)
      bottom_padding = top_padding + remainder
      if use_color:
        color = _canonicalize_color(next(color_iter)[:3])
        text_color = _get_text_color(color)
        top_padding += 1
        bottom_padding += 1
        left_padding += 1
        right_padding += 1
      else:
        color = None
        text_color = None
      padding = (
          max(top_padding, 0),
          max(right_padding, 0),
          max(bottom_padding, 0),
          max(left_padding, 0),
      )
      col.append(
          rich.padding.Padding(
            rich.align.Align(entry, "center", vertical="middle"), padding,
            style=rich.style.Style(bgcolor=color,
              color=text_color)))
    table.add_row(*col)
  console.print(table, end='\n\n')

def inspect_array_sharding(value, *, callback: Callable[[Sharding], None]):
  """让你可以在经 JIT 处理的函数内部检查数组分片。

  给定一个由数组组成的 Pytree，该函数会针对每个数组的分片进行
  回调，并且能在经 ``jax.jit`` 处理的计算中工作，从而可以检查
  所选中间值（intermediate）的分片。

  ``callback`` 的调用时机策略是：一旦分片信息可用就*尽早*调用。
  这意味着如果在没有任何变换的情况下调用 ``inspect_array_callback``，
  回调会立即发生，因为数组及其分片信息随手可得。
  在 ``jax.jit`` 内部，回调会发生在降级（lowering）阶段，
  也就是说你可以用 AOT API（``jit(f).lower(...)``）来
  触发回调。在 ``jax.jit`` 内部时，由于分片由 XLA 决定，
  回调发生在*编译期*。你可以用 JAX 的 AOT API
  （``jax.jit(f).lower(...).compile()``）来触发回调。
  无论哪种情况，只要运行该函数就会触发回调，
  因为运行函数必然先降级并编译它。
  不过，一旦函数编译完成并被缓存，
  回调就不会再发生了。

  该函数是实验性的，其行为将来可能改变。

  Args:
    value: 由 JAX 数组组成的 Pytree。
    callback: 接收一个 ``Sharding`` 且不返回值的可调用对象。

  下面的例子会打印出经 ``jax.jit`` 处理的计算中
  某个中间值的分片：

  >>> import jax
  >>> import jax.numpy as jnp
  >>> from jax.sharding import Mesh, PartitionSpec
  >>>
  >>> x = jnp.arange(8, dtype=jnp.float32)
  >>> def f_(x):
  ...   x = jnp.sin(x)
  ...   jax.debug.inspect_array_sharding(x, callback=print)
  ...   return jnp.square(x)
  >>> f = jax.jit(f_, in_shardings=PartitionSpec('dev'),
  ...             out_shardings=PartitionSpec('dev'))
  >>> with jax.set_mesh(Mesh(jax.devices(), ('dev',))):
  ...   f.lower(x).compile()  # doctest: +SKIP
  ...
  NamedSharding(mesh={'dev': 8}, partition_spec=PartitionSpec(('dev',),))
  """
  def _inspect(val):
    inspect_sharding_p.bind(val, callback=callback)
  tree_util.tree_map(_inspect, value)

def visualize_array_sharding(arr, **kwargs):
  """可视化一个数组的分片。"""
  def _visualize(sharding):
    return visualize_sharding(arr.shape, sharding, **kwargs)
  inspect_array_sharding(arr, callback=_visualize)


# TODO(mattjj): 绕过疑似 XLA 或 PjRt 的 bug，最终应移除
def _debug_callback_eager_rule(
    mesh,
    *args,
    callback: Callable[..., Any],
    effect: DebugEffect,
    partitioned: bool,
):
  del effect
  with core.eval_context():
    all_blocks = zip(*map(list, args))
  for (idx, device), blocks in zip(np.ndenumerate(mesh.devices), all_blocks):
    callback(*blocks)
  return []
shard_map.eager_rules[debug_callback_p] = _debug_callback_eager_rule


def _debug_print_eager_rule(
    mesh,
    *args,
    fmt: str,
    ordered,
    partitioned,
    in_tree,
    static_args,
    np_printoptions,
    has_placeholders,
    logging_record,
):
  del ordered, partitioned
  callback = partial(
      _format_print_callback, fmt, dict(np_printoptions), has_placeholders,
      logging_record,
  )
  callback = _make_flat_callback(in_tree, callback, static_args)
  with core.eval_context():
    all_blocks = zip(*map(list, args))
  for (idx, device), blocks in zip(np.ndenumerate(mesh.devices), all_blocks):
    callback(*blocks)
  return []


shard_map.eager_rules[debug_print_p] = _debug_print_eager_rule
