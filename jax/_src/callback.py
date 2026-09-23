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
# 文件职责：实现 JAX 的外部回调（callback）机制，让已暂存的 JAX 程序在运行时回调主机上的 Python 函数。
# 核心是 `pure_callback_p`（纯回调，无副作用，可被变换重排或省略）与 `io_callback_p`
# （不纯回调，带 IO 效果，可选有序执行）这两个原语，以及它们的抽象求值、JVP/转置、
# 批处理规则、分片注解和 MLIR 降级规则。
# 降级路径按后端区分：TPU 走 `send_to_host`/`receive_from_host` 这一主机传输通信机制，
# CPU/GPU 则生成 XLA FFI 自定义调用；降级时会结合 SPMD/Shardy 的分片上下文决定回调在哪些设备上执行。
"""Module for JAX callbacks."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
import logging
from typing import Any, cast

from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import ffi
from jax._src import pickle_util
from jax._src import sharding_impls
from jax._src import tree_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.sharding_impls import SdyArray, SdyArrayList, SdyDim, SingleDeviceSharding
from jax._src.sharding import Sharding
from jax._src.typing import Array
import numpy as np

logger = logging.getLogger(__name__)

# `pure_callback_p` 是把 Python 纯回调暂存出去的主原语。
pure_callback_p = core.Primitive("pure_callback")
pure_callback_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(pure_callback_p)

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


@dataclasses.dataclass(frozen=True, slots=True, weakref_slot=True)
class _FlatCallback:
  """可用扁平参数和结果调用的 Python 函数。

  该类的实例被用作回调原语的参数。
  我们更偏好它而不是匿名的扁平化函数，因为当我们用相同的参数结构调用同一个
  Python 函数时，它会产生相等的对象。
  """
  callback_func: Callable[..., Any]
  in_tree: tree_util.PyTreeDef  # `callback_func` 的 (args, kwargs) pytree。

  def __call__(self, *flat_args: Array) -> Sequence[Array]:
    args, kwargs = tree_util.tree_unflatten(self.in_tree, flat_args)
    return tree_util.tree_leaves(self.callback_func(*args, **kwargs))


def pure_callback_impl(
    *args,
    result_avals,
    callback: _FlatCallback,
    sharding: Sharding | None,
    vmap_method: str | None,
):
  del sharding, vmap_method, result_avals
  try:
    cpu_device, *_ = xb.local_devices(backend="cpu")
  except RuntimeError as e:
    raise RuntimeError(
        "jax.pure_callback failed to find a local CPU device to place the"
        " inputs on. Make sure \"cpu\" is listed in --jax_platforms or the"
        " JAX_PLATFORMS environment variable."
    ) from e
  args = api.device_put(args, cpu_device)
  with config.default_device(cpu_device):
    try:
      return tree_util.tree_map(np.asarray, callback(*args))
    except BaseException:
      logger.exception("jax.pure_callback failed")
      raise


pure_callback_p.def_impl(functools.partial(dispatch.apply_primitive,
                                           pure_callback_p))


@pure_callback_p.def_abstract_eval
def pure_callback_abstract_eval(
    *avals,
    callback: _FlatCallback,
    result_avals,
    sharding: Sharding | None,
    vmap_method: str | None,
):
  del avals, callback, sharding, vmap_method
  return result_avals


def pure_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Pure callbacks do not support JVP. "
      "Please use `jax.custom_jvp` to use callbacks while taking gradients.")


ad.primitive_jvps[pure_callback_p] = pure_callback_jvp_rule


def pure_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Pure callbacks do not support transpose. "
      "Please use `jax.custom_vjp` to use callbacks while taking gradients.")

ad.primitive_transposes[pure_callback_p] = pure_callback_transpose_rule


batching.primitive_batchers[pure_callback_p] = functools.partial(
    ffi.ffi_batching_rule, pure_callback_p
)

def _get_sdy_array_list_for_callbacks(avals: Sequence[core.ShapedArray]) -> SdyArrayList:
  """返回一个 SdyArrayList，其中带有 `max(1, len(avals))` 个复制分片。"""
  ndims = [0]
  if avals:
    ndims = [x.ndim for x in avals if isinstance(x, core.ShapedArray)]
  return SdyArrayList(tuple(
      SdyArray(
          mesh_shape=(),
          dim_shardings=(SdyDim(axes=(), is_open=False),) * ndim,
          logical_device_ids=())
      for ndim in ndims))


def _callback_op_sharding(
    axis_context, sharding: Sharding | None, avals_out
):
  if isinstance(axis_context, sharding_impls.SPMDAxisContext):
    # 如果在降级期间处于完全手动分片状态，那就意味着该 JAX 程序具有按设备语义，
    # 因此我们在每个设备上运行回调。
    if axis_context.manual_axes != frozenset(axis_context.mesh.axis_names):
      raise NotImplementedError(
          "callbacks are only supported in spmd computations when all mesh"
          " axes are partitioned manually (no partial automatic sharding)."
      )
    if sharding is not None:
      raise NotImplementedError(
          "callbacks do not support specifying sharding inside spmd"
          " computations"
      )
    if config.use_shardy_partitioner.value:
      op_sharding = _get_sdy_array_list_for_callbacks(avals_out)
    else:
      op_sharding = xc.OpSharding()
      op_sharding.type = xc.OpSharding.Type.MANUAL
    return op_sharding

  if isinstance(axis_context, sharding_impls.ShardingContext):
    if sharding is not None:
      if (isinstance(sharding, sharding_impls.NamedSharding) and
          sharding.mesh.is_scalar):  # pyrefly: ignore[missing-attribute]
        pass
      elif not isinstance(sharding, SingleDeviceSharding):
        raise NotImplementedError(
            "pure_callback only supports SingleDeviceSharding, but got"
            f" {type(sharding)}"
        )
      device = next(iter(sharding.device_set))
      device_assignment = axis_context.device_assignment
      if device_assignment is None:
        raise AssertionError(
            "Please file a bug at https://github.com/jax-ml/jax/issues")
      try:
        device_index = device_assignment.index(device)
      except IndexError as e:
        raise ValueError(
            "Sharding provided to pure_callback specifies a device"
            f" {device} that is not in the device assignment"
            f" ({device_assignment})") from e
    else:
      device_index = 0

    # 如果在降级期间处于完全自动分片状态，那就意味着该 JAX 程序具有整体数组语义，
    # 因此我们以 MAXIMAL 分片运行回调，也就是只在完整的逻辑值上执行一次。
    if config.use_shardy_partitioner.value:
      # 对于 shardy，分片注解的个数必须与结果算子的个数相同。如果没有结果算子，
      # 则需要 1 个 shardy 注解。
      num_sdy_shardings = max(1, len(avals_out))
      op_sharding = SdyArrayList((
          SdyArray(mesh_shape=(), dim_shardings=(),
                   logical_device_ids=(device_index,)),) * num_sdy_shardings)
    else:
      op_sharding = xc.OpSharding()
      op_sharding.type = xc.OpSharding.Type.MAXIMAL
      op_sharding.tile_assignment_dimensions = [1]
      op_sharding.tile_assignment_devices = [device_index]
    return op_sharding

  # 没有 SPMD 分区时，不要标注分片。
  return None


def pure_callback_lowering(
    ctx, *args, callback: _FlatCallback, sharding: Sharding | None, **params
):
  def _callback(*flat_args):
    return tuple(
        pure_callback_impl(
            *flat_args,
            callback=callback,
            sharding=None,  # 未使用。
            **params,
        )
    )

  op_sharding = _callback_op_sharding(
      ctx.module_context.axis_context, sharding, ctx.avals_out)
  result, _, _ = emit_python_callback(
      ctx,
      _callback,
      None,
      list(args),
      ctx.avals_in,
      ctx.avals_out,
      has_side_effect=False,
      returns_token=False,
      sharding=op_sharding,
  )
  return result


# TODO(phawkins): 在 TPU 上，这些回调带有内嵌的通道 ID，该 ID 对每个回调都应当唯一。
# 缓存会破坏这一点。
mlir.register_lowering(pure_callback_p, pure_callback_lowering, cacheable=False)

def _check_shape_dtype(shape_dtype):
  dt = np.dtype(shape_dtype.dtype)
  if dtypes.canonicalize_dtype(dt) != dt:
    raise ValueError(
        "result_shape_dtypes cannot specify 64-bit types when `jax_enable_x64` is disabled")


def pure_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: Sharding | None = None,
    vmap_method: str | None = None,
    **kwargs: Any,
):
  """调用一个纯 Python 回调。可在 :func:`jit`/:func:`~vmap`/等变换下工作。

  更多说明请参阅 `External Callbacks`_。

  ``pure_callback`` 使得在经 JIT 编译的 JAX 函数中可以调用 Python 函数。
  输入 ``callback`` 会收到被放置到本地 CPU 上的 JAX 数组，
  它也应返回位于 CPU 上的 JAX 数组。

  该回调被视为函数式纯函数，也就是说它没有副作用，其输出值只依赖参数值。
  因此，它可以安全地被多次调用（例如被 :func:`~vmap` 或 :func:`~pmap` 变换时），
  或者在 `jit` 装饰的函数的输出与其值之间没有数据依赖时完全不被调用。
  在数据依赖允许的情况下，纯回调还可以被重排。

  .. warning::

     在 JAX 变换的语境下，Python 异常应当被视为副作用：这意味着在
     `pure_callback` 中故意抛出错误会违反 API 约定，由此产生的程序行为是未定义的。

  被 `vmap` 作用时，其行为取决于 ``vmap_method`` 的取值。

  * 在没有显式指定 ``vmap_method`` 的情况下对回调调用 :func:`~jax.vmap`
    会抛出 ``NotImplementedError``。
  * ``vmap_method="sequential"`` 使用 :func:`~jax.lax.map` 遍历批处理后的
    参数，对每个批元素调用一次 ``callback``。
  * ``vmap_method="sequential_unrolled"`` 与 ``sequential`` 类似，但循环会被展开。
  * ``vmap_method="expand_dims"`` 在未批处理的输入的前导维度上新增大小为 ``1``
    的轴来调用 ``callback``。
  * ``vmap_method="broadcast_all"`` 的行为与 ``expand_dims`` 类似，但输入会被
    平铺到预期的批处理形状。



  当前的默认行为是在未指定时使用 ``vmap_method="sequential"``，但该行为已被弃用，
  未来除非显式指定 ``vmap_method``，否则默认将抛出 ``NotImplementedError``。

  Args:
    callback: 在主机上执行的函数。该回调被假定为纯函数（即没有副作用）：
      如果传入不纯的函数，它可能以意外的方式运行，尤其是在变换之下。该可调用对象
      会收到以数组 pytree 形式给出的参数，并应返回与 ``result_shape_dtypes``
      匹配的数组 pytree。
    result_shape_dtypes: 一个 pytree，其叶子具有 ``shape`` 和 ``dtype`` 属性，
      其结构与回调函数在运行时预期的输出结构匹配。
      :class:`jax.ShapeDtypeStruct` 常用于定义叶子值。
    *args: 传给回调函数的参数
    sharding: 可选的分片，用于指定应该从哪个设备调用该回调。
    vmap_method: 字符串，按上文所述指定该回调在
      :func:`~jax.vmap` 下如何变换。
    **kwargs: 传给回调函数的关键字参数

  Returns:
    result: 一个 :class:`jax.Array` 对象的 pytree，其结构与
      ``result_shape_dtypes`` 的结构匹配。

  See Also:
    - :func:`jax.experimental.io_callback`：为不纯函数设计的回调。
    - :func:`jax.debug.callback`：为通用调试设计的回调。
    - :func:`jax.debug.print`：为打印设计的回调。

  Examples:
    ``pure_callback`` 在 :func:`~jax.vmap` 下的行为由上文所述的 ``vmap_method``
    参数控制。考虑一些展示其语义的明确例子会很有帮助。例如，
    考虑如下函数：

    >>> def callback(x, y):
    ...   print(jnp.shape(x), jnp.shape(y))
    ...   return x + y

    >>> def fun(x, y, *, vmap_method):
    ...   shape = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y))
    ...   dtype = jnp.result_type(x, y)
    ...   out_type = jax.ShapeDtypeStruct(shape, dtype)
    ...   return jax.pure_callback(callback, out_type, x, y,
    ...                            vmap_method=vmap_method)

    以 ``vmap_method="expand_dims"`` 调用它会给 ``y`` 添加一个大小为 ``1`` 的新轴：

    >>> from functools import partial
    >>> x = jnp.arange(4)
    >>> y = 1.0
    >>> jax.vmap(partial(fun, vmap_method="expand_dims"), in_axes=(0, None))(x, y)
    (4,) (1,)
    Array([1., 2., 3., 4.], dtype=float32)

    而 ``vmap_method="broadcast_all"`` 会给 ``y`` 添加一个大小为 ``4`` 的轴：

    >>> jax.vmap(partial(fun, vmap_method="broadcast_all"),
    ...          in_axes=(0, None))(x, y)
    (4,) (4,)
    Array([1., 2., 3., 4.], dtype=float32)

  .. _External Callbacks: https://docs.jax.dev/en/latest/external-callbacks.html
  """

  allowed_vmap_methods = ["sequential", "sequential_unrolled", "expand_dims",
                          "broadcast_all", "legacy_vectorized", None]
  if vmap_method not in allowed_vmap_methods:
    raise ValueError(
        f"vmap_method must be on of the allowed methods {allowed_vmap_methods}, "
        f"but got: {vmap_method}")

  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  tree_util.tree_map(_check_shape_dtype, result_shape_dtypes)
  result_avals = tree_util.tree_map(
      lambda x: core.ShapedArray(x.shape, x.dtype), result_shape_dtypes)
  flat_result_avals, out_tree = tree_util.tree_flatten(result_avals)
  out_flat = pure_callback_p.bind(
      *flat_args,
      callback=_FlatCallback(callback, in_tree),
      result_avals=tuple(flat_result_avals),
      sharding=sharding,
      vmap_method=vmap_method,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)


# IO Callback

io_callback_p = core.Primitive("io_callback")
io_callback_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(io_callback_p)

class IOEffect(effects.Effect):
  __str__ = lambda _: "IO"

class OrderedIOEffect(effects.Effect):
  __str__ = lambda _: "OrderedIO"

_IOEffect = IOEffect()
_OrderedIOEffect = OrderedIOEffect()
effects.lowerable_effects.add_type(IOEffect)
effects.lowerable_effects.add_type(OrderedIOEffect)
effects.control_flow_allowed_effects.add_type(IOEffect)
effects.control_flow_allowed_effects.add_type(OrderedIOEffect)
effects.ordered_effects.add_type(OrderedIOEffect)
effects.shardable_ordered_effects.add_type(OrderedIOEffect)


def io_callback_impl(
    *args,
    result_avals,
    callback: _FlatCallback,
    sharding: Sharding | None,
    ordered: bool,
):
  del result_avals, sharding, ordered
  try:
    cpu_device, *_ = xb.local_devices(backend="cpu")
  except RuntimeError as e:
    raise RuntimeError(
        "jax.io_callback failed to find a local CPU device to place the"
        " inputs on. Make sure \"cpu\" is listed in --jax_platforms or the"
        " JAX_PLATFORMS environment variable."
    ) from e
  args = api.device_put(args, cpu_device)
  with config.default_device(cpu_device):
    try:
      return tree_util.tree_map(np.asarray, callback(*args))
    except BaseException:
      logger.exception("jax.io_callback failed")
      raise


io_callback_p.def_impl(functools.partial(dispatch.apply_primitive,
                                         io_callback_p))


@io_callback_p.def_effectful_abstract_eval
def io_callback_abstract_eval(
    *avals,
    callback: _FlatCallback,
    result_avals,
    sharding: Sharding | None,
    ordered: bool,
):
  del avals, sharding, callback
  effect = _OrderedIOEffect if ordered else _IOEffect
  return result_avals, {effect}


def io_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError("IO callbacks do not support JVP.")
ad.primitive_jvps[io_callback_p] = io_callback_jvp_rule


def io_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError("IO callbacks do not support transpose.")
ad.primitive_transposes[io_callback_p] = io_callback_transpose_rule


def io_callback_batching_rule(
    args, dims, callback, result_avals, sharding, ordered
):
  from jax._src.lax.control_flow.loops import map as lax_map  # pyrefly: ignore[missing-import]
  if ordered:
    raise ValueError("Cannot `vmap` ordered IO callback.")
  is_batched = [d is not None for d in dims]
  new_args = [arg if dim is None else
              batching.moveaxis(arg, dim, 0) for arg, dim in zip(args, dims)]
  unbatched_args, batched_args = util.partition_list(is_batched, new_args)
  def _batch_fun(batched_args):
    merged = util.merge_lists(is_batched, unbatched_args, batched_args)
    return io_callback_p.bind(*merged, callback=callback, sharding=sharding,
                              result_avals=result_avals, ordered=False)
  out_vals = lax_map(_batch_fun, batched_args)
  return out_vals, (0,) * len(out_vals)
batching.primitive_batchers[io_callback_p] = io_callback_batching_rule


def io_callback_lowering(ctx, *args, callback, sharding, ordered, **params):
  def _callback(*flat_args):
    return tuple(
        io_callback_impl(
            *flat_args,
            callback=callback,
            sharding=None,  # 未使用。
            ordered=ordered,
            **params,
        )
    )

  op_sharding = _callback_op_sharding(
      ctx.module_context.axis_context, sharding, ctx.avals_out)
  if ordered:
    token = ctx.tokens_in.get(_OrderedIOEffect)
    result, token, _ = emit_python_callback(
        ctx,
        _callback,
        token,
        list(args),
        ctx.avals_in,
        ctx.avals_out,
        has_side_effect=True,
        returns_token=True,
        sharding=op_sharding,
    )
    ctx.set_tokens_out(
        ctx.tokens_in.update_tokens(mlir.TokenSet({_OrderedIOEffect: token})))
  else:
    result, _, _ = emit_python_callback(
        ctx,
        _callback,
        None,
        list(args),
        ctx.avals_in,
        ctx.avals_out,
        has_side_effect=True,
        returns_token=False,
        sharding=op_sharding,
    )
  return result

# TODO(phawkins): 在 TPU 上，这些回调带有内嵌的通道 ID，该 ID 对每个回调都应当唯一。
# 缓存会破坏这一点。
mlir.register_lowering(io_callback_p, io_callback_lowering, cacheable=False)

def io_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: Sharding | None = None,
    ordered: bool = False,
    **kwargs: Any,
):
  """调用一个不纯的 Python 回调。

  更多说明请参阅 `External Callbacks`_。

  Args:
    callback: 在主机上执行的函数。它被假定为不纯函数。
      如果 ``callback`` 是纯函数，改用 :func:`jax.pure_callback` 可能带来
      更高效的执行。
    result_shape_dtypes: 一个 pytree，其叶子具有 ``shape`` 和 ``dtype`` 属性，
      其结构与回调函数在运行时预期的输出结构匹配。
      :class:`jax.ShapeDtypeStruct` 常用于定义叶子值。
    *args: 传给回调函数的参数
    sharding: 可选的分片，用于指定应该从哪个设备调用该回调。
    ordered: 布尔值，指定对回调的连续调用是否必须保持有序。
    **kwargs: 传给回调函数的关键字参数

  Returns:
    result: 一个 :class:`jax.Array` 对象的 pytree，其结构与
      ``result_shape_dtypes`` 的结构匹配。

  See Also:
    - :func:`jax.pure_callback`：为纯函数设计的回调。
    - :func:`jax.debug.callback`：为通用调试设计的回调。
    - :func:`jax.debug.print`：为打印设计的回调。

  .. _External Callbacks: https://docs.jax.dev/en/latest/notebooks/external_callbacks.html
  """
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  tree_util.tree_map(_check_shape_dtype, result_shape_dtypes)
  flat_shape_dtypes, out_tree = tree_util.tree_flatten(result_shape_dtypes)
  flat_result_avals = map(lambda x: core.ShapedArray(x.shape, x.dtype),
                          flat_shape_dtypes)
  out_flat = io_callback_p.bind(
      *flat_args,
      callback=_FlatCallback(callback, in_tree),
      result_avals=tuple(flat_result_avals),
      sharding=sharding,
      ordered=ordered,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)


def is_empty_shape(s: core.Shape) -> bool:
  return any(d == 0 for d in s)


_XLA_HOST_TRANSFER_PJRT_RENDEZVOUS_HANDLER_NAME = "pjrt_rendezvous"


def send_to_host(
    ctx: mlir.ModuleContext,
    channel: int,
    token: ir.Value[hlo.TokenType],
    operand: Any,
    name: str | None = None,
    *,
    sharding: SdyArrayList | xc.OpSharding | None = None,
) -> ir.Value:
  channel_handle = hlo.ChannelHandle.get(channel, mlir.SEND_TO_HOST_TYPE)
  send_op = hlo.SendOp([operand], token, channel_handle,
                        is_host_transfer=ir.BoolAttr.get(True))
  send_op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      dict(
          _xla_host_transfer_handler_name=ir.StringAttr.get(
              _XLA_HOST_TRANSFER_PJRT_RENDEZVOUS_HANDLER_NAME
          ),
          _xla_host_transfer_rendezvous=ir.StringAttr.get(str(channel)),
      )
  )
  if sharding is not None:
    if config.use_shardy_partitioner.value:
      # `SendOp` 的返回类型是 StableHLO 的 `TokenType`。但 JAX 传入的是
      # 数组类型的最大（MAXIMAL）分片。由于 token 没有秩，我们需要创建一个
      # 等价的、没有维度的分片。如果有多个分片，
      # 直接取第一个即可，因为这些分片应当都是相同的。
      assert isinstance(sharding, SdyArrayList)
      assert len(sharding.shardings) >= 1
      sharding = SdyArrayList((SdyArray(
          mesh_shape=(), dim_shardings=(),
          logical_device_ids=sharding.shardings[0].logical_device_ids),))
    mlir.set_sharding(ctx, send_op, sharding)
  return send_op.result


def receive_from_host(
    ctx: mlir.ModuleContext,
    channel: int,
    token: ir.Value[hlo.TokenType],
    out_aval: core.ShapedArray,
    name: str | None = None,
    *,
    sharding: SdyArrayList | xc.OpSharding | None = None,
) -> tuple[ir.Value, ir.Value]:
  channel_handle = hlo.ChannelHandle.get(channel, mlir.RECV_FROM_HOST_TYPE)
  out_type = mlir.aval_to_ir_type(ctx, out_aval)
  recv_op = hlo.RecvOp([out_type,
                        hlo.TokenType.get()], token, channel_handle,
                        is_host_transfer=ir.BoolAttr.get(True))
  recv_op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      dict(
          _xla_host_transfer_handler_name=ir.StringAttr.get(
              _XLA_HOST_TRANSFER_PJRT_RENDEZVOUS_HANDLER_NAME
          ),
          _xla_host_transfer_rendezvous=ir.StringAttr.get(str(channel)),
      )
  )
  if sharding is not None:
    if config.use_shardy_partitioner.value:
      assert isinstance(sharding, SdyArrayList)
      assert len(sharding.shardings) >= 1
      # `RecvOp` 的最后一个参数是 `TokenType`。由于 Shardy 要求分片数量与结果
      # 数量一致，而 JAX 只看到数组结果，我们需要为 token 补上一个等价的分片。
      # 注意即使一个函数返回 N 个结果，最终也会有 N 个 `RecvOp`，
      # 因此我们只需要取第一个分片。所有分片反正都是一样的，
      # 都作用于同一个设备 ID。
      sharding = SdyArrayList((
          sharding.shardings[0],
          SdyArray(mesh_shape=(), dim_shardings=(),
                   logical_device_ids=sharding.shardings[0].logical_device_ids)))
    mlir.set_sharding(ctx, recv_op, sharding)
  # token 应位于结果的末尾
  result, token = recv_op.results
  return token, result


def _aval_to_xla_shape(aval: core.AbstractValue) -> xc.Shape:
  try:
    return _xla_shape_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No xla_shape_handler for type: {type(aval)}") from err

_xla_shape_handlers: dict[type[core.AbstractValue],
                         Callable[[Any], xc.Shape]] = {}

def _make_array_shape(aval: core.ShapedArray) -> xc.Shape:
  aval = core.physical_aval(aval)
  dtype = np.dtype('bool') if aval.dtype == dtypes.float0 else aval.dtype
  return xc.Shape.array_shape(dtype, aval.shape)
_xla_shape_handlers[core.ShapedArray] = _make_array_shape

_xla_shape_handlers[core.AbstractToken] = lambda _: xc.Shape.token_shape()


def _emit_tpu_python_callback(
    backend: xc.Client,
    ctx: mlir.LoweringRuleContext,
    callback,
    token: Any | None,
    operands: Sequence[ir.Value],
    operand_avals: Sequence[core.ShapedArray],
    operand_shapes: Sequence[xc.Shape],
    result_avals: Sequence[core.ShapedArray],
    result_shapes: Sequence[xc.Shape],
    *,
    returns_token: bool,
    sharding: SdyArrayList | xc.OpSharding | None = None,
) -> tuple[Sequence[ir.Value], Any]:
  token = token or hlo.create_token()
  _wrapped_callback = callback

  send_channels = []
  if not operand_avals:
    # 如果回调没有任何操作数，我们需要插入一个哑元 send 算子，
    # 否则该回调永远不会被触发！
    # TODO(sharadmv,chky): 在运行时而不是在 MLIR 构建器中启用此修复。
    callback_without_args = _wrapped_callback
    def _wrapped_callback(*args):
      del args
      return callback_without_args()
    send_channel = ctx.module_context.new_channel()
    dummy_send_aval = core.ShapedArray((1,), np.float32)
    dummy_send_val = mlir.ir_constant(np.zeros(1, np.float32))
    operand_shapes = [*operand_shapes, _aval_to_xla_shape(dummy_send_aval)]
    token = send_to_host(ctx.module_context, send_channel, token, dummy_send_val,
                         sharding=sharding)
    send_channels.append(send_channel)
  else:
    for operand in operands:
      channel = ctx.module_context.new_channel()
      token = send_to_host(ctx.module_context, channel, token, operand, sharding=sharding)
      send_channels.append(channel)

  recv_channels = []
  outputs = []
  if returns_token and not result_avals:
    # 如果调用方期望一个 token，我们至少需要一个结果，这样来自 recv 的 token
    # 才能被用作回调已完成的标志。否则我们只会等待 send 结束。
    callback_without_results = _wrapped_callback
    def _wrapped_callback(*args):
      callback_without_results(*args)
      return 0.0,
    dummy_recv_aval = core.ShapedArray((), np.float32)
    result_shapes = [_aval_to_xla_shape(dummy_recv_aval)]
    channel = ctx.module_context.new_channel()
    token, _ = receive_from_host(
        ctx.module_context, channel, token, dummy_recv_aval, sharding=sharding
    )
    recv_channels.append(channel)
  else:
    for result_aval in result_avals:
      channel = ctx.module_context.new_channel()
      assert isinstance(result_aval, core.ShapedArray)
      token, out = receive_from_host(
          ctx.module_context, channel, token, result_aval, sharding=sharding
      )
      outputs.append(out)
      recv_channels.append(channel)
  ifrt_callback = backend.make_python_callback_from_host_send_and_recv(
      _wrapped_callback, operand_shapes, result_shapes, send_channels,
      recv_channels, pickle_util.dumps)
  ctx.module_context.add_host_callback(ifrt_callback)
  return outputs, token


def emit_python_callback(
    ctx: mlir.LoweringRuleContext,
    callback,
    token: Any | None,
    operands: Sequence[ir.Value],
    operand_avals: Sequence[core.ShapedArray],
    result_avals: Sequence[core.ShapedArray],
    *,
    has_side_effect: bool,
    returns_token: bool = True,
    partitioned: bool = False,
    sharding: SdyArrayList | xc.OpSharding | None = None,
) -> tuple[Sequence[mlir.IrValues], Any, Any]:
  """生成回调到给定 Python 函数的 MLIR。

  Args:
    ctx: 降级上下文。
    callback: Python 回调函数。
    token: 用于该回调的 token。
    operands: 该回调的操作数。
    operand_avals: 操作数的抽象值。
    result_avals: 结果的抽象值。
    has_side_effect: 该回调是否有副作用。
    returns_token: 该回调是否应返回一个 token。
    partitioned: 若为 True，则 `callback` 只在本地分片上被调用。
      若为 False，则 `callback` 在所有分片上被调用。
    sharding: 该回调的分片。

  Returns:
    MLIR 结果值的元组、新的 token（如果有），以及主机回调对象。
  """
  if len(ctx.module_context.platforms) > 1:
    raise NotImplementedError("multi-platform lowering for python_callback")
  platform = ctx.module_context.platforms[0]
  if platform not in {"cpu", "cuda", "rocm", "tpu", "oneapi"}:
    raise ValueError(
        f"`EmitPythonCallback` not supported on {platform} backend.")
  if partitioned:
    if platform not in {"cpu", "cuda", "rocm", "oneapi"}:
      raise NotImplementedError(
          f"Partitioned callback not implemented on {platform} backend.")
    if result_avals:
      raise ValueError("Partitioned callback not supported with return values.")
  backend: xc.Client = cast(xc.Client, ctx.module_context.get_backend())
  result_shapes = [_aval_to_xla_shape(aval) for aval in result_avals]
  operand_shapes = [_aval_to_xla_shape(aval) for aval in operand_avals]

  # 首先我们施加检查，确保输出的形状和数据类型与预期一致。
  def _wrapped_callback(*args):
    out_vals = callback(*args)
    if len(out_vals) != len(result_avals):
      raise RuntimeError(
          "Mismatched number of outputs from callback. "
          "Expected: {}, Actual: {}".format(len(result_avals), len(out_vals)))
    # 处理 Python 字面量以及自定义数组，例如 tf.Tensor。
    out_vals = tuple(dtypes.canonicalize_value(np.asarray(a)) for a in out_vals)
    for i, (out_val, out_aval) in enumerate(zip(out_vals, result_avals)):
      if out_val.shape != out_aval.shape:
        raise RuntimeError(
            f"Incorrect output shape for return value #{i}: "
            f"Expected: {out_aval.shape}, Actual: {out_val.shape}")
      if out_val.dtype != out_aval.dtype:
        raise RuntimeError(
            f"Incorrect output dtype for return value #{i}: "
            f"Expected: {out_aval.dtype}, Actual: {out_val.dtype}")

    if platform == "tpu":
      # 在 TPU 上我们无法接收空数组。因此我们从被包装的回调中只返回非空结果，
      # 并在接收计算中创建空常量。
      # TODO(b/238239458): 修复 TPU Recv 使其能处理空数组。
      non_empty_out_vals = tuple(
          out_val
          for out_val, result_aval in zip(out_vals, result_avals)
          if not is_empty_shape(result_aval.shape))
      return non_empty_out_vals
    else:
      return out_vals

  if platform == "tpu":
    non_empty_result_avals, non_empty_result_shapes = util.unzip2([
        (aval, shape)
        for aval, shape in zip(result_avals, result_shapes)
        if not is_empty_shape(aval.shape)])
    non_empty_outputs, token = _emit_tpu_python_callback(
        backend, ctx, _wrapped_callback,  token,
        operands, operand_avals, operand_shapes,
        non_empty_result_avals, non_empty_result_shapes,
        returns_token=returns_token, sharding=sharding)
    non_empty_outputs_iter = iter(non_empty_outputs)
    outputs = [
        mlir.ir_constant(np.zeros(result_aval.shape, dtype=result_aval.dtype))
        if is_empty_shape(result_aval.shape) else next(non_empty_outputs_iter)
        for result_aval in result_avals]
    return outputs, token, None

  device = "gpu" if platform in {"cuda", "rocm", "oneapi"} else "cpu"
  partition = "_partitioned" if partitioned else ""
  call_target_name = f"xla_ffi{partition}_python_{device}_callback"
  if token:
    callback_without_token = _wrapped_callback
    def _wrapped_callback(token, *args):
      return (token, *callback_without_token(*args))
    operands = [token, *operands]
    if (
        config.use_shardy_partitioner.value
        and sharding is not None
        and len(ctx.avals_out) > 0
        and isinstance(sharding, SdyArrayList)
    ):
      # 如果我们至少有一个输出，就为 token 添加一个分片注解。
      # 否则，所有算子（即使没有任何结果）都需要的那个 shardy 注解
      # 可以标注在 token 上。
      sharding = SdyArrayList((
          SdyArray(mesh_shape=(), dim_shardings=(),
                   logical_device_ids=sharding.shardings[0].logical_device_ids),
          *sharding.shardings))
    ctx = dataclasses.replace(
        ctx,
        avals_in=[core.abstract_token, *ctx.avals_in],
        avals_out=[core.abstract_token, *ctx.avals_out],
    )

  # TODO(dsuo): 一旦我们弃用 XLA 自定义调用处理器，就删除这一行。
  ifrt_callback = _wrapped_callback
  ctx.module_context.add_host_callback(ifrt_callback)
  index = np.uint64(len(ctx.module_context.host_callbacks) - 1)
  result = ffi.build_ffi_lowering_function(
      call_target_name,
      has_side_effect=has_side_effect,
  )(ctx, *operands, index=np.uint64(index))

  if sharding is not None:
    mlir.set_sharding(ctx.module_context, result, sharding)

  results = result.results

  if token:
    token, *results = results

  return results, token, ifrt_callback
