# Copyright 2025 The JAX Authors.
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

# 文件职责：实现实验性的 `jax.experimental.buffer_callback`，注册一种直接在设备
# 缓冲区上原地写入输出的宿主回调。
# `buffer_callback` 把用户 Python 函数接入 JAX 的原语系统：本文件登记了它的
# 效果化抽象求值、JVP/转置规则、批处理规则与 MLIR 降级规则；降级时借助 XLA 的
# 宿主回调机制调用该函数，使 numpy、PyTorch、Cupy 等库可以直接读写设备内存。

from collections.abc import Callable, Sequence
import functools
from typing import Any

import numpy as np

from jax._src import core
from jax._src import dispatch
from jax._src import effects
from jax._src import ffi
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lib import ffi as ffi_lib

export = util.set_module("jax.experimental.buffer_callback")
Buffer = export(ffi_lib.Buffer)
ExecutionStage = export(ffi_lib.ExecutionStage)
ExecutionContext = export(ffi_lib.ExecutionContext)


def buffer_callback(
    callback: Callable[..., None],
    result_shape_dtypes: object,
    *,
    has_side_effect: bool = False,
    vmap_method: str | None = None,
    input_output_aliases: dict[int, int] | None = None,
    command_buffer_compatible: bool = False,
):
  """一种在设备缓冲区上原地操作的实验性回调。

  仅在 CPU 和 GPU 后端上受支持。

  注意，计划是最终由基于 JAX 可变数组构建的统一回调 API 来取代它，
  但就目前而言，它提供了一种机制，用于借助其他 Python 库来
  原型验证计算内核，这些库包括 Numpy、PyTorch、
  Cupy 以及其它类似的库。

  让我们从一个简单的例子开始：

    >>> def py_add_one_inplace(ctx, out, x):
    ...   np.asarray(out)[...] = np.asarray(x) + 1
    ...
    >>> x = jnp.array(41, dtype=jnp.int32)
    >>> out_type = jax.ShapeDtypeStruct.like(x)
    >>> add_one = buffer_callback(py_add_one_inplace, out_type)
    >>> add_one(x)  # doctest: +SKIP
    Array(42, dtype=int32)

  在这个例子中，我们通过 JAX 执行一个 numpy 计算，
  它本来也可以使用 :func:`jax.pure_callback` 来实现，
  但在这里，输出是由回调函数原地填充的，这意味着
  JAX 在从回调返回时无需再复制输出数组。注意，即使
  回调函数操作的是可变缓冲区，JAX 仍然把它视为一个
  消费并产生普通不可变 JAX 数组的操作。

  与其它 JAX 回调 API 不同，``buffer_callback`` 要求用户定义的
  Python 函数具有如下签名：

  .. code-block:: python

    def callback(ctx: ExecutionContext, out, *args) -> None:
      ...

  其中 ``ctx`` 是
  :class:`~jax.experimental.buffer_callback.ExecutionContext` 的实例，
  它主要在 GPU 上运行时提供对 XLA 计算流的访问；``out`` 是可变的
  :class:`~jax.experimental.buffer_callback.Buffer` 对象组成的 pytree，
  而 ``args`` 参数与输入具有相同的 pytree 结构，但每个叶子都是
  :class:`~jax.experimental.buffer_callback.Buffer`。该回调不应返回
  任何值，而是应当原地覆写 ``out`` 缓冲区，
  从而把值输出回 JAX。

  需要特别注意的是，这个 Python 函数实际上无法在别处调用，只能经由
  ``buffer_callback`` 本身调用，因为目前（还！）无法在 Python 中
  直接构造可变的 JAX 缓冲区。

  专门设计的 :class:`~jax.experimental.buffer_callback.Buffer` 类型是一种
  类数组对象，它在 CPU 上支持 ``__array__`` 协议，在 GPU 上支持
  ``__cuda_array_interface__`` 协议，并且在 CPU 和 GPU 上都支持
  ``__dlpack__`` 协议。

  Args:
    callback: 具有上述签名与行为的 Python 函数。
    result_shape_dtypes: 一个 pytree，其叶子带有 ``shape`` 和
      ``dtype`` 属性，并且其结构与回调函数在运行时的预期输出
      相匹配。:class:`jax.ShapeDtypeStruct` 常被用来定义其中的
      叶子值。
    has_side_effect: 回调是否具有副作用。
    vmap_method: 一个字符串，用于指定回调在
      :func:`~jax.vmap` 下如何变换，详见 :func:`jax.pure_callback` 的文档。
    input_output_aliases: 一个字典，把某些输入的索引映射到与它们
      形成别名的输出的索引。这些索引是相对于展平后的输入和
      输出而言的。
    command_buffer_compatible: 若为 ``True``，回调会被追踪进
      命令缓冲区。这意味着其中的 Python 代码应当只被
      执行一次，随后每次调用都会重放这些操作。

  Returns:
    一个新的可调用对象，它接受 :class:`jax.Array` 输入
    （以及由它们构成的 pytree），并返回由
    :class:`jax.Array` 对象构成的 pytree，其结构与
    ``result_shape_dtypes`` 相匹配。

  See Also:
    - :func:`jax.pure_callback`：为纯主机函数设计的回调。
    - :func:`jax.experimental.io_callback`：为非纯主机函数
      设计的回调。
    - :func:`jax.debug.callback`：为通用调试设计的
      回调。
    - :func:`jax.debug.print`：为打印设计的回调。
  """
  flat_shape_dtypes, out_tree = tree_util.tree_flatten(result_shape_dtypes)
  flat_result_avals = tuple(
      core.ShapedArray(x.shape, x.dtype) for x in flat_shape_dtypes
  )

  def wrapped_callback(*args, **kwargs):
    flat_args, in_tree = tree_util.tree_flatten((args, kwargs))

    in_avals = [core.typeof(x) for x in flat_args]
    static_input_output_aliases: list[tuple[int, int]] = []
    if input_output_aliases is not None:
      for i_idx, o_idx in sorted(input_output_aliases.items()):
        i_idx, o_idx = int(i_idx), int(o_idx)
        if i_idx >= len(args):
          raise ValueError(
              f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
              f"with input index {i_idx} outside the range [0, "
              f"{len(args)}).")
        if o_idx >= len(flat_result_avals):
          raise ValueError(
              f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
              f"with output index {o_idx} outside the range [0, "
              f"{len(flat_result_avals)}).")
        in_aval = in_avals[i_idx]
        out_aval = flat_result_avals[o_idx]
        if not ffi._check_compatible_avals(in_aval, out_aval):
          raise ValueError(
              f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
              f"referring to an input with abstract value {in_aval} and an "
              f"output with a different abstract value {out_aval}.")
        static_input_output_aliases.append((i_idx, o_idx))

    out_flat = buffer_callback_p.bind(
        *flat_args,
        callback=callback,
        result_avals=flat_result_avals,
        in_tree=in_tree,
        out_tree=out_tree,
        vmap_method=vmap_method,
        has_side_effect=has_side_effect,
        input_output_aliases=tuple(static_input_output_aliases),
        command_buffer_compatible=command_buffer_compatible,
    )
    return tree_util.tree_unflatten(out_tree, out_flat)

  return wrapped_callback


buffer_callback_p = core.Primitive("buffer_callback")
buffer_callback_p.multiple_results = True
dispatch.simple_impl(buffer_callback_p)


class BufferCallbackEffect(effects.Effect):
  def __str__(self):
    return "BufferCallback"

_BufferCallbackEffect = BufferCallbackEffect()
effects.lowerable_effects.add_type(BufferCallbackEffect)
effects.control_flow_allowed_effects.add_type(BufferCallbackEffect)


@buffer_callback_p.def_effectful_abstract_eval
def _buffer_callback_abstract_eval(
    *args,
    result_avals: tuple[core.ShapedArray, ...],
    has_side_effect: bool,
    **_,
):
  del args
  effects = {_BufferCallbackEffect} if has_side_effect else core.no_effects
  return result_avals, effects


def _buffer_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Buffer callbacks do not support JVP. "
      "Please use `jax.custom_jvp` to use callbacks while taking gradients.")
ad.primitive_jvps[buffer_callback_p] = _buffer_callback_jvp_rule


def _buffer_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Buffer callbacks do not support transpose. "
      "Please use `jax.custom_vjp` to use callbacks while taking gradients.")
ad.primitive_transposes[buffer_callback_p] = _buffer_callback_transpose_rule

batching.primitive_batchers[buffer_callback_p] = functools.partial(
    ffi.ffi_batching_rule, buffer_callback_p
)


def _buffer_callback_lowering(
    ctx: mlir.LoweringRuleContext,
    *args: Any,
    callback,
    in_tree: Any,
    out_tree: Any,
    has_side_effect: bool,
    input_output_aliases: Sequence[tuple[int, int]],
    command_buffer_compatible: bool,
    **_,
):

  if len(ctx.module_context.platforms) > 1:
    raise NotImplementedError("multi-platform lowering for buffer_callback")
  platform = ctx.module_context.platforms[0]
  target_name = {
      "cpu": "xla_buffer_python_cpu_callback",
      "cuda": "xla_buffer_python_gpu_callback",
      "rocm": "xla_buffer_python_gpu_callback",
      "oneapi": "xla_buffer_python_gpu_callback",
  }.get(platform)
  if target_name is None:
    raise ValueError(f"`buffer_callback` not supported on {platform} backend.")

  if command_buffer_compatible and platform in ("cuda", "rocm", "oneapi"):
    target_name += "_cmd_buffer"

  def wrapped_callback(exec_ctx, *args: Any):
    args_in, args_out = util.split_list(args, [in_tree.num_leaves])
    py_args_in, py_kwargs_in = tree_util.tree_unflatten(in_tree, args_in)
    py_args_out = tree_util.tree_unflatten(out_tree, args_out)
    if callback(exec_ctx, py_args_out, *py_args_in, **py_kwargs_in) is not None:
      raise ValueError("buffer_callback callback must not return any values.")
    return ()

  ctx.module_context.add_host_callback(wrapped_callback)
  index = np.uint64(len(ctx.module_context.host_callbacks) - 1)
  rule = ffi.ffi_lowering(
      target_name,
      has_side_effect=has_side_effect,
      operand_output_aliases=dict(input_output_aliases),
  )
  return rule(ctx, *args, index=index)
mlir.register_lowering(buffer_callback_p, _buffer_callback_lowering)
