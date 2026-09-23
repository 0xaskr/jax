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

# 文件职责：实现 JAX 在 Python 侧的 FFI（外部函数接口）接入层。
# 它既提供注册入口（`register_ffi_target`/`register_ffi_type`/`pycapsule`
# 等），把外部编译库里的函数注册为 XLA custom call 目标；也提供调用入口
# `ffi_call`，把带目标名、布局、输入输出别名与属性参数的调用封装成
# `ffi_call` 原语，并为其注册抽象求值、JVP/转置、批处理（vmap）与 MLIR
# 降级规则，使外部内核能像普通 JAX 运算一样参与追踪、变换与编译。

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import ctypes
import dataclasses
import functools
import os
from typing import Any, TypedDict, NotRequired, overload

import numpy as np

from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import effects
from jax._src import util
from jax._src import xla_bridge
from jax._src.hashable_array import HashableArray
from jax._src.frozen_dict import FrozenDict
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.layout import Layout
from jax._src.lib import jaxlib
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.typing import Array, ArrayLike, DuckTypedArray, Shape

map, unsafe_map = util.safe_map, map
FfiLayoutOptions = Sequence[int] | Layout | None


def register_ffi_target(
    name: str,
    fn: Any,
    platform: str = "cpu",
    api_version: int = 1,
    **kwargs: Any,
) -> None:
  """注册一个外部函数目标。

  Args:
    name: 目标的名称。
    fn: 一个包含函数指针的 ``PyCapsule`` 对象，或者一个 ``dict``，
      其键是 FFI 阶段名（例如 `"execute"`），值是指向该阶段处理函数的
      ``PyCapsule`` 对象。
    platform: 目标平台。
    api_version: 要使用的 XLA custom call API 版本。支持的版本有：
      1（默认）表示带类型的 FFI，0 表示更早的 "custom call" API。
    kwargs: 任何额外的关键字参数都会直接传给
      :func:`~jaxlib.xla_client.register_custom_call_target`，以应对更高级的
      用法。
  """
  return xla_client.register_custom_call_target(name, fn, platform, api_version,
                                                **kwargs)


class TypeRegistration(TypedDict):
  """用于注册 FFI 类型的字典类型。

  Attributes:
    type_id: 一个 ``PyCapsule`` 对象，包含指向
      ``XLA_FFI_TypeId`` 的指针。
    type_info: 一个可选的 ``PyCapsule`` 对象，包含指向类型
      ``XLA_FFI_TypeInfo`` 的指针。
  """

  type_id: Any
  type_info: NotRequired[Any]


def register_ffi_type_id(
    name: str,
    obj: Any,
    platform: str = "cpu",
) -> None:
  """为 FFI 目标注册自定义类型 ID。

  Args:
    name: 类型 ID 的名称。该名称在进程内必须唯一。
    obj: 一个 ``PyCapsule`` 对象，封装了指向该类型 ID 的指针。
    platform: 目标平台。
  """
  raise ValueError(
      "register_ffi_type_id is not supported after jaxlib version 381.")

def register_ffi_type(
    name: str,
    type_registration: TypeRegistration,
    platform: str = "cpu",
) -> None:
  """为 FFI 目标注册自定义类型。

  Args:
    name: 类型的名称。该名称在进程内必须唯一。
    type_registration: 定义该外部类型的 ``TypeRegistration``。
    platform: 目标平台。
  """
  return xla_client.register_custom_type(
      name, type_registration, platform=platform
  )


def register_ffi_target_as_batch_partitionable(name: str) -> None:
  """把一个 FFI 目标注册为可按批次切分。

  Args:
    name: 目标的名称。
  """
  xla_client.register_custom_call_as_batch_partitionable(name)
  xla_bridge.register_plugin_callbacks(
      functools.partial(xla_client.register_custom_call_as_batch_partitionable,
                        name))


def pycapsule(funcptr):
  """把 ctypes 函数指针包装成 PyCapsule。

  这个函数的主要用途，也是它被放在 ``jax.ffi`` 子模块里的原因，
  是把外部编译库中的函数调用包装起来，以便注册为 XLA custom call。

  示例用法::

    import ctypes
    import jax
    from jax.lib import xla_client

    libfoo = ctypes.cdll.LoadLibrary('./foo.so')
    xla_client.register_custom_call_target(
        name="bar",
        fn=jax.ffi.pycapsule(libfoo.bar),
        platform=PLATFORM,
        api_version=API_VERSION
    )

  Args:
    funcptr: 用 ``ctypes`` 从动态库中加载的函数指针。

  Returns:
    一个包装了 ``funcptr`` 的不透明 ``PyCapsule`` 对象。
  """
  destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
  builder = ctypes.pythonapi.PyCapsule_New
  builder.restype = ctypes.py_object
  builder.argtypes = (ctypes.c_void_p, ctypes.c_char_p, destructor)
  return builder(funcptr, None, destructor(0))


def include_dir() -> str:
  """获取 jaxlib 自带头文件所在目录的路径"""
  # 同时处理常规包（设置了 __file__）和命名空间包
  # （__file__ 为 None，但 __path__ 可用）两种情况
  if jaxlib.__file__ is not None:
    jaxlib_dir = os.path.dirname(os.path.abspath(jaxlib.__file__))
  elif hasattr(jaxlib, '__path__') and jaxlib.__path__:
    # 对于命名空间包，使用第一个路径条目
    jaxlib_dir = jaxlib.__path__[0]
  else:
    raise RuntimeError(
        "Cannot determine jaxlib directory: neither __file__ nor __path__ is available")
  return os.path.join(jaxlib_dir, "include")


def _aval_shape(aval: core.AbstractValue) -> Shape:
  if isinstance(aval, core.AbstractFuture):
    # AbstractFuture 是数组的 future。虽然该数组有形状，
    # 但 future 本身更像一个没有形状的 token。
    return ()
  return () if aval is core.abstract_token else core.physical_aval(aval).shape  # pyrefly: ignore[missing-attribute]


def _convert_layout_for_lowering(
    aval: core.AbstractValue, layout: FfiLayoutOptions = None) -> Sequence[int]:
  """把布局转换为 custom call API 使用的从次到主（minor-to-major）顺序。"""
  if layout is None:
    return tuple(reversed(range(len(_aval_shape(aval)))))
  elif isinstance(layout, Layout):
    if layout.tiling is not None:
      raise ValueError("The FFI does not support layouts with tiling")
    return layout.major_to_minor[::-1]
  else:
    return tuple(layout)


def build_ffi_lowering_function(
    call_target_name: str,
    *,
    operand_layouts: Sequence[FfiLayoutOptions] | None = None,
    result_layouts: Sequence[FfiLayoutOptions] | None = None,
    backend_config: Mapping[str, ir.Attribute] | str | None = None,
    skip_ffi_layout_processing: bool = False,
    **lowering_args: Any,
) -> Callable[..., ir.OpView]:
  """为外部函数接口（FFI）目标构建一个降级算子。

  默认情况下，该降级规则可以利用输入和输出的抽象值，
  在假定为行主序布局的前提下，计算出 custom call 的输入与输出类型和形状。

  注意，以元组形式传给本函数的布局应采用从次到主（minor-to-major）顺序
  （XLA 所期望的顺序），而不是 :func:`~jax.ffi.ffi_call` 与 ``Layout``
  所使用的从主到次（major-to-minor）顺序。

  如果向该降级规则传入关键字参数，它们会被当作属性，
  并被加入到 `backend_config` 中。

  Args:
    call_target_name: custom call 目标的名称。
    operand_layouts: 每个操作数的布局（维度顺序）序列。
      默认假定操作数为行主序。
    result_layouts: 每个结果的布局（维度顺序）序列。
      默认假定结果为行主序。
    backend_config: custom call 的配置数据。任何传给该降级规则的
      关键字参数都会被加入这个字典。
    lowering_args: 如果作为额外参数传给本函数，任何其他传给
      :func:`mlir.custom_call` 的参数也会一并传递。
    skip_ffi_layout_processing: 若为 true，则跳过对传给该降级规则的
      操作数与结果布局参数的处理。
  """

  def _lowering_op(
      ctx: mlir.LoweringRuleContext, *operands: ir.Value, **params: Any
  ) -> ir.OpView:
    kwargs = dict(lowering_args)
    kwargs.setdefault("api_version", 4)
    if kwargs["api_version"] >= 4:
      if backend_config is not None and not isinstance(backend_config, dict):
        raise ValueError(
            "When api_version > 4, backend_config must be a dictionary.")
      kwargs["backend_config"] = dict(
        backend_config or {}, **{k: mlir.ir_attribute(v) for k, v in params.items()})
    else:
      if params:
        raise ValueError(
            "The use of ffi_call attributes requires a custom call API version "
            f"of at least 4; got api_version={kwargs['api_version']}.")
      kwargs["backend_config"] = backend_config
    if "result_types" not in kwargs:
      flat_res_types, _ = mlir.ir_tree_registry.flatten(
          [mlir._aval_to_ir_types(ctx.module_context, a) for a in ctx.avals_out])
      kwargs["result_types"] = flat_res_types
    if not skip_ffi_layout_processing:
      if operand_layouts is None:
        kwargs["operand_layouts"] = map(
            _convert_layout_for_lowering, ctx.avals_in
        )
      else:
        kwargs["operand_layouts"] = [
            _convert_layout_for_lowering(*args)
            for args in zip(ctx.avals_in, operand_layouts)
        ]
      if result_layouts is None:
        kwargs["result_layouts"] = map(
            _convert_layout_for_lowering, ctx.avals_out
        )
      else:
        kwargs["result_layouts"] = [
            _convert_layout_for_lowering(*args)
            for args in zip(ctx.avals_out, result_layouts)
        ]
    if "result_shapes" not in kwargs and not all(
        core.is_constant_shape(_aval_shape(aval)) for aval in ctx.avals_out):
      kwargs["result_shapes"] = [
          mlir.shape_tensor(ctx.module_context, mlir.eval_dynamic_shape_as_ivals(ctx, _aval_shape(aval)))
          for aval in ctx.avals_out]

    return mlir.custom_call(call_target_name, operands=operands, **kwargs)

  return _lowering_op


def ffi_lowering(
    call_target_name: str,
    *,
    operand_layouts: Sequence[FfiLayoutOptions] | None = None,
    result_layouts: Sequence[FfiLayoutOptions] | None = None,
    backend_config: Mapping[str, ir.Attribute] | str | None = None,
    skip_ffi_layout_processing: bool = False,
    **lowering_args: Any
) -> mlir.LoweringRule:
  """为外部函数接口（FFI）目标构建一个降级规则。

  默认情况下，该降级规则可以利用输入和输出的抽象值，
  在假定为行主序布局的前提下，计算出 custom call 的输入与输出类型和形状。

  注意，以元组形式传给本函数的布局应采用从次到主（minor-to-major）顺序
  （XLA 所期望的顺序），而不是 :func:`~jax.ffi.ffi_call` 与 ``Layout``
  所使用的从主到次（major-to-minor）顺序。

  如果向该降级规则传入关键字参数，它们会被当作属性，
  并被加入到 `backend_config` 中。

  Args:
    call_target_name: custom call 目标的名称。
    operand_layouts: 每个操作数的布局（维度顺序）序列。
      默认假定操作数为行主序。
    result_layouts: 每个结果的布局（维度顺序）序列。
      默认假定结果为行主序。
    backend_config: custom call 的配置数据。任何传给该降级规则的
      关键字参数都会被加入这个字典。
    lowering_args: 如果作为额外参数传给本函数，任何其他传给
      :func:`mlir.custom_call` 的参数也会一并传递。
    skip_ffi_layout_processing: 若为 true，则跳过对传给该降级规则的
      操作数与结果布局参数的处理。
  """

  def _lowering(
    ctx: mlir.LoweringRuleContext, *operands: ir.Value, **params: Any
  ) -> Sequence[ir.Value | Sequence[ir.Value]]:
    result = build_ffi_lowering_function(
        call_target_name,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
        backend_config=backend_config,
        skip_ffi_layout_processing=skip_ffi_layout_processing,
        **lowering_args,
    )(ctx, *operands, **params)

    return result.results

  return _lowering


ResultMetadata = DuckTypedArray | core.AbstractToken


def _result_avals(results: Sequence[ResultMetadata]) -> tuple[core.AbstractValue, ...]:
  avals: list[core.AbstractValue] = []
  for idx, result in enumerate(results):
    if result is core.abstract_token:
      avals.append(result)
    else:
      if not hasattr(result, "shape") or not hasattr(result, "dtype"):
        raise ValueError(
            "All elements of result_shape_dtypes must have 'shape' and 'dtype' "
            f"attributes. Got {result} at position {idx}.")
      # 我们使用 explicit_x64_dtypes("allow")，这样 shaped_abstractify
      # 就不会对 result_shape_dtypes 上显式的 64 位数据类型做规范化。
      with config.explicit_x64_dtypes("allow"):
        avals.append(core.shaped_abstractify(result))
  return tuple(avals)

def _check_compatible_avals(a: core.AbstractValue, b: core.AbstractValue) -> bool:
  if isinstance(a, core.AbstractToken) and isinstance(b, core.AbstractToken):
    return True
  if getattr(a, "shape", ()) != getattr(b, "shape", ()):
    return False
  if getattr(a, "dtype", ()) != getattr(b, "dtype", ()):
    return False
  return True


def _convert_layouts_for_ffi_call(
    avals: Sequence[core.AbstractValue],
    layouts: Sequence[FfiLayoutOptions]) -> tuple[Sequence[int], ...]:
  return tuple(
      _convert_layout_for_lowering(
          aval,
          layout if layout is None or isinstance(layout, Layout)
          else layout[::-1]
      )
      for aval, layout in zip(avals, layouts))


# ffi_call() 返回的结果数量与 result_shape_dtypes 一样多。
@overload
def ffi_call(
    target_name: str,
    result_shape_dtypes: ResultMetadata,
    *,
    has_side_effect: bool = ...,
    vmap_method: str | None = ...,
    input_layouts: Sequence[FfiLayoutOptions] | None = ...,
    output_layouts: FfiLayoutOptions | Sequence[FfiLayoutOptions] | None = ...,
    input_output_aliases: dict[int, int] | None = ...,
    custom_call_api_version: int = ...,
    legacy_backend_config: str | None = ...,
) -> Callable[..., Array]:
  ...


@overload
def ffi_call(
    target_name: str,
    result_shape_dtypes: Sequence[ResultMetadata],
    *,
    has_side_effect: bool = ...,
    vmap_method: str | None = ...,
    input_layouts: Sequence[FfiLayoutOptions] | None = ...,
    output_layouts: FfiLayoutOptions | Sequence[FfiLayoutOptions] | None = ...,
    input_output_aliases: dict[int, int] | None = ...,
    custom_call_api_version: int = ...,
    legacy_backend_config: str | None = ...,
) -> Callable[..., Sequence[Array]]:
  ...


def ffi_call(
    target_name: str,
    result_shape_dtypes: ResultMetadata | Sequence[ResultMetadata],
    *,
    has_side_effect: bool = False,
    vmap_method: str | None = None,
    input_layouts: Sequence[FfiLayoutOptions] | None = None,
    output_layouts: FfiLayoutOptions | Sequence[FfiLayoutOptions] | None = None,
    input_output_aliases: dict[int, int] | None = None,
    custom_call_api_version: int = 4,
    legacy_backend_config: str | None = None,
) -> Callable[..., Array | Sequence[Array]]:
  """调用一个外部函数接口（FFI）目标。

  更多信息请参见 :ref:`ffi-tutorial` 教程。

  与 :func:`~jax.pure_callback` 类似，``ffi_call`` 在 :func:`~jax.vmap`
  下的行为取决于 ``vmap_method`` 的取值。关于允许的取值及其行为示例，
  详见 :func:`~jax.pure_callback` 的文档。

  当前的默认行为是：未指定时使用 ``vmap_method="sequential"``，
  但该行为已被弃用；将来除非显式指定 ``vmap_method``，
  默认行为将改为抛出 ``NotImplementedError``。

  Args:
    target_name: 通过 :func:`~jax.ffi.register_ffi_target` 注册的
      XLA FFI custom call 目标的名称。
    result_shape_dtypes: 一个对象或对象序列，其 ``shape`` 与 ``dtype``
      属性应当与 custom call 输出的形状和数据类型匹配。
      通常用 :class:`~jax.ShapeDtypeStruct` 来定义
      ``result_shape_dtypes`` 的元素。
      可以用 ``jax.core.abstract_token`` 表示 token 类型的输出。
    has_side_effect: 布尔值，指定该 custom call 是否有副作用。
      当为 ``True`` 时，即使输出未被使用，FFI 调用也会被执行。
    vmap_method: 字符串，按上文所述指定 FFI 调用在 :func:`~jax.vmap`
      下如何变换。
    input_layouts: 每个输入参数对应的布局序列。每种情况下，
      布局可以是 (a) ``None``，表示该输入采用默认的行主序，
      (b) 一个指定轴顺序的 ``Layout``，
      或 (c) 一个整数序列，指定从主到次的轴顺序。
      熟悉 XLA 布局的用户应注意，本函数期望的布局是从主到次顺序，
      而不是 XLA 使用的从次到主顺序。例如，一批行主序矩阵
      可以用布局 ``[0, 1, 2]`` 表示，而一批列主序矩阵的布局
      则为 ``[0, 2, 1]``。在这两个例子中，前导/批次维度都是“最慢”的轴。
      ``input_layouts`` 参数用于请求 FFI 调用目标所期望的内存布局，
      XLA 会确保处理函数执行前缓冲区具有正确的布局。
    output_layouts: 与 ``input_layouts`` 类似，但指定的是输出数组
      所需的布局。
    input_output_aliases: 一个字典，其键是输入索引，值是输出索引。
      该映射指明了哪些输出数组与特定的输入数组互为别名。
    custom_call_api_version: FFI 目标 ``target_name`` 所实现的
      custom call API 版本号。唯一正式支持的版本是
      ``custom_call_api_version=4`` 的带类型 FFI API，
      但更早的、不受支持的 custom call 也可以用该参数执行。
    legacy_backend_config: 对于用 ``custom_call_api_version<4``
      实现的旧式目标，属性通过该参数提供的不透明字符串表示来传递。
      该参数不能与 ``custom_call_api_version>=4`` 一起使用。

  Returns:
    一个函数，可以把输入数组作为位置参数调用它，以执行 FFI 处理函数。
    任何关键字参数都会通过 XLA 的 FFI 接口，作为具名属性传给
    FFI 处理函数。
  """

  allowed_vmap_methods = ["sequential", "sequential_unrolled", "expand_dims",
                          "broadcast_all", "legacy_vectorized", None]
  if vmap_method not in allowed_vmap_methods:
    raise ValueError(
        f"vmap_method must be on of the allowed methods {allowed_vmap_methods}, "
        f"but got: {vmap_method}")

  output_layouts_: Sequence[FfiLayoutOptions] | None
  if isinstance(result_shape_dtypes, Sequence):
    output_layouts_ = output_layouts  # pyrefly: ignore[bad-assignment]
    multiple_results = True
    result_avals = _result_avals(result_shape_dtypes)
  else:
    multiple_results = False
    result_avals = _result_avals([result_shape_dtypes])
    output_layouts_ = (output_layouts,)  # pyrefly: ignore[bad-assignment]

  if custom_call_api_version >= 4 and legacy_backend_config is not None:
    raise ValueError(
        "The use of the legacy_backend_config parameter requires "
        f"custom_call_api_version < 4; got {custom_call_api_version}.")

  def wrapped(*args: ArrayLike, **kwargs: Any):
    in_avals = [core.typeof(x) for x in args]

    if input_layouts is None:
      static_input_layouts = tuple(map(_convert_layout_for_lowering, in_avals))
    else:
      if len(input_layouts) != len(in_avals):
        raise ValueError(
            f"The number of input arguments ({len(in_avals)}) must equal the "
            f"number of input layouts ({len(input_layouts)}).")
      static_input_layouts = _convert_layouts_for_ffi_call(in_avals,
                                                           input_layouts)
    if output_layouts_ is None:
      static_output_layouts = tuple(map(_convert_layout_for_lowering,
                                        result_avals))
    else:
      if len(output_layouts_) != len(result_avals):
        raise ValueError(
            f"The number of outputs ({len(result_avals)}) must equal the "
            f"number of output layouts ({len(output_layouts_)}).")
      static_output_layouts = _convert_layouts_for_ffi_call(result_avals,
                                                            output_layouts_)

    static_input_output_aliases: list[tuple[int, int]] = []
    if input_output_aliases is not None:
      for i_idx, o_idx in sorted(input_output_aliases.items()):
        i_idx, o_idx = int(i_idx), int(o_idx)
        if i_idx >= len(args):
          raise ValueError(
              f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
              f"with input index {i_idx} outside the range [0, "
              f"{len(args)}).")
        if o_idx >= len(result_avals):
          raise ValueError(
              f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
              f"with output index {o_idx} outside the range [0, "
              f"{len(result_avals)}).")
        in_aval = in_avals[i_idx]
        out_aval = result_avals[o_idx]
        if not _check_compatible_avals(in_aval, out_aval):
          raise ValueError(
              f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
              f"referring to an input with abstract value {in_aval} and an "
              f"output with a different abstract value {out_aval}.")
        if static_input_layouts[i_idx] != static_output_layouts[o_idx]:
          raise ValueError(
              f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
              f"referring to an input with layout {static_input_layouts[i_idx]} "
              "and an output with a different layout "
              f"{static_output_layouts[o_idx]}.")
        static_input_output_aliases.append((i_idx, o_idx))
    args = core.auto_insert_reshard(*args)
    results = ffi_call_p.bind(
        *args,
        result_avals=result_avals,
        vmap_method=vmap_method,
        target_name=target_name,
        has_side_effect=has_side_effect,
        input_layouts=static_input_layouts,
        output_layouts=static_output_layouts,
        input_output_aliases=tuple(static_input_output_aliases),
        custom_call_api_version=custom_call_api_version,
        legacy_backend_config=legacy_backend_config,
        attributes=_wrap_kwargs_hashable(kwargs),
    )
    if multiple_results:
      if isinstance(result_shape_dtypes, tuple):
        return tuple(results)
      return results
    else:
      return results[0]

  return wrapped


# ffi_call 必须支持一些不可哈希的小型输入参数，例如 np.array
# 和 dict，以便支持用数组输入或用户定义的结构体调用 FFI 目标。
# 由于这些参数最终会作为稠密属性嵌入 HLO 中，
# 我们假定它们很小，于是通过创建不可变副本、
# 并按值进行哈希的方式来哈希。
def _wrap_kwargs_hashable(kwargs: dict[str, Any]) -> Sequence[tuple[str, Any]]:
  hashable_kwargs: list[tuple[str, Any]] = []
  for k, v in sorted(kwargs.items()):
    if isinstance(v, np.ndarray):
      hashable_kwargs.append((k, HashableArray(v)))
    elif isinstance(v, dict):
      hashable_kwargs.append((k, FrozenDict(v)))
    else:
      try:
        hash(v)
      except TypeError as e:
        raise TypeError(
            f"Non-hashable keyword argument to ffi_call {k}: {v}") from e
      else:
        hashable_kwargs.append((k, v))
  return tuple(hashable_kwargs)


def _unwrap_kwargs_hashable(kwargs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
  unwrapped_kwargs: dict[str, Any] = {}
  for k, v in kwargs:
    if isinstance(v, HashableArray):
      unwrapped_kwargs[k] = v.val
    elif isinstance(v, FrozenDict):
      unwrapped_kwargs[k] = v._d
    else:
      unwrapped_kwargs[k] = v
  return unwrapped_kwargs


@dataclasses.dataclass(frozen=True, slots=True)
class FfiEffect(effects.Effect):
  def __str__(self):
    return "FFI"


_FfiEffect = FfiEffect()
effects.lowerable_effects.add_type(FfiEffect)
effects.control_flow_allowed_effects.add_type(FfiEffect)
effects.remat_allowed_effects.add_type(FfiEffect)
effects.custom_derivatives_allowed_effects.add_type(FfiEffect)
effects.partial_eval_kept_effects.add_type(FfiEffect)


def ffi_call_abstract_eval(
    *avals_in,
    result_avals: tuple[core.AbstractValue, ...],
    has_side_effect: bool,
    **_,
):
  core.standard_vma_rule('ffi_call', *avals_in)
  effects = {_FfiEffect} if has_side_effect else core.no_effects
  return tuple(r if r is core.abstract_token else
               r.update(sharding=(core.get_cur_mesh_sharding()
                                  if r.sharding.mesh.empty else r.sharding))  # pyrefly: ignore[missing-attribute]
               for r in result_avals), effects


def ffi_call_jvp(*args, target_name, **_):
  del args
  raise ValueError(
      f"The FFI call to `{target_name}` cannot be differentiated. "
      "You can use `jax.custom_jvp` or `jax.custom_jvp` to add support.")


def ffi_call_transpose(*args, target_name, **_):
  del args
  raise ValueError(
      f"The FFI call to `{target_name}` cannot be differentiated. "
      "You can use `jax.custom_jvp` or `jax.custom_jvp` to add support.")


def ffi_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *operands: ir.Value,
    target_name: str,
    has_side_effect: bool,
    input_layouts: Sequence[Sequence[int]],
    output_layouts: Sequence[Sequence[int]],
    input_output_aliases: Sequence[tuple[int, int]],
    custom_call_api_version: int,
    legacy_backend_config: str | None,
    attributes: Sequence[tuple[str, Any]],
    **_,
) -> Sequence[ir.Value | Sequence[ir.Value]]:
  rule = ffi_lowering(target_name, has_side_effect=has_side_effect,
                      operand_layouts=input_layouts,
                      result_layouts=output_layouts,
                      operand_output_aliases=dict(input_output_aliases),
                      api_version=custom_call_api_version,
                      backend_config=legacy_backend_config)
  return rule(ctx, *operands, **_unwrap_kwargs_hashable(attributes))


def ffi_batching_rule(
    prim,
    args,
    dims,
    *,
    vmap_method: str | None,
    result_avals: Sequence[core.ShapedArray],
    **kwargs: Any,
):
  from jax._src.lax import control_flow  # pyrefly: ignore[missing-import]
  from jax._src.lax import lax  # pyrefly: ignore[missing-import]

  axis_size, = {a.shape[d] for a, d in zip(args, dims)
                if d is not None}
  new_args = [arg if dim is None else
              batching.moveaxis(arg, dim, 0) for arg, dim in zip(args, dims)]
  batched_result_avals = tuple(
      core.unmapped_aval(axis_size, 0, aval) for aval in result_avals)

  # 对于 FFI 调用，我们必须更新布局。这里处理输出布局，
  # 而输入布局的更新取决于 vmap_method 参数。
  if (
      vmap_method not in ("sequential", "sequential_unrolled") and
      kwargs.get("output_layouts") is not None
  ):
    kwargs["output_layouts"] = tuple(
        None if layout is None else tuple(n + 1 for n in layout) + (0,)
        for layout in kwargs["output_layouts"])

  if vmap_method == "legacy_vectorized":
    # 保留该方法是为了支持以前使用 `vectorized=True` 时
    # 所暴露的行为。
    if kwargs.get("input_layouts") is not None:
      kwargs["input_layouts"] = tuple(
          layout if d is None else
          (None if layout is None else tuple(n + 1 for n in layout) + (0,))
          for layout, d in zip(kwargs["input_layouts"], dims))
    outvals = prim.bind(
        *new_args,
        vmap_method=vmap_method,
        result_avals=batched_result_avals,
        **kwargs,
    )
  elif vmap_method == "expand_dims" or vmap_method == "broadcast_all":
    size = axis_size if vmap_method == "broadcast_all" else 1
    bcast_args = [
        lax.broadcast(x, (size,)) if d is None else x
        for x, d in zip(new_args, dims)]
    if kwargs.get("input_layouts") is not None:
      kwargs["input_layouts"] = tuple(
          None if layout is None else tuple(n + 1 for n in layout) + (0,)
          for layout in kwargs["input_layouts"])
    outvals = prim.bind(
      *bcast_args,
      vmap_method=vmap_method,
      result_avals=batched_result_avals,
      **kwargs,
    )
  elif vmap_method == "sequential" or vmap_method == "sequential_unrolled":
    is_batched = [d is not None for d in dims]
    unbatched_args, batched_args = util.partition_list(is_batched, new_args)
    def _batch_fun(batched_args):
      merged_args = util.merge_lists(is_batched, unbatched_args, batched_args)
      return prim.bind(
          *merged_args,
          result_avals=result_avals,
          vmap_method=vmap_method,
          **kwargs,
      )
    unroll = vmap_method == "sequential_unrolled"
    g = lambda _, x: ((), _batch_fun(x))
    _, outvals = control_flow.scan(g, (), batched_args, unroll=unroll)
  else:
    raise NotImplementedError(
        f"vmap is only supported for the {prim.name} primitive when vmap_method "
        "is one of 'sequential', 'sequential_unrolled', 'expand_dims', "
        f"'broadcast_all', or 'legacy_vectorized'. Got {vmap_method=}.")
  return tuple(outvals), (0,) * len(outvals)


ffi_call_p = core.Primitive("ffi_call")
ffi_call_p.multiple_results = True
dispatch.simple_impl(ffi_call_p)
ffi_call_p.def_effectful_abstract_eval(ffi_call_abstract_eval)
ad.primitive_jvps[ffi_call_p] = ffi_call_jvp
ad.primitive_transposes[ffi_call_p] = ffi_call_transpose
batching.primitive_batchers[ffi_call_p] = functools.partial(
    ffi_batching_rule, ffi_call_p)
mlir.register_lowering(ffi_call_p, ffi_call_lowering)
