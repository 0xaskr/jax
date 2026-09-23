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
# 文件职责：定义 JAX 编译流程中各阶段的公开接口类型，以及让内部实现适配这些接口的工具。
# 模块为编译流水线建模两个阶段：降级（lowering，产出编译器输入）与编译
# （compilation，产出编译器输出），分别对应公开类型 `Lowered` 与 `Compiled`。
# 内部的 `Lowering`/`Executable` 协议为上述公开类型提供支撑，并由下面的辅助类
# 把 JAX 各种基于 XLA 的内部 lowering 与可执行文件适配到这些协议。
# 这些类型由 `jax.jit`、`jax.pmap` 等面向用户的 API 返回，供用户检查已暂存、
# 已降级、已编译的计算（文本表示、代价分析、内存分析、分片与布局信息）。
"""
与 JAX 各编译步骤交互的接口，以及用于适配这些接口的工具。

本模块定义了一组面向公开的类型，它们反映编译过程中
各中间阶段的输出。目前建模了两个阶段：降级（lowering，
产出编译器输入）与编译（compilation，产出编译器输出）。

它还定义了一些面向内部的类型，用以指导 JAX 能以这种
统一形式呈现什么：内部的 ``Lowering`` 足以支撑面向公开的
``Lowered``，内部的 ``Executable`` 足以支撑面向公开的
``Compiled``。

最后，本模块还定义了几个类，用于把 JAX 内部各种基于 XLA 的
lowering 与可执行文件方便地适配到上文所述的 lowering 与
可执行协议。
"""
from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import enum
import itertools as it
from typing import Any, NamedTuple, Protocol, runtime_checkable

from jax._src import config
from jax._src import core
from jax._src import sharding as sharding_lib
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src.core import typeof
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.layout import AutoLayoutSingleton, Format, Layout
from jax._src.lib import _jax
from jax._src.lib import hlo
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.sharding_impls import UnspecifiedValue
from jax._src.mesh import AbstractMesh
from jax._src import flattree as ft
from jax._src.tree_util import tree_unflatten
from jax._src.typing import ArrayLike

source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

CompilerOptions = dict[str, str | bool]


# -- 内部类型


class Executable:

  def xla_extension_executable(self) -> xc.LoadedExecutable:
    raise NotImplementedError(
        "compiled executable carries no loaded XLA executable. It may be "
        f"that {type(self)} defines an incomplete implementation.")

  def call(self, *args_flat) -> Sequence[Any]:
    """在扁平参数列表上执行，返回扁平的输出。"""
    raise NotImplementedError("compiled executable does not support invocation")

  def create_cpp_call(self, params: CompiledCallParams) -> Any:
    """可选地构造一个快速的 C++ 分派器。"""
    return None

  def input_shardings(self) -> Sequence[sharding_lib.Sharding]:
    """扁平的输入分片序列。

    在不可用时（例如取决于后端、
    编译器或运行时）可能抛出 ``NotImplementedError``。
    """
    raise NotImplementedError(
        "compiled executable carries no input sharding information")

  def output_shardings(self) -> Sequence[sharding_lib.Sharding]:
    """扁平的输出分片序列。

    在不可用时（例如取决于后端、
    编译器或运行时）可能抛出 ``NotImplementedError``。
    """
    raise NotImplementedError(
        "compiled executable carries no output sharding information")

  def input_formats(self):
    raise NotImplementedError(
        "compiled executable carries no input layout information")

  def output_formats(self):
    raise NotImplementedError(
        "compiled executable carries no output layout information")

  def as_text(self) -> str:
    """此可执行文件的、人类可读的文本表示。

    用于可视化和调试目的。它不必是有效或可靠的
    序列化形式。它会被直接转交给外部调用者。

    在不可用时（例如取决于后端、
    编译器或运行时）可能抛出 ``NotImplementedError``。
    """
    xla_ext_exe = self.xla_extension_executable()
    err_msg = ("text view unsupported on current XLA backend: "
               f"{type(xla_ext_exe)}")

    if hasattr(xla_ext_exe, "get_hlo_text"):
      try:
        return xla_ext_exe.get_hlo_text()
      except _jax.JaxRuntimeError as e:
        msg, *_ = e.args
        if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
          raise NotImplementedError(err_msg) from e
        else:
          raise
    else:
      if not hasattr(xla_ext_exe, "hlo_modules"):
        raise NotImplementedError(err_msg)
      try:
        return "\n\n".join([m.to_string() for m in xla_ext_exe.hlo_modules()])
      except _jax.JaxRuntimeError as e:
        msg, *_ = e.args
        if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
          raise NotImplementedError(err_msg) from e
        else:
          raise

  def cost_analysis(self) -> Any:
    """执行代价估计的摘要。

    用于可视化和调试目的。它输出的对象
    是一些易于打印或序列化的简单数据结构
    （例如以数值为叶子的嵌套 dict、list 和 tuple）。不过
    其结构可以是任意的：它不必在 JAX 与 jaxlib 的
    不同版本之间保持一致，甚至不必在多次调用之间保持一致。它会被
    直接转交给外部调用者。

    在不可用时（例如取决于后端、
    编译器或运行时）可能抛出 ``NotImplementedError``。
    """
    xla_ext_exe = self.xla_extension_executable()

    if hasattr(xla_ext_exe, "cost_analysis"):
      try:
        return xla_ext_exe.cost_analysis()
      except _jax.JaxRuntimeError as e:
        msg, *_ = e.args
        if not (type(msg) is str and msg.startswith("UNIMPLEMENTED")):
          raise

    if (
        xla_ext_exe is None
        and hasattr(self, "unsafe_call")
        and hasattr(self.unsafe_call, "compiled")
        and hasattr(self.unsafe_call.compiled, "cost_analysis")
    ):
      return self.unsafe_call.compiled.cost_analysis()

    raise NotImplementedError(
        f"cost analysis unsupported on current XLA backend: {type(xla_ext_exe)}"
    )

  def memory_analysis(self) -> Any:
    """估计内存需求的摘要。

    用于可视化和调试目的。它输出的对象
    是一些易于打印或序列化的简单数据结构
    （例如以数值为叶子的嵌套 dict、list 和 tuple）。不过
    其结构可以是任意的：它不必在 JAX 与 jaxlib 的
    不同版本之间保持一致，甚至不必在多次调用之间保持一致。它会被
    直接转交给外部调用者。

    在不可用时（例如取决于后端、
    编译器或运行时）可能抛出 ``NotImplementedError``。
    """
    xla_ext_exe = self.xla_extension_executable()
    err_msg = ("memory analysis unsupported on current XLA backend: "
               f"{type(xla_ext_exe)}")
    if not hasattr(xla_ext_exe, "get_compiled_memory_stats"):
      raise NotImplementedError(err_msg)
    try:
      return xla_ext_exe.get_compiled_memory_stats()
    except _jax.JaxRuntimeError as e:
      msg, *_ = e.args
      if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
        raise NotImplementedError(err_msg) from e
      else:
        raise

  def runtime_executable(self) -> Any:
    """此可执行文件的任意对象表示。

    用于调试目的。它不必是有效或可靠的
    序列化形式。它会被直接转交给外部调用者，对其类型、
    结构或多次调用之间的一致性不作任何保证。

    在不可用时（例如取决于后端或
    编译器）可能抛出 ``NotImplementedError``。
    """
    return self.xla_extension_executable()


class Lowering:

  compile_args: dict[str, Any]
  # 已被提到外层的常量，必须作为最前面的参数传入，
  # 位于 tokens 之后。
  # 参见 https://docs.jax.dev/en/latest/internals/constants.html。
  const_args: list[ArrayLike]

  def hlo(self) -> xc.XlaComputation:
    """返回此计算的 HLO 表示。"""
    hlo = self.stablehlo()
    m: str | bytes
    m = mlir.module_to_bytecode(hlo)
    return _jax.mlir.mlir_module_to_xla_computation(
        m, use_tuple_args=self.compile_args["tuple_args"])

  def stablehlo(self) -> ir.Module:
    """返回此计算的 StableHLO 表示。"""
    raise NotImplementedError(
        f"cost analysis unsupported on XLA computation: {type(self)}")

  def compile(
      self, compiler_options: CompilerOptions | None = None, *,
      device_assignment: tuple[xc.Device, ...] | None = None) -> Executable:
    """编译并返回对应的 ``Executable``。"""
    raise NotImplementedError(
        f"cost analysis unsupported on XLA computation: {type(self)}")

  def as_text(self, dialect: str | None = None,
              *,
              debug_info: bool = False) -> str:
    """此 lowering 的人类可读文本表示。

    用于可视化和调试目的。它不必是有效或可靠的序列化形式。
    它会被直接转交给外部调用者。
    """
    if dialect is None:
      dialect = "stablehlo"
    if dialect == "stablehlo":
      return mlir.module_to_string(self.stablehlo(),
                                   enable_debug_info=debug_info)
    elif dialect == "hlo":
      print_opts = hlo.HloPrintOptions.short_parsable()
      print_opts.print_metadata = debug_info
      return self.hlo().as_hlo_module().to_string(print_opts)
    else:
      raise ValueError(f"unknown dialect: {dialect}")

  def compiler_ir(self, dialect: str | None = None) -> Any:
    """此 lowering 的任意对象表示。

    用于调试目的。它不必是有效或可靠的序列化形式。
    它会被直接转交给外部调用者，对其类型、
    结构或多次调用之间的一致性不作任何保证。

    在不可用时（例如取决于后端或
    编译器）可能抛出 ``NotImplementedError``。

    Args:
      dialect: 可选字符串，指定表示所用的方言
      （例如 "stablehlo"）
    """
    if dialect is None:
      dialect = "stablehlo"
    if dialect == "stablehlo":
      return self.stablehlo()
    elif dialect == "hlo":
      return self.hlo()
    else:
      raise ValueError(f"unknown dialect: {dialect}")

  def cost_analysis(self) -> Any:
    """执行代价估计的摘要。

    用于可视化和调试目的。它输出的对象
    是一些易于打印或序列化的简单数据结构
    （例如以数值为叶子的嵌套 dict、list 和 tuple）。不过
    其结构可以是任意的：它不必在 JAX 与 jaxlib 的
    不同版本之间保持一致，甚至不必在多次调用之间保持一致。它会被
    直接转交给外部调用者。

    此函数估计的是在没有编译器优化时的执行代价，
    优化可能大幅改变代价。若需要优化之后的执行代价
    估计，请编译此 lowering 并参见
    ``Compiled.cost_analysis``。

    在不可用时（例如取决于后端、
    编译器或运行时）可能抛出 ``NotImplementedError``。
    """
    raise NotImplementedError(
        f"cost analysis unsupported on XLA computation: {type(self)}")


# -- 面向公开的 API，以及辅助函数

@dataclass(frozen=True, slots=True)
class ArgInfo:
  _aval: core.AbstractValue
  donated: bool

  @property
  def shape(self):
    if not hasattr(self._aval, "shape"):
      raise TypeError(f"No shape attribute with aval of type {type(self._aval)}")
    return self._aval.shape

  @property
  def dtype(self):
    if not hasattr(self._aval, "dtype"):
      raise TypeError(f"No dtype attribute with aval of type {type(self._aval)}")
    return self._aval.dtype


class Stage:
  args_info: Any  # ArgInfo 的 PyTree

  @property
  def in_tree(self) -> tree_util.PyTreeDef:
    """由（位置参数, 关键字参数）组成的对的树结构。"""
    return tree_util.tracing_registry.flatten(self.args_info)[1]

  @property
  def in_avals(self):
    """输入 aval 的树。"""
    return tree_util.tree_map(lambda x: x._aval, self.args_info)

  @property
  def donate_argnums(self):
    """被捐赠参数索引组成的扁平元组。"""
    return tuple(
        i for i, x in enumerate(tree_util.tree_leaves(self.args_info))
        if x.donated)


def make_args_info(in_tree, in_avals, donate_argnums):
  donate_argnums = frozenset(donate_argnums)
  flat_avals, _ = tree_util.tree_flatten(in_avals)  # todo: 待移除
  return in_tree.unflatten([
      ArgInfo(aval, i in donate_argnums)
      for i, aval in enumerate(flat_avals)])


class CompiledCallParams(NamedTuple):
  executable: Executable
  no_kwargs: bool
  in_tree: tree_util.PyTreeDef  # lo tree（低层树）
  out_tree: tree_util.PyTreeDef  # lo tree（低层树）
  const_args: list[ArrayLike]  # https://docs.jax.dev/en/latest/internals/constants.html
  in_types: tuple[tree_util.PyTreeDef, list[core.AbstractValue]] | None
  out_types: tuple[tree_util.PyTreeDef, list[core.AbstractValue]] | None

  @property
  def is_high(self):
    return self.in_types and self.out_types and any(
        a.is_high for a in it.chain(self.in_types[1], self.out_types[1]))


def _traced_args_info(self):
  don8_rgn = tuple(i for i, d in enumerate(self._params['donated_invars']) if d)
  arg_avals = self.jaxpr.in_avals[self._num_consts:]
  return make_args_info(self._in_tree, arg_avals, don8_rgn)

def _traced_out_info(self):
  out_shardings = [None if isinstance(s, UnspecifiedValue) else s
                   for s in self._params['out_shardings']]
  out_layouts = [None if isinstance(l, AutoLayoutSingleton) else l
                 for l in self._params['out_layouts']]
  out = []
  for a, out_s, out_l in zip(self.jaxpr.out_avals, out_shardings, out_layouts):
    if isinstance(a, core.ShapedArray):
      s = ((a.sharding if a.sharding.mesh._are_all_axes_explicit_or_manual
            else out_s) if out_s is None else out_s)
      out.append(
          core.ShapeDtypeStruct(
              a.shape, a.dtype, sharding=Format(out_l, s),
              weak_type=a.weak_type,
              manual_axis_type=(a.mat if config._check_vma.value else None),
              _memory_space=a.memory_space))
    else:
      out.append(a)
  return tree_util.tree_unflatten(self.out_tree, out)


class Traced(Stage):
  """针对参数类型和值特化后的函数的已追踪形式。

  已追踪的计算即可进行降级（lowering）。此类携带
  追踪后的表示，以及后续对其进行降级、编译和执行
  所需的其余信息。

  分别通过 `.jaxpr` 与 `.lojax` 属性提供对
  hijax（高层）和 lojax（低层）表示的访问。
  """
  __slots__ = ['_meta_tys_flat', '_params', '_in_tree', 'out_tree', '_consts',
               '_fun_sourceinfo', '_lojax', '_closure_converted']

  def __init__(self, meta_tys_flat, params, in_tree, out_tree, consts,
               fun_sourceinfo):
    self._meta_tys_flat = meta_tys_flat
    self._params = params
    self._in_tree = in_tree
    self.out_tree = out_tree
    self._consts = consts
    self._fun_sourceinfo = fun_sourceinfo
    self._lojax = None
    self._closure_converted = None

  jaxpr = property(lambda self: self._params['jaxpr'])
  fun_name = property(lambda self: self._params['name'])
  args_info = property(_traced_args_info)  # pyrefly: ignore[bad-override]
  out_info = property(_traced_out_info)
  _num_consts = property(lambda self: len(self._consts))

  @property
  def out_avals(self):
    return tree_unflatten(self.out_tree, self.jaxpr.out_avals)

  def __call__(self, *args, **kwargs):
    args_flat = tree_util.tree_leaves_checked(self.in_tree, (args, kwargs))
    out_flat = core.eval_jaxpr_p.bind(*args_flat, call_jaxpr=self.jaxpr)
    return tree_unflatten(self.out_tree, out_flat)

  def closure_convert(self):
    """闭包转换：把此 Traced 捕获的常量显式化。

    返回一对 ``(consts, fun)``，其中 ``consts`` 是此 Traced
    在追踪期间从其函数闭包中捕获的值（不是 Python 的
    ``__closure__`` 单元：而是追踪期间遇到的、任何
    决定输出的值），而 ``fun`` 是一个封闭函数，满足
    ``fun(consts, *args, **kwargs)`` 的计算结果与此 Traced
    作用于 ``args`` 和 ``kwargs`` 时相同。环境 ``consts`` 作为
    单个前导参数传入，并且可以替换为具有相同类型的
    值组成的任意 pytree。
    """
    if self._closure_converted is None:
      consts = [*self.jaxpr.consts, *self._consts]
      _, consts_tree = tree_util.tracing_registry.flatten(consts)
      jaxpr = self.jaxpr.replace(consts=None)
      in_tree = tree_util.treedef_tuple_tracing_registry(
          (consts_tree, *self.in_tree.children()))
      out_tree = self.out_tree
      def fun(consts, *args, **kwargs):
        args_flat = tree_util.tree_leaves_checked(in_tree, (consts, args, kwargs))
        out_flat = core.eval_jaxpr_p.bind(*args_flat, call_jaxpr=jaxpr)
        return tree_unflatten(out_tree, out_flat)
      self._closure_converted = (consts, fun)
    return self._closure_converted

  def with_consts_as_arg(self) -> tuple[list[Any], Traced]:
    """返回 consts 以及一个把它们作为单个前导参数的等价 Traced。

    基于 ``closure_convert`` 构建：转换后的函数会以
    consts 环境作为单个前导参数被重新追踪，从而让追踪
    机制一致地重建所有按参数维护的记账信息。非 const
    参数类型取自此 Traced。重新追踪只会暂存
    单个 call 方程，而不会重新追踪原函数。
    """
    from jax._src.api import jit  # type: ignore
    consts, fun = self.closure_convert()
    arg_avals = self.jaxpr.in_avals[len(self._consts):]
    args, kwargs = tree_unflatten(self.in_tree, arg_avals)
    traced = jit(fun).trace(consts, *args, **kwargs)
    jaxpr = traced.jaxpr
    if (not jaxpr.consts and len(jaxpr.eqns) == 1 and
        (eqn := jaxpr.eqns[0]).primitive is core.eval_jaxpr_p and
        list(eqn.invars) == list(jaxpr.invars) and
        list(eqn.outvars) == list(jaxpr.outvars)):
      traced._params = dict(traced._params, jaxpr=eqn.params['call_jaxpr'])
    traced._params = dict(traced._params, name=self.fun_name)
    traced._fun_sourceinfo = self._fun_sourceinfo
    return consts, traced

  @property
  def lojax(self) -> LoJax:
    if self._lojax is not None:
      return self._lojax

    if not self.jaxpr.is_high:
      self._lojax = LoJax(
          self._meta_tys_flat, self._params, self._in_tree, self.out_tree,
          (self._in_tree, self.jaxpr.in_avals),
          (self.out_tree, self.jaxpr.out_avals),
          self._consts, self._fun_sourceinfo)
      return self._lojax

    # TODO(mattjj): 当 pmap 被删除后，与 pjit.py 的 BUILD 规则合并
    from jax._src.pjit import _lojax_expand_params  # pyrefly: ignore[missing-import]
    hi_jaxpr = self.jaxpr
    in_avals = ft.flatten(([a.lo_ty() for a in hi_jaxpr.in_avals], {}))
    lo_jaxpr, out_avals = pe.lower_jaxpr(hi_jaxpr, in_avals)
    params = dict(_lojax_expand_params(in_avals, out_avals, **self._params), jaxpr=lo_jaxpr)
    if any(a.is_high for a in hi_jaxpr.in_avals):
      in_tree = lojax_pytree(hi_jaxpr.in_avals, self._in_tree)
    else:
      in_tree = self._in_tree
    if any(a.is_high for a in hi_jaxpr.out_avals):
      out_tree = lojax_pytree(hi_jaxpr.out_avals, self.out_tree)
    else:
      out_tree = self.out_tree
    lo_meta_tys = [mty.replace(aval=lo_ty)
                   for mty in self._meta_tys_flat
                   for lo_ty in mty.aval.lo_ty()]
    self._lojax = LoJax(
        lo_meta_tys, params, in_tree, out_tree,
        (self._in_tree, hi_jaxpr.in_avals),
        (self.out_tree, hi_jaxpr.out_avals),
        self._consts, self._fun_sourceinfo)
    return self._lojax

  def lower(self, *, lowering_platforms: tuple[str, ...] | None = None,
            _private_parameters: mlir.LoweringParameters | None = None):
    """降级为编译器输入，返回 ``Lowered`` 实例。"""
    from jax._src.pjit import _resolve_and_lower  # pyrefly: ignore[missing-import]
    lo = self.lojax
    if _private_parameters is None:
      _private_parameters = mlir.LoweringParameters()
    try:
      ctx_mesh = lo._params['ctx_mesh']
      if (lowering_platforms is None and isinstance(ctx_mesh, AbstractMesh)
          and (abd := ctx_mesh.abstract_device) is not None):
        lowering_platforms = (abd.platform,)
      lowering = _resolve_and_lower(
          lo._meta_tys_flat, **lo._params, lowering_platforms=lowering_platforms,
          lowering_parameters=_private_parameters, pgle_profiler=None)
    except DeviceAssignmentMismatchError as e:
      fails, = e.args
      msg = _device_assignment_mismatch_error(
          lo._params['name'], fails, lo._meta_tys_flat, 'jit',
          lo.jaxpr.debug_info.safe_arg_names(len(lo.jaxpr.in_avals)))
      raise ValueError(msg) from None
    return Lowered(lowering, lo.args_info, lo.out_tree,
                   in_types=lo._in_types, out_types=lo._out_types)


def lojax_pytree(hi_avals, tree):
  lo_avals = [t.lo_ty() for t in hi_avals]
  return tree_util.tracing_registry.flatten(tree_unflatten(tree, lo_avals))[1]


class LoJax:
  __slots__ = ['_meta_tys_flat', '_params', '_in_tree', 'out_tree',
               '_consts', '_fun_sourceinfo', '_in_types', '_out_types']

  def __init__(self, meta_tys_flat, params, in_tree, out_tree, in_types, out_types,
               consts, fun_sourceinfo):
    self._meta_tys_flat = meta_tys_flat
    self._params = params
    self._in_tree = in_tree
    self.out_tree = out_tree
    self._consts = consts
    self._fun_sourceinfo = fun_sourceinfo
    self._in_types = in_types  # hi types（高层类型）
    self._out_types = out_types

  jaxpr = property(lambda self: self._params['jaxpr'])
  fun_name = property(lambda self: self._params['name'])
  args_info = property(_traced_args_info)
  out_info = property(_traced_out_info)
  _num_consts = property(lambda self: len(self._consts))


class Lowered(Stage):
  """针对参数类型和值特化后的函数的降级结果。

  一个 lowering 就是一份可编译的计算。此类
  携带一个 lowering，以及后续编译和执行它
  所需的其余信息。它还提供统一的 API，
  用于在 JAX 各种降级路径（:func:`~jax.jit`、:func:`~jax.pmap` 等）
  上查询已降级计算的属性。
  """
  __slots__ = ["_lowering", "args_info", "out_tree", "_no_kwargs",
               "_in_types", "_out_types"]

  _lowering: Lowering
  args_info: Any  # ArgInfo 的 PyTree，不包含 const_args
  out_tree: tree_util.PyTreeDef
  _no_kwargs: bool
  _in_types: tuple[tree_util.PyTreeDef, list[core.AbstractValue]] | None
  _out_types: list[core.AbstractValue] | None

  def __init__(self, lowering: Lowering, args_info,
               out_tree: tree_util.PyTreeDef, no_kwargs: bool = False,
               in_types=None, out_types=None):

    self._lowering = lowering
    self.args_info = args_info
    self.out_tree = out_tree
    self._no_kwargs = no_kwargs
    self._in_types = in_types
    self._out_types = out_types

  @property
  def in_avals(self):
    in_avals_ = self._lowering.compile_args["global_in_avals"]
    kept_var_idx = self._lowering.compile_args["kept_var_idx"]
    non_dce_avals = self._lowering.compile_args["all_args_info"].in_avals
    if self.in_tree.num_leaves > len(in_avals_):
      iter_in_avals = iter(in_avals_)
      in_avals_ = [
          next(iter_in_avals) if i in kept_var_idx
          else a for i, a in zip(range(self.in_tree.num_leaves), non_dce_avals)]
    return self.in_tree.unflatten(in_avals_)

  @property
  def out_info(self):  # OutInfo 的 PyTree
    out_avals = self._lowering.compile_args["global_out_avals"]
    out_shardings = self._lowering.compile_args["out_shardings"]
    out_layouts = self._lowering.compile_args["out_layouts"]
    outs = []
    for o, l, s in zip(out_avals, out_layouts, out_shardings):
      s = None if isinstance(s, UnspecifiedValue) else s
      l = None if isinstance(l, AutoLayoutSingleton) else l
      format = Format(l, s)
      outs.append(core.ShapeDtypeStruct(o.shape, o.dtype, sharding=format))
    return self.out_tree.unflatten(outs)

  def compile(
      self, compiler_options: CompilerOptions | None = None, *,
      device_assignment: tuple[xc.Device, ...] | None = None) -> Compiled:
    """编译，返回对应的 ``Compiled`` 实例。"""
    kw: dict[str, Any] = {"compiler_options": compiler_options,
                          "device_assignment": device_assignment}
    return Compiled(self._lowering.compile(**kw), self._lowering.const_args,
                    self.args_info, self.out_tree, self._no_kwargs,
                    self._in_types, self._out_types)

  def as_text(self, dialect: str | None = None, *,
              debug_info: bool = False) -> str:
    """此 lowering 的人类可读文本表示。

    用于可视化和调试目的。它不必是有效或可靠的
    序列化形式。
    若需要可靠且可移植的序列化，请使用 `jax.export`。

    Args:
      dialect: 可选字符串，指定降级方言（例如 "stablehlo"
        或 "hlo"）。
      debug_info: 是否包含调试信息，
        例如源码位置。
    """
    return self._lowering.as_text(dialect, debug_info=debug_info)

  def compiler_ir(self, dialect: str | None = None) -> Any | None:
    """此 lowering 的任意对象表示。

    用于调试目的。它不是有效或可靠的
    序列化形式。其输出不保证在多次调用之间
    保持一致。
    若需要可靠且可移植的序列化，请使用 `jax.export`。

    在不可用时返回 ``None``，例如取决于后端、编译器或
    运行时。

    Args:
      dialect: 可选字符串，指定降级方言（例如 "stablehlo"
        或 "hlo"）。
    """
    try:
      return self._lowering.compiler_ir(dialect)
    except NotImplementedError:
      return None

  def cost_analysis(self) -> Any | None:
    """执行代价估计的摘要。

    用于可视化和调试目的。它输出的对象
    是一些易于打印或序列化的简单数据结构
    （例如以数值为叶子的嵌套 dict、list 和 tuple）。不过
    其结构可以是任意的：它可能在 JAX 与 jaxlib 的
    不同版本之间不一致，甚至可能在多次调用之间不一致。

    在不可用时返回 ``None``，例如取决于后端、编译器或
    运行时。
    """
    # TODO(frostig): 改进类型注解（任意结构的基础 pytree）
    try:
      return self._lowering.cost_analysis()
    except NotImplementedError:
      return None


class Compiled(Stage):
  """针对类型/值特化后的函数的已编译表示。

  已编译的计算关联着一个可执行文件，以及执行它
  所需的其余信息。它还提供统一的 API，
  用于在 JAX 各种编译路径和后端上查询
  已编译计算的属性。
  """
  __slots__ = ["args_info", "out_tree", "_executable", "_no_kwargs", "_params"]

  # ArgInfo 的 PyTree，包含死参数，但不包含 const_args
  args_info: Any
  out_tree: tree_util.PyTreeDef
  _executable: Executable
  _no_kwargs: bool
  _params: CompiledCallParams

  def __init__(self, executable, const_args: list[ArrayLike],
               args_info, out_tree, no_kwargs=False, in_types=None, out_types=None):
    self._executable = executable
    self._no_kwargs = no_kwargs
    self.args_info = args_info
    self.out_tree = out_tree
    self._params = CompiledCallParams(
        self._executable, self._no_kwargs, self.in_tree, self.out_tree,
        const_args, in_types, out_types)
    self._call = None

  def as_text(self) -> str | None:
    """此可执行文件的人类可读文本表示。

    用于可视化和调试目的。它不是有效或可靠的
    序列化形式。

    在不可用时返回 ``None``，例如取决于后端、编译器或
    运行时。
    """
    try:
      return self._executable.as_text()
    except NotImplementedError:
      return None

  def cost_analysis(self) -> Any | None:
    """执行代价估计的摘要。

    用于可视化和调试目的。它输出的对象
    是一些易于打印或序列化的简单数据结构
    （例如以数值为叶子的嵌套 dict、list 和 tuple）。不过
    其结构可以是任意的：它可能在 JAX 与 jaxlib 的
    不同版本之间不一致，甚至可能在多次调用之间不一致。

    在不可用时返回 ``None``，例如取决于后端、编译器或
    运行时。
    """
    # TODO(frostig): 改进类型注解（任意结构的基础 pytree）
    try:
      return self._executable.cost_analysis()
    except NotImplementedError:
      return None

  def memory_analysis(self) -> Any | None:
    """估计内存需求的摘要。

    用于可视化和调试目的。它输出的对象
    是一些易于打印或序列化的简单数据结构
    （例如以数值为叶子的嵌套 dict、list 和 tuple）。不过
    其结构可以是任意的：它可能在 JAX 与 jaxlib 的
    不同版本之间不一致，甚至可能在多次调用之间不一致。

    在不可用时返回 ``None``，例如取决于后端、编译器或
    运行时。
    """
    # TODO(frostig): 改进类型注解（任意结构的基础 pytree）
    try:
      return self._executable.memory_analysis()
    except NotImplementedError:
      return None

  @property
  def in_avals(self):
    # 包含死参数，但不包含 const_args
    nr_const_args = len(self._params.const_args)
    in_avals_ = self._executable.in_avals[nr_const_args:]  # pyrefly: ignore[missing-attribute]
    if self.in_tree.num_leaves > len(in_avals_):
      iter_in_avals = iter(in_avals_)
      non_dce_avals = self._executable._all_args_info.in_avals[nr_const_args:]  # pyrefly: ignore[missing-attribute]
      in_avals_ = [
          next(iter_in_avals) if i + nr_const_args in self._executable._kept_var_idx # pyrefly: ignore[missing-attribute]
          else a for i, a in zip(range(self.in_tree.num_leaves), non_dce_avals)]
    return self.in_tree.unflatten(in_avals_)

  @property
  def out_info(self):  # jax.ShapeDtypeStruct 的 PyTree
    out_avals = self._executable.out_avals  # pyrefly: ignore[missing-attribute]
    out_formats_flat = self._output_formats_flat
    return self.out_tree.unflatten(
        [core.ShapeDtypeStruct(o.shape, o.dtype, sharding=f)
         for o, f in zip(out_avals, out_formats_flat)])

  def runtime_executable(self) -> Any | None:
    """此可执行文件的任意对象表示。

    用于调试目的。它不是有效或可靠的
    序列化形式。其输出不保证在多次调用之间
    保持一致。

    在不可用时返回 ``None``，例如取决于后端、编译器或
    运行时。
    """
    return self._executable.runtime_executable()

  def _input_shardings_flat(self):
    nr_const_args = len(self._params.const_args)
    shardings_flat = self._executable._in_shardings[nr_const_args:]  # pyrefly: ignore[missing-attribute]
    # 一些输入分片被 DCE（死代码消除）掉了
    if self.in_tree.num_leaves > len(shardings_flat):
      iter_shardings_flat = iter(shardings_flat)
      shardings_flat = [next(iter_shardings_flat) if i + nr_const_args in self._executable._kept_var_idx  # pyrefly: ignore[missing-attribute]
                        else None for i in range(self.in_tree.num_leaves)]
    return shardings_flat

  @property
  def input_shardings(self):  # -> PyTree[sharding.Sharding]
    # 包含死参数，但不包含 const_args
    shardings_flat = self._input_shardings_flat()
    return tree_util.tree_unflatten(self.in_tree, shardings_flat)

  @property
  def output_shardings(self):  # -> PyTree[sharding.Sharding]
    shardings_flat = self._executable._out_shardings  # pyrefly: ignore[missing-attribute]
    return tree_util.tree_unflatten(self.out_tree, shardings_flat)

  def _input_layouts_flat(self):
    nr_const_args = len(self._params.const_args)
    layouts_flat = self._executable._xla_in_layouts[nr_const_args:]  # pyrefly: ignore[missing-attribute]
    # 一些输入布局被 DCE 掉了
    if self.in_tree.num_leaves > len(layouts_flat):
      iter_layouts_flat = iter(layouts_flat)
      layouts_flat = [next(iter_layouts_flat) if i + nr_const_args in self._executable._kept_var_idx  # pyrefly: ignore[missing-attribute]
                      else None for i in range(self.in_tree.num_leaves)]
    return layouts_flat

  @property
  def input_formats(self):
    # 包含死参数，但不包含 const_args
    layouts_flat = self._input_layouts_flat()
    shardings_flat = self._input_shardings_flat()
    formats_flat = [Format(l, s) for l, s in zip(layouts_flat, shardings_flat)]
    return tree_util.tree_unflatten(self.in_tree, formats_flat)

  @property
  def _output_formats_flat(self):
    layouts_flat = self._executable._xla_out_layouts  # pyrefly: ignore[missing-attribute]
    shardings_flat = self._executable._out_shardings  # pyrefly: ignore[missing-attribute]
    assert all(isinstance(l, Layout) for l in layouts_flat)
    return [Format(l, s) for l, s in zip(layouts_flat, shardings_flat)]

  @property
  def output_formats(self):
    formats_flat = self._output_formats_flat
    return tree_util.tree_unflatten(self.out_tree, formats_flat)

  @staticmethod
  def call(*args, **kwargs):
    util.test_event("stages_compiled_call")
    # 这是因为 `__call__` 会把 `self._params` 作为第一个参数传入。
    # 这里没有把调用签名写成 `call(params, *args, **kwargs)`，
    # 而是从 args 中把它提取出来，因为用户可能以关键字参数的形式传入
    # `params`，那样会与此处冲突。
    params = args[0]
    args = args[1:]  # 不包含 const_args
    if params.no_kwargs and kwargs:
      kws = ', '.join(kwargs.keys())
      raise NotImplementedError(
          "function was compiled by a transformation that does not support "
          f"keyword arguments, but called with keyword arguments: {kws}")

    if params.is_high:
      hi_args_flat, hi_tree = tree_util.tracing_registry.flatten((args, kwargs))
      args_flat = [typeof(x).lower_val(x) for x in hi_args_flat]
      args_flat, in_tree = tree_util.tracing_registry.flatten(
          tree_util.tree_unflatten(hi_tree, args_flat))
    else:
      args_flat, in_tree = tree_util.tracing_registry.flatten((args, kwargs))

    # TODO(mattjj): 改进参数个数错误的报错
    if in_tree != params.in_tree:
      errs = list(tree_util.equality_errors_pytreedef(in_tree, params.in_tree))
      msg = []
      msg.append(
          "Function compiled with input pytree does not match the input pytree"
          f" it was called with. There are {len(errs)} mismatches, including:")
      for path, thing1, thing2, explanation in errs:
        fst, *rest = path
        base = ['args', 'kwargs'][fst.idx]
        msg.append(
            f"    * at {base}{tree_util.keystr(tuple(rest))}, seen {thing2} but now"
            f" given {thing1}, so {explanation}")
      raise TypeError('\n'.join(msg))

    if not core.trace_state_clean():
      # 只有在处于某个变换之下时我们才检查追踪器，在常见路径上
      # 跳过该检查。我们无法变换提前（ahead-of-time）编译的
      # 调用，因为我们是针对固定的函数签名做降级和编译的，
      # 而 JAX 的变换会改变签名。
      for arg in args_flat:
        if isinstance(arg, core.Tracer):
          raise TypeError(
              "Cannot apply JAX transformations to a function lowered and "
              "compiled for a particular signature. Detected argument of "
              f"Tracer type {type(arg)}.")
    lo_outs = params.executable.call(*params.const_args, *args_flat)

    if params.is_high:
      out_hi_tree, out_hi_types = params.out_types
      out_flat = raise_lo_outs(out_hi_types, lo_outs)
      outs = tree_util.tree_unflatten(out_hi_tree, out_flat)
    else:
      out_flat = lo_outs
      outs = tree_util.tree_unflatten(params.out_tree, out_flat)

    return outs, out_flat, args_flat

  def __call__(self, *args, **kwargs):
    if self._call is None:
      self._call = self._executable.create_cpp_call(self._params)
      if self._call is None:
        params = self._params
        def cpp_call_fallback(*args, **kwargs):
          outs, _, _ = Compiled.call(params, *args, **kwargs)
          return outs
        self._call = cpp_call_fallback
    return self._call(*args, **kwargs)

# TODO(mattjj): 与 partial_eval.py 去重
def raise_lo_outs(hi_avals, lo_outs):
  lo_outs_ = iter(lo_outs)
  hi_outs = [t.raise_val(*it.islice(lo_outs_, len(t.lo_ty()))) for t in hi_avals]
  assert next(lo_outs_, None) is None
  return hi_outs

@runtime_checkable
class Wrapped(Protocol):
  """一个已准备好被追踪、降级和编译的函数。

  此协议反映 `jax.jit` 等函数的返回值。
  调用它会进行 JIT（即时）降级、
  编译和执行。它也可以在编译之前显式降级，
  并在执行之前编译其结果。
  """

  def __call__(self, *args, **kwargs):
    """执行被包装的函数，按需进行降级和编译。"""
    raise NotImplementedError

  def trace(self, *args, **kwargs) -> Traced:
    """针对给定参数显式追踪此函数。

    被追踪的函数会从 Python 中暂存出来并翻译成 jaxpr。它
    已准备好降级，但尚未降级。

    Returns:
      表示本次追踪的 ``Traced`` 实例。
    """
    raise NotImplementedError

  def lower(self, *args, **kwargs) -> Lowered:
    """针对给定参数显式降级此函数。

    这是 ``self.trace(*args, **kwargs).lower()`` 的快捷方式。

    被降级的函数会从 Python 中暂存出来，并翻译为
    编译器的输入语言，这一过程可能依赖于具体后端。
    它已准备好编译，但尚未编译。

    Returns:
      表示本次降级的 ``Lowered`` 实例。
    """
    raise NotImplementedError


class MismatchType(enum.Enum):
  ARG_SHARDING = enum.auto()
  CONST_SHARDING = enum.auto()
  OUT_SHARDING = enum.auto()
  SHARDING_INSIDE_COMPUTATION = enum.auto()
  CONTEXT_DEVICES = enum.auto()
  IN_SHARDING = enum.auto()

  def __str__(self):
    if self.name == 'IN_SHARDING':
      return 'explicit input sharding'
    if self.name == 'CONST_SHARDING':
      return 'closed over constant sharding'
    elif self.name == 'OUT_SHARDING':
      return 'explicit output sharding'
    elif self.name == 'CONTEXT_DEVICES':
      return 'context mesh'
    return f'{self.name}'


class SourceInfo(NamedTuple):
  source_info: source_info_util.SourceInfo
  eqn_name: str


@dataclasses.dataclass(slots=True)
class DeviceAssignmentMismatch:
  da: Sequence[xc.Device] | int
  m_type: MismatchType
  source_info: SourceInfo | None

  @property
  def device_ids(self) -> Sequence[int]:
    return [d.id for d in self.da]  # pyrefly: ignore[not-iterable]

  @property
  def platform(self) -> str:
    return self.da[0].platform.upper()  # pyrefly: ignore[bad-index]

  def _maybe_api_name(self, api_name) -> str:
    return f" {api_name}'s" if self.m_type == MismatchType.CONTEXT_DEVICES else ""

  @property
  def source_info_str(self):
    return (
        "" if self.source_info is None
        else f" at {source_info_util.summarize(self.source_info.source_info)}"
    )

  @property
  def _dev_ids_plat_str(self):
    if isinstance(self.da, int):
      return f"as AbstractMesh of size {self.da}"
    else:
      return f"with device ids {self.device_ids} on platform {self.platform}"

  def m_type_str(self, api_name):
    return (f'{self.source_info and self.source_info.eqn_name} inside {api_name}'
            if self.m_type == MismatchType.SHARDING_INSIDE_COMPUTATION else self.m_type)

  def _str(self, api_name):
    return (f"{self._maybe_api_name(api_name)} {self.m_type_str(api_name)} "
            f"{self._dev_ids_plat_str}{self.source_info_str}")


class DeviceAssignmentMismatchError(Exception):
  pass


def _find_arg_mismatch(arg_list, fails, fun_name):
  mismatched_args_msg = []
  def mismatch(err):
    for name, inp_da, aval in arg_list:
      if err.m_type == MismatchType.ARG_SHARDING and err.da == inp_da:
        mismatched_args_msg.append(
            f"argument {name} of {fun_name} with shape {aval.str_short()} and "
            f"{err._dev_ids_plat_str}")
        break
  first_err, second_err = fails
  mismatch(first_err)
  mismatch(second_err)
  return mismatched_args_msg


def _device_assignment_mismatch_error(fun_name, fails, args_flat, api_name,
                                      arg_names):
  arg_list = []
  if arg_names is None:
    arg_names = [''] * len(args_flat)
  for a, n in zip(args_flat, arg_names):
    da = a.sharding._device_assignment if a.sharding is not None else None
    arg_list.append((n, da, a.aval))

  mismatched_args_msg = _find_arg_mismatch(arg_list, fails, fun_name)

  if len(mismatched_args_msg) == 2:
    first, second = mismatched_args_msg
    extra_msg = f" Got {first} and {second}"
  elif len(mismatched_args_msg) == 1:
    first, second = fails
    # 选择尚未被 ARG_SHARDING 覆盖的那一侧失败。
    left = second if first.m_type == MismatchType.ARG_SHARDING else first
    extra_msg = f" Got {mismatched_args_msg[0]} and{left._str(api_name)}"
  else:
    first, second = fails
    extra_msg = f" Got{first._str(api_name)} and{second._str(api_name)}"
  msg = (f"Received incompatible devices for {api_name}ted computation.{extra_msg}")
  return msg
