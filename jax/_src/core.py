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

# 文件职责：JAX 追踪（tracing）与中间表示的核心。
# 定义 Tracer/Trace（追踪器与追踪层级）、AbstractValue/ShapedArray 等抽象值(aval)、
# Jaxpr 与 JaxprEqn（含效果 effect 与调试信息），以及 Var/Atom/Literal 等 jaxpr 构件，
# 并提供 trace_to_jaxpr、eval_jaxpr、new_ref 等基础设施。
# 它是 jit/grad/vmap 等所有变换与降级（lowering）共同依赖的底座；公开 API 见 `jax.core`。
from __future__ import annotations

from collections import Counter, defaultdict, deque, namedtuple
from collections.abc import (Callable, Collection, Hashable, Iterable, Iterator,
                             Sequence, MutableSet, MutableMapping)
from contextlib import contextmanager
from dataclasses import dataclass
import enum
import functools
from functools import partial, total_ordering
import gc
import inspect
import itertools as it
import math
import operator
import re
import sys
import threading
import types
from typing import (Any, ClassVar, NamedTuple, final, overload,
                    TYPE_CHECKING, Literal as Literal_)
import warnings
import weakref

import numpy as np

from jax._src import dtypes
from jax._src import config
from jax._src import effects
from jax._src.frozen_dict import FrozenDict
from jax._src import mesh as mesh_lib
from jax._src.mesh import AxisType
from jax._src.partition_spec import PartitionSpec as P, UnreducedKind
from jax._src.errors import (
    ConcretizationTypeError, TracerArrayConversionError, TracerBoolConversionError,
    TracerIntegerConversionError, UnexpectedTracerError)
from jax._src import flattree as ft
from jax._src import linear_util as lu
from jax._src.tree_util import tree_map
from jax._src import source_info_util
from jax._src.util import (safe_zip, safe_map, curry, tuple_insert,
                           tuple_delete, cache, HashableWrapper,
                           weakref_lru_cache, partition_list, StrictABCMeta,
                           foreach, weakref_cache_key_types, set_module,
                           weak_value_interner, immutable)
import jax._src.pretty_printer as pp
from jax._src.named_sharding import NamedSharding, get_replicated_axes
from jax._src import named_sharding as ns
from jax._src.sharding import Sharding
from jax._src.layout import Format, AutoLayout, AutoLayoutSingleton
from jax._src.lib import _jax
from jax._src import traceback_util
from jax._src.typing import Array, ArrayLike, DimSize, Shape
from jax._src import xla_metadata_lib

traceback_util.register_exclusion(__file__)

zip, unsafe_zip = safe_zip, zip
map, unsafe_map = safe_map, map

config_ext = _jax.config

PyTree = Any


_TRACER_ERROR_NUM_TRACEBACK_FRAMES = config.int_flag(
    'jax_tracer_error_num_traceback_frames',
    config.int_env('JAX_TRACER_ERROR_NUM_TRACEBACK_FRAMES', 5),
    help='Set the number of stack frames in JAX tracer error messages.'
)

def identity(x): return x

# -------------------- jaxprs --------------------

Effect = effects.Effect
Effects = effects.Effects
EffectTypeSet = effects.EffectTypeSet
no_effects: Effects = effects.no_effects


DebugInfo = lu.DebugInfo
InitialResultPaths = lu.InitialResultPaths
initial_result_paths = lu.initial_result_paths

class Jaxpr:
  __slots__ = [
      "__weakref__",
      "_all_invars",
      "_outvars",
      "_eqns",
      "_effects",
      "_debug_info",
      "_is_high",
      "_consts",
  ]
  # 所有输入变量；前 len(_consts) 个已附常量值，其余由调用者传入。
  _all_invars: list[Var]
  # 按返回顺序排列的输出，可以是计算结果 Var 或已知值 Literal。
  _outvars: list[Atom]
  # 程序体中的方程，按求值顺序记录 primitive 调用。
  _eqns: list[JaxprEqn]
  # 整个 Jaxpr 的效果集合，由方程的 effects 汇总而来。
  _effects: Effects
  # 被追踪函数、输入名和结果路径等信息，用于诊断与报错。
  _debug_info: DebugInfo
  # 是否涉及 HiJAX 原语或抽象值；决定何时需要转成 LoJAX。
  _is_high: bool
  # 与 _all_invars 前缀中的变量一一对应的已附常量值。
  _consts: list[Any]

  @property
  def all_invars(self) -> list[Var]:
    return self._all_invars

  @property
  def constvars(self) -> list[Var]:
    # 常量输入恰好就是那些附带了值的输入；没有附加值
    # 的输入是普通的 invar。
    return self._all_invars[: len(self._consts)]

  @property
  def consts(self) -> list[Any]:
    return self._consts

  literals = consts  # consts 的旧 ClosedJaxpr 名称

  @property
  def num_consts(self) -> int:
    return len(self._consts)

  @property
  def jaxpr(self) -> Jaxpr:
    # 来自 ClosedJaxpr 时代的旧访问器，ClosedJaxpr 曾包装一个 Jaxpr。
    # TODO(dougalm): 移除用法并删除。
    return self

  @property
  def invars(self) -> list[Var]:
    num_consts = len(self._consts)
    return self._all_invars[num_consts:] if num_consts else self._all_invars

  @property
  def outvars(self) -> list[Atom]:
    return self._outvars

  @property
  def eqns(self) -> list[JaxprEqn]:
    return self._eqns

  @property
  def effects(self) -> Effects:
    return self._effects

  @property
  def debug_info(self) -> DebugInfo:
    return self._debug_info

  @property
  def is_high(self) -> bool:
    return self._is_high

  @property
  def in_avals(self):
    return [v.aval for v in self.invars]

  @property
  def out_avals(self):
    return [v.aval for v in self.outvars]

  def __init__(
      self,
      constvars: Sequence[Var] | Jaxpr,
      invars: Sequence[Var] | Sequence[Any] | None = None,
      outvars: Sequence[Atom] | None = None,
      eqns: Sequence[JaxprEqn] | None = None,
      effects: Effects = no_effects,
      # 我们希望所有调用都传入 DebugInfo 对象，但为了向后兼容，
      # 必须允许 debug_info 缺失时的调用。
      debug_info: DebugInfo = None,  # pyrefly: ignore[bad-function-definition]
      is_high: bool = False,
      consts: Sequence[Any] | None = None,
  ):
    if isinstance(constvars, Jaxpr):
      # 旧的 ClosedJaxpr(jaxpr, consts) 构造方式：共享 `jaxpr` 的
      # 结构，并把 `consts` 附加为其常量参数值。
      # TODO(dougalm): 迁移调用方并移除。
      jaxpr = constvars
      assert outvars is None and eqns is None and debug_info is None
      if consts is None:
        jaxpr, consts = constvars, invars
      else:
        assert invars is None
      assert consts is not None and len(consts) <= len(jaxpr._all_invars)
      self._all_invars = jaxpr._all_invars
      self._outvars = jaxpr._outvars
      self._eqns = jaxpr._eqns
      self._effects = jaxpr._effects
      self._debug_info = _shift_arg_names(jaxpr._debug_info,
                                          len(jaxpr._consts) - len(consts))
      self._is_high = jaxpr._is_high
      self._consts = list(consts)
      return
    assert invars is not None and outvars is not None and eqns is not None
    self._consts = [] if consts is None else list(consts)
    assert (
        not self._consts or len(self._consts) == len(constvars)
    ), "consts, when attached, must pair with constvars"
    self._all_invars = [*constvars, *invars]
    self._outvars = list(outvars)
    self._eqns = list(eqns)
    self._effects = effects
    # TODO(https://github.com/jax-ml/jax/issues/26480)
    debug_info = debug_info or lu._missing_debug_info("core.Jaxpr")
    debug_info = debug_info.resolve_result_paths()
    if constvars and not self._consts:
      # 没有附加值的 constvars 就是普通的前导 invars。
      debug_info = _shift_arg_names(debug_info, len(constvars))
    self._debug_info = debug_info
    config.enable_checks.value and self._debug_info.assert_arg_names(len(self.invars))
    config.enable_checks.value and self._debug_info.assert_result_paths(len(outvars))
    self._is_high = is_high

  def __str__(self):
    return str(self.pretty_print())

  __repr__ = __str__

  def pretty_print(self, *, source_info=False, print_shapes=True,
                   custom_pp_eqn_rules=True, name_stack=False,
                   print_effects: bool = False, **kwargs):
    doc = pp_toplevel_jaxpr(
      self, source_info=source_info, print_shapes=print_shapes,
      custom_pp_eqn_rules=custom_pp_eqn_rules, name_stack=name_stack,
      print_effects=print_effects)
    return doc.format(**kwargs)

  def _repr_pretty_(self, p, cycle):
    return p.text(self.pretty_print(use_color=True))

  def with_consts(self, consts: Sequence[Any]) -> Jaxpr:
    """返回此 jaxpr 的一个副本，其中把 `consts` 附加为其前
    `len(consts)` 个输入的值，这些输入因而成为它的 constvars。
    共享所有其它结构。"""
    consts = list(consts)
    assert len(consts) <= len(self._all_invars)
    new = Jaxpr.__new__(Jaxpr)
    new._all_invars = self._all_invars
    new._outvars = self._outvars
    new._eqns = self._eqns
    new._effects = self._effects
    new._debug_info = _shift_arg_names(self._debug_info,
                                       len(self._consts) - len(consts))
    new._is_high = self._is_high
    new._consts = consts
    return new

  def map_jaxpr(self, f):
    # 旧的 ClosedJaxpr 方法：把 f 应用到 jaxpr 上，并保留 consts。
    return Jaxpr(f(self), self.consts)

  def replace(self, **kwargs):
    if "jaxpr" in kwargs:
      # 旧的 ClosedJaxpr.replace(jaxpr=..., consts=...) 形式。
      # TODO(dougalm): 迁移调用方并移除。
      jaxpr = kwargs.pop("jaxpr")
      consts = kwargs.pop("consts", None)
      if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs}")
      jaxpr = self if jaxpr is None else jaxpr
      consts = self.consts if consts is None else consts
      return Jaxpr(jaxpr, consts)
    debug_default = self.debug_info
    if (kwargs.get('invars', self.invars) != self.invars or
        kwargs.get('outvars', self.outvars) != self.outvars):
      debug_default = debug_default.with_unknown_names()
    # 替换 constvars 会使 consts 的配对失效，因此除非显式给出新的
    # consts，否则结果不会附加任何 consts。
    consts_default = () if "constvars" in kwargs else self.consts
    jaxpr = Jaxpr(
        constvars=kwargs.pop("constvars", self.constvars),
        invars=kwargs.pop("invars", self.invars),
        outvars=kwargs.pop("outvars", self.outvars),
        eqns=kwargs.pop("eqns", self.eqns),
        effects=kwargs.pop("effects", self.effects),
        debug_info=kwargs.pop("debug_info", debug_default),
        is_high=kwargs.pop("is_high", self.is_high),
        consts=kwargs.pop("consts", consts_default),
    )
    if kwargs:
      raise ValueError(f"Unknown keyword arguments: {kwargs}")
    return jaxpr

weakref_cache_key_types.add(Jaxpr)

def _shift_arg_names(dbg: DebugInfo, delta: int) -> DebugInfo:
  # 随着 const/invar 边界移动，保持调试用 arg_names 与 `invars` 对齐：
  # delta > 0 会把这么多的 constvars 暴露为 invars（用无名条目填充），
  # delta < 0 会把这么多的前导 invars 变成 constvars。
  if not delta or dbg.arg_names is None:
    return dbg
  if delta > 0:
    return dbg._replace(arg_names=("",) * delta + tuple(dbg.arg_names))
  return dbg._replace(arg_names=tuple(dbg.arg_names[-delta:]))

def join_effects(*effects: Effects) -> Effects:
  return set().union(*effects) if effects else no_effects

def jaxprs_in_params(params) -> Iterator[Jaxpr]:
  for val in params.values():
    vals = val if isinstance(val, tuple) else (val,)
    for v in vals:
      if isinstance(v, Jaxpr):
        yield v


def subjaxprs(jaxpr: Jaxpr) -> Iterator[Jaxpr]:
  """生成器，用于生成在 jaxpr.eqns 的 params 中找到的所有子 jaxpr。
  不会递归下降到找到的子 jaxprs 中。
  """
  for eqn in jaxpr.eqns:
    yield from jaxprs_in_params(eqn.params)


# ClosedJaxpr 和 Jaxpr 已经合并为单个类：Jaxpr 携带一个
# 可能为空的常量参数值列表 `consts`。ClosedJaxpr 这个名字
# 仍作为别名保留，供那些通过 ClosedJaxpr(jaxpr, consts) 构造
# 封闭 jaxpr、或在 isinstance 检查和注解中使用它的调用方使用。
# TODO(dougalm): 迁移这些调用方，并移除该别名。
ClosedJaxpr = Jaxpr


@curry
def jaxpr_as_fun(closed_jaxpr: Jaxpr, *args):
  # TODO(dougalm): 当我们给 jaxpr 添加上下文后，移除这个 hack。
  # debug_nans 有时会在可追踪层面被那些内部处理 nan 的算子
  # （如 jnp.var）局部禁用。正确的做法是给我们的 jaxpr
  # 表示添加上下文，这样我们就能捕获这些局部上下文修改。
  # 在此期间，在往返过程中禁用这些检查可以防止
  # 那些算子产生虚假的错误。
  with config.debug_nans(False):
    return eval_jaxpr(closed_jaxpr, closed_jaxpr.consts, *args)


# 这个上下文管理器处于热点路径，因为它会针对每个 jaxpr 方程被频繁调用。
# 这个上下文管理器被实现为一个带有显式 __enter__ 和 __exit__
# 方法的类，因为 @contextlib.contextmanager 明显更慢。
# 我们还实际上把另外四个上下文管理器融合为一个，主要是
# 为了节省内存分配。
class JaxprEqnContextManager:
  __slots__ = ['context', 'prev_compute_type', 'prev_threefry_partitionable',
               'prev_xla_metadata', 'prev_abstract_mesh',
               'prev_remove_size_one_mesh_axis']

  def __init__(self, context):
    self.context = context

  def __enter__(self):
    self.prev_xla_metadata = config.xla_metadata_context_manager.get_local()
    if (self.context.xla_metadata and
        (self.prev_xla_metadata is None or
         self.prev_xla_metadata is config_ext.unset or
         self.prev_xla_metadata.val != self.context.xla_metadata)):
      updated = xla_metadata_lib.update_metadata(
          self.prev_xla_metadata, self.context.xla_metadata)
      config.xla_metadata_context_manager.set_local(updated)

    self.prev_threefry_partitionable = config.threefry_partitionable.swap_local(
        self.context.threefry_partitionable)
    self.prev_compute_type = config.compute_on_context_manager.swap_local(
        self.context.compute_type)
    self.prev_abstract_mesh = config.abstract_mesh_context_manager.swap_local(
        self.context.cur_abstract_mesh)
    self.prev_remove_size_one_mesh_axis = config.remove_size_one_mesh_axis_from_type.swap_local(
        self.context.remove_size_one_mesh_axis)

  def __exit__(self, exc_type, exc_value, traceback):
    config.xla_metadata_context_manager.set_local(self.prev_xla_metadata)
    config.threefry_partitionable.set_local(self.prev_threefry_partitionable)
    config.compute_on_context_manager.set_local(self.prev_compute_type)
    config.abstract_mesh_context_manager.set_local(self.prev_abstract_mesh)
    config.remove_size_one_mesh_axis_from_type.set_local(self.prev_remove_size_one_mesh_axis)


@immutable
class JaxprEqnContext:

  __slots__ = ['compute_type', 'threefry_partitionable', 'cur_abstract_mesh',
               'remove_size_one_mesh_axis', 'xla_metadata', 'configs',
               '__weakref__']

  compute_type: str | None
  threefry_partitionable: bool
  xla_metadata: dict[str, Any] | None
  cur_abstract_mesh: mesh_lib.AbstractMesh
  remove_size_one_mesh_axis: bool

  @staticmethod
  @weak_value_interner
  def _create(compute_type, threefry_partitionable, cur_abstract_mesh,
              remove_size_one_mesh_axis, xla_metadata):
    obj = object.__new__(JaxprEqnContext)
    object.__setattr__(obj, 'compute_type', compute_type)
    object.__setattr__(obj, 'threefry_partitionable', threefry_partitionable)
    object.__setattr__(obj, 'cur_abstract_mesh', cur_abstract_mesh)
    object.__setattr__(obj, 'remove_size_one_mesh_axis', remove_size_one_mesh_axis)
    object.__setattr__(obj, 'xla_metadata',
                       None if xla_metadata is None else dict(xla_metadata))
    return obj

  def __new__(cls):
    compute_type = config.compute_on_context_manager.value
    threefry_partitionable = config.threefry_partitionable.value
    cur_abstract_mesh = mesh_lib.get_abstract_mesh()
    remove_size_one_mesh_axis = config.remove_size_one_mesh_axis_from_type.value
    xla_metadata = xla_metadata_lib.current_xla_metadata()
    xla_metadata = (None if xla_metadata is None else
                    tuple(sorted(xla_metadata.items())))
    return JaxprEqnContext._create(
        compute_type, threefry_partitionable, cur_abstract_mesh,
        remove_size_one_mesh_axis, xla_metadata)

  # 没有 __eq__ 或 __hash__：驻留的类使用对象标识。

  @property
  def manager(self):
    return JaxprEqnContextManager(self)

  def __repr__(self):
    return (f"JaxprEqnContext(compute_type={self.compute_type}, "
            f"threefry_partitionable={self.threefry_partitionable}, "
            f"cur_abstract_mesh={self.cur_abstract_mesh}, "
            f"remove_size_one_mesh_axis={self.remove_size_one_mesh_axis}, "
            f"xla_metadata={self.xla_metadata})")


@cache()  # 上下文中的一切同样是追踪缓存键。
def current_jaxpr_eqn_context():
  return JaxprEqnContext()

class JaxprEqn:
  invars: list[Atom]
  outvars: list[Var]
  primitive: Primitive
  params: dict[str, Any]
  effects: Effects

  # source_info.name_stack 始终（仅）相对于外层 jaxpr，
  # 并且不包含来自 jaxpr 调用方的任何名称上下文。毕竟
  # 一个 jaxpr 可能有多个调用方。
  # TODO(phawkins): 把 source_info.tracebacks 也更新为相对于
  # 外层 jaxpr。
  source_info: source_info_util.SourceInfo
  ctx: JaxprEqnContext

  # 使用带 __slots__ 的类比使用 NamedTuple 略快一些。
  __slots__ = ['invars', 'outvars', 'primitive', 'params', 'effects',
               'source_info', 'ctx']

  def __init__(self, invars, outvars, primitive, params, effs, source_info,
               ctx):
    self.invars = invars
    self.outvars = outvars
    self.primitive = primitive
    self.params = params
    self.effects = effs
    self.source_info = source_info
    self.ctx = ctx

  def __repr__(self):
    return str(pp_eqn(self, JaxprPpContext(), JaxprPpSettings())).rstrip()

  def replace(
      self,
      invars: list[Atom] | None = None,
      outvars: list[Var] | None = None,
      primitive: Primitive | None = None,
      params: dict[str, Any] | None = None,
      effects: Effects | None = None,
      source_info: source_info_util.SourceInfo | None = None,
      ctx: JaxprEqnContext | None = None
  ):
    return JaxprEqn(
      self.invars if invars is None else invars,
      self.outvars if outvars is None else outvars,
      self.primitive if primitive is None else primitive,
      self.params if params is None else params,
      self.effects if effects is None else effects,
      self.source_info if source_info is None else source_info,
      self.ctx if ctx is None else ctx,
    )


# TODO(mattjj): 在这里调用类型检查规则，这样我们就不会形成错误的方程
def new_jaxpr_eqn(invars, outvars, primitive, params, effects, source_info=None,
                  ctx=None) -> JaxprEqn:
  source_info = source_info or source_info_util.new_source_info()
  ctx = ctx or current_jaxpr_eqn_context()
  effects = resolve_input_effects(effects, invars)
  if config.enable_checks.value:
    assert all(isinstance(x, (Var, Literal)) for x in  invars)
    assert all(isinstance(v,  Var)           for v in outvars)
  return JaxprEqn(invars, outvars, primitive, params, effects, source_info, ctx)


def resolve_input_effects(effs, invars) -> Effects:
  if not any(isinstance(e, effects.JaxprInputEffect) and isinstance(e.input, int)
             for e in effs):
    return effs
  out_effs = set()
  for eff in effs:
    if isinstance(eff, effects.JaxprInputEffect) and isinstance(eff.input, int):
      invar = invars[eff.input]
      if isinstance(invar, Literal):
        continue
      eff = eff.replace(invar)
    out_effs.add(eff)
  return out_effs

class Var:
  __slots__ = ["aval"]

  aval: AbstractValue

  def __init__(self, aval: AbstractValue):
    assert isinstance(aval, AbstractValue), aval
    self.aval = aval

  def __repr__(self):
    return f'Var(id={id(self)}):{self.aval.str_short()}'

  def pretty_print(self, context: JaxprPpContext, *, print_dtype: bool = True):
    del print_dtype  # 未使用
    return f"{context.var_names[self]}"


gensym = lambda: Var

# 在 jaxpr 中，`dropvar` 可以出现在绑定变量的位置上，以表示
# 该赋值被丢弃，即某个表达式的输出值永远不会
# 被读取。就此而言，`dropvar` 并不是一个变量，但把它
# 当作变量的一个特例来处理很方便。它的 `aval` 同样是不精确的。
class DropVar(Var):
  def __init__(self, aval: AbstractValue):
    super().__init__(aval)
  def __repr__(self): return '_'
  def pretty_print(self, context: JaxprPpContext, *, print_dtype: bool = True):
    del context, print_dtype  # 未使用
    return '_'

@final
class Literal:
  # 参见 https://docs.jax.dev/en/latest/internals/constants.html
  __slots__ = ["val", "aval"]

  val: Any
  aval: AbstractValue

  def __init__(self, val, aval):
    self.val = val
    self.aval = aval

  @property
  def hash(self):
    try:
      return hash(self.val)
    except TypeError:
      if type(self.val) in literalable_types:
        try:
          return hash((self.val.item(), self.val.dtype))
        except (TypeError, AttributeError, ValueError):
          return None

  __hash__ = None

  def pretty_print(self, context: JaxprPpContext, *, print_dtype: bool = True):
    del context  # 未使用
    dtype = getattr(self.aval, 'dtype', None)
    if not np.shape(self.val):
      val_str = str(np.asarray(self.val).item())
    else:
      val_str = "[...]"
    if print_dtype and dtype:
      return f'{val_str}:{self.aval.str_short(short_dtypes=True)}'
    else:
      return val_str

  def __repr__(self):
    return f'Literal({self.val})'

# 可与 core.Literal 一起使用的常量类型。其它常量
# 最终会成为 `constvars`。
literalable_types: set[type] = set()
literalable_scalar_types: set[type] = set()

def is_literalable(x: Any, for_ad: bool = False) -> bool:
  x_type = type(x)
  # 标量类型的快速路径，可避免一次 np.ndarray 转换。
  if x_type in literalable_scalar_types:
    return True

  # 参见 https://docs.jax.dev/en/latest/internals/constants.html
  # for_ad: 我们希望在 AD 下保留
  if config.use_simplified_jaxpr_constants.value:
    from jax._src.array import ArrayImpl  # pyrefly: ignore[missing-import]
    do_lit_array = not for_ad
    if isinstance(x, ArrayImpl):
      return do_lit_array
  else:
    do_lit_array = False
  for t in x_type.__mro__:
    if t in literalable_types:
      return (do_lit_array or not np.ndim(x))
  return False

def is_hoistable(v: Literal) -> bool:
  return (np.ndim(v.val) > 0 and
          getattr(v.val, "nbytes", 4) > config.embedded_constants_max_bytes.value)

@partial(weakref_lru_cache, trace_context_in_key=False)
def jaxpr_const_args(jaxpr: Jaxpr) -> list[tuple[ArrayLike, AbstractValue]]:
  # 整个 Jaxpr 中 core.Literal 里的非标量常量，
  # 按 id 去重。它们将作为 const 参数被提升到
  # 它们所出现的函数中。
  # 参见 https://docs.jax.dev/en/latest/internals/constants.html
  if not config.use_simplified_jaxpr_constants.value:
    return []
  consts_by_id: dict[int, tuple[ArrayLike, AbstractValue]] = {}
  for v in jaxpr.outvars:
    if type(v) is Literal and is_hoistable(v):
      consts_by_id[id(v)] = (v.val, v.aval)

  for eqn in jaxpr.eqns:
    for v in eqn.invars:
      if type(v) is Literal and is_hoistable(v):
        consts_by_id[id(v)] = (v.val, v.aval)
    consts_by_id.update({id(v_aval[0]): v_aval
                         for v_aval in eqn_params_const_args(eqn.params)})
  return list(consts_by_id.values())

def eqn_params_const_args(params) -> list[tuple[ArrayLike, AbstractValue]]:
  consts_by_id: dict[int, tuple[ArrayLike, AbstractValue]] = {}
  for j in jaxprs_in_params(params):
    consts_by_id.update(
        {id(v_aval[0]): v_aval for v_aval in jaxpr_const_args(j)}
    )
  return list(consts_by_id.values())

Atom = Var | Literal

class Primitive:
  name: str
  # 为多输出原语设置。
  multiple_results: bool = False
  # 为以最终风格处理的 call 原语设置。
  call_primitive: bool = False
  # 为 ref 原语设置
  ref_primitive: bool = False
  # 为可以跳过值规范化(canonicalization)的原语设置
  skip_canonicalization: bool = False
  # 为分配引用的原语设置
  ref_allocating: bool = False
  is_effectful = None

  def __init__(self, name: str):
    self.name = name

  def __repr__(self):
    return f'{self.name}'

  def bind(self, *args, **params):
    canonical_args = []
    avals = []
    # 这一段是 规范化 + 获取类型值 + 调整mesh(如果不对，reshard)
    for i, arg in enumerate(args):
      try:
        c_arg = dtypes.canonicalize_value(arg)
        aval = typeof(c_arg)
      except TypeError as e:
        raise TypeError(
          f"Error interpreting argument to {self} as a JAX value."
          f" The problematic value is of type {type(arg)} and was passed to"
          f" {self} at position {i}.\n"
        ) from e
      if (not self.skip_canonicalization and isinstance(aval, ShapedArray)
          and not aval.sharding.mesh.empty):
        cur_mesh = mesh_lib.get_abstract_mesh()
        if cur_mesh != aval.sharding.mesh:
          # TODO(yashkatariya): 目前还不允许转换为 Explicit。也许我们
          # 需要 cast_and_slice_p，因为形状可能会改变？
          # 至少应有 1 个 mesh 轴是 Manual，且所有其它轴应是
          # Manual 或 Auto，以允许转换。
          if cur_mesh._any_axis_manual and cur_mesh._are_all_axes_auto_or_manual:
            if aval.sharding.mesh.are_all_axes_auto:
              from jax._src.pjit import reshard  # pyrefly: ignore[missing-import]
              c_arg = reshard(c_arg, NamedSharding(cur_mesh, P(*[None] * aval.ndim)))
              aval = typeof(c_arg)
            elif aval.sharding.mesh._any_axis_explicit:
              raise NotImplementedError(
                  "Closing over inputs to shard_map where the input is sharded "
                  "on `Explicit` axes is not implemented. As a workaround, "
                  "please pass those inputs as an argument to shard_map. Got "
                  f"input with shape {aval.str_short(True, True)}")
      if isinstance(c_arg, Tracer) and not c_arg._trace.is_valid():
        raise escaped_tracer_error(c_arg)
      canonical_args.append(c_arg)
      avals.append(aval)

    args = canonical_args

    # 这等价于 "with take_current_trace()"，但 bind() 代码
    # 被频繁调用，避免使用上下文管理器对象会略快一些。
    prev_trace = trace_ctx.trace
    trace_ctx.set_trace(None)
    try:
      return self.bind_with_trace(prev_trace, args, avals, params)
    finally:
      trace_ctx.set_trace(prev_trace)

  def bind_with_trace(self, trace, args, avals, params, /):
    if self.is_high(*avals, **params) and trace.requires_low:
      with set_current_trace(trace):
        return self.to_lojax(*args, **params)
    return trace.process_primitive(self, args, params)

  def def_impl(self, impl):
    self.impl = impl
    return impl

  def def_abstract_eval(self, abstract_eval):
    self.abstract_eval = _effect_free_abstract_eval(abstract_eval)
    return abstract_eval

  def def_effectful_abstract_eval(self, effectful_abstract_eval):
    self.abstract_eval = effectful_abstract_eval
    return effectful_abstract_eval

  def def_effectful_abstract_eval2(self, abstract_eval):
    self.abstract_eval = _generic_effectful_abstract_eval(abstract_eval, self)
    return abstract_eval

  def def_bind_with_trace(self, bind_with_trace):
    self.bind_with_trace = bind_with_trace
    return bind_with_trace

  def impl(self, *args, **params):
    raise NotImplementedError("Evaluation rule for '{}' not implemented"
                              .format(self.name))

  def abstract_eval(self, *args, **params):
    raise NotImplementedError("Abstract evaluation for '{}' not implemented"
                              .format(self.name))

  def get_bind_params(self, params):
    return params

  def to_lojax(self, *args, **params):
    raise NotImplementedError(
        f"Primitive '{self.name}' has is_high=True but no to_lojax "
        f"lowering rule.")

  def is_high(self, *avals, **params) -> bool:
    for v in params.values():
      if isinstance(v, Jaxpr) and v.is_high:
        return True
    return False


def _effect_free_abstract_eval(abstract_eval):
  def abstract_eval_(*args, **kwargs):
    return abstract_eval(*args, **kwargs), no_effects
  return abstract_eval_

@dataclass(frozen=True, slots=True)
class GenericEffect(Effect):
  prim: Primitive
effects.lowerable_effects.add_type(GenericEffect)
effects.control_flow_allowed_effects.add_type(GenericEffect)
effects.custom_derivatives_allowed_effects.add_type(GenericEffect)

def _generic_effectful_abstract_eval(abstract_eval, prim):
  def abstract_eval_(*args, **kwargs):
    return abstract_eval(*args, **kwargs), {GenericEffect(prim)}
  return abstract_eval_

# -------------------- 提升 --------------------

def eval_jaxpr(jaxpr: Jaxpr, consts, *args, propagate_source_info=True) -> list[Any]:
  def read(v: Atom) -> Any:
    return v.val if isinstance(v, Literal) else env[v]

  def write(v: Var, val: Any) -> None:
    if config.enable_checks.value:
      assert typecheck(v.aval, val), (v.aval, typeof(val), val)
    env[v] = val

  env: dict[Var, Any] = {}
  foreach(write, jaxpr.all_invars, [*consts, *args])
  lu = last_used(jaxpr)
  for eqn in jaxpr.eqns:
    bind_params = eqn.primitive.get_bind_params(eqn.params)
    name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
    traceback = eqn.source_info.traceback if propagate_source_info else None
    with (source_info_util.user_context(traceback, name_stack=name_stack),
          eqn.ctx.manager):
      ans = eqn.primitive.bind(*map(read, eqn.invars), **bind_params)
    if eqn.primitive.multiple_results:
      foreach(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
    clean_up_dead_vars(eqn, env, lu)
  return map(read, jaxpr.outvars)

def check_avals_context_mesh(avals, prim_name):
  cur_mesh = mesh_lib.get_abstract_mesh()
  for a in avals:
    if not isinstance(a.memory_space, MemorySpace):
      raise TypeError(
          f"Primitive {prim_name} got aval {a} with unknown memory_space type:"
          f" {type(a.memory_space)}")
    if cur_mesh.empty or a.sharding.mesh.empty:
      continue
    # aval 可以带有 axis_names 各不相同的 mesh，因此
    # 在全自动模式下允许这种情况。
    if a.sharding.mesh.are_all_axes_auto and cur_mesh.are_all_axes_auto:
      continue
    if a.sharding.mesh != cur_mesh:
      raise ValueError(
          f"For primitive {prim_name}, context mesh {cur_mesh} should match"
          f" the aval mesh {a.sharding.mesh} for shape {a.str_short()}. This"
          " error occurs at source: "
          f" {source_info_util.summarize(source_info_util.current())}")

# -------------------- 追踪 --------------------

class Trace:
  __slots__ = ("__weakref__", "_invalidated", "_weakref", "requires_low")

  def __init__(self):
    self._invalidated = False
    # 我们经常需要某个追踪（Trace）的弱引用，所以预先计算一个。
    self._weakref = weakref.ref(self)
    self.requires_low = True

  def stage_value(self, val):
    """把一个值提升到某个追踪中。

    语义上等价于对 identity 原语调用 process_primitive，
    但可能避免（例如）构造 jaxpr 方程。"""
    raise NotImplementedError("must override")

  def process_primitive(self, primitive, tracers, params, /):
    raise NotImplementedError("must override")

  def invalidate(self):
    self._invalidated = True

  def is_valid(self):
    return not self._invalidated

  def __repr__(self):
    return f'{self.__class__.__name__}'

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers, /, *,
                              symbolic_zeros):
    msg = (f"{type(self)} must override process_custom_jvp_call "
           "to handle custom_jvp primitives")
    raise NotImplementedError(msg)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, /, *,
                              out_trees, symbolic_zeros):
    msg = (f"{type(self)} must override process_custom_vjp_call "
           "to handle custom_vjp primitives")
    raise NotImplementedError(msg)

  # TODO(dougalm): 弃用/删除
  def full_raise(self, x):
    return x

  # TODO(dougalm): 弃用/删除
  @property
  def main(self):
    return getattr(self, "tag", None)

def escaped_tracer_error(tracer, detail=None):
  num_frames = _TRACER_ERROR_NUM_TRACEBACK_FRAMES.value
  msg = ('Encountered an unexpected tracer. A function transformed by JAX '
         'had a side effect, allowing for a reference to an intermediate value '
         f'with type {tracer.aval.str_short()} wrapped in a '
         f'{type(tracer).__name__} to escape the scope of the transformation.\n'
         'JAX transformations require that functions explicitly return their '
         'outputs, and disallow saving intermediate values to global state.')
  dbg = getattr(tracer, '_debug_info', None)
  if dbg is not None:
    msg += ('\nThe function being traced when the value leaked was '
            f'{dbg.func_src_info} traced for {dbg.traced_for}.')
  line_info = getattr(tracer, '_line_info', None)
  if line_info is not None:
    divider = '\n' + '-'*30 + '\n'
    msg += divider
    msg += ('The leaked intermediate value was created on line '
            f'{source_info_util.summarize(line_info)}. ')
    msg += divider
    if num_frames > 0:
      msg += (f'When the value was created, the final {num_frames} stack '
              'frames (most recent last) excluding JAX-internal frames were:')
      msg += divider + source_info_util.summarize(
          line_info, num_frames=num_frames) + divider
  msg += ('\nTo catch the leak earlier, try setting the environment variable '
          'JAX_CHECK_TRACER_LEAKS or using the `jax.checking_leaks` context '
          'manager.')
  if detail:
    msg += f'Detail: {detail}'
  return UnexpectedTracerError(msg)


def check_scalar_conversion(arr: Array):
  if arr.ndim > 0:
    raise TypeError("Only scalar arrays can be converted to Python scalars; "
                    f"got {arr.ndim=}")


def check_integer_conversion(arr: Array):
  if not (arr.shape == () and dtypes.issubdtype(arr.dtype, np.integer)):
    raise TypeError("Only integer scalar arrays can be converted to a scalar index.")


def check_bool_conversion(arr: Array):
  if arr.size == 0:
    raise ValueError("The truth value of an empty array is ambiguous. Use"
                     " `array.size > 0` to check that an array is not empty.")
  if arr.size > 1:
    raise ValueError("The truth value of an array with more than one element"
                     " is ambiguous. Use a.any() or a.all()")


pytype_aval_mappings: dict[type, Callable[[Any], AbstractValue]] = {}

def _str_abstractify(x):
  raise TypeError(f"Argument '{x}' of type {type(x)} is not a valid JAX type")
pytype_aval_mappings[str] = _str_abstractify


def _aval_property(name):
  return property(lambda self: getattr(self.aval, name))


if TYPE_CHECKING:
  # 我们希望 Python 类型检查器能够接受 `some_tracer: jax.Array`，尽管
  # 追踪器可以表示非数组。也就是说，理想情况下我们只应在 Tracer 实例
  # 拥有 ShapedArray 抽象值(aval)时才接受该标注，但我们无法
  # 在 Python 类型检查时做出这一判断。因此我们改为过度宽松，
  # 允许所有 Tracer 实例都能通过 jax.Array 标注的类型检查。
  TracerBase = Array
  TracerMeta = StrictABCMeta
else:
  class TracerBase:
    __slots__ = ()
  TracerMeta = type


class Tracer[TraceType: Trace](TracerBase, metaclass=TracerMeta):
  __array_priority__ = 1000
  __slots__ = ['__weakref__', '_trace', '_line_info', 'aval']
  __hash__ = None

  _trace: TraceType
  _line_info: source_info_util.SourceInfo | None
  aval: AbstractValue

  dtype = _aval_property('dtype')
  ndim = _aval_property('ndim')
  size = _aval_property('size')
  shape = _aval_property('shape')

  # dimension_as_value 会被频繁访问，我们显式地将其定义为
  # None，以避免走到 __getattr__ 路径，那条路径会构造一条错误
  # 消息（因此很慢）。
  dimension_as_value = None

  # 我们将 __jax_array__ 定义为属性，以便在 self.aval.__jax_array__
  # 存在时（例如对于 Flax NNX 变量这类 aval）委托给它。
  # 这避免了对没有该属性的追踪器走到较慢的 __getattr__ 路径。
  @property
  def __jax_array__(self):
    m = getattr(self.aval, '__jax_array__', None)
    if m is not None:
      return lambda: m.fun(self)
    return None

  def __init__(self, trace: TraceType, aval: AbstractValue):
    self._trace = trace
    self.aval = aval

  def _error_repr(self):
    if self.aval is None:
      return f"traced array with aval {self.aval}"
    return f"traced array with shape {self.aval.str_short()}"

  def __array__(self, *args, **kw):
    raise TracerArrayConversionError(self)

  # 用于 isinstance(tracer, jax.Array) 的辅助函数，放在这里以避免循环导入
  def _is_traced_array(self):
    return isinstance(self.aval, ShapedArray)

  def __dlpack__(self, *args, **kw):
    raise ConcretizationTypeError(self,
      f"The __dlpack__() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def tolist(self):
    raise ConcretizationTypeError(self,
      f"The tolist() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def tobytes(self, order="C"):
    del order
    raise ConcretizationTypeError(self,
      f"The tobytes() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  # TODO(dougalm): 弃用/删除
  def full_lower(self):
    return self

  def __iter__(self):
    if not hasattr(self.aval, "_iter"):
      raise TypeError(f"Value of type {type(self)} is not iterable.")
    return iter(self.aval._iter(self))

  def __reversed__(self):
    return iter(self[::-1])

  def __len__(self):
    if not hasattr(self.aval, "_len"):
      raise TypeError(f"Value of type {type(self)} has no length.")
    return self.aval._len(self)

  def to_concrete_value(self):
    # 如果存在具体值则应返回它，否则返回 None。
    return None

  @property
  def sharding(self):
    # 该属性是 jax.Array API 的一部分，但只在具体数组上定义。
    # 抛出 ConcretizationTypeError 是合理的，但为了向后兼容，
    # 我们抛出 AttributeError，以便 hasattr() 和 getattr() 按预期工作。
    raise AttributeError(
        f"The 'sharding' attribute is not available on {self._error_repr()}."
        f"{self._origin_msg()}")

  @property
  def committed(self):
    raise ConcretizationTypeError(
        self,
        f"The 'committed' attribute is not available on {self._error_repr()}."
        f"{self._origin_msg()}")

  @property
  def device(self):
    # 该属性是 jax.Array API 的一部分，但只在具体数组上定义。
    # 抛出 ConcretizationTypeError 是合理的，但为了向后兼容，
    # 我们抛出 AttributeError，以便 hasattr() 和 getattr() 按预期工作。
    raise AttributeError(
      f"The 'device' attribute is not available on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def addressable_shards(self):
    raise ConcretizationTypeError(self,
      f"The 'addressable_shards' attribute is not available on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def at(self):
    if not hasattr(self.aval, "at"):
      raise TypeError(f"Value of type {type(self)} does not support at().")
    return self.aval.at.fget(self)

  def get_referent(self) -> Any:
    return self  # 重写用于对象等价性检查

  def __bool__(self):
    if is_concrete(self): return bool(self.to_concrete_value())
    check_bool_conversion(self)
    if not hasattr(self.aval, "_bool"):
      raise TypeError(f"Value of type {type(self)} is not convertible to boolean.")
    return self.aval._bool(self)

  def __int__(self):
    if is_concrete(self): return int(self.to_concrete_value())
    check_scalar_conversion(self)
    if not hasattr(self.aval, "_int"):
      raise TypeError(f"Value of type {type(self)} is not convertible to integer.")
    return self.aval._int(self)

  def __float__(self):
    check_scalar_conversion(self)
    if not hasattr(self.aval, "_float"):
      raise TypeError(f"Value of type {type(self)} is not convertible to float.")
    return self.aval._float(self)

  def __complex__(self):
    check_scalar_conversion(self)
    if not hasattr(self.aval, "_complex"):
      raise TypeError(f"Value of type {type(self)} is not convertible to complex.")
    return self.aval._complex(self)

  def __hex__(self):
    if is_concrete(self): return hex(self.to_concrete_value())
    check_integer_conversion(self)
    if not hasattr(self.aval, "_hex"):
      raise TypeError(f"Value of type {type(self)} is not convertible to hex.")
    return self.aval._hex(self)

  def __oct__(self):
    if is_concrete(self): return oct(self.to_concrete_value())
    check_integer_conversion(self)
    if not hasattr(self.aval, "_oct"):
      raise TypeError(f"Value of type {type(self)} is not convertible to oct.")
    return self.aval._oct(self)

  def __index__(self):
    if is_concrete(self): return operator.index(self.to_concrete_value())
    check_integer_conversion(self)
    if not hasattr(self.aval, "_index"):
      raise TypeError(f"Value of type {type(self)} is not convertible to integer index.")
    return self.aval._index(self)

  # 在尝试 pickle 一个 Tracer 时抛出一条有用的错误。
  def __reduce__(self):
    raise ConcretizationTypeError(
      self, ("The error occurred in the __reduce__ method, which may "
             "indicate an attempt to serialize/pickle a traced value."))

  # 抛出 ShapedArray 提供的更好错误消息
  def __setitem__(self, key, value):
    if not hasattr(self.aval, "_setitem"):
      raise TypeError(f"Value of type {type(self)} is not indexable.")
    return self.aval._setitem(self, key, value)

  # NumPy 也只在类上查找特殊方法。
  def __array_module__(self, types):
    if not hasattr(self.aval, "_array_module"):
      raise TypeError(f"Value of type {type(self)} is not compatible with the Array API.")
    return self.aval._array_module(self, types)

  def __getattr__(self, name):
    # 如果 aval 属性抛出 AttributeError，会在这里被捕获
    assert not config.enable_checks.value or name != "aval"

    # 为了向后兼容，这些在基类中必须抛出 AttributeError。
    # TODO(jakevdp): 我们能改掉这一点并让它们改为抛出 NotImplementedError 吗？
    if name in ["block_until_ready", "copy_to_host_async"]:
      raise AttributeError(
        f"The '{name}' method is not available on {self._error_repr()}."
        f"{self._origin_msg()}")

    if name == 'sharding':
      raise AttributeError(
        f"The 'sharding' attribute is not available on {self._error_repr()}. "
        "To query sharding information on tracers, use `jax.typeof(x)`.")

    try:
      attr = getattr(self.aval, name)
    except AttributeError as err:
      raise AttributeError(
          f"{self.__class__.__name__} has no attribute {name}"
      ) from err
    else:
      t = type(attr)
      if t is aval_property:
        return attr.fget(self)
      elif t is aval_method:
        return types.MethodType(attr.fun, self)
      else:
        return attr

  def _short_repr(self) -> str:
    return f'{self.__class__.__name__}<{self.aval}>'

  def _pretty_print(self, verbose: bool = False) -> pp.Doc:
    if not verbose:
      return pp.text(self._short_repr())

    base = pp.text(f'Traced<{self.aval}>with<{self._trace}>')
    contents = [(name, attr._pretty_print() if isinstance(attr, Tracer)
                 else pp.text(repr(attr))) for name, attr in self._contents()]
    if contents:
      base = pp.group(pp.nest(2, pp.concat([
        base, pp.text(' with'), pp.brk(), pp.join(pp.brk(), [
          pp.text(f'{name} = ') + pp_payload
          for name, pp_payload in contents])
      ])))
    return base

  def __repr__(self):
    return self._pretty_print(verbose=False).format()

  def _contents(self):
    try:
      return [(name, getattr(self, name)) for name in self.__slots__]
    except AttributeError:
      return ()

  def _origin_msg(self) -> str:
    return ""

  # 仅对已实体化(materialized)数组有效的方法
  def addressable_data(self, index):
    raise ConcretizationTypeError(self,
      f"The addressable_data() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def delete(self):
    raise ConcretizationTypeError(self,
      f"The delete() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def devices(self):
    raise ConcretizationTypeError(self,
      f"The devices() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def global_shards(self):
    raise ConcretizationTypeError(self,
      f"The global_shards property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def is_deleted(self):
    raise ConcretizationTypeError(self,
      f"The is_deleted() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def is_fully_addressable(self):
    raise ConcretizationTypeError(self,
      f"The is_fully_addressable property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def is_fully_replicated(self):
    raise ConcretizationTypeError(self,
      f"The is_fully_replicated property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def on_device_size_in_bytes(self):
    raise ConcretizationTypeError(self,
      f"The on_device_size_in_bytes() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def traceback(self):
    raise ConcretizationTypeError(self,
      f"The traceback property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def unsafe_buffer_pointer(self):
    raise ConcretizationTypeError(self,
      f"The unsafe_buffer_pointer() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

_jax.set_tracer_class(Tracer)

# 这些可用于设置从 Tracer 实例到其底层 aval 的
# 属性和实例方法转发
aval_property = namedtuple("aval_property", ["fget"])
aval_method = namedtuple("aval_method", ["fun"])

pytype_aval_mappings[Tracer] = lambda x: x.aval
dtypes.register_canonicalize_value_handler(Tracer, None)

def check_eval_args(args):
  for arg in args:
    if isinstance(arg, Tracer):
      raise escaped_tracer_error(arg)

stage_p = Primitive('stage')

class EvalTrace(Trace):

  def stage_value(self, val):
    if isinstance(val, Array):
      return val
    return self.process_primitive(stage_p, [val], {})

  def process_primitive(self, primitive, args, params, /):
    with set_current_trace(self):
      if config.debug_key_reuse.value:
        from jax.experimental.key_reuse._core import call_impl_with_key_reuse_checks  # pyrefly: ignore[missing-import]
        return call_impl_with_key_reuse_checks(primitive, primitive.impl, *args, **params)
      else:
        # TODO(dougalm): 删除。这应该是不必要的
        args = map(full_lower, args)
        check_eval_args(args)
        return primitive.impl(*args, **params)

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers, /, **_):
    del primitive, jvp, _  # 未使用。
    with set_current_trace(self):
      return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, /, **_):
    del primitive, fwd, bwd, _  # 未使用。
    with set_current_trace(self):
      return fun.call_wrapped(*tracers)

class TraceTag:
  # TODO: 这之所以能工作，原因微妙得令人意外。像
  # `jvp_subtrace` 这样的函数变换由一个 tag 参数化，该 tag 标识出
  # 我们希望在变换过程中解开(unpack)的那组既存追踪器。在外层作用域中
  # 定义的函数不可能有任何闭包捕获的追踪，因此该 tag 无关紧要。在当前
  # 作用域中定义的函数可能有闭包捕获的追踪，但该 tag 永远不会改变，
  # 所以我们永远不会遇到虚假的缓存命中。计划是完全去掉 `lu.cache`，
  # 并采用一种只缓存顶层函数的更简单的缓存方案。那时我们就可以
  # 移除这个 hack。
  def __hash__(self):
    return hash(TraceTag)
  def __eq__(self, other):
    return isinstance(other, TraceTag)

ParamDict = dict[str, Any]
AxisName = Hashable

no_axis_name = object()

@immutable
class AxisEnv:
  __slots__ = ('axis_sizes', 'spmd_axis_names', 'explicit_mesh_axis_names',
               '__weakref__')

  axis_sizes : FrozenDict[AxisName, int]
  spmd_axis_names : frozenset[AxisName]
  explicit_mesh_axis_names: frozenset[AxisName]

  @staticmethod
  @weak_value_interner
  def _create(axis_sizes, spmd_axis_names, explicit_mesh_axis_names):
    obj = object.__new__(AxisEnv)
    object.__setattr__(obj, 'axis_sizes', axis_sizes)
    object.__setattr__(obj, 'spmd_axis_names', spmd_axis_names)
    object.__setattr__(obj, 'explicit_mesh_axis_names', explicit_mesh_axis_names)
    return obj

  def __new__(cls, axis_sizes=FrozenDict({}), spmd_axis_names=frozenset(),
              explicit_mesh_axis_names=frozenset()):
    return cls._create(axis_sizes, spmd_axis_names, explicit_mesh_axis_names)

  def axis_size(self, axis_name):
    if axis_name not in self.axis_sizes:
      raise NameError(f"unbound axis name: {axis_name}")
    else:
      return self.axis_sizes[axis_name]

  def axis_exists(self, axis_name):
    return axis_name in self.axis_sizes

  def axis_names(self):
    return tuple(k for k in self.axis_sizes)

  def pop_pure(self, axis_name):
    new_sizes = dict(self.axis_sizes)
    new_sizes.pop(axis_name)
    return AxisEnv(FrozenDict(new_sizes), self.spmd_axis_names,
                   self.explicit_mesh_axis_names)

  def extend_pure(self, name_size_pairs):
    new_sizes = dict(self.axis_sizes)
    new_sizes.update((name, size) for name, size in name_size_pairs
                    if name is not no_axis_name)
    return AxisEnv(FrozenDict(new_sizes), self.spmd_axis_names,
                   self.explicit_mesh_axis_names)

  def add_spmd_axis_names(self, axis_names):
    new_spmd_axis_names = self.spmd_axis_names | frozenset(axis_names)
    return AxisEnv(self.axis_sizes, new_spmd_axis_names,
                   self.explicit_mesh_axis_names)

  def add_explicit_mesh_axis_names(self, axis_names):
    new_ema = self.explicit_mesh_axis_names | frozenset(axis_names)
    return AxisEnv(self.axis_sizes, self.spmd_axis_names, new_ema)

  def remove_explicit_mesh_axis_names(self, axis_names):
    new_ema = self.explicit_mesh_axis_names - frozenset(axis_names)
    return AxisEnv(self.axis_sizes, self.spmd_axis_names, new_ema)

eval_trace = EvalTrace()
top_axis_env = AxisEnv(FrozenDict({}), frozenset(), frozenset())

# 对追踪状态的弱引用。例如它会被包含在 jit key 中。
trace_state = config_ext.Config(
    'trace_state', eval_trace._weakref, include_in_jit_key=True)

# 对追踪状态的强引用。它不应被包含在任何
# jit 或缓存键中，但我们需要一个线程局部强引用来确保它
# 保持存活。
trace_state_strong_ref = config_ext.Config(
  'trace_state_strong_ref', eval_trace, include_in_jit_key=False,
  include_in_trace_context=False)

axis_env_state = config_ext.Config(
    'axis_env_state',
    top_axis_env,
    include_in_jit_key=True,
    include_in_trace_context=True,
)


class TracingContext:
  __slots__ = ()

  @staticmethod
  def reset():
    trace_state.set_local(config_ext.unset)
    trace_state_strong_ref.set_local(config_ext.unset)
    axis_env_state.set_local(config_ext.unset)

  @property
  def trace(self):
    return trace_state_strong_ref.value

  @property
  def axis_env(self):
    return axis_env_state.value

  def is_top_level(self) -> bool:
    return self.trace is eval_trace and self.axis_env is top_axis_env

  def is_empty(self) -> bool:
    return self.trace is None

  def set_trace(self, trace):
    trace_state_strong_ref.set_local(trace)
    ts = trace._weakref if trace is not None else None
    trace_state.set_local(ts)

  def set_axis_env(self, axis_env):
    axis_env_state.set_local(axis_env)

trace_ctx = TracingContext()


class TakeCurrentTraceContextManager:
  __slots__ = ['prev']

  def __enter__(self):
    self.prev = trace_ctx.trace
    trace_ctx.set_trace(None)
    return self.prev

  def __exit__(self, exc_type, exc_value, traceback):
    trace_ctx.set_trace(self.prev)

take_current_trace = TakeCurrentTraceContextManager


class SetCurrentTraceContextManager:
  __slots__ = ['trace', 'check_leaks', 'prev']

  def __init__(self, trace, check_leaks=False):
    self.trace = trace
    self.check_leaks = check_leaks

  def __enter__(self):
    self.prev = trace_ctx.trace
    trace_ctx.set_trace(self.trace)

  def __exit__(self, exc_type, exc_value, traceback):
    trace_ctx.set_trace(self.prev)
    if self.check_leaks and config.check_tracer_leaks.value:
      self.trace.invalidate()
      trace_ref = self.trace._weakref
      del self.trace
      live_trace = trace_ref()
      if live_trace is not None:
        leaked_tracers = maybe_find_leaked_tracers(live_trace)
        if leaked_tracers:
          raise leaked_tracer_error("trace", live_trace, leaked_tracers)

set_current_trace = SetCurrentTraceContextManager

class ExtendAxisEnvNdContextManager:
  __slots__ = ['prev', 'name_size_pairs']

  def __init__(self, name_size_pairs: Iterable[tuple[AxisName, int]]):
    self.name_size_pairs = name_size_pairs

  def __enter__(self):
    self.prev = trace_ctx.axis_env
    trace_ctx.set_axis_env(self.prev.extend_pure(self.name_size_pairs))

  def __exit__(self, exc_type, exc_value, traceback):
    trace_ctx.set_axis_env(self.prev)

extend_axis_env_nd = ExtendAxisEnvNdContextManager


class AddSpmdAxisNamesContextManager:
  __slots__ = ['prev', 'axis_names']

  def __init__(self, axis_names: AxisName | None):
    self.axis_names = axis_names

  def __enter__(self):
    self.prev = trace_ctx.axis_env
    if self.axis_names is not None:
      trace_ctx.set_axis_env(self.prev.add_spmd_axis_names(self.axis_names))

  def __exit__(self, exc_type, exc_value, traceback):
    trace_ctx.set_axis_env(self.prev)

add_spmd_axis_names = AddSpmdAxisNamesContextManager

# TODO(yashkatariya): 等 vmap 能正确处理 mesh 上下文后删除这里。
class AddExplicitMeshAxisNamesContextManager:
  __slots__ = ['prev', 'axis_names']

  def __init__(self, axis_names: AxisName | None):
    self.axis_names = axis_names

  def __enter__(self):
    self.prev = trace_ctx.axis_env
    if self.axis_names is not None:
      trace_ctx.set_axis_env(self.prev.add_explicit_mesh_axis_names(
          self.axis_names))

  def __exit__(self, exc_type, exc_value, traceback):
    trace_ctx.set_axis_env(self.prev)

add_explicit_mesh_axis_names = AddExplicitMeshAxisNamesContextManager

# TODO(yashkatariya): 等 vmap 能正确处理 mesh 上下文后删除这里。
class RemoveExplicitMeshAxisNamesContextManager:
  __slots__ = ['prev', 'axis_names']

  def __init__(self, axis_names: AxisName | None):
    self.axis_names = axis_names

  def __enter__(self):
    self.prev = trace_ctx.axis_env
    if self.axis_names is not None:
      trace_ctx.set_axis_env(self.prev.remove_explicit_mesh_axis_names(
          self.axis_names))

  def __exit__(self, exc_type, exc_value, traceback):
    trace_ctx.set_axis_env(self.prev)

remove_explicit_mesh_axis_names = RemoveExplicitMeshAxisNamesContextManager


def get_axis_env():
  return trace_ctx.axis_env


def trace_state_clean() -> bool:
  return trace_ctx.is_empty() or trace_ctx.is_top_level()

def reset_trace_state() -> bool:
  """重置全局追踪状态，如果它原本就是干净的则返回 True。"""
  if not trace_ctx.is_top_level():
    trace_ctx.reset()
    return False
  else:
    return True

TRACER_LEAK_DEBUGGER_WARNING = """\
JAX check_tracer_leaks behavior can trigger false positives when used with a debugger.
To avoid false positives and silence this warning, you can disable thread tracing using
the following:

  import threading
  threading.current_thread().pydev_do_not_trace = True
"""

@contextmanager
def ensure_no_leaks(trace:Trace):
  yield
  trace.invalidate()
  if config.check_tracer_leaks.value:
    trace_ref = trace._weakref
    del trace
    live_trace = trace_ref()
    if live_trace is not None:
      leaked_tracers = maybe_find_leaked_tracers(live_trace)
      if leaked_tracers:
        raise leaked_tracer_error("trace", live_trace, leaked_tracers)


def maybe_find_leaked_tracers(trace: Trace) -> list[Tracer]:
  """查找持有对 Trace 引用的泄漏追踪器
  """
  if not getattr(threading.current_thread(), 'pydev_do_not_trace', True):
    warnings.warn(TRACER_LEAK_DEBUGGER_WARNING)
  # 触发垃圾回收，以过滤掉仅因循环依赖而存活的不可达对象。
  # （我们不关心不可达的泄漏追踪器，因为它们无法与用户代码
  # 交互并造成问题。）
  gc.collect()
  tracers = list(filter(lambda x: isinstance(x, Tracer), gc.get_referrers(trace)))
  return tracers

def leaked_tracer_error(name: str, t, tracers: list[Tracer]) -> Exception:
  assert tracers
  why = partial(_why_alive, {id(tracers)})
  msgs = []
  for tracer in tracers:  # 不用生成器表达式：它会被 gc 看到，并把自身也报告为引用者
    chain = why(tracer)
    label = f'<{type(tracer).__name__} {id(tracer)}>'
    chain += ''.join(f'\n{label} is referred to by {h}' for h in
                     _held_in_frame_locals(tracer, {id(tracers)}))
    if not chain:
      chain = (f'\n{label} has no referrers visible to the gc module; it may '
               'be held by an object that does not cooperate with the garbage '
               'collector, such as one implemented in a C extension')
    msgs.append(f'{tracer}{tracer._origin_msg()}{chain}')
  return Exception(f'Leaked {name} {t}. Leaked tracer(s):\n\n'
                   + '\n\n'.join(msgs) + '\n')

def _held_in_frame_locals(x, ignore_ids: set[int]) -> list[str]:
  """查找活跃的栈帧，其局部变量引用（或包含）x。

  在 CPython 3.11+ 上，正在执行的函数的帧通常不是 gc 跟踪的对象，
  因此其局部变量持有的引用对 gc.get_referrers 不可见，
  对 _why_alive 也不可见。改为直接遍历当前栈。
  """
  skip_codes = (leaked_tracer_error.__code__, _held_in_frame_locals.__code__)
  holders = []
  frame = sys._getframe(1)
  while frame is not None:
    if frame.f_code not in skip_codes:
      for name, val in frame.f_locals.items():
        if id(val) in ignore_ids:
          continue
        if val is x:
          via = ''
        elif _contains_ref(val, x):
          via = f', a {type(val).__name__} containing it,'
        else:
          continue
        code = frame.f_code
        holders.append(f"the local variable {name!r}{via} of the frame "
                       f"{code.co_qualname} ({code.co_filename}:{frame.f_lineno})")
    frame = frame.f_back
  return holders

def _contains_ref(val, x, depth: int = 0) -> bool:
  if depth >= 3:
    return False
  if isinstance(val, (list, tuple, set, frozenset)):
    return any(v is x or _contains_ref(v, x, depth + 1) for v in val)
  if isinstance(val, dict):
    return any(v is x or _contains_ref(v, x, depth + 1) for v in val.values())
  return False

def _why_alive(ignore_ids: set[int], x: Any) -> str:
  parents = lambda x: [r for r in gc.get_referrers(x) if id(r) not in ignore_ids]
  child, lines, seen = x, [], set()
  while (id(child) not in seen and type(child) is not types.ModuleType
         and parents(child)):
    parent = parents(child)[0]  # 只挑一个父对象

    # 对于命名空间（如模块和类实例）以及闭包，这些引用
    # 可能形成一条简单链：例如实例引用它自己的 __dict__，
    # 而 __dict__ 又引用 child；或者函数引用它的 __closure__，
    # 后者引用各个 cell，cell 再引用 child。在这些情况下，
    # 我们可以把这条链折叠成一次父->子跳转，从而给出
    # 更直观的描述。做法是：在这里把 `parent` 设为 `child` 的
    # 祖父（或曾祖父），然后在 _why_alive_container_info 中
    # 处理该情形。参见示例：
    #  https://github.com/jax-ml/jax/pull/13022#discussion_r1008456599
    # 要阻止这种折叠行为，只需注释掉这个代码块。
    try:
      if (isinstance(parent, dict) and
          getattr(parents(parent)[0], '__dict__', None) is parents(child)[0]):
        parent = parents(parent)[0]
      elif type(parent) is types.CellType:
        parent = parents(parents(parent)[0])[0]
    except IndexError:
      pass  # 引用者列表可能为空，例如某个容器只被
            # 活跃栈帧的局部变量持有，因为 gc.get_referrers 看不到活跃栈帧

    line = f'<{type(child).__name__} {id(child)}> is referred to by '
    lines.append(line + _why_alive_container_info(parent, id(child)))
    seen.add(id(child))
    child = parent
  return '\n' + '\n'.join(lines) if lines else ''

def _why_alive_container_info(container, obj_id) -> str:
  name = f'<{type(container).__name__} {id(container)}>'
  if type(container) is types.ModuleType:
    name = getattr(container, '__name__', name)
  if type(container) is types.FunctionType:
    name_ = getattr(container, '__name__', '<no-name>')
    closure = inspect.getclosurevars(container)
    keys = [k for k, v in dict(closure.nonlocals, **closure.globals).items()
            if id(v) == obj_id]
    if len(keys) == 1: return f'{name} ({name_}) closed-over variable {keys[0]}'
    elif len(keys) > 1: return (f'{name} in closed-over variables ' +
                                ', '.join(map(repr, keys)))
  if hasattr(container, '__dict__'):
    keys = [k for k in vars(container) if id(vars(container)[k]) == obj_id]
    if len(keys) == 1: return f'{name}.{keys[0]}'
    elif len(keys) > 1: return f'{name} in vars ' + ', '.join(map(repr, keys))
  if isinstance(container, (list, tuple)):
    idxs = [i for i, x in enumerate(container) if id(x) == obj_id]
    if len(idxs) == 1: return f'{name}[{idxs[0]}]'
    else: return f'{name} at indices ' + ', '.join(map(str, idxs))
  if isinstance(container, dict):
    keys = [k for k in container if id(container[k]) == obj_id]
    if len(keys) == 1: return f'{name}[{keys[0]!r}]'
    else: return f'{name} at keys ' + ', '.join(map(repr, keys))
  if isinstance(container, types.ModuleType):
    return f' named {container.__name__}'
  return name

@contextmanager
def ensure_compile_time_eval():
  """上下文管理器，确保在追踪/编译期完成求值（否则报错）。

  一些 JAX API（如 :func:`jax.jit` 和 :func:`jax.lax.scan`）
  涉及暂存，即延迟数值表达式的求值（如 :mod:`jax.numpy`
  函数调用），这样，这些计算就不是在求值相应的
  Python 表达式时即时执行，而是被单独进行，
  例如在优化编译之后。但这种延迟可能并不理想。
  例如，求值 Python 控制流时可能需要数值，
  因此不能延迟对它们的求值。
  再举一个例子，出于性能原因，确保编译期求值
  （或“常量折叠”）可能是有益的。

  该上下文管理器确保 JAX 计算被即时执行(eager)。如果
  无法即时求值，则会抛出 ``ConcretizationTypeError``。

  下面是一个人为构造的示例::

    import jax
    import jax.numpy as jnp

    @jax.jit
    def f(x):
      with jax.ensure_compile_time_eval():
        y = jnp.sin(3.0)
        z = jnp.sin(y)
        z_positive = z > 0
      if z_positive:  # z_positive 可用于 Python 控制流
        return jnp.sin(x)
      else:
        return jnp.cos(x)

  下面是一个来自 https://github.com/jax-ml/jax/issues/3974 的真实示例::

    import jax
    import jax.numpy as jnp
    from jax import random

    @jax.jit
    def jax_fn(x):
      with jax.ensure_compile_time_eval():
        y = random.randint(random.key(0), (1000,1000), 0, 100)
      y2 = y @ y
      x2 = jnp.sum(y2) * x
      return x2

  类似的行为通常可以简单地通过把常量表达式“提升”到
  相应的暂存 API 之外来实现::

    y = random.randint(random.key(0), (1000,1000), 0, 100)

    @jax.jit
    def jax_fn(x):
      y2 = y @ y
      x2 = jnp.sum(y2)*x
      return x2

  但在某些情况下，使用这个上下文管理器可能更方便。
  """
  with config.eager_constant_folding(True):
    yield

@contextmanager
def eval_context():
  with set_current_trace(eval_trace):
    yield

# TODO(dougalm): 弃用/删除
def full_lower(val):
  if isinstance(val, Tracer):
    return val.full_lower()
  else:
    return val

def get_referent(x: Any) -> Any:
  return x.get_referent() if isinstance(x, Tracer) else x

def same_referent(x: Any, y: Any) -> bool:
  return get_referent(x) is get_referent(y)

def dedup_referents(itr: Iterable[Any]) -> list[Any]:
  return list({HashableWrapper(get_referent(x)):x for x in itr}.values())

def definitely_equal(x, y):
  if isinstance(x, Tracer) or isinstance(y, Tracer):
    return same_referent(x, y)
  elif x is y:
    return True
  try:
    return x == y
  except InconclusiveDimensionOperation:
    return False

# -------------------- 抽象值 --------------------

class AbstractValue:
  __slots__: list[str] = []

  @property
  def is_high(self) -> bool:
    return False

  def to_tangent_aval(self) -> AbstractValue:
    raise NotImplementedError("must override")

  def to_ct_aval(self) -> AbstractValue:
    raise NotImplementedError("must override")

  # TODO(dougalm): 弃用这个别名
  def at_least_vspace(self):
    return self.to_tangent_aval()

  def __repr__(self):
    try:
      kv_pairs = (f'{k}={v}' for k, v in self.__dict__.items())
      return '{}({})'.format(self.__class__.__name__, ','.join(kv_pairs))
    except AttributeError:
      return self.__class__.__name__

  def update_weak_type(self, weak_type):
    return self

  def update_manual_axis_type(self, mat):
    return self

  def strip_weak_type(self) -> AbstractValue:
    return self.update_weak_type(False)

  def normalize(self) -> AbstractValue:
    return self.strip_weak_type()

  def update(self, **kwargs):
    raise NotImplementedError("must override")

  def lo_ty(self):
    return [self]

  def lower_val(self, val, /):
    return [val]

  def raise_val(self, *vals):
    val, = vals
    return val

  def str_short(self, short_dtypes=False, mesh_axis_types=False):
    return str(self)

  def dec_rank(self, size, spec):
    return mapped_aval(size, spec, self)

  def inc_rank(self, size, spec):
    return unmapped_aval(size, spec, self)

  def leading_axis_spec(self):
    return 0

  def shard(self, mesh, manual_axes, check_vma, spec):
    return shard_aval(mesh, manual_axes, check_vma, spec, self)

  def unshard(self, mesh, check_vma, spec):
    return unshard_aval(mesh, check_vma, spec, self)

  def vspace_add(self, x, y):
    from jax._src.ad_util import add_jaxvals  # pyrefly: ignore[missing-import]
    return add_jaxvals(x, y)

  def raise_val2(self, lo_vals_ft):
    return self.raise_val(*lo_vals_ft.unflatten())  # pyrefly: ignore[missing-attribute]

  def lower_val2(self, hi_val):
    return ft.flatten(self.lower_val(hi_val))  # pyrefly: ignore[missing-attribute]

InputType = tuple[AbstractValue, ...]
OutputType = tuple[AbstractValue, ...]

# 用于类型标注，表示一个 Tracer 或一个 `valid_jaxtype`。
Value = Any

def valid_jaxtype(x) -> bool:
  try:
    aval = typeof(x)
  except TypeError:
    return False
  else:
    if hasattr(aval, "dtype") and aval.dtype == dtypes.string_dtype:
      return False
    else:
      return True


def mem_kind_to_space(mem_kind: str | None) -> MemorySpace:
  if mem_kind == 'pinned_host':
    return MemorySpace.Host
  return MemorySpace.Device


def mem_space_to_kind(mem_space: Any) -> str:
  """将内存空间转换为其对应的 XLA memory kind 字符串。

  支持标准的 MemorySpace 枚举值，以及定义了 `memory_kind` 属性的自定义
  内存空间。
  """
  if isinstance(mem_space, MemorySpace):
    if mem_space == MemorySpace.Device:
      return "device"
    elif mem_space == MemorySpace.Host:
      return "pinned_host"
  elif hasattr(mem_space, "memory_kind"):
    return mem_space.memory_kind
  assert False, f"unreachable: {mem_space}"


@cache(max_size=4096,
       trace_context_in_key=lambda: config.remove_size_one_mesh_axis_from_type.value)
def update_aval_with_sharding(aval, sharding, mat=None):
  if isinstance(sharding, NamedSharding):
    s = NamedSharding(sharding.mesh.abstract_mesh,
                      sharding.spec._normalized_spec_for_aval(aval.ndim))
    return aval.update(
        sharding=s, manual_axis_type=(aval.mat if mat is None else mat),
        memory_space=mem_kind_to_space(sharding.memory_kind))
  return aval if mat is None else aval.update(manual_axis_type=mat)


# 这里有两套抽象化 API，它们过去各自有独立的实现。现在它们实际上
# 是等价的，区别如下：
#
# - typeof 为合法的类数组对象（包括追踪器）返回 aval。
# - shaped_abstractify 类似于 typeof，但也接受鸭子类型的数组。
#

def shaped_abstractify(x):
  typ = type(x)
  if (aval_fn := pytype_aval_mappings.get(typ)):  # 快速路径
    return aval_fn(x)
  for t in typ.__mro__[1:]:
    if (aval_fn := pytype_aval_mappings.get(t)):
      return aval_fn(x)
  if isinstance(x, AbstractValue):
    return x
  if getattr(x, '__jax_array__', None) is not None:
    raise ValueError(
        'Triggering __jax_array__() during abstractification is no longer'
        ' supported. To avoid this error, either explicitly convert your object'
        ' using jax.numpy.array(), or register your object as a pytree.'
    )
  if hasattr(x, 'dtype'):
    aval = ShapedArray(
        np.shape(x),
        dtypes.canonicalize_dtype(x.dtype, allow_extended_dtype=True),
        weak_type=getattr(x, "weak_type", False),
    )
    return update_aval_with_sharding(aval, getattr(x, 'sharding', None))
  raise TypeError(
      f"Cannot interpret value of type {typ} as an abstract array; it "
      "does not have a dtype attribute")


# TODO(phawkins): 返回类型应该是 AbstractValue。
def typeof(x: Any) -> Any:
  """返回输入的 JAX 类型（即 :class:`AbstractValue`）。

  如果 ``x`` 不是合法的 JAX 类型，则抛出 ``TypeError``。
  """
  typ = type(x)
  if (aval_fn := pytype_aval_mappings.get(typ)):  # 快速路径
    return aval_fn(x)
  for t in typ.__mro__[1:]:
    if (aval_fn := pytype_aval_mappings.get(t)):
      return aval_fn(x)
  if getattr(x, '__jax_array__', None) is not None:
    raise ValueError(
        'Triggering __jax_array__() during abstractification is no longer'
        ' supported. To avoid this error, either explicitly convert your object'
        ' using jax.numpy.array(), or register your object as a pytree.'
    )
  raise TypeError(f"Argument '{x}' of type '{typ}' is not a valid JAX type")

def is_concrete(x):
  return to_concrete_value(x) is not None

def to_concrete_value(x):
  if isinstance(x, Tracer):
    return x.to_concrete_value()
  else:
    return x

def concretization_function_error(fun, suggest_astype=False):
  fname = getattr(fun, "__name__", fun)
  fname_context = f"The problem arose with the `{fname}` function. "
  if suggest_astype:
    fname_context += ("If trying to convert the data type of a value, "
                      f"try using `x.astype({fun.__name__})` "
                      f"or `jnp.array(x, {fun.__name__})` instead.")
  if fun is bool:
    def error(self, arg):
      raise TracerBoolConversionError(arg)
  elif fun in (hex, oct, operator.index):
    def error(self, arg):
      raise TracerIntegerConversionError(arg)
  else:
    def error(self, arg):
      raise ConcretizationTypeError(arg, fname_context)
  return error

def concrete_or_error(force: Any, val: Any, context=""):
  """类似于 force(val)，但会在错误信息中给出上下文。"""
  if force is None:
    force = lambda x: x
  if isinstance(val, Tracer):
    maybe_concrete = val.to_concrete_value()
    if maybe_concrete is None:
      raise ConcretizationTypeError(val, context)
    else:
      return force(maybe_concrete)
  else:
    return force(val)

def concrete_dim_or_error(val: Any, context=""):
  """类似于 concrete_or_error(operator.index)，但允许符号维度。"""
  if is_symbolic_dim(val):
    return val
  else:
    return concrete_or_error(operator.index, val, context=context)

### 扩展 dtype
#
# 扩展 dtype 是 JAX 特有的 dtype，用于表示逻辑数组，这些数组的元素类型
# 在编译器中并没有显而易见的直接对应的基本类型（“物理”）数组。特别地，其元素类型
# 与 XLA 和 NumPy 的元素类型（例如 int32）不同。这些 dtype 只有 JAX 知道。
# 它们的实现由以下两部分决定：
# a) 一个表示该扩展 dtype 的对象，可通过相应 JAX 数组上的 `dtype` 属性访问，
#    在内部也可通过对应于这类 JAX 数组的 aval（例如 ShapedArray）访问；
# b) 一组规则，可通过 (a) 中扩展 dtype 对象的私有属性获得。
# (b) 中的规则告诉 JAX 内部代码如何将元素类型落地（ground out），以便与编译器
# 和运行时交互，例如在降级到编译器语言时。

@overload
def physical_aval(aval: ShapedArray) -> ShapedArray:
  ...
@overload                       # TODO(frostig): 删除这种情况
def physical_aval(aval: AbstractValue) -> AbstractValue:
  ...

def physical_aval(aval):
  if (isinstance(aval, ShapedArray) and
      isinstance(aval.dtype, dtypes.ExtendedDType)):
    elt_aval = physical_element_aval(aval.dtype)
    from jax._src.sharding_impls import physical_sharding  # pyrefly: ignore[missing-import]
    return ShapedArray((*aval.shape, *elt_aval.shape), elt_aval.dtype,
                       sharding=physical_sharding(aval, aval.sharding),
                       manual_axis_type=aval.mat,
                       memory_space=aval.memory_space)
  return aval

def physical_shape(logical_shape, dtype):
  elt_aval = physical_element_aval(dtype)
  return (*logical_shape, *elt_aval.shape)

def physical_element_aval(edtype: dtypes.ExtendedDType) -> ShapedArray:
  duck = edtype._rules.physical_element_aval(edtype)
  return ShapedArray(duck.shape, dtypes.dtype(duck.dtype))


_dtype_object_types = (np.dtype, dtypes.ExtendedDType)

def _dtype_object(dtype):
  return dtype if isinstance(dtype, _dtype_object_types) else np.dtype(dtype)

def _canonicalize_dimension(dim: DimSize) -> DimSize:
  # 维度绝大多数情况下是整数（且远超其他情况），所以我们先检查这一情况。
  try:
    return operator.index(dim)
  except TypeError as e:
    type_error = e
  if is_dim(dim):
    return dim
  else:
    raise type_error

def canonicalize_shape(shape: Shape, context: str="") -> tuple[Any, ...]:
  """规范化用户提供的 shape 值，并检查其中的错误。

  Args:
    shape: 表示一个 shape 的 Python 值。

  Returns:
    由规范化后的维度值组成的元组。
  """
  if isinstance(shape, int):
    shape = shape,
  try:
    return tuple(unsafe_map(_canonicalize_dimension, shape))
  except TypeError:
    pass
  raise _invalid_shape_error(shape, context)

def canonicalize_dim(d: DimSize, context: str="") -> DimSize:
  """规范化用户提供的 shape 维度值，并检查其中的错误。

  Args:
    d: 表示一个维度的 Python 值。

  Returns:
    规范化后的维度值。
  """
  return canonicalize_shape((d,), context)[0]

def _invalid_shape_error(shape: Shape, context: str=""):
  msg = ("Shapes must be 1D sequences of concrete values of integer type, "
         f"got {shape}.")
  if context:
    msg += f" {context}."
  if any(isinstance(x, Tracer) and isinstance(typeof(x), ShapedArray)
         and not is_concrete(x) for x in shape):
    msg += ("\nIf using `jit`, try using `static_argnums` or applying `jit` to "
            "smaller subfunctions.")
    for x in shape:
      if isinstance(x, Tracer) and hasattr(x, "_origin_msg"):
        msg += x._origin_msg()

  return TypeError(msg)


class ShardingTypeError(Exception):
  pass


class MemorySpace(enum.Enum):
  Device = enum.auto()
  Host = enum.auto()
  Any = enum.auto()

  def __repr__(self):
    return f"MemorySpace.{self.name}"

  # 出于我无法理解的原因，Enum.__hash__ 会对名称字符串做哈希。
  # 这里改用基于对象标识的哈希。
  __hash__ = object.__hash__


def get_cur_mesh_sharding(spec=None):
  spec = P() if spec is None else spec
  return NamedSharding(mesh_lib.get_abstract_mesh(), spec)

def getu(aval, kind=UnreducedKind.sum):
  if aval.sharding.mesh.are_all_axes_manual:
    out_u = aval.mat.unreduced
    if out_u:
      if (aval_k := aval.mat.unreduced_kind) is not kind:
        raise ValueError(f'Expected unreduced_kind={kind} but got {aval_k}')
    return out_u
  if aval.sharding.mesh.are_all_axes_explicit:
    out_u = aval.sharding.spec.unreduced
    if out_u:
      if (aval_k := aval.sharding.spec.unreduced_kind) is not kind:
        raise ValueError(f'Expected unreduced_kind={kind} but got {aval_k}')
    return out_u
  # 在支持部分手动 unreduced 之后修改这里
  assert not aval.mat.unreduced
  assert not aval.sharding.spec.unreduced
  return frozenset()

def getr(aval):
  if aval.sharding.mesh.are_all_axes_manual:
    return aval.mat.reduced
  if aval.sharding.mesh.are_all_axes_explicit:
    return aval.sharding.spec.reduced
  # 在支持部分手动 reduced 之后修改这里
  assert not aval.mat.reduced
  assert not aval.sharding.spec.reduced
  return frozenset()

def _make_lengths_same(sharding, ndim):
  pspec = sharding.spec
  if ndim > len(pspec):
    return sharding.update(spec=pspec._normalized_spec_for_aval(ndim))
  if ((sharding.mesh.empty or sharding.mesh._are_all_axes_auto_or_manual) and
      ndim < len(pspec)):
    assert all(s is None for s in pspec.partitions[ndim:])
    return sharding.update(spec=sharding.spec.update(
        partitions=pspec.partitions[:ndim]))
  return sharding

def modify_spec_for_auto_manual(spec, mesh) -> P:
  new_spec: list[Any] = []
  # PartitionSpec 只能提及 Explicit 类型的 mesh 轴。
  for s in spec.partitions:
    if s is None:
      new_spec.append(s)
    elif isinstance(s, tuple):
      new_spec.append(tuple(
          p for p in s if mesh._name_to_type[p] == AxisType.Explicit))
    else:
      new_spec.append(s if mesh._name_to_type[s] == AxisType.Explicit else None)
  new_unreduced = {u for u in spec.unreduced
                   if mesh._name_to_type[u] == AxisType.Explicit}
  new_reduced = {u for u in spec.reduced
                 if mesh._name_to_type[u] == AxisType.Explicit}
  u_kind = spec.unreduced_kind if new_unreduced else None
  return P(*new_spec, unreduced=new_unreduced, reduced=new_reduced,
           unreduced_kind=u_kind)


def _maybe_modify_sharding(sharding, ndim):
  if len(sharding.spec) == 0 or all(s is None for s in sharding.spec.partitions):
    out = sharding
  elif sharding.mesh.are_all_axes_explicit:
    out = sharding
  else:
    out = sharding.update(spec=modify_spec_for_auto_manual(
        sharding.spec, sharding.mesh))
  if config.remove_size_one_mesh_axis_from_type.value:
    out = out.update(spec=ns.remove_size_one_mesh_axis_from_spec(out.spec, out.mesh))
  if len(out.spec) != ndim:
    out = _make_lengths_same(out, ndim)
  return out

def _check_divisibility(sharding, shape):
  mesh = sharding.mesh
  for dim, (spec, sh) in enumerate(zip(sharding.spec.partitions, shape)):
    if spec is None:
      continue
    spec = spec if isinstance(spec, tuple) else (spec,)
    size = math.prod(mesh.shape[s] for s in spec)
    _, remainder = divmod(sh, size)
    if remainder != 0:
      raise ValueError(
          f"Sharding spec {spec} implies that array axis {dim} is partitioned"
          f" {size} times, but does not evenly divide the dimension size {sh}."
          f" Got shape: {shape} and sharding {sharding}")

@cache(max_size=4096,
       trace_context_in_key=lambda: config.remove_size_one_mesh_axis_from_type.value)
def get_sharding(sharding, shape):
  """修改并检查分片。

  其中一些修改/检查包括：
    * 使 spec 的长度与 ndim 相同
    * 如果 pspec 中提及的某个 mesh 轴是 Auto/Manual，则将其替换为 None
    * 检查 len(spec) 与 ndim 是否匹配
    * 检查 mesh 是否为 AbstractMesh。
  """
  ndim = len(shape)
  if sharding is None:
    return _empty_sharding(ndim)

  out_s = _maybe_modify_sharding(sharding, ndim)
  if len(out_s.spec) != ndim:
    raise ValueError(
        f"Length of sharding.spec ({len(out_s.spec)}) must be equal to aval's"
        f" ndim ({ndim}). Got sharding.spec {out_s.spec}, aval.ndim {ndim} and"
        f" sharding {out_s}")
  if not isinstance(out_s.mesh, mesh_lib.AbstractMesh):
    raise ValueError("Mesh of an aval must be an AbstractMesh. "
                     f"Got {out_s.mesh} of type {type(out_s.mesh)}")
  _check_divisibility(out_s, shape)
  if out_s.memory_kind is not None:
    raise ValueError(
        "sharding with memory_kind is not allowed. Please use `jax.device_put`"
        f" to transfer to different memory spaces. Got {sharding=}")
  return out_s


@cache(max_size=4096,
       trace_context_in_key=lambda: config.remove_size_one_mesh_axis_from_type.value)
def get_mat(mat, mesh):
  if mesh.empty:
    assert mat.empty, mat
    return mat

  axis_env = get_axis_env()
  in_axis_env = lambda i: axis_env.axis_exists(i) and i not in mesh._name_to_type
  for i in it.chain(mat.varying, mat.unreduced, mat.reduced):
    if in_axis_env(i):
      continue
    if mesh._name_to_type[i] != AxisType.Manual:
      raise ValueError(
          "Axes mentioned in `manual_axis_type` field of ShapedArray should be"
          f" of type `Manual`. Got manual_axis_type={mat} with axis: {i} of"
          f" type {mesh._name_to_type[i]}")
  if config.remove_size_one_mesh_axis_from_type.value:
    varying = frozenset(i for i in mat.varying
                        if in_axis_env(i) or mesh.shape[i] != 1)
    unreduced = frozenset(u for u in mat.unreduced if mesh.shape[u] != 1)
    reduced = frozenset(r for r in mat.reduced if mesh.shape[r] != 1)
    u_kind = mat.unreduced_kind if unreduced else None
    return mat.update(varying=varying, unreduced=unreduced, reduced=reduced,
                      unreduced_kind=u_kind)
  return mat


def get_memory_space(memory_space):
  assert memory_space is not None
  return memory_space


def _check_mat(varying, unreduced, reduced, unreduced_kind):
  if varying & unreduced:
    raise ValueError(
        "varying and unreduced cannot have common mesh axes. Got"
        f" varying={varying} and unreduced={unreduced}")
  if varying & reduced:
    raise ValueError(
        "varying and reduced cannot have common mesh axes. Got"
        f" varying={varying} and reduced={reduced}")
  assert not (varying & unreduced & reduced)

  if unreduced_kind is not None and not isinstance(unreduced_kind, UnreducedKind):
    raise TypeError(
        "Expected unreduced_kind to be of type `jax.sharding.UnreducedKind`"
        f" but got {type(unreduced_kind)}")
  if not unreduced and unreduced_kind is not None:
    raise ValueError(
        "`unreduced_kind` should be `None` when `unreduced` is an empty set."
        f" Got {unreduced_kind=} and {unreduced=}")

def _canonicalize_mat(name, val):
  if not isinstance(val, frozenset):
    if not isinstance(val, set):
      raise TypeError(
          f"{name} argument of ManualAxisType should "
          f"of type `frozenset` or `set`. Got type {type(val)}")
    val = frozenset(val)
  return val


@immutable
class ManualAxisType:
  __slots__ = ('varying', 'unreduced', 'reduced', 'unreduced_kind',
               '__weakref__')

  varying: frozenset
  unreduced: frozenset
  reduced: frozenset
  unreduced_kind: UnreducedKind | None

  @staticmethod
  @weak_value_interner
  def _create(varying, unreduced, reduced, unreduced_kind):
    # 我们不能修改驻留函数内部的参数，但可以
    # 自由地抛出异常。
    _check_mat(varying, unreduced, reduced, unreduced_kind)
    obj = object.__new__(ManualAxisType)
    object.__setattr__(obj, 'varying', varying)
    object.__setattr__(obj, 'unreduced', unreduced)
    object.__setattr__(obj, 'reduced', reduced)
    object.__setattr__(obj, 'unreduced_kind', unreduced_kind)
    return obj

  def __new__(cls, *, varying=frozenset(), unreduced=frozenset(),
              reduced=frozenset(), unreduced_kind: UnreducedKind | None = None):
    varying = _canonicalize_mat('varying', varying)
    unreduced = _canonicalize_mat('unreduced', unreduced)
    reduced = _canonicalize_mat('reduced', reduced)
    if unreduced and unreduced_kind is None:
      unreduced_kind = UnreducedKind.sum
    return cls._create(varying, unreduced, reduced, unreduced_kind)

  # 没有 __eq__ 或 __hash__：驻留类使用对象标识。

  def __repr__(self):
    return (f"ManualAxisType(varying={self.varying}, "
            f"unreduced={self.unreduced}, reduced={self.reduced}), "
            f"unreduced_kind={self.unreduced_kind}")

  def __getnewargs_ex__(self):
    return (), {'varying': self.varying, 'unreduced': self.unreduced,
                'reduced': self.reduced, 'unreduced_kind': self.unreduced_kind}

  def update(self, **kwargs):
    if 'varying' not in kwargs:
      kwargs['varying'] = self.varying
    if 'unreduced' not in kwargs:
      kwargs['unreduced'] = self.unreduced
    if 'reduced' not in kwargs:
      kwargs['reduced'] = self.reduced
    if 'unreduced_kind' not in kwargs:
      kwargs['unreduced_kind'] = self.unreduced_kind
    return ManualAxisType(**kwargs)

  def to_ct_mat(self):
    assert self.unreduced_kind is None or self.unreduced_kind is UnreducedKind.sum
    kind = UnreducedKind.sum if self.reduced else None
    return self.update(unreduced=self.reduced, reduced=self.unreduced,
                       unreduced_kind=kind)

  @property
  def empty(self):
    return self is empty_mat

  def invarying(self, mesh) -> frozenset:
    return frozenset(mesh.manual_axes) - (
        self.varying | self.unreduced | self.reduced)

  @property
  def vur(self) -> frozenset:
    return self.varying | self.unreduced | self.reduced

empty_mat = ManualAxisType()

@functools.cache
def _empty_sharding(ndim):
  return NamedSharding(mesh_lib.empty_abstract_mesh, P(*[None] * ndim))


@immutable
class ShapedArray(AbstractValue):
  # 从父类继承 slots
  __slots__ = ['shape', 'dtype', 'weak_type', 'sharding', 'manual_axis_type',
               'memory_space', 'layout', '_stripped_weak_type', '__weakref__']
  array_abstraction_level = 2

  shape: Any
  dtype: Any
  weak_type: Any
  sharding: Any
  manual_axis_type: Any
  memory_space: Any
  layout: Any
  _stripped_weak_type: Any

  @staticmethod
  @weak_value_interner
  def _create(shape, dtype, weak_type, sharding, manual_axis_type,
              memory_space, layout):
    obj = object.__new__(ShapedArray)
    object.__setattr__(obj, 'shape', shape)
    object.__setattr__(obj, 'dtype', dtype)
    object.__setattr__(obj, 'weak_type', weak_type)
    object.__setattr__(obj, 'sharding', sharding)
    object.__setattr__(obj, 'manual_axis_type', manual_axis_type)
    object.__setattr__(obj, 'memory_space', memory_space)
    object.__setattr__(obj, 'layout', layout)
    object.__setattr__(obj, '_stripped_weak_type', None)
    return obj

  def __new__(cls, shape, dtype, weak_type=False, *, sharding=None,
              manual_axis_type: ManualAxisType = empty_mat,
              memory_space: MemorySpace = MemorySpace.Device,
              layout=AutoLayout):
    shape = canonicalize_shape(shape)
    dtype = _dtype_object(dtype)
    if sharding is None:
      sharding = _empty_sharding(len(shape))
      assert manual_axis_type.empty, manual_axis_type
    else:
      sharding = get_sharding(sharding, shape)
      # https://docs.jax.dev/en/latest/notebooks/shard_map.html#tracking-how-values-vary-over-manual-mesh-axes-and-check-vma-true
      manual_axis_type = get_mat(manual_axis_type, sharding.mesh)
    # 参见 https://github.com/jax-ml/jax/pull/30556 的说明
    memory_space = get_memory_space(memory_space)
    return cls._create(shape, dtype, weak_type, sharding, manual_axis_type,
                       memory_space, layout)

  # 驻留类型不需要 __eq__ 或 __hash__。

  @property
  def mat(self):
    return self.manual_axis_type

  def update(self, shape=None, dtype=None, weak_type=None, **kwargs):
    if shape is None:
      shape = self.shape
    if dtype is None:
      dtype = self.dtype
    if weak_type is None:
      weak_type = self.weak_type
    if 'sharding' not in kwargs:
      kwargs['sharding'] = self.sharding
    if 'manual_axis_type' not in kwargs:
      kwargs['manual_axis_type'] = self.manual_axis_type
    if 'memory_space' not in kwargs:
      kwargs['memory_space'] = self.memory_space
    if 'layout' not in kwargs:
      kwargs['layout'] = self.layout
    return ShapedArray(shape, dtype, weak_type, **kwargs)

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self:
                  0 if any(type(d) is int and d == 0 for d in self.shape)
                  else math.prod(self.shape))

  broadcast: ClassVar[aval_method | None] = None
  transpose: ClassVar[aval_method | None] = None
  reshape: ClassVar[aval_method | None] = None
  _iter: ClassVar[staticmethod | None] = None

  def __getnewargs_ex__(self):
    return (self.shape, self.dtype, self.weak_type), {
        'sharding': self.sharding,
        'manual_axis_type': self.manual_axis_type,
        'memory_space': self.memory_space,
        'layout': self.layout,
    }

  def str_short(self, short_dtypes=False, mesh_axis_types=False):
    return str_short_aval(
        self.shape, self.dtype, self.sharding.mesh, self.sharding.spec,
        self.mat, self.memory_space, short_dtypes, mesh_axis_types)

  def __repr__(self):
    wt_str = ", weak_type=True" if self.weak_type else ""
    return f'ShapedArray({self.str_short()}{wt_str})'

  def __str__(self):
    wt_str = "~" if self.weak_type else ""
    return f'{wt_str}{self.str_short()}'

  def to_tangent_aval(self):
    return ShapedArray._create(
        self.shape, primal_dtype_to_tangent_dtype(self.dtype),
        self.weak_type, self.sharding, self.mat, self.memory_space,
        self.layout)

  def to_ct_aval(self):
    dtype = primal_dtype_to_tangent_dtype(self.dtype)
    sharding = primal_sharding_to_cotangent_sharding(self.sharding)
    ct_mat = self.mat.to_ct_mat()
    return ShapedArray._create(
        self.shape, dtype, self.weak_type, sharding, ct_mat, self.memory_space,
        self.layout)

  def _len(self, ignored_tracer):
    try:
      return self.shape[0]
    except IndexError as err:
      raise TypeError("len() of unsized object") from err  # 与 numpy 报错相同

  def update_manual_axis_type(self, mat):
    mat = get_mat(mat, self.sharding.mesh)
    if mat is self.manual_axis_type:
      return self
    return ShapedArray._create(self.shape, self.dtype, self.weak_type,
                               self.sharding, mat, self.memory_space,
                               self.layout)

  def update_weak_type(self, weak_type):
    if weak_type == self.weak_type:
      return self
    return ShapedArray._create(self.shape, self.dtype, weak_type, self.sharding,
                               self.manual_axis_type, self.memory_space,
                               self.layout)

  def strip_weak_type(self) -> AbstractValue:
    if not self.weak_type:
      return self
    # _stripped_weak_type 不受锁保护，但访问应该是
    # 安全的，因为 ShapedArray 值是驻留的：如果两个线程竞争设置
    # 该值，得到的将是同一个对象。
    val = self._stripped_weak_type
    if val is None:
      val = self.update_weak_type(False)
      object.__setattr__(self, '_stripped_weak_type', val)
    return val

  def nospec(self, mesh, check_vma, all_names) -> P:
    # TODO(mattjj, yashkatariya): 在 check_vma 路径中是否应该使用新的 all_names？
    sh_names = (order_wrt_mesh(mesh, self.mat.varying)
                if check_vma else all_names)
    u_names = self.mat.unreduced if check_vma else frozenset()
    r_names = self.mat.reduced if check_vma else frozenset()
    u_kind = self.mat.unreduced_kind if check_vma else None
    return (P(sh_names, unreduced=u_names, reduced=r_names, unreduced_kind=u_kind)
            if sh_names else
            P(unreduced=u_names, reduced=r_names, unreduced_kind=u_kind))

  _bool    = concretization_function_error(bool)
  _int     = concretization_function_error(int, True)
  _float   = concretization_function_error(float, True)
  _complex = concretization_function_error(complex, True)
  _hex     = concretization_function_error(hex)
  _oct     = concretization_function_error(oct)
  _index   = concretization_function_error(operator.index)


def _get_shape_sharding_str(shape, spec):
  out = []
  for s1, s2 in zip(shape, spec.partitions):
    if s2 is None:
      out.append(f"{s1}")
    elif isinstance(s2, tuple):
      ss = ','.join(s for s in s2)
      out.append(f"{s1}@({ss})")
    else:
      out.append(f"{s1}@{s2}")
  return ','.join(out)

@cache(max_size=1024, trace_context_in_key=False)
def _axis_types_dict(mesh):
  if not mesh.axis_names:
    return {}
  d = defaultdict(list)
  for n, t in safe_zip(mesh.axis_names, mesh.axis_types):
    d[t].append(n)
  return {t: tuple(n) for t, n in d.items()}

def str_short_aval(shape, dtype, mesh, spec, mat, memory_space,
                   short_dtypes=False, mesh_axis_types=False) -> str:
  dt_str = dtypes.short_dtype_name(dtype) if short_dtypes else dtype.name
  dt_str = dt_str.replace('void', 'float0')
  shapestr = _get_shape_sharding_str(shape, spec)
  mesh_axes = f'({_axis_types_dict(mesh)})' if mesh_axis_types else ''
  vma_ur = _vma_ur_str(mat, spec.unreduced, spec.reduced, spec.unreduced_kind,
                       mesh)
  ms_str = ("" if memory_space == MemorySpace.Device else
            f"<{memory_space.name.lower()}>")
  return f'{dt_str}{ms_str}[{shapestr}]{vma_ur}{mesh_axes}'

def _create_str(x, prefix):
  x_str = f"{','.join(i for i in x)}"
  x_str = x_str if len(x) == 1 else f"({x_str})"
  return f"{prefix}:{x_str}, "

def order_wrt_mesh(mesh, x):
  return tuple(a for a in mesh.axis_names if a in x)

def _vma_ur_str(mat, spec_unreduced, spec_reduced, u_kind, mesh):
  vma = mat.varying
  # TODO(yashkatariya): 显式 unreduced 与手动 unreduced 之间的差异
  unreduced = mat.unreduced | spec_unreduced
  reduced = mat.reduced | spec_reduced
  if not vma and not unreduced and not reduced:
    return ''
  vma_str = _create_str(order_wrt_mesh(mesh, vma), 'V') if vma else ''
  u_str = ''
  if unreduced:
    u_prefix = ('U' if u_kind is None or u_kind is UnreducedKind.sum else
                f'U_{u_kind.name}')
    u_str = _create_str(order_wrt_mesh(mesh, unreduced), u_prefix)
  r_str = _create_str(order_wrt_mesh(mesh, reduced), 'R') if reduced else ''
  m_str = f"{vma_str}{u_str}{r_str}".rstrip(', ')
  return f"{{{m_str}}}"

def primal_dtype_to_tangent_dtype(primal_dtype):
  if isinstance(primal_dtype, dtypes.ExtendedDType):
    return primal_dtype._rules.tangent_dtype(primal_dtype)
  elif not dtypes.issubdtype(primal_dtype, np.inexact):
    return dtypes.float0
  else:
    return primal_dtype

def primal_sharding_to_cotangent_sharding(sharding):
  return sharding.update(spec=sharding.spec.to_ct_spec())

############################## pvary #################################

# Invariant -> Variant 的空操作转换
def pvary(x, axis_name):
  axes = (axis_name,) if not isinstance(axis_name, tuple) else axis_name
  if not axis_name:
    return x
  cur_mesh = mesh_lib.get_abstract_mesh()
  if not config._check_vma.value and all(a in cur_mesh.manual_axes for a in axes):
    return x
  new_axes = axes if cur_mesh.empty else order_wrt_mesh(cur_mesh, axes)
  assert set(new_axes) == set(axes)
  del axes
  # TODO(yashkatariya): 移除这一处理，并从 JAX 中彻底移除
  # remove_size_one_mesh_axis_from_type。
  if config.remove_size_one_mesh_axis_from_type.value and not cur_mesh.empty:
    new_axes = tuple(i for i in new_axes if cur_mesh.shape[i] != 1)
    if not new_axes:
      return x
  return tree_map(lambda leaf: pvary_p.bind(leaf, axes=new_axes), x)

pvary_p = Primitive('pvary')

####################### reduced_vary_cast #############################

# Reduced -> Varying 的空操作转换
def reduced_vary_cast(x, axis_name):
  axes = (axis_name,) if not isinstance(axis_name, tuple) else axis_name
  if not axis_name:
    return x
  cur_mesh = mesh_lib.get_abstract_mesh()
  if not config._check_vma.value and all(a in cur_mesh.manual_axes for a in axes):
    return x
  new_axes = axes if cur_mesh.empty else order_wrt_mesh(cur_mesh, axes)
  assert set(new_axes) == set(axes)
  del axes
  return tree_map(lambda leaf: reduced_vary_cast_p.bind(leaf, axes=new_axes), x)

reduced_vary_cast_p = Primitive('reduced_vary_cast_p')

#######################################################################

def check_unreduced_args(args, axes, name, kind=UnreducedKind.sum):
  axes = axes if isinstance(axes, (tuple, list)) else (axes,)
  axes = set(axes)
  for a in args:
    if a.mat.unreduced & axes:
      raise ValueError(
          f"{name} cannot accept args which are unreduced. Got"
          f" {a.str_short(True)} and axes={axes}")
    if a.mat.unreduced and a.mat.unreduced_kind is not kind:
      raise ValueError(
          f"{name} cannot accept args with"
          f" unreduced_kind={a.mat.unreduced_kind}. Expected"
          f" unreduced_kind={kind}")
    if a.mat.reduced & axes:
      raise ValueError(
          f"{name} cannot accept args which are reduced. Got"
          f" {a.str_short(True)} and axes={axes}")

def insert_reduced_reshard(args):
  cur_mesh = mesh_lib.get_abstract_mesh()
  if not cur_mesh.are_all_axes_explicit:
    return args
  # TODO(yashkatariya): 也要处理多于 2 个参数的情况
  if len(args) != 2:
    return args
  in_reduced = [aval.sharding.spec.reduced
                if isinstance(aval := shaped_abstractify(a), ShapedArray)
                else frozenset() for a in args]
  out_reduced = frozenset.union(*in_reduced)
  out = []
  for arg, src_reduced in zip(args, in_reduced):
    aval = shaped_abstractify(arg)
    if (isinstance(aval, ShapedArray) and aval.ndim == 0 and out_reduced and
        (get_replicated_axes(aval.sharding.spec, cur_mesh) & out_reduced) == out_reduced):
      from jax._src.pjit import reshard  # type: ignore
      out.append(reshard(arg, P(reduced=out_reduced)))
    else:
      out.append(arg)
  return out

def auto_insert_reshard(*args):
  if not args:
    return args
  if not config._check_vma.value:
    return insert_reduced_reshard(args)
  if not config.auto_pcast.value:
    return args
  in_vma = [aval.mat.varying if isinstance(aval := typeof(a), ShapedArray)
            else frozenset() for a in args]
  in_reduced = [aval.mat.reduced
                if isinstance(aval := typeof(a), ShapedArray) else frozenset()
                for a in args]
  out_vma = frozenset.union(*in_vma)
  out = []
  for arg, src_vma, src_reduced in zip(args, in_vma, in_reduced):
    if (isinstance(typeof(arg), ShapedArray) and
        (rest_vma := out_vma - src_vma)):
      # TODO(yashkatariya): 处理部分 reduced_vary_cast 和部分 pvary。
      # 需要对 pvary 做更多修改才能支持这种部分性。
      if src_reduced == rest_vma:
        out.append(
            reduced_vary_cast(arg, tuple(n for n in out_vma if n in rest_vma)))
      else:
        out.append(pvary(arg, tuple(n for n in out_vma if n in rest_vma)))
    else:
      out.append(arg)
  return out

def standard_vma_rule(prim_name, *avals, **kwargs) -> frozenset[AxisName]:
  if not config._check_vma.value:
    return frozenset()
  avals = tuple(a for a in avals if a is not abstract_token)
  if not avals:
    return frozenset()
  vma, *vmas = (a.mat.varying for a in avals)
  if not all(vma == vma_ for vma_ in vmas):
    raise ValueError(
        f'Primitive {prim_name} requires varying manual axes '
        f'to match, but got {[vma, *vmas]}. Please open an issue at '
        'https://github.com/jax-ml/jax/issues and as a temporary '
        'workaround pass the check_vma=False argument to `jax.shard_map`')
  return vma

@dataclass(frozen=True, slots=True)
class bint(dtypes.ExtendedDType):
  bound: int

  @property
  def type(self) -> type:
    return dtypes.extended

  @property
  def name(self) -> str:
    return f'bint{{≤{self.bound}}}'

  def __str__(self) -> str:
    return self.name

AxisSize = int | Tracer | Var


class RefMeta(type):
  def __instancecheck__(self, inst):
    from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
    return (super().__instancecheck__(inst) or
            isinstance(inst, Tracer) and isinstance(inst.aval, AbstractRef))

class Ref(metaclass=RefMeta):
  """可变的数组引用。

  在大多数情况下不应直接构造它，而应
  通过 :func:`jax.ref.new_ref` 构造。有关如何使用它的示例，
  请参阅 `Ref guide`_。

  .. _Ref guide: https://docs.jax.dev/en/latest/array_refs.html
  """
  _aval: AbstractValue
  _refs: PyTree  # ArrayRefImpl 的列表

  def __init__(self, aval, refs):
    from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
    assert isinstance(aval, AbstractRef)
    self._aval = aval
    self._refs = refs

  def __repr__(self) -> str:
    if self._aval.is_high:
      return f"Ref({self._refs})"
    return "Ref" + repr(self._refs._buf)[5:]

  # 将类型层面的信息转发给 aval
  aval = property(lambda self: self._aval)
  shape = property(lambda self: self._aval.shape)
  size = property(lambda self: self._aval.size)
  ndim = property(lambda self: len(self._aval.shape))
  dtype = property(lambda self: self._aval.dtype)

  # 从 aval 获取操作，并改写名称
  def __getitem__(self, idx): return self._aval._getitem(self, idx)  # pyrefly: ignore[missing-attribute]
  def __setitem__(self, idx, x): return self._aval._setitem(self, idx, x)  # pyrefly: ignore[missing-attribute]
  def __len__(self) -> int: return self._aval._len(self)  # pyrefly: ignore[missing-attribute]
  def addupdate(self, x, idx=()): return self._aval._addupdate(self, idx, x)  # pyrefly: ignore[missing-attribute]

  # 某些属性/方法仅对 lojax 引用有效
  sharding = property(lambda self: self._refs._buf.sharding)
  format = property(lambda self: self._refs._buf.format)
  committed = _committed = property(lambda self: True)
  def unsafe_buffer_pointer(self): return self._refs._buf.unsafe_buffer_pointer()

  @property
  def at(self): raise NotImplementedError()  # TODO(mattjj)

class ArrayRefImpl:
  _aval: AbstractValue
  _buf: Array  # 可变字段

  def __init__(self, aval, buf):
    from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
    assert isinstance(aval, AbstractRef) and isinstance(aval.inner_aval, ShapedArray)
    self._aval = aval
    self._buf = buf

pytype_aval_mappings[Ref] = lambda x: x._aval
dtypes.register_canonicalize_value_handler(Ref, None)


class InternalMutableArrayEffect(effects.Effect):
  pass
array_ref_effect = internal_mutable_array_effect = InternalMutableArrayEffect()
effects.control_flow_allowed_effects.add_type(InternalMutableArrayEffect)
effects.remat_allowed_effects.add_type(InternalMutableArrayEffect)


def new_ref(init_val: Any, *, memory_space: Any = None, kind: Any = None,
            pin: bool = False):
  """创建一个初值为 ``init_val`` 的可变数组引用。

  更多讨论请参阅 `Ref guide`_。

  Args:
    init_val: 一个 :class:`jax.Array`，表示缓冲区的初始
      状态。
    memory_space: 可选的 Ref 内存空间属性。
    kind: 可选的字符串，指示重物化（rematerialization）下的
      变更语义。
    pin: 是否在 HLO 中把该 ref 降级为 pinned 缓冲区。

  Returns:
    一个 :class:`jax.ref.Ref`，其中包含对可变缓冲区的引用。

  .. _Ref guide: https://docs.jax.dev/en/latest/array_refs.html
  """
  return ref_p.bind(init_val, memory_space=memory_space, kind=kind, pin=pin)
ref_p = Primitive('new_ref')
ref_p.is_effectful = lambda params: True
ref_p.ref_primitive = True
ref_p.ref_allocating = True

ref_p.is_high = lambda aval, *, memory_space, kind, pin: aval.is_high
def _ref_to_lojax(init_val, *, memory_space, kind, pin):
  from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
  val_ty = typeof(init_val)
  hival_of_refs = val_ty.raise_val(*map(new_ref, val_ty.lower_val(init_val)))
  return Ref(AbstractRef(val_ty), hival_of_refs)
ref_p.to_lojax = _ref_to_lojax

@ref_p.def_effectful_abstract_eval
def _ref_abstract_eval(init_aval, *, memory_space: Any, kind: Any, pin: bool):
  from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
  # 如果未指定内存空间，则使用初始值的内存空间，
  # 但我们确保将其重置为 Device，因为该 Ref 拥有该内存空间
  if (memory_space is None
      and isinstance(init_aval, ShapedArray)):
    if init_aval.memory_space is not MemorySpace.Device:
      memory_space = init_aval.memory_space
    init_aval = init_aval.update(memory_space=MemorySpace.Device)
  return (AbstractRef(init_aval, memory_space=memory_space, kind=kind),
          {internal_mutable_array_effect})

@ref_p.def_impl
def _ref_impl(init_val, *, memory_space: Any, kind: Any, pin: bool):
  if memory_space is not None:
    raise NotImplementedError(
        "array ref with memory space only works inside of a `jit`.")
  if pin:
    raise NotImplementedError(
        "pinned array ref only works inside of a `jit`.")
  from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
  from jax._src.lax.lax import _array_copy  # pyrefly: ignore[missing-import]
  aval = AbstractRef(typeof(init_val), kind=kind)
  return Ref(aval, ArrayRefImpl(aval, _array_copy(init_val)))

# TODO(mattjj,dougalm): 与 ref_p 合并
def empty_ref(ty, memory_space=None, pin=False):
  aval = shaped_abstractify(ty)
  return empty_ref_p.bind(ty=aval, memory_space=memory_space, pin=pin)
empty_ref_p = Primitive('empty_ref')
empty_ref_p.ref_primitive = True
empty_ref_p.is_effectful = lambda _: True
empty_ref_p.ref_allocating = True
empty_ref_p.is_high = lambda *, ty, memory_space, pin: ty.is_high

def _empty_ref_to_lojax(*, ty, memory_space, pin):
  from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
  n = len(ty.lo_ty())
  hival_of_refs = ty.raise_val(
      *map(empty_ref, ty.lo_ty(), [memory_space] * n, [pin] * n))
  return Ref(AbstractRef(ty), hival_of_refs)
empty_ref_p.to_lojax = _empty_ref_to_lojax


@empty_ref_p.def_effectful_abstract_eval
def _empty_ref_abstract_eval(*, ty, memory_space, pin):
  from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
  return (AbstractRef(ty, memory_space=memory_space),
          {internal_mutable_array_effect})


# TODO(mattjj,dougalm): 与 freeze_p 合并
def free_ref(ref: Ref):
  """使给定的引用失效。"""
  free_ref_p.bind(ref)
  return ()

free_ref_p = Primitive('free_ref')
free_ref_p.multiple_results = True
free_ref_p.is_effectful = lambda _: True
free_ref_p.ref_primitive = True


@free_ref_p.def_effectful_abstract_eval
def _free_ref_abstract_eval(ref_aval):
  # 没有效果，但存在一个自定义的 DCE 规则，它阻止 free_ref
  # 被 DCE 消除。
  return (), {}


@free_ref_p.def_impl
def _free_ref_impl(ref):
  return ()

def freeze(ref: Ref) -> Array:
  """使给定的引用失效并返回其最终值。

  有关可变数组引用的更多信息，请参阅
  `Ref guide`_。

  Args:
    ref: 一个 :class:`jax.ref.Ref` 对象。

  Returns:
    一个 :class:`jax.Array`，包含 ``ref`` 的内容。

  Examples:
    >>> import jax
    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> ref[3] = 100
    >>> ref
    Ref([  0,   1,   2, 100,   4], dtype=int32)

    >>> jax.ref.freeze(ref)
    Array([  0,   1,   2, 100,   4], dtype=int32)

  .. _Ref guide: https://docs.jax.dev/en/latest/array_refs.html
  """
  return freeze_p.bind(ref)
freeze_p = Primitive('freeze')
freeze_p.is_effectful = lambda params: True
freeze_p.ref_primitive = True
freeze_p.is_high = lambda aval: aval.is_high
def _freeze_to_lojax(ref):
  aval = typeof(ref._refs)
  lovals = aval.lower_val(ref._refs)
  vals = [freeze(loval) for loval in lovals]
  return aval.raise_val(*vals)
freeze_p.to_lojax = _freeze_to_lojax

@freeze_p.def_effectful_abstract_eval
def freeze_abstract_eval(ref_aval):
  return ref_aval.inner_aval, {internal_mutable_array_effect}

@freeze_p.def_impl
def _freeze_impl(ref):
  return ref[()]

def accum_grad_in_ref(x):
  return accum_grad_in_ref_p.bind(x)

accum_grad_in_ref_p = Primitive('accum_grad_in_ref')
accum_grad_in_ref_p.is_high = lambda *_: True
accum_grad_in_ref_p.to_lojax = lambda x: x
accum_grad_in_ref_p.def_abstract_eval(lambda x: x)
accum_grad_in_ref_p.def_impl(lambda x: x)


class AbstractToken(AbstractValue):
  def str_short(self, short_dtypes=False, mesh_axis_types=False): return 'Tok'
  def to_tangent_aval(self): return self
  def to_ct_aval(self): return self
abstract_token: AbstractToken = AbstractToken()

# 当需要形状/数据类型时，所有抽象 token 共用的单例 ShapedArray。
def get_token_aval():
  return ShapedArray((0,), np.dtype(np.bool_), sharding=None)

# 具体 token 对象
class Token:
  # token 包装的底层数据，可以传入和传出计算，从而构建数据依赖。
  _buf: Array
  def __init__(self, buf):
    self._buf = buf
  def block_until_ready(self):
    self._buf.block_until_ready()
pytype_aval_mappings[Token] = lambda _: abstract_token
dtypes.register_canonicalize_value_handler(Token, None)


class AbstractFuture(AbstractValue):
  def __init__(self, inner_aval, done_fun):
    self.inner_aval = inner_aval
    self.done_fun = done_fun

  def __eq__(self, other):
    return (isinstance(other, AbstractFuture) and
            self.inner_aval == other.inner_aval)

  def __hash__(self):
    return hash(self.inner_aval)

  def str_short(self, short_dtypes=False, mesh_axis_types=False) -> str:
    return f'AbstractFuture{{{self.inner_aval.str_short(True)}}}'

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: math.prod(self.shape))

  @aval_method
  def done(tracer):
    return tracer.aval.done_fun(tracer)  # type: ignore

  @property
  def shape(self):
    try:
      return self.inner_aval.shape
    except AttributeError:
      raise AttributeError(f"{self!r} has no `sharding`.") from None

  @property
  def dtype(self):
    try:
      return self.inner_aval.dtype
    except AttributeError:
      raise AttributeError(f"{self!r} has no `sharding`.") from None

  @property
  def sharding(self):
    try:
      return self.inner_aval.sharding
    except AttributeError:
      raise AttributeError(f"{self!r} has no `sharding`.") from None

  @property
  def manual_axis_type(self):
    try:
      return self.inner_aval.manual_axis_type
    except AttributeError:
      raise AttributeError(f"{self!r} has no `manual_axis_type`.") from None

  @property
  def mat(self):
    return self.manual_axis_type


### 对形状和维度大小的操作。

@set_module("jax.errors")
class InconclusiveDimensionOperation(Exception):
  """当无法对符号维度进行确定性计算时抛出。"""

def is_symbolic_dim(v: Any) -> bool:
  """检查某个值是否是为形状多态而使用的符号维度。

  这应当极少使用，因为符号维度重载了所有运算符，
  本应直接可用。
  """
  return getattr(v, "dimension_as_value", None) is not None

def is_constant_dim(d: DimSize) -> bool:
  # 该维度是否为静态整数常量。
  # 对非具体（non-concrete）的 Tracer 尝试使用快速路径。
  if isinstance(d, Tracer) and not is_concrete(d):
    return False
  try:
    operator.index(d)
    return True
  except:
    return False

def is_dim(v: Any) -> bool:
  return is_symbolic_dim(v) or is_constant_dim(v)

def is_constant_shape(s: Shape) -> bool:
  # 形状是否为静态常量。
  return all(is_constant_dim(d) for d in s)

def definitely_equal_one_of_dim(d1: DimSize, dlist: Sequence[DimSize]) -> bool:
  return any(definitely_equal(d1, d) for d in dlist)

def definitely_equal_shape(s1: Shape, s2: Shape) -> bool:
  """检查两个形状是否保证逐元素相等。

  在存在动态形状时，即使这些形状在运行时可能相等，
  也可能返回 False。
  """
  return (len(s1) == len(s2) and
          all(unsafe_map(definitely_equal, s1, s2)))

def divide_shape_sizes(s1: Shape, s2: Shape) -> DimSize:
  """返回一个整数 "i"，使得 i * size(s2) == size(s1)。
  如果不存在这样的整数，则抛出 InconclusiveDimensionOperation。"""
  sz1 = math.prod(s1)
  sz2 = math.prod(s2)
  if definitely_equal(sz1, sz2):  # 处理 sz1 和 sz2 为 0 的情况
    return 1
  q, r = divmod(sz1, sz2)
  if isinstance(r, Tracer) or r != 0:
    raise InconclusiveDimensionOperation(
        f"Cannot divide evenly the sizes of shapes {tuple(s1)} and {tuple(s2)}. "
        f"The remainder {r} should be 0.")
  return q

def cancel_divide_tracers(num, denom):
  partition = lambda l: partition_list([isinstance(d, Tracer) for d in l], l)
  num, num_tracers = partition(num)
  denom, denom_tracers = partition(denom)
  if num_tracers or denom_tracers:
    factor = _cancel_divide(num_tracers, denom_tracers)
    if factor is not None:
      size1 = math.prod(num)
      size2 = math.prod(denom)
      if size1 == size2 or size2 != 0:
        return factor * (size1 // size2 if size1 != size2 else 1)

def _cancel_divide(num, denom):
  num = list(num)
  for a in denom:
    i = next((i for i, b in enumerate(num) if definitely_equal(a, b)), None)
    if i is None:
      break  # 无法约去
    del num[i]
  else:
    return math.prod(num)

def is_empty_shape(s: Shape) -> bool:
  return any(definitely_equal(d, 0) for d in s)

def dilate_dim(d: DimSize, dilation: DimSize) -> DimSize:
  """max(0, 1 + dilation * (d - 1))。

  假定 dilation >= 1。
  """
  if definitely_equal(dilation, 1):  # 快速路径
    return d
  return max_dim(1 + dilation * (d - 1), 0)

def stride_dim(d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
  """max(0, (d - window_size) // window_stride + 1)

  如果 d < window_size，则返回 0。
  我们假定 window_size >= 1 且 window_stride >= 1。
  """
  # 如果 d < window_size，那么 (d - window_size) // window_stride < 0
  return max_dim((d - window_size) // window_stride + 1, 0)

def min_dim(d1: DimSize, d2: DimSize) -> DimSize:
  """类似于 min(d1, d2)，但适用于常量和符号维度。"""
  d1_is_constant = is_constant_dim(d1)
  if d1_is_constant and is_constant_dim(d2):
    return min(d1, d2)
  d1 = concrete_dim_or_error(d1, "argument `d1` of `core.min_dim`")
  d2 = concrete_dim_or_error(d2, "argument `d2` of `core.min_dim`")
  if d1_is_constant:
    return d2.rmin(d1)
  else:
    return d1.min(d2)

def max_dim(d1: DimSize, d2: DimSize) -> DimSize:
  """类似于 max(d1, d2)，但适用于常量和符号维度。"""
  d1_is_constant = is_constant_dim(d1)
  if d1_is_constant and is_constant_dim(d2):
      return max(d1, d2)
  d1 = concrete_dim_or_error(d1, "argument `d1` of `core.max_dim`")
  d2 = concrete_dim_or_error(d2, "argument `d2` of `core.max_dim`")
  if d1_is_constant:
    return d2.rmax(d1)
  else:
    return d1.max(d2)

def dimension_as_value(d: DimSize):
  """将维度大小转换为 JAX 数组。
     对于常量维度，这是恒等函数。

     其抽象值与 Python 常量相同。
     """
  if isinstance(d, (int, Tracer, np.int32, np.int64)): return d
  # 用于 shape_poly._DimPolynomial
  m = getattr(d, "dimension_as_value", None)
  if m is not None: return m()
  return operator.index(d)

def canonicalize_slice(
    s: slice,
    axis_size: DimSize
  ) -> tuple[DimSize, DimSize, DimSize]:
  """计算切片 `x[s]` 的起始索引、步长和大小。

  这与 `s.indices(axis_size)` 类似，区别在于它返回
  `(start, step, size)`，并且当切片和/或
  `axis_size` 是符号值时它也能工作。

  参见 https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding
  """
  def convert_to_index(d: DimSize) -> DimSize:
    # 将 np.array 和 jax.Array 转换为 int，保留符号维度不变
    try:
      return operator.index(d)
    except:
      return d

  # 如果 step 属于 {<0, ==0, >0}，则必须静态解析
  step = convert_to_index(s.step) if s.step is not None else 1
  try:
    if step == 0:
      raise ValueError("slice step cannot be zero")
    step_gt_0 = (step > 0)
  except InconclusiveDimensionOperation as e:
    raise InconclusiveDimensionOperation(
        f"In slice with non-constant elements the step ({step}) must " +
        f"be resolved statically if it is > 0 or < 0.\nDetails: {e}")

  def clamp_index(i: DimSize, which: str):
    try:
      i_ge_0 = (i >= 0)
    except InconclusiveDimensionOperation as e:
      raise InconclusiveDimensionOperation(
          f"In slice with non-constant elements the {which} ({i}) must " +
          f"be resolved statically if it is >= 0.\nDetails: {e}")
    if i_ge_0:
      if step_gt_0:
        return min_dim(axis_size, i)
      else:
        return min_dim(axis_size - 1, i)
    else:
      if step_gt_0:
        return max_dim(0, axis_size + i)
      else:
        return max_dim(-1, axis_size + i)

  if s.start is None:
    start = 0 if step_gt_0 else axis_size - 1
  else:
    start = clamp_index(convert_to_index(s.start), "start")

  if s.stop is None:
    stop = axis_size if step_gt_0 else -1
  else:
    stop = clamp_index(convert_to_index(s.stop), "stop")

  gap = step if step_gt_0 else - step
  distance = (stop - start) if step_gt_0 else (start - stop)
  slice_size = max_dim(0, distance + gap - 1) // gap
  return start, step, slice_size


class SomeTracer:
  __slots__ = ()
  def __repr__(self): return "[dynamic]"

def replace_tracer_for_error_message(obj):
  # TODO(mattjj): 有很多改进的想法。遍历栈看看是否
  # 有用户变量的值 == 这个对象？或者至少搜索
  # 正在被变换的函数的参数？或者至少为它们分配
  # 简短且唯一的 id？
  if isinstance(obj, Tracer):
    return SomeTracer()
  else:
    return obj

def evaluate_shape(shape: Shape, dim_vars: Sequence[str],
                   *dim_values: Array) -> Sequence[Array]:
  """对可能包含非常量的 shape 求值。

  Args:
    shape: 待求值的 shape。
    dim_vars: 可能出现在 `shape` 中的维度变量名。
    dim_values: 与 `dim_vars` 对应的维度值。

  Returns:
     与 `shape` 对应的 JAX 值组成的元组，其类型为
     `dim_value_dtype`。
  """
  env = dict(zip(dim_vars, dim_values))
  def eval_one_dim(d: DimSize):
    try:
      return operator.index(d)
    except:
      # 是一个 _DimExpr
      return d._evaluate(env)  # pyrefly: ignore[missing-attribute]
  return tuple(eval_one_dim(d) for d in shape)

def dim_value_dtype():
  """用于维度值的 dtype。"""
  return dtypes.default_int_dtype()

def dim_constant(ct: int):
  dtype = dim_value_dtype()
  assert dtype in (np.int32, np.int64)
  if dtype == np.int32:
    return np.int32(ct)
  elif dtype == np.int64:
    return np.int64(ct)

def dim_value_aval() -> AbstractValue:
  return ShapedArray((), dim_value_dtype(), weak_type=True, sharding=None)

# ------------------- 调用 -------------------

# eval_jaxpr_p 是一个类似 call 的原语，它由 jaxpr 而非
# Python 可调用对象参数化：应用它会求值该 jaxpr，将其暂存出去是 O(1) 的，
# 只生成一个方程，该方程在重新追踪时保持不变。它的
# 变换规则位于 partial_eval.py 和 lax/eval_jaxpr.py 中。
eval_jaxpr_p = Primitive('eval_jaxpr')
eval_jaxpr_p.multiple_results = True
eval_jaxpr_p.def_impl(lambda *args, call_jaxpr, **_: jaxpr_as_fun(call_jaxpr)(*args))
eval_jaxpr_p.def_effectful_abstract_eval(
    lambda *_, call_jaxpr, **__: (call_jaxpr.out_avals, positional_effects(call_jaxpr)))

# 已删除的 final 风格 call 原语的别名，供在解释 jaxpr 时
# 按这些名称进行匹配的下游代码使用。
call_p = closed_call_p = eval_jaxpr_p


# ------------------- 映射 -------------------

def mapped_aval(size: AxisSize, axis, aval: AbstractValue) -> AbstractValue:
  from jax._src.hijax import HiType  # pyrefly: ignore[missing-import]
  if isinstance(aval, HiType):
    return aval.dec_rank(size, axis)  # pyrefly: ignore[bad-argument-type]
  handler, _ = aval_mapping_handlers.get(type(aval), (None, None))
  if handler is not None:
    return handler(size, axis, aval)
  else:
    raise TypeError(f"no mapping handler for {aval} of type {type(aval)}")

def mapped_leading_aval(size, aval) -> AbstractValue:
  return mapped_aval(size, aval.leading_axis_spec(), aval)

# TODO(yashkatariya): 接收轴数据
def unmapped_aval(size: AxisSize, axis: int | None,
                  aval: AbstractValue, explicit_mesh_axis=None) -> AbstractValue:
  from jax._src.hijax import HiType  # pyrefly: ignore[missing-import]
  if isinstance(aval, HiType):
    return aval.inc_rank(size, axis)  # pyrefly: ignore[bad-argument-type]
  _, handler = aval_mapping_handlers.get(type(aval), (None, None))
  if handler is not None:
    return handler(size, axis, explicit_mesh_axis, aval)
  else:
    raise TypeError(f"no unmapping handler for {aval} of type {type(aval)}")

def unmapped_leading_aval(size, aval) -> AbstractValue:
  return unmapped_aval(size, aval.leading_axis_spec(), aval)

def _map_shaped_array(
    size: int, axis: int | None, aval: ShapedArray) -> ShapedArray:
  assert axis is None or aval.shape[axis] == size
  if axis is None:
    return aval
  aval_s = aval.sharding
  sharding = aval_s.update(
      spec=aval_s.spec.update(partitions=tuple_delete(aval_s.spec.partitions, axis)))
  return ShapedArray(tuple_delete(aval.shape, axis), aval.dtype,
                     weak_type=aval.weak_type, sharding=sharding,
                     manual_axis_type=aval.mat, memory_space=aval.memory_space)

def _unmap_shaped_array(
    size: int, axis: int | None, explicit_mesh_axis, aval: ShapedArray
    ) -> ShapedArray:
  if axis is None:
    return aval
  elif type(axis) is int:
    aval_s = aval.sharding
    sharding = aval_s.update(spec=aval_s.spec.update(partitions=tuple_insert(
        aval_s.spec.partitions, axis, explicit_mesh_axis)))
    return ShapedArray(tuple_insert(aval.shape, axis, size), aval.dtype,
                       weak_type=aval.weak_type, sharding=sharding,
                       manual_axis_type=aval.mat,
                       memory_space=aval.memory_space)
  else:
    raise TypeError(axis)

AvalMapHandlerPair = tuple[Callable, Callable]
aval_mapping_handlers: dict[type, AvalMapHandlerPair] = {
    ShapedArray:   (_map_shaped_array, _unmap_shaped_array),
    AbstractToken: (lambda _, __, a: a, lambda _, __, ____, a: a)
}

# 当被映射的函数没有给出轴名时，我们基于函数对象的 id 生成
# 一个名称对象。冲突并不重要，因为这个名称无法在集合通信中
# 使用，用户代码永远拿不到对该对象的引用。我们不想使用函数
# 对象本身，因为那可能会持久保留对函数对象的引用。
# TODO(mattjj): 重新审视这个唯一轴名策略
@total_ordering
class _TempAxisName:

  def __init__(self, obj):
    self.id = id(obj)

  def __repr__(self):
    return f'<axis {hex(self.id)}>'

  def __hash__(self):
    return hash(self.id)

  def __eq__(self, other):
    return type(other) is _TempAxisName and self.id == other.id

  def __lt__(self, other):
    return type(other) is _TempAxisName and self.id < other.id


@dataclass(frozen=True, slots=True)
class NamedAxisEffect(effects.Effect):
  """一种将新的命名轴引入当前作用域的副作用。"""
  name: AxisName

effects.control_flow_allowed_effects.add_type(NamedAxisEffect)
effects.custom_derivatives_allowed_effects.add_type(NamedAxisEffect)
effects.lowerable_effects.add_type(NamedAxisEffect)
effects.remat_allowed_effects.add_type(NamedAxisEffect)


def filter_named_axis_effects(
    effects: Effects, names: Collection[AxisName]
) -> Effects:
  return {e for e in effects
          if not isinstance(e, NamedAxisEffect) or e.name not in names}


def remove_named_axis_effects(
    jaxpr: Jaxpr, names: Collection[AxisName]
) -> Jaxpr:
  if not names or not jaxpr.effects:
    return jaxpr
  return jaxpr.replace(effects=filter_named_axis_effects(jaxpr.effects, names))

def replace_jaxpr_effects(jaxpr: Jaxpr, effects: Effects):
  return _replace_jaxpr_effects(jaxpr, frozenset(effects))

@weakref_lru_cache
def _replace_jaxpr_effects(jaxpr: Jaxpr, effects: frozenset[Effect]):
  return jaxpr.replace(effects=set(effects))

# ------------------- jaxpr 检查 -------------------

def typecheck(aval: AbstractValue, x) -> bool:
  return typecompat(aval, typeof(x))

def typecompat(aval_ref: AbstractValue, aval: AbstractValue) -> bool:
  """判断 `aval` 是否符合 `aval_ref`。忽略 weak_type。"""
  try:
    return typematch(aval_ref, aval)
  except TypeError:
    return False

def typematch(t1: AbstractValue, t2: AbstractValue,
              no_dtype_check: bool = False) -> bool:
  """判断 `t1` 和 `t2` 是否等价。忽略 weak_type。"""
  t1 = t1.normalize()
  t2 = t2.normalize()
  from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
  if t1 == t2:
    return True
  elif isinstance(t1, ShapedArray) and isinstance(t2, ShapedArray):
    if no_dtype_check:
      return cmp_shape_shd_mat_memsp(t1, t2)
    return t1.dtype == t2.dtype and cmp_shape_shd_mat_memsp(t1, t2)
  elif isinstance(t1, AbstractRef) and isinstance(t2, AbstractRef):
    # 这里我们想对 ShapedArray 使用常规的类型检查。
    return (typematch(t1.inner_aval, t2.inner_aval, no_dtype_check) and
            (t1.memory_space is None or t2.memory_space is None or
             t1.memory_space == t2.memory_space))
  else:
    return False

def cmp_shape_shd_mat_memsp(t1, t2):
  # TODO(yashkatariya): 将其扩展到 Manual 和 Auto 模式。
  # 参见 https://github.com/jax-ml/jax/issues/26474
  t1_mesh, t2_mesh = t1.sharding.mesh, t2.sharding.mesh
  if not t1_mesh.empty and not t2_mesh.empty:
    if t1_mesh._any_axis_explicit or t2_mesh._any_axis_explicit:
      shd_eq = t1.sharding == t2.sharding
    else:
      shd_eq = True
  else:
    shd_eq = True
  return (shd_eq and definitely_equal_shape(t1.shape, t2.shape) and
          t1.mat == t2.mat and t1.memory_space == t2.memory_space)

def aval_mismatch_extra(a1: AbstractValue, a2: AbstractValue) -> str:
  assert not typematch(a1, a2)
  if isinstance(a1, ShapedArray) and isinstance(a2, ShapedArray):
    mismatches = []
    if a1.dtype != a2.dtype:
      mismatches.append('the dtypes do not match')
    if a1.shape != a2.shape:
      mismatches.append('the shapes do not match')
    if a1.mat != a2.mat:
      mismatches.append('the manual axis types do not match')
    # TODO(yashkatariya,mattjj): 添加对类型中分片（sharding-in-types）不匹配的检查

    if len(mismatches) == 0:
      return ''
    elif len(mismatches) == 1:
      return ', so ' + mismatches[0]
    else:
      return ', so ' + ', '.join(mismatches[:-1]) + ', and ' + mismatches[-1]
  return ''

@set_module("jax.errors")
class JaxprTypeError(TypeError):
  pass

custom_typechecks: dict[Primitive, Callable] = {}

def _check_closed_call(_, *in_atoms, call_jaxpr, **__):
  in_avals = [x.aval for x in in_atoms]
  if not all(map(typecompat, call_jaxpr.in_avals, in_avals)):
    raise JaxprTypeError("Closed call in_avals mismatch")
  return call_jaxpr.out_avals, positional_effects(call_jaxpr)
custom_typechecks[eval_jaxpr_p] = _check_closed_call

def check_jaxpr(jaxpr: Jaxpr):
  """检查 jaxpr 的良构性。

  具体来说，检查：
  - 被读取的变量此前已经绑定
  - 同一变量在整个 jaxpr 中类型保持一致
  - 变量的类型标注与其绑定表达式兼容

  如果判定 `jaxpr` 无效，则抛出 `JaxprTypeError`。否则
  返回 `None`。
  """
  @functools.cache
  def ctx_factory():
    ctx = JaxprPpContext(_dropvars(jaxpr))
    pp_settings = JaxprPpSettings()
    try: pp_jaxpr(jaxpr, ctx, pp_settings)  # 对 ctx 产生副作用，构建变量名
    except: pass
    return ctx, pp_settings

  try:
    _check_jaxpr(ctx_factory, jaxpr)
  except JaxprTypeError as e:
    ctx, pp_settings = ctx_factory()
    if len(e.args) == 2:
      msg, eqnidx = e.args
      jaxpr_str = str(pp_jaxpr_eqn_range(jaxpr, eqnidx - 10, eqnidx + 10, ctx,
                                         pp_settings))
    else:
      msg, = e.args
      jaxpr_str = str(pp_jaxpr_eqn_range(jaxpr, 0, 20, ctx, pp_settings))
    msg = "\n\n".join([msg, "while checking jaxpr:", jaxpr_str])
    raise JaxprTypeError(msg) from None

  # 在验证 jaxpr 之后运行键复用检查器：
  if config.debug_key_reuse.value:
    # 在此处导入以避免循环导入
    from jax.experimental.key_reuse._core import check_key_reuse_jaxpr  # pyrefly: ignore[missing-import]
    check_key_reuse_jaxpr(jaxpr)

@partial(weakref_lru_cache, trace_context_in_key=False)
def _dropvars(jaxpr: Jaxpr) -> dict[Var, Literal_['_']]:
  varnames: dict[Var, Literal_['_']] = {}
  used: set[Var] = {atom for atom in jaxpr.outvars if isinstance(atom, Var)}
  for eqn in jaxpr.eqns[::-1]:
    for v in eqn.outvars:
      if not v in used:
        varnames[v] = '_'
    used.update(atom for atom in eqn.invars if isinstance(atom, Var))
  return varnames


def _check_jaxpr(
    ctx_factory: Callable[[], tuple[JaxprPpContext, JaxprPpSettings]],
    jaxpr: Jaxpr
  ) -> None:
  env: dict[Var, Atom] = {}

  def read(x: Atom) -> Atom:
    # 检查类型标注本身是良类型的。
    check_type(ctx_factory, env, x.aval)
    if isinstance(x, Var):
      # 检查变量在作用域内且类型一致。
      if x not in env:
        ctx, _ = ctx_factory()
        raise JaxprTypeError(f"Variable '{x.pretty_print(ctx)}' not defined")
      return env[x]
    elif isinstance(x, Literal):
      # 检查字面量与其类型标注匹配。
      if not typecheck(x.aval, x.val):
        ctx, _ = ctx_factory()
        raise JaxprTypeError(
            f"Literal value {x.val} does not match its type annotation "
            f"{pp_aval(x.aval, ctx)}")
      return x
    else:
      assert False, "syntactically invalid jaxpr"

  def write(v: Var, aval: AbstractValue) -> None:
    assert isinstance(v, Var), "syntactically invalid jaxpr"
    # 检查绑定者的类型标注本身是良类型的。
    check_type(ctx_factory, env, v.aval)
    # 检查该变量尚未被绑定。
    if v in env:
      ctx, _ = ctx_factory()
      raise JaxprTypeError(f"Variable '{v.pretty_print(ctx)}' already bound")
    # 检查计算出的类型与绑定者标注一致。
    if not typematch(v.aval, aval):
      ctx, _ = ctx_factory()
      raise JaxprTypeError(
          f"Value for variable '{v.pretty_print(ctx)}' inconsistently typed "
          f"as {pp_aval(aval, ctx)} for let-binder of type {pp_aval(v.aval, ctx)}")

    # 如果变量不是 DropVar，则将其加入环境。
    if not isinstance(v, DropVar):
      env[v] = v

  # # 不要返回 ref
  if config.mutable_array_checks.value:
    from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
    for v in jaxpr.outvars:
      if isinstance(v.aval, AbstractRef):
        raise JaxprTypeError("returned a ref!")

  # 检查 lambda 绑定者上的类型标注。
  for v in it.chain(jaxpr.constvars, jaxpr.invars):
    check_type(ctx_factory, env, v.aval)
    write(v, v.aval)

  # 检查每个方程。
  input_vars = set(it.chain(jaxpr.constvars, jaxpr.invars))
  mut_arrays = set()
  for eqn_idx, eqn in enumerate(jaxpr.eqns):
    prim = eqn.primitive
    try:
      in_atoms = map(read, eqn.invars)
      in_avals = [x.aval for x in in_atoms]  # 对动态 shape 使用 in_atoms

      # 计算该原语应用的类型。
      with eqn.ctx.manager:
        if prim in custom_typechecks:
          out_type, eqn_effects = custom_typechecks[prim](
            ctx_factory, *in_atoms, **eqn.params)
        else:
          out_type, eqn_effects = check_eqn(prim, in_avals, eqn.params)

      # 检查计算出的效果类型与方程的标注匹配，并且
      # 包含在 jaxpr 的标注中。
      if prim.ref_primitive:
        if prim.ref_allocating:
          outvar, = eqn.outvars
          mut_arrays.add(outvar)
      eqn_effects = resolve_input_effects(eqn_effects, eqn.invars)
      if eqn.effects != eqn_effects:
        raise JaxprTypeError("Inferred effects do not match equation effects. "
                             f"Equation effects: {eqn.effects}. "
                             f"Inferred effects: {eqn_effects}")
      for eff in eqn.effects:
        if isinstance(eff, effects.JaxprInputEffect):
          if eff.input in mut_arrays:
            continue
          if eff.input not in input_vars:
            raise JaxprTypeError(
                "Invalid `JaxprInputEffect`: must correspond to a jaxpr invar")
          if eff not in jaxpr.effects:
            raise JaxprTypeError(
                "Invalid `JaxprInputEffect`: must be present in jaxpr. "
                f"{eff} is not in {jaxpr.effects}.")
        elif isinstance(eff, NamedAxisEffect):
          # 原语可以合法地消解（discharge）命名轴效果。
          continue
        elif eff not in jaxpr.effects:
          raise JaxprTypeError("Equation effect not present in jaxpr effects. "
                               f"Equation effect: {eff}. "
                               f"Jaxpr effects: {jaxpr.effects}")

      # 检查 out_type 与 let 绑定者的标注匹配（替换之后）。
      foreach(write, eqn.outvars, out_type)

    except JaxprTypeError as e:
      ctx, settings = ctx_factory()
      msg, = e.args
      src = source_info_util.summarize(eqn.source_info)
      msg = "\n\n".join([msg, "in equation:", str(pp.nest(2, pp_eqn(eqn, ctx, settings))),
                         f"from source: {src}"])
      raise JaxprTypeError(msg, eqn_idx) from None

  # 检查没有输出 ref
  # TODO(mattjj): 改进这条错误信息
  if config.mutable_array_checks.value:
    from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
    for v in jaxpr.outvars:
      if isinstance(v.aval, AbstractRef): raise TypeError("returned ref")

  # TODO(mattjj): 在 jaxpr 上包含输出类型标注并在此处检查它
  foreach(read, jaxpr.outvars)

def check_type(
    ctx_factory: Callable[[], tuple[JaxprPpContext, JaxprPpSettings]],
    env: dict[Var, Atom],
    ty: AbstractValue,
  ) -> None:
  return  # 除上述情况外，所有语法形式都是有效的

def check_eqn(prim, in_avals, params):
  for jaxpr in jaxprs_in_params(params):
    check_jaxpr(jaxpr)

  out_avals, effects = prim.abstract_eval(*in_avals, **params)
  if not prim.multiple_results:
    out_avals = [out_avals]
  return out_avals, effects

def _check_call(ctx_factory, prim, in_atoms, params):
  if "call_jaxpr" not in params:
    raise JaxprTypeError(
        f"Call primitive {prim} missing 'call_jaxpr' parameter")
  call_jaxpr = params["call_jaxpr"]
  if len(in_atoms) != len(call_jaxpr.invars):
    raise JaxprTypeError(f"Call primitive {prim} with {len(in_atoms)} "
                         f"operands cannot call jaxpr with "
                         f"{len(call_jaxpr.invars)} inputs")

  # 检查 `call_jaxpr` 是否可以应用到 in_atoms。
  env: dict[Var, Atom] = {}
  for v, x in zip(call_jaxpr.invars, in_atoms):
    if not typecompat(v.aval, x.aval):
      # TODO(mattjj): 由于 Var.__repr__，错误消息中的变量令人困惑
      raise JaxprTypeError(f"Call primitive {prim} passes operand {x} of type "
                           f"{x.aval} to jaxpr expecting type "
                           f"{v.aval}")
    env[v] = x.val if type(x) is Literal else x

  check_jaxpr(call_jaxpr)

  out_avals = [x.aval for x in call_jaxpr.outvars]
  return out_avals, positional_effects(call_jaxpr)

def eqn_effects(jaxpr, invars) -> Effects:
  return resolve_input_effects(positional_effects(jaxpr), invars)

def subst_input_effects(effs, env) -> Effects:
  return {e.replace(env.get(e.input, e.input))
          if isinstance(e, effects.JaxprInputEffect) else e for e in effs}

def positional_effects(jaxpr) -> Effects:
  if not any(isinstance(e, effects.JaxprInputEffect) for e in jaxpr.effects):
    return jaxpr.effects
  idx = {v: i for i, v in enumerate(jaxpr.invars)}
  out_effs = set()
  for eff in jaxpr.effects:
    if isinstance(eff, effects.JaxprInputEffect):
      i = idx.get(eff.input)
      if i is None:
        continue
      eff = eff.replace(i)
    out_effs.add(eff)
  return out_effs


# ------------------- ShapeDtypeStruct -------------------

def _check_sharding(sharding, shape):
  if sharding is None:
    return
  if isinstance(sharding, P):
    sharding._check_compatible_wrt_shape(shape)
  else:
    sharding.check_compatible_aval(shape)

@set_module("jax")
class ShapeDtypeStruct:
  """用于保存数组的形状、dtype 以及其他静态属性的容器。

  ``ShapeDtypeStruct`` 通常与 :func:`jax.eval_shape` 配合使用。

  Args:
    shape: 表示数组形状的整数序列
    dtype: 一个类似 dtype 的对象
    sharding: （可选）一个 :class:`jax.Sharding` 对象
  """
  __slots__ = ["shape", "dtype", "_sharding", "_dll", "weak_type",
               "manual_axis_type", "is_ref", "_memory_space"]

  shape: Any
  dtype: Any
  _sharding: Any
  _dll: Any
  weak_type: Any
  manual_axis_type: Any
  is_ref: Any
  _memory_space: Any

  def __init__(self, shape, dtype, *, sharding=None, weak_type=False,
               manual_axis_type=None, is_ref=False, _memory_space=None):
    shape = tuple(shape)
    if any(s is None for s in shape):
      raise ValueError('`shape` passed to `ShapeDtypeStruct` cannot have '
                       f'None in it. Got {shape=}')
    object.__setattr__(self, 'shape', shape)
    if dtype is None:
      raise ValueError("ShapeDtypeStruct: dtype must be specified.")
    dtype = dtype if dtypes.issubdtype(dtype, dtypes.extended) else np.dtype(dtype)
    object.__setattr__(self, 'dtype', dtype)
    if sharding is not None and not isinstance(sharding, (Sharding, Format, P)):
      raise ValueError(
          "sharding should be an instance of `jax.sharding.Sharding`, "
          "`jax.sharding.PartitionSpec` or"
          f" `jax.experimental.layout.Format`. Got {sharding} of type"
          f" {type(sharding)}.")
    if (isinstance(sharding, Format) and
        isinstance(sharding.layout, AutoLayoutSingleton)):
      raise TypeError(
          "`Layout.AUTO` cannot be used in place of a device-local"
          f" layout in a `ShapeDtypeStruct`. Got {sharding}")
    object.__setattr__(
      self, '_dll', sharding.layout if isinstance(sharding, Format) else None)
    object.__setattr__(
      self, '_sharding', sharding.sharding if isinstance(sharding, Format) else sharding)
    _check_sharding(self._sharding, self.shape)
    object.__setattr__(self, 'weak_type', weak_type)
    if (manual_axis_type is not None
        and not isinstance(manual_axis_type, ManualAxisType)):
      raise TypeError(
          "`manual_axis_type` argument passed to ShapeDtypeStruct should be of"
          " type `jax.sharding.ManualAxisType`. Got type"
          f" {type(manual_axis_type)}")
    object.__setattr__(self, 'manual_axis_type', manual_axis_type)
    object.__setattr__(self, 'is_ref', is_ref)
    object.__setattr__(self, '_memory_space', _memory_space)

  def __setattr__(self, name, value):
    if hasattr(self, name):
      if getattr(self, name) == value:
        # 这可能发生在两个线程竞争时，例如两个线程
        # 试图对同一个 SDS 实例求哈希。
        return
      raise RuntimeError(
          f"Cannot reassign attributes ({name}) of immutable ShapeDtypeStruct"
          " objects")
    super().__setattr__(name, value)

  size = property(lambda self: math.prod(self.shape))
  ndim = property(lambda self: len(self.shape))

  @classmethod
  def like(cls, x) -> ShapeDtypeStruct:
    from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
    from jax._src.array import ArrayImpl  # pyrefly: ignore[missing-import]
    from jax._src.hijax import HiType  # pyrefly: ignore[missing-import]
    aval = shaped_abstractify(x)
    if isinstance(aval, AbstractRef):
      raise NotImplementedError(
          "Ref/AbstractRef support is not implemented. Please file an issue at"
          " https://github.com/jax-ml/jax/issues")
    if isinstance(aval, HiType):
      raise NotImplementedError(
          "Passing a HiVal to `ShapeDtypeStruct.like` is not implemented.")
    aval_s = None if aval.sharding.mesh.empty else aval.sharding
    if getattr(x, '_committed', True):
      sharding = (x.format if isinstance(x, ArrayImpl) else
                  getattr(x, 'sharding', aval_s))
    else:
      sharding = None
    mat = None if aval.mat.empty else aval.mat
    return cls(aval.shape, aval.dtype, sharding=sharding,
               weak_type=aval.weak_type, manual_axis_type=mat, is_ref=False,
               _memory_space=aval.memory_space)

  @property
  def sharding(self):
    if isinstance(self._sharding, P):
      # TODO(yashkatariya): 也许可以在这里使用 `get_abstract_mesh()`，但要根据
      # `core.trace_state_clean()` 来切换？
      cur_mesh = mesh_lib.get_concrete_mesh()
      if cur_mesh.empty:
        raise TypeError(
            "When specifying PartitionSpec to `ShapeDtypeStruct`, the context"
            " mesh cannot be empty. Please use `jax.set_mesh` to set"
            " the mesh context.")
      return NamedSharding(cur_mesh, self._sharding)
    else:
      return self._sharding

  @property
  def format(self):
    return Format(self._dll, self.sharding)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError as e:
      raise TypeError("len() of unsized object") from e  # 与 numpy 的报错相同

  def __repr__(self):
    sh = f", sharding={self.sharding}" if self.sharding is not None else ""
    l = f", format={self._dll}" if self._dll is not None else ""
    wt = f", weak_type={self.weak_type}" if self.weak_type else ""
    mat = (f", manual_axis_type={self.manual_axis_type}"
           if self.manual_axis_type else "")
    is_ref = f", is_ref={self.is_ref}" if self.is_ref else ""
    return (f"{type(self).__name__}(shape={self.shape}, "
            f"dtype={self.dtype.name}{sh}{l}{wt}{mat}{is_ref})")

  __str__ = __repr__

  def __eq__(self, other):
    if not isinstance(other, ShapeDtypeStruct):
      return False
    else:
      return ((self.shape, self.dtype, self.sharding, self._dll,
               self.weak_type, self.manual_axis_type, self.is_ref) ==
              (other.shape, other.dtype, other.sharding, other._dll,
               other.weak_type, other.manual_axis_type, other.is_ref))

  def __hash__(self):
    return hash((self.shape, self.dtype, self.sharding, self._dll,
                 self.weak_type, self.manual_axis_type, self.is_ref))

  def update(self, **kwargs):
    if 'sharding' in kwargs:
      s = kwargs['sharding']
      if self._dll is not None and isinstance(s, Sharding):
        raise ValueError(
            f"You are updating ShapeDtypeStruct with a {type(s)} when the"
            f" original ShapeDtypeStruct had a concrete layout {self.format}."
            " This might lead to bugs. If you want to do this, create a new"
            " ShapeDtypeStruct via the constructor.")
      sharding = s
    else:
      sharding = self.format
    return ShapeDtypeStruct(
        shape=kwargs.pop('shape', self.shape),
        dtype=kwargs.pop('dtype', self.dtype),
        sharding=sharding,
        weak_type=kwargs.pop('weak_type', self.weak_type),
        manual_axis_type=kwargs.pop('manual_axis_type', self.manual_axis_type),
        is_ref=kwargs.pop('is_ref', self.is_ref),
        _memory_space=kwargs.pop('_memory_space', self._memory_space))


def _sds_aval_mapping(x):
  dtype = dtypes.check_and_canonicalize_user_dtype(
      x.dtype, "ShapeDtypeStruct", allow_non_jax_dtypes=True)
  memory_space = getattr(x, "_memory_space", None) or MemorySpace.Device
  aval = ShapedArray(x.shape, dtype, weak_type=x.weak_type,
                     memory_space=memory_space)
  aval = update_aval_with_sharding(aval, x.sharding, mat=x.manual_axis_type)
  if x.is_ref:
    from jax._src.state.types import AbstractRef  # pyrefly: ignore[missing-import]
    return AbstractRef(aval)
  return aval

def _canonicalize_sds(sds):
  dt = dtypes.check_and_canonicalize_user_dtype(
      sds.dtype, "ShapeDtypeStruct", allow_non_jax_dtypes=True)
  if dt == sds.dtype:
    return sds
  return sds.update(dtype=dt)

pytype_aval_mappings[ShapeDtypeStruct] = _sds_aval_mapping
dtypes.register_canonicalize_value_handler(ShapeDtypeStruct, _canonicalize_sds)

# ------------------- Jaxpr 打印表示 -------------------

def pp_toplevel_jaxpr(jaxpr_to_print: Jaxpr, *,
                      source_info: bool = False,
                      print_shapes: bool = True,
                      custom_pp_eqn_rules : bool = True,
                      name_stack: bool = False,
                      print_effects: bool = False) -> pp.Doc:
    context = JaxprPpContext(_dropvars(jaxpr_to_print))
    settings = JaxprPpSettings(
        source_info=source_info,
        print_shapes=print_shapes,
        custom_pp_eqn_rules=custom_pp_eqn_rules,
        name_stack=name_stack,
        print_effects=print_effects)

    # 统计每个 jaxpr 被使用的次数。
    names = defaultdict[Jaxpr, str](lambda: "jaxpr")
    jaxpr_counts = Counter[Jaxpr]()
    s = deque([jaxpr_to_print])
    while s:
      jaxpr = s.popleft()
      jaxpr_counts[jaxpr] += 1
      if jaxpr is not jaxpr_to_print and len(jaxpr.eqns) > 10:
        jaxpr_counts[jaxpr] += 1
      for eqn in jaxpr.eqns:
        # TODO(slebedev): 为 name= 想出一个更精细的启发式规则。
        name = eqn.params.get("name")
        if name is None:
          s.extend(jaxprs_in_params(eqn.params))
          continue
        name = name.strip("<>")  # <lambda> -> lambda
        for subjaxpr in jaxprs_in_params(eqn.params):
          s.append(subjaxpr)
          names.setdefault(subjaxpr, name)

    # 把出现多次的 jaxpr 提到顶层，并确保
    # 它们的名称是唯一的。
    name_counts = Counter[str]()
    shared = []
    for jaxpr, c in jaxpr_counts.items():
      if c == 1:
        continue
      name = names[jaxpr]
      if (count := name_counts[name]) > 0:
        name_counts[name] += 1
        name += str(count)
        name_counts[name] += 1
      else:
        name_counts[name] += 1
      context.shared_jaxpr_names.add(name)
      context.shared_jaxprs[jaxpr] = name
      shared.append((name, jaxpr))

    docs = []
    for name, jaxpr in shared:
      docs.append(pp_shared_jaxpr(name, jaxpr, context, settings))
    docs.append(pp_jaxpr(jaxpr_to_print, context, settings))
    return pp.concat(docs)


class JaxprPpSettings(NamedTuple):
  print_shapes: bool = True
  source_info: bool = False
  name_stack: bool = False
  custom_pp_eqn_rules: bool = True
  print_effects: bool = False

def _encode_digits_alphabetic(n: int) -> str:
  if n == -1:
    return '*'
  s = ''
  while len(s) == 0 or n:
    n, i = n // 26, n % 26
    s = chr(97 + i % 26) + s
  return s

# JaxprPpContext 允许在嵌套的 Jaxpr 中使用全局唯一的变量名。
class JaxprPpContext:
  var_names: defaultdict[Var, str]
  # 共享 jaxpr 指那些被多次使用、并且会最先打印的 jaxpr。
  shared_jaxprs: MutableMapping[Jaxpr, str]  # 将共享 jaxpr 映射到其名称
  shared_jaxpr_names: MutableSet[str]

  def __init__(self, var_names: dict | None = None) -> None:
    self.shared_jaxprs = {}
    self.shared_jaxpr_names = set()
    fresh_names: Iterator[str] = (
        name for i in it.count()
        if (name := _encode_digits_alphabetic(i)) not in self.shared_jaxpr_names)
    self.var_names = defaultdict(fresh_names.__next__, var_names or {})

  def suggest_same_var_names(self,
                             for_vars: Sequence[Atom],
                             like_vars: Sequence[Atom]) -> None:
    """建议 `for_vars` 的名称，使其与 `like_vars` 的名称一致。

    `for_vars` 是互不相同的 Var，并被作为 `like_vars` 的别名。
    """
    used_like_vars: set[Var] = set()
    if len(for_vars) != len(like_vars):
      # 当调用带有 subjaxpr 的原语时传入的参数个数有误，例如打印一个无效的
      # Jaxpr 时，就可能出现这种不匹配。
      return
    for for_v, like_v in zip(for_vars, like_vars):
      if (isinstance(like_v, Var) and
          like_v not in used_like_vars and
          isinstance(for_v, Var) and
          for_v not in self.var_names):
        used_like_vars.add(like_v)
        self.var_names[for_v] = like_v.pretty_print(self)


def pp_var(v: Var | Literal, context: JaxprPpContext, *,
           print_literal_dtype: bool = True,
           is_binder: bool = False) -> pp.Doc:
  name = v.pretty_print(context, print_dtype=print_literal_dtype)
  if (isinstance(v, Var) and not isinstance(v, DropVar)):
    if is_binder:
      return pp.text(name, anchor=f"v_{name}")
    else:
      return pp.text(name, href=f"#v_{name}")
  return pp.text(name)

def pp_aval(a: AbstractValue, context: JaxprPpContext) -> str:
  return a.str_short(short_dtypes=True)

def pp_vars(vs: Sequence[Atom], context: JaxprPpContext,
            *, separator="", print_shapes: bool = False,
            is_binder: bool = False) -> pp.Doc:
  if print_shapes:
    return pp.nest(2, pp.group(
      pp.join(pp.text(separator) + pp.group(pp.brk()), [
        pp_var(v, context, is_binder=is_binder) +
        pp.type_annotation(pp.text(":" + pp_aval(v.aval, context)))
        for v in vs
      ])
    ))
  else:
    return pp.nest(2, pp.group(
      pp.join(pp.text(separator) + pp.group(pp.brk()),
              [pp_var(v, context, is_binder=is_binder) for v in vs])
    ))

def pp_kv_pair(k:str, v: Any, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  if type(v) is tuple and all(isinstance(j, Jaxpr) for j in v):
    pp_v = pp_jaxprs(v, context, settings)
  elif isinstance(v, Jaxpr):
    pp_v = pp_jaxpr(v, context, settings)
  elif isinstance(v, frozenset):
    pp_v = pp.text(f"frozenset({{{', '.join(repr(e) for e in sorted(v))}}})")
  else:
    s = str(v)
    s = re.sub(
      r' at 0x([0-9a-fA-F]+)', lambda m: ' at 0x' + 'X' * len(m.group(1)), s)
    pp_v = pp.text(s)
  return pp.text(f'{k}=') + pp_v

def pp_kv_pairs(kv_pairs, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  if not kv_pairs:
    return pp.nil()
  return pp.group(pp.concat([
    pp.nest(2, pp.concat([
      pp.text("["),  pp.brk(""),
      pp.join(pp.brk(), [pp_kv_pair(k, v, context, settings) for k, v in kv_pairs])
    ])),
    pp.brk(""), pp.text("]")
  ]))

def pp_eqn(eqn: JaxprEqn, context: JaxprPpContext, settings: JaxprPpSettings
           ) -> pp.Doc:
  rule = (_pp_eqn if not settings.custom_pp_eqn_rules else
          pp_eqn_rules.get(eqn.primitive, _pp_eqn))
  doc = rule(eqn, context, settings)
  return (doc if eqn.source_info.traceback is None
          else pp.source_map(doc, eqn.source_info.traceback))

def _pp_eqn(eqn: JaxprEqn, context: JaxprPpContext, settings: JaxprPpSettings,
            params: Sequence[str] | None = None) -> pp.Doc:
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  if params is None:
    params = sorted(eqn.params)
  name_stack_annotation = f'[{eqn.source_info.name_stack}]' if settings.name_stack else None
  lhs = pp_vars(eqn.outvars, context, print_shapes=settings.print_shapes,
                is_binder=True)
  rhs = [pp.text(eqn.primitive.name, annotation=name_stack_annotation),
         pp_kv_pairs([(p, eqn.params[p]) for p in params], context, settings),
         pp.text(" ") + pp_vars(eqn.invars, context)]
  if eqn.outvars:
    return pp.concat([lhs, pp.text(" = ", annotation=annotation), *rhs])
  else:
    return pp.concat(rhs)
CustomPpEqnRule = Callable[[JaxprEqn, JaxprPpContext, JaxprPpSettings], pp.Doc]
pp_eqn_rules: dict[Primitive, CustomPpEqnRule] = {}

def pp_eqns(eqns: Sequence[JaxprEqn],
            context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  return pp.join(
    pp.brk("; "),
    [pp_eqn(e, context, settings) for e in eqns])

def pp_jaxpr_skeleton(jaxpr: Jaxpr, eqns_fn, context: JaxprPpContext,
                      settings: JaxprPpSettings) -> pp.Doc:
  constvars = pp_vars(jaxpr.constvars, context,
                      print_shapes=settings.print_shapes, is_binder=True)
  invars = pp_vars(jaxpr.invars, context, print_shapes=settings.print_shapes,
                   is_binder=True)
  eqns = eqns_fn()
  outvars = pp.concat([
    pp.text("("), pp_vars(jaxpr.outvars, context, separator=","),
    pp.text(")" if len(jaxpr.outvars) != 1 else ",)")])
  if settings.print_effects:
    # TODO(sharadmv): 在此渲染完整的签名
    eff_text = [pp.text(" : { ")]
    for i, eff in enumerate(jaxpr.effects):
      if i > 0:
        eff_text.append(pp.text(", "))
      eff_text.append(pp_effect(eff, context))
    eff_text.append(pp.text(" }"))
  else:
    eff_text = []
  return pp.group(pp.nest(2, pp.concat([
    pp.text("{ "), pp.keyword(pp.text("lambda ")),
    constvars, pp.text("; "), invars,
    pp.text(". "), pp.keyword(pp.text("let")),
    pp.nest(2, pp.brk() + eqns), pp.brk(),
    pp.keyword(pp.text("in ")), outvars,
    pp.concat(eff_text)
  ])) + pp.text(" }"))


def pp_shared_jaxpr(
    name: str,
    jaxpr: Jaxpr,
    context: JaxprPpContext,
    settings: JaxprPpSettings,
) -> pp.Doc:
  eqns_fn = lambda: pp_eqns(jaxpr.eqns, context, settings)
  pp_skeleton = pp_jaxpr_skeleton(jaxpr, eqns_fn, context, settings)
  return pp.concat([
      pp.text("let "),
      pp.text(name, anchor=f"g_{name}"),
      pp.text(" = "),
      pp_skeleton,
      pp.text(" in"),
      pp.brk(),
  ])


def pp_jaxpr(
    jaxpr: Jaxpr,
    context: JaxprPpContext,
    settings: JaxprPpSettings,
) -> pp.Doc:
  if name := context.shared_jaxprs.get(jaxpr):
    return pp.text(name, href=f"#g_{name}")
  eqns_fn = lambda: pp_eqns(jaxpr.eqns, context, settings)
  return pp_jaxpr_skeleton(jaxpr, eqns_fn, context, settings)


def pp_jaxprs(
    jaxprs: Sequence[Jaxpr], context: JaxprPpContext, settings: JaxprPpSettings
) -> pp.Doc:
  return pp.group(pp.concat([pp.nest(2, pp.concat([
      pp.text('('), pp.brk(""),
      pp.join(pp.brk(), map(lambda x: pp_jaxpr(x, context, settings), jaxprs))]
    )), pp.brk(""), pp.text(')')])
  )


def pp_jaxpr_eqn_range(jaxpr: Jaxpr, lo: int, hi: int, context: JaxprPpContext,
                       settings: JaxprPpSettings) -> pp.Doc:
  lo = max(lo, 0)
  hi = max(lo, min(hi, len(jaxpr.eqns)))
  eqns = jaxpr.eqns[lo:hi]
  def eqns_fn():
    pps = []
    if len(eqns) == 0 and len(jaxpr.eqns) != 0:
      pps.append(pp.text('...'))
    else:
      if lo != 0:
        pps.append(pp.text('...'))
      pps.extend(map((lambda e: pp_eqn(e, context, settings)), eqns))
      if hi != len(jaxpr.eqns):
        pps.append(pp.text('...'))
    return pp.join(pp.brk("; "), pps)
  return pp_jaxpr_skeleton(jaxpr, eqns_fn, context, settings)

def pp_effect(effect: Effect, context: JaxprPpContext) -> pp.Doc:
  if hasattr(effect, "_pretty_print"):
    return effect._pretty_print(context)
  return pp.text(str(effect))

# ------------------- Jaxpr 工具 -------------------

def last_used(jaxpr: Jaxpr) -> dict[Var, JaxprEqn | None]:
  """返回一个从 jaxpr 中每个变量到最后一个使用它的方程的映射。"""
  last_used: dict[Var, JaxprEqn | None] = {
      v: None for v in jaxpr.outvars if not isinstance(v, Literal)}
  for eqn in reversed(jaxpr.eqns):
    for v in eqn.invars:
      if not isinstance(v, Literal) and v not in last_used:
        last_used[v] = eqn
  return last_used

def clean_up_dead_vars(eqn: JaxprEqn, env: dict[Var, Any],
                       last_used: dict[Var, JaxprEqn | None]):
  """若 eqn 是这些变量最后一次被使用的地方，则从 env 中移除所有 eqn.invars。"""
  for v in {v for v in eqn.invars if not isinstance(v, Literal)}:
    if last_used[v] is eqn:
      # 当变量不再被后续方程需要时，删除对它的引用。
      del env[v]

# 在 shard_map 中用于转换 aval
shard_aval_handlers = {}
unshard_aval_handlers = {}

def shard_aval(mesh, manual_axes, check_vma, spec, aval: AbstractValue
               ) -> AbstractValue:
  from jax._src.hijax import HiType  # pyrefly: ignore[missing-import]
  if isinstance(aval, HiType):
    return aval.shard(mesh, manual_axes, check_vma, spec)
  if (handler := shard_aval_handlers.get(type(aval))):
    return handler(mesh, manual_axes, check_vma, spec, aval)
  raise NotImplementedError(f"Unsupported aval type: {type(aval)}")

def unshard_aval(mesh, check_vma, spec, aval: AbstractValue
                 ) -> AbstractValue:
  from jax._src.hijax import HiType  # pyrefly: ignore[missing-import]
  if isinstance(aval, HiType):
    return aval.unshard(mesh, check_vma, spec)
  if (handler := unshard_aval_handlers.get(type(aval))):
    return handler(mesh, check_vma, spec, aval)
  raise NotImplementedError(f"Unsupported aval type: {type(aval)}")


# ----------------- 用于查询追踪上下文的外部 API -----------------

# TODO(dougalm, jakevdp): 通过 jax.extend 暴露这些接口

# 用于检查 JAX 的追踪状态是否发生变化的可比较对象。
class OpaqueTraceState:
  def __init__(self, trace_ref):
    self._trace_ref = trace_ref

  def __eq__(self, other):
    if isinstance(other, OpaqueTraceState):
      return self._trace_ref == other._trace_ref
    else:
      return False

def get_opaque_trace_state(convention=None):
  del convention
  assert trace_ctx.trace is not None
  return OpaqueTraceState(trace_ctx.trace._weakref)

def nonempty_axis_env() -> bool:
  return bool(trace_ctx.axis_env.axis_sizes)

def unsafe_am_i_under_a_jit() -> bool:
  return 'DynamicJaxprTrace' in str(unsafe_get_trace_stack(trace_ctx.trace))

def unsafe_am_i_under_a_vmap() -> bool:
  return 'BatchTrace' in str(unsafe_get_trace_stack(trace_ctx.trace))

# TODO(douglam): 弃用/删除
def find_top_trace(_):
  return unsafe_get_current_trace()


def unsafe_get_current_trace():
  return trace_ctx.trace

def unsafe_get_trace_stack(trace):
  if hasattr(trace, "parent_trace"):
    return unsafe_get_trace_stack(trace.parent_trace) + [trace]
  else:
    return [trace]

def unsafe_get_axis_names() -> list[Any]:
  return list(trace_ctx.axis_env.axis_sizes)

# TODO(douglam): 弃用/删除
def axis_frame(axis_name):
  return trace_ctx.axis_env.axis_size(axis_name)
