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
# 文件职责：提供自动微分（AD）所需的基础抽象，主要是各种“零”的表示与求和/截断梯度原语。
# 其中 `Zero` 是 jaxpr 内部使用的抽象零，只携带抽象值（aval），
# 而 `SymbolicZero` 是面向用户的符号零，属性访问会转发给其 aval。
# 还定义 `add_jaxvals`（对应原语 `add_any`）用于把两个值相加，
# 以及 `stop_gradient` 原语，并给出由原值/抽象值构造零切向量、
# 零余切向量的辅助函数，供线性化与转置等 AD 规则使用。
from __future__ import annotations

from collections.abc import Callable
import types
from typing import Any

from jax._src import core
from jax._src.core import typeof
from jax._src import traceback_util
from jax._src.core import Primitive
from jax._src.tree_util import register_pytree_node, tree_map
from jax._src.typing import Array, ArrayLike
from jax._src.util import safe_map

traceback_util.register_exclusion(__file__)

map = safe_map

def add_jaxvals(x: ArrayLike, y: ArrayLike) -> Array:
  from jax._src.hijax import HiType  # pyrefly: ignore[missing-import]
  ty = typeof(x)
  if isinstance(ty, HiType):
    return ty.vspace_add(x, y)
  x, y = core.auto_insert_reshard(x, y)
  return add_jaxvals_p.bind(x, y)

add_jaxvals_p = Primitive('add_any')
add_any_p = add_jaxvals_p

@add_jaxvals_p.def_impl
def add_impl(x, y):
  return raw_jaxval_adders[type(x)](x, y)
raw_jaxval_adders = {}

@add_jaxvals_p.def_abstract_eval
def add_abstract(x, y):
  assert core.typematch(x, y), (x, y)
  return x

def zeros_like_aval(aval: core.AbstractValue) -> Array:
  from jax._src.hijax import HiType  # pyrefly: ignore[missing-import]
  if isinstance(aval, HiType):
    return aval.vspace_zero()
  return aval_zeros_likers[type(aval)](aval)
aval_zeros_likers: dict[type, Callable[[Any], Array]] = {}

def zeros_like_jaxval(val):
  return zeros_like_aval(core.typeof(val))

def empty_like_aval(aval):
  from jax._src.hijax import HiType  # pyrefly: ignore[missing-import]
  if isinstance(aval, HiType):
    return aval.raise_val(*map(empty_like_aval, aval.lo_ty()))
  return aval_empty_likers[type(aval)](aval)
aval_empty_likers: dict[type, Callable[[Any], Array]] = {}

def instantiate(z: Zero | Array) -> Array:
  if isinstance(z, Zero):
    return zeros_like_aval(z.aval)
  return z


class Zero:
  __slots__ = ['aval']
  def __init__(self, aval: core.AbstractValue):
    self.aval = aval
  def __repr__(self) -> str:
    return f'Zero({self.aval})'
  def instantiate(self):
    return zeros_like_aval(self.aval)

register_pytree_node(Zero, lambda z: ((), z.aval), lambda aval, _: Zero(aval))

def p2tz(primal_value):
  return Zero(typeof(primal_value).to_tangent_aval())

def p2cz(primal_value):
  return Zero(typeof(primal_value).to_ct_aval())

def a2tz(primal_aval):
  return Zero(primal_aval.to_tangent_aval())


def _stop_gradient_impl[T](x: T) -> T:
  if not core.valid_jaxtype(x):
    raise TypeError("stop_gradient only works on valid JAX arrays, but "
                    f"input argument is: {x}")
  return x

stop_gradient_p : Primitive = Primitive('stop_gradient')
stop_gradient_p.def_impl(_stop_gradient_impl)
stop_gradient_p.def_abstract_eval(lambda x: x)


# `Zero` 的面向用户版本
class SymbolicZero:
  def __init__(self, aval: core.AbstractValue) -> None:
    self.aval = aval

  def __repr__(self) -> str:
    return self.__class__.__name__

  # TODO(mattjj,frostig): 这里把属性查找转发给 self.aval 委托对象；
  # 应与做同样事情的 core.Tracer.__getattr__ 去重
  def __getattr__(self, name):
    # 若 aval 属性抛出 AttributeError，会在这里被捕获
    try:
      attr = getattr(self.aval, name)
    except KeyError as err:
      raise AttributeError(
          f"{self.__class__.__name__} has no attribute {name}"
      ) from err
    else:
      t = type(attr)
      if t is core.aval_property:
        return attr.fget(self)
      elif t is core.aval_method:
        return types.MethodType(attr.fun, self)
      else:
        return attr

  @staticmethod
  def from_primal_value(val: Any) -> SymbolicZero:
    return SymbolicZero(typeof(val).to_tangent_aval())

def zero_from_primal(val, symbolic_zeros=False):
  def f(x):
    t_aval = typeof(x).to_tangent_aval()
    return SymbolicZero(t_aval) if symbolic_zeros else zeros_like_aval(t_aval)
  return tree_map(f, val)


JaxTypeOrTracer = Any

def replace_internal_symbolic_zeros(
    x: JaxTypeOrTracer | Zero) -> JaxTypeOrTracer | SymbolicZero:
  return SymbolicZero(x.aval) if type(x) is Zero else x

def replace_rule_output_symbolic_zeros(
    x: JaxTypeOrTracer | SymbolicZero) -> JaxTypeOrTracer | Zero:
  return Zero(x.aval) if type(x) is SymbolicZero else x


# TODO(mattjj): 在修复依赖这些的调用方之后移除它们
zeros_like_p: Primitive = Primitive('zeros_like')
