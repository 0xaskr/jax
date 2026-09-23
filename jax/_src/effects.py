# Copyright 2023 The JAX Authors.
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
# 文件职责：定义 JAX 的效果（effect）机制，用于描述带有副作用、不能被优化的计算。
# 效果依附于 JAX 原语实例与 jaxpr，带有效果的原语即使结果未被使用也不会被死代码消除。
# 其中“有序效果”会在降级后的计算里各引入一个 i1[0] 类型的 token 输入与输出，
# 并串接在各条指令之间，以保证编译器不会删除、复制或重排这些指令。
# 本模块还维护各效果类型集合（可分片、可降级、允许控制流等）与跨分派的当前 token，
# 供核心追踪、降级与 barrier 实现使用。
"""JAX 效果。

JAX 用效果来描述可能带有副作用的计算。效果与 JAX 原语实例以及
Jaxpr 相关联。

带有效果的原语实例会被保护起来，不会因为结果未被使用而被死代码消除。

一类特殊的效果是**有序**效果（`effects.ordered_effects` 的成员）。
带有序效果的计算在降级后会为每个有序效果额外增加一个输入和一个输出。
它们出现在常规输入/输出之前，类型为 `i1[0]`。这些 token
会在带有序效果的指令之间串接，以确保编译器不会消除、复制或重排
相应的指令。

为了确保跨多个计算的顺序，我们为每个线程维护一个集合，记录最近一次
分派的计算所返回的 token。每个有序效果对应一个 token，并且它可能被分片到
最近一次分派的计算所使用的设备上。在分派一个
带有序效果的新计算时，我们取出当前 token，把它分片到
将要分派的计算所在的设备上，并作为输入传入。
随后我们更新当前 token，使其指向该次
分派计算输出的 token。

在存在有序效果时，我们还用当前 token 实现
`jax.barrier`，它会等待当前 token 就绪。

对于无序效果，`jax.barrier` 的实现略有不同，
因为这类效果不会在分派计算的输入输出之间串接 token。取而代之的是
使用 `RuntimeToken`，它是分派计算时返回的对象，我们可以
在其上阻塞直到就绪。我们为每个线程保存最近一次
分派计算所返回的 `RuntimeToken`。

更多细节请参见设计说明：
https://docs.jax.dev/en/latest/jep/10657-sequencing-effects.html。
"""

from __future__ import annotations

from collections.abc import Iterable, Set
from typing import Any


class Effect:
  """一种通用副作用。"""

Effects = Set[Effect]

class JaxprInputEffect(Effect):
  """与 `JaxprEqn` 或 `Jaxpr` 的某个输入相关的副作用。

  它被用作与输入相关的效果的基类，例如对可变输入的读/写。

  在 `JaxprEqn` 或 `Jaxpr` 中，`input` 是该效果所关联输入的 `core.Var`。
  抽象求值规则的作用域中没有变量，因此在那里 `input` 是一个 int，
  即对应原语输入的位置。追踪机制在构造方程时会把位置解析为
  变量（参见 `core.resolve_input_effects`）。
  """

  def __init__(self, input: Any):
    self.input = input

  def replace(self, input: Any):
    return self.__class__(input)

  def __eq__(self, other):
    if not isinstance(other, JaxprInputEffect):
      return NotImplemented
    return self.input == other.input

  def __hash__(self):
    return hash((self.__class__, self.input))

  def __repr__(self):
    return f"{self.__class__.__name__}({self.input})"

class EffectTypeSet:

  def __init__(self):
    self._effect_types: set[type[Effect]] = set()

  def __repr__(self):
    return f"EffectTypeSet({self._effect_types})"

  def add_type(self, effect_type: type[Effect]):
    self._effect_types.add(effect_type)

  def contains(self, eff: Effect) -> bool:
    return any(isinstance(eff, eff_type) for eff_type in self._effect_types)

  def filter_in(self, effects: Iterable[Effect]) -> list[Effect]:
    return [eff for eff in effects if self.contains(eff)]

  def filter_not_in(self, effects: Iterable[Effect]) -> list[Effect]:
    return [eff for eff in effects if not self.contains(eff)]


no_effects: Effects = frozenset()
ordered_effects: EffectTypeSet = EffectTypeSet()

# 默认情况下，有序效果不允许出现在多设备计算中，
# 因为我们无法保证全序。效果可以选择性地被
# 声明为可分片，这意味着效果会按程序顺序出现，
# 但在某个程序点上我们可能会看到参与设备上的多个副作用，
# 且它们的相对顺序没有任何保证。
shardable_ordered_effects: EffectTypeSet = EffectTypeSet()

lowerable_effects: EffectTypeSet = EffectTypeSet()
control_flow_allowed_effects: EffectTypeSet = EffectTypeSet()
custom_derivatives_allowed_effects: EffectTypeSet = EffectTypeSet()
remat_allowed_effects: EffectTypeSet = EffectTypeSet()

partial_eval_kept_effects: EffectTypeSet = EffectTypeSet()
