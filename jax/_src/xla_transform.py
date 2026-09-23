# Copyright 2026 The JAX Authors.
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
# 文件职责：实现 `jax.extend.xla` 中注册/清除自定义 XLA 编译器 pass 的内部机制。
# 用户经 `register_hlo_module_transformation` 注册回调，在 XLA 编译的指定流水线阶段
# （调度前或调度后）接收序列化后的 `HloModuleProto` 字节，并可返回修改后的字节或 None。
# 同一阶段注册的多个回调会按注册顺序排队执行；平台参数可取 cpu、tpu 等，为 None 时
# 对所有已知后端生效，并分别走非 PJRT 的 C API 路径或 PJRT 插件路径进行注册。

"""内部实现：`jax.extend.xla` 的编译器 pass 注册。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import enum

from jax._src import xla_bridge
from jax._src.lib import _jax
from jax._src.lib import _xla


class PipelineStage(enum.Enum):
  PRE_SCHEDULER = 0
  POST_SCHEDULER = 1


def _normalize_platforms(
    platforms: Sequence[str] | str | None,
) -> list[str]:
  """把 platforms 规范化为平台字符串列表。"""
  if platforms is None:
    return list(xla_bridge.backends().keys())
  if isinstance(platforms, str):
    return [platforms]
  return list(set(platforms))


def register_hlo_module_transformation(
    callback: Callable[[bytes], bytes | None],
    *,
    name: str,
    stage: PipelineStage = PipelineStage.PRE_SCHEDULER,
    platforms: Sequence[str] | str | None = None,
) -> None:
  """注册一个用于变换 HLO 模块的自定义编译器 pass。

  注册的 pass 会在 XLA 编译的指定流水线阶段被调用。回调接收序列化后的
  ``HloModuleProto`` 字节，并按以下两种方式之一返回结果：

  - 若模块被修改，返回修改后的序列化 ``HloModuleProto`` 字节。
  - 若未做任何修改，返回 ``None``。

  在同一阶段多次注册（使用不同回调）时，这些注册会加入队列，
  并按注册的顺序依次被调用。

  Args:
    callback: 一个 ``(bytes) -> bytes | None`` 函数，接收序列化的
      HloModuleProto，并可选地返回修改后的版本。
    name: 编译器 pass 的名称。
    stage: pass 运行的流水线阶段，必须是
      ``PipelineStage`` 枚举值。
    platforms: 为哪些平台注册该 pass（例如 ``"cpu"``、
      ``"tpu"``）。若为 ``None``，默认对所有已知后端注册。
      可以是单个平台字符串，也可以是字符串序列。
  """
  # 无条件触发后端初始化，以便在访问各 PJRT 插件之前
  # 先把它们全部加载好。
  xla_bridge.backends()

  for platform in _normalize_platforms(platforms):
    if platform == "cpu":
      # 直接注册 CPU，因为它是唯一非 PJRT 的 C API 后端。
      _xla.register_xla_transform(name, stage.value, callback)
      continue
    c_api = _jax.get_pjrt_plugin(platform)
    _xla.register_xla_transform_c_api(c_api, name, stage.value, callback)


def clear_hlo_module_transformation(
    name: str,
    stage: PipelineStage = PipelineStage.PRE_SCHEDULER,
    platforms: Sequence[str] | str | None = None,
) -> bool:
  """清除一个已注册的自定义编译器 pass。

  Args:
    name: 要清除的编译器 pass 名称。
    stage: pass 所在的流水线阶段，必须是 ``PipelineStage`` 枚举值。
    platforms: 要为哪些平台清除该 pass。若为 ``None``，
      则对所有已知后端清除该 pass。

  Returns:
    若找到并清除了该 pass 则返回 True，否则返回 False。
  """
  # 无条件触发后端初始化，以便在访问各 PJRT 插件之前
  # 先把它们全部加载好。
  xla_bridge.backends()

  cleared = False
  for platform in _normalize_platforms(platforms):
    if platform == "cpu":
      # 直接清除 CPU，因为它是唯一非 PJRT 的 C API 后端。
      cleared |= _xla.clear_xla_transform(name, stage.value)
      continue
    c_api = _jax.get_pjrt_plugin(platform)
    cleared |= _xla.clear_xla_transform_c_api(c_api, name, stage.value)

  return cleared
