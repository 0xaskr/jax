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

"""用于对代码进行插桩的工具。

可以把代码中的某些位置标记为具名事件。程序执行过程中每次到达某个事件时，
已注册的监听器都会被调用。

监听器回调的典型用途是把事件发送给指标收集器，以便聚合或导出。
"""

# 文件职责：实现 JAX 内部轻量的事件与指标埋点（instrumentation）机制。
# 调用方在代码路径上埋下具名事件，通过 `record_event`、
# `record_event_duration_secs`、`record_event_time_span` 与 `record_scalar`
# 上报事件、耗时、时间区间和标量摘要；模块内维护四类监听器列表，
# 由 `register_*` / `unregister_*` 增删、`get_*_listeners` 读取，
# 监听器通常把数据转发给外部指标收集器，用于监控与性能分析。
from __future__ import annotations

from typing import Protocol


class EventListenerWithMetadata(Protocol):

  def __call__(self, event: str, **kwargs: str | int) -> None:
    ...


class EventDurationListenerWithMetadata(Protocol):

  def __call__(self, event: str, duration_secs: float,
               **kwargs: str | int) -> None:
    ...


class EventTimeSpanListenerWithMetadata(Protocol):

  def __call__(
      self, event: str, start_time: float, end_time: float, **kwargs: str | int
  ) -> None:
    ...

class ScalarListenerWithMetadata(Protocol):

  def __call__(
      self, event: str, value: float | int, **kwargs: str | int,
  ) -> None:
    ...


_event_listeners: list[EventListenerWithMetadata] = []
_event_duration_secs_listeners: list[EventDurationListenerWithMetadata] = []
_event_time_span_listeners: list[EventTimeSpanListenerWithMetadata] = []
_scalar_listeners: list[ScalarListenerWithMetadata] = []


def record_event(event: str, **kwargs: str | int) -> None:
  """记录一个事件。

  若指定了 **kwargs，那么对同一事件的所有调用中，这些具名参数
  都必须以相同的顺序传入。
  """
  for callback in _event_listeners:
    callback(event, **kwargs)


def record_event_duration_secs(event: str, duration: float,
                               **kwargs: str | int) -> None:
  """以秒（float）记录一个事件的持续时间。

  若指定了 **kwargs，那么对同一事件的所有调用中，这些具名参数
  都必须以相同的顺序传入。
  """
  for callback in _event_duration_secs_listeners:
    callback(event, duration, **kwargs)


def record_event_time_span(
    event: str, start_time: float, end_time: float, **kwargs: str | int
) -> None:
  """以秒（float）记录一个事件的开始与结束时间。"""
  for callback in _event_time_span_listeners:
    callback(event, start_time, end_time, **kwargs)


def record_scalar(
    event: str, value: float | int, **kwargs: str | int
) -> None:
  """记录一个标量摘要值。"""
  for callback in _scalar_listeners:
    callback(event, value, **kwargs)


def register_event_listener(
    callback: EventListenerWithMetadata,
) -> None:
  """注册一个在 record_event() 期间被调用的回调。"""
  _event_listeners.append(callback)


def register_event_time_span_listener(
    callback: EventTimeSpanListenerWithMetadata,
) -> None:
  """注册一个在 record_event_time_span() 期间被调用的回调。"""
  _event_time_span_listeners.append(callback)


def register_event_duration_secs_listener(
    callback : EventDurationListenerWithMetadata) -> None:
  """注册一个在 record_event_duration_secs() 期间被调用的回调。"""
  _event_duration_secs_listeners.append(callback)


def register_scalar_listener(
    callback : ScalarListenerWithMetadata,
) -> None:
  """注册一个在 record_scalar() 期间被调用的回调。"""
  _scalar_listeners.append(callback)


def get_event_duration_listeners() -> list[EventDurationListenerWithMetadata]:
  """获取事件持续时间监听器。"""
  return list(_event_duration_secs_listeners)


def get_event_time_span_listeners() -> list[EventTimeSpanListenerWithMetadata]:
  """获取事件时间区间监听器。"""
  return list(_event_time_span_listeners)


def get_event_listeners() -> list[EventListenerWithMetadata]:
  """获取事件监听器。"""
  return list(_event_listeners)


def get_scalar_listeners() -> list[ScalarListenerWithMetadata]:
  """获取标量事件监听器。"""
  return list(_scalar_listeners)


def clear_event_listeners():
  """清空事件监听器。"""
  global _event_listeners, _event_duration_secs_listeners, _event_time_span_listeners
  _event_listeners = []
  _event_duration_secs_listeners = []
  _event_time_span_listeners = []
  _scalar_listeners = []


def unregister_event_duration_listener(
    callback: EventDurationListenerWithMetadata,
) -> None:
  """按回调注销一个事件持续时间监听器。"""
  assert callback in _event_duration_secs_listeners
  _event_duration_secs_listeners.remove(callback)


def unregister_event_time_span_listener(
    callback: EventTimeSpanListenerWithMetadata,
) -> None:
  """按回调注销一个事件时间区间监听器。"""
  assert callback in _event_time_span_listeners
  _event_time_span_listeners.remove(callback)


def unregister_event_listener(
    callback: EventListenerWithMetadata,
) -> None:
  """按回调注销一个事件监听器。"""
  assert callback in _event_listeners
  _event_listeners.remove(callback)


def unregister_scalar_listener(
    callback: ScalarListenerWithMetadata,
) -> None:
  """按回调注销一个标量事件监听器。"""
  assert callback in _scalar_listeners
  _scalar_listeners.remove(callback)
