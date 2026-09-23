# Copyright 2020 The JAX Authors.
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

# 文件职责：实现 JAX 的异常回溯过滤（traceback filtering）基础设施。
# 它登记需要从回溯中剔除的 JAX 内部文件路径，并提供 filter_traceback 等
# 工具来重建只保留用户代码帧的回溯。
# api_boundary 装饰器把 JAX 变换产出的函数标记为过滤边界：当函数内部抛出
# 异常时，会在原异常上挂一条过滤后的回溯，并按 JAX_TRACEBACK_FILTERING
# 的取值改为设置 __tracebackhide__ 或追加说明等其他模式。

from __future__ import annotations

import functools
import os
import traceback
import types
from typing import cast

from jax._src import config
from jax._src import util
from jax._src.lib import _jax


_exclude_paths: list[str] = []

def register_exclusion(path: str):
  _exclude_paths.append(path)
  _jax.add_exclude_path(path)

register_exclusion(__file__)
register_exclusion(util.__file__)

_jax_message_append = (
    'The stack trace below excludes JAX-internal frames.\n'
    'The preceding is the original exception that occurred, unmodified.\n'
    '\n--------------------')

def _path_starts_with(path: str, path_prefix: str) -> bool:
  path = os.path.abspath(path)
  path_prefix = os.path.abspath(path_prefix)
  try:
    common = os.path.commonpath([path, path_prefix])
  except ValueError:
    # path 与 path_prefix 都是绝对路径，唯一会抛出
    # ValueError 的情况是它们位于不同的盘符。
    # https://docs.python.org/3/library/os.path.html#os.path.commonpath
    return False
  try:
    return common == path_prefix or os.path.samefile(common, path_prefix)
  except OSError:
    # 其中一个路径可能不存在。
    return False

def include_frame(f: types.FrameType) -> bool:
  return include_filename(f.f_code.co_filename)

def include_filename(filename: str) -> bool:
  return not any(_path_starts_with(filename, path) for path in _exclude_paths)

# 扫描堆栈回溯时，我们可能遇到 CPython 中那些不会出现在打印结果里的
# 帧，例如 importlib 某些部分的帧。我们根据源码与名称的匹配
# 启发式地忽略这些帧。
def _ignore_known_hidden_frame(f: types.FrameType) -> bool:
  return 'importlib._bootstrap' in f.f_code.co_filename

def _add_tracebackhide_to_hidden_frames(tb: types.TracebackType | None):
  if tb is None:
    return
  for f, _lineno in traceback.walk_tb(tb):
    if not include_frame(f) and not _is_reraiser_frame(f):
      f.f_locals["__tracebackhide__"] = True

def filter_traceback(tb: types.TracebackType) -> types.TracebackType | None:
  out = None
  # 扫描回溯并收集相关的帧。
  frames = list(traceback.walk_tb(tb))
  for f, lineno in reversed(frames):
    if include_frame(f):
      out = types.TracebackType(out, f, f.f_lasti, lineno)
  return out

def _add_call_stack_frames(tb: types.TracebackType) -> types.TracebackType:
  # 继续沿调用栈向上走。
  #
  # 我们希望避免向上走得太远，例如越过 IPython 这类 REPL 的
  # exec/eval 位置。为此，如果遇到了模块级帧，我们会在第一段
  # 连续的模块级帧之后停止。这是一个启发式规则，可能会在
  # REPL 边界之前就提前停下。例如，如果调用栈中包含了当前
  # 模块 A 的模块级帧，而当前模块 A 又是在别处的函数 F 中被
  # 导入的，那么我们生成的堆栈回溯就会在 F 的帧处被截断。
  out = tb

  reached_module_level = False
  for f, lineno in traceback.walk_stack(tb.tb_frame):
    if _ignore_known_hidden_frame(f):
      continue
    if reached_module_level and f.f_code.co_name != '<module>':
      break
    if include_frame(f):
      out = types.TracebackType(out, f, f.f_lasti, lineno)
    if f.f_code.co_name == '<module>':
      reached_module_level = True
  return out

def _is_reraiser_frame(f: traceback.FrameSummary | types.FrameType) -> bool:
  if isinstance(f, traceback.FrameSummary):
    filename, name = f.filename, f.name
  else:
    filename, name = f.f_code.co_filename, f.f_code.co_name
  return filename == __file__ and name == 'reraise_with_filtered_traceback'

def _is_under_reraiser(e: BaseException) -> bool:
  if e.__traceback__ is None:
    return False
  tb = traceback.extract_stack(e.__traceback__.tb_frame)
  return any(_is_reraiser_frame(f) for f in tb[:-1])

def format_exception_only(e: BaseException) -> str:
  return ''.join(traceback.format_exception_only(type(e), e)).strip()

class UnfilteredStackTrace(Exception): pass

_simplified_tb_msg = ("For simplicity, JAX has removed its internal frames from the "
                      "traceback of the following exception. Set "
                      "JAX_TRACEBACK_FILTERING=off to include these.")

class SimplifiedTraceback(Exception):
  def __str__(self):
    return _simplified_tb_msg

SimplifiedTraceback.__module__ = "jax.errors"

def _running_under_ipython() -> bool:
  """若我们看起来处于 IPython 会话中则返回 true。"""
  try:
    get_ipython()  # pyrefly: ignore[unknown-name]
    return True
  except NameError:
    return False

def _ipython_supports_tracebackhide() -> bool:
  """若该 IPython 版本支持 __tracebackhide__ 则返回 true。"""
  import IPython  # pyrefly: ignore[missing-import]
  return IPython.version_info[:2] >= (7, 17)

def _filtering_mode() -> str:
  mode = config.traceback_filtering.value
  if mode is None or mode == "auto":
    if (_running_under_ipython() and _ipython_supports_tracebackhide()):
      mode = "tracebackhide"
    else:
      mode = "quiet_remove_frames"
  return mode


# TODO(slebedev): 待 facebook/pyrefly#3329 修复后，改用 [C: Callable[..., Any]]。
def api_boundary[C](
    fun: C, *,
    repro_api_name: str | None = None,
    repro_user_func: bool = False) -> C:
  '''包装 ``fun``，使其成为过滤异常回溯的边界。

  当 ``fun`` 之下发生异常时，会为该异常附加一个自定义的 ``__cause__``，
  其中携带过滤后的回溯。该回溯模仿原始异常的堆栈回溯，但去掉了
  JAX 内部的帧。

  这种边界标注可以自我组合。最靠上的那个 :func:`~api_boundary` 对应的帧
  就是回溯过滤的起点。换句话说，如果 ``api_boundary(f)`` 直接或间接地
  调用 ``api_boundary(g)``，那么最终提供的过滤后堆栈回溯，与
  ``api_boundary(f)`` 直接调用 ``g`` 时相同。

  该标注主要用于包装 JAX 变换产出的函数。例如，设 ``g = jax.jit(f)``。
  调用 ``g`` 时会启动 JAX 的 JIT 编译机制，进而调用 ``f`` 以对其进行
  追踪与转换。如果函数 ``f`` 抛出异常，调用栈会穿过 JAX 的 JIT 内部实现
  一路展开到 ``g`` 的原始调用点。由于 :func:`~jax.jit` 返回的函数被标注为
  :func:`~api_boundary`，这样的异常会附带一条额外的回溯，其中不含
  JAX 实现特有的帧。

  "repro" 相关关键字参数见 `repro.boundary` 的注释。
  '''

  @functools.wraps(fun)  # pyrefly: ignore[bad-argument-type]
  def reraise_with_filtered_traceback(*args, **kwargs):
    __tracebackhide__ = True
    try:
      return fun(*args, **kwargs)  # pyrefly: ignore[not-callable]
    except Exception as e:
      mode = _filtering_mode()
      if _is_under_reraiser(e) or mode == "off":
        raise
      if mode == "tracebackhide":
        _add_tracebackhide_to_hidden_frames(e.__traceback__)
        raise

      tb = e.__traceback__
      if tb is None:
        raise TypeError("Traceback is None") from e
      try:
        e.with_traceback(filter_traceback(tb))
        if mode == "quiet_remove_frames":
          e.add_note("--------------------\n" + _simplified_tb_msg)
        else:
          if mode == "remove_frames":
            msg = format_exception_only(e)
            msg = f'{msg}\n\n{_jax_message_append}'
            jax_error = UnfilteredStackTrace(msg)
            jax_error.with_traceback(_add_call_stack_frames(tb))
          else:
            raise ValueError(f"JAX_TRACEBACK_FILTERING={mode} is not a valid value.")
          jax_error.__cause__ = e.__cause__
          jax_error.__context__ = e.__context__
          jax_error.__suppress_context__ = e.__suppress_context__
          e.__cause__ = jax_error
          e.__context__ = None
          del jax_error
        raise
      finally:
        del mode, tb
  if repro and (repro_api_name or repro_user_func):
    reraise_with_filtered_traceback = repro.boundary(
        reraise_with_filtered_traceback, api_name=repro_api_name,
        is_user=repro_user_func)
  return cast(C, reraise_with_filtered_traceback)

try:
  # TODO: 待最终位置确定后，从那里导入
  from jax._src import repro  # pyrefly: ignore[missing-module-attribute]
  repro_is_enabled = repro.is_enabled
except (ImportError, AttributeError):
  repro = None
  repro_is_enabled = lambda: False
