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

# 文件职责：提供线程安全的 Python 警告捕获与断言工具，供 JAX 测试使用。
# 由于标准库 `warnings` 模块（截至 Python 3.13）并非线程安全，
# `catch_warnings()` 天然存在竞态，这里改用自定义 `showwarning` 钩子。
# 每个线程维护一份处理器栈，从而提供自己的警告过滤与记录机制。
# 主要接口：`raise_on_warnings`、`record_warnings`、`ignore_warning`
# 三个上下文管理器，以及安装钩子的 `install_threadsafe_warning_handlers`。

# 用于捕获与测试警告的线程安全工具。
#
# Python 的 `warnings` 模块（至少到 Python 3.13 为止）不是线程安全的。
# `catch_warnings()` 这一特性天然存在竞态，参见
# https://py-free-threading.github.io/porting/#the-warnings-module-is-not-thread-safe
#
# 本模块提供了一种线程安全地捕获并记录警告的方式。我们向 Python 的
# `warnings` 模块安装一个自定义的 showwarning 钩子，然后依赖
# CPython 的 `warnings` 模块调用我们自己的显示警告函数。接着我们用它
# 来构造自己的线程安全警告过滤工具。

import contextlib
import re
import threading
import warnings


class _WarningContext(threading.local):
  "保存警告处理器列表的线程局部状态。"

  def __init__(self):
    self.handlers = []


_context = _WarningContext()


# 回调函数：按相反顺序应用各处理器。若没有处理器匹配，
# 我们就抛出错误。
def _showwarning(message, category, filename, lineno, file=None, line=None):
  for handler in reversed(_context.handlers):
    if handler(message, category, filename, lineno, file, line):
      return
  raise category(message)


@contextlib.contextmanager
def raise_on_warnings():
  "在出现警告时抛出异常的上下文管理器。"
  if warnings.showwarning is not _showwarning:
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      yield
    return

  def handler(message, category, filename, lineno, file=None, line=None):
    raise category(message)

  _context.handlers.append(handler)
  try:
    yield
  finally:
    _context.handlers.pop()


@contextlib.contextmanager
def record_warnings():
  "产出所抛出警告列表的上下文管理器。"
  if warnings.showwarning is not _showwarning:
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      yield w
    return

  log = []

  def handler(message, category, filename, lineno, file=None, line=None):
    log.append(warnings.WarningMessage(message, category, filename, lineno, file, line))
    return True

  _context.handlers.append(handler)
  try:
    yield log
  finally:
    _context.handlers.pop()


@contextlib.contextmanager
def ignore_warning(*, message: str | None = None, category: type = Warning):
  "忽略所有匹配警告的上下文管理器。"
  if warnings.showwarning is not _showwarning:
    with warnings.catch_warnings():
      warnings.filterwarnings(
        "ignore", message="" if message is None else message, category=category)
      yield
    return

  if message:
    message_re = re.compile(message)
  else:
    message_re = None

  category_cls = category

  def handler(message, category, filename, lineno, file=None, line=None):
    text = str(message) if isinstance(message, Warning) else message
    if (message_re is None or message_re.match(text)) and issubclass(
        category, category_cls
    ):
      return True
    return False

  _context.handlers.append(handler)
  try:
    yield
  finally:
    _context.handlers.pop()


def install_threadsafe_warning_handlers():
  # 挂接 showwarning 方法。`warnings` 模块明确指出
  # 这是一个允许用户替换的函数。
  warnings.showwarning = _showwarning

  # 让 `warnings` 模块始终显示警告。我们通过
  # 覆盖 "showwarning" 方法来挂接，因此所有警告都必须
  # 由常规机制“显示”出来，这一点很重要。
  warnings.simplefilter("always")
