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

# 文件职责：集中管理 JAX 的日志配置，包括全局日志级别与按模块的调试日志。
# 它同时设置 Python 侧 `jax`/`jaxlib` logger 的级别与处理器，
# 并通过 `jax._src.lib.utils.absl_set_min_log_level` 同步 C++ 运行时级别。
# `update_logging_level_global` 面向全局级别配置项，
# `update_debug_log_modules` 则为指定模块开启逐模块 debug 输出到 stderr。
# 这些函数由 JAX 配置项（如 `jax_debug_log_modules`）在值变化时回调调用。

import logging
import sys
from jax._src.lib import utils

# 日志消息示例：
# DEBUG:2023-06-07 00:14:40,280:jax._src.xla_bridge:590: Initializing backend 'cpu'
logging_formatter = logging.Formatter(
    "{levelname}:{asctime}:{name}:{lineno}: {message}", style='{')

_logging_level_set: dict[str, int] = {}

_jax_logger_handler = logging.StreamHandler(sys.stderr)
_jax_logger_handler.setFormatter(logging_formatter)

_nameToLevel = {
    'CRITICAL': logging.CRITICAL,
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}

_tf_cpp_map = {
    'CRITICAL': 3,
    'FATAL': 3,
    'ERROR': 2,
    'WARN': 1,
    'WARNING': 1,
    'INFO': 0,
    'DEBUG': 0,
}

def _set_cpp_min_log_level(logging_level: str | None = None):
  if logging_level in (None, "NOTSET"):
    return
  # 只要级别不是 NOTSET，就设置 C++ 运行时的日志级别
  if logging_level not in _tf_cpp_map:
    raise ValueError(f"Attempting to set log level \"{logging_level}\" which"
                      f" isn't one of the supported:"
                      f" {list(_tf_cpp_map.keys())}.")
  # 配置 C++ 日志级别 0 - debug，1 - info，2 - warning，3 - error
  log_level = _tf_cpp_map[logging_level]
  utils.absl_set_min_log_level(log_level)

def update_logging_level_global(logging_level: str | None) -> None:
  # 移除此前的处理器
  for logger_name, level in _logging_level_set.items():
    logger = logging.getLogger(logger_name)
    logger.removeHandler(_jax_logger_handler)
    logger.setLevel(level)
  _logging_level_set.clear()
  _set_cpp_min_log_level(logging_level)

  if logging_level is None:
    return

  logging_level_num = _nameToLevel[logging_level]

  # 更新 jax 与 jaxlib 根 logger 以支持传播
  root_loggers = [logging.getLogger("jax"), logging.getLogger("jaxlib")]
  for logger in root_loggers:
    logger.setLevel(logging_level_num)
    if logging_level_num != logging.NOTSET:
      logger.addHandler(_jax_logger_handler)
    _logging_level_set[logger.name] = logger.level

# 按模块的调试日志

_jax_logger = logging.getLogger("jax")

class _DebugHandlerFilter(logging.Filter):
  def filter(self, record):
    del record  # 未使用。
    return _jax_logger.level > logging.DEBUG

_debug_handler = logging.StreamHandler(sys.stderr)
_debug_handler.setLevel(logging.DEBUG)
_debug_handler.setFormatter(logging_formatter)
_debug_handler.addFilter(_DebugHandlerFilter())

_debug_enabled_loggers = []

def _enable_debug_logging(logger_name):
  """让指定的 logger 把所有内容都记录到 stderr。

  同时为日志消息添加更有用的调试信息，例如时间。

  Args:
    logger_name: logger 的名称，例如 "jax._src.xla_bridge"。
  """
  logger = logging.getLogger(logger_name)
  _debug_enabled_loggers.append((logger, logger.level))

  logger.addHandler(_debug_handler)
  logger.setLevel(logging.DEBUG)


def _disable_all_debug_logging():
  """停用所有通过 `enable_debug_logging` 启用的调试日志。

  默认的日志行为仍然生效，即 WARNING 及以上级别
  会以不带额外消息格式的方式记录到 stderr。
  """
  for logger, prev_level in _debug_enabled_loggers:
    logger: logging.Logger
    logger.removeHandler(_debug_handler)
    logger.setLevel(prev_level)
  _debug_enabled_loggers.clear()

def update_debug_log_modules(module_names_str: str | None):
  _disable_all_debug_logging()
  if not module_names_str:
    return
  module_names = module_names_str.split(',')
  for module_name in module_names:
    _enable_debug_logging(module_name)
