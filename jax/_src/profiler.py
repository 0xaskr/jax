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

# 文件职责：实现 JAX 运行时性能分析（profiling）的 Python 前端，负责启动/停止分析器
# 服务器与追踪，并把 CPU/GPU/TPU 的执行追踪（含 Python 函数与 JAX 设备端操作）导出到
# TensorBoard/Perfetto 日志目录，是 `jax.profiler` 公共 API 的底层实现。
# 关键概念：`ProfileOptions` 用于配置采集器，`_ProfileState` 保存单例分析会话状态，
# `TraceAnnotation`/`StepTraceAnnotation`/`annotate_function` 用于给代码打标记；设备内存
# 分析（pprof 格式）与 PGLE 的 FDO profile 采集也在此实现。

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
import gzip
import http.server
import json
import logging
import os
import pathlib
import socketserver
import threading
from typing import Any
import warnings

from jax._src import traceback_util
traceback_util.register_exclusion(__file__)

from jax._src import xla_bridge
from jax._src.lib import _profiler
from jax._src.lib import _profile_data
from jax import version as jax_version_module
from jax._src.lib import version as version_lib

ProfileData = _profile_data.ProfileData
ProfileEvent = _profile_data.ProfileEvent
ProfilePlane = _profile_data.ProfilePlane

_profiler_server: _profiler.ProfilerServer | None = None

logger = logging.getLogger(__name__)


class ProfileOptions(_profiler.ProfileOptions):
  """用于配置分析器采集器的分析器选项。"""


def start_server(
    port: int, requires_backend: bool = True
) -> _profiler.ProfilerServer:
  """在 `port` 端口上启动分析器服务器。

  使用 `TensorBoard <https://www.tensorflow.org/tensorboard>`_ 2.2 或更新
  版本中的“TensorFlow profiler”功能，你可以连接到该分析器服务器，
  并采样执行追踪，
  这些追踪会展示 CPU、GPU 和/或 TPU 设备活动。

  Args:
    port: 用于启动分析器服务器的端口。
    requires_backend: 若为 False，分析器服务器在启动前不会等待后端
      初始化。默认为 True。
  """
  global _profiler_server
  if _profiler_server is not None:
    raise ValueError("Only one profiler server can be active at a time.")

  # 确保在创建分析会话之前后端已初始化。
  # 否则在 Cloud TPU 上，libtpu 可能还未在创建追踪器之前完成初始化，
  # 这会导致 TPU 追踪器初始化失败，
  # 并且分析结果中不会包含任何 TPU 操作。
  # NOTE(skyewm): 我不确定 start_server 是否也需要这样做（start_trace 肯定
  # 需要），但为了保险起见还是放在这里。
  if requires_backend:
    xla_bridge.get_backend()

  _profiler_server = _profiler.start_server(port)
  return _profiler_server


def stop_server():
  """停止正在运行的分析器服务器。"""
  global _profiler_server
  if _profiler_server is None:
    raise ValueError("No active profiler server.")
  _profiler_server = None # 应当会销毁该分析器服务器


def register_subprocess(pid: int, port: int) -> Callable[[], None]:
  """注册某个子进程的分析器服务器，使其与当前进程一起被分析。

  当当前进程收集分析数据时（无论是通过编程方式还是经由其分析器服务器），
  它都会把该请求传播到所有已注册子进程的分析器服务器，
  随后把所有响应聚合到本进程分析器服务器返回的
  主响应中。

  当在独立进程中运行工作进程、且这些进程可能影响主进程性能时
  （例如 PyGrain），这很有用。

  NOTE: 目前仅支持对子进程进行 CPU 分析。

  Args:
    pid: 子进程的进程 ID。
    port: 子进程中分析器服务器的端口。

  Returns:
    一个函数，调用它或对其做垃圾回收时，会把该子进程从主进程的分析器中
    注销。

  Raises:
    RuntimeError: 若子进程注册失败（例如已注册、
    无法连接等）。
  """
  return _profiler.register_subprocess(pid, port)


class _ProfileState:
  def __init__(self):
    self.profile_session = None
    self.log_dir: str | None = None
    self.create_perfetto_link = False
    self.create_perfetto_trace = False
    self.lock = threading.Lock()

  def reset(self):
    self.profile_session = None
    self.create_perfetto_link = False
    self.create_perfetto_trace = False
    self.log_dir = None


_profile_state = _ProfileState()


def set_metadata(key: str, value: str) -> None:
  """为当前分析会话设置元数据。"""
  if hasattr(_profiler, "set_metadata"):
    return _profiler.set_metadata(key, value)


def clear_metadata() -> None:
  """清除当前分析会话的元数据。"""
  if hasattr(_profiler, "clear_metadata"):
    return _profiler.clear_metadata()


def start_trace(
    log_dir: os.PathLike | str,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
    profiler_options: ProfileOptions | None = None,
) -> None:
  """启动一次分析器追踪。

  该追踪会捕获 CPU、GPU 和/或 TPU 活动，
  包括 Python 函数和 JAX 设备端操作。
  使用 :func:`stop_trace` 结束追踪
  并把结果保存到 ``log_dir``。

  生成的追踪可以用 TensorBoard 查看。注意收集追踪时不需要 TensorBoard
  处于运行状态。

  同一时间只能收集一次追踪。若在另一个追踪运行期间调用
  :func:`start_trace`，将抛出 RuntimeError。

  Args:
    log_dir: 保存分析器追踪的目录（通常是
      TensorBoard 日志目录）。
    create_perfetto_link: 布尔值，若为 true，则创建并打印指向
      Perfetto 追踪查看器 UI（https://ui.perfetto.dev）的链接。程序会
      阻塞，直到该链接被打开并且 Perfetto 加载完追踪。
    create_perfetto_trace: 布尔值，若为 true，则额外导出一个可与
      Perfetto 追踪查看器 UI（https://ui.perfetto.dev）上传兼容的
      ``perfetto_trace.json.gz`` 文件。若 ``create_perfetto_link`` 为 true
      也会生成该文件。如果你想生成与 Perfetto 兼容的追踪而又不想阻塞进程，
      这会很有用。
    profiler_options: 用于配置分析器采集行为的分析器选项。
  """
  with _profile_state.lock:
    if _profile_state.profile_session is not None:
      raise RuntimeError("Profile has already been started. "
                         "Only one profile may be run at a time.")
    clear_metadata()
    # 确保在创建分析会话之前后端已初始化。
    # 否则在 Cloud TPU 上，libtpu 可能还未在创建追踪器之前完成初始化，
    # 这会导致 TPU 追踪器初始化失败，
    # 并且分析结果中不会包含任何 TPU 操作。
    xla_bridge.get_backend()

    options = profiler_options
    if options is None:
      options = ProfileOptions()
    set_metadata("jax_version", jax_version_module.__version__)
    jaxlib_version_str = ".".join(map(str, version_lib))
    set_metadata("jaxlib_version", jaxlib_version_str)
    for backend_name in xla_bridge.backends():
      try:
        backend = xla_bridge.get_backend(backend_name)
        set_metadata(f"{backend.platform}_version", backend.platform_version)
      except RuntimeError:
        pass
    _profile_state.profile_session = _profiler.ProfilerSession(options)
    _profile_state.create_perfetto_link = create_perfetto_link
    _profile_state.create_perfetto_trace = (
        create_perfetto_trace or create_perfetto_link)
    _profile_state.log_dir = str(log_dir)


def _write_perfetto_trace_file(log_dir: os.PathLike | str):
  # 进入包含最新追踪转储的文件夹，以找到 `trace.json.jz`
  trace_folders = (pathlib.Path(log_dir).absolute() / "plugins" / "profile").iterdir()
  latest_trace_folder = max(trace_folders, key=os.path.getmtime)
  trace_jsons = latest_trace_folder.glob("*.trace.json.gz")
  try:
    trace_json, = trace_jsons
  except ValueError as value_error:
    raise ValueError(f"Invalid trace folder: {latest_trace_folder}") from value_error

  logger.info("Loading trace.json.gz and removing its metadata...")
  # Perfetto 不喜欢 `trace.json` 中的 `metadata` 字段，所以我们
  # 把它移除。
  # TODO(sharadmv): 通过更新生成的 `trace.json` 使其尽可能不包含元数据，
  # 来加快这一步。
  with gzip.open(trace_json, "rb") as fp:
    trace = json.load(fp)
    del trace["metadata"]
  perfetto_trace = latest_trace_folder / "perfetto_trace.json.gz"
  logger.info("Writing perfetto_trace.json.gz...")
  with gzip.open(perfetto_trace, "w") as fp:
    fp.write(json.dumps(trace).encode("utf-8"))
  return perfetto_trace

class _PerfettoServer(http.server.SimpleHTTPRequestHandler):
  """处理来自 `ui.perfetto.dev` 对 `trace.json` 的请求。"""

  def end_headers(self):
    self.send_header('Access-Control-Allow-Origin', '*')
    return super().end_headers()

  def do_GET(self):
    self.server.last_request = self.path  # pyrefly: ignore[missing-attribute]
    return super().do_GET()

  def do_POST(self):
    self.send_error(404, "File not found")

def _host_perfetto_trace_file(path: os.PathLike | str):
  # ui.perfetto.dev 会在 `127.0.0.1:9001` 上查找托管的文件。我们搭建一个
  # TCP 服务器来托管 `perfetto_trace.json.gz` 文件。
  port = 9001
  orig_directory = pathlib.Path.cwd()
  directory, filename = os.path.split(path)
  try:
    os.chdir(directory)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('127.0.0.1', port), _PerfettoServer) as httpd:
      url = f"https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{port}/{filename}"
      print(f"Open URL in browser: {url}")

      # 一旦 ui.perfetto.dev 从这个服务器获取到 trace.json，我们就可以
      # 把它关掉。
      while httpd.__dict__.get('last_request') != '/' + filename:
        httpd.handle_request()
  finally:
    os.chdir(orig_directory)

def stop_trace():
  """停止当前正在运行的分析器追踪。

  追踪会被保存到对应的 :func:`start_trace` 调用所传入的 ``log_dir``。
  若尚未启动任何追踪，则抛出 RuntimeError。
  """
  with _profile_state.lock:
    profile_session = _profile_state.profile_session
    if profile_session is None:
      raise RuntimeError("No profile started")
    profile_session.stop_and_export(str(_profile_state.log_dir))
    if _profile_state.create_perfetto_trace:
      abs_filename = _write_perfetto_trace_file(str(_profile_state.log_dir))
      if _profile_state.create_perfetto_link:
        _host_perfetto_trace_file(abs_filename)
    _profile_state.reset()
    clear_metadata()


def stop_and_get_fdo_profile() -> bytes | str:
  """停止当前正在运行的分析器追踪并导出 fdo_profile。

  目前仅支持 GPU。
  若尚未启动任何追踪，则抛出 RuntimeError。
  """
  with _profile_state.lock:
    profile_session = _profile_state.profile_session
    if profile_session is None:
      raise RuntimeError("No profile started")
    xspace = profile_session.stop()
    fdo_profile = _profiler.get_fdo_profile(xspace)
    _profile_state.reset()
    clear_metadata()
    return fdo_profile


@contextmanager
def trace(
    log_dir: os.PathLike | str,
    create_perfetto_link=False,
    create_perfetto_trace=False,
    profiler_options: ProfileOptions | None = None,
):
  """用于采集分析器追踪的上下文管理器。

  该追踪会捕获 CPU、GPU 和/或 TPU 活动，
  包括 Python 函数和 JAX 设备端操作。

  生成的追踪可以用 TensorBoard 查看。注意收集追踪时不需要 TensorBoard
  处于运行状态。

  同一时间只能收集一次追踪。若在另一个追踪运行期间启动追踪，将抛出
  RuntimeError。

  Args:
    log_dir: 保存分析器追踪的目录（通常是
      TensorBoard 日志目录）。
    create_perfetto_link: 布尔值，若为 true，则创建并打印指向
      Perfetto 追踪查看器 UI（https://ui.perfetto.dev）的链接。程序会
      阻塞，直到该链接被打开并且 Perfetto 加载完追踪。
    create_perfetto_trace: 布尔值，若为 true，则额外导出一个可与
      Perfetto 追踪查看器 UI（https://ui.perfetto.dev）上传兼容的
      ``perfetto_trace.json.gz`` 文件。若 ``create_perfetto_link`` 为 true
      也会生成该文件。如果你想生成与 Perfetto 兼容的追踪而又不想阻塞进程，
      这会很有用。
    profiler_options: 用于配置分析器采集行为的分析器选项。
  """
  start_trace(
      log_dir, create_perfetto_link, create_perfetto_trace, profiler_options
  )
  try:
    yield
  finally:
    stop_trace()


class TraceAnnotation(_profiler.TraceMe):
  """在分析器中生成一个追踪事件的上下文管理器。

  该追踪事件的时间跨度覆盖上下文所包含代码的执行时长。

  例如：

  >>> x = jnp.ones((1000, 1000))
  >>> with jax.profiler.TraceAnnotation("my_label"):
  ...   result = jnp.dot(x, x.T).block_until_ready()

  如果该事件发生在进程被追踪期间，它会使一个 "my_label" 事件出现在
  追踪时间线上。
  """


class StepTraceAnnotation(TraceAnnotation):
  """在分析器中生成一个步进追踪事件的上下文管理器。

  该步进追踪事件的时间跨度覆盖上下文所包含代码的执行时长。
  分析器会为每个步进追踪事件提供性能分析。

  例如，可以用它来标记训练步，
  让分析器能够提供逐步的性能分析：

  >>> while global_step < NUM_STEPS:                                           # doctest: +SKIP
  ...   with jax.profiler.StepTraceAnnotation("train", step_num=global_step):  # doctest: +SKIP
  ...     train_step()                                                         # doctest: +SKIP
  ...     global_step += 1                                                     # doctest: +SKIP

  如果该事件发生在进程被 TensorBoard 追踪期间，
  它会使一个 "train xx" 事件出现在追踪时间线上。此外，如果使用
  加速器，设备追踪时间线上也会显示一个 "train xx" 事件。
  注意 "step_num" 可以作为关键字参数传入，
  以便把全局步号传递给分析器。

  """

  def __init__(self, name: str, **kwargs):
    super().__init__(name, _r=1, **kwargs)


def annotate_function(func: Callable, name: str | None = None,
                      **decorator_kwargs):
  """为函数执行生成追踪事件的装饰器。

  例如：

  >>> @jax.profiler.annotate_function
  ... def f(x):
  ...   return jnp.dot(x, x.T).block_until_ready()
  >>>
  >>> result = f(jnp.ones((1000, 1000)))

  如果函数执行发生在进程被 TensorBoard 追踪期间，它会使一个 "f" 事件
  出现在追踪时间线上。

  可以通过 :py:func:`functools.partial` 给该装饰器传递参数。

  >>> from functools import partial

  >>> @partial(jax.profiler.annotate_function, name="event_name")
  ... def f(x):
  ...   return jnp.dot(x, x.T).block_until_ready()

  >>> result = f(jnp.ones((1000, 1000)))
  """

  name = name or getattr(func, '__qualname__', None)
  name = name or func.__name__
  @wraps(func)
  def wrapper(*args, **kwargs):
    with TraceAnnotation(name, **decorator_kwargs):
      return func(*args, **kwargs)
  return wrapper


def device_memory_profile(backend: str | None = None) -> bytes:
  """以 ``pprof`` 格式的 protocol buffer 捕获 JAX 设备内存分析数据。

  设备内存分析是内存状态的一份快照，
  它描述了内存中存在的 JAX :class:`~jax.Array` 与可执行对象，
  以及它们各自的分配位置。

  关于如何使用设备内存分析器的更多信息，请参见
  :doc:`/device_memory_profiling`。

  该分析系统通过插桩 JAX 的设备端分配来工作，会为每次分配捕获一份
  Python 栈回溯。插桩始终处于启用状态；:func:`device_memory_profile`
  提供了用于捕获它的 API。

  :func:`device_memory_profile` 的输出是一个二进制 protocol buffer，
  可以用 `pprof 工具
  <https://github.com/google/pprof>`_ 解释和可视化。

  Args:
    backend: 可选；应为其收集设备内存分析数据的
      JAX 后端名称。

  Returns:
    一个包含二进制 `pprof` 格式 protocol buffer 的字节串。
  """
  client = xla_bridge.get_backend(backend)
  return gzip.compress(client.heap_profile())


def save_device_memory_profile(filename, backend: str | None = None) -> None:
  """收集设备内存分析数据并把它写入文件。

  :func:`save_device_memory_profile` 是 :func:`device_memory_profile` 的
  便捷包装器，它把输出保存到 ``filename``。更多信息请参见
  :func:`device_memory_profile` 的文档。

  Args:
    filename: 分析数据应写入的文件名。
    backend: 可选；应为其收集设备内存分析数据的
      JAX 后端名称。
  """
  profile = device_memory_profile(backend)
  with open(filename, "wb") as f:
    f.write(profile)


# 允许以分析器运行模型给定次数。在达到所需的
# 重试次数后，客户端就可以收集 FDO 数据。
class PGLEProfiler:

  def __init__(self, retries: int, percentile: int):
    self.retries: int = retries
    self.percentile: int = percentile
    self.collected_fdo: bytes | None = None
    self.called_times: int = 0
    self.fdo_profiles: list[Any] = []
    self.current_session: _profiler.ProfilerSession | None = None

  def consume_fdo_profile(self) -> bytes | None:
    if self.collected_fdo is not None:
      return self.collected_fdo

    if not self.is_enabled() or self.called_times != self.retries:
      return None

    self.collected_fdo = _profiler.aggregate_profiled_instructions(
        self.fdo_profiles, self.percentile
    )
    return self.collected_fdo

  def is_fdo_consumed(self):
    return self.collected_fdo is not None

  def disable(self):
    self.retries = 0

  def is_enabled(self):
    return self.retries > 0

  def is_running(self):
    return self.current_session is not None

  @classmethod
  @contextmanager
  def trace(cls, runner: PGLEProfiler | None):
    if (runner is None or runner.is_running()
        or not runner.is_enabled() or runner.is_fdo_consumed()):
      yield
    else:
      options = _profiler.ProfileOptions()
      options.enable_hlo_proto = True
      options.raise_error_on_start_failure = True
      runner.current_session = _profiler.ProfilerSession(options)

      try:
        yield
      finally:
        xspace = runner.current_session.stop()
        runner.fdo_profiles.append(
            _profiler.get_fdo_profile(xspace)
        )
        runner.current_session = None

        runner.called_times += 1
        if runner.fdo_profiles[-1] == b'':
          warnings.warn(
              "PGLE collected an empty trace, may be due to contention with "
              "another tool that subscribes to CUPTI, such as Nsight Systems - check "
              "for CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED from XLA. "
              "Consider populating a persistent compilation cache with PGLE enabled, "
              "and then profiling a second run that has the "
              "JAX_COMPILATION_CACHE_EXPECT_PGLE option enabled.",
              RuntimeWarning)
