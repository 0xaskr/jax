# Copyright 2021 The JAX Authors.
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

# 文件职责：实现 JAX 的持久化编译缓存，把已编译的 XLA 可执行文件按 cache_key
# 落盘保存，使进程重启后再次运行时无需重新编译。
# 主要使用者是 jax.jit 的编译路径与各后端（TPU/GPU/CPU/neuron）；用户可通过
# `config.update("jax_compilation_cache_dir", ...)` 或 `set_cache_dir` 启用。
# 关键概念：cache_key、LRU 容量上限、zstd/zlib 压缩、条目布局
# [4 字节大端编译时间]+[序列化可执行文件]，以及可选的 VerificationCache 校验。

from __future__ import annotations

from collections.abc import Sequence
import logging
import threading
from typing import Any
import warnings
import zlib

import numpy as np

# 若安装了 zstandard，则使用 zstd 压缩，否则使用 zlib。
try:
  # Python 3.14+ 中应当自带 compression.zstd
  from compression import zstd
except ImportError:
  zstd = None

if zstd is None:
  # TODO(phawkins): 当我们放弃对 Python 3.13 的支持后，移除这个分支。
  try:
    import zstandard  # pyrefly: ignore[missing-import]
  except ImportError:
    zstandard = None
else:
  zstandard = None

from jax._src import cache_key
from jax._src import config
from jax._src import monitoring
from jax._src.compilation_cache_interface import CacheInterface
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.lru_cache import LRUCache


logger = logging.getLogger(__name__)

_cache: CacheInterface | None = None

_cache_initialized: bool = False

_cache_checked: bool = False

_cache_used: bool = False

# 用于保护 _cache_initialized、_cache_checked 和 _cache_used 的互斥锁。
_cache_initialized_mutex = threading.Lock()

_UNSUPPORTED_RUNTIMES: set[str] = set()

_TIME_BYTES = 4

def is_cache_used(backend: xla_client.Client) -> bool:
  """检查缓存是否被使用，并在每个任务中只上报一次采用情况指标。
  缓存的初始化可能发生在本函数的首次调用期间。
  """
  # 若 _cache_checked 为 True，则直接返回 _cache_used。若 _cache_checked 为
  # False，则将其置为 True，上报指标，并返回缓存是否被使用。这提供了一种
  # 每个任务只上报一次指标的机制。注意，reset_cache() 也会重置
  # _cache_checked 与 _cache_used。
  global _cache_checked, _cache_used
  with _cache_initialized_mutex:
    if _cache_checked:
      return _cache_used

  with _cache_initialized_mutex:
    if not _cache_checked:
      _cache_checked = True

      # 持久化编译缓存只在 TPU 和 GPU，以及支持可执行文件
      # 序列化的后端上实现。
      # TODO(skye): 在不支持的默认平台上初始化缓存时
      # 给出警告
      supported_platforms = ["tpu", "gpu", "cpu", "neuron"]

      if not _is_cache_enabled():
        monitoring.record_event('/jax/compilation_cache/task_disabled_cache')
      elif (
          backend.platform in supported_platforms
          and getattr(backend, "supports_executable_serialization", True)
      ):
        monitoring.record_event('/jax/compilation_cache/tasks_using_cache')
        _cache_used = True
      return _cache_used

  return False


def get_file_cache(path: str) -> tuple[CacheInterface, str] | None:
  """返回文件缓存以及该缓存的路径。"""
  max_size = config.compilation_cache_max_size.value
  cache = LRUCache(path, max_size=max_size)
  if config.compilation_cache_check_contents.value:
    return VerificationCache(cache), path
  return cache, path


class CacheVerificationError(RuntimeError):
  """当刚编译出的可执行文件与编译缓存中同一键下的可执行文件不完全
  一致时抛出的错误。
  """

  def __init__(
      self,
      message: str,
      cache_key: str,
      executable_on_disk: bytes,
      executable_new: bytes,
  ):
    super().__init__(message)
    self.cache_key = cache_key
    self.executable_on_disk = executable_on_disk
    self.executable_new = executable_new


class VerificationCache(CacheInterface):
  """一个包装另一个缓存并校验其内容的缓存。

  若 jax_compilation_cache_check_contents 为 True，则本进程首次在磁盘
  缓存中遇到一个新键时，即使磁盘缓存里已经存在这样的条目，
  我们也让 get() 返回 None，从而强制重新编译。随后，当以
  编译好的可执行文件调用 put() 时，我们会校验它与磁盘上的
  内容是否一致。
  """

  def __init__(self, base_cache: CacheInterface):
    self._base_cache = base_cache
    self._verified_keys: set[str] = set()
    self.base_cache_hits: dict[str, int] = {}

  @property
  def _path(self):  # pyrefly: ignore[bad-override]
    return self._base_cache._path

  @_path.setter
  def _path(self, value):
    self._base_cache._path = value

  def get(self, key: str) -> bytes | None:
    cache_value = self._base_cache.get(key)
    if cache_value is not None:
      self.base_cache_hits[key] = self.base_cache_hits.get(key, 0) + 1

    if key not in self._verified_keys:
      # 首次遇到某个键时，强制重新编译。
      return None

    return cache_value

  def put(self, key: str, value: bytes) -> None:
    if key not in self._verified_keys:
      on_disk = self._base_cache.get(key)
      if on_disk is not None:
        # 缓存内容为 [时间戳] + [可执行文件]。
        # 我们将两者都解压后比较，比较时跳过时间戳，因为它对于刚完成的
        # 编译必然不同。
        decompressed_on_disk = decompress_executable(on_disk)
        decompressed_new = decompress_executable(value)
        executable_on_disk, _ = extract_executable_and_time(decompressed_on_disk)
        executable_new, _ = extract_executable_and_time(decompressed_new)
        if executable_on_disk != executable_new:
          raise CacheVerificationError(
              f"Persistent compilation cache inconsistency for key {key}. "
              "Executable found in the disk cache does not match the "
              "freshly compiled executable.",
              key,
              executable_on_disk,
              executable_new,
          )
      self._verified_keys.add(key)

    self._base_cache.put(key, value)

  def clear(self):
    self._verified_keys.clear()


def set_cache_dir(path) -> None:
  """设置持久化编译缓存目录。

  调用之后，jit 编译的函数会被保存到 `path`，因此当进程重启或以其他方式
  再次运行时，它们无需重新编译。这同时也告诉 Jax 在编译前到哪里查找
  已编译的函数。

  更多信息参见 :ref:`持久化编译缓存指南 <persistent-compilation-cache>`。

  .. warning::
     编译缓存被视为可信的。不要与你所不信任的用户共享
     编译缓存。例如，若你把编译缓存放在其他用户可写的目录中，
     这些用户就能触发你的 JAX 进程运行任意代码。
     共享编译缓存等同于允许任何能写入该缓存目录的人
     在你的机器上运行代码。
  """
  config.config.update("jax_compilation_cache_dir", path)


def initialize_cache(path) -> None:
  """此 API 已废弃；请改用 set_cache_dir。

  设置路径。为使其生效，应在任何对 get_executable_and_time() 和
  put_executable_and_time() 的调用之前调用它。

  更多信息参见 :ref:`持久化编译缓存指南 <persistent-compilation-cache>`。

  .. warning::
     编译缓存被视为可信的。不要与你所不信任的用户共享
     编译缓存。例如，若你把编译缓存放在其他用户可写的目录中，
     这些用户就能触发你的 JAX 进程运行任意代码。
     共享编译缓存等同于允许任何能写入该缓存目录的人
     在你的机器上运行代码。
  """
  config.config.update("jax_compilation_cache_dir", path)


def default_min_cache_entry_size() -> int:
  """返回尺寸低于多少的条目就不应被缓存的最小尺寸。"""
  return 0


def _is_cache_enabled() -> bool:
  return config.enable_compilation_cache.value


def _initialize_cache() -> None:
  # 最多尝试初始化缓存一次。
  global _cache_initialized
  with _cache_initialized_mutex:
    if _cache_initialized:
      return

    path: str | None = config.compilation_cache_dir.value
    # 若未设置路径，则不会构建缓存。
    if not path:
      return

    # 若缓存被禁用，则无需做任何事。
    if not _is_cache_enabled():
      logger.debug("_initialize_cache: cache is disabled!")
      return

    _cache_initialized = True

    # 仅当标志 --jax_persistent_cache_min_entry_size_bytes 尚未被设置时，
    # 才设置最小缓存条目尺寸。
    if config.persistent_cache_min_entry_size_bytes.value == 0:
      config.config.update("jax_persistent_cache_min_entry_size_bytes",
                           default_min_cache_entry_size())

    global _cache
    assert _cache is None, "The cache has already been initialized!"

    cache_and_path = get_file_cache(path)
    if cache_and_path is None:
      logger.debug("_initialize_cache: cache initialization failed!")
    else:
      _cache, path = cache_and_path
      logger.debug("Initialized persistent compilation cache at %s", path)

def is_persistent_cache_enabled() -> bool:
  return (config.compilation_cache_dir.value is not None
          and config.enable_compilation_cache.value)


def _get_cache(backend) -> CacheInterface | None:
  # TODO(b/289098047): 考虑把它变成公开 API，并修改 get_executable_and_time()
  # 与 put_executable_and_time() 的调用方，让它们调用 get_cache()
  # 并把结果传给它俩。
  if backend.runtime_type in _UNSUPPORTED_RUNTIMES:
    log_priority = (logging.WARNING if is_persistent_cache_enabled()
                    else logging.DEBUG)
    logger.log(log_priority, "_get_cache: Unsupported runtime: %s",
               backend.runtime_type)
    return None
  if _cache is None:
    _initialize_cache()  # 初始化最多执行一次；见上文
  return _cache


def compress_executable(executable: bytes) -> bytes:
  if zstd:
    return zstd.compress(executable)
  elif zstandard:
    compressor = zstandard.ZstdCompressor()
    return compressor.compress(executable)
  else:
    return zlib.compress(executable)

def decompress_executable(executable: bytes) -> bytes:
  if zstd:
    return zstd.decompress(executable)
  elif zstandard:
    decompressor = zstandard.ZstdDecompressor()
    return decompressor.decompress(executable)
  else:
    return zlib.decompress(executable)


def is_executable_in_cache(backend, cache_key: str) -> bool:
  """检查该可执行文件是否在缓存中。"""
  cache = _get_cache(backend)
  if cache is None:
    return False

  # TODO(patrios): 向缓存接口添加检查缓存键的方法。
  executable_and_time = cache.get(cache_key)
  return executable_and_time is not None


def get_executable_and_time(
    cache_key: str, compile_options, backend, executable_devices,
    host_callbacks: Sequence[Any] = (),
) -> tuple[xla_client.LoadedExecutable | None, int | None]:
  """若存在，则返回缓存的已编译可执行文件及其编译时间，否则返回
  None。
  """
  cache = _get_cache(backend)
  if cache is None:
    logger.debug("get_executable_and_time: cache is disabled/not initialized")
    return None, None
  executable_and_time = cache.get(cache_key)
  if executable_and_time is None:
    return None, None

  executable_and_time = decompress_executable(executable_and_time)
  serialized_executable, compile_time = extract_executable_and_time(
      executable_and_time)
  if host_callbacks:
    xla_executable_deserialized = backend.deserialize_executable(
        serialized_executable,
        executable_devices,
        compile_options,
        host_callbacks,
    )
  else:
    xla_executable_deserialized = backend.deserialize_executable(
        serialized_executable, executable_devices, compile_options
    )
  return xla_executable_deserialized, compile_time


def put_executable_and_time(
    cache_key: str,
    module_name: str,
    executable: xla_client.LoadedExecutable,
    backend,
    compile_time: int
) -> None:
  """把 'executable' 及其编译时间加入缓存，可能
  会淘汰较旧的条目。
  """
  log_priority = (logging.WARNING
                  if config.explain_cache_misses.value
                  and is_persistent_cache_enabled()
                  else logging.DEBUG)
  cache = _get_cache(backend)
  if cache is None:
    logger.log(log_priority,
               "Not writing persistent cache entry with key %r"
               " since cache is disabled/not initialized", cache_key)
    return

  serialized_executable = executable.serialize()
  executable_and_time = combine_executable_and_time(
      serialized_executable, compile_time)
  executable_and_time = compress_executable(executable_and_time)

  min_entry_size = config.persistent_cache_min_entry_size_bytes.value
  entry_size = len(executable_and_time)
  if entry_size < min_entry_size:
    logger.log(log_priority,
        "Not writing persistent cache entry with key %r since its size"
        " (%d bytes) is less than threshold (%d bytes)", cache_key, entry_size,
        min_entry_size)
  else:
    logger.log(log_priority,
               "Writing %s to persistent compilation cache with key %r",
               module_name, cache_key)
    monitoring.record_event('/jax/compilation_cache/cache_misses')
    if config.compilation_cache_expect_pgle.value:
      # 用户断言编译缓存中应当已经包含经过 PGLE 优化的可执行文件。
      # 由于尺寸/编译时间阈值的限制，预计仍会发生
      # 一些小模块的编译，但这不应导致
      # 对编译缓存的写入。
      warnings.warn(
          f"PERSISTENT CACHE WRITE with key {cache_key}, this is unexpected because "
          "JAX_COMPILATION_CACHE_EXPECT_PGLE is set. The execution that populated the "
          "cache may lack coverage, "
          "https://docs.jax.dev/en/latest/persistent_compilation_cache.html may "
          "help debug why this has happened")

    cache.put(cache_key, executable_and_time)


def get_cache_key(
    module: ir.Module,
    devices: np.ndarray,
    compile_options,
    backend,
    ignore_custom_partitioning: bool = False,
) -> str:
  return cache_key.get(
      module,
      devices,
      compile_options,
      backend,
      "zstandard" if zstandard is not None else "zlib",
      ignore_custom_partitioning,
  )


def is_initialized() -> bool:
  """
  已废弃。

  返回缓存是否启用。初始化可能被延迟，因此
  这里不检查初始化状态。保留该名称是为了
  向后兼容。
  """
  return _is_cache_enabled()


def reset_cache() -> None:
  """恢复到最初未初始化的状态。

  更多信息参见 :ref:`持久化编译缓存指南 <persistent-compilation-cache>`。
  """
  global _cache
  global _cache_initialized
  global _cache_checked
  global _cache_used
  logger.info("Resetting cache at %s.",
               _cache._path if _cache is not None else "<empty>")
  _cache = None
  with _cache_initialized_mutex:
    _cache_initialized = False
    _cache_checked = False
    _cache_used = False


def combine_executable_and_time(
    serialized_executable: bytes, compile_time: int
) -> bytes:
  """给定序列化后的可执行文件与编译时间，按下述格式生成一条缓存
  条目。

  缓存条目的形式为：
  字节:     0    1    2    3    4 ...
  内容:     编译时间    序列化后的可执行文件
            （大端整数）
  """
  return (
      int(compile_time).to_bytes(_TIME_BYTES, byteorder="big")
      + serialized_executable
  )


def extract_executable_and_time(
    executable_and_time: bytes
) -> tuple[bytes, int]:
  """给定下述格式的缓存条目，提取其中序列化后的可执行文件
  与编译时间。

  缓存条目 'executable_and_time' 的形式为：
  字节:     0    1    2    3    4 ...
  内容:     编译时间    序列化后的可执行文件
            （大端整数）
  """
  return executable_and_time[_TIME_BYTES:], int.from_bytes(
      executable_and_time[:_TIME_BYTES], byteorder='big')
