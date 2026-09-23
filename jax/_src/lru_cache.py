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

# 文件职责：实现 JAX 编译缓存的 LRU（最近最少使用）磁盘后端。
# `LRUCache` 实现 `CacheInterface`，把编译产物按键写为缓存目录下的文件，
# 并用同名 `-atime` 附属文件记录访问时间；写入前用 `filelock` 加锁，
# 再用优先队列按访问时间淘汰最久未用的条目，使目录总大小不超过 `max_size`。
# 当 `max_size` 为 -1 时不淘汰，此时它退化为无容量限制的普通缓存；
# 路径为远程文件系统时依赖 `etils[epath]`，并使用软文件锁。

from __future__ import annotations

import heapq
import logging
import time
from typing import Any
import warnings

filelock: Any | None = None
try:
  import filelock
except ImportError:
  pass

from jax._src import path as pathlib
from jax._src.compilation_cache_interface import CacheInterface

logger = logging.getLogger(__name__)


_CACHE_SUFFIX = "-cache"
_ATIME_SUFFIX = "-atime"


def _is_local_filesystem(path: str) -> bool:
  return path.startswith("file://") or "://" not in path


class LRUCache(CacheInterface):
  """具有最近最少使用（LRU）淘汰策略的有界缓存。

  该实现包含缓存的读取、写入与淘汰，
  三者都基于 LRU 策略。

  特别地，当 ``max_size`` 被设为 -1 时缓存淘汰会被禁用，
  此时该 LRU 缓存的行为与普通缓存一致，
  不再有任何容量限制。
  """

  def __init__(self, path: str, *, max_size: int, lock_timeout_secs: float | None = 10):
    """Args:

      path: 缓存目录的路径。
      max_size: 缓存的最大字节数。若该值被设为 ``0``，
        则禁用缓存。特殊值 ``-1`` 表示不设限制，
        缓存大小可以无限增长。
      lock_timeout_secs:（可选）获取文件锁的超时时间。
    """
    if not _is_local_filesystem(path) and not pathlib.epath_installed:
      raise RuntimeError("Please install the `etils[epath]` package to specify a cache directory on a non-local filesystem")

    self.path = self._path = pathlib.Path(path)
    self.path.mkdir(parents=True, exist_ok=True)

    self.eviction_enabled = max_size != -1  # 若 `max_size` 为 -1 则不淘汰

    if self.eviction_enabled:
      if filelock is None:
        raise RuntimeError("Please install the `filelock` package to set `jax_compilation_cache_max_size`")

      self.max_size = max_size
      self.lock_timeout_secs = lock_timeout_secs

      self.lock_path = self.path / ".lockfile"
      if _is_local_filesystem(path):
        self.lock = filelock.FileLock(self.lock_path)
      else:
        self.lock = filelock.SoftFileLock(self.lock_path)

  def get(self, key: str) -> bytes | None:
    """获取给定键对应的缓存值。

    Args:
      key: 要获取缓存值所用的键。

    Returns:
      若存在则返回以字节表示的缓存数据，否则返回 ``None``。
    """
    if not key:
      raise ValueError("key cannot be empty")

    cache_path = self.path / f"{key}{_CACHE_SUFFIX}"

    if self.eviction_enabled:
      self.lock.acquire(timeout=self.lock_timeout_secs)

    try:
      if not cache_path.exists():
        logger.debug(f"Cache miss for key: {key!r}")
        return None

      logger.debug(f"Cache hit for key: {key!r}")

      val = cache_path.read_bytes()

      if self.eviction_enabled:
        timestamp = time.time_ns().to_bytes(8, "little")
        atime_path = self.path / f"{key}{_ATIME_SUFFIX}"
        atime_path.write_bytes(timestamp)

      return val

    finally:
      if self.eviction_enabled:
        self.lock.release()

  def put(self, key: str, value: bytes) -> None:
    """向缓存中添加一个新条目。

    如果已存在相同键的缓存项，则不做任何操作，
    即使其值不同。

    Args:
      key: 存储数据所使用的键。
      val: 要存储的数据。
    """
    if not key:
      raise ValueError("key cannot be empty")

    # 防止加入大小超过缓存最大容量限制的条目
    if self.eviction_enabled and len(value) > self.max_size:
      msg = (f"Cache value for key {key!r} of size {len(value)} bytes exceeds "
             f"the maximum cache size of {self.max_size} bytes")
      warnings.warn(msg)
      return

    cache_path = self.path / f"{key}{_CACHE_SUFFIX}"

    if self.eviction_enabled:
      self.lock.acquire(timeout=self.lock_timeout_secs)

    try:
      if cache_path.exists():
        return

      self._evict_if_needed(additional_size=len(value))

      cache_path.write_bytes(value)

      if self.eviction_enabled:
        timestamp = time.time_ns().to_bytes(8, "little")
        atime_path = self.path / f"{key}{_ATIME_SUFFIX}"
        atime_path.write_bytes(timestamp)

    finally:
      if self.eviction_enabled:
        self.lock.release()

  def _evict_if_needed(self, *, additional_size: int = 0) -> None:
    """如有必要则从缓存中淘汰最近最少使用的条目，
    以确保缓存不超过其最大容量。

    Args:
      additional_size: 即将加入缓存的新条目的大小。
        在判断是否需要淘汰时把它计入，
        以便将新条目也考虑在内。
    """
    if not self.eviction_enabled:
      return

    # 一个优先队列，每个元素是一个元组 `(file_atime, key, file_size)`
    h: list[tuple[int, str, int]] = []
    dir_size = 0
    for cache_path in self.path.glob(f"*{_CACHE_SUFFIX}"):
      file_stat = cache_path.stat()

      # `pathlib` 与 `etils[epath]` 获取文件大小的 API 不同，
      # 而这两种情况我们都需要支持。
      # 另见 https://github.com/google/etils/issues/630
      file_size = file_stat.st_size if not pathlib.epath_installed else file_stat.length  # pyrefly: ignore[missing-attribute]

      key = cache_path.name.removesuffix(_CACHE_SUFFIX)
      atime_path = self.path / f"{key}{_ATIME_SUFFIX}"
      file_atime = int.from_bytes(atime_path.read_bytes(), "little")

      dir_size += file_size
      heapq.heappush(h, (file_atime, key, file_size))

    target_size = self.max_size - additional_size
    # 不断淘汰文件，直到目录大小小于或等于
    # `target_size`
    while dir_size > target_size:
      file_atime, key, file_size = heapq.heappop(h)

      logger.debug("Evicting cache entry %r: file size %d bytes, "
                   "target cache size %d bytes", key, file_size, target_size)

      cache_path = self.path / f"{key}{_CACHE_SUFFIX}"
      atime_path = self.path / f"{key}{_ATIME_SUFFIX}"

      cache_path.unlink()
      atime_path.unlink()

      dir_size -= file_size
