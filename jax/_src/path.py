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

# 文件职责：为 JAX 提供统一的路径抽象，屏蔽本地文件系统与云端对象存储的差异。
# 优先使用 `etils.epath`（pip 中的 `etils[epath]`），因为它可以读写 GCS 桶等远程路径；
# 若未安装则回退到标准库 `pathlib`，此时只能读写本地文件系统。
# 模块用 `PathProtocol` 声明 `Path` 的构造签名，让两种实现共享同一个类型接口。
# `make_jax_dump_dir` 供 IR dump 等场景创建输出目录，并支持 `sponge` 这一测试专用取值。

from typing import cast, Protocol
import logging
import os
import pathlib

__all__ = ["Path"]

logger = logging.getLogger(__name__)

epath_installed: bool


class PathProtocol(Protocol):
  """创建 `PurePath` 的工厂。"""
  def __call__(self, *pathsegments: str | os.PathLike) -> pathlib.Path:
    ...

Path: PathProtocol

# 若存在 etils.epath（在 pip 中即 etils[epath]），我们优先使用它，因为它
# 可以读写诸如 GCS 桶之类的路径。否则使用内置的 pathlib，
# 此时只能读写本地文件系统。
try:
  from etils import epath  # pyrefly: ignore[missing-import]
except ImportError:
  logger.debug("etils.epath was not found. Using pathlib for file I/O.")
  Path = pathlib.Path
  epath_installed = False
else:
  logger.debug("etils.epath found. Using etils.epath for file I/O.")
  # 归根结底，epath.Path 实现了 pathlib.Path。参见：
  # https://github.com/google/etils/blob/2083f3d932a88d8a135ef57112cd1f9aff5d559e/etils/epath/abstract_path.py#L47
  Path = epath.Path
  epath_installed = True

def make_jax_dump_dir(out_dir_path: str) -> pathlib.Path | None:
  """创建目录；若为 `sponge` 则返回未声明的输出目录。"""
  if not out_dir_path:
    return None
  if out_dir_path == "sponge":
    out_dir_path = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
    if not out_dir_path:
      raise ValueError(
          "Got output directory (e.g., via JAX_DUMP_IR_TO) 'sponge' but"
          " TEST_UNDECLARED_OUTPUTS_DIR is not defined."
      )
  out_dir = Path(out_dir_path)
  out_dir.mkdir(parents=True, exist_ok=True)
  return cast(pathlib.Path, out_dir)
