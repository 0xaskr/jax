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

# 文件职责：收集并展示本地环境与 JAX 安装信息，供提问或提交 issue 时一并附上。
# `print_environment_info` 汇总 jax、jaxlib、numpy、python 的版本、设备与进程数量、
# 平台信息，以及所有以 `JAX_`、`XLA_` 开头的环境变量；可用时还会附上 `nvidia-smi` 输出。
# 它以 `jax.print_environment_info` 的名字对外导出，是用户报告问题时的标准诊断入口。

from __future__ import annotations

import os
import platform
import subprocess
import sys
import textwrap

from jax._src import lib
from jax._src import xla_bridge as xb
import numpy as np

def try_nvidia_smi() -> str | None:
  try:
    return subprocess.check_output(['nvidia-smi']).decode()
  except Exception:
    return None


def print_environment_info(return_string: bool = False) -> str | None:
  """返回一个包含本地环境与 JAX 安装信息的字符串。

  在提问或提交 bug 时，附上这些信息很有用。

  Args: return_string (bool) : 若为 True，则返回该字符串而不打印到 stdout。
  """
  from jax import version

  # TODO(jakevdp): 是否应包含其他信息，例如 jax.config.values？
  python_version = sys.version.replace('\n', ' ')
  info = textwrap.dedent(f"""\
  jax:    {version.__version__}
  jaxlib: {lib.version_str}
  numpy:  {np.__version__}
  python: {python_version}
  device info: {xb.devices()[0].device_kind}-{xb.device_count()}, {xb.local_device_count()} local devices"
  process_count: {xb.process_count()}
  platform: {platform.uname()}""")
  for key, value in os.environ.items():
    if key.startswith(("JAX_", "XLA_")):
      info += f"\n{key}={value}"
  nvidia_smi = try_nvidia_smi()
  if nvidia_smi:
    info += '\n\n$ nvidia-smi\n' + nvidia_smi
  if return_string:
    return info
  else:
    return print(info)
