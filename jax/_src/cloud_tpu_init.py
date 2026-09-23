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

# 文件职责：在 Cloud TPU 虚拟机上自动完成 TPU 运行时所需的环境准备。
# 它探测本机的 TPU 硬件与 libtpu 库路径，并在加载 TPU 运行时之前设置
# 拓扑、平台与遥测相关的环境变量，还会对透明大页未开启等情况给出警告。
# 模块同时提供 libtpu 版本比较工具，供其他组件判断可用特性。

import logging
import os
import re
import warnings

from jax._src import config
from jax._src import hardware_utils

logger = logging.getLogger(__name__)

running_in_cloud_tpu_vm: bool = False


def maybe_import_libtpu():
  try:
    import libtpu  # pyrefly: ignore[missing-import]
  except ImportError:
    return None
  else:
    return libtpu


def get_tpu_library_path() -> str | None:
  path_from_env = os.getenv('TPU_LIBRARY_PATH')
  if path_from_env is not None:
    if os.path.isfile(path_from_env):
      return path_from_env
    warning_message = (
        f'TPU_LIBRARY_PATH is set to a non-existent path: {path_from_env}.'
        ' Falling back to default libtpu path. Please unset TPU_LIBRARY_PATH'
        ' or set it to a valid path.'
    )
    warnings.warn(warning_message)

  libtpu_module = maybe_import_libtpu()
  if libtpu_module is not None:
    return libtpu_module.get_library_path()

  return None


def jax_force_tpu_init() -> bool:
  return 'JAX_FORCE_TPU_INIT' in os.environ


def cloud_tpu_init() -> None:
  """自动设置 Cloud TPU 的拓扑以及其他环境变量。

  **必须在加载 TPU 运行时之前调用本函数，而 JAX 的 C++ 后端一被加载
  TPU 运行时就会被加载！也就是说，要在导入 xla_bridge 或 xla_client 之前
  调用它。**

  在非 Cloud TPU 环境中调用是安全的。

  其中一些环境变量用于告诉 TPU 运行时使用何种 mesh 拓扑。它默认假定为
  单主机拓扑，因此我们在这里手动设置这些变量，以便在适用时默认为整个
  pod 切片。

  若已设置任一与拓扑相关的环境变量，本函数不会设置任何环境变量。
  """
  global running_in_cloud_tpu_vm

  from jax import version

  # 若不在 Cloud TPU VM 上运行或没有安装 libtpu，则提前退出。
  libtpu_path = get_tpu_library_path()
  num_tpu_chips, tpu_id = hardware_utils.num_available_tpu_chips_and_device_id()
  if num_tpu_chips == 0:
    os.environ['TPU_SKIP_MDS_QUERY'] = '1'
  if (
      tpu_id is not None
      and tpu_id >= hardware_utils.TpuVersion.v5e
      and not hardware_utils.transparent_hugepages_enabled()
  ):
    warnings.warn(
        'Transparent hugepages are not enabled. TPU runtime startup and'
        ' shutdown time should be significantly improved on TPU v5e and newer.'
        ' If not already set, you may need to enable transparent hugepages in'
        ' your VM image (sudo sh -c "echo always >'
        ' /sys/kernel/mm/transparent_hugepage/enabled")'
    )
  if (libtpu_path is None or num_tpu_chips == 0) and not jax_force_tpu_init():
    return

  running_in_cloud_tpu_vm = True

  os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
  os.environ.setdefault('TPU_ML_PLATFORM', 'JAX')
  os.environ.setdefault('TPU_ML_PLATFORM_VERSION', version.__version__)
  os.environ.setdefault('ENABLE_RUNTIME_UPTIME_TELEMETRY', '1')
  if '--xla_tpu_use_enhanced_launch_barrier' not in os.environ.get(
      'LIBTPU_INIT_ARGS', ''
  ):
    os.environ['LIBTPU_INIT_ARGS'] = (
        os.environ.get('LIBTPU_INIT_ARGS', '')
        + ' --xla_tpu_use_enhanced_launch_barrier=true'
    )

  # 这能让 tensorstore 序列化在 TPU 上表现得更好
  os.environ.setdefault('TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS', '60')
  os.environ.setdefault('TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES', '256')

  # 若未设置 JAX_PLATFORMS 环境变量，config.jax_platforms 的默认值
  # 为 None。这种情况下我们把它设为 'tpu,cpu'，以确保 JAX
  # 会使用 TPU 后端。
  if config.jax_platforms.value is None:
    config.update('jax_platforms', 'tpu,cpu')

  if config.jax_pjrt_client_create_options.value is None:
    config.update(
        'jax_pjrt_client_create_options',
        f'ml_framework_name:JAX;ml_framework_version:{version.__version__}',
    )


_version_regex = re.compile(r'([0-9]+(?:\.[0-9]+)*)(?:(rc|dev).*)?')


def _parse_version(v: str) -> tuple[int, ...]:
  m = _version_regex.match(v)
  if m is None:
    raise ValueError(f"Unable to parse version '{v}'")
  return tuple(int(x) for x in m.group(1).split('.'))


def is_libtpu_at_least(version_str: str) -> bool:
  """若不在 Cloud TPU 上运行则返回 True。

  若在 Cloud TPU 上运行，则当已安装的 libtpu 版本不低于
  `version_str` 时返回 True。

  Note: 这里检查的是已安装的 `libtpu` Python 包的版本。
  若 `TPU_LIBRARY_PATH` 指向的路径与该包安装后的默认路径不同，
  则会发出警告，因为实际加载的库可能与我们检查的包版本并不一致。
  """
  if not running_in_cloud_tpu_vm:
    return True

  tpu_library_path = get_tpu_library_path()
  libtpu = maybe_import_libtpu()
  if libtpu is None:
    if tpu_library_path:
      warnings.warn(
          (
              'libtpu Python package is not installed, but TPU_LIBRARY_PATH is'
              f' set to {tpu_library_path}. Cannot determine libtpu version.'
              f' Assuming it is newer than {version_str}.'
          ),
          stacklevel=2,
      )
    else:
      warnings.warn(
          (
              'libtpu Python package is not installed, but we appear to be on a'
              ' Cloud TPU VM. Cannot determine libtpu version. Assuming it is'
              f' newer than {version_str}.'
          ),
          stacklevel=2,
      )
    return True

  if tpu_library_path and tpu_library_path != libtpu.get_library_path():
    logger.info(
        'TPU_LIBRARY_PATH is set to %s, which differs from the installed'
        ' package default (%s). Using the custom path set by TPU_LIBRARY_PATH'
        ' and assuming the version of libtpu is head for version tests.',
        tpu_library_path,
        libtpu.get_library_path(),
    )
    return True

  actual_version = _parse_version(libtpu.__version__)
  required_version = _parse_version(version_str)

  return actual_version >= required_version
