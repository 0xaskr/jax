# Copyright 2018 The JAX Authors.
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

# 文件职责：把 JAX 与 XLA/PJRT 运行时连接起来，并对外提供“后端/设备”查询入口。
# 它负责注册 CPU、GPU、TPU 等后端工厂，发现并加载 `jax_plugins` 命名空间包或
# `PJRT_NAMES_AND_LIBRARY_PATHS` 环境变量中的 PJRT 插件，再把插件的 C API
# （`PJRT_Api*`）统一包装成 `xla_client.Client`，并管理设备拓扑的构建。
# 上层模块通过 `get_backend`、`devices`、`local_devices`、`process_index` 等
# 函数获取后端与设备信息；本文件同时定义相关的 config 标志（可见设备、
# 跨主机传输、CPU 异步分派等）以及设备/进程数量的便捷查询工具。

"""与 XLA 交互的接口与工具函数。

本模块包装 XLA 客户端与构建器，以统一它们的接口，并提供一些在 Numpy 与 XLA
之间转换的自动类型映射逻辑。此外还有少量相关的类型转换工具。
"""
from __future__ import annotations

import atexit
from collections.abc import Callable, Mapping
import dataclasses
from functools import partial
import importlib
import json
import logging
import os
import pkgutil
import platform as py_platform
import threading
from typing import Any
from collections.abc import Sequence
import warnings

from jax._src import config
from jax._src import distributed
from jax._src import hardware_utils
from jax._src import traceback_util
from jax._src import util
from jax._src.cloud_tpu_init import get_tpu_library_path
from jax._src.lib import xla_client
from jax._src.lib import _jax
from jax._src.lib import _profiler

logger = logging.getLogger(__name__)

jax_plugins: Any | None
try:
  import jax_plugins  # pyrefly: ignore[missing-import]
except ModuleNotFoundError:
  jax_plugins = None
except ImportError as e:
  logger.error("Failed to import jax_plugins: %s", e)
  jax_plugins = None

traceback_util.register_exclusion(__file__)

# 此集合中的运行时会强制对降级启用向前兼容。
FORCE_FORWARD_COMPAT_LOWERING_RUNTIMES: set[str] = set()

MIN_COMPUTE_CAPABILITY = 52

# TODO(phawkins): 移除 jax_xla_backend。
_XLA_BACKEND = config.string_flag(
    'jax_xla_backend', '',
    help='Deprecated, please use --jax_platforms instead.')
BACKEND_TARGET = config.string_flag(
    'jax_backend_target',
    os.getenv('JAX_BACKEND_TARGET', '').lower(),
    help='Either "local" or "rpc:address" to connect to a remote service target.')
# TODO(skye): 等我们对 --jax_platforms 测试一段时间后，在此项被使用时给出警告
_PLATFORM_NAME = config.string_flag(
    'jax_platform_name',
    os.getenv('JAX_PLATFORM_NAME', '').lower(),
    help='Deprecated, please use --jax_platforms instead.')
CUDA_VISIBLE_DEVICES = config.string_flag(
    'jax_cuda_visible_devices', 'all',
    help=(
      'Restricts the set of CUDA devices that JAX will use. Either "all", or a '
      'comma-separate list of integer device IDs.'))
_ROCM_VISIBLE_DEVICES = config.string_flag(
    'jax_rocm_visible_devices', 'all',
    help=(
      'Restricts the set of ROCM devices that JAX will use. Either "all", or a '
      'comma-separate list of integer device IDs.'))
_ONEAPI_VISIBLE_DEVICES = config.string_flag(
    'jax_oneapi_visible_devices', 'all',
    help=(
      'Restricts the set of ONEAPI devices that JAX will use. Either "all", or a '
      'comma-separate list of integer device IDs.'))

MOCK_NUM_GPU_PROCESSES = config.int_flag(
    name="mock_num_gpu_processes",
    default=0,
    help="Mock number of JAX processes in GPU client. Value zero turns "
         "off mocking.",
)
MOCK_GPU_TOPOLOGY = config.string_flag(
    name="jax_mock_gpu_topology",
    default="",
    help='Mock multi-host GPU topology in GPU client. The value should '
         'be of the form "<number-of-slices> x <number-of-hosts-per-slice> x '
         '<number-of-devices-per-host>". Empty string turns off mocking.',
)

_CPU_ENABLE_ASYNC_DISPATCH = config.bool_flag(
    name="jax_cpu_enable_async_dispatch",
    default=True,
    help="Only applies to non-parallel computations. If False, run computations"
    "inline without async dispatch.",
)

FORCE_DCN_CROSS_HOST_TRANSFERS = config.bool_flag(
    name="jax_force_dcn_cross_host_transfers",
    default=False,
    help="Force cross host transfers to use the DCN socket transfer library "
         "even when the plugin supports cross-host transfers."
)

SORT_DEVICES_BY_PROCESS_INDEX = config.bool_flag(
    name="jax_sort_devices_by_process_index",
    default=True,
    help="Sort JAX devices by process index first, then by device id. "
         "If False, sort devices only by device id, which preserves the "
         "global device ordering assigned by the PJRT client."
)

CROSS_HOST_TRANSFER_SOCKET_ADDRESS = config.string_flag(
    name="jax_cross_host_transfer_socket_address",
    default="",
    help="Socket address to use for cross host device transfers via DCN. "
    "Necessary only if the PjRt plugin does not support cross host transfers.",
)

CROSS_HOST_TRANSPORT_ADDRESSES = config.string_flag(
    name="jax_cross_host_transport_addresses",
    default="",
    help=(
        "Comma-separated list of transport addresses to use for cross host "
        "device transfers via DCN. If not set, defaults to [0.0.0.0:0] * 4."
    ),
)

CROSS_HOST_TRANSFER_TIMEOUT_SECONDS = config.int_flag(
    "jax_cross_host_transfer_timeout_seconds",
    None,
    help=(
      "Timeout for cross host transfer metadata exchange through KV store. "
      "Default is one minute."
    ),
)

CROSS_HOST_TRANSFER_TRANSFER_SIZE = config.int_flag(
    "jax_cross_host_transfer_transfer_size",
    None,
    help="Chunk size for chunked transfer requests."
)

# 如果用户调用了 fork() 就给出警告，因为这对他们来说不会有好结果。
def _at_fork():
  warnings.warn(
    "os.fork() was called. os.fork() is incompatible with multithreaded code, "
    "and JAX is multithreaded, so this will likely lead to a deadlock.",
    RuntimeWarning, stacklevel=2)

_at_fork_handler_installed = False

# 后端

_NameValueMapping = Mapping[str, str | int | list[int] | float | bool]

def _make_transfer_server_factory(
) -> _jax.TransferServerInterfaceFactory | None:
  """创建传输服务器接口工厂。"""
  if (not CROSS_HOST_TRANSFER_SOCKET_ADDRESS.value or not
      hasattr(_jax, "make_transfer_server_interface_factory")):
    return None
  transport_addresses = []
  if CROSS_HOST_TRANSPORT_ADDRESSES.value:
    transport_addresses = CROSS_HOST_TRANSPORT_ADDRESSES.value.split(",")
  transfer_server_kwargs = {
      "distributed_client": distributed.global_state.client,
      "socket_address": CROSS_HOST_TRANSFER_SOCKET_ADDRESS.value,
      "transport_addresses": transport_addresses,
  }
  if CROSS_HOST_TRANSFER_TIMEOUT_SECONDS.value is not None:
    transfer_server_kwargs["cross_host_transfer_timeout_seconds"] = (
        CROSS_HOST_TRANSFER_TIMEOUT_SECONDS.value)
  if CROSS_HOST_TRANSFER_TRANSFER_SIZE.value is not None:
    transfer_server_kwargs["transfer_size"] = (
        CROSS_HOST_TRANSFER_TRANSFER_SIZE.value)
  return _jax.make_transfer_server_interface_factory(**transfer_server_kwargs)


def make_tpu_client(
    library_path: str | None = None, options: _NameValueMapping | None = None
):
  """返回一个 TPU 客户端。默许最多 32 个在途计算。"""
  if not _jax.pjrt_plugin_loaded('tpu'):
    c_api = xla_client.load_pjrt_plugin_dynamically(
        "tpu", library_path or "libtpu.so"
    )
    _profiler.register_plugin_profiler(c_api)
    assert _jax.pjrt_plugin_loaded('tpu')
  if not _jax.pjrt_plugin_initialized('tpu'):
    _jax.initialize_pjrt_plugin('tpu')
  if options is None:
    options = {}
  return _jax.get_c_api_client(
      "tpu",
      options,
      distributed.global_state.client,
      _make_transfer_server_factory(),
      FORCE_DCN_CROSS_HOST_TRANSFERS.value,
      SORT_DEVICES_BY_PROCESS_INDEX.value,
  )


def tpu_client_timer_callback(timer_secs: float) -> xla_client.Client | None:
  def _log_warning():
    warnings.warn(
      f'TPU backend initialization is taking more than {timer_secs} seconds. '
      'Did you run your code on all TPU hosts? '
      'See https://docs.jax.dev/en/latest/multi_process.html '
      'for more information.')

  # 在 `timer_secs` 之后记录一条警告。
  t = threading.Timer(timer_secs, _log_warning)
  t.start()

  try:
    client = make_tpu_client(
        get_tpu_library_path(),
        _options_from_jax_configs("tpu"))
  finally:
    t.cancel()

  return client


# 后端
#
# 我们对“后端”与“设备”之间的关系没有特别的预设。例如，可能存在多个后端
# 提供同一种设备。

BackendFactory = Callable[[], xla_client.Client | None]
TopologyFactory = Callable[..., xla_client.DeviceTopology | None]

@dataclasses.dataclass(slots=True)
class BackendRegistration:
  factory: BackendFactory

  # 选择默认后端时该后端的优先级。数值越大表示越优先。
  priority: int

  # 如果此后端初始化失败，我们是否应记录一条用户可见的警告？
  # 对于插件（例如 TPU），我们通常希望失败是可见的，因为如果你不打算使用它，
  # 又何必安装这个插件呢？
  fail_quietly: bool = False

  # 这个插件是实验性的吗？如果某个插件被认为是实验性的，我们会在它被初始化时
  # 发出警告。这主要是为了正确设定用户预期：我们不希望用户因为一个有缺陷的插件
  # 而认为 JAX 有问题。
  experimental: bool = False

  # 若此后端是插件，则为其 C API（`PJRT_Api*`）。
  c_api: Any | None = None

_backend_factories: dict[str, BackendRegistration] = {}
_default_backend: xla_client.Client | None = None
_backends : dict[str, xla_client.Client] = {}
_backend_errors : dict[str, str] = {}
_backend_lock = threading.Lock()
_plugins_registered: bool = False
_plugin_lock = threading.Lock()
_topology_factories: dict[str, TopologyFactory] = {}
_plugin_callbacks: list[Any] = []
_plugin_callback_lock = threading.Lock()

# 已知的非实验性插件集合。
#
# 如果某个插件通过了 JAX 测试套件，就可以把它加入下面的允许列表。
# 如果你希望被加入，请提交一个 PR。
#
# 一个插件不必实现 JAX 用到的每一个特性，只要它实现了合理的特性集合、并且对
# 未实现的特性优雅地失败即可。错误的输出是不可接受的。
_nonexperimental_plugins: set[str] = {'cuda', 'rocm'}

# 在 JAX 代码库中有注册信息的已知实验性插件集合。
_experimental_plugins: set[str] = {"oneapi"}

def register_backend_factory(name: str, factory: BackendFactory, *,
                             priority: int = 0,
                             fail_quietly: bool = True,
                             experimental: bool = False,
                             make_topology: TopologyFactory | None = None,
                             c_api: Any | None = None) -> None:
  with _backend_lock:
    if name in _backends:
      raise RuntimeError(f"Backend {name} already initialized")
  _backend_factories[name] = BackendRegistration(
    factory, priority, fail_quietly, experimental, c_api)
  if make_topology is not None:
    _topology_factories[name] = make_topology


def make_cpu_client(
    collectives: _jax.CpuCollectives | None = None,
) -> xla_client.Client:
  """创建使用所请求的集合通信实现的 CPU 客户端。

  客户端所使用的 CPU 集合通信实现由标志 `--jax_cpu_collectives_implementation`
  决定——除非提供了 `collectives`，此时该标志会被覆盖，改用 `collectives`。

  Args:
    collectives: 可选的 CPU 集合通信实现，若提供则由客户端使用。

  Raises:
    RuntimeError: 如果 `--jax_cpu_collectives_implementation` 未知。

  Returns:
    所创建的 CPU 客户端。
  """
  # TODO(skyewm): 等 https://github.com/jax-ml/jax/pull/26172 合入后，
  # 改用 distributed.is_initialized()。
  if collectives is None and distributed.global_state.client is not None:
    collectives_impl = config.cpu_collectives_implementation.value
    if collectives_impl == 'gloo':
      collectives = _jax.make_gloo_tcp_collectives(
        distributed_client=distributed.global_state.client,
      )
    elif collectives_impl == 'mpi':
      collectives = _jax.make_mpi_collectives()
      collectives.Init()
      atexit.register(collectives.Finalize)
    else:
      # 已由 config 模块校验过
      assert collectives_impl is None

  num_devices = num_cpu_devices.value if num_cpu_devices.value >= 0 else None
  return xla_client.make_cpu_client(
      asynchronous=_CPU_ENABLE_ASYNC_DISPATCH.value,
      distributed_client=distributed.global_state.client,
      node_id=distributed.global_state.process_id,
      num_nodes=distributed.global_state.num_processes,
      collectives=collectives,
      num_devices=num_devices,
      get_local_topology_timeout_minutes=cpu_get_local_topology_timeout_minutes.value,
      get_global_topology_timeout_minutes=cpu_get_global_topology_timeout_minutes.value,
      transfer_server_factory=_make_transfer_server_factory(),
  )


register_backend_factory(
    "cpu", make_cpu_client, priority=0, fail_quietly=False
)

def get_num_nodes_from_gpu_topology(topology: str) -> int:
    try:
      slices_str, hosts_per_slice_str, _ = topology.split("x", 2)
      return int(slices_str) * int(hosts_per_slice_str)
    except (IndexError, ValueError):
      raise ValueError('Mock topology must be of the form '
                       '"<number-of-slices> x <number-of-hosts-per-slice> x '
                       '<number-of-devices-per-host>".')

# TODO(phawkins,skyewm): 把 TPU 插件改为使用 PJRT 插件机制，
# 然后在初始化失败时大声报错。
register_backend_factory(
  'tpu', partial(tpu_client_timer_callback, timer_secs=60.0), priority=300,
  fail_quietly=True)


def _get_pjrt_plugin_names_and_library_paths(
    plugins_from_env: str,
) -> dict[str, str]:
  """从环境变量获取待加载 PJRT 插件的名称与库路径。

  Args:
    plugins_from_env: 来自环境变量的插件名称与路径，格式为
      'name1:path1,name2:path2'（Windows 上为 'name1;path1,name2;path2'）。

  Returns:
    待加载 PJRT 插件的 {插件名: 库路径} 字典。
  """
  if not plugins_from_env:
    return {}

  pjrt_plugins = {}
  for plugin in plugins_from_env.split(','):
    try:
      name, library_path = plugin.split(os.path.pathsep)
      pjrt_plugins[name] = library_path
    except ValueError:
      logger.warning(
          'invalid value %s in env var PJRT_NAMES_AND_LIBRARY_PATHS %s',
          plugin,
          plugins_from_env,
      )
  return pjrt_plugins


def _get_pjrt_plugin_config(
    json_path: str,
) -> tuple[
    str, Mapping[str, str | int | list[int] | float | bool] | None
]:
  """从 json 文件获取 PJRT 插件配置。

  该 json 文件需要有一个 "library_path" 字段以给出插件库路径。它还可以有一个
  可选的 "create_option" 字段，用于给出创建 PJRT 插件客户端时所用的选项。
  "create_option" 的值是键值对。关于支持的值类型，请参见
  xla_client._NameValueMapping。
  """
  with open(json_path) as f:
    config = json.load(f)
  if 'library_path' not in config.keys():
    raise ValueError(
        'PJRT plugin config file should contain "library_path" field.'
    )
  return (config['library_path'], config.get('create_options'))

def discover_pjrt_plugins() -> None:
  """发现命名空间包 `jax_plugins` 中的插件并导入它们。

  有两种用于发现插件模块的方法。实现者应当同时使用这两种方法，以覆盖所有打包
  与开发场景：

  1. 在 `jax_plugins` 命名空间包下定义一个全局唯一的模块（也就是说，只需创建
     一个 `jax_plugins` 目录并在其下定义你的模块）。
  2. 若通过 pyproject.toml 或 setup.py 构建包，则在 `jax_plugins` 组下添加一个
     指向你完整模块名的 entry-point，以声明你的插件模块名。

  在 JAX 启动期间，JAX 会加载以这种方式发现的每个模块并调用其 `initialize()`
  函数。该函数应当通过调用
  `jax._src.xla_bridge.register_plugin(name, priority=, library_paty=,
  options=)` 来注册其具体的插件名称/实现。由于所有已安装插件的 `initialize()`
  函数都会被调用，它们应避免执行与注册无关的昂贵工作。

  TODO: 我们应当提供 `register_plugin` 的一个变体，允许通过回调来解析
  library_path 与 options。这样在需要从重量级系统初始化中推导选项的场景下，
  就能实现轻量级的插件注册。
  """
  plugin_modules = set()
  # 扫描 |jax_plugins| 下已安装的模块。注意并非所有打包场景都适合这种扫描，
  # 因此我们还使用 entry-point 方法来为列表补充种子。
  if jax_plugins:
    for _, name, _ in pkgutil.iter_modules(
        jax_plugins.__path__, jax_plugins.__name__ + '.'
    ):
      logger.debug("Discovered path based JAX plugin: %s", name)
      plugin_modules.add(name)
  else:
    logger.debug("No jax_plugins namespace packages available")

  # 用声明的 entrypoint 加以补充。
  from importlib.metadata import entry_points

  for entry_point in entry_points(group="jax_plugins"):
    logger.debug("Discovered entry-point based JAX plugin: %s",
                 entry_point.value)
    plugin_modules.add(entry_point.value)

  # 现在加载并初始化它们全部。
  for plugin_module_name in plugin_modules:
    logger.debug("Loading plugin module %s", plugin_module_name)
    plugin_module = None
    try:
      plugin_module = importlib.import_module(plugin_module_name)
    except ModuleNotFoundError:
      logger.warning("Jax plugin configuration error: Plugin module %s "
                     "does not exist", plugin_module_name)
    except ImportError:
      logger.exception("Jax plugin configuration error: Plugin module %s "
                       "could not be loaded")

    if plugin_module:
      try:
        plugin_module.initialize()
      except:
        logger.exception("Jax plugin configuration error: Exception when "
                         "calling %s.initialize()", plugin_module_name)


def _options_from_jax_configs(plugin_name):
  options = {}

  pjrt_client_options = config.jax_pjrt_client_create_options.value
  if isinstance(pjrt_client_options, str):
    pjrt_client_option_list = []
    if pjrt_client_options:
      pjrt_client_option_list = pjrt_client_options.split(";")

    for option in pjrt_client_option_list:
      option_list = option.split(":")
      if (len(option_list) != 2):
        raise RuntimeError(
            "Multiple ':' separators for option in "
            f"jax_pjrt_client_create_options: '{option}'. "
            "Should be in format 'key:value'")
      options[option_list[0]] = option_list[1]
  elif isinstance(pjrt_client_options, dict):
    options.update(pjrt_client_options)

  _visible_device_configs = {
      "cuda": CUDA_VISIBLE_DEVICES,
      "rocm": _ROCM_VISIBLE_DEVICES,
      "oneapi": _ONEAPI_VISIBLE_DEVICES,
  }
  if plugin_name in _visible_device_configs:
    visible_devices = _visible_device_configs[plugin_name].value
    if visible_devices != 'all':
      options['visible_devices'] = [int(x) for x in visible_devices.split(',')]
    mock_gpu_topology = MOCK_GPU_TOPOLOGY.value or None
    mock_num_processes = (get_num_nodes_from_gpu_topology(mock_gpu_topology) if
        mock_gpu_topology else MOCK_NUM_GPU_PROCESSES.value)
    options['enable_mock_nccl'] = mock_num_processes > 0
    if mock_num_processes > 0:
      options['num_nodes'] = mock_num_processes
      if mock_gpu_topology:
        options['mock_gpu_topology'] = mock_gpu_topology

  return options

OptionsDict = Mapping[str, str | int | list[int] | float | bool]


def make_pjrt_c_api_client(
    plugin_name: str,
    options: OptionsDict | Callable[[], OptionsDict] | None = None,
) -> xla_client.Client:
  """为给定的插件创建 PjRt 客户端。

  Args:
    plugin_name: 插件的名称。
    options: 可选。在创建 PJRT 插件客户端时使用。它可以是一个可调用对象，
      此时它会在插件初始化时被调用，并且应当返回一个选项字典。
  """
  if not xla_client.pjrt_plugin_initialized(plugin_name):
    xla_client.initialize_pjrt_plugin(plugin_name)
  updated_options: dict[str, Any] = {}
  if options is not None:
    updated_options.update(options() if callable(options) else options)
  updated_options.update(_options_from_jax_configs(plugin_name))
  if distributed.global_state.client is None:
    return xla_client.make_c_api_client(plugin_name, updated_options, None)

  distribute_options = {
      'node_id': distributed.global_state.process_id,
      'num_nodes': distributed.global_state.num_processes,
  }
  if (partition_index := distributed.global_state.partition_index) is not None:
    distribute_options['partition_index'] = partition_index
  if options is not None:
    distribute_options.update(updated_options)
  return xla_client.make_c_api_client(
      plugin_name,
      distribute_options,
      distributed.global_state.client,
      _make_transfer_server_factory(),
      FORCE_DCN_CROSS_HOST_TRANSFERS.value,
      SORT_DEVICES_BY_PROCESS_INDEX.value,
  )


def register_plugin(
    plugin_name: str,
    *,
    priority: int = 400,
    library_path: str | None = None,
    options: OptionsDict | Callable[[], OptionsDict] | None = None,
    c_api: Any | None = None,
    factory: BackendFactory | None = None,
    make_topology: TopologyFactory | None = None,
) -> Any:
  """为 PJRT 插件注册一个后端工厂。

  Args:
    plugin_name: 插件的名称。
    priority: 该插件在 jax 后端中注册时应具有的优先级。默认为 400。
    library_path: 可选。插件 .so 文件的完整路径。插件需要提供 library_path
      或 c_api 中的一个。
    options: 可选。在创建 PJRT 插件客户端时使用。它可以是一个可调用对象，
      此时它会在插件初始化时被调用，并且应当返回一个选项字典。
    c_api: 可选。插件可以提供要注册的 PJRT C API。
    factory: 可选。创建 PJRT 客户端的工厂函数。若未提供，则使用默认工厂。
  """

  if library_path and c_api:
    logger.error(
        "Both library_path and c_api are provided when registering PJRT plugin"
        " %s",
        plugin_name,
    )
    return
  if not library_path and not c_api:
    logger.error(
        "Neither library_path nor c_api provided when registering PJRT plugin"
        " %s",
        plugin_name,
    )
    return

  if factory is not None and options is not None:
    raise ValueError(
        "Cannot provide both 'factory' and 'options' when registering PJRT"
        " plugin. When providing a custom factory, the factory's must handle"
        " its own options."
    )
  if factory is None:
    factory = partial(make_pjrt_c_api_client, plugin_name, options=options)

  logger.debug(
      'registering PJRT plugin %s from %s', plugin_name, library_path
  )
  if library_path is not None:
    c_api = xla_client.load_pjrt_plugin_dynamically(plugin_name, library_path)
    _profiler.register_plugin_profiler(c_api)
  else:
    assert c_api is not None
    xla_client.load_pjrt_plugin_with_c_api(plugin_name, c_api)

  make_topology = make_topology or partial(xla_client.make_c_api_device_topology, c_api)
  experimental = plugin_name not in _nonexperimental_plugins
  register_backend_factory(plugin_name, factory, priority=priority,
                           fail_quietly=False, experimental=experimental,
                           make_topology=make_topology, c_api=c_api)
  return c_api


def register_pjrt_plugin_factories_from_env() -> None:
  """为 PJRT 插件注册后端工厂。

  对于输入字符串中的每个 PJRT 插件，都会注册一个后端工厂，格式为
  'name1:path1,name2:path2'（Windows 上为 'name1;path1,name2;path2'）。该路径
  既可以是插件库的路径，也可以是插件配置 json 文件的路径。该 json 文件需要
  有一个 "library_path" 字段以给出插件库路径。它还可以有一个可选的
  "create_option" 字段，用于给出创建 PJRT 插件客户端时所用的选项。
  "create_option" 的值是键值对。关于支持的值类型，请参见
  xla_client._NameValueMapping。

  TPU PJRT 插件将在 make_tpu_client 中单独加载并注册。
  """
  pjrt_plugins = _get_pjrt_plugin_names_and_library_paths(
      os.getenv('PJRT_NAMES_AND_LIBRARY_PATHS', '')
  )
  for plugin_name, path in pjrt_plugins.items():
    if path.endswith('.json'):
      library_path, options = _get_pjrt_plugin_config(path)
    else:
      library_path = path
      options = None
    logger.debug(
        'registering PJRT plugin %s from %s', plugin_name, library_path
    )
    register_plugin(plugin_name, library_path=library_path, options=options)


def _discover_and_register_pjrt_plugins():
  global _plugins_registered

  # 需要一个单独的锁，因为 register_backend_factory（由 register_plugin 调用）
  # 需要持有 _backend_lock。
  with _plugin_lock:
    if not _plugins_registered:
      # 位于命名空间包 `jax_plugins` 中、或在 `jax_plugins` 组下拥有 entry-point
      # 的插件将被导入。
      discover_pjrt_plugins()
      # 注册环境变量 PJRT_NAMES_AND_LIBRARY_PATHS 中设置的插件名称与路径，
      # 格式为 'name1:path1,name2:path2'（Windows 上为 'name1;path1,name2;path2'）。
      register_pjrt_plugin_factories_from_env()
      with _plugin_callback_lock:
        for factory in _backend_factories.values():
          if factory.c_api is not None:
            for callback in _plugin_callbacks:
              callback(c_api=factory.c_api)
      _plugins_registered = True
_platform_aliases = {
  "cuda": "gpu",
  "rocm": "gpu",
  "oneapi": "gpu",
}

_alias_to_platforms: dict[str, list[str]] = {}
for _platform, _alias in _platform_aliases.items():
  _alias_to_platforms.setdefault(_alias, []).append(_platform)


def known_platforms() -> set[str]:
  platforms = set()
  platforms |= set(_nonexperimental_plugins)
  platforms |= set(_backend_factories.keys())
  platforms |= set(_platform_aliases.values())
  platforms |= set(_platform_aliases.keys())
  return platforms


def is_known_platform(platform: str) -> bool:
  # 如果某个平台有已注册的工厂，它就是有效的。我们是否能够初始化该平台并不重要；
  # 我们只关心我们听说过它，并且它不是（例如）拼写错误。
  return platform in known_platforms()


def canonicalize_platform(platform: str) -> str:
  """把平台别名替换为它们的具体等价形式。

  具体来说，会根据实际存在的硬件，把 "gpu" 替换为 "cuda"、"oneapi" 或 "rocm"
  之一。出于 MLIR 降级规则等目的，我们希望区分 "cuda"、"oneapi" 和 "rocm"，
  但在许多情况下我们并不想强迫用户去关心这些区别。
  """
  platforms = _alias_to_platforms.get(platform, None)
  if platforms is None:
    return platform

  b = backends()
  for p in platforms:
    if p in b.keys():
      return p
  raise RuntimeError(f"Unknown backend: '{platform}' requested, but no "
                     f"platforms that are instances of {platform} are present. "
                     "Platforms present are: " + ",".join(b.keys()))


def expand_platform_alias(platform: str) -> list[str]:
  """把诸如 "gpu" 的别名展开为 ["cuda", "rocm", "oneapi"]。

  这是出于便利考虑：由于 cuda 与 rocm 共享大部分相同代码，我们预期它们在
  许多方面表现相似。
  """
  return _alias_to_platforms.get(platform, [platform])


def backends_are_initialized() -> bool:
  "如果后端已经初始化，则返回 true。"
  with _backend_lock:
    return len(_backends) != 0


def register_plugin_callbacks(callback):
  """注册一个在插件发现之后带着 c_api 被调用的回调。

  该回调会在所有已发现的 PJRT C API 插件上被调用。如果在插件被发现之前调用
  `register_plugin_callbacks`，回调会在插件刚被发现之后被调用。否则，回调会在
  调用 `register_plugin_callbacks` 时立即被调用。

  Args:
    callback: 要带着 c_api 被调用的回调。
  """
  with _plugin_callback_lock:
    if _plugins_registered:
      for factory in _backend_factories.values():
        if factory.c_api is not None:
          callback(c_api=factory.c_api)
    else:
      _plugin_callbacks.append(callback)


def backends() -> dict[str, xla_client.Client]:
  global _backends
  global _backend_errors
  global _default_backend
  global _at_fork_handler_installed

  _discover_and_register_pjrt_plugins()

  with _backend_lock:
    if _backends:
      return _backends

    # os.register_at_fork 只存在于 Unix 上。
    if not _at_fork_handler_installed and hasattr(os, "register_at_fork"):
      os.register_at_fork(before=_at_fork)
      _at_fork_handler_installed = True

    if jax_platforms := config.jax_platforms.value:
      platforms = []
      # 允许在平台列表中使用平台别名。
      for platform in jax_platforms.split(","):
        platforms.extend(expand_platform_alias(platform))
      priorities = range(len(platforms), 0, -1)
      # 如果用户显式指定了平台列表，则总是大声地失败。
      fail_quietly_list = [False] * len(platforms)
      platform_registrations = list(
        zip(platforms, priorities, fail_quietly_list))
    else:
      platform_registrations = [
          (platform, registration.priority, registration.fail_quietly)
          for platform, registration
          in _backend_factories.items()
      ]
    default_priority = -1000
    for platform, priority, fail_quietly in platform_registrations:
      try:
        if platform == "cuda" and not hardware_utils.has_visible_nvidia_gpu():
          continue

        backend = _init_backend(platform)
        _backends[platform] = backend

        if priority > default_priority:
          _default_backend = backend
          default_priority = priority
      except Exception as err:
        err_msg = f"Unable to initialize backend '{platform}': {err}"
        if fail_quietly:
          _backend_errors[platform] = str(err)
          logger.info(err_msg)
        else:
          if config.jax_platforms.value:
            err_msg += " (set JAX_PLATFORMS='' to automatically choose an available backend)"
          else:
            err_msg += " (you may need to uninstall the failing plugin package, or set JAX_PLATFORMS=cpu to skip this backend.)"
          raise RuntimeError(err_msg)

    assert _default_backend is not None
    if not config.jax_platforms.value:
      _suggest_missing_backends()
    return _backends

# 用于建议应安装哪些插件的代码。
#
# 欢迎插件厂商向这个列表添加代码，前提是存在一种轻量级的方式来判断硬件是否存在，
# 而不需要安装相关插件。

def _suggest_missing_backends():
  if py_platform.system() != "Linux":
    # 如果你不使用 Linux（或 WSL2），我们目前没有任何建议。
    return

  assert _default_backend is not None
  default_platform = _default_backend.platform
  if "cuda" not in _backends and hardware_utils.has_visible_nvidia_gpu():
    if hasattr(_jax, "GpuAllocatorConfig") and "cuda" in _backend_errors:
      err = _backend_errors["cuda"]
      warning_msg = f"CUDA backend failed to initialize: {err}."
      if "no supported devices found for platform CUDA." in err:
        warning_msg += (
          "This may be due to JAX pre-allocating too much device "
          "memory, leaving too little for CUDA library initialization. See "
          "https://docs.jax.dev/en/latest/gpu_memory_allocation.html "
          "for more details and potential workarounds."
        )
      warning_msg += "(Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)"

      logger.warning(warning_msg)
    else:
      logger.warning("An NVIDIA GPU may be present on this machine, but a "
                     "CUDA-enabled jaxlib is not installed. Falling back to "
                     f"{default_platform}.")
  elif "tpu" not in _backends and hardware_utils.num_available_tpu_chips_and_device_id()[0] > 0:
    logger.warning("A Google TPU may be present on this machine, but either a "
                    "TPU-enabled jaxlib or libtpu is not installed. Falling "
                    f"back to {default_platform}.")


def _clear_backends() -> None:
  global _backends
  global _backend_errors
  global _default_backend

  logger.debug("Clearing JAX backend caches.")
  with _backend_lock:
    _backends = {}
    _backend_errors = {}
    _default_backend = None


def _init_backend(platform: str) -> xla_client.Client:
  registration = _backend_factories.get(platform, None)
  if registration is None:
    raise RuntimeError(
        f"Backend '{platform}' is not in the list of known backends: "
        f"{list(_backend_factories.keys())}.")

  if registration.experimental:
    logger.warning(f"Platform '{platform}' is experimental and not all JAX "
                   "functionality may be correctly supported!")
  logger.debug("Initializing backend '%s'", platform)
  backend = registration.factory()
  # TODO(skye): 考虑直接由后端工厂抛出更具描述性的错误，而不是返回 None。
  if backend is None:
    raise RuntimeError(f"Could not initialize backend '{platform}'")
  # TODO(b/356678989): 只有当 `backend.device_count()` 统计的是纯 CPU 设备时，
  # 才检查它。
  if backend.device_count() == 0 and len(backend._get_all_devices()) == 0:
    raise RuntimeError(f"Backend '{platform}' provides no devices.")
  util.distributed_debug_log(("Initialized backend", backend.platform),
                             ("process_index", backend.process_index()),
                             ("device_count", backend.device_count()),
                             ("local_devices", backend.local_devices()))
  logger.debug("Backend '%s' initialized", platform)
  return backend


def _get_backend_uncached(
    platform: None | str | xla_client.Client = None
) -> xla_client.Client:
  # TODO(mattjj,skyewm): 等我们理清 'backend' 值的处理方式后，移除这里的
  # 输入多态。
  if platform is not None and not isinstance(platform, str):
    return platform

  platform = (platform or _XLA_BACKEND.value or _PLATFORM_NAME.value or None)

  bs = backends()
  if platform is not None:
    platform = canonicalize_platform(platform)
    backend = bs.get(platform, None)
    if backend is None:
      if platform in _backend_errors:
        raise RuntimeError(f"Backend '{platform}' failed to initialize: "
                           f"{_backend_errors[platform]}. "
                           f'Available backends are {list(bs)}')
      raise RuntimeError(
          f"Unknown backend {platform}. Available backends are {list(bs)}")
    return backend
  else:
    assert _default_backend is not None
    return _default_backend


@util.cache(max_size=None, trace_context_in_key=False)  # 不要用 util.memoize，因为这里不依赖 X64。
def get_backend(
    platform: None | str | xla_client.Client = None
) -> xla_client.Client:
  return _get_backend_uncached(platform)


def get_device_backend(
    device: xla_client.Device | None = None,
) -> xla_client.Client:
  """返回与 `device` 关联的后端，或默认后端。"""
  if device is not None:
    return device.client
  return get_backend()


def device_count(
    backend: str | xla_client.Client | None = None
) -> int:
  """返回设备总数。

  在大多数平台上，这与 :py:func:`jax.local_device_count` 相同。不过，在不同设备
  关联到不同进程的多进程平台上，它会返回所有进程的设备总数。

  Args:
    backend: 这是一个实验性特性，API 很可能会变化。可选，一个表示 xla 后端的
      字符串：``'cpu'``、``'gpu'`` 或 ``'tpu'``。

  Returns:
    设备数量。

  """
  return int(get_backend(backend).device_count())


def local_device_count(
    backend: str | xla_client.Client | None = None
) -> int:
  """返回本进程可寻址的设备数量。"""
  return int(get_backend(backend).local_device_count())


def devices(
    backend: str | xla_client.Client | None = None
) -> list[xla_client.Device]:
  """返回给定后端的所有设备列表。

  .. currentmodule:: jaxlib._jax

  每个设备由 :class:`Device` 的一个子类表示（例如 :class:`CpuDevice`、
  :class:`GpuDevice`）。返回列表的长度等于 ``device_count(backend)``。把
  :attr:`Device.process_index` 与 :py:func:`jax.process_index` 返回的值相比较，
  即可识别本地设备。

  如果 ``backend`` 为 ``None``，则返回默认后端的所有设备。默认后端通常是
  ``'gpu'`` 或 ``'tpu'``（若可用），否则为 ``'cpu'``。

  Args:
    backend: 这是一个实验性特性，API 很可能会变化。可选，一个表示 xla 后端的
      字符串：``'cpu'``、``'gpu'`` 或 ``'tpu'``。

  Returns:
    Device 子类的列表。
  """
  return get_backend(backend).devices()


def default_backend() -> str:
  """返回默认 XLA 后端的平台名。"""
  return get_backend(None).platform


def backend_pjrt_c_api_version(platform=None) -> tuple[int, int] | None:
  """返回后端的 PJRT C API 版本。

  如果后端不使用 PJRT C API，则返回 None。
  """
  backend = get_backend(platform)
  if hasattr(backend, "pjrt_c_api_major_version") and hasattr(
      backend, "pjrt_c_api_minor_version"
  ):
    return (backend.pjrt_c_api_major_version, backend.pjrt_c_api_minor_version)
  return None


def backend_xla_version(platform=None) -> int | None:
  """返回后端的 XLA 版本。

  如果后端不使用 PJRT C API，或插件属性中没有 xla_version，则返回 None。若后端
  是一个使用 xla_version 的插件，可以用这个方法来跳过在某个 xla_version 之前
  不可用的特性。
  """
  backend = get_backend(platform)
  return getattr(backend, "xla_version", None)

def backend_stablehlo_version(platform=None) -> Sequence[int] | None:
  """返回后端的 StableHLO 版本。

  如果后端不使用 PJRT C API，或插件属性中没有 stablehlo_current_version，则返回
  None。若后端是一个使用 stablehlo_current_version 的插件，可以用这个方法来跳过
  在某个 stablehlo_current_version 之前不可用的特性。
  """
  backend = get_backend(platform)
  return getattr(backend, "stablehlo_current_version", None)

@util.cache(max_size=None, trace_context_in_key=False)
def local_devices(process_index: int | None = None,
                  backend: str | xla_client.Client | None = None,
                  host_id: int | None = None) -> list[xla_client.Device]:
  """类似于 :py:func:`jax.devices`，但只返回某个给定进程的本地设备。

  如果 ``process_index`` 为 ``None``，则返回本进程的本地设备。

  Args:
    process_index: 进程的整数索引。进程索引可以通过
      ``len(jax.process_count())`` 获取。
    backend: 这是一个实验性特性，API 很可能会变化。可选，一个表示 xla 后端的
      字符串：``'cpu'``、``'gpu'`` 或 ``'tpu'``。

  Returns:
    Device 子类的列表。
  """
  if host_id is not None:
    warnings.warn(
        "The argument to jax.local_devices has been renamed from `host_id` to "
        "`process_index`. This alias will eventually be removed; please update "
        "your code.")
    process_index = host_id
  if process_index is None:
    process_index = get_backend(backend).process_index()
  if not (0 <= process_index < process_count(backend)):
    raise ValueError(f"Unknown process_index {process_index}")
  return [d for d in devices(backend) if d.process_index == process_index]


def process_index(
    backend: str | xla_client.Client | None = None
) -> int:
  """返回本进程的整数进程索引。

  在大多数平台上，它始终为 0。不过在多进程平台上它会有所不同。

  Args:
    backend: 这是一个实验性特性，API 很可能会变化。可选，一个表示 xla 后端的
      字符串：``'cpu'``、``'gpu'`` 或 ``'tpu'``。

  Returns:
    整数进程索引。
  """
  return get_backend(backend).process_index()


# TODO: 在 jax 0.2.13 发布之后的某个时候移除这个
def host_id(backend: str | xla_client.Client | None = None) -> int:
  warnings.warn(
      "jax.process_index has been renamed to jax.process_index. This alias "
      "will eventually be removed; please update your code.")
  return process_index(backend)


@util.cache(max_size=None, trace_context_in_key=False)
def process_count(
    backend: str | xla_client.Client | None = None
) -> int:
  """返回与该后端关联的 JAX 进程数量。"""
  gen = (d.process_index for d in devices(backend))
  return max(gen, default=0) + 1


# TODO: 在 jax 0.2.13 发布之后的某个时候移除这个
def host_count(backend: str | xla_client.Client | None = None) -> int:
  warnings.warn(
      "jax.process_count has been renamed to jax.process_count. This alias "
      "will eventually be removed; please update your code.")
  return process_count(backend)


def process_indices(
    backend: str | xla_client.Client | None = None
) -> list[int]:
  """返回与该后端关联的所有 JAX 进程索引的列表。

  Args:
    backend: 这是一个实验性特性，API 很可能会变化。可选，一个表示 xla 后端的
      字符串：``'cpu'``、``'gpu'`` 或 ``'tpu'``。

  Returns:
    整数进程索引的列表。
  """
  return list(range(process_count(backend)))


# TODO: 在 jax 0.2.13 发布之后的某个时候移除这个
def host_ids(
    backend: str | xla_client.Client | None = None
) -> list[int]:
  warnings.warn(
      "jax.process_indexs has been renamed to jax.process_indices. This alias "
      "will eventually be removed; please update your code.")
  return process_indices(backend)


def using_pjrt_c_api(backend=None):
  return "PJRT C API" in get_backend(backend).platform_version

def make_pjrt_topology(platform: str, topology_name='', **kwargs):
  _discover_and_register_pjrt_plugins()
  actual_platform = canonicalize_platform(platform)
  with _backend_lock:
    if actual_platform in _topology_factories:
      return _topology_factories[actual_platform](topology_name, **kwargs)
  raise NotImplementedError("topology not implemented for %s" % platform)


# TODO(parkers): 去掉这个，改用获取拓扑的通用方式。
def make_pjrt_tpu_topology(topology_name='', **kwargs):
  if not xla_client.pjrt_plugin_loaded("tpu"):
    library_path = get_tpu_library_path()
    if library_path is None:
      raise RuntimeError(
          "JAX TPU support not installed; cannot generate TPU topology. See"
          " https://github.com/jax-ml/jax#installation")
    c_api = xla_client.load_pjrt_plugin_dynamically("tpu", library_path)
    _profiler.register_plugin_profiler(c_api)
  assert xla_client.pjrt_plugin_loaded("tpu")
  if not xla_client.pjrt_plugin_initialized("tpu"):
    xla_client.initialize_pjrt_plugin("tpu")
  return xla_client.make_tfrt_tpu_c_api_device_topology(
      topology_name, **kwargs
  )

def _validate_backend_not_initialized(name, new_val):
  if backends_are_initialized():
    if getattr(config.config, name) == new_val:
      return
    raise RuntimeError(
        f"{name} config should be updated before backends are"
        " initialized i.e. before any JAX operation is executed. You should"
        " initialize this config immediately after `import jax`.")

num_cpu_devices = config.int_state(
    name="jax_num_cpu_devices",
    default=-1,
    help=(
        "Number of CPU devices to use. If not provided, the value of "
        "the XLA flag --xla_force_host_platform_device_count is used."
        " Must be set before JAX is initialized."),
    validator=partial(_validate_backend_not_initialized, "jax_num_cpu_devices"),
)

cpu_get_local_topology_timeout_minutes = config.int_state(
    name="jax_cpu_get_local_topology_timeout_minutes",
    default=2,
    help=(
        "Timeout in minutes for getting the local topology of each CPU device"
        " when building the global topology."
    ),
    validator=partial(_validate_backend_not_initialized,
                      "jax_cpu_get_local_topology_timeout_minutes"),
)

cpu_get_global_topology_timeout_minutes = config.int_state(
    name="jax_cpu_get_global_topology_timeout_minutes",
    default=5,
    help=(
        "Timeout in minutes for getting the global topology of CPU devices;"
        " should be strictly greater than"
        " `--jax_cpu_get_local_topology_timeout_minutes`."
    ),
    validator=partial(_validate_backend_not_initialized,
                      "jax_cpu_get_global_topology_timeout_minutes"),
)
