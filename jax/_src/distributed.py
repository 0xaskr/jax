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

# 文件职责：实现 JAX 的多主机分布式运行时入口（`jax.distributed.initialize` 等）。
# 它在用户执行任何 JAX 计算之前启动分布式运行时：进程 0 上创建协调器服务，
# 各进程分别创建运行时客户端并连接协调器，从而让进程互相发现、共享拓扑并做健康检查。
# 模块用全局单例 `global_state`（`State`）保存服务、客户端、进程号、分区索引等状态，
# 并负责可见设备环境变量、代理环境变量告警以及抢占同步管理器的初始化与关闭。

from __future__ import annotations

from collections.abc import Sequence
import logging
import os
from typing import Any
import warnings

from jax._src import clusters
from jax._src import config
from jax._src import xla_bridge
from jax._src.lib import _jax

logger = logging.getLogger(__name__)


_CHECK_PROXY_ENVS = config.bool_flag(
    name="jax_check_proxy_envs",
    default=True,
    help="Checks proxy vars in user envs and emit warnings.",
)


_ENABLE_RECOVERABILITY = config.bool_state(
    name="jax_enable_recoverability",
    default=False,
    help=(
        "Allows a multi-controller JAX job to continue running, even after some"
        " tasks have failed."
    ),
)

_ENABLE_PREEMPTION_SERVICE = config.bool_state(
    name='jax_enable_preemption_service',
    default=True,
    help=(
        "Enables the preemption service. See"
        " multihost_utils.reached_preemption_sync_point for details."
    ),
)

class State:
  process_id: int = 0
  num_processes: int = 1
  service: _jax.DistributedRuntimeService | Any | None = None
  client: _jax.DistributedRuntimeClient | Any | None = None
  preemption_sync_manager: Any | None = None
  coordinator_address: str | None = None
  partition_index: int | None = None

  def initialize(self,
                 coordinator_address: str | None = None,
                 num_processes: int | None = None,
                 process_id: int | None = None,
                 local_device_ids: int | Sequence[int] | None = None,
                 cluster_detection_method: str | None = None,
                 initialization_timeout: int = 300,
                 coordinator_bind_address: str | None = None,
                 heartbeat_timeout_seconds: int = 100,
                 shutdown_timeout_seconds: int = 300,
                 partition_index: int | None = None):
    coordinator_address = (coordinator_address or
                           os.environ.get('JAX_COORDINATOR_ADDRESS'))
    if isinstance(local_device_ids, int):
      local_device_ids = [local_device_ids]

    if local_device_ids is None and (env_ids := os.environ.get('JAX_LOCAL_DEVICE_IDS')):
      local_device_ids = list(map(int, env_ids.split(",")))

    if (cluster_detection_method != 'deactivate' and
        None in (coordinator_address, num_processes, process_id, local_device_ids)):
      (coordinator_address, num_processes, process_id, local_device_ids) = (
          clusters.ClusterEnv.auto_detect_unset_distributed_params(
              coordinator_address,
              num_processes,
              process_id,
              local_device_ids,
              cluster_detection_method,
              initialization_timeout,
          )
      )

    if coordinator_address is None:
      raise ValueError('coordinator_address should be defined.')
    if num_processes is None:
      raise ValueError('Number of processes must be defined.')
    if process_id is None:
      raise ValueError('The process id of the current process must be defined.')
    if not isinstance(process_id, int):
      raise TypeError("process_id must be a nonnegative int. "
                      f"Got process_id={process_id} of type {type(process_id)}.")
    if not isinstance(num_processes, int):
      raise TypeError("num_processes must be a positive int. "
                      f"Got num_processes={num_processes} of type {type(num_processes)}.")
    if not (0 <= process_id < num_processes):
      raise ValueError("process_id and num_processes must be nonnegative, with process_id < num_processes. "
                       f"Got process_id={process_id}, num_processes={num_processes}.")

    self.coordinator_address = coordinator_address

    # [::]:port 这个默认值告诉协调器在与 coordinator_address 相同的端口上
    # 绑定所有可用地址。
    default_coordinator_bind_address = '[::]:' + coordinator_address.rsplit(':', 1)[1]
    coordinator_bind_address = (coordinator_bind_address or
                                os.environ.get('JAX_COORDINATOR_BIND_ADDRESS',
                                               default_coordinator_bind_address))
    if coordinator_bind_address is None:
      raise ValueError('coordinator_bind_address should be defined.')

    if local_device_ids:
      visible_devices = ','.join(str(x) for x in local_device_ids)
      logger.info('JAX distributed initialized with visible devices: %s', visible_devices)
      config.update("jax_cuda_visible_devices", visible_devices)
      config.update("jax_rocm_visible_devices", visible_devices)

    self.process_id = process_id

    proxy_vars = []
    if _CHECK_PROXY_ENVS.value:
      proxy_vars = [key for key in os.environ.keys()
                    if '_proxy' in key.lower()]

    if len(proxy_vars) > 0:
      vars = " ".join(proxy_vars) + ". "
      warning = (
        f'JAX detected proxy variable(s) in the environment as distributed setup: {vars}'
        'On some systems, this may cause a hang of distributed.initialize and '
        'you may need to unset these ENV variable(s)'
      )
      logger.warning(warning)

    if process_id == 0:
      if self.service is not None:
        raise RuntimeError('distributed.initialize should only be called once.')
      logger.info(
          'Starting JAX distributed service on %s', coordinator_bind_address
      )
      self.service = _jax.get_distributed_runtime_service(
          coordinator_bind_address,
          num_processes,
          heartbeat_timeout=heartbeat_timeout_seconds,
          shutdown_timeout=shutdown_timeout_seconds,
          recoverable=_ENABLE_RECOVERABILITY.value,
      )

    self.num_processes = num_processes

    if self.client is not None:
      raise RuntimeError('distributed.initialize should only be called once.')

    self.client = _jax.get_distributed_runtime_client(
        coordinator_address,
        process_id,
        init_timeout=initialization_timeout,
        use_compression=True,
        heartbeat_timeout=heartbeat_timeout_seconds,
    )
    logger.info('Connecting to JAX distributed service on %s', coordinator_address)
    self.client.connect()

    self.initialize_preemption_sync_manager()

    if partition_index is None:
      jax_partition_index = os.environ.get('JAX_PARTITION_INDEX')
      jax_slice_index = os.environ.get('JAX_SLICE_INDEX')
      if jax_partition_index is not None:
        partition_index = int(jax_partition_index)
      elif jax_slice_index is not None:
        # 弃用于 2025-08-05 添加。应在 3 个月后移除。
        warnings.warn(
            'JAX_SLICE_INDEX has been deprecated. Please use'
            ' JAX_PARTITION_INDEX instead.',
            DeprecationWarning,
        )
        partition_index = int(jax_slice_index)
    self.partition_index = partition_index

  def shutdown(self):
    if self.preemption_sync_manager:
      # 必须在客户端之前关闭抢占同步管理器，
      # 因为抢占同步管理器依赖客户端。
      self.preemption_sync_manager.shutdown()
      self.preemption_sync_manager = None
    if self.client:
      self.client.shutdown()
      self.client = None
    if self.service:
      self.service.shutdown()
      self.service = None

  def initialize_preemption_sync_manager(self):
    if not _ENABLE_PREEMPTION_SERVICE.value:
      logger.info(
          'The JAX preemption service is disabled. You can enable it using the'
          ' jax_enable_preemption_service configuration option.'
      )
      return
    if self.preemption_sync_manager is not None:
      raise RuntimeError(
          'Preemption sync manager should only be initialized once.')
    self.preemption_sync_manager = (
        _jax.create_preemption_sync_manager())
    assert self.client is not None
    self.preemption_sync_manager.initialize(self.client)

global_state = State()

def initialize(coordinator_address: str | None = None,
               num_processes: int | None = None,
               process_id: int | None = None,
               local_device_ids: int | Sequence[int] | None = None,
               cluster_detection_method: str | None = None,
               initialization_timeout: int = 300,
               heartbeat_timeout_seconds: int = 100,
               shutdown_timeout_seconds: int = 300,
               coordinator_bind_address: str | None = None,
               slice_index: int | None = None,
               partition_index: int | None = None):
  """初始化 JAX 分布式系统。

  调用 :func:`~jax.distributed.initialize` 会让 JAX 做好在多主机 GPU 和
  Cloud TPU 上执行的准备。必须在执行任何 JAX 计算之前调用
  :func:`~jax.distributed.initialize`。

  JAX 分布式系统承担多项职责：

    * 让各个 JAX 进程能够相互发现并共享拓扑信息，
    * 执行健康检查，确保任一进程死亡时所有进程都会关闭，以及
    * 用于分布式检查点。

  如果你使用 TPU、Slurm 或 Open MPI，所有参数都是可选的：省略时会
  自动选取。

  可以用 ``cluster_detection_method`` 指定检测这些分布式参数的具体方法。
  你可以把任意一种自动检测的 ``spec_detect_methods`` 传给这个参数，
  不过在 TPU、Slurm 或 Open MPI 场景下没有必要。对于其他 MPI
  安装，如果你的 ``mpi4py`` 可以正常工作，可以传入
  ``cluster_detection_method="mpi4py"`` 来引导出所需参数。

  否则，你必须向 :func:`~jax.distributed.initialize` 提供
  ``coordinator_address``、``num_processes``、``process_id`` 和
  ``local_device_ids`` 参数。当这四个参数全部提供时，将跳过集群
  环境自动检测。

  请注意：在某些系统上，尤其是只能通过 HTTP_PROXY、HTTPS_PROXY 等
  代理变量访问外部网络的 HPC 集群，对
  :func:`~jax.distributed.initialize` 的调用可能会超时。你可能需要在启动
  应用之前取消设置这些变量。

  Args:
    coordinator_address: 进程 `0` 的 IP 地址，以及该进程用于
      启动协调器服务的端口。端口选择
      无关紧要，只要该端口在协调器上可用，
      且所有进程对该端口达成一致即可。
      仅在受支持的环境中可以为 ``None``，此时会自动选取。
      注意，像 ``localhost`` 或 ``127.0.0.1`` 这类特殊地址通常意味着程序
      会绑定到本地接口，不适合在多主机环境中运行。
    num_processes: 进程数量。仅在受支持的环境中可以为 ``None``，
      此时会自动选取。
    process_id: 当前进程的 ID 编号。整个集群中的 ``process_id`` 取值
      必须是稠密区间 ``0``、``1``、…、``num_processes - 1``。
      仅在受支持的环境中可以为 ``None``；若为 ``None`` 则自动选取。
    local_device_ids: 把当前进程可见的设备限制为 ``local_device_ids``。
      若为 ``None``，默认当前进程可见所有本地设备；但当进程是通过 Slurm 和 Open MPI
      在 GPU 上启动时例外，此时默认为每个进程一个设备。
    cluster_detection_method: 可选字符串，用于尝试自动检测分布式运行的
      配置。注意 "mpi4py" 方式要求环境中已安装可用的 ``mpi4py``，
      并且要用 ``mpiexec`` 或 ``mpirun`` 这类兼容 MPI 的作业启动器来启动应用。
      旧版自动检测选项 "ompi"（OMPI）和 "slurm"（Slurm）仍然可用。"deactivate" 会绕过
      自动集群检测。
    initialization_timeout: 连接将被重试的时间长度（秒）。
      如果初始化耗时超过指定的超时时间，
      初始化将报错。默认 300 秒，即 5 分钟。
    heartbeat_timeout_seconds: 若某进程在此时间（秒）内未成功
      发送任何心跳，则被视为已死亡。
      默认 100 秒。
    shutdown_timeout_seconds: 正在终止的进程等待其他所有进程也终止的
      时间（秒）。默认 300 秒。
    coordinator_bind_address: 进程 `0` 上的协调器服务应绑定的地址和端口。
      若未指定，默认绑定到与 ``coordinator_address`` 相同端口上的
      所有可用地址。在每节点有多个网络接口的系统上，
      只让协调器服务监听一个地址/接口
      可能不够。
    slice_index: 已弃用：请改用 ``partition_index``。
    partition_index: 分配给本进程本地设备的分区索引。如果有任何进程设置了 ``partition_index``，
      那么所有进程都必须设置。若为 ``None``，分区索引将自动选取。

  Raises:
    RuntimeError: 如果 :func:`~jax.distributed.initialize` 被调用多次，
      或者在后端已经初始化之后才被调用。

  Examples:

  假设有两个 GPU 进程，进程 0 是指定的协调器，
  地址为 ``10.0.0.1:1234``。要初始化 GPU 集群，请在其他任何操作
  之前先运行以下命令。

  在进程 0 上：

  >>> jax.distributed.initialize(coordinator_address='10.0.0.1:1234', num_processes=2, process_id=0)  # doctest: +SKIP

  在进程 1 上：

  >>> jax.distributed.initialize(coordinator_address='10.0.0.1:1234', num_processes=2, process_id=1)  # doctest: +SKIP
  """
  if xla_bridge.backends_are_initialized():
    raise RuntimeError("jax.distributed.initialize() must be called before "
                        "any JAX calls that might initialise the XLA backend. "
                        "This includes any computation, but also calls to jax.devices, jax.device_put, and others.")
  if partition_index is None:
    if slice_index is not None:
      # 弃用于 2025-08-05 添加。应在 3 个月后移除。
      warnings.warn(
          '`slice_index` has been deprecated. Please use `partition_index` instead.',
          DeprecationWarning,
      )
    partition_index = slice_index
  global_state.initialize(coordinator_address, num_processes, process_id,
                          local_device_ids, cluster_detection_method,
                          initialization_timeout, coordinator_bind_address,
                          heartbeat_timeout_seconds=heartbeat_timeout_seconds,
                          shutdown_timeout_seconds=shutdown_timeout_seconds,
                          partition_index=partition_index)


def is_initialized() -> bool:
  """检查 JAX 分布式系统是否已初始化。"""
  return global_state.client is not None

def shutdown():
  """关闭分布式系统。

  如果分布式系统未在运行，则不做任何事。
  """
  global_state.shutdown()
