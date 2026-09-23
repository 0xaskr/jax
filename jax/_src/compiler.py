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

# 文件职责：本模块是 JAX 通往 XLA 运行时的编译器接口层。
# 它把降级得到的 MLIR 模块交给 XLA 客户端编译、加载与序列化，并负责组装
# `CompileOptions`（副本数、分区、设备分配、优化等级与内存适配等级、FDO profile）。
# 主要入口 `compile_or_get_cached` 先查持久化编译缓存，未命中才真正编译；多进程下
# 还可跨主机共享二进制与 FDO profile，并支持 PGLE 采样后的再编译优化。
# 此外还提供 XLA 运行时错误处理钩子，以及缓存命中/未命中和详细编译日志的开关。

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import copy
import enum
from functools import partial
import logging
import time
from typing import Any
import warnings

from jax._src import compilation_cache
from jax._src import config as config
from jax._src import distributed
from jax._src import lib
from jax._src import monitoring
from jax._src import path as pathlib
from jax._src import profiler
from jax._src import traceback_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir
from jax._src.lib import _jax
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
import numpy as np


class CompilerEffortLevel(enum.Enum):
  """XLA 的投入等级（effort level）枚举。

  用于指定 XLA 编译器应针对运行时性能还是内存适配做多大程度的优化，详见
  https://openxla.org/xla/effort_levels。投入等级越高，编译时间越长，但在
  相应维度上应能取得更好的结果。
  """

  UNKNOWN = 0
  O0 = 9
  O1 = 19
  O2 = 29
  O3 = 39

  @classmethod
  def _missing_(cls, value: object) -> CompilerEffortLevel | None:
    return _effort_from_string.get(value)


_effort_from_string: dict[Any, CompilerEffortLevel] = {
    "UNKNOWN": CompilerEffortLevel.UNKNOWN,
    "O0": CompilerEffortLevel.O0,
    "O1": CompilerEffortLevel.O1,
    "O2": CompilerEffortLevel.O2,
    "O3": CompilerEffortLevel.O3,
}
_DISABLE_MOST_OPTIMIZATIONS = config.bool_flag(
    'jax_disable_most_optimizations',
    config.bool_env('JAX_DISABLE_MOST_OPTIMIZATIONS', False),
    'Try not to do much optimization work. This can be useful if the cost of '
    'optimization is greater than that of running a less-optimized program.')

_COMPILER_DETAILED_LOGGING_MIN_OPS = config.int_flag(
    "jax_compiler_detailed_logging_min_ops",
    config.int_env("JAX_COMPILER_DETAILED_LOGGING_MIN_OPS", 10),
    help=(
        'How big should a module be in MLIR operations before JAX enables '
        'detailed compiler logging? The intent of this flag is to suppress '
        'detailed logging for small/uninteresting computations.'
    ),
)

# 特殊的 XLA-AutoFDO profile 版本号：它表示 profile 不可用，
# 因此不应尝试去获取。
_NO_PROFILE_DONT_RETRIEVE = -1

traceback_util.register_exclusion(__file__)

CompileOptions = xc.CompileOptions

logger = logging.getLogger(__name__)


def _add_disabled_hlo_pass(disabled_passes: str, pass_name: str) -> str:
  """向逗号分隔的 pass 列表中添加一个 pass，同时保留已有条目。

  ``DebugOptions.xla_disable_hlo_passes`` 里可能已经存有用户通过
  ``XLA_FLAGS=--xla_disable_hlo_passes=...`` 禁用的 pass，这些不能被覆盖。
  条目会被去掉首尾空白，因为 XLA 对 pass 名称做精确匹配。
  """
  passes = [p.strip() for p in disabled_passes.split(",") if p.strip()]
  if pass_name not in passes:
    passes.append(pass_name)
  return ",".join(passes)


def _get_cross_compile_backend(compile_only_backend):
  """返回真实后端，用来借助真实客户端做交叉编译。

  当通过仅编译客户端做交叉编译时，检查同一平台上是否有可用的真实后端。
  若有则返回它，以便编译过程能利用真实硬件
  （例如用于 GPU kernel 自动调优）。
  """
  platform = compile_only_backend.platform
  try:
    real_backend = xb.get_backend(platform)
  except Exception:
    return None
  # 如果真实后端本身也是仅编译客户端，就不要使用它。
  if isinstance(real_backend, _jax.CompileOnlyPyClient):
    return None
  # 如果平台版本不同，就不要使用真实后端，
  # 否则会导致超时和挂起。
  if real_backend.platform_version != compile_only_backend.platform_version:
    return None
  return real_backend


# 本函数会被 monkeypatch 为获取 XLA-AutoFDO profile 版本的函数。
# 默认实现（-1）负责处理出错的情况。
# TODO(b/289098047): 考虑重构这个接口。
def get_latest_profile_version(backend: xc.Client) -> int:
  del backend
  return -1


def _walk_operations(op, k):
  k -= 1
  if k < 0:
    return k
  for region in op.regions:
    for block in region:
      for child_op in block:
        k = _walk_operations(child_op, k)
        if k < 0:
          return k
  return k


def use_detailed_logging(module: ir.Module) -> bool:
  """如果应为 'module' 启用详细日志，则返回 'true'。"""
  bound = _COMPILER_DETAILED_LOGGING_MIN_OPS.value
  return _walk_operations(module.operation, bound) < 0


def log_persistent_cache_hit(module_name: str, cache_key: str) -> None:
  hit_log_priority = (logging.WARNING if config.log_compiles.value
                      else logging.DEBUG)
  logger.log(hit_log_priority, "Persistent compilation cache hit for '%s' with key %r",
             module_name, cache_key)


def log_persistent_cache_miss(module_name: str, cache_key: str) -> None:
  miss_log_priority = (logging.WARNING
                        if config.explain_cache_misses.value
                        and compilation_cache.is_persistent_cache_enabled()
                        else logging.DEBUG)
  # 全部大写，以便与追踪缓存的 "TRACING CACHE MISS" 保持一致
  logger.log(miss_log_priority, "PERSISTENT COMPILATION CACHE MISS for '%s' with key %r",
             module_name, cache_key)


def get_compile_options(
    num_replicas: int,
    num_partitions: int,
    device_assignment=None,
    env_options_overrides: dict[str, str] | None = None,
    fdo_profile: bytes | None = None,
    detailed_logging: bool = True,
    backend: xc.Client | None = None,
) -> xc.CompileOptions:
  """返回应使用的编译选项，其取值来自各个 flag。

  Args:
    num_replicas: 要编译的副本（replica）数量。
    num_partitions: 要编译的分区数量。
    device_assignment: 可选的 jax 设备 ndarray，表示逻辑副本到物理设备的
      分配（默认沿用 xla_client.CompileOptions）。必须与 `num_replicas`
      和 `num_partitions` 保持一致。
    env_options_overrides: 由编译器解析的额外选项字典
    fdo_profile: 可选的反馈导向优化（FDO）profile，会传给 XLA。
    detailed_logging: 这是否是一个值得关注的计算，XLA 是否值得
      为它记录编译信息？
    backend: 可用的客户端。
  """
  compile_options = xc.CompileOptions()
  compile_options.num_replicas = num_replicas
  compile_options.num_partitions = num_partitions
  build_options = compile_options.executable_build_options
  build_options.use_spmd_partitioning = True
  build_options.use_shardy_partitioner = config.use_shardy_partitioner.value
  if fdo_profile is not None:
    build_options.fdo_profile = fdo_profile
  if device_assignment is not None:
    logger.debug(
        'get_compile_options: num_replicas=%s num_partitions=%s device_assignment=%s',
        num_replicas, num_partitions, device_assignment)
    device_assignment = np.array(device_assignment)

    # 当 num_partitions 为 1 时，允许使用一维的设备分配。
    if (device_assignment.ndim == 1) and (num_partitions == 1):
      device_assignment = device_assignment[:, None]

    if num_replicas != device_assignment.shape[0]:
      msg = 'device_assignment does not match num_replicas: {} vs {}.'
      raise ValueError(msg.format(device_assignment, num_replicas))

    if num_partitions != device_assignment.shape[1]:
      msg = 'device_assignment does not match num_partitions: {} vs {}.'
      raise ValueError(msg.format(device_assignment, num_partitions))

    if device_assignment.dtype == object:
      device_assignment = np.vectorize(lambda d: d.id, otypes=[int])(
          device_assignment)
    device_assignment = xc.DeviceAssignment.create(device_assignment)
    assert device_assignment.replica_count() == num_replicas
    assert device_assignment.computation_count() == num_partitions
    compile_options.device_assignment = device_assignment

  build_options.optimization_level = CompilerEffortLevel(
      config.optimization_level.value
  ).value
  build_options.memory_fitting_level = CompilerEffortLevel(
      config.memory_fitting_level.value
  ).value

  if env_options_overrides is not None:
    # 有些覆盖项是直接作用在 build_options 上的。
    overrides_on_build_options = ["optimization_level", "memory_fitting_level"]

    env_options_overrides = dict(env_options_overrides)
    for name in overrides_on_build_options:
      if name in env_options_overrides:
        setattr(
            build_options,
            name,
            CompilerEffortLevel(env_options_overrides.pop(name)).value,
        )
    compile_options.env_option_overrides = list(env_options_overrides.items())

  debug_options = compile_options.executable_build_options.debug_options
  if lib.cuda_path is not None:
    debug_options.xla_gpu_cuda_data_dir = lib.cuda_path

  if _DISABLE_MOST_OPTIMIZATIONS.value:
    debug_options.xla_backend_optimization_level = 0
    debug_options.xla_llvm_disable_expensive_passes = True
    debug_options.xla_test_all_input_layouts = False

  if not config.enable_remat_opt_pass.value:
    debug_options.xla_disable_hlo_passes = _add_disabled_hlo_pass(
        debug_options.xla_disable_hlo_passes, "rematerialization")

  # XLA-AutoFDO profile 版本的优先级顺序为：
  # 1. 以 --jax_xla_profile_version 的取值为准。
  # 2. 若未设置 --jax_xla_profile_version（即为 0），则调用
  #    get_latest_profile_version 中设置的函数，若非零就使用其返回值。
  #    若该函数返回 0，则置为 -1；这是一种错误。
  # -1 表示后续不应尝试获取最新的 profile。
  jax_xla_profile_version = config.jax_xla_profile_version.value
  if jax_xla_profile_version > 0:
    compile_options.profile_version = jax_xla_profile_version
    logger.debug("get_compile_options XLA-AutoFDO profile: " +
                 "using JAX XLA profile version %d from flag",
                 jax_xla_profile_version)
  else:
    compile_options.profile_version = _NO_PROFILE_DONT_RETRIEVE
    if backend is None:
      logger.info("get_compile_options: no backend supplied; "
                   "disabling XLA-AutoFDO profile")
    else:
      fdo_profile_version = get_latest_profile_version(backend)
      if fdo_profile_version != 0:
        compile_options.profile_version = fdo_profile_version
        logger.debug("get_compile_options XLA-AutoFDO profile: " +
                     "using XLA-AutoFDO profile version %d",
                     fdo_profile_version)
      else:
        logger.error("get_compile_options XLA-AutoFDO profile: " +
                     "XLA-AutoFDO profile version is 0; this should not happen")

  debug_options.xla_detailed_logging = detailed_logging

  # 如果启用了持久化缓存，还要同时开启额外的 XLA 缓存特性。
  if compilation_cache.is_persistent_cache_enabled():
    # 这里的 compilation_cache_dir 不可能是 None，只是类型检查器比较严格。
    path = pathlib.Path(config.compilation_cache_dir.value or "")
    enabled_flags = config.persistent_cache_enable_xla_caches.value or ""

    if enabled_flags == "all" or "xla_gpu_kernel_cache_file" in enabled_flags:
      kernel_cache_path = path / "xla_gpu_kernel_cache_file"
      debug_options.xla_gpu_kernel_cache_file = str(kernel_cache_path)
      logger.debug("Enabling XLA kernel cache at '%s'", kernel_cache_path)

    if enabled_flags == "all" or "xla_gpu_per_fusion_autotune_cache_dir" in enabled_flags:
      autotune_cache_path = path / "xla_gpu_per_fusion_autotune_cache_dir"
      debug_options.xla_gpu_per_fusion_autotune_cache_dir = str(autotune_cache_path)
      logger.debug("Enabling XLA autotuning cache at '%s'", autotune_cache_path)

      # 设置缓存模式，使得只有进程 0 可以向缓存写入。
      if distributed.global_state.process_id == 0:
        debug_options.xla_gpu_experimental_autotune_cache_mode = xc.AutotuneCacheMode.UPDATE
      else:
        debug_options.xla_gpu_experimental_autotune_cache_mode = xc.AutotuneCacheMode.READ

  return compile_options


@profiler.annotate_function
def backend_compile_and_load(
    backend: xc.Client,
    module: ir.Module,
    executable_devices: xc.DeviceList,
    options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
) -> xc.LoadedExecutable:
  sym_name = module.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value

  if (options.executable_build_options.fdo_profile is not None
      and len(options.executable_build_options.fdo_profile)):
    logger.debug(
        "Compiling module %s with FDO profile of length %d",
        module_name,
        len(options.executable_build_options.fdo_profile),
    )

  try:
    # 这里通过一次单独的函数调用，确保 XLA 编译过程
    # 在 Python 性能分析结果中单独出现
    # TODO(dsuo): 等删除 _jax.CompileOnlyPyClient 之后简化这段逻辑。
    if isinstance(backend, _jax.CompileOnlyPyClient):
      # 当同一平台上存在可用的真实后端时，用它来做交叉编译。真实客户端
      # 提供硬件访问能力（例如用于 GPU kernel 自动调优），而拓扑结构
      # 则指定目标设备。
      cross_compile_backend = _get_cross_compile_backend(backend)
      if cross_compile_backend is not None and not host_callbacks:
        cross_compile_topology = xc.get_topology_for_devices(backend.devices())
        return cross_compile_backend.compile(
            module,
            topology=cross_compile_topology,
            compile_options=options,
        )
      if host_callbacks:
        return backend.compile(  # pyrefly: ignore[bad-return]
            module,
            executable_devices=executable_devices,
            compile_options=options,
            host_callbacks=host_callbacks,
        )
      # 有些后端还不支持 `host_callbacks` 选项
      # TODO(sharadmv): 当所有后端都允许 `compile` 接收
      # `host_callbacks` 之后，删除这个回退分支
      return backend.compile(  # pyrefly: ignore[bad-return]
          module,
          executable_devices=executable_devices,
          compile_options=options,
      )
    else:
      if host_callbacks:
        return backend.compile_and_load(
            module,
            executable_devices=executable_devices,
            compile_options=options,
            host_callbacks=host_callbacks,
        )
      # 有些后端还不支持 `host_callbacks` 选项
      # TODO(sharadmv): 当所有后端都允许 `compile` 接收
      # `host_callbacks` 之后，删除这个回退分支
      return backend.compile_and_load(
          module,
          executable_devices=executable_devices,
          compile_options=options,
      )
  except _jax.JaxRuntimeError as e:
    for error_handler in _XLA_RUNTIME_ERROR_HANDLERS:
      handler_result = error_handler(e)
      if handler_result is not None:
        raise handler_result from e
    raise e


_XLA_RUNTIME_ERROR_HANDLERS = []


def register_xla_runtime_error_handler(
    handler_fn: Callable[[_jax.JaxRuntimeError], Exception | None],
):
  """为 XLA 运行时错误注册自定义异常处理器。

  注册自定义处理器后，可以在遇到 XLARuntimeError 之后重新抛出信息更丰富的
  异常。

  Args:
    handler_fn: 一个函数，返回用于替换原始 XLA 运行时错误的新异常；若原始
      错误应当继续传播则返回 None。

  Returns:
    一个新的异常，或 None。
  """
  _XLA_RUNTIME_ERROR_HANDLERS.append(handler_fn)


def compile_or_get_cached(
    backend: xc.Client,
    computation: ir.Module,
    devices: np.ndarray,
    compile_options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
    executable_devices: xc.DeviceList,
    pgle_profiler: profiler.PGLEProfiler | None = None,
) -> xc.LoadedExecutable:
  sym_name = computation.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value

  if dumped_to := mlir.dump_module_to_file(computation, "compile"):
    logger.info("Dumped the module to %s.", dumped_to)

  is_multi_process = (
      len({device.process_index for device in devices.flatten()}) > 1
  )
  min_device_process_id = min(
      devices.flatten(), key=lambda device: device.id
  ).process_index

  # cache_key：如果编译缓存被禁用，它可能为 None
  cache_key, compile_options = _resolve_compilation_strategy(
    computation,
    devices,
    compile_options,
    backend,
    pgle_profiler,
    is_multi_process,
    module_name,
    min_device_process_id,
  )

  if cache_key is None:
    return backend_compile_and_load(
        backend, computation, executable_devices, compile_options,
        host_callbacks)

  monitoring.record_event('/jax/compilation_cache/compile_requests_use_cache')

  cache_retrieval_start = time.monotonic()
  retrieved_executable, retrieved_compile_time = _cache_read(
      module_name, cache_key, compile_options, backend, executable_devices,
      host_callbacks)
  cache_retrieval_time = time.monotonic() - cache_retrieval_start

  if retrieved_executable is not None:
    assert retrieved_compile_time is not None
    log_persistent_cache_hit(module_name, cache_key)

    monitoring.record_event('/jax/compilation_cache/cache_hits')
    monitoring.record_event_duration_secs(
        '/jax/compilation_cache/compile_time_saved_sec',
        retrieved_compile_time - cache_retrieval_time)

    monitoring.record_event_duration_secs(
        "/jax/compilation_cache/cache_retrieval_time_sec", cache_retrieval_time)

    return retrieved_executable
  util.test_event("compile_after_persistent_compilation_miss")
  if (
      config.share_binary_between_hosts.value
      and is_multi_process
      and distributed.global_state.client is not None
  ):
    log_persistent_cache_miss(module_name, cache_key)
    return _compile_and_share_module(
        backend,
        computation,
        executable_devices,
        compile_options,
        host_callbacks,
        distributed.global_state.client,
        module_name,
        cache_key,
        min_device_process_id
    )
  else:
    log_persistent_cache_miss(module_name, cache_key)
    return _compile_and_write_cache(
        backend,
        computation,
        executable_devices,
        compile_options,
        host_callbacks,
        module_name,
        cache_key,
    )


# 启用 PGLE 时，可能出现 3 种情况：
# 1. PGLE 优化后的模块（即用 FDO profile 重新编译过的那个模块）已经在持久化
# 缓存中。此时应从缓存返回该模块，并对该模块禁用 PGLE。该模块存放在持久化
# 缓存的 "pgle_optimized_cache_key" 键下，这个键由 FDO profile 替换为一个
# 哨兵值计算得到，该哨兵值标识出这个模块是用 PGLE 优化过的。
# 2. PGLE 采样过的模块不在持久化缓存中，且该模块正带着 FDO profile 构建。
# 此时我们需要把 FDO profile 分享给其他进程，并把结果存放在
# "pgle_optimized_cache_key" 下，这样之后在情况 1 中就能找到该模块。
# 3. PGLE 采样过的模块不在持久化缓存中，且该模块正被编译为待 PGLE 优化的
# 版本（FDO profile 为空）。此时如果持久化缓存中存在非 PGLE 采样的模块，
# 我们只需直接返回它；否则就编译它。
#
# 如果设置了 compilation_cache_expect_pgle 选项，那么在情况 1 中即使当前
# 进程未启用 PGLE，也会加载 PGLE 优化后的模块。当我们想把 PGLE 与其他性能
# 分析工具（例如 Nsight Systems）结合使用时这很有用，因为这些工具会争用
# CUPTI 资源，无法与 PGLE 共存。
def _resolve_compilation_strategy(
    computation: ir.Module,
    devices: np.ndarray,
    compile_options: xc.CompileOptions,
    backend: xc.Client,
    pgle_profiler: profiler.PGLEProfiler | None,
    is_multi_process: bool,
    module_name: str,
    min_device_process_id: int,
) -> tuple[str | None, xc.CompileOptions]:
  is_auto_pgle_used = (
      config.enable_pgle.value and config.pgle_profiling_runs.value > 0
  )

  get_cache_key = partial(_get_cache_key, backend=backend,
                          computation=computation, devices=devices)

  if is_auto_pgle_used or config.compilation_cache_expect_pgle.value:
    # 如果 cache key 生成失败，它可能为 None。
    pgle_optimized_cache_key = get_cache_key(compile_options,
                                             override_fdo_profile=b"pgle profiled")
    # TODO(b/376647494): 该 bug 修复后移除这个变通方案；如果启用了
    # command buffer / CUDA graph，JAX profiler 就无法为 PGLE 采集到足够
    # 详细的 profile 数据。因此在为 PGLE 数据采集而编译时要禁用 command
    # buffer，但在 AutoPGLE 未启用时不禁用，在用 PGLE 数据重新编译时也不禁用。
    # 这一条件包含 `compilation_cache_expect_pgle`，这样那些编译很慢、
    # 执行次数不足以触发重新编译的模块，仍然能在 "enable_pgle" 运行与
    # "expect_pgle" 运行之间被缓存下来。
    first_pass_compile_options = copy.deepcopy(compile_options)
    first_pass_compile_options.env_option_overrides += [
      ("xla_gpu_enable_command_buffer", ""),
    ]
  else:
    pgle_optimized_cache_key = None
    first_pass_compile_options = compile_options

  # 如果 cache key 生成失败或缓存被禁用，它可能为 None
  cache_key = get_cache_key(first_pass_compile_options)

  if cache_key is not None and pgle_optimized_cache_key is not None:
    # 编译缓存已启用，且 AutoPGLE 已启用或已预期使用
    if _is_executable_in_cache(backend, pgle_optimized_cache_key):
      if config.compilation_cache_expect_pgle.value:
        logger.info(f"PGLE-optimized {module_name} loaded from compilation cache")
      # 这种情况下不需要再记录 N 次 profile 采样
      if pgle_profiler is not None:
        pgle_profiler.disable()
      return pgle_optimized_cache_key, compile_options
    elif (config.compilation_cache_expect_pgle.value
          and _is_executable_in_cache(backend, cache_key)):
      # 在持久化缓存中没有找到 PGLE 优化后的模块，
      # 而用户（通过 expect_pgle）断言这次未命中是不应发生的
      warnings.warn(f"PERSISTENT CACHE MISS for PGLE-optimized {module_name} "
                    "despite non-PGLE hit; it may not have been executed "
                    "enough times when the cache was populated")

  if (is_auto_pgle_used
      and compile_options.executable_build_options.fdo_profile is not None
      and len(compile_options.executable_build_options.fdo_profile)):
    # 已有 profile 数据，可以触发 PGLE 优化的重新编译；
    # 如果缓存已启用，就把结果存到 `pgle_optimized_cache_key` 下
    if is_multi_process and distributed.global_state.client is not None:
      compile_options.executable_build_options.fdo_profile = (
        _share_fdo_profiles(
            computation,
            devices,
            compile_options,
            backend,
            distributed.global_state.client,
            min_device_process_id,
        )
      )
    return pgle_optimized_cache_key, compile_options
  else:
    # 为 PGLE 采样而编译；如果缓存已启用，就把结果存到 `cache_key` 下。
    # 这也是 AutoPGLE 被禁用时走的路径。
    return cache_key, first_pass_compile_options

def _get_cache_key(
    options: xc.CompileOptions,
    backend: xc.Client,
    computation: ir.Module,
    devices: np.ndarray,
    override_fdo_profile: bytes | None = None) -> str | None:
  if not compilation_cache.is_cache_used(backend):
    return None
  if override_fdo_profile is not None:
    options = copy.deepcopy(options)
    options.executable_build_options.fdo_profile = override_fdo_profile
  try:
    return compilation_cache.get_cache_key(
        computation,
        devices,
        options,
        backend,
        config.remove_custom_partitioning_ptr_from_cache_key.value,
    )
  except _jax.JaxRuntimeError as ex:
    logger.error("compile_or_get_cached: unable to generate cache key, "
                  "skipping the cache: %s", ex)
  return None

# 拥有最小设备 ID 的进程应当在编译之前
# 把 FDO profile 分享给其他进程。
def _share_fdo_profiles(
    computation: ir.Module,
    devices: np.ndarray,
    compile_options: xc.CompileOptions,
    backend: xc.Client,
    global_client: lib._jax.DistributedRuntimeClient,
    min_process_id
) -> bytes:
  sym_name = computation.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value
  fdo_profile = compile_options.executable_build_options.fdo_profile
  if len(fdo_profile) == 0:
    return fdo_profile

  compile_options.executable_build_options.fdo_profile = b""
  try:
    profile_key = (
        compilation_cache.get_cache_key(
            computation,
            devices,
            compile_options,
            backend,
            ignore_custom_partitioning=True,
        )
        + "_fdo_sync"
    )
  except _jax.JaxRuntimeError as ex:
    logger.error(
        "compile_or_get_cached: unable to generate cache key, "
        "skipping the fdo profile sharing: %s",
        ex,
    )
    return fdo_profile

  if profile_key in _share_fdo_profiles.modules_profiles:  # pyrefly: ignore[missing-attribute]
    return _share_fdo_profiles.modules_profiles[profile_key]  # pyrefly: ignore[missing-attribute]

  share_timeout = config.share_binary_between_hosts_timeout_ms.value
  if distributed.global_state.process_id == min_process_id:
    logger.debug(
        "Module %s. Sharing FDO profile. Process %d.",
        module_name,
        min_process_id,
    )
    global_client.key_value_set_bytes(profile_key, fdo_profile)
  else:
    logger.debug(
        "Module %s. Waiting for FDO profile which should be set by process %d.",
        module_name,
        min_process_id,
    )
    fdo_profile = global_client.blocking_key_value_get_bytes(
        profile_key, share_timeout
    )

  _share_fdo_profiles.modules_profiles[profile_key] = fdo_profile  # pyrefly: ignore[missing-attribute]
  return fdo_profile


_share_fdo_profiles.modules_profiles = {}  # pyrefly: ignore[missing-attribute]


# first_process_id 对应的进程应当编译该模块，
# 并把它写入 K-V 存储。
def _compile_and_share_module(
    backend: xc.Client,
    computation: ir.Module,
    executable_devices: xc.DeviceList,
    compile_options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
    global_client: lib._jax.DistributedRuntimeClient,
    module_name: str,
    cache_key: str,
    first_process_id: int
) -> xc.LoadedExecutable:
  share_timeout = config.share_binary_between_hosts_timeout_ms.value

  if cache_key in _compile_and_share_module.modules_cache:  # pyrefly: ignore[missing-attribute]
    return _compile_and_share_module.modules_cache[cache_key]  # pyrefly: ignore[missing-attribute]

  if distributed.global_state.process_id == first_process_id:
    logger.debug("Process %d compiling and sharing module: %s",
                 first_process_id, module_name)
    executable = _compile_and_write_cache(
        backend,
        computation,
        executable_devices,
        compile_options,
        host_callbacks,
        module_name,
        cache_key,
    )
    serialized_executable = backend.serialize_executable(executable)
    serialized_executable = compilation_cache.compress_executable(
        serialized_executable
    )
    global_client.key_value_set_bytes(cache_key, serialized_executable)
  else:
    logger.debug("Waiting for module: %s from process %d", module_name,
                 first_process_id)
    serialized_executable = global_client.blocking_key_value_get_bytes(
        cache_key, share_timeout
    )
    serialized_executable = compilation_cache.decompress_executable(
        serialized_executable
    )
    executable = backend.deserialize_executable(
        serialized_executable, executable_devices, compile_options,
        host_callbacks)

  _compile_and_share_module.modules_cache[cache_key] = executable  # pyrefly: ignore[missing-attribute]
  return executable


_compile_and_share_module.modules_cache = {}  # pyrefly: ignore[missing-attribute]


def _compile_and_write_cache(
    backend: xc.Client,
    computation: ir.Module,
    executable_devices: xc.DeviceList,
    compile_options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
    module_name: str,
    cache_key: str,
) -> xc.LoadedExecutable:
  start_time = time.monotonic()
  executable = backend_compile_and_load(
      backend, computation, executable_devices, compile_options, host_callbacks
  )
  compile_time = time.monotonic() - start_time
  _cache_write(cache_key, compile_time, module_name, backend, executable)
  return executable


def _should_raise_persistent_cache_error(ex: Exception) -> bool:
  """如果该异常应当抛出则返回 True，若应当发出警告则返回 False。"""
  return (
      config.raise_persistent_cache_errors.value or
      isinstance(ex, compilation_cache.CacheVerificationError)
  )


def _is_executable_in_cache(backend, cache_key) -> bool:
  """检查在给定的键上，缓存中是否已存在可执行文件
  """
  try:
    return compilation_cache.is_executable_in_cache(backend, cache_key)
  except Exception as ex:
    if _should_raise_persistent_cache_error(ex):
      raise
    warnings.warn(
        f"Error reading persistent compilation cache entry for "
        f"'{cache_key}': {type(ex).__name__}: {ex}")
    return False


def _cache_read(
    module_name: str, cache_key: str, compile_options: xc.CompileOptions,
    backend: xc.Client, executable_devices: xc.DeviceList,
    host_callbacks: Sequence[Any],
) -> tuple[xc.LoadedExecutable | None, int | None]:
  """在持久化编译缓存仓库中查找 `computation` 及其编译时间。
  """
  try:
    return compilation_cache.get_executable_and_time(
        cache_key, compile_options, backend, executable_devices,
        host_callbacks)
  except Exception as ex:
    if _should_raise_persistent_cache_error(ex):
      raise
    warnings.warn(
        f"Error reading persistent compilation cache entry for "
        f"'{module_name}': {type(ex).__name__}: {ex}")
    return None, None


def _cache_write(cache_key: str,
                 compile_time_secs: float,
                 module_name: str,
                 backend: xc.Client,
                 executable: xc.LoadedExecutable) -> None:
  """把 `serialized_computation` 及其编译时间写入持久化编译缓存仓库。
  """
  # 只从第一个进程写入缓存条目。否则在某些文件系统（例如 GCS）上
  # 会产生写入争用问题。
  log_priority = (logging.WARNING
                  if config.explain_cache_misses.value
                  and compilation_cache.is_persistent_cache_enabled()
                  else logging.DEBUG)
  if distributed.global_state.process_id != 0:
    logger.log(log_priority,
               "Not writing persistent cache entry since process_id != 0")
    return

  min_compile_time = config.persistent_cache_min_compile_time_secs.value
  if compile_time_secs < min_compile_time:
    logger.log(
        log_priority,
        "Not writing persistent cache entry for '%s' because it took < %.2f "
        "seconds to compile (%.2fs)", module_name, min_compile_time,
        compile_time_secs)
    return
  else:
    logger.debug(
        "'%s' took at least %.2f seconds to compile (%.2fs)",
        module_name, min_compile_time, compile_time_secs)

  try:
    compilation_cache.put_executable_and_time(
        cache_key, module_name, executable, backend, int(compile_time_secs))
  except Exception as ex:
    if _should_raise_persistent_cache_error(ex):
      raise
    warnings.warn(
        f"Error writing persistent compilation cache entry for "
        f"'{module_name}': {type(ex).__name__}: {ex}")
