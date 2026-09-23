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

# 文件职责：为 XLA 编译缓存生成稳定且可复现的缓存键（cache key）。
# `get()` 汇集输入 IR、jaxlib 与后端版本、XLA flags、编译选项、
# 加速器拓扑与压缩算法等条目，逐项哈希后返回 `<模块名>-<sha256>`。
# 生成前会对 IR 做规范化：清除调试信息、剥离 Mosaic 内核字节码中的
# 源文件路径与自定义分区回调指针，避免非确定性内容造成缓存未命中。
# 环境变量与命令行中与编译结果无关的 XLA dump 类 flags 会被排除，
# GPU 多进程场景下还会剔除设备分配以保证跨进程缓存键一致。

import base64
import copy
import hashlib
import io
import json
import logging
import os
import sys
from typing import cast as type_cast

from jax._src import config
from jax._src.lib import version_str as jaxlib_version_str
from jax._src.lib import _jax
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager as pm
import numpy as np


logger = logging.getLogger(__name__)

_extra_flag_prefixes: list[str] = []

def add_flag_prefixes(flag_prefixes: list[str]) -> None:
  """添加要纳入缓存键的 flag 前缀。请在调用 get() 之前调用。
  """
  global _extra_flag_prefixes
  _extra_flag_prefixes += flag_prefixes


def clear_flag_prefixes() -> None:
  """清除由 add_flag_prefixes() 添加的 flag 前缀。
  """
  global _extra_flag_prefixes
  _extra_flag_prefixes = []


def get_flag_prefixes() -> list[str]:
  """返回由 add_flag_prefixes() 添加的 flag 前缀。
  """
  return _extra_flag_prefixes


def custom_hook() -> str:
  """用于向缓存键追加内容的自定义钩子。

  每次调用 get() 时都会调用该自定义钩子，可将其定义为返回一个字符串，
  该字符串会被哈希进缓存键。
  """
  return ""


def get(
    module: ir.Module,
    devices: np.ndarray,
    compile_options: xla_client.CompileOptions,
    backend: xla_client.Client,
    compression_algorithm: str = "zstandard",
    ignore_custom_partitioning: bool = False,
) -> str:
  """生成一个哈希字符串，用作编译缓存的键。

  依据各参数创建一个缓存键，它是唯一哈希的十六进制编码字符串。
  该十六进制编码字符串长度为 256 个字符。

  Args:
    module: 输入程序
    devices: 程序将在其上运行的加速器设备数组
    compile_options: 传给 XLA 编译器的选项
    backend: 平台描述（例如 TPU 版本）
    compression_algorithm: 表示可执行文件在持久化到缓存之前所用压缩
      算法的字符串
    ignore_custom_partitioning: 是否从计算中移除 custom_partitioning
      回调指针。

  典型返回值示例：
   'jit__psum-14ac577cdb2ef6d986078b4054cc9893a9a14a16dbb0d8f37b89167c1f1aacdf'
  """
  entries = [
      (
          "computation",
          lambda hash_obj: _hash_computation(
              hash_obj, module, ignore_custom_partitioning
          ),
      ),
      (
          "jax_lib version",
          lambda hash_obj: hash_obj.update(
              bytes(jaxlib_version_str.encode("utf-8"))
          ),
      ),
      (
          "backend version",
          lambda hash_obj: _hash_platform(hash_obj, backend)
      ),
      (
          "XLA flags",
          lambda hash_obj: _hash_xla_flags(hash_obj, get_flag_prefixes()),
      ),
      (
          "compile_options",
          lambda hash_obj: _hash_serialized_compile_options(
              hash_obj,
              compile_options,
              # 在 GPU 多进程任务中，需要剔除设备分配，以便把缓存键
              # 用作各进程之间的不变量。
              strip_device_assignment=(backend.platform == "gpu"),
          ),
      ),
      (
          "accelerator_config",
          lambda hash_obj: _hash_accelerator_config(hash_obj, devices),
      ),
      (
          "compression",
          lambda hash_obj: _hash_string(hash_obj, compression_algorithm),
      ),
      ("custom_hook", lambda hash_obj: _hash_string(hash_obj, custom_hook())),
  ]

  hash_obj = hashlib.sha256()
  for name, hashfn in entries:
    hashfn(hash_obj)
    _log_cache_key_hash(hash_obj, name, hashfn)
  sym_name = module.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value
  return module_name + "-" + hash_obj.digest().hex()


def _log_cache_key_hash(hash_obj, last_serialized: str, hashfn):
  if logger.isEnabledFor(logging.DEBUG):
    # 只记录该条目自身的哈希值
    fresh_hash_obj = hashlib.sha256()
    hashfn(fresh_hash_obj)
    logger.debug(
        "get_cache_key hash of serialized %s: %s",
        last_serialized,
        fresh_hash_obj.digest().hex(),
    )
    # 记录累计哈希值
    logger.debug(
        "get_cache_key hash after serializing %s: %s",
        last_serialized,
        hash_obj.digest().hex(),
    )


def _remove_custom_partitioning_callbacks(m: ir.Module):
  """从预编译 IR 中移除 custom_partitioning 回调指针。

  Python 函数指针在不同次执行之间不是确定性的。
  """
  def _update_bc_attribute(op: ir.Operation) -> ir.WalkResult:
    if "call_target_name" not in op.attributes:
      return ir.WalkResult.ADVANCE
    call_target_name = op.attributes["call_target_name"]
    assert isinstance(call_target_name, ir.StringAttr)
    if (
        op.name == "stablehlo.custom_call"
        and call_target_name.value == "CustomSPMDPartitioning"
    ):
      op.attributes["backend_config"] = ir.StringAttr.get("REMOVED")
    return ir.WalkResult.ADVANCE

  m.operation.walk(_update_bc_attribute)
  return m


def _strip_mosaic_debug_info(m: ir.Module) -> None:
  """剥离 tpu_custom_call 算子中 Mosaic 内核字节码的调试信息。

  顶层的 strip-debuginfo pass 不会深入到 backend_config 内嵌的序列化内核
  MLIR，因此源文件路径会泄漏进缓存键。
  """
  try:
    from jax._src.lib import tpu  # pylint: disable=g-import-not-at-top
  except ImportError:
    return

  def _strip_kernel(op: ir.Operation) -> ir.WalkResult:
    if (op.name != "stablehlo.custom_call"
        or op.attributes["call_target_name"].value != "tpu_custom_call"):  # type: ignore
      return ir.WalkResult.ADVANCE
    bc = json.loads(op.attributes["backend_config"].value)  # type: ignore
    body = bc.get("custom_call_config", {}).get("body")
    if not body:
      return ir.WalkResult.ADVANCE
    ctx = m.context
    tpu.register_dialect(ctx)
    ctx.allow_unregistered_dialects = True
    with ctx:
      try:
        kernel = ir.Module.parse(base64.b64decode(body), context=ctx)
      except ir.MLIRError:
        return ir.WalkResult.ADVANCE
      pm.PassManager.parse("builtin.module(strip-debuginfo)").run(
          kernel.operation)
      out = io.BytesIO()
      kernel.operation.write_bytecode(out)
    bc["custom_call_config"]["body"] = base64.b64encode(
        out.getvalue()).decode()
    op.attributes["backend_config"] = ir.StringAttr.get(
        json.dumps(bc, separators=(",", ":")))
    return ir.WalkResult.ADVANCE

  m.operation.walk(_strip_kernel)


def _serialize_ir(m: ir.Module, ignore_custom_partitioning: bool) -> bytes:
  output = io.BytesIO()
  if ignore_custom_partitioning:
    m = _remove_custom_partitioning_callbacks(
        type_cast(ir.Module, m.operation.clone(ip=False))
    )
  m.operation.write_bytecode(file=output)
  return output.getvalue()


def _canonicalize_ir(
    m_original: ir.Module, ignore_custom_partitioning: bool
) -> bytes:
  with m_original.context:
    m = type_cast(ir.Module, m_original.operation.clone(ip=False))
    passes = pm.PassManager.parse(
        "builtin.module(strip-debuginfo)"
    )
    passes.run(m.operation)
    _strip_mosaic_debug_info(m)
    return _serialize_ir(m, ignore_custom_partitioning)


def _hash_computation(hash_obj, module, ignore_custom_partitioning: bool):
  if config.compilation_cache_include_metadata_in_key.value:
    canonical_ir = _serialize_ir(module, ignore_custom_partitioning)
  else:
    canonical_ir = _canonicalize_ir(module, ignore_custom_partitioning)
  hash_obj.update(canonical_ir)


def _hash_devices(hash_obj, devices: np.ndarray) -> None:
  for device in devices.flat:
    _hash_string(hash_obj, device.device_kind)


def _hash_accelerator_config(hash_obj, accelerators: np.ndarray):
  accelerator_devices = []
  for accelerator in accelerators.flat:
    accelerator_devices.append(accelerator)
  try:
    topology = xla_client.get_topology_for_devices(accelerator_devices)
    hash_obj.update(topology.fingerprint().to_bytes(8, byteorder="big"))
  except _jax.JaxRuntimeError as ex:
    # 对那些尚不支持序列化 PjRtTopologyDescription 的后端做回退。
    logger.info("get (_hash_accelerator_config): unable to hash "
                "accelerator config, falling back to hashing "
                "devices %s (type %s)", ex, type(ex))
    _hash_devices(hash_obj, accelerators)

# LINT.IfChange(xla_flags)
xla_flags_to_exclude_from_cache_key = [
    "--xla_dump_compress_protos",
    "--xla_dump_module_metadata",
    "--xla_dump_max_hlo_modules",
    "--xla_dump_include_timestamp",
    "--xla_dump_hlo_pass_re",
    "--xla_dump_hlo_module_re",
    "--xla_dump_hlo_snapshots",
    "--xla_dump_fusion_visualization",
    "--xla_dump_hlo_as_url",
    "--xla_dump_hlo_as_proto",
    "--xla_dump_hlo_as_text",
    "--xla_dump_hlo_as_long_text",
    "--xla_dump_hlo_as_html",
    "--xla_dump_hlo_as_dot",
    "--xla_dump_to",
    "--xla_force_host_platform_device_count",
    "--xla_dump_disable_metadata",
    "--xla_dump_hlo_pipeline_re",
    "--xla_tpu_sdc_checker_streamz_metric",
    "--xla_tpu_sdc_checker_enable_sdc_event_callbacks",
    "--xla_tpu_sdc_checker_enable_coresweep_ng_callbacks",
    "--xla_tpu_sdc_checker_no_logging_if_callbacks_are_present",
    "--xla_gpu_cuda_data_dir",
    "--xla_gpu_experimental_autotune_cache_mode",
    "--xla_gpu_per_fusion_autotune_cache_dir",
    "--xla_tpu_compiler_variant",
]

env_override_flags_to_exclude_from_cache_key = {
    x.strip("-") for x in xla_flags_to_exclude_from_cache_key
}
# LINT.ThenChange(:debug_options)

def _hash_serialized_compile_options(hash_obj, compile_options_obj,
                                     strip_device_assignment=False):
  # 不要改动原始的 CompileOptions 对象，因为它会被传给编译器。
  # 为生成缓存键创建一个深拷贝。
  compile_options_copy = copy.deepcopy(compile_options_obj)

  # 某些调试选项不影响编译结果，因此不应成为缓存键的一部分，
  # 否则会导致不必要的缓存未命中。这里通过把布尔值设为 False、整数设为 0、
  # 字符串设为空来清除它们。只要每个字段每次都使用相同的值，
  # 具体用什么值来清除并不重要。
  debug_options = compile_options_copy.executable_build_options.debug_options
  # LINT.IfChange(debug_options)
  debug_options.xla_force_host_platform_device_count = 0
  debug_options.xla_dump_to = ""
  debug_options.xla_dump_hlo_module_re = ""
  debug_options.xla_dump_hlo_pass_re = ""
  debug_options.xla_dump_hlo_as_text = False
  debug_options.xla_dump_hlo_as_proto = False
  debug_options.xla_dump_hlo_as_dot = False
  debug_options.xla_dump_hlo_as_url = False
  debug_options.xla_dump_hlo_as_html = False
  debug_options.xla_dump_fusion_visualization = False
  debug_options.xla_dump_hlo_snapshots = False
  debug_options.xla_dump_max_hlo_modules = False
  debug_options.xla_dump_module_metadata = False
  debug_options.xla_dump_compress_protos = False
  debug_options.xla_dump_hlo_as_long_text = False
  debug_options.xla_dump_disable_metadata = False
  debug_options.xla_dump_hlo_pipeline_re = ""
  debug_options.xla_gpu_experimental_autotune_cache_mode = 0

  # 可选地指定编译器使用的 cuda 安装路径。
  # 它可能影响编译所用的 cuda 版本，但这一点本应已经包含在平台信息中
  # （而且也可能并不反映在 cuda 路径上，因为这里只对目录名做哈希而不对
  # 内容做哈希）。若 cuda 路径跨多次运行发生变化而版本相同，
  # 它还会造成无谓的缓存未命中，因此我们在此清除它。
  debug_options.xla_gpu_cuda_data_dir = ""
  debug_options.xla_gpu_per_fusion_autotune_cache_dir = ""
  # LINT.ThenChange(:xla_flags)

  compile_options_copy.env_option_overrides = [
      flag_value
      for flag_value in compile_options_copy.env_option_overrides
      if flag_value[0] not in env_override_flags_to_exclude_from_cache_key
  ]
  if strip_device_assignment and compile_options_copy.device_assignment:
    replica_count = compile_options_copy.device_assignment.replica_count()
    computation_count = compile_options_copy.device_assignment.computation_count()
    compile_options_copy.device_assignment = xla_client.DeviceAssignment.create(
        np.arange(replica_count * computation_count).reshape(  # pyrefly: ignore[bad-argument-type]
          [replica_count, computation_count])
    )
  return hash_obj.update(compile_options_copy.SerializeAsString())


def _hash_platform(hash_obj, backend):
  _hash_string(hash_obj, backend.platform)
  _hash_string(hash_obj, backend.platform_version)


def _hash_xla_flags(hash_obj, extra_flag_prefixes: list[str]):
  xla_flags = []

  xla_flags_env_var = os.getenv("XLA_FLAGS")
  if xla_flags_env_var:
    xla_flags.extend(xla_flags_env_var.split())
  libtpu_init_args_env_var = os.getenv("LIBTPU_INIT_ARGS")
  if libtpu_init_args_env_var:
    xla_flags.extend(libtpu_init_args_env_var.split())

  for arg in sys.argv:
    if arg.startswith("--xla") or any(
        arg.startswith(p) for p in extra_flag_prefixes
    ):
      xla_flags.append(arg)

  # 注意：所有带参数的 XLA flag 都必须使用 '=' 而不能用空格
  # （例如 --xla_force_host_platform_device_count=8）（我想是这样）。
  for flag in sorted(xla_flags):
    if flag.split("=")[0] in xla_flags_to_exclude_from_cache_key:
      logger.debug("Not including XLA flag in cache key: %s", flag)
      continue
    logger.debug("Including XLA flag in cache key: %s", flag)
    _hash_string(hash_obj, flag)


def _hash_string(hash_obj, str_var):
  hash_obj.update(str_var.encode("utf-8").strip())
