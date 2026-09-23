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

# 文件职责：原语分派与 jit 分派，是 JAX 把抽象计算落到具体设备并执行的最后一环。
# 一方面为每个原语生成“单步编译并运行”的实现（op-by-op 即时执行），
# 另一方面实现 `device_put` 原语，负责数组的跨设备、跨主机搬运与重分片
# （含分片提交状态、拷贝语义与 DCN 传输回退）。
# 同时用 `RuntimeTokenSet` 维护有序效果（effect）的 token 顺序，保证多次分派的
# 副作用按序发生，并提供编译耗时的事件与日志记录工具。

# 原语分派与 jit 分派。
from __future__ import annotations

import atexit
from collections.abc import Sequence
import dataclasses
from functools import partial
import logging
import threading
import time
from typing import Any

from jax._src import api
from jax._src import array
from jax._src import basearray
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import literals
from jax._src import pjit
from jax._src import traceback_util
from jax._src import util

from jax._src import xla_bridge as xb
from jax._src.abstract_arrays import array_types
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval
from jax._src.interpreters import pxla
from jax._src.api_util import InternalFloatingPointError
from jax._src.layout import Layout, Format
from jax._src.lib import _jax
from jax._src.lib import xla_client as xc
from jax._src.mesh import AbstractMesh, Mesh
from jax._src.monitoring import record_scalar, record_event_duration_secs, record_event_time_span
from jax._src.partition_spec import PartitionSpec, UnreducedKind
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (
    NamedSharding, make_single_device_sharding, GSPMDSharding)
from jax._src.stages import SourceInfo
import numpy as np


JAXPR_TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"
JAXPR_TO_MLIR_MODULE_EVENT = "/jax/core/compile/jaxpr_to_mlir_module_duration"
BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"

traceback_util.register_exclusion(__file__)

Backend = _jax.Client
Device = xc.Device
ArrayCopySemantics = xc.ArrayCopySemantics

CompileOptions = xc.CompileOptions

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

logger = logging.getLogger(__name__)

# 该标志在退出时被置位；此时不应再尝试记录日志
_on_exit = False

### op-by-op execution

def apply_primitive(prim, *args, **params):
  """实现规则：用 XLA 编译并运行单个原语 'prim'。"""
  fun = xla_primitive_callable(prim, **params)
  # TODO(yashkatariya): 调研在 jit 上加入 is_primitive 并永远不再
  # 触发禁用 jit 的路径，而不是在这里对它做各种临时处理。
  prev = config.disable_jit.swap_local(False)
  try:
    outs = fun(*args)
  finally:
    config.disable_jit.set_local(prev)
  return outs

# TODO(necula): 这个缓存会持有 `params` 中所有 Jaxpr 的强引用
# （针对高阶原语）。
# 这无法立刻通过改用
# util.multi_weakref_lru_cache 来修复，因为 `params`（包括其中的 Jaxpr）
# 被 `prim_fun` 这个 lambda 闭包捕获了。此修复留到后续 PR。
@util.cache()
def xla_primitive_callable(prim: core.Primitive, **params):
  util.test_event("xla_primitive_callable_cache_miss")
  def prim_fun(*args):
    with config.eager_constant_folding(False):
      return prim.bind(*args, **params)
  prim_fun.__name__ = prim.name
  prim_fun.__qualname__ = prim.name
  prim_fun._apply_primitive = True  # pyrefly: ignore[missing-attribute]
  return api.jit(prim_fun)


def simple_impl(prim):
  prim.def_impl(partial(apply_primitive, prim))

RuntimeToken = Any

class RuntimeTokenSet(threading.local):
  """关于 token 的调用约定，见 effects.py 模块的文档字符串。"""

  # 对于每个有序效果，记录最近一次被分派计算返回的 token，
  # 该 token 已按该次计算涉及的设备做了分片。
  current_tokens: dict[core.Effect, core.Token]

  # 对于每个设备，记录该设备上最近一次被分派计算返回的运行时 token。
  output_runtime_tokens: dict[Device, RuntimeToken]

  def __init__(self):
    self.current_tokens = {}
    self.output_runtime_tokens = {}

  def get_token_input(
      self, eff: core.Effect, devices: list[Device]
  ) -> core.Token:
    tok = self.current_tokens.get(eff, np.zeros(0, np.bool_))

    if isinstance(tok, core.Token):
      # 设备的顺序可能发生变化，因此必要时需要重新分片。
      # TODO(yueshengys): 在多进程 SPMD 场景下这里可能仍有 bug。
      # 该逻辑以后再修订。可能需要在 XLA 程序内部加入一个
      # 分布式关停屏障。
      return api.device_put(
          tok, NamedSharding(Mesh(devices, 'x'), PartitionSpec('x')))

    # 只有在有序效果的 token 尚未创建时，我们才第一次使用
    # 复制式分片。
    s = GSPMDSharding.get_replicated(devices)
    sharded_tok = core.Token(
        pxla.shard_args(
            [s], [None], [xc.ArrayCopySemantics.REUSE_INPUT], [tok]
        )[0]
    )
    self.current_tokens[eff] = sharded_tok
    return sharded_tok

  def set_token_result(self, eff: core.Effect, token: core.Token):
    self.current_tokens[eff] = token

  def set_output_runtime_token(self, device: Device, token: RuntimeToken):
    # 我们可以随意覆盖先前的输出 token，因为在每个设备上计算都是
    # 全序的。只有最近一次计算的 token 才有意义。
    self.output_runtime_tokens[device] = token

  def clear(self):
    self.current_tokens = {}
    self.output_runtime_tokens = {}

  def block_until_ready(self):
    for token in self.current_tokens.values():
      token.block_until_ready()
    for token in self.output_runtime_tokens.values():
      token.block_until_ready()
    self.clear()

runtime_tokens: RuntimeTokenSet = RuntimeTokenSet()

@atexit.register
def wait_for_tokens():
  runtime_tokens.block_until_ready()


class LogElapsedTimeContextManager:
  __slots__ = ['fmt', 'fun_name', 'event', 'start_time']

  def __init__(self, fmt: str, fun_name: str, event: str | None = None):
    self.fmt = fmt
    self.fun_name = fun_name
    self.event = event

  def __enter__(self):
    self.start_time = time.time()
    if self.event is not None:
      record_scalar(
          self.event, self.start_time, fun_name=self.fun_name
      )

  def __exit__(self, exc_type, exc_value, traceback):
    if _on_exit:
      return

    end_time = time.time()
    elapsed_time = end_time - self.start_time
    log_priority = logging.WARNING if config.log_compiles.value else logging.DEBUG
    if logger.isEnabledFor(log_priority):
      logger.log(log_priority, self.fmt.format(
          fun_name=self.fun_name, elapsed_time=elapsed_time))
    if self.event is not None:
      record_event_duration_secs(
          self.event, elapsed_time, fun_name=self.fun_name
      )
      record_event_time_span(
          self.event, self.start_time, end_time, fun_name=self.fun_name
      )

log_elapsed_time = LogElapsedTimeContextManager


def should_tuple_args(num_args: int, platform: str) -> bool:
  # CPU 和 GPU 不需要元组，因为它们使用的主机端数据结构
  # 没有很小的数量上限。
  # TPU 只在列表非常长时才需要元组
  if platform == "tpu":
    return num_args > 2000
  else:
    return False

def jaxpr_has_primitive(jaxpr: core.Jaxpr, prim_name: str) -> bool:
  """Jaxpr 内部任意位置是否存在用户给定的某个原语。"""
  for eqn in jaxpr.eqns:
    if prim_name in eqn.primitive.name:
      return True
  for subjaxpr in core.subjaxprs(jaxpr):
    if jaxpr_has_primitive(subjaxpr, prim_name):
      return True
  return False


# 请谨慎使用这个注册表。它会破坏“降级到 stablehlo 时不感知
# 物理设备”这一保证。
prim_requires_devices_during_lowering: set[core.Primitive] = set()

@util.weakref_lru_cache
def jaxpr_has_prim_requiring_devices(jaxpr: core.Jaxpr) -> bool:
  for eqn in jaxpr.eqns:
    if eqn.primitive in prim_requires_devices_during_lowering:
      return True
  for subjaxpr in core.subjaxprs(jaxpr):
    if jaxpr_has_prim_requiring_devices(subjaxpr):
      return True
  return False


@util.weakref_lru_cache
def get_intermediate_shardings(
    jaxpr: core.Jaxpr) -> Sequence[tuple[Sharding, SourceInfo]]:
  from jax._src import shard_map  # pyrefly: ignore[missing-module-attribute]

  out = []
  for eqn in jaxpr.eqns:
    if eqn.primitive is pjit.sharding_constraint_p:
      s = eqn.params['sharding']
      if isinstance(s, NamedSharding) and isinstance(s.mesh, AbstractMesh):
        continue
      source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
      out.append((s, source_info))
    elif eqn.primitive is pjit.jit_p:
      source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
      out.extend((i, source_info) for i in eqn.params['in_shardings'])
      out.extend((o, source_info) for o in eqn.params['out_shardings'])
    elif eqn.primitive is shard_map.shard_map_p:
      mesh = eqn.params['mesh']
      if isinstance(mesh, AbstractMesh):
        continue
      source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
      out.extend((NamedSharding(mesh, spec), source_info)
                 for spec in [*eqn.params['in_specs'], *eqn.params['out_specs']])
    elif eqn.primitive is device_put_p:
      source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
      out.extend((s, source_info) for s in eqn.params['devices']
                 if isinstance(s, Sharding) and s.memory_kind is not None)
  for subjaxpr in core.subjaxprs(jaxpr):
    out.extend(get_intermediate_shardings(subjaxpr))
  return out


def check_arg(arg: Any):
  if not core.valid_jaxtype(arg):
    raise TypeError(f"Argument '{arg}' of type {type(arg)} is not a valid "
                    "JAX type.")


def needs_check_special() -> bool:
  return config.debug_infs.value or config.debug_nans.value

def check_special(name: str, bufs: Sequence[basearray.Array]) -> None:
  if needs_check_special():
    for buf in bufs:
      _check_special(name, buf.dtype, buf)


def check_special_array(name: str, arr: array.ArrayImpl) -> array.ArrayImpl:
  if needs_check_special():
    if dtypes.issubdtype(arr.dtype, np.inexact):
      for buf in arr._arrays:
        _check_special(name, buf.dtype, buf)
  return arr


def _check_special(name: str, dtype: np.dtype, buf: basearray.Array) -> None:
  if dtypes.issubdtype(dtype, np.inexact):
    if config.debug_nans.value and np.any(np.isnan(np.asarray(buf))):
      raise InternalFloatingPointError(name, "nan")
    if config.debug_infs.value and np.any(np.isinf(np.asarray(buf))):
      raise InternalFloatingPointError(name, "inf")

def _device_put_reshard(x): return x


@util.cache(max_size=2048, trace_context_in_key=False)
def _cached_logical_device_ids(
    inp_device_list: xc.DeviceList,
    target_device_list: xc.DeviceList
) -> tuple[int, ...]:
  device_to_index = {d: i for i, d in enumerate(target_device_list)}
  return tuple(device_to_index[d] for d in inp_device_list)


def _different_device_order_reshard(
    x: array.ArrayImpl, target_sharding: NamedSharding, copy: ArrayCopySemantics
) -> array.ArrayImpl:
  x._check_if_deleted()
  inp_sharding = x.sharding
  assert isinstance(inp_sharding, NamedSharding)

  inp_device_list = inp_sharding._internal_device_list
  target_device_list = target_sharding._internal_device_list

  donate_argnums = 0 if copy == ArrayCopySemantics.DONATE_INPUT else None
  if inp_device_list == target_device_list:
    return api.jit(_device_put_reshard, out_shardings=target_sharding,
                   donate_argnums=donate_argnums)(x)

  if inp_sharding.is_fully_replicated:
    logical_device_ids = None
  else:
    logical_device_ids = _cached_logical_device_ids(
        inp_device_list, target_device_list,
    )

  new_mesh = Mesh(
      target_sharding.mesh.devices.reshape(inp_sharding.mesh.axis_sizes),
      inp_sharding.mesh.axis_names)
  new_s = NamedSharding(
      new_mesh, inp_sharding.spec, memory_kind=target_sharding.memory_kind,
      _logical_device_ids=logical_device_ids)
  new_x = xc.reorder_shards(x, new_s, ArrayCopySemantics.REUSE_INPUT)
  return api.jit(_device_put_reshard, out_shardings=target_sharding,
                donate_argnums=donate_argnums)(new_x)


@util.cache(max_size=2048, trace_context_in_key=False)
def _is_supported_cross_host_transfer(ndim, src_sharding, dst_sharding):
  """若 src->dst 是受支持的跨主机传输，则返回 True。"""
  if (src_sharding._internal_device_list.device_kind !=
      dst_sharding._internal_device_list.device_kind):
    return False
  if (src_sharding._to_xla_hlo_sharding(ndim) !=
      dst_sharding._to_xla_hlo_sharding(ndim)):
    return False
  # 该检查排除了以下情形：源分片与目标分片具有相同的进程索引集合，
  # 但其中存在需要跨主机传输的分片。这种情形是可以支持的，
  # 只是检查代价很高。
  different_process_inds = (
      src_sharding._internal_device_list.process_indices !=
      dst_sharding._internal_device_list.process_indices)
  backend = xb.get_backend()
  # 如果请求了跨主机设备传输，但后端不支持，那么用户必须设置
  # 相应标志以启用基于 DCN 的传输。
  if (different_process_inds and
      (xb.FORCE_DCN_CROSS_HOST_TRANSFERS.value
      or not getattr(backend, "supports_cross_host_transfers", False)) and
      not xb.CROSS_HOST_TRANSFER_SOCKET_ADDRESS.value):
    if xb.FORCE_DCN_CROSS_HOST_TRANSFERS.value:
      msg = ("DCN-based cross-host transfers were requested with the "
             "jax_force_dcn_cross_host_transfers flag.")
    else:
      msg = ("The backend ({backend.platform}, {backend.platform_version}) "
             "does not support cross-host device transfers.")
    raise ValueError(
        f"{msg} Please set jax_cross_host_transfer_socket_address and "
        "(optionally) jax_cross_host_transport_addresses flags to enable "
        "DCN-based cross host device transfers.")
  return different_process_inds

@dataclasses.dataclass(frozen=True, slots=True)
class _DeferredShardArg:
  """对 `pxla.shard_args` 的延迟调用。

  各数组的实现会返回此对象而不是结果数组，以表示一次
  延迟的 `shard_args` 调用。随后 `_batched_device_put_impl` 会把所有
  `_DeferredShardArg` 对象合并为一次 `shard_args` 调用。
  """

  x: Any
  s: Sharding
  aval: core.AbstractValue
  committed: bool
  copy_semantics: ArrayCopySemantics

  def result_handler(self, shard_arg_result):
    return pxla.global_aval_to_result_handler(
        self.aval, self.s, self.committed)(shard_arg_result)

@dataclasses.dataclass(frozen=True, slots=True)
class _DeferredCrossHostTransferArg:
  """对 `xc.batched_copy_array_to_devices_with_sharding` 的延迟调用，
  用于跨主机数据传输。

  各数组的实现会返回此对象而不是结果数组，以表示一次延迟的
  `batched_copy_array_to_devices_with_sharding` 调用，用于跨主机
  数据传输。随后 `_batched_device_put_impl` 会把所有
  `_DeferredCrossHostTransferArg` 对象合并为一次
  `_batched_device_put_impl` 调用。

  对于任意 _DeferredCrossHostTransferArg，都有
  _is_supported_cross_host_transfer(
  x.ndim, x.sharding, dst_sharding) == True。
  """

  x: array.ArrayImpl
  dst_sharding: Sharding
  copy_semantics: ArrayCopySemantics


def _device_put_sharding_impl(
    x: Any,
    aval: core.ShapedArray,
    device: Device | Sharding | None,
    copy: ArrayCopySemantics,
):
  from jax.experimental import multihost_utils  # pyrefly: ignore[missing-import]

  # 这里使用动态类型，因为静态类型取决于
  # ``x_is_jax_array`` 的取值。
  x_sharding: Any
  if isinstance(x, array.ArrayImpl):
    x_is_jax_array = True
    x_is_fully_addressable, x_sharding = x.is_fully_addressable, x.sharding
  else:
    x_is_jax_array = False
    x_is_fully_addressable, x_sharding = None, None

  if isinstance(device, Sharding):
    s = device
    s_is_fully_addressable = s.is_fully_addressable
    if (getattr(x, 'sharding', None) == s and getattr(x, '_committed', False)
        and copy == ArrayCopySemantics.REUSE_INPUT):
      return x

    if isinstance(s, NamedSharding) and s.spec.unreduced:
      if s.spec.unreduced_kind != UnreducedKind.sum:
        raise NotImplementedError
      norm = lambda p: p._normalized_spec_for_aval(x.ndim)
      if (xb.process_count() == 1 and x_is_jax_array and
          isinstance(x_sharding, NamedSharding) and
          norm(x_sharding.spec) == norm(s.spec) and
          x_sharding.mesh.size == s.mesh.size):
        return _DeferredShardArg(x, s, aval, True, copy)
      # TODO(mattjj,yashkatariya): 处理捐赠（donation）
      return api.jit(_device_put_reshard, out_shardings=s)(x)

    if (not s_is_fully_addressable and
        x_is_jax_array and not x_is_fully_addressable and
        s.device_set == x_sharding.device_set):
      assert isinstance(s, NamedSharding), s
      return _different_device_order_reshard(x, s, copy)

    if (s_is_fully_addressable and x_is_jax_array and
        x_is_fully_addressable and s.num_devices > 1 and
        s._internal_device_list != x_sharding._internal_device_list and
        s.device_set == x_sharding.device_set):
      assert isinstance(s, NamedSharding), s
      return _different_device_order_reshard(x, s, copy)

    if (x_is_jax_array and x._committed and xb.process_count() > 1
        and _is_supported_cross_host_transfer(x.ndim, x_sharding, s)):
      return _DeferredCrossHostTransferArg(x, s, copy)

    if not s_is_fully_addressable:
      # 如果源分片和目标分片都不是完全可寻址的，并且上述条件
      # 均未满足，那么假定用户正在尝试不同设备顺序的重分片。
      if (x_is_jax_array and not x_is_fully_addressable
          and s.device_set != x_sharding.device_set):
        inp_ids = [d.id for d in x_sharding._device_assignment]
        inp_plat = x_sharding._device_assignment[0].platform.upper()
        target_ids = [d.id for d in s._device_assignment]
        target_plat = s._device_assignment[0].platform.upper()
        raise ValueError(
            "For a cross-host reshard in multi-controller JAX, input and target"
            " sharding should have the same set of devices. Got input's device"
            f" set ids: {inp_ids} on platform {inp_plat} and target sharding's"
            f" device set ids: {target_ids} on platform {target_plat}.\n\n"
            "There is experimental support for cross-host transfers with "
            "different device sets, when input/output shardings have the same "
            "indices and layouts, in the TFRT TPU runtime only.")

      if ((x_is_jax_array and not x._committed) or
          type(x) in array_types or type(x) in dtypes.python_scalar_types):
        # 如果所有主机都参与该分片，则断言输入在所有主机上相同。
        # 如果某些主机在该分片中没有任何可寻址设备，则跳过该检查，
        # 因为我们无法轻易区分以下两种情形：(1) 该分片在所有主机上
        # 包含同一组全局设备的子集（与分片中没有任何可寻址设备的主机
        # 不传输数据）；(2) 该分片在每台主机上包含不同的设备子集。
        # 对于 (1)，输入在所有主机上应当相同；而对于 (2)，则不必相同。
        if xb.process_count() == len(s._internal_device_list.process_indices):
          multihost_utils.assert_equal(
              x, fail_message=(
                  f"{type(x)} passed to device_put is not the same on each"
                  " process. Make sure you are passing the same value of"
                  f" {type(x)} on each process."))
        return _DeferredShardArg(x, s, aval, True, copy)
      # TODO(yashkatariya,mattjj): 链接到一篇关于 McJAX 与 jax.Array 的文档。
      raise ValueError(
          "device_put's second argument must be a Device or a Sharding which"
          f" represents addressable devices, but got {s}. Please pass device or"
          " Sharding which represents addressable devices.")
    return _DeferredShardArg(x, s, aval, True, copy)

  # 下面只剩下 `Device` 的情形。`Sharding` 实例已在上面处理。
  if x_is_jax_array:
    if not x_is_fully_addressable and not x_sharding.num_devices == 1:
      raise ValueError(
          "When the second argument to `device_put` is a Device, the first "
          "argument must be a fully addressable array or a non-addressable "
          "array with a single device sharding. Got value with devices "
          f"{x.devices()}")
    if device is None:
      if copy == ArrayCopySemantics.REUSE_INPUT:
        return x
      else:
        return _DeferredShardArg(x, x_sharding, aval, x.committed, copy)
    elif x_sharding.num_devices == 1:
      device = x_sharding._device_assignment[0] if device is None else device
      sharding = make_single_device_sharding(device)
      if not x._committed and not sharding.has_addressable_devices:
        # 对于 McJAX 中未提交的数组，每个进程都持有一份该数组的本地副本。
        # 如果目标分片不可寻址，则不需要传输数据，因为数据已经在
        # 该分片可寻址的那个进程中完成了传输。
        shards, devices = [], []
      else:
        shards, devices = [x], [device]
      if copy == ArrayCopySemantics.ALWAYS_COPY:
        return xc.batched_device_put(aval, sharding, shards, devices, True,
                                     True)
      return pxla.batched_device_put(aval, sharding, shards, devices)

  sh = make_single_device_sharding(pxla.get_default_device()
                                   if device is None else device)
  return _DeferredShardArg(x, sh, aval, device is not None, copy)


def _device_put_impl(
    x, *, device: Device | Sharding | Format | None,
    src: Device | Sharding | Format | None, copy: ArrayCopySemantics, aval):
  if aval is None:
    try:
      if isinstance(x, core.Tracer):
        raise TypeError(f"Argument '{x}' of type '{type(x)}' is not a valid JAX type")
      aval = core.typeof(x)
      aval = update_dp_aval(aval, device)
    except TypeError as err:
      raise TypeError(
          f"Argument '{x}' of type {type(x)} is not a valid JAX type") from err

  if isinstance(device, core.MemorySpace):
    return apply_primitive(device_put_p, x, devices=(device,), srcs=(src,),
                           copy_semantics=(copy,))[0]

  if isinstance(device, Format):
    l = device
    dll = l.layout
    x_dll = x.format.layout if hasattr(x, 'format') else None
    if dll is None and l.sharding is None:
      return _device_put_sharding_impl(x, aval, l.sharding, copy)
    if (not isinstance(l.sharding, Sharding) or
        not isinstance(dll, (Layout, type(None)))):
      raise ValueError(
          "sharding and layout in `Layout` instance should be"
          f" concrete. Got layout: {l} for input {aval.str_short()}")
    if (getattr(x, 'format', None) == l and getattr(x, '_committed', False) and
        copy == ArrayCopySemantics.REUSE_INPUT):
      return x
    if x_dll is None and dll is None:
      return _device_put_sharding_impl(x, aval, l.sharding, copy)
    return api.jit(
        _device_put_reshard,
        out_shardings=l,
        donate_argnums=(0 if copy == ArrayCopySemantics.DONATE_INPUT else None),
    )(x)

  return _device_put_sharding_impl(x, aval, device, copy)


def _batched_device_put_impl(
    *xs,
    devices: Sequence[Device | Sharding | Format | None],
    srcs: Sequence[Device | Sharding | Format | None],
    copy_semantics: Sequence[ArrayCopySemantics],
    dst_avals: Sequence[core.ShapedArray | None]):
  ys = []
  dsa_indices, dsa_xs, dsa_shardings, dsa_copy_semantics = [], [], [], []
  dca_indices, dca_xs, dca_shardings, dca_device_lists, dca_copy_semantics = \
    [], [], [], [], []

  for i, (x, device, src, cp, aval) in enumerate(
      zip(xs, devices, srcs, copy_semantics, dst_avals)):
    y = _device_put_impl(x, device=device, src=src, copy=cp, aval=aval)
    if isinstance(y, _DeferredShardArg):
      dsa_indices.append(i)
      dsa_xs.append(y.x)
      dsa_shardings.append(y.s)
      dsa_copy_semantics.append(y.copy_semantics)
    elif isinstance(y, _DeferredCrossHostTransferArg):
      dca_indices.append(i)
      dca_xs.append(y.x)
      dca_shardings.append(y.dst_sharding)
      dca_device_lists.append(y.dst_sharding._internal_device_list)
      dca_copy_semantics.append(y.copy_semantics)
    ys.append(y)

  if dsa_xs:
    shard_arg_results = pxla.shard_args(dsa_shardings, [None] * len(dsa_xs),
                                        dsa_copy_semantics, dsa_xs)
    for i, shard_arg_result in zip(dsa_indices, shard_arg_results):
      assert isinstance(ys[i], _DeferredShardArg)
      ys[i] = ys[i].result_handler(shard_arg_result)
  if dca_xs:
    copy_array_results = xc.batched_copy_array_to_devices_with_sharding(
      dca_xs, dca_device_lists, dca_shardings, dca_copy_semantics)
    for i, copy_array_result in zip(dca_indices, copy_array_results):
      assert isinstance(ys[i], _DeferredCrossHostTransferArg)
      ys[i] = copy_array_result

  return ys

def batched_device_put_impl(
    *xs,
    devices: Sequence[Device | Sharding | Format | None],
    srcs: Sequence[Device | Sharding | Format | None],
    copy_semantics: Sequence[ArrayCopySemantics]):
  return _batched_device_put_impl(
      *xs, devices=devices, srcs=srcs, copy_semantics=copy_semantics,
      dst_avals=[None] * len(devices))


device_put_p = core.Primitive('device_put')
device_put_p.multiple_results = True
device_put_p.def_impl(batched_device_put_impl)


def _device_put_folding_rule(consts, params, out_avals):
  # 我们会消除那些什么都不做的 device_put；例如 jnp.array
  # 就可能生成这类调用。
  if (all(x is None for x in params["devices"])
      and all(isinstance(x, literals.TypedNdArray) for x in consts)
      and all(x == ArrayCopySemantics.REUSE_INPUT for x in params["copy_semantics"])):
    return consts
  return None

partial_eval.const_fold_rules[device_put_p] = _device_put_folding_rule


def update_dp_aval(aval, d):
  if not isinstance(aval, core.ShapedArray):
    return aval
  if isinstance(d, Sharding):
    aval = (aval.update(sharding=aval.sharding.update(mesh=d.mesh.abstract_mesh,
                                                      spec=d.spec))
            if isinstance(d, NamedSharding) else aval.update(sharding=None))
    if d.memory_kind is not None:
      aval = aval.update(memory_space=core.mem_kind_to_space(d.memory_kind))
    return aval
  elif isinstance(d, core.MemorySpace):
    return aval.update(memory_space=d)
  return aval

def _device_put_abstract_eval(*xs, devices, srcs, copy_semantics):
  return [update_dp_aval(x, d) for x, d in zip(xs, devices)]
device_put_p.def_abstract_eval(_device_put_abstract_eval)

def _device_put_transpose(cts, *args, devices, srcs, copy_semantics):
  results: list[Any | None] = [None] * len(cts)
  dp_cts = []
  for i, (ct, arg, device, src, cp) in enumerate(zip(
      cts, args, devices, srcs, copy_semantics)):
    if ad.is_undefined_primal(arg):
      if type(ct) is ad.Zero:
        results[i] = ad.Zero(arg.aval)
      else:
        dp_cts.append((i, ct, arg, device, src, cp))

  if dp_cts:
    indices, dp_ct, args, devices, srcs, copy_semantics = list(zip(*dp_cts))
    # TODO(yashkatariya): 也许可以去掉针对 Host 的特殊处理？
    srcs = tuple(a.aval.memory_space
                 if s is None and a.aval.memory_space == core.MemorySpace.Host
                 else s for s, a in zip(srcs, args))
    new_copy_semantics = []
    for cp in copy_semantics:
      if cp == ArrayCopySemantics.DONATE_INPUT:
        raise ValueError(
            "donate=True is not allowed during tranposition of device_put."
            " Please file an issue if you want this to be supported.")
      elif cp == ArrayCopySemantics.REUSE_INPUT:
        new_copy_semantics.append(ArrayCopySemantics.ALWAYS_COPY)
      else:
        assert cp == ArrayCopySemantics.ALWAYS_COPY
        new_copy_semantics.append(ArrayCopySemantics.ALWAYS_COPY)
    ys = device_put_p.bind(*dp_ct, devices=srcs, srcs=devices,
                           copy_semantics=tuple(new_copy_semantics))
    for i, y in zip(indices, ys):
      results[i] = y
  return results
ad.primitive_jvps[device_put_p] = partial(ad.linear_jvp, device_put_p)
ad.primitive_transposes[device_put_p] = _device_put_transpose

def _device_put_batcher(batched_args, batch_dims, **params):
  mapped_batch_dims = [bd for bd in batch_dims if bd is not None]
  assert not mapped_batch_dims or all(
      mapped_batch_dims[0] == bd for bd in mapped_batch_dims[1:]
  ), batch_dims
  return device_put_p.bind(*batched_args, **params), batch_dims
batching.primitive_batchers[device_put_p] = _device_put_batcher

def _tpu_gpu_device_put_lowering(ctx, *xs, devices, srcs, copy_semantics):
  # TODO(yashkatariya): 或许无论如何都应该加上这些自定义调用，如果它正被用在 jit 内部的话？
  # 至少就目前而言，这样可以保持旧有的行为。
  if ctx.module_context.all_default_mem_kind:
    return xs
  def lower(x, device, aval, out_aval):
    if ((isinstance(device, Sharding) and device.memory_kind is not None) or
        isinstance(device, core.MemorySpace)):
      if isinstance(device, Sharding):
        if config.use_shardy_partitioner.value:
          x = mlir.wrap_with_sharding_op(
              ctx, x, out_aval,
              device._to_sdy_sharding(aval.ndim))
        else:
          x = mlir.wrap_with_sharding_op(
              ctx, x, out_aval,
              device._to_xla_hlo_sharding(aval.ndim).to_proto())
      mem_kind = (core.mem_space_to_kind(device)
                  if isinstance(device, core.MemorySpace) else device.memory_kind)
      assert mem_kind is not None
      x = mlir.wrap_with_memory_kind(ctx.module_context, x, mem_kind, out_aval)
      return x
    return x
  return list(map(lower, xs, devices, ctx.avals_in, ctx.avals_out))

mlir.register_lowering(
  device_put_p, _tpu_gpu_device_put_lowering, platform='tpu')
mlir.register_lowering(
  device_put_p, _tpu_gpu_device_put_lowering, platform='gpu')


def _common_device_put_lowering(ctx, *xs, devices, srcs, copy_semantics):
  return xs
mlir.register_lowering(device_put_p, _common_device_put_lowering)
