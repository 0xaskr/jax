# Copyright 2025 The JAX Authors.
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

# 文件职责：为已暂存的 JAX 计算提供运行时错误检查机制。
# `set_error_if` 在计算内部记录错误码与回溯但不立即中断执行，
# `raise_if_error` 在计算结束后统一抛出 `JaxValueError`。
# 错误码保存在线程局部的 `core.Ref` 中，其形状随上下文 mesh 变化，
# 因此多设备显式模式下需用 `error_checking_context` 重新初始化。
# `wrap_for_export` / `unwrap_from_import` 让错误检查状态可随 AOT
# 导出与导入函数一起序列化，从而接入导入进程的全局错误状态。

from __future__ import annotations

import dataclasses
from functools import partial
import json
import threading
import traceback as tb_lib
from types import TracebackType
import warnings

import numpy as np

from jax._src import core
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import tree_util
import jax._src.mesh as mesh_lib
from jax._src import shard_map
from jax._src.export import _export
from jax._src.lax import lax
from jax._src.sharding_impls import NamedSharding, PartitionSpec as P
from jax._src.typing import Array, ArrayLike


traceback_util.register_exclusion(__file__)


class JaxValueError(ValueError):
  """在 JAX 计算内部检测到运行时错误时抛出的异常。"""


#: 表示无错误的默认错误码。
#:
#: 选择该值是因为在做规约时可以用 `jnp.min()` 取得第一个错误。
_NO_ERROR = np.iinfo(np.uint32).max


_error_list_lock = threading.RLock()
# (错误信息, 回溯) 对。从 AOT 导入时回溯为 `str`。
_error_list: list[tuple[str, TracebackType | str]] = []

# AOT 导入时无效/损坏错误码的标准错误信息。
_INVALID_ERROR_CODE_MSG = (
    "An unknown error occurred during execution of an AOT-imported function. "
    "This may indicate data corruption during AOT serialization/deserialization, "
    "or a version mismatch between the exporting and importing JAX versions."
)
_INVALID_ERROR_CODE_TRACEBACK = "Traceback not available for corrupted error codes."


class _ErrorStorage(threading.local):

  def __init__(self):
    self.ref: core.Ref | None = None


_error_storage = _ErrorStorage()


def _initialize_error_code_ref() -> None:
  """在当前线程中初始化错误码引用。

  错误码数组的形状与大小取决于上下文中的 mesh。
  在单设备环境中该数组是标量；在多设备环境中，
  其形状与大小与 mesh 一致。
  """
  # 从上下文中获取 mesh。
  mesh = mesh_lib.get_concrete_mesh()

  if mesh.empty:  # 单设备情形。
    error_code: ArrayLike = np.uint32(_NO_ERROR)

  else:  # 多设备情形。
    sharding = NamedSharding(mesh, P(*mesh.axis_names))
    error_code = lax.full(
        mesh.axis_sizes,
        np.uint32(_NO_ERROR),
        sharding=sharding,
    )

  _error_storage.ref = core.new_ref(error_code)


class error_checking_context:
  """依据上下文中的 mesh 重新定义内部错误状态。

  在显式模式下于多设备环境中使用 JAX 时，错误跟踪需要与设备 mesh
  正确对齐。该上下文管理器确保内部错误状态能依据当前 mesh 配置被
  正确初始化。

  在启动多设备计算时，或在不同设备 mesh 之间切换时，应使用该上下文
  管理器。

  进入该上下文时，它会依据上下文中的 mesh 初始化新的错误状态；
  退出该上下文时，它会恢复先前的错误状态。
  """

  __slots__ = ("old_ref",)

  def __init__(self):
    self.old_ref: core.Ref | None = None

  def __enter__(self):
    self.old_ref = _error_storage.ref
    with core.eval_context():
      _initialize_error_code_ref()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    _error_storage.ref = self.old_ref


def set_error_if(pred: Array, /, msg: str) -> None:
  """若 `pred` 的任一元素为 `True`，则设置内部错误状态。

  该函数在 JAX 计算内部使用，用于在不立即中止执行的前提下检测运行时
  错误。当该函数被追踪时（例如在 :func:`jax.jit` 内部），相应的错误
  信息及其回溯会被记录下来。在执行时，若 `pred` 含有任何 `True` 值，
  则会设置错误状态，但执行会继续而不被中断。记录下来的错误随后可用
  :func:`raise_if_error` 抛出。

  若错误状态已被设置，则后续错误会被忽略，不会覆盖已有的错误。

  对于多设备环境，在显式模式下用户必须调用
  :func:`error_checking_context` 来初始化一个与设备 mesh 匹配的新错误
  跟踪状态。在自动模式下，该函数内部可能发生隐式的跨设备通信，这可能
  影响性能；此时会发出警告。

  在使用 `jax.export` 导出函数时，必须在导出前用 :func:`wrap_for_export`
  显式包装错误检查，并在导入后用 :func:`unwrap_from_import` 解开包装。

  Args:
    pred: 一个 JAX 布尔数组。若 `pred` 的任一元素为 `True`，
      则会设置内部错误状态。
    msg: 稍后要抛出的对应错误信息。
  """
  # TODO(jakevdp): 移除这个导入，改用 lax API 表达下面的逻辑。
  import jax.numpy as jnp  # pyrefly: ignore[missing-import]

  if _error_storage.ref is None:
    with core.eval_context():
      _initialize_error_code_ref()
    assert _error_storage.ref is not None

  # 获取回溯。
  traceback = source_info_util.current().traceback
  assert traceback is not None
  traceback = traceback.as_python_traceback()
  assert isinstance(traceback, TracebackType)
  traceback = traceback_util.filter_traceback(traceback)
  assert isinstance(traceback, TracebackType)

  with _error_list_lock:
    new_error_code = np.uint32(len(_error_list))
    _error_list.append((msg, traceback))

  out_sharding = core.typeof(_error_storage.ref).sharding
  in_sharding: NamedSharding = core.typeof(pred).sharding

  # 对 `pred` 做规约。
  if all(dim is None for dim in out_sharding.spec):  # 单设备情形。
    pred = pred.any()
  else:  # 多设备情形。
    has_auto_axes = mesh_lib.AxisType.Auto in in_sharding.mesh.axis_types
    if has_auto_axes:  # 自动模式。
      warnings.warn(
          "When at least one mesh axis of `pred` is in auto mode, calling"
          " `set_error_if` will cause implicit communication between devices."
          " To avoid this, consider converting the mesh axis in auto mode to"
          " explicit mode.",
          RuntimeWarning,
      )
      pred = pred.any()  # 规约为单个标量
    else:  # 显式模式。
      if out_sharding.mesh != in_sharding.mesh:
        raise ValueError(
            "The error code state and the predicate must be on the same mesh, "
            f"but got {out_sharding.mesh} and {in_sharding.mesh} respectively. "
            "Please use `with error_checking_context()` to redefine the error "
            "code state based on the mesh."
        )
      pred = shard_map.shard_map(
          partial(jnp.any, keepdims=True),
          mesh=out_sharding.mesh,
          in_specs=in_sharding.spec,
          out_specs=out_sharding.spec,
      )(pred)  # 执行逐设备规约

  error_code = _error_storage.ref[...]
  should_update = jnp.logical_and(error_code == jnp.uint32(_NO_ERROR), pred)
  error_code = jnp.where(should_update, new_error_code, error_code)
  # TODO(ayx): 支持 vmap 与 shard_map。
  _error_storage.ref[...] = error_code


def raise_if_error() -> None:
  """若内部错误状态已被设置，则抛出异常。

  该函数应在计算完成后调用，以检查执行期间通过 `set_error_if()` 标记的
  任何错误。若存在错误，它会抛出带有相应错误信息的 `JaxValueError`。

  不应在被追踪的函数内部（例如 :func:`jax.jit` 内部）调用该函数。
  这样做会抛出 `ValueError`。

  Raises:
    JaxValueError: 若内部错误状态已被设置。
    ValueError: 若在被追踪的 JAX 函数内部调用。
  """
  if _error_storage.ref is None:  # 若未初始化，则什么也不做
    return

  error_code = _error_storage.ref[...].min()  # 规约为单个错误码
  if isinstance(error_code, core.Tracer):
    raise ValueError(
        "raise_if_error() should not be called within a traced context, such as"
        " within a jitted function."
    )
  if error_code == np.uint32(_NO_ERROR):
    return
  _error_storage.ref[...] = lax.full(
      _error_storage.ref.shape,
      np.uint32(_NO_ERROR),
      sharding=_error_storage.ref.sharding,
  )  # 清除错误码

  with _error_list_lock:
    if error_code < 0 or error_code >= len(_error_list):
      # 用标准错误信息优雅地处理无效错误码。
      # 这可能由损坏的 AOT 序列化数据引起，也可能由会导致索引错误的
      # 负错误码引起。
      msg, traceback = _INVALID_ERROR_CODE_MSG, _INVALID_ERROR_CODE_TRACEBACK
    else:
      msg, traceback = _error_list[error_code]
  if isinstance(traceback, str):  # 来自导入的 AOT 函数
    exc = JaxValueError(
        f"{msg}\nThe original traceback is shown below:\n{traceback}"
    )
    raise exc
  else:
    exc = JaxValueError(msg)
    raise exc.with_traceback(traceback)


@dataclasses.dataclass(frozen=True, slots=True)
class _ErrorClass:
  """用于为 AOT 编译保存错误信息的类。

  该类由包装函数 `wrap_for_export` 与 `unwrap_from_import` 在内部使用，
  以便把与错误相关的数据封装在导出的函数中。

  Attributes:
    error_code (jax.Array): 一个 JAX 数组，表示待导出函数的最终错误状态。
      该值局部于包装函数。
    error_list (list[tuple[str, str]]): 一个 `(error_message, traceback)`
      对的列表，包含错误信息及对应的栈回溯。该错误列表局部于包装函数，
      不包含来自其他函数的错误信息对。
  """

  error_code: Array
  error_list: list[tuple[str, str]]


tree_util.register_dataclass(
    _ErrorClass, data_fields=("error_code",), meta_fields=("error_list",)
)
_export.register_pytree_node_serialization(
    _ErrorClass,
    serialized_name=f"{_ErrorClass.__module__}.{_ErrorClass.__name__}",
    serialize_auxdata=lambda x: json.dumps(x, ensure_ascii=False).encode(
        "utf-8"
    ),
    deserialize_auxdata=lambda x: json.loads(x.decode("utf-8")),
)


def _traceback_to_str(traceback: TracebackType) -> str:
  """把回溯转换为字符串以便导出。"""
  return "".join(tb_lib.format_list(tb_lib.extract_tb(traceback))).rstrip("\n")


def wrap_for_export(f):
  """用错误检查包装函数，使其兼容 AOT 模式。

  错误检查依赖全局状态，而全局状态无法跨进程序列化。该包装器确保错误
  状态保持在函数作用域内，从而可以导出函数并在之后于其他进程中导入。

  当该函数之后被导入时，必须用 :func:`unwrap_from_import` 包装，以便把
  被导入函数的错误检查机制接入当前进程的全局错误检查机制。

  该函数只应作用于一个函数一次；对同一函数多次包装是不必要的。
  """

  def inner(*args, **kwargs):
    global _error_list

    # 1. 保存旧状态并初始化新状态。
    with core.eval_context():
      old_ref = _error_storage.ref
    _initialize_error_code_ref()
    with _error_list_lock:
      old_error_list, _error_list = _error_list, []

      # 2. 追踪该函数。
      out = f(*args, **kwargs)
      assert _error_storage.ref is not None
      error_code = _error_storage.ref[...].min()

      # 3. 恢复旧状态。
      _error_list, new_error_list = old_error_list, _error_list
    with core.eval_context():
      _error_storage.ref = old_ref

    new_error_list = [
        (msg, _traceback_to_str(traceback)) for msg, traceback in new_error_list
    ]
    return out, _ErrorClass(error_code, new_error_list)

  return inner


def unwrap_from_import(f):
  """在 AOT 导入后解开函数包装以恢复错误检查。

  当 AOT 导出的函数在新进程中被导入时，其错误状态与当前进程的全局错误
  状态是分离的。该包装器确保执行期间检测到的错误被正确接入当前进程的
  全局错误检查机制。

  该函数只应作用于导出前曾用 :func:`wrap_for_export` 包装过的函数。
  """
  if _error_storage.ref is None:
    with core.eval_context():
      _initialize_error_code_ref()
    assert _error_storage.ref is not None

  def inner(*args, **kwargs):
    out, error_class = f(*args, **kwargs)
    new_error_code, error_list = error_class.error_code, error_class.error_list

    # 更新全局错误列表。
    with _error_list_lock:
      offset = len(_error_list)
      _error_list.extend(error_list)

    # 更新全局错误码数组。
    assert _error_storage.ref is not None
    error_code = _error_storage.ref[...]
    should_update = lax.bitwise_and(
        error_code == np.uint32(_NO_ERROR),
        new_error_code != np.uint32(_NO_ERROR),
    )
    error_code = lax.select(should_update, new_error_code + offset, error_code)

    # TODO(ayx): 支持 vmap 与 shard_map。
    _error_storage.ref[...] = error_code

    return out

  return inner
