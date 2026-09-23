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
#
# 文件职责：提供旧版并行映射 API `jax.pmap` 及其多主机分片实现。
# 本模块把 `pmap` 表达为 `jit` 与 `shard_map` 的组合：先由 `in_axes`/`out_axes`
# 推导每个参数的 PartitionSpec 与 Mesh，再在全局数组与主机本地数组之间做转换。
# 关键概念包括静态广播参数、被捐赠（donation）的缓冲区，以及多进程下
# 集合通信跨越所有进程设备的行为；`pmap` 已建议由 `shard_map` 取代。
from __future__ import annotations

from collections.abc import Hashable
from functools import partial
from typing import Any, NamedTuple
from collections.abc import Callable, Sequence, Iterable
import warnings

from jax._src import api
from jax._src import array
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import pjit as pjit_lib
from jax._src.random import prng
from jax._src import sharding_impls
from jax._src import stages
from jax._src import traceback_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.api_util import (
    _ensure_index_tuple, argnums_partial, check_callable, donation_vector,
    fun_signature, fun_sourceinfo, rebase_donate_argnums)
from jax._src.interpreters import pxla
from jax._src.lax import lax
from jax._src.lib import xla_client as xc
from jax._src.mesh import Mesh
from jax._src.shard_map import _axes_to_pspec, shard_map
from jax._src.traceback_util import api_boundary
from jax._src.tree_util import (
    broadcast_flattened_prefix_with_treedef, broadcast_prefix,
    prefix_errors, tree_flatten, tree_leaves, tree_map, tree_unflatten)
import numpy as np

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

traceback_util.register_exclusion(__file__)

AxisName = Hashable


@partial(api_boundary, repro_api_name="jax.pmap")
def pmap(
    fun: Callable,
    axis_name: AxisName | None = None,
    *,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    static_broadcasted_argnums: int | Iterable[int] = (),
    devices: Sequence[xc.Device] | None = None,  # noqa: F811
    backend: str | None = None,
    axis_size: int | None = None,
    donate_argnums: int | Iterable[int] = (),
  ) -> Any:
  """旧的并行 map 实现方式。请改用 :py:func:`jax.shard_map`。

  .. note::
    虽然 :py:func:`jax.pmap` 可以使用，但通常应改用
    :py:func:`jax.shard_map` 或 ``jax.smap``。shard_map 支持更
    高效的自动微分，在多控制器场景下也具有更好的可组合性。
    示例见 https://docs.jax.dev/en/latest/notebooks/shard_map.html。

  .. note::
    :py:func:`pmap` 现在基于 :py:func:`jit` 和
    :py:func:`shard_map` 实现。更多信息请参阅
    `迁移指南
    <https://docs.jax.dev/en/latest/migrate_pmap.html>`_。

  :py:func:`pmap` 的用途是表达单程序多数据
  （SPMD）程序。把 :py:func:`pmap` 应用于一个函数会用 XLA 编译该函数
  （与 :py:func:`jit` 类似），然后在 XLA 设备上并行执行它，
  例如多块 GPU 或多个 TPU 核心。在语义上它可与 :py:func:`vmap` 相比，
  因为二者都把函数映射到数组轴上，但 :py:func:`vmap` 是通过把
  被映射的轴下推到原语操作来向量化函数的，而 :py:func:`pmap` 则是
  复制函数，并在各自的 XLA 设备上并行执行
  每个副本。

  被映射轴的大小必须小于或等于可用的本地 XLA
  设备数，即 :py:func:`jax.local_device_count()` 的返回值（除非
  指定了 ``devices``，见下文）。对于嵌套的 :py:func:`pmap` 调用，
  各被映射轴大小的乘积必须小于或等于
  XLA 设备数。

  .. note::
    :py:func:`pmap` 会编译 ``fun``，因此虽然它可以与
    :py:func:`jit` 组合使用，但通常没有必要。

  :py:func:`pmap` 要求所有参与设备完全相同。
  例如，无法用 :py:func:`pmap` 把一个计算并行化到
  两种不同型号的 GPU 上。目前，同一设备在同一个 `pmap` 中
  参与两次会导致错误。

  **多进程平台：** 在 TPU pod 等多进程平台上，
  :py:func:`pmap` 被设计用于 SPMD Python 程序，其中每个
  进程都运行相同的 Python 代码，从而所有进程按相同顺序
  运行同一个 pmap 过的函数。每个进程仍应以等于 *本地*
  设备数的被映射轴大小来调用该 pmap 函数（除非指定了
  ``devices``，见下文），并且照常返回一个具有相同前导轴
  大小的数组。但是，``fun`` 中的任何集合通信操作
  都会通过设备间通信在 *所有* 参与设备（包括其他
  进程上的设备）上计算。概念上，可以把这理解为
  在一个跨进程分片的单个数组上运行 pmap，其中每个
  进程只“看到”输入和输出中属于它的本地分片。
  SPMD 模型要求相同的多进程 pmap 必须在所有设备上
  以相同顺序运行，但它们之间可以穿插任意
  在单个进程中运行的运算。

  Args:
    fun: 要在参数轴上映射的函数。它的参数和返回值
      应当是数组、标量，或由它们构成的（嵌套）标准 Python
      容器（tuple/list/dict）。由 ``static_broadcasted_argnums``
      指明的位置参数可以是任意对象，只要它们
      可哈希且定义了相等运算。
    axis_name: 可选，一个可哈希的 Python 对象，用于标识被映射的轴，
      以便施加并行集合通信操作。
    in_axes: 一个非负整数、None，或由它们构成的嵌套 Python 容器，
      用于指明要在位置参数的哪些轴上映射。以关键字方式
      传入的参数总是在其前导轴（即轴索引 0）上被映射。
      详见 :py:func:`vmap`。
    out_axes: 一个非负整数、None，或由它们构成的嵌套 Python 容器，
      指明被映射的轴应出现在输出中的位置。所有带被映射轴的
      输出都必须有非 None 的 ``out_axes`` 指定
      （见 :py:func:`vmap`）。
    static_broadcasted_argnums: 一个整数或一组整数，指明把哪些
      位置参数视为静态（编译期常量）。仅依赖静态参数的
      操作会被常量折叠。以不同的值调用该 pmap 函数
      会触发重新编译。若调用 pmap 函数时传入的位置参数
      少于 ``static_broadcasted_argnums`` 所指示的数量，
      则会抛出错误。每个静态参数都会被广播到所有设备。
      不是数组或数组容器的参数必须标记为静态，
      即视为编译期常量。
      默认为 ()。

      静态参数必须可哈希，即同时实现了 ``__hash__`` 和 ``__eq__``，
      并且应当是不可变的。

    devices: 这是实验性功能，API 可能会变化。
      可选，要在其上映射的设备序列。（可用设备可通过
      jax.devices() 获取。）在多进程场景下，每个进程必须
      给出完全相同的值（因此会包含跨进程的设备）。
      若指定，被映射轴的大小必须等于该序列中属于
      给定进程的本地设备数。在内层或外层
      :py:func:`pmap` 中指定了 ``devices`` 的嵌套 :py:func:`pmap`
      尚不支持。
    backend: 这是实验性功能，API 可能会变化。可选，一个表示 XLA
      后端的字符串：'cpu'、'gpu' 或 'tpu'。
    axis_size: 可选；被映射轴的大小。
    donate_argnums: 指明把哪些位置参数的缓冲区“捐赠”
      给该计算。如果计算结束后你不再需要这些参数
      缓冲区，那么捐赠是安全的。某些情况下 XLA 可以利用
      捐赠的缓冲区来减少执行某个计算所需的内存量，
      例如回收某个输入缓冲区来存放结果。
      你不应重复使用捐赠给计算的缓冲区，
      若这样做 JAX 会抛出错误。
      注意 donate_argnums 只对位置参数有效，
      以关键字传入的参数不会被捐赠。

      关于缓冲区捐赠的更多细节，见
      `FAQ <https://docs.jax.dev/en/latest/faq.html#buffer-donation>`_。

  Returns:
    ``fun`` 的并行化版本，其参数与 ``fun`` 的参数相对应，
    但会在 ``in_axes`` 所指明的位置上带有额外的数组轴，
    并且输出带有一个额外的前导数组轴（大小相同）。

  例如，假设有 8 个可用的 XLA 设备，:py:func:`pmap` 可以用作
  沿前导数组轴的 map：

  >>> import jax.numpy as jnp
  >>>
  >>> out = pmap(lambda x: x ** 2)(jnp.arange(8))  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [0, 1, 4, 9, 16, 25, 36, 49]

  当该前导维度小于可用设备数时，JAX 只会在
  设备的子集上运行：

  >>> x = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2))
  >>> y = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2)) ** 2
  >>> out = pmap(jnp.dot)(x, y)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [[[    4.     9.]
    [   12.    29.]]
   [[  244.   345.]
    [  348.   493.]]
   [[ 1412.  1737.]
    [ 1740.  2141.]]]

  如果你的前导维度大于可用设备数，
  就会得到错误：

  >>> pmap(lambda x: x ** 2)(jnp.arange(9))  # doctest: +SKIP
  ValueError: ... requires 9 replicas, but only 8 XLA devices are available

  与 :py:func:`vmap` 一样，在 ``in_axes`` 中使用 ``None`` 表示
  该参数没有额外轴，应在各副本之间广播，而不是
  映射：

  >>> x, y = jnp.arange(2.), 4.
  >>> out = pmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None))(x, y)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  ([4., 5.], [8., 8.])

  注意 :py:func:`pmap` 总是返回值在其前导轴上的映射结果，
  等价于在 :py:func:`vmap` 中使用 ``out_axes=0``。

  除了表达纯 map 之外，:py:func:`pmap` 还可以用来表达
  通过集合通信操作进行通信的并行单程序多数据（SPMD）
  程序。例如：

  >>> f = lambda x: x / jax.lax.psum(x, axis_name='i')
  >>> out = pmap(f, axis_name='i')(jnp.arange(4.))  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [ 0.          0.16666667  0.33333334  0.5       ]
  >>> print(out.sum())  # doctest: +SKIP
  1.0

  在这个例子中，``axis_name`` 是一个字符串，但它可以是任何定义了
  ``__hash__`` 和 ``__eq__`` 的 Python 对象。

  :py:func:`pmap` 的 ``axis_name`` 参数为被映射轴命名，以便
  :func:`jax.lax.psum` 之类的集合通信操作可以引用它。轴名
  在嵌套的 :py:func:`pmap` 函数中尤为重要，
  因为此时集合通信操作可以作用于不同的轴：

  在多进程平台上，集合通信操作会作用于所有设备，
  包括其他进程上的设备。例如，假设下面的代码
  运行在两个进程上，每个进程有 4 个 XLA 设备：

  >>> f = lambda x: x + jax.lax.psum(x, axis_name='i')
  >>> data = jnp.arange(4) if jax.process_index() == 0 else jnp.arange(4, 8)
  >>> out = pmap(f, axis_name='i')(data)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [28 29 30 31] # on process 0
  [32 33 34 35] # on process 1

  每个进程传入一个不同的长度为 4 的数组，对应它的 4 个
  本地设备，而 psum 会作用于全部 8 个值。概念上，这两个
  长度为 4 的数组可以看作一个被分片的长度为 8 的数组（在此例中
  等价于 jnp.arange(8)）在其上映射，其中长度为 8 的被映射
  轴命名为 'i'。于是每个进程上的 pmap 调用会返回
  对应的长度为 4 的输出分片。

  ``devices`` 参数可以用来精确指定使用哪些设备
  来运行并行计算。例如，仍假设单个进程有 8 个设备，
  下面的代码定义了两个并行计算，一个运行在前六个设备上，
  另一个运行在剩余两个设备上：

  >>> from functools import partial
  >>> @partial(pmap, axis_name='i', devices=jax.devices()[:6])
  ... def f1(x):
  ...   return x / jax.lax.psum(x, axis_name='i')
  >>>
  >>> @partial(pmap, axis_name='i', devices=jax.devices()[-2:])
  ... def f2(x):
  ...   return jax.lax.psum(x ** 2, axis_name='i')
  >>>
  >>> print(f1(jnp.arange(6.)))  # doctest: +SKIP
  [0.         0.06666667 0.13333333 0.2        0.26666667 0.33333333]
  >>> print(f2(jnp.array([2., 3.])))  # doctest: +SKIP
  [ 13.  13.]
  """
  if devices is not None:
    if not devices:
      raise ValueError("'devices' argument to pmap must be non-empty, or None.")
    devices = tuple(devices)
  axis_name, static_broadcasted_tuple, donate_tuple = _prepare_pmap(
      fun, axis_name, static_broadcasted_argnums, donate_argnums, in_axes,
      out_axes)
  wrapped_fun = _pmap_wrap_init(fun, static_broadcasted_tuple)
  out_axes_flat, out_axes_tree = tree_flatten(out_axes)
  out_axes_flat = tuple(out_axes_flat)

  def infer_params(*args, **kwargs):
    process_count = xb.process_count(backend)
    trace_state_clean = core.trace_state_clean()
    dyn_f, dyn_argnums, dyn_args = _get_dyn_args(
        wrapped_fun, static_broadcasted_tuple, args)
    dyn_args_flat, dyn_args_tree = tree_flatten((dyn_args, kwargs))
    in_axes_flat = _get_in_axes_flat(
        in_axes, dyn_argnums, dyn_args, kwargs, len(dyn_args_flat),
        dyn_args_tree)
    local_axis_size = _mapped_axis_size(dyn_args_flat, in_axes_flat)
    donated_invars = _get_donated_invars(
        donate_tuple, dyn_args_tree, len(dyn_args_flat))
    mesh_devices = _get_mesh_devices(
        devices, backend, local_axis_size, axis_size, trace_state_clean)
    cached = _cached_shard_map(
        dyn_f, dyn_args_tree, in_axes_flat, out_axes_flat, out_axes_tree,
        donated_invars, mesh_devices, axis_name)
    jitted_f = (cached.jitted_f_with_shardings if trace_state_clean
                else cached.jitted_f)
    if process_count > 1:
      dyn_args_flat = host_local_array_to_global_array(
          dyn_args_flat, cached, trace_state_clean, donated_invars
      )
    return (cached, jitted_f, dyn_args_flat, dyn_args_tree, donate_tuple,
            process_count, trace_state_clean)

  @util.wraps(fun)
  def wrapped(*args, **kwargs):
    cached, jitted_f, dyn_args_flat, _, _, process_count, trace_state_clean = (
        infer_params(*args, **kwargs))
    out = jitted_f(*dyn_args_flat)
    if process_count > 1:
      out = global_array_to_host_local_array(out, cached, trace_state_clean)
    return out

  def lower(*args, **kwargs):
    _, jitted_f, args_flat, in_tree, donated_tuple, _, _ = infer_params(
        *args, **kwargs
    )
    abstract_args = list(map(core.shaped_abstractify, args_flat))
    args_info = stages.make_args_info(in_tree, abstract_args, donated_tuple)
    lowered = jitted_f.trace(*args_flat).lower()
    # NOTE(dsuo): 调用 .compile()(*inputs) 会失败，因为我们 jit 过的函数
    # 不具备主机本地 <> 全局转换的概念。
    return stages.Lowered(
        lowered._lowering,
        args_info,
        lowered.out_tree,
        no_kwargs=lowered._no_kwargs,
    )

  wrapped.lower = lower  # pyrefly: ignore[missing-attribute]
  return wrapped


def _prepare_pmap(fun, axis_name, static_broadcasted_argnums,
                      donate_argnums, in_axes, out_axes):
  # axis_size 是一个可选整数，表示全局轴大小。被映射轴在所有进程上
  # 聚合后的大小必须与
  # 给定值匹配。
  check_callable(fun)
  axis_name = "_internal_pmap_axis_name" if axis_name is None else axis_name
  static_broadcasted_tuple = _ensure_index_tuple(static_broadcasted_argnums)
  donate_tuple = rebase_donate_argnums(
      _ensure_index_tuple(donate_argnums), static_broadcasted_tuple)

  if not all(type(l) is int for l in tree_leaves(in_axes)):
    raise TypeError("pmap in_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {in_axes}.")
  if not all(type(l) is int for l in tree_leaves(out_axes)):
    raise TypeError("pmap out_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {out_axes}.")

  return axis_name, static_broadcasted_tuple, donate_tuple


class CachedShardMap(NamedTuple):
  """pmap 的核心缓存结果。

  Attributes:
    pmapped: 经 shard_map 变换后的函数。
    in_specs_flat: 用于数组转换的展平后输入 PartitionSpec。
    local_devices: 本地 mesh 中的设备列表。
    in_local_shardings: 每个输入在本地 mesh 上的 NamedSharding。
    in_global_shardings: 每个输入在全局 mesh 上的 NamedSharding。
    mesh: 本次 pmap 调用使用的全局 Mesh。
    out_specs: 作为 pytree 前缀的输出 PartitionSpec。
    out_local_shardings_thunk: 缓存的 thunk，为输出 pspec 返回
      (local, global) 分片对。
    donate_argnums: 被捐赠参数的索引。
    out_global_shardings: 作为 pytree 的输出 NamedSharding。
    jitted_f: 预先缓存的、不带显式分片的 jit 包装器。
    jitted_f_with_shardings: 预先缓存的、带输入/输出分片的 jit 包装器。
  """

  pmapped: Callable[..., Any]
  in_specs_flat: tuple[sharding_impls.PartitionSpec, ...]
  local_devices: list[xc.Device]
  in_local_shardings: list[sharding_impls.NamedSharding]
  in_global_shardings: list[sharding_impls.NamedSharding]
  mesh: Mesh
  out_specs: Any  # PartitionSpec 的 pytree
  out_local_shardings_thunk: Callable[
      [sharding_impls.PartitionSpec],
      tuple[sharding_impls.NamedSharding, sharding_impls.NamedSharding],
  ]
  donate_argnums: list[int]
  out_global_shardings: Any  # NamedSharding 的 pytree
  jitted_f: Any
  jitted_f_with_shardings: Any


@lu.cache
def _cached_shard_map(fun, in_tree, in_axes_flat, out_axes_flat, out_axes_tree,
                      donated_invars, mesh_devices, axis_name):
  mesh = Mesh(mesh_devices, (axis_name,))
  out_axes = tree_unflatten(out_axes_tree, list(out_axes_flat))
  in_specs = tuple(map(partial(_axes_to_pspec, axis_name), in_axes_flat))
  out_specs = tree_map(
      partial(_axes_to_pspec, axis_name), out_axes, is_leaf=lambda x: x is None
  )
  def _fun(*flat_args):
    args = tree_map(
        lambda x, ax: x if ax is None else lax.squeeze(x, [ax]),
        flat_args,
        in_axes_flat,
    )
    args, kwargs = tree_unflatten(in_tree, args)
    out = fun.call_wrapped(*args, **kwargs)
    out_flat, out_tree = tree_flatten(out)
    out_axes_flat = broadcast_prefix(out_axes, out, is_leaf=lambda x: x is None)
    out_flat = tree_map(
        lambda x, ax: x if ax is None else lax.expand_dims(x, [ax]),
        out_flat,
        out_axes_flat,
    )
    return tree_unflatten(out_tree, out_flat)
  _pmapped = shard_map(_fun, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                       check_vma=False, axis_names=set(mesh.axis_names))
  # 现在多主机模式下捐赠是安全的，因为 host_local_array_to_global_array
  # 会复制捐赠的数组，而不是重新包装它们（重新包装会共享缓冲区）。
  donate_argnums = [i for i, val in enumerate(donated_invars) if val]

  # out_specs 是 pytree，因此用 tree_map 把它转换为分片
  get_sharding = (
      lambda spec: sharding_impls.NamedSharding(mesh, spec)
      if spec is not None else spec)
  out_global_shardings = tree_map(
      get_sharding, out_specs, is_leaf=lambda x: x is None)

  @util.cache()
  def out_local_shardings_thunk(pspec):
    return (
        sharding_impls.NamedSharding(mesh.local_mesh, pspec),
        sharding_impls.NamedSharding(mesh, pspec),
    )

  local_devices = list(mesh.local_mesh.devices.flat)
  in_local_shardings = [
      sharding_impls.NamedSharding(mesh.local_mesh, p) for p in in_specs]
  in_global_shardings = [
      sharding_impls.NamedSharding(mesh, p) for p in in_specs]
  jitted_f = api.jit(_pmapped, donate_argnums=donate_argnums)
  jitted_f_with_shardings = api.jit(
      _pmapped,
      donate_argnums=donate_argnums,
      in_shardings=tuple(in_global_shardings),
      out_shardings=out_global_shardings,
  )
  return CachedShardMap(
      pmapped=_pmapped,
      in_specs_flat=in_specs,
      local_devices=local_devices,
      in_local_shardings=in_local_shardings,
      in_global_shardings=in_global_shardings,
      mesh=mesh,
      out_specs=out_specs,
      out_local_shardings_thunk=out_local_shardings_thunk,
      donate_argnums=donate_argnums,
      out_global_shardings=out_global_shardings,
      jitted_f=jitted_f,
      jitted_f_with_shardings=jitted_f_with_shardings,
  )


def _mapped_axis_size(args, in_axes):
  """从第一个被映射的参数推断轴大小。

  shard_map 已经对所有参数做了检查，所以只需看第一个参数。

  Args:
    args: 展平的参数列表。
    in_axes: 展平的轴索引元组（每个参数为 int 或 None）。

  Returns:
    被映射轴的大小。

  Raises:
    ValueError: 若没有任何参数带被映射轴。
  """
  if args and in_axes:
    # 快速路径：检查第一个参数/轴（最常见的情况）。
    if in_axes[0] is not None and hasattr(args[0], "shape"):
      return int(args[0].shape[in_axes[0]])
    # 慢速路径：扫描第一个被映射的参数。
    if isinstance(in_axes, tuple):
      for arg, ax in zip(args, in_axes):
        if ax is not None and hasattr(arg, "shape"):
          return int(arg.shape[ax])
  raise ValueError("pmap requires at least one argument with a mapped axis.")


def _pmap_wrap_init(f, static_broadcasted_tuple):
  """为 pmap 创建带 DebugInfo 的包装后的函数。

  Args:
    f: 要包装的函数。
    static_broadcasted_tuple: 静态参数索引的元组。

  Returns:
    可用于 pmap 的 lu.WrappedFun。
  """
  # 由签名计算 arg_names，排除静态参数序号
  if (signature := fun_signature(f)) is not None:
    static_set = frozenset(static_broadcasted_tuple)
    arg_names = tuple(
        name
        for i, name in enumerate(signature.parameters.keys())
        if i not in static_set
    )
  else:
    arg_names = None
  dbg = lu.DebugInfo("pmap", fun_sourceinfo(f), arg_names, None)
  return lu.wrap_init(f, debug_info=dbg)


def _get_dyn_args(wrapped_f, static_broadcasted_tuple, args):
  """在处理静态参数之后提取动态参数与参数序号。

  Args:
    wrapped_f: 包装后的函数。
    static_broadcasted_tuple: 静态参数索引的元组。
    args: 位置参数。

  Returns:
    dyn_f: 已绑定静态参数的函数
    dyn_argnums: 动态参数索引列表（若无静态参数则为 None）
    dyn_args: 动态位置参数（已移除静态参数之后）

  Raises:
    ValueError: 若 static_broadcasted_argnums 超出参数个数。
  """

  if static_broadcasted_tuple:
    if max(static_broadcasted_tuple) >= len(args):
      raise ValueError(
          "pmapped function has"
          f" static_broadcasted_argnums={static_broadcasted_tuple} but was"
          f" called with only {len(args)} positional"
          f" argument{'s' if len(args) > 1 else ''}. All static broadcasted"
          " arguments must be passed positionally."
      )
    dyn_argnums = [
        i for i in range(len(args)) if i not in static_broadcasted_tuple
    ]
    wrapped_f, dyn_args = argnums_partial(wrapped_f, dyn_argnums, args)
  else:
    dyn_argnums = None
    dyn_args = args
  return wrapped_f, dyn_argnums, dyn_args


def _get_in_axes_flat(
    in_axes, dyn_argnums, dyn_args, kwargs, num_flat_args, in_tree
):
  """根据 in_axes 前缀与参数结构计算展平的 in_axes 元组。

  Args:
    in_axes: 原始的 in_axes 指定。
    dyn_argnums: 动态（非静态）位置参数的索引；
      若无静态参数则为 None。
    dyn_args: 动态位置参数（已移除静态参数之后）。
    kwargs: 关键字参数。
    num_flat_args: 展平参数的总数。
    in_tree: (dyn_args, kwargs) 的 PyTreeDef。

  Returns:
    展平的轴索引元组（每个展平参数为 int 或 None）。

  Raises:
    ValueError: 若 in_axes 不是参数结构的合法前缀。
  """
  # 由 in_axes 和 dyn_argnums 计算 dyn_in_axes
  if dyn_argnums is not None and isinstance(in_axes, tuple):
    dyn_in_axes = tuple(in_axes[i] for i in dyn_argnums)
  else:
    dyn_in_axes = in_axes

  # 快速路径：对常见简单情形避免使用 broadcast_prefix
  in_axes_flat = None

  if isinstance(dyn_in_axes, int):
    if dyn_in_axes == 0:
      # 最常见的情况：所有参数都在轴 0 上映射，包括 kwargs（它们也得到 0）
      in_axes_flat = (0,) * num_flat_args
    elif not kwargs:
      # 无 kwargs：把单个 in_axes 广播到所有位置参数的叶子
      in_axes_flat = (dyn_in_axes,) * num_flat_args
  elif dyn_in_axes is None and not kwargs:
    # 少见的情况：不映射，也没有 kwargs
    in_axes_flat = (None,) * num_flat_args
  elif (
      not kwargs
      and isinstance(dyn_in_axes, tuple)
      and all(isinstance(ax, int) or ax is None for ax in dyn_in_axes)
  ):
    # 无 kwargs：检查它是否是匹配位置参数的简单展平元组
    if len(dyn_in_axes) == len(dyn_args) and num_flat_args == len(dyn_args):
      # 每个位置参数都是叶子（没有嵌套结构）
      in_axes_flat = dyn_in_axes

  # 慢速路径：对复杂情况使用 broadcast_flattened_prefix_with_treedef
  if in_axes_flat is None:
    try:
      # 展平 in_axes 前缀树（把 None 视为叶子）
      flat_in_axes_prefix, in_axes_tree = tree_flatten(
          (dyn_in_axes, 0), is_leaf=lambda x: x is None
      )
      in_axes_flat = tuple(
          broadcast_flattened_prefix_with_treedef(
              flat_in_axes_prefix, in_axes_tree, in_tree
          )
      )
    except ValueError:
      e, *_ = prefix_errors((dyn_in_axes, 0), (dyn_args, kwargs))
      ex = e("pmap in_axes")
      (msg,) = ex.args
      msg += (
          "\n\nThe 'full pytree' here is the tuple of arguments passed "
          "positionally to the pmapped function, and the value of `in_axes` "
          "must be a tree prefix of that tuple. But it was not a prefix."
      )
      if kwargs:
        msg += (
            "\n\nWhen some arguments are passed by keyword to the pmapped "
            "function, they are not included in the comparison to `in_axes`. "
            "Instead, each argument passed by keyword is mapped over its "
            "leading axis. See the description of `in_axes` in the `pmap` "
            "docstring: "
            "https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html#jax.pmap"
        )
      msg += (
          "\n\nCheck that the value of the `in_axes` argument to `pmap` "
          "is a tree prefix of the tuple of arguments passed positionally to "
          "the pmapped function."
      )
      raise ValueError(msg) from None

  return in_axes_flat


def _get_donated_invars(donate_tuple, in_tree, num_flat_args):
  """为参数计算捐赠向量。

  Args:
    donate_tuple: 被捐赠参数索引的元组。
    in_tree: 输入结构的 PyTreeDef。
    num_flat_args: 展平参数的个数。

  Returns:
    布尔值元组，指示哪些展平参数被捐赠。
  """

  if donate_tuple and not config.debug_nans.value:
    return donation_vector(donate_tuple, (), in_tree)
  else:
    return (False,) * num_flat_args


@util.cache()
def _get_mesh_devices(devices, backend, local_axis_size, axis_size,
                      trace_state_clean):
  """根据上下文计算实际生效的 mesh 设备。

  Args:
    devices: mesh 设备元组。
    backend: 要使用的后端。
    local_axis_size: 本地（每进程）轴大小。
    axis_size: 用户指定的全局轴大小（可选）。
    trace_state_clean: 若处于执行模式（而非追踪）则为 True。

  Returns:
    经过适当切片的实际生效 mesh 设备元组。

  Raises:
    ValueError: 若单进程下 axis_size 与推断出的大小不匹配。
  """
  process_count = xb.process_count(backend)

  # 在单进程模式下校验显式指定的 axis_size
  if (process_count == 1 and axis_size is not None and
      axis_size != local_axis_size):
    raise ValueError(
        f"Specified axis_size {axis_size} doesn't match received "
        f"axis_size {local_axis_size}.")

  # 计算 global_axis_size
  if axis_size is not None:
    global_axis_size = axis_size
  elif process_count > 1:
    global_axis_size = local_axis_size * process_count
    # 校验所有进程拥有相同数量的本地设备
    assert all(
        len(xb.local_devices(pi, backend)) == xb.local_device_count(backend)
        for pi in range(process_count))
  else:
    global_axis_size = local_axis_size

  # 确定 mesh 设备
  if devices is not None:
    mesh_devices = devices
  elif process_count > 1:
    # 多进程：按进程（主机）对设备分组，以获得最优的集合通信
    # 性能。这与旧 pmap 的设备排序一致，后者在嵌套循环中使用
    # local_devices(process_index)，从而保证来自
    # 同一主机的设备在 mesh 中连续。
    mesh_devices = tuple(
        d
        for process_index in range(process_count)
        for d in xb.local_devices(process_index, backend)
    )
  elif backend is not None:
    mesh_devices = tuple(xb.devices(backend=backend))
  else:
    mesh_devices = tuple(xb.devices())

  if not trace_state_clean and process_count > 1:
    # 多主机下的追踪：使用本地设备
    return tuple(xb.local_devices(backend=backend)[:local_axis_size])
  else:
    return mesh_devices[:global_axis_size]


@util.cache()
def _local_to_global_aval(shape, dtype, sharding):
  """由本地形状计算全局 aval。"""
  pspec_prepared = sharding_impls.prepare_axis_resources(sharding.spec, "pspec")
  local_aval = core.ShapedArray(shape, dtype)
  return pxla.mesh_local_to_global(
      sharding.mesh,
      sharding_impls.get_array_mapping(pspec_prepared),
      local_aval,
  )


@util.cache()
def _global_to_local_aval(shape, dtype, sharding):
  """由全局形状计算本地 aval。"""
  pspec_prepared = sharding_impls.prepare_axis_resources(sharding.spec, "pspec")
  global_aval = core.ShapedArray(shape, dtype)
  return pxla.mesh_global_to_local(
      sharding.mesh,
      sharding_impls.get_array_mapping(pspec_prepared),
      global_aval,
  )


@util.cache()
def _local_device_indices(local_sharding, shape):
  """为切片数组而缓存的设备索引。"""
  return tuple(local_sharding.devices_indices_map(shape).values())


@util.cache()
def _is_sharding_equivalent(sharding_a, sharding_b, ndim):
  """检查分片是否等价于 NamedSharding(mesh.local_mesh, pspec)。"""
  return sharding_a.is_equivalent_to(sharding_b, ndim)


@util.cache()
def _get_out_shardings(out_tree, pspecs, out_shardings_thunk):
  """获取展平的输出分片，同时完成 pspec 展平与分片查找。"""
  out_pspecs_flat = pjit_lib.flatten_axis_resources(
      "output pspecs", out_tree, pspecs, tupled_args=True
  )
  return tuple(zip(*[out_shardings_thunk(p) for p in out_pspecs_flat]))


def host_local_array_to_global_array(
    dyn_args_flat, cached, trace_state_clean, donated_invars
):
  """为多主机 pmap 把主机本地数组转换为全局数组。

  Args:
    dyn_args_flat: 展平的输入数组列表。
    cached: 带 mesh 与分片信息的 CachedPmap 元组。
    trace_state_clean: 若处于执行模式（而非追踪）则为 True。
    donated_invars: 布尔值元组，指示哪些参数被捐赠。对于需要走
      慢速路径的捐赠参数，我们会删除原数组以释放
      内存。

  Returns:
    转换后的全局数组。
  """
  if not trace_state_clean:
    import jax.experimental.multihost_utils as mhu  # pyrefly: ignore[missing-import]

    return list(
        mhu.host_local_array_to_global_array(
            tuple(dyn_args_flat), cached.mesh, cached.in_specs_flat
        )
    )

  in_local_shardings = cached.in_local_shardings
  in_global_shardings = cached.in_global_shardings

  if dyn_args_flat and isinstance(
      dyn_args_flat[0], (core.Tracer, core.AbstractValue)
  ):
    return dyn_args_flat

  for i, arr in enumerate(dyn_args_flat):
    local_sharding = in_local_shardings[i]
    global_sharding = in_global_shardings[i]
    donated = donated_invars[i]
    prng_impl = None
    typ = type(arr)
    if typ is array.ArrayImpl and not arr.is_fully_addressable:
      continue
    if typ is not array.ArrayImpl:
      if typ is prng.PRNGKeyArray:
        prng_impl = arr.dtype._impl
        arr = arr._base_array
      arr = np.asarray(arr)
      dtype = arr.dtype
      if dtype == dtypes.float0:
        arr = np.zeros(arr.shape, dtype=bool)
      if dtype != dtypes.canonicalize_dtype(dtype):
        arr = dtypes.canonicalize_value(arr)
    shape, dtype = arr.shape, arr.dtype
    typ = type(arr)

    global_aval = _local_to_global_aval(shape, dtype, global_sharding)
    if typ == array.ArrayImpl and _is_sharding_equivalent(
        arr.sharding, local_sharding, len(arr.shape)
    ):
      # 快速路径：不复制地重新包装（与原数组共享缓冲区）。
      # 对于捐赠参数，jit 的捐赠会使共享缓冲区失效，
      # 这正是预期行为——原数组会变为无效。
      dyn_args_flat[i] = arr._rewrap_with_aval_and_sharding(
          global_aval, global_sharding
      )
    else:
      # 慢速路径：切片并 device_put（会创建新缓冲区）。
      # 对于捐赠参数，我们必须显式删除原数组以释放内存。
      arrays = [
          arr[idx] for idx in _local_device_indices(local_sharding, shape)
      ]
      dyn_args_flat[i] = pxla.batched_device_put(
          global_aval,
          global_sharding,
          arrays,
          list(local_sharding._device_assignment),
      )
      if donated and typ is array.ArrayImpl:
        warnings.warn(
            "Donated pmap argument required resharding. This causes a brief "
            "2x memory spike before the original is freed. For optimal "
            "donation, ensure inputs are correctly sharded before pmap.",
            stacklevel=4,
        )
        arr.delete()
    if prng_impl is not None:
      dyn_args_flat[i] = prng.PRNGKeyArray(prng_impl, dyn_args_flat[i])

  return dyn_args_flat


def global_array_to_host_local_array(out, cached, trace_state_clean):
  """为多主机 pmap 输出把全局数组转换为主机本地数组。

  Args:
    out: jit 过的函数产生的输出 pytree。
    cached: 带 mesh 与分片信息的 CachedPmap 元组。
    trace_state_clean: 若处于执行模式（而非追踪）则为 True。

  Returns:
    主机本地的输出 pytree。
  """
  if not trace_state_clean:
    import jax.experimental.multihost_utils as mhu  # pyrefly: ignore[missing-import]

    return mhu.global_array_to_host_local_array(
        out, cached.mesh, cached.out_specs
    )

  out_flat, out_tree = tree_flatten(out)
  out_local_shardings, out_global_shardings = _get_out_shardings(
      out_tree, cached.out_specs, cached.out_local_shardings_thunk
  )

  if out_flat and isinstance(out_flat[0], (core.Tracer, core.AbstractValue)):
    return out

  for i, arr in enumerate(out_flat):
    local_sharding = out_local_shardings[i]
    global_sharding = out_global_shardings[i]
    prng_impl = None
    typ = type(arr)
    if typ is array.ArrayImpl and arr.is_fully_addressable:
      continue
    if typ is not array.ArrayImpl:
      if typ is prng.PRNGKeyArray:
        prng_impl = arr.dtype._impl
        arr = arr._base_array
      try:
        _ = arr.shape
      except AttributeError:
        arr = np.array(arr)
      dtype = arr.dtype
      if dtype == dtypes.float0:
        arr = np.zeros(arr.shape, dtype=bool)
      if dtype != dtypes.canonicalize_dtype(dtype):
        arr = dtypes.canonicalize_value(arr)
    shape, dtype = arr.shape, arr.dtype
    typ = type(arr)

    local_aval = _global_to_local_aval(shape, dtype, global_sharding)
    if typ == array.ArrayImpl:
      if not _is_sharding_equivalent(arr.sharding, global_sharding, len(shape)):
        arr = api.device_put(arr, global_sharding)
      out_flat[i] = arr._rewrap_with_aval_and_sharding(
          local_aval, local_sharding
      )
    else:
      arrays = [
          arr[idx] for idx in _local_device_indices(local_sharding, shape)
      ]
      out_flat[i] = pxla.batched_device_put(
          local_aval,
          local_sharding,
          arrays,
          list(local_sharding._device_assignment),
      )
    if prng_impl is not None:
      out_flat[i] = prng.PRNGKeyArray(prng_impl, out_flat[i])

  return tree_unflatten(out_tree, out_flat)
