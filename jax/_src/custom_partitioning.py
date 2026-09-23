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

# 文件职责：实现 JAX 的“自定义分区”（custom partitioning）公开 API。
# 该模块提供 `custom_partitioning` 装饰器，让用户为自定义算子注册 SPMD 分区
# 规则：它把被装饰函数包装成 `custom_partitioning` 原语，在 XLA 图中插入
# `CustomCallOp`，并把 `partition`、`propagate_user_sharding`、
# `infer_sharding_from_operands` 以及 Shardy 的 `sharding_rule` 等回调交给
# XLA 的 SPMD 分区器；本模块原先位于 `jax.experimental`，为打破导入循环而迁出。

"""自定义分区 API 的实现。

该模块原先位于 ``jax.experimental``，为避免导入循环而迁出。
"""

from __future__ import annotations

from functools import partial
import inspect
from typing import Any
from collections.abc import Callable
import weakref

import numpy as np

from jax._src import api
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import custom_api_util
from jax._src import dispatch
from jax._src import errors
from jax._src import flattree as ft
from jax._src import mesh as mesh_lib
from jax._src import sharding_impls
from jax._src import tree_util
from jax._src import xla_bridge as xb
from jax._src.custom_partitioning_sharding_rule import sdy_sharding_rule_to_mlir, SdyShardingRule, str_to_sdy_sharding_rule
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.sharding import Sharding


def _resolve_kwargs(fun, args, kwargs):
  ba = inspect.signature(fun).bind(*args, **kwargs)
  ba.apply_defaults()
  if ba.kwargs:
    raise TypeError("keyword arguments could not be resolved to positions")
  else:
    return ba.args


class _ShardingCallbackInfo:

  def __init__(self, propagate_user_sharding, partition, to_mesh_pspec_sharding,
      in_tree, out_tree, infer_sharding_from_operands, module_context, mesh,
      static_args):
    self.propagate_user_sharding = propagate_user_sharding
    self.partition = partition
    self.to_mesh_pspec_sharding = to_mesh_pspec_sharding
    self.in_tree = in_tree
    self.out_tree = out_tree
    self.infer_sharding_from_operands = infer_sharding_from_operands
    self.module_context = module_context
    self.mesh = mesh
    self.static_args = static_args

  def unflatten_arg_shape(self, s, sharding):
    return _to_jax_sharded_shape(
        s, self.to_mesh_pspec_sharding(sharding, len(s.dimensions()))
    )

  def unflatten_arg_shapes(self, arg_shapes, arg_shardings):
    return self.in_tree.unflatten(
        [
            self.unflatten_arg_shape(s, sharding)
            for s, sharding in zip(arg_shapes, arg_shardings)
        ]
    )


_sharding_callbacks = weakref.WeakValueDictionary()

_CUSTOM_PARTITIONING_CALL_NAME = "CustomSPMDPartitioning"


def _to_jax_shape(s):
  return core.ShapedArray(s.dimensions(), s.numpy_dtype())


def _to_jax_sharded_shape(s, sharding):
  return api.ShapeDtypeStruct(
      s.dimensions(), s.numpy_dtype(), sharding=sharding
  )


def _pack_result_sharding(shape, result_shardings):
  if shape.is_tuple():
    return xc.HloSharding.tuple_sharding(shape, result_shardings)
  else:
    return result_shardings[0]


def _flatten_sharding(tree, shardings, shapes):
  return [
      _to_hlo_sharding(sharding, len(shape.dimensions()))
      for sharding, shape in zip(
          tree.flatten_up_to(shardings), shapes
      )
  ]


def _custom_partitioning_propagate_user_sharding(user_sharding, shape,
                                                 backend_string):
  info = _sharding_callbacks[backend_string]
  if info.propagate_user_sharding is None:
    return user_sharding
  if shape.is_tuple():
    user_shapes = shape.tuple_shapes()
    user_shardings = user_sharding.tuple_elements()
  else:
    user_shapes = (shape,)
    user_shardings = (user_sharding,)
  user_shape = info.out_tree.unflatten(
      [
          info.unflatten_arg_shape(s, sharding)
          for s, sharding in zip(user_shapes, user_shardings)
      ]
  )
  result_sharding = info.propagate_user_sharding(
      *info.static_args, info.mesh, user_shape
  )
  result_shardings = _flatten_sharding(
      info.out_tree, result_sharding, user_shapes)
  return _pack_result_sharding(shape, result_shardings)


def _to_hlo_sharding(sharding, num_dimensions):
  if not isinstance(sharding, Sharding):
    raise ValueError("Custom Partitioning rules must return Sharding.")
  return sharding._to_xla_hlo_sharding(num_dimensions)


def _custom_partitioning_partition(arg_shapes, arg_shardings, result_shape,
                                   result_sharding, backend_string):
  info = _sharding_callbacks[backend_string]
  if result_shape.is_tuple():
    result_shapes = result_shape.tuple_shapes()
    result_shardings = result_sharding.tuple_elements()
  else:
    result_shapes = (result_shape,)
    result_shardings = (result_sharding,)
  mesh, lower_fn, result_sharding, arg_shardings = info.partition(
      *info.static_args,
      info.mesh,
      info.unflatten_arg_shapes(arg_shapes, arg_shardings),
      info.out_tree.unflatten(
          [
              info.unflatten_arg_shape(s, sharding)
              for s, sharding in zip(result_shapes, result_shardings)
          ]
      ),
  )
  module_context = info.module_context

  result_shardings = _flatten_sharding(
      info.out_tree, result_sharding, result_shapes)
  arg_shardings = _flatten_sharding(info.in_tree, arg_shardings, arg_shapes)
  tiled_args = [
      _to_jax_shape(sharding.tile(s))
      for sharding, s in zip(arg_shardings, arg_shapes)
  ]
  tiled_results = [
      _to_jax_shape(sharding.tile(s))
      for sharding, s in zip(result_shardings, result_shapes)
  ]
  closed_jaxpr = api.make_jaxpr(lower_fn, axis_env=list(mesh.shape.items()))(
      *info.in_tree.unflatten(tiled_args)
  )
  if ([(o.shape, o.dtype) for o in closed_jaxpr.out_avals] !=
      [(t.shape, t.dtype) for t in tiled_results]):
    raise ValueError(
        "Mismatch in result shapes. %s vs %s"
        % (repr(closed_jaxpr.out_avals), repr(tiled_results))
    )
  axis_context = sharding_impls.SPMDAxisContext(mesh, frozenset(mesh.axis_names))
  with core.extend_axis_env_nd(mesh.shape.items()):
    module = mlir.build_mlir_module_helper(
        closed_jaxpr,
        name="tmp_xla_computation",
        platforms=module_context.platforms,
        backend=module_context.backend,
        axis_context=axis_context,
    )
  result_sharding = _pack_result_sharding(result_shape, result_shardings)
  return mlir.module_to_bytecode(module), arg_shardings, result_sharding


def _custom_partitioning_infer_sharding_from_operands(arg_shapes, arg_shardings,
                                                      result_shape,
                                                      backend_string):
  info = _sharding_callbacks[backend_string]
  if result_shape.is_tuple():
    result_shapes = result_shape.tuple_shapes()
  else:
    result_shapes = (result_shape,)
  result_sharding = info.infer_sharding_from_operands(
      *info.static_args,
      info.mesh,
      info.unflatten_arg_shapes(arg_shapes, arg_shardings),
      info.out_tree.unflatten([_to_jax_shape(s) for s in result_shapes]),
  )
  result_shardings = _flatten_sharding(
      info.out_tree, result_sharding, result_shapes)
  return _pack_result_sharding(result_shape, result_shardings)


custom_partitioning_p = core.Primitive("custom_partitioning")
custom_partitioning_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(custom_partitioning_p)


def _custom_partitioning_abstract_eval(*avals, call, in_tree, out_tree,
                                       propagate_user_sharding, partition,
                                       infer_sharding_from_operands,
                                       decode_shardings,
                                       sharding_rule,
                                       static_args):
  del in_tree, out_tree, propagate_user_sharding, partition
  del infer_sharding_from_operands, decode_shardings, sharding_rule
  del static_args
  return call.out_avals


def _custom_partitioning_impl(*args, call, in_tree, out_tree,
                              propagate_user_sharding,
                              partition, infer_sharding_from_operands,
                              decode_shardings, sharding_rule, static_args):
  del in_tree, out_tree, propagate_user_sharding, partition
  del infer_sharding_from_operands, decode_shardings, static_args, sharding_rule
  return core.jaxpr_as_fun(call)(*args)


custom_partitioning_p.def_abstract_eval(_custom_partitioning_abstract_eval)
custom_partitioning_p.def_impl(_custom_partitioning_impl)


def _check_for_tracers(x):
  if any(isinstance(leaf, core.Tracer) for leaf in tree_util.tree_leaves(x)):
    raise errors.UnexpectedTracerError(
        "Found a JAX Tracer object passed as an argument to a"
        "custom_partitioning function in a position indicated as static by"
        "static_argnums. "
    )


@custom_api_util.register_custom_decorator_type
class custom_partitioning:
  """把带有自定义 SPMD 降级规则的 ``CustomCallOp`` 插入 XLA 图。

  .. code-block:: python

    @custom_partitioning
    def f(*args):
      return ...

    def propagate_user_sharding(mesh, user_shape):
      '''根据用户的 shape.sharding 更新该算子的分片。'''
      user_sharding = jax.tree.map(lambda x: x.sharding, user_shape)

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(*args):
        ... builds computation on per-device shapes ...
      result_shardings = jax.tree.map(lambda x: x.sharding, result_shape)
      arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
      # result_sharding 与 arg_shardings 可以选择性地修改，分区器会插入
      # 集合通信来重塑形状。
      return mesh, lower_fn, result_sharding, arg_shardings

    def infer_sharding_from_operands(mesh, arg_shapes, shape):
      '''由操作数的分片计算结果分片。'''
      arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)


    f.def_partition(partition, propagate_user_sharding,
                    infer_sharding_from_operands=infer_sharding_from_operands,
                    sharding_rule='i j -> 'i j')

  传递给 ``def_partition`` 的参数如下：

  * ``propagate_user_sharding``：可调用对象，接收用户（DAG 中）的分片，
    并返回一个新的 `NamedSharding` 建议值。默认值为 None。
    最简单的实现就是原样返回输入分片。
  * ``partition``：可调用对象，接收 SPMD 建议的分片形状和
    分片规格，返回 mesh、逐分片降级函数，以及最终的
    输入与输出分片规格（SPMD 分区器会重新分区输入以匹配它们）。
    返回 mesh 是为了在未提供 mesh 时配置集合通信的 axis_names。
  * ``infer_sharding_from_operands``：可调用对象，由为每个参数选定的
    ``NamedSharding`` 计算输出的 ``NamedSharding``。
  * ``decode_shardings``：设为 True 时，尽可能把输入的 ``GSPMDSharding``
    转换为 ``NamedSharding``。如果用户没有提供上下文 mesh，
    则可能无法转换。
  * ``sharding_rule``：一个 SdyShardingRule 对象、描述分片规则的类 Einsum 记法
    字符串，或者能产出上述两者之一的可调用对象。我们把 Einsum 记法中的
    索引标签称为分片规则中的因子（factor）。我们借鉴了
    einops.rearrange 字符串的做法，用空格分隔因子，
    并允许因子名包含多个字母。默认情况下，一个因子对应
    透传/逐元素维度。其他维度对应的因子可以通过下文描述的关键字
    参数指定。更多细节与示例参见
    `jax-shardy-guide <https://colab.sandbox.google.com/github/openxla/shardy/blob/main/docs/getting_started_jax.ipynb>`_
  * ``reduction_factors``：字符串元组，为字符串 `sharding_rule` 指定归约
    因子。归约因子对应出现在操作数中但不出现在结果中的维度，
    例如 matmul 运算中的收缩维度。如果归约因子被分片，
    结果就需要沿相同的轴做 all-reduce。
  * ``need_replication_factors``：字符串元组，为字符串 `sharding_rule`
    指定 need_replication 因子。need_replication 因子对应
    为了支持该实现而不应被分片的维度。
  * ``permutation_factors``：字符串元组，为字符串 `sharding_rule` 指定
    置换因子。置换因子对应一旦被分片就会触发 collective permute
    的维度。
  * ``factor_sizes``：由可变关键字参数组成的字典，为字符串
    `sharding_rule` 中仅用于复合因子的因子指定大小。

  当 config.use_shardy_partitioner.value 为 True 时使用 `sharding_rule`；
  否则使用 `propagate_user_sharding` 与
  `infer_sharding_from_operands`。

  可以用 static_argnums 把位置参数指定为静态参数。JAX 使用
  :code:`inspect.signature(fun)` 来解析这些位置参数。

  Examples:

    举例来说，假设我们想增强现有的 ``jax.numpy.fft.fft``。该函数沿最后一个
    维度计算 N 维输入的离散傅里叶变换，并沿前 N-1 个维度做批处理。
    但默认情况下，它会忽略输入的分片，把输入聚集到所有设备上。
    然而，由于 ``jax.numpy.fft.fft`` 是沿前 N-1 个维度做批处理的，
    这样做并无必要。我们将创建一个新的 ``my_fft`` 算子，
    它不改动前 `N-1` 个维度上的分片，只在需要时沿最后一个维度聚集输入。

    .. code-block:: python

      import jax
      from jax.sharding import NamedSharding
      from jax.experimental.custom_partitioning import custom_partitioning
      from jax.experimental.pjit import pjit
      from jax.sharding import PartitionSpec as P
      from jax.sharding import Mesh
      from jax.numpy.fft import fft
      import regex as re
      import numpy as np

      # 用于检测生成的 HLO 中 all-gather 或 dynamic-slice 的正则模式
      _PATTERN = '(dynamic-slice|all-gather)'

      # 对 N 维输入，保留前 N-1 个维度上的分片，
      # 但在最后一个维度上做复制
      def supported_sharding(sharding, shape):
          rank = len(shape.shape)
          max_shared_dims = min(len(sharding.spec), rank-1)
          names = tuple(sharding.spec[:max_shared_dims]) + tuple(None for _ in range(rank - max_shared_dims))
          return NamedSharding(sharding.mesh, P(*names))

      def partition(mesh, arg_shapes, result_shape):
          result_shardings = jax.tree.map(lambda x: x.sharding, result_shape)
          arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
          return mesh, fft, \
              supported_sharding(arg_shardings[0], arg_shapes[0]), \
              (supported_sharding(arg_shardings[0], arg_shapes[0]),)

      def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
          arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
          return supported_sharding(arg_shardings[0], arg_shapes[0])

      @custom_partitioning
      def my_fft(x):
          return fft(x)

      # 使用类 Einsum 记法指定分片规则。
      my_fft.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule='...i -> ...i')
      # 使用 SdyShardingRule 对象指定分片规则。
      my_fft.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule=SdyShardingRule(operand_mappings=((BATCHING, 'i'),), result_mappings=((BATCHING, 'i'),))))

    现在创建一个沿第一个轴分片的 2D 数组，把它传入 ``my_fft``，
    可以看到它仍然按预期分片，且输出与 ``fft`` 相同。
    不过，查看 HLO（使用
    ``lower(x).compile().runtime_executable().hlo_modules()``）会发现，
    ``my_fft`` 不产生任何 all-gather 或 dynamic-slice，而 ``fft`` 会产生。

    .. code-block::

      with Mesh(np.array(jax.devices()), ('x',)):
        x = np.asarray(np.random.randn(32*1024, 1024), dtype=np.complex64)
        y = pjit(lambda x: x, in_shardings=None, out_shardings=P('x'))(x)
        pjit_my_fft = pjit(my_fft, in_shardings=P('x'), out_shardings=P('x'))
        pjit_fft    = pjit(fft,    in_shardings=P('x'), out_shardings=P('x'))
        print(pjit_my_fft(y))
        print(pjit_fft(y))
        # 因为 x 是 2D 数组，my_fft 的 HLO 中不存在 dynamic-slice 或 all-gather
        assert(re.search(_PATTERN, pjit_my_fft.lower(x).compile().runtime_executable().hlo_modules()[0].to_string()) is None)
        # fft 的 HLO 中存在 dynamic-slice 或 all-gather
        assert(re.search(_PATTERN, pjit_fft.lower(x).compile().runtime_executable().hlo_modules()[0].to_string())    is not None)

    .. code-block::

      # my_fft
      [[-38.840824   +0.j        -40.649452  +11.845365j
      ...
        -1.6937828  +0.8402481j  15.999859   -4.0156755j]]

      # jax.numpy.fft.fft
      [[-38.840824   +0.j        -40.649452  +11.845365j
        ...
        -1.6937828  +0.8402481j  15.999859   -4.0156755j]]

    由于 ``supported_sharding`` 中的逻辑，``my_fft`` 也能处理一维数组。
    不过在这种情况下，``my_fft`` 的 HLO 中确实会出现 dynamic-slice，因为最后一个
    维度是计算 FFT 所沿的维度，需要在计算开始前把它复制到所有
    设备上。

    .. code-block::

      with Mesh(np.array(jax.devices()), ('x',)):
        x = np.asarray(np.random.randn(32*1024*1024), dtype=np.complex64)
        y = pjit(lambda x: x, in_shardings=None, out_shardings=P('x'))(x)
        pjit_my_fft = pjit(my_fft, in_shardings=P('x'), out_shardings=P('x'))
        pjit_fft    = pjit(fft,    in_shardings=P('x'), out_shardings=P('x'))
        print(pjit_my_fft(y))
        print(pjit_fft(y))
        # 因为 x 是 1D 数组，my_fft 的 HLO 中存在 dynamic-slice 或 all-gather
        assert(re.search(_PATTERN, pjit_my_fft.lower(x).compile().runtime_executable().hlo_modules()[0].to_string()) is None)
        # fft 的 HLO 中存在 dynamic-slice 或 all-gather
        assert(re.search(_PATTERN, pjit_fft.lower(x).compile().runtime_executable().hlo_modules()[0].to_string())    is not None)

    .. code-block::

      # my_fft
      [    7.217285   +0.j     -3012.4937  +4287.635j   -405.83594 +3042.984j
      ...  1422.4502  +7271.4297j  -405.84033 -3042.983j
      -3012.4963  -4287.6343j]

      # jax.numpy.fft.fft
      [    7.217285   +0.j     -3012.4937  +4287.635j   -405.83594 +3042.984j
      ...  1422.4502  +7271.4297j  -405.84033 -3042.983j
      -3012.4963  -4287.6343j]

  """

  def __init__(self, fun, static_argnums=()):
    self.fun = fun
    self.partition = None
    self.static_argnums = static_argnums
    self.propagate_user_sharding = None
    self.infer_sharding_from_operands = None
    self.sharding_rule = None

  __getattr__: Any = custom_api_util.forward_attr

  def def_partition(self, partition, infer_sharding_from_operands=None,
                    propagate_user_sharding=None, decode_shardings=True,
                    sharding_rule=None, *, reduction_factors=(),
                    need_replication_factors=(), permutation_factors=(),
                    **factor_sizes):
    self.partition = partition
    self.propagate_user_sharding = propagate_user_sharding
    self.infer_sharding_from_operands = infer_sharding_from_operands
    self.decode_shardings = decode_shardings
    if (sharding_rule is None or isinstance(sharding_rule, Callable) or
        isinstance(sharding_rule, SdyShardingRule)):
      sharding_rule_dict = factor_sizes
      if len(reduction_factors) > 0:
        sharding_rule_dict["reduction_factors"] = reduction_factors
      if len(need_replication_factors) > 0:
        sharding_rule_dict["need_replication_factors"] = need_replication_factors
      if len(permutation_factors) > 0:
        sharding_rule_dict["permutation_factors"] = permutation_factors
      if sharding_rule_dict:
        raise ValueError(f"Unknown keyword arguments: {sharding_rule_dict}")
      self.sharding_rule = sharding_rule
    else:
      self.sharding_rule = str_to_sdy_sharding_rule(
          sharding_rule,
          reduction_factors=reduction_factors,
          need_replication_factors=need_replication_factors,
          permutation_factors=permutation_factors,
          **factor_sizes)
    return partition

  def __call__(self, *args, **kwargs):
    args = _resolve_kwargs(self.fun, args, kwargs)
    debug = api_util.debug_info("custom_partitioning", self.fun,
                                args, {},
                                static_argnums=self.static_argnums)
    if self.static_argnums:
      static_argnums = set(self.static_argnums)
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      static_args = tuple(args[i] for i in self.static_argnums)
      _check_for_tracers(static_args)
      dyn_args = tuple(args[i] for i in dyn_argnums)
      full_args = list(args)
      def f_(*dyn):
        for i, x in zip(dyn_argnums, dyn): full_args[i] = x
        return self.fun(*full_args)
    else:
      static_args = ()
      f_, dyn_args = self.fun, args
    args_flat, in_tree = tree_util.tree_flatten(dyn_args)
    in_avals = [core.typeof(x) for x in args_flat]
    mesh = mesh_lib.thread_resources.env.physical_mesh
    with core.extend_axis_env_nd(mesh.shape.items()):
      closed_call, out_avals = pe.trace_to_jaxpr(
          f_, ft.pack((ft.treedef_args_to_ft(in_tree, in_avals), {})), debug)
    assert not closed_call.consts

    propagate_user_sharding = None
    infer_sharding_from_operands = None
    sharding_rule = None
    if config.use_shardy_partitioner.value:
      if (self.sharding_rule is None and
          (self.propagate_user_sharding is not None or
            self.infer_sharding_from_operands is not None)):
        raise NotImplementedError(
            "Shardy is used, but sharding propagation callbacks instead of "
            "sharding_rule are provided. Need to provide sharding_rule to "
            "migrate to Shardy."
        )
      sharding_rule = self.sharding_rule
    else:
      propagate_user_sharding = self.propagate_user_sharding
      infer_sharding_from_operands = self.infer_sharding_from_operands

    out_flat = custom_partitioning_p.bind(
        *args_flat,
        call=closed_call,
        partition=self.partition,
        propagate_user_sharding=propagate_user_sharding,
        infer_sharding_from_operands=infer_sharding_from_operands,
        decode_shardings=self.decode_shardings,
        sharding_rule=sharding_rule,
        in_tree=in_tree,
        out_tree=out_avals.tree,
        static_args=static_args
    )
    return tree_util.tree_unflatten(out_avals.tree, out_flat)


def _custom_partitioning_lowering_rule(ctx: mlir.LoweringRuleContext, *values,
                                       call, in_tree, out_tree,
                                       propagate_user_sharding, partition,
                                       infer_sharding_from_operands,
                                       decode_shardings,
                                       sharding_rule,
                                       static_args):
  axis_context = ctx.module_context.axis_context
  if (isinstance(axis_context, sharding_impls.SPMDAxisContext) and
      set(axis_context.manual_axes) == set(axis_context.mesh.axis_names)):
    return mlir.lower_fun(core.jaxpr_as_fun(call), multiple_results=True)(ctx, *values)

  mesh = mesh_lib.thread_resources.env.physical_mesh
  if isinstance(axis_context, sharding_impls.ShardingContext):
    devices = axis_context.device_assignment
    if devices is None:
      raise AssertionError(
          'Please file a bug at https://github.com/jax-ml/jax/issues')
    am = axis_context.abstract_mesh
    if am is not None:
      mesh = mesh_lib.Mesh(np.array(devices).reshape(am.axis_sizes),
                           am.axis_names)
  elif isinstance(axis_context, sharding_impls.SPMDAxisContext):
    devices = axis_context.mesh._flat_devices_tuple
  else:
    devices = None

  if not devices or len(devices) == 1:
    return mlir.lower_fun(
        core.jaxpr_as_fun(call), multiple_results=True)(ctx, *values)

  if (not config.use_shardy_partitioner.value and
      infer_sharding_from_operands is None):
    function = call.jaxpr.debug_info.func_src_info
    raise NotImplementedError(
        f"Custom-partitioned function {function!r} does not support GSPMD "
        "sharding propagation rules. GSPMD is deprecated; please upgrade "
        "to and enable the Shardy partitioner "
        "(jax_use_shardy_partitioner=True, which is the default)."
    )

  def to_mesh_pspec_sharding(hlo_sharding: xc.HloSharding | None, ndim):
    if hlo_sharding is None:
      return hlo_sharding
    if mesh.empty or not decode_shardings:
      assert devices is not None
      return sharding_impls.GSPMDSharding(devices, hlo_sharding)
    pspec = sharding_impls.parse_flatten_op_sharding(
        hlo_sharding, mesh)[0]
    pspec = sharding_impls.PartitionSpec(*pspec, *((None,) * (ndim - len(pspec))))
    return sharding_impls.NamedSharding(mesh, pspec)

  sharding_callback_info = _ShardingCallbackInfo(propagate_user_sharding,
      partition, to_mesh_pspec_sharding, in_tree, out_tree,
      infer_sharding_from_operands, ctx.module_context, mesh, static_args)
  key = str(id(sharding_callback_info))
  _sharding_callbacks[bytes(key, 'utf8')] = sharding_callback_info
  # 需要保证 SPMD 分区器运行时 `sharding_callback_info` 仍然存活，
  # 因此把它附加到可执行文件上以维持其存活。
  ctx.module_context.add_keepalive(sharding_callback_info)

  result_types, _ = mlir.ir_tree_registry.flatten(
      [mlir.aval_to_ir_types(ctx.module_context, a) for a in call.out_avals])
  out = hlo.CustomCallOp(
      result_types,
      list(values),
      call_target_name=ir.StringAttr.get(_CUSTOM_PARTITIONING_CALL_NAME),
      has_side_effect=ir.BoolAttr.get(False),
      api_version=mlir.i32_attr(2),
      called_computations=ir.ArrayAttr.get([]),
      backend_config=ir.StringAttr.get(key),
      operand_layouts=None,
      result_layouts=None)
  if sharding_rule is not None:
    value_types, _ = mlir.ir_tree_registry.flatten(
        [mlir.aval_to_ir_types(ctx.module_context, a) for a in call.in_avals])
    if callable(sharding_rule):
      sharding_rule = sharding_rule(*static_args, mesh, value_types, result_types)
      if isinstance(sharding_rule, (list, tuple)) and len(sharding_rule) == 2:
        sharding_rule, sharding_rule_dict = sharding_rule
      else:
        sharding_rule_dict = {}
      if isinstance(sharding_rule, str):
        sharding_rule = str_to_sdy_sharding_rule(sharding_rule, **sharding_rule_dict)
      elif not isinstance(sharding_rule, SdyShardingRule):
          raise ValueError("sharding_rule callable must produce either an "
                           "SdyShardingRule object or an Einsum-like notation "
                           "string.")
    out.attributes['sdy.sharding_rule'] = sdy_sharding_rule_to_mlir(
      sharding_rule, value_types, result_types)
  return out.results

mlir.register_lowering(custom_partitioning_p,
                       _custom_partitioning_lowering_rule)

xc.register_custom_call_partitioner(
    _CUSTOM_PARTITIONING_CALL_NAME,
    _custom_partitioning_propagate_user_sharding,
    _custom_partitioning_partition,
    _custom_partitioning_infer_sharding_from_operands,
    can_side_effecting_have_replicated_sharding=True,
)
xb.register_plugin_callbacks(
    partial(
        xc.register_custom_call_partitioner,
        name=_CUSTOM_PARTITIONING_CALL_NAME,
        prop_user_sharding=_custom_partitioning_propagate_user_sharding,
        partition=_custom_partitioning_partition,
        infer_sharding_from_operands=_custom_partitioning_infer_sharding_from_operands,
        can_side_effecting_have_replicated_sharding=True,
    )
)
