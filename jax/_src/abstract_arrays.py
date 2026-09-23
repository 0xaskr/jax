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

# 文件职责：注册 Python / NumPy 各种类型到 JAX 抽象值（aval）的映射，并转发
# `core` 中的抽象数组设施（`ShapedArray`、`AbstractToken`、`canonicalize_shape`）。
# 这里为 numpy 标量、numpy 数组、Python 内建标量、typed scalar 等注册
# `core.pytype_aval_mappings`，同时收集可字面化的类型集合供后续追踪与常量化使用。
# 它还让 numpy 掩码数组（masked array）在作为 JAX 函数输入时直接抛出错误，
# 从而在追踪入口处给出清晰的失败信息。

from __future__ import annotations

import numpy as np

from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import literals
from jax._src import traceback_util
traceback_util.register_exclusion(__file__)

ShapedArray = core.ShapedArray
AbstractToken = core.AbstractToken
abstract_token = core.abstract_token
canonicalize_shape = core.canonicalize_shape

numpy_scalar_types: set[type] = {
    dtypes.int4, np.int8, np.int16, np.int32, np.int64,
    dtypes.uint4, np.uint8, np.uint16, np.uint32, np.uint64,
    np.complex64, np.complex128,
    np.bool_, np.longlong, np.intc,
} | {np.dtype(dt).type for dt in dtypes._float_types}

if dtypes.int2 is not None:
  assert dtypes.uint2 is not None
  numpy_scalar_types.add(dtypes.int2)
  numpy_scalar_types.add(dtypes.uint2)

if dtypes.int1 is not None:
  assert dtypes.uint1 is not None
  numpy_scalar_types.add(dtypes.int1)
  numpy_scalar_types.add(dtypes.uint1)

core.literalable_scalar_types.update(numpy_scalar_types)

array_types: set[type] = {literals.TypedNdArray, np.ndarray} | numpy_scalar_types


def masked_array_error(*args, **kwargs):
  raise ValueError(
      "numpy masked arrays are not supported as direct inputs to JAX functions."
      " Use arr.filled() to convert the value to a standard numpy array.")

core.pytype_aval_mappings[np.ma.MaskedArray] = masked_array_error


def _make_shaped_array_for_numpy_array(x: np.ndarray) -> ShapedArray:
  dtype = x.dtype
  dtypes.check_valid_dtype(dtype)
  return ShapedArray(x.shape, dtypes.canonicalize_dtype(dtype), sharding=None)

core.pytype_aval_mappings[np.ndarray] = _make_shaped_array_for_numpy_array
core.pytype_aval_mappings[literals.TypedNdArray] = lambda x: x.aval


def _make_shaped_array_for_numpy_scalar(x: np.generic) -> ShapedArray:
  dtype = np.dtype(x)
  dtypes.check_valid_dtype(dtype)
  shape = np.shape(x)
  return ShapedArray(shape, dtypes.canonicalize_dtype(dtype), sharding=None)

for t in numpy_scalar_types:
  core.pytype_aval_mappings[t] = _make_shaped_array_for_numpy_scalar

core.literalable_types.update(array_types)

_int32_min = np.iinfo(np.int32).min
_int32_max = np.iinfo(np.int32).max
_int64_min = np.iinfo(np.int64).min
_int64_max = np.iinfo(np.int64).max

# 注意：除 `bool` 之外，所有 Python 标量类型都是弱类型，
# 因为 `bool` 只有一种宽度。
_bool_aval = ShapedArray((), dtype=np.dtype(bool))
_int32_aval = ShapedArray((), dtype=np.dtype(np.int32), weak_type=True)
_int64_aval = ShapedArray((), dtype=np.dtype(np.int64), weak_type=True)
_float32_aval = ShapedArray((), dtype=np.dtype(np.float32), weak_type=True)
_float64_aval = ShapedArray((), dtype=np.dtype(np.float64), weak_type=True)
_complex64_aval = ShapedArray((), dtype=np.dtype(np.complex64), weak_type=True)
_complex128_aval = ShapedArray((), dtype=np.dtype(np.complex128), weak_type=True)

core.pytype_aval_mappings[bool] = lambda v: _bool_aval

def _int_aval(value):
  if config.enable_x64.value:
    if value < _int64_min or value > _int64_max:
      raise OverflowError(f"Python int {value} too large to convert to int64")
    return _int64_aval
  else:
    if value < _int32_min or value > _int32_max:
      raise OverflowError(f"Python int {value} too large to convert to int32")
    return _int32_aval
core.pytype_aval_mappings[int] = _int_aval

_float_aval = lambda v: _float64_aval if config.enable_x64.value else _float32_aval
core.pytype_aval_mappings[float] = _float_aval

_complex_aval = lambda v: _complex128_aval if config.enable_x64.value else _complex64_aval
core.pytype_aval_mappings[complex] = _complex_aval

core.literalable_scalar_types.update(dtypes.python_scalar_types)
core.literalable_types.update(dtypes.python_scalar_types)


for t in literals.typed_scalar_types:
  core.pytype_aval_mappings[t] = lambda x: x.aval
core.literalable_scalar_types.update(literals.typed_scalar_types)
core.literalable_types.update(literals.typed_scalar_types)

def _canonicalize_masked_array_dtype(x):
  raise ValueError("numpy masked arrays are not supported as direct inputs to JAX functions. "
                   "Use arr.filled() to convert the value to a standard numpy array.")

dtypes.register_canonicalize_value_handler(
    np.ma.MaskedArray, _canonicalize_masked_array_dtype
)
