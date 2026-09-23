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

# 文件职责：定义在追踪期携带 JAX 类型信息的标量与主机侧数组字面量类型。
# TypedInt、TypedFloat、TypedComplex 让 Python 内建标量可以携带 JAX 数据
# 类型；TypedNdArray 则是带有 aval 与弱类型标记的 np.ndarray 子类。这些类型
# 在规范化过程中不会被改写，从而在不同 jax_enable_x64 模式下保持 dtype 稳定。

from jax._src import dtypes
from jax._src.core import ShapedArray
from jax._src.lib import _jax
import numpy as np

# TypedInt、TypedFloat 和 TypedComplex 是 int、float 和 complex 的子类，
# 它们携带 JAX 数据类型。规范化会从 int、float 和 complex 构造出这些
# 类型。重复规范化（包括在不同 jax_enable_x64 模式下进行）会保留
# 原有的数据类型。

# 预先计算好的弱标量 aval
_weak_int32_aval = ShapedArray((), np.dtype(np.int32), weak_type=True)
_weak_int64_aval = ShapedArray((), np.dtype(np.int64), weak_type=True)
_weak_float32_aval = ShapedArray((), np.dtype(np.float32), weak_type=True)
_weak_float64_aval = ShapedArray((), np.dtype(np.float64), weak_type=True)
_weak_complex64_aval = ShapedArray((), np.dtype(np.complex64), weak_type=True)
_weak_complex128_aval = ShapedArray((), np.dtype(np.complex128), weak_type=True)

_int32_dtype = np.dtype(np.int32)
_int64_dtype = np.dtype(np.int64)
_float32_dtype = np.dtype(np.float32)
_float64_dtype = np.dtype(np.float64)
_complex64_dtype = np.dtype(np.complex64)
_complex128_dtype = np.dtype(np.complex128)

class TypedInt(int):
  dtype: np.dtype
  aval: ShapedArray

  def __new__(cls, value: int, dtype: np.dtype):
    v = super().__new__(cls, value)
    v.dtype = dtype
    if dtype == _int32_dtype:
      v.aval = _weak_int32_aval
    elif dtype == _int64_dtype:
      v.aval = _weak_int64_aval
    else:
      v.aval = ShapedArray((), dtype, weak_type=True)
    return v

  def __repr__(self):
    return f'TypedInt({int(self)}, dtype={self.dtype.name})'

  def __getnewargs__(self):
    return (int(self), self.dtype)


class TypedFloat(float):
  __slots__ = ('dtype', 'aval')

  dtype: np.dtype
  aval: ShapedArray

  def __new__(cls, value: float, dtype: np.dtype):
    v = super().__new__(cls, value)
    v.dtype = dtype
    if dtype == _float32_dtype:
      v.aval = _weak_float32_aval
    elif dtype == _float64_dtype:
      v.aval = _weak_float64_aval
    else:
      v.aval = ShapedArray((), dtype, weak_type=True)
    return v

  def __repr__(self):
    return f'TypedFloat({float(self)}, dtype={self.dtype.name})'

  def __str__(self):
    return str(float(self))

  def __getnewargs__(self):
    return (float(self), self.dtype)


class TypedComplex(complex):
  __slots__ = ('dtype', 'aval')

  dtype: np.dtype
  aval: ShapedArray

  def __new__(cls, value: complex, dtype: np.dtype):
    v = super().__new__(cls, value)
    v.dtype = dtype
    if dtype == _complex64_dtype:
      v.aval = _weak_complex64_aval
    elif dtype == _complex128_dtype:
      v.aval = _weak_complex128_aval
    else:
      v.aval = ShapedArray((), dtype, weak_type=True)
    return v

  def __repr__(self):
    return f'TypedComplex({complex(self)}, dtype={self.dtype.name})'

  def __getnewargs__(self):
    return (complex(self), self.dtype)


typed_scalar_types: set[type] = {TypedInt, TypedFloat, TypedComplex}


class TypedNdArray(np.ndarray):
  """TypedNdArray 是 JAX 在追踪期间使用的主机侧数组。

  TypedNdArray 是 np.ndarray 的子类，携带额外的 JAX 类型信息：
  * 无论 jax_enable_x64 模式如何，它的类型都不会被 JAX 规范化
  * 它可以是弱类型的。
  """
  __slots__ = ('_aval', '_weak_type')

  def __new__(cls, val: np.ndarray, aval: ShapedArray | None = None):
    obj = np.asarray(val).view(cls)
    if aval is not None:
      obj._aval = aval
    return obj

  def __array_finalize__(self, obj):
    self._aval = None
    self._weak_type = (obj.aval.weak_type
                       if isinstance(obj, TypedNdArray) else False)

  @property
  def aval(self) -> ShapedArray:
    result = self._aval
    if result is None:
      # 可能有多个线程竞争到达这里。不过这似乎是安全的，
      # 因为它们都会设置相同的值。
      result = ShapedArray(self.shape, self.dtype, weak_type=self._weak_type)
      self._aval = result
    return result

  @property
  def weak_type(self) -> bool:
    return self.aval.weak_type

  @property
  def val(self) -> np.ndarray:
    return np.asarray(self)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    inputs = tuple(
        np.asarray(x) if isinstance(x, TypedNdArray) else x for x in inputs
    )
    if 'out' in kwargs:
      kwargs['out'] = tuple(
          np.asarray(x) if isinstance(x, TypedNdArray) else x
          for x in kwargs['out']
      )
    return getattr(ufunc, method)(*inputs, **kwargs)

  def __repr__(self):
    prefix = 'TypedNdArray('
    if self.aval.weak_type:
      dtype_str = f'dtype={self.dtype.name}, weak_type=True)'
    else:
      dtype_str = f'dtype={self.dtype.name})'

    line_width = np.get_printoptions()['linewidth']
    if self.size == 0:
      s = f'[], shape={self.shape}'
    else:
      s = np.array2string(
          np.asarray(self),
          prefix=prefix,
          suffix=',',
          separator=', ',
          max_line_width=line_width,
      )
    last_line_len = len(s) - s.rfind('\n') + 1
    sep = ' '
    if last_line_len + len(dtype_str) + 1 > line_width:
      sep = ' ' * len(prefix)
    return f'{prefix}{s},{sep}{dtype_str}'

  def __reduce__(self):
    return (TypedNdArray, (np.asarray(self), self.aval.weak_type))

  def __getnewargs__(self):
    return (np.asarray(self), self.aval.weak_type)


_jax.set_typed_ndarray_type(TypedNdArray)
dtypes.register_type_whose_dtype_should_not_be_canonicalized(TypedNdArray)

_jax.set_typed_int_type(TypedInt)
_jax.set_typed_float_type(TypedFloat)
_jax.set_typed_complex_type(TypedComplex)

for _typ in typed_scalar_types:
  dtypes.register_weak_scalar_type(_typ)
  dtypes.register_type_whose_dtype_should_not_be_canonicalized(_typ)
