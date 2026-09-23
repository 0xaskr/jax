# Copyright 2019 The JAX Authors.
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
# 文件职责：定义 JAX 的数据类型系统，包括扩展 dtype、类型提升格与 dtype 规范化。
# 本模块为 `jax.numpy`、`jax.lax` 等前端提供公共 dtype API：`canonicalize_dtype`
# 按 x64 配置把 dtype 归一化，`result_type` / `promote_types` 按 JAX 自己的规则
# 求类型提升的最小上界，`issubdtype` / `isdtype` 负责类型层级判断。
# 它还注册并校验自定义浮点（bfloat16、float8/float6/float4）、子字节整型
# （int1/2/4）以及 `prng_key`、`float0` 等扩展 dtype，其规则与 NumPy 并不相同。

# 数组类型相关函数。
#
# JAX 的 dtype 与 NumPy 在两方面存在差异：
# a) 类型提升规则不同，
# b) 支持的类型集合不同（例如 bfloat16），
# 因此我们需要自己的实现，并在若干地方有意偏离 NumPy。

from __future__ import annotations

import abc
from collections.abc import Callable
import dataclasses
import functools
import types
from typing import Any, Literal, cast, overload
import warnings

import ml_dtypes
import numpy as np

from jax._src import config
from jax._src import traceback_util
from jax._src.lib import _jax
from jax._src.typing import Array, DType, DTypeLike
from jax._src.util import StrictABC, set_module, cache

traceback_util.register_exclusion(__file__)

try:
  _ml_dtypes_version = tuple(map(int, ml_dtypes.__version__.split('.')[:3]))
except:
  pass
else:
  if _ml_dtypes_version < (0, 5):
    raise ValueError("JAX requires ml_dtypes version 0.5 or newer; "
                     f"installed version is {ml_dtypes.__version__}.")

export = set_module('jax.dtypes')

@export
class extended(np.generic):
  """扩展 dtype 的标量类。

  这是一个抽象类，绝不应该被实例化，它的存在只是为了支持
  `jnp.issubdtype`。

  Examples:
    >>> from jax import random
    >>> from jax import dtypes
    >>> key = random.key(0)
    >>> jnp.issubdtype(key.dtype, dtypes.extended)
    True
  """


@export
class prng_key(extended):
  """PRNG 密钥 dtype 的标量类。

  这是一个抽象类，绝不应该被实例化，它的存在只是为了支持
  `jnp.issubdtype`。

  Examples:
    >>> from jax import random
    >>> from jax import dtypes
    >>> key = random.key(0)
    >>> jnp.issubdtype(key.dtype, dtypes.prng_key)
    True
  """


class ExtendedDType(StrictABC):
  """扩展 dtype 的抽象基类"""
  @property
  @abc.abstractmethod
  def type(self) -> type: ...

  _rules: Any = None

# fp8 支持
float8_e3m4: type[np.generic] = ml_dtypes.float8_e3m4
float8_e4m3: type[np.generic] = ml_dtypes.float8_e4m3
float8_e8m0fnu: type[np.generic] = ml_dtypes.float8_e8m0fnu
float8_e4m3b11fnuz: type[np.generic] = ml_dtypes.float8_e4m3b11fnuz
float8_e4m3fn: type[np.generic] = ml_dtypes.float8_e4m3fn
float8_e4m3fnuz: type[np.generic] = ml_dtypes.float8_e4m3fnuz
float8_e5m2: type[np.generic] = ml_dtypes.float8_e5m2
float8_e5m2fnuz: type[np.generic] = ml_dtypes.float8_e5m2fnuz

_float8_e3m4_dtype: np.dtype = np.dtype(float8_e3m4)
_float8_e4m3_dtype: np.dtype = np.dtype(float8_e4m3)
_float8_e8m0fnu_dtype: np.dtype = np.dtype(float8_e8m0fnu)
_float8_e4m3b11fnuz_dtype: np.dtype = np.dtype(float8_e4m3b11fnuz)
_float8_e4m3fn_dtype: np.dtype = np.dtype(float8_e4m3fn)
_float8_e4m3fnuz_dtype: np.dtype = np.dtype(float8_e4m3fnuz)
_float8_e5m2_dtype: np.dtype = np.dtype(float8_e5m2)
_float8_e5m2fnuz_dtype: np.dtype = np.dtype(float8_e5m2fnuz)

# fp6 支持
float6_e2m3fn: type[np.generic] = ml_dtypes.float6_e2m3fn
float6_e3m2fn: type[np.generic] = ml_dtypes.float6_e3m2fn

_float6_e2m3fn_dtype: np.dtype = np.dtype(float6_e2m3fn)
_float6_e3m2fn_dtype: np.dtype = np.dtype(float6_e3m2fn)

# fp4 支持
float4_e2m1fn: type[np.generic] = ml_dtypes.float4_e2m1fn

_float4_e2m1fn_dtype: np.dtype = np.dtype(float4_e2m1fn)

def supports_inf(dtype: DTypeLike) -> bool:
  """如果该 dtype 支持无穷大则返回 True，否则返回 False。"""
  typ = np.dtype(dtype).type
  if typ in {float8_e4m3b11fnuz, float8_e4m3fn, float8_e4m3fnuz, float8_e5m2fnuz}:
    return False
  return issubdtype(dtype, np.inexact)

# bfloat16 支持
bfloat16: type[np.generic] = ml_dtypes.bfloat16
_bfloat16_dtype: np.dtype = np.dtype(bfloat16)

_custom_float_scalar_types = [
    float4_e2m1fn,
    float6_e2m3fn,
    float6_e3m2fn,
    float8_e3m4,
    float8_e4m3,
    float8_e8m0fnu,
    float8_e4m3b11fnuz,
    float8_e4m3fn,
    float8_e4m3fnuz,
    float8_e5m2,
    float8_e5m2fnuz,
    bfloat16,
]
_custom_float_dtypes = [
    _float4_e2m1fn_dtype,
    _float6_e2m3fn_dtype,
    _float6_e3m2fn_dtype,
    _float8_e3m4_dtype,
    _float8_e4m3_dtype,
    _float8_e8m0fnu_dtype,
    _float8_e4m3b11fnuz_dtype,
    _float8_e4m3fn_dtype,
    _float8_e4m3fnuz_dtype,
    _float8_e5m2_dtype,
    _float8_e5m2fnuz_dtype,
    _bfloat16_dtype,
]
_float8_dtypes = [
    _float8_e3m4_dtype,
    _float8_e4m3_dtype,
    _float8_e8m0fnu_dtype,
    _float8_e4m3b11fnuz_dtype,
    _float8_e4m3fn_dtype,
    _float8_e4m3fnuz_dtype,
    _float8_e5m2_dtype,
    _float8_e5m2fnuz_dtype,
]

_float6_dtypes: list[np.dtype] = [
    _float6_e2m3fn_dtype,
    _float6_e3m2fn_dtype,
]

_float4_dtypes: list[np.dtype] = [
    _float4_e2m1fn_dtype,
]

int1: type[np.generic] | None = None
uint1: type[np.generic] | None = None
_int1_dtype: np.dtype | None = None
_uint1_dtype: np.dtype | None = None

int2: type[np.generic] = ml_dtypes.int2
uint2: type[np.generic] = ml_dtypes.uint2

_int2_dtype: np.dtype = np.dtype(int2)
_uint2_dtype: np.dtype = np.dtype(uint2)

# 4 位整数支持
int4: type[np.generic] = ml_dtypes.int4
uint4: type[np.generic] = ml_dtypes.uint4
_int4_dtype = np.dtype(int4)
_uint4_dtype = np.dtype(uint4)

_intn_dtypes = [
    _int2_dtype,
    _uint2_dtype,
    _int4_dtype,
    _uint4_dtype,
]

if hasattr(ml_dtypes, 'int1'):
  int1 = ml_dtypes.int1
  _int1_dtype = np.dtype(int1)
  _intn_dtypes.append(_int1_dtype)

if hasattr(ml_dtypes, 'uint1'):
  uint1 = ml_dtypes.uint1
  _uint1_dtype = np.dtype(uint1)
  _intn_dtypes.append(_uint1_dtype)

# 默认类型。
bool_ = np.bool_
int_: type[Any] = np.int64
uint: type[Any] = np.uint64
float_: type[Any] = np.float64
complex_: type[Any] = np.complex128


# 默认 dtype。它们意在具有与（例如）canonicalize_dtype(np.float64)
# 相同的语义，但这样划分是为了将来减少我们执行
# dtype 规范化的调用点数量。


def default_int_dtype() -> DType:
  return np.dtype(np.int64) if config.enable_x64.value else np.dtype(np.int32)


def default_uint_dtype() -> DType:
  return np.dtype(np.uint64) if config.enable_x64.value else np.dtype(np.uint32)


def default_float_dtype() -> DType:
  return (
      np.dtype(np.float64) if config.enable_x64.value else np.dtype(np.float32)
  )


def default_complex_dtype() -> DType:
  return (
      np.dtype(np.complex128)
      if config.enable_x64.value
      else np.dtype(np.complex64)
  )


default_types: dict[str, Callable[[], DType]] = {
    'b': lambda: np.dtype(bool),
    'i': default_int_dtype,
    'u': default_uint_dtype,
    'f': default_float_dtype,
    'c': default_complex_dtype,
}

def jax_dtype(obj: DTypeLike | None, *, align: bool = False,
              copy: bool = False) -> DType:
  """把对象转换为 dtype，并遵循 JAX 的默认 dtype 规则。

  参数与 :func:`numpy.dtype` 一致。
  """
  if obj is None:
    obj = default_float_dtype()
  elif issubdtype(obj, extended):
    return obj  # pyrefly: ignore[bad-return]
  elif isinstance(obj, type) and (f := _DEFAULT_TYPEMAP.get(obj)) is not None:
    obj = f()
  return np.dtype(obj, align=align, copy=copy)

_DEFAULT_TYPEMAP: dict[type, Callable[[], np.dtype]] = {
  bool: lambda: np.dtype(bool),
  int: default_int_dtype,
  float: default_float_dtype,
  complex: default_complex_dtype,
}

def itemsize_bits(dtype: DTypeLike) -> int:
  """该 dtype 每个元素占用的位数。"""
  # 注意：这里不能使用 dtype.itemsize，
  # 因为对子字节整型来说它是不正确的。
  if dtype is None:
    raise ValueError("dtype cannot be None.")
  if dtype == np.dtype(bool):
    return 8  # 布尔 dtype 的物理位布局
  elif issubdtype(dtype, np.integer):
    return iinfo(dtype).bits
  elif issubdtype(dtype, np.floating):
    return finfo(dtype).bits
  elif issubdtype(dtype, np.complexfloating):
    return 2 * finfo(dtype).bits
  else:
    raise ValueError(f"unexpected input: {dtype=}")

# int/bool 原始值的切向量所需的平凡向量空间数据类型
float0: np.dtype = np.dtype([('float0', np.void, 0)])

_dtype_to_32bit_dtype: dict[DType, DType] = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}

# 注意：为与早期做法保持向后兼容，这里把窄类型
# 提升为 float32。我们可能会重新考虑这一点，
# 或者把该逻辑与类型提升格结合得更紧密。
_dtype_to_inexact: dict[DType, DType] = {
    np.dtype(k): np.dtype(v) for k, v in [
        ('bool', 'float32'),
        ('uint4', 'float32'), ('int4', 'float32'),
        ('uint8', 'float32'), ('int8', 'float32'),
        ('uint16', 'float32'), ('int16', 'float32'),
        ('uint32', 'float32'), ('int32', 'float32'),
        ('uint64', 'float64'), ('int64', 'float64')
    ]
}

def to_numeric_dtype(dtype: DTypeLike) -> DType:
  """若该 dtype 还不是数值 dtype，则将其提升为数值 dtype。"""
  dtype_ = np.dtype(dtype)
  return np.dtype('int32') if dtype_ == np.dtype('bool') else dtype_


def to_inexact_dtype(dtype: DTypeLike) -> DType:
  """若该 dtype 还不是非精确 dtype，则将其提升为非精确 dtype。"""
  dtype_ = np.dtype(dtype)
  return _dtype_to_inexact.get(dtype_, dtype_)


def to_floating_dtype(dtype: DTypeLike) -> DType:
  """将该 dtype 提升为非复数浮点 dtype。"""
  dtype_ = np.dtype(dtype)
  return finfo(_dtype_to_inexact.get(dtype_, dtype_)).dtype


def to_complex_dtype(dtype: DTypeLike) -> DType:
  ftype = to_inexact_dtype(dtype)
  if ftype in [np.dtype('float64'), np.dtype('complex128')]:
    return np.dtype('complex128')
  return np.dtype('complex64')


@functools.cache
def _canonicalize_dtype(x64_enabled: bool, allow_extended_dtype: bool, dtype: Any) -> DType | ExtendedDType:
  if issubdtype(dtype, extended):
    if not allow_extended_dtype:
      raise ValueError(f"Internal: canonicalize_dtype called on extended dtype {dtype} "
                       "with allow_extended_dtype=False")
    return dtype
  try:
    dtype_ = np.dtype(dtype)
  except TypeError as e:
    raise TypeError(f'dtype {dtype!r} not understood') from e

  if x64_enabled:
    return dtype_
  else:
    return _dtype_to_32bit_dtype.get(dtype_, dtype_)

@overload


def canonicalize_dtype(
    dtype: Any, allow_extended_dtype: Literal[False] = False
) -> DType:
  ...


@overload


def canonicalize_dtype(
    dtype: Any, allow_extended_dtype: bool = False
) -> DType | ExtendedDType:
  ...


@export
def canonicalize_dtype(dtype: Any, allow_extended_dtype: bool = False) -> DType | ExtendedDType:
  """根据 config.x64_enabled 把 dtype 转换为其规范形式。"""
  return _canonicalize_dtype(config.enable_x64.value, allow_extended_dtype, dtype)

class InvalidInputException(TypeError):
  pass

_jax.set_invalid_input_exception(InvalidInputException)

register_canonicalize_value_handler = _jax.register_canonicalize_value_handler
canonicalize_value = _jax.canonicalize_value

# 向后兼容垫片。
class _CanonicalizeValueHandlersDict:

  def __getitem__(self, key):
    return lambda x: canonicalize_value(np.asarray(x))

  def __setitem__(self, key, value):
    register_canonicalize_value_handler(key, value)

canonicalize_value_handlers = _CanonicalizeValueHandlersDict()


# 所有已知 Python 标量类型的列表。
python_scalar_types: set[type] = {bool, int, float, complex}

# Python 标量对应的默认 dtype。
python_scalar_types_to_dtypes: dict[type, DType] = {
  bool: np.dtype('bool'),
  int: np.dtype('int64'),
  float: np.dtype('float64'),
  complex: np.dtype('complex128'),
}

@export
def scalar_type_of(x: Any) -> type:
  """返回与 JAX 值关联的标量类型。"""
  typ = dtype(x)
  if typ in _custom_float_dtypes:
    return float
  elif typ in _intn_dtypes:
    return int
  elif np.issubdtype(typ, np.bool_):
    return bool
  elif np.issubdtype(typ, np.integer):
    return int
  elif np.issubdtype(typ, np.floating):
    return float
  elif np.issubdtype(typ, np.complexfloating):
    return complex
  else:
    raise TypeError(f"Invalid scalar value {x}")


def scalar_type_to_dtype(typ: type, value: Any = None) -> DType:
  """返回给定标量类型对应的 numpy dtype。

  Raises
  ------
  OverflowError：当 `typ` 为 `int` 且该值对 int64 而言过大时抛出。

  Examples
  --------
  >>> scalar_type_to_dtype(int)
  dtype('int32')
  >>> scalar_type_to_dtype(float)
  dtype('float32')
  >>> scalar_type_to_dtype(complex)
  dtype('complex64')
  >>> scalar_type_to_dtype(int)
  dtype('int32')
  >>> scalar_type_to_dtype(int, 0)
  dtype('int32')
  >>> scalar_type_to_dtype(int, 1 << 63)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  OverflowError: Python int 9223372036854775808 too large to convert to int32
  """
  dtype = canonicalize_dtype(python_scalar_types_to_dtypes[typ])
  if typ is int and value is not None:
    iinfo = np.iinfo(dtype)
    if value < iinfo.min or value > iinfo.max:
      raise OverflowError(f"Python int {value} too large to convert to {dtype}")
  return dtype


def coerce_to_array(x: Any, dtype: DTypeLike | None = None) -> np.ndarray:
  """把标量或 NumPy 数组强制转换为 np.array。

  按照 JAX 的规则（而不是 NumPy 的规则）处理
  Python 标量类型提升。
  """
  if dtype is None and type(x) in python_scalar_types:
    dtype = scalar_type_to_dtype(type(x), x)
  return np.asarray(x, dtype)

iinfo = ml_dtypes.iinfo
finfo = ml_dtypes.finfo

def _issubclass(a: Any, b: Any) -> bool:
  """判断 ``a`` 是否为 ``b`` 的子类。

  与 issubclass 类似，但当 `a` 不是类时返回 False，
  而不是抛出异常。
  """
  try:
    return issubclass(a, b)
  except TypeError:
    return False


_types_for_issubdtype = (type, np.dtype, ExtendedDType)

# TODO(jakevdp): 考虑是否在此禁止 None。我们允许它，
# 因为 np.issubdtype 允许（并将其视为等价于 float64）。
@set_module('jax.numpy')
def issubdtype(a: DTypeLike | ExtendedDType | None,
               b: DTypeLike | ExtendedDType | None) -> bool:
  """如果第一个参数的类型码在类型层级中更低或相等，则返回 True。

  它类似 :func:`numpy.issubdtype`，但能处理诸如
  :obj:`jax.dtypes.bfloat16` 和 `jax.dtypes.prng_key` 这类 dtype 扩展。
  """
  # 与 np.issubdtype 的主要差异在于：
  # - “扩展”dtype（如 prng key 类型）不是普通的 numpy dtype，因此
  #   我们需要专门处理它们。不过它们的标量类型确实符合
  #   numpy 标量类型层级。
  # - 自定义 dtype（如 bfloat16、int4 等）是普通的 numpy dtype，但它们
  #   不符合标准的 numpy 类型层级（例如 bfloat16 标量类型并不是 np.floating
  #   的子类），所以也必须专门处理。

  # 我们不能对所有输入都直接使用带缓存的版本，因为有些输入可能不可哈希
  # （例如带 dtype 属性的自定义对象）。下面这个检查很快，且覆盖了
  # JAX 库代码中调用本函数的大多数情况。
  return _issubdtype_cached(
      a if isinstance(a, _types_for_issubdtype) else np.dtype(a),
      b if isinstance(b, _types_for_issubdtype) else np.dtype(b),
  )


@cache(max_size=512, trace_context_in_key=False)  # 不要用 util.memoize，因为这里不依赖 X64。
def _issubdtype_cached(a: type | np.dtype | ExtendedDType,
                       b: type | np.dtype | ExtendedDType) -> bool:
  # 先处理扩展 dtype，它们需要自己的逻辑。
  a_is_type = isinstance(a, type)
  b_is_type = isinstance(b, type)
  if b_is_type and _issubclass(b, extended):
    if isinstance(a, ExtendedDType):
      return _issubclass(a.type, b)
    if a_is_type and _issubclass(a, np.generic):
      return _issubclass(a, b)
    return _issubclass(np.dtype(a).type, b)
  if isinstance(b, ExtendedDType):
    return isinstance(a, ExtendedDType) and a == b
  if isinstance(a, ExtendedDType):
    a = a.type
    a_is_type = isinstance(a, type)

  # 对于其他情况，把输入归一化为标量类型。
  a_sctype = a if a_is_type and _issubclass(a, np.generic) else np.dtype(a).type
  b_sctype = b if b_is_type and _issubclass(b, np.generic) else np.dtype(b).type

  # 现在对自定义浮点与整数类型做特殊处理，因为它们不符合
  # 常规的标量类型层级。
  if a_sctype in _custom_float_scalar_types:
    return b_sctype in {a_sctype, np.floating, np.inexact, np.number, np.generic}
  if a_sctype in [int2, int4] or (int1 is not None and a_sctype == int1):
    return b_sctype in {a_sctype, np.signedinteger, np.integer, np.number, np.generic}
  if a_sctype in [uint2, uint4] or (uint1 is not None and a_sctype == uint1):
    return b_sctype in {a_sctype, np.unsignedinteger, np.integer, np.number, np.generic}

  # 其他情况回退到 numpy.issubdtype
  return bool(np.issubdtype(a_sctype, b_sctype))

can_cast = np.can_cast

JAXType = type | DType

# 按顺序枚举所有合法的 JAX 类型。
_weak_types: list[JAXType] = [int, float, complex]
_bool_types: list[JAXType] = [np.dtype(bool)]
_signed_types: list[JAXType]
_unsigned_types: list[JAXType]
_int_types: list[JAXType]
_unsigned_types = [
    np.dtype(uint2),
    np.dtype(uint4),
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64'),
]
_signed_types = [
    np.dtype(int2),
    np.dtype(int4),
    np.dtype('int8'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64'),
]

if int1 is not None:
  _signed_types.insert(0, np.dtype(int1))
if uint1 is not None:
  _unsigned_types.insert(0, np.dtype(uint1))

_int_types = _unsigned_types + _signed_types

_float_types: list[JAXType] = [
    *_custom_float_dtypes,
    np.dtype('float16'),
    np.dtype('float32'),
    np.dtype('float64'),
]
_complex_types: list[JAXType] = [
    np.dtype('complex64'),
    np.dtype('complex128'),
]


# 我们只把 StringDType 加入 `_jax_dtype_set`，而不加入 `_jax_types` 和
# `_dtype_kinds`。这是因为，尽管这个名字听起来非常相似，
# `_jax_types` 只用于类型提升相关的逻辑，而 StringDType
# 目前并不参与类型提升。同理，`_dtype_kinds`
# 也只用于 `jnp.isdtype`，我们希望保守一些，不允许
# 在其中使用 StringDType。
string_dtype = np.dtypes.StringDType()

_jax_dtype_set = {
    float0,
    string_dtype,
    *_bool_types,
    *_int_types,
    *_float_types,
    *_complex_types,
}

_jax.set_valid_dtypes(_jax_dtype_set)

_jax_types = (_bool_types + _int_types + _float_types + _complex_types)

_dtype_kinds: dict[str, set] = {
    'bool': {*_bool_types},
    'signed integer': {*_signed_types},
    'unsigned integer': {*_unsigned_types},
    'integral': {*_signed_types, *_unsigned_types},
    'real floating': {*_float_types},
    'complex floating': {*_complex_types},
    'numeric': {*_signed_types, *_unsigned_types, *_float_types, *_complex_types},
}


@set_module('jax.numpy')
def isdtype(dtype: DTypeLike, kind: str | DTypeLike | tuple[str | DTypeLike, ...]) -> bool:
  """返回一个布尔值，表示给定 dtype 是否属于指定的类别。

  Args:
    dtype : 输入的 dtype
    kind : 数据类型类别。
      如果 ``kind`` 是 dtype 形式的，则返回 ``dtype = kind``。
      如果 ``kind`` 是字符串，则当 dtype 属于指定类别时返回 True：

      - ``'bool'``: ``{bool}``
      - ``'signed integer'``: ``{int4, int8, int16, int32, int64}``
      - ``'unsigned integer'``: ``{uint4, uint8, uint16, uint32, uint64}``
      - ``'integral'``: ``('signed integer', 'unsigned integer')`` 的简写
      - ``'real floating'``: ``{float8_*, float16, bfloat16, float32, float64}``
      - ``'complex floating'``: ``{complex64, complex128}``
      - ``'numeric'``: ``('integral', 'real floating', 'complex floating')`` 的简写

      如果 ``kind`` 是元组，则当 dtype 匹配元组中任意一项时返回 True。

  Returns:
    True 或 False
  """
  the_dtype = np.dtype(dtype)
  kind_tuple: tuple[str | DTypeLike, ...] = (
    kind if isinstance(kind, tuple) else (kind,)
  )
  options: set[DType] = set()
  for kind in kind_tuple:
    if isinstance(kind, str) and kind in _dtype_kinds:
      options.update(_dtype_kinds[kind])
      continue
    try:
      _dtype = np.dtype(kind)
    except TypeError as e:
      if isinstance(kind, str):
        raise ValueError(
          f"Unrecognized {kind=} expected one of {list(_dtype_kinds.keys())}, "
          "or a compatible input for jnp.dtype()")
      raise TypeError(
        f"Expected kind to be a dtype, string, or tuple; got {kind=}"
      ) from e
    options.add(_dtype)
  return the_dtype in options


def _jax_type(dtype: DType, weak_type: bool) -> JAXType:
  """返回给定 dtype 与弱类型标志对应的 jax 类型。"""
  if weak_type:
    if dtype == bool:
      return dtype
    if dtype in _custom_float_dtypes:
      return float
    return type(dtype.type(0).item())
  return dtype

def _dtype_and_weaktype(value: Any) -> tuple[DType, bool]:
  """返回给定输入的 (dtype, weak_type) 元组。"""
  return dtype(value), any(value is typ for typ in _weak_types) or is_weakly_typed(value)

def _type_promotion_lattice(strict: bool, x64: bool) -> dict[JAXType, list[JAXType]]:
  """
  以 DAG 的形式返回类型提升格。
  该 DAG 把每个类型映射到它在格上紧邻的更高类型。

  Args:
    strict: 是否使用严格类型提升格？
    x64: 是否允许由非 x64 输入提升出 x64 类型？
  """
  b1, = _bool_types
  u1, i1 = None, None
  if _int1_dtype is not None:
    assert _uint1_dtype is not None
    u1, u2, u4, u8, u16, u32, u64, i1, i2, i4, i8, i16, i32, i64 = _int_types
  else:
    u2, u4, u8, u16, u32, u64, i2, i4, i8, i16, i32, i64 = _int_types
  *small_float_types, bf16, f16, f32, f64 = _float_types
  c64, c128 = _complex_types
  i_, f_, c_ = _weak_types
  if not strict:
    out: dict[JAXType, list[JAXType]] = {
        b1: [i_],
        i_: [u8, u2, u4, i8, i2, i4],
        u2: [],
        u4: [],
        u8: [i16, u16],
        u16: [i32, u32],
        u32: [i64, u64],
        u64: [f_],
        i2: [],
        i4: [],
        i8: [i16],
        i16: [i32],
        i32: [i64],
        i64: [f_],
        f_: [*small_float_types, bf16, f16, c_],
        **{t: [] for t in small_float_types},
        bf16: [f32],
        f16: [f32],
        f32: [f64, c64],
        f64: [c128],
        c_: [c64],
        c64: [c128],
        c128: [],
    }
    if i1 is not None:
      out[i_].append(i1)
      out[i1] = []
    if u1 is not None:
      out[i_].append(u1)
      out[u1] = []
    # 如果未启用 x64 模式，我们希望避免任何由非 64 位输入
    # 产生 64 位类型的提升。整个提升格中只有一处这样的情况，
    # 即 u4xi4->i8，我们可以通过把它替换为 u4xi4->i4
    # 来避免。
    if not x64:
      out[u32] = [i32, u64]
    return out
  else:
    return {
      i_: [f_] + _int_types,
      f_: [c_] + _float_types,
      c_: _complex_types,
      **{t: [] for t in _jax_types}
    }

def _make_lattice_upper_bounds(strict: bool, x64: bool) -> dict[JAXType, set[JAXType]]:
  lattice = _type_promotion_lattice(strict, x64)
  upper_bounds = {node: {node} for node in lattice}
  for n in lattice:
    while True:
      new_upper_bounds = set().union(*(lattice[b] for b in upper_bounds[n]))
      if n in new_upper_bounds:
        raise ValueError(f"cycle detected in type promotion lattice for node {n}")
      if new_upper_bounds.issubset(upper_bounds[n]):
        break
      upper_bounds[n] |= new_upper_bounds
  return upper_bounds

_standard_x64_lattice_ubs = _make_lattice_upper_bounds(strict=False, x64=True)
_standard_x32_lattice_ubs = _make_lattice_upper_bounds(strict=False, x64=False)
_strict_lattice_ubs = _make_lattice_upper_bounds(strict=True, x64=True)


@export
class TypePromotionError(ValueError):
  """当 JAX 类型提升失败时抛出。"""
  pass


# 我们没有使用 util.memoize，因为这里不存在隐式的 X64 依赖。
@functools.lru_cache(512)
def _least_upper_bound(jax_numpy_dtype_promotion: config.NumpyDtypePromotion,
                       x64: bool, *nodes: JAXType) -> JAXType:
  """计算一组节点的最小上界。

  Args:
    nodes: 来自 _jax_types + _weak_types 的条目序列
  Returns:
    在提升格上表示输入节点最小上界的
      _jax_type。
  """
  # 该函数计算节点集合 N 的最小上界，其中 N 位于上面生成的
  # 格所定义的偏序集之内。
  # 给定偏序集 S，令 n ∈ S 的上界集合为
  #   UB(n) ≡ {m ∈ S | n ≤ m}
  # 进而，对于节点集合 N ⊆ S，其公共上界集合定义为
  #   CUB(N) ≡ {a ∈ S | ∀ b ∈ N: a ∈ UB(b)}
  # 那么 N 的最小上界定义为
  #   LUB(N) ≡ {c ∈ CUB(N) | ∀ d ∈ CUB(N), c ≤ d}
  # 上界的定义意味着 c ≤ d 当且仅当 d ∈ UB(c)，
  # 于是 LUB 可以表示为：
  #   LUB(N) = {c ∈ CUB(N) | ∀ d ∈ CUB(N): d ∈ UB(c)}
  # 或者等价地：
  #   LUB(N) = {c ∈ CUB(N) | CUB(N) ⊆ UB(c)}
  # 按定义，对于偏序集而言 LUB(N) 的基数为 1。
  # 注意一个可能的算法捷径：由 CUB(N) 的定义可得
  #   ∀ c ∈ N: CUB(N) ⊆ UB(c)
  # 因此若 N ∩ CUB(N) 非空，则可推出 LUB(N) = N ∩ CUB(N)。
  N = set(nodes)
  if jax_numpy_dtype_promotion == config.NumpyDtypePromotion.STRICT:
    UB = _strict_lattice_ubs
  elif jax_numpy_dtype_promotion == config.NumpyDtypePromotion.STANDARD:
    if x64:
      UB = _standard_x64_lattice_ubs
    else:
      UB = _standard_x32_lattice_ubs
  else:
    raise ValueError(
      f"Unexpected value of jax_numpy_dtype_promotion={jax_numpy_dtype_promotion!r}")
  try:
    bounds = [UB[n] for n in N]
  except KeyError:
    dtype = next(n for n in N if n not in UB)
    raise ValueError(f"{dtype=} is not a valid dtype for JAX type promotion.")
  CUB = set.intersection(*bounds)
  LUB = (CUB & N) or {c for c in CUB if CUB.issubset(UB[c])}
  if len(LUB) == 1:
    return LUB.pop()
  elif len(LUB) == 0:
    if config.numpy_dtype_promotion.value == config.NumpyDtypePromotion.STRICT:
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path when jax_numpy_dtype_promotion=strict. Try explicitly casting "
        "inputs to the desired output type, or set jax_numpy_dtype_promotion=standard.")
    elif any(n in _float8_dtypes for n in nodes):
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. To avoid unintended promotion, 8-bit floats do not support "
        "implicit promotion. If you'd like your inputs to be promoted to another type, "
        "you can do so explicitly using e.g. x.astype('float32')")
    elif any(n in _float6_dtypes for n in nodes):
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. To avoid unintended promotion, 6-bit floats do not support "
        "implicit promotion. If you'd like your inputs to be promoted to another type, "
        "you can do so explicitly using e.g. x.astype('float32')")
    elif any(n in _float4_dtypes for n in nodes):
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. To avoid unintended promotion, 4-bit floats do not support "
        "implicit promotion. If you'd like your inputs to be promoted to another type, "
        "you can do so explicitly using e.g. x.astype('float32')")
    elif any(n in _intn_dtypes for n in nodes):
      msg = (
          f'Input dtypes {tuple(str(n) for n in nodes)} have no available'
          ' implicit dtype promotion path. To avoid unintended promotion,'
          ' 1-bit, 2-bit and 4-bit integers do not support implicit promotion.'
          " If you'd like your inputs to be promoted to another type, you can"
          " do so explicitly using e.g. x.astype('int32')"
      )
    else:
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. Try explicitly casting inputs to the desired output type.")
    raise TypePromotionError(msg)
  else:
    # 执行到这里说明该格的结构有问题。
    raise TypePromotionError(
      f"Internal Type Promotion error: {nodes} do not have a unique least upper bound "
      f"on the specified lattice; options are {LUB}. This is an unexpected error in "
      "JAX's internal logic; please report it to the JAX maintainers."
    )

@set_module('jax.numpy')
def promote_types(a: DTypeLike, b: DTypeLike) -> DType:
  """返回二元运算把其参数转换成的类型。

  这是 :func:`numpy.promote_types` 的 JAX 实现。关于 JAX 类型提升语义的
  细节，参见 :ref:`type-promotion`。

  Args:
    a: 一个 :class:`numpy.dtype` 或 dtype 说明符。
    b: 一个 :class:`numpy.dtype` 或 dtype 说明符。

  Returns:
    一个 :class:`numpy.dtype` 对象。

  Examples:
    类型说明符可以是字符串、dtype 或标量类型，
    返回值始终是一个 dtype：

    >>> jnp.promote_types('int32', 'float32')  # strings
    dtype('float32')
    >>> jnp.promote_types(jnp.dtype('int32'), jnp.dtype('float32'))  # dtypes
    dtype('float32')
    >>> jnp.promote_types(jnp.int32, jnp.float32)  # scalar types
    dtype('float32')

    内置标量类型（:type:`int`、:type:`float` 或 :type:`complex`）被视为弱类型，
    它们不会改变与之对应的强类型值的位宽
    （讨论见 :ref:`type-promotion`）：

    >>> jnp.promote_types('uint8', int)
    dtype('uint8')
    >>> jnp.promote_types('float16', float)
    dtype('float16')

    这与该函数的 NumPy 版本不同：后者把内置标量类型
    视为等价于 64 位类型：

    >>> import numpy
    >>> numpy.promote_types('uint8', int)
    dtype('int64')
    >>> numpy.promote_types('float16', float)
    dtype('float64')
  """
  # 注意：这里刻意避免使用 `if a in _weak_types`，因为我们要检查的是
  # 对象同一性而非对象相等性，这是由 np.dtype.__eq__ 的行为决定的
  a_tp = cast(JAXType, a if any(a is t for t in _weak_types) else np.dtype(a))
  b_tp = cast(JAXType, b if any(b is t for t in _weak_types) else np.dtype(b))
  return np.dtype(_least_upper_bound(
      config.numpy_dtype_promotion.value, config.enable_x64.value, a_tp, b_tp))


def register_weak_scalar_type(typ: type):
  """把一个标量类型注册为弱类型。"""
  _registered_weak_types.add(typ)

_registered_weak_types: set[JAXType] = set()


def is_weakly_typed(x: Any) -> bool:
  if type(x) in _weak_types or type(x) in _registered_weak_types:
    return True
  try:
    return x.aval.weak_type
  except AttributeError:
    return False

def is_weakly_typed_scalar(x: Any) -> bool:
  try:
    return x.aval.weak_type and np.ndim(x) == 0
  except AttributeError:
    return type(x) in python_scalar_types

def check_valid_dtype(dtype: DType) -> None:
  if dtype not in _jax_dtype_set:
    raise TypeError(f"Dtype {dtype} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")

def _maybe_canonicalize_explicit_dtype(dtype: DType, fun_name: str) -> DType:
  "根据 explicit_x64_dtypes 对显式请求的 dtype 做规范化。"
  allow = config.explicit_x64_dtypes.value
  if allow == config.ExplicitX64Mode.ALLOW or config.enable_x64.value:
    return dtype
  canonical_dtype = canonicalize_dtype(dtype)
  if canonical_dtype == dtype:
    return dtype
  fun_name = f" requested in {fun_name}" if fun_name else ""
  if allow == config.ExplicitX64Mode.ERROR:
    msg = ("Explicitly requested dtype {}{} is not available. To enable more "
           "dtypes, set the jax_enable_x64 or allow_explicit_x64_dtypes "
           "configuration options."
          "See https://github.com/jax-ml/jax#current-gotchas for more.")
    msg = msg.format(dtype, fun_name, canonical_dtype.name)
    raise ValueError(msg)
  else:  # 警告
    msg = ("Explicitly requested dtype {}{} is not available, "
          "and will be truncated to dtype {}. To enable more dtypes, set the "
          "jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell "
          "environment variable. "
          "See https://github.com/jax-ml/jax#current-gotchas for more.")
    msg = msg.format(dtype, fun_name, canonical_dtype.name)
    warnings.warn(msg, stacklevel=4)
    return canonical_dtype


_types_whose_dtype_should_not_be_canonicalized: tuple[type, ...] = (
    Array,
)

def register_type_whose_dtype_should_not_be_canonicalized(typ: type):
  global _types_whose_dtype_should_not_be_canonicalized
  _types_whose_dtype_should_not_be_canonicalized += (typ,)

def dtype(x: Any) -> DType:
  """返回值或类型对应的 dtype 对象。

  Python 标量、Python 标量类型、NumPy 标量类型、NumPy dtype 以及非 JAX
  数组，它们的 dtype 都会被规范化。

  Note: 这个函数与 jax.numpy.dtype 不是同一个函数，后者只是
  numpy.dtype 的别名。"""
  # TODO(phawkins): 将来我们希望：
  # - 对 Python 标量类型和值返回默认 dtype
  # - 规范化 NumPy 数组和标量类型
  # - 原样返回 NumPy dtype，不做规范化。
  if x is None:
    raise ValueError(f"Invalid argument to dtype: {x}.")
  if isinstance(x, type):
    # Python 标量类型，例如 int、float
    if (dt := python_scalar_types_to_dtypes.get(x)) is not None:
      return canonicalize_dtype(dt)

    # NumPy 标量类型，例如 np.int32、np.float32
    if _issubclass(x, np.generic):
      dt = np.dtype(x)
      return _maybe_canonicalize_explicit_dtype(dt, "dtype")

  # Python 标量值，例如 int(3)、float(3.14)
  elif (dt := python_scalar_types_to_dtypes.get(type(x))) is not None:
    return canonicalize_dtype(dt)
  # JAX 数组、字面量数组和标量。
  # 我们有意不对这些类型做规范化：一旦构造出 x64 值，
  # 无论 x64 模式如何，我们都会尊重它。
  elif isinstance(x, _types_whose_dtype_should_not_be_canonicalized):
    return x.dtype

  if isinstance(x, (str, np.dtype)):
    dt = np.dtype(x)
    if dt not in _jax_dtype_set and not issubdtype(dt, extended):
      raise TypeError(f"Value '{x}' with dtype {dt} is not a valid JAX array "
                      "type. Only arrays of numeric types are supported by JAX.")
    return _maybe_canonicalize_explicit_dtype(dt, "dtype")

  # 如果 x 带有 dtype 属性，且它是合法 dtype，就直接使用它。这样可以避免
  # 对可能带有 .dtype 但并非标准 NumPy 数组类对象的对象调用 np.result_type，
  # 否则在 NumPy 2.4+ 中可能产生警告。
  dt_attr = getattr(x, 'dtype', None)
  if issubdtype(dt_attr, extended) or isinstance(dt_attr, np.dtype):
    dt = dt_attr
  else:
    try:
      dt = np.result_type(x)
    except TypeError as err:
      raise TypeError(f"Cannot determine dtype of {x}") from err
  if dt not in _jax_dtype_set and not issubdtype(dt, extended):
    raise TypeError(f"Value '{x}' with dtype {dt} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")
  # TODO(jakevdp): 修正返回类型标注并移除这个 ignore。
  return canonicalize_dtype(dt, allow_extended_dtype=True)  # pyrefly: ignore[bad-return]

def lattice_result_type(*args: Any) -> tuple[DType, bool]:
  dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  if len(dtypes) == 1:
    out_dtype = dtypes[0]
    out_weak_type = weak_types[0]
  elif len(set(dtypes)) == 1 and not all(weak_types):
    # 平凡的提升情形。这样可以允许扩展 dtype 通过。
    out_dtype = dtypes[0]
    out_weak_type = False
  elif all(weak_types) and config.numpy_dtype_promotion.value != config.NumpyDtypePromotion.STRICT:
    # 如果所有输入都是弱类型，我们先计算其强类型对应物的上界，
    # 最后再施加弱类型。这样可以避免因非规范弱类型
    # （例如弱 int16）而返回错误结果。
    # TODO(jakevdp): 探索移除这个特殊情形。
    result_type = _least_upper_bound(
        config.numpy_dtype_promotion.value, config.enable_x64.value,
        *{_jax_type(dtype, False) for dtype in dtypes})
    out_dtype = dtype(result_type)
    out_weak_type = True
  else:
    result_type = _least_upper_bound(
        config.numpy_dtype_promotion.value, config.enable_x64.value,
        *{_jax_type(d, w) for d, w in zip(dtypes, weak_types)})
    out_dtype = dtype(result_type)
    out_weak_type = any(result_type is t for t in _weak_types)
  return out_dtype, (out_dtype != bool_) and out_weak_type

@overload
def result_type(*args: Any, return_weak_type_flag: Literal[True]) -> tuple[DType, bool]: ...

@overload
def result_type(*args: Any, return_weak_type_flag: Literal[False] = False) -> DType: ...

@overload
def result_type(*args: Any, return_weak_type_flag: bool = False) -> DType | tuple[DType, bool]: ...

@export
def result_type(*args: Any, return_weak_type_flag: bool = False) -> DType | tuple[DType, bool]:
  """应用 JAX 参数 dtype 提升的便捷函数。

  Args:
    return_weak_type_flag : 若为 True，则返回 ``(dtype, weak_type)`` 元组。
      若为 False，则只返回 `dtype`

  Returns:
    取决于 ``return_weak_type`` 参数的值，返回 dtype 或 (dtype, weak_type)。
  """
  if len(args) == 0:
    raise ValueError("at least one array or dtype is required")
  dtype: DType | ExtendedDType
  dtype, weak_type = lattice_result_type(*(default_float_dtype() if arg is None else arg for arg in args))
  if weak_type:
    dtype = default_types['f' if dtype in _custom_float_dtypes else dtype.kind]()
  return (dtype, weak_type) if return_weak_type_flag else dtype

def check_and_canonicalize_user_dtype(
    dtype, fun_name=None, *, allow_non_jax_dtypes: bool = False
) -> DType:
  """检查用户提供的 dtype 是否合法，并返回其规范形式。

  对于 Python 标量类型，该函数返回相应的默认 dtype。
  """
  if dtype is None:
    raise ValueError("dtype must be specified.")
  if isinstance(dtype, Array):
    raise ValueError("Passing an array as a dtype argument is no longer "
                     "supported; instead of dtype=arr use dtype=arr.dtype.")
  if issubdtype(dtype, extended):
    return dtype
  # 避免使用 `dtype in [...]`，因为 numpy dtype 重载了相等比较。
  if isinstance(dtype, type) and (f := _DEFAULT_TYPEMAP.get(dtype)) is not None:
    return f()
  np_dtype = np.dtype(dtype)
  if np_dtype not in _jax_dtype_set:
    if allow_non_jax_dtypes:
      return np_dtype
    msg = (
        f'JAX only supports number, bool, and string dtypes, got dtype {dtype}'
    )
    msg += f" in {fun_name}" if fun_name else ""
    raise TypeError(msg)
  return _maybe_canonicalize_explicit_dtype(np_dtype, fun_name or "")

def safe_to_cast(input_dtype_or_value: Any,
                 output_dtype_or_value: Any) -> bool:
  """检查某个 dtype/值是否可以安全地转换到另一个 dtype/值

  Args:
    input_dtype_or_value: 表示源 dtype 的 dtype 或值
      （会被传给 result_type）。
    output_dtype_or_value: 表示目标 dtype 的 dtype 或值
      （会被传给 result_type）。

  Returns:
    布尔值，表示按默认类型提升语义
    这些值是否可以安全转换。

  Raises:
    TypePromotionError: 当输入类型不同、且在当前的 jax_numpy_dtype_promotion
    设置下不存在类型提升路径时抛出。

  Examples:

    >>> safe_to_cast('int16', 'float32')
    True
    >>> safe_to_cast('float32', 'int16')
    False
    >>> safe_to_cast('float32', 'complex64')
    True
    >>> safe_to_cast('complex64', 'float32')
    False
  """
  input_dtype = dtype(input_dtype_or_value)
  output_dtype = dtype(output_dtype_or_value)
  if input_dtype == output_dtype:
    return True
  # 这里我们刻意使用 output_dtype 而不是 output_dtype_or_value：
  # 这相当于把输出 dtype 始终视为强类型。
  return result_type(input_dtype_or_value, output_dtype) == output_dtype

class primal_tangent_dtype_scalar(extended): ...

@dataclasses.dataclass(frozen=True, slots=True)
class PrimalTangentDType(ExtendedDType):
  primal_dtype: Any
  tangent_dtype: Any
  name: str
  type = primal_tangent_dtype_scalar
  def __repr__(self): return self.name
  @property
  def _rules(self):  # pyrefly: ignore[bad-override]
    return types.SimpleNamespace(
      physical_element_aval=
      lambda dtype: types.SimpleNamespace(shape=(), dtype=self.primal_dtype),
      tangent_dtype=lambda dtype: self.tangent_dtype,
      allow_conversion=True)

def primal_tangent_dtype(primal_dtype, tangent_dtype,
                         name: str | None = None) -> ExtendedDType:
  primal_dtype, tangent_dtype = map(dtype, (primal_dtype, tangent_dtype))
  name_ = name or (f'PrimalTangentDType{{{short_dtype_name(primal_dtype)}'
                   f'/{short_dtype_name(tangent_dtype)}}}')
  return PrimalTangentDType(primal_dtype, tangent_dtype, name_)

@functools.cache
def short_dtype_name(dtype) -> str:
  if isinstance(dtype, ExtendedDType):
    return str(dtype)
  else:
    return (dtype.name.replace('float', 'f').replace('uint'   , 'u')
                      .replace('int'  , 'i').replace('complex', 'c'))
