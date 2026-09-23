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
# 文件职责：定义 `jax.errors` 对外暴露的全部 JAX 专有异常类型，并为追踪器等场景给出可读的诊断信息。
# 这些异常覆盖 JAX 变换（`jit`、`vmap` 等）中常见的抽象值误用：抽象追踪器被当作具体值使用、
# 非具体的布尔索引、数组/整数/布尔转换失败、`Tracer` 泄漏到变换之外，以及 PRNG key 的不安全复用。
# 每个错误类都继承 `_JAXErrorMixin`，在其报错信息末尾附加指向官方 errors 文档页面中
# 对应条目的链接，用户可据类名直接跳转查阅相应的成因与修复建议。
# 该模块是 `jax.errors` 公共错误 API 的实现本体，经 `set_module` 把类导出为 `jax.errors` 成员。
from __future__ import annotations

from jax._src import core
from jax._src.util import set_module

export = set_module('jax.errors')


class _JAXErrorMixin:
  """JAX 专有错误的混入类"""
  _error_page = 'https://docs.jax.dev/en/latest/errors.html'
  _module_name = "jax.errors"

  def __init__(self, message: str):
    error_page = self._error_page
    module_name = self._module_name
    class_name = self.__class__.__name__
    error_msg = f'{message}\nSee {error_page}#{module_name}.{class_name}'
    super().__init__(error_msg)  # pyrefly: ignore[bad-argument-count]


@export
class JAXTypeError(_JAXErrorMixin, TypeError):
  """JAX 专有的 :class:`TypeError`"""


@export
class JAXIndexError(_JAXErrorMixin, IndexError):
  """JAX 专有的 :class:`IndexError`"""


@export
class ConcretizationTypeError(JAXTypeError):
  """
  当 JAX 追踪器对象被用在需要具体值的上下文中时，
  就会抛出这个错误，关于追踪器具体是什么，
  参见 :ref:`faq-different-kinds-of-jax-values`。
  在某些情形下，只需把有问题的值标记为静态即可轻松修复；
  在另一些情形下，它可能表明你的程序正在执行的操作
  并不被 JAX 的 JIT 编译模型直接支持。

  Examples:

  在期望静态值的地方使用了被追踪的值
    这个错误的一个常见原因，是在需要静态值的地方使用了被追踪的值。
    例如：

      >>> from functools import partial
      >>> from jax import jit
      >>> import jax.numpy as jnp
      >>> @jit
      ... def func(x, axis):
      ...   return x.min(axis)

      >>> func(jnp.arange(4), 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      ConcretizationTypeError: Abstract tracer value encountered where concrete
      value is expected: axis argument to jnp.min().

    通常可以通过把有问题的参数标记为静态来修复::

        >>> @jit(static_argnums=1)
        ... def func(x, axis):
        ...   return x.min(axis)

        >>> func(jnp.arange(4), 0)
        Array(0, dtype=int32)

  形状依赖于被追踪的值
    当 JIT 编译的计算中某个形状依赖于被追踪量的取值时，
    也可能出现这类错误。例如::

      >>> @jit
      ... def func(x):
      ...     return jnp.where(x < 0)

      >>> func(jnp.arange(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected:
      The error arose in jnp.nonzero.

    这是一个与 JAX 的 JIT 编译模型不兼容的操作示例，
    该模型要求数组大小在编译期已知。
    这里返回数组的大小取决于 `x` 的内容，
    因此这类代码无法被 JIT 编译。

    很多情况下，可以通过修改函数中使用的逻辑来绕过这一问题；
    例如下面这段代码也有类似的毛病::

      >>> @jit
      ... def func(x):
      ...     indices = jnp.where(x > 1)
      ...     return x[indices].sum()

      >>> func(jnp.arange(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      ConcretizationTypeError: Abstract tracer value encountered where concrete
      value is expected: The error arose in jnp.nonzero.

    下面则展示了如何用避免创建动态大小索引数组的方式
    来表达同样的操作::

      >>> @jit
      ... def func(x):
      ...   return jnp.where(x > 1, x, 0).sum()

      >>> func(jnp.arange(4))
      Array(5, dtype=int32)

  如果你想深入了解追踪器与普通值、具体值与抽象值
  之间还有哪些微妙之处，可以阅读
  :ref:`faq-different-kinds-of-jax-values`。
  """
  def __init__(self, tracer: core.Tracer, context: str = ""):
    super().__init__(
        "Abstract tracer value encountered where concrete value is expected: "
        f"{tracer._error_repr()}\n{context}{tracer._origin_msg()}\n")


@export
class NonConcreteBooleanIndexError(JAXIndexError):
  """
  当程序在追踪的索引操作中使用非具体的布尔索引时，
  就会抛出这个错误。在 JIT 编译下，
  JAX 数组必须具有静态形状（即在编译期已知的形状），
  因此使用布尔掩码时必须格外小心。
  某些通过布尔掩码实现的逻辑在 :func:`jax.jit` 函数中根本无法完成；
  而在另一些情况下，这些逻辑可以用 JIT 兼容的方式重新表达，
  通常要借助 :func:`~jax.numpy.where` 的三参数版本。

  下面是可能触发这个错误的几个示例。

  通过布尔掩码构造数组
    最常见的情形是在 JIT 上下文中尝试用布尔掩码创建数组。
    例如::

      >>> import jax
      >>> import jax.numpy as jnp

      >>> @jax.jit
      ... def positive_values(x):
      ...   return x[x > 0]

      >>> positive_values(jnp.arange(-5, 5))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      NonConcreteBooleanIndexError: Array boolean indices must be concrete: ShapedArray(bool[10])

    这个函数试图只返回输入数组中的正值；
    除非把 `x` 标记为静态，否则返回数组的大小
    无法在编译期确定，因此像这样的操作无法
    在 JIT 编译下执行。

  可重新表达的布尔逻辑
    尽管 JAX 并不直接支持创建动态大小的数组，
    但在许多情况下，可以把计算的逻辑重新表达
    为 JIT 兼容的操作。例如，下面这个函数
    也因为同样的原因在 JIT 下失败::

      >>> @jax.jit
      ... def sum_of_positive(x):
      ...   return x[x > 0].sum()

      >>> sum_of_positive(jnp.arange(-5, 5))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      NonConcreteBooleanIndexError: Array boolean indices must be concrete: ShapedArray(bool[10])

    不过在这个例子中，有问题的数组只是一个中间值，
    我们可以改用 JIT 兼容的
    :func:`jax.numpy.where` 三参数版本来表达同样的逻辑::

      >>> @jax.jit
      ... def sum_of_positive(x):
      ...   return jnp.where(x > 0, x, 0).sum()

      >>> sum_of_positive(jnp.arange(-5, 5))
      Array(10, dtype=int32)

    用三参数 :func:`~jax.numpy.where` 取代布尔掩码，
    是解决这类问题的常见做法。

  用布尔索引访问 JAX 数组
    另一个经常出现该错误的情形是使用布尔索引，
    例如 :code:`.at[...].set(...)`。下面是一个简单的例子::

      >>> @jax.jit
      ... def manual_clip(x):
      ...   return x.at[x < 0].set(0)

      >>> manual_clip(jnp.arange(-2, 2))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      NonConcreteBooleanIndexError: Array boolean indices must be concrete: ShapedArray(bool[4])

    这个函数试图把小于零的值设为某个标量填充值。
    与上面一样，可以把该逻辑重新表达为
    :func:`~jax.numpy.where` 的形式来解决::

      >>> @jax.jit
      ... def manual_clip(x):
      ...   return jnp.where(x < 0, 0, x)

      >>> manual_clip(jnp.arange(-2, 2))
      Array([0, 0, 0, 1], dtype=int32)
  """
  def __init__(self, tracer: core.Tracer):
    super().__init__(
        f"Array boolean indices must be concrete; got {tracer}\n")


@export
class TracerArrayConversionError(JAXTypeError):
  """
  当程序试图把 JAX 追踪器对象转换为标准 NumPy 数组时，就会抛出这个错误
  （关于追踪器是什么，参见 :ref:`faq-different-kinds-of-jax-values`）。
  它通常出现在以下几种情形之中。

  在 JAX 变换中使用非 JAX 函数
    如果你在 JAX 变换（:func:`~jax.jit`、:func:`~jax.grad`、
    :func:`jax.vmap` 等）内部使用 ``numpy`` 或 ``scipy`` 这类
    非 JAX 库，就可能出现这个错误。例如::

      >>> from jax import jit
      >>> import numpy as np

      >>> @jit
      ... def func(x):
      ...   return np.sin(x)

      >>> func(np.arange(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerArrayConversionError: The numpy.ndarray conversion method
      __array__() was called on traced array with shape int32[4]

    在这个例子中，可以用 :func:`jax.numpy.sin` 代替
    :func:`numpy.sin` 来修复问题::

      >>> import jax.numpy as jnp
      >>> @jit
      ... def func(x):
      ...   return jnp.sin(x)

      >>> func(jnp.arange(4))
      Array([0.        , 0.84147096, 0.9092974 , 0.14112   ], dtype=float32)

    关于从变换后的 JAX 代码回调主机侧计算的方案，
    另见 `External Callbacks`_。

  用追踪器索引 numpy 数组
    如果这个错误出现在涉及数组索引的代码行上，
    可能是被索引的数组 ``x`` 是标准 numpy.ndarray，
    而索引 ``idx`` 是追踪的 JAX 数组。例如::

      >>> x = np.arange(10)

      >>> @jit
      ... def func(i):
      ...   return x[i]

      >>> func(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerArrayConversionError: The numpy.ndarray conversion method
      __array__() was called on traced array with shape int32[0]

    视具体上下文而定，你可以把该 numpy 数组
    转换成 JAX 数组来修复::

      >>> @jit
      ... def func(i):
      ...   return jnp.asarray(x)[i]

      >>> func(0)
      Array(0, dtype=int32)

    或者把索引声明为静态参数::

      >>> from functools import partial
      >>> @jit(static_argnums=(0,))
      ... def func(i):
      ...   return x[i]

      >>> func(0)
      Array(0, dtype=int32)

  如果你想深入了解追踪器与普通值、具体值与抽象值
  之间还有哪些微妙之处，可以阅读
  :ref:`faq-different-kinds-of-jax-values`。

  .. _External Callbacks: https://docs.jax.dev/en/latest/notebooks/external_callbacks.html
  """
  def __init__(self, tracer: core.Tracer):
    super().__init__(
        "The numpy.ndarray conversion method __array__() was called on "
        f"{tracer._error_repr()}{tracer._origin_msg()}")


@export
class TracerIntegerConversionError(JAXTypeError):
  """
  当 JAX 追踪器对象被用在期望 Python 整数的上下文中时，就可能出现这个错误
  （关于追踪器是什么，参见 :ref:`faq-different-kinds-of-jax-values`）。
  它通常出现在以下几种情形之中。

  用追踪器代替整数传入
    如果你试图把被追踪的值传给需要静态整数参数的函数，
    就可能出现这个错误；例如::

      >>> from jax import jit
      >>> import numpy as np

      >>> @jit
      ... def func(x, axis):
      ...   return np.split(x, 2, axis)

      >>> func(np.arange(4), 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerIntegerConversionError: The __index__() method was called on
      traced array with shape int32[0]

    出现这种情况时，通常的解决办法是把有问题的参数
    标记为静态::

      >>> from functools import partial
      >>> @jit(static_argnums=1)
      ... def func(x, axis):
      ...   return np.split(x, 2, axis)

      >>> func(np.arange(10), 0)
      [Array([0, 1, 2, 3, 4], dtype=int32),
       Array([5, 6, 7, 8, 9], dtype=int32)]

    另一种做法是把变换应用到一个封装了待保护参数的闭包上，
    既可以像下面这样手工完成，
    也可以借助 :func:`functools.partial`::

      >>> jit(lambda arr: np.split(arr, 2, 0))(np.arange(4))
      [Array([0, 1], dtype=int32), Array([2, 3], dtype=int32)]

    **注意：每次调用都会创建一个新的闭包，这会破坏编译缓存机制，
    因此更推荐使用 static_argnums。**

  用追踪器索引列表
    如果你试图用被追踪的量去索引 Python 列表，
    就可能出现这个错误。
    例如::

      >>> import jax.numpy as jnp
      >>> from jax import jit

      >>> L = [1, 2, 3]

      >>> @jit
      ... def func(i):
      ...   return L[i]

      >>> func(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerIntegerConversionError: The __index__() method was called on
      traced array with shape int32[0]

    视具体上下文而定，通常可以把该列表
    转换成 JAX 数组来修复::

      >>> @jit
      ... def func(i):
      ...   return jnp.array(L)[i]

      >>> func(0)
      Array(1, dtype=int32)

    或者把索引声明为静态参数::

      >>> from functools import partial
      >>> @jit(static_argnums=0)
      ... def func(i):
      ...   return L[i]

      >>> func(0)
      Array(1, dtype=int32, weak_type=True)

  如果你想深入了解追踪器与普通值、具体值与抽象值
  之间还有哪些微妙之处，可以阅读
  :ref:`faq-different-kinds-of-jax-values`。
  """
  def __init__(self, tracer: core.Tracer):
    super().__init__(
        f"The __index__() method was called on {tracer._error_repr()}"
        f"{tracer._origin_msg()}")


@export
class TracerBoolConversionError(ConcretizationTypeError):
  """
  当 JAX 中被追踪的值被用在期望布尔值的上下文中时，就会抛出这个错误
  （关于追踪器是什么，参见
  :ref:`faq-different-kinds-of-jax-values`）。

  这种布尔转换可能是显式的（例如 ``bool(x)``），也可能是隐式的：
  来自控制流（例如 ``if x > 0`` 或 ``while x``）、
  Python 布尔运算符（例如 ``z = x and y``、``z = x or y``、``z = not x``），
  或使用了这些运算符的函数（例如 ``z = max(x, y)``、``z = min(x, y)`` 等）。

  在某些情况下，把被追踪的值标记为静态即可轻松解决这个问题；
  在另一些情况下，它可能表明你的程序正在执行的操作
  并不被 JAX 的 JIT 编译模型直接支持。

  Examples:

  在控制流中使用被追踪的值
    常见的一种情形是在 Python 控制流中
    使用了被追踪的值。例如::

      >>> from jax import jit
      >>> import jax.numpy as jnp
      >>> @jit
      ... def func(x, y):
      ...   return x if x.sum() < y.sum() else y

      >>> func(jnp.ones(4), jnp.zeros(4))  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerBoolConversionError: Attempted boolean conversion of JAX Tracer [...]

    我们可以把两个输入 ``x`` 和 ``y`` 都标记为静态，
    但那样就失去了在这里使用 :func:`jax.jit` 的意义。
    另一种选择是用三项的 :func:`jax.numpy.where` 重新表达这个 if 语句::

      >>> @jit
      ... def func(x, y):
      ...   return jnp.where(x.sum() < y.sum(), x, y)

      >>> func(jnp.ones(4), jnp.zeros(4))
      Array([0., 0., 0., 0.], dtype=float32)

    关于包含循环在内的更复杂控制流，参见
    :ref:`lax-control-flow`。

  针对被追踪值的控制流
    这个错误的另一个常见原因是，你不小心把某个布尔标志
    也纳入了追踪。例如::

      >>> @jit
      ... def func(x, normalize=True):
      ...   if normalize:
      ...     return x / x.sum()
      ...   return x

      >>> func(jnp.arange(5), True)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerBoolConversionError: Attempted boolean conversion of JAX Tracer ...

    这里由于标志 ``normalize`` 是被追踪的，
    它不能用于 Python 控制流。在这种情况下，
    最好的办法大概是把这个值标记为静态::

      >>> from functools import partial
      >>> @jit(static_argnames=['normalize'])
      ... def func(x, normalize=True):
      ...   if normalize:
      ...     return x / x.sum()
      ...   return x

      >>> func(jnp.arange(5), True)
      Array([0. , 0.1, 0.2, 0.3, 0.4], dtype=float32)

    关于 ``static_argnums`` 的更多内容，参见 :func:`jax.jit` 的文档。

  使用不感知 JAX 的函数
    另一个常见原因是在 JAX 代码中使用了
    不感知 JAX 的函数。例如：

      >>> @jit
      ... def func(x):
      ...   return min(x, 0)

      >>> func(2)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      TracerBoolConversionError: Attempted boolean conversion of JAX Tracer ...

    在这个例子中，出错是因为 Python 内置的 ``min`` 函数
    与 JAX 变换不兼容。可以把它替换为
    ``jnp.minimum`` 来修复：

      >>> @jit
      ... def func(x):
      ...   return jnp.minimum(x, 0)

      >>> print(func(2))
      0

  如果你想深入了解追踪器与普通值、具体值与抽象值
  之间还有哪些微妙之处，可以阅读
  :ref:`faq-different-kinds-of-jax-values`。
  """
  def __init__(self, tracer: core.Tracer):
    JAXTypeError.__init__(self,
        f"Attempted boolean conversion of {tracer._error_repr()}."
        f"{tracer._origin_msg()}")


@export
class UnexpectedTracerError(JAXTypeError):
  """
  当你使用了从函数中泄漏出来的 JAX 值时，就会抛出这个错误。
  什么叫泄漏一个值？如果你对函数 ``f`` 使用 JAX 变换，
  而该函数把某个中间值的引用存到了 ``f`` 之外的某个作用域中，
  那么这个值就被视为已经泄漏。泄漏值是一种副作用。
  （关于如何避免副作用，可阅读
  `Pure Functions <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions>`_）

  JAX 会在你之后于另一个操作中使用这个泄漏值时检测到泄漏，
  此时它会抛出 ``UnexpectedTracerError``。
  要修复这个问题，请避免副作用：如果某个函数计算出的值
  在外层作用域中需要用到，就应当从被变换的函数中显式返回该值。

  具体来说，``Tracer`` 是 JAX 在变换期间对函数中间值的内部表示，
  例如在 :func:`~jax.jit`、:func:`~jax.pmap`、
  :func:`~jax.vmap` 等之中。在变换之外遇到 ``Tracer``
  就意味着发生了泄漏。

  泄漏值的生命周期
    请看下面这个例子：一个被变换的函数
    把值泄漏到了外层作用域::

      >>> from jax import jit
      >>> import jax.numpy as jnp

      >>> outs = []
      >>> @jit                   # 1
      ... def side_effecting(x):
      ...   y = x + 1            # 3
      ...   outs.append(y)       # 4

      >>> x = 1
      >>> side_effecting(x)      # 2
      >>> outs[0] + 1            # 5  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      UnexpectedTracerError: Encountered an unexpected tracer.

    在这个例子中，我们把一个被追踪的值从内部变换作用域泄漏到了
    外层作用域。``UnexpectedTracerError`` 是在泄漏值被使用时
    抛出的，而不是在值泄漏时抛出。

    这个例子也展示了泄漏值的生命周期：

      1. 函数被变换（这里是经 :func:`~jax.jit` 变换）
      2. 被变换的函数被调用（由此启动对该函数的抽象追踪，
         并把 ``x`` 变成一个 ``Tracer``）
      3. 之后会被泄漏的中间值 ``y`` 被创建出来
         （被追踪函数的中间值同样是 ``Tracer``）
      4. 该值被泄漏（它被追加到外层作用域的一个列表中，
         通过旁路从函数中逃逸出去）
      5. 泄漏的值被使用，于是抛出 ``UnexpectedTracerError``。

    ``UnexpectedTracerError`` 的消息会通过包含每个阶段的信息，
    尽力指出你代码中这些位置。它们依次是：

      1. 被变换函数的名称（``side_effecting``），以及是哪个变换
         发起了这次追踪 :func:`~jax.jit`）。
      2. 重建的栈回溯，指出泄漏的 ``Tracer`` 是在哪里创建的，
         其中包含被变换函数的调用位置。
         （``When the Tracer was created, the final 5 stack frames were...``）。
      3. 根据重建的栈回溯，指出创建该泄漏 ``Tracer``
         的那行代码。
      4. 错误消息中不包含泄漏位置，因为这一点很难确定！
         JAX 只能告诉你泄漏的值长什么样
         （它是什么形状、在哪里创建），
         以及它是越过哪个边界泄漏的（变换的名称
         和被变换函数的名称）。
      5. 当前错误的栈回溯指向该值被使用的位置。

    把该值从被变换的函数中返回出来
    即可修复这个错误::

      >>> from jax import jit
      >>> import jax.numpy as jnp

      >>> outs = []
      >>> @jit
      ... def not_side_effecting(x):
      ...   y = x+1
      ...   return y

      >>> x = 1
      >>> y = not_side_effecting(x)
      >>> outs.append(y)
      >>> outs[0] + 1  # all good! no longer a leaked value.
      Array(3, dtype=int32, weak_type=True)

  泄漏检查器
    如上面第 2 点和第 3 点所述，JAX 会显示重建的栈回溯，
    指出泄漏的值是在哪里创建的。这是因为
    JAX 只在泄漏值被使用时抛出错误，而不是在值泄漏时。
    这并不是抛出该错误最有用的位置，
    因为要修复错误，你需要知道 ``Tracer``
    是在哪里泄漏的。

    为了让这个位置更容易定位，你可以使用泄漏检查器。
    启用泄漏检查器后，一旦有 ``Tracer`` 泄漏就会立刻抛出错误。
    （更准确地说，它会在 ``Tracer`` 所泄漏自的
    被变换函数返回时抛出错误）

    要启用泄漏检查器，可以使用 ``JAX_CHECK_TRACER_LEAKS``
    环境变量，或 ``with jax.checking_leaks()`` 上下文管理器。

    .. note::
      注意该工具是实验性的，可能会报告误报。
      它的工作原理是禁用 JAX 的部分缓存，因此会对
      性能产生负面影响，只应在调试时使用。

    用法示例::

      >>> from jax import jit
      >>> import jax.numpy as jnp

      >>> outs = []
      >>> @jit
      ... def side_effecting(x):
      ...   y = x+1
      ...   outs.append(y)

      >>> x = 1
      >>> with jax.checking_leaks():
      ...   y = side_effecting(x)  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
          ...
      Exception: Leaked Trace

  """

  def __init__(self, msg: str):
    super().__init__(msg)


@export
class KeyReuseError(JAXTypeError):
  """
  当 PRNG key 以不安全的方式被复用时，就会抛出这个错误。
  只有在 `jax_debug_key_reuse` 被设为 `True` 时，
  才会检查 key 复用。

  下面是一个会导致此类错误的简单代码示例::

    >>> with jax.debug_key_reuse(True):  # doctest: +SKIP
    ...   key = jax.random.key(0)
    ...   value = jax.random.uniform(key)
    ...   new_value = jax.random.uniform(key)
    ...
    ---------------------------------------------------------------------------
    KeyReuseError                             Traceback (most recent call last)
    ...
    KeyReuseError: Previously-consumed key passed to jit-compiled function at index 0

  这类 key 复用之所以成为问题，是因为 JAX 的 PRNG 是无状态的，
  key 必须手动拆分；更多信息参见
  `伪随机数教程 <https://docs.jax.dev/en/latest/random-numbers.html>`_。
  """
