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

# 文件职责：为 JAX 的各类变换（jit、vjp、grad 等）提供函数包装与线性化辅助设施。
# 核心是 `WrappedFun`：它把被包装的函数 `f` 与一串嵌套的变换生成器叠成栈，
# 在调用时依次改写动态参数、关键字参数与返回值，并可收集辅助输出（aux）。
# 相关设施包括 `Store`/`EqualStore` 辅助输出存储、`wrap_init` 包装入口、
# `cache` 记忆化、`DebugInfo` 追踪信息，以及 `merge_linear_aux` 辅助输出合并。

"""
用于定义与变换组合而成的函数的工具。

例如，

   from jax._src import linear_util as lu

   # 生成一个 WrappedFun，用于对 `f` 施加变换
   wf = lu.wrap_init(f, debug_info=api_util.debug_info("test", f, (), {}))

`WrappedFun` 对象表示一个函数 `f`，并携带一串嵌套的变换。这些变换需要在
调用时施加于位置参数与关键字参数，在返回时施加于函数的返回值。
一个变换可以接受一些在包装时给定的静态位置参数，
并且还可以返回一些辅助输出，
如下所示：

    wf, aux_out_thunk = trans1(wf, static_arg)

我们可以调用被变换后的函数。首先，变换会施加于动态参数与关键字参数，
以产生新的动态参数与关键字参数；
然后调用底层函数，并把变换施加于结果之上。
如果存在多个变换，它们会构成一个栈。
参数按“最后施加的变换优先”的顺序被处理，
结果则按“最先施加的变换优先”的顺序被处理，
也就是说，变换栈对参数与结果的作用次序是相反的。

    res = wf.call_wrapped(dynamic_args, kwargs)
    # 现在 `aux_out_thunk()` 就是辅助输出。

一个变换写成一个生成器函数，它接受零个或多个静态位置参数
（这些参数在变换被实例化时给出），
以及待变换的位置参数与关键字参数。
该生成器会 yield 两次：

    @lu.transformation_with_aux
    def trans1(static_arg, *dynamic_args, **kwargs):
      ...
      # 第一次 yield：变换后的 (args, kwargs) 二元组；随后取回结果。
      results = yield (new_dynamic_args, new_kwargs)
      ...
      # 第二次 yield：(变换后的结果, 辅助输出) 二元组
      yield new_results, auxiliary_output


`WrappedFun` 对象显式地表示这组变换，因此可以把它们用作记忆化的字典键。
`WrappedFun` 对象只有在计算同一个函数时才比较相等。
生成器的静态位置参数与动态位置参数，
以及辅助输出数据，都必须是不可变的，
因为它们会被存放在函数记忆化表中。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
import re
import time
from typing import Any, NamedTuple
from collections.abc import Hashable
import warnings
import weakref

from jax._src import config
from jax._src import core
from jax._src import traceback_util
from jax._src.tree_util import KeyPath, generate_key_paths, keystr
from jax._src.util import curry, fun_name, register_cache


traceback_util.register_exclusion(__file__)


class StoreException(Exception): pass


class EmptyStoreValue: pass
_EMPTY_STORE_VALUE = EmptyStoreValue()

class Store:
  """用于存放一个值的存储，会在覆盖写入或读取空存储时进行检查。"""
  __slots__ = ("_val",)

  def __init__(self):
    self._val = _EMPTY_STORE_VALUE

  def store(self, val):
    if self._val is not _EMPTY_STORE_VALUE:
      raise StoreException("Store occupied")
    self._val = val

  def reset(self):
    # 只应在异常情形下调用（例如调试）。
    self._val = _EMPTY_STORE_VALUE

  @property
  def val(self):
    if not self:
      raise StoreException("Store empty")
    return self._val

  def __nonzero__(self):
    return self._val is not _EMPTY_STORE_VALUE

  __bool__ = __nonzero__

class EqualStore:
  __slots__ = ('_store',)

  def __init__(self):
    self._store = Store()

  @property
  def val(self):
    return self._store.val

  def store(self, val):
    try:
      self._store.store(val)
    except StoreException as e:
      try:
        okay = bool(self._store._val == val)
      except:
        raise e from None
      else:
        if not okay:
          raise StoreException("Store occupied with not-equal value") from None

  def reset(self):
    self._store.reset()


class WrappedFun:
  """表示一个待施加 `transforms` 的函数 `f`。

  Args:
    f: 待变换的函数。
    f_transformed: 变换后的函数。
    transforms: 由 `(gen, gen_static_args)` 二元组构成的元组，表示要施加到
      `f` 上的变换。这里 `gen` 是一个生成器函数，
      而 `gen_static_args` 是传给该生成器的一个静态参数元组，
      用于在包装时确定该变换的静态行为。
      关于生成器应有的行为，见本模块开头的描述。
    stores: 存放 `transforms` 辅助输出的 out_store 列表。
    params: 由 `(name, param)` 二元组构成的元组，表示需要作为关键字参数
      传给 `f` 的额外参数，它们会与变换后的关键字参数
      一起传入。
    in_type: 可选的输入类型
    debug_info: 关于被包装函数的调试信息。
  """
  __slots__ = ("f", "f_transformed", "transforms", "stores", "params", "in_type", "debug_info")

  f: Callable
  f_transformed: Callable
  transforms: tuple[tuple[Callable, tuple[Hashable, ...]], ...]
  stores: tuple[Store | EqualStore | None, ...]
  params: tuple[tuple[str, Any], ...]
  in_type: core.InputType | None
  debug_info: DebugInfo

  def __init__(self, f: Callable,
               f_transformed: Callable,
               transforms: tuple[tuple[Callable, tuple[Hashable, ...]], ...],
               stores: tuple[Store | EqualStore | None, ...],
               params: tuple[tuple[str, Hashable], ...],
               in_type: core.InputType | None,
               debug_info: DebugInfo):
    self.f = f
    self.f_transformed = f_transformed
    self.transforms = transforms
    self.stores = stores
    self.params = params
    self.in_type = in_type
    self.debug_info = debug_info

  @property
  def __name__(self):
    return fun_name(self.f, "<unnamed wrapped function>")

  def wrap(self, gen, gen_static_args,
           out_store: Store | EqualStore | None) -> WrappedFun:
    """再添加一个变换及其存储。"""
    if out_store is None:
      return WrappedFun(self.f, partial(gen, self.f_transformed, *gen_static_args),
                        ((gen, gen_static_args),) + self.transforms,
                        (out_store,) + self.stores, self.params, None, self.debug_info)
    else:
      return WrappedFun(self.f, partial(gen, self.f_transformed, out_store, *gen_static_args),
                        ((gen, gen_static_args),) + self.transforms,
                        (out_store,) + self.stores, self.params, None, self.debug_info)

  def populate_stores(self, stores):
    """把 `stores` 中的值复制到 `self.stores` 中。"""
    for self_store, other_store in zip(self.stores, stores):
      if self_store is not None:
        self_store.store(other_store.val)

  def call_wrapped(self, *args, **kwargs):
    """调用变换后的函数"""
    return self.f_transformed(*args, **kwargs)

  def __repr__(self):
    def transform_to_str(x):
      i, (gen, args) = x
      return f"{i}   : {fun_name(gen)}   {fun_name(args)}"
    transformation_stack = map(transform_to_str, enumerate(self.transforms))
    return "Wrapped function:\n" + '\n'.join(transformation_stack) + '\nCore: ' + fun_name(self.f) + '\n'

  def __hash__(self):
    return hash((self.f, self.transforms, self.params, self.in_type,
                 self.debug_info))

  def __eq__(self, other):
    return (self.f == other.f and self.transforms == other.transforms and
            self.params == other.params and self.in_type == other.in_type and
            self.debug_info == other.debug_info)

  def replace_debug_info(self, dbg: core.DebugInfo) -> WrappedFun:
    return WrappedFun(self.f, self.f_transformed, self.transforms,
                      self.stores, self.params, self.in_type,
                      dbg)

  def with_unknown_names(self) -> WrappedFun:
    return self.replace_debug_info(self.debug_info.with_unknown_names())

@curry
def transformation2(gen, fun: WrappedFun, *gen_static_args) -> WrappedFun:
  """向一个 WrappedFun 再添加一个变换。

  Args:
    gen: 变换生成器函数
    fun: 要施加该变换的 WrappedFun
    gen_static_args: 生成器函数的静态参数
  """
  return fun.wrap(gen, gen_static_args, None)

# 仅为向后兼容。TODO: 弃用
@curry
def transformation(gen, fun: WrappedFun, *gen_static_args) -> WrappedFun:
  def gen2(f, *args, **kwargs):
    gen_inst = gen(*args, **kwargs)
    args_, kwargs_ = next(gen_inst)
    return gen_inst.send(f(*args_, **kwargs_))
  return transformation2(gen2, fun, *gen_static_args)()

# 仅为向后兼容。TODO: 弃用
@curry
def transformation_with_aux(gen, fun: WrappedFun, *gen_static_args) -> WrappedFun:
  def gen2(f, store, *args, **kwargs):
    gen_inst = gen(*args, **kwargs)
    args_, kwargs_ = next(gen_inst)
    ans, aux = gen_inst.send(f(*args_, **kwargs_))
    store.store(aux)
    return ans
  return transformation_with_aux2(gen2, fun, *gen_static_args)()

@curry
def transformation_with_aux2(
    gen, fun: WrappedFun, *gen_static_args, use_eq_store: bool = False,
    unk_names: bool = False) -> tuple[WrappedFun, Callable[[], Any]]:
  """向一个 WrappedFun 再添加一个带辅助输出的变换。"""
  out_store = Store() if not use_eq_store else EqualStore()
  out_thunk = lambda: out_store.val
  fun = fun.wrap(gen, gen_static_args, out_store)
  fun = fun.with_unknown_names() if unk_names else fun
  return fun, out_thunk

class InitialResultPaths:
  pass
initial_result_paths = InitialResultPaths()

class DebugInfo(NamedTuple):
  """关于某个函数及其参数与结果的调试信息。"""
  traced_for: str             # 例如 'jit'、'scan' 等

  func_src_info: str
  """例如 ``f'{fun.__name__} at {filename}:{lineno}'``；如果我们没有源位置
  信息，则形如 ``'{fun.__name__}'``。第一个词始终是函数名，
  该名字可能是 '<unknown>'。
  """

  arg_names: tuple[str, ...] | None
  """展平后的非静态参数名的路径，
  例如 ``('x', 'dict_arg["a"]', ...)``。
  对于并不对应用户命名参数的那些参数（例如 ``jax.jvp`` 中的切向量参数），
  或者对于我们还未能正确追踪的参数，使用空字符串表示。
  取值 ``None`` 表示参数名未知。

  目前，``arg_names`` 的准确性是尽力而为的。
  请使用 ``safe_arg_names`` 来检测并处理 ``arg_names`` 中
  元素数量不符合预期的情况。
  """

  result_paths: tuple[str, ...] | InitialResultPaths | Callable[[], tuple[str, ...]] | None
  """展平后结果的路径。例如，对于返回数组元组的函数，
  形如 `('result[0]', result[1])`；对于返回单个数组的函数，
  形如 `(result,)`。取值 `None` 表示路径未知。

  在最初创建 `DebugInfo` 时，我们可能使用取值
  `initial_result_paths`；在开始追踪之前，当我们把调试信息放入
  `lu.WrappedFun` 时，会把它替换成一个 thunk。追踪结束后，
  我们调用 `self.resolve_result_paths()` 来执行该 thunk，
  并把结果路径替换为一个元组。

  请使用 `safe_result_paths` 来检测并处理 `result_paths` 中
  元素数量不符合预期的情况。
  """

  def resolve_result_paths(self) -> DebugInfo:
    """返回一个已解析结果路径的调试信息。"""
    assert self.result_paths is not initial_result_paths
    if callable(self.result_paths):
      paths = tuple(self.result_paths())
      return self._replace(result_paths=paths)
    return self

  @property
  def func_name(self) -> str:
    return self.func_src_info.split(" ")[0]

  def replace_func_name(self, name: str) -> DebugInfo:
    func_src_comps = self.func_src_info.split(" ")
    func_src_comps[0] = name
    return self._replace(func_src_info=" ".join(func_src_comps))

  def set_result_paths(self, ans):
    result_paths = tuple(f"result{_clean_keystr_arg_names(path)}"
                         for path, _ in generate_key_paths(ans))
    return self._replace(result_paths=result_paths)

  @property
  def func_filename(self) -> str | None:
    m = _re_func_src_info.match(self.func_src_info)
    if not m: return None
    return m.group(3)

  @property
  def func_lineno(self) -> int | None:
    m = _re_func_src_info.match(self.func_src_info)
    if not m or m.group(4) is None: return None
    return int(m.group(4))

  def safe_arg_names(self, expected_count: int) -> tuple[str, ...]:
    """在带安全检查的情况下获取 arg_names。"""
    self.assert_arg_names(expected_count)
    if self.arg_names is not None:
      return self.arg_names
    return ("",) * expected_count

  def assert_arg_names(self, expected_count: int):
    assert self.arg_names is None or len(self.arg_names) == expected_count, (
        expected_count, self)

  def filter_arg_names(self, keep: Sequence[bool]) -> tuple[str, ...] | None:
    """只保留 `keep` 为 True 的那些 arg_names。"""
    if self.arg_names is None:
      return None
    return tuple(v for v, b in zip(self.safe_arg_names(len(keep)), keep) if b)

  def safe_result_paths(self, expected_count: int) -> tuple[str, ...]:
    """在带安全检查的情况下获取结果路径。空路径表示未知。"""
    assert not isinstance(self.result_paths, InitialResultPaths) and not callable(self.result_paths), self
    self.assert_result_paths(expected_count)
    if self.result_paths is not None:
      return self.result_paths

    return ("",) * expected_count

  def assert_result_paths(self, expected_count: int):
    if self.result_paths is None:
      return
    assert isinstance(self.result_paths, tuple), self
    assert len(self.result_paths) == expected_count, (expected_count, self)

  def filter_result_paths(self, keep: Sequence[bool]) -> tuple[str, ...] | None:
    """只保留 `keep` 为 True 的那些 result_paths。"""
    assert not isinstance(self.result_paths, InitialResultPaths) and not callable(self.result_paths), self
    if self.result_paths is None: return None
    return tuple(v for v, b in zip(self.result_paths, keep) if b)

  def with_unknown_names(self) -> DebugInfo:
    return self._replace(arg_names=None, result_paths=None)


_re_func_src_info = re.compile(r"([^ ]+)( at (.+):(\d+))?$")

def _missing_debug_info(for_what: str) -> DebugInfo:
  warnings.warn(
      f"{for_what} is missing a DebugInfo object. "
      "This behavior is deprecated, use api_util.debug_info() to "
      "construct a proper DebugInfo object and propagate it to this function. "
      "See https://github.com/jax-ml/jax/issues/26480 for more details.",
      DeprecationWarning, stacklevel=2)
  return DebugInfo("missing_debug_info", "<missing_debug_info>", None, None)

def wrap_init(f: Callable, params=None, *, debug_info: DebugInfo) -> WrappedFun:
  """把函数 `f` 包装成一个 `WrappedFun`，以便对其施加变换。"""
  params_dict = {} if params is None else params
  params = () if params is None else tuple(sorted(params.items()))
  debug_info = debug_info._replace(result_paths=None)
  fun = WrappedFun(f, partial(f, **params_dict), (), (), params, None, debug_info)
  return fun


# 我们把 <flat index 0> 替换为 0
_re_clean_keystr_arg_names = re.compile(r"<flat index ([^>]+)>")
def _clean_keystr_arg_names(k: KeyPath) -> str:
  res = keystr(k)
  return _re_clean_keystr_arg_names.sub(r"\1", res)

def annotate(f: WrappedFun, in_type: core.InputType | None) -> WrappedFun:
  assert f.in_type is None
  if in_type is None:
    return f
  _check_input_type(in_type)
  return WrappedFun(f.f, f.f_transformed, f.transforms, f.stores, f.params,
                    in_type, f.debug_info)

def _check_input_type(in_type: core.InputType) -> None:
  # 检查 in_type 在语法上是否良构
  assert type(in_type) is tuple
  assert all(isinstance(a, core.AbstractValue) for a in in_type)

def cache(call: Callable, *,
          explain: Callable[[WrappedFun, bool, dict, tuple, float], None] | None = None):
  """用于首个参数为 WrappedFun 的函数的记忆化装饰器。

  Args:
    call: 一个 Python 可调用对象，其第一个参数是 WrappedFun。
      该 WrappedFun 上底层的变换（transforms）与参数（params）
      会作为记忆化缓存键的一部分。

    explain: 一个在缓存未命中时被调用的函数，
      用于记录未命中的原因说明。
      调用时传入 `(fun, is_cache_first_use, cache, key, elapsed_sec)`。

  Returns:
     返回 ``call`` 的记忆化版本。
  """
  fun_caches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

  def memoized_fun(fun: WrappedFun, *args):
    cache = fun_caches.setdefault(fun.f, new_cache := {})
    key = (fun.transforms, fun.params, fun.in_type, args, config.trace_context())
    result = cache.get(key, None)
    if result is not None:
      ans, stores = result
      fun.populate_stores(stores)
    else:
      start = 0.0
      if do_explain := explain and config.explain_cache_misses.value:
        start = time.time()
      ans = call(fun, *args)
      if do_explain:
        assert explain
        explain(fun, cache is new_cache, cache, key, time.time() - start)
      cache[key] = (ans, fun.stores)

    return ans

  def _evict_function(f):
    fun_caches.pop(f, None)

  memoized_fun.evict_function = _evict_function  # pyrefly: ignore[missing-attribute]
  memoized_fun.cache_clear = fun_caches.clear  # pyrefly: ignore[missing-attribute]
  register_cache(memoized_fun, str(call))
  return memoized_fun

@transformation2
def hashable_partial(f, *args):
  return f(*args)


def merge_linear_aux(aux1, aux2):
  try:
    out1 = aux1()
  except StoreException:
    # 存储 1 未被占用，所以存储 2 最好已被占用
    try:
      out2 = aux2()
    except StoreException:
      raise StoreException("neither store occupied") from None
    else:
      return False, out2
  else:
    # 存储 1 已被占用，所以来检查存储 2 未被占用
    try:
      out2 = aux2()
    except StoreException:
      return True, out1
    else:
      raise StoreException("both stores occupied")
