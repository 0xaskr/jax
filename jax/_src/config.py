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

# 文件职责：JAX 运行时配置系统（`jax.config`）的核心实现，统一管理全部配置选项。
# 每个选项同时对应环境变量、可选的 absl flag，以及一个 `State` 对象——后者既
# 保存进程级取值，也可作为上下文管理器临时切换线程局部取值。
# 本模块提供 `bool_state`/`enum_state`/`int_state` 等状态构造器，供本文件后续部分
# 和各子模块声明 `jax_*` 配置；用户可通过 `config.update(...)` 或
# `with config.jax_xxx(...)` 切换运行行为，相关状态可计入 JIT 缓存键与追踪上下文。

from __future__ import annotations

from collections.abc import Callable, Sequence
from collections.abc import Generator
import contextlib
import enum
import functools
import itertools
import logging
import os
import sys
from typing import Any, NoReturn, Protocol, TypeVar, cast
from typing import TYPE_CHECKING

from jax._src import logging_config
from jax._src.lib import _jax
from jax._src.lib import guard_lib
from jax._src.lib import jax_jit
from jax._src.lib import xla_client

config_ext = _jax.config

logger = logging.getLogger(__name__)

_T = TypeVar('_T')


def bool_env(varname: str, default: bool) -> bool:
  """读取环境变量并将其解释为布尔值。

  真值（不区分大小写）为：'y'、'yes'、't'、'true'、'on' 和 '1'；
  假值为 'n'、'no'、'f'、'false'、'off' 和 '0'。

  Args:
    varname: 变量名
    default: 默认布尔值
  Raises: 若环境变量为上述之外的任何值，则抛出 ValueError。
  """
  val = os.getenv(varname, str(default))
  val = val.lower()
  if val in ('y', 'yes', 't', 'true', 'on', '1'):
    return True
  elif val in ('n', 'no', 'f', 'false', 'off', '0'):
    return False
  else:
    raise ValueError(f"invalid truth value {val!r} for environment {varname!r}")

def int_env(varname: str, default: int) -> int:
  """读取环境变量并将其解释为整数。"""
  return int(os.getenv(varname, str(default)))


class ValueHolder[ValueType](Protocol):
  """配置值的持有者。

  值持有者有两类：``Flag``，它恰好被赋值一次，
  之后永不修改；以及 ``State``，它可以通过上下文
  管理器在线程内被局部修改。
  """

  value: ValueType

  def _set(self, value: ValueType) -> None: ...


class Config:
  _HAS_DYNAMIC_ATTRIBUTES = True
  if TYPE_CHECKING:

    def __getattr__(self, name: str) -> Any:
      ...

    def __setattr__(self, name: str, value: Any) -> None:
      ...

  def __init__(self):
    self._value_holders: dict[str, ValueHolder] = {}
    self.meta = {}
    self.use_absl = False
    self._contextmanager_flags = set()

  def update(self, name, val):
    if name not in self._value_holders:
      raise AttributeError(f"Unrecognized config option: {name}")
    self._value_holders[name]._set(val)

  def read(self, name):
    if name in self._contextmanager_flags:
      raise AttributeError(
          "For flags with a corresponding contextmanager, read their value "
          f"via e.g. `config.{name}` rather than `config.FLAGS.{name}`.")
    return self._read(name)

  def _read(self, name):
    try:
      return self._value_holders[name].value
    except KeyError:
      raise AttributeError(f"Unrecognized config option: {name}")

  @property
  def values(self):
    return {name: holder.value for name, holder in self._value_holders.items()}

  def add_option(self, name, holder, opt_type, meta_args, meta_kwargs):
    if name in self._value_holders:
      raise Exception(f"Config option {name} already defined")
    self._value_holders[name] = holder
    self.meta[name] = (opt_type, meta_args, meta_kwargs)

  def config_with_absl(self):
    """为 JAX 配置注册 absl flag。

    例如，对于每个用 bool_state() 定义的 JAX 配置，本方法都会注册一个同名的
    absl 布尔 flag。

    如果你使用 `app.run(main)` 并且需要
    JAX flag，推荐调用本方法。

    Examples:

    ```python
    from absl import app
    import jax
    ...

    if __name__ == '__main__':
      jax.config.config_with_absl()
      app.run(main)
    ```

    """
    import absl.flags as absl_FLAGS  # noqa: F401  # pyrefly: ignore[missing-import]
    from absl import app, flags as absl_flags  # pyrefly: ignore[missing-import]

    self.use_absl = True
    self.absl_flags = absl_flags
    absl_defs = {
        bool: absl_flags.DEFINE_bool,
        int: absl_flags.DEFINE_integer,
        float: absl_flags.DEFINE_float,
        str: absl_flags.DEFINE_string,
        'enum': absl_flags.DEFINE_enum,
        'enum_class': absl_flags.DEFINE_enum_class,
    }

    for name, (flag_type, meta_args, meta_kwargs) in self.meta.items():
      holder = self._value_holders[name]
      absl_defs[flag_type](name, holder.value, *meta_args, **meta_kwargs)
    app.call_after_init(lambda: self.complete_absl_config(absl_flags))

  def complete_absl_config(self, absl_flags):
    # NOTE：避免从本模块外部调用本方法。请改用
    # `config_with_absl()`，以及（极少数情况下）`parse_flags_with_absl()`。
    for name, holder in self._value_holders.items():
      try:
        flag = absl_flags.FLAGS[name]
      except KeyError:
        # 如果在调用 config_with_absl() 之后、运行 complete_absl_config 之前
        # 新增了 flag，就会出现这种情况。原则上我们可以在 DEFINE_... 中加入
        # 代码，以便在 config_with_absl() 已被调用时把新增的 flag 注册到
        # ABSL，但更合理的做法或许是让用户晚一些再调用
        # config_with_absl()。
        continue
      if flag.present:
        holder._set(flag.value)

  def parse_flags_with_absl(self):
    """解析以 --jax 开头的命令行参数。

    本方法只应供高级用户使用。大多数用户应改用
    :meth:`config_with_absl`。

    本方法有严重的局限：例如，尽管它只解析
    --jax* 命令行参数，却会运行所有已注册 absl
    flag 的校验器，甚至包括尚未设置的非 JAX
    flag；因此对于非 JAX flag，校验器作用在
    flag 的默认值上，而不是命令行参数所指示的值上。
    """
    global already_configured_with_absl
    if not already_configured_with_absl:
      # 只从 argv 中提取 --jax... 这些 flag（位于第一个 -- 之前）。在某些
      # 环境（例如 ipython/colab）中，argv 可能混杂着 absl 可解析的内容以及
      # 其他无用内容。
      jax_argv = itertools.takewhile(lambda a: a != '--', sys.argv)
      jax_argv = ['', *(a for a in jax_argv if a.startswith('--jax'))]

      import absl.flags  # pyrefly: ignore[missing-import]
      self.config_with_absl()
      absl.flags.FLAGS(jax_argv, known_only=True)
      self.complete_absl_config(absl.flags)
      already_configured_with_absl = True

register_trace_context_callback = []

trace_context = config_ext.trace_context
trace_context_names = config_ext.trace_context_names

config = Config()

_read = config._read
update = config.update
parse_flags_with_absl = config.parse_flags_with_absl


class NoDefault: pass
no_default = NoDefault()

config_states = {}

class State(config_ext.Config[_T]):

  __slots__ = (
      '_name', '_update_thread_local_hook', '_update_global_hook',
      '_parser', '_default_context_manager_value', '__doc__', '__name__',
  )

  def __init__(
      self,
      name: str,
      default: _T,
      help,
      update_global_hook: Callable[[_T], None] | None = None,
      update_thread_local_hook: Callable[[_T | None], None] | None = None,
      parser: Callable[[Any], Any] | None = None,
      extra_description: str = '',
      default_context_manager_value: Any = no_default,
      include_in_jit_key: bool = False,
      include_in_trace_context: bool = False,
  ):
    if parser is not None:
      default = parser(default)
    super().__init__(name, default, include_in_jit_key=include_in_jit_key,
                     include_in_trace_context=include_in_trace_context)
    self._name = name
    self.__name__ = name[4:] if name.startswith('jax_') else name
    self.__doc__ = (f"Context manager for `{name}` config option"
                    f"{extra_description}.\n\n{help}")
    self._update_global_hook = update_global_hook
    self._update_thread_local_hook = update_thread_local_hook
    self._parser = parser
    self._default_context_manager_value = default_context_manager_value
    if self._update_global_hook:
      self._update_global_hook(default)
    config_states[name] = self

  @property
  def name(self):
    return self._name

  def __bool__(self) -> NoReturn:
    raise TypeError(
        "bool() not supported for instances of type '{0}' "
        "(did you mean to use '{0}.value' instead?)".format(
            type(self).__name__))

  def _set(self, value: _T) -> None:
    if self._parser:
      value = self._parser(value)
    self.set_global(value)
    if self._update_global_hook:
      self._update_global_hook(value)

  def __call__(self, new_val: Any = no_default):
    return StateContextManager(self, new_val)

  def _add_hooks(self, update_global_hook, update_thread_local_hook):
    """为已有上下文管理器添加钩子的私有方法。

    用于避免循环导入依赖。"""
    self._update_thread_local_hook = update_thread_local_hook
    self._update_global_hook = update_global_hook
    update_global_hook(self.get_global())


class StateContextManager[FuncType: Callable[..., Any]]:
  __slots__ = ['state', 'new_val', 'prev']

  def __init__(self, state, new_val):
    self.state = state

    if new_val is no_default:
      if state._default_context_manager_value is not no_default:
        new_val = state._default_context_manager_value  # 构造函数已提供 default_context_manager_value
      else:
        # 构造函数未提供 default_value，调用时也未提供作为参数的值，
        # 因此抛出错误
        raise TypeError(f"Context manager for {state.__name__} config option "
                        "requires an argument representing the new value for "
                        "the config option.")
    if state._parser:
      self.new_val = state._parser(new_val)
    else:
      self.new_val = new_val

  def __enter__(self):
    self.prev = self.state.swap_local(self.new_val)
    if self.state._update_thread_local_hook:
      self.state._update_thread_local_hook(self.new_val)

  def __exit__(self, exc_type, exc_value, traceback):
    self.state.set_local(self.prev)
    if self.state._update_thread_local_hook:
      if self.prev is config_ext.unset:
        self.state._update_thread_local_hook(None)
      else:
        self.state._update_thread_local_hook(cast(Any | None, self.prev))

  def __call__(self, func: FuncType) -> FuncType:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      with StateContextManager(self.state, self.new_val):
        return func(*args, **kwargs)
    return cast(FuncType, wrapper)


UPGRADE_BOOL_HELP = (
    " This will be enabled by default in future versions of JAX, at which "
    "point all uses of the flag will be considered deprecated (following "
    "the `API compatibility policy "
    "<https://docs.jax.dev/en/latest/api_compatibility.html>`_).")

UPGRADE_BOOL_EXTRA_DESC = " (transient)"


def bool_state(
    name: str,
    default: bool,
    help: str,
    *,
    update_global_hook: Callable[[bool], None] | None = None,
    update_thread_local_hook: Callable[[bool | None], None] | None = None,
    upgrade: bool = False,
    extra_description: str = '',
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
    validator: Callable[[str], None] | None = None,
) -> State[bool]:
  """设置线程局部状态，并返回用于管理它的上下文管理器。

  本函数是一个便捷包装器。它定义一个 flag、一个环境变量，
  以及对应的线程局部状态，这些都可以通过它返回的
  上下文管理器来管理。

  线程局部状态的值可以通过 ``config.<option_name>``
  属性读取，其中 ``config`` 是单例 ``Config`` 实例。

  Args:
    name: 字符串，会被转换为小写以定义配置选项（以及
      absl flag）的名字。它会被转换为大写，以定义
      对应的 shell 环境变量。
    default: 布尔值，该选项的默认值。
    help: 字符串，用于填充 flag 的帮助信息，同时也作为
      所返回上下文管理器的文档字符串。
    update_global_hook: 可选回调，在全局状态被修改或
      初次设置时，会以更新后的取值调用它。
    update_thread_local_hook: 可选回调，当线程局部状态被
      修改或初次设置时，会以更新后的取值
      调用它。
    upgrade: 可选指示符，表示该 flag 控制一项规范的
      功能升级，因此对即将引入的功能为 `True`，对
      将要废弃的旧功能为 `False`。
    extra_description: 字符串，可选：要添加到
      摘要描述中的额外信息。
    include_in_jit_key: 布尔值，可选：是否将该状态纳入
      JIT 缓存键。
    include_in_trace_context: 布尔值，可选：是否将该状态纳入
      追踪上下文。
    validator: 可选函数，用于校验该配置选项的值。

  Returns:
    用于控制线程局部状态取值的上下文管理器。

  Examples:

    ENABLE_FOO = config.bool_state(
        name='jax_enable_foo',
        default=False,
        help='Enable foo.')

    # 现在可以用 JAX_ENABLE_FOO shell 环境变量和 --jax_enable_foo
    # 命令行 flag 控制该配置选项在进程级的取值，此外还可以
    # 直接使用例如
    # ``config.update("jax_enable_foo", True)``。我们也可以使用
    # 上下文管理器：

    with enable_foo(True):
      ...

  线程局部状态或 flag 的值可以通过
  ``config.jax_enable_foo`` 访问。通过 ``config.FLAGS.jax_enable_foo``
  读取它会报错。
  """
  if not isinstance(default, bool):
    raise TypeError(f"Default value must be of type bool, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")
  default = bool_env(name.upper(), default)
  name = name.lower()
  if upgrade:
    help += ' ' + UPGRADE_BOOL_HELP
    extra_description += UPGRADE_BOOL_EXTRA_DESC
  config._contextmanager_flags.add(name)

  def parser(val):
    if validator:
      validator(val)
    return bool(val)

  s = State[bool](
      name, default, help, update_global_hook=update_global_hook,
      update_thread_local_hook=update_thread_local_hook,
      extra_description=extra_description, default_context_manager_value=True,
      parser=parser, include_in_jit_key=include_in_jit_key,
      include_in_trace_context=include_in_trace_context)
  config.add_option(name, s, bool, meta_args=[], meta_kwargs={"help": help})
  setattr(Config, name, property(lambda _: s.value))
  return s

def optional_bool_state(
    name: str,
    default: bool | None,
    help: str,
    *,
    update_global_hook: Callable[[bool], None] | None = None,
    update_thread_local_hook: Callable[[bool | None], None] | None = None,
    upgrade: bool = False,
    extra_description: str = '',
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
    validator: Callable[[str], None] | None = None):
  if default is not None and not isinstance(default, bool):
    raise TypeError(f"Default value must be of type bool, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")
  default = None if default is None else bool_env(name.upper(), default)
  name = name.lower()
  if upgrade:
    help += ' ' + UPGRADE_BOOL_HELP
    extra_description += UPGRADE_BOOL_EXTRA_DESC
  config._contextmanager_flags.add(name)

  def parser(val):
    if validator is not None:
      validator(val)
    return None if val is None else bool(val)

  s = State[bool | None](
      name, default, help, update_global_hook=update_global_hook,  # type: ignore
      update_thread_local_hook=update_thread_local_hook,
      extra_description=extra_description, default_context_manager_value=True,
      parser=parser, include_in_jit_key=include_in_jit_key,
      include_in_trace_context=include_in_trace_context)
  config.add_option(name, s, bool, meta_args=[], meta_kwargs={"help": help})
  setattr(Config, name, property(lambda _: s.value))
  return s


def enum_state(
    name: str,
    enum_values: Sequence[str],
    default: str,
    help: str,
    *,
    update_global_hook: Callable[[str], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
    extra_validator: Callable[[str], None] | None = None,
) -> State[str]:
  """设置线程局部状态，并返回用于管理它的上下文管理器。

  参见 ``bool_state`` 的文档字符串。

  Args:
    name: 字符串，会被转换为小写以定义配置选项（以及
      absl flag）的名字。它会被转换为大写，以定义
      对应的 shell 环境变量。
    enum_values: 字符串列表，表示该选项
      可能的取值。
    default: 字符串，默认值。
    help: 字符串，用于填充 flag 的帮助信息，同时也作为
      所返回上下文管理器的文档字符串。
    include_in_jit_key: 布尔值，可选：是否将该状态纳入
      JIT 缓存键。
    extra_validator: 可选函数，用于校验该配置选项
      的值。

  Returns:
    用于控制线程局部状态取值的上下文管理器。
  """
  if not isinstance(default, str):
    raise TypeError(f"Default value must be of type str, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")
  name = name.lower()
  default = os.getenv(name.upper(), default)
  if default not in enum_values:
    raise ValueError(f"Invalid value \"{default}\" for JAX flag {name}")
  config._contextmanager_flags.add(name)

  def parser(new_val):
    if type(new_val) is not str or new_val not in enum_values:
      raise ValueError(f"new enum value must be in {enum_values}, "
                       f"got {new_val} of type {type(new_val)}.")
    if extra_validator is not None:
      extra_validator(new_val)
    return new_val

  s = State[str](
      name,
      default,
      help,
      update_global_hook=update_global_hook,
      update_thread_local_hook=update_thread_local_hook,
      parser=parser,
      include_in_jit_key=include_in_jit_key,
      include_in_trace_context=include_in_trace_context,
  )
  config.add_option(
      name, s, 'enum',
      meta_args=[],
      meta_kwargs={"enum_values": enum_values, "help": help}
  )
  setattr(Config, name, property(lambda _: s.value))
  return s


def optional_enum_state(
    name: str,
    enum_values: Sequence[str],
    default: str | None,
    help: str,
    *,
    update_global_hook: Callable[[str | None], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
) -> State[str | None]:
  """设置线程局部状态，并返回用于管理它的上下文管理器。

  参见 ``bool_state`` 的文档字符串。

  Args:
    name: 字符串，会被转换为小写以定义配置选项（以及
      absl flag）的名字。它会被转换为大写，以定义
      对应的 shell 环境变量。
    enum_values: 字符串列表，表示该选项
      可能的取值。
    default: 可选字符串，默认值。
    help: 字符串，用于填充 flag 的帮助信息，同时也作为
      所返回上下文管理器的文档字符串。

  Returns:
    用于控制线程局部状态取值的上下文管理器。
  """
  if default is not None and not isinstance(default, str):
    raise TypeError(f"Default value must be of type str or None, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")
  name = name.lower()
  default = os.getenv(name.upper(), default)
  if default is not None and default not in enum_values:
    raise ValueError(f"Invalid value \"{default}\" for JAX flag {name}")
  config._contextmanager_flags.add(name)

  def parser(new_val):
    if (new_val is not None and
      (type(new_val) is not str or new_val not in enum_values)):
      raise ValueError(f"new enum value must be None or in {enum_values}, "
                       f"got {new_val} of type {type(new_val)}.")
    return new_val

  s = State[str | None](
      name, default, help, update_global_hook, update_thread_local_hook,
      parser, include_in_jit_key=include_in_jit_key,
      include_in_trace_context=include_in_trace_context,
  )
  config.add_option(
      name, s, 'enum',
      meta_args=[],
      meta_kwargs={"enum_values": enum_values, "help": help}
  )
  setattr(Config, name, property(lambda _: s.value))
  return s


def enum_class_state[EnumType: enum.Enum](
    name: str,
    enum_class: type[EnumType],
    default: EnumType,
    help: str,
    *,
    update_global_hook: Callable[[EnumType], None] | None = None,
    update_thread_local_hook: Callable[[EnumType | None], None] | None = None,
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
    extra_validator: Callable[[EnumType], None] | None = None,
) -> State[EnumType]:
  """设置线程局部状态，并返回用于管理它的上下文管理器。

  参见 ``bool_state`` 的文档字符串。

  Args:
    name: 字符串，会被转换为小写以定义配置选项（以及
      absl flag）的名字。它会被转换为大写，以定义
      对应的 shell 环境变量。
    enum_class: enum.Enum 的子类型。
    default: enum_class 的一个实例，作为默认值。
    help: 字符串，用于填充 flag 的帮助信息，同时也作为
      所返回上下文管理器的文档字符串。
    include_in_jit_key: 布尔值，可选：是否将该状态纳入
      JIT 缓存键。
    include_in_trace_context: 布尔值，可选：是否将该状态
      纳入追踪上下文。
    extra_validator: 可选函数，用于校验该配置选项
      的值。

  Returns:
    用于控制线程局部状态取值的上下文管理器。
  """
  if not isinstance(default, enum_class):
    raise TypeError(
        f'Default value must be of type {enum_class}, got {default} '
        f"of type {getattr(type(default), '__name__', type(default))}"
    )
  name = name.lower()
  default_str = os.getenv(name.upper(), None)
  if default_str is not None:
    try:
      default = enum_class(default_str)
    except ValueError as e:
      raise ValueError(f"Invalid value \"{default_str}\" for JAX flag {name}") from e
  config._contextmanager_flags.add(name)

  def parser(new_val):
    if isinstance(new_val, str):
      return enum_class(new_val)
    if not isinstance(new_val, enum_class):
      raise TypeError(
          f'new enum value must be an instance of {enum_class}, got'
          f' {new_val} of type {type(new_val)}.'
      )
    if extra_validator is not None:
      extra_validator(new_val)
    return new_val

  s = State[EnumType](
      name,
      default,
      help,
      update_global_hook=update_global_hook,
      update_thread_local_hook=update_thread_local_hook,
      parser=parser,
      include_in_jit_key=include_in_jit_key,
      include_in_trace_context=include_in_trace_context,
  )
  config.add_option(
      name, s, 'enum_class',
      meta_args=[],
      meta_kwargs={"enum_class": enum_class, "help": help}
  )
  setattr(Config, name, property(lambda _: s.value))
  return s


def int_state(
    name: str,
    default: int,
    help: str,
    *,
    update_global_hook: Callable[[int], None] | None = None,
    update_thread_local_hook: Callable[[int | None], None] | None = None,
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
    validator: Callable[[Any], None] | None = None,
) -> State[int]:
  """设置线程局部状态，并返回用于管理它的上下文管理器。

  参见 ``bool_state`` 的文档字符串。

  Args:
    name: 字符串，会被转换为小写以定义配置选项（以及
      absl flag）的名字。它会被转换为大写，以定义
      对应的 shell 环境变量。
    default: 可选整数，默认值。
    help: 字符串，用于填充 flag 的帮助信息，同时也作为
      所返回上下文管理器的文档字符串。

  Returns:
    用于控制线程局部状态取值的上下文管理器。
  """
  if not isinstance(default, int):
    raise TypeError(f"Default value must be of type int, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")
  name = name.lower()
  default_env = os.getenv(name.upper())
  if default_env is not None:
    try:
      default = int(default_env)
    except ValueError:
      raise ValueError(f"Invalid value \"{default_env}\" for JAX flag {name}")
  config._contextmanager_flags.add(name)

  def parser(new_val):
    if new_val is not None and not isinstance(new_val, int):
      raise ValueError(f'new int config value must be None or of type int, '
                       f'got {new_val} of type {type(new_val)}')
    if new_val is not None and validator is not None:
      validator(new_val)
    return new_val

  s = State[int](name, default, help, update_global_hook,
                 update_thread_local_hook, parser,
                 include_in_jit_key=include_in_jit_key,
                 include_in_trace_context=include_in_trace_context)
  config.add_option(name, s, int, meta_args=[], meta_kwargs={"help": help})
  setattr(Config, name, property(lambda _: s.value))
  return s


def float_state(
    name: str,
    default: float,
    help: str,
    *,
    update_global_hook: Callable[[float], None] | None = None,
    update_thread_local_hook: Callable[[float | None], None] | None = None,
) -> State[float]:
  """设置线程局部状态，并返回一个用于管理它的 contextmanager。

  参见 ``bool_state`` 的文档字符串。

  Args:
    name: 字符串，会转换为小写以定义配置项（以及 absl flag）的名称。
      它会被转换为大写以定义
      对应的 shell 环境变量。
    default: 默认值。
    help: 字符串，用于填充 flag 的帮助信息，以及所返回的 context manager
      的文档字符串。

  Returns:
    一个用于控制线程局部状态值的 contextmanager。
  """
  if not isinstance(default, float):
    raise TypeError(f"Default value must be of type float, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")
  name = name.lower()
  default_env = os.getenv(name.upper())
  if default_env is not None:
    try:
      default = float(default_env)
    except ValueError:
      raise ValueError(f"Invalid value \"{default_env}\" for JAX flag {name}")
  config._contextmanager_flags.add(name)

  def parser(new_val):
    if new_val is not None and not isinstance(new_val, (float, int)):
      raise ValueError(
        f'new float config value must be None or of type float, '
        f'got {new_val} of type {type(new_val)}')
    return new_val

  s = State[float](name, default, help, update_global_hook,
                   update_thread_local_hook, parser)
  config.add_option(name, s, float, meta_args=[], meta_kwargs={"help": help})
  setattr(Config, name, property(lambda _: s.value))
  return s


def string_state(
    name: str,
    default: str,
    help: str,
    *,
    update_global_hook: Callable[[str], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
) -> State[str]:
  """设置线程局部状态，并返回一个用于管理它的 contextmanager。

  参见 ``bool_state`` 的文档字符串。

  Args:
    name: 字符串，会转换为小写以定义配置项（以及 absl flag）的名称。
      它会被转换为大写以定义
      对应的 shell 环境变量。
    default: 字符串，该配置项的默认值。
    help: 字符串，用于填充 flag 的帮助信息，以及所返回的 context manager
      的文档字符串。
    update_global_hook: 可选回调，当全局状态的值被修改或首次设置时，
      会以更新后的值调用它。
    update_thread_local_hook: 可选回调，当线程局部状态的值被修改或
      首次设置时，会以更新后的
      值调用它。

  Returns:
    一个用于控制线程局部状态值的 contextmanager。
  """
  if not isinstance(default, str):
    raise TypeError(f"Default value must be of type str, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")

  def validator(new_val):
    if not isinstance(new_val, str):
      raise TypeError('new string config value must be of type str,'
                       f' got {new_val} of type {type(new_val)}.')

  return string_or_object_state(
      name, default, help,
      update_global_hook=update_global_hook,
      update_thread_local_hook=update_thread_local_hook,
      validator=validator)


def optional_string_state(
    name: str,
    default: str | None,
    help: str,
    *,
    update_global_hook: Callable[[str], None] | None = None,
    update_thread_local_hook: Callable[[str | None], None] | None = None,
    include_in_trace_context: bool = False,
) -> State[str | None]:
  """设置线程局部状态，并返回一个用于管理它的 contextmanager。

  参见 ``bool_state`` 的文档字符串。

  Args:
    name: 字符串，会转换为小写以定义配置项（以及 absl flag）的名称。
      它会被转换为大写以定义
      对应的 shell 环境变量。
    default: 可选的字符串，该配置项的默认值。
    help: 字符串，用于填充 flag 的帮助信息，以及所返回的 context manager
      的文档字符串。
    update_global_hook: 可选回调，当全局状态的值被修改或首次设置时，
      会以更新后的值调用它。
    update_thread_local_hook: 可选回调，当线程局部状态的值被修改或
      首次设置时，会以更新后的
      值调用它。

  Returns:
    一个用于控制线程局部状态值的 contextmanager。
  """
  if default is not None and not isinstance(default, str):
    raise TypeError(f"Default value must be of type str or None, got {default} "
                    f"of type {getattr(type(default), '__name__', type(default))}")

  def validator(new_val):
    if new_val is not None and not isinstance(new_val, str):
      raise ValueError('new string config value must be None or of type str,'
                       f' got {new_val} of type {type(new_val)}.')

  return string_or_object_state(
      name, default, help,
      update_global_hook=update_global_hook,
      update_thread_local_hook=update_thread_local_hook,
      validator=validator,
      include_in_trace_context=include_in_trace_context)

def string_or_object_state(
    name: str,
    default: Any,
    help: str,
    *,
    update_global_hook: Callable[[Any], None] | None = None,
    update_thread_local_hook: Callable[[Any], None] | None = None,
    validator: Callable[[Any], None] | None = None,
    include_in_jit_key: bool = False,
    include_in_trace_context: bool = False,
) -> State[Any]:
  """设置线程局部状态，并返回一个用于管理它的 contextmanager。

  与 ``string_state`` 类似，区别在于该 context manager 可以接受任何对象，
  而不只是字符串。任何通过命令行 flag 或环境变量传入的
  值都会被当作字符串处理。

  Args:
    name: 字符串，会转换为小写以定义配置项（以及 absl flag）的名称。
      它会被转换为大写以定义
      对应的 shell 环境变量。
    default: 字符串，该配置项的默认值。
    help: 字符串，用于填充 flag 的帮助信息，以及所返回的 context manager
      的文档字符串。
    update_global_hook: 可选回调，当全局状态的值被修改或首次设置时，
      会以更新后的值调用它。
    update_thread_local_hook: 可选回调，当线程局部状态的值被修改或
      首次设置时，会以更新后的
      值调用它。
    validator: 可选回调，在每次更新时都会以新值调用它；
      如果新值无效，它应当
      抛出错误。

  Returns:
    一个用于控制线程局部状态值的 contextmanager。
  """
  name = name.lower()
  default = os.getenv(name.upper(), default)
  config._contextmanager_flags.add(name)

  def parser(new_val):
    if validator is not None:
      validator(new_val)
    return new_val

  s = State[Any](
      name, default, help, update_global_hook, update_thread_local_hook,
      parser, include_in_jit_key=include_in_jit_key,
      include_in_trace_context=include_in_trace_context)
  setattr(Config, name, property(lambda _: s.value))
  config.add_option(name, s, str, meta_args=[], meta_kwargs={"help": help})
  return s


class Flag[ValueType]:

  __slots__ = ("_name", "value", "_update_hook")

  _name: str
  value: ValueType
  _update_hook: Callable[[Any], None] | None

  def __init__(self, name: str, default: ValueType,
               update_hook: Callable[[Any], None] | None = None):
    self._name = name
    self._update_hook = update_hook
    self._set(default)

  def __bool__(self) -> NoReturn:
    raise TypeError(
        "bool() not supported for instances of type '{0}' "
        "(did you mean to use '{0}.value' instead?)".format(
            type(self).__name__))

  def _set(self, value: ValueType) -> None:
    self.value = value
    if self._update_hook is not None:
      self._update_hook(value)


def bool_flag(name, default, *args, **kwargs) -> Flag[bool]:
  update_hook = kwargs.pop("update_hook", None)
  holder = Flag(name, default, update_hook)
  config.add_option(name, holder, bool, args, kwargs)
  return holder


def int_flag(name, default, *args, **kwargs) -> Flag[int]:
  update_hook = kwargs.pop("update_hook", None)
  holder = Flag(name, default, update_hook)
  config.add_option(name, holder, int, args, kwargs)
  return holder


def float_flag(name, default, *args, **kwargs) -> Flag[float]:
  update_hook = kwargs.pop("update_hook", None)
  holder = Flag(name, default, update_hook)
  config.add_option(name, holder, float, args, kwargs)
  return holder


def string_flag(name, default, *args, **kwargs) -> Flag[str]:
  update_hook = kwargs.pop("update_hook", None)
  holder = Flag(name, default, update_hook)
  config.add_option(name, holder, str, args, kwargs)
  return holder


def enum_flag(name, default, *args, **kwargs) -> Flag[str]:
  update_hook = kwargs.pop("update_hook", None)
  holder = Flag(name, default, update_hook)
  config.add_option(name, holder, 'enum', args, kwargs)
  return holder


already_configured_with_absl = False

mesh_context_manager = config_ext.Config(
    'mesh_context_manager',
    (),
    include_in_jit_key=True,
    include_in_trace_context=True,
)
abstract_mesh_context_manager = config_ext.Config(
    'abstract_mesh_context_manager',
    None,
    include_in_jit_key=True,
    include_in_trace_context=True,
)
device_context = config_ext.Config(
    'device_context', None, include_in_jit_key=True
)
compute_on_context_manager = config_ext.Config(
    'compute_on_context_manager',
    None,
    include_in_jit_key=True,
    include_in_trace_context=True,
)
xla_metadata_context_manager = config_ext.Config(
    'xla_metadata_context_manager',
    None,
    include_in_jit_key=True,
    include_in_trace_context=True,
)
pallas_tpu_interpret_mode_context_manager = config_ext.Config(
    'pallas_tpu_interpret_mode_context_manager',
    None,
    include_in_jit_key=True,
    include_in_trace_context=True,
)


class UserContext:
  __slots__ = ["_config", "_new_value", "_prev_value"]

  def __init__(self, config, new_value):
    self._config = config
    self._new_value = new_value

  def __enter__(self):
    self._prev_value = self._config.swap_local(self._new_value)

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._config.set_local(self._prev_value)


class UserConfig:
  def __init__(self, default_value):
    self._obj = config_ext.Config(
        "user_context", default_value, include_in_jit_key=True,
        include_in_trace_context=True)

  @property
  def value(self):
    return self._obj.value

  def get_global(self):
    return self._obj.get_global()

  def set_global(self, new_value):
    return self._obj.set_global(new_value)

  def __call__(self, new_value):
    return UserContext(self._obj, new_value)


def make_user_context(default_value=None):
  """创建一个对 `jax.jit` 缓存敏感（cache sensitive）的上下文。

  如果该上下文的值发生变化，JAX 的追踪、降级与编译
  缓存都不会命中，被 jit 的函数将被重新追踪、
  重新降级并重新编译。

  新增用户上下文不是线程安全的。不要与其他 JAX API
  并发调用 make_user_context。不过，一旦用户上下文
  构造完成，使用它就是线程安全的。

  Example:

  ```
  @jax.jit
  def f(x):
    return x * 2

  my_context = jax.make_user_context(default_value=None)
  with my_context(1):
    f(1.)
  with my_context(2):
    f(1.)  # 追踪缓存未命中
  ```
  """
  return UserConfig(default_value)

jax_jit_cpp_cache_obj = make_user_context(None)

# TODO(b/214340779): 待 XLA:CPU 改进后移除该 flag。
jax2tf_associative_scan_reductions = bool_state(
    name='jax2tf_associative_scan_reductions',
    default=False,
    help=(
        'JAX has two separate lowering rules for the cumulative reduction '
        'primitives (cumsum, cumprod, cummax, cummin). On CPUs and GPUs it uses '
        'a lax.associative_scan, while for TPUs it uses the HLO ReduceWindow. '
        'The latter has a slow implementation on CPUs and GPUs. '
        'By default, jax2tf uses the TPU lowering. Set this flag to True to '
        'use the associative scan lowering usage, and only if it makes a difference '
        'for your application. '
        'See the jax2tf README.md for more details.'
    )
)

jax2tf_default_native_serialization = bool_state(
    name='jax2tf_default_native_serialization',
    default=bool_env('JAX2TF_DEFAULT_NATIVE_SERIALIZATION', True),
    help=(
        'Sets the default value of the native_serialization parameter to '
        'jax2tf.convert. Prefer using the parameter instead of the flag, '
        'the flag may be removed in the future. '
        'Starting with JAX 0.4.31 non-native serialization is deprecated.'
    )
)

jax_serialization_version = int_state(
    name='jax_serialization_version',
    default=int_env('JAX_SERIALIZATION_VERSION', 0),  # 用 0 来检测默认值。
    help=(
        'DEPRECATED: use jax_export_calling_convention_version.'
    )
)

jax_export_calling_convention_version = int_state(
    name='jax_export_calling_convention_version',
    # 注意：在更新 XlaCallModule 以支持新版本之后，至少再等一个月才提升
    # 默认的调用约定版本，这样序列化后的模块才能与已部署的
    # XlaCallModule 版本保持向前兼容。
    # XlaCallModule 的第 10 版自 2025 年 5 月 20 日起获得支持。
    default=int_env('JAX_EXPORT_CALLING_CONVENTION_VERSION', 10),
    help=(
        'The calling convention version number to use for exporting. This must be '
        'within the range of versions supported by the tf.XlaCallModule '
        'used in your deployment environment. '
        'See https://docs.jax.dev/en/latest/export/shape_poly.html#calling-convention-versions.'
    )
)

export_ignore_forward_compatibility = bool_state(
    name='jax_export_ignore_forward_compatibility',
    default=bool_env('JAX_EXPORT_IGNORE_FORWARD_COMPATIBILIY', False),
    help=(
        'Whether to ignore the forward compatibility lowering rules. '
        'See https://docs.jax.dev/en/latest/export/export.html#compatibility-guarantees-for-custom-calls.'
    )
)

export_deserialize_expired_versions = bool_state(
    name='jax_export_deserialize_expired_versions',
    default=bool_env('JAX_EXPORT_DESERIALIZE_EXPIRED_VERSIONS', False),
    help=(
        'Whether to allow deserialization of expired versions of JAX exports.'
        'If you turn this on, you may see obscure downstream errors in JAX or '
        'the compiler and runtime. Furthermore, you accept the fact that the '
        'behavior of the deserialized model may change at any time. '
        'Read carefully '
        'https://docs.jax.dev/en/latest/export/export.html#compatibility-guarantees.'
    )
)

jax_platforms = optional_string_state(
    name='jax_platforms',
    default=None,
    help=(
        'Comma-separated list of platform names specifying which platforms jax '
        'should initialize. If any of the platforms in this list are not successfully '
        'initialized, an exception will be raised and the program will be aborted. '
        'The first platform in the list will be the default platform. '
        'For example, config.jax_platforms=cpu,tpu means that CPU and TPU backends '
        'will be initialized, and the CPU backend will be used unless otherwise '
        'specified. If TPU initialization fails, it will raise an exception. '
        'By default, jax will try to initialize all available '
        'platforms and will default to GPU or TPU if available, and fallback to CPU '
        'otherwise.'
        ))

def _validate_jax_pjrt_client_create_options(new_val):
  if new_val is not None and not isinstance(new_val, (str, dict)):
      raise ValueError('new string config value must be None or of type dict'
                       f' | str, got {new_val} of type {type(new_val)}.')

jax_pjrt_client_create_options = string_or_object_state(
    name='jax_pjrt_client_create_options',
    default=None,
    help=('A set of key-value pairs in the format of "k1:v1;k2:v2" strings '
          'provided to a device platform pjrt client as extra arguments.'),
    validator=_validate_jax_pjrt_client_create_options)

enable_checks = bool_state(
    name='jax_enable_checks',
    default=False,
    help='Turn on invariant checking for JAX internals. Makes things slower.')

debug_key_reuse = bool_state(
    name='jax_debug_key_reuse',
    default=False,
    help=('Turn on experimental key reuse checking. With this configuration enabled,'
          ' typed PRNG keys (i.e. keys created with jax.random.key()) will have their'
          ' usage tracked, and incorrect reuse of a previously-used key will lead to'
          ' an error. Currently enabling this leads to a small Python overhead on'
          ' every call to a JIT-compiled function with keys as inputs or outputs.'),
    include_in_trace_context=True,
    include_in_jit_key=True)

use_hlo_logistic_lowering = optional_bool_state(
    name='jax_use_hlo_logistic_lowering',
    default=None,
    help=(
        'Uses `hlo.logistic` during lowering of logistic_p on TPUs. There are 3'
        ' valid values:\n'
        '  `None` - JAX chooses (Uses `hlo.logistic` for TPU gen >= 8 else uses '
        '           current lowering)\n'
        '  `True` - use `hlo.logistic`\n'
        '  `False` - use the current lowering of logistic_p.'
    ),
    include_in_trace_context=True,
    include_in_jit_key=True,
)

check_tracer_leaks = bool_state(
    name='jax_check_tracer_leaks',
    default=False,
    help=('Turn on checking for leaked tracers as soon as a trace completes. '
          'Enabling leak checking may have performance impacts: some caching '
          'is disabled, and other overheads may be added. Additionally, be aware '
          'that some Python debuggers can cause false positives, so it is recommended '
          'to disable any debuggers while leak checking is enabled.'))
checking_leaks = functools.partial(check_tracer_leaks, True)

check_static_indices = bool_state(
    name='jax_check_static_indices',
    default=False,
    help=('Turn on bounds checks for static indices during array indexing operations.'
          ' These will only be checked when indexing mode is PROMISE_IN_BOUNDS, which'
          ' is the default for gather-type operations.'),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

captured_constants_warn_bytes = int_state(
    name='jax_captured_constants_warn_bytes',
    default=2 * 10 ** 9,
    help=('The number of bytes of parameters that may be captured as constants '
          'before a warning is issued. Defaults to approximately 2GB. '
          'Set to -1 to disable issuing a warning.'
    )
)

captured_constants_report_frames = int_state(
    name='jax_captured_constants_report_frames',
    default=0,
    help=('The number of stack frames reported for each captured constant '
          'indicating the file and operation where the constant was captured. '
          'Set to -1 to print the complete set of frames, or 0 to disable. '
          'N.b. the report is only generated if the total amount of captured '
          'constants exceeds `jax_captured_constants_warn_bytes`, as it is expensive'
          'to generate the report.'
    )
)

raise_on_ppermute_sort_diff = bool_state(
    name='jax_raise_on_ppermute_sort_diff',
    default=True,
    help=(
        'Raises an error if ppermute axis_name are not in the same order as the'
        ' mesh axis_names because it leads to wrong answers.'
    ),
    include_in_jit_key=True,
    include_in_trace_context=True,
)


debug_leaked_clients_on_clear_backends = bool_state(
    name='jax_debug_leaked_clients_on_clear_backends',
    default=False,
    help=('Check that clear_backends actually clears the backends.'))

debug_nans = bool_state(
    name='jax_debug_nans',
    default=False,
    help=('Add nan checks to every operation. When a nan is detected on the '
          'output of a jit-compiled computation, call into the un-compiled '
          'version in an attempt to more precisely identify the operation '
          'which produced the nan.'))

debug_infs = bool_state(
    name='jax_debug_infs',
    default=False,
    help=('Add inf checks to every operation. When an inf is detected on the '
          'output of a jit-compiled computation, call into the un-compiled '
          'version in an attempt to more precisely identify the operation '
          'which produced the inf.'))

log_compiles = bool_state(
    name='jax_log_compiles',
    default=False,
    help=(
        'Log a message each time `jit` or `pmap` compiles an XLA computation.'
        ' Logging is performed with `logging`. When this option is set, the log'
        ' level is WARNING; otherwise the level is DEBUG.\n\nSee'
        ' https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html'
        ' for more details.'
    ),
)

explain_cache_misses = bool_state(
    name='jax_explain_cache_misses',
    default=False,
    help=(
        'Each time there is a miss on one of the main caches (e.g. the tracing'
        ' cache), log an explanation. Logging is performed with `logging`. When'
        ' this option is set, the log level is WARNING; otherwise the level is'
        ' DEBUG.\n\nSee'
        ' https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html'
        ' for more details.'
    ),
)

log_checkpoint_residuals = bool_state(
    name='jax_log_checkpoint_residuals',
    default=False,
    help=('Log a message every time jax.checkpoint (aka jax.remat) is '
          'partially evaluated (e.g. for autodiff), printing what residuals '
          'are saved.'))

scan3 = bool_state(
    name='jax_scan3',
    default=False,
    upgrade=True,
    help='If True, embrace the future of loops.',
    include_in_jit_key=True,
    include_in_trace_context=True,
)


remat3 = bool_state(
    name='jax_remat3',
    default=False,
    upgrade=True,
    help='If True, embrace the future of remat.',
    include_in_jit_key=True,
    include_in_trace_context=True,
)

custom_vjp3 = bool_state(
    name='jax_custom_vjp3',
    default=False,
    upgrade=True,
    help='If True, embrace the future of custom autodiff rules.',
    include_in_jit_key=True,
    include_in_trace_context=True,
)

custom_jvp3 = bool_state(
    name='jax_custom_jvp3',
    default=False,
    upgrade=True,
    help='If True, embrace the future of custom autodiff rules.',
    include_in_jit_key=True,
    include_in_trace_context=True,
)

remat_barrier_no_cotangents = bool_state(
    name='jax_remat_barrier_no_cotangents',
    default=False,
    upgrade=True,
    help=('If True, embrace the future of remat CSE-prevention barriers, '
          'which do not pin cotangents.'),
    include_in_jit_key=True,
    include_in_trace_context=True,
)


distributed_debug = bool_state(
    name='jax_distributed_debug',
    default=False,
    help=('Enable logging useful for debugging multi-process distributed '
          'computations. Logging is performed with `logging` at WARNING '
          'level.'))

random_seed_offset = int_state(
    name='jax_random_seed_offset',
    default=0,
    help=('Offset to all random seeds (e.g. argument to jax.random.key()).'),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

class LegacyPrngKeyState(enum.StrEnum):
  ALLOW = 'allow'
  WARN = 'warn'
  ERROR = 'error'

legacy_prng_key = enum_class_state(
    name='jax_legacy_prng_key',
    enum_class=LegacyPrngKeyState,
    default=LegacyPrngKeyState.ALLOW,
    help=('Specify the behavior when raw PRNG keys are passed to '
          'jax.random APIs.')
)

enable_custom_prng = bool_state(
    name='jax_enable_custom_prng',
    default=False,
    upgrade=True,
    help=('Enables an internal upgrade that allows one to define custom '
          'pseudo-random number generator implementations.'))

default_prng_impl = enum_state(
    name='jax_default_prng_impl',
    enum_values=['threefry2x32', 'threefry4x32', 'rbg', 'unsafe_rbg', 'philox2x32', 'philox4x32'],
    default='threefry2x32',
    help=('Select the default PRNG implementation, used when one is not '
          'explicitly provided at seeding time.'))

threefry_partitionable = bool_state(
    name='jax_threefry_partitionable',
    default=True,
    upgrade=True,
    help=('Enables internal threefry PRNG implementation changes that '
          'render it automatically partitionable in some cases. Without this '
          'flag, using the standard jax.random pseudo-random number generation '
          'may result in extraneous communication and/or redundant distributed '
          'computation. With this flag, the communication overheads disappear '
          'in some cases.'),
    include_in_jit_key=True,
    include_in_trace_context=True)

threefry_gpu_kernel_lowering = bool_state(
    name='jax_threefry_gpu_kernel_lowering',
    default=False,
    help=('On GPU, lower threefry PRNG operations to a kernel implementation. '
          'This makes compile times faster at a potential runtime memory '
          'cost.'),
    include_in_jit_key=True,
    include_in_trace_context=True)

use_direct_linearize = bool_state(
    name='jax_use_direct_linearize',
    default=True,
    help=('Use direct linearization instead JVP followed by partial eval'),
    include_in_jit_key=True,
    include_in_trace_context=True)

use_simplified_jaxpr_constants = bool_state(
    name='jax_use_simplified_jaxpr_constants',
    default=False,
    help=('Enable a simplification of the handling of closed-over constants '
          'in Jaxpr. The value `True` enables the new behavior. '
          'This flag will exist only briefly, while we transition '
          'users. See https://docs.jax.dev/en/latest/internals/constants.html.'
          'DO NOT RELY ON THIS FLAG.'),
    include_in_jit_key=True,
    include_in_trace_context=True)

embedded_constants_max_bytes = int_state(
    name='jax_embedded_constants_max_bytes',
    default=32,
    help=('Maximum size in bytes of a constant that is allowed to be '
          'embedded in the lowered HLO. Constants larger than this '
          'are hoisted as additional arguments to the executable. '
          'See https://docs.jax.dev/en/latest/internals/constants.html.'),
    include_in_jit_key=True,
    include_in_trace_context=True)

# 这个配置是临时性的，本应被移除，因为问题出在用户侧。
# 如果用户不希望大小为 1 的 mesh 轴名出现在 sharding 与 vma
# 位（位于 ShapedArray 上）中，那么他们传给 set_mesh 的 mesh
# 就根本不应该包含这些轴。
remove_size_one_mesh_axis_from_type = bool_state(
    name='jax_remove_size_one_mesh_axis_from_type',
    default=False,
    help="Removes mesh axes of size 1 from ShapedArray.sharding and vma",
    include_in_jit_key=True,
    include_in_trace_context=True)

# TODO 想办法让人们不要使用这个，它属于内部实现...
_check_vma = bool_state(
    name='check_vma',
    default=False,
    help='internal implementation detail of shard_map, DO NOT USE',
    include_in_jit_key=True,
    include_in_trace_context=True)

softmax_custom_jvp = bool_state(
    name='jax_softmax_custom_jvp',
    default=False,
    upgrade=True,
    help=('Use a new custom_jvp rule for jax.nn.softmax. The new rule should '
          'improve memory usage and stability. Set True to use new '
          'behavior. See https://github.com/jax-ml/jax/pull/15677'),
    include_in_jit_key=True,
    include_in_trace_context=True)

raise_persistent_cache_errors = bool_state(
    name='jax_raise_persistent_cache_errors',
    default=False,
    help=('If true, exceptions raised when reading or writing to the '
          'persistent compilation cache will be allowed through, halting '
          'program execution if not manually caught. If false, exceptions are '
          'caught and raised as warnings, allowing program execution to '
          'continue. Defaults to false so cache bugs or intermittent issues '
          'are non-fatal.'))

persistent_cache_min_compile_time_secs = float_state(
    name='jax_persistent_cache_min_compile_time_secs',
    default=1.,
    help=('The minimum compile time of a computation to be written to the '
          'persistent compilation cache. This threshold can be raised to '
          'decrease the number of entries written to the cache.'))

persistent_cache_min_entry_size_bytes = int_state(
    name='jax_persistent_cache_min_entry_size_bytes',
    default=0,
    help=('The minimum size (in bytes) of an entry that will be cached in the '
          'persistent compilation cache: '
          '* -1: disable the size restriction and prevent overrides. '
          '* Leave at default (0) to allow for overrides. The override will '
          '  typically ensure that the minimum size is optimal for the '
          '  filesystem being used for the cache. '
          '* > 0: the actual minimum size desired; no overrides.'))

# TODO: 把默认值改为 all
persistent_cache_enable_xla_caches = optional_string_state(
    name='jax_persistent_cache_enable_xla_caches',
    default='xla_gpu_per_fusion_autotune_cache_dir',
    help=('When the persistent cache is enabled, additional XLA caching will '
          'also be enabled automatically. This option can be used to configure'
          'which XLA caching methods will be enabled.'),
)

compilation_cache_include_metadata_in_key = bool_state(
    name='jax_compilation_cache_include_metadata_in_key',
    default=False,
    help=(
        'Include metadata, such as file names and line numbers, in the'
        ' compilation cache key. If false, the cache will still get hits even'
        ' if functions or files are moved, etc. However, it means that'
        ' executables loaded from the cache may have stale metadata, which'
        ' may show up in, e.g., profiles.'
    ),
)

hlo_source_file_canonicalization_regex = optional_string_state(
    name='jax_hlo_source_file_canonicalization_regex',
    default=None,
    help=('Used to canonicalize the source_path metadata of HLO instructions '
          'by removing the given regex. If set, re.sub() is called on each '
          'source_file with the given regex, and all matches are removed.'),
    include_in_trace_context=True)

source_url_schema = optional_string_state(
    name='jax_source_url_schema',
    default=None,
    help=('URL format string used to generate links to source files in the HTML'
          ' jaxpr dumps. Can contain `{file}` and `{line}` placeholders.'))

include_full_tracebacks_in_locations = bool_state(
    name='jax_include_full_tracebacks_in_locations',
    default=True,
    help=(
        'Include Python tracebacks in MLIR locations in IR emitted by JAX.'
    ),
)

traceback_in_locations_limit = int_state(
    name='jax_traceback_in_locations_limit',
    default=100,
    help=(
        'Limit the number of frames at the Python traceback frames included in '
        'MLIR locations. If set to the negative value, traceback will not be '
        'limited.'
    ),
)

share_binary_between_hosts = bool_state(
    name='jax_share_binary_between_hosts',
    default=False,
    help=(
        'If set to True, the compiled module will be shared between hosts '
        'directly.'
    ),
)

share_binary_between_hosts_timeout_ms = int_state(
    name='jax_share_binary_between_hosts_timeout_ms',
    default=20 * 60 * 1000,
    help='Timeout for the compiled module share.',
)

enable_pgle = bool_state(
    name='jax_enable_pgle',
    default=False,
    help=(
      'If set to True and the property jax_pgle_profiling_runs is set to '
      'greater than 0, the modules will be recompiled after running specified '
      'number times with collected data provided to the profile guided latency '
      'estimator.'
    ),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

pgle_profiling_runs = int_state(
    name='jax_pgle_profiling_runs',
    default=3,
    help=(
        'Amount of times module should be profiled before recompilation when '
        'PGLE is used.'
    ),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

pgle_aggregation_percentile = int_state(
    name='jax_pgle_aggregation_percentile',
    default=90,
    help='Percentile used to aggregate performance data between devices when '
         'PGLE is used.',
)

enable_compilation_cache = bool_state(
    name='jax_enable_compilation_cache',
    default=True,
    help=('If set to False, the compilation cache will be disabled regardless '
          'of whether set_cache_dir() was called. If set to True, the '
          'path could be set to a default value or via a call to '
          'set_cache_dir().'),
)

compilation_cache_dir = optional_string_state(
    name='jax_compilation_cache_dir',
    default=None,
    help=('Path for the cache. '
          'Precedence: '
          '1. A call to compilation_cache.set_cache_dir(). '
          '2. The value of this flag set in the command line or by default.'),
)

compilation_cache_check_contents = bool_state(
    name='jax_compilation_cache_check_contents',
    default=False,
    help=(
        'When the compilation cache is enabled, check that the value '
        'found in the disk cache matches the result of a fresh compilation. '
        'This check is performed only the first time a key is encountered '
        'in a process.'
    ),
)

compilation_cache_expect_pgle = bool_state(
    name='jax_compilation_cache_expect_pgle',
    default=False,
    help=('If set to True, compilation cache entries that were compiled with '
          'profile data (i.e. PGLE was enabled and the requisite number of '
          'executions were profiled) will be preferentially loaded, even if '
          'PGLE is not currently enabled. A warning will be printed when no '
          'preferred cache entry is found.')
)

compilation_cache_max_size = int_state(
    name='jax_compilation_cache_max_size',
    default=-1,
    help=('The maximum size (in bytes) allowed for the persistent compilation '
          'cache. When set, the least recently accessed cache entry(s) '
          'will be deleted once the total cache directory size '
          'exceeds the specified limit. '
          'Caching will be disabled if this value is set to 0. A '
          'special value of -1 indicates no limit, allowing the cache '
          'size to grow indefinitely.'),
)

remove_custom_partitioning_ptr_from_cache_key = bool_state(
    name='jax_remove_custom_partitioning_ptr_from_cache_key',
    default=False,
    help=('If set to True, remove the custom partitioning pointer '
          'present in the precompiled stableHLO before hashing  '
          'during cache key computation. This is a potentially '
          'unsafe flag to set and only users who are sure of '
          'what they are trying to achieve should set it.'),
)


class ExplicitX64Mode(enum.IntEnum):
  WARN = enum.auto()
  ERROR = enum.auto()
  ALLOW = enum.auto()

  @classmethod
  def _missing_(cls, value: object) -> ExplicitX64Mode | None:
    if value == "warn":
      return cls.WARN
    if value == "error":
      return cls.ERROR
    if value == "allow":
      return cls.ALLOW
    return None


explicit_x64_dtypes = enum_class_state(
    name='jax_explicit_x64_dtypes',
    enum_class=ExplicitX64Mode,
    default=ExplicitX64Mode.WARN,
    help=(
        'If set to ALLOW, explicit specification of 64-bit types will be '
        'respected even if enable_x64 is false. If set to WARN, a warning will '
        'be issued, and if set to ERROR, an error will be raised.'
    ),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

class NumpyDtypePromotion(enum.StrEnum):
  STANDARD = 'standard'
  STRICT = 'strict'

numpy_dtype_promotion = enum_class_state(
    name='jax_numpy_dtype_promotion',
    enum_class=NumpyDtypePromotion,
    default=NumpyDtypePromotion.STANDARD,
    help=('Specify the rules used for implicit type promotion in operations '
          'between arrays. Options are "standard" or "strict"; in strict-mode, '
          'binary operations between arrays of differing strongly-specified '
          'dtypes will result in an error.'),
    include_in_jit_key=True,
    include_in_trace_context=True)

disallow_mesh_context_manager = bool_state(
    name='jax_disallow_mesh_context_manager',
    default=False,
    help=(
        'If set to True, trying to use a mesh as a context manager will'
        ' result in a RuntimeError.'
    ),
)

# TODO(ayx): 等我们有了用户级的扩展机制、可以添加 jit 缓存敏感的上下文
# 之后，就把这 3 个 flag 从 config 中移出去。
error_checking_behavior_nan = enum_state(
    name='jax_error_checking_behavior_nan',
    enum_values=['ignore', 'raise'],
    default='ignore',
    help=(
        'Specify the behavior when a NaN is encountered. Options are "ignore"'
        ' or "raise".'
    ),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

error_checking_behavior_divide = enum_state(
    name='jax_error_checking_behavior_divide',
    enum_values=['ignore', 'raise'],
    default='ignore',
    help=(
        'Specify the behavior when a divide by zero is encountered. Options are'
        ' "ignore" or "raise".'
    ),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

error_checking_behavior_oob = enum_state(
    name='jax_error_checking_behavior_oob',
    enum_values=['ignore', 'raise'],
    default='ignore',
    help=(
        'Specify the behavior when an out of bounds access is encountered.'
        ' Options are "ignore" or "raise".'
    ),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

enable_x64 = bool_state(
    name='jax_enable_x64',
    default=False,
    help='Enable 64-bit types to be used',
    include_in_jit_key=True,
    include_in_trace_context=True)

jax_jit.set_enable_x64_state(enable_x64)

# TODO(phawkins): 修好 FLAGS.x64_enabled 的使用者后移除。
config._contextmanager_flags.remove('jax_enable_x64')

setattr(Config, "x64_enabled", property(lambda _: enable_x64.value))

def _validate_default_device(val):
  if (val is not None and
      not isinstance(val, xla_client.Device) and
      val not in ['cpu', 'gpu', 'tpu']):
    # TODO(skyewm): 这是针对非 PJRT Device 类型的变通方案。等所有 JAX 后端
    # 都改用统一的 C++ 设备接口后即可移除。
    if 'Device' in str(type(val)):
      logger.info(
          'Allowing non-`xla_client.Device` default device: %s, type: %s',
          repr(val), type(val))
      return
    raise ValueError('jax.default_device must be passed either a Device object (e.g. '
                     f"`jax.devices('cpu')[0]`) or a platform name string like 'cpu' or 'gpu'"
                     f", got: {val!r}")


default_device = string_or_object_state(
    name='jax_default_device',
    default=None,
    help=(
        'Configure the default device for JAX operations. Set to a Device '
        'object (e.g. ``jax.devices("cpu")[0]``) to use that Device as the '
        'default device for JAX operations and jit\'d function calls (there is '
        'no effect on multi-device computations, e.g. pmapped function calls). '
        'Set to None to use the system default device.'),
    validator=_validate_default_device,
    include_in_jit_key=True,
    include_in_trace_context=True)

disable_jit = bool_state(
    name='jax_disable_jit',
    default=False,
    help=('Disable JIT compilation and just call original Python.'),
    include_in_trace_context=True)

jax_jit.set_disable_jit_state(disable_jit)

numpy_rank_promotion = enum_state(
    name='jax_numpy_rank_promotion',
    enum_values=['allow', 'warn', 'raise'],
    default='allow',
    help=('Control NumPy-style automatic rank promotion broadcasting '
          '("allow", "warn", or "raise").'),
    include_in_jit_key=True,
    include_in_trace_context=True)

auto_pcast = bool_state(
    name='jax_auto_pcast',
    default=True,  # TODO(yashkatariya): False
    help=('If True, automatically insert `pvary` to match VMAs on simple ops.'),
    include_in_jit_key=True,
    include_in_trace_context=True)

default_matmul_precision = optional_enum_state(
    name='jax_default_matmul_precision',
    enum_values=[
        # 旧版精度 API 取值
        'default', 'high', 'highest', 'bfloat16', 'tensorfloat32', 'float32',
        # dot 算法预设
        'ANY_F8_ANY_F8_F32', 'ANY_F8_ANY_F8_F32_FAST_ACCUM', 'ANY_F8_ANY_F8_ANY',
        'ANY_F8_ANY_F8_ANY_FAST_ACCUM', 'F16_F16_F16', 'F16_F16_F32',
        'BF16_BF16_BF16', 'BF16_BF16_F32', 'BF16_BF16_F32_X3',
        'BF16_BF16_F32_X6', 'BF16_BF16_F32_X9', 'TF32_TF32_F32',
        'TF32_TF32_F32_X3', 'F32_F32_F32', 'F64_F64_F64',
    ],
    default=None,
    help=('Control the default matmul and conv precision for 32bit inputs.\n\n'

          'Some platforms, like TPU, offer configurable precision levels for '
          'matrix multiplication and convolution computations, trading off '
          'accuracy for speed. The precision can be controlled for each '
          'operation; for example, see the :func:`jax.lax.conv_general_dilated` '
          'and :func:`jax.lax.dot` docstrings. But it can be useful to control '
          'the default behavior obtained when an operation is not given a '
          'specific precision.\n\n'

          'This option can be used to control the default precision '
          'level for computations involved in matrix multiplication and '
          'convolution on 32bit inputs. The levels roughly describe the '
          "precision at which scalar products are computed. The 'bfloat16' "
          "option is the fastest and least precise; 'float32' is similar to "
          "full float32 precision; 'tensorfloat32' is intermediate.\n\n"

          'This parameter can also be used to specify an accumulation '
          '"algorithm" for functions that perform matrix multiplications, like '
          ':func:`jax.lax.dot`. To specify an algorithm, set this option to '
          'the name of a :class:`~jax.lax.DotAlgorithmPreset`.\n\n'),
    include_in_jit_key=True,
    include_in_trace_context=True)

allow_f16_reductions = bool_state(
    name='jax_allow_f16_reductions',
    default=True,
    help=('If False, `reduce_sum` on `f16` or `bf16` inputs will raise an error.'
          'Defaults to True.'),
    include_in_jit_key=True,
    include_in_trace_context=True)


traceback_filtering = enum_state(
    name = 'jax_traceback_filtering',
    enum_values=["off", "tracebackhide", "remove_frames", "quiet_remove_frames",
                 "auto"],
    default="auto",
    help="Controls how JAX filters internal frames out of tracebacks. Valid values are:\n"
         "- ``off``: disables traceback filtering.\n"
         "- ``auto``: use ``tracebackhide`` if running under a sufficiently "
         "new IPython, or ``remove_frames`` otherwise.\n"
         "- ``tracebackhide``: adds ``__tracebackhide__`` annotations to "
         "hidden stack frames, which some traceback printers support.\n"
         "- ``remove_frames``: removes hidden frames from tracebacks, and adds "
         "the unfiltered traceback as a ``__cause__`` of the exception.\n"
         "- ``quiet_remove_frames``: removes hidden frames from tracebacks, and adds "
         "a brief message (to the ``__cause__`` of the exception) describing that this has "
         "happened.\n\n")

# TODO(rdyro): 等我们始终启用 emit_pipeline 原语后移除。
use_emit_pipeline_primitive = bool_state(
    name = 'jax_use_emit_pipeline_primitive',
    default=False,
    help="Controls whether to use the emit_pipeline primitive.",
    include_in_jit_key=True,
    include_in_trace_context=True)


# 这个 flag 供内部使用。
# TODO(tianjianlu): 等我们始终启用 cusparse 降级后移除。
# TODO(b/262050896): bug 修复后设为 true
bcoo_cusparse_lowering = bool_state(
    name='jax_bcoo_cusparse_lowering',
    default=False,
    help=('Enables lowering BCOO ops to cuSparse.'))

# 这是为了与 equinox 等实现保持无栈（stackless）向后兼容
eager_constant_folding = bool_state(
    name='eager_constant_folding',
    default=False,
    help=('Attempt constant folding during staging.'),
    include_in_jit_key=True,
    include_in_trace_context=True)

enable_remat_opt_pass = bool_state(
    name='jax_compiler_enable_remat_pass',
    default=True,
    help=('Config to enable / disable the rematerialization HLO pass. '
          'Useful to allow XLA to automatically trade off memory and '
          'compute when encountering OOM errors. However, you are '
          'likely to get better results manually with jax.checkpoint'))

no_tracing = bool_state(
    name='jax_no_tracing',
    default=False,
    help='Disallow tracing for JIT compilation.')

no_execution = bool_state(
    name='jax_no_execution',
    default=False,
    help='Disallow JAX executions.',
    include_in_jit_key=True,
    include_in_trace_context=True)

disable_vmap_shmap_error = bool_state(
    name='jax_disable_vmap_shmap_error',
    default=False,
    upgrade=False,
    help='Temporary workaround to disable an error check in vmap-of-shmap.')

mutable_array_checks = bool_state(
    name='jax_mutable_array_checks',
    default=True,
    upgrade=True,
    help='Enable error checks for mutable arrays that rule out aliasing.',
    include_in_trace_context=True)

# TODO(mattjj, yashkatariya): 等我们落地 box plumbing 后移除
disable_bwd_checks = bool_state(
    name='jax_disable_bwd_checks',
    default=False,
    upgrade=True,
    help='Disables all bwd pass checks')

use_rgv3 = bool_state(
    name='jax_use_rgv3',
    default=True,
    help=(
        'Whether to use StableHLO RGV3 (mesh-axes based replica groups) during'
        ' shard_map lowering.'
    ),
)

xla_runtime_errors = bool_state(
    name='jax_experimental_unsafe_xla_runtime_errors',
    default=False,
    help=('Enable XLA runtime errors for jax.experimental.checkify.checks '
          'on CPU and GPU. These errors are async, might get lost and are not '
          'very readable. But, they crash the computation and enable you '
          'to write jittable checks without needing to checkify. Does not '
          'work under pmap/pjit.')
)

jax_xla_profile_version = int_state(
    name='jax_xla_profile_version',
    default=0,
    help=(
        'Optional profile version for XLA compilation. This is meaningful '
        'only when XLA is configured to support the remote compilation '
        'profile feature.'),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

@contextlib.contextmanager
def explicit_device_put_scope() -> Generator[None]:
  """表示当前上下文是一次显式的 device_put*() 调用。"""
  state = guard_lib.thread_local_state()
  prev = state.explicit_device_put
  state.explicit_device_put = True
  try:
    yield
  finally:
    state.explicit_device_put = prev

@contextlib.contextmanager
def explicit_device_get_scope() -> Generator[None]:
  """表示当前上下文是一次显式的 device_get() 调用。"""
  state = guard_lib.thread_local_state()
  prev = state.explicit_device_get
  state.explicit_device_get = True
  try:
    yield
  finally:
    state.explicit_device_get = prev

def _update_transfer_guard(state, key, val):
  """在 `guard_lib` 中应用传输防护等级。"""
  if val is None:
    setattr(state, key, None)
  elif val == 'allow':
    setattr(state, key, guard_lib.TransferGuardLevel.ALLOW)
  elif val == 'log':
    setattr(state, key, guard_lib.TransferGuardLevel.LOG)
  elif val == 'disallow':
    setattr(state, key, guard_lib.TransferGuardLevel.DISALLOW)
  elif val == 'log_explicit':
    setattr(state, key, guard_lib.TransferGuardLevel.LOG_EXPLICIT)
  elif val == 'disallow_explicit':
    setattr(state, key, guard_lib.TransferGuardLevel.DISALLOW_EXPLICIT)
  else:
    assert False, f'Invalid transfer guard level {val}'

transfer_guard_host_to_device = optional_enum_state(
    name='jax_transfer_guard_host_to_device',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # 默认值由 guard_lib 应用。这里用 None，以免意外
    # 覆盖 --jax_transfer_guard。
    default=None,
    help=('Select the transfer guard level for host-to-device transfers. '
          'Default is "allow".'),
    update_global_hook=lambda val: _update_transfer_guard(
        guard_lib.global_state(), 'host_to_device', val),
    update_thread_local_hook=lambda val: _update_transfer_guard(
        guard_lib.thread_local_state(), 'host_to_device', val))

transfer_guard_device_to_device = optional_enum_state(
    name='jax_transfer_guard_device_to_device',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # 默认值由 guard_lib 应用。这里用 None，以免意外
    # 覆盖 --jax_transfer_guard。
    default=None,
    help=('Select the transfer guard level for device-to-device transfers. '
          'Default is "allow".'),
    update_global_hook=lambda val: _update_transfer_guard(
        guard_lib.global_state(), 'device_to_device', val),
    update_thread_local_hook=lambda val: _update_transfer_guard(
        guard_lib.thread_local_state(), 'device_to_device', val))

transfer_guard_device_to_host = optional_enum_state(
    name='jax_transfer_guard_device_to_host',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # 默认值由 guard_lib 应用。这里用 None，以免
    # 意外覆盖 --jax_transfer_guard。
    default=None,
    help=('Select the transfer guard level for device-to-host transfers. '
          'Default is "allow".'),
    update_global_hook=lambda val: _update_transfer_guard(
        guard_lib.global_state(), 'device_to_host', val
    ),
    update_thread_local_hook=lambda val: _update_transfer_guard(
        guard_lib.thread_local_state(), 'device_to_host', val))

def _update_all_transfer_guard_global(val):
  for name in ('jax_transfer_guard_host_to_device',
               'jax_transfer_guard_device_to_device',
               'jax_transfer_guard_device_to_host'):
    config.update(name, val)

_transfer_guard = optional_enum_state(
    name='jax_transfer_guard',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # 默认值由 guard_lib 应用。这里用 None，以免意外
    # 覆盖 --jax_transfer_guard_*。
    default=None,
    help=('Select the transfer guard level for all transfers. This option is '
          'set-only; the transfer guard level for a specific direction should '
          'be read using the per-transfer direction option. '
          'Default is "allow".'),
    update_global_hook=_update_all_transfer_guard_global)

@contextlib.contextmanager
def transfer_guard(new_val: str) -> Generator[None]:
  """用于控制所有传输的传输防护等级的上下文管理器。

  更多信息请见
  https://docs.jax.dev/en/latest/transfer_guard.html

  Args:
    new_val: 所有传输的新线程局部传输防护等级。

  Yields:
    None.
  """
  with contextlib.ExitStack() as stack:
    stack.enter_context(transfer_guard_host_to_device(new_val))
    stack.enter_context(transfer_guard_device_to_device(new_val))
    stack.enter_context(transfer_guard_device_to_host(new_val))
    stack.enter_context(_transfer_guard(new_val))
    yield


def _update_garbage_collection_guard(state, key, val):
  """在 `guard_lib` 中应用传输防护等级。"""
  if val is None:
    setattr(state, key, None)
  elif val == 'allow':
    setattr(state, key, guard_lib.GarbageCollectionGuardLevel.ALLOW)
  elif val == 'log':
    setattr(state, key, guard_lib.GarbageCollectionGuardLevel.LOG)
  elif val == 'fatal':
    setattr(state, key, guard_lib.GarbageCollectionGuardLevel.FATAL)
  else:
    assert False, f'Invalid garbage collection guard level {val}'

array_garbage_collection_guard = optional_enum_state(
    name='jax_array_garbage_collection_guard',
    enum_values=['allow', 'log', 'fatal'],
    # 默认值由 guard_lib 应用。
    default=None,
    help=(
        'Select garbage collection guard level for ``jax.Array`` objects.\n\n'
        'This option can be used to control what happens when a ``jax.Array``'
        ' object is garbage collected. It is desirable for ``jax.Array``'
        ' objects to be freed by Python reference counting rather than garbage'
        ' collection in order to avoid device memory being held by the arrays'
        ' until garbage collection occurs.\n\n'
        'Valid values are:\n\n'
        '* ``allow``: do not log garbage collection of ``jax.Array`` objects.\n'
        '* ``log``: log an error when a ``jax.Array`` is garbage collected.\n'
        '* ``fatal``: fatal error if a ``jax.Array`` is garbage collected.\n\n'
        'Default is ``allow``. Note that not all cycles may be detected.'
    ),
    update_global_hook=lambda val: _update_garbage_collection_guard(
        guard_lib.global_state(), 'garbage_collect_array', val
    ),
    update_thread_local_hook=lambda val: _update_garbage_collection_guard(
        guard_lib.thread_local_state(), 'garbage_collect_array', val
    ),
)

thread_guard = bool_state(
    name='jax_thread_guard',
    default=False,
    help=(
        'If True, an error will be raised at runtime if a multi-process JAX '
        'operation is called from a thread other than the one in which the '
        'thread guard was set. This is useful for detecting cases where '
        'threads may schedule operations in different orders in different '
        'processes, leading to non-deterministic crashes.'
    ),
    update_thread_local_hook=(
        # 若该状态为 None，则把它设为 False。
        lambda val: guard_lib.update_thread_guard_global_state(val or False)),
)

class RuntimeTracebackMode(enum.StrEnum):
  OFF = 'off'
  ON = 'on'
  FULL = 'full'

  @classmethod
  def _missing_(cls, value):
    if isinstance(value, str):
      try:
        return cls[value.upper()]
      except KeyError:
        pass
    return None

  def as_cpp_enum(self):
    return getattr(_jax.RuntimeTracebackMode, self.name)

send_traceback_to_runtime = enum_class_state(
    name='jax_send_traceback_to_runtime',
    enum_class=RuntimeTracebackMode,
    default=RuntimeTracebackMode.OFF,
    help=(
        'Controls the level of Python traceback information sent to the'
        ' runtime at dispatch time:\n- "OFF": (default) No Python traceback'
        ' information is sent.\n- "ON": Only the most recent user frame call'
        ' location is sent.\n- "FULL": The full Python traceback of the call'
        ' location is sent. This has a high fixed cost on the dispatch path'
        ' and should be used only for debugging.'
    ),
    update_global_hook=lambda val: _jax.set_send_traceback_to_runtime_global(
        val.as_cpp_enum() if val is not None else _jax.RuntimeTracebackMode.OFF),
    update_thread_local_hook=lambda val: _jax.set_send_traceback_to_runtime_thread_local(
        val.as_cpp_enum() if val is not None else None),
)

# 不要定义上下文管理器，因为这里不是线程安全的。
string_state(
    name='jax_debug_log_modules',
    default='',
    help=('Comma-separated list of module names (e.g. "jax" or '
          '"jax._src.xla_bridge,jax._src.dispatch") to enable debug logging '
          'for.'),
    update_global_hook=logging_config.update_debug_log_modules)

# 不要定义上下文管理器，因为这里不是线程安全的。
optional_enum_state(
    name='jax_logging_level',
    enum_values=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    default=logging.getLevelName(logging.getLogger("jax").level),
    help=('Set the corresponding logging level on all jax loggers. Only string'
          ' values from ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR",'
          ' "CRITICAL"] are accepted. If None, the logging level will not be'
          ' set. Includes C++ logging.'),
    update_global_hook=lambda logging_level: \
      logging_config.update_logging_level_global(logging_level=logging_level)
)


use_shardy_partitioner = bool_state(
    name='jax_use_shardy_partitioner',
    default=True,
    upgrade=True,
    help=(
        'Whether to lower to Shardy. See the migration guide for more '
        'information: https://docs.jax.dev/en/latest/shardy_jax_migration.html.'
    ),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

use_cpp_shard_args = bool_state(
    name='jax_use_cpp_shard_args',
    default=False,
    help='Whether to use C++ implementation for sharding arguments.',
)

gpu_use_magma = enum_state(
    name='jax_use_magma',
    enum_values=['off', 'on', 'auto'],
    default='auto',
    help=(
        'Enable experimental support for MAGMA-backed lax.linalg.eig on GPU. '
        'See the documentation for lax.linalg.eig for more details about how '
        'to use this feature.'
    ),
)

optimization_level = enum_state(
    name='jax_optimization_level',
    enum_values=[
        'UNKNOWN',
        'O0',
        'O1',
        'O2',
        'O3',
    ],
    default='UNKNOWN',
    help='The degree to which the compiler should optimize for execution time',
    include_in_jit_key=True
)

memory_fitting_level = enum_state(
    name='jax_memory_fitting_level',
    enum_values=[
        'UNKNOWN',
        'O0',
        'O1',
        'O2',
        'O3',
    ],
    default='O2',
    help=(
        'The degree to which the compiler should attempt to make the program'
        ' fit in memory'
    ),
    include_in_jit_key=True,
)

DEFAULT_CPU_COLLECTIVES_IMPL = "gloo"

cpu_collectives_implementation = optional_enum_state(
    name='jax_cpu_collectives_implementation',
    enum_values=["gloo", "mpi", "megascale"],
    default=DEFAULT_CPU_COLLECTIVES_IMPL,
    help=(
        "Cross-process collective implementation used on CPU. Must be one of "
        '("gloo", "mpi")'),
)

use_high_dynamic_range_gumbel = bool_state(
    name='jax_high_dynamic_range_gumbel',
    default=False,
    help='If True, gumbel noise draws two samples to cover low probability '
         'events with more precision.',
    include_in_trace_context=True,
)

jax_dump_ir_to = string_flag(
    name='jax_dump_ir_to',
    default=os.getenv('JAX_DUMP_IR_TO', ''),
    help="Path to which IR(s) emitted by JAX should be dumped as text files."
         "If omitted, JAX will not dump any IR. "
         "Supports the special value 'sponge' to pick the path from the "
         "environment variable TEST_UNDECLARED_OUTPUTS_DIR. See "
         "jax_dump_ir_modes for options governing what is dumped.")

jax_include_debug_info_in_dumps = bool_flag(
    name='jax_include_debug_info_in_dumps',
    default=bool_env('JAX_INCLUDE_DEBUG_INFO_IN_DUMPS', True),
    help='Determine whether or not to keep debug symbols and location '
        'information when dumping IR code. By default, debug information will '
        'be preserved in the IR dump. To avoid exposing source code and '
        'potentially sensitive information, set to false ')

# TODO(dsuo): 把它改造成取值为列表的 flag。
jax_dump_ir_modes = string_flag(
    name="jax_dump_ir_modes",
    default=os.getenv("JAX_DUMP_IR_MODES", "stablehlo"),
    help="Comma-delimited modes in which to dump IR. Can be 'stablehlo' (the "
         "default), 'jaxpr', 'jaxpr_html', or 'eqn_count_pprof' for "
         "jaxpr equation count pprof profile.")

jax_ragged_dot_use_ragged_dot_instruction = bool_state(
    name='jax_ragged_dot_use_ragged_dot_instruction',
    default=True,
    help=(
        '(TPU only) If True, use chlo.ragged_dot instruction for ragged_dot()'
        ' lowering. Otherwise, rely on the rollout logic in lowering rule for'
        ' ragged_dot_general_p.'
    ),
)

jax_ragged_dot_use_gpu_pallas_triton_lowering = bool_state(
    name='jax_ragged_dot_use_gpu_pallas_triton_lowering',
    default=False,
    help=(
        '(GPU only) If True, use Pallas Triton lowering for ragged_dot()'
        ' lowering. Otherwise, rely on the default lowering for'
        ' ragged_dot_general_p.'
    ),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

jax_pallas_verbose_errors = bool_flag(
    'jax_pallas_verbose_errors',
    default=bool_env('JAX_PALLAS_VERBOSE_ERRORS', False),
    help='If True, print verbose error messages for Pallas kernels.',
)

jax_pallas_enable_debug_checks = bool_state(
    name='jax_pallas_enable_debug_checks',
    default=False,
    help=(
        'If set, ``pl.debug_check`` calls are checked at runtime. Otherwise,'
        ' they are a noop.'
    ),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

jax_pallas_use_mosaic_gpu = bool_state(
    name='jax_pallas_use_mosaic_gpu',
    default=bool_env('JAX_PALLAS_USE_MOSAIC_GPU', True),
    help=(
        'If True, lower Pallas kernels to the experimental Mosaic GPU'
        ' dialect, instead of Triton IR.'
    ),
    include_in_jit_key=True,
    include_in_trace_context=True,
)

jax_mosaic_allow_hlo = bool_state(
    name='jax_mosaic_allow_hlo',
    default=False,
    help='Allow hlo dialects in Mosaic',
)

jax_pallas_poison_buffers = bool_state(
    name="jax_pallas_poison_buffers",
    default=False,
    help=(
        "If set, scratch buffers allocated by Pallas (e.g., in run_scoped)"
        " are initialized with poison values (NaN for floats) at allocation time."
    ),
)

jax_pallas_auto_assign_collective_ids = enum_state(
    name='jax_pallas_auto_assign_collective_ids',
    enum_values=['no', 'yes', 'override'],
    default='no',
    help=(
        'Auto-assign or override the collective ids in pallas_call with'
        ' auto-assigned ones, based on the serialized kernel (module) hash.'
    ),
    include_in_jit_key=True,
)
