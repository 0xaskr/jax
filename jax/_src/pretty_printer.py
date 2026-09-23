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

# 文件职责：为 JAX 提供 Wadler-Lindig 风格的文档美化打印（pretty printing）组合子。
# 这些原语（`text`、`concat`、`brk`、`group`、`nest`、`color`、`source_map` 等）
# 是打印 jaxpr、HLO 等 IR 的底层设施：调用方先构造 `Doc` 文档树，再用 `format`
# 以指定宽度渲染为字符串，必要时附加 ANSI 颜色和“输出区域到来源”的映射。
# 真正的排版算法由 C++ 扩展 `_pretty_printer` 实现，这里只是它的 Python 包装层，
# 并通过 `use_cpp_class` / `use_cpp_method` 让 `Doc` 与 C++ 类型互操作。
#
# Wadler-Lindig 美化打印器。
#
# 参考文献：
# Wadler, P., 1998. A prettier printer. Journal of Functional Programming,
# pp.223-244.
#
# Lindig, C. 2000. Strictly Pretty.
# https://lindig.github.io/papers/strictly-pretty-2000.pdf
#
# Hafiz, A. 2021. Strictly Annotated: A Pretty-Printer With Support for
# Annotations. https://ayazhafiz.com/articles/21/strictly-annotated
#

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import sys
from typing import Any

from jax._src import config
from jax._src.lib import _pretty_printer as _pretty_printer
from jax._src.util import use_cpp_class, use_cpp_method


_PPRINT_USE_COLOR = config.bool_state(
    'jax_pprint_use_color',
    True,
    help='Enable jaxpr pretty-printing with colorful syntax highlighting.'
)

def _can_use_color() -> bool:
  try:
    # 检查是否处于 IPython 或 Colab 环境
    ipython = get_ipython()  # pyrefly: ignore[unknown-name]
    shell = ipython.__class__.__name__
    if shell == "ZMQInteractiveShell":
      # Jupyter Notebook
      return True
    elif "colab" in str(ipython.__class__):
      # Google Colab（外部或内部）
      return True
  except NameError:
    pass
  # 否则检查是否处于终端环境
  return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

CAN_USE_COLOR = _can_use_color()

Color = _pretty_printer.Color
Intensity = _pretty_printer.Intensity

OutputFormat = _pretty_printer.OutputFormat


@use_cpp_class(_pretty_printer.Doc)
class Doc:

  @use_cpp_method()
  def __add__(self, other: Doc) -> Doc:
    raise NotImplementedError

  @use_cpp_method()
  def __repr__(self) -> str:
    raise NotImplementedError

  def __str__(self) -> str:
    return self.format()

  @use_cpp_method()
  def _format(
      self,
      width: int,
      *,
      use_color: bool,
      annotation_prefix: str,
      source_map: list[list[tuple[int, int, Any]]] | None,
      separable_lines: bool = False,
      output_format: OutputFormat | None = None,
  ) -> str:
    raise NotImplementedError

  def format(
      self,
      width: int = 80,
      *,
      use_color: bool | None = None,
      output_format: OutputFormat | None = None,
      separable_lines: bool = False,
      annotation_prefix: str = " # ",
      source_map: list[list[tuple[int, int, Any]]] | None = None,
  ) -> str:
    """把美化打印器的文档格式化为字符串。

    Args:

    source_map: 对输出中的每一行，包含一个
      (起始列, 结束列, 来源) 元组列表。每个元组把一段
      输出文本与一个来源关联起来。
    """
    if use_color is None:
      use_color = CAN_USE_COLOR and _PPRINT_USE_COLOR.value

    if output_format is None:
      output_format = OutputFormat.TEXT
    return self._format(
        width,
        use_color=use_color,
        output_format=output_format,
        separable_lines=separable_lines,
        annotation_prefix=annotation_prefix,
        source_map=source_map,
    )


def nil() -> Doc:
  """空文档。"""
  return _pretty_printer.nil()  # pyrefly: ignore[bad-return]


def text(
    text: str,
    annotation: str | None = None,
    anchor: str | None = None,
    href: str | None = None,
) -> Doc:
  """字面文本。

  Args:
    text: 要打印的文本内容。
    annotation: 该文本的可选注解。
    anchor: 该文本可选的 HTML 锚点 ID。以 HTML 格式输出时，
      会把文本包裹在 <a id="..."> 标签中。
    href: 该文本可选的 HTML href。以 HTML 格式输出时，
      会把文本包裹在 <a href="..."> 标签中。
  """
  return _pretty_printer.text(text, annotation, anchor, href)  # pyrefly: ignore[bad-return]


def concat(children: Sequence[Doc]) -> Doc:
  """文档的拼接。"""
  return _pretty_printer.concat(children)  # pyrefly: ignore[bad-argument-type, bad-return]


def brk(text: str = " ") -> Doc:
  """一个换行点。

  根据所在分组的状态，打印为换行符或 `text`。
  """
  return _pretty_printer.brk(text)  # pyrefly: ignore[bad-return]


def group(doc: Doc) -> Doc:
  """布局备选分组。

  如果整个分组按把换行点当作其文本（通常是空格）打印时能容纳在一行内，
  则把这些换行点打印为各自的文本；否则，分组内部的换行点
  打印为换行符。
  """
  return _pretty_printer.group(doc)  # pyrefly: ignore[bad-argument-type, bad-return]


def nest(n: int, doc: Doc) -> Doc:
  """把缩进层级增加 `n`。"""
  return _pretty_printer.nest(n, doc)  # pyrefly: ignore[bad-argument-type, bad-return]


def color(
    child: Doc,
    foreground: Color | None = None,
    background: Color | None = None,
    intensity: Intensity | None = None,
) -> Doc:
  """ANSI 颜色。

  覆盖子文档文本的前景色/背景色/强度。
  打印时需要设置 use_colors=True，否则不起作用。
  """
  return _pretty_printer.color(child, foreground, background, intensity)  # pyrefly: ignore[bad-argument-type, bad-return]


def source_map(doc: Doc, source: Any) -> Doc:
  """来源映射。

  来源映射把美化打印器输出的某段文本与产生它的来源位置关联起来。
  对美化打印器而言，``source`` 可以是任意对象：
  我们只要求来源之间可以相互比较是否相等。
  文本区域到来源对象的映射可以作为
  ``format`` 方法的附带输出被填充。
  """
  return _pretty_printer.source_map(doc, source)  # pyrefly: ignore[bad-argument-type, bad-return]


type_annotation = partial(color, intensity=Intensity.NORMAL,
                          foreground=Color.MAGENTA)
keyword = partial(color, intensity=Intensity.BRIGHT, foreground=Color.BLUE)


def join(sep: Doc, docs: Sequence[Doc]) -> Doc:
  """用 `sep` 分隔地拼接 `docs`。"""
  docs = list(docs)
  if len(docs) == 0:
    return nil()
  if len(docs) == 1:
    return docs[0]
  xs = [docs[0]]
  for doc in docs[1:]:
    xs.append(sep)
    xs.append(doc)
  return concat(xs)
