# Copyright 2024 The JAX Authors.
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

# 文件职责：实现遵循 TC39 source map 规范的源码映射（sourcemap）读写。
# 供 JAX 在生成代码时记录生成代码与原始源码的位置对应，
# 让报错与调试信息能映射回用户写的原始源码。
# 提供 SourceMap 数据类（JSON 序列化/反序列化）、Base-64-VLQ 与
# segment 编解码，以及 TC39 mappings 字符串的解析与生成。
# MappingsGenerator 以绝对索引为输入，负责转成 TC39 的相对增量形式。

"""
遵循 `TC39 <https://tc39.es/source-map>`_ 的 sourcemap 实现。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json

# 一个 Segment 编码生成源码中的各部分与原始源码之间的对应关系。
# 每个 segment 由 1、4 或 5 个变长字段组成，其语义见
# https://tc39.es/source-map/#mappings-structure
Segment = (
    tuple[int] | tuple[int, int, int, int] | tuple[int, int, int, int, int]
)

# Mappings 是生成源码中每一行的 segment 序列。
Mappings = Sequence[Sequence[Segment]]


@dataclass(frozen=True, slots=True)
class SourceMap:
  version: int
  # file: str
  # source_root: str
  sources: Sequence[str]
  sources_content: Sequence[str]
  names: Sequence[str]
  mappings: Mappings

  @classmethod
  def from_json(cls, json_data: str) -> SourceMap:
    """从 JSON 反序列化出一个 source map。"""
    data = json.loads(json_data)
    return cls(
        version=data["version"],
        sources=data["sources"],
        sources_content=data["sourcesContent"],
        names=data["names"],
        mappings=deserialize_mappings(data["mappings"]),
    )

  def to_json(self) -> str:
    """把 source map 序列化为 JSON。"""
    data = {
        "version": self.version,
        "sources": self.sources,
        "sourcesContent": self.sources_content,
        "names": self.names,
        "mappings": serialize_mappings(self.mappings),
    }
    return json.dumps(data)


VLQ_SIGN_MASK = 0x01
VLQ_MORE_MASK = 0x20
VLQ_VALUE_MASK = 0x1F
VLQ_VALUE_BITWIDTH = 5
VLQ_ALPHABET = (
    list(range(ord("A"), ord("Z") + 1))
    + list(range(ord("a"), ord("z") + 1))
    + list(range(ord("0"), ord("9") + 1))
    + [ord("+"), ord("/")]
)


def make_vlq_decode_table():
  lookup = {c: d for d, c in enumerate(VLQ_ALPHABET)}
  return [lookup.get(i, None) for i in range(256)]


VLQ_DECODE_TABLE = make_vlq_decode_table()


def decode_vlq(enc: Iterable[int]) -> int:
  """把 Base-64-VLQ 解码为整数。"""
  enc_iter = iter(enc)
  d = VLQ_DECODE_TABLE[next(enc_iter)]
  sign = bool(d & VLQ_SIGN_MASK)
  value = (d & VLQ_VALUE_MASK) >> 1
  # 补偿第一个量元把符号位放在最低位的情况：
  shift = -1

  while d & VLQ_MORE_MASK:
    shift += VLQ_VALUE_BITWIDTH
    d = VLQ_DECODE_TABLE[next(enc_iter)]
    value |= (d & VLQ_VALUE_MASK) << shift

  return -value if sign else value


def encode_vlq(value: int) -> bytes:
  """把整数编码为 Base-64-VLQ。"""
  # 把符号位移到最低位
  value = ((-value) << 1 | 1) if value < 0 else value << 1
  buf = []

  while True:
    d = value & VLQ_VALUE_MASK
    value >>= VLQ_VALUE_BITWIDTH
    more = value > 0
    if more:
      d |= VLQ_MORE_MASK
    buf.append(VLQ_ALPHABET[d])
    if not more:
      break
  return bytes(buf)


def decode_segment(enc: Iterable[int]) -> Segment:
  """把一串 VLQ 解码为一个 segment。"""
  enc_iter = iter(enc)
  col = decode_vlq(enc_iter)
  try:
    source = decode_vlq(enc_iter)
  except StopIteration:
    # 在这里停止是可以的（1 字段 segment）。
    return (col,)
  source_line = decode_vlq(enc_iter)
  source_col = decode_vlq(enc_iter)
  try:
    name = decode_vlq(enc_iter)
  except StopIteration:
    # 在这里停止也可以（4 字段 segment）。
    return col, source, source_line, source_col
  # （5 字段 segment）
  return col, source, source_line, source_col, name


def encode_segment(seg: Segment) -> bytes:
  """把 segment 编码为一串 VLQ。"""
  return b"".join(encode_vlq(value) for value in seg)


def deserialize_mappings(mappings_str: str) -> Mappings:
  """解码 TC39 映射数据字符串。"""
  mappings_bytes = bytes(mappings_str, encoding="ascii")
  return [
      list(map(decode_segment, mapping.split(b","))) if mapping else []
      for mapping in mappings_bytes.split(b";")
  ]


def serialize_mappings(mappings: Mappings) -> str:
  """把 mappings 编码为 TC39 映射数据字符串。"""
  enc = b";".join(
      b",".join(encode_segment(seg) for seg in segs) for segs in mappings
  )
  return enc.decode("ascii")


class MappingsGenerator:
  """MappingsGenerator 是用于构建 mappings 的构造器 API。

  TC39 映射数据不便于直接生成：为了压缩数据，
  它用相对于前一个元素的数值来编码大多数索引。
  MappingsGenerator 通过在所有地方接受绝对索引来简化这件事。
  """

  def __init__(self):
    self._last_col = None
    self._last_source = 0
    self._last_source_line = 0
    self._last_source_col = 0
    self._last_name = 0
    self._mappings = []
    self._cur_group = None

  def new_group(self):
    """开始一个新的组（行）。"""
    self._last_col = 0
    self._cur_group = []
    self._mappings.append(self._cur_group)

  def new_segment(self, *seg):
    """在当前组中开始一个新的源码映射 segment。

    Args:
      *seg: 与 TC39 中相同的 segment，但所有索引都是绝对索引。详见
        https://tc39.es/source-map/#mappings-structure。

    Raises:
      RuntimeError: 若当前不存在组。
    """
    assert len(seg) >= 1
    group = self._cur_group
    if group is None:
      raise RuntimeError("No current group. Forgot to call new_group()?")

    col = seg[0] - self._last_col
    self._last_col = seg[0]

    if len(seg) == 1:
      group.append((col,))
      return

    source = seg[1] - self._last_source
    self._last_source = seg[1]
    source_line = seg[2] - self._last_source_line
    self._last_source_line = seg[2]
    source_col = seg[3] - self._last_source_col
    self._last_source_col = seg[3]

    if len(seg) == 4:
      group.append((col, source, source_line, source_col))
      return

    name = seg[4] - self._last_name
    self._last_name = seg[4]

    if len(seg) == 5:
      group.append((col, source, source_line, source_col, name))
      return

    assert False, "invalid segment"

  def mappings(self) -> Mappings:
    """把映射按行返回为 segment 列表。"""
    return self._mappings
