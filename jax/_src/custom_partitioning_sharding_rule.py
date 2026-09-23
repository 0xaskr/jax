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

# 文件职责：实现 Shardy 的自定义分片规则表示及其与 MLIR 属性的互转。
# 该模块是 `jax.experimental.custom_partitioning` 中 `infer_sharding_from_operands`
# 等回调（用户用类 Einsum 记号字符串 `"a b -> a"` 描述分片规则）的底层支撑，
# 负责把这类字符串解析校验成 `SdyShardingRule`，再结合操作数/结果的类型
# 构建出 `sdy.OpShardingRuleAttr`，供 Shardy 分区器使用。

"""实现 SdyShardingRule。"""

from collections import OrderedDict

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import sdy


# 用一个字符替换 ... 以简化解析。
BATCHING: str = "…"

# 批处理维度因子名的前缀，用于把开头的 ... 展开成若干因子。
_BATCHING_DIM_FACTOR_PREFIX = "?"


def _check_factor(factor:str):
  """校验一个因子。

  因子是以字母开头、且只包含字母、数字或下划线的字符串。
  """
  if not factor[0].isalpha():
    raise ValueError(f"Factor names have to start with a letter, but got '{factor[0]}'")
  for char in factor[1:]:
    if char != "_" and not char.isdigit() and not char.isalpha():
      raise ValueError(f"Unknown character '{char}'")

def _is_batching(factor: str) -> bool:
  """检查一个因子是否表示开头的批处理维度。

  开头的批处理维度由含 ... 的因子表示，其后可选地跟一个数字，
  并且 ... 等价于 ...0。
  """
  if len(factor) < 1 or factor[0] != BATCHING:
    return False
  return len(factor) == 1 or factor[1:].isdigit()

def _get_batching_group(factor: str) -> str:
  """从表示开头批处理维度的因子中取出批处理组。"""
  return factor[1:] if len(factor) > 1 else "0"

class CompoundFactor(tuple):
  """描述一个复合因子的各个因子。

  复合因子至少要包含两个因子，例如
  * CompoundFactor('b', 'c')。
  """
  def __init__(self, *factors):
    if len(factors) < 2:
      raise ValueError("A compound factor should contain at least two factors")
    for factor in factors:
      if not isinstance(factor, str):
        raise ValueError(f"Each element of CompoundFactor must be a str, but got {type(factor)}")
      if _is_batching(factor):
        raise ValueError("Ellipsis can't be used in a compound factor")
      else:
        _check_factor(factor)

  def __new__(cls, *factors):
    return tuple.__new__(CompoundFactor, factors)


class ArrayMapping(tuple):
  """描述一个操作数或结果的各个因子。

  每个元素要么是一个因子，要么是一个 CompoundFactor。开头的元素也可以是
  BATCHING，它表示批处理维度。示例：
  * ArrayMapping('a')
  * ArrayMapping('b', 'c')
  * ArrayMapping(CompoundFactor('b', 'c'), 'd')
  * ArrayMapping(BATCHING, CompoundFactor('b', 'c'), 'd')
  """
  def __init__(self, *dim_mappings):
    for i, d in enumerate(dim_mappings):
      if not isinstance(d, str) and not isinstance(d, CompoundFactor):
        raise ValueError(
            "Each element of ArrayMapping must be a str or CompoundFactor, but"
            f" got {type(d)}")
      if isinstance(d, str):
        if _is_batching(d):
          if i != 0:
            raise ValueError("Ellipsis can only be used at the beginning of a dimension")
        else:
          _check_factor(d)

  def __new__(cls, *dim_mappings):
    return tuple.__new__(ArrayMapping, dim_mappings)


class SdyShardingRule:
  """表示一条 Shardy 分片规则。

  SdyShardingRule 包含各操作数与结果的 ArrayMapping、可选的
  特殊因子以及可选的因子大小。因子是 ArrayMapping 中使用的名字。
  若某个因子只用在 CompoundFactor 中，则必须指定它的大小。

  默认情况下，因子是直通（passthrough）因子。可以用关键字参数指定
  其他因子种类，包括 reduction_factors、need_replication_factors
  和 permutation_factors。
  """
  operand_mappings: tuple[ArrayMapping, ...]
  result_mappings: tuple[ArrayMapping, ...]
  factor_sizes: dict[str, int]
  reduction_factors: tuple[str, ...]
  need_replication_factors: tuple[str, ...]
  permutation_factors: tuple[str, ...]

  def __init__(self, operand_mappings: tuple[ArrayMapping, ...],
               result_mappings: tuple[ArrayMapping, ...],
               *, reduction_factors: tuple[str, ...] = (),
               need_replication_factors: tuple[str, ...] = (),
               permutation_factors: tuple[str, ...] = (),
               **factor_sizes: int):
    # 找出所有因子，并标记它们的大小能否被推断出来。
    factors_inferrable = {}
    for value in operand_mappings + result_mappings:
      for dim in value:
        if isinstance(dim, str):
          factors_inferrable[dim] = True
        else:
          for factor in dim:
            if factor not in factors_inferrable.keys():
              factors_inferrable[factor] = False

    # 检查 factor_sizes 中的因子确实被这条规则用到。
    for factor in factor_sizes:
      if factor not in factors_inferrable:
        raise ValueError(
          f"Factor {factor} is not used in the rule, but size is provided")

    # 检查用于整个维度的因子不在 factor_sizes 中，而从未用于整个维度的因子
    # 都在 factor_sizes 中。
    for factor, inferable in factors_inferrable.items():
      if factor not in factor_sizes and not inferable:
        raise ValueError(
          f"Factor {factor} is only used in compound factors; must specify"
          " its size")
      if factor in factor_sizes and inferable:
        raise ValueError(
          f"Factor {factor} represents a whole dimension; do not specify its"
          " size")

    special_factors = set()
    def check_special_factors(kind, factors):
      if not isinstance(factors, tuple):
        raise ValueError(f"{kind} must be a tuple of factors")

      if len(factors) != len(set(factors)):
        raise ValueError(f"{kind} contains duplicated factors")

      for factor in factors:
        if factor not in factors_inferrable:
          raise ValueError(
            f"Factor {factor} in {kind} is not used in the rule")
        if factor in special_factors:
          raise ValueError(f"Factor {factor} can only be in one of the "
              f"reduction, need replication, or permutation factor sets.")
        special_factors.add(factor)

    check_special_factors("reduction_factors", reduction_factors)
    check_special_factors("need_replication_factors", need_replication_factors)
    check_special_factors("permutation_factors", permutation_factors)

    self.operand_mappings = operand_mappings
    self.result_mappings = result_mappings
    self.factor_sizes = factor_sizes
    self.reduction_factors = reduction_factors
    self.need_replication_factors = need_replication_factors
    self.permutation_factors = permutation_factors


  def __str__(self):
    def to_str(kind, factors):
      if len(factors) > 0:
        return f" {kind}={factors}"
      return ""

    special_factors = (to_str("reduction_factors", self.reduction_factors) +
        to_str("need_replication_factors", self.need_replication_factors) +
        to_str("permutation_factors", self.permutation_factors))
    return (f"SdyShardingRule({self.operand_mappings}, {self.result_mappings}, "
            f"{self.factor_sizes}{special_factors})")


def _get_batching_dim_factor_name(batch_group: str,batch_dim_order : int):
  """为一个批处理维度构造因子名。

  为了支持构建该分片规则的 MLIR 表示，我们会把开头的 ... 展开为表示各
  批处理维度的因子。因此，需要为这些批处理维度构造一个用户不会用到的
  因子名。
  """
  return f"{_BATCHING_DIM_FACTOR_PREFIX}{batch_group}_{batch_dim_order}"

def _parse_values(
    rule: str,
) -> tuple[ArrayMapping, ...]:
  """解析类 Einsum 记号字符串的左侧或右侧。

  把类 Einsum 记号字符串中的每个操作数或结果转换为一个 ArrayMapping 元组。
  这与 einops 在 einops/parsing.py 中解析其规则的方式非常接近。

  Args:
    rule: 某个运算各操作数或结果的类 Einsum 记号。

  Returns:
    ArrayMapping 构成的元组。

  Raises:
    ValueError: 若规则不对称或包含未知字符。
  """

  # 去掉规则中不必要的空格，以简化解析过程。
  words = rule.split()
  rule = " ".join(words)

  # 与 einops 的规则类似，空的左侧/右侧表示一个标量值。
  if not rule:
    return (ArrayMapping(),)

  all_values = []
  # 表示某个值的所有维度。当 value[0]==BATCHING 时，该值可能有 0 个或
  # 更多个开头维度。
  value = []
  current_factor: str | None = None
  # 值为 None 表示当前维度不是复合维度，而值为 [] 表示我们刚开始解析
  # 一个复合维度。
  current_compound_dim: list[str] | None = None

  def add_factor(x):
    if current_compound_dim is None:
      value.append(x)
    else:
      current_compound_dim.append(x)

  rule_len = len(rule)
  rule_index = 0
  while rule_index < rule_len:
    char = rule[rule_index]
    rule_index += 1
    if char == BATCHING:
      if (current_factor is not None or current_compound_dim is not None
          or value):
        raise ValueError(
            "Ellipsis can only be used at the beginning of a dimension")
      if rule_index < rule_len and rule[rule_index].isdigit():
        batching_group_str = ""
        while rule_index < rule_len and rule[rule_index].isdigit():
          batching_group_str += rule[rule_index]
          rule_index += 1
        batching_group = str(int(batching_group_str))
      else:
        batching_group = "0"

      add_factor(f"{BATCHING}{batching_group}")
      continue
    if char in "(), ":
      if current_factor is not None:
        add_factor(current_factor)
        current_factor = None
      if char == "(":
        if current_compound_dim is not None:
          raise ValueError(
              "Compound factors should be one level, nested brackets are not"
              " allowed")
        current_compound_dim = []
      elif char == ")":
        if current_compound_dim is None:
          raise ValueError("Brackets are not balanced")
        if len(current_compound_dim) <= 1:
          raise ValueError("Brackets should contain at least two factors")
        value.append(CompoundFactor(*current_compound_dim))
        current_compound_dim = None
      elif char == ",":
        all_values.append(ArrayMapping(*value))
        value = []
    elif char == "_" or char.isdigit() or char.isalpha():
      if current_factor is None:
        if str.isdigit(char):
          raise ValueError(f"Factor names have to start with a letter, but got '{char}'")
        current_factor = char
      else:
        current_factor += char
    else:
      raise ValueError(f"Unknown character '{char}'")

  if current_compound_dim is not None:
    raise ValueError(f"Brackets are not balanced in rule: '{rule}'")
  if current_factor is not None:
    add_factor(current_factor)
  all_values.append(ArrayMapping(*value))

  return tuple(all_values)

def str_to_sdy_sharding_rule(rule: str, *,
                             reduction_factors: tuple[str, ...] = (),
                             need_replication_factors: tuple[str, ...] = (),
                             permutation_factors: tuple[str, ...] = (),
                             **factor_sizes: int) -> SdyShardingRule:
  """由类 Einsum 记号字符串构造一个 SdyShardingRule 对象。

  做法是：验证输入的类 Einsum 记号字符串以及可选的
  特殊因子和因子大小确实构成一条合法的分片规则，并把它转换为
  内部表示。

  Args:
    rule: 某个运算的类 Einsum 记号字符串。
    reduction_factors: 由约简因子构成的元组。
    need_replication_factors: 由需要复制的因子构成的元组。
    permutation_factors: 由置换因子构成的元组。
    **factor_sizes: 可选的因子大小。

  Raises:
    ValueError: 若规则或 factor_sizes 存在任何问题。
  """
  if not isinstance(rule, str):
    raise TypeError(f"rule must be a str, but got {type(rule)}")
  if not all(isinstance(size, int) for size in factor_sizes.values()):
    raise TypeError(
        f"factor_sizes must be a dict of str to int, but got {factor_sizes}")

  # 把 ... 替换成单个字符以简化解析。
  if BATCHING in rule:
    raise ValueError(f"Unknown character '{BATCHING}'")
  if "." in rule:
    rule = rule.replace("...", BATCHING)
    if "." in rule:
      raise ValueError("Character '.' must be used inside ellipsis '...'")

  try:
    operands, results = rule.split("->")
  except ValueError as e:
    raise ValueError(f"There is no -> in rule: '{rule}'") from e

  operand_mappings = _parse_values(operands)
  result_mappings = _parse_values(results)
  return SdyShardingRule(operand_mappings, result_mappings,
                         reduction_factors=reduction_factors,
                         need_replication_factors=need_replication_factors,
                         permutation_factors=permutation_factors,
                         **factor_sizes)


def sdy_sharding_rule_to_mlir(
  rule: SdyShardingRule,
  operand_types: list[ir.Type],
  result_types: list[ir.Type],) -> ir.Attribute:
  """构建该分片规则的 MLIR 表示。

  做法是：验证规则与该运算的各类型一致，并把类 Einsum 记号字符串
  转换为 OpShardingRuleAttr。
  """
  if len(rule.operand_mappings) != len(operand_types):
    raise ValueError(
      f"Sharding rule has {len(rule.operand_mappings)} operands, but the operation"
      f" has {len(operand_types)} operands")
  if len(rule.result_mappings) != len(result_types):
    raise ValueError(
      f"Sharding rule has {len(rule.result_mappings)} results, but the operation"
      f" has {len(result_types)} results")
  if not all(isinstance(t, ir.Type) for t in operand_types + result_types):
    raise TypeError(
        f"operand_types and result_types must be a list of ir.Type, but got"
        f" {operand_types} and {result_types}")

  factors_to_indices_sizes: OrderedDict[str, list[int]] = OrderedDict()
  types = operand_types + result_types
  UNKNOWN = -1  # 未知因子大小或因子索引的表示。

  def get_message_for_value(i):
    if i >= len(operand_types):
      return f"{i - len(operand_types)}th result"
    else:
      return f"{i}th operand"

  def get_rank_for_value(i):
    return ir.ShapedType(types[i]).rank

  def get_size_for_value_dim(i, j):
    return ir.ShapedType(types[i]).shape[j]

  def add_factor(factor, size):
    """把一个因子加入 factors_to_indices_sizes。

    `size` 可以是一个维度的大小、用户指定的因子大小，或者当某个因子
    先出现在复合因子中、之后又用于整个维度时的 UNKNOWN。若某个因子不是
    用于开头的批处理维度，且它对应多个大小，则取其中最小的大小。
    """
    factor_index, factor_size = factors_to_indices_sizes.get(factor, [UNKNOWN, UNKNOWN])
    if factor_index != UNKNOWN:
      # 不是第一次见到这个因子。
      if size != UNKNOWN and factor_size != UNKNOWN and factor_size != size:
        if _BATCHING_DIM_FACTOR_PREFIX in factor:
          raise ValueError(f"Batching dimension {factor[1:]} corresponds to "
                           f"two sizes: {factor_size} and {size}")
        else:
          if size < factor_size:
            # 用较小的大小更新该因子的大小。
            factor_size = UNKNOWN
      if size != UNKNOWN and factor_size == UNKNOWN:
        factors_to_indices_sizes[factor] = [factor_index, size]
    else:
      # 第一次见到这个因子。
      factor_index = len(factors_to_indices_sizes)
      factors_to_indices_sizes[factor] = [factor_index, size]

  def add_batching_dim_factor(batch_grp, batch_dim_order, factor_size):
    add_factor(_get_batching_dim_factor_name(batch_grp, batch_dim_order), factor_size)

  def build_dim_mapping_for_compound_factors(i, j, factors):
    accumulated_size = 1
    all_indices = []
    for factor in factors:
      factor_index, factor_size = factors_to_indices_sizes[factor]
      accumulated_size *= factor_size
      all_indices.append(factor_index)

    dim_size = get_size_for_value_dim(i, j)
    if accumulated_size != dim_size:
      raise ValueError(
          f"{get_message_for_value(i)} actual size {dim_size} doesn't match"
          f" the size {accumulated_size} derived from the compound factors"
          f" {factors}")

    return sdy.DimMappingAttr.get(factor_indices=all_indices)

  def factors_to_indices(factors):
    return [factors_to_indices_sizes[factor][0] for factor in factors]

  # 按因子在规则中出现的顺序加入它们及其大小，
  # 其中也包括由省略号表示的批处理维度。
  batching_group_to_rank: dict[str, int] = {}
  for i, mapping in enumerate(rule.operand_mappings + rule.result_mappings):
    value = tuple(mapping)
    if value and _is_batching(value[0]):
      batching_group = _get_batching_group(value[0])
      value = value[1:]
    else:
      batching_group = None
    rule_rank = len(value)
    op_rank = get_rank_for_value(i)
    # 省略号所表示的维度个数。
    current_batching_rank = 0
    if batching_group is not None and op_rank >= rule_rank:
      current_batching_rank = op_rank - rule_rank
    if batching_group is not None:
      ellipsis_rank = batching_group_to_rank.get(batching_group, None)
      if ellipsis_rank is None:
        ellipsis_rank = current_batching_rank
        batching_group_to_rank[batching_group] = ellipsis_rank
      elif ellipsis_rank != current_batching_rank:
        raise ValueError(
          "Ellipsis represents different number of leading dimensions"
          f" {ellipsis_rank} and {current_batching_rank}")
    rule_rank += current_batching_rank
    if rule_rank != op_rank:
      msg = get_message_for_value(i)
      raise ValueError(
        f"Sharding rule {msg} has rank {rule_rank}, but the operation"
        f" {msg} has rank {op_rank}")

    for j in range(current_batching_rank):
      add_batching_dim_factor(batching_group, j, get_size_for_value_dim(i, j))

    for j, dim in enumerate(value):
      if isinstance(dim, str):
        add_factor(dim, get_size_for_value_dim(i, j + current_batching_rank))
      else:
        for factor in dim:
          add_factor(factor, rule.factor_sizes.get(factor, UNKNOWN))

  # 为每个操作数和结果构建张量映射。
  tensor_mappings = []
  for i, mapping in enumerate(rule.operand_mappings + rule.result_mappings):
    value = tuple(mapping)
    dim_mappings = []
    if value and _is_batching(value[0]):
      batching_group = _get_batching_group(value[0])
      value = value[1:]
      if batching_group in batching_group_to_rank:
        current_batching_rank = batching_group_to_rank[batching_group]
      else:
        raise ValueError("Unreachabled code")
    else:
      current_batching_rank = 0
      batching_group = None

    for j in range(current_batching_rank):
      assert batching_group is not None
      dim_mappings.append(
        sdy.DimMappingAttr.get(factor_indices=[
          factors_to_indices_sizes[_get_batching_dim_factor_name(batching_group, j)][0]]))

    for j, dim in enumerate(value):
      if isinstance(dim, str):
        dim_mappings.append(
          sdy.DimMappingAttr.get(
            factor_indices=[factors_to_indices_sizes[dim][0]]))
      else:
        dim_mappings.append(
          build_dim_mapping_for_compound_factors(
            i, j + current_batching_rank, dim))

    tensor_mappings.append(
      sdy.TensorMappingAttr.get(dim_mappings=dim_mappings))

  return sdy.OpShardingRuleAttr.get(
      factor_sizes=[item[1] for item in factors_to_indices_sizes.values()],
      operand_mappings=tensor_mappings[0:len(operand_types)],
      result_mappings=tensor_mappings[len(operand_types):],
      is_custom=True,
      reduction_factors=factors_to_indices(rule.reduction_factors),
      need_replication_factors=factors_to_indices(rule.need_replication_factors),
      permutation_factors=factors_to_indices(rule.permutation_factors),
      )
