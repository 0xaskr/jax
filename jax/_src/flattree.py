# Copyright 2026 The JAX Authors.
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

# 文件职责：提供 `FlatTree`——用 Python 面向对象方式编码 pytree 结构的内部函子。
# JAX 的暂存与追踪机制（`partial_eval`、`ad`、`stages`、`interpreters/mlir`、
# `interpreters/remat`、Pallas 等）靠它把「扁平叶子序列 + 树结构」表示成代数数据类型，
# 每个分支对应一个子类：元组、pytree、单例、静态项、过滤项、字典。
# 核心操作包括 map/update/unzip2/unpack/filter/unfilter/unflatten，
# 以及 pack/flatten/statics_mask 等按静态与动态拆分参数并打包的辅助函数。

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property
import itertools as it
from typing import Any

from jax._src.tree_util import (
    PyTree, PyTreeDef, tracing_registry, treedef_children, treedef_tuple_tracing_registry)
from jax._src.util import (
    unzip2, Either, safe_map, safe_zip, split_list_checked, PyArgs)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

class FlatTree:
  """`FlatTree` 是这个函子的 Python 面向对象版本。
  每个分支都实现为 `FlatTree` 的一个子类，
  固定的那组模式匹配操作则由子类实现的方法来承担。

    data FlatTree a = Tuple [FlatTree a]
                    | Pytree [a] PyTreeDef
                    | Singleton a
                    | Static Aux
                    | Filtered (FlatTree (Either a Aux))
                    -- TODO: 移除这个分支
                    | Dict [Key] [FlatTree a]
  """

  # === 每个子类（ADT 分支）都应实现以下方法 ===
  def __init__(self, *a, **k): assert False, f"subclass {self.__class__} should implement"
  def __len__(self):           assert False, f"subclass {self.__class__} should implement"
  def __eq__(self, other):     assert False, f"subclass {self.__class__} should implement"
  def __hash__(self):          assert False, f"subclass {self.__class__} should implement"
  def __repr__(self):          assert False, f"subclass {self.__class__} should implement"
  def map(self, f):            assert False, f"subclass {self.__class__} should implement"
  @property
  def tree(self) -> PyTreeDef: assert False, f"subclass {self.__class__} should implement"

  # === 派生方法 ===
  def map2(self: FlatTree, t2: Sequence[Any], f: Callable) -> FlatTree:
    n = len(self)
    assert len(t2) == n
    return self.update(f(x1, x2) for x1, x2 in zip(self.vals, list(t2)))

  def map3(
      self: FlatTree, t2: Sequence[Any], t3: Sequence[Any], f: Callable) -> FlatTree:
    n = len(self)
    assert len(t2) == n and len(t3) == n
    return self.update(f(x1, x2, x3)
                       for x1, x2, x3 in zip(self.vals, list(t2), list(t3)))

  def void(self): return self.map(lambda _: None)
  def unzip2(self: FlatTree) -> tuple[FlatTree, FlatTree]:
    ys = []
    zs = []
    for y, z in self.vals:
      ys.append(y)
      zs.append(z)
    return self.update(ys), self.update(zs)

  @property
  def unpackable(self):
    if isinstance(self, FTTuple):
      return True
    elif isinstance(self, FTPyTree):
      return (isinstance(nd := self.tree.node_data(), (tuple, list))
              and nd and nd[0] in (tuple, list))
    else:
      return False

  def unpack(self: FlatTree) -> tuple[FlatTree, ...]:
    if isinstance(self, FTTuple):
      return self.elts
    elif isinstance(self, FTPyTree):
      # TODO: 我们应当能够去掉这个分支，办法是保证返回元组的高阶原语
      # 其输出上总是带有 FTTuple。
      treedefs = treedef_children(self.tree)
      valss = split_list_checked(self.xs, [t.num_leaves for t in treedefs])
      return tuple(FTPyTree(vals, treedef) for vals, treedef in zip(valss, treedefs))
    elif isinstance(self, FTFiltered):
      return tuple(FTFiltered(t) for t in self.val.unpack())
    else:
      raise TypeError(f"Not a FlatTree tuple: {self}")

  def unpack2(self): return tuple(x.unpack() for x in self.unpack())

  def with_aux(self:FlatTree, aux:Any) -> FlatTree:
    return pack((self, FTStatic(aux)))

  def unpack_aux(self:FlatTree) -> tuple[FlatTree, Any]:
    x, static = self.unpack()
    assert isinstance(static, FTStatic)
    return x, static.val

  def unflatten(self) -> PyTree:
    if isinstance(self, FTPyTree):
      return self.tree.unflatten(self.vals)
    elif isinstance(self, FTTuple):
      return tuple(elt.unflatten() for elt in self.elts)
    elif isinstance(self, FTDict):
      return {k: v.unflatten() for k, v in zip(self.keys, self._vals)}
    elif isinstance(self, FTStatic):
      return self.val
    elif isinstance(self, FTSingleton):
      return self.val
    else:
      raise TypeError(f"Not a FlatTree pytree or tuple: {self}")

  def update(self, new_vals) -> FlatTree:
    new_vals = new_vals if type(new_vals) is tuple else tuple(new_vals)
    assert len(self) == len(new_vals)
    vals_iter = iter(new_vals)
    return self.map(lambda _: next(vals_iter))

  @cached_property
  def paths(self) -> FlatTree:
    # TODO(dougalm): 找一个不用来回转换就能做到这件事的办法
    try:
      paths, _ = unzip2(tracing_registry.flatten_with_path(self.unflatten())[0])
      assert len(paths) == len(self.vals)
      return self.update(paths)
    except:
      return self.update([()] * len(self.vals))  # 不是我们的问题

  @cached_property
  def vals(self):
    vals = []
    self.map(vals.append)
    return tuple(vals)

  def __iter__(self):
    return iter(self.vals)

  def __getitem__(self, i):
    return self.vals[i]

  def filter(self, f):
    # `list.filter` 的 FlatTree 版本。与后者不同，
    # 它把被过滤掉的数据保留在 pytree 结构中，
    # 以便之后能用 `unfilter` 重新实例化。
    return self.filter_with_mask(map(f, self))

  # True 表示保留
  def filter_with_mask(self, mask):
    return ft_filtered(self.map2(mask,
        lambda x, kept: Either.right(x) if kept else Either.left(x)))

  def unfilter(self):
    if isinstance(self, FTTuple):
      return FTTuple(*(t.unfilter() for t in self.elts))
    elif isinstance(self, FTFiltered):
      return self.val.map(lambda x: x.val)
    else:
      raise TypeError("expected a FTTuple or a FTFiltered")

  def enumerate(self):
    idxs = it.count()
    return self.map(lambda x: (next(idxs), x))
  # TODO: 按需补充 map3、zip、unzip3 之类的其他辅助方法

class FTTuple(FlatTree):
  # TODO: 改掉这个做法。我们不应该从 FlatTree 里取 treedef
  @property
  def tree(self):
    return treedef_tuple_tracing_registry(t.tree for t in self.elts)

  # Tuple [FlatTree a]
  def __init__(self, *elts):
    assert all(isinstance(v, FlatTree) for v in elts)
    self.elts = elts
  def __len__(self): return sum(len(elt) for elt in self.elts)
  def __eq__(self, other):
    return isinstance(other, FTTuple) and self.elts == other.elts
  def __hash__(self): return hash(self.elts)
  def __repr__(self): return repr(self.elts)
  def map(self, f): return FTTuple(*(elt.map(f) for elt in self.elts))

# TODO(dougalm): 删掉这个分支
class FTDict(FlatTree):
  # TODO: 改掉这个做法
  @property
  def tree(self):
    # 我们必须来回转换，因为不存在与 `treedef_tuple` 对应的字典版本
    _, t = tracing_registry.flatten(self.update([0] * len(self)).unflatten())
    return t

  # keys 应当是排好序的
  def __init__(self, keys, vals):
    keys = keys if type(keys) is tuple else tuple(keys)
    assert all(isinstance(v, FlatTree) for v in vals)
    vals = vals if type(vals) is tuple else tuple(vals)
    self.keys = keys
    self._vals = vals  # 加下划线以避免与 FlatTree.vals 冲突
  def __len__(self): return sum(len(val) for val in self._vals)
  def __eq__(self, other):
    return (isinstance(other, FTDict) and
            self.keys == other.keys and
            self._vals == other._vals)
  def __hash__(self): return hash((self.keys, self._vals))
  def __repr__(self): return repr(dict(zip(self.keys, self._vals)))
  def map(self, f):
    return FTDict(self.keys, tuple(elt.map(f) for elt in self._vals))

class FTPyTree(FlatTree):
  # Pytree [a] PyTreeDef
  def __init__(self, xs, treedef):
    assert isinstance(treedef, PyTreeDef)
    xs = xs if isinstance(xs, tuple) else tuple(xs)
    self.xs = xs
    self._tree = treedef
  @property
  def tree(self) -> PyTreeDef: return self._tree
  def __len__(self): return len(self.xs)
  def __eq__(self, other):
    return (isinstance(other, FTPyTree) and
            self.xs == other.xs and self.tree == other.tree)
  def __hash__(self): return hash((self.xs, self.tree))
  def __repr__(self): return f"Pytree(vals={list(self.xs)}, tree={self.tree})"
  def map(self, f):
    return FTPyTree(tuple(f(x) for x in self.xs), self.tree)

class FTSingleton(FlatTree):
  # Singleton a
  def __init__(self, val): self.val = val
  def __len__(self): return 1
  def __eq__(self, other):
    return isinstance(other, FTSingleton) and self.val == other.val
  def __hash__(self): return hash(self.val)
  def __repr__(self): return repr(self.val)
  def map(self, f): return FTSingleton(f(self.val))

class FTStatic(FlatTree):
  # Static Aux
  def __init__(self, val): self.val = val
  def __len__(self): return 0
  def __eq__(self, other):
    return (
        isinstance(other, FTStatic) and
        type(self.val) is type(other.val) and  # 参见 https://github.com/jax-ml/jax/pull/9311
        self.val == other.val)
  def __hash__(self): return hash(self.val)
  def __repr__(self): return f"Static({self.val})"
  def map(self, f): return self
  @property
  def tree(self): return tracing_registry.flatten(0)[1]  # 叶子的 treedef

class FTFiltered(FlatTree):
  # Filtered (FlatTree (Either Aux a))
  def __init__(self, val): self.val = val
  @cached_property
  def _len(self): return sum(1 for x in self.val if x.is_right)
  def __len__(self): return self._len
  def __eq__(self, other):
    return isinstance(other, FTFiltered) and self.val == other.val
  def __hash__(self): return hash(self.val)
  def __repr__(self): return f"RightsOnly({self.val})"
  def map(self, f):
    def apply_f_to_rights(x):
      if x.is_left:
        return Either.left(x.from_left())
      else:
        return Either.right(f(x.from_right()))
    return FTFiltered(self.val.map(apply_f_to_rights))

def pack(tree):
  # 我们本可以把它推广到 `FlatTree` 的任意 pytree，但目前
  # 元组与字典已经够用了。
  if isinstance(tree, FlatTree):
    return tree
  elif isinstance(tree, tuple):
    return FTTuple(*map(pack, tree))
  elif isinstance(tree, dict):
    keys, vals = unzip2(sorted(tree.items(), key=lambda pair: pair[0]))
    return FTDict(keys, tuple(map(pack, vals)))
  else:
    assert False, type(tree)

def pack_args(*args, **kwargs):
  # TODO: 检查 args 与 kwargs 的元素是否都是扁平树
  return pack((args, kwargs))

def flatten(tree: PyTree, is_leaf=None, registry=tracing_registry) -> FlatTree:
  vals, tree = registry.flatten(tree, is_leaf)
  return FTPyTree(vals, tree)

def flatten_args(*arg_trees: PyTree, registry=tracing_registry) -> FlatTree:
  arg_fts = tuple(flatten(t, registry=registry) for t in arg_trees)
  return pack((arg_fts, {}))

# 静态项放左边，动态项放右边
def statics_mask(args: PyArgs, static_argnums, static_argnames):
  num_args = len(args.args)
  static_argnums = [i % num_args if i < 0 else i for i in static_argnums]
  return ([i in static_argnums for i in range(num_args)] +
          [k in static_argnames for k in args.kwargs.keys()])

def flatten_static_argnums_argnames(
    args, kwargs, static_argnums, static_argnames, registry=tracing_registry):
  pargs = PyArgs(args, kwargs)
  statics = statics_mask(pargs, static_argnums, static_argnames)
  fts = pargs.map2(
      statics,
      lambda x, is_static:
          FTStatic(x) if is_static else flatten(x, registry=registry))
  return pack((fts.args, fts.kwargs))

# TODO: 这个设计很糟糕。改成在 pjit 里用 FlatTree 代替 treedef 就能把它删掉。
def flatten_static_argnums_argnames_and_return_various_trees(
    args, kwargs, static_argnums, static_argnames):
  registry = tracing_registry
  pargs = PyArgs(args, kwargs)
  statics = statics_mask(pargs, static_argnums, static_argnames)
  ans_ft = pack(pargs.map2(
      statics,
      lambda x, static: FTStatic(x) if static else flatten(x, registry=registry)
      ).args_kwargs)
  _, tree_nones = registry.flatten(
      pargs.map2(statics, lambda x, static: None if static else x).args_kwargs)
  _, tree_filtered = registry.flatten(
      pargs.filter_with_mask([not s for s in statics]).args_kwargs)
  return ans_ft, tree_nones, tree_filtered

def flatten_list(xs):
  # [a] -> FlatTree[a]。把列表元素当作叶子。
  return pack(tuple(FTSingleton(x) for x in xs))

def nones(n):
  return flatten_list([None] * n)

def zipstar(ft):
  return pack(tuple(zip(*ft.unpack2())))

def ft_filtered(tree):
  if isinstance(tree, FTTuple):
    return FTTuple(*map(ft_filtered, tree.elts))
  else:
    return FTFiltered(tree)

def treedef_to_ft(treedef):
  vals = [None] * treedef.num_leaves
  return FTPyTree(vals, treedef)

def treedef_args_to_ft(tree, vals):
  arg_trees = treedef_children(tree)
  return FTTuple(*map(treedef_to_ft, arg_trees)).update(vals)
