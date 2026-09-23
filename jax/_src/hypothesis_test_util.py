# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 文件职责：为 JAX 的 hypothesis 属性测试提供分片支持与统一配置。
# hypothesis 默认把整个测试函数当作单个测试，无法被 Bazel/多线程的测试
# 分片机制拆分；本模块包装 hypothesis 的内部测试，按方法签名的确定性哈希
# 把每个生成的示例分派到某个分片，从而让慢速属性测试也能并行展开。
# 它还注册并加载 "deterministic" / "interactive" 两个配置档，并抑制若干
# 在 JAX 测试环境下无意义的健康检查，供各测试模块经 `jtu` 基类使用。

import functools
import hashlib
import itertools
import logging
import os
import unittest

import hypothesis as hp
from hypothesis import errors as hp_errors
from hypothesis.internal import detection
from hypothesis.internal import reflection
from hypothesis.strategies._internal import core as hps_internal_core
from jax._src import config
from jax._src import test_util as jtu
from jax._src.test_loader import JaxTestLoader, TEST_NUM_THREADS


HYPOTHESIS_PROFILE = config.string_flag(
    "hypothesis_profile",
    os.getenv("JAX_HYPOTHESIS_PROFILE", "deterministic"),
    help=(
        "Select the hypothesis profile to use for testing. Available values: "
        "deterministic, interactive"
    ),
)


_TEST_SHARD_INDEX = int(os.environ.get("TEST_SHARD_INDEX", "0"))
_TEST_TOTAL_SHARDS = int(os.environ.get("TEST_TOTAL_SHARDS", "1"))


def hypothesis_inner_test_shard(inner_test, args, kwargs, total_shards):
  """返回为某个生成的 Hypothesis 示例预期的分片索引。"""
  text_repr = reflection.repr_call(inner_test, args, kwargs)
  test_hash = int(hashlib.md5(text_repr.encode()).hexdigest(), 16)
  return test_hash % total_shards


def _shard_aware_hypothesis_inner_test(inner_test):
  @functools.wraps(inner_test)
  def shard_aware_inner_test_fn(*args, **kwargs):
    self = kwargs["self"] if "self" in kwargs else args[0]
    proc_shards = _TEST_TOTAL_SHARDS
    proc_idx = _TEST_SHARD_INDEX
    thread_shards = self._thread_total_shards
    thread_idx = self._thread_shard_index

    total_shards = proc_shards * thread_shards
    global_shard_idx = proc_idx * thread_shards + thread_idx

    if total_shards == 1:
      return inner_test(*args, **kwargs)

    # 如果本次测试用例执行中已经遇到过失败，就假定我们正处于收缩/解释阶段，
    # 该阶段无法与分片配合工作。
    if getattr(self, "_hypothesis_failed", False):
      return inner_test(*args, **kwargs)

    mod = hypothesis_inner_test_shard(inner_test, args, kwargs, total_shards)
    if mod != global_shard_idx:
      # 这里不调用 `googletest.skip`，因为在 hypothesis 内部跳过会跳过*整个*
      # 测试，而不只是当前示例。
      return None
    else:
      try:
        return inner_test(*args, **kwargs)
      # 被跳过的测试不算失败，也不会触发收缩阶段。
      except (hp_errors.UnsatisfiedAssumption, unittest.SkipTest):
        raise
      except Exception:
        self._hypothesis_failed = True
        raise

  return shard_aware_inner_test_fn


def _shard_aware_test(test, shard_index):
  """包装一个非 Hypothesis 测试，使其在错误的分片上被跳过。

  `HypothesisShardedTestLoader` 绕过了 absltest 常规的轮转分片，因此
  `HypothesisShardedTestCase` 中的非 Hypothesis 测试原本会在每个分片上运行。
  这个包装器在方法层级上重新实现了轮转。
  """

  @functools.wraps(test)
  def shard_aware_test_fn(*args, **kwargs):
    if _TEST_SHARD_INDEX != shard_index:
      return unittest.skip(f"Running on shard {shard_index}")(test)(
          *args, **kwargs
      )
    else:
      return test(*args, **kwargs)

  return shard_aware_test_fn


def _apply_sharding_to_tests(test_runner):

  shards_index_iter = itertools.cycle(range(_TEST_TOTAL_SHARDS))
  for name in dir(test_runner):
    if name.startswith("test"):
      test = getattr(test_runner, name)
      if detection.is_hypothesis_test(test):
        handle = test.hypothesis
        assert isinstance(handle, hp.core.HypothesisHandle)
        # 不支持 `@given(..., data())`，原因如下：
        # - 分片要求所有抽取的参数都是已知值。
        # - 必须在给定分片的情况下调用测试主体。
        # - 使用 `data()` 时，部分或全部参数要到调用测试主体
        #   时才可知。
        for val in handle._given_kwargs.values():
          if isinstance(val, hps_internal_core.DataStrategy):
            raise ValueError(
                "Sharded hypothesis runner does not support `data()` inside"
                " `@given`. All parameters must be drawn before the test body"
                " is called. Consider using `@composite` instead."
            )
        handle.inner_test = _shard_aware_hypothesis_inner_test(
            handle.inner_test
        )
      else:
        # 如果这些测试不是 hypothesis 测试（或者我们不对
        # hypothesis 测试做分片），我们只需以轮转方式把它们
        # 分配到各分片即可。
        if _TEST_TOTAL_SHARDS > 1:
          shard_index = next(shards_index_iter)
          setattr(test_runner, name, _shard_aware_test(test, shard_index))


class HypothesisShardedTestCase(jtu.JaxTestCase):
  """以分片方式运行 Hypothesis 测试。

  hypothesis 的工作方式是：它会使用不同的参数来运行同一个
  测试函数，并且还会使用相同的参数把同一个测试函数重复
  运行多次。

  这使 Bazel 测试分片难以生效，因为测试加载器把该测试函数视为
  单个测试——于是会把它调度到单个分片上。这样一来，一个含
  300 个示例的 hypothesis 测试就只会运行在一个分片上，耗时
  很长。

  本类绕开这个问题的方式是：在每个分片上都运行
  每个 hypothesis 测试，然后过滤掉那些不属于
  当前分片的测试。

  必须与 `HypothesisShardedTestLoader` 配合使用。
  """

  _thread_shard_index: int = 0
  _thread_total_shards: int = 1

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if _TEST_TOTAL_SHARDS > 1 or TEST_NUM_THREADS.value > 1:
      _apply_sharding_to_tests(cls)


class HypothesisShardedTestLoader(JaxTestLoader):
  """一个绕过方法级分片的 `TestLoader`。

  与 `jtu.HypothesisShardedTestCase` 配合使用，为较慢的 hypothesis 测试
  实现内层测试分片。
  """

  def getTestCaseNames(self, testCaseClass):
    self._current_test_class = testCaseClass
    return super().getTestCaseNames(testCaseClass)

  def shardTestCaseNames(self, iterator, ordered_names, shard_index):
    if issubclass(self._current_test_class, HypothesisShardedTestCase):
      return ordered_names
    return super().shardTestCaseNames(iterator, ordered_names, shard_index)

  def loadTestsFromTestCase(self, testCaseClass):
    num_threads = TEST_NUM_THREADS.value
    if (
        issubclass(testCaseClass, HypothesisShardedTestCase)
        and num_threads > 1
    ):
      cases: list[unittest.TestCase] = []
      for name in self.getTestCaseNames(testCaseClass):
        test_method = getattr(testCaseClass, name, None)
        if test_method is not None and detection.is_hypothesis_test(test_method):
          for thread_idx in range(num_threads):
            thread_name = f"{name}__thread_{thread_idx}"
            setattr(testCaseClass, thread_name, test_method)
            case = testCaseClass(thread_name)
            case._thread_shard_index = thread_idx
            case._thread_total_shards = num_threads
            cases.append(case)
        else:
          cases.append(testCaseClass(name))
      return self.suiteClass(cases)
    return super().loadTestsFromTestCase(testCaseClass)


def hypothesis_is_thread_safe() -> bool:
  """如果安装的 hypothesis 版本是线程安全的，则返回 True。

  Hypothesis 6.136.9 及以上的版本是线程安全的。
  """
  return tuple(int(x) for x in hp.__version__.split(".")) >= (6, 136, 9)


def setup_hypothesis(max_examples=30) -> None:
  """设置 hypothesis 的各配置档。

  设置 hypothesis 测试配置档，并选用由 ``JAX_HYPOTHESIS_PROFILE``
  环境变量（或 ``--jax_hypothesis_profile`` 配置项）所指定的
  那一个。

  Args:
    max_examples: 使用默认的 "deterministic" 配置档时，尝试的 hypothesis
      示例数量上限。
  """
  # 在我们的测试中，经常使用类变量略有不同的子类，
  # 来生成整套参数化测试，但这种方式不能很好地
  # 与 Hypothesis 数据库配合，因为后者会用方法标识的某种
  # 函数来生成键。但是，如果方法定义在超类中，
  # 所有子类就会共享同一个键。这种键冲突可能导致
  # 在其他健康检查中出现令人困惑的误报。
  #
  # 不过据我所知，只要我们不使用示例数据库，
  # 抑制这项健康检查就应该是完全安全的。这似乎
  # 比改写那些会触发该行为的测试更简单。更多
  # 背景见 https://github.com/HypothesisWorks/hypothesis/issues/3446
  # 的末尾。
  suppressed_checks = []
  if hasattr(hp.HealthCheck, "differing_executors"):
    suppressed_checks.append(hp.HealthCheck.differing_executors)
  if jtu.is_asan() or jtu.is_msan() or jtu.is_tsan():
    suppressed_checks.append(hp.HealthCheck.too_slow)

  hp.settings.register_profile(
      "deterministic",
      database=None,
      derandomize=True,
      deadline=None,
      max_examples=max_examples,
      print_blob=True,
      suppress_health_check=suppressed_checks,
  )
  hp.settings.register_profile(
      "interactive",
      parent=hp.settings.load_profile("deterministic"),
      max_examples=1,
      report_multiple_bugs=False,
      verbosity=hp.Verbosity.verbose,
      # 不要尝试做收缩
      phases=(
          hp.Phase.explicit,
          hp.Phase.reuse,
          hp.Phase.generate,
          hp.Phase.target,
          hp.Phase.explain,
      ),
  )
  profile = HYPOTHESIS_PROFILE.value
  logging.info("Using hypothesis profile: %s", profile)
  hp.settings.load_profile(profile)
