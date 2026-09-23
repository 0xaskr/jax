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

# 文件职责：为 JAX 的测试提供自定义的 unittest 加载器与并行测试套件。
# 通过 config 标志暴露 JAX_TEST_TARGETS / JAX_EXCLUDE_TEST_TARGETS，
# 在收集测试用例名时用正则做包含与排除过滤。
# 当 JAX_TEST_NUM_THREADS >= 1 时，JaxTestSuite 用线程池并行执行线程安全
# 的测试，并借助 ThreadSafeTestResult 在测试结束时加锁批量回放结果；
# 线程不安全的用例则退回主线程串行执行。

"""
包含自定义的 unittest 加载器与测试套件。

实现：
- 基于 JAX_TEST_TARGETS 与 JAX_EXCLUDE_TEST_TARGETS 环境变量的
  测试过滤。
- 当 JAX_TEST_NUM_THREADS >= 1 时，使用线程并行运行测试的
  测试套件。
- 将测试用例或测试类标记为线程不安全的测试装饰器。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import threading
import time
import unittest

from absl.testing import absltest
from jax._src import config
from jax._src import test_warning_util

logger = logging.getLogger(__name__)


_TEST_TARGETS = config.string_flag(
  'test_targets', os.getenv('JAX_TEST_TARGETS', ''),
  'Regular expression specifying which tests to run, called via re.search on '
  'the test name. If empty or unspecified, run all tests.'
)

_EXCLUDE_TEST_TARGETS = config.string_flag(
  'exclude_test_targets', os.getenv('JAX_EXCLUDE_TEST_TARGETS', ''),
  'Regular expression specifying which tests NOT to run, called via re.search '
  'on the test name. If empty or unspecified, run all tests.'
)

TEST_NUM_THREADS = config.int_flag(
    'jax_test_num_threads', int(os.getenv('JAX_TEST_NUM_THREADS', '0')),
    help='Number of threads to use for running tests. 0 means run everything '
    'in the main thread. Using > 1 thread is experimental.'
)




def thread_unsafe_test(condition: bool = True):
  """用于标记非线程安全测试的装饰器。

  Args:
    condition: 若为 True，则把该测试标记为线程不安全；若为 False，该测试
      可以与其他测试并行运行。默认为 True。
  """
  def decorator(func):
    setattr(func, "thread_unsafe", condition)
    return func
  return decorator


def thread_unsafe_test_class(condition: bool = True):
  """将某个 TestCase 类标记为线程不安全的装饰器。

  Args:
    condition: 若为 True，则把该测试类标记为线程不安全；若为 False，该
      测试类照常运行。默认为 True。
  """
  def f(klass):
    assert issubclass(klass, unittest.TestCase), type(klass)
    klass.thread_unsafe = condition
    return klass
  return f


class ThreadSafeTestResult:
  """
  包装一个 TestResult 使其线程安全。

  做法是累积 API 调用，并在每个测试用例结束时于锁的保护下批量
  应用它们。

  我们采用鸭子类型而不是继承 TestResult，因为我们实际上并不是
  TestResult 的完整实现，对于尚未实现的部分，我们更希望得到
  明显的报错。
  """
  def __init__(self, lock: threading.Lock, result: unittest.TestResult):
    self.lock = lock
    self.test_result = result
    self.actions: list[Callable[[], None]] = []

  def startTest(self, test: unittest.TestCase):
    logger.info("Test start: %s", test.id())
    self.start_time = time.time()

  def stopTest(self, test: unittest.TestCase):
    logger.info("Test stop: %s", test.id())
    stop_time = time.time()
    with self.lock:
      # 如果 test_result 是 ABSL 的 _TextAndXMLTestResult，我们就覆盖它
      # 获取时间的方式。这会影响 CI 消费的 XML 输出中显示的计时。
      time_getter = getattr(self.test_result, "time_getter", None)
      try:
        self.test_result.time_getter = lambda: self.start_time  # pyrefly: ignore[missing-attribute]
        self.test_result.startTest(test)
        for callback in self.actions:
          callback()
        self.test_result.time_getter = lambda: stop_time  # pyrefly: ignore[missing-attribute]
        self.test_result.stopTest(test)
      finally:
        if time_getter is not None:
          self.test_result.time_getter = time_getter  # pyrefly: ignore[missing-attribute]

  def addSuccess(self, test: unittest.TestCase):
    self.actions.append(lambda: self.test_result.addSuccess(test))

  def addSkip(self, test: unittest.TestCase, reason: str):
    self.actions.append(lambda: self.test_result.addSkip(test, reason))

  def addError(self, test: unittest.TestCase, err):
    self.actions.append(lambda: self.test_result.addError(test, err))

  def addFailure(self, test: unittest.TestCase, err):
    self.actions.append(lambda: self.test_result.addFailure(test, err))

  def addExpectedFailure(self, test: unittest.TestCase, err):
    self.actions.append(lambda: self.test_result.addExpectedFailure(test, err))

  def addDuration(self, test: unittest.TestCase, elapsed):
    self.actions.append(lambda: self.test_result.addDuration(test, elapsed))


def _is_thread_unsafe(test: unittest.TestCase) -> bool:
  if not isinstance(test, unittest.TestCase):
    return False
  if getattr(test.__class__, "thread_unsafe", False):
    return True
  method = getattr(test, test._testMethodName)
  if getattr(method, "thread_unsafe", False):
    return True
  return False


class JaxTestSuite(unittest.TestSuite):
  """当 TEST_NUM_THREADS > 1 时使用线程并行运行测试。

  注意：启用线程并行时，该测试套件不会运行 setUpClass 或 setUpModule
  方法。
  """

  def __init__(self, suite: unittest.TestSuite):
    super().__init__(list(suite))

  def run(self, result: unittest.TestResult, debug: bool = False) -> unittest.TestResult:
    if TEST_NUM_THREADS.value <= 0:
      return super().run(result)

    test_warning_util.install_threadsafe_warning_handlers()

    executor = ThreadPoolExecutor(TEST_NUM_THREADS.value)
    lock = threading.Lock()
    futures = []

    thread_safe_tests = []
    thread_unsafe_tests = []

    def partition_test(test):
      if isinstance(test, unittest.TestSuite):
        for subtest in test:
          partition_test(subtest)
      else:
        if _is_thread_unsafe(test):
          thread_unsafe_tests.append(test)
        else:
          thread_safe_tests.append(test)

    partition_test(self)

    with executor:
      for test in thread_safe_tests:
        test_result = ThreadSafeTestResult(lock, result)
        futures.append(executor.submit(test, test_result))

      for future in futures:
        future.result()

    for test in thread_unsafe_tests:
      test_result = ThreadSafeTestResult(lock, result)
      test(test_result)

    return result


class JaxTestLoader(absltest.TestLoader):
  suiteClass = JaxTestSuite  # pyrefly: ignore[bad-assignment]

  def getTestCaseNames(self, testCaseClass):
    names = super().getTestCaseNames(testCaseClass)
    if _TEST_TARGETS.value:
      pattern = re.compile(_TEST_TARGETS.value)
      names = [name for name in names
               if pattern.search(f"{testCaseClass.__name__}.{name}")]
    if _EXCLUDE_TEST_TARGETS.value:
      pattern = re.compile(_EXCLUDE_TEST_TARGETS.value)
      names = [name for name in names
               if not pattern.search(f"{testCaseClass.__name__}.{name}")]
    return names
