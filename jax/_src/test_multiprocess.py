# Copyright 2025 The JAX Authors.
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

# 文件职责：为 JAX 的多进程（多 worker）测试提供统一的启动与协调辅助。
# 它在主进程里按 flag 指定的数量拉起多个子进程，为每个子进程分配各自的
# GPU 或 TPU 芯片，并借助 `jax._src.distributed` 初始化控制器与协调服务，
# 最后收集、汇总各子进程的 stdout/stderr 日志并判定整体成败；
# 子进程（worker）复用同一入口初始化分布式运行时并执行测试用例。
# 对外主要暴露 `main()` 入口、`MultiProcessTest` 基类以及一组 absl flags。

"""用于运行多进程测试的辅助工具。"""

import functools
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

from absl import app
import absl.flags
from absl.testing import absltest
from absl.testing import parameterized

from jax._src import distributed
from jax._src import xla_bridge as xb
from jax._src import test_util as jtu
from jax._src.config import config
from jax._src.lib import cuda_versions
from jax._src.lib import _jax
from jax._src import hardware_utils
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm

try:
  import portpicker
except ImportError:
  portpicker = None

NUM_PROCESSES = absl.flags.DEFINE_integer(
    "num_processes", None, "Number of processes to use."
)

_GPUS_PER_PROCESS = absl.flags.DEFINE_integer(
    "gpus_per_process",
    0,
    "Number of GPUs per worker process.",
)

_TPU_CHIPS_PER_PROCESS = absl.flags.DEFINE_integer(
    "tpu_chips_per_process",
    0,
    "Number of TPU chips per worker process.",
)

CPU_COLLECTIVES_IMPLEMENTATION = absl.flags.DEFINE_string(
    "cpu_collectives_implementation",
    "",
    "CPU collectives implementation to use. Uses default if empty.",
)

EXTRA_TEST_ARGS = absl.flags.DEFINE_multi_string(
    "extra_test_args", [], "Extra flags to pass to worker process."
)

# 仅供内部使用。
MULTIPROCESS_TEST_WORKER_ID = absl.flags.DEFINE_integer(
    "multiprocess_test_worker_id",
    -1,
    "Worker id. Set by main test process; should not be set by users.",
)

_MULTIPROCESS_TEST_CONTROLLER_ADDRESS = absl.flags.DEFINE_string(
    "multiprocess_test_controller_address",
    "",
    "Address of the JAX controller. Set by the main test process; should not be"
    " set by users.",
)

_DEVICE_IDS = absl.flags.DEFINE_list(
    "device_ids",
    None,
    "List of device ids to use. Set by main test process; should not be set by"
    " users.",
)

_ENABLE_MEGASCALE = absl.flags.DEFINE_bool(
    "enable_megascale", False, "If true, enable Megascale runtime."
)

_HEARTBEAT_TIMEOUT = absl.flags.DEFINE_integer(
    "heartbeat_timeout",
    30,
    "Timeout in seconds for heartbeat checks. Set to a higher number when"
    " running under sanitizers.",
)

_SHUTDOWN_TIMEOUT = absl.flags.DEFINE_integer(
    "shutdown_timeout",
    30,
    "JAX shutdown timeout duration in seconds for each subprocess worker. If "
    "your test is timing out, try increasing this value.",
)

_BARRIER_TIMEOUT = absl.flags.DEFINE_integer(
    "barrier_timeout",
    120,
    "Barrier timeout in seconds. Set to a higher number when running under"
    " sanitizers.",
)

_INITIALIZATION_TIMEOUT = absl.flags.DEFINE_integer(
    "initialization_timeout",
    60,
    "Coordination service initialization timeout in seconds. Set to a higher"
    " number when running under sanitizers.",
)

_DUMP_HLO = absl.flags.DEFINE_bool(
    "dump_hlo",
    False,
    "If true, dump per-process HLO to undeclared outputs. They will show up in"
    " sponge artifacts under the directory 'jax_%process_idx%_hlo_dump'.",
)

expect_failures_with_regex = None


def main(shard_main=None):
  config.config_with_absl()
  app.run(functools.partial(_main, shard_main=shard_main))


class GracefulKiller:
  """添加一个信号处理器：捕获到 SIGINT 或 SIGTERM 时设置标志。"""

  # 源自 https://stackoverflow.com/a/31464349
  kill_now = False

  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, sig_num, unused_stack_frame):
    print(f"Caught signal: {signal.Signals(sig_num).name} ({sig_num})")
    self.kill_now = True


def _main(argv, shard_main):
  # TODO(emilyaf): 在 Windows 上启用多进程测试。
  if sys.platform == "win32":
    print("Multiprocess tests are not supported on Windows.")
    return

  _, tpu_version = hardware_utils.num_available_tpu_chips_and_device_id()
  if running_in_cloud_tpu_vm and tpu_version in (
      hardware_utils.TpuVersion.v4,
      hardware_utils.TpuVersion.v5e,
  ):
    print(f"Skipping multiprocess tests on TPU {tpu_version.name} in Cloud.")
    return
  num_processes = NUM_PROCESSES.value
  if MULTIPROCESS_TEST_WORKER_ID.value >= 0:
    local_device_ids = _DEVICE_IDS.value
    if local_device_ids is not None:
      local_device_ids = [int(device_id) for device_id in local_device_ids]
    distributed.initialize(
        _MULTIPROCESS_TEST_CONTROLLER_ADDRESS.value,
        num_processes=num_processes,
        process_id=MULTIPROCESS_TEST_WORKER_ID.value,
        local_device_ids=local_device_ids,
        heartbeat_timeout_seconds=_HEARTBEAT_TIMEOUT.value,
        shutdown_timeout_seconds=_SHUTDOWN_TIMEOUT.value,
        initialization_timeout=_INITIALIZATION_TIMEOUT.value,
    )
    if shard_main is not None:
      return shard_main()
    return absltest.main(testLoader=jtu.JaxTestLoader())

  if not argv[0].endswith(".py"):  # 若存在解释器路径则跳过。
    argv = argv[1:]

  if num_processes is None:
    raise ValueError("num_processes must be set")
  gpus_per_process = _GPUS_PER_PROCESS.value
  tpu_chips_per_process = _TPU_CHIPS_PER_PROCESS.value
  num_tpu_chips = num_processes * tpu_chips_per_process
  if num_tpu_chips == 0:
    tpu_host_bounds = ""
    tpu_chips_per_host_bounds = ""
  elif num_tpu_chips == 1:
    assert tpu_chips_per_process == 1
    tpu_host_bounds = "1,1,1"
    tpu_chips_per_host_bounds = "1,1,1"
  elif num_tpu_chips == 4:
    if tpu_chips_per_process == 1:
      tpu_host_bounds = "2,2,1"
      tpu_chips_per_host_bounds = "1,1,1"
    elif tpu_chips_per_process == 2:
      tpu_host_bounds = "2,1,1"
      tpu_chips_per_host_bounds = "1,2,1"
    elif tpu_chips_per_process == 4:
      tpu_host_bounds = "1,1,1"
      tpu_chips_per_host_bounds = "2,2,1"
    else:
      raise ValueError(
          "Invalid number of TPU chips per worker {}".format(
              tpu_chips_per_process
          )
      )
  elif num_tpu_chips == 8:
    if tpu_chips_per_process == 1:
      tpu_host_bounds = "4,2,1"
      tpu_chips_per_host_bounds = "1,1,1"
    elif tpu_chips_per_process == 4:
      # 注意：该分支假定使用的是 2x4 的 v6e LitePod，
      # 在 4x2 的 v5e LitePod 上无法工作。
      tpu_host_bounds = "1,2,1"
      tpu_chips_per_host_bounds = "2,2,1"
    elif tpu_chips_per_process == 8:
      tpu_host_bounds = "1,1,1"
      tpu_chips_per_host_bounds = "2,4,1"
    else:
      # TODO(phawkins): 实现其他情况。
      raise ValueError(
          "Invalid number of TPU chips per worker {}".format(
              tpu_chips_per_process
          )
      )
  else:
    raise ValueError(f"Invalid number of TPU chips {num_tpu_chips}")

  if portpicker is None:
    slicebuilder_ports = [10000 + i for i in range(num_processes)]
  else:
    portserver_address = os.environ.get("JAX_PORTSERVER_ADDRESS")
    slicebuilder_ports = [
        portpicker.pick_unused_port(portserver_address=portserver_address)
        for _ in range(num_processes)
    ]
  slicebuilder_addresses = ",".join(
      f"localhost:{port}" for port in slicebuilder_ports
  )
  megascale_coordinator_port = None

  if gpus_per_process > 0:
    # 在不初始化运行时的情况下，获取本进程可见的 GPU 数量
    if cuda_versions is not None:
      local_device_count = cuda_versions.cuda_device_count()
      if num_processes * gpus_per_process > local_device_count:
        print(
          f"Cannot run {num_processes} processes with {gpus_per_process} GPU(s) "
          f"each on a system with only {local_device_count} local GPU(s), "
          f"starting {local_device_count // gpus_per_process} instead - test "
          "cases will likely be skipped!"
        )
        num_processes = local_device_count // gpus_per_process

  if portpicker is None:
    jax_port = 9876
  else:
    # TODO(emilyaf): 如果因各测试间 pick_unused_port() 竞争而出现偶发端口冲突，
    # 就改用端口服务器。
    portserver_address = os.environ.get("JAX_PORTSERVER_ADDRESS")
    jax_port = portpicker.pick_unused_port(portserver_address=portserver_address)
  subprocesses = []
  output_filenames = []
  output_files = []
  sys_path = os.pathsep.join(sys.path)

  for i in range(num_processes):
    device_ids = None
    env = os.environ.copy()

    # 注意：这是针对 rules_python >= 1.7.0（Strict Hermeticity，严格封闭性）的修复：
    # 父进程通过 sys.path 看到依赖，但新版 rules_python
    # 默认不会把它导出到 PYTHONPATH。我们必须手动传递，
    # 以便子工作进程能够定位依赖。
    path_parts = [sys_path, env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(p for p in path_parts if p)

    args = [
        "/proc/self/exe",
        *argv,
        f"--num_processes={num_processes}",
        f"--multiprocess_test_worker_id={i}",
        f"--multiprocess_test_controller_address=localhost:{jax_port}",
        f"--heartbeat_timeout={_HEARTBEAT_TIMEOUT.value}",
        f"--shutdown_timeout={_SHUTDOWN_TIMEOUT.value}",
        f"--barrier_timeout={_BARRIER_TIMEOUT.value}",
        f"--initialization_timeout={_INITIALIZATION_TIMEOUT.value}",
        "--logtostderr",
    ]

    if num_tpu_chips > 0:
      device_ids = range(
          i * tpu_chips_per_process, (i + 1) * tpu_chips_per_process)
      env["CLOUD_TPU_TASK_ID"] = str(i)
      env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = tpu_chips_per_host_bounds
      env["TPU_PROCESS_BOUNDS"] = tpu_host_bounds
      env["TPU_PROCESS_ADDRESSES"] = slicebuilder_addresses
      env["TPU_PROCESS_PORT"] = str(slicebuilder_ports[i])
      env["TPU_VISIBLE_CHIPS"] = ",".join(map(str, device_ids))
      env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"

    if gpus_per_process > 0:
      device_ids = range(i * gpus_per_process, (i + 1) * gpus_per_process)
      args.append(f"--jax_cuda_visible_devices={','.join(map(str, device_ids))}")

    if device_ids is not None:
      args.append(f"--device_ids={','.join(map(str, device_ids))}")

    cpu_collectives_impl = CPU_COLLECTIVES_IMPLEMENTATION.value
    if cpu_collectives_impl:
      args.append(
          f"--jax_cpu_collectives_implementation={cpu_collectives_impl}"
      )

    if _ENABLE_MEGASCALE.value or cpu_collectives_impl == "megascale":
      if portpicker is None:
        megascale_port = 9877
      else:
        portserver_address = os.environ.get("JAX_PORTSERVER_ADDRESS")
        megascale_port = portpicker.pick_unused_port(
            portserver_address=portserver_address
        )
      if megascale_coordinator_port is None:
        megascale_coordinator_port = megascale_port
      args += [
          f"--megascale_coordinator_address=localhost:{megascale_coordinator_port}",
          f"--megascale_port={megascale_port}",
      ]

    args += EXTRA_TEST_ARGS.value

    undeclared_outputs = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")
    stdout_name = f"{undeclared_outputs}/jax_{i}_stdout.log"
    stderr_name = f"{undeclared_outputs}/jax_{i}_stderr.log"

    if _DUMP_HLO.value:
      hlo_dump_path = f"{undeclared_outputs}/jax_{i}_hlo_dump/"
      os.makedirs(hlo_dump_path, exist_ok=True)
      env["XLA_FLAGS"] = f"--xla_dump_to={hlo_dump_path}"

    stdout = open(stdout_name, "wb")
    stderr = open(stderr_name, "wb")
    print(f"Launching process {i}:")
    print(f"  stdout: {stdout_name}")
    print(f"  stderr: {stderr_name}")
    proc = subprocess.Popen(args, env=env, stdout=stdout, stderr=stderr)
    subprocesses.append(proc)
    output_filenames.append((stdout_name, stderr_name))
    output_files.append((stdout, stderr))

  print(" All launched, running ".center(80, "="), flush=True)

  # 等待所有子进程结束，或等待来自 bazel 的 SIGTERM。若收到
  # SIGTERM，我们仍希望收集它们的日志，因此先杀掉它们再继续。
  killer = GracefulKiller()
  running_procs = dict(enumerate(subprocesses))
  while not killer.kill_now and running_procs:
    time.sleep(0.1)
    for i, proc in list(running_procs.items()):
      if proc.poll() is not None:
        print(f"Process {i} finished.", flush=True)
        running_procs.pop(i)
  if killer.kill_now and running_procs:
    print("Caught termination, terminating remaining children.", flush=True)

    # 向每个子进程发送 SIGTERM，通知它应当终止。
    for i, proc in running_procs.items():
      proc.terminate()
      print(f"Process {i} terminated.", flush=True)

    # 我们给子进程几秒钟做自身的清理，
    # 余下时间（最多 15 秒）用于把子进程日志复制到我们自己的输出中。
    time.sleep(5)

    # 向每个子进程发送 SIGKILL（“硬”杀）。这至关重要：若不这样做，
    # 本进程可能长时间阻塞在下面的 proc.wait() 上，
    # 以致永远无法保存子进程日志，
    # 从而使测试超时问题变得非常难以调试。
    for i, proc in running_procs.items():
      proc.kill()
      print(f"Process {i} killed.")
    print("Killed all child processes.", flush=True)

  retvals = []
  stdouts = []
  stderrs = []
  for proc, fds, (stdout, stderr) in zip(
      subprocesses, output_files, output_filenames
  ):
    retvals.append(proc.wait())
    for fd in fds:
      fd.close()
    stdouts.append(pathlib.Path(stdout).read_text(errors="replace"))
    stderrs.append(pathlib.Path(stderr).read_text(errors="replace"))

  print(" All finished ".center(80, "="), flush=True)

  print(" Summary ".center(80, "="))
  for i, (retval, stdout, stderr) in enumerate(zip(retvals, stdouts, stderrs)):
    m = re.search(r"Ran \d+ tests? in [\d.]+s\n\n.*", stderr, re.MULTILINE)
    result = m.group().replace("\n\n", "; ") if m else "Test crashed?"
    print(
        f"Process {i}, ret: {retval}, len(stdout): {len(stdout)}, "
        f"len(stderr): {len(stderr)}; {result}"
    )

  print(" Detailed logs ".center(80, "="))
  for i, (retval, stdout, stderr) in enumerate(zip(retvals, stdouts, stderrs)):
    print(f" Process {i}: return code: {retval} ".center(80, "="))
    if stdout:
      print(f" Process {i} stdout ".center(80, "-"))
      print(stdout)
    if stderr:
      print(f" Process {i} stderr ".center(80, "-"))
      print(stderr)

  print(" Done detailed logs ".center(80, "="), flush=True)
  for i, (retval, stderr) in enumerate(zip(retvals, stderrs)):
    if retval != 0:
      if expect_failures_with_regex is not None:
        assert re.search(
            expect_failures_with_regex, stderr
        ), f"process {i} failed, expected regex: {expect_failures_with_regex}"
      else:
        assert retval == 0, f"process {i} failed, return value: {retval}"


class MultiProcessTest(parameterized.TestCase):

  def setUp(self):
    """让各测试一起开始。"""
    super().setUp()
    if xb.process_count() == 1:
      self.skipTest("Test requires multiple processes.")
    assert xb.process_count() == NUM_PROCESSES.value, (
        xb.process_count(),
        NUM_PROCESSES.value,
    )
    # 确保所有进程都处于同一个测试用例。
    client = distributed.global_state.client
    if client is None:
      raise TypeError("client cannot be None")
    try:
      client.wait_at_barrier(
          f"{self._testMethodName}_start", _BARRIER_TIMEOUT.value * 1000)
    except _jax.JaxRuntimeError as e:
      msg, *_ = e.args
      if msg.startswith("DEADLINE_EXCEEDED"):
        raise RuntimeError(
            f"Init or some test executed earlier than {self._testMethodName} "
            "failed. Check logs from earlier tests to debug further. We "
            "recommend debugging that specific failed test with "
            "`--test_filter` before running the full test suite again."
        ) from e

  def tearDown(self):
    """让各测试一起结束。"""
    client = distributed.global_state.client
    if client is None:
      raise TypeError("client cannot be None")
    # 对于一部分进程运行不同测试断言的测试，确保它们的命运与共
    # （即某些进程可能通过、某些进程可能失败，
    # 但整体测试应当失败）。
    try:
      client.wait_at_barrier(
          f"{self._testMethodName}_end", _BARRIER_TIMEOUT.value * 1000)
    except _jax.JaxRuntimeError as e:
      msg, *_ = e.args
      if msg.startswith("DEADLINE_EXCEEDED"):
        raise RuntimeError(
            f"Test {self._testMethodName} failed in another process.  We "
            "recommend debugging that specific failed test with "
            "`--test_filter` before running the full test suite again."
        ) from e
    super().tearDown()
