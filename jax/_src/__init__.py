# 文件职责：JAX 内部实现包 `jax._src` 的包初始化文件（本文件除许可证外没有任何代码）。
# 它本身不导出任何符号，唯一作用是把目录标记为 Python 包，使 `jax._src.*` 下的
# 各内部模块（追踪、原语、解释器等）可以被导入。
# 注意：`jax._src` 是私有实现层，不是公开 API；用户应使用 `jax` 顶层的公开接口，
# JAX 不保证这里的模块与符号在版本间保持兼容。

# Copyright 2020 The JAX Authors.
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
