# Copyright 2022 The JAX Authors.
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

# 文件职责：为 JAX 自定义导数与自定义分片等自定义 API 提供装饰器包装器的注册表与属性转发基建。
# `register_custom_decorator_type` 把包装器类登记到内部注册表 `_custom_wrapper_types`，
# 供 `jax.custom_jvp`、`jax.custom_vjp`、`jax.custom_batching.custom_vmap`
# 以及 `jax.custom_partitioning` 等自定义变换 API 使用。
# `forward_attr` 充当包装器的 `__getattr__`：名字以 `def` 开头时转发到被包装的 `fun`，
# 从而让包装后的对象保留原函数的 `__name__`、`__doc__` 等元信息，其余名字抛 `AttributeError`。


_custom_wrapper_types = set()

def register_custom_decorator_type(cls):
  _custom_wrapper_types.add(cls)
  return cls

def forward_attr(self_, name):
  if name.startswith('def'):
    return getattr(self_.fun, name)
  else:
    raise AttributeError
