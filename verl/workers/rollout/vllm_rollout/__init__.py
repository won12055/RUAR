# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib.metadata import PackageNotFoundError, version

from packaging.version import parse as parse_version

try:
    package_version = version("vllm")
except PackageNotFoundError as exc:
    raise ModuleNotFoundError("RUAR training requires vLLM.") from exc

if parse_version(package_version) < parse_version("0.7.0"):
    raise ValueError(f"RUAR public training requires vLLM >= 0.7.0, found {package_version}.")

from .vllm_rollout_spmd import vLLMRollout

__all__ = ["vLLMRollout"]
