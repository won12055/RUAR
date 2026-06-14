# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import torch

import verl.utils.torch_functional as verl_F


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, clipranges):
    """PPO clipped policy-gradient loss used by RUAR actor updates."""

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clipranges[0], 1.0 + clipranges[1])
    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Return the token-level KL penalty used when actor.use_kl_loss=True."""

    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        return torch.exp(kl) - kl - 1
    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}")
