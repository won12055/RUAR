from __future__ import annotations

import torch

import verl
import verl.utils.torch_functional as verl_F


def _response_lengths(eos_mask: torch.Tensor) -> torch.Tensor:
    return eos_mask.sum(dim=-1).float()


def _correct_group_sigmoid_length_penalty(
    data: verl.DataProto,
    response_lengths: torch.Tensor,
    n_samples: int,
    cfg,
):
    eps = float(cfg.get("eps", 1e-6))
    grouped = response_lengths.view(-1, n_samples)

    if "acc" not in data.batch:
        raise ValueError("RUAR length penalty requires data.batch['acc']")

    std_floor = float(cfg.get("std_floor", 0.0))
    correct = (data.batch["acc"].float().view(-1, n_samples) > 0.5).float()
    correct_count = correct.sum(dim=-1, keepdim=True)
    safe_count = torch.clamp(correct_count, min=1.0)
    group_mean = (grouped * correct).sum(dim=-1, keepdim=True) / safe_count
    centered = (grouped - group_mean) * correct
    group_var = (centered * centered).sum(dim=-1, keepdim=True) / safe_count
    group_std = torch.sqrt(torch.clamp(group_var, min=0.0))
    denom = torch.clamp(group_std, min=max(std_floor, eps))
    penalty = torch.sigmoid((grouped - group_mean) / denom)
    metrics = {
        "length_penalty/correct_group_response_length_mean": group_mean.mean().detach().item(),
        "length_penalty/correct_group_response_length_std": group_std.mean().detach().item(),
        "length_penalty/correct_group_count_mean": correct_count.mean().detach().item(),
        "length_penalty/std_floor": std_floor,
    }
    return penalty.reshape_as(response_lengths), metrics


def _length_penalty(data: verl.DataProto, eos_mask: torch.Tensor, config):
    cfg = config.algorithm.get("length_penalty", {})
    if not cfg.get("enable", False):
        return None, {}

    coef = float(cfg.get("coef", 0.0))
    if coef == 0.0:
        return None, {}

    n_samples = int(config.actor_rollout_ref.rollout.n)
    response_lengths = _response_lengths(eos_mask)
    if response_lengths.numel() % n_samples != 0:
        raise ValueError(f"Batch size {response_lengths.numel()} is not divisible by rollout.n={n_samples}")

    penalty, extra_metrics = _correct_group_sigmoid_length_penalty(data, response_lengths, n_samples, cfg)

    if cfg.get("only_correct", True) and "acc" in data.batch:
        penalty = penalty * data.batch["acc"].float()

    penalty = penalty * coef
    metrics = {
        "length_penalty/mean": penalty.mean().detach().item(),
        "length_penalty/max": penalty.max().detach().item(),
        "length_penalty/coef": coef,
        **extra_metrics,
    }
    return penalty, metrics


def _attach_metrics(data: verl.DataProto, metrics: dict) -> None:
    data.meta_info["adv_metrics"] = {
        **data.meta_info.get("adv_metrics", {}),
        **metrics,
    }


def _masked_leave_one_out(reward_tensor_original: torch.Tensor, mask_tensor: torch.Tensor, n_samples: int):
    if n_samples <= 1:
        raise ValueError(f"RLOO requires n_samples > 1, got {n_samples}")
    if reward_tensor_original.shape[0] % n_samples != 0:
        raise ValueError(f"Batch size {reward_tensor_original.shape[0]} is not divisible by rollout.n={n_samples}")

    reward_tensor = reward_tensor_original.clone()
    reward_tensor[~mask_tensor] = 0
    for start_pos in range(0, reward_tensor.shape[0], n_samples):
        current_rewards = torch.cat([
            reward_tensor[pos:pos + 1][mask_tensor[pos:pos + 1]].mean(dim=0, keepdim=True)
            for pos in range(start_pos, start_pos + n_samples)
        ], dim=0)
        current_reward_sum = current_rewards.sum()
        current_reward_baseline = current_reward_sum / (n_samples - 1)
        reward_tensor[start_pos:start_pos + n_samples][mask_tensor[start_pos:start_pos + n_samples]] = (
            reward_tensor[start_pos:start_pos + n_samples][mask_tensor[start_pos:start_pos + n_samples]]
            * (n_samples / (n_samples - 1))
            - current_reward_baseline
        )

    return reward_tensor


def compute_rloo_advantage_return(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config):
    reward_tensors = []
    with torch.no_grad():
        if "acc" in data.batch and config.algorithm.reward_gt_coef != 0.0:
            reward_tensor = torch.zeros_like(eos_mask, dtype=torch.float32)
            reward_mask = torch.zeros_like(eos_mask, dtype=torch.bool)

            prompt_ids = data.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(-1)
            terminal_rows = torch.arange(
                0,
                valid_response_length.shape[0],
                dtype=torch.long,
                device=valid_response_length.device,
            )
            reward_mask[terminal_rows, valid_response_length - 1] = True
            reward_tensor[terminal_rows, valid_response_length - 1] = data.batch["acc"].float()
            reward_tensors.append(
                _masked_leave_one_out(reward_tensor, reward_mask, n_samples) * config.algorithm.reward_gt_coef
            )

        length_penalty, length_metrics = _length_penalty(data, eos_mask, config)
        if length_penalty is not None:
            reward_tensor = torch.zeros_like(eos_mask, dtype=torch.float32)
            reward_mask = torch.zeros_like(eos_mask, dtype=torch.bool)

            prompt_ids = data.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(-1)
            terminal_rows = torch.arange(
                0,
                valid_response_length.shape[0],
                dtype=torch.long,
                device=valid_response_length.device,
            )
            reward_mask[terminal_rows, valid_response_length - 1] = True
            reward_tensor[terminal_rows, valid_response_length - 1] = -length_penalty
            reward_tensors.append(_masked_leave_one_out(reward_tensor, reward_mask, n_samples))
            _attach_metrics(data, length_metrics)

        if not reward_tensors:
            raise ValueError("RLOO requires at least one enabled reward source.")

        final_reward_tensor = sum(reward_tensors)
        returns = (final_reward_tensor * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        scaling_cfg = config.algorithm.get("advantage_scaling", {})
        if (
            scaling_cfg.get("enable", False)
            and "adv_gamma_pos" in data.batch
            and "adv_gamma_neg" in data.batch
        ):
            pos_gamma = data.batch["adv_gamma_pos"].float()
            neg_gamma = data.batch["adv_gamma_neg"].float()
            token_gamma = torch.where(returns >= 0, pos_gamma, neg_gamma)
            if scaling_cfg.get("only_correct", False) and "acc" in data.batch:
                correct_mask = (data.batch["acc"].float() > 0.5).unsqueeze(-1)
                token_gamma = torch.where(correct_mask, token_gamma, torch.ones_like(token_gamma))
            token_gamma = token_gamma * eos_mask
            active = eos_mask > 0
            changed = active & ((token_gamma - 1.0).abs() > 1e-6)
            scaled = active & (token_gamma < 0.999)
            scaling_metrics = {
                "advantage_scaling/gamma_mean": token_gamma[active].mean().detach().item() if active.any() else 1.0,
                "advantage_scaling/scaled_token_rate": (
                    scaled[active].float().mean().detach().item() if active.any() else 0.0
                ),
                "advantage_scaling/changed_token_rate": (
                    changed[active].float().mean().detach().item() if active.any() else 0.0
                ),
                "advantage_scaling/positive_gamma_mean": (
                    pos_gamma[active].mean().detach().item() if active.any() else 1.0
                ),
                "advantage_scaling/negative_gamma_mean": (
                    neg_gamma[active].mean().detach().item() if active.any() else 1.0
                ),
            }
            returns = returns * token_gamma
            _attach_metrics(data, scaling_metrics)

        advantages = verl_F.masked_whiten(returns.clone(), eos_mask)
        return advantages, returns


def compute_return_abs_accuracy(returns, acc):
    return (torch.sign(returns[:, 0]) == torch.sign(acc * 2 - 1)).float().mean()


def compute_return_smoothness(returns):
    return ((returns[:, :-1] - returns[:, 1:]) ** 2).sum(dim=-1).mean()
