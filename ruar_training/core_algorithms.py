from __future__ import annotations

import torch

import verl
import verl.utils.torch_functional as verl_F


def _terminal_indices(data: verl.DataProto) -> torch.Tensor:
    prompt_length = data.batch["prompts"].shape[-1]
    response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1).long()
    return torch.clamp(response_lengths, min=1) - 1


def _group_rloo(values: torch.Tensor, n_samples: int) -> torch.Tensor:
    if n_samples <= 1:
        raise ValueError(f"RLOO requires n_samples > 1, got {n_samples}")
    if values.numel() % n_samples != 0:
        raise ValueError(f"Batch size {values.numel()} is not divisible by rollout.n={n_samples}")

    grouped = values.float().view(-1, n_samples)
    group_sum = grouped.sum(dim=-1, keepdim=True)
    baseline = (group_sum - grouped) / (n_samples - 1)
    return (grouped - baseline).reshape_as(values).to(values.dtype)


def _terminal_reward(data: verl.DataProto, eos_mask: torch.Tensor, values: torch.Tensor, n_samples: int) -> torch.Tensor:
    reward = torch.zeros_like(eos_mask, dtype=torch.float32)
    terminal = _terminal_indices(data)
    rows = torch.arange(values.shape[0], device=values.device)
    reward[rows, terminal] = _group_rloo(values.float(), n_samples)
    return reward


def _response_lengths(eos_mask: torch.Tensor) -> torch.Tensor:
    return eos_mask.sum(dim=-1).float()


def _basic_length_penalty(response_lengths: torch.Tensor, cfg) -> torch.Tensor:
    target = float(cfg.get("target", 512))
    if target <= 0:
        return torch.zeros_like(response_lengths)

    mode = str(cfg.get("mode", "excess_ratio"))
    if mode == "ratio":
        penalty = response_lengths / target
    elif mode in ("excess_ratio", "capped_excess_ratio", "capped_excess", "clipped_excess_ratio"):
        penalty = torch.clamp(response_lengths - target, min=0.0) / target
        if mode != "excess_ratio":
            penalty = torch.clamp(penalty, max=float(cfg.get("cap", 1.0)))
    elif mode == "log_excess":
        denom = torch.log1p(torch.tensor(target, device=response_lengths.device, dtype=response_lengths.dtype))
        penalty = torch.log1p(torch.clamp(response_lengths - target, min=0.0)) / denom
    else:
        raise ValueError(f"Unknown length penalty mode: {mode}")
    return penalty


def _group_length_penalty(data: verl.DataProto, response_lengths: torch.Tensor, n_samples: int, cfg):
    mode = str(cfg.get("mode", "excess_ratio"))
    eps = float(cfg.get("eps", 1e-6))
    grouped = response_lengths.view(-1, n_samples)

    if mode in ("group_normalized_excess", "group_norm_excess", "normalized_excess"):
        group_mean = grouped.mean(dim=-1, keepdim=True)
        group_std = grouped.std(dim=-1, keepdim=True, unbiased=False)
        penalty = torch.clamp((grouped - group_mean) / torch.clamp(group_std, min=eps), min=0.0)
        metrics = {
            "length_penalty/group_response_length_mean": group_mean.mean().detach().item(),
            "length_penalty/group_response_length_std": group_std.mean().detach().item(),
        }
        return penalty.reshape_as(response_lengths), metrics

    if mode not in ("correct_group_sigmoid", "correct_group_normalized_sigmoid"):
        return None, {}
    if "acc" not in data.batch:
        raise ValueError(f"{mode} requires data.batch['acc']")

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

    mode = str(cfg.get("mode", "excess_ratio"))
    if mode in (
        "group_normalized_excess",
        "group_norm_excess",
        "normalized_excess",
        "correct_group_sigmoid",
        "correct_group_normalized_sigmoid",
    ):
        penalty, extra_metrics = _group_length_penalty(data, response_lengths, n_samples, cfg)
    else:
        penalty = _basic_length_penalty(response_lengths, cfg)
        extra_metrics = {}

    if cfg.get("only_correct", True) and "acc" in data.batch:
        penalty = penalty * data.batch["acc"].float()

    penalty = penalty * coef
    metrics = {
        "length_penalty/mean": penalty.mean().detach().item(),
        "length_penalty/max": penalty.max().detach().item(),
        "length_penalty/target": float(cfg.get("target", 512)),
        "length_penalty/coef": coef,
        "length_penalty/cap": float(cfg.get("cap", 0.0)),
        **extra_metrics,
    }
    return penalty, metrics


def _attach_metrics(data: verl.DataProto, metrics: dict) -> None:
    data.meta_info["adv_metrics"] = {
        **data.meta_info.get("adv_metrics", {}),
        **metrics,
    }


def _apply_advantage_scaling(data: verl.DataProto, returns: torch.Tensor, eos_mask: torch.Tensor, config):
    cfg = config.algorithm.get("advantage_scaling", {})
    if not cfg.get("enable", False):
        return returns
    if "adv_gamma_pos" not in data.batch or "adv_gamma_neg" not in data.batch:
        return returns

    pos_gamma = data.batch["adv_gamma_pos"].float()
    neg_gamma = data.batch["adv_gamma_neg"].float()
    token_gamma = torch.where(returns >= 0, pos_gamma, neg_gamma)
    if cfg.get("only_correct", False) and "acc" in data.batch:
        correct = (data.batch["acc"].float() > 0.5).unsqueeze(-1)
        token_gamma = torch.where(correct, token_gamma, torch.ones_like(token_gamma))

    token_gamma = token_gamma * eos_mask
    active = eos_mask > 0
    if active.any():
        changed = active & ((token_gamma - 1.0).abs() > 1e-6)
        scaled = active & (token_gamma < 0.999)
        _attach_metrics(data, {
            "advantage_scaling/gamma_mean": token_gamma[active].mean().detach().item(),
            "advantage_scaling/scaled_token_rate": scaled[active].float().mean().detach().item(),
            "advantage_scaling/changed_token_rate": changed[active].float().mean().detach().item(),
            "advantage_scaling/positive_gamma_mean": pos_gamma[active].mean().detach().item(),
            "advantage_scaling/negative_gamma_mean": neg_gamma[active].mean().detach().item(),
        })

    return returns * token_gamma


def compute_rloo_advantage_return(data: verl.DataProto, eos_mask: torch.Tensor, n_samples, config):
    reward_terms = []
    with torch.no_grad():
        if "acc" in data.batch and config.algorithm.reward_gt_coef != 0.0:
            reward_terms.append(
                _terminal_reward(data, eos_mask, data.batch["acc"].float(), n_samples)
                * float(config.algorithm.reward_gt_coef)
            )

        length_penalty, length_metrics = _length_penalty(data, eos_mask, config)
        if length_penalty is not None:
            reward_terms.append(_terminal_reward(data, eos_mask, -length_penalty, n_samples))
            _attach_metrics(data, length_metrics)

        if not reward_terms:
            raise ValueError("RLOO requires at least one enabled reward source.")

        token_reward = torch.stack(reward_terms, dim=0).sum(dim=0)
        returns = (token_reward * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        returns = _apply_advantage_scaling(data, returns, eos_mask, config)
        advantages = verl_F.masked_whiten(returns.clone(), eos_mask)
        return advantages, returns


def compute_return_abs_accuracy(returns, acc):
    return (torch.sign(returns[:, 0]) == torch.sign(acc * 2 - 1)).float().mean()


def compute_return_smoothness(returns):
    return ((returns[:, :-1] - returns[:, 1:]) ** 2).sum(dim=-1).mean()
