"""Debug metrics for sac_continuous_action.py."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def compute_grad_norm(*modules_or_tensors) -> float:
    """L2 norm over all gradients of the given modules / leaf tensors.
    Call right after .backward() and before .step()."""
    total = 0.0
    for item in modules_or_tensors:
        params = item.parameters() if isinstance(item, nn.Module) else [item]
        for p in params:
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
    return total ** 0.5


def log_grad_norms(
    writer: SummaryWriter,
    global_step: int,
    qf_grad_norm: float,
    actor_grad_norm: float,
    alpha_grad_norm: Optional[float] = None,
) -> None:
    writer.add_scalar("debug/qf_grad_norm", qf_grad_norm, global_step)
    writer.add_scalar("debug/actor_grad_norm", actor_grad_norm, global_step)
    if alpha_grad_norm is not None:
        writer.add_scalar("debug/alpha_grad_norm", alpha_grad_norm, global_step)


def log_q_stats(
    writer: SummaryWriter,
    global_step: int,
    qf1_a_values: torch.Tensor,
    qf2_a_values: torch.Tensor,
    next_q_value: torch.Tensor,
    prefix: str = "debug",
) -> None:
    writer.add_scalar(f"{prefix}/qf1_values_min", qf1_a_values.min().item(), global_step)
    writer.add_scalar(f"{prefix}/qf1_values_max", qf1_a_values.max().item(), global_step)
    writer.add_scalar(f"{prefix}/qf1_values_std", qf1_a_values.std().item(), global_step)
    writer.add_scalar(f"{prefix}/qf2_values_min", qf2_a_values.min().item(), global_step)
    writer.add_scalar(f"{prefix}/qf2_values_max", qf2_a_values.max().item(), global_step)
    writer.add_scalar(f"{prefix}/qf2_values_std", qf2_a_values.std().item(), global_step)
    writer.add_scalar(f"{prefix}/target_q_mean", next_q_value.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/target_q_std", next_q_value.std().item(), global_step)


def log_log_pi_stats(
    writer: SummaryWriter,
    global_step: int,
    log_pi: torch.Tensor,
    next_state_log_pi: torch.Tensor,
    prefix: str = "debug",
) -> None:
    writer.add_scalar(f"{prefix}/log_pi_mean", log_pi.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/log_pi_std", log_pi.std().item(), global_step)
    writer.add_scalar(f"{prefix}/next_state_log_pi_mean", next_state_log_pi.mean().item(), global_step)
    writer.add_scalar(f"{prefix}/next_state_log_pi_std", next_state_log_pi.std().item(), global_step)
