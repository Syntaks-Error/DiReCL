from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RewardTemplate:
    name: str
    code: str


def _wrap_expr(expr: str) -> str:
    return "def reward_fn(obs, act):\n" "    # obs: [B, obs_dim], act: [B, act_dim]\n" f"    return {expr}\n"


def default_ant_templates() -> List[RewardTemplate]:
    exprs = [
        "1.0 * obs[:, 13] - 0.5 * torch.sum(act * act, dim=1) - 0.05 * torch.sum(obs[:, 0:2] * obs[:, 0:2], dim=1)",
        "1.2 * obs[:, 13] + 0.3 * obs[:, 14] - 0.6 * torch.sum(act * act, dim=1)",
        "0.8 * obs[:, 13] - 0.4 * torch.square(obs[:, 0]) - 0.4 * torch.square(obs[:, 1]) - 0.08 * torch.sum(act * act, dim=1)",
        "1.0 * obs[:, 13] + 0.2 * torch.abs(obs[:, 0]) - 0.7 * torch.sum(act * act, dim=1)",
    ]
    out = []
    for i, expr in enumerate(exprs):
        out.append(RewardTemplate(name=f"seed_{i}", code=_wrap_expr(expr)))
    return out


def reflected_candidates(best_code: str, round_id: int) -> List[RewardTemplate]:
    """Simple Eureka-style reflection: mutate structure while keeping differentiability."""
    mut_exprs = [
        "1.0 * obs[:, 13] + 0.1 * obs[:, 14] - 0.6 * torch.sum(act * act, dim=1) - 0.04 * torch.sum(obs[:, 0:2] * obs[:, 0:2], dim=1)",
        "1.1 * obs[:, 13] - 0.5 * torch.sum(act * act, dim=1) + 0.05 * torch.tanh(obs[:, 14])",
        "0.9 * obs[:, 13] + 0.1 * torch.square(obs[:, 14]) - 0.65 * torch.sum(act * act, dim=1)",
    ]
    return [RewardTemplate(name=f"reflect_{round_id}_{i}", code=_wrap_expr(expr)) for i, expr in enumerate(mut_exprs)]
