from __future__ import annotations

import sys
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch

from .reward_parser import ParameterizedReward


@dataclass
class InnerLoopConfig:
    env_name: str = "Ant-v4"
    seed: int = 2026
    device: str = "cpu"
    sac_epochs: int = 2
    sac_steps_per_epoch: int = 2000
    sac_log_step_interval: int = 1000
    max_ep_len: int = 1000
    training_trajs: int = 4
    irl_iterations: int = 2
    reward_grad_steps: int = 20
    reward_lr: float = 5e-3


@dataclass
class CandidateResult:
    name: str
    final_loss: float
    final_gap: float
    param_values: List[float]
    code: str


class MLIRLNumericOptimizer:
    def __init__(self, workspace_root: Path, cfg: InnerLoopConfig):
        self.workspace_root = workspace_root
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        mlirl_root = workspace_root.parent / "ML-IRL"
        if str(mlirl_root) not in sys.path:
            sys.path.insert(0, str(mlirl_root))

        importlib.import_module("envs")
        self.SAC = importlib.import_module("common.sac").SAC
        self.collect = importlib.import_module("utils.collect")

        self.env = gym.make(cfg.env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.expert_states, self.expert_actions = self._load_expert_data()

    def _load_expert_data(self):
        env_tag = self.cfg.env_name.split("-")[0]
        expert_dir = self.workspace_root / "expert_data" / env_tag
        states = np.load(expert_dir / "states.npy").astype(np.float32)
        actions = np.load(expert_dir / "actions.npy").astype(np.float32)
        states = states.reshape(-1, states.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        return states, actions

    def _make_sac(self, reward_model: ParameterizedReward):
        env_fn = lambda: gym.make(self.cfg.env_name)
        state_indices = list(range(self.obs_dim))

        sac = self.SAC(
            env_fn,
            seed=self.cfg.seed,
            steps_per_epoch=self.cfg.sac_steps_per_epoch,
            epochs=self.cfg.sac_epochs,
            max_ep_len=self.cfg.max_ep_len,
            log_step_interval=self.cfg.sac_log_step_interval,
            reward_state_indices=state_indices,
            device=self.device,
            reinitialize=False,
            sa=True,
            policy="MlpPolicy",
            policy_kwargs={"net_arch": [256, 256]},
            learning_rate=3e-4,
            batch_size=256,
            learning_starts=1000,
            train_freq=1,
            gradient_steps=1,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            buffer_size=200000,
            num_test_episodes=3,
        )

        sac.reward_function = lambda sa_vec: reward_model.scalar_reward_from_state_action(sa_vec, self.obs_dim)
        return sac

    def _ml_irl_loss(self, reward_model: ParameterizedReward, s_agent, a_agent):
        sA = torch.as_tensor(s_agent, dtype=torch.float32, device=self.device)
        aA = torch.as_tensor(a_agent, dtype=torch.float32, device=self.device)
        sE = torch.as_tensor(self.expert_states, dtype=torch.float32, device=self.device)
        aE = torch.as_tensor(self.expert_actions, dtype=torch.float32, device=self.device)

        rA = reward_model(sA, aA)
        rE = reward_model(sE, aE)

        loss = rA.mean() - rE.mean()
        gap = (rE.mean() - rA.mean()).detach().cpu().item()
        return loss, float(gap)

    def optimize_candidate(self, name: str, code: str) -> CandidateResult:
        reward_model = ParameterizedReward(code=code, fn_name="reward_fn", device=self.cfg.device).to(self.device)
        optimizer = torch.optim.Adam([reward_model.params], lr=self.cfg.reward_lr)

        sac = self._make_sac(reward_model)

        last_loss = 0.0
        last_gap = 0.0

        for _ in range(self.cfg.irl_iterations):
            sac.learn_mujoco(print_out=False)

            samples = self.collect.collect_trajectories_policy_single(
                self.env,
                sac,
                n=self.cfg.training_trajs,
                state_indices=list(range(self.obs_dim)),
            )
            sA, aA, _ = samples
            sA = sA.reshape(-1, sA.shape[-1])
            aA = aA.reshape(-1, aA.shape[-1])

            for _ in range(self.cfg.reward_grad_steps):
                loss, gap = self._ml_irl_loss(reward_model, sA, aA)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())
                last_gap = gap

        return CandidateResult(
            name=name,
            final_loss=last_loss,
            final_gap=last_gap,
            param_values=reward_model.params.detach().cpu().tolist(),
            code=code,
        )

    def optimize_batch(self, candidates: Dict[str, str]) -> List[CandidateResult]:
        results = []
        for name, code in candidates.items():
            try:
                results.append(self.optimize_candidate(name=name, code=code))
            except Exception as e:
                results.append(
                    CandidateResult(
                        name=name,
                        final_loss=float("inf"),
                        final_gap=float("-inf"),
                        param_values=[],
                        code=f"# failed: {e}\n" + code,
                    )
                )
        return results
