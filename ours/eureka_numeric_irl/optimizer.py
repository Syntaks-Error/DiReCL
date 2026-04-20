from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

from .reward_parser import ParameterizedReward


@dataclass
class InnerLoopConfig:
    env_name: str = "Ant-v4"
    seed: int = 2026
    device: str = "cuda:0"
    objective: str = "maxentirl_sa"
    expert_episodes: int = 25
    resample_episodes: int = 1
    sac_epochs: int = 5
    sac_steps_per_epoch: int = 4000
    sac_log_step_interval: int = 5000
    max_ep_len: int = 1000
    training_trajs: int = 10
    irl_iterations: int = 1
    eval_episodes: int = 20
    reward_grad_steps: int = 1
    reward_lr: float = 1e-4
    reward_weight_decay: float = 1e-3
    reward_momentum: float = 0.9
    learning_starts: int = 10000
    buffer_size: int = 1000000
    num_test_episodes: int = 10
    grad_clip_norm: float = 10.0


@dataclass
class CandidateResult:
    name: str
    final_loss: float
    final_gap: float
    param_values: List[float]
    code: str
    loss_trace: List[float]
    gap_trace: List[float]
    error: Optional[str] = None
    parsed_param_count: int = 0
    parse_source_fn: Optional[str] = None
    parse_mode: Optional[str] = None
    train_log: List[str] = None
    sac_info: Dict[str, List[float]] = None
    initial_param_values: List[float] = None
    trained_reward_code: Optional[str] = None


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
        self.eval = importlib.import_module("utils.eval")

        self.env = gym.make(cfg.env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.expert_states, self.expert_actions, self.expert_trajs, self.expert_actions_trajs = self._load_expert_data()

    def _load_expert_data(self):
        env_tag = self.cfg.env_name.split("-")[0]
        expert_dir = self.workspace_root / "expert_data" / env_tag
        states = np.load(expert_dir / "states.npy").astype(np.float32)
        actions = np.load(expert_dir / "actions.npy").astype(np.float32)
        states = states[: self.cfg.expert_episodes]
        actions = actions[: self.cfg.expert_episodes]
        states_flat = states.reshape(-1, states.shape[-1])
        actions_flat = actions.reshape(-1, actions.shape[-1])
        return states_flat, actions_flat, states, actions

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
            learning_starts=self.cfg.learning_starts,
            train_freq=1,
            gradient_steps=1,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            buffer_size=self.cfg.buffer_size,
            num_test_episodes=self.cfg.num_test_episodes,
        )

        sac.reward_function = lambda sa_vec: reward_model.scalar_reward_from_state_action(sa_vec, self.obs_dim)
        return sac

    def _sample_expert_sa(self):
        if self.cfg.resample_episodes > self.cfg.expert_episodes:
            idx = np.random.choice(self.expert_trajs.shape[0], self.cfg.resample_episodes, replace=True)
            sE = self.expert_trajs[idx].copy()
            aE = self.expert_actions_trajs[idx].copy()
        elif self.cfg.resample_episodes > 0:
            k = min(self.cfg.resample_episodes, self.expert_trajs.shape[0])
            idx = np.random.choice(self.expert_trajs.shape[0], k, replace=False)
            sE = self.expert_trajs[idx].copy()
            aE = self.expert_actions_trajs[idx].copy()
        else:
            sE = self.expert_trajs
            aE = self.expert_actions_trajs

        return sE.reshape(-1, sE.shape[-1]), aE.reshape(-1, aE.shape[-1])

    def _ml_irl_loss(self, reward_model: ParameterizedReward, agent_samples):
        sA, aA, _ = agent_samples
        _, T, _ = sA.shape

        sA_vec = torch.as_tensor(sA.reshape(-1, sA.shape[-1]), dtype=torch.float32, device=self.device)
        aA_vec = torch.as_tensor(aA.reshape(-1, aA.shape[-1]), dtype=torch.float32, device=self.device)

        sE_np, aE_np = self._sample_expert_sa()
        sE_vec = torch.as_tensor(sE_np, dtype=torch.float32, device=self.device)
        aE_vec = torch.as_tensor(aE_np, dtype=torch.float32, device=self.device)

        if self.cfg.objective == "maxentirl":
            rA = reward_model(sA_vec, None)
            rE = reward_model(sE_vec, None)
        elif self.cfg.objective == "maxentirl_sa":
            rA = reward_model(sA_vec, aA_vec)
            rE = reward_model(sE_vec, aE_vec)
        else:
            raise ValueError(f"Unsupported objective: {self.cfg.objective}")

        loss = T * (rA.mean() - rE.mean())
        gap = float((rE.mean() - rA.mean()).detach().cpu().item())
        return loss, gap

    def optimize_candidate(
        self, name: str, code: str, log_fn: Optional[Callable[[str], None]] = None
    ) -> CandidateResult:
        def _log(msg: str):
            if log_fn is not None:
                log_fn(msg)
            else:
                print(msg, flush=True)

        train_log: List[str] = []

        _log(f"[{name}] reward parameter parsing begins")
        train_log.append("reward parameter parsing begins")
        reward_model = ParameterizedReward(code=code, fn_name="reward_fn", device=self.cfg.device).to(self.device)
        parse_report = reward_model.report()
        _log(
            f"[{name}] reward parsing success: source_fn={parse_report.source_fn_name}, "
            f"mode={parse_report.mode}, parsed_params={len(parse_report.constants)}"
        )
        train_log.append(
            f"reward parsing success source_fn={parse_report.source_fn_name} mode={parse_report.mode} "
            f"parsed_params={len(parse_report.constants)}"
        )

        _log(f"[{name}] reward parameter initializes: {parse_report.constants}")
        train_log.append(f"reward parameter initializes: {parse_report.constants}")

        optimizer = torch.optim.Adam(
            [reward_model.params],
            lr=self.cfg.reward_lr,
            weight_decay=self.cfg.reward_weight_decay,
            betas=(self.cfg.reward_momentum, 0.999),
        )

        _log(f"[{name}] ML-IRL policy optimizer (SAC) setup begins")
        train_log.append("ML-IRL policy optimizer (SAC) setup begins")
        sac = self._make_sac(reward_model)

        last_loss = 0.0
        last_gap = 0.0
        loss_trace: List[float] = []
        gap_trace: List[float] = []
        sac_info = {"test_rets": [], "alphas": [], "log_pis": [], "time_steps": []}
        sac_info["real_det_returns"] = []
        sac_info["real_sto_returns"] = []

        _log(f"[{name}] ML-IRL training begins: irl_iterations={self.cfg.irl_iterations}")
        train_log.append(f"ML-IRL training begins irl_iterations={self.cfg.irl_iterations}")

        for irl_itr in range(self.cfg.irl_iterations):
            _log(f"[{name}] [irl_iter={irl_itr}] SAC learning begins")
            train_log.append(f"irl_iter={irl_itr} SAC learning begins")
            sac_ret = sac.learn_mujoco(print_out=True)
            if isinstance(sac_ret, list) and len(sac_ret) == 4:
                sac_info["test_rets"].extend([float(x) for x in sac_ret[0]])
                sac_info["alphas"].extend([float(x) for x in sac_ret[1]])
                sac_info["log_pis"].extend([float(x) for x in sac_ret[2]])
                sac_info["time_steps"].extend([float(x) for x in sac_ret[3]])
            _log(f"[{name}] [irl_iter={irl_itr}] SAC learning done")
            train_log.append(f"irl_iter={irl_itr} SAC learning done")

            samples = self.collect.collect_trajectories_policy_single(
                self.env,
                sac,
                n=self.cfg.training_trajs,
                state_indices=list(range(self.obs_dim)),
            )
            _log(
                f"[{name}] [irl_iter={irl_itr}] trajectory collection done: "
                f"state_samples={samples[0].shape[0] * samples[0].shape[1]}, "
                f"action_samples={samples[1].shape[0] * samples[1].shape[1]}"
            )
            train_log.append(
                (
                    f"irl_iter={irl_itr} trajectory collection done "
                    f"state_samples={samples[0].shape[0] * samples[0].shape[1]} "
                    f"action_samples={samples[1].shape[0] * samples[1].shape[1]}"
                )
            )

            for grad_step in range(self.cfg.reward_grad_steps):
                loss, gap = self._ml_irl_loss(reward_model, samples)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite ML-IRL loss detected at grad_step={grad_step}: {loss}")
                optimizer.zero_grad()
                loss.backward()
                if reward_model.params.grad is None:
                    raise RuntimeError("Reward parameter gradient is None. Reward code likely detached from graph.")
                torch.nn.utils.clip_grad_norm_([reward_model.params], self.cfg.grad_clip_norm)
                grad_norm = float(torch.norm(reward_model.params.grad.detach()).cpu().item())
                if not np.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite gradient norm detected at grad_step={grad_step}: {grad_norm}")
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())
                last_gap = gap
                loss_trace.append(last_loss)
                gap_trace.append(last_gap)
                _log(
                    f"[{name}] [irl_iter={irl_itr}] [grad_step={grad_step}] "
                    f"ml_irl_loss={last_loss:.6f}, gap={last_gap:.6f}, grad_norm={grad_norm:.6f}, "
                    f"params={reward_model.params.detach().cpu().tolist()}"
                )
                train_log.append(
                    f"irl_iter={irl_itr} grad_step={grad_step} ml_irl_loss={last_loss:.6f} "
                    f"gap={last_gap:.6f} grad_norm={grad_norm:.6f}"
                )

            eval_env_det = gym.make(self.cfg.env_name)
            eval_env_sto = gym.make(self.cfg.env_name)
            real_return_det = self.eval.evaluate_real_return(
                sac.get_action,
                eval_env_det,
                self.cfg.eval_episodes,
                self.cfg.max_ep_len,
                True,
            )
            real_return_sto = self.eval.evaluate_real_return(
                sac.get_action,
                eval_env_sto,
                self.cfg.eval_episodes,
                self.cfg.max_ep_len,
                False,
            )
            eval_env_det.close()
            eval_env_sto.close()
            sac_info["real_det_returns"].append(float(real_return_det))
            sac_info["real_sto_returns"].append(float(real_return_sto))
            _log(
                f"[{name}] [irl_iter={irl_itr}] real reward eval: "
                f"det_return={float(real_return_det):.4f}, sto_return={float(real_return_sto):.4f}"
            )
            train_log.append(
                f"irl_iter={irl_itr} real reward eval det_return={float(real_return_det):.4f} "
                f"sto_return={float(real_return_sto):.4f}"
            )

        _log(f"[{name}] ML-IRL training completed")
        train_log.append("ML-IRL training completed")

        return CandidateResult(
            name=name,
            final_loss=last_loss,
            final_gap=last_gap,
            param_values=reward_model.params.detach().cpu().tolist(),
            code=code,
            loss_trace=loss_trace,
            gap_trace=gap_trace,
            parsed_param_count=len(parse_report.constants),
            parse_source_fn=parse_report.source_fn_name,
            parse_mode=parse_report.mode,
            train_log=train_log,
            sac_info=sac_info,
            initial_param_values=parse_report.constants,
            trained_reward_code=reward_model.export_trained_code(),
        )

    def optimize_batch(
        self,
        candidates: Dict[str, str],
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> List[CandidateResult]:
        results = []
        for name, code in candidates.items():
            try:
                results.append(self.optimize_candidate(name=name, code=code, log_fn=log_fn))
            except Exception as e:
                tb = traceback.format_exc()
                if log_fn is not None:
                    log_fn(f"[{name}] failed: {e}")
                    for line in tb.rstrip().splitlines():
                        log_fn(f"[{name}] traceback: {line}")
                results.append(
                    CandidateResult(
                        name=name,
                        final_loss=float("inf"),
                        final_gap=float("-inf"),
                        param_values=[],
                        code=f"# failed: {e}\n" + code,
                        loss_trace=[],
                        gap_trace=[],
                        error=tb,
                        train_log=[f"failed: {e}", tb],
                        sac_info={"test_rets": [], "alphas": [], "log_pis": [], "time_steps": []},
                        initial_param_values=[],
                        trained_reward_code=None,
                    )
                )
        return results
