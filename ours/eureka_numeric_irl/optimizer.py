from __future__ import annotations

import os
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
    agent_algo: str = "sac"  # sac | ppo
    env_name: str = "Ant-v4"
    seed: int = 2026
    device: str = "cuda:0"
    objective: str = "maxentirl_sa"
    expert_episodes: int = 25
    resample_episodes: int = 1
    expert_data_dir: Optional[str] = None

    # CommonRoad env construction (used when agent_algo=ppo or env_name starts with commonroad)
    commonroad_rl_root: str = "../commonroad-rl"
    meta_scenario_path: Optional[str] = None
    train_reset_config_path: Optional[str] = None
    test_reset_config_path: Optional[str] = None
    config_file: Optional[str] = None
    logging_path: Optional[str] = None
    test_env: bool = True
    play: bool = False
    render_mode: Optional[str] = None

    # PPO-specific
    ppo_n_envs: int = 1
    ppo_timesteps_per_iter: int = 32768
    ppo_reinitialize: bool = False
    ppo_hyperparams: Optional[Dict] = None

    # SAC-specific
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
    best_policy_path: Optional[str] = None


class MLIRLNumericOptimizer:
    def __init__(self, workspace_root: Path, cfg: InnerLoopConfig):
        self.workspace_root = workspace_root
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.is_commonroad = self.cfg.agent_algo.lower() == "ppo" or self.cfg.env_name.startswith("commonroad")

        mlirl_root = workspace_root.parent / "ML-IRL"
        if str(mlirl_root) not in sys.path:
            sys.path.insert(0, str(mlirl_root))

        importlib.import_module("envs")
        self.SAC = importlib.import_module("common.sac").SAC
        self.collect = importlib.import_module("utils.collect")
        self.eval = importlib.import_module("utils.eval")

        self.PPO = None
        self.read_commonroad_ppo_hyperparams = None
        self.env_kwargs = {}

        if self.is_commonroad:
            self._setup_commonroad_imports()
            ppo_mod = importlib.import_module("common.ppo")
            self.PPO = ppo_mod.PPO
            self.read_commonroad_ppo_hyperparams = ppo_mod.read_commonroad_ppo_hyperparams
            self.env_kwargs = self._build_commonroad_env_kwargs()
            self.env_fn = lambda: gym.make(self.cfg.env_name, **self.env_kwargs)
        else:
            self.env_fn = lambda: gym.make(self.cfg.env_name)

        self.env = self.env_fn()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.state_indices = list(range(self.obs_dim))

        (
            self.expert_states,
            self.expert_actions,
            self.expert_state_eps,
            self.expert_action_eps,
            self.expert_lens,
        ) = self._load_expert_data()

    def _abs_from_workspace(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(str(self.workspace_root), path))

    def _setup_commonroad_imports(self):
        commonroad_root = self._abs_from_workspace(self.cfg.commonroad_rl_root)
        if commonroad_root not in sys.path:
            sys.path.insert(0, commonroad_root)
        import commonroad_rl.gym_commonroad  # noqa: F401

    def _build_commonroad_env_kwargs(self) -> Dict:
        kwargs = {}
        keys = [
            "meta_scenario_path",
            "train_reset_config_path",
            "test_reset_config_path",
            "config_file",
            "logging_path",
            "test_env",
            "play",
            "render_mode",
        ]
        for k in keys:
            v = getattr(self.cfg, k)
            if v is None:
                continue
            if k in {
                "meta_scenario_path",
                "train_reset_config_path",
                "test_reset_config_path",
                "config_file",
                "logging_path",
            } and isinstance(v, str):
                kwargs[k] = self._abs_from_workspace(v)
            else:
                kwargs[k] = v
        return kwargs

    def _load_expert_data(self):
        if self.cfg.expert_data_dir is not None:
            expert_dir = Path(self._abs_from_workspace(self.cfg.expert_data_dir))
        else:
            env_tag = "highD" if self.is_commonroad else self.cfg.env_name.split("-")[0]
            expert_dir = self.workspace_root / "expert_data" / env_tag

        states = np.load(expert_dir / "states.npy").astype(np.float32)
        actions = np.load(expert_dir / "actions.npy").astype(np.float32)
        lens_path = expert_dir / "lens.npy"
        if lens_path.exists():
            lens = np.load(lens_path).astype(np.int32)
        else:
            lens = np.full((states.shape[0],), states.shape[1], dtype=np.int32)

        n_eps = min(int(self.cfg.expert_episodes), states.shape[0])
        states = states[:n_eps]
        actions = actions[:n_eps]
        lens = lens[:n_eps]

        sa_rows = []
        s_rows = []
        for ep in range(n_eps):
            L = int(lens[ep])
            s = states[ep, :L, :]
            a = actions[ep, :L, :]
            s_rows.append(s)
            sa_rows.append(a)

        states_flat = np.concatenate(s_rows, axis=0).astype(np.float32)
        actions_flat = np.concatenate(sa_rows, axis=0).astype(np.float32)
        return states_flat, actions_flat, states, actions, lens

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

    @staticmethod
    def _reset_env(env):
        out = env.reset()
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _step_env(env, action):
        out = env.step(action)
        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
            return obs, rew, bool(terminated or truncated), info
        return out

    def _make_ppo(self, reward_model: ParameterizedReward):
        if self.PPO is None or self.read_commonroad_ppo_hyperparams is None:
            raise RuntimeError("PPO backend not initialized. Check CommonRoad setup.")

        commonroad_root = self._abs_from_workspace(self.cfg.commonroad_rl_root)
        policy, ppo_kwargs, normalize, normalize_kwargs = self.read_commonroad_ppo_hyperparams(
            env_id=self.cfg.env_name,
            commonroad_root=commonroad_root,
            n_envs=int(self.cfg.ppo_n_envs),
            overrides=self.cfg.ppo_hyperparams,
        )

        ppo_agent = self.PPO(
            self.env_fn,
            env_id=self.cfg.env_name,
            env_kwargs=self.env_kwargs,
            seed=int(self.cfg.seed),
            n_envs=int(self.cfg.ppo_n_envs),
            total_timesteps_per_itr=int(self.cfg.ppo_timesteps_per_iter),
            policy=policy,
            ppo_kwargs=ppo_kwargs,
            normalize=normalize,
            normalize_kwargs=normalize_kwargs,
            reward_state_indices=self.state_indices,
            device=self.device,
            reinitialize=bool(self.cfg.ppo_reinitialize),
            max_ep_len=int(self.cfg.max_ep_len),
        )
        ppo_agent.reward_function = lambda sa_vec: reward_model.scalar_reward_from_state_action(sa_vec, self.obs_dim)
        return ppo_agent

    def _collect_agent_rollouts_ppo(self, ppo_agent, n: int, horizon: int):
        env = self.env_fn()
        states = np.zeros((n, horizon, len(self.state_indices)), dtype=np.float32)
        actions = np.zeros((n, horizon, self.act_dim), dtype=np.float32)
        lens = np.zeros((n,), dtype=np.int32)

        for ep in range(n):
            obs = self._reset_env(env)
            ep_len = 0
            for t in range(horizon):
                if (
                    ppo_agent.normalize
                    and ppo_agent.train_env is not None
                    and hasattr(ppo_agent.train_env, "normalize_obs")
                ):
                    obs_for_store = ppo_agent.train_env.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])[0]
                else:
                    obs_for_store = np.asarray(obs, dtype=np.float32)

                act = ppo_agent.get_action(obs, deterministic=False)
                obs2, _, done, _ = self._step_env(env, act)

                states[ep, t] = obs_for_store[self.state_indices]
                actions[ep, t] = np.asarray(act, dtype=np.float32)

                obs = obs2
                ep_len += 1
                if done:
                    break
            lens[ep] = ep_len

        env.close()
        return states, actions, lens

    def _sample_expert_sa(self):
        n_eps = self.expert_state_eps.shape[0]
        if self.cfg.resample_episodes > self.cfg.expert_episodes:
            idx = np.random.choice(n_eps, self.cfg.resample_episodes, replace=True)
        elif self.cfg.resample_episodes > 0:
            k = min(self.cfg.resample_episodes, n_eps)
            idx = np.random.choice(n_eps, k, replace=False)
        else:
            idx = np.arange(n_eps)

        s_rows = []
        a_rows = []
        for ep in idx:
            L = int(self.expert_lens[int(ep)])
            s_rows.append(self.expert_state_eps[int(ep), :L, :])
            a_rows.append(self.expert_action_eps[int(ep), :L, :])

        sE = np.concatenate(s_rows, axis=0).astype(np.float32)
        aE = np.concatenate(a_rows, axis=0).astype(np.float32)
        return sE, aE

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
        self,
        name: str,
        code: str,
        log_fn: Optional[Callable[[str], None]] = None,
        policy_save_dir: Optional[Path] = None,
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

        algo_name = "PPO" if self.is_commonroad else "SAC"
        _log(f"[{name}] ML-IRL policy optimizer ({algo_name}) setup begins")
        train_log.append(f"ML-IRL policy optimizer ({algo_name}) setup begins")
        agent = self._make_ppo(reward_model) if self.is_commonroad else self._make_sac(reward_model)

        last_loss = 0.0
        last_gap = 0.0
        loss_trace: List[float] = []
        gap_trace: List[float] = []
        sac_info = {"test_rets": [], "alphas": [], "log_pis": [], "time_steps": []}
        sac_info["real_det_returns"] = []
        sac_info["real_sto_returns"] = []
        best_policy_score = float("-inf")
        best_policy_path = None

        if self.is_commonroad and policy_save_dir is not None:
            policy_save_dir.mkdir(parents=True, exist_ok=True)
            best_policy_path = policy_save_dir / f"{name}_best_model.zip"

        _log(f"[{name}] ML-IRL training begins: irl_iterations={self.cfg.irl_iterations}")
        train_log.append(f"ML-IRL training begins irl_iterations={self.cfg.irl_iterations}")

        for irl_itr in range(self.cfg.irl_iterations):
            _log(f"[{name}] [irl_iter={irl_itr}] {algo_name} learning begins")
            train_log.append(f"irl_iter={irl_itr} {algo_name} learning begins")
            sac_ret = agent.learn_mujoco(print_out=True)
            if isinstance(sac_ret, list) and len(sac_ret) == 4:
                sac_info["test_rets"].extend([float(x) for x in sac_ret[0]])
                sac_info["alphas"].extend([float(x) for x in sac_ret[1]])
                sac_info["log_pis"].extend([float(x) for x in sac_ret[2]])
                sac_info["time_steps"].extend([float(x) for x in sac_ret[3]])
            _log(f"[{name}] [irl_iter={irl_itr}] {algo_name} learning done")
            train_log.append(f"irl_iter={irl_itr} {algo_name} learning done")

            if self.is_commonroad:
                samples = self._collect_agent_rollouts_ppo(
                    agent,
                    n=int(self.cfg.training_trajs),
                    horizon=int(self.cfg.max_ep_len),
                )
                samples = (samples[0], samples[1], np.zeros_like(samples[0][:, :, 0]))
            else:
                samples = self.collect.collect_trajectories_policy_single(
                    self.env,
                    agent,
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

            eval_env_det = self.env_fn()
            eval_env_sto = self.env_fn()
            real_return_det = self.eval.evaluate_real_return(
                agent.get_action,
                eval_env_det,
                self.cfg.eval_episodes,
                self.cfg.max_ep_len,
                True,
            )
            real_return_sto = self.eval.evaluate_real_return(
                agent.get_action,
                eval_env_sto,
                self.cfg.eval_episodes,
                self.cfg.max_ep_len,
                False,
            )
            eval_env_det.close()
            eval_env_sto.close()
            sac_info["real_det_returns"].append(float(real_return_det))
            sac_info["real_sto_returns"].append(float(real_return_sto))

            if (
                self.is_commonroad
                and best_policy_path is not None
                and hasattr(agent, "model")
                and agent.model is not None
            ):
                policy_score = float(real_return_det)
                if policy_score > best_policy_score:
                    best_policy_score = policy_score
                    agent.model.save(str(best_policy_path))
                    _log(
                        f"[{name}] [irl_iter={irl_itr}] new best PPO policy saved: {best_policy_path} "
                        f"(det_return={best_policy_score:.4f})"
                    )
                    train_log.append(
                        f"irl_iter={irl_itr} new best PPO policy saved path={best_policy_path} "
                        f"det_return={best_policy_score:.4f}"
                    )

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
            best_policy_path=str(best_policy_path) if best_policy_path is not None else None,
        )

    def optimize_batch(
        self,
        candidates: Dict[str, str],
        log_fn: Optional[Callable[[str], None]] = None,
        policy_save_dir: Optional[Path] = None,
    ) -> List[CandidateResult]:
        results = []
        for name, code in candidates.items():
            try:
                results.append(
                    self.optimize_candidate(
                        name=name,
                        code=code,
                        log_fn=log_fn,
                        policy_save_dir=policy_save_dir,
                    )
                )
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
                        best_policy_path=None,
                    )
                )
        return results
