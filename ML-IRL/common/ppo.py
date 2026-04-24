import copy
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from ruamel.yaml import YAML
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import gymnasium as gym


def _convert_schedule(value):
    if isinstance(value, str) and value.startswith("lin_"):
        initial_value = float(value.split("_", 1)[1])

        def _linear(progress_remaining: float):
            return progress_remaining * initial_value

        return _linear
    return value


def read_commonroad_ppo_hyperparams(
    env_id: str,
    commonroad_root: str,
    n_envs: int = 1,
    overrides: Optional[Dict] = None,
):
    """
    Parse PPO hyperparameters with the same conversion logic as
    commonroad_rl/train_model_sb3.py.
    """
    ppo_hyperparams_file = os.path.join(commonroad_root, "commonroad_rl/hyperparams/ppo.yml")
    legacy_ppo2_file = os.path.join(commonroad_root, "commonroad_rl/hyperparams/ppo2.yml")
    source_file = ppo_hyperparams_file if os.path.isfile(ppo_hyperparams_file) else legacy_ppo2_file

    with open(source_file, "r") as f:
        hyperparams_dict = YAML(typ="safe").load(f)

    if env_id not in hyperparams_dict:
        raise ValueError(f"Hyperparameters not found for ppo-{env_id} in {source_file}")

    hp = copy.deepcopy(hyperparams_dict[env_id])
    if overrides is not None:
        hp.update(overrides)

    normalize = hp.pop("normalize", False)
    normalize_kwargs = {}
    if isinstance(normalize, str):
        normalize_kwargs = eval(normalize)
        normalize = True

    if "policy_kwargs" in hp and isinstance(hp["policy_kwargs"], str):
        hp["policy_kwargs"] = eval(hp["policy_kwargs"])

    hp["learning_rate"] = _convert_schedule(hp.get("learning_rate", 3e-4))

    if "cliprange" in hp and "clip_range" not in hp:
        hp["clip_range"] = _convert_schedule(hp.pop("cliprange"))
    elif "clip_range" in hp:
        hp["clip_range"] = _convert_schedule(hp["clip_range"])

    if "cliprange_vf" in hp and "clip_range_vf" not in hp:
        hp["clip_range_vf"] = _convert_schedule(hp.pop("cliprange_vf"))
    elif "clip_range_vf" in hp:
        hp["clip_range_vf"] = _convert_schedule(hp["clip_range_vf"])

    if "lam" in hp and "gae_lambda" not in hp:
        hp["gae_lambda"] = hp.pop("lam")
    if "noptepochs" in hp and "n_epochs" not in hp:
        hp["n_epochs"] = hp.pop("noptepochs")

    nminibatches = hp.pop("nminibatches", None)
    if nminibatches is not None and "batch_size" not in hp:
        n_steps = int(hp.get("n_steps", 2048))
        total_rollout = max(n_steps * int(n_envs), 1)
        hp["batch_size"] = max(total_rollout // int(nminibatches), 1)

    for key in ["n_timesteps", "n_envs", "env_wrapper", "frame_stack"]:
        hp.pop(key, None)

    policy = hp.pop("policy", "MlpPolicy")
    return policy, hp, normalize, normalize_kwargs


class RewardFnWrapper(gym.Wrapper):
    """Override environment reward with an external reward function."""

    def __init__(self, env, reward_getter, reward_state_indices: Optional[List[int]] = None, sa: bool = True):
        super().__init__(env)
        self.reward_getter = reward_getter
        self.reward_state_indices = reward_state_indices
        self.sa = sa
        self._last_obs = None

    @staticmethod
    def _as_obs(reset_out):
        if isinstance(reset_out, tuple):
            return reset_out[0]
        return reset_out

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0])
        if isinstance(value, np.ndarray):
            return float(np.asarray(value).reshape(-1)[0])
        return float(value)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._last_obs = self._as_obs(out)
        return out

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, env_rew, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, env_rew, done, info = out
            terminated = bool(done)
            truncated = False

        rew = env_rew
        reward_fn = self.reward_getter()
        if reward_fn is not None and self._last_obs is not None:
            state = self._last_obs
            if self.reward_state_indices is not None:
                state = state[self.reward_state_indices]
            if self.sa:
                state = np.concatenate([state, np.asarray(action)], axis=-1)
            rew = self._to_scalar(reward_fn(state))

        self._last_obs = obs
        if len(out) == 5:
            return obs, rew, terminated, truncated, info
        return obs, rew, done, info


class PPO:
    def __init__(
        self,
        env_fn,
        env_id: str,
        env_kwargs: Dict,
        seed: int = 0,
        n_envs: int = 1,
        total_timesteps_per_itr: int = 32768,
        policy: str = "MlpPolicy",
        ppo_kwargs: Optional[Dict] = None,
        normalize: bool = False,
        normalize_kwargs: Optional[Dict] = None,
        reward_state_indices: Optional[List[int]] = None,
        device=torch.device("cpu"),
        reinitialize: bool = False,
        max_ep_len: int = 1000,
    ):
        self.env_fn = env_fn
        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self.seed = int(seed)
        self.n_envs = int(n_envs)
        self.total_timesteps_per_itr = int(total_timesteps_per_itr)
        self.policy_name = policy
        self.ppo_kwargs = copy.deepcopy(ppo_kwargs or {})
        self.normalize = bool(normalize)
        self.normalize_kwargs = copy.deepcopy(normalize_kwargs or {})
        self.reward_state_indices = reward_state_indices
        self.device = device
        self.reinitialize = bool(reinitialize)
        self.max_ep_len = int(max_ep_len)

        # SB3 PPO is typically faster on CPU for non-CNN policies.
        # Auto-switch to CPU to avoid the known warning/performance pitfall.
        device_str = str(self.device)
        if "CnnPolicy" not in str(self.policy_name) and device_str.startswith("cuda"):
            print(
                "[PPO] Non-CNN policy detected; switching PPO device from GPU to CPU "
                "for better performance and to avoid SB3 warning."
            )
            self.device = torch.device("cpu")

        self.reward_function = None
        self.model = None
        self.train_env = None
        self.ac = None

        self._ensure_model_env()

    def _make_single_env(self, rank: int, use_reward_wrapper: bool, monitor_log_dir=None, subproc=False):
        env_id = self.env_id
        env_kwargs = copy.deepcopy(self.env_kwargs)
        seed = self.seed

        def _init():
            local_kwargs = copy.deepcopy(env_kwargs)
            if subproc and ("commonroad" in env_id or env_id == "cr-monitor-v0"):
                from commonroad_rl.gym_commonroad.constants import PATH_PARAMS

                train_reset_config_path = local_kwargs.pop("train_reset_config_path", PATH_PARAMS["train_reset_config"])
                test_reset_config_path = local_kwargs.pop("test_reset_config_path", PATH_PARAMS["test_reset_config"])
                env = gym.make(
                    env_id,
                    train_reset_config_path=os.path.join(train_reset_config_path, str(rank)),
                    test_reset_config_path=os.path.join(test_reset_config_path, str(rank)),
                    **local_kwargs,
                )
            else:
                env = gym.make(env_id, **local_kwargs)

            if use_reward_wrapper:
                env = RewardFnWrapper(
                    env,
                    reward_getter=lambda: self.reward_function,
                    reward_state_indices=self.reward_state_indices,
                    sa=True,
                )

            try:
                env.reset(seed=seed + rank)
            except TypeError:
                env.seed(seed + rank)

            if monitor_log_dir is not None:
                os.makedirs(monitor_log_dir, exist_ok=True)
                env = Monitor(env, os.path.join(monitor_log_dir, str(rank)))
            return env

        return _init

    def _make_vec_env(self, use_reward_wrapper: bool, monitor_log_dir=None):
        if self.n_envs == 1:
            vec_env = DummyVecEnv(
                [self._make_single_env(0, use_reward_wrapper=use_reward_wrapper, monitor_log_dir=monitor_log_dir)]
            )
        else:
            vec_env = SubprocVecEnv(
                [
                    self._make_single_env(
                        i,
                        use_reward_wrapper=use_reward_wrapper,
                        monitor_log_dir=monitor_log_dir,
                        subproc=True,
                    )
                    for i in range(self.n_envs)
                ],
                start_method="spawn",
            )

        if self.normalize:
            vec_env = VecNormalize(vec_env, **copy.deepcopy(self.normalize_kwargs))
        return vec_env

    def _ensure_model_env(self, monitor_log_dir=None):
        if self.model is None or self.reinitialize:
            train_env = self._make_vec_env(use_reward_wrapper=True, monitor_log_dir=monitor_log_dir)
            self.model = SB3PPO(
                policy=self.policy_name,
                env=train_env,
                seed=self.seed,
                device=self.device,
                verbose=0,
                **self.ppo_kwargs,
            )
            self.train_env = train_env
            self.ac = self.model.policy
            self.reinitialize = False

    def learn_mujoco(self, print_out=False):
        self._ensure_model_env()
        start = time.time()
        self.model.learn(total_timesteps=self.total_timesteps_per_itr, reset_num_timesteps=False, progress_bar=False)
        if print_out:
            print(f"PPO Training End: time {time.time() - start:.0f}s")
        return []

    def learn(self, print_out=False, n_parallel=1):
        return self.learn_mujoco(print_out=print_out)

    def get_action(self, obs, deterministic=False, get_logprob=False):
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]

        if self.normalize and self.train_env is not None and hasattr(self.train_env, "normalize_obs"):
            obs_in = self.train_env.normalize_obs(obs_arr.copy())
        else:
            obs_in = obs_arr

        action, _ = self.model.predict(obs_in, deterministic=deterministic)
        action = np.asarray(action)
        if action.ndim > 1 and action.shape[0] == 1:
            action_out = action[0]
        else:
            action_out = action

        if get_logprob:
            return action_out, 0.0
        return action_out
