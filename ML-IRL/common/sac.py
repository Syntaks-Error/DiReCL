import copy
import time
from typing import Callable, Optional

import numpy as np
import torch
from stable_baselines3 import SAC as SB3SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


class RewardFnWrapper(gym.Wrapper):
    """Override environment reward with an external reward function."""

    def __init__(
        self, env, reward_getter: Callable[[], Optional[Callable]], reward_state_indices=None, sa: bool = False
    ):
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
    def _step_unpack(step_out):
        if len(step_out) == 5:
            obs, rew, terminated, truncated, info = step_out
            done = terminated or truncated
            return obs, rew, bool(terminated), bool(truncated), done, info
        obs, rew, done, info = step_out
        return obs, rew, bool(done), False, bool(done), info

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs = self._as_obs(out)
        self._last_obs = obs
        return out

    def _to_scalar(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0])
        if isinstance(value, np.ndarray):
            return float(np.asarray(value).reshape(-1)[0])
        return float(value)

    def step(self, action):
        step_out = self.env.step(action)
        obs, env_rew, terminated, truncated, done, info = self._step_unpack(step_out)

        reward_fn = self.reward_getter()
        rew = env_rew
        if reward_fn is not None and self._last_obs is not None:
            state = self._last_obs
            if self.reward_state_indices is not None:
                state = state[self.reward_state_indices]
            if self.sa:
                state = np.concatenate([state, np.asarray(action)], axis=-1)
            rew = self._to_scalar(reward_fn(state))

        self._last_obs = obs

        if len(step_out) == 5:
            return obs, rew, terminated, truncated, info
        return obs, rew, done, info


class SAC:

    def __init__(
        self,
        env_fn,
        seed=0,
        steps_per_epoch=4000,
        epochs=100,
        num_test_episodes=10,
        max_ep_len=1000,
        log_step_interval=None,
        reward_state_indices=None,
        device=torch.device("cpu"),
        reinitialize=True,
        sa=False,
        **kwargs,
    ):

        self.env_fn = env_fn
        self.env = env_fn()
        self.test_env = env_fn()

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        self.max_ep_len = max_ep_len
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.num_test_episodes = num_test_episodes
        self.log_step_interval = log_step_interval or steps_per_epoch

        self.device = device
        self.reinitialize = reinitialize
        self.reward_function = None
        self.reward_state_indices = reward_state_indices
        self.sa = sa

        self.alpha = torch.tensor(0.0, device=device)

        # SB3 SAC parameters
        sb3_policy = kwargs.get("policy", "MlpPolicy")
        sb3_learning_rate = kwargs.get("learning_rate", 3e-4)
        sb3_buffer_size = kwargs.get("buffer_size", int(1e6))
        sb3_learning_starts = kwargs.get("learning_starts", 10000)
        sb3_tau = kwargs.get("tau", 0.005)
        sb3_gamma = kwargs.get("gamma", 0.99)
        sb3_train_freq = kwargs.get("train_freq", 1)
        sb3_gradient_steps = kwargs.get("gradient_steps", 1)
        sb3_batch_size = kwargs.get("batch_size", 256)
        sb3_policy_kwargs = copy.deepcopy(kwargs.get("policy_kwargs", {}))
        sb3_ent_coef = kwargs.get("ent_coef", "auto")

        model_kwargs = dict(
            policy=sb3_policy,
            learning_rate=sb3_learning_rate,
            buffer_size=sb3_buffer_size,
            learning_starts=sb3_learning_starts,
            batch_size=sb3_batch_size,
            tau=sb3_tau,
            gamma=sb3_gamma,
            train_freq=sb3_train_freq,
            gradient_steps=sb3_gradient_steps,
            ent_coef=sb3_ent_coef,
            policy_kwargs=sb3_policy_kwargs,
            verbose=0,
            seed=seed,
            device=device,
        )

        # Optional SB3 args passthrough
        optional_keys = ["target_update_interval", "target_entropy", "use_sde", "sde_sample_freq", "use_sde_at_warmup"]
        for key in optional_keys:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        self._model_kwargs = model_kwargs
        self.model = None
        self.ac = None
        self._ensure_model_env()

        self.test_fn = self.test_agent

    @staticmethod
    def _reset_env(env):
        out = env.reset()
        if isinstance(out, tuple):
            return out[0]
        return out

    @staticmethod
    def _step_env(env, action):
        out = env.step(action)
        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
            return obs, rew, (terminated or truncated), info
        return out

    def _make_train_env(self):
        if self.reinitialize:
            return self.env
        return RewardFnWrapper(
            self.env,
            reward_getter=lambda: self.reward_function,
            reward_state_indices=self.reward_state_indices,
            sa=self.sa,
        )

    def _ensure_model_env(self):
        train_env = self._make_train_env()
        vec_env = DummyVecEnv([lambda: train_env])
        if self.model is None:
            self.model = SB3SAC(env=vec_env, **self._model_kwargs)
        else:
            self.model.set_env(vec_env)
        self.ac = self.model.policy

    def _refresh_alpha(self):
        if hasattr(self.model, "ent_coef_tensor") and self.model.ent_coef_tensor is not None:
            self.alpha = self.model.ent_coef_tensor.detach().cpu()
        else:
            ent_coef = self.model.ent_coef
            if isinstance(ent_coef, (float, int)):
                self.alpha = torch.tensor(float(ent_coef))

    def get_action(self, o, deterministic=False, get_logprob=False):
        obs = np.asarray(o, dtype=np.float32)
        if obs.ndim == 1:
            batch_obs = obs[None, :]
        else:
            batch_obs = obs

        action, _ = self.model.predict(batch_obs, deterministic=deterministic)
        action = np.asarray(action)

        if action.ndim > 1 and action.shape[0] == 1:
            action_out = action[0]
        else:
            action_out = action

        if not get_logprob:
            return action_out

        try:
            obs_t = torch.as_tensor(batch_obs, dtype=torch.float32, device=self.model.device)
            with torch.no_grad():
                _, logp_t = self.model.policy.actor.action_log_prob(obs_t)
            logp = logp_t.detach().cpu().numpy()
            if np.asarray(logp).ndim > 0:
                return action_out, float(np.asarray(logp).reshape(-1)[0])
            return action_out, float(logp)
        except Exception:
            return action_out, 0.0

    def get_action_batch(self, o, deterministic=False):
        obs = np.asarray(o, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        action = np.asarray(action)
        try:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.model.device)
            with torch.no_grad():
                _, logp_t = self.model.policy.actor.action_log_prob(obs_t)
            logp = logp_t.detach().cpu().numpy()
        except Exception:
            logp = np.zeros((obs.shape[0],), dtype=np.float32)
        return action, logp

    def reset(self):
        pass

    def test_agent(self):
        avg_ep_return = 0.0
        for _ in range(self.num_test_episodes):
            o = self._reset_env(self.test_env)
            ep_return = 0.0
            for _ in range(self.max_ep_len):
                a = self.get_action(o, deterministic=True)
                o_next, _, done, _ = self._step_env(self.test_env, a)

                if self.reward_function is not None:
                    state = o
                    if self.reward_state_indices is not None:
                        state = state[self.reward_state_indices]
                    if self.sa:
                        state = np.concatenate([state, np.asarray(a)], axis=-1)
                    r = self.reward_function(state)
                    if isinstance(r, torch.Tensor):
                        r = float(r.detach().cpu().reshape(-1)[0])
                    elif isinstance(r, np.ndarray):
                        r = float(np.asarray(r).reshape(-1)[0])
                    else:
                        r = float(r)
                    ep_return += r

                o = o_next
                if done:
                    break
            avg_ep_return += ep_return
        return avg_ep_return / self.num_test_episodes

    def test_agent_ori_env(self, deterministic=True):
        rets = []
        for _ in range(self.num_test_episodes):
            ret = 0.0
            o = self._reset_env(self.test_env)
            for _ in range(self.max_ep_len):
                a = self.get_action(o, deterministic)
                o, r, done, _ = self._step_env(self.test_env, a)
                ret += r
                if done:
                    break
            rets.append(ret)
        return np.mean(rets)

    def test_agent_batch(self):
        # Retained for compatibility with old API.
        return self.test_agent_ori_env(deterministic=False), 0.0

    def learn(self, print_out=False, n_parallel=1):
        # For backward compatibility with non-MuJoCo callers.
        return self.learn_mujoco(print_out=print_out)

    def learn_mujoco(self, print_out=False, save_path=None):
        self._ensure_model_env()

        total_steps = int(self.steps_per_epoch * self.epochs)
        start_time = time.time()
        local_time = time.time()
        best_eval = -np.inf

        test_rets, alphas, log_pis, test_time_steps = [], [], [], []

        if total_steps <= 0:
            return [test_rets, alphas, log_pis, test_time_steps]

        log_interval = max(int(self.log_step_interval), 1)
        trained_steps = 0

        while trained_steps < total_steps:
            chunk = min(log_interval, total_steps - trained_steps)
            self.model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
            trained_steps += chunk

            test_epret = self.test_fn()
            if print_out:
                print(
                    f"SAC Training | Evaluation: {test_epret:.3f} Timestep: {trained_steps:d} Elapsed {time.time() - local_time:.0f}s"
                )

            if save_path is not None and test_epret > best_eval:
                best_eval = test_epret
                torch.save(self.ac.state_dict(), save_path)

            self._refresh_alpha()
            alphas.append(float(self.alpha.item()) if isinstance(self.alpha, torch.Tensor) else float(self.alpha))
            test_rets.append(float(test_epret))
            log_pis.append(0.0)
            test_time_steps.append(int(trained_steps))
            local_time = time.time()

        print(f"SAC Training End: time {time.time() - start_time:.0f}s")
        return [test_rets, alphas, log_pis, test_time_steps]

    @property
    def networks(self):
        nets = []
        if self.model is None:
            return nets
        if hasattr(self.model.policy, "actor"):
            nets.append(self.model.policy.actor)
        if hasattr(self.model.policy, "critic"):
            nets.append(self.model.policy.critic)
        if hasattr(self.model.policy, "critic_target"):
            nets.append(self.model.policy.critic_target)
        return nets
