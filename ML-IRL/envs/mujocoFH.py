# Fixed Horizon wrapper of mujoco environments
import gymnasium as gym
import numpy as np


class MujocoFH(gym.Env):
    def __init__(self, env_name, T=1000, r=None, obs_mean=None, obs_std=None, seed=1):
        self.env = gym.make(env_name)
        self.T = T
        self.r = r
        assert (obs_mean is None and obs_std is None) or (obs_mean is not None and obs_std is not None)
        self.obs_mean, self.obs_std = obs_mean, obs_std

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.seed(seed)

    def seed(self, seed):
        try:
            self.env.reset(seed=seed)
        except TypeError:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

    @staticmethod
    def _as_obs(reset_out):
        if isinstance(reset_out, tuple):
            return reset_out[0]
        return reset_out

    @staticmethod
    def _step_unpack(step_out):
        if len(step_out) == 5:
            obs, r, terminated, truncated, info = step_out
            return obs, r, bool(terminated), bool(truncated), info
        obs, r, done, info = step_out
        return obs, r, bool(done), False, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.t = 0
        self.terminated = False
        self.terminal_state = None

        reset_out = self.env.reset() if options is None else self.env.reset(options=options)
        self.obs = self._as_obs(reset_out)
        self.obs = self.normalize_obs(self.obs)
        return self.obs.copy(), {}

    def step(self, action):
        self.t += 1

        if self.terminated:
            truncated = self.t >= self.T
            return self.terminal_state.copy(), 0.0, False, truncated, {}
        else:
            prev_obs = self.obs.copy()
            self.obs, r, terminated, truncated, info = self._step_unpack(self.env.step(action))
            self.obs = self.normalize_obs(self.obs)

            if self.r is not None:  # from irl model
                r = self.r(prev_obs)

            done = terminated or truncated
            if done:
                self.terminated = True
                self.terminal_state = self.obs.copy()

            if self.t >= self.T and not terminated:
                truncated = True

            return self.obs.copy(), r, terminated, truncated, info

    def normalize_obs(self, obs):
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std
        return obs
