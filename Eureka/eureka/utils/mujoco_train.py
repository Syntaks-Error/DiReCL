import argparse
import importlib.util
import inspect
import json
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


class RewardFnWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.last_obs = None
        self.reward_signature = inspect.signature(reward_fn)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = torch.as_tensor(np.asarray(obs).copy(), dtype=torch.float32)
        return obs, info

    @staticmethod
    def _to_float_scalar(x):
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().reshape(-1)[0].item())
        return float(np.asarray(x).reshape(-1)[0])

    def _call_reward_fn(self, obs, action, next_obs, info, gt_reward, terminated, truncated):
        context = {
            "obs": obs,
            "observation": obs,
            "observations": obs,
            "action": action,
            "actions": action,
            "act": action,
            "next_obs": next_obs,
            "next_observation": next_obs,
            "next_observations": next_obs,
            "new_obs": next_obs,
            "info": info,
            "infos": info,
            "gt_reward": gt_reward,
            "env_reward": gt_reward,
            "reward": gt_reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": terminated or truncated,
        }

        kwargs = {}
        for name in self.reward_signature.parameters.keys():
            if name not in context:
                raise ValueError(f"Unsupported reward function argument: {name}")
            kwargs[name] = context[name]

        output = self.reward_fn(**kwargs)
        if isinstance(output, tuple) and len(output) == 2:
            rew, rew_dict = output
        else:
            rew, rew_dict = output, {}

        rew = self._to_float_scalar(rew)
        rew_dict = rew_dict or {}
        rew_dict = {str(k): self._to_float_scalar(v) for k, v in rew_dict.items()}
        return rew, rew_dict

    def step(self, action):
        obs, gt_reward, terminated, truncated, info = self.env.step(action)

        obs_t = self.last_obs
        action_t = torch.as_tensor(np.asarray(action).copy(), dtype=torch.float32)
        next_obs_t = torch.as_tensor(np.asarray(obs).copy(), dtype=torch.float32)
        gt_reward_t = torch.as_tensor(gt_reward, dtype=torch.float32)
        terminated_t = torch.as_tensor(terminated)
        truncated_t = torch.as_tensor(truncated)

        rew, rew_dict = self._call_reward_fn(
            obs_t,
            action_t,
            next_obs_t,
            info,
            gt_reward_t,
            terminated_t,
            truncated_t,
        )
        self.last_obs = next_obs_t

        info = dict(info)
        info["gt_reward"] = float(gt_reward)
        info["gpt_reward"] = float(rew)
        for k, v in rew_dict.items():
            info[k] = float(v)
        return obs, rew, terminated, truncated, info


class MetricsCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, metrics_file, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = int(n_eval_episodes)
        self.metrics_file = metrics_file

        self.metrics = defaultdict(list)
        self._reward_sums = defaultdict(float)
        self._reward_counts = defaultdict(int)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            for key, value in info.items():
                if not isinstance(value, (int, float, np.number)):
                    continue
                self._reward_sums[key] += float(value)
                self._reward_counts[key] += 1

        if self.num_timesteps % self.eval_freq == 0:
            mean_eval_return, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )

            self.metrics["task_score"].append(float(mean_eval_return))

            for key, value in self._reward_sums.items():
                count = max(self._reward_counts[key], 1)
                self.metrics[key].append(float(value / count))

            self._reward_sums = defaultdict(float)
            self._reward_counts = defaultdict(int)

            with open(self.metrics_file, "w") as f:
                json.dump(dict(self.metrics), f, indent=2)

        return True


def load_reward_function(reward_file: str):
    spec = importlib.util.spec_from_file_location("eureka_generated_reward", reward_file)
    module = importlib.util.module_from_spec(spec)
    # Some generated files use `torch.*` in type hints or reward math without importing torch.
    # Provide torch in the module globals to avoid NameError during import-time evaluation.
    module.__dict__.setdefault("torch", torch)
    spec.loader.exec_module(module)

    fn_candidates = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and not name.startswith("_"):
            fn_candidates.append((name, obj))

    if not fn_candidates:
        raise RuntimeError("No callable reward function found in generated reward file.")

    # Prefer canonical function name if present.
    for name, fn in fn_candidates:
        if name == "compute_reward":
            return fn

    return fn_candidates[0][1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--reward-file", type=str, required=True)
    parser.add_argument("--metrics-file", type=str, required=True)
    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()

    reward_fn = load_reward_function(args.reward_file)

    train_env = RewardFnWrapper(Monitor(gym.make(args.env_id)), reward_fn)
    eval_env = Monitor(gym.make(args.env_id))

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=None,
        seed=args.seed,
        device=args.device,
    )

    callback = MetricsCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        metrics_file=args.metrics_file,
    )

    print("TRAINING_START")
    print(f"Metrics File: {args.metrics_file}")
    model.learn(total_timesteps=int(args.total_timesteps), callback=callback)

    if "task_score" not in callback.metrics:
        mean_eval_return, _ = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )
        callback.metrics["task_score"].append(float(mean_eval_return))
        with open(args.metrics_file, "w") as f:
            json.dump(dict(callback.metrics), f, indent=2)


if __name__ == "__main__":
    main()
