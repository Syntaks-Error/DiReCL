import argparse
import glob
import os

import gymnasium as gym
import numpy as np
import torch
from ruamel.yaml import YAML

import envs  # noqa: F401
from common.sac import SAC
from utils import system


def _reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        return obs, rew, (terminated or truncated), info
    return out


def _resolve_policy_path(env_name, seed, explicit_path=None):
    if explicit_path is not None:
        return explicit_path

    default_path = f"expert_data/optimal_policy/CustomAntgd_{env_name}_{seed}.pt"
    if os.path.exists(default_path):
        return default_path

    pattern = f"expert_data/optimal_policy/*_{env_name}_{seed}.pt"
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find policy checkpoint for env={env_name}, seed={seed}. "
        f"Tried {default_path} and pattern {pattern}."
    )


def collect_expert_trajs(cfg_path, policy_path=None):
    yaml = YAML()
    v = yaml.load(open(cfg_path))

    env_name, env_T = v["env"]["env_name"], v["env"]["T"]
    samples_episode = int(v["expert"]["samples_episode"])
    seed = int(v["seed"])

    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v["cuda"] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)

    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()

    sac_agent = SAC(env_fn, steps_per_epoch=env_T, max_ep_len=env_T, seed=seed, device=device, **v["sac"])

    ckpt_path = _resolve_policy_path(env_name, seed, policy_path)
    print(f"Loading policy checkpoint: {ckpt_path}")
    sac_agent.ac.load_state_dict(torch.load(ckpt_path, map_location=device))

    state_dim = gym_env.observation_space.shape[0]
    action_dim = gym_env.action_space.shape[0]

    states = np.zeros((samples_episode, env_T, state_dim), dtype=np.float32)
    actions = np.zeros((samples_episode, env_T, action_dim), dtype=np.float32)
    rewards = np.zeros((samples_episode, env_T), dtype=np.float32)
    next_states = np.zeros((samples_episode, env_T, state_dim), dtype=np.float32)
    dones = np.zeros((samples_episode, env_T), dtype=np.bool_)
    lens = np.zeros((samples_episode,), dtype=np.int32)

    for ep in range(samples_episode):
        obs = _reset_env(gym_env)
        ep_len = 0
        for t in range(env_T):
            act = sac_agent.get_action(obs, deterministic=True)
            obs2, rew, done, _ = _step_env(gym_env, act)

            states[ep, t] = obs
            actions[ep, t] = act
            rewards[ep, t] = rew
            next_states[ep, t] = obs2
            dones[ep, t] = done

            obs = obs2
            ep_len += 1
            if done:
                break

        lens[ep] = ep_len
        print(f"Collected episode {ep + 1}/{samples_episode} | length={ep_len}")

    env_tag = env_name.split("-")[0]
    out_dir = os.path.join("expert_data", env_tag)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "states.npy"), states)
    np.save(os.path.join(out_dir, "actions.npy"), actions)
    np.save(os.path.join(out_dir, "rewards.npy"), rewards)
    np.save(os.path.join(out_dir, "next_states.npy"), next_states)
    np.save(os.path.join(out_dir, "dones.npy"), dones)
    np.save(os.path.join(out_dir, "lens.npy"), lens)

    print(f"Saved expert data to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--policy-path", type=str, default=None)
    args = parser.parse_args()

    collect_expert_trajs(args.config, args.policy_path)
