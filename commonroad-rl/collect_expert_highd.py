import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import commonroad_rl.gym_commonroad  # noqa: F401

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def collect_expert_demos(model_dir: str, meta_path: str, reset_config_path: str, out_dir: str, num_episodes: int = 25):
    model_path = os.path.join(model_dir, "best_model.zip")
    vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
    config_path = os.path.join(model_dir, "environment_configurations.yml")

    def make_env():
        return gym.make(
            "commonroad-v1",
            meta_scenario_path=meta_path,
            test_reset_config_path=reset_config_path,
            test_env=True,
            play=False,
            config_file=config_path,
            logging_path=model_dir,
        )

    base_env = DummyVecEnv([make_env])
    env = VecNormalize.load(vecnorm_path, base_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    states_list = []
    actions_list = []
    next_states_list = []
    lens = []

    obs = env.reset()
    for ep in range(num_episodes):
        ep_s = []
        ep_a = []
        ep_ns = []
        done = [False]
        steps = 0

        while not done[0]:
            state = obs[0].copy()
            action, _ = model.predict(obs, deterministic=True)
            obs2, _, done, info = env.step(action)

            next_state = obs2[0].copy()
            if done[0] and isinstance(info, (list, tuple)) and len(info) > 0 and "terminal_observation" in info[0]:
                term_obs = info[0]["terminal_observation"]
                if term_obs is not None:
                    next_state = np.array(term_obs, dtype=np.float32)

            ep_s.append(state)
            ep_a.append(action[0].copy())
            ep_ns.append(next_state)

            obs = obs2
            steps += 1

        states_list.append(np.asarray(ep_s, dtype=np.float32))
        actions_list.append(np.asarray(ep_a, dtype=np.float32))
        next_states_list.append(np.asarray(ep_ns, dtype=np.float32))
        lens.append(steps)
        print(f"Collected episode {ep + 1}/{num_episodes} length={steps}")

        obs = env.reset()

    max_len = int(max(lens))
    state_dim = int(states_list[0].shape[1])
    action_dim = int(actions_list[0].shape[1])

    states = np.zeros((num_episodes, max_len, state_dim), dtype=np.float32)
    actions = np.zeros((num_episodes, max_len, action_dim), dtype=np.float32)
    next_states = np.zeros((num_episodes, max_len, state_dim), dtype=np.float32)

    for i in range(num_episodes):
        L = lens[i]
        states[i, :L] = states_list[i]
        actions[i, :L] = actions_list[i]
        next_states[i, :L] = next_states_list[i]

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "states.npy"), states)
    np.save(os.path.join(out_dir, "actions.npy"), actions)
    np.save(os.path.join(out_dir, "next_states.npy"), next_states)
    np.save(os.path.join(out_dir, "lens.npy"), np.asarray(lens, dtype=np.int32))

    print("Saved to", out_dir)
    print("states", states.shape, states.dtype)
    print("actions", actions.shape, actions.dtype)
    print("next_states", next_states.shape, next_states.dtype)
    print("lens", np.asarray(lens).shape, min(lens), max(lens))


def main():
    parser = argparse.ArgumentParser(description="Collect expert demos from a CommonRoad SB3 PPO model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/wangar/Workspace/hw/commonroad-rl/logs/ppo/commonroad-v1_6",
        help="Directory containing best_model.zip, vecnormalize.pkl, and environment_configurations.yml",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="/home/wangar/Workspace/hw/commonroad-rl/pickles/meta_scenario",
        help="Path to meta_scenario directory",
    )
    parser.add_argument(
        "--reset-config-path",
        type=str,
        default="/home/wangar/Workspace/hw/commonroad-rl/pickles/problem_train",
        help="Path to scenario reset config directory (*.pickle)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/wangar/Workspace/hw/ML-IRL/expert_data/highD",
        help="Output directory for states.npy/actions.npy/next_states.npy",
    )
    parser.add_argument("--num-episodes", type=int, default=25, help="Number of expert episodes to collect")
    args = parser.parse_args()

    collect_expert_demos(
        model_dir=args.model_dir,
        meta_path=args.meta_path,
        reset_config_path=args.reset_config_path,
        out_dir=args.out_dir,
        num_episodes=args.num_episodes,
    )


if __name__ == "__main__":
    main()
