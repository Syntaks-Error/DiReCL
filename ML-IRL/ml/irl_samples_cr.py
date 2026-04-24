"""
IRL training entry point for CommonRoad/highD expert demonstrations.

Run:
    python ml/irl_samples_cr.py configs/samples/agents/highD.yml
"""

import datetime
import json
import os
import sys
from typing import Dict, List, Tuple

import dateutil.tz
import numpy as np
import torch
import gymnasium as gym
from ruamel.yaml import YAML

from ml.models.reward import MLPReward
from common.ppo import PPO, read_commonroad_ppo_hyperparams
from utils import logger, system, eval


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs_from_project(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_project_root(), path))


def _setup_commonroad_imports(cfg: Dict):
    commonroad_root = cfg["env"].get("commonroad_rl_root", "../commonroad-rl")
    commonroad_root = _abs_from_project(commonroad_root)
    if commonroad_root not in sys.path:
        sys.path.insert(0, commonroad_root)

    # Register env ids such as "commonroad-v1"
    import commonroad_rl.gym_commonroad  # noqa: F401


def _build_env_kwargs(cfg: Dict) -> Dict:
    env_cfg = cfg["env"]
    kwargs = {}

    mapping = {
        "meta_scenario_path": "meta_scenario_path",
        "train_reset_config_path": "train_reset_config_path",
        "test_reset_config_path": "test_reset_config_path",
        "config_file": "config_file",
        "logging_path": "logging_path",
        "test_env": "test_env",
        "play": "play",
        "render_mode": "render_mode",
    }
    for src_k, dst_k in mapping.items():
        if src_k in env_cfg:
            kwargs[dst_k] = env_cfg[src_k]

    for k in ["meta_scenario_path", "train_reset_config_path", "test_reset_config_path", "config_file", "logging_path"]:
        if k in kwargs and isinstance(kwargs[k], str):
            kwargs[k] = _abs_from_project(kwargs[k])

    return kwargs


def _make_env_fn(cfg: Dict):
    env_name = cfg["env"]["env_name"]
    env_kwargs = _build_env_kwargs(cfg)
    return lambda: gym.make(env_name, **env_kwargs)


def _reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        return obs, rew, (terminated or truncated), info
    return out


def _load_expert_sa(cfg: Dict, state_size: int, action_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert_dir = cfg["env"].get("expert_data_dir", os.path.join(_project_root(), "expert_data", "highD"))
    expert_dir = _abs_from_project(expert_dir)

    states = np.load(os.path.join(expert_dir, "states.npy"))
    actions = np.load(os.path.join(expert_dir, "actions.npy"))
    lens_path = os.path.join(expert_dir, "lens.npy")

    if os.path.exists(lens_path):
        lens = np.load(lens_path).astype(np.int32)
    else:
        lens = np.full((states.shape[0],), states.shape[1], dtype=np.int32)

    assert states.shape[0] == actions.shape[0], "expert states/actions episode mismatch"
    assert states.shape[2] == state_size, f"state dim mismatch: expert={states.shape[2]} env={state_size}"
    assert actions.shape[2] == action_size, f"action dim mismatch: expert={actions.shape[2]} env={action_size}"

    state_indices = cfg["env"].get("state_indices", "all")
    if state_indices == "all":
        state_indices = list(range(state_size))

    sa_rows = []
    s_rows = []
    for ep in range(states.shape[0]):
        L = int(lens[ep])
        s = states[ep, :L, :][:, state_indices]
        a = actions[ep, :L, :]
        sa_rows.append(np.concatenate([s, a], axis=1))
        s_rows.append(s)

    expert_sa = np.concatenate(sa_rows, axis=0).astype(np.float32)
    expert_s = np.concatenate(s_rows, axis=0).astype(np.float32)
    return expert_sa, expert_s, np.array(state_indices, dtype=np.int64)


def _collect_agent_rollouts(env_fn, ppo_agent: PPO, n: int, horizon: int, state_indices: List[int]):
    env = env_fn()
    action_dim = env.action_space.shape[0]

    states = np.zeros((n, horizon, len(state_indices)), dtype=np.float32)
    actions = np.zeros((n, horizon, action_dim), dtype=np.float32)
    lens = np.zeros((n,), dtype=np.int32)

    for ep in range(n):
        obs = _reset_env(env)
        ep_len = 0
        for t in range(horizon):
            if (
                ppo_agent.normalize
                and ppo_agent.train_env is not None
                and hasattr(ppo_agent.train_env, "normalize_obs")
            ):
                obs_for_store = ppo_agent.train_env.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])[0]
            else:
                obs_for_store = obs

            act = ppo_agent.get_action(obs, deterministic=False)
            obs2, _, done, _ = _step_env(env, act)

            states[ep, t] = obs_for_store[state_indices]
            actions[ep, t] = act

            obs = obs2
            ep_len += 1
            if done:
                break

        lens[ep] = ep_len

    sa_rows = []
    s_rows = []
    for ep in range(n):
        L = int(lens[ep])
        s = states[ep, :L, :]
        a = actions[ep, :L, :]
        sa_rows.append(np.concatenate([s, a], axis=1))
        s_rows.append(s)

    agent_sa = np.concatenate(sa_rows, axis=0).astype(np.float32)
    agent_s = np.concatenate(s_rows, axis=0).astype(np.float32)
    return agent_sa, agent_s, lens


def _ml_sa_loss(agent_sa: np.ndarray, expert_sa: np.ndarray, reward_func: MLPReward, device, scale: float):
    sA = torch.as_tensor(agent_sa, dtype=torch.float32, device=device)
    sE = torch.as_tensor(expert_sa, dtype=torch.float32, device=device)

    tA = reward_func.r(sA).view(-1)
    tE = reward_func.r(sE).view(-1)
    return scale * (tA.mean() - tE.mean())


def _try_evaluate(cfg: Dict, itr: int, ppo_agent: PPO, env_fn, expert_states: np.ndarray, agent_states: np.ndarray):
    env_steps = int((itr + 1) * ppo_agent.total_timesteps_per_itr)
    metrics = eval.KL_summary(expert_states, agent_states, env_steps, "Running")

    real_return_det = eval.evaluate_real_return(
        ppo_agent.get_action, env_fn(), cfg["irl"]["eval_episodes"], cfg["env"]["T"], True
    )
    logger.record_tabular("Real Det Return", round(real_return_det, 2))

    real_return_sto = eval.evaluate_real_return(
        ppo_agent.get_action, env_fn(), cfg["irl"]["eval_episodes"], cfg["env"]["T"], False
    )
    logger.record_tabular("Real Sto Return", round(real_return_sto, 2))

    logger.record_tabular("Running Env Steps", env_steps)
    return metrics, real_return_det, real_return_sto


def main(cfg_path: str):
    yaml = YAML()
    cfg = yaml.load(open(cfg_path))

    assert cfg["obj"] == "maxentirl_sa", "irl_samples_cr.py currently supports obj=maxentirl_sa"
    assert cfg["IS"] is False, "Importance sampling mode is not supported in this script"

    _setup_commonroad_imports(cfg)

    device = torch.device(f"cuda:{cfg['cuda']}" if torch.cuda.is_available() and cfg["cuda"] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(int(cfg["seed"]))

    env_fn = _make_env_fn(cfg)
    env_kwargs = _build_env_kwargs(cfg)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]

    expert_sa, expert_s, state_indices = _load_expert_sa(cfg, state_size, action_size)

    exp_id = f"logs/{cfg['env']['env_name']}/exp-{cfg['irl']['expert_episodes']}/{cfg['obj']}_cr"
    os.makedirs(exp_id, exist_ok=True)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = os.path.join(exp_id, now.strftime("%Y_%m_%d_%H_%M_%S"))
    logger.configure(dir=log_folder)

    with open(os.path.join(logger.get_dir(), "variant.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    os.makedirs(os.path.join(log_folder, "model"), exist_ok=True)

    reward_func = MLPReward(len(state_indices) + action_size, **cfg["reward"], device=device).to(device)
    reward_optimizer = torch.optim.Adam(
        reward_func.parameters(),
        lr=cfg["reward"]["lr"],
        weight_decay=cfg["reward"]["weight_decay"],
        betas=(cfg["reward"]["momentum"], 0.999),
    )

    commonroad_root = _abs_from_project(cfg["env"].get("commonroad_rl_root", "../commonroad-rl"))
    ppo_policy, ppo_kwargs, normalize, normalize_kwargs = read_commonroad_ppo_hyperparams(
        env_id=cfg["env"]["env_name"],
        commonroad_root=commonroad_root,
        n_envs=int(cfg.get("ppo", {}).get("n_envs", 1)),
        overrides=cfg.get("ppo", {}).get("hyperparams", None),
    )
    ppo_cfg = cfg.get("ppo", {})
    n_envs = int(ppo_cfg.get("n_envs", 1))
    default_timesteps = int(cfg["env"]["T"] * ppo_kwargs.get("n_steps", 1024))
    n_timesteps_per_itr = int(ppo_cfg.get("n_timesteps_per_iter", default_timesteps))

    max_real_return_det, max_real_return_sto = -np.inf, -np.inf
    ppo_agent = None

    for itr in range(int(cfg["irl"]["n_itrs"])):
        if ppo_agent is None:
            ppo_agent = PPO(
                env_fn,
                env_id=cfg["env"]["env_name"],
                env_kwargs=env_kwargs,
                seed=int(cfg["seed"]),
                n_envs=n_envs,
                policy=ppo_policy,
                ppo_kwargs=ppo_kwargs,
                normalize=normalize,
                normalize_kwargs=normalize_kwargs,
                total_timesteps_per_itr=n_timesteps_per_itr,
                reward_state_indices=state_indices.tolist(),
                device=device,
                reinitialize=bool(ppo_cfg.get("reinitialize", False)),
                max_ep_len=int(cfg["env"]["T"]),
            )

        ppo_agent.reward_function = reward_func.get_scalar_reward
        ppo_agent.learn_mujoco(print_out=True)

        agent_sa, agent_s, agent_lens = _collect_agent_rollouts(
            env_fn,
            ppo_agent,
            n=int(cfg["irl"]["training_trajs"]),
            horizon=int(cfg["env"]["T"]),
            state_indices=state_indices.tolist(),
        )

        scale = float(cfg["env"]["T"])
        for _ in range(int(cfg["reward"]["gradient_step"])):
            loss = _ml_sa_loss(agent_sa, expert_sa, reward_func, device, scale=scale)
            reward_optimizer.zero_grad()
            loss.backward()
            reward_optimizer.step()

        _, real_return_det, real_return_sto = _try_evaluate(cfg, itr, ppo_agent, env_fn, expert_s, agent_s)

        if real_return_det > max_real_return_det and real_return_sto > max_real_return_sto:
            max_real_return_det, max_real_return_sto = real_return_det, real_return_sto
            torch.save(
                reward_func.state_dict(),
                os.path.join(
                    logger.get_dir(),
                    f"model/reward_model_itr{itr}_det{max_real_return_det:.0f}_sto{max_real_return_sto:.0f}.pkl",
                ),
            )

        logger.record_tabular("Iteration", itr)
        logger.record_tabular("Reward Loss", float(loss.item()))
        logger.record_tabular("Agent Avg Episode Len", round(float(agent_lens.mean()), 2))
        logger.dump_tabular()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python ml/irl_samples_cr.py <config.yml>")
    main(sys.argv[1])
