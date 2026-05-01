#!/usr/bin/env python3
"""
Evaluate a Stable-Baselines3 best_model.zip policy in CommonRoad-RL/highD.

This script reports:
  - return, length, episode time
  - goal reached / collision / off-road / timeout / friction violation
  - approximate highway TTC / THW metrics from closed-loop rollouts
  - progress, speed, acceleration, jerk comfort metrics

Assumptions:
  - You trained with SB3 and saved best_model.zip.
  - Your CommonRoad-RL env id is "commonroad-v1".
  - highD scenarios are already converted to CommonRoad-RL pickles.
  - For TTC/THW, vehicles are approximately highway-aligned; this is suitable
    for highD-style car-following/lane-change evaluation.

Example:
python evalute_policy_highd.py \
  --algo PPO \
  --model_path logs/best_model.zip \
    --config evalute_policy_highd.yaml \
  --out_dir eval_best_model
"""

import argparse
import json
import math
import os
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import gymnasium as gym
except ImportError:
    import gym

import commonroad_rl.gym_commonroad  # noqa: F401

from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


INFO_KEYS = (
    "is_collision",
    "is_time_out",
    "is_off_road",
    "is_friction_violation",
    "is_goal_reached",
)


ALGOS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "A2C": A2C,
    "DQN": DQN,
}


DEFAULT_EVAL_CONFIG = {
    "vecnormalize_path": None,
    "env_config": None,
    "meta_scenario_path": None,
    "problem_path": None,
    "n_episodes": 100,
    "max_steps": 2000,
    "device": "auto",
    "deterministic": True,
    "lane_margin": 1.0,
    "default_dt": 0.1,
}


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def make_commonroad_env(args):
    env_configs = load_yaml(args.env_config)

    # Evaluation should use test mode and test reset path.
    env_configs["test_env"] = True

    env = gym.make(
        "commonroad-v1",
        meta_scenario_path=args.meta_scenario_path,
        test_reset_config_path=args.problem_path,
        **env_configs,
    )

    # Monitor is important for terminal info flags and episode statistics.
    env = Monitor(env, filename=None, info_keywords=INFO_KEYS)
    return env


def build_vec_env(args):
    vec_env = DummyVecEnv([lambda: make_commonroad_env(args)])

    if args.vecnormalize_path is not None and os.path.exists(args.vecnormalize_path):
        vec_env = VecNormalize.load(args.vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    return vec_env


def unwrap_to_base_env(vec_env):
    """
    Unwrap DummyVecEnv/VecNormalize/Monitor to access CommonRoadEnv.
    """
    env = vec_env
    if isinstance(env, VecNormalize):
        env = env.venv
    if hasattr(env, "envs"):
        env = env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def safe_getattr(obj: Any, names: List[str], default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            return value() if callable(value) and name.startswith("get_") else value
    return default


def safe_scalar(x: Any, default=np.nan) -> float:
    if x is None:
        return float(default)
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)


def as_np_position(state) -> Optional[np.ndarray]:
    pos = safe_getattr(state, ["position"])
    if pos is None:
        return None
    return np.asarray(pos, dtype=float)


def state_orientation(state) -> float:
    ori = safe_getattr(state, ["orientation"], 0.0)
    try:
        return float(ori)
    except Exception:
        return 0.0


def state_speed(state) -> float:
    v = safe_getattr(state, ["velocity"], 0.0)
    try:
        return float(v)
    except Exception:
        return 0.0


def velocity_vector(state) -> np.ndarray:
    """
    CommonRoad states usually provide scalar velocity + orientation.
    """
    v = state_speed(state)
    psi = state_orientation(state)
    return np.array([v * math.cos(psi), v * math.sin(psi)], dtype=float)


def shape_length_width(obj, default_length=4.8, default_width=2.0) -> Tuple[float, float]:
    """
    Try several CommonRoad/CommonRoad-RL shape conventions.
    """
    shape = safe_getattr(obj, ["obstacle_shape", "shape"], None)
    if shape is not None:
        length = safe_getattr(shape, ["length"], default_length)
        width = safe_getattr(shape, ["width"], default_width)
        try:
            return float(length), float(width)
        except Exception:
            pass

    # Some ego vehicle implementations store parameters differently.
    params = safe_getattr(obj, ["parameters", "vehicle_params"], None)
    if params is not None:
        length = safe_getattr(params, ["l", "length"], default_length)
        width = safe_getattr(params, ["w", "width"], default_width)
        try:
            return float(length), float(width)
        except Exception:
            pass

    return default_length, default_width


def get_ego_vehicle_and_state(base_env):
    ego = safe_getattr(base_env, ["ego_vehicle", "ego"], None)
    if ego is None:
        return None, None

    state = safe_getattr(
        ego,
        ["state", "current_state", "initial_state"],
        None,
    )
    return ego, state


def get_current_time_step(base_env, ego_state) -> int:
    if ego_state is not None and hasattr(ego_state, "time_step"):
        return int(ego_state.time_step)

    t = safe_getattr(
        base_env,
        ["current_step", "_current_step", "time_step", "step_count"],
        0,
    )
    try:
        return int(t)
    except Exception:
        return 0


def get_scenario_dt(base_env, default_dt=0.1) -> float:
    scenario = safe_getattr(base_env, ["scenario"], None)
    if scenario is not None and hasattr(scenario, "dt"):
        try:
            return float(scenario.dt)
        except Exception:
            pass

    dt = safe_getattr(base_env, ["dt", "time_step_size"], default_dt)
    try:
        return float(dt)
    except Exception:
        return default_dt


def get_dynamic_obstacles(base_env):
    scenario = safe_getattr(base_env, ["scenario"], None)
    if scenario is None:
        return []
    return list(safe_getattr(scenario, ["dynamic_obstacles"], []) or [])


def compute_highway_ttc_thw(
    base_env,
    lane_margin: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Approximate TTC and THW for highD/highway scenarios.

    TTC is computed for same-lane vehicles ahead:
      TTC = longitudinal_gap / closing_speed
    where closing_speed > 0 means ego is approaching the front vehicle.

    THW is:
      THW = longitudinal_gap / ego_speed

    This is a practical closed-loop metric for highD. For official CommonRoad
    criticality measures such as TTS/TTR, use CommonRoad-CriMe after exporting
    the ego trajectory.
    """
    ego, ego_state = get_ego_vehicle_and_state(base_env)
    if ego is None or ego_state is None:
        return {
            "ttc": np.inf,
            "thw": np.inf,
            "front_gap": np.inf,
            "n_front_same_lane": 0,
        }

    ego_pos = as_np_position(ego_state)
    if ego_pos is None:
        return {
            "ttc": np.inf,
            "thw": np.inf,
            "front_gap": np.inf,
            "n_front_same_lane": 0,
        }

    ego_psi = state_orientation(ego_state)
    ego_vel = velocity_vector(ego_state)
    ego_speed = np.linalg.norm(ego_vel)
    ego_len, ego_width = shape_length_width(ego)

    # World -> ego frame rotation.
    c, s = math.cos(ego_psi), math.sin(ego_psi)
    rot = np.array([[c, s], [-s, c]], dtype=float)

    t = get_current_time_step(base_env, ego_state)

    best_ttc = np.inf
    best_thw = np.inf
    best_gap = np.inf
    n_front_same_lane = 0

    for obs in get_dynamic_obstacles(base_env):
        try:
            obs_state = obs.state_at_time(t)
        except Exception:
            obs_state = None

        if obs_state is None:
            continue

        obs_pos = as_np_position(obs_state)
        if obs_pos is None:
            continue

        obs_vel = velocity_vector(obs_state)
        obs_len, obs_width = shape_length_width(obs)

        rel_pos_ego = rot @ (obs_pos - ego_pos)
        rel_vel_ego = rot @ (obs_vel - ego_vel)

        longitudinal_center = rel_pos_ego[0]
        lateral_center = rel_pos_ego[1]

        lateral_limit = 0.5 * (ego_width + obs_width) + lane_margin
        same_lane = abs(lateral_center) <= lateral_limit

        # Positive gap means obstacle is in front of ego, accounting for length.
        gap = longitudinal_center - 0.5 * (ego_len + obs_len)

        if same_lane and gap > 0:
            n_front_same_lane += 1
            best_gap = min(best_gap, gap)

            # rel_vel_x < 0 means the front vehicle is slower in ego frame.
            closing_speed = -rel_vel_ego[0]
            if closing_speed > eps:
                best_ttc = min(best_ttc, gap / closing_speed)

            if ego_speed > eps:
                best_thw = min(best_thw, gap / ego_speed)

    return {
        "ttc": float(best_ttc),
        "thw": float(best_thw),
        "front_gap": float(best_gap),
        "n_front_same_lane": int(n_front_same_lane),
    }


def collect_ego_kinematics(base_env) -> Dict[str, float]:
    ego, state = get_ego_vehicle_and_state(base_env)
    if ego is None or state is None:
        return {
            "x": np.nan,
            "y": np.nan,
            "speed": np.nan,
            "orientation": np.nan,
        }

    pos = as_np_position(state)
    if pos is None:
        x, y = np.nan, np.nan
    else:
        x, y = float(pos[0]), float(pos[1])

    return {
        "x": x,
        "y": y,
        "speed": state_speed(state),
        "orientation": state_orientation(state),
    }


def collect_goal_progress(base_env) -> Dict[str, float]:
    """
    Read goal-distance related values from CommonRoad observation dict, when available.
    """
    obs_dict = safe_getattr(base_env, ["observation_dict"], None)
    if not isinstance(obs_dict, dict):
        return {
            "distance_goal_long": np.nan,
            "distance_goal_lat": np.nan,
            "distance_goal_time": np.nan,
        }

    return {
        "distance_goal_long": safe_scalar(obs_dict.get("distance_goal_long", np.nan), default=np.nan),
        "distance_goal_lat": safe_scalar(obs_dict.get("distance_goal_lat", np.nan), default=np.nan),
        "distance_goal_time": safe_scalar(obs_dict.get("distance_goal_time", np.nan), default=np.nan),
    }


def finite_or_nan(x: float) -> float:
    if x is None:
        return np.nan
    if np.isinf(x):
        return np.nan
    return float(x)


def summarize_episode(
    episode_id: int,
    rewards: List[float],
    infos: List[Dict[str, Any]],
    step_metrics: List[Dict[str, float]],
    dt: float,
    ttc_thresholds=(1.5, 2.0),
) -> Dict[str, Any]:
    arr_ttc = np.array([m["ttc"] for m in step_metrics], dtype=float)
    arr_thw = np.array([m["thw"] for m in step_metrics], dtype=float)
    arr_gap = np.array([m["front_gap"] for m in step_metrics], dtype=float)

    xs = np.array([m["x"] for m in step_metrics], dtype=float)
    ys = np.array([m["y"] for m in step_metrics], dtype=float)
    speeds = np.array([m["speed"] for m in step_metrics], dtype=float)
    headings = np.array([m["orientation"] for m in step_metrics], dtype=float)

    valid_pos = np.isfinite(xs) & np.isfinite(ys)
    if valid_pos.sum() >= 2:
        dx = np.diff(xs[valid_pos])
        dy = np.diff(ys[valid_pos])
        distance_travelled = float(np.sum(np.sqrt(dx**2 + dy**2)))

        init_heading = headings[np.where(valid_pos)[0][0]]
        direction = np.array([math.cos(init_heading), math.sin(init_heading)])
        pos0 = np.array([xs[valid_pos][0], ys[valid_pos][0]])
        pos_last = np.array([xs[valid_pos][-1], ys[valid_pos][-1]])
        forward_progress = float(np.dot(pos_last - pos0, direction))
    else:
        distance_travelled = np.nan
        forward_progress = np.nan

    # Acceleration and jerk from speed profile.
    valid_speed = np.isfinite(speeds)
    if valid_speed.sum() >= 3:
        sp = speeds[valid_speed]
        acc = np.diff(sp) / dt
        jerk = np.diff(acc) / dt
        mean_abs_acc = float(np.mean(np.abs(acc)))
        max_abs_acc = float(np.max(np.abs(acc)))
        mean_abs_jerk = float(np.mean(np.abs(jerk))) if len(jerk) else np.nan
        max_abs_jerk = float(np.max(np.abs(jerk))) if len(jerk) else np.nan
    else:
        mean_abs_acc = max_abs_acc = mean_abs_jerk = max_abs_jerk = np.nan

    # Lateral acceleration approximation: a_lat = v * yaw_rate.
    valid_heading = valid_speed & np.isfinite(headings)
    if valid_heading.sum() >= 2:
        hd = np.unwrap(headings[valid_heading])
        sp = speeds[valid_heading]
        yaw_rate = np.diff(hd) / dt
        sp_mid = sp[1:]
        lat_acc = sp_mid * yaw_rate
        mean_abs_lat_acc = float(np.mean(np.abs(lat_acc)))
        max_abs_lat_acc = float(np.max(np.abs(lat_acc)))
    else:
        mean_abs_lat_acc = max_abs_lat_acc = np.nan

    flags = {k: False for k in INFO_KEYS}
    termination_reason = None
    time_step = np.nan
    max_time_steps = np.nan
    ttc_lead_seq: List[float] = []
    ttc_follow_seq: List[float] = []
    for info in infos:
        if info.get("termination_reason") is not None:
            termination_reason = info.get("termination_reason")
        if "current_episode_time_step" in info:
            time_step = safe_scalar(info.get("current_episode_time_step"), default=time_step)
        if "max_episode_time_steps" in info:
            max_time_steps = safe_scalar(info.get("max_episode_time_steps"), default=max_time_steps)
        if "ttc_lead" in info:
            ttc_lead_seq.append(safe_scalar(info.get("ttc_lead"), default=np.nan))
        if "ttc_follow" in info:
            ttc_follow_seq.append(safe_scalar(info.get("ttc_follow"), default=np.nan))
        for k in INFO_KEYS:
            if k in info:
                flags[k] = flags[k] or bool(info[k])

    finite_ttc_lead_info = np.array(ttc_lead_seq, dtype=float)
    finite_ttc_lead_info = finite_ttc_lead_info[np.isfinite(finite_ttc_lead_info)]
    finite_ttc_follow_info = np.array(ttc_follow_seq, dtype=float)
    finite_ttc_follow_info = finite_ttc_follow_info[np.isfinite(finite_ttc_follow_info)]

    finite_ttc = arr_ttc[np.isfinite(arr_ttc)]
    finite_thw = arr_thw[np.isfinite(arr_thw)]
    finite_gap = arr_gap[np.isfinite(arr_gap)]

    out = {
        "episode": episode_id,
        "return": float(np.sum(rewards)),
        "length_steps": int(len(rewards)),
        "episode_time_s": float(len(rewards) * dt),
        "is_goal_reached": int(flags["is_goal_reached"]),
        "is_collision": int(flags["is_collision"]),
        "is_off_road": int(flags["is_off_road"]),
        "is_time_out": int(flags["is_time_out"]),
        "is_friction_violation": int(flags["is_friction_violation"]),
        "termination_reason": termination_reason,
        "episode_completion_ratio": (
            float(np.clip(time_step / max_time_steps, 0.0, 1.0))
            if np.isfinite(time_step) and np.isfinite(max_time_steps) and max_time_steps > 0
            else np.nan
        ),
        "min_ttc_s": float(np.min(finite_ttc)) if len(finite_ttc) else np.nan,
        "mean_ttc_s": float(np.mean(finite_ttc)) if len(finite_ttc) else np.nan,
        "min_ttc_lead_s": (float(np.min(finite_ttc_lead_info)) if len(finite_ttc_lead_info) else np.nan),
        "mean_ttc_lead_s": (float(np.mean(finite_ttc_lead_info)) if len(finite_ttc_lead_info) else np.nan),
        "min_ttc_follow_s": (float(np.min(finite_ttc_follow_info)) if len(finite_ttc_follow_info) else np.nan),
        "mean_ttc_follow_s": (float(np.mean(finite_ttc_follow_info)) if len(finite_ttc_follow_info) else np.nan),
        "min_thw_s": float(np.min(finite_thw)) if len(finite_thw) else np.nan,
        "mean_thw_s": float(np.mean(finite_thw)) if len(finite_thw) else np.nan,
        "min_front_gap_m": float(np.min(finite_gap)) if len(finite_gap) else np.nan,
        "distance_travelled_m": distance_travelled,
        "forward_progress_m": forward_progress,
        "mean_speed_mps": float(np.nanmean(speeds)) if np.isfinite(speeds).any() else np.nan,
        "max_speed_mps": float(np.nanmax(speeds)) if np.isfinite(speeds).any() else np.nan,
        "mean_abs_acc_mps2": mean_abs_acc,
        "max_abs_acc_mps2": max_abs_acc,
        "mean_abs_lat_acc_mps2": mean_abs_lat_acc,
        "max_abs_lat_acc_mps2": max_abs_lat_acc,
        "mean_abs_jerk_mps3": mean_abs_jerk,
        "max_abs_jerk_mps3": max_abs_jerk,
    }

    arr_goal_long = np.array([m.get("distance_goal_long", np.nan) for m in step_metrics], dtype=float)
    finite_goal_long = arr_goal_long[np.isfinite(arr_goal_long)]
    if len(finite_goal_long) >= 2:
        initial_goal_long = float(np.abs(finite_goal_long[0]))
        final_goal_long = float(np.abs(finite_goal_long[-1]))
        if initial_goal_long > 1e-6:
            route_completion = np.clip((initial_goal_long - final_goal_long) / initial_goal_long, 0.0, 1.0)
        else:
            route_completion = 1.0
        out["route_completion_ratio"] = float(route_completion)
    elif flags["is_goal_reached"]:
        out["route_completion_ratio"] = 1.0
    else:
        out["route_completion_ratio"] = np.nan

    for thr in ttc_thresholds:
        below = np.isfinite(arr_ttc) & (arr_ttc < thr)
        out[f"tet_ttc_below_{thr:.1f}s_s"] = float(np.sum(below) * dt)
        # TIT: time-integrated TTC deficit.
        deficit = np.where(below, thr - arr_ttc, 0.0)
        out[f"tit_ttc_below_{thr:.1f}s_s2"] = float(np.sum(deficit) * dt)

    return out


def aggregate_results(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    summary = {"n_episodes": int(n)}

    rate_cols = [
        "is_goal_reached",
        "is_collision",
        "is_off_road",
        "is_time_out",
        "is_friction_violation",
    ]
    for c in rate_cols:
        summary[c.replace("is_", "") + "_rate"] = float(df[c].mean())

    if "termination_reason" in df.columns:
        tr_counter = Counter([str(x) for x in df["termination_reason"].fillna("none").tolist()])
        summary["termination_reason_count"] = dict(tr_counter)
        summary["termination_reason_rate"] = {k: float(v / n) for k, v in tr_counter.items()} if n > 0 else {}

    scalar_cols = [
        "return",
        "length_steps",
        "episode_time_s",
        "min_ttc_s",
        "mean_ttc_s",
        "min_ttc_lead_s",
        "mean_ttc_lead_s",
        "min_ttc_follow_s",
        "mean_ttc_follow_s",
        "min_thw_s",
        "mean_thw_s",
        "min_front_gap_m",
        "episode_completion_ratio",
        "route_completion_ratio",
        "distance_travelled_m",
        "forward_progress_m",
        "mean_speed_mps",
        "max_speed_mps",
        "mean_abs_acc_mps2",
        "max_abs_acc_mps2",
        "mean_abs_lat_acc_mps2",
        "max_abs_lat_acc_mps2",
        "mean_abs_jerk_mps3",
        "max_abs_jerk_mps3",
        "tet_ttc_below_1.5s_s",
        "tit_ttc_below_1.5s_s2",
        "tet_ttc_below_2.0s_s",
        "tit_ttc_below_2.0s_s2",
    ]

    for c in scalar_cols:
        if c in df.columns:
            summary[c + "_mean"] = float(df[c].mean(skipna=True))
            summary[c + "_std"] = float(df[c].std(skipna=True))
            summary[c + "_median"] = float(df[c].median(skipna=True))

    # Useful robust safety statistic: 5th percentile of episode-level min TTC.
    if "min_ttc_s" in df.columns:
        summary["min_ttc_s_p5"] = float(df["min_ttc_s"].quantile(0.05))
        summary["min_ttc_s_p10"] = float(df["min_ttc_s"].quantile(0.10))

    return summary


def evaluate(args):
    cfg = load_yaml(args.config)
    run_args = argparse.Namespace(**vars(args))
    for k, v in DEFAULT_EVAL_CONFIG.items():
        setattr(run_args, k, cfg.get(k, v))

    required_from_cfg = ["meta_scenario_path", "problem_path"]
    missing = [k for k in required_from_cfg if getattr(run_args, k, None) in (None, "")]
    if missing:
        raise ValueError("Missing required config keys in YAML: " + ", ".join(missing))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vec_env = build_vec_env(run_args)
    base_env = unwrap_to_base_env(vec_env)
    dt = get_scenario_dt(base_env, default_dt=run_args.default_dt)

    model_cls = ALGOS[args.algo]
    model = model_cls.load(args.model_path, env=vec_env, device=run_args.device)

    episode_rows = []

    for ep in range(run_args.n_episodes):
        obs = vec_env.reset()
        done = False

        rewards = []
        infos = []
        step_metrics = []

        # Refresh base env after reset because wrappers may replace internals.
        base_env = unwrap_to_base_env(vec_env)
        dt = get_scenario_dt(base_env, default_dt=run_args.default_dt)

        while not done:
            action, _ = model.predict(obs, deterministic=run_args.deterministic)
            obs, reward, dones, info_list = vec_env.step(action)

            reward_scalar = float(np.asarray(reward).reshape(-1)[0])
            info = dict(info_list[0])
            done = bool(np.asarray(dones).reshape(-1)[0])

            base_env = unwrap_to_base_env(vec_env)

            ttc_thw = compute_highway_ttc_thw(
                base_env,
                lane_margin=run_args.lane_margin,
            )
            ego_kin = collect_ego_kinematics(base_env)
            goal_prog = collect_goal_progress(base_env)

            step_metrics.append({**ttc_thw, **ego_kin, **goal_prog})
            rewards.append(reward_scalar)
            infos.append(info)

            if len(rewards) >= run_args.max_steps:
                # Safety guard against environments that never terminate.
                info["is_time_out"] = True
                done = True

        row = summarize_episode(
            episode_id=ep,
            rewards=rewards,
            infos=infos,
            step_metrics=step_metrics,
            dt=dt,
            ttc_thresholds=(1.5, 2.0),
        )
        episode_rows.append(row)

        print(
            f"[{ep + 1:04d}/{run_args.n_episodes}] "
            f"R={row['return']:.2f}, "
            f"goal={row['is_goal_reached']}, "
            f"collision={row['is_collision']}, "
            f"offroad={row['is_off_road']}, "
            f"minTTC={row['min_ttc_s']:.3f}"
        )

    df = pd.DataFrame(episode_rows)
    summary = aggregate_results(df)

    per_episode_path = out_dir / "per_episode_metrics.csv"
    summary_path = out_dir / "summary_metrics.json"

    df.to_csv(per_episode_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(f"  per-episode metrics: {per_episode_path}")
    print(f"  summary metrics:     {summary_path}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, required=True, choices=ALGOS.keys())
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="eval_results")

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
