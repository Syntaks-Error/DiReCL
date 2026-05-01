"""
Module for evaluating trained model using Stable-Baselines3
"""

import argparse
import csv
import glob
import importlib
import logging
import os

os.environ["KMP_WARNINGS"] = "off"
os.environ["KMP_AFFINITY"] = "none"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").disabled = True
import pickle
from typing import Union
import numpy as np
import yaml
import gym
from gym import Env
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.gym_commonroad.constants import PATH_PARAMS
import commonroad_rl.gym_commonroad  # noqa: F401

try:
    MPI = importlib.import_module("mpi4py").MPI
except Exception:
    print("ImportFailure MPI")
    MPI = None

LOGGER = logging.getLogger(__name__)
LOGGING_MODE_CHOICES = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

ALGOS = {
    "ppo": PPO,
    "ppo2": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
    "dqn": DQN,
}


def _resolve_model_file(model_path: str, preferred_name: str = "best_model.zip") -> str:
    if os.path.isfile(model_path):
        return model_path

    files = os.listdir(model_path)
    if preferred_name in files:
        return os.path.join(model_path, preferred_name)

    candidates = sorted(glob.glob(os.path.join(model_path, "rl_model*.zip")))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No model checkpoint found under: {model_path}")

    def extract_number(f):
        import re

        s = re.findall("\\d+", f)
        return int(s[-1]) if s else -1, f

    return max(candidates, key=extract_number)


def _get_wrapper_class(hyperparams: dict):
    wrapper_name = hyperparams.get("env_wrapper")
    if wrapper_name is None:
        return None

    if isinstance(wrapper_name, list):
        wrapper_names = wrapper_name
    else:
        wrapper_names = [wrapper_name]

    wrapper_classes = []
    wrapper_kwargs = []

    for wn in wrapper_names:
        if isinstance(wn, dict):
            assert len(wn) == 1
            wn_key = list(wn.keys())[0]
            kwargs = wn[wn_key]
            wn = wn_key
        else:
            kwargs = {}

        module_name = ".".join(wn.split(".")[:-1])
        class_name = wn.split(".")[-1]
        wrapper_module = importlib.import_module(module_name)
        wrapper_class = getattr(wrapper_module, class_name)
        wrapper_classes.append(wrapper_class)
        wrapper_kwargs.append(kwargs)

    def _wrap(env):
        for wc, kwargs in zip(wrapper_classes, wrapper_kwargs):
            env = wc(env, **kwargs)
        return env

    return _wrap


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluates SB3 trained model with specified test scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env_id", type=str, default="commonroad-v1", help="environment ID")
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument(
        "--scenario_path",
        "-i",
        type=str,
        help="Path to pickled test scenarios",
        default=PATH_PARAMS["test_reset_config"],
    )
    parser.add_argument("--test_folder", type=str, help="Folder for problem pickles", default="problem_test")
    parser.add_argument(
        "--model_path",
        "-model",
        type=str,
        help="Path to trained model",
        default=PATH_PARAMS["log"] + "/ppo2/commonroad-v0_3",
    )
    parser.add_argument(
        "--evaluation_path", "-eval_path", type=str, help="Folder to store evaluation data", default="evaluation"
    )
    parser.add_argument("--viz_path", "-viz", type=str, default="")
    parser.add_argument("--num_scenarios", "-n", default=-1, type=int, help="Maximum number of scenarios to draw")
    parser.add_argument("--multiprocessing", "-mpi", action="store_true")
    parser.add_argument(
        "--combine_frames", "-1", action="store_true", help="Combine rendered environments into one picture"
    )
    parser.add_argument(
        "--skip_timesteps", "-st", type=int, default=1, help="Only render every nth frame (including first and last)"
    )
    parser.add_argument("--no_render", "-nr", action="store_true", help="Whether store render images")
    parser.add_argument("--no_csv", action="store_true", help="Not store statistic results to speed up MPI processes")
    parser.add_argument("--hyperparam_filename", "-hyperparam_f", type=str, default="model_hyperparameters.yml")
    parser.add_argument("--config_filename", "-config_f", type=str, default="environment_configurations.yml")
    parser.add_argument("--log_action_curve", action="store_true", help="Store action curve plot for analysis")
    parser.add_argument("--log_step_info", action="store_true", help="Store all info dict in step function")
    parser.add_argument("--logging_mode", type=str, default="INFO", choices=LOGGING_MODE_CHOICES)

    return parser


def create_environments(
    env_id: str,
    model_path: str,
    viz_path: str,
    hyperparam_filename: str = "model_hyperparameters.yml",
    config_filename: str = "environment_configurations.yml",
    logging_mode: str = "INFO",
    test_path=None,
    meta_path=None,
    play: bool = True,
    test_env: bool = True,
    **kwargs,
):
    """
    Create CommonRoad vectorized environment environment

    :param config_filename:
    :param env_id: Environment gym id
    :param test_path: Path to the test files
    :param meta_path: Path to the meta-scenarios
    :param model_path: Path to the trained model
    :param viz_path: Output path for rendered images
    :param hyperparam_filename: The filename of the hyperparameters
    :param env_kwargs: Keyword arguments to be passed to the environment
    """
    # Get environment keyword arguments including observation and reward configurations
    config_fn = os.path.join(model_path, config_filename)
    env_kwargs = {
        "logging_mode": logging_mode,
        "visualization_path": viz_path,
        "play": play,
        "test_env": test_env,
        "config_file": config_fn,
    }

    env_kwargs.update(kwargs)
    if meta_path is not None:
        env_kwargs.update({"meta_scenario_path": meta_path})
    if test_path is not None:
        env_kwargs.update({"test_reset_config_path": test_path})

    # Load model hyperparameters:
    hyperparam_fn = os.path.join(model_path, hyperparam_filename)
    with open(hyperparam_fn, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.Loader)

    env_wrapper = _get_wrapper_class(hyperparams)

    env = gym.make(env_id, logging_path=None, **env_kwargs)
    if env_wrapper:
        env = env_wrapper(env)
    env = Monitor(env, filename=None)

    return env, bool(hyperparams.get("normalize", False)), env_kwargs


def build_obs_normalizer(normalize: bool, model_path: str, env_id: str, env_kwargs: dict):
    if not normalize:
        return None

    vecnormalize_path = os.path.join(model_path, "vecnormalize.pkl")
    if not os.path.exists(vecnormalize_path):
        LOGGER.warning("normalize=True but vecnormalize.pkl not found, evaluating without obs normalization")
        return None

    def _factory():
        import gym

        e = gym.make(env_id, logging_path=None, **env_kwargs)
        return Monitor(e, filename=None)

    dummy_vec = DummyVecEnv([_factory])
    vec_norm = VecNormalize.load(vecnormalize_path, dummy_vec)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return vec_norm


def main():
    args = get_parser().parse_args()

    if MPI is None:
        args.multiprocessing = False

    log_level = getattr(logging, str(args.logging_mode).upper(), logging.INFO)
    LOGGER.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    LOGGER.addHandler(handler)

    meta_path = os.path.join(args.scenario_path, "meta_scenario")

    # mpi for parallel processing
    rank = 0
    size = 1
    comm = None
    if args.multiprocessing:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        test_path = os.path.join(args.scenario_path, args.test_folder, str(rank))

    else:
        test_path = os.path.join(args.scenario_path, args.test_folder)

    # create evaluation folder in model_path
    evaluation_path = os.path.join(args.model_path, args.evaluation_path)
    os.makedirs(evaluation_path, exist_ok=True)
    if args.viz_path == "":
        args.viz_path = os.path.join(evaluation_path, "img")

    env, normalize, env_kwargs = create_environments(
        args.env_id,
        args.model_path,
        args.viz_path,
        args.hyperparam_filename,
        args.config_filename,
        args.logging_mode,
        test_path=test_path,
        meta_path=meta_path,
    )

    LOGGER.info(f"Testing a maximum of {args.num_scenarios} scenarios")

    algo_key = args.algo.lower()
    if algo_key not in ALGOS:
        raise ValueError(f"Unsupported SB3 algorithm: {args.algo}. Supported: {list(ALGOS.keys())}")
    model_file = _resolve_model_file(args.model_path)
    model = ALGOS[algo_key].load(model_file)
    vec_norm = build_obs_normalizer(normalize, args.model_path, args.env_id, env_kwargs)

    set_random_seed(1)
    (
        num_valid_collisions,
        num_collisions,
        num_valid_off_road,
        num_off_road,
        num_goal_reaching,
        num_timeout,
        total_scenarios,
    ) = (0, 0, 0, 0, 0, 0, 0)

    if args.log_action_curve:
        accelerations = {}
        jerks = {}

    # In case there a no scenarios at all
    try:
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    except IndexError:
        args.num_scenarios = 0

    count = 0
    info_dict = {}

    while count != args.num_scenarios:
        done = False
        if not args.no_render:
            env.render()
        benchmark_id = env.benchmark_id
        LOGGER.debug(benchmark_id)
        if args.log_action_curve:
            accelerations[benchmark_id] = []
            jerks[benchmark_id] = []
        while not done:
            if vec_norm is not None:
                obs_in = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])[0]
            else:
                obs_in = obs

            action, _states = model.predict(obs_in, deterministic=True)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out
            if not args.no_render:
                env.render()
            if args.log_step_info:
                for kwargs in info:
                    if kwargs not in info_dict:
                        info_dict[kwargs] = [info[kwargs]]
                    else:
                        info_dict[kwargs].append(info[kwargs])

            if done:
                LOGGER.info(f'scenario={info["scenario_name"]}, termination_reason={info["termination_reason"]}')

            if args.log_action_curve:
                dt = env.scenario.dt
                ego_vehicle = env.ego_action.vehicle
                jerk_x = (ego_vehicle.state.acceleration - ego_vehicle.previous_state.acceleration) / dt
                jerk_y = (ego_vehicle.state.acceleration_y - ego_vehicle.previous_state.acceleration_y) / dt
                accelerations[benchmark_id].append([ego_vehicle.state.acceleration, ego_vehicle.state.acceleration_y])
                jerks[benchmark_id].append([jerk_x, jerk_y])

        # log collision rate, off-road rate, and goal-reaching rate
        total_scenarios += 1
        num_valid_collisions += info.get("valid_collision", info["is_collision"])
        num_collisions += info["is_collision"]
        num_timeout += info["is_time_out"]
        num_valid_off_road += info.get("valid_off_road", info["is_off_road"])
        num_off_road += info["is_off_road"]
        num_goal_reaching += info["is_goal_reached"]

        try:
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            out_of_scenarios = False
        except IndexError:
            out_of_scenarios = True

        if not args.no_csv:
            with open(os.path.join(evaluation_path, f"{rank}_results.csv"), "a") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    (info["scenario_name"], info["current_episode_time_step"], info["termination_reason"])
                )

        if out_of_scenarios:
            break
        count += 1

    if args.log_action_curve:
        for key in accelerations.keys():
            accelerations[key] = np.array(accelerations[key])
            jerks[key] = np.array(jerks[key])
        with open(f"{args.model_path.split('/')[-1]}_actions.pkl", "wb") as f:
            pickle.dump({"accelerations": accelerations, "jerks": jerks}, f)

    if args.log_step_info:
        with open(f"{os.path.abspath(args.model_path)}_step_info_{rank}.pkl", "wb") as f:
            pickle.dump(info_dict, f)
    if args.multiprocessing:
        data = (
            total_scenarios,
            num_valid_collisions,
            num_collisions,
            num_valid_off_road,
            num_off_road,
            num_timeout,
            num_goal_reaching,
        )
        data = comm.gather(data)
        if rank == 0:
            (
                g_num_scenarios,
                g_num_valid_collisions,
                g_num_collisions,
                g_num_valid_off_road,
                g_num_off_road,
                g_num_timeout,
                g_num_goal_reaching,
            ) = zip(*data)
            total_scenarios = sum(g_num_scenarios)
            num_valid_collisions = sum(g_num_valid_collisions)
            num_collisions = sum(g_num_collisions)
            num_valid_off_road = sum(g_num_valid_off_road)
            num_off_road = sum(g_num_off_road)
            num_timeout = sum(g_num_timeout)
            num_goal_reaching = sum(g_num_goal_reaching)
        else:
            return
    if not args.no_csv:
        # save evaluation results
        with open(os.path.join(evaluation_path, "results.csv"), "w") as fd_result:
            fd_result.write("benchmark_id, time_steps, termination_reason\n")
            for i in range(size):
                path = os.path.join(evaluation_path, f"{i}_results.csv")
                with open(path, "r") as f:
                    fd_result.write(f.read())
                os.remove(path)

        with open(os.path.join(evaluation_path, "overview.yml"), "w") as f:
            yaml.dump(
                {
                    "total_scenarios": total_scenarios,
                    "num_collisions": num_collisions,
                    "num_valid_collisions": num_valid_collisions,
                    "num_valid_off_road": num_valid_off_road,
                    "num_timeout": num_timeout,
                    "num_off_road": num_off_road,
                    "num_goal_reached": num_goal_reaching,
                    "percentage_goal_reached": 100.0 * num_goal_reaching / total_scenarios,
                    "percentage_off_road": 100.0 * num_off_road / total_scenarios,
                    "percentage_collisions": 100.0 * num_collisions / total_scenarios,
                    "percentage_valid_off_road": 100.0 * num_valid_off_road / total_scenarios,
                    "percentage_valid_collisions": 100.0 * num_valid_collisions / total_scenarios,
                    "percentage_timeout": 100.0 * num_timeout / total_scenarios,
                },
                f,
            )

    if not args.no_csv:
        # TODO: fix for no_render mode
        # Reorganize the rendered images according to result of the scenario
        # Flatten directory
        if not args.no_render:
            img_path = args.viz_path
            for d in os.listdir(img_path):
                dir_path = os.path.join(img_path, d)
                if not os.path.isdir(dir_path):
                    continue

                for f in os.listdir(dir_path):
                    os.rename(os.path.join(dir_path, f), os.path.join(img_path, f))

                os.rmdir(dir_path)

            # Split into different termination reasons
            with open(os.path.join(evaluation_path, "results.csv"), "r") as f:
                os.mkdir(os.path.join(img_path, "time_out"))
                os.mkdir(os.path.join(img_path, "off_road"))
                os.mkdir(os.path.join(img_path, "collision"))
                os.mkdir(os.path.join(img_path, "collision", "valid_collision"))
                os.mkdir(os.path.join(img_path, "collision", "collision_caused_by_other_vehicle"))
                os.mkdir(os.path.join(img_path, "goal_reached"))
                os.mkdir(os.path.join(img_path, "other"))

                reader = csv.reader(f)
                reader.__next__()
                for [scenario_id, _, t_reason] in reader:
                    if args.no_render:
                        if t_reason == "valid_collision" or t_reason == "collision_caused_by_other_vehicle":
                            dest_path = os.path.join(img_path, "collision", t_reason)
                        else:
                            dest_path = os.path.join(img_path, t_reason)
                    else:
                        if t_reason == "valid_collision" or t_reason == "collision_caused_by_other_vehicle":
                            os.mkdir(os.path.join(img_path, "collision", t_reason, scenario_id))
                            dest_path = os.path.join(img_path, "collision", t_reason, scenario_id)
                        else:
                            os.mkdir(os.path.join(img_path, t_reason, scenario_id))
                            dest_path = os.path.join(img_path, t_reason, scenario_id)
                    for file_path in glob.glob(os.path.join(img_path, scenario_id + "*")):
                        os.rename(file_path, os.path.join(dest_path, os.path.basename(file_path)))


if __name__ == "__main__":
    main()
