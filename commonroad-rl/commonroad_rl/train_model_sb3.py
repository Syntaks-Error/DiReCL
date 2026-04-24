"""
Train CommonRoad agents with Stable-Baselines3 PPO (PyTorch only).
"""

import argparse
import copy
import difflib
import glob
import importlib
import logging
import os
import sys
import time
import uuid
from pprint import pformat

# If TensorFlow is imported indirectly by dependencies, suppress noisy startup logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    import gymnasium as gym
except ImportError:
    import gym

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import commonroad_rl.gym_commonroad
from commonroad_rl.gym_commonroad.constants import ROOT_STR, PATH_PARAMS


LOGGER = logging.getLogger(__name__)


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, 'args2': {'a': 1}}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)


def linear_schedule(initial_value: float):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def get_latest_run_id(log_path: str, env_id: str) -> int:
    max_run_id = 0
    for path in glob.glob(log_path + f"/{env_id}_[0-9]*"):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def construct_logger(save_path: str):
    LOGGER.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(save_path, "console_copy.txt"))
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    LOGGER.handlers.clear()
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", type=str, default="commonroad-v1", help="environment ID")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo"], help="RL algorithm")
    parser.add_argument("-tb", "--tensorboard-log", default="", type=str, help="Tensorboard log dir")
    parser.add_argument("-i", "--trained-agent", type=str, default="", help="Path to a pretrained .zip agent")
    parser.add_argument("-n", "--n-timesteps", default=int(1e6), type=int, help="Number of timesteps")
    parser.add_argument("--log-interval", default=-1, type=int, help="Override log interval")
    parser.add_argument("--eval-freq", default=10000, type=int, help="Evaluate every n steps (negative disables)")
    parser.add_argument(
        "--eval_timesteps", default=1000, type=int, help="Not used for PPO eval, kept for compatibility"
    )
    parser.add_argument("--eval_episodes", default=5, type=int, help="Evaluation episodes")
    parser.add_argument("--save-freq", default=-1, type=int, help="Checkpoint frequency")
    parser.add_argument("-f", "--log-folder", type=str, default="logs", help="Log folder")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--configs-path", type=str, default="", help="Override env config file")
    parser.add_argument("--hyperparams-path", type=str, default="", help="Override hyperparams file")
    parser.add_argument(
        "-params", "--hyperparams", type=str, nargs="+", action=StoreDict, help="Overwrite model hyperparameters"
    )
    parser.add_argument("-uuid", "--uuid", choices=["top", "none", "true"], type=str, default="none")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Overwrite env kwargs")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of vectorized environments")
    parser.add_argument("--gym-packages", type=str, nargs="+", default=[], help="Additional gym env packages to import")
    parser.add_argument("--info_keywords", type=str, nargs="+", default=(), help="Extra info keys to monitor")
    return parser.parse_args(sys.argv[1:])


def construct_save_path(args):
    log_path = os.path.join(args.log_folder, args.algo)
    if args.uuid == "top":
        return args.log_folder
    if args.uuid == "true":
        uuid_str = f"_{uuid.uuid4()}"
    else:
        uuid_str = ""
    run_id = get_latest_run_id(log_path, args.env) + 1
    return os.path.join(log_path, f"{args.env}_{run_id}{uuid_str}")


def read_env_kwargs(args):
    env_kwargs = {}
    if "commonroad" in args.env or args.env == "cr-monitor-v0":
        with open(PATH_PARAMS["configs"][args.env], "r") as config_file:
            configs = yaml.safe_load(config_file)
            env_kwargs.update(configs["env_configs"])

    if os.path.isfile(args.configs_path):
        with open(args.configs_path, "r") as configs_file:
            env_kwargs.update(yaml.safe_load(configs_file))

    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    return env_kwargs


def _convert_schedule(value):
    if isinstance(value, str) and value.startswith("lin_"):
        return linear_schedule(float(value.split("_", 1)[1]))
    return value


def _count_pickle_scenarios(path: str) -> int:
    if not path or not os.path.isdir(path):
        return 0
    return len(glob.glob(os.path.join(path, "*.pickle")))


def read_ppo_hyperparams(args):
    ppo_hyperparams_file = os.path.join(ROOT_STR, "commonroad_rl/hyperparams/ppo.yml")
    legacy_ppo2_file = os.path.join(ROOT_STR, "commonroad_rl/hyperparams/ppo2.yml")

    source_file = ppo_hyperparams_file if os.path.isfile(ppo_hyperparams_file) else legacy_ppo2_file
    with open(source_file, "r") as f:
        hyperparams_dict = yaml.safe_load(f)

    if args.env not in hyperparams_dict:
        raise ValueError(f"Hyperparameters not found for ppo-{args.env} in {source_file}")

    hp = copy.deepcopy(hyperparams_dict[args.env])

    if os.path.isfile(args.hyperparams_path):
        with open(args.hyperparams_path, "r") as hyperparams_file:
            hp.update(yaml.safe_load(hyperparams_file))
    if args.hyperparams is not None:
        hp.update(args.hyperparams)

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
        total_rollout = max(n_steps * args.n_envs, 1)
        hp["batch_size"] = max(total_rollout // int(nminibatches), 1)

    for key in ["n_timesteps", "n_envs", "env_wrapper", "frame_stack"]:
        hp.pop(key, None)

    return hp, normalize, normalize_kwargs


def make_single_env(env_id, rank, seed, log_dir, logging_path, env_kwargs, info_keywords, subproc=False):
    def _init():
        local_kwargs = copy.deepcopy(env_kwargs)
        if subproc and ("commonroad" in env_id or env_id == "cr-monitor-v0"):
            train_reset_config_path = local_kwargs.pop("train_reset_config_path", PATH_PARAMS["train_reset_config"])
            test_reset_config_path = local_kwargs.pop("test_reset_config_path", PATH_PARAMS["test_reset_config"])
            env = gym.make(
                env_id,
                train_reset_config_path=os.path.join(train_reset_config_path, str(rank)),
                test_reset_config_path=os.path.join(test_reset_config_path, str(rank)),
                logging_path=logging_path,
                **local_kwargs,
            )
        else:
            env = gym.make(env_id, logging_path=logging_path, **local_kwargs)

        try:
            env.reset(seed=seed + rank)
        except TypeError:
            env.seed(seed + rank)

        log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
        return Monitor(env, log_file, info_keywords=tuple(info_keywords))

    return _init


def create_vec_env(args, env_kwargs, save_path, normalize=False, normalize_kwargs=None, eval_env=False):
    if normalize_kwargs is None:
        normalize_kwargs = {}

    kwargs = copy.deepcopy(env_kwargs)
    kwargs["logging_mode"] = logging.INFO
    if ("commonroad" in args.env or args.env == "cr-monitor-v0") and eval_env:
        kwargs["test_env"] = True

    log_dir = os.path.join(save_path, "test") if eval_env else save_path

    if args.n_envs == 1:
        vec_env = DummyVecEnv(
            [
                make_single_env(
                    args.env,
                    rank=0,
                    seed=args.seed,
                    log_dir=log_dir,
                    logging_path=save_path,
                    env_kwargs=kwargs,
                    info_keywords=args.info_keywords,
                    subproc=False,
                )
            ]
        )
    else:
        vec_env = SubprocVecEnv(
            [
                make_single_env(
                    args.env,
                    rank=i,
                    seed=args.seed + args.n_envs if eval_env else args.seed,
                    log_dir=log_dir,
                    logging_path=save_path,
                    env_kwargs=kwargs,
                    info_keywords=args.info_keywords,
                    subproc=True,
                )
                for i in range(args.n_envs)
            ],
            start_method="spawn",
        )

    if normalize:
        vn_kwargs = copy.deepcopy(normalize_kwargs)
        if eval_env:
            vn_kwargs["norm_reward"] = False
            vec_env = VecNormalize(vec_env, training=False, **vn_kwargs)
        else:
            vec_env = VecNormalize(vec_env, **vn_kwargs)

    return vec_env


def run_training():
    args = parse_args()
    args.info_keywords = tuple(args.info_keywords)

    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    try:
        registered_envs = set(gym.envs.registry.keys())
    except AttributeError:
        registered_envs = set(gym.envs.registry.env_specs.keys())
    if args.env not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(args.env, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "no close match found"
        raise ValueError(f"{args.env} not found in gym registry, maybe you meant {closest_match}?")

    if args.seed < 0:
        args.seed = np.random.randint(2**32 - 1)
    set_random_seed(args.seed)

    if args.trained_agent:
        valid_extension = args.trained_agent.endswith(".zip")
        assert valid_extension and os.path.isfile(args.trained_agent), "--trained-agent must be a valid .zip file"

    save_path = args.save_path if args.save_path is not None else construct_save_path(args)
    os.makedirs(save_path, exist_ok=True)
    construct_logger(save_path)

    t1 = time.time()
    LOGGER.info(f"Environment id: {args.env}")
    LOGGER.info(f"Seed: {args.seed}")
    LOGGER.info(f"Using {args.n_envs} environments")
    LOGGER.info(f"Learning with {args.n_timesteps} timesteps")

    env_kwargs = read_env_kwargs(args)
    with open(os.path.join(save_path, "environment_configurations.yml"), "w") as output_file:
        yaml.dump(env_kwargs, output_file)

    hyperparams, normalize, normalize_kwargs = read_ppo_hyperparams(args)
    with open(os.path.join(save_path, "model_hyperparameters.yml"), "w") as output_file:
        yaml.dump(hyperparams, output_file)
    LOGGER.info("Model hyperparameters loaded")
    LOGGER.debug(pformat(hyperparams))

    env = create_vec_env(
        args, env_kwargs, save_path, normalize=normalize, normalize_kwargs=normalize_kwargs, eval_env=False
    )

    callbacks = []
    if args.save_freq > 0:
        save_freq = max(args.save_freq // args.n_envs, 1)
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=save_path,
                name_prefix="rl_model",
                save_vecnormalize=True,
            )
        )

    if args.eval_freq > 0:
        eval_reset_config_path = env_kwargs.get("test_reset_config_path", PATH_PARAMS["test_reset_config"])
        n_eval_scenarios = _count_pickle_scenarios(eval_reset_config_path)

        if n_eval_scenarios <= 0:
            LOGGER.warning(
                "Evaluation disabled: no scenarios found in %s. "
                "Populate the test dataset or set --eval-freq -1 to silence this warning.",
                eval_reset_config_path,
            )
        else:
            eval_freq = max(args.eval_freq // args.n_envs, 1)
            eval_env = create_vec_env(
                args,
                env_kwargs,
                save_path,
                normalize=normalize,
                normalize_kwargs=normalize_kwargs,
                eval_env=True,
            )
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=save_path,
                    log_path=save_path,
                    eval_freq=eval_freq,
                    n_eval_episodes=args.eval_episodes,
                    deterministic=True,
                )
            )

    tensorboard_log = None if args.tensorboard_log == "" else os.path.join(args.tensorboard_log, args.env)

    policy = hyperparams.pop("policy", "MlpPolicy")
    if args.trained_agent:
        model = PPO.load(args.trained_agent, env=env, tensorboard_log=tensorboard_log)
        reset_num_timesteps = False
    else:
        model = PPO(policy=policy, env=env, tensorboard_log=tensorboard_log, verbose=1, seed=args.seed, **hyperparams)
        reset_num_timesteps = True

    learn_kwargs = {"reset_num_timesteps": reset_num_timesteps}
    if args.log_interval > -1:
        learn_kwargs["log_interval"] = args.log_interval
    if callbacks:
        learn_kwargs["callback"] = callbacks

    try:
        model.learn(total_timesteps=args.n_timesteps, **learn_kwargs)
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted by user")

    model.save(os.path.join(save_path, args.env))
    if normalize and model.get_vec_normalize_env() is not None:
        model.get_vec_normalize_env().save(os.path.join(save_path, "vecnormalize.pkl"))

    LOGGER.info(f"Training finished in {time.time() - t1:.2f}s")
    LOGGER.info(f"Saved outputs to {save_path}")


if __name__ == "__main__":
    run_training()
