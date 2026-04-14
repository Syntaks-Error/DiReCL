import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openai

from utils.extract_task_code import file_to_string, get_function_signature
from utils.misc import block_until_training, filter_traceback

EUREKA_ROOT_DIR = os.getcwd()
MUJOCO_TRAIN_SCRIPT = f"{EUREKA_ROOT_DIR}/utils/mujoco_train.py"


def _extract_code_string(response_cur: str) -> str:
    patterns = [
        r"```python(.*?)```",
        r"```(.*?)```",
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    code_string = None
    for pattern in patterns:
        code_string = re.search(pattern, response_cur, re.DOTALL)
        if code_string is not None:
            code_string = code_string.group(1).strip()
            break

    code_string = response_cur if not code_string else code_string

    lines = code_string.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            code_string = "\n".join(lines[i:])
            break
    return code_string


def _query_llm_samples(cfg, model: str, messages):
    responses = []
    response_cur = None
    total_samples = 0
    total_token = 0
    total_completion_token = 0
    chunk_size = cfg.sample

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.deepseek.com")

    while True:
        if total_samples >= cfg.sample:
            break
        for attempt in range(1000):
            try:
                response_cur = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=cfg.temperature,
                    n=chunk_size,
                )
                total_samples += chunk_size
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    print("Current Chunk Size", chunk_size)
                logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()

        choices = getattr(response_cur, "choices", None)
        if choices is None:
            choices = response_cur.get("choices", [])

        for choice in choices:
            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message", {})

            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content", "")

            responses.append(content or "")

        usage = getattr(response_cur, "usage", None)
        if usage is None and isinstance(response_cur, dict):
            usage = response_cur.get("usage", {})

        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        if prompt_tokens is None and isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        total_tokens = int(total_tokens or 0)

        total_completion_token += completion_tokens
        total_token += total_tokens

    return responses, prompt_tokens, total_completion_token, total_token


def _build_mujoco_feedback(policy_feedback: str, code_feedback: str, tensorboard_logs: Dict[str, list]) -> str:
    max_iterations = max(len(v) for v in tensorboard_logs.values()) if len(tensorboard_logs) > 0 else 1
    epoch_freq = max(int(max_iterations // 10), 1)

    content = policy_feedback.format(epoch_freq=epoch_freq)
    for metric in tensorboard_logs:
        if len(tensorboard_logs[metric]) == 0:
            continue
        metric_cur = ["{:.2f}".format(x) for x in tensorboard_logs[metric][::epoch_freq]]
        metric_cur_max = max(tensorboard_logs[metric])
        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
        metric_cur_min = min(tensorboard_logs[metric])
        content += (
            f"{metric}: {metric_cur}, Max: {metric_cur_max:.2f}, "
            f"Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
        )

    content += code_feedback
    return content


def _load_env_metadata(env_name: str) -> Dict[str, str]:
    env_meta_file = Path(EUREKA_ROOT_DIR) / "envs" / "mujoco" / f"{env_name}.py"
    if not env_meta_file.exists():
        raise FileNotFoundError(f"Unsupported MuJoCo environment '{env_name}'. Missing: {env_meta_file}")

    spec = importlib.util.spec_from_file_location(f"eureka_mujoco_{env_name}", str(env_meta_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return {
        "task": getattr(module, "TASK_NAME", f"{env_name.capitalize()}Mujoco"),
        "env_id": getattr(module, "ENV_ID", None),
        "description": getattr(module, "TASK_DESCRIPTION", ""),
    }


def _resolve_mujoco_env(cfg) -> Tuple[str, str, str, str]:
    env_name = str(cfg.env.get("env_name", "")).lower()
    if not env_name:
        task_guess = str(cfg.env.get("task", "")).lower()
        env_name = task_guess.replace("mujoco", "").replace("_", "").strip()

    if not env_name:
        raise ValueError("Cannot infer MuJoCo environment name. Please set cfg.env.env_name.")

    metadata = _load_env_metadata(env_name)

    task = str(cfg.env.get("task", metadata["task"]))
    env_id = str(cfg.env.get("env_id", metadata["env_id"]))
    task_description = str(cfg.env.get("description", metadata["description"]))

    if env_id in ("None", "", None):
        raise ValueError(
            f"MuJoCo env '{env_name}' is missing env_id. Set cfg/env/<env>.yaml env_id or define ENV_ID in env metadata."
        )

    return env_name, task, env_id, task_description


def run_mujoco(cfg):
    env_name, task, env_id, task_description = _resolve_mujoco_env(cfg)
    suffix = cfg.suffix
    model = cfg.model

    task_obs_file = f"{EUREKA_ROOT_DIR}/envs/mujoco/{env_name}_obs.py"
    if not os.path.exists(task_obs_file):
        raise FileNotFoundError(f"Missing MuJoCo observation context file: {task_obs_file}")

    shutil.copy(task_obs_file, "env_init_obs.py")
    task_obs_code_string = file_to_string(task_obs_file)

    prompt_dir = f"{EUREKA_ROOT_DIR}/utils/prompts"
    initial_system = file_to_string(f"{prompt_dir}/initial_system.txt")
    code_output_tip = file_to_string(f"{prompt_dir}/code_output_tip.txt")
    code_feedback = file_to_string(f"{prompt_dir}/code_feedback.txt")
    initial_user = file_to_string(f"{prompt_dir}/initial_user.txt")
    reward_signature = file_to_string(f"{prompt_dir}/reward_signature_mujoco.txt")
    policy_feedback = file_to_string(f"{prompt_dir}/policy_feedback.txt")
    execution_error_feedback = file_to_string(f"{prompt_dir}/execution_error_feedback.txt")

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    DUMMY_FAILURE = -10000.0
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None

    for iter in range(cfg.iteration):
        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")
        responses, prompt_tokens, total_completion_token, total_token = _query_llm_samples(cfg, model, messages)

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0] + "\n")

        logging.info(
            f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, "
            f"Completion Tokens: {total_completion_token}, Total Tokens: {total_token}"
        )

        code_runs = []
        rl_runs = []
        metrics_files = []
        run_response_ids = []

        num_responses = min(cfg.sample, len(responses))
        for response_id in range(num_responses):
            response_cur = responses[response_id]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            code_string = _extract_code_string(response_cur)

            try:
                _, _ = get_function_signature(code_string)
            except Exception:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                continue

            code_runs.append(code_string)
            reward_file = f"env_iter{iter}_response{response_id}_rewardonly.py"
            with open(reward_file, "w") as file:
                file.writelines("from typing import Any, Dict, Tuple\n")
                file.writelines("import numpy as np\n")
                file.writelines(code_string + "\n")

            generated_code_path = f"env_iter{iter}_response{response_id}.py"
            shutil.copy(reward_file, generated_code_path)

            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            metrics_file = f"env_iter{iter}_response{response_id}.metrics.json"
            metrics_files.append(metrics_file)
            run_response_ids.append(response_id)

            with open(rl_filepath, "w") as f:
                process = subprocess.Popen(
                    [
                        "python",
                        "-u",
                        MUJOCO_TRAIN_SCRIPT,
                        "--env-id",
                        env_id,
                        "--reward-file",
                        reward_file,
                        "--metrics-file",
                        metrics_file,
                        "--total-timesteps",
                        str(cfg.max_iterations),
                        "--seed",
                        str(response_id),
                    ],
                    stdout=f,
                    stderr=f,
                )

            block_until_training(
                rl_filepath,
                log_status=True,
                iter_num=iter,
                response_id=response_id,
                success_markers=["TRAINING_START"],
            )
            rl_runs.append(process)

        contents = []
        successes = []
        reward_correlations = []
        code_paths = []

        exec_success = False
        for idx, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            response_id = run_response_ids[idx]
            rl_run.communicate()
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            metrics_file = metrics_files[idx]
            code_paths.append(f"env_iter{iter}_response{response_id}.py")

            try:
                with open(rl_filepath, "r") as f:
                    stdout_str = f.read()
            except Exception:
                content = execution_error_feedback.format(
                    traceback_msg=(
                        "Code Run cannot be executed due to function signature error! "
                        "Please re-write an entirely new reward function!"
                    )
                )
                content += code_output_tip
                contents.append(content)
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            traceback_msg = filter_traceback(stdout_str)
            if traceback_msg != "":
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content = execution_error_feedback.format(traceback_msg=traceback_msg) + code_output_tip
                contents.append(content)
                continue

            if not os.path.exists(metrics_file):
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content = (
                    execution_error_feedback.format(traceback_msg="Training finished without metrics output.")
                    + code_output_tip
                )
                contents.append(content)
                continue

            exec_success = True
            with open(metrics_file, "r") as f:
                metric_logs = json.load(f)

            content = _build_mujoco_feedback(policy_feedback, code_feedback, metric_logs)

            if "task_score" in metric_logs and len(metric_logs["task_score"]) > 0:
                successes.append(max(metric_logs["task_score"]))
            elif "gt_reward" in metric_logs and len(metric_logs["gt_reward"]) > 0:
                successes.append(max(metric_logs["gt_reward"]))
            else:
                successes.append(DUMMY_FAILURE)

            if "gt_reward" in metric_logs and "gpt_reward" in metric_logs:
                gt_reward = np.array(metric_logs["gt_reward"])
                gpt_reward = np.array(metric_logs["gpt_reward"])
                if len(gt_reward) > 1 and len(gpt_reward) > 1:
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)
                else:
                    reward_correlations.append(DUMMY_FAILURE)
            else:
                reward_correlations.append(DUMMY_FAILURE)

            content += code_output_tip
            contents.append(content)

        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.0)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue

        if len(successes) == 0:
            execute_rates.append(0.0)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("No valid code sample to evaluate in this iteration.")
            continue

        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]

        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.0) / cfg.sample

        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(
            f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, "
            f"Max Success Reward Correlation: {max_success_reward_correlation}"
        )
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" + responses[best_sample_idx] + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f"{task}")
        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig("summary.png")
        np.savez(
            "summary.npz",
            max_successes=max_successes,
            execute_rates=execute_rates,
            best_code_paths=best_code_paths,
            max_successes_reward_correlation=max_successes_reward_correlation,
        )

        if len(messages) == 2:
            messages += [{"role": "assistant", "content": responses[best_sample_idx]}]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]}
            messages[-1] = {"role": "user", "content": best_content}

        with open("messages.json", "w") as file:
            json.dump(messages, file, indent=4)

    if max_reward_code_path is None:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()

    logging.info(
        f"Task: {task}, Max Training Success {max_success_overall}, "
        f"Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}"
    )
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")

    reward_code_final_successes = []
    reward_code_correlations_final = []
    eval_runs = []
    eval_metrics_files = []

    for i in range(cfg.num_eval):
        rl_filepath = f"reward_code_eval{i}.txt"
        metrics_file = f"reward_code_eval{i}.metrics.json"
        eval_metrics_files.append(metrics_file)
        with open(rl_filepath, "w") as f:
            process = subprocess.Popen(
                [
                    "python",
                    "-u",
                    MUJOCO_TRAIN_SCRIPT,
                    "--env-id",
                    env_id,
                    "--reward-file",
                    max_reward_code_path,
                    "--metrics-file",
                    metrics_file,
                    "--total-timesteps",
                    str(cfg.max_iterations),
                    "--seed",
                    str(i),
                ],
                stdout=f,
                stderr=f,
            )
        block_until_training(rl_filepath, success_markers=["TRAINING_START"])
        eval_runs.append(process)

    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        metrics_file = eval_metrics_files[i]
        if not os.path.exists(metrics_file):
            continue

        with open(metrics_file, "r") as f:
            metric_logs = json.load(f)

        if "task_score" in metric_logs and len(metric_logs["task_score"]) > 0:
            reward_code_final_successes.append(max(metric_logs["task_score"]))
        elif "gt_reward" in metric_logs and len(metric_logs["gt_reward"]) > 0:
            reward_code_final_successes.append(max(metric_logs["gt_reward"]))

        if "gt_reward" in metric_logs and "gpt_reward" in metric_logs:
            gt_reward = np.array(metric_logs["gt_reward"])
            gpt_reward = np.array(metric_logs["gpt_reward"])
            if len(gt_reward) > 1 and len(gpt_reward) > 1:
                reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                reward_code_correlations_final.append(reward_correlation)

    logging.info(
        f"Final Success Mean: {np.mean(reward_code_final_successes)}, "
        f"Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}"
    )
    logging.info(
        f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, "
        f"Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}"
    )
    np.savez(
        "final_eval.npz",
        reward_code_final_successes=reward_code_final_successes,
        reward_code_correlations_final=reward_code_correlations_final,
    )
