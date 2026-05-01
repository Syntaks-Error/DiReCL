from __future__ import annotations

import asyncio
import ast
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openai

from .optimizer import CandidateResult, InnerLoopConfig, MLIRLNumericOptimizer


def _extract_code_string(response_cur: str) -> str:
    patterns = [r"```python(.*?)```", r"```(.*?)```", r'"""(.*?)"""', r'""(.*?)""', r'"(.*?)"']
    code_string = None
    for pattern in patterns:
        found = re.search(pattern, response_cur, re.DOTALL)
        if found is not None:
            code_string = found.group(1).strip()
            break

    code_string = response_cur if not code_string else code_string

    lines = code_string.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("@")
            or stripped.startswith("def ")
        ):
            return "\n".join(lines[i:])
    return code_string


def _validate_reward_code(code: str) -> Tuple[bool, str]:
    try:
        module = ast.parse(code)
    except Exception as exc:
        return False, f"Syntax error: {exc}"

    fn_nodes = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if not fn_nodes:
        return False, "No function definition found."

    fn_node = fn_nodes[0]
    if not fn_node.body:
        return False, f"Function '{fn_node.name}' has an empty body."

    return True, ""


def _file_to_string(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@dataclass
class OuterLoopConfig:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    iteration: int = 2
    sample: int = 2
    output_dir: str = "outputs/eureka_numeric_irl"
    env_name: str = "ant"
    env_description: str = "Train an Ant policy to move forward stably with smooth control."
    inner: InnerLoopConfig = field(default_factory=InnerLoopConfig)


class EurekaNumericIRL:
    def __init__(self, workspace_root: Path, config: OuterLoopConfig):
        self.workspace_root = workspace_root
        self.config = config
        self.optimizer = MLIRLNumericOptimizer(workspace_root, config.inner)

        self.eureka_root = self.workspace_root.parent / "Eureka" / "eureka"
        self.prompt_dir = self.eureka_root / "utils" / "prompts"
        self.env_obs_file = self.eureka_root / "envs" / "mujoco" / f"{self.config.env_name}_obs.py"
        self.env_fallback_file = self.eureka_root / "envs" / "mujoco" / f"{self.config.env_name}.py"

        if self.config.env_name == "commonroad":
            self.env_fallback_file = self.eureka_root / "envs" / "commonroad_highd.py"

    @staticmethod
    def _append_log(log_file: Path, msg: str):
        lines = msg.splitlines() if msg else [""]
        with log_file.open("a", encoding="utf-8") as f:
            for entry in lines:
                line = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} | {entry}"
                print(line, flush=True)
                f.write(line + "\n")

    def _build_initial_messages(self) -> List[Dict[str, str]]:
        initial_system = _file_to_string(self.prompt_dir / "initial_system.txt")
        code_output_tip = _file_to_string(self.prompt_dir / "code_output_tip.txt")
        initial_user = _file_to_string(self.prompt_dir / "initial_user.txt")
        reward_signature = _file_to_string(self.prompt_dir / "reward_signature_mujoco.txt")

        initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
        initial_system += (
            "\nHard constraints:\n"
            "- Output should contain one complete Python function body, not pseudocode.\n"
            "- Do not reference environment keys unless they are guaranteed by the environment.\n"
            "- Keep the code syntactically valid and self-contained.\n"
            "- If the environment is CommonRoad, use only stable observation/action inputs and generic geometric terms.\n"
        )

        if self.env_obs_file.exists():
            task_obs = _file_to_string(self.env_obs_file)
        elif self.env_fallback_file.exists():
            task_obs = _file_to_string(self.env_fallback_file)
        else:
            task_obs = (
                "Observation context file is not available in Eureka templates for this environment. "
                "Use only variables from function inputs (obs/action/next_obs/info) and keep code differentiable."
            )
        initial_user = initial_user.format(task_obs_code_string=task_obs, task_description=self.config.env_description)

        return [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    async def _query_llm_samples(self, messages: List[Dict[str, str]]) -> Tuple[List[str], Dict[str, int]]:
        responses: List[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=self.config.base_url)
        chunk_size = self.config.sample
        total_samples = 0

        while total_samples < self.config.sample:
            response_cur = None
            for attempt in range(1000):
                try:
                    response_cur = await client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=self.config.temperature,
                        n=chunk_size,
                    )
                    total_samples += chunk_size
                    break
                except Exception:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                    await asyncio.sleep(1)

            if response_cur is None:
                raise RuntimeError("LLM querying failed after retries.")

            choices = getattr(response_cur, "choices", None) or []
            for choice in choices:
                message = getattr(choice, "message", None)
                content = getattr(message, "content", "") if message is not None else ""
                responses.append(content or "")

            usage_cur = getattr(response_cur, "usage", None)
            if usage_cur is not None:
                usage["prompt_tokens"] += int(getattr(usage_cur, "prompt_tokens", 0) or 0)
                usage["completion_tokens"] += int(getattr(usage_cur, "completion_tokens", 0) or 0)
                usage["total_tokens"] += int(getattr(usage_cur, "total_tokens", 0) or 0)

        return responses[: self.config.sample], usage

    @staticmethod
    def _feedback_from_result(result: CandidateResult) -> str:
        policy_feedback = (
            "We trained a RL policy using the provided reward function code and tracked metrics over optimization:\n"
        )

        if result.error is not None:
            return (
                f"Executing the reward function code above has the following error: {result.error}. "
                "Please fix the bug and provide a new, improved reward function!\n"
            )

        def _summ(v: List[float]) -> str:
            if len(v) == 0:
                return "[]"
            arr = np.array(v, dtype=np.float32)
            return (
                f"samples={list(np.round(arr[:: max(len(arr)//10, 1)], 4))}, "
                f"Max={arr.max():.4f}, Mean={arr.mean():.4f}, Min={arr.min():.4f}"
            )

        content = policy_feedback
        content += f"final_gap: {_summ(result.gap_trace)}\n"
        content += f"ml_irl_loss: {_summ(result.loss_trace)}\n"
        content += f"parsed_learnable_params: {result.parsed_param_count}\n"
        content += (
            "Please carefully analyze the policy feedback and provide a new, improved reward function that can better "
            "solve the task. If metrics are flat or poor, rewrite the reward function instead of small edits.\n"
        )
        content += (
            "The output of the reward function should be python code only. Keep the signature consistent with the task "
            "and make the reward fully differentiable.\n"
            "Avoid env-specific dictionary keys that may be missing; prefer direct tensor computations from the inputs.\n"
        )
        return content

    def _write_candidate_artifacts(
        self,
        out_dir: Path,
        iter_id: int,
        response_id: int,
        response_text: str,
        code: str,
        result: CandidateResult,
    ):
        (out_dir / f"env_iter{iter_id}_response{response_id}.txt").write_text(response_text, encoding="utf-8")
        (out_dir / f"env_iter{iter_id}_response{response_id}_rewardonly.py").write_text(code + "\n", encoding="utf-8")
        if result.trained_reward_code is not None:
            (out_dir / f"env_iter{iter_id}_response{response_id}_trained_reward.py").write_text(
                result.trained_reward_code, encoding="utf-8"
            )
        (out_dir / f"env_iter{iter_id}_response{response_id}.metrics.json").write_text(
            json.dumps(
                {
                    "name": result.name,
                    "final_loss": result.final_loss,
                    "final_gap": result.final_gap,
                    "initial_param_values": result.initial_param_values,
                    "param_values": result.param_values,
                    "parsed_param_count": result.parsed_param_count,
                    "parse_source_fn": result.parse_source_fn,
                    "parse_mode": result.parse_mode,
                    "error": result.error,
                    "loss_trace": result.loss_trace,
                    "gap_trace": result.gap_trace,
                    "train_log": result.train_log,
                    "sac_info": result.sac_info,
                    "best_policy_path": result.best_policy_path,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def run(self) -> Dict:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.workspace_root / self.config.output_dir / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        run_log = out_dir / "run.log"
        self._append_log(run_log, "run start")

        messages = self._build_initial_messages()
        self._append_log(run_log, "initial Eureka prompts loaded")
        history = []
        best_overall = None

        for iter_id in range(self.config.iteration):
            self._append_log(run_log, f"iteration {iter_id} begins: requesting {self.config.sample} reward codes")

            llm_input_file = out_dir / f"iter{iter_id}_llm_input_messages.txt"
            llm_input_text = [
                f"model: {self.config.model}",
                f"base_url: {self.config.base_url}",
                f"temperature: {self.config.temperature}",
                f"sample: {self.config.sample}",
                "",
                "messages:",
                json.dumps(messages, indent=2, ensure_ascii=False),
            ]
            llm_input_file.write_text("\n".join(llm_input_text), encoding="utf-8")
            self._append_log(run_log, f"iteration {iter_id}: saved LLM input messages to {llm_input_file.name}")

            responses, token_usage = asyncio.run(self._query_llm_samples(messages))
            self._append_log(
                run_log,
                (
                    f"iteration {iter_id}: reward code generation done, "
                    f"prompt_tokens={token_usage['prompt_tokens']} completion_tokens={token_usage['completion_tokens']} "
                    f"total_tokens={token_usage['total_tokens']}"
                ),
            )

            candidates = {}
            candidate_meta = []
            for response_id, response_text in enumerate(responses):
                code = _extract_code_string(response_text)
                name = f"iter{iter_id}_response{response_id}"
                candidates[name] = code
                candidate_meta.append((response_id, response_text, code, name))
                self._append_log(
                    run_log,
                    f"iteration {iter_id} candidate {response_id} parsed from LLM text (chars={len(code)})",
                )

            validated_candidates = {}
            validated_meta = []
            for response_id, response_text, code, name in candidate_meta:
                ok, reason = _validate_reward_code(code)
                if not ok:
                    self._append_log(
                        run_log,
                        f"iteration {iter_id} candidate {response_id} rejected before ML-IRL: {reason}",
                    )
                    continue
                validated_candidates[name] = code
                validated_meta.append((response_id, response_text, code, name))

            if not validated_candidates:
                self._append_log(
                    run_log,
                    f"iteration {iter_id}: no valid reward code candidates after validation; reflecting on syntax errors only",
                )
                candidates = {}
                candidate_meta = []
                history.append(
                    {
                        "iteration": iter_id,
                        "token_usage": token_usage,
                        "best_name": None,
                        "best_gap": float("-inf"),
                        "best_loss": float("inf"),
                        "best_params": [],
                        "results": [],
                    }
                )
                best_feedback = (
                    "The previous generation produced invalid Python code. "
                    "Return only syntactically valid Python code.\n"
                )
                if len(messages) == 2:
                    messages += [{"role": "assistant", "content": responses[0] if responses else ""}]
                    messages += [{"role": "user", "content": best_feedback}]
                else:
                    messages[-2] = {"role": "assistant", "content": responses[0] if responses else ""}
                    messages[-1] = {"role": "user", "content": best_feedback}
                (out_dir / "messages.json").write_text(json.dumps(messages, indent=2), encoding="utf-8")
                (out_dir / "summary.json").write_text(
                    json.dumps({"config": asdict(self.config), "history": history}, indent=2), encoding="utf-8"
                )
                continue

            candidates = validated_candidates
            candidate_meta = validated_meta

            self._append_log(
                run_log, f"iteration {iter_id}: ML-IRL optimization begins for {len(candidates)} candidates"
            )
            policy_save_dir = out_dir / "policies"
            results = self.optimizer.optimize_batch(
                candidates,
                log_fn=lambda m: self._append_log(run_log, m),
                policy_save_dir=policy_save_dir,
            )
            self._append_log(run_log, f"iteration {iter_id}: ML-IRL optimization done")
            result_by_name = {r.name: r for r in results}

            for response_id, response_text, code, name in candidate_meta:
                self._write_candidate_artifacts(
                    out_dir, iter_id, response_id, response_text, code, result_by_name[name]
                )

            ranked = sorted(results, key=lambda x: x.final_gap, reverse=True)
            best = ranked[0]
            self._append_log(
                run_log,
                (
                    f"iteration {iter_id}: best candidate={best.name} best_gap={best.final_gap} "
                    f"best_loss={best.final_loss} parsed_params={best.parsed_param_count} "
                    f"initial_params={best.initial_param_values} trained_params={best.param_values}"
                ),
            )

            if best_overall is None or best.final_gap > best_overall.final_gap:
                best_overall = best
                self._append_log(run_log, f"iteration {iter_id}: best overall updated -> {best_overall.name}")

            best_response_idx = int(best.name.split("response")[-1]) if "response" in best.name else 0
            best_response_text = responses[best_response_idx]
            best_feedback = self._feedback_from_result(best)

            if len(messages) == 2:
                messages += [{"role": "assistant", "content": best_response_text}]
                messages += [{"role": "user", "content": best_feedback}]
            else:
                messages[-2] = {"role": "assistant", "content": best_response_text}
                messages[-1] = {"role": "user", "content": best_feedback}

            history.append(
                {
                    "iteration": iter_id,
                    "token_usage": token_usage,
                    "best_name": best.name,
                    "best_gap": best.final_gap,
                    "best_loss": best.final_loss,
                    "best_params": best.param_values,
                    "results": [
                        {
                            "name": r.name,
                            "final_gap": r.final_gap,
                            "final_loss": r.final_loss,
                            "initial_param_values": r.initial_param_values,
                            "param_values": r.param_values,
                            "parsed_param_count": r.parsed_param_count,
                            "parse_source_fn": r.parse_source_fn,
                            "parse_mode": r.parse_mode,
                            "error": r.error,
                            "train_log": r.train_log,
                            "sac_info": r.sac_info,
                            "best_policy_path": r.best_policy_path,
                        }
                        for r in ranked
                    ],
                }
            )

            (out_dir / "messages.json").write_text(json.dumps(messages, indent=2), encoding="utf-8")
            (out_dir / "summary.json").write_text(
                json.dumps({"config": asdict(self.config), "history": history}, indent=2), encoding="utf-8"
            )

        if best_overall is None:
            raise RuntimeError("No valid generated reward candidate was produced.")

        (out_dir / "best_reward.py").write_text(best_overall.code + "\n", encoding="utf-8")
        if best_overall.trained_reward_code is not None:
            (out_dir / "best_reward_trained.py").write_text(best_overall.trained_reward_code, encoding="utf-8")
        self._append_log(run_log, f"run completed. best candidate={best_overall.name}")

        return {
            "output_dir": str(out_dir),
            "best_name": best_overall.name,
            "best_gap": best_overall.final_gap,
            "best_loss": best_overall.final_loss,
            "best_params": best_overall.param_values,
        }
