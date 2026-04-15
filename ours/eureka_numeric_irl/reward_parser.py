from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


_NON_LEARNABLE_KEYWORDS = {"dim", "axis", "keepdim", "dtype", "device"}


@dataclass
class ParseReport:
    fn_name: str
    source_fn_name: str
    mode: str
    constants: List[float]


class _AnnotationStripper(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        node = self.generic_visit(node)
        for arg in node.args.args:
            arg.annotation = None
        for arg in node.args.kwonlyargs:
            arg.annotation = None
        if node.args.vararg is not None:
            node.args.vararg.annotation = None
        if node.args.kwarg is not None:
            node.args.kwarg.annotation = None
        node.returns = None
        return node


class _GradientSafeRewriter(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)

        if isinstance(node.func, ast.Attribute) and node.func.attr == "item" and len(node.args) == 0:
            return node.func.value

        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "tensor"
        ):
            node.func.attr = "as_tensor"

        return node


class _NumericParameterizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.parents: List[ast.AST] = []
        self.constants: List[float] = []

    def visit(self, node):
        self.parents.append(node)
        out = super().visit(node)
        self.parents.pop()
        return out

    def _parent(self) -> ast.AST | None:
        if len(self.parents) < 2:
            return None
        return self.parents[-2]

    def _should_parameterize(self, node: ast.Constant) -> bool:
        parent = self._parent()
        if parent is None:
            return False

        if isinstance(parent, (ast.Slice, ast.Subscript)):
            return False
        if isinstance(parent, ast.keyword) and parent.arg in _NON_LEARNABLE_KEYWORDS:
            return False
        if isinstance(parent, ast.Call):
            fn = parent.func
            if isinstance(fn, ast.Name) and fn.id in {"range", "len", "int", "float"}:
                return False

        # Learn only float literals. Integer literals are frequently used as
        # tensor indices/shapes and must stay integers to avoid runtime errors.
        return isinstance(node.value, float)

    def visit_Constant(self, node: ast.Constant):
        if not self._should_parameterize(node):
            return node

        idx = len(self.constants)
        self.constants.append(float(node.value))
        return ast.copy_location(
            ast.Subscript(value=ast.Name(id="params", ctx=ast.Load()), slice=ast.Constant(value=idx), ctx=ast.Load()),
            node,
        )


class ParameterizedReward(nn.Module):
    def __init__(self, code: str, fn_name: str = "reward_fn", device: str = "cpu"):
        super().__init__()
        self.code = code
        self.fn_name = fn_name
        self.device = torch.device(device)
        (
            self._compiled_fn,
            constants,
            self._source_fn_name,
            self._mode,
            self._transformed_source,
            self._target_fn_name,
        ) = self._compile(code, fn_name)
        self.params = nn.Parameter(torch.tensor(constants, dtype=torch.float32, device=self.device))

    @staticmethod
    def _compile(code: str, fn_name: str):
        module = ast.parse(code)

        module = _AnnotationStripper().visit(module)
        module = _GradientSafeRewriter().visit(module)
        ast.fix_missing_locations(module)

        all_fn_nodes = [n for n in module.body if isinstance(n, ast.FunctionDef)]
        if not all_fn_nodes:
            raise ValueError("No function definition found in reward code.")

        candidate_names = [fn_name, "reward_fn", "compute_reward"]
        fn_nodes = [n for n in all_fn_nodes if n.name in candidate_names]
        if not fn_nodes:
            fn_nodes = [all_fn_nodes[0]]

        fn_node = fn_nodes[0]
        source_fn_name = fn_node.name
        fn_node.args.args.append(ast.arg(arg="params"))

        transformer = _NumericParameterizer()
        transformed_module = transformer.visit(module)
        ast.fix_missing_locations(transformed_module)

        mode = "direct"
        if source_fn_name == "compute_reward":
            mode = "compute_reward_wrapper"
            wrapper_src = (
                "def reward_fn(obs, act, params):\n"
                "    vals = []\n"
                "    n = int(obs.shape[0])\n"
                "    for i in range(n):\n"
                "        out = compute_reward(obs[i], act[i], obs[i], {}, params)\n"
                "        r = out[0] if isinstance(out, tuple) else out\n"
                "        if torch.is_tensor(r):\n"
                "            vals.append(r.reshape(-1)[0])\n"
                "        else:\n"
                "            vals.append(torch.as_tensor(r, device=obs.device, dtype=obs.dtype))\n"
                "    return torch.stack(vals)\n"
            )
            wrapper_module = ast.parse(wrapper_src)
            transformed_module.body.extend(wrapper_module.body)

        namespace = {"torch": torch}
        exec(compile(transformed_module, filename="<reward_code>", mode="exec"), namespace, namespace)

        target_name = fn_name if mode == "direct" else "reward_fn"
        compiled_fn = namespace.get(target_name)
        if compiled_fn is None:
            compiled_fn = namespace.get(target_name)
        if compiled_fn is None:
            raise RuntimeError(f"Failed to compile function '{target_name}'.")

        transformed_source = ast.unparse(transformed_module)
        return compiled_fn, transformer.constants, source_fn_name, mode, transformed_source, target_name

    def forward(self, obs: torch.Tensor, act: torch.Tensor | None = None) -> torch.Tensor:
        if act is None:
            act = torch.zeros((obs.shape[0], 0), device=obs.device, dtype=obs.dtype)
        out = self._compiled_fn(obs, act, self.params)
        if out.ndim == 0:
            out = out.unsqueeze(0)
        return out.reshape(-1)

    def scalar_reward_from_state_action(self, state_action, obs_dim: int) -> float:
        x = torch.as_tensor(state_action, dtype=torch.float32, device=self.device).reshape(1, -1)
        obs = x[:, :obs_dim]
        act = x[:, obs_dim:]
        with torch.no_grad():
            val = self.forward(obs, act).reshape(-1)[0]
        return float(val.detach().cpu().item())

    def report(self) -> ParseReport:
        return ParseReport(
            fn_name=self.fn_name,
            source_fn_name=self._source_fn_name,
            mode=self._mode,
            constants=self.params.detach().cpu().tolist(),
        )

    def export_trained_code(self) -> str:
        trained_params = self.params.detach().cpu().tolist()
        return (
            "import torch\n\n"
            "# Original LLM-generated code\n"
            f"{self.code.strip()}\n\n"
            "# Parameterized transformed reward code used during ML-IRL optimization\n"
            f"{self._transformed_source.strip()}\n\n"
            f"TRAINED_PARAMS = {trained_params}\n\n"
            "def reward_fn_trained(obs, act):\n"
            "    params = torch.as_tensor(TRAINED_PARAMS, device=obs.device, dtype=obs.dtype)\n"
            f"    return {self._target_fn_name}(obs, act, params)\n"
        )
