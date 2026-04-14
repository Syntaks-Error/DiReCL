from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


_NON_LEARNABLE_KEYWORDS = {"dim", "axis", "keepdim", "dtype", "device"}


@dataclass
class ParseReport:
    fn_name: str
    constants: List[float]


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

        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)

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
        self._compiled_fn, constants = self._compile(code, fn_name)
        self.params = nn.Parameter(torch.tensor(constants, dtype=torch.float32, device=self.device))

    @staticmethod
    def _compile(code: str, fn_name: str):
        module = ast.parse(code)

        fn_nodes = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == fn_name]
        if not fn_nodes:
            raise ValueError(f"Function '{fn_name}' not found in reward code.")

        fn_node = fn_nodes[0]
        fn_node.args.args.append(ast.arg(arg="params"))

        transformer = _NumericParameterizer()
        transformed_module = transformer.visit(module)
        ast.fix_missing_locations(transformed_module)

        safe_globals = {"torch": torch}
        safe_locals = {}
        exec(compile(transformed_module, filename="<reward_code>", mode="exec"), safe_globals, safe_locals)

        compiled_fn = safe_locals.get(fn_name)
        if compiled_fn is None:
            compiled_fn = safe_globals.get(fn_name)
        if compiled_fn is None:
            raise RuntimeError(f"Failed to compile function '{fn_name}'.")

        return compiled_fn, transformer.constants

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
        return ParseReport(fn_name=self.fn_name, constants=self.params.detach().cpu().tolist())
