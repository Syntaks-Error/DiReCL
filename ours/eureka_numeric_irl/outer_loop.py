from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .optimizer import CandidateResult, InnerLoopConfig, MLIRLNumericOptimizer
from .templates import RewardTemplate, default_ant_templates, reflected_candidates


@dataclass
class OuterLoopConfig:
    rounds: int = 2
    keep_top_k: int = 2
    output_dir: str = "outputs/eureka_numeric_irl"
    inner: InnerLoopConfig = InnerLoopConfig()


class EurekaNumericIRL:
    def __init__(self, workspace_root: Path, config: OuterLoopConfig):
        self.workspace_root = workspace_root
        self.config = config
        self.optimizer = MLIRLNumericOptimizer(workspace_root, config.inner)

    @staticmethod
    def _to_dict(templates: List[RewardTemplate]) -> Dict[str, str]:
        return {t.name: t.code for t in templates}

    def _reflect(self, ranked: List[CandidateResult], round_id: int) -> List[RewardTemplate]:
        best = ranked[0]
        return reflected_candidates(best.code, round_id)

    def run(self) -> Dict:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.workspace_root / self.config.output_dir / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        pool = default_ant_templates()
        history = []

        for rid in range(self.config.rounds):
            batch = self._to_dict(pool)
            results = self.optimizer.optimize_batch(batch)
            ranked = sorted(results, key=lambda x: x.final_gap, reverse=True)
            topk = ranked[: self.config.keep_top_k]

            history.append(
                {
                    "round": rid,
                    "results": [asdict(r) for r in ranked],
                    "reflection": (
                        f"Keep high forward-velocity terms and smooth action penalty. "
                        f"Best gap={ranked[0].final_gap:.4f}."
                    ),
                }
            )

            if rid < self.config.rounds - 1:
                reflected = self._reflect(ranked, rid)
                pool = [RewardTemplate(name=r.name, code=r.code) for r in topk] + reflected

        best = sorted(history[-1]["results"], key=lambda x: x["final_gap"], reverse=True)[0]

        (out_dir / "best_reward.py").write_text(best["code"] + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps({"config": asdict(self.config), "history": history}, indent=2), encoding="utf-8"
        )

        return {
            "output_dir": str(out_dir),
            "best_name": best["name"],
            "best_gap": best["final_gap"],
            "best_loss": best["final_loss"],
            "best_params": best["param_values"],
        }
