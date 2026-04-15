from __future__ import annotations

import argparse
from pathlib import Path

from ruamel.yaml import YAML

from eureka_numeric_irl import EurekaNumericIRL
from eureka_numeric_irl.optimizer import InnerLoopConfig
from eureka_numeric_irl.outer_loop import OuterLoopConfig


def load_config(path: Path) -> OuterLoopConfig:
    yaml = YAML(typ="safe")
    raw = yaml.load(path.read_text(encoding="utf-8"))

    inner = InnerLoopConfig(**raw.get("inner", {}))
    cfg = OuterLoopConfig(
        model=raw.get("model", "deepseek-chat"),
        base_url=raw.get("base_url", "https://api.deepseek.com"),
        temperature=raw.get("temperature", 0.7),
        iteration=raw.get("iteration", 2),
        sample=raw.get("sample", 2),
        output_dir=raw.get("output_dir", "outputs/eureka_numeric_irl"),
        env_name=raw.get("env_name", "ant"),
        env_description=raw.get("env_description", "Train an Ant policy to move forward stably with smooth control."),
        inner=inner,
    )
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eureka_numeric_irl_ant.yaml")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent
    cfg = load_config(workspace_root / args.config)

    runner = EurekaNumericIRL(workspace_root=workspace_root, config=cfg)
    result = runner.run()

    print("Finished Eureka-like numeric IRL run")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
