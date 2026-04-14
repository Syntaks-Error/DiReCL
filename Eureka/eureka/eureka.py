import hydra
import logging
import os
from pathlib import Path

import openai

from eureka_mujoco import run_mujoco

EUREKA_ROOT_DIR = os.getcwd()


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    run_mujoco(cfg)


if __name__ == "__main__":
    main()
