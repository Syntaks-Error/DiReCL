"""Step 1/2a: Clone and install dataset-converters."""

from pathlib import Path
import argparse
import subprocess
from typing import List, Optional


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install -r requirements.txt")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parents[2]
    io_parent = base_path.parent / "external" / "commonroad-io"
    repo_path = io_parent / "dataset-converters"

    io_parent.mkdir(parents=True, exist_ok=True)
    if not repo_path.exists():
        run(["git", "clone", "https://gitlab.lrz.de/tum-cps/dataset-converters.git"], cwd=io_parent)
    else:
        print(f"Repository already exists: {repo_path}")

    if not args.skip_install:
        run(["pip", "install", "-r", "requirements.txt"], cwd=repo_path)
