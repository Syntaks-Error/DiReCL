"""Step 7: Split train/test pickles into multiple environment subfolders."""

from pathlib import Path
import argparse
import subprocess
import sys
from typing import List, Optional


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-folders", type=int, default=5)
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parents[2]
    train_dir = base_path / "tutorials" / "data" / "highD" / "pickles" / "problem_train"
    test_dir = base_path / "tutorials" / "data" / "highD" / "pickles" / "problem_test"

    run(
        [
            sys.executable,
            "-m",
            "commonroad_rl.tools.pickle_scenario.copy_files",
            "-i",
            str(train_dir),
            "-o",
            str(train_dir),
            "-f",
            "*.pickle",
            "-n",
            str(args.num_folders),
        ],
        cwd=base_path.parent,
    )

    run(
        [
            sys.executable,
            "-m",
            "commonroad_rl.tools.pickle_scenario.copy_files",
            "-i",
            str(test_dir),
            "-o",
            str(test_dir),
            "-f",
            "*.pickle",
            "-n",
            str(args.num_folders),
        ],
        cwd=base_path.parent,
    )
