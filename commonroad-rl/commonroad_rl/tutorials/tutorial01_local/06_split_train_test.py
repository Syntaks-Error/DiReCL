"""Step 6: Split pickle scenarios into train/test sets."""

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
    parser.add_argument("--train-ratio", type=float, default=0.7)
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parents[2]
    input_dir = base_path / "tutorials" / "data" / "highD" / "pickles" / "problem"
    output_train = base_path / "tutorials" / "data" / "highD" / "pickles" / "problem_train"
    output_test = base_path / "tutorials" / "data" / "highD" / "pickles" / "problem_test"

    output_train.mkdir(parents=True, exist_ok=True)
    output_test.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "-m",
            "commonroad_rl.utils_run.split_dataset",
            "-i",
            str(input_dir),
            "-otrain",
            str(output_train),
            "-otest",
            str(output_test),
            "-tr_r",
            str(args.train_ratio),
        ],
        cwd=base_path.parent,
    )
