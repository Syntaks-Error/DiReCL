"""Step 2/2b: Convert highD raw CSV files to CommonRoad XML files."""

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
    parser.add_argument("--num-time-steps", type=int, default=1000)
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parents[2]
    raw_path = base_path / "tutorials" / "data" / "highD" / "raw"
    xml_path = base_path / "tutorials" / "data" / "highD" / "xmls"
    dc_path = base_path.parent / "external" / "commonroad-io" / "dataset-converters"

    xml_path.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "-m",
            "src.main",
            "highD",
            str(raw_path),
            str(xml_path),
            "--num_time_steps_scenario",
            str(args.num_time_steps),
        ],
        cwd=dc_path,
    )
