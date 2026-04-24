"""Step 5: Convert XML scenarios to pickles."""

from pathlib import Path
import subprocess
import sys
from typing import List, Optional


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[2]
    input_xml_dir = base_path / "tutorials" / "data" / "highD" / "xmls"
    output_pickle_dir = base_path / "tutorials" / "data" / "highD" / "pickles"

    output_pickle_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "-m",
            "commonroad_rl.tools.pickle_scenario.xml_to_pickle",
            "-i",
            str(input_xml_dir),
            "-o",
            str(output_pickle_dir),
        ],
        cwd=base_path.parent,
    )
