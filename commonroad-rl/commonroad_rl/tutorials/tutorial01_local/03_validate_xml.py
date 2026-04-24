"""Step 3: Validate XML scenarios against CommonRoad XSD schema."""

from pathlib import Path
import glob
import subprocess
import sys
from typing import List, Optional


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[2]
    xsd_path = base_path / "tools" / "XML_commonRoad_XSD_2020a.xsd"
    xml_glob = str(base_path / "tutorials" / "data" / "highD" / "xmls" / "*")

    xml_files = sorted(glob.glob(xml_glob))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in: {xml_glob}")

    run(
        [
            sys.executable,
            "-m",
            "commonroad_rl.tools.validate_cr",
            "-s",
            str(xsd_path),
            *xml_files,
        ],
        cwd=base_path.parent,
    )
