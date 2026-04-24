"""Step 0: Prepare environment for Tutorial 01 code snippets."""

from pathlib import Path
import sys
import warnings


def resolve_base_path() -> Path:
    # Script is located at commonroad_rl/tutorials/tutorial01_local
    base_path = Path(__file__).resolve().parents[2]
    if base_path.name != "commonroad_rl":
        warnings.warn(
            f"Expected script under commonroad_rl, got: {base_path}",
            stacklevel=2,
        )
    return base_path


if __name__ == "__main__":
    base_path = resolve_base_path()
    print(f"Base path: {base_path}")
    print(f"Python executable: {sys.executable}")
