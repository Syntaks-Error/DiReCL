"""Step 4: Visualize a sample CommonRoad XML scenario."""

from pathlib import Path
import glob
import argparse
import os

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--figsize-w", type=float, default=25)
    parser.add_argument("--figsize-h", type=float, default=10)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save rendered frames. Useful on remote servers.")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open interactive windows.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parents[2]
    files = str(base_path / "tutorials" / "data" / "highD" / "xmls" / "*.xml")
    matches = sorted(glob.glob(files))

    if not matches:
        raise FileNotFoundError(f"No XML files found with: {files}")

    file_path = matches[0]
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, args.steps):
        fig = plt.figure(figsize=(args.figsize_w, args.figsize_h))
        rnd = MPRenderer()
        draw_params = MPDrawParams()
        draw_params.time_begin = i
        draw_params.time_end = i
        scenario.draw(rnd, draw_params=draw_params)
        planning_problem_set.draw(rnd)
        rnd.render()

        if save_dir is not None:
            fig.savefig(save_dir / f"{scenario.scenario_id}_{i:04d}.png", dpi=args.dpi, bbox_inches="tight")

        if not args.no_show:
            plt.show(block=False)
            plt.pause(0.001)

        # Close all opened figures after each render to avoid accumulating windows/memory.
        plt.close("all")

    if save_dir is not None:
        print(f"Saved {args.steps} frame(s) to: {save_dir}")
