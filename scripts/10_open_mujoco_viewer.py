from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.config import load_config
from openarm_retarget.viewer import launch_viewer


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the OpenArm MuJoCo viewer")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/openarm.yaml")
    )
    parser.add_argument("--keyframe", default="home")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--walls", action="store_true")
    parser.add_argument("--no-sheet", action="store_true")
    args = parser.parse_args()
    return launch_viewer(
        load_config(args.config),
        keyframe=args.keyframe,
        static=args.static,
        walls=args.walls,
        no_sheet=args.no_sheet,
    )


if __name__ == "__main__":
    raise SystemExit(main())

