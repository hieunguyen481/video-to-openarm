from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.config import load_config
from openarm_retarget.io import load_npz, save_npz
from openarm_retarget.plots import plot_target
from openarm_retarget.retargeting import retarget_wrist


def main() -> int:
    parser = argparse.ArgumentParser(description="Map wrist motion to OpenArm workspace")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/retarget.yaml"))
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    data = load_npz(args.input)
    config = load_config(args.config)
    target = retarget_wrist(data["wrist_smooth"], config)
    payload = {
        "timestamps": data["timestamps"],
        "wrist_smooth": data["wrist_smooth"],
        "target_pos": target,
        "gripper_cmd": data["gripper_cmd"],
    }
    save_npz(
        args.output,
        payload,
        stage="robot_target",
        metadata={"source": str(args.input), "config": config},
    )
    if args.plot:
        plot_target(target, args.plot)
    print(f"Saved {len(target)} OpenArm targets to {args.output}")
    print(f"Workspace min={target.min(axis=0)}, max={target.max(axis=0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

