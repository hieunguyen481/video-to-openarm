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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bimanual_retarget.yaml"),
    )
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    data = load_npz(args.input)
    config = load_config(args.config)
    if "left_wrist_smooth" in data:
        payload = {"timestamps": data["timestamps"]}
        all_targets = []
        for side in ("left", "right"):
            target = retarget_wrist(
                data[f"{side}_wrist_smooth"], config[side]
            )
            all_targets.append(target)
            payload.update(
                {
                    f"{side}_wrist_smooth": data[f"{side}_wrist_smooth"],
                    f"{side}_target_pos": target,
                    f"{side}_gripper_cmd": data[f"{side}_gripper_cmd"],
                }
            )
            if args.plot:
                side_plot = args.plot.with_name(
                    f"{args.plot.stem}_{side}{args.plot.suffix}"
                )
                plot_target(target, side_plot)
        stage = "bimanual_robot_target"
    else:
        single_config = config["left"] if "left" in config else config
        target = retarget_wrist(data["wrist_smooth"], single_config)
        all_targets = [target]
        payload = {
            "timestamps": data["timestamps"],
            "wrist_smooth": data["wrist_smooth"],
            "target_pos": target,
            "gripper_cmd": data["gripper_cmd"],
        }
        if args.plot:
            plot_target(target, args.plot)
        stage = "robot_target"
    save_npz(
        args.output,
        payload,
        stage=stage,
        metadata={"source": str(args.input), "config": config},
    )
    print(f"Saved {len(all_targets[0])} OpenArm targets to {args.output}")
    for index, target in enumerate(all_targets):
        label = ("left", "right")[index] if len(all_targets) == 2 else "arm"
        print(
            f"{label} workspace min={target.min(axis=0)}, "
            f"max={target.max(axis=0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
