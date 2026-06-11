from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.config import load_config
from openarm_retarget.bimanual import prefix_fields, side_pose
from openarm_retarget.io import load_npz, save_npz
from openarm_retarget.pinch import detect_pinch
from openarm_retarget.plots import plot_pinch


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect pinch with hysteresis")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/pinch.yaml"))
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    data = load_npz(args.input)
    config = load_config(args.config)
    if "left_wrist" in data:
        results = {
            side: detect_pinch(side_pose(data, side), **config)
            for side in ("left", "right")
        }
        payload = dict(data)
        for side in ("left", "right"):
            payload.update(prefix_fields(side, results[side], exclude=()))
            if args.plot:
                side_plot = args.plot.with_name(
                    f"{args.plot.stem}_{side}{args.plot.suffix}"
                )
                plot_pinch(
                    data["timestamps"],
                    results[side]["pinch_distance"],
                    results[side]["gripper_cmd"],
                    side_plot,
                    close_threshold=float(config["close_threshold"]),
                    open_threshold=float(config["open_threshold"]),
                )
        transitions = sum(
            int(
                (
                    results[side]["gripper_cmd"][1:]
                    != results[side]["gripper_cmd"][:-1]
                ).sum()
            )
            for side in ("left", "right")
        )
        stage = "bimanual_pinch"
    else:
        result = detect_pinch(data, **config)
        payload = {**data, **result}
        if args.plot:
            plot_pinch(
                data["timestamps"],
                result["pinch_distance"],
                result["gripper_cmd"],
                args.plot,
                close_threshold=float(config["close_threshold"]),
                open_threshold=float(config["open_threshold"]),
            )
        transitions = int(
            (result["gripper_cmd"][1:] != result["gripper_cmd"][:-1]).sum()
        )
        stage = "pinch"
    save_npz(
        args.output,
        payload,
        stage=stage,
        metadata={"source": str(args.input), "config": config},
    )
    print(f"Saved pinch signal to {args.output}; transitions={transitions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
