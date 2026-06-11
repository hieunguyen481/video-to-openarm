from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.io import load_npz, save_npz
from openarm_retarget.plots import plot_wrist
from openarm_retarget.smoothing import smooth_wrist


def main() -> int:
    parser = argparse.ArgumentParser(description="Interpolate and smooth wrist trajectory")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--max-speed", type=float, default=2.0)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    data = load_npz(args.input)
    if "left_wrist" in data:
        payload = {"timestamps": data["timestamps"]}
        for side in ("left", "right"):
            smooth = smooth_wrist(
                data[f"{side}_wrist"],
                data[f"{side}_valid"],
                window=args.window,
                max_speed=args.max_speed,
                timestamps=data["timestamps"],
            )
            payload.update(
                {
                    f"{side}_valid": data[f"{side}_valid"],
                    f"{side}_wrist_raw": data[f"{side}_wrist"],
                    f"{side}_wrist_smooth": smooth,
                    f"{side}_gripper_cmd": data.get(
                        f"{side}_gripper_cmd",
                        data[f"{side}_valid"].astype("float32") * 0,
                    ),
                }
            )
            if args.plot:
                side_plot = args.plot.with_name(
                    f"{args.plot.stem}_{side}{args.plot.suffix}"
                )
                plot_wrist(
                    data["timestamps"],
                    data[f"{side}_wrist"],
                    smooth,
                    side_plot,
                )
        stage = "bimanual_smooth_wrist"
    else:
        smooth = smooth_wrist(
            data["wrist"],
            data["valid"],
            window=args.window,
            max_speed=args.max_speed,
            timestamps=data["timestamps"],
        )
        payload = {
            "timestamps": data["timestamps"],
            "valid": data["valid"],
            "wrist_raw": data["wrist"],
            "wrist_smooth": smooth,
            "gripper_cmd": data.get(
                "gripper_cmd", data["valid"].astype("float32") * 0
            ),
        }
        if args.plot:
            plot_wrist(data["timestamps"], data["wrist"], smooth, args.plot)
        stage = "smooth_wrist"
    save_npz(
        args.output,
        payload,
        stage=stage,
        metadata={"source": str(args.input), "window": args.window},
    )
    print(f"Saved smoothed wrist trajectory to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
