from __future__ import annotations

import argparse
from pathlib import Path

from openarm_retarget.config import load_config
from openarm_retarget.io import load_npz
from openarm_retarget.mujoco_replay import replay_trajectory
from openarm_retarget.openarm_model import load_openarm


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay OpenArm trajectory in MuJoCo")
    parser.add_argument("--traj", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/openarm.yaml"))
    parser.add_argument("--mode", choices=("kinematic", "actuator"), default="kinematic")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--substeps", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    data = load_npz(args.traj)
    model, info = load_openarm(config)
    replay_trajectory(
        model,
        info,
        data["qpos"],
        data["gripper_cmd"],
        mode=args.mode,
        output=args.output,
        fps=float(config.get("fps", 30)),
        width=int(config.get("render_width", 960)),
        height=int(config.get("render_height", 720)),
        camera=config.get("camera"),
        substeps=args.substeps,
    )
    target = f" and rendered {args.output}" if args.output else ""
    print(f"Replayed {len(data['qpos'])} frames in {args.mode} mode{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

