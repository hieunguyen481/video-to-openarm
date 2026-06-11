from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_bimanual_synthetic, run_bimanual_video
from .viewer import launch_viewer


def _demo(args: argparse.Namespace) -> int:
    common = {
        "name": args.name,
        "root": args.root,
        "config_dir": args.config_dir,
        "render": args.render,
        "replay_mode": args.replay_mode,
    }
    if args.video:
        _, quality = run_bimanual_video(args.video, **common)
    else:
        _, quality = run_bimanual_synthetic(frames=args.frames, **common)
    print(json.dumps(quality, indent=2, ensure_ascii=False))
    return 0 if quality["max_ik_error_m"] < 0.20 else 2


def _viewer(args: argparse.Namespace) -> int:
    return launch_viewer(
        load_config(args.config),
        keyframe=args.keyframe,
        static=args.static,
        walls=args.walls,
        no_sheet=args.no_sheet,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openarm-retarget",
        description="Video hand motion to OpenArm MuJoCo retargeting",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run the complete pipeline")
    demo.add_argument("--video", type=Path, help="Input MP4; omit for synthetic data")
    demo.add_argument("--name", default="demo")
    demo.add_argument("--frames", type=int, default=180)
    demo.add_argument("--root", type=Path, default=Path("."))
    demo.add_argument("--config-dir", type=Path, default=Path("configs"))
    demo.add_argument("--render", action="store_true")
    demo.add_argument(
        "--replay-mode",
        choices=("kinematic", "actuator"),
        default="kinematic",
    )
    demo.set_defaults(handler=_demo)

    viewer = subparsers.add_parser(
        "viewer", help="Open the OpenArm MuJoCo interactive viewer"
    )
    viewer.add_argument(
        "--config", type=Path, default=Path("configs/openarm.yaml")
    )
    viewer.add_argument("--keyframe", default="home")
    viewer.add_argument("--static", action="store_true")
    viewer.add_argument("--walls", action="store_true")
    viewer.add_argument("--no-sheet", action="store_true")
    viewer.set_defaults(handler=_viewer)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
