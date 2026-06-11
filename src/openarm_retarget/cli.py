from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_synthetic, run_video


def _demo(args: argparse.Namespace) -> int:
    common = {
        "name": args.name,
        "root": args.root,
        "config_dir": args.config_dir,
        "render": args.render,
        "replay_mode": args.replay_mode,
    }
    if args.video:
        _, quality = run_video(args.video, **common)
    else:
        _, quality = run_synthetic(frames=args.frames, **common)
    print(json.dumps(quality, indent=2, ensure_ascii=False))
    return 0 if quality["max_ik_error_m"] < 0.20 else 2


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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

