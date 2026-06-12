from __future__ import annotations

import argparse
import json
from pathlib import Path

from .camera_recorder import record_camera
from .comparison_video import create_comparison_video
from .config import load_config
from .live_teleop import run_live_teleoperation
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


def _record(args: argparse.Namespace) -> int:
    result = record_camera(
        args.output,
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=args.backend,
        mirror_preview=not args.no_mirror_preview,
        auto_start=args.auto_start,
        duration=args.duration,
        overwrite=args.overwrite,
    )
    print(
        f"Saved {result.frames} frames ({result.duration_seconds:.1f}s) "
        f"at {result.width}x{result.height}, {result.fps:.1f} FPS\n"
        f"{result.output}"
    )
    return 0


def _compare(args: argparse.Namespace) -> int:
    result = create_comparison_video(
        args.human,
        args.robot,
        args.output,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        overwrite=args.overwrite,
    )
    print(
        f"Saved synchronized comparison: {result.frames} frames, "
        f"{result.width}x{result.height}, {result.fps:.1f} FPS\n"
        f"{result.output}"
    )
    return 0


def _live(args: argparse.Namespace) -> int:
    summary = run_live_teleoperation(
        config_dir=args.config_dir,
        camera=args.camera,
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
        inference_width=args.inference_width,
        delegate=args.delegate,
        swap_left_right=args.swap_left_right,
        mirror_horizontal=args.mirror_horizontal,
        mirror_depth=args.mirror_depth,
        duration=args.duration,
        show_viewer=not args.no_viewer,
        show_preview=not args.no_preview,
        record_session=args.record_session,
        report_path=args.report,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


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

    record = subparsers.add_parser(
        "record", help="Record a two-hand video from the computer camera"
    )
    record.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw_videos/demo_001.mp4"),
    )
    record.add_argument("--camera", type=int, default=0)
    record.add_argument("--width", type=int, default=1280)
    record.add_argument("--height", type=int, default=720)
    record.add_argument("--fps", type=float, default=30.0)
    record.add_argument(
        "--backend",
        choices=("auto", "any", "dshow", "msmf", "v4l2"),
        default="auto",
    )
    record.add_argument(
        "--duration",
        type=float,
        help="Stop after this many recorded seconds",
    )
    record.add_argument("--auto-start", action="store_true")
    record.add_argument("--overwrite", action="store_true")
    record.add_argument("--no-mirror-preview", action="store_true")
    record.set_defaults(handler=_record)

    compare = subparsers.add_parser(
        "compare", help="Compose human tracking and robot replay side by side"
    )
    compare.add_argument("--human", type=Path, required=True)
    compare.add_argument("--robot", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--panel-width", type=int, default=960)
    compare.add_argument("--panel-height", type=int, default=720)
    compare.add_argument("--overwrite", action="store_true")
    compare.set_defaults(handler=_compare)

    live = subparsers.add_parser(
        "live", help="Control both OpenArm arms live from the webcam"
    )
    live.add_argument("--config-dir", type=Path, default=Path("configs"))
    live.add_argument("--camera", type=int)
    live.add_argument(
        "--backend",
        choices=("auto", "any", "dshow", "msmf", "v4l2"),
    )
    live.add_argument("--width", type=int)
    live.add_argument("--height", type=int)
    live.add_argument("--fps", type=float)
    live.add_argument("--inference-width", type=int)
    live.add_argument(
        "--delegate",
        choices=("auto", "cpu", "gpu"),
    )
    live.add_argument(
        "--swap-left-right",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Swap camera handedness to match the opposing-camera layout",
    )
    live.add_argument(
        "--mirror-horizontal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mirror horizontal hand motion in the robot coordinate frame",
    )
    live.add_argument(
        "--mirror-depth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reverse near/far motion along the robot forward axis",
    )
    live.add_argument("--duration", type=float)
    live.add_argument("--no-viewer", action="store_true")
    live.add_argument("--no-preview", action="store_true")
    live.add_argument("--record-session", type=Path)
    live.add_argument("--report", type=Path)
    live.set_defaults(handler=_live)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
