"""Run the external HaWoR demo with project-local environment defaults."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HaWoR demo.py safely.")
    parser.add_argument(
        "--hawor-root",
        type=Path,
        default=Path("external_repos/HaWoR"),
        help="Path to the external HaWoR checkout.",
    )
    parser.add_argument(
        "--video",
        default="example/video_0.mp4",
        help="Video path relative to HaWoR root, or absolute path.",
    )
    parser.add_argument("--vis-mode", choices=("world", "cam"), default="world")
    parser.add_argument("--img-focal", type=float)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for HaWoR.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.hawor_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HaWoR root not found: {root}")

    env = os.environ.copy()
    env["YOLO_CONFIG_DIR"] = str((Path(".cache") / "ultralytics").resolve())
    env["PATH"] = str(root) + os.pathsep + env.get("PATH", "")

    command = [
        args.python,
        "demo.py",
        "--video_path",
        args.video,
        "--vis_mode",
        args.vis_mode,
    ]
    if args.img_focal is not None:
        command.extend(["--img_focal", str(args.img_focal)])

    print(" ".join(command))
    return subprocess.run(command, cwd=root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
