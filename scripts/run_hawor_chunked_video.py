"""Run HaWoR/OpenArm on a long video by processing short chunks sequentially."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from openarm_retarget.comparison_video import create_comparison_video
from openarm_retarget.io import save_npz
from stitch_hawor_chunk_poses import stitch_chunks
from stabilize_hawor_pose import stabilize_pose


DOCKER_IMAGE = "video-to-openarm/hawor:cu117-torch113"
DOCKER_PYTHONPATH = "/workspace/src:/opt/droid-slam:/workspace/external_repos/HaWoR"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return float(result.stdout.strip())


def make_chunk(
    source: Path,
    destination: Path,
    *,
    start: float,
    duration: float,
    overwrite: bool,
) -> None:
    if destination.is_file() and not overwrite:
        print(f"skip existing chunk: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(destination),
        ]
    )


def docker_hawor(video: Path) -> None:
    run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--ipc=host",
            "--shm-size=16g",
            "-e",
            "YOLO_CONFIG_DIR=/tmp/Ultralytics",
            "-e",
            f"PYTHONPATH={DOCKER_PYTHONPATH}",
            "-v",
            f"{Path.home() / 'work' / 'video_to_openarm'}:/workspace",
            "-w",
            "/workspace",
            DOCKER_IMAGE,
            "python",
            "scripts/run_hawor_inference.py",
            "--video",
            str(video),
        ]
    )


def docker_convert(world_path: Path, output_npz: Path) -> None:
    run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-e",
            f"PYTHONPATH={DOCKER_PYTHONPATH}",
            "-v",
            f"{Path.home() / 'work' / 'video_to_openarm'}:/workspace",
            "-w",
            "/workspace",
            DOCKER_IMAGE,
            "python",
            "scripts/convert_hawor_world_pose.py",
            "--input",
            str(world_path),
            "--output",
            str(output_npz),
            "--hawor-root",
            "external_repos/HaWoR",
        ]
    )


def run_openarm(pose: Path, name: str) -> Path:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    run(
        [
            sys.executable,
            "scripts/run_hawor_world.py",
            "--pose",
            str(pose),
            "--name",
            name,
            "--render",
        ],
        env=env,
    )
    return Path("outputs/replay_videos") / f"{name}_bimanual_openarm.mp4"


def concat_videos(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    list_file = output.with_suffix(".concat.txt")
    list_file.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in inputs),
        encoding="utf-8",
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(output),
        ]
    )


def read_quality(name: str) -> dict[str, Any]:
    path = Path("outputs") / f"{name}_bimanual_quality_report.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a long video through HaWoR/OpenArm in short chunks."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=0.0,
        help="Overlap adjacent chunks and cosine-blend their pose trajectories.",
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Total source duration to process. Defaults to the remaining video.",
    )
    parser.add_argument(
        "--chunk-root",
        type=Path,
        default=Path("data/raw_videos/chunks"),
        help="Where generated chunk videos and HaWoR chunk folders are stored.",
    )
    parser.add_argument("--overwrite-chunks", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing chunk videos/world_space_res/NPZ/replay/comparison outputs.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    parser.add_argument(
        "--stitch-full",
        action="store_true",
        help="After chunks finish, stitch pose chunks and render one full-length replay.",
    )
    parser.add_argument(
        "--stitch-align",
        action="store_true",
        help="Align chunk boundaries during full pose stitching. Off by default because per-chunk normalization can over-correct.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = args.video
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if args.chunk_seconds <= 0:
        raise ValueError("chunk-seconds must be positive")
    if args.overlap_seconds < 0 or args.overlap_seconds >= args.chunk_seconds:
        raise ValueError("overlap-seconds must be >= 0 and smaller than chunk-seconds")

    source_duration = ffprobe_duration(video)
    duration = (
        source_duration - args.start_seconds
        if args.duration_seconds is None
        else min(args.duration_seconds, source_duration - args.start_seconds)
    )
    if duration <= 0:
        raise ValueError("No positive duration to process")

    stride = args.chunk_seconds - args.overlap_seconds
    chunk_count = max(1, int(math.ceil(max(0.0, duration - args.chunk_seconds) / stride)) + 1)
    if args.max_chunks is not None:
        chunk_count = min(chunk_count, args.max_chunks)

    run_stem = (
        f"{video.stem}_overlap_{args.overlap_seconds:g}s"
        if args.overlap_seconds
        else video.stem
    )
    root = args.chunk_root / run_stem
    comparison_paths: list[Path] = []
    pose_paths: list[Path] = []
    summary: list[dict[str, Any]] = []

    for index in range(chunk_count):
        start = args.start_seconds + index * stride
        chunk_duration = min(args.chunk_seconds, args.start_seconds + duration - start)
        chunk_stem = f"{run_stem}_chunk_{index:03d}"
        chunk_video = root / f"{chunk_stem}.mp4"
        chunk_name = f"{chunk_stem}_hawor_world"
        world_path = root / chunk_stem / "world_space_res.pth"
        pose_npz = (
            Path("outputs/external_trials/hawor_world/chunks")
            / run_stem
            / f"{chunk_name}_hand_pose.npz"
        )
        replay_video = Path("outputs/replay_videos") / f"{chunk_name}_bimanual_openarm.mp4"
        comparison = (
            Path("outputs/comparison/chunks")
            / run_stem
            / f"{chunk_name}_human_vs_robot.mp4"
        )

        print(f"\\n=== Chunk {index + 1}/{chunk_count}: {start:.2f}s + {chunk_duration:.2f}s ===", flush=True)
        make_chunk(
            video,
            chunk_video,
            start=start,
            duration=chunk_duration,
            overwrite=args.overwrite_chunks,
        )
        if not world_path.is_file() or not args.skip_existing:
            docker_hawor(chunk_video)
        else:
            print(f"skip existing HaWoR output: {world_path}", flush=True)
        if not pose_npz.is_file() or not args.skip_existing:
            docker_convert(world_path, pose_npz)
        else:
            print(f"skip existing pose NPZ: {pose_npz}", flush=True)
        if not replay_video.is_file() or not args.skip_existing:
            replay_video = run_openarm(pose_npz, chunk_name)
        else:
            print(f"skip existing replay: {replay_video}", flush=True)
        if not comparison.is_file() or not args.skip_existing:
            create_comparison_video(
                chunk_video,
                replay_video,
                comparison,
                panel_width=960,
                panel_height=720,
                overwrite=True,
            )
        else:
            print(f"skip existing comparison: {comparison}", flush=True)

        comparison_paths.append(comparison)
        pose_paths.append(pose_npz)
        quality = read_quality(chunk_name)
        summary.append(
            {
                "chunk": index,
                "start_seconds": start,
                "duration_seconds": chunk_duration,
                "chunk_video": str(chunk_video),
                "world_space_res": str(world_path),
                "pose_npz": str(pose_npz),
                "replay_video": str(replay_video),
                "comparison_video": str(comparison),
                "mean_ik_error_m": quality.get("mean_ik_error_m"),
                "max_ik_error_m": quality.get("max_ik_error_m"),
                "ik_converged_ratio": quality.get("ik_converged_ratio"),
            }
        )

    final_comparison = Path("outputs/comparison") / f"{run_stem}_hawor_world_chunked_human_vs_robot.mp4"
    concat_videos(comparison_paths, final_comparison)
    stitched_comparison = None
    if args.stitch_full:
        stitched_pose = Path("outputs/stiched_hawor_world") / f"{run_stem}_stitched_hawor_world_hand_pose.npz"
        stitched_pose_data, diagnostics = stitch_chunks(
            pose_paths,
            fps=30.0,
            align_boundaries=args.stitch_align or bool(args.overlap_seconds),
            overlap_frames=int(round(args.overlap_seconds * 30.0)),
        )
        if args.overlap_seconds:
            stitched_pose_data = stabilize_pose(stitched_pose_data)
        save_npz(
            stitched_pose,
            stitched_pose_data,
            stage="hawor_world_chunked_bimanual_hand_pose",
            metadata={
                "inputs": [str(path) for path in pose_paths],
                "fps": 30.0,
                "align_boundaries": args.stitch_align,
                "overlap_seconds": args.overlap_seconds,
            },
        )
        diagnostics_path = stitched_pose.with_suffix(".diagnostics.json")
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        stitched_name = f"{run_stem}_hawor_world_stitched"
        stitched_replay = run_openarm(stitched_pose, stitched_name)
        stitched_comparison = (
            Path("outputs/comparison")
            / f"{run_stem}_hawor_world_stitched_human_vs_robot.mp4"
        )
        create_comparison_video(
            video,
            stitched_replay,
            stitched_comparison,
            panel_width=960,
            panel_height=720,
            overwrite=True,
        )

    summary_path = Path("outputs") / f"{run_stem}_hawor_world_chunked_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\\nDone")
    print(f"Final comparison: {final_comparison}")
    if stitched_comparison is not None:
        print(f"Stitched comparison: {stitched_comparison}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
