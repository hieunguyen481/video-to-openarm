from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import load_config
from .dataset import build_dataset
from .hand_tracking import extract_video_hand_pose
from .ik_solver import JacobianIKSolver
from .io import save_npz
from .mujoco_replay import replay_trajectory
from .openarm_model import load_openarm
from .pinch import detect_pinch
from .plots import plot_ik_error, plot_pinch, plot_target, plot_wrist
from .retargeting import retarget_wrist
from .smoothing import smooth_wrist
from .synthetic import generate_hand_pose


@dataclass(frozen=True)
class PipelineArtifacts:
    hand_pose: Path
    pinch: Path
    smooth: Path
    target: Path
    trajectory: Path
    dataset: Path
    pinch_plot: Path
    smoothing_plot: Path
    target_plot: Path
    ik_plot: Path
    quality_report: Path
    replay_video: Path | None


def _artifact_paths(root: Path, name: str, render: bool) -> PipelineArtifacts:
    return PipelineArtifacts(
        hand_pose=root / "data" / "hand_pose" / f"{name}_hand_pose.npz",
        pinch=root / "data" / "hand_pose" / f"{name}_pinch.npz",
        smooth=root / "data" / "hand_pose" / f"{name}_smooth.npz",
        target=root / "data" / "robot_targets" / f"{name}_target.npz",
        trajectory=root / "data" / "robot_traj" / f"{name}_qpos.npz",
        dataset=root / "data" / "datasets" / f"{name}_dataset.npz",
        pinch_plot=root / "outputs" / "plots" / f"{name}_pinch.png",
        smoothing_plot=root / "outputs" / "plots" / f"{name}_smoothing.png",
        target_plot=root / "outputs" / "plots" / f"{name}_target.png",
        ik_plot=root / "outputs" / "plots" / f"{name}_ik_error.png",
        quality_report=root / "outputs" / f"{name}_quality_report.json",
        replay_video=(
            root / "outputs" / "replay_videos" / f"{name}_openarm.mp4"
            if render
            else None
        ),
    )


def _configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        name: load_config(config_dir / f"{name}.yaml")
        for name in ("hand_tracking", "pinch", "retarget", "ik", "openarm")
    }


def run_from_pose(
    pose_data: Mapping[str, Any],
    *,
    name: str,
    root: str | Path = ".",
    config_dir: str | Path = "configs",
    source: str = "synthetic",
    render: bool = False,
    replay_mode: str = "kinematic",
) -> tuple[PipelineArtifacts, dict[str, Any]]:
    root_path = Path(root).resolve()
    configs = _configs(Path(config_dir))
    artifacts = _artifact_paths(root_path, name, render)

    pose = {key: np.asarray(value) for key, value in pose_data.items()}
    save_npz(
        artifacts.hand_pose,
        pose,
        stage="hand_pose",
        metadata={"source": source},
    )

    pinch_result = detect_pinch(pose, **configs["pinch"])
    pinch_data = {**pose, **pinch_result}
    save_npz(
        artifacts.pinch,
        pinch_data,
        stage="pinch",
        metadata={"source": str(artifacts.hand_pose), "config": configs["pinch"]},
    )
    plot_pinch(
        pose["timestamps"],
        pinch_result["pinch_distance"],
        pinch_result["gripper_cmd"],
        artifacts.pinch_plot,
        close_threshold=float(configs["pinch"]["close_threshold"]),
        open_threshold=float(configs["pinch"]["open_threshold"]),
    )

    wrist_smooth = smooth_wrist(
        pose["wrist"],
        pose["valid"],
        window=7,
        max_speed=2.0,
        timestamps=pose["timestamps"],
    )
    smooth_data = {
        "timestamps": pose["timestamps"],
        "valid": pose["valid"],
        "wrist_raw": pose["wrist"],
        "wrist_smooth": wrist_smooth,
        "gripper_cmd": pinch_result["gripper_cmd"],
    }
    save_npz(
        artifacts.smooth,
        smooth_data,
        stage="smooth_wrist",
        metadata={"source": str(artifacts.pinch), "window": 7, "max_speed": 2.0},
    )
    plot_wrist(
        pose["timestamps"],
        pose["wrist"],
        wrist_smooth,
        artifacts.smoothing_plot,
    )

    target_pos = retarget_wrist(wrist_smooth, configs["retarget"])
    target_data = {
        "timestamps": pose["timestamps"],
        "wrist_smooth": wrist_smooth,
        "target_pos": target_pos,
        "gripper_cmd": pinch_result["gripper_cmd"],
    }
    save_npz(
        artifacts.target,
        target_data,
        stage="robot_target",
        metadata={"source": str(artifacts.smooth), "config": configs["retarget"]},
    )
    plot_target(target_pos, artifacts.target_plot)

    if configs["ik"].get("solver", "jacobian_dls") != "jacobian_dls":
        raise ValueError("Only solver=jacobian_dls is currently supported")
    model, model_info = load_openarm(configs["openarm"])
    ik_result = JacobianIKSolver(model, model_info, configs["ik"]).solve(target_pos)
    trajectory_data = {
        "timestamps": pose["timestamps"],
        "qpos": ik_result.qpos,
        "arm_qpos": ik_result.arm_qpos,
        "ee_pos": ik_result.ee_pos,
        "target_pos": ik_result.target_pos,
        "ik_error": ik_result.ik_error,
        "ik_converged": ik_result.converged,
        "ik_iterations": ik_result.iterations,
        "gripper_cmd": pinch_result["gripper_cmd"],
    }
    save_npz(
        artifacts.trajectory,
        trajectory_data,
        stage="openarm_ik",
        metadata={
            "source": str(artifacts.target),
            "model_path": str(model_info.model_path),
            "ee_site": model_info.ee_site,
            "arm_joint_names": model_info.arm_joint_names,
            "config": configs["ik"],
        },
    )
    plot_ik_error(
        pose["timestamps"],
        ik_result.ik_error,
        artifacts.ik_plot,
        tolerance=float(configs["ik"]["tolerance"]),
    )

    dataset = build_dataset(smooth_data, target_data, trajectory_data)
    save_npz(
        artifacts.dataset,
        dataset,
        stage="dataset",
        metadata={"trajectory_source": str(artifacts.trajectory)},
    )

    if artifacts.replay_video is not None:
        replay_trajectory(
            model,
            model_info,
            ik_result.qpos,
            pinch_result["gripper_cmd"],
            mode=replay_mode,
            output=artifacts.replay_video,
            fps=float(configs["openarm"].get("fps", 30)),
            width=int(configs["openarm"].get("render_width", 960)),
            height=int(configs["openarm"].get("render_height", 720)),
            camera=configs["openarm"].get("camera"),
        )

    command = pinch_result["gripper_cmd"]
    quality = {
        "name": name,
        "source": source,
        "frames": int(len(pose["timestamps"])),
        "valid_tracking_ratio": float(np.mean(pose["valid"])),
        "gripper_transitions": int(np.count_nonzero(np.diff(command))),
        "target_min_m": target_pos.min(axis=0).tolist(),
        "target_max_m": target_pos.max(axis=0).tolist(),
        "mean_ik_error_m": float(np.mean(ik_result.ik_error)),
        "max_ik_error_m": float(np.max(ik_result.ik_error)),
        "ik_converged_ratio": float(np.mean(ik_result.converged)),
        "model_path": str(model_info.model_path),
        "ee_site": model_info.ee_site,
        "artifacts": {
            key: str(value) if value is not None else None
            for key, value in asdict(artifacts).items()
        },
    }
    artifacts.quality_report.parent.mkdir(parents=True, exist_ok=True)
    artifacts.quality_report.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifacts, quality


def run_synthetic(
    *,
    name: str = "synthetic",
    frames: int = 180,
    root: str | Path = ".",
    config_dir: str | Path = "configs",
    render: bool = False,
    replay_mode: str = "kinematic",
) -> tuple[PipelineArtifacts, dict[str, Any]]:
    pose = generate_hand_pose(frames=frames)
    return run_from_pose(
        pose,
        name=name,
        root=root,
        config_dir=config_dir,
        source="synthetic",
        render=render,
        replay_mode=replay_mode,
    )


def run_video(
    video: str | Path,
    *,
    name: str,
    root: str | Path = ".",
    config_dir: str | Path = "configs",
    render: bool = False,
    replay_mode: str = "kinematic",
) -> tuple[PipelineArtifacts, dict[str, Any]]:
    root_path = Path(root).resolve()
    configs = _configs(Path(config_dir))
    debug_video = root_path / "outputs" / "debug_videos" / f"{name}_hand_debug.mp4"
    tracking = extract_video_hand_pose(
        video,
        configs["hand_tracking"],
        debug_video=debug_video,
    )
    return run_from_pose(
        tracking.data,
        name=name,
        root=root_path,
        config_dir=config_dir,
        source=str(Path(video).resolve()),
        render=render,
        replay_mode=replay_mode,
    )

