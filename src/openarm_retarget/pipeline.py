from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import load_config
from .bimanual import prefix_fields, side_pose
from .dataset import build_bimanual_dataset, build_dataset
from .hand_tracking import (
    extract_video_bimanual_hand_pose,
    extract_video_hand_pose,
)
from .ik_solver import BimanualJacobianIKSolver, JacobianIKSolver
from .io import save_npz
from .mujoco_replay import replay_bimanual_trajectory, replay_trajectory
from .openarm_model import load_bimanual_openarm, load_openarm
from .pinch import detect_pinch
from .plots import plot_ik_error, plot_pinch, plot_target, plot_wrist
from .retargeting import retarget_wrist
from .smoothing import smooth_wrist
from .synthetic import generate_bimanual_hand_pose, generate_hand_pose


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


@dataclass(frozen=True)
class BimanualPipelineArtifacts:
    hand_pose: Path
    pinch: Path
    smooth: Path
    target: Path
    trajectory: Path
    dataset: Path
    left_pinch_plot: Path
    right_pinch_plot: Path
    left_smoothing_plot: Path
    right_smoothing_plot: Path
    left_target_plot: Path
    right_target_plot: Path
    left_ik_plot: Path
    right_ik_plot: Path
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


def _bimanual_artifact_paths(
    root: Path, name: str, render: bool
) -> BimanualPipelineArtifacts:
    plot_root = root / "outputs" / "plots"
    return BimanualPipelineArtifacts(
        hand_pose=root / "data" / "hand_pose" / f"{name}_bimanual_hand_pose.npz",
        pinch=root / "data" / "hand_pose" / f"{name}_bimanual_pinch.npz",
        smooth=root / "data" / "hand_pose" / f"{name}_bimanual_smooth.npz",
        target=root / "data" / "robot_targets" / f"{name}_bimanual_target.npz",
        trajectory=root / "data" / "robot_traj" / f"{name}_bimanual_qpos.npz",
        dataset=root / "data" / "datasets" / f"{name}_bimanual_dataset.npz",
        left_pinch_plot=plot_root / f"{name}_left_pinch.png",
        right_pinch_plot=plot_root / f"{name}_right_pinch.png",
        left_smoothing_plot=plot_root / f"{name}_left_smoothing.png",
        right_smoothing_plot=plot_root / f"{name}_right_smoothing.png",
        left_target_plot=plot_root / f"{name}_left_target.png",
        right_target_plot=plot_root / f"{name}_right_target.png",
        left_ik_plot=plot_root / f"{name}_left_ik_error.png",
        right_ik_plot=plot_root / f"{name}_right_ik_error.png",
        quality_report=root / "outputs" / f"{name}_bimanual_quality_report.json",
        replay_video=(
            root / "outputs" / "replay_videos" / f"{name}_bimanual_openarm.mp4"
            if render
            else None
        ),
    )


def _configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    configs = {
        name: load_config(config_dir / f"{name}.yaml")
        for name in ("hand_tracking", "pinch", "retarget", "ik", "openarm")
    }
    bimanual_path = config_dir / "bimanual_retarget.yaml"
    if bimanual_path.is_file():
        configs["bimanual_retarget"] = load_config(bimanual_path)
    return configs


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


def run_bimanual_from_pose(
    pose_data: Mapping[str, Any],
    *,
    name: str,
    root: str | Path = ".",
    config_dir: str | Path = "configs",
    source: str = "synthetic_bimanual",
    render: bool = False,
    replay_mode: str = "kinematic",
) -> tuple[BimanualPipelineArtifacts, dict[str, Any]]:
    root_path = Path(root).resolve()
    configs = _configs(Path(config_dir))
    if "bimanual_retarget" not in configs:
        raise FileNotFoundError(
            f"Missing bimanual config: {Path(config_dir) / 'bimanual_retarget.yaml'}"
        )
    artifacts = _bimanual_artifact_paths(root_path, name, render)
    pose = {key: np.asarray(value) for key, value in pose_data.items()}
    save_npz(
        artifacts.hand_pose,
        pose,
        stage="bimanual_hand_pose",
        metadata={"source": source},
    )

    side_poses = {
        side: side_pose(pose, side) for side in ("left", "right")
    }
    pinch_results = {
        side: detect_pinch(side_poses[side], **configs["pinch"])
        for side in ("left", "right")
    }
    pinch_data: dict[str, Any] = {"timestamps": pose["timestamps"]}
    for side in ("left", "right"):
        pinch_data.update(prefix_fields(side, side_poses[side]))
        pinch_data.update(prefix_fields(side, pinch_results[side], exclude=()))
        plot_pinch(
            pose["timestamps"],
            pinch_results[side]["pinch_distance"],
            pinch_results[side]["gripper_cmd"],
            getattr(artifacts, f"{side}_pinch_plot"),
            close_threshold=float(configs["pinch"]["close_threshold"]),
            open_threshold=float(configs["pinch"]["open_threshold"]),
        )
    save_npz(
        artifacts.pinch,
        pinch_data,
        stage="bimanual_pinch",
        metadata={"source": str(artifacts.hand_pose), "config": configs["pinch"]},
    )

    smooth_data: dict[str, Any] = {"timestamps": pose["timestamps"]}
    targets: dict[str, np.ndarray] = {}
    target_data: dict[str, Any] = {"timestamps": pose["timestamps"]}
    for side in ("left", "right"):
        side_data = side_poses[side]
        wrist_smooth = smooth_wrist(
            side_data["wrist"],
            side_data["valid"],
            window=7,
            max_speed=2.0,
            timestamps=pose["timestamps"],
        )
        if "palm_scale" in side_data:
            depth_proxy = smooth_wrist(
                np.asarray(side_data["palm_scale"])[:, None],
                side_data["valid"],
                window=7,
                max_speed=2.0,
                timestamps=pose["timestamps"],
            )[:, 0]
            wrist_smooth = wrist_smooth.copy()
            wrist_smooth[:, 2] = depth_proxy
        command = pinch_results[side]["gripper_cmd"]
        smooth_data.update(
            {
                f"{side}_valid": side_data["valid"],
                f"{side}_wrist_raw": side_data["wrist"],
                f"{side}_wrist_smooth": wrist_smooth,
                f"{side}_depth_proxy": wrist_smooth[:, 2],
                f"{side}_gripper_cmd": command,
            }
        )
        plot_wrist(
            pose["timestamps"],
            side_data["wrist"],
            wrist_smooth,
            getattr(artifacts, f"{side}_smoothing_plot"),
        )
        targets[side] = retarget_wrist(
            wrist_smooth, configs["bimanual_retarget"][side]
        )
        target_data.update(
            {
                f"{side}_wrist_smooth": wrist_smooth,
                f"{side}_target_pos": targets[side],
                f"{side}_gripper_cmd": command,
            }
        )
        plot_target(targets[side], getattr(artifacts, f"{side}_target_plot"))
    save_npz(
        artifacts.smooth,
        smooth_data,
        stage="bimanual_smooth_wrist",
        metadata={"source": str(artifacts.pinch), "window": 7, "max_speed": 2.0},
    )
    save_npz(
        artifacts.target,
        target_data,
        stage="bimanual_robot_target",
        metadata={
            "source": str(artifacts.smooth),
            "config": configs["bimanual_retarget"],
        },
    )

    model, model_info = load_bimanual_openarm(configs["openarm"])
    ik_result = BimanualJacobianIKSolver(
        model, model_info, configs["ik"]
    ).solve(targets["left"], targets["right"])
    trajectory_data = {
        "timestamps": pose["timestamps"],
        "qpos": ik_result.qpos,
        "left_arm_qpos": ik_result.left_arm_qpos,
        "right_arm_qpos": ik_result.right_arm_qpos,
        "left_ee_pos": ik_result.left_ee_pos,
        "right_ee_pos": ik_result.right_ee_pos,
        "left_target_pos": ik_result.left_target_pos,
        "right_target_pos": ik_result.right_target_pos,
        "left_ik_error": ik_result.left_ik_error,
        "right_ik_error": ik_result.right_ik_error,
        "ik_converged": ik_result.converged,
        "ik_iterations": ik_result.iterations,
        "left_gripper_cmd": pinch_results["left"]["gripper_cmd"],
        "right_gripper_cmd": pinch_results["right"]["gripper_cmd"],
    }
    save_npz(
        artifacts.trajectory,
        trajectory_data,
        stage="bimanual_openarm_ik",
        metadata={
            "source": str(artifacts.target),
            "model_path": str(model_info.model_path),
            "left_ee_site": model_info.sides["left"].ee_site,
            "right_ee_site": model_info.sides["right"].ee_site,
            "config": configs["ik"],
        },
    )
    for side in ("left", "right"):
        plot_ik_error(
            pose["timestamps"],
            getattr(ik_result, f"{side}_ik_error"),
            getattr(artifacts, f"{side}_ik_plot"),
            tolerance=float(configs["ik"]["tolerance"]),
        )

    dataset = build_bimanual_dataset(
        smooth_data, target_data, trajectory_data
    )
    save_npz(
        artifacts.dataset,
        dataset,
        stage="bimanual_dataset",
        metadata={"trajectory_source": str(artifacts.trajectory)},
    )
    if artifacts.replay_video is not None:
        replay_bimanual_trajectory(
            model,
            model_info,
            ik_result.qpos,
            pinch_results["left"]["gripper_cmd"],
            pinch_results["right"]["gripper_cmd"],
            mode=replay_mode,
            output=artifacts.replay_video,
            fps=float(configs["openarm"].get("fps", 30)),
            width=int(configs["openarm"].get("render_width", 960)),
            height=int(configs["openarm"].get("render_height", 720)),
            camera=configs["openarm"].get("camera"),
        )

    quality: dict[str, Any] = {
        "name": name,
        "source": source,
        "mode": "bimanual",
        "frames": int(len(pose["timestamps"])),
        "model_path": str(model_info.model_path),
        "ik_converged_ratio": float(np.mean(ik_result.converged)),
        "artifacts": {
            key: str(value) if value is not None else None
            for key, value in asdict(artifacts).items()
        },
    }
    all_errors = []
    for side in ("left", "right"):
        command = pinch_results[side]["gripper_cmd"]
        error = getattr(ik_result, f"{side}_ik_error")
        all_errors.append(error)
        quality[side] = {
            "valid_tracking_ratio": float(
                np.mean(side_poses[side]["valid"])
            ),
            "gripper_transitions": int(np.count_nonzero(np.diff(command))),
            "target_min_m": targets[side].min(axis=0).tolist(),
            "target_max_m": targets[side].max(axis=0).tolist(),
            "mean_ik_error_m": float(np.mean(error)),
            "max_ik_error_m": float(np.max(error)),
            "ee_site": model_info.sides[side].ee_site,
            "gripper_actuator": model_info.sides[side].gripper_actuator,
        }
    combined_errors = np.concatenate(all_errors)
    quality["mean_ik_error_m"] = float(np.mean(combined_errors))
    quality["max_ik_error_m"] = float(np.max(combined_errors))
    artifacts.quality_report.parent.mkdir(parents=True, exist_ok=True)
    artifacts.quality_report.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifacts, quality


def run_bimanual_synthetic(
    *,
    name: str = "synthetic",
    frames: int = 180,
    root: str | Path = ".",
    config_dir: str | Path = "configs",
    render: bool = False,
    replay_mode: str = "kinematic",
) -> tuple[BimanualPipelineArtifacts, dict[str, Any]]:
    return run_bimanual_from_pose(
        generate_bimanual_hand_pose(frames=frames),
        name=name,
        root=root,
        config_dir=config_dir,
        source="synthetic_bimanual",
        render=render,
        replay_mode=replay_mode,
    )


def run_bimanual_video(
    video: str | Path,
    *,
    name: str,
    root: str | Path = ".",
    config_dir: str | Path = "configs",
    render: bool = False,
    replay_mode: str = "kinematic",
) -> tuple[BimanualPipelineArtifacts, dict[str, Any]]:
    root_path = Path(root).resolve()
    configs = _configs(Path(config_dir))
    debug_video = (
        root_path / "outputs" / "debug_videos" / f"{name}_bimanual_hand_debug.mp4"
    )
    tracking = extract_video_bimanual_hand_pose(
        video,
        configs["hand_tracking"],
        debug_video=debug_video,
    )
    return run_bimanual_from_pose(
        tracking.data,
        name=name,
        root=root_path,
        config_dir=config_dir,
        source=str(Path(video).resolve()),
        render=render,
        replay_mode=replay_mode,
    )
