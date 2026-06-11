from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .openarm_model import BimanualOpenArmInfo, OpenArmModelInfo, reset_home


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError('Replay requires MuJoCo. Install with: python -m pip install -e ".[simulation]"') from exc
    return mujoco


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            'Video output requires OpenCV. Install with: python -m pip install -e ".[vision]"'
        ) from exc
    return cv2


def _object_id(model: Any, object_type: Any, name: str) -> int:
    mujoco = _mujoco()
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise ValueError(f"Model does not contain {name!r}")
    return int(value)


def _gripper_target(command: float, control_range: np.ndarray) -> float:
    lower, upper = np.asarray(control_range, dtype=float)
    closed_target = float(np.clip(0.0, lower, upper))
    open_target = float(
        lower if abs(lower - closed_target) > abs(upper - closed_target) else upper
    )
    return float(np.interp(command, [0.0, 1.0], [open_target, closed_target]))


def _gripper_qpos_indices(model: Any, actuator_id: int) -> np.ndarray:
    mujoco = _mujoco()
    primary_joint_id = int(model.actuator_trnid[actuator_id, 0])
    primary_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, primary_joint_id
    )
    joint_ids = [primary_joint_id]
    if primary_name and primary_name.endswith("joint1"):
        secondary_name = f"{primary_name[:-1]}2"
        secondary_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, secondary_name
        )
        if secondary_id >= 0:
            joint_ids.append(int(secondary_id))
    return model.jnt_qposadr[joint_ids].astype(int)


def replay_trajectory(
    model: Any,
    info: OpenArmModelInfo,
    qpos: np.ndarray,
    gripper_cmd: np.ndarray,
    *,
    mode: str = "kinematic",
    output: str | Path | None = None,
    fps: float = 30.0,
    width: int = 960,
    height: int = 720,
    camera: str | int | None = None,
    substeps: int = 8,
) -> np.ndarray:
    mujoco = _mujoco()
    trajectory = np.asarray(qpos, dtype=float)
    command = np.asarray(gripper_cmd, dtype=float).reshape(-1)
    if trajectory.ndim != 2 or trajectory.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape [T, {model.nq}]")
    if len(command) != len(trajectory):
        raise ValueError("gripper_cmd and qpos must have the same length")
    if mode not in {"kinematic", "actuator"}:
        raise ValueError("mode must be 'kinematic' or 'actuator'")
    if substeps < 1:
        raise ValueError("substeps must be positive")

    data = mujoco.MjData(model)
    reset_home(model, data, info.home_keyframe)
    arm_joint_ids = [
        _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in info.arm_joint_names
    ]
    arm_qpos_indices = model.jnt_qposadr[arm_joint_ids].astype(int)
    arm_actuator_ids = [
        _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in info.arm_actuator_names
    ]
    gripper_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, info.gripper_actuator
    )
    gripper_qpos_indices = _gripper_qpos_indices(model, gripper_id)
    gripper_range = model.actuator_ctrlrange[gripper_id]

    renderer = None
    writer = None
    cv2_module = None
    if output:
        cv2_module = _cv2()
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        renderer = mujoco.Renderer(model, height=height, width=width)
        writer = cv2_module.VideoWriter(
            str(destination),
            cv2_module.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create replay video: {destination}")

    achieved = np.empty_like(trajectory)
    try:
        for frame_index, desired in enumerate(trajectory):
            gripper_target = _gripper_target(
                command[frame_index], gripper_range
            )
            if mode == "kinematic":
                data.qpos[:] = desired
                data.qpos[gripper_qpos_indices] = gripper_target
                mujoco.mj_forward(model, data)
            else:
                data.ctrl[arm_actuator_ids] = desired[arm_qpos_indices]
                data.ctrl[gripper_id] = gripper_target
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
            achieved[frame_index] = data.qpos

            if renderer is not None and writer is not None:
                if camera is None:
                    renderer.update_scene(data)
                else:
                    renderer.update_scene(data, camera=camera)
                rgb = renderer.render()
                writer.write(
                    cv2_module.cvtColor(rgb, cv2_module.COLOR_RGB2BGR)
                )
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
    return achieved


def replay_bimanual_trajectory(
    model: Any,
    info: BimanualOpenArmInfo,
    qpos: np.ndarray,
    left_gripper_cmd: np.ndarray,
    right_gripper_cmd: np.ndarray,
    *,
    mode: str = "kinematic",
    output: str | Path | None = None,
    fps: float = 30.0,
    width: int = 960,
    height: int = 720,
    camera: str | int | None = None,
    substeps: int = 8,
) -> np.ndarray:
    mujoco = _mujoco()
    trajectory = np.asarray(qpos, dtype=float)
    commands = {
        "left": np.asarray(left_gripper_cmd, dtype=float).reshape(-1),
        "right": np.asarray(right_gripper_cmd, dtype=float).reshape(-1),
    }
    if trajectory.ndim != 2 or trajectory.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape [T, {model.nq}]")
    if any(len(command) != len(trajectory) for command in commands.values()):
        raise ValueError("Both gripper commands must match qpos length")
    if mode not in {"kinematic", "actuator"}:
        raise ValueError("mode must be 'kinematic' or 'actuator'")
    if substeps < 1:
        raise ValueError("substeps must be positive")

    data = mujoco.MjData(model)
    reset_home(model, data, info.home_keyframe)
    arm_qpos_indices: dict[str, np.ndarray] = {}
    arm_actuator_ids: dict[str, list[int]] = {}
    gripper_ids: dict[str, int] = {}
    gripper_qpos_indices: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        arm = info.sides[side]
        joint_ids = [
            _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in arm.arm_joint_names
        ]
        arm_qpos_indices[side] = model.jnt_qposadr[joint_ids].astype(int)
        arm_actuator_ids[side] = [
            _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in arm.arm_actuator_names
        ]
        gripper_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, arm.gripper_actuator
        )
        gripper_ids[side] = gripper_id
        gripper_qpos_indices[side] = _gripper_qpos_indices(
            model, gripper_id
        )

    renderer = None
    writer = None
    cv2_module = None
    if output:
        cv2_module = _cv2()
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        renderer = mujoco.Renderer(model, height=height, width=width)
        writer = cv2_module.VideoWriter(
            str(destination),
            cv2_module.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create replay video: {destination}")

    achieved = np.empty_like(trajectory)
    try:
        for frame_index, desired in enumerate(trajectory):
            gripper_targets = {
                side: _gripper_target(
                    commands[side][frame_index],
                    model.actuator_ctrlrange[gripper_ids[side]],
                )
                for side in ("left", "right")
            }
            if mode == "kinematic":
                data.qpos[:] = desired
                for side in ("left", "right"):
                    data.qpos[gripper_qpos_indices[side]] = gripper_targets[side]
                mujoco.mj_forward(model, data)
            else:
                for side in ("left", "right"):
                    data.ctrl[arm_actuator_ids[side]] = desired[
                        arm_qpos_indices[side]
                    ]
                    data.ctrl[gripper_ids[side]] = gripper_targets[side]
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
            achieved[frame_index] = data.qpos

            if renderer is not None and writer is not None:
                if camera is None:
                    renderer.update_scene(data)
                else:
                    renderer.update_scene(data, camera=camera)
                rgb = renderer.render()
                writer.write(
                    cv2_module.cvtColor(rgb, cv2_module.COLOR_RGB2BGR)
                )
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
    return achieved
