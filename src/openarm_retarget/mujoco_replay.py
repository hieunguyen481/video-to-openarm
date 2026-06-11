from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .openarm_model import OpenArmModelInfo, reset_home


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
    gripper_joint_id = int(model.actuator_trnid[gripper_id, 0])
    gripper_qpos_index = int(model.jnt_qposadr[gripper_joint_id])
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
            gripper_target = float(
                np.interp(command[frame_index], [0.0, 1.0], gripper_range)
            )
            if mode == "kinematic":
                data.qpos[:] = desired
                data.qpos[gripper_qpos_index] = gripper_target
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
