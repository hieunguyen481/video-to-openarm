from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .validation import require_time_length


def build_dataset(
    smooth_data: Mapping[str, Any],
    target_data: Mapping[str, Any],
    trajectory_data: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    length = require_time_length(
        smooth_data, ("timestamps", "wrist_smooth", "gripper_cmd")
    )
    if require_time_length(target_data, ("target_pos", "gripper_cmd")) != length:
        raise ValueError("Target data length does not match smooth data")
    if require_time_length(
        trajectory_data, ("qpos", "arm_qpos", "ee_pos", "gripper_cmd")
    ) != length:
        raise ValueError("Trajectory data length does not match smooth data")

    timestamps = np.asarray(smooth_data["timestamps"], dtype=np.float64)
    qpos = np.asarray(trajectory_data["qpos"], dtype=np.float32)
    if len(qpos) > 1:
        qvel = np.gradient(qpos, timestamps, axis=0).astype(np.float32)
    else:
        qvel = np.zeros_like(qpos)
    gripper = np.asarray(trajectory_data["gripper_cmd"], dtype=np.float32).reshape(-1, 1)

    payload = {
        "timestamps": timestamps,
        "obs_wrist_smooth": np.asarray(smooth_data["wrist_smooth"], dtype=np.float32),
        "obs_target_pos": np.asarray(target_data["target_pos"], dtype=np.float32),
        "obs_qpos": qpos,
        "obs_qvel": qvel,
        "obs_ee_pos": np.asarray(trajectory_data["ee_pos"], dtype=np.float32),
        "obs_gripper_state": gripper,
        "action_arm_joint_target": np.asarray(
            trajectory_data["arm_qpos"], dtype=np.float32
        ),
        "action_gripper_cmd": gripper.copy(),
    }
    for key, value in payload.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Dataset field {key} contains NaN or infinite values")
    return payload


def build_bimanual_dataset(
    smooth_data: Mapping[str, Any],
    target_data: Mapping[str, Any],
    trajectory_data: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    smooth_keys = (
        "timestamps",
        "left_wrist_smooth",
        "right_wrist_smooth",
        "left_gripper_cmd",
        "right_gripper_cmd",
    )
    length = require_time_length(smooth_data, smooth_keys)
    target_keys = (
        "left_target_pos",
        "right_target_pos",
        "left_gripper_cmd",
        "right_gripper_cmd",
    )
    if require_time_length(target_data, target_keys) != length:
        raise ValueError("Bimanual target data length does not match smooth data")
    trajectory_keys = (
        "qpos",
        "left_arm_qpos",
        "right_arm_qpos",
        "left_ee_pos",
        "right_ee_pos",
        "left_gripper_cmd",
        "right_gripper_cmd",
    )
    if require_time_length(trajectory_data, trajectory_keys) != length:
        raise ValueError("Bimanual trajectory length does not match smooth data")

    timestamps = np.asarray(smooth_data["timestamps"], dtype=np.float64)
    qpos = np.asarray(trajectory_data["qpos"], dtype=np.float32)
    qvel = (
        np.gradient(qpos, timestamps, axis=0).astype(np.float32)
        if len(qpos) > 1
        else np.zeros_like(qpos)
    )
    left_gripper = np.asarray(
        trajectory_data["left_gripper_cmd"], dtype=np.float32
    ).reshape(-1, 1)
    right_gripper = np.asarray(
        trajectory_data["right_gripper_cmd"], dtype=np.float32
    ).reshape(-1, 1)
    payload = {
        "timestamps": timestamps,
        "obs_left_wrist_smooth": np.asarray(
            smooth_data["left_wrist_smooth"], dtype=np.float32
        ),
        "obs_right_wrist_smooth": np.asarray(
            smooth_data["right_wrist_smooth"], dtype=np.float32
        ),
        "obs_left_target_pos": np.asarray(
            target_data["left_target_pos"], dtype=np.float32
        ),
        "obs_right_target_pos": np.asarray(
            target_data["right_target_pos"], dtype=np.float32
        ),
        "obs_qpos": qpos,
        "obs_qvel": qvel,
        "obs_left_ee_pos": np.asarray(
            trajectory_data["left_ee_pos"], dtype=np.float32
        ),
        "obs_right_ee_pos": np.asarray(
            trajectory_data["right_ee_pos"], dtype=np.float32
        ),
        "obs_left_gripper_state": left_gripper,
        "obs_right_gripper_state": right_gripper,
        "action_left_arm_joint_target": np.asarray(
            trajectory_data["left_arm_qpos"], dtype=np.float32
        ),
        "action_right_arm_joint_target": np.asarray(
            trajectory_data["right_arm_qpos"], dtype=np.float32
        ),
        "action_left_gripper_cmd": left_gripper.copy(),
        "action_right_gripper_cmd": right_gripper.copy(),
    }
    for key, value in payload.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Dataset field {key} contains NaN or infinite values")
    return payload
