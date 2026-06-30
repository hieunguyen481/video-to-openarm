from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


LANDMARK_INDICES = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}
PALM_MCP_INDICES = (5, 9, 13, 17)
HAWOR_HAND_INDEX = {"left": 0, "right": 1}


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def load_hawor_world_result(path: str | Path) -> tuple[np.ndarray, ...]:
    """Load HaWoR's joblib-backed ``world_space_res.pth`` result."""
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "Reading HaWoR world_space_res.pth requires joblib in this "
            "environment. Run conversion in the HaWoR Docker image or install "
            "the project with the 'all' extra."
        ) from exc

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"HaWoR result not found: {source}")
    data = joblib.load(source)
    if not isinstance(data, (list, tuple)) or len(data) != 5:
        raise ValueError(
            "Expected HaWoR result to contain "
            "[pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid]."
        )
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = data
    return (
        _to_numpy(pred_trans).astype(np.float32),
        _to_numpy(pred_rot).astype(np.float32),
        _to_numpy(pred_hand_pose).astype(np.float32),
        _to_numpy(pred_betas).astype(np.float32),
        np.asarray(pred_valid, dtype=bool),
    )


@contextlib.contextmanager
def _hawor_import_context(hawor_root: Path):
    previous_cwd = Path.cwd()
    root = hawor_root.resolve()
    sys.path.insert(0, str(root))
    try:
        os.chdir(root)
        yield
    finally:
        os.chdir(previous_cwd)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(root))


def decode_hawor_mano_joints(
    pred_trans: np.ndarray,
    pred_rot: np.ndarray,
    pred_hand_pose: np.ndarray,
    pred_betas: np.ndarray,
    *,
    hawor_root: str | Path,
    use_cuda: bool = False,
) -> np.ndarray:
    """Decode HaWoR MANO parameters into [left/right, frame, 21, xyz] joints."""
    root = Path(hawor_root)
    with _hawor_import_context(root):
        from hawor.utils.process import run_mano, run_mano_left

        import torch

        tensors = [
            torch.from_numpy(value)
            for value in (pred_trans, pred_rot, pred_hand_pose, pred_betas)
        ]
        trans, rot, pose, betas = tensors
        left = run_mano_left(
            trans[0:1],
            rot[0:1],
            pose[0:1],
            betas=betas[0:1],
            use_cuda=use_cuda,
        )["joints"][0]
        right = run_mano(
            trans[1:2],
            rot[1:2],
            pose[1:2],
            betas=betas[1:2],
            use_cuda=use_cuda,
        )["joints"][0]
    return np.stack([_to_numpy(left), _to_numpy(right)], axis=0).astype(np.float32)


def _proxy_joints_from_wrist(wrist: np.ndarray) -> np.ndarray:
    frames = wrist.shape[0]
    joints = np.repeat(wrist[:, None, :], 21, axis=1)
    offsets = {
        4: np.array([0.035, -0.025, -0.010], dtype=np.float32),
        8: np.array([0.095, -0.055, -0.005], dtype=np.float32),
        12: np.array([0.070, -0.090, 0.000], dtype=np.float32),
        16: np.array([0.035, -0.095, 0.005], dtype=np.float32),
        20: np.array([0.005, -0.080, 0.010], dtype=np.float32),
        5: np.array([0.045, -0.030, -0.004], dtype=np.float32),
        9: np.array([0.035, -0.050, 0.000], dtype=np.float32),
        13: np.array([0.020, -0.055, 0.003], dtype=np.float32),
        17: np.array([0.000, -0.045, 0.006], dtype=np.float32),
    }
    for index, offset in offsets.items():
        joints[:, index, :] = wrist + offset[None, :]
    if frames:
        joints[:, :, 0] += np.linspace(-0.01, 0.01, frames, dtype=np.float32)[:, None]
    return joints.astype(np.float32)


def _apply_hawor_world_axis(points: np.ndarray) -> np.ndarray:
    # HaWoR's own demo applies this transform before world visualization.
    transform = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    return np.einsum("ij,...j->...i", transform, points)


def _normalize_points(
    points: np.ndarray,
    valid: np.ndarray,
    *,
    workspace_scale: float,
) -> np.ndarray:
    if workspace_scale <= 0:
        raise ValueError("workspace_scale must be positive")
    valid_mask = np.broadcast_to(valid[:, :, None, None], points.shape)
    valid_points = points[valid_mask].reshape(-1, 3)
    if valid_points.size == 0:
        center = np.zeros(3, dtype=np.float32)
    else:
        center = np.nanmean(valid_points.reshape(-1, 3), axis=0).astype(np.float32)
    normalized = (points - center[None, None, None, :]) / workspace_scale + 0.5
    return normalized.astype(np.float32)


def _palm_scale(joints: np.ndarray) -> np.ndarray:
    wrist = joints[:, 0, :]
    distances = [
        np.linalg.norm(joints[:, index, :] - wrist, axis=1)
        for index in PALM_MCP_INDICES
    ]
    return np.mean(np.stack(distances, axis=1), axis=1).astype(np.float32)


def convert_hawor_world_to_bimanual_pose(
    path: str | Path,
    *,
    fps: float = 30.0,
    hawor_root: str | Path | None = None,
    use_mano: bool = True,
    use_cuda: bool = False,
    apply_world_axis: bool = True,
    normalize: bool = True,
    workspace_scale: float = 0.5,
) -> dict[str, np.ndarray]:
    """Convert HaWoR world-space output to this project's bimanual pose schema."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = (
        load_hawor_world_result(path)
    )
    if pred_trans.shape[:2] != pred_valid.shape or pred_trans.shape[-1] != 3:
        raise ValueError(
            "Expected pred_trans shape [2, T, 3] and pred_valid shape [2, T]."
        )
    if pred_trans.shape[0] != 2:
        raise ValueError(f"Expected two hands, got shape {pred_trans.shape}")

    if use_mano and hawor_root is not None:
        joints = decode_hawor_mano_joints(
            pred_trans,
            pred_rot,
            pred_hand_pose,
            pred_betas,
            hawor_root=hawor_root,
            use_cuda=use_cuda,
        )
    else:
        joints = np.stack(
            [_proxy_joints_from_wrist(pred_trans[index]) for index in range(2)],
            axis=0,
        )

    if apply_world_axis:
        joints = _apply_hawor_world_axis(joints)

    valid = np.asarray(pred_valid, dtype=bool)
    if normalize:
        joints_out = _normalize_points(
            joints,
            valid,
            workspace_scale=workspace_scale,
        )
    else:
        joints_out = joints.astype(np.float32)

    frames = joints_out.shape[1]
    result: dict[str, np.ndarray] = {
        "timestamps": np.arange(frames, dtype=np.float64) / fps
    }
    for side in ("left", "right"):
        hand_index = HAWOR_HAND_INDEX[side]
        side_valid = valid[hand_index]
        result[f"{side}_valid"] = side_valid
        for name, joint_index in LANDMARK_INDICES.items():
            values = joints_out[hand_index, :, joint_index, :].astype(np.float32)
            values = values.copy()
            values[~side_valid] = np.nan
            result[f"{side}_{name}"] = values
        palm = _palm_scale(joints_out[hand_index])
        palm[~side_valid] = np.nan
        # The existing pipeline uses palm_scale as a depth proxy when present.
        # For HaWoR, preserve real wrist Z motion by storing normalized wrist Z.
        if normalize:
            wrist_z = result[f"{side}_wrist"][:, 2].astype(np.float32)
            result[f"{side}_palm_scale"] = wrist_z
        else:
            result[f"{side}_palm_scale"] = palm
    return result
