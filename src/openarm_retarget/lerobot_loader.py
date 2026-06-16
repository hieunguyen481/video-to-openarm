"""Download and convert LeRobot EgoWorld dataset for the OpenArm pipeline.

EgoWorld (haoyang-li/EgoWorld) provides ego-centric bimanual manipulation
data with world-frame 3D hand poses (MANO), camera tracking, depth, and
gripper proxy signals — all in LeRobot v3.0 format (Parquet).

This module converts EgoWorld episodes into the dict format expected by
``run_bimanual_from_pose()`` so they can flow through the existing
smoothing → retarget → IK → replay pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# EgoWorld column names (state: 40-D world-frame bimanual hand poses)
# Layout per hand (20 dims): wrist(3) + thumb_tip(3) + index_tip(3)
#   + middle_tip(3) + ring_tip(3) + pinky_tip(3) + gripper_proxy(2)
# Left hand occupies state[0:20], right hand state[20:40].
# ---------------------------------------------------------------------------

KEYPOINT_NAMES = ("wrist", "thumb_tip", "index_tip", "middle_tip",
                  "ring_tip", "pinky_tip")

# Indices into the 20-D per-hand state vector
_KEYPOINT_SLICES: dict[str, slice] = {
    "wrist": slice(0, 3),
    "thumb_tip": slice(3, 6),
    "index_tip": slice(6, 9),
    "middle_tip": slice(9, 12),
    "ring_tip": slice(12, 15),
    "pinky_tip": slice(15, 18),
}
_GRIPPER_PROXY_SLICE = slice(18, 20)  # (thumb-index dist, openness)

LEFT_STATE_OFFSET = 0
RIGHT_STATE_OFFSET = 20


def _try_import_huggingface():
    """Import huggingface_hub lazily; raise helpful message if missing."""
    try:
        import huggingface_hub  # noqa: F401
        return huggingface_hub
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to download EgoWorld data. "
            "Install with: pip install huggingface-hub>=0.20"
        ) from exc


def _try_import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq  # noqa: F401
        return pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read EgoWorld parquet files. "
            "Install with: pip install pyarrow>=14.0"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_available_episodes(
    repo_id: str = "haoyang-li/EgoWorld",
    *,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List available episodes in the EgoWorld dataset.

    Returns a list of dicts with ``episode_index``, ``num_frames``, etc.
    This reads only the meta/episodes.jsonl file from the Hub.
    """
    hf = _try_import_huggingface()
    try:
        episodes_path = hf.hf_hub_download(
            repo_id=repo_id,
            filename="meta/episodes.jsonl",
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception:
        # Fallback: try info.json for older format
        try:
            info_path = hf.hf_hub_download(
                repo_id=repo_id,
                filename="meta/info.json",
                repo_type="dataset",
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            total = info.get("total_episodes", 0)
            return [{"episode_index": i} for i in range(total)]
        except Exception as exc2:
            raise RuntimeError(
                f"Cannot list episodes from {repo_id}. "
                "Make sure you have accepted the access conditions on "
                "https://huggingface.co/datasets/haoyang-li/EgoWorld"
            ) from exc2

    episodes = []
    with open(episodes_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def download_episode_parquet(
    episode_index: int,
    repo_id: str = "haoyang-li/EgoWorld",
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download a single episode's parquet chunk from EgoWorld.

    Returns the local path to the downloaded parquet file.
    """
    hf = _try_import_huggingface()
    # LeRobot v3 format: data/chunk-000/episode_NNNNNN.parquet
    chunk_index = episode_index // 1000
    filename = (
        f"data/chunk-{chunk_index:03d}/"
        f"episode_{episode_index:06d}.parquet"
    )
    try:
        local_path = hf.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception:
        # Fallback: try single-chunk format
        filename_alt = f"data/chunk-000/episode_{episode_index:06d}.parquet"
        try:
            local_path = hf.hf_hub_download(
                repo_id=repo_id,
                filename=filename_alt,
                repo_type="dataset",
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot download episode {episode_index} from {repo_id}. "
                f"Tried: {filename} and {filename_alt}. "
                "Make sure you have accepted the access conditions."
            ) from exc
    return Path(local_path)


def read_episode_parquet(parquet_path: str | Path) -> dict[str, np.ndarray]:
    """Read a parquet file and return raw columns as numpy arrays."""
    pq = _try_import_pyarrow_parquet()
    table = pq.read_table(str(parquet_path))
    result: dict[str, np.ndarray] = {}
    for col_name in table.column_names:
        column = table.column(col_name)
        try:
            result[col_name] = column.to_numpy()
        except Exception:
            # Some columns may be nested — try converting to Python list
            result[col_name] = np.asarray(column.to_pylist())
    return result


def _extract_hand_keypoints(
    state_array: np.ndarray,
    side: str,
) -> dict[str, np.ndarray]:
    """Extract wrist + fingertip positions from the 40-D state vector.

    Parameters
    ----------
    state_array : (T, 40) float
        World-frame bimanual hand poses.
    side : 'left' or 'right'

    Returns
    -------
    dict with keys like ``wrist``, ``thumb_tip``, ..., ``gripper_proxy``.
    """
    offset = LEFT_STATE_OFFSET if side == "left" else RIGHT_STATE_OFFSET
    result: dict[str, np.ndarray] = {}
    for name, sl in _KEYPOINT_SLICES.items():
        start = offset + sl.start
        stop = offset + sl.stop
        result[name] = state_array[:, start:stop].astype(np.float32)
    # Gripper proxy
    gp_start = offset + _GRIPPER_PROXY_SLICE.start
    gp_stop = offset + _GRIPPER_PROXY_SLICE.stop
    result["gripper_proxy"] = state_array[:, gp_start:gp_stop].astype(
        np.float32
    )
    return result


def _world_to_normalized(
    positions: np.ndarray,
    *,
    workspace_center: np.ndarray | None = None,
    workspace_scale: float = 1.0,
) -> np.ndarray:
    """Convert world-frame positions (meters) to normalized [0,1] coords.

    The pipeline expects normalized image-like coordinates where:
    - x ∈ [0, 1] : horizontal (left to right)
    - y ∈ [0, 1] : vertical (top to bottom)
    - z : depth proxy (palm_scale or similar)

    For ego-centric EgoWorld data the world axes are:
    - World X → forward (away from person)
    - World Y → left (person's left)
    - World Z → up

    We map: image_x ← -world_Y (left→right), image_y ← -world_Z (up→down),
             depth  ← world_X (forward).
    """
    if workspace_center is None:
        workspace_center = np.mean(positions, axis=0)
    centered = positions - workspace_center
    scaled = centered / max(workspace_scale, 1e-6)
    # Map to [0, 1] range centered at 0.5
    normalized = np.zeros_like(scaled)
    normalized[:, 0] = 0.5 - scaled[:, 1]  # -Y → image X (left to right)
    normalized[:, 1] = 0.5 - scaled[:, 2]  # -Z → image Y (up to down)
    normalized[:, 2] = scaled[:, 0]  # X → depth
    return normalized.astype(np.float32)


def _compute_palm_scale(
    wrist: np.ndarray,
    index_tip: np.ndarray,
    middle_tip: np.ndarray,
    ring_tip: np.ndarray,
    pinky_tip: np.ndarray,
) -> np.ndarray:
    """Compute palm_scale as average distance from wrist to MCP-like points."""
    distances = np.mean([
        np.linalg.norm(tip - wrist, axis=1)
        for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
    ], axis=0)
    return distances.astype(np.float32)


def convert_egoworld_to_pose(
    raw: dict[str, np.ndarray],
    *,
    fps: float = 30.0,
    normalize: bool = True,
    workspace_scale: float = 0.5,
) -> dict[str, np.ndarray]:
    """Convert raw EgoWorld episode data to bimanual pose format.

    The output dict matches the format expected by
    ``run_bimanual_from_pose()``: timestamps, left_valid, left_wrist, ...,
    right_valid, right_wrist, ..., left_palm_scale, right_palm_scale.

    Parameters
    ----------
    raw : dict
        Raw columns from ``read_episode_parquet()``.
    fps : float
        Frames per second (EgoWorld default is 30).
    normalize : bool
        If True, convert world-frame meters to normalized [0,1] coords.
        If False, pass through world-frame positions directly.
    workspace_scale : float
        Scale factor for normalization (meters per unit).
    """
    # Find the state column
    state_key = None
    for candidate in ("observation.state", "state", "observation.hand_state"):
        if candidate in raw:
            state_key = candidate
            break
    if state_key is None:
        raise KeyError(
            f"Cannot find state column in EgoWorld data. "
            f"Available columns: {list(raw.keys())}"
        )
    state = np.asarray(raw[state_key], dtype=np.float64)
    if state.ndim == 1:
        # Might be a list of lists
        state = np.stack(state)
    num_frames = state.shape[0]
    if state.shape[1] < 40:
        raise ValueError(
            f"Expected state dimension >= 40, got {state.shape[1]}. "
            "EgoWorld state should be 40-D (20 per hand)."
        )

    timestamps = np.arange(num_frames, dtype=np.float64) / fps
    data: dict[str, np.ndarray] = {"timestamps": timestamps}

    # Compute workspace center from both wrists
    left_kp = _extract_hand_keypoints(state, "left")
    right_kp = _extract_hand_keypoints(state, "right")

    all_wrists = np.concatenate([left_kp["wrist"], right_kp["wrist"]], axis=0)
    workspace_center = np.mean(all_wrists, axis=0)

    for side, kp in (("left", left_kp), ("right", right_kp)):
        valid = np.all(np.isfinite(kp["wrist"]), axis=1)
        data[f"{side}_valid"] = valid.astype(bool)

        if normalize:
            for name in KEYPOINT_NAMES:
                data[f"{side}_{name}"] = _world_to_normalized(
                    kp[name],
                    workspace_center=workspace_center,
                    workspace_scale=workspace_scale,
                )
            # Palm scale from normalized coords
            data[f"{side}_palm_scale"] = _compute_palm_scale(
                data[f"{side}_wrist"],
                data[f"{side}_index_tip"],
                data[f"{side}_middle_tip"],
                data[f"{side}_ring_tip"],
                data[f"{side}_pinky_tip"],
            )
        else:
            for name in KEYPOINT_NAMES:
                data[f"{side}_{name}"] = kp[name]
            data[f"{side}_palm_scale"] = _compute_palm_scale(
                kp["wrist"], kp["index_tip"],
                kp["middle_tip"], kp["ring_tip"], kp["pinky_tip"],
            )

        # Set NaN for invalid frames
        for name in KEYPOINT_NAMES:
            data[f"{side}_{name}"][~valid] = np.nan
        data[f"{side}_palm_scale"][~valid] = np.nan

    return data


def convert_egoworld_world_frame_to_targets(
    raw: dict[str, np.ndarray],
    *,
    fps: float = 30.0,
) -> dict[str, np.ndarray]:
    """Convert EgoWorld world-frame data directly to robot target positions.

    This bypasses the normal retarget step since EgoWorld already provides
    world-frame 3D coordinates in meters. The coordinates are directly
    mapped to the OpenArm workspace.

    Returns a dict suitable for ``BimanualJacobianIKSolver.solve()``.
    """
    state_key = None
    for candidate in ("observation.state", "state", "observation.hand_state"):
        if candidate in raw:
            state_key = candidate
            break
    if state_key is None:
        raise KeyError(f"Cannot find state column. Available: {list(raw.keys())}")

    state = np.asarray(raw[state_key], dtype=np.float64)
    if state.ndim == 1:
        state = np.stack(state)
    num_frames = state.shape[0]

    left_kp = _extract_hand_keypoints(state, "left")
    right_kp = _extract_hand_keypoints(state, "right")

    # Gripper commands from gripper_proxy
    left_gripper = (left_kp["gripper_proxy"][:, 1] < 0.5).astype(np.float32)
    right_gripper = (right_kp["gripper_proxy"][:, 1] < 0.5).astype(np.float32)

    return {
        "timestamps": np.arange(num_frames, dtype=np.float64) / fps,
        "left_wrist_world": left_kp["wrist"],
        "right_wrist_world": right_kp["wrist"],
        "left_target_pos": left_kp["wrist"],
        "right_target_pos": right_kp["wrist"],
        "left_gripper_cmd": left_gripper,
        "right_gripper_cmd": right_gripper,
    }


def download_and_convert_episode(
    episode_index: int,
    *,
    repo_id: str = "haoyang-li/EgoWorld",
    output_dir: str | Path = "data/egoworld",
    cache_dir: str | Path | None = None,
    fps: float = 30.0,
    normalize: bool = True,
) -> tuple[Path, dict[str, np.ndarray]]:
    """Download one EgoWorld episode and convert to pipeline format.

    Returns (saved_path, pose_data).
    """
    parquet_path = download_episode_parquet(
        episode_index, repo_id, cache_dir=cache_dir,
    )
    raw = read_episode_parquet(parquet_path)
    pose = convert_egoworld_to_pose(raw, fps=fps, normalize=normalize)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f"egoworld_ep{episode_index:04d}_pose.npz"
    np.savez_compressed(str(save_path), **pose)
    return save_path, pose
