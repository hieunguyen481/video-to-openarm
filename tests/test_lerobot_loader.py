from __future__ import annotations

import numpy as np

from openarm_retarget.lerobot_loader import (
    convert_egoworld_to_pose,
    convert_egoworld_world_frame_to_targets,
)


def _mock_egoworld_state(num_frames: int = 5) -> dict[str, np.ndarray]:
    """Create a mock 40-D state array simulating EgoWorld format."""
    state = np.zeros((num_frames, 40), dtype=np.float64)
    # Left hand: 0:20. Wrist=0:3, thumb=3:6, index=6:9... proxy=18:20
    # Right hand: 20:40.
    
    # Left wrist at (0.2, 0.3, 0.4), right wrist at (0.5, 0.6, 0.7)
    state[:, 0:3] = [0.2, 0.3, 0.4]
    state[:, 20:23] = [0.5, 0.6, 0.7]

    # Index tips
    state[:, 6:9] = [0.2, 0.3, 0.5]  # left index
    state[:, 26:29] = [0.5, 0.6, 0.8]  # right index

    # Gripper proxy: (dist, openness). Openness < 0.5 -> closed
    state[:, 18:20] = [0.05, 1.0]  # left open
    state[2:, 18:20] = [0.01, 0.0]  # left closes after frame 2

    state[:, 38:40] = [0.02, 0.0]  # right closed
    state[3:, 38:40] = [0.08, 1.0]  # right opens after frame 3

    return {"state": state}


def test_convert_egoworld_to_pose_extracts_and_normalizes():
    raw = _mock_egoworld_state(num_frames=5)
    
    pose = convert_egoworld_to_pose(
        raw, fps=30.0, normalize=True, workspace_scale=0.5
    )

    assert len(pose["timestamps"]) == 5
    np.testing.assert_allclose(pose["timestamps"], np.arange(5) / 30.0)
    
    # Test valid masks
    assert np.all(pose["left_valid"])
    assert np.all(pose["right_valid"])

    # Test normalization. 
    # Left wrist world: (0.2, 0.3, 0.4)
    # Right wrist world: (0.5, 0.6, 0.7)
    # Mean (center): (0.35, 0.45, 0.55)
    # Left centered: (-0.15, -0.15, -0.15)
    # Scaled (scale=0.5): (-0.3, -0.3, -0.3)
    # Normalized: x=0.5 - (-0.3) = 0.8, y=0.5 - (-0.3) = 0.8, z = -0.3
    np.testing.assert_allclose(pose["left_wrist"][0], [0.8, 0.8, -0.3], atol=1e-6)

    # Right centered: (0.15, 0.15, 0.15)
    # Scaled: (0.3, 0.3, 0.3)
    # Normalized: x=0.5 - 0.3 = 0.2, y=0.5 - 0.3 = 0.2, z = 0.3
    np.testing.assert_allclose(pose["right_wrist"][0], [0.2, 0.2, 0.3], atol=1e-6)


def test_convert_egoworld_to_pose_keeps_world_coordinates_if_not_normalized():
    raw = _mock_egoworld_state(num_frames=2)
    
    pose = convert_egoworld_to_pose(
        raw, fps=30.0, normalize=False
    )

    np.testing.assert_allclose(pose["left_wrist"][0], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(pose["right_wrist"][0], [0.5, 0.6, 0.7])


def test_convert_egoworld_world_frame_to_targets():
    raw = _mock_egoworld_state(num_frames=5)

    targets = convert_egoworld_world_frame_to_targets(raw, fps=30.0)

    np.testing.assert_allclose(targets["left_target_pos"][0], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(targets["right_target_pos"][0], [0.5, 0.6, 0.7])

    # Left gripper: open (1.0) then closed (0.0)
    # Gripper command = 1.0 if closed, 0.0 if open
    np.testing.assert_allclose(targets["left_gripper_cmd"], [0, 0, 1, 1, 1])

    # Right gripper: closed (0.0) then open (1.0)
    np.testing.assert_allclose(targets["right_gripper_cmd"], [1, 1, 1, 0, 0])
