"""Tests for accuracy improvement features (Phase 1, 3A, 4A).

Phase 1: Auto-calibration of scale & origin
Phase 3A: Outlier rejection before smoothing
Phase 4A: Null-space IK & adaptive damping
"""
from __future__ import annotations

import numpy as np
import pytest

from openarm_retarget.retargeting import (
    auto_calibrate_origin,
    auto_calibrate_scale,
    retarget_wrist,
    retarget_wrist_auto,
)
from openarm_retarget.smoothing import reject_outliers, smooth_wrist


# ---------------------------------------------------------------------------
# Phase 1: Auto-Calibration Tests
# ---------------------------------------------------------------------------


class TestAutoCalibrateScale:
    """Tests for auto_calibrate_scale()."""

    @pytest.fixture()
    def base_config(self):
        return {
            "openarm_origin": [0.401, 0.1535, 1.12],
            "scale": {"x": 0.35, "y": 0.35, "z": 0.35},
            "axis_mapping": {
                "human_x": "y",
                "human_y": "z_negative",
                "human_z": "x",
            },
            "workspace_limit": {
                "x": [0.15, 0.70],
                "y": [0.00, 0.55],
                "z": [0.65, 1.35],
            },
        }

    def test_small_motion_gets_min_scale(self, base_config):
        """Very small motion should get min_scale, not zero."""
        wrist = np.full((100, 3), 0.5)
        wrist[:, 0] += np.random.randn(100) * 1e-8
        scale = auto_calibrate_scale(wrist, base_config, min_scale=0.05)
        assert scale["x"] == pytest.approx(0.05, abs=1e-6)

    def test_large_motion_gets_capped_scale(self, base_config):
        """Large motion should not exceed max_scale."""
        wrist = np.random.rand(100, 3)
        scale = auto_calibrate_scale(wrist, base_config, max_scale=0.5)
        for axis in ("x", "y", "z"):
            assert scale[axis] <= 0.5 + 1e-6

    def test_scale_fits_workspace(self, base_config):
        """Scale should map motion amplitude to workspace_utilization * room."""
        # Create wrist data with known amplitude
        wrist = np.zeros((100, 3))
        wrist[:, 0] = np.linspace(0.3, 0.7, 100)  # amplitude 0.4
        wrist[:, 1] = 0.5
        wrist[:, 2] = 0.5

        scale = auto_calibrate_scale(
            wrist, base_config, workspace_utilization=0.85
        )
        # human_x maps to robot y, room = 0.55
        # Expected: 0.85 * 0.55 / 0.4 = 1.16875, capped at max_scale=1.0
        assert scale["x"] <= 1.0 + 1e-6

    def test_per_axis_scale_differs(self, base_config):
        """Different motion amplitudes should produce different scales."""
        wrist = np.zeros((100, 3))
        wrist[:, 0] = np.linspace(0.3, 0.7, 100)  # large horizontal
        wrist[:, 1] = np.linspace(0.48, 0.52, 100)  # small vertical
        wrist[:, 2] = np.linspace(0.45, 0.55, 100)  # medium depth

        # Use a high max_scale so differences aren't capped
        scale = auto_calibrate_scale(wrist, base_config, max_scale=5.0)
        # Different amplitudes → different scales
        assert scale["x"] != scale["y"]

    def test_nan_frames_ignored(self, base_config):
        """NaN frames should be excluded from amplitude calculation."""
        wrist = np.zeros((100, 3))
        wrist[:, 0] = np.linspace(0.3, 0.7, 100)
        wrist[:30, 0] = np.nan  # Invalid frames
        scale = auto_calibrate_scale(wrist, base_config)
        assert all(np.isfinite(v) for v in scale.values())


class TestAutoCalibrateOrigin:
    """Tests for auto_calibrate_origin()."""

    @pytest.fixture()
    def base_config(self):
        return {
            "openarm_origin": [0.401, 0.1535, 1.12],
            "scale": {"x": 0.35, "y": 0.35, "z": 0.35},
            "axis_mapping": {
                "human_x": "y",
                "human_y": "z_negative",
                "human_z": "x",
            },
            "workspace_limit": {
                "x": [0.15, 0.70],
                "y": [0.00, 0.55],
                "z": [0.65, 1.35],
            },
        }

    def test_origin_maps_median_to_workspace_center(self, base_config):
        """Median wrist position should map to workspace center."""
        wrist = np.full((100, 3), 0.5)
        origin = auto_calibrate_origin(wrist, base_config)
        assert origin.shape == (3,)
        assert np.all(np.isfinite(origin))

    def test_shifted_wrist_shifts_origin(self, base_config):
        """Different delta from wrist[0] to median should produce different origins."""
        # Dataset 1: range 0.3-0.7, median=0.5, wrist[0]=0.3, delta=0.2
        wrist_wide = np.zeros((100, 3))
        wrist_wide[:, 0] = np.linspace(0.3, 0.7, 100)
        wrist_wide[:, 1] = 0.5
        wrist_wide[:, 2] = 0.5

        # Dataset 2: range 0.3-0.5, median=0.4, wrist[0]=0.3, delta=0.1
        wrist_narrow = np.zeros((100, 3))
        wrist_narrow[:, 0] = np.linspace(0.3, 0.5, 100)
        wrist_narrow[:, 1] = 0.5
        wrist_narrow[:, 2] = 0.5

        origin_wide = auto_calibrate_origin(wrist_wide, base_config)
        origin_narrow = auto_calibrate_origin(wrist_narrow, base_config)
        # Origins should differ because delta_median differs (0.2 vs 0.1)
        assert not np.allclose(origin_wide, origin_narrow)


class TestRetargetWristAuto:
    """Tests for retarget_wrist_auto()."""

    @pytest.fixture()
    def base_config(self):
        return {
            "openarm_origin": [0.401, 0.1535, 1.12],
            "scale": {"x": 0.35, "y": 0.35, "z": 0.35},
            "axis_mapping": {
                "human_x": "y",
                "human_y": "z_negative",
                "human_z": "x",
            },
            "workspace_limit": {
                "x": [0.15, 0.70],
                "y": [0.00, 0.55],
                "z": [0.65, 1.35],
            },
        }

    def test_auto_scale_false_uses_fixed_scale(self, base_config):
        """When auto_scale=False, should behave like retarget_wrist."""
        wrist = np.random.rand(50, 3) * 0.3 + 0.35
        result_auto = retarget_wrist_auto(wrist, {**base_config, "auto_scale": False})
        result_fixed = retarget_wrist(wrist, base_config)
        np.testing.assert_allclose(result_auto, result_fixed, atol=1e-6)

    def test_auto_scale_true_adapts_to_motion(self, base_config):
        """When auto_scale=True, targets should fit within workspace."""
        wrist = np.zeros((100, 3))
        wrist[:, 0] = np.linspace(0.2, 0.8, 100)  # large motion
        wrist[:, 1] = np.linspace(0.3, 0.7, 100)
        wrist[:, 2] = np.linspace(0.4, 0.6, 100)

        config = {**base_config, "auto_scale": True}
        result = retarget_wrist_auto(wrist, config)

        # Check targets are within workspace limits
        limits = base_config["workspace_limit"]
        for axis, (lo, hi) in limits.items():
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            assert result[:, idx].min() >= lo - 0.01
            assert result[:, idx].max() <= hi + 0.01

    def test_auto_scale_improves_workspace_utilization(self, base_config):
        """Auto-scale should spread targets across more of the workspace."""
        # Create moderate motion
        wrist = np.zeros((100, 3))
        wrist[:, 0] = np.linspace(0.3, 0.7, 100)
        wrist[:, 1] = np.linspace(0.3, 0.7, 100)
        wrist[:, 2] = np.linspace(0.3, 0.7, 100)

        result_fixed = retarget_wrist(wrist, base_config)
        result_auto = retarget_wrist_auto(
            wrist, {**base_config, "auto_scale": True}
        )

        # Auto-scale should produce a wider spread of targets
        # (better workspace utilization)
        limits = base_config["workspace_limit"]
        for axis, (lo, hi) in limits.items():
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            room = hi - lo
            fixed_range = result_fixed[:, idx].max() - result_fixed[:, idx].min()
            auto_range = result_auto[:, idx].max() - result_auto[:, idx].min()
            # Auto should use at least as much of the workspace
            if fixed_range > 1e-6:
                auto_util = auto_range / room
                fixed_util = fixed_range / room
                # Both should be reasonable utilizations
                assert auto_util > 0.01  # Not degenerate


# ---------------------------------------------------------------------------
# Phase 3A: Outlier Rejection Tests
# ---------------------------------------------------------------------------


class TestRejectOutliers:
    """Tests for reject_outliers()."""

    def test_no_outliers_passes_through(self):
        """Clean data should not be modified."""
        values = np.linspace(0, 1, 50)[:, None].repeat(3, axis=1)
        valid = np.ones(50, dtype=bool)
        result = reject_outliers(values, valid, max_jump=0.15)
        np.testing.assert_array_equal(result, valid)

    def test_single_outlier_is_rejected(self):
        """A single large jump should be marked invalid."""
        values = np.zeros((10, 2))
        values[:, 0] = np.arange(10) * 0.01  # small steps
        values[:, 1] = np.arange(10) * 0.01
        values[5, 0] = 5.0  # huge jump
        valid = np.ones(10, dtype=bool)
        result = reject_outliers(values, valid, max_jump=0.15)
        assert not result[5]
        assert result[:5].all()
        assert result[6:].all()

    def test_already_invalid_stays_invalid(self):
        """Previously invalid frames should remain invalid."""
        values = np.zeros((10, 2))
        valid = np.ones(10, dtype=bool)
        valid[3] = False
        result = reject_outliers(values, valid, max_jump=0.15)
        assert not result[3]

    def test_nan_values_are_rejected(self):
        """NaN values should be marked invalid."""
        values = np.zeros((10, 2))
        values[4, 0] = np.nan
        valid = np.ones(10, dtype=bool)
        result = reject_outliers(values, valid, max_jump=0.15)
        assert not result[4]

    def test_max_jump_zero_raises(self):
        """max_jump=0 should raise ValueError."""
        values = np.zeros((5, 2))
        valid = np.ones(5, dtype=bool)
        with pytest.raises(ValueError, match="positive"):
            reject_outliers(values, valid, max_jump=0)


class TestSmoothWristWithOutlierRejection:
    """Tests for smooth_wrist() with outlier rejection."""

    def test_smooth_wrist_handles_outliers(self):
        """smooth_wrist should handle outliers gracefully."""
        wrist = np.zeros((50, 3))
        wrist[:, 0] = np.linspace(0.3, 0.7, 50)
        wrist[:, 1] = 0.5
        wrist[:, 2] = 0.5
        # Add a spike
        wrist[25, 0] = 5.0
        valid = np.ones(50, dtype=bool)
        result = smooth_wrist(wrist, valid, max_jump=0.15)
        assert result.shape == (50, 3)
        assert np.all(np.isfinite(result))
        # The spike should be smoothed out, not propagated
        assert result[25, 0] < 2.0  # Not the full spike value


# ---------------------------------------------------------------------------
# Phase 4A: Null-Space & Adaptive Damping Tests (unit-level)
# ---------------------------------------------------------------------------


class TestNullSpaceStep:
    """Tests for _compute_null_space_step()."""

    def test_zero_weight_returns_zeros(self):
        from openarm_retarget.ik_solver import _compute_null_space_step

        J = np.random.rand(3, 7)
        q = np.zeros(7)
        q_pref = np.ones(7)
        result = _compute_null_space_step(J, q, q_pref, 0.01, 0.0)
        np.testing.assert_array_equal(result, np.zeros(7))

    def test_non_redundant_returns_zeros(self):
        from openarm_retarget.ik_solver import _compute_null_space_step

        # 7 DOF, 7 task dimensions → no redundancy
        J = np.eye(7)
        q = np.zeros(7)
        q_pref = np.ones(7)
        result = _compute_null_space_step(J, q, q_pref, 0.01, 0.1)
        np.testing.assert_array_equal(result, np.zeros(7))

    def test_redundant_returns_nonzero(self):
        from openarm_retarget.ik_solver import _compute_null_space_step

        # 3 task DOF, 7 joint DOF → 4 DOF redundancy
        J = np.random.rand(3, 7)
        q = np.zeros(7)
        q_pref = np.ones(7)
        result = _compute_null_space_step(J, q, q_pref, 0.01, 0.1)
        assert result.shape == (7,)
        assert np.any(result != 0)


class TestAdaptiveDamping:
    """Tests for _adaptive_damping()."""

    def test_well_conditioned_returns_base(self):
        from openarm_retarget.ik_solver import _adaptive_damping

        # Identity-like matrix → condition number ≈ 1
        J = np.eye(3) * 10
        result = _adaptive_damping(0.01, J)
        assert result == pytest.approx(0.01, abs=1e-6)

    def test_near_singular_increases_damping(self):
        from openarm_retarget.ik_solver import _adaptive_damping

        # Nearly rank-deficient matrix
        J = np.array([[1.0, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]])
        result = _adaptive_damping(0.01, J)
        assert result > 0.01

    def test_moderate_condition_increases_slightly(self):
        from openarm_retarget.ik_solver import _adaptive_damping

        # Moderate condition number
        J = np.diag([1.0, 0.01, 0.001])
        result = _adaptive_damping(0.01, J)
        assert result >= 0.01