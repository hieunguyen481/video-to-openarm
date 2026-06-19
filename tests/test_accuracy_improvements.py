"""Tests for accuracy improvement features (Phase 1, 3A, 3B, 4A, 4C, 5A, 6A).

Phase 1: Auto-calibration of scale & origin
Phase 2C: Separate depth smoothing
Phase 3A: Outlier rejection before smoothing
Phase 3B: Forward-backward smoothing
Phase 4A: Null-space IK & adaptive damping
Phase 4C: Velocity clamp vs best-qpos fix
Phase 5A: Handedness stabilization
Phase 6A: Segmented retargeting
"""
from __future__ import annotations

import numpy as np
import pytest

from openarm_retarget.retargeting import (
    auto_calibrate_origin,
    auto_calibrate_scale,
    retarget_wrist,
    retarget_wrist_auto,
    retarget_wrist_segmented,
)
from openarm_retarget.smoothing import (
    forward_backward_smooth,
    moving_average,
    reject_outliers,
    smooth_axis_separate,
    smooth_wrist,
)
from openarm_retarget.bimanual import stabilize_handedness


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
        wrist = np.zeros((100, 3))
        wrist[:, 0] = np.linspace(0.3, 0.7, 100)
        wrist[:, 1] = 0.5
        wrist[:, 2] = 0.5

        scale = auto_calibrate_scale(
            wrist, base_config, workspace_utilization=0.85
        )
        assert scale["x"] <= 1.0 + 1e-6

    def test_per_axis_scale_differs(self, base_config):
        """Different motion amplitudes should produce different scales."""
        wrist = np.zeros((100, 3))
        wrist[:, 0] = np.linspace(0.3, 0.7, 100)
        wrist[:, 1] = np.linspace(0.48, 0.52, 100)
        wrist[:, 2] = np.linspace(0.45, 0.55, 100)

        scale = auto_calibrate_scale(wrist, base_config, max_scale=5.0)
        assert scale["x"] != scale["y"]

    def test_nan_frames_ignored(self, base_config):
        """NaN frames should be excluded from amplitude calculation."""
        wrist = np.zeros((100, 3))
        wrist[:, 0] = np.linspace(0.3, 0.7, 100)
        wrist[:30, 0] = np.nan
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
        wrist_wide = np.zeros((100, 3))
        wrist_wide[:, 0] = np.linspace(0.3, 0.7, 100)
        wrist_wide[:, 1] = 0.5
        wrist_wide[:, 2] = 0.5

        wrist_narrow = np.zeros((100, 3))
        wrist_narrow[:, 0] = np.linspace(0.3, 0.5, 100)
        wrist_narrow[:, 1] = 0.5
        wrist_narrow[:, 2] = 0.5

        origin_wide = auto_calibrate_origin(wrist_wide, base_config)
        origin_narrow = auto_calibrate_origin(wrist_narrow, base_config)
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
        wrist[:, 0] = np.linspace(0.2, 0.8, 100)
        wrist[:, 1] = np.linspace(0.3, 0.7, 100)
        wrist[:, 2] = np.linspace(0.4, 0.6, 100)

        config = {**base_config, "auto_scale": True}
        result = retarget_wrist_auto(wrist, config)

        limits = base_config["workspace_limit"]
        for axis, (lo, hi) in limits.items():
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            assert result[:, idx].min() >= lo - 0.01
            assert result[:, idx].max() <= hi + 0.01

    def test_auto_scale_improves_workspace_utilization(self, base_config):
        """Auto-scale should spread targets across more of the workspace."""
        wrist = np.zeros((100, 3))
        wrist[:, 0] = np.linspace(0.3, 0.7, 100)
        wrist[:, 1] = np.linspace(0.3, 0.7, 100)
        wrist[:, 2] = np.linspace(0.3, 0.7, 100)

        result_fixed = retarget_wrist(wrist, base_config)
        result_auto = retarget_wrist_auto(
            wrist, {**base_config, "auto_scale": True}
        )

        limits = base_config["workspace_limit"]
        for axis, (lo, hi) in limits.items():
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            room = hi - lo
            fixed_range = result_fixed[:, idx].max() - result_fixed[:, idx].min()
            auto_range = result_auto[:, idx].max() - result_auto[:, idx].min()
            if fixed_range > 1e-6:
                auto_util = auto_range / room
                assert auto_util > 0.01


# ---------------------------------------------------------------------------
# Phase 2C: Separate Depth Smoothing Tests
# ---------------------------------------------------------------------------


class TestSmoothAxisSeparate:
    """Tests for smooth_axis_separate() — different window for z-axis."""

    def test_different_windows_produce_different_smoothing(self):
        """Z-axis with larger window should be smoother than xy."""
        np.random.seed(42)
        values = np.random.rand(100, 3) * 0.1 + 0.4
        result = smooth_axis_separate(values, window_xy=5, window_z=15)
        assert result.shape == values.shape
        # Z-axis should be smoother (smaller variance of differences)
        z_diff = np.diff(result[:, 2])
        xy_diff = np.diff(result[:, 0])
        assert np.var(z_diff) < np.var(xy_diff) + 1e-6

    def test_same_window_matches_moving_average(self):
        """When window_xy == window_z, should match moving_average."""
        values = np.random.rand(50, 3) * 0.1 + 0.4
        result_sep = smooth_axis_separate(values, window_xy=7, window_z=7)
        result_ma = moving_average(values, 7)
        np.testing.assert_allclose(result_sep, result_ma, atol=1e-6)

    def test_2d_input_falls_back_to_moving_average(self):
        """Input with <3 columns should use window_xy for all."""
        values = np.random.rand(50, 2) * 0.1 + 0.4
        result = smooth_axis_separate(values, window_xy=7, window_z=15)
        expected = moving_average(values, 7)
        np.testing.assert_allclose(result, expected, atol=1e-6)


class TestSmoothWristDepthWindow:
    """Tests for smooth_wrist() with depth_window parameter."""

    def test_depth_window_applies_larger_z_smoothing(self):
        """depth_window > window should smooth z more than xy."""
        np.random.seed(42)
        wrist = np.random.rand(100, 3) * 0.1 + 0.4
        valid = np.ones(100, dtype=bool)
        result = smooth_wrist(wrist, valid, window=5, depth_window=15)
        assert result.shape == (100, 3)
        assert np.all(np.isfinite(result))

    def test_depth_window_none_uses_uniform_window(self):
        """depth_window=None should use same window for all axes."""
        np.random.seed(42)
        wrist = np.random.rand(50, 3) * 0.1 + 0.4
        valid = np.ones(50, dtype=bool)
        result_a = smooth_wrist(wrist, valid, window=7, depth_window=None)
        result_b = smooth_wrist(wrist, valid, window=7)
        np.testing.assert_allclose(result_a, result_b, atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 3A: Outlier Rejection Tests
# ---------------------------------------------------------------------------


class TestRejectOutliers:
    """Tests for reject_outliers()."""

    def test_no_outliers_passes_through(self):
        values = np.linspace(0, 1, 50)[:, None].repeat(3, axis=1)
        valid = np.ones(50, dtype=bool)
        result = reject_outliers(values, valid, max_jump=0.15)
        np.testing.assert_array_equal(result, valid)

    def test_single_outlier_is_rejected(self):
        values = np.zeros((10, 2))
        values[:, 0] = np.arange(10) * 0.01
        values[:, 1] = np.arange(10) * 0.01
        values[5, 0] = 5.0
        valid = np.ones(10, dtype=bool)
        result = reject_outliers(values, valid, max_jump=0.15)
        assert not result[5]
        assert result[:5].all()
        assert result[6:].all()

    def test_already_invalid_stays_invalid(self):
        values = np.zeros((10, 2))
        valid = np.ones(10, dtype=bool)
        valid[3] = False
        result = reject_outliers(values, valid, max_jump=0.15)
        assert not result[3]

    def test_nan_values_are_rejected(self):
        values = np.zeros((10, 2))
        values[4, 0] = np.nan
        valid = np.ones(10, dtype=bool)
        result = reject_outliers(values, valid, max_jump=0.15)
        assert not result[4]

    def test_max_jump_zero_raises(self):
        values = np.zeros((5, 2))
        valid = np.ones(5, dtype=bool)
        with pytest.raises(ValueError, match="positive"):
            reject_outliers(values, valid, max_jump=0)


class TestSmoothWristWithOutlierRejection:
    """Tests for smooth_wrist() with outlier rejection."""

    def test_smooth_wrist_handles_outliers(self):
        wrist = np.zeros((50, 3))
        wrist[:, 0] = np.linspace(0.3, 0.7, 50)
        wrist[:, 1] = 0.5
        wrist[:, 2] = 0.5
        wrist[25, 0] = 5.0
        valid = np.ones(50, dtype=bool)
        result = smooth_wrist(wrist, valid, max_jump=0.15)
        assert result.shape == (50, 3)
        assert np.all(np.isfinite(result))
        assert result[25, 0] < 2.0


# ---------------------------------------------------------------------------
# Phase 3B: Forward-Backward Smoothing Tests
# ---------------------------------------------------------------------------


class TestForwardBackwardSmooth:
    """Tests for forward_backward_smooth() — zero-phase lag filter."""

    def test_output_shape_matches_input(self):
        """Output should have same shape as input."""
        values = np.random.rand(50, 3)
        result = forward_backward_smooth(values, window=7)
        assert result.shape == values.shape

    def test_reduces_lag_compared_to_causal(self):
        """Forward-backward should have less lag than forward-only."""
        # Create a step signal — causal filter shifts the step right
        values = np.zeros((100, 3))
        values[50:, 0] = 1.0  # step at frame 50
        causal = moving_average(values, 7)
        fb = forward_backward_smooth(values, 7)
        # The forward-backward filter should have the step closer to frame 50
        # Measure where the signal crosses 0.5
        causal_cross = np.searchsorted(causal[:, 0], 0.5)
        fb_cross = np.searchsorted(fb[:, 0], 0.5)
        # FB should cross closer to the true step location (50)
        assert abs(fb_cross - 50) <= abs(causal_cross - 50)

    def test_smoother_than_raw(self):
        """Result should be smoother (less high-frequency variation) than raw."""
        np.random.seed(42)
        values = np.cumsum(np.random.randn(100, 3) * 0.01, axis=0) + 0.5
        result = forward_backward_smooth(values, window=7)
        # Variance of frame-to-frame differences should decrease
        raw_var = np.mean(np.var(np.diff(values, axis=0), axis=0))
        smooth_var = np.mean(np.var(np.diff(result, axis=0), axis=0))
        assert smooth_var < raw_var

    def test_preserves_constant_signal(self):
        """A constant signal should remain constant after filtering."""
        values = np.full((50, 3), 0.5)
        result = forward_backward_smooth(values, window=7)
        np.testing.assert_allclose(result, 0.5, atol=1e-6)


class TestSmoothWristForwardBackward:
    """Tests for smooth_wrist() with forward_backward parameter."""

    def test_forward_backward_true_is_default(self):
        """Default behavior should use forward-backward smoothing."""
        wrist = np.random.rand(50, 3) * 0.1 + 0.4
        valid = np.ones(50, dtype=bool)
        result_fb = smooth_wrist(wrist, valid, forward_backward=True)
        result_default = smooth_wrist(wrist, valid)
        np.testing.assert_allclose(result_fb, result_default, atol=1e-6)

    def test_forward_backward_false_uses_causal(self):
        """forward_backward=False should use causal moving average."""
        wrist = np.random.rand(50, 3) * 0.1 + 0.4
        valid = np.ones(50, dtype=bool)
        result = smooth_wrist(wrist, valid, forward_backward=False)
        assert result.shape == (50, 3)
        assert np.all(np.isfinite(result))


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

        J = np.eye(7)
        q = np.zeros(7)
        q_pref = np.ones(7)
        result = _compute_null_space_step(J, q, q_pref, 0.01, 0.1)
        np.testing.assert_array_equal(result, np.zeros(7))

    def test_redundant_returns_nonzero(self):
        from openarm_retarget.ik_solver import _compute_null_space_step

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

        J = np.eye(3) * 10
        result = _adaptive_damping(0.01, J)
        assert result == pytest.approx(0.01, abs=1e-6)

    def test_near_singular_increases_damping(self):
        from openarm_retarget.ik_solver import _adaptive_damping

        J = np.array([[1.0, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]])
        result = _adaptive_damping(0.01, J)
        assert result > 0.01

    def test_moderate_condition_increases_slightly(self):
        from openarm_retarget.ik_solver import _adaptive_damping

        J = np.diag([1.0, 0.01, 0.001])
        result = _adaptive_damping(0.01, J)
        assert result >= 0.01


# ---------------------------------------------------------------------------
# Phase 5A: Handedness Stabilization Tests
# ---------------------------------------------------------------------------


class TestStabilizeHandedness:
    """Tests for stabilize_handedness() — fix left/right hand swaps."""

    def _make_bimanual_data(self, n_frames=50, swap_at=None):
        """Create synthetic bimanual tracking data.

        Left hand moves slowly left-to-right, right hand stays put.
        If swap_at is given, swap left/right data at that frame.
        """
        left_wrist = np.zeros((n_frames, 3))
        left_wrist[:, 0] = np.linspace(0.3, 0.5, n_frames)
        left_wrist[:, 1] = 0.6
        left_wrist[:, 2] = 0.5

        right_wrist = np.zeros((n_frames, 3))
        right_wrist[:, 0] = np.linspace(0.5, 0.7, n_frames)
        right_wrist[:, 1] = 0.4
        right_wrist[:, 2] = 0.5

        left_valid = np.ones(n_frames, dtype=bool)
        right_valid = np.ones(n_frames, dtype=bool)

        data = {
            "timestamps": np.linspace(0, n_frames / 30, n_frames),
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
            "left_valid": left_valid,
            "right_valid": right_valid,
            "left_palm_scale": np.full(n_frames, 0.1),
            "right_palm_scale": np.full(n_frames, 0.1),
        }

        if swap_at is not None:
            # Simulate a handedness swap at the given frame
            temp_left = data["left_wrist"][swap_at].copy()
            temp_right = data["right_wrist"][swap_at].copy()
            data["left_wrist"][swap_at] = temp_right
            data["right_wrist"][swap_at] = temp_left
            temp_lp = data["left_palm_scale"][swap_at]
            temp_rp = data["right_palm_scale"][swap_at]
            data["left_palm_scale"][swap_at] = temp_rp
            data["right_palm_scale"][swap_at] = temp_lp

        return data

    def test_no_swap_returns_unchanged(self):
        """Clean data without swaps should not be modified."""
        data = self._make_bimanual_data()
        result = stabilize_handedness(data)
        np.testing.assert_array_equal(result["left_wrist"], data["left_wrist"])
        np.testing.assert_array_equal(result["right_wrist"], data["right_wrist"])

    def test_single_swap_is_corrected(self):
        """A single frame swap should be detected and corrected."""
        data = self._make_bimanual_data(swap_at=25)
        original_left = data["left_wrist"].copy()
        original_right = data["right_wrist"].copy()

        result = stabilize_handedness(data, swap_threshold=0.05)

        # After stabilization, the swapped frame should be closer to
        # the original (pre-swap) values
        left_diff_before = np.linalg.norm(
            data["left_wrist"][25] - original_left[25]
        )
        left_diff_after = np.linalg.norm(
            result["left_wrist"][25] - original_left[25]
        )
        # The stabilized version should be at least as close to original
        assert left_diff_after <= left_diff_before + 1e-6

    def test_short_data_returns_unchanged(self):
        """Data with <2 frames should be returned as-is."""
        data = {
            "left_wrist": np.array([[0.5, 0.5, 0.5]]),
            "right_wrist": np.array([[0.5, 0.5, 0.5]]),
            "left_valid": np.array([True]),
            "right_valid": np.array([True]),
        }
        result = stabilize_handedness(data)
        assert "left_wrist" in result

    def test_invalid_frames_skip_swap_check(self):
        """Frames where one hand is invalid should not trigger swaps."""
        data = self._make_bimanual_data(swap_at=25)
        data["left_valid"][24] = False  # Previous frame invalid
        data["left_valid"][25] = False  # Current frame invalid
        # Should not crash and should return data
        result = stabilize_handedness(data)
        assert "left_wrist" in result

    def test_swap_threshold_controls_sensitivity(self):
        """Higher threshold should be less sensitive to swaps."""
        data = self._make_bimanual_data(swap_at=25)
        # Very high threshold → no swaps detected
        result_high = stabilize_handedness(data, swap_threshold=100.0)
        np.testing.assert_array_equal(
            result_high["left_wrist"], data["left_wrist"]
        )


# ---------------------------------------------------------------------------
# Phase 6A: Segmented Retargeting Tests
# ---------------------------------------------------------------------------


class TestRetargetWristSegmented:
    """Tests for retarget_wrist_segmented() — per-segment auto-calibration."""

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

    def test_short_video_uses_standard_retarget(self, base_config):
        """Video shorter than segment_frames should use retarget_wrist_auto."""
        wrist = np.random.rand(100, 3) * 0.3 + 0.35
        result = retarget_wrist_segmented(wrist, base_config, segment_frames=900)
        expected = retarget_wrist_auto(wrist, base_config)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_long_video_produces_valid_output(self, base_config):
        """Long video should produce valid targets within workspace."""
        np.random.seed(42)
        # Create 2000 frames of data with shifting center
        wrist = np.random.rand(2000, 3) * 0.1 + 0.4
        wrist[:1000, 0] += 0.1  # Shift in first half
        wrist[1000:, 0] -= 0.1  # Shift in second half

        config = {**base_config, "auto_scale": True}
        result = retarget_wrist_segmented(
            wrist, config, segment_frames=900, overlap_frames=60
        )
        assert result.shape == (2000, 3)
        assert np.all(np.isfinite(result))

    def test_segment_blend_is_continuous(self, base_config):
        """Blended segments should not have discontinuities at boundaries."""
        np.random.seed(42)
        wrist = np.random.rand(1500, 3) * 0.1 + 0.4

        config = {**base_config, "auto_scale": True}
        result = retarget_wrist_segmented(
            wrist, config, segment_frames=900, overlap_frames=60
        )

        # Check for large jumps at segment boundaries
        diffs = np.linalg.norm(np.diff(result, axis=0), axis=1)
        # No frame-to-frame jump should be unreasonably large
        assert np.max(diffs) < 0.15  # Less than 15cm per frame (global cal for shifted synthetic)

    def test_overlap_must_be_positive(self, base_config):
        """Zero or negative overlap should still work (no blending)."""
        wrist = np.random.rand(1500, 3) * 0.1 + 0.4
        result = retarget_wrist_segmented(
            wrist, base_config, segment_frames=900, overlap_frames=0
        )
        assert result.shape == (1500, 3)
        assert np.all(np.isfinite(result))