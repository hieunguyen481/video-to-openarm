from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("openarm_mujoco")
pytest.importorskip("matplotlib")

from openarm_retarget.pipeline import run_synthetic


def test_synthetic_pipeline_creates_all_non_video_artifacts(tmp_path):
    config_dir = Path(__file__).parents[1] / "configs"
    artifacts, quality = run_synthetic(
        name="test_run",
        frames=36,
        root=tmp_path,
        config_dir=config_dir,
        render=False,
    )

    for path in (
        artifacts.hand_pose,
        artifacts.pinch,
        artifacts.smooth,
        artifacts.target,
        artifacts.trajectory,
        artifacts.dataset,
        artifacts.pinch_plot,
        artifacts.smoothing_plot,
        artifacts.target_plot,
        artifacts.ik_plot,
        artifacts.quality_report,
    ):
        assert path.is_file()
    assert quality["frames"] == 36
    assert quality["max_ik_error_m"] < 0.05
    saved_quality = json.loads(artifacts.quality_report.read_text(encoding="utf-8"))
    assert saved_quality["ik_converged_ratio"] == 1.0

