from __future__ import annotations

from types import SimpleNamespace

import pytest

from openarm_retarget.camera_recorder import (
    RecordingResult,
    camera_backend_id,
    record_camera,
)


CV2 = SimpleNamespace(
    CAP_ANY=0,
    CAP_DSHOW=700,
    CAP_MSMF=1400,
    CAP_V4L2=200,
)


def test_auto_camera_backend_prefers_dshow_on_windows():
    assert camera_backend_id("auto", cv2_module=CV2, system="Windows") == 700
    assert camera_backend_id("auto", cv2_module=CV2, system="Linux") == 0


def test_explicit_camera_backends_and_invalid_value(tmp_path):
    assert camera_backend_id("msmf", cv2_module=CV2) == 1400
    assert camera_backend_id("v4l2", cv2_module=CV2) == 200
    with pytest.raises(ValueError, match="Unknown camera backend"):
        camera_backend_id("invalid", cv2_module=CV2)

    result = RecordingResult(tmp_path / "video.mp4", 90, 30.0, 1280, 720)
    assert result.duration_seconds == 3.0


def test_recorder_refuses_to_overwrite_before_opening_camera(tmp_path):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="--overwrite"):
        record_camera(output)

    assert output.read_bytes() == b"keep"
