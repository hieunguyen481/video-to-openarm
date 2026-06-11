from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from openarm_retarget.comparison_video import (
    compose_comparison_frame,
    fit_frame,
)


def test_fit_frame_letterboxes_without_changing_canvas_size():
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)
    result = fit_frame(frame, 100, 100)

    assert result.shape == (100, 100, 3)
    assert np.all(result[25:75] == 255)
    assert np.all(result[:20] == 18)


def test_comparison_frame_places_two_panels_side_by_side():
    human = np.zeros((720, 1280, 3), dtype=np.uint8)
    robot = np.zeros((720, 960, 3), dtype=np.uint8)

    result = compose_comparison_frame(
        human, robot, panel_width=960, panel_height=720, timestamp=1.5
    )

    assert result.shape == (720, 1920, 3)
    assert np.all(result[80:650, 960] == 255)
