from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            'Comparison video requires OpenCV. Install with: '
            'python -m pip install -e ".[vision]"'
        ) from exc
    return cv2


@dataclass(frozen=True)
class ComparisonResult:
    output: Path
    frames: int
    fps: float
    width: int
    height: int


def fit_frame(
    frame: np.ndarray,
    width: int,
    height: int,
    *,
    background: tuple[int, int, int] = (18, 18, 18),
) -> np.ndarray:
    cv2 = _cv2()
    source_height, source_width = frame.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Input frame has invalid dimensions")
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def compose_comparison_frame(
    human_frame: np.ndarray,
    robot_frame: np.ndarray,
    *,
    panel_width: int = 960,
    panel_height: int = 720,
    timestamp: float = 0.0,
) -> np.ndarray:
    cv2 = _cv2()
    human = fit_frame(human_frame, panel_width, panel_height)
    robot = fit_frame(robot_frame, panel_width, panel_height)
    combined = np.hstack((human, robot))
    cv2.line(
        combined,
        (panel_width, 0),
        (panel_width, panel_height),
        (255, 255, 255),
        3,
    )
    for label, x in (("HUMAN HAND TRACKING", 20), ("OPENARM MUJOCO", panel_width + 20)):
        cv2.rectangle(combined, (x - 8, 12), (x + 310, 52), (0, 0, 0), -1)
        cv2.putText(
            combined,
            label,
            (x, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
    time_label = f"{timestamp:06.2f} s"
    cv2.rectangle(
        combined,
        (panel_width - 65, panel_height - 46),
        (panel_width + 65, panel_height - 10),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        combined,
        time_label,
        (panel_width - 50, panel_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    return combined


def create_comparison_video(
    human_video: str | Path,
    robot_video: str | Path,
    output: str | Path,
    *,
    panel_width: int = 960,
    panel_height: int = 720,
    overwrite: bool = False,
) -> ComparisonResult:
    cv2 = _cv2()
    human_path = Path(human_video)
    robot_path = Path(robot_video)
    destination = Path(output)
    for path in (human_path, robot_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input video not found: {path}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Pass --overwrite to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.composing{destination.suffix}"
    )
    temporary.unlink(missing_ok=True)

    human = cv2.VideoCapture(str(human_path))
    robot = cv2.VideoCapture(str(robot_path))
    if not human.isOpened() or not robot.isOpened():
        human.release()
        robot.release()
        raise RuntimeError("Cannot open one or both input videos")

    human_fps = float(human.get(cv2.CAP_PROP_FPS))
    robot_fps = float(robot.get(cv2.CAP_PROP_FPS))
    if human_fps <= 0 or robot_fps <= 0:
        human.release()
        robot.release()
        raise ValueError("Both input videos must report a positive FPS")
    if abs(human_fps - robot_fps) > 0.01:
        human.release()
        robot.release()
        raise ValueError(
            f"Input FPS differs: human={human_fps}, robot={robot_fps}"
        )

    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        human_fps,
        (panel_width * 2, panel_height),
    )
    if not writer.isOpened():
        human.release()
        robot.release()
        raise RuntimeError(f"Cannot create comparison video: {temporary}")

    frame_index = 0
    try:
        while True:
            human_ok, human_frame = human.read()
            robot_ok, robot_frame = robot.read()
            if not human_ok or not robot_ok:
                break
            writer.write(
                compose_comparison_frame(
                    human_frame,
                    robot_frame,
                    panel_width=panel_width,
                    panel_height=panel_height,
                    timestamp=frame_index / human_fps,
                )
            )
            frame_index += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        human.release()
        robot.release()
        writer.release()

    if frame_index == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("No synchronized frames could be composed")
    temporary.replace(destination)
    return ComparisonResult(
        output=destination.resolve(),
        frames=frame_index,
        fps=human_fps,
        width=panel_width * 2,
        height=panel_height,
    )

