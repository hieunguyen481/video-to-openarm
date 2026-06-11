from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            'Camera recording requires OpenCV. Install with: '
            'python -m pip install -e ".[vision]"'
        ) from exc
    return cv2


def camera_backend_id(
    backend: str,
    *,
    cv2_module: Any,
    system: str | None = None,
) -> int:
    name = backend.lower()
    if name == "auto":
        current_system = system or platform.system()
        if current_system == "Windows":
            return int(cv2_module.CAP_DSHOW)
        return int(cv2_module.CAP_ANY)
    mapping = {
        "any": cv2_module.CAP_ANY,
        "dshow": getattr(cv2_module, "CAP_DSHOW", cv2_module.CAP_ANY),
        "msmf": getattr(cv2_module, "CAP_MSMF", cv2_module.CAP_ANY),
        "v4l2": getattr(cv2_module, "CAP_V4L2", cv2_module.CAP_ANY),
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown camera backend {backend!r}; "
            f"choose from {', '.join(mapping)} or auto"
        )
    return int(mapping[name])


@dataclass(frozen=True)
class RecordingResult:
    output: Path
    frames: int
    fps: float
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.fps if self.fps > 0 else 0.0


def _draw_preview(
    frame: np.ndarray,
    *,
    recording: bool,
    recorded_frames: int,
    fps: float,
    output: Path,
) -> np.ndarray:
    cv2 = _cv2()
    preview = frame.copy()
    status = "RECORDING" if recording else "READY / PAUSED"
    color = (40, 40, 230) if recording else (40, 210, 240)
    if recording:
        cv2.circle(preview, (24, 25), 9, color, -1)
    cv2.putText(
        preview,
        status,
        (43, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )
    elapsed = recorded_frames / fps if fps > 0 else 0.0
    cv2.putText(
        preview,
        f"Recorded: {elapsed:05.1f}s | {output.name}",
        (16, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        preview,
        "SPACE: start/pause   Q or ESC: save and exit",
        (16, preview.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return preview


def record_camera(
    output: str | Path,
    *,
    camera: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    backend: str = "auto",
    mirror_preview: bool = True,
    auto_start: bool = False,
    duration: float | None = None,
    overwrite: bool = False,
) -> RecordingResult:
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height and fps must be positive")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")

    cv2 = _cv2()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. "
            "Choose another name or pass --overwrite."
        )
    temporary = destination.with_name(
        f".{destination.stem}.recording{destination.suffix}"
    )
    temporary.unlink(missing_ok=True)
    backend_id = camera_backend_id(backend, cv2_module=cv2)
    capture = cv2.VideoCapture(camera, backend_id)
    if not capture.isOpened() and backend.lower() == "auto":
        capture.release()
        capture = cv2.VideoCapture(camera, cv2.CAP_ANY)
    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open camera {camera}. Close other camera applications "
            "and check Windows camera privacy permissions."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"Camera {camera} opened but returned no frames")

    actual_height, actual_width = first_frame.shape[:2]
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    output_fps = reported_fps if np.isfinite(reported_fps) and reported_fps > 1 else fps
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (actual_width, actual_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {temporary}")

    window_name = "Video to OpenArm - Camera Recorder"
    recording = auto_start
    recorded_frames = 0
    frame = first_frame
    last_status_time = time.monotonic()
    try:
        try:
            while True:
                if recording:
                    writer.write(frame)
                    recorded_frames += 1
                    if duration is not None and recorded_frames / output_fps >= duration:
                        break

                display = cv2.flip(frame, 1) if mirror_preview else frame
                preview = _draw_preview(
                    display,
                    recording=recording,
                    recorded_frames=recorded_frames,
                    fps=output_fps,
                    output=destination,
                )
                cv2.imshow(window_name, preview)
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    recording = not recording
                    last_status_time = time.monotonic()
                elif key in (ord("q"), ord("Q"), 27):
                    break

                ok, next_frame = capture.read()
                if not ok or next_frame is None:
                    if time.monotonic() - last_status_time > 2.0:
                        raise RuntimeError("Camera stopped returning frames")
                    continue
                frame = next_frame
        finally:
            capture.release()
            writer.release()
            cv2.destroyAllWindows()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    if recorded_frames == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("No frames were recorded; press SPACE before exiting")
    temporary.replace(destination)
    return RecordingResult(
        output=destination.resolve(),
        frames=recorded_frames,
        fps=output_fps,
        width=actual_width,
        height=actual_height,
    )
