from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .camera_recorder import camera_backend_id
from .config import load_config
from .hand_tracking import (
    _create_tracker,
    _draw_landmarks,
    _import_cv2,
    _map_hands_to_robot_sides,
    _select_hands_by_side,
)
from .ik_solver import StatefulBimanualJacobianIKSolver
from .live_control import (
    LiveMetrics,
    LivePinchDetector,
    LiveRetargeter,
    sample_from_landmarks,
)
from .mujoco_replay import _gripper_qpos_indices, _gripper_target
from .openarm_model import load_bimanual_openarm, reset_home


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    captured_at: float
    image: np.ndarray


class LatestFrameCamera:
    def __init__(
        self,
        *,
        index: int,
        backend: str,
        width: int,
        height: int,
        fps: float,
        buffer_size: int = 1,
        pixel_format: str | None = "MJPG",
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("Camera width, height and fps must be positive")
        cv2 = _import_cv2()
        backend_id = camera_backend_id(backend, cv2_module=cv2)
        capture = cv2.VideoCapture(index, backend_id)
        if not capture.isOpened() and backend.lower() == "auto":
            capture.release()
            capture = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        if pixel_format:
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*pixel_format.upper()),
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, max(int(buffer_size), 1))
        self.capture = capture
        self.actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        self._condition = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._stopped = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="openarm-camera",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _capture_loop(self) -> None:
        sequence = 0
        try:
            while True:
                with self._condition:
                    if self._stopped:
                        return
                ok, image = self.capture.read()
                if not ok or image is None:
                    raise RuntimeError("Camera stopped returning frames")
                sequence += 1
                frame = CapturedFrame(sequence, time.monotonic(), image)
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._stopped = True
                self._condition.notify_all()

    def read_latest(
        self,
        *,
        after_sequence: int = 0,
        timeout: float = 2.0,
    ) -> CapturedFrame:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError("Camera capture failed") from self._error
                if (
                    self._latest is not None
                    and self._latest.sequence > after_sequence
                ):
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a camera frame")
                self._condition.wait(remaining)

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self.capture.release()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class LiveHandTracker:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        inference_width: int,
        delegate: str,
        swap_left_right: bool | None = None,
    ) -> None:
        if inference_width <= 0:
            raise ValueError("inference_width must be positive")
        tracker_config = dict(config)
        tracker_config["max_num_hands"] = 2
        tracker_config["delegate"] = delegate
        (self.detector, self.detect), self.backend = _create_tracker(
            tracker_config
        )
        self.inference_width = inference_width
        self.mirror_input = bool(tracker_config.get("mirror_input", False))
        self.swap_left_right = (
            bool(tracker_config.get("swap_left_right", False))
            if swap_left_right is None
            else swap_left_right
        )
        self._last_timestamp_ms = -1

    def process(
        self,
        frame: np.ndarray,
        *,
        timestamp_s: float,
    ) -> tuple[dict[str, Any | None], np.ndarray]:
        cv2 = _import_cv2()
        display = (
            cv2.flip(frame, 1) if self.mirror_input else frame.copy()
        )
        height, width = display.shape[:2]
        inference_height = max(
            int(round(height * self.inference_width / width)), 1
        )
        inference = cv2.resize(
            display,
            (self.inference_width, inference_height),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(inference, cv2.COLOR_BGR2RGB)
        timestamp_ms = max(
            int(round(timestamp_s * 1000)),
            self._last_timestamp_ms + 1,
        )
        self._last_timestamp_ms = timestamp_ms
        hands, labels = self.detect(rgb, timestamp_ms)
        selected = _select_hands_by_side(hands, labels)
        selected = _map_hands_to_robot_sides(
            selected,
            swap_left_right=self.swap_left_right,
        )
        return selected, display

    def close(self) -> None:
        self.detector.close()

    def toggle_left_right(self) -> bool:
        self.swap_left_right = not self.swap_left_right
        return self.swap_left_right


class LiveSessionRecorder:
    def __init__(self) -> None:
        self.values: dict[str, list[Any]] = {
            "timestamps": [],
            "qpos": [],
            "left_target_pos": [],
            "right_target_pos": [],
            "left_gripper_cmd": [],
            "right_gripper_cmd": [],
            "latency_ms": [],
            "inference_ms": [],
            "ik_ms": [],
            "render_ms": [],
        }

    def append(self, **values: Any) -> None:
        for key in self.values:
            self.values[key].append(values[key])

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            **{
                key: np.asarray(value)
                for key, value in self.values.items()
            },
        )
        return destination.resolve()


def _draw_status(
    frame: np.ndarray,
    *,
    tracked: Mapping[str, Any | None],
    paused: bool,
    latency_ms: float,
    inference_ms: float,
    ik_ms: float,
    grippers: Mapping[str, float],
    swap_left_right: bool,
    mirror_horizontal: bool,
) -> None:
    cv2 = _import_cv2()
    colors = {"left": (50, 205, 50), "right": (255, 150, 30)}
    for side in ("left", "right"):
        hand = tracked[side]
        if hand is not None:
            _draw_landmarks(frame, hand, color=colors[side])
    lines = [
        (
            f"{'PAUSED' if paused else 'LIVE'} | "
            f"L:{'OK' if tracked['left'] is not None else 'LOST'} "
            f"R:{'OK' if tracked['right'] is not None else 'LOST'}"
        ),
        (
            f"latency {latency_ms:5.1f} ms | inference "
            f"{inference_ms:5.1f} ms | IK {ik_ms:5.1f} ms"
        ),
        (
            f"gripper L:{'CLOSED' if grippers['left'] else 'OPEN'} "
            f"R:{'CLOSED' if grippers['right'] else 'OPEN'}"
        ),
        (
            f"swap L/R: {'ON' if swap_left_right else 'OFF'} | "
            f"mirror X: {'ON' if mirror_horizontal else 'OFF'} | "
            "S swap | R recalibrate | H home | P pause | Q quit"
        ),
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (16, 30 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )


def _fit_panel(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    cv2 = _import_cv2()
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(
        frame,
        (
            max(int(round(source_width * scale)), 1),
            max(int(round(source_height * scale)), 1),
        ),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _compose_live_display(
    human: np.ndarray | None,
    robot: np.ndarray | None,
    *,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    cv2 = _import_cv2()
    panels = []
    for label, frame in (("HUMAN", human), ("OPENARM MUJOCO", robot)):
        if frame is None:
            continue
        panel = _fit_panel(
            frame,
            width=panel_width,
            height=panel_height,
        )
        cv2.putText(
            panel,
            label,
            (16, panel_height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        panels.append(panel)
    if not panels:
        raise ValueError("At least one live display panel is required")
    return panels[0] if len(panels) == 1 else np.hstack(panels)


def _load_live_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        name: load_config(config_dir / f"{name}.yaml")
        for name in (
            "hand_tracking",
            "pinch",
            "ik",
            "openarm",
            "bimanual_retarget",
            "live",
        )
    }


def run_live_teleoperation(
    *,
    config_dir: str | Path = "configs",
    camera: int | None = None,
    backend: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    inference_width: int | None = None,
    delegate: str | None = None,
    swap_left_right: bool | None = None,
    mirror_horizontal: bool | None = None,
    duration: float | None = None,
    show_viewer: bool = True,
    show_preview: bool = True,
    record_session: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")
    config_path = Path(config_dir)
    configs = _load_live_configs(config_path)
    live = configs["live"]
    camera_config = dict(live.get("camera", {}))
    tracking_config = dict(live.get("tracking", {}))
    control_config = dict(live.get("control", {}))
    display_config = dict(live.get("display", {}))
    ik_config = {
        **configs["ik"],
        **dict(live.get("ik", {})),
    }

    tracker = LiveHandTracker(
        configs["hand_tracking"],
        inference_width=int(
            inference_width
            if inference_width is not None
            else tracking_config.get("inference_width", 640)
        ),
        delegate=str(
            delegate
            if delegate is not None
            else tracking_config.get("delegate", "cpu")
        ),
        swap_left_right=(
            swap_left_right
            if swap_left_right is not None
            else bool(tracking_config.get("swap_left_right", True))
        ),
    )
    model, info = load_bimanual_openarm(configs["openarm"])
    solver = StatefulBimanualJacobianIKSolver(model, info, ik_config)
    retargeter = LiveRetargeter(
        configs["bimanual_retarget"],
        smoothing_tau_s=float(
            control_config.get("smoothing_tau_s", 0.08)
        ),
        max_target_speed_m_s=float(
            control_config.get("max_target_speed_m_s", 0.8)
        ),
        return_home_speed_m_s=float(
            control_config.get("return_home_speed_m_s", 0.2)
        ),
        max_home_displacement_m=control_config.get(
            "max_home_displacement_m", 0.18
        ),
        max_human_jump=float(
            control_config.get("max_human_jump", 0.2)
        ),
        lost_hand_timeout_s=float(
            control_config.get("lost_hand_timeout_s", 0.5)
        ),
    )
    retargeter.set_mirror_horizontal(
        mirror_horizontal
        if mirror_horizontal is not None
        else bool(tracking_config.get("mirror_horizontal", True))
    )
    pinch_config = configs["pinch"]
    pinch_detectors = {
        side: LivePinchDetector(
            close_threshold=float(pinch_config["close_threshold"]),
            open_threshold=float(pinch_config["open_threshold"]),
            initial_state=int(pinch_config.get("initial_state", 0)),
            close_confirm_frames=int(
                pinch_config.get("close_confirm_frames", 1)
            ),
            open_confirm_frames=int(
                pinch_config.get("open_confirm_frames", 1)
            ),
            lost_hand_timeout_s=float(
                control_config.get("lost_hand_timeout_s", 0.5)
            ),
            open_on_loss=bool(
                control_config.get("open_gripper_on_loss", True)
            ),
        )
        for side in ("left", "right")
    }

    import mujoco

    display_data = mujoco.MjData(model)
    reset_home(model, display_data, info.home_keyframe)
    gripper_ids = {}
    gripper_indices = {}
    for side in ("left", "right"):
        actuator_name = info.sides[side].gripper_actuator
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        gripper_ids[side] = actuator_id
        gripper_indices[side] = _gripper_qpos_indices(model, actuator_id)

    renderer = None
    cached_robot_frame = None
    render_every_n_frames = max(
        int(display_config.get("render_every_n_frames", 1)), 1
    )
    if show_viewer:
        panel_width = int(display_config.get("panel_width", 640))
        panel_height = int(display_config.get("panel_height", 480))
        model.vis.global_.offwidth = max(
            model.vis.global_.offwidth, panel_width
        )
        model.vis.global_.offheight = max(
            model.vis.global_.offheight, panel_height
        )
        renderer = mujoco.Renderer(
            model,
            height=panel_height,
            width=panel_width,
        )
        robot_rgb = None
        for _ in range(3):
            renderer.update_scene(
                display_data,
                camera=configs["openarm"].get("camera"),
            )
            robot_rgb = renderer.render()
        cv2 = _import_cv2()
        cached_robot_frame = cv2.cvtColor(
            robot_rgb, cv2.COLOR_RGB2BGR
        )

    camera_device = LatestFrameCamera(
        index=int(
            camera if camera is not None else camera_config.get("index", 0)
        ),
        backend=str(
            backend
            if backend is not None
            else camera_config.get("backend", "auto")
        ),
        width=int(
            width if width is not None else camera_config.get("width", 1280)
        ),
        height=int(
            height
            if height is not None
            else camera_config.get("height", 720)
        ),
        fps=float(
            fps if fps is not None else camera_config.get("fps", 30)
        ),
        buffer_size=int(camera_config.get("buffer_size", 1)),
        pixel_format=str(camera_config.get("pixel_format", "MJPG")),
    )
    metrics = LiveMetrics()
    recorder = LiveSessionRecorder() if record_session else None
    started_at = time.monotonic()
    last_sequence = 0
    paused = False
    returning_home = False
    window_name = "OpenArm Live Teleoperation"
    camera_device.start()
    try:
        while True:
            if duration is not None and time.monotonic() - started_at >= duration:
                break
            captured = camera_device.read_latest(
                after_sequence=last_sequence,
                timeout=2.0,
            )
            last_sequence = captured.sequence
            frame_started = time.monotonic()
            inference_started = frame_started
            selected, preview = tracker.process(
                captured.image,
                timestamp_s=captured.captured_at,
            )
            inference_ms = (time.monotonic() - inference_started) * 1000
            samples = {
                side: (
                    sample_from_landmarks(selected[side])
                    if selected[side] is not None
                    else None
                )
                for side in ("left", "right")
            }
            if returning_home:
                targets = retargeter.step_return_home(
                    captured.captured_at
                )
                grippers = {"left": 0.0, "right": 0.0}
                if retargeter.at_home():
                    returning_home = False
                    paused = True
            elif not paused:
                targets = {
                    side: retargeter.update(
                        side, samples[side], captured.captured_at
                    )
                    for side in ("left", "right")
                }
                grippers = {
                    side: pinch_detectors[side].update(
                        samples[side], captured.captured_at
                    )
                    for side in ("left", "right")
                }
            else:
                targets = {
                    side: retargeter.target(side)
                    for side in ("left", "right")
                }
                grippers = {
                    side: pinch_detectors[side].state
                    for side in ("left", "right")
                }

            ik_started = time.monotonic()
            ik_frame = solver.solve_frame(
                targets["left"], targets["right"]
            )
            ik_ms = (time.monotonic() - ik_started) * 1000
            display_data.qpos[:] = ik_frame.qpos
            for side in ("left", "right"):
                target = _gripper_target(
                    grippers[side],
                    model.actuator_ctrlrange[gripper_ids[side]],
                )
                display_data.qpos[gripper_indices[side]] = target
            mujoco.mj_forward(model, display_data)

            render_started = time.monotonic()
            if (
                renderer is not None
                and metrics.processed_frames % render_every_n_frames == 0
            ):
                renderer.update_scene(
                    display_data,
                    camera=configs["openarm"].get("camera"),
                )
                robot_rgb = renderer.render()
                cv2 = _import_cv2()
                cached_robot_frame = cv2.cvtColor(
                    robot_rgb, cv2.COLOR_RGB2BGR
                )
            render_ms = (time.monotonic() - render_started) * 1000
            latency_ms = (time.monotonic() - captured.captured_at) * 1000
            display_frame = None
            if show_preview or renderer is not None:
                human_frame = preview if show_preview else None
                if human_frame is not None:
                    _draw_status(
                        human_frame,
                        tracked=selected,
                        paused=paused,
                        latency_ms=latency_ms,
                        inference_ms=inference_ms,
                        ik_ms=ik_ms,
                        grippers=grippers,
                        swap_left_right=tracker.swap_left_right,
                        mirror_horizontal=retargeter.mirror_horizontal,
                    )
                display_frame = _compose_live_display(
                    human_frame,
                    cached_robot_frame,
                    panel_width=int(
                        display_config.get("panel_width", 640)
                    ),
                    panel_height=int(
                        display_config.get("panel_height", 480)
                    ),
                )
            metrics.record(
                sequence=captured.sequence,
                latency_ms=latency_ms,
                inference_ms=inference_ms,
                ik_ms=ik_ms,
                render_ms=render_ms,
            )
            if recorder is not None:
                recorder.append(
                    timestamps=time.monotonic() - started_at,
                    qpos=display_data.qpos.copy(),
                    left_target_pos=targets["left"],
                    right_target_pos=targets["right"],
                    left_gripper_cmd=grippers["left"],
                    right_gripper_cmd=grippers["right"],
                    latency_ms=latency_ms,
                    inference_ms=inference_ms,
                    ik_ms=ik_ms,
                    render_ms=render_ms,
                )
            if display_frame is not None:
                cv2 = _import_cv2()
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    retargeter.reset_reference()
                    returning_home = False
                    paused = False
                if key in (ord("s"), ord("S")):
                    tracker.toggle_left_right()
                    retargeter.toggle_horizontal_direction()
                    returning_home = False
                if key in (ord("h"), ord("H")):
                    retargeter.start_return_home(
                        time.monotonic()
                    )
                    for detector in pinch_detectors.values():
                        detector.open()
                    returning_home = True
                    paused = False
                if key in (ord("p"), ord("P")):
                    paused = not paused
                    returning_home = False
    finally:
        camera_device.close()
        tracker.close()
        if renderer is not None:
            renderer.close()
        if show_preview or show_viewer:
            cv2 = _import_cv2()
            cv2.destroyAllWindows()

    summary = {
        **metrics.summary(),
        "camera": {
            "index": int(
                camera
                if camera is not None
                else camera_config.get("index", 0)
            ),
            "requested_fps": float(
                fps if fps is not None else camera_config.get("fps", 30)
            ),
            "actual_width": camera_device.actual_width,
            "actual_height": camera_device.actual_height,
            "reported_fps": camera_device.actual_fps,
        },
        "tracking": {
            "backend": tracker.backend,
            "delegate": str(
                delegate
                if delegate is not None
                else tracking_config.get("delegate", "cpu")
            ),
            "inference_width": tracker.inference_width,
            "swap_left_right": tracker.swap_left_right,
            "mirror_horizontal": retargeter.mirror_horizontal,
        },
        "display": {
            "mujoco_enabled": show_viewer,
            "render_every_n_frames": render_every_n_frames,
        },
    }
    if recorder is not None and record_session is not None:
        summary["session_path"] = str(recorder.save(record_session))
    destination = Path(
        report_path
        if report_path is not None
        else live.get(
            "report_path", "outputs/live/latest_live_report.json"
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    summary["report_path"] = str(destination.resolve())
    return summary
