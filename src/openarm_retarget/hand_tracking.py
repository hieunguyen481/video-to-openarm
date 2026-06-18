from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LANDMARK_INDICES = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}
PALM_MCP_INDICES = (5, 9, 13, 17)

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class TrackingResult:
    data: dict[str, np.ndarray]
    fps: float
    frame_size: tuple[int, int]


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video tracking. Install with: "
            'python -m pip install -e ".[vision]"'
        ) from exc
    return cv2


def _select_hand(
    hands: list[Any],
    labels: list[str],
    preferred: str | None,
) -> tuple[Any | None, str]:
    if not hands:
        return None, ""
    if preferred:
        for hand, label in zip(hands, labels, strict=False):
            if label.lower() == preferred.lower():
                return hand, label
    return hands[0], labels[0] if labels else ""


def _select_hands_by_side(
    hands: list[Any],
    labels: list[str],
) -> dict[str, Any | None]:
    selected: dict[str, Any | None] = {"left": None, "right": None}
    for hand, label in zip(hands, labels, strict=False):
        side = label.lower()
        if side in selected and selected[side] is None:
            selected[side] = hand
    return selected


def _map_hands_to_robot_sides(
    selected: dict[str, Any | None],
    *,
    swap_left_right: bool,
) -> dict[str, Any | None]:
    if not swap_left_right:
        return selected
    return {
        "left": selected["right"],
        "right": selected["left"],
    }


def _palm_scale(landmarks: Any) -> float:
    wrist = np.asarray([landmarks[0].x, landmarks[0].y], dtype=float)
    distances = [
        np.linalg.norm(
            np.asarray([landmarks[index].x, landmarks[index].y], dtype=float)
            - wrist
        )
        for index in PALM_MCP_INDICES
    ]
    return float(np.mean(distances))


def _draw_landmarks(
    frame: np.ndarray,
    landmarks: Any,
    *,
    color: tuple[int, int, int] = (40, 210, 90),
) -> None:
    cv2 = _import_cv2()
    height, width = frame.shape[:2]
    points = [
        (int(np.clip(item.x, 0, 1) * (width - 1)), int(np.clip(item.y, 0, 1) * (height - 1)))
        for item in landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 2)
    for point in points:
        cv2.circle(frame, point, 3, (30, 80, 255), -1)


def _task_tracker(config: dict[str, Any]):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("MediaPipe Tasks is not available") from exc

    model_path = Path(config.get("model_path", "models/hand_landmarker.task"))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe task model not found: {model_path}. "
            "Download hand_landmarker.task and set model_path in the config."
        )
    delegate_name = str(config.get("delegate", "cpu")).lower()
    if delegate_name == "auto":
        delegate_name = "cpu"
    if delegate_name not in {"cpu", "gpu"}:
        raise ValueError("MediaPipe delegate must be auto, cpu or gpu")
    delegate = (
        python.BaseOptions.Delegate.GPU
        if delegate_name == "gpu"
        else python.BaseOptions.Delegate.CPU
    )
    try:
        base_options = python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=delegate,
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(config.get("max_num_hands", 1)),
            min_hand_detection_confidence=float(
                config.get("min_detection_confidence", 0.5)
            ),
            min_hand_presence_confidence=float(
                config.get(
                    "min_hand_presence_confidence",
                    config.get("min_detection_confidence", 0.5),
                )
            ),
            min_tracking_confidence=float(
                config.get("min_tracking_confidence", 0.5)
            ),
        )
        detector = vision.HandLandmarker.create_from_options(options)
    except (RuntimeError, NotImplementedError) as exc:
        if delegate_name == "gpu":
            platform_note = (
                " The official MediaPipe Python GPU delegate is not "
                "available in the current Windows wheel."
                if sys.platform == "win32"
                else ""
            )
            raise RuntimeError(
                f"Cannot initialize MediaPipe GPU delegate.{platform_note}"
            ) from exc
        raise

    def detect(rgb: np.ndarray, timestamp_ms: int):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(image, timestamp_ms)
        labels = [
            categories[0].category_name if categories else ""
            for categories in result.handedness
        ]
        return result.hand_landmarks, labels, result.hand_world_landmarks

    return detector, detect


def _legacy_tracker(config: dict[str, Any]):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is not installed") from exc
    if not hasattr(mp, "solutions"):
        raise RuntimeError("This MediaPipe build does not include the legacy solutions API")

    detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=int(config.get("max_num_hands", 1)),
        min_detection_confidence=float(
            config.get("min_detection_confidence", 0.5)
        ),
        min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
    )

    def detect(rgb: np.ndarray, timestamp_ms: int):
        del timestamp_ms
        result = detector.process(rgb)
        hands = [
            item.landmark for item in (result.multi_hand_landmarks or [])
        ]
        labels = [
            item.classification[0].label
            for item in (result.multi_handedness or [])
        ]
        # Legacy API does not provide world landmarks
        world_hands: list[Any] = []
        return hands, labels, world_hands

    return detector, detect


def _create_tracker(config: dict[str, Any]):
    backend = str(config.get("backend", "auto")).lower()
    errors: list[str] = []
    if backend in {"auto", "tasks"}:
        try:
            return _task_tracker(config), "tasks"
        except (RuntimeError, FileNotFoundError) as exc:
            errors.append(str(exc))
            if backend == "tasks":
                raise
    if backend in {"auto", "legacy"}:
        try:
            return _legacy_tracker(config), "legacy"
        except RuntimeError as exc:
            errors.append(str(exc))
            if backend == "legacy":
                raise
    raise RuntimeError("No MediaPipe backend available:\n- " + "\n- ".join(errors))


def extract_video_hand_pose(
    video: str | Path,
    config: dict[str, Any],
    *,
    debug_video: str | Path | None = None,
) -> TrackingResult:
    cv2 = _import_cv2()
    source = Path(video)
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    (detector, detect), backend = _create_tracker(config)
    writer = None
    if debug_video:
        debug_path = Path(debug_video)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(debug_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create debug video: {debug_path}")

    values = {name: [] for name in LANDMARK_INDICES}
    palm_scale: list[float] = []
    timestamps: list[float] = []
    valid: list[bool] = []
    selected_labels: list[str] = []
    preferred = config.get("handedness")
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if bool(config.get("mirror_input", False)):
                frame = cv2.flip(frame, 1)
            timestamp = frame_index / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands, labels, _world_hands = detect(rgb, int(round(timestamp * 1000)))
            hand, label = _select_hand(hands, labels, preferred)

            timestamps.append(timestamp)
            valid.append(hand is not None)
            selected_labels.append(label)
            if hand is None:
                palm_scale.append(np.nan)
                for name in LANDMARK_INDICES:
                    values[name].append([np.nan, np.nan, np.nan])
            else:
                palm_scale.append(_palm_scale(hand))
                for name, index in LANDMARK_INDICES.items():
                    item = hand[index]
                    values[name].append([item.x, item.y, item.z])
                if writer is not None:
                    _draw_landmarks(frame, hand)

            if writer is not None:
                status = f"{backend} | {'TRACKED' if hand is not None else 'LOST'}"
                cv2.putText(
                    frame,
                    status,
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        detector.close()

    if not timestamps:
        raise ValueError(f"Video contains no readable frames: {source}")
    data = {
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "valid": np.asarray(valid, dtype=bool),
        "handedness": np.asarray(
            next((label for label in selected_labels if label), preferred or "Unknown")
        ),
        "palm_scale": np.asarray(palm_scale, dtype=np.float32),
        **{
            name: np.asarray(points, dtype=np.float32)
            for name, points in values.items()
        },
    }
    return TrackingResult(data=data, fps=fps, frame_size=(width, height))


def extract_video_bimanual_hand_pose(
    video: str | Path,
    config: dict[str, Any],
    *,
    debug_video: str | Path | None = None,
) -> TrackingResult:
    cv2 = _import_cv2()
    source = Path(video)
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bimanual_config = dict(config)
    bimanual_config["max_num_hands"] = max(2, int(config.get("max_num_hands", 2)))
    (detector, detect), backend = _create_tracker(bimanual_config)
    writer = None
    if debug_video:
        debug_path = Path(debug_video)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(debug_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create debug video: {debug_path}")

    values = {
        side: {name: [] for name in LANDMARK_INDICES}
        for side in ("left", "right")
    }
    world_values = {
        side: {name: [] for name in LANDMARK_INDICES}
        for side in ("left", "right")
    }
    palm_scale = {"left": [], "right": []}
    valid = {"left": [], "right": []}
    timestamps: list[float] = []
    colors = {"left": (50, 205, 50), "right": (255, 150, 30)}
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if bool(config.get("mirror_input", False)):
                frame = cv2.flip(frame, 1)
            timestamp = frame_index / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands, labels, world_hands = detect(rgb, int(round(timestamp * 1000)))
            selected = _select_hands_by_side(hands, labels)
            selected = _map_hands_to_robot_sides(
                selected,
                swap_left_right=bool(config.get("swap_left_right", False)),
            )
            # Match world landmarks to the same hands by index
            world_selected = _select_hands_by_side(world_hands, labels)
            world_selected = _map_hands_to_robot_sides(
                world_selected,
                swap_left_right=bool(config.get("swap_left_right", False)),
            )
            timestamps.append(timestamp)

            for side in ("left", "right"):
                hand = selected[side]
                world_hand = world_selected[side]
                valid[side].append(hand is not None)
                if hand is None:
                    palm_scale[side].append(np.nan)
                    for name in LANDMARK_INDICES:
                        values[side][name].append([np.nan, np.nan, np.nan])
                        world_values[side][name].append([np.nan, np.nan, np.nan])
                else:
                    palm_scale[side].append(_palm_scale(hand))
                    for name, index in LANDMARK_INDICES.items():
                        item = hand[index]
                        values[side][name].append([item.x, item.y, item.z])
                    # Extract world landmarks (3D coordinates in cm)
                    if world_hand is not None:
                        for name, index in LANDMARK_INDICES.items():
                            w_item = world_hand[index]
                            world_values[side][name].append(
                                [w_item.x, w_item.y, w_item.z]
                            )
                    else:
                        for name in LANDMARK_INDICES:
                            world_values[side][name].append([np.nan, np.nan, np.nan])
                    if writer is not None:
                        _draw_landmarks(frame, hand, color=colors[side])

            if writer is not None:
                status = (
                    f"{backend} | "
                    f"L:{'OK' if selected['left'] is not None else 'LOST'} "
                    f"R:{'OK' if selected['right'] is not None else 'LOST'}"
                )
                cv2.putText(
                    frame,
                    status,
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        detector.close()

    if not timestamps:
        raise ValueError(f"Video contains no readable frames: {source}")
    data: dict[str, np.ndarray] = {
        "timestamps": np.asarray(timestamps, dtype=np.float64)
    }
    for side in ("left", "right"):
        data[f"{side}_valid"] = np.asarray(valid[side], dtype=bool)
        data[f"{side}_palm_scale"] = np.asarray(
            palm_scale[side], dtype=np.float32
        )
        for name, points in values[side].items():
            data[f"{side}_{name}"] = np.asarray(points, dtype=np.float32)
        # Include world landmarks (3D coordinates from MediaPipe)
        for name, points in world_values[side].items():
            data[f"{side}_world_{name}"] = np.asarray(points, dtype=np.float32)
    return TrackingResult(data=data, fps=fps, frame_size=(width, height))
