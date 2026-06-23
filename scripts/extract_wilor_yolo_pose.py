"""Extract bimanual hand-pose NPZ files from a WiLoR YOLO detector.

This script intentionally uses only WiLoR's YOLO keypoint detector output. It
does not require MANO assets or the full WiLoR 3D reconstruction stack.
"""
from __future__ import annotations

import argparse
import json
import os
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
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _load_runtime_modules() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "WiLoR YOLO extraction requires opencv-python, torch, and "
            "ultralytics. Install them in the environment used for WiLoR."
        ) from exc
    return cv2, torch, YOLO


def _patch_torch_load(torch_module: Any) -> None:
    original_load = torch_module.load

    def patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch_module.load = patched_load


def _empty_side() -> dict[str, list[Any]]:
    return {
        "valid": [],
        "palm_scale": [],
        **{name: [] for name in LANDMARK_INDICES},
    }


def _choose_by_side(
    result: Any, names: dict[int, str], conf_threshold: float
) -> dict[str, dict[str, Any] | None]:
    selected: dict[str, dict[str, Any] | None] = {"left": None, "right": None}
    if result.boxes is None or result.keypoints is None:
        return selected
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy()
    for index, class_id in enumerate(classes):
        side = names.get(int(class_id), "").lower()
        if side not in selected or confs[index] < conf_threshold:
            continue
        current = selected[side]
        if current is None or confs[index] > current["conf"]:
            selected[side] = {
                "conf": float(confs[index]),
                "keypoints": keypoints[index],
            }
    return selected


def _normalized_landmarks(
    keypoints: np.ndarray, width: int, height: int
) -> np.ndarray:
    normalized = np.empty((21, 3), dtype=np.float32)
    normalized[:, 0] = keypoints[:, 0] / max(width - 1, 1)
    normalized[:, 1] = keypoints[:, 1] / max(height - 1, 1)
    normalized[:, 2] = 0.0
    return normalized


def _wrist_xy(
    detection: dict[str, Any] | None, width: int, height: int
) -> np.ndarray | None:
    if detection is None:
        return None
    landmarks = _normalized_landmarks(detection["keypoints"], width, height)
    return landmarks[0, :2]


def _reject_unstable_detections(
    selected: dict[str, dict[str, Any] | None],
    *,
    width: int,
    height: int,
    last_wrist: dict[str, np.ndarray | None],
    max_wrist_jump: float,
    min_lr_distance: float,
    edge_margin: float,
) -> dict[str, dict[str, Any] | None]:
    filtered = dict(selected)
    wrists = {
        side: _wrist_xy(filtered[side], width, height)
        for side in ("left", "right")
    }

    if wrists["left"] is not None and wrists["right"] is not None:
        if np.linalg.norm(wrists["left"] - wrists["right"]) < min_lr_distance:
            costs = {}
            for side in ("left", "right"):
                previous = last_wrist.get(side)
                costs[side] = (
                    np.inf
                    if previous is None
                    else float(np.linalg.norm(wrists[side] - previous))
                )
            drop_side = "left" if costs["left"] >= costs["right"] else "right"
            filtered[drop_side] = None
            wrists[drop_side] = None

    for side in ("left", "right"):
        wrist = wrists[side]
        if wrist is None:
            continue
        near_edge = (
            wrist[0] < edge_margin
            or wrist[0] > 1.0 - edge_margin
            or wrist[1] < edge_margin
            or wrist[1] > 1.0 - edge_margin
        )
        previous = last_wrist.get(side)
        jumped = (
            previous is not None
            and np.linalg.norm(wrist - previous) > max_wrist_jump
        )
        if near_edge or jumped:
            filtered[side] = None
    return filtered


def _palm_scale(landmarks: np.ndarray) -> float:
    wrist = landmarks[0, :2]
    distances = [
        np.linalg.norm(landmarks[index, :2] - wrist)
        for index in PALM_MCP_INDICES
    ]
    return float(np.mean(distances))


def _append_side(
    side_data: dict[str, list[Any]],
    detection: dict[str, Any] | None,
    width: int,
    height: int,
    min_kp_conf: float,
) -> None:
    if detection is None:
        side_data["valid"].append(False)
        side_data["palm_scale"].append(np.nan)
        for name in LANDMARK_INDICES:
            side_data[name].append([np.nan, np.nan, np.nan])
        return

    keypoints = detection["keypoints"]
    required = [0, 4, 8, 12, 16, 20, *PALM_MCP_INDICES]
    valid = bool(np.all(keypoints[required, 2] >= min_kp_conf))
    landmarks = _normalized_landmarks(keypoints, width, height)
    side_data["valid"].append(valid)
    side_data["palm_scale"].append(_palm_scale(landmarks) if valid else np.nan)
    for name, index in LANDMARK_INDICES.items():
        value = landmarks[index].tolist() if valid else [np.nan, np.nan, np.nan]
        side_data[name].append(value)


def _draw_detection(
    cv2: Any,
    frame: np.ndarray,
    detection: dict[str, Any] | None,
    color: tuple[int, int, int],
) -> None:
    if detection is None:
        return
    points = detection["keypoints"][:, :2].astype(int)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, tuple(points[start]), tuple(points[end]), color, 2)
    for point in points:
        cv2.circle(frame, tuple(point), 3, (30, 80, 255), -1)


def convert(args: argparse.Namespace) -> dict[str, float | int | str]:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    cv2, torch_module, yolo_class = _load_runtime_modules()
    _patch_torch_load(torch_module)

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.model.is_file():
        raise FileNotFoundError(f"WiLoR detector weights not found: {args.model}")

    model = yolo_class(str(args.model))
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.debug_video:
        args.debug_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.debug_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create debug video: {args.debug_video}")

    sides = {"left": _empty_side(), "right": _empty_side()}
    timestamps: list[float] = []
    last_wrist: dict[str, np.ndarray | None] = {"left": None, "right": None}
    one_hand_frames = 0
    two_hand_frames = 0
    frame_index = 0
    colors = {"left": (50, 205, 50), "right": (255, 150, 30)}

    try:
        while True:
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            result = model(frame, verbose=False, conf=args.conf)[0]
            selected = _choose_by_side(result, result.names, args.conf)
            if args.temporal_filter:
                selected = _reject_unstable_detections(
                    selected,
                    width=width,
                    height=height,
                    last_wrist=last_wrist,
                    max_wrist_jump=args.max_wrist_jump,
                    min_lr_distance=args.min_lr_distance,
                    edge_margin=args.edge_margin,
                )

            timestamps.append(frame_index / fps)
            valid_count = 0
            for side in ("left", "right"):
                _append_side(
                    sides[side],
                    selected[side],
                    width,
                    height,
                    args.min_kp_conf,
                )
                if sides[side]["valid"][-1]:
                    valid_count += 1
                    last_wrist[side] = np.asarray(
                        sides[side]["wrist"][-1][:2], dtype=np.float32
                    )
                if writer is not None:
                    _draw_detection(cv2, frame, selected[side], colors[side])

            if valid_count >= 1:
                one_hand_frames += 1
            if valid_count >= 2:
                two_hand_frames += 1
            if writer is not None:
                cv2.putText(
                    frame,
                    f"WiLoR YOLO | L:{'OK' if sides['left']['valid'][-1] else 'LOST'} "
                    f"R:{'OK' if sides['right']['valid'][-1] else 'LOST'}",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                writer.write(frame)

            frame_index += 1
            if args.progress and frame_index % args.progress == 0:
                print(f"processed {frame_index} frames", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    payload: dict[str, np.ndarray] = {
        "timestamps": np.asarray(timestamps, dtype=np.float64),
    }
    for side in ("left", "right"):
        payload[f"{side}_valid"] = np.asarray(sides[side]["valid"], dtype=bool)
        payload[f"{side}_palm_scale"] = np.asarray(
            sides[side]["palm_scale"], dtype=np.float32
        )
        for name in LANDMARK_INDICES:
            payload[f"{side}_{name}"] = np.asarray(
                sides[side][name], dtype=np.float32
            )
    payload["_schema_version"] = np.asarray("1.0")
    payload["_stage"] = np.asarray("wilor_yolo_bimanual_hand_pose")
    payload["_metadata_json"] = np.asarray(
        json.dumps(
            {
                "source": str(args.video),
                "model": str(args.model),
                "conf": args.conf,
                "min_kp_conf": args.min_kp_conf,
                "temporal_filter": args.temporal_filter,
                "max_wrist_jump": args.max_wrist_jump,
                "min_lr_distance": args.min_lr_distance,
                "edge_margin": args.edge_margin,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    total = max(frame_index, 1)
    summary = {
        "frames": frame_index,
        "left_valid_ratio": float(np.mean(payload["left_valid"])),
        "right_valid_ratio": float(np.mean(payload["right_valid"])),
        "one_or_more_ratio": one_hand_frames / total,
        "two_hand_ratio": two_hand_frames / total,
        "output": str(args.output),
        "debug_video": str(args.debug_video) if args.debug_video else "",
    }
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WiLoR YOLO hand keypoints to a bimanual pose NPZ."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-video", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external_repos/WiLoR/pretrained_models/detector.pt"),
        help="Path to WiLoR detector.pt weights.",
    )
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--min-kp-conf", type=float, default=0.3)
    parser.add_argument(
        "--temporal-filter",
        action="store_true",
        help="Drop obvious left/right overlaps, edge detections, and jumps.",
    )
    parser.add_argument("--max-wrist-jump", type=float, default=0.12)
    parser.add_argument("--min-lr-distance", type=float, default=0.08)
    parser.add_argument("--edge-margin", type=float, default=0.02)
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick checks.",
    )
    return parser.parse_args()


def main() -> int:
    convert(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
