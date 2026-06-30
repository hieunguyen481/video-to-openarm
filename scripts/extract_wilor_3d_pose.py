"""Extract camera-space bimanual 3D hand pose with the full WiLoR model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.renderer import cam_crop_to_full


LANDMARK_INDICES = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}
PALM_MCP_INDICES = (5, 9, 13, 17)
SIDES = ("left", "right")


def _empty_side() -> dict[str, list[Any]]:
    return {
        "valid": [],
        "palm_scale": [],
        "global_orient": [],
        **{name: [] for name in LANDMARK_INDICES},
    }


def _append_missing(side_data: dict[str, list[Any]]) -> None:
    side_data["valid"].append(False)
    side_data["palm_scale"].append(np.nan)
    side_data["global_orient"].append(np.full((3, 3), np.nan, dtype=np.float32))
    for name in LANDMARK_INDICES:
        side_data[name].append(np.full(3, np.nan, dtype=np.float32))


def _append_prediction(
    side_data: dict[str, list[Any]],
    joints: np.ndarray,
    global_orient: np.ndarray,
) -> None:
    side_data["valid"].append(True)
    wrist = joints[0]
    palm_scale = np.mean(
        [np.linalg.norm(joints[index] - wrist) for index in PALM_MCP_INDICES]
    )
    side_data["palm_scale"].append(float(palm_scale))
    side_data["global_orient"].append(global_orient.astype(np.float32))
    for name, index in LANDMARK_INDICES.items():
        side_data[name].append(joints[index].astype(np.float32))


def _select_detections(
    result: Any,
    confidence: float,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    for box, class_id, score in zip(boxes, classes, scores, strict=True):
        side = str(result.names.get(int(class_id), "")).lower()
        if side not in SIDES or score < confidence:
            continue
        if side not in selected or score > selected[side]["score"]:
            selected[side] = {
                "side": side,
                "box": box.astype(np.float32),
                "score": float(score),
                "is_right": 1.0 if side == "right" else 0.0,
            }
    return [selected[side] for side in SIDES if side in selected]


def _infer_frame(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    model: Any,
    model_cfg: Any,
    device: torch.device,
    rescale_factor: float,
) -> dict[str, dict[str, np.ndarray]]:
    if not detections:
        return {}
    boxes = np.stack([item["box"] for item in detections])
    right = np.asarray([item["is_right"] for item in detections], dtype=np.float32)
    dataset = ViTDetDataset(
        model_cfg,
        frame,
        boxes,
        right,
        rescale_factor=rescale_factor,
        fp16=False,
    )
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), num_workers=0)))
    batch = recursive_to(batch, device)
    with torch.no_grad():
        output = model(batch)

    multiplier = 2.0 * batch["right"] - 1.0
    pred_cam = output["pred_cam"].clone()
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    image_size = batch["img_size"].float()
    focal = (
        model_cfg.EXTRA.FOCAL_LENGTH
        / model_cfg.MODEL.IMAGE_SIZE
        * image_size.max()
    )
    camera_translation = cam_crop_to_full(
        pred_cam,
        batch["box_center"].float(),
        batch["box_size"].float(),
        image_size,
        focal,
    )
    joints = output["pred_keypoints_3d"].clone()
    joints[:, :, 0] *= multiplier[:, None]
    joints = joints + camera_translation[:, None, :]
    orientations = output["pred_mano_params"]["global_orient"].reshape(-1, 3, 3)

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for index, detection in enumerate(detections):
        predictions[detection["side"]] = {
            "joints": joints[index].detach().cpu().float().numpy(),
            "global_orient": orientations[index].detach().cpu().float().numpy(),
            "box": detection["box"],
        }
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract full WiLoR camera-space 3D hand pose from video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-video", type=Path)
    parser.add_argument(
        "--wilor-root",
        type=Path,
        default=Path("external_repos/WiLoR"),
    )
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--progress", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.wilor_root.resolve()
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)

    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    previous_cwd = Path.cwd()
    os.chdir(root)
    try:
        model, model_cfg = load_wilor(
            checkpoint_path="./pretrained_models/wilor_final.ckpt",
            cfg_path="./pretrained_models/model_config.yaml",
        )
        detector = YOLO("./pretrained_models/detector.pt")
    finally:
        os.chdir(previous_cwd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    detector = detector.to(device)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
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

    sides = {side: _empty_side() for side in SIDES}
    timestamps: list[float] = []
    frame_index = 0
    try:
        while args.max_frames is None or frame_index < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            result = detector(frame, conf=args.confidence, verbose=False)[0]
            detections = _select_detections(result, args.confidence)
            predictions = _infer_frame(
                frame,
                detections,
                model=model,
                model_cfg=model_cfg,
                device=device,
                rescale_factor=args.rescale_factor,
            )
            timestamps.append(frame_index / fps)
            for side in SIDES:
                prediction = predictions.get(side)
                if prediction is None:
                    _append_missing(sides[side])
                    continue
                _append_prediction(
                    sides[side],
                    prediction["joints"],
                    prediction["global_orient"],
                )
                if writer is not None:
                    x1, y1, x2, y2 = prediction["box"].astype(int)
                    color = (60, 210, 60) if side == "left" else (255, 150, 30)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{side} 3D",
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
            if writer is not None:
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
    for side in SIDES:
        payload[f"{side}_valid"] = np.asarray(sides[side]["valid"], dtype=bool)
        payload[f"{side}_palm_scale"] = np.asarray(
            sides[side]["palm_scale"], dtype=np.float32
        )
        payload[f"{side}_global_orient"] = np.asarray(
            sides[side]["global_orient"], dtype=np.float32
        )
        for name in LANDMARK_INDICES:
            payload[f"{side}_{name}"] = np.asarray(
                sides[side][name], dtype=np.float32
            )
    payload["_schema_version"] = np.asarray("1.0")
    payload["_stage"] = np.asarray("wilor_full_3d_camera_pose")
    payload["_metadata_json"] = np.asarray(
        json.dumps(
            {
                "source": str(video),
                "coordinate_frame": "camera",
                "units": "meters",
                "uses_pred_cam_t": True,
                "uses_pred_keypoints_3d": True,
            },
            sort_keys=True,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    args.output.chmod(0o644)

    summary = {
        "frames": frame_index,
        "left_valid_ratio": float(np.mean(payload["left_valid"])),
        "right_valid_ratio": float(np.mean(payload["right_valid"])),
        "output": str(args.output),
        "debug_video": str(args.debug_video) if args.debug_video else None,
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
