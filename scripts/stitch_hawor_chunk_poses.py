"""Stitch converted HaWoR chunk pose NPZ files into one continuous pose."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openarm_retarget.bimanual import LANDMARK_KEYS
from openarm_retarget.io import load_npz, save_npz


SIDES = ("left", "right")


def _first_valid_index(valid: np.ndarray, values: np.ndarray) -> int | None:
    usable = np.asarray(valid, dtype=bool) & np.isfinite(values).all(axis=1)
    indices = np.flatnonzero(usable)
    return int(indices[0]) if indices.size else None


def _last_valid_index(valid: np.ndarray, values: np.ndarray) -> int | None:
    usable = np.asarray(valid, dtype=bool) & np.isfinite(values).all(axis=1)
    indices = np.flatnonzero(usable)
    return int(indices[-1]) if indices.size else None


def _apply_side_offset(
    chunk: dict[str, np.ndarray],
    side: str,
    offset: np.ndarray,
) -> None:
    for key in LANDMARK_KEYS:
        field = f"{side}_{key}"
        if field in chunk:
            chunk[field] = np.asarray(chunk[field], dtype=np.float32).copy()
            chunk[field] += offset[None, :].astype(np.float32)
    palm_key = f"{side}_palm_scale"
    wrist_key = f"{side}_wrist"
    if palm_key in chunk and wrist_key in chunk:
        # In the HaWoR adapter palm_scale is used as the pipeline depth proxy.
        chunk[palm_key] = chunk[wrist_key][:, 2].astype(np.float32)


def _renormalize_pose(
    pose: dict[str, np.ndarray],
    *,
    workspace_scale: float,
) -> None:
    if workspace_scale <= 0:
        raise ValueError("workspace_scale must be positive")
    valid_points: list[np.ndarray] = []
    for side in SIDES:
        valid = np.asarray(pose[f"{side}_valid"], dtype=bool)
        for key in LANDMARK_KEYS:
            values = np.asarray(pose[f"{side}_{key}"], dtype=np.float32)
            usable = valid & np.isfinite(values).all(axis=1)
            if np.any(usable):
                valid_points.append(values[usable])
    if not valid_points:
        return
    center = np.nanmean(np.concatenate(valid_points, axis=0), axis=0).astype(np.float32)
    for side in SIDES:
        valid = np.asarray(pose[f"{side}_valid"], dtype=bool)
        for key in LANDMARK_KEYS:
            field = f"{side}_{key}"
            values = np.asarray(pose[field], dtype=np.float32).copy()
            values = (values - center[None, :]) / workspace_scale + 0.5
            values[~valid] = np.nan
            pose[field] = values.astype(np.float32)
        palm_key = f"{side}_palm_scale"
        if palm_key in pose:
            pose[palm_key] = pose[f"{side}_wrist"][:, 2].astype(np.float32)


def _copy_pose_fields(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value).copy()
        for key, value in data.items()
        if not key.startswith("_")
    }


def _swap_sides(chunk: dict[str, np.ndarray]) -> None:
    suffixes = {
        key.removeprefix("left_")
        for key in chunk
        if key.startswith("left_") and f"right_{key.removeprefix('left_')}" in chunk
    }
    for suffix in suffixes:
        left_key = f"left_{suffix}"
        right_key = f"right_{suffix}"
        chunk[left_key], chunk[right_key] = chunk[right_key], chunk[left_key]


def _overlap_identity_cost(
    previous: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    overlap_frames: int,
    *,
    swapped: bool,
) -> float:
    costs: list[np.ndarray] = []
    for side in SIDES:
        current_side = ("right" if side == "left" else "left") if swapped else side
        before = np.asarray(previous[f"{side}_wrist"][-overlap_frames:])
        after = np.asarray(current[f"{current_side}_wrist"][:overlap_frames])
        usable = np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
        if np.any(usable):
            delta = before[usable] - after[usable]
            delta -= np.nanmedian(delta, axis=0, keepdims=True)
            costs.append(np.linalg.norm(delta, axis=1))
    return float(np.nanmedian(np.concatenate(costs))) if costs else np.inf


def _blend_overlap(
    previous: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    overlap_frames: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    alpha = (
        0.5
        - 0.5
        * np.cos(np.linspace(0.0, np.pi, overlap_frames, dtype=np.float32))
    )
    keys = sorted(set(previous) & set(current) - {"timestamps"})
    for key in keys:
        before = np.asarray(previous[key])
        after = np.asarray(current[key])
        if key.endswith("_valid"):
            overlap = before[-overlap_frames:] | after[:overlap_frames]
        else:
            shape = (overlap_frames,) + (1,) * (before.ndim - 1)
            weight = alpha.reshape(shape)
            left = before[-overlap_frames:]
            right = after[:overlap_frames]
            overlap = left * (1.0 - weight) + right * weight
            overlap = np.where(np.isfinite(left), overlap, right)
            overlap = np.where(np.isfinite(right), overlap, left)
        result[key] = np.concatenate(
            (before[:-overlap_frames], overlap, after[overlap_frames:]),
            axis=0,
        )
    return result


def stitch_chunks(
    inputs: list[Path],
    *,
    fps: float,
    align_boundaries: bool = True,
    renormalize: bool = False,
    workspace_scale: float = 0.5,
    overlap_frames: int = 0,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    if not inputs:
        raise ValueError("At least one input NPZ is required")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must not be negative")

    chunks = [_copy_pose_fields(load_npz(path)) for path in inputs]
    adjusted: list[dict[str, np.ndarray]] = []
    offsets = {side: np.zeros(3, dtype=np.float32) for side in SIDES}
    last_wrist = {side: None for side in SIDES}
    diagnostics: list[dict[str, object]] = []

    for chunk_index, chunk in enumerate(chunks):
        diag: dict[str, object] = {"chunk": chunk_index, "input": str(inputs[chunk_index])}
        if overlap_frames and adjusted:
            available = min(
                overlap_frames,
                len(adjusted[-1]["left_wrist"]),
                len(chunk["left_wrist"]),
            )
            direct_cost = _overlap_identity_cost(
                adjusted[-1], chunk, available, swapped=False
            )
            swapped_cost = _overlap_identity_cost(
                adjusted[-1], chunk, available, swapped=True
            )
            if swapped_cost < direct_cost:
                _swap_sides(chunk)
                diag["swapped_hands"] = True
            diag["identity_cost"] = min(direct_cost, swapped_cost)
        for side in SIDES:
            valid = np.asarray(chunk[f"{side}_valid"], dtype=bool)
            wrist = np.asarray(chunk[f"{side}_wrist"], dtype=np.float32)
            if align_boundaries and last_wrist[side] is not None:
                if overlap_frames and adjusted:
                    available = min(
                        overlap_frames,
                        len(adjusted[-1][f"{side}_wrist"]),
                        len(wrist),
                    )
                    before = adjusted[-1][f"{side}_wrist"][-available:]
                    after = wrist[:available]
                    usable = (
                        np.isfinite(before).all(axis=1)
                        & np.isfinite(after).all(axis=1)
                    )
                    boundary_offset = (
                        np.nanmedian(before[usable] - after[usable], axis=0)
                        if np.any(usable)
                        else None
                    )
                else:
                    first_index = _first_valid_index(valid, wrist)
                    boundary_offset = (
                        last_wrist[side] - wrist[first_index]
                        if first_index is not None
                        else None
                    )
                if boundary_offset is not None:
                    # The offset is measured against an already adjusted previous
                    # chunk, so it is the complete current-chunk transform.
                    offsets[side] = boundary_offset.astype(np.float32)
                    diag[f"{side}_boundary_offset"] = boundary_offset.tolist()
            _apply_side_offset(chunk, side, offsets[side])

            wrist_after = np.asarray(chunk[f"{side}_wrist"], dtype=np.float32)
            last_index = _last_valid_index(valid, wrist_after)
            if last_index is not None:
                last_wrist[side] = wrist_after[last_index].copy()
            diag[f"{side}_cumulative_offset"] = offsets[side].tolist()
        diagnostics.append(diag)
        adjusted.append(chunk)

    if overlap_frames:
        result = adjusted[0]
        for chunk in adjusted[1:]:
            available = min(
                overlap_frames,
                len(result["left_wrist"]),
                len(chunk["left_wrist"]),
            )
            result = _blend_overlap(result, chunk, available)
    else:
        result = {}
        keys = sorted({key for chunk in adjusted for key in chunk if key != "timestamps"})
        for key in keys:
            result[key] = np.concatenate(
                [np.asarray(chunk[key]) for chunk in adjusted], axis=0
            )
    frames = len(next(iter(result.values())))
    result["timestamps"] = np.arange(frames, dtype=np.float64) / fps
    if renormalize:
        _renormalize_pose(result, workspace_scale=workspace_scale)
    return result, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch HaWoR chunk pose NPZ files into one continuous NPZ."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Concatenate chunks without boundary wrist alignment.",
    )
    parser.add_argument(
        "--renormalize",
        action="store_true",
        help="Normalize stitched metric coordinates once after stitching.",
    )
    parser.add_argument("--workspace-scale", type=float, default=0.5)
    parser.add_argument("--overlap-frames", type=int, default=0)
    parser.add_argument("--diagnostics", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pose, diagnostics = stitch_chunks(
        args.inputs,
        fps=args.fps,
        align_boundaries=not args.no_align,
        renormalize=args.renormalize,
        workspace_scale=args.workspace_scale,
        overlap_frames=args.overlap_frames,
    )
    save_npz(
        args.output,
        pose,
        stage="hawor_world_chunked_bimanual_hand_pose",
        metadata={
            "inputs": [str(path) for path in args.inputs],
            "fps": args.fps,
            "align_boundaries": not args.no_align,
            "renormalize": args.renormalize,
            "workspace_scale": args.workspace_scale,
            "overlap_frames": args.overlap_frames,
        },
    )
    diagnostics_path = args.diagnostics or args.output.with_suffix(".diagnostics.json")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": int(len(pose["timestamps"])),
                "duration_seconds": float(pose["timestamps"][-1] + 1.0 / args.fps),
                "diagnostics": str(diagnostics_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
