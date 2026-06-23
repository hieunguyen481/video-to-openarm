# Project Report - Video to OpenArm Bimanual

Updated: 2026-06-23

## Goal

Convert human two-hand motion from video into OpenArm bimanual motion in
MuJoCo:

```text
video / extracted hand pose
-> left/right landmarks
-> pinch commands
-> wrist smoothing
-> per-side target retargeting
-> bimanual Jacobian DLS IK
-> MuJoCo replay and comparison video
```

The maintained pipeline controls 7 joints and 1 gripper per arm in a shared
OpenArm MuJoCo model.

## Current Decision

Keep PhaseB as the stable baseline artifact:

```text
outputs/final_videos/factory002_middle_phaseB/human_vs_robot_comparison.mp4
```

Use WiLoR YOLO tuned as the selected improved result for `factory002_middle`.
The tuned result uses WiLoR's YOLO hand keypoints as the detector input, then
keeps the existing OpenArm retargeting and IK pipeline.

Selected preset:

```text
configs/presets/wilor_yolo_tuned/
```

Key tuned values:

```text
bimanual retarget scale.y = 0.35
ik max_frame_delta_q      = 0.15
```

## Final Metrics

```text
PhaseB baseline:
mean=0.0225 m  max=0.1807 m  converged=71.0%
left=0.0288 m  right=0.0163 m

WiLoR YOLO raw:
mean=0.0201 m  max=0.1921 m  converged=73.7%
left=0.0251 m  right=0.0151 m

WiLoR YOLO tuned:
mean=0.0122 m  max=0.1189 m  converged=93.1%
left=0.0137 m  right=0.0107 m

WiLoR post-filter tuned:
mean=0.0124 m  max=0.1194 m  converged=88.6%
left=0.0140 m  right=0.0107 m
```

The selected tuned preset reduces mean IK error from 2.25 cm to 1.22 cm versus
PhaseB and improves convergence from 71.0% to 93.1%.

## Demo Videos

Primary comparison:

```text
outputs/external_trials/method_comparison/factory002_middle_phaseB_vs_wilor_yolo_vs_tuned_variants.mp4
```

Individual tuned comparison:

```text
outputs/external_trials/wilor_yolo_scaleY035_fdq015/factory002_middle_wilor_yolo_scaleY035_fdq015_human_vs_robot.mp4
```

Post-filter tuned comparison, useful for checking the few left-hand detection
failure frames:

```text
outputs/external_trials/wilor_yolo_postfiltered_scaleY035_fdq015/factory002_middle_wilor_yolo_postfiltered_scaleY035_fdq015_human_vs_robot.mp4
```

These videos are local generated artifacts and are intentionally ignored by
git.

## Reproduce The Selected Run

After exporting WiLoR YOLO hand pose to:

```text
outputs/external_trials/wilor_yolo/factory002_middle_wilor_yolo_hand_pose.npz
```

run:

```powershell
python scripts/run_factory002_wilor_tuned.py
```

The script writes the OpenArm trajectory, quality report, robot replay video,
and a side-by-side comparison video if the WiLoR debug video is available.

## External Repository Findings

WiLoR:

- Useful immediately because its detector returns 21 two-dimensional hand
  keypoints and handedness labels.
- Full 3D reconstruction was blocked locally by missing MANO assets.
- The detector output was sufficient to improve the current OpenArm replay
  after retarget and IK tuning.

HaWoR:

- Promising for egocentric world-space hand reconstruction.
- Not used in the selected result because local execution requires a heavier
  stack: MANO assets, PyTorch3D, DROID-SLAM/CUDA components, and depth model
  weights.

## Maintained Code

Core modules:

- `hand_tracking.py`: MediaPipe hand landmarks for the original pipeline.
- `pinch.py`: hysteresis pinch detection.
- `smoothing.py`: interpolation, moving average, velocity clamp.
- `retargeting.py`: fixed per-axis retargeting with workspace clamp.
- `ik_solver.py`: single-arm and bimanual Jacobian DLS IK.
- `mujoco_replay.py`: kinematic/actuator replay.
- `comparison_video.py`: human-vs-robot video composition.
- `live_teleop.py` and `live_control.py`: webcam teleoperation.
- `dataset.py`: NPZ dataset export.

Local/ignored areas:

- `external_repos/`: cloned external repos and experiment-only scripts.
- `external_weights/`: downloaded external model weights.
- `outputs/external_trials/`: generated trial videos and reports.
- `report_local/`: local report material not intended for git.

## Remaining Work

- Verify the tuned result visually on more videos, not only `factory002_middle`.
- Decide whether to use the post-filter variant when left-hand detector glitches
  are visually obvious.
- Add a tracked WiLoR adapter only if the WiLoR dependency/weight/MANO setup is
  accepted as part of the official project workflow.
- Add collision avoidance and real OpenArm hardware control later if the project
  moves beyond MuJoCo replay.
