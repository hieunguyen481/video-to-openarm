# Project Report - Video to OpenArm Bimanual

Updated after cleanup: factory002_middle_phaseB is the selected factory run.

## Scope

The project converts human two-hand motion into OpenArm bimanual motion in
MuJoCo:

```text
video/webcam
-> left/right MediaPipe landmarks
-> left/right pinch commands
-> wrist smoothing
-> per-side target retargeting
-> bimanual Jacobian DLS IK
-> MuJoCo replay, comparison video, or live teleoperation
```

The maintained pipeline controls:

- 7 left arm joints and left gripper.
- 7 right arm joints and right gripper.
- Two end-effector targets in one MuJoCo model state.

## Selected Result

Keep this artifact as the best current factory002_middle comparison:

```text
outputs/final_videos/factory002_middle_phaseB/human_vs_robot_comparison.mp4
```

Selected method:

```text
PhaseB per-axis retargeting
scale x = 0.35
scale y = 0.50
scale z = 0.25
depth source = palm_scale proxy
IK = bimanual Jacobian DLS
```

Later experiments using extra smoothing, Yolo/EgoForce adapters, and world3d
depth were not kept because they did not improve the chosen factory002_middle
result.

## Synthetic Acceptance

`openarm-retarget demo --name bimanual_synthetic --frames 180 --render`

```text
frames            180
left tracking     97.22%
right tracking    97.22%
left mean error   1.158 cm
right mean error  1.282 cm
combined mean     1.220 cm
max error         1.999 cm
IK converged      100%
```

## Live Teleoperation

The live path remains available:

```powershell
openarm-retarget live
```

Latest local benchmark:

```text
processing FPS      19.90
mean latency        58.83 ms
p95 latency         93.00 ms
MediaPipe backend   tasks / CPU
inference width     640
```

## Maintained Code

Core modules kept:

- `hand_tracking.py`: MediaPipe hand landmarks.
- `pinch.py`: hysteresis pinch detection.
- `smoothing.py`: interpolation, moving average, velocity clamp.
- `retargeting.py`: fixed per-axis retargeting with workspace clamp.
- `ik_solver.py`: single-arm and bimanual Jacobian DLS IK.
- `mujoco_replay.py`: kinematic/actuator replay.
- `live_teleop.py` and `live_control.py`: webcam teleoperation.
- `dataset.py`: NPZ dataset export.

Removed from the maintained path:

- Yolo/EgoForce adapter scripts.
- World3d comparison/render scripts.
- Accuracy-improvement tests for abandoned functions.
- Obsolete generated videos except the selected PhaseB comparison.

## Remaining Limitations

- No collision avoidance between the two OpenArm arms.
- No real OpenArm hardware control.
- Factory videos still depend strongly on camera placement, tracking quality,
  handedness stability, and retarget scale.
- The selected PhaseB result is a practical best-so-far, not a final accuracy
  guarantee across all videos.
