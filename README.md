# Video to OpenArm

Pipeline bimanual chuyen video/webcam hai tay nguoi thanh chuyen dong OpenArm
trong MuJoCo: 14 joint tay, hai end-effector va hai gripper.

Trang thai hien tai: giu pipeline on dinh dung MediaPipe hand landmarks,
pinch hysteresis, palm-scale depth proxy, retarget per-axis va bimanual
Jacobian DLS IK. Cac nhanh thu nghiem sau do nhu Yolo/EgoForce, world3d depth
va smoothing nang cao da bi loai khoi code chinh vi khong cho ket qua tot hon
tren factory002_middle.

## Ket qua chinh

Synthetic bimanual:

```text
frames          = 180
mean IK error   = 1.22 cm
max IK error    = 2.00 cm
IK converged    = 100%
tracking L/R    = 97.22% / 97.22%
```

Factory002 selected run:

```text
selected video  = outputs/final_videos/factory002_middle_phaseB/human_vs_robot_comparison.mp4
method          = PhaseB per-axis retarget scale
scale           = x=0.35, y=0.50, z=0.25
status          = selected as the best current factory002_middle result
```

## Pipeline

```text
Video / Webcam
-> MediaPipe left/right hand landmarks
-> independent pinch detection
-> interpolate + moving-average wrist smoothing
-> per-side wrist retarget + workspace clamp
-> bimanual Jacobian DLS IK
-> MuJoCo replay / live teleoperation
-> optional NPZ dataset
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[all,dev]"
python scripts/download_hand_model.py
python scripts/00_check_install.py
```

The MediaPipe model is not committed. It is downloaded to
`models/hand_landmarker.task`.

## Common Commands

Record a two-hand video:

```powershell
openarm-retarget record
```

Run the full offline pipeline on synthetic data:

```powershell
openarm-retarget demo --name bimanual_synthetic --frames 180 --render
```

Run the pipeline on a real video:

```powershell
openarm-retarget demo --video data/raw_videos/demo_001.mp4 --name demo_001 --render
```

Open the MuJoCo viewer:

```powershell
openarm-retarget viewer
```

Run live webcam teleoperation:

```powershell
openarm-retarget live
```

Compose human/robot comparison video:

```powershell
openarm-retarget compare `
  --human outputs/debug_videos/demo_001_bimanual_hand_debug.mp4 `
  --robot outputs/replay_videos/demo_001_bimanual_openarm.mp4 `
  --output outputs/comparison/demo_001_human_vs_robot.mp4
```

## Maintained Scripts

The maintained step scripts are `scripts/00_check_install.py` through
`scripts/12_create_comparison_video.py`, plus `download_hand_model.py`,
`download_egoworld.py`, `generate_synthetic_hand_pose.py`, and `read_quality.py`.

Experimental Yolo/EgoForce/world3d helper scripts were removed from the tracked
repo because they are not part of the selected PhaseB pipeline.

## Reports

- `PROJECT_REPORT.md`: compact current status and accepted result.
- `PROJECT_PLAN.md`: original project plan and architecture.
- `ACCURACY_IMPROVEMENT_PLAN_2026-06-18.md`: short archive of the accuracy
  experiments and the decision to keep PhaseB.

## Tests

```powershell
pytest -q
```
