# Video to OpenArm

Pipeline bimanual chuyen video/webcam hai tay nguoi thanh chuyen dong OpenArm
trong MuJoCo: 14 joint tay, hai end-effector va hai gripper.

Trang thai hien tai: giu pipeline on dinh dung MediaPipe hand landmarks,
pinch hysteresis, palm-scale depth proxy, retarget per-axis va bimanual
Jacobian DLS IK. Voi factory002_middle, WiLoR YOLO keypoints + preset tuned
hien la ket qua cai thien tot nhat; PhaseB duoc giu lam baseline on dinh.

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
baseline video  = outputs/final_videos/factory002_middle_phaseB/human_vs_robot_comparison.mp4
baseline        = PhaseB per-axis retarget scale
selected preset = configs/presets/wilor_yolo_tuned/
selected result = WiLoR YOLO tuned, mean IK error 1.22 cm, converged 93.1%
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

Extract WiLoR YOLO keypoints to the bimanual pose schema:

```powershell
python scripts/extract_wilor_yolo_pose.py `
  --video data/raw_videos/factory002_middle.mp4 `
  --model external_repos/WiLoR/pretrained_models/detector.pt `
  --output outputs/external_trials/wilor_yolo/factory002_middle_wilor_yolo_hand_pose.npz `
  --debug-video outputs/external_trials/wilor_yolo/factory002_middle_wilor_yolo_hand_debug.mp4
```

Run the selected tuned OpenArm replay from the extracted WiLoR pose:

```powershell
python scripts/run_factory002_wilor_tuned.py
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
`download_egoworld.py`, `generate_synthetic_hand_pose.py`, `read_quality.py`,
`extract_wilor_yolo_pose.py`, and `run_factory002_wilor_tuned.py`.

Experimental Yolo/EgoForce/world3d helper scripts were removed from the tracked
repo. WiLoR YOLO extraction is now tracked because it produced the selected
factory002_middle improvement without requiring MANO or full 3D reconstruction.
HaWoR remains a future egocentric/world-space 3D direction because it needs a
heavier dependency stack and external assets.

## Reports

- `PROJECT_REPORT.md`: compact current status and accepted result.
- `PROJECT_PLAN.md`: original project plan and architecture.
- `ACCURACY_IMPROVEMENT_PLAN_2026-06-18.md`: short archive of the accuracy
  experiments and the decision to keep PhaseB.

## Tests

```powershell
pytest -q
```
