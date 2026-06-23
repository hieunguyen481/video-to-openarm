# HaWoR Integration Notes

HaWoR is the next external method to evaluate after WiLoR YOLO. Its value for
this project is world-space hand motion reconstruction from egocentric videos:

```text
egocentric video
-> HaWoR hand reconstruction
-> world-space hand translation / pose
-> OpenArm retargeting
-> bimanual IK and replay
```

## Why It Is Optional

HaWoR is heavier than the tracked WiLoR YOLO adapter. It requires:

- HaWoR model weights.
- WiLoR detector weights in HaWoR's expected folder.
- MANO right and left model files under HaWoR `_DATA`.
- DROID-SLAM installed from HaWoR `thirdparty/DROID-SLAM`.
- DROID-SLAM weights.
- Metric3D weights.
- PyTorch3D and CUDA-oriented dependencies.
- `ffmpeg` on PATH.

These assets are not committed to this repo.

## Readiness Check

Run from the project root:

```powershell
python scripts/check_hawor_setup.py
```

The checker reports every missing dependency or asset and prints the expected
demo command once the setup is ready.

## HaWoR Demo Command

When the checker passes, run from the HaWoR checkout:

```powershell
cd external_repos/HaWoR
python demo.py --video_path ./example/video_0.mp4 --vis_mode world
```

Camera-view visualization:

```powershell
python demo.py --video_path ./example/video_0.mp4 --vis_mode cam
```

## Expected HaWoR Outputs

For `example/video_0.mp4`, HaWoR writes into:

```text
external_repos/HaWoR/example/video_0/
```

Important artifacts:

```text
extracted_images/
tracks_0_N/
cam_space/
SLAM/hawor_slam_w_scale_0_N.npz
world_space_res.pth
vis_0_N/
```

`world_space_res.pth` contains the main world-space hand reconstruction:

```text
pred_trans
pred_rot
pred_hand_pose
pred_betas
pred_valid
```

## Planned OpenArm Adapter

Once HaWoR runs locally, the adapter should convert `world_space_res.pth` into
the project hand/target schema:

```text
world_space_res.pth
-> left/right world hand translations
-> normalized or metric target trajectory
-> bimanual OpenArm target NPZ
-> existing IK/replay pipeline
```

The first evaluation should compare against:

- PhaseB baseline.
- WiLoR YOLO tuned.
- HaWoR world-space retarget.

Use the same metrics:

```text
mean IK error
max IK error
IK converged ratio
left/right valid ratio
visual comparison video
```
