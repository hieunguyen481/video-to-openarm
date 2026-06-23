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

Prepare public assets:

```powershell
python scripts/prepare_hawor_public_assets.py
```

Run from the project root:

```powershell
python scripts/check_hawor_setup.py
```

The checker reports every missing dependency or asset and prints the expected
demo command once the setup is ready.

Current local progress:

- HaWoR checkpoint, infiller, and model config are downloaded.
- WiLoR detector, DROID-SLAM weight, and Metric3D weight are downloaded.
- A local `ffmpeg.exe` can be placed in `external_repos/HaWoR/` and is accepted
  by the checker.
- MANO files are still required from the official MANO website.
- PyTorch3D still needs a compatible install. On Windows this usually requires
  either a matching prebuilt wheel or a build environment with Visual Studio C++
  tools and CUDA toolkit (`cl` and `nvcc` on PATH).

## HaWoR Demo Command

When the checker passes, run from the HaWoR checkout:

```powershell
python scripts/run_hawor_demo.py --video example/video_0.mp4 --vis-mode world
```

Camera-view visualization:

```powershell
python scripts/run_hawor_demo.py --video example/video_0.mp4 --vis-mode cam
```

The runner sets `YOLO_CONFIG_DIR` inside the project and prepends the HaWoR root
to `PATH` so the local `ffmpeg.exe` can be found.

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

## Required Manual Asset

MANO is license-gated and must be downloaded manually from:

```text
https://mano.is.tue.mpg.de
```

Place the files as:

```text
external_repos/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl
external_repos/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
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
