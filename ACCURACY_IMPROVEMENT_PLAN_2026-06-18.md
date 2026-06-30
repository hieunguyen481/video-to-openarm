# Accuracy Experiments Archive - 2026-06-18

This file is now an archive, not an active implementation plan.

## Decision

Keep `factory002_middle_phaseB` as the best current factory002_middle result:

```text
outputs/final_videos/factory002_middle_phaseB/human_vs_robot_comparison.mp4
```

The selected configuration is the simple PhaseB per-axis retargeting path:

```text
scale x = 0.35
scale y = 0.50
scale z = 0.25
depth source = palm_scale proxy
IK = bimanual Jacobian DLS
```

## Why Other Branches Were Removed

The following experiments were useful for comparison but are no longer part of
the maintained repo:

- Extra smoothing variants.
- World3d / MediaPipe world-depth comparison scripts.
- Yolo/EgoForce adapter and inference scripts.
- Phase7/diagnostic one-off scripts.
- Tests for functions that were not kept in `src/openarm_retarget`.

They were removed because they made the repo inconsistent and did not beat the
selected PhaseB output on the target factory002_middle comparison.

## Current Maintained Direction

Use the stable pipeline:

```text
MediaPipe landmarks
-> pinch hysteresis
-> interpolate + moving-average smoothing
-> fixed per-axis retargeting
-> bimanual Jacobian DLS IK
-> MuJoCo replay/comparison
```

Future accuracy work should start from the selected PhaseB baseline and add one
change at a time, with the comparison video and quality report saved under a new
run name.
