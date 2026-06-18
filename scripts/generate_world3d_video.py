"""Generate comparison video with world3d depth."""
from openarm_retarget.pipeline import run_bimanual_video

# Run with world landmarks AND render
_, quality = run_bimanual_video(
    "data/raw_videos/factory002_end.mp4",
    name="factory002_end_world3d",
    render=True,
)

print(f"Mean IK Error: {quality['mean_ik_error_m']:.4f}m")
print(f"IK Converged: {quality['ik_converged_ratio']:.1%}")
print(f"Left: {quality['left']['mean_ik_error_m']:.4f}m")
print(f"Right: {quality['right']['mean_ik_error_m']:.4f}m")