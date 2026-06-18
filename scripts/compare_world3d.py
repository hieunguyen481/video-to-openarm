"""Compare world3d depth vs palm_scale baseline."""
import json
from openarm_retarget.pipeline import run_bimanual_video

# Run with world landmarks
_, quality = run_bimanual_video(
    "data/raw_videos/factory002_end.mp4",
    name="factory002_end_world3d_rescaled",
    render=False,
)

# Load baseline
with open("outputs/factory002_end_improved_bimanual_quality_report.json") as f:
    baseline = json.load(f)

print("=== SO SANH: WORLD Z (raw) vs BASELINE (palm_scale) ===")
print(f"Video: factory002_end.mp4 ({quality['frames']} frames)")
print()
for side in ("left", "right"):
    b = baseline[side]
    q = quality[side]
    diff = q["mean_ik_error_m"] - b["mean_ik_error_m"]
    arrow = "↓" if diff < 0 else "↑"
    print(f"--- {side.upper()} ---")
    print(f"  Mean IK Error:  {b['mean_ik_error_m']:.4f}m -> {q['mean_ik_error_m']:.4f}m  ({arrow} {abs(diff):.4f}m)")
    print(f"  Max IK Error:   {b['max_ik_error_m']:.4f}m -> {q['max_ik_error_m']:.4f}m")
print()
print(f"--- TONG ---")
print(f"  Mean IK Error: {baseline['mean_ik_error_m']:.4f}m -> {quality['mean_ik_error_m']:.4f}m")
print(f"  IK Converged:  {baseline['ik_converged_ratio']:.2%} -> {quality['ik_converged_ratio']:.2%}")