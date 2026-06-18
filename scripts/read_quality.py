"""Read and display quality reports."""
import json
import sys

files = sys.argv[1:] if len(sys.argv) > 1 else [
    "outputs/factory002_end_improved_bimanual_quality_report.json",
    "outputs/factory002_end_world3d_final_bimanual_quality_report.json",
]

for f in files:
    try:
        q = json.load(open(f))
        name = q.get("name", "?")
        mean_err = q.get("mean_ik_error_m", -1)
        max_err = q.get("max_ik_error_m", -1)
        conv = q.get("ik_converged_ratio", -1)
        left_err = q.get("left", {}).get("mean_ik_error_m", -1)
        right_err = q.get("right", {}).get("mean_ik_error_m", -1)
        print(f"{name}: mean={mean_err:.4f}m max={max_err:.4f}m conv={conv:.1%} L={left_err:.4f}m R={right_err:.4f}m")
    except Exception as e:
        print(f"ERROR reading {f}: {e}")