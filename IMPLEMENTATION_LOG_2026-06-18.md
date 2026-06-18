# Nhật ký triển khai cải thiện độ chính xác — 18/06/2026

## Tóm tắt

Đã triển khai 3 cải thiện chính cho pipeline video-to-openarm:
1. **Giai đoạn 1: Tự động hiệu chuẩn Scale & Origin** — loại bỏ việc tinh chỉnh thủ công
2. **Giai đoạn 3A: Loại bỏ ngoại lai (Outlier Rejection)** — ngăn glitch theo dõi lan truyền
3. **Giai đoạn 4A: Null-Space IK & Damping thích ứng** — cải thiện hội tụ và ổn định IK

Tất cả 62 test đều pass (40 cũ + 22 mới).

---

## Giai đoạn 1: Tự động hiệu chuẩn Scale & Origin

### Vấn đề
Giá trị scale/origin cố định trong `retarget.yaml` gây ra:
- Target tràn workspace khi chuyển động người lớn → IK thất bại
- Target hầu như không di chuyển khi chuyển động người nhỏ → robot cử động kém
- Phải tinh chỉnh thủ công cho mỗi video mới

### Giải pháp
Thêm 3 hàm mới vào `src/openarm_retarget/retargeting.py`:

1. **`auto_calibrate_scale()`** — Tính scale theo từng trục từ biên độ chuyển động:
   - Dùng percentile (5th-95th) để bỏ qua outlier
   - Map biên độ chuyển động người vào `workspace_utilization × robot_room`
   - Giới hạn trong `[min_scale, max_scale]`
   - Mặc định: sử dụng 85% workspace

2. **`auto_calibrate_origin()`** — Tính origin từ vị trí median cổ tay:
   - Map vị trí median của cổ tay người vào tâm workspace robot
   - Điều chỉnh theo axis mapping và dấu
   - Đảm bảo targets nằm giữa workspace

3. **`retarget_wrist_auto()`** — Entry point thống nhất:
   - Khi `auto_scale: true`, tính scale VÀ tự điều chỉnh origin
   - Khi `auto_origin: true`, chỉ tính origin
   - Fallback về `retarget_wrist()` khi cả hai đều false

### Thay đổi config
- `configs/retarget.yaml`: Thêm `auto_scale`, `auto_origin`, `workspace_utilization`, `max_scale`, `min_scale`
- `configs/bimanual_retarget.yaml`: Tương tự cho bên trái/phải
- Mặc định: `auto_scale: false` (bật khi xử lý video thực)

### File đã sửa
- `src/openarm_retarget/retargeting.py` — Hàm mới
- `src/openarm_retarget/pipeline.py` — Dùng `retarget_wrist_auto()` thay `retarget_wrist()`
- `configs/retarget.yaml` — Config mới
- `configs/bimanual_retarget.yaml` — Config mới

---

## Giai đoạn 3A: Loại bỏ ngoại lai

### Vấn đề
MediaPipe đôi khi tạo ra các bước nhảy đột ngột (landmark glitch) gây:
- Vị trí target không thực tế
- IK phân kỳ hoặc tạo chuyển động giật
- Lan truyền qua smoothing thành các transient lớn

### Giải pháp
Thêm `reject_outliers()` vào `src/openarm_retarget/smoothing.py`:

- So sánh các frame liên tiếp hợp lệ
- Đánh dấu frame có displacement > `max_jump` là không hợp lệ
- Cũng loại bỏ giá trị NaN/Inf
- Tích hợp vào pipeline `smooth_wrist()` (chạy trước interpolation)

### Tham số
- `max_jump: 0.15` — Displacement tối đa giữa các frame liên tiếp (tọa độ ảnh chuẩn hóa)

### File đã sửa
- `src/openarm_retarget/smoothing.py` — Hàm `reject_outliers()` mới, cập nhật `smooth_wrist()`

---

## Giai đoạn 4A: Null-Space IK & Damping thích ứng

### Vấn đề
IK Damped Least Squares hiện tại có 2 vấn đề:
1. **Gần singularities**: Damping thấp → vận tốc khớp lớn → bất ổn
2. **Lãng phí redundancy**: Tay 7-DOF với tác vụ 3-DOF → 4 DOF không dùng, dẫn đến cấu hình khớp khó đoán

### Giải pháp
Thêm vào `src/openarm_retarget/ik_solver.py`:

1. **`_compute_null_space_step()`** — Tận dụng redundancy động học:
   - Chiếu tư thế ưu tiên (khớp mid-range) lên null-space của Jacobian
   - Di chuyển khớp về giữa range mà không ảnh hưởng end-effector
   - Trọng số `null_space_weight` điều khiển cường độ (0 = tắt)

2. **`_adaptive_damping()`** — Điều chỉnh damping theo condition number của Jacobian:
   - Condition tốt (cond < 100): damping cơ bản → hội tụ nhanh
   - Trung bình (100 < cond < 10⁴): damping × 1.5
   - Gần singular (10⁴ < cond < 10⁶): damping × 3
   - Rất gần singular (cond > 10⁶): damping × 10 → ổn định

3. **Tính tư thế ưu tiên** — Trong cả `JacobianIKSolver` và `BimanualJacobianIKSolver`:
   - Tính vị trí khớp mid-range từ `model.jnt_range`
   - Dùng làm mục tiêu null-space

### Thay đổi config
- `configs/ik.yaml`: Thêm `null_space_weight: 0.0` và `adaptive_damping: false`
- Mặc định: tắt (bật khi xử lý video thực)
- Khuyến nghị cho video thực: `null_space_weight: 0.1`, `adaptive_damping: true`

### File đã sửa
- `src/openarm_retarget/ik_solver.py` — Hàm mới, cập nhật cả 2 IK solver
- `configs/ik.yaml` — Config mới

---

## Kiểm thử

### File test mới
`tests/test_accuracy_improvements.py` — 22 test bao phủ:

| Lớp | Số test | Giai đoạn |
|-----|---------|-----------|
| TestAutoCalibrateScale | 5 | Giai đoạn 1 |
| TestAutoCalibrateOrigin | 2 | Giai đoạn 1 |
| TestRetargetWristAuto | 3 | Giai đoạn 1 |
| TestRejectOutliers | 5 | Giai đoạn 3A |
| TestSmoothWristWithOutlierRejection | 1 | Giai đoạn 3A |
| TestNullSpaceStep | 3 | Giai đoạn 4A |
| TestAdaptiveDamping | 3 | Giai đoạn 4A |

### Kết quả toàn bộ
```
62 passed in 4.55s
```

---

## Cách bật cải thiện

Để xử lý video thực, chỉnh config:

**`configs/retarget.yaml`** (tay đơn):
```yaml
auto_scale: true    # Bật tự động hiệu chuẩn scale
auto_origin: false  # Thường không cần nếu auto_scale đã xử lý
```

**`configs/bimanual_retarget.yaml`** (hai tay):
```yaml
left:
  auto_scale: true
right:
  auto_scale: true
```

**`configs/ik.yaml`**:
```yaml
null_space_weight: 0.1    # Bật tối ưu tư thế null-space
adaptive_damping: true    # Bật damping nhận biết singularities
```

---

## Kết quả đánh giá thực tế

### Video test: `factory002_end.mp4` (1793 frames)

| Metric | Trước | Sau v2 | Thay đổi |
|--------|-------|--------|----------|
| Mean IK Error | 0.053m | 0.036m | ↓ 32% |
| Max IK Error | 0.357m | 0.220m | ↓ 38% |
| IK Converged | 40.0% | 47.7% | ↑ 7.7% |
| Left Mean Error | 0.035m | 0.007m | ↓ 80% |
| Right Mean Error | 0.070m | 0.065m | ↓ 7% |

### Cấu hình tốt nhất
- Per-axis scale: x=0.35, y=0.50, z=0.25
- Null-space weight: 0.1
- Adaptive damping: true
- Auto-scale: false (quá mạnh cho video này)

### Video so sánh
- `outputs/final_videos/factory002_end_v2/human_vs_robot_comparison.mp4`

### Bài học
- `auto_scale` với `workspace_utilization=0.85` tạo scale quá lớn → target tràn workspace → IK thất bại
- Per-axis scale thủ công (từ phaseB trước đó) vẫn tốt hơn cho video cụ thể
- Null-space + adaptive damping giúp cải thiện IK convergence
- Cần điều chỉnh `auto_scale` để tính đến reachable workspace, không chỉ workspace limit

---

## Công việc tương lai còn lại

1. **Giai đoạn 2: Ước lượng độ sâu** — Dùng palm scale hoặc stereo cho độ chính xác trục Z
2. **Giai đoạn 3B: Nhất quán thời gian** — Smoothing forward-backward, bộ lọc Kalman
3. **Giai đoạn 5: IK warm-start đa frame** — Dùng qpos frame trước làm khởi tạo
4. **Giai đoạn 6: Framework đánh giá** — Metric độ chính xác tự động trên video benchmark
5. **Tự động tuning per-video** — Cờ CLI để bật auto_scale cho từng lần chạy

---

## Phase D: Tích hợp MediaPipe World Landmarks (3D Depth Thật)

### Phát hiện quan trọng
MediaPipe HandLandmarker **đã cung cấp sẵn** `hand_world_landmarks` (tọa độ 3D thực tế) nhưng code cũ **không sử dụng**! Không cần thêm PyTorch/EgoForce.

### Thay đổi code
1. **`hand_tracking.py`**: Sửa `detect()` trả về `result.hand_world_landmarks` (thêm tham số thứ 3)
2. **`bimanual.py`**: Thêm `WORLD_LANDMARK_KEYS` và truyền `world_*` qua `side_pose()`
3. **`pipeline.py`**: Ưu tiên dùng `world_wrist.z` (rescale matching x,y amplitude) thay vì `palm_scale`

### Kỹ thuật rescale
- World z ở đơn vị meters (~0.04-0.10m), x,y normalized [0,1]
- Rescale: `z_new = (z - z_center) * (xy_range / z_range) + 0.5`
- Giữ nguyên retargeting scale/origin config

### Kết quả so sánh (factory002_end.mp4, 1793 frames)

| Approach | Left Mean | Right Mean | Total Mean | IK Converged |
|---|---|---|---|---|
| Baseline (palm_scale) | 0.0457m | 0.0583m | 0.0520m | 27.05% |
| **+ World z (rescaled)** | **0.0103m** | **0.0598m** | **0.0351m** | **47.74%** |

**Cải thiện: Left ↓77%, Total ↓33%, IK Converged ↑77%**

### Test: 62/62 pass
