# Báo cáo triển khai Video to OpenArm Bimanual

**Ngày cập nhật:** 11/06/2026
**Môi trường:** Windows, Python 3.12.10, MuJoCo 3.9.0,
openarm-mujoco 2.0.0, MediaPipe 0.10.35, OpenCV 4.13.0

## 1. Phạm vi hoàn thành

Pipeline hiện điều khiển đồng thời:

- 7 joint tay trái + gripper trái.
- 7 joint tay phải + gripper phải.
- Hai end-effector trên cùng một trạng thái MuJoCo.

Luồng xử lý:

```text
Video hai tay
-> left/right hand landmarks
-> left/right pinch + smoothing
-> left/right robot targets
-> bimanual Jacobian IK (6D error, 14 joints)
-> two-gripper replay
-> bimanual dataset
```

API một tay cũ vẫn được giữ để tương thích, nhưng CLI `openarm-retarget demo`
mặc định chạy bimanual.

### Tiến độ theo kế hoạch

| Tuần | Nội dung kế hoạch | Trạng thái |
|---|---|---|
| Tuần 1 | OpenArm MuJoCo, inspect model, hand tracking | Hoàn thành |
| Tuần 2 | Pinch, smoothing, retargeting | Hoàn thành cho hai tay |
| Tuần 3 | Bimanual IK, replay, hai gripper | Hoàn thành |
| Tuần 4 | Tối ưu, dataset, baseline, README/report, demo | Hoàn thành offline |
| Tuần 5 | Live webcam, stateful IK, MuJoCo realtime | Hoàn thành |

Ngoài kế hoạch 4 tuần đã có thêm:

- Schema bimanual `left_*` và `right_*`.
- Jacobian gộp 6D cho 14 joint.
- Hai gripper độc lập.
- CLI mở viewer: `openarm-retarget viewer`.

Chưa hoàn thành:

- Collision avoidance giữa hai cánh tay.
- Điều khiển robot OpenArm phần cứng thật.

Đã bổ sung công cụ thu video hai tay từ webcam:

```bash
openarm-retarget record
```

Live teleoperation:

```bash
openarm-retarget live
```

Kiến trúc:

```text
MSMF latest-frame camera
-> MediaPipe Tasks VIDEO / XNNPACK CPU
-> live pinch + smoothing + tracking-loss safety
-> warm-start bimanual Jacobian IK
-> integrated human + MuJoCo display
-> optional live NPZ dataset
```

Benchmark trên i7-12700H, RTX 3060 Laptop, Windows:

| Chế độ | FPS xử lý | Mean latency | P95 latency |
|---|---:|---:|---:|
| Tracking + IK | 22.46 | 28.77 ms | 78.00 ms |
| Camera + MuJoCo | 12.67 | 57.93 ms | 94.00 ms |

GPU delegate MediaPipe đã được thử trực tiếp nhưng wheel Windows báo
`GPU processing is disabled in build flags`. Vì vậy inference dùng XNNPACK
CPU; RTX 3060 chưa được dùng trong phiên bản Python hiện tại.

## 2. Kết quả nghiệm thu bimanual

Lệnh:

```bash
openarm-retarget demo --name bimanual_synthetic --frames 180 --render
```

| Chỉ số | Tay trái | Tay phải | Tiêu chí |
|---|---:|---:|---:|
| Tracking hợp lệ | 97.22% | 97.22% | >= 90% |
| Gripper transitions | 4 | 4 | 2 close + 2 open |
| Mean IK error | 1.158 cm | 1.282 cm | < 5 cm |
| Max IK error | 1.981 cm | 1.999 cm | < 20 cm |
| IK đồng thời hội tụ | 100% | 100% | Không diverge |

Chỉ số gộp:

```text
mean IK error = 1.220 cm
max IK error  = 1.999 cm
frames        = 180
video         = 960x720, 30 FPS
```

## 3. OpenArm model

```text
LEFT
  joints: openarm_left_joint1 ... openarm_left_joint7
  EE: left_ee_control_point
  gripper: left_finger1_ctrl
  home: [0.401, 0.1535, 1.12] m

RIGHT
  joints: openarm_right_joint1 ... openarm_right_joint7
  EE: right_ee_control_point
  gripper: right_finger1_ctrl
  home: [0.401, -0.1535, 1.12] m
```

Gripper phải dùng control range âm. Replay đã được sửa để `0=open`, `1=closed`
cho cả hai bên thay vì nội suy trực tiếp theo thứ tự min/max.

## 4. Data schema

Hand pose:

```text
left_valid, left_wrist, left_thumb_tip, ...
right_valid, right_wrist, right_thumb_tip, ...
```

Robot trajectory:

```text
qpos [T, 19]
left_arm_qpos [T, 7]
right_arm_qpos [T, 7]
left_ee_pos/right_ee_pos [T, 3]
left_gripper_cmd/right_gripper_cmd [T]
```

Dataset action:

```text
action_left_arm_joint_target [T, 7]
action_right_arm_joint_target [T, 7]
action_left_gripper_cmd [T, 1]
action_right_gripper_cmd [T, 1]
```

## 5. Baseline

Temporal split: 143 train / 36 test sample.

| Chỉ số | Kết quả |
|---|---:|
| 14-joint arm RMSE | 0.2315 rad |
| Left gripper accuracy | 63.89% |
| Right gripper accuracy | 75.00% |

Đây chỉ là sanity check trên một trajectory synthetic nhỏ. Các chỉ số gripper
chưa đủ để xem là policy học tốt; cần nhiều episode video thật và dữ liệu cân
bằng hơn.

## 6. Kiểm thử và artefact

```text
19 passed
```

- [Video bimanual MuJoCo](docs/demo/bimanual_openarm_replay.mp4)
- [Frame hai tay](docs/images/bimanual_replay_frame.png)
- [Left pinch](docs/images/left_pinch_detection.png)
- [Right pinch](docs/images/right_pinch_detection.png)
- [Left IK error](docs/images/left_ik_error.png)
- [Right IK error](docs/images/right_ik_error.png)

## 7. Giới hạn còn lại

Chưa có video hai tay thật trong workspace, nên chưa đo được:

- Tỷ lệ MediaPipe giữ đúng handedness khi hai tay giao nhau.
- Ngưỡng pinch phù hợp với camera/người vận hành thật.
- Collision giữa hai tay robot khi target tiến gần nhau.
- Chất lượng actuator replay khi chuyển động nhanh.

`mirror_input: true` được bật mặc định để webcam hoạt động theo kiểu gương.
Nếu nguồn video đã mirror sẵn, đổi thành `false`.

## 8. Nghiệm thu video thật `demo_001.mp4`

Video quay ngày 11/06/2026:

```text
327 frames
1280x720
30 FPS
10.9 seconds
```

Kết quả:

| Chỉ số | Tay trái | Tay phải |
|---|---:|---:|
| Tracking hợp lệ | 97.86% | 99.69% |
| Gripper transitions | 10 | 4 |
| Mean IK error | 1.29 cm | 1.08 cm |
| Max IK error | 2.00 cm | 2.14 cm |

IK gộp:

```text
mean error = 1.19 cm
max error  = 2.14 cm
converged  = 100%
```

Video thật phát hiện hai vấn đề mà synthetic không thể hiện:

1. Tay vừa vào khung tạo một pinch giả dài 1 frame. Đã thêm xác nhận chuyển
   trạng thái trong 3 frame, loại xung giả và giảm transitions tay phải từ 6
   xuống 4.
2. Tọa độ `z` của wrist landmark là mốc tương đối nên gần như không đổi. Đã
   thay tín hiệu độ sâu bằng `palm_scale` (kích thước lòng bàn tay trên ảnh).
   Sau sửa, target depth thay đổi khoảng 6.5 cm bên trái và 9.7 cm bên phải.
3. Quy ước gripper của model OpenArm ngược với giả định ban đầu: joint `0` là
   đóng, còn biên `+0.7854` bên trái và `-0.7854` bên phải là mở. Replay đã
   sửa thành `gripper_cmd = 0` mở, `gripper_cmd = 1` đóng và gán đồng thời hai
   finger joint.
4. Camera thật đặt đối diện người nên trái/phải trên video ngược với giải phẫu
   người. Pipeline giữ ảnh gốc, sau đó ánh xạ chéo tay phải thật -> robot trái,
   tay trái thật -> robot phải và đảo dấu trục ngang. Nhờ đó phía trái/phải
   của video khớp trực quan với `camera_ceiling` trong MuJoCo. Trục dọc điều
   khiển cao/thấp, `palm_scale` điều khiển tiến/lùi.

Kiểm tra lại với `demo_002.mp4`:

```text
left tracking  = 99.44%
right tracking = 99.16%
mean IK error  = 1.23 cm
max IK error   = 2.00 cm
converged      = 100%
```

Artefact:

```text
outputs/debug_videos/demo_001_real_bimanual_hand_debug.mp4
outputs/replay_videos/demo_001_real_bimanual_openarm.mp4
outputs/demo_001_real_bimanual_quality_report.json
outputs/comparison/demo_002_human_vs_robot.mp4
```
