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
