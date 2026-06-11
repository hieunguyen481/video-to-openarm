# Báo cáo triển khai Video to OpenArm

**Ngày nghiệm thu:** 11/06/2026  
**Môi trường:** Windows, Python 3.12.10, MuJoCo 3.9.0,
openarm-mujoco 2.0.0, MediaPipe 0.10.35, OpenCV 4.13.0

## 1. Phạm vi hoàn thành

Pipeline đã triển khai đủ các module trong `PROJECT_PLAN.md`:

1. Kiểm tra môi trường.
2. Inspect OpenArm MJCF.
3. Hand tracking video bằng MediaPipe Tasks.
4. Pinch detection có hysteresis.
5. Nội suy và smoothing wrist.
6. Retarget vào workspace OpenArm.
7. Jacobian damped least-squares IK.
8. Replay kinematic/actuator và render MP4.
9. Dataset logging.
10. Baseline MLP + gripper classifier.

## 2. Kết quả nghiệm thu synthetic

Lệnh:

```bash
openarm-retarget demo --name final_synthetic --frames 180 --render
```

| Chỉ số | Kết quả | Tiêu chí |
|---|---:|---:|
| Số frame | 180 | 180 |
| Tracking hợp lệ | 97.22% | >= 90% |
| Khoảng mất tracking cố ý | 5 frame | Nội suy thành công |
| Gripper transitions | 4 | 2 close + 2 open |
| Mean IK error | 1.699 cm | < 5 cm |
| Max IK error | 1.998 cm | < 20 cm |
| IK converged | 100% | Không diverge |
| Replay video | 180 frame, 960x720, 30 FPS | Đọc lại được |
| Dataset | 180 sample | Không NaN/Inf |

Workspace target:

```text
min = [0.3022, 0.1058, 1.0643] m
max = [0.4942, 0.1535, 1.1833] m
```

Toàn bộ target nằm trong giới hạn `configs/retarget.yaml`.

## 3. Model OpenArm được xác nhận

```text
MJCF: cell.xml
nq/nv/nu: 19/19/17
EE site: left_ee_control_point
Arm joints: openarm_left_joint1 ... openarm_left_joint7
Arm actuators: left_joint1_ctrl ... left_joint7_ctrl
Gripper actuator: left_finger1_ctrl
Home EE: [0.401, 0.1535, 1.12] m
```

Origin trong kế hoạch cũ (`z=0.35`) không khớp OpenArm v2. Config triển khai đã
được hiệu chỉnh theo home pose thực tế (`z=1.12`).

## 4. Baseline learning

Temporal split: 143 train / 36 test sample.

| Chỉ số | Kết quả |
|---|---:|
| Arm RMSE | 0.1521 rad |
| Gripper accuracy | 97.22% |

Baseline chỉ dùng để xác nhận dataset có thể học; IK vẫn là controller chính.

## 5. Kiểm thử

```text
12 passed
```

Các test gồm unit, integration trên model OpenArm thật và pipeline synthetic
end-to-end. GitHub Actions chạy Python 3.12 với dependency simulation.

## 6. Artefact trực quan

- [Video MuJoCo synthetic](docs/demo/synthetic_openarm_replay.mp4)
- [Pinch detection](docs/images/pinch_detection.png)
- [Wrist smoothing](docs/images/wrist_smoothing.png)
- [Target trajectory](docs/images/target_trajectory.png)
- [IK error](docs/images/ik_error.png)

## 7. Giới hạn còn lại

Workspace ban đầu không có video tay thật. Vì vậy:

- MediaPipe Tasks, model và video code path đã khởi tạo thành công.
- Chưa thể đo `valid_tracking_ratio` hoặc tune threshold trên người/camera thật.
- Demo synthetic không thay thế bước calibration theo camera và khoảng cách tay.

Khi thêm video thật vào `data/raw_videos/`, lệnh end-to-end trong README tạo
quality report riêng để quyết định có cần chỉnh `pinch.yaml` và
`retarget.yaml` hay không.

