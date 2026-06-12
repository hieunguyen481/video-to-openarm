# OpenArm Bimanual Wrist & Pinch Retargeting — Project Plan v2

> **Phiên bản:** 2.0 | **Cập nhật triển khai:** 11/06/2026 | **Trạng thái:** MVP implemented

---

## 0. Cập nhật phạm vi bimanual

Phạm vi triển khai cuối cùng đã được mở rộng từ một tay sang **cả hai tay**:

```text
left human hand  -> 7 left arm joints  + left gripper
right human hand -> 7 right arm joints + right gripper
```

Các phần đặc tả một biến `wrist`, `target_pos`, `gripper_cmd` ở tài liệu gốc
được hiểu là chạy độc lập cho hai namespace `left_*` và `right_*`. IK cuối dùng
sai số vị trí gộp 6D và Jacobian 14 joint trên cùng một model MuJoCo. Chi tiết
kết quả thực tế nằm trong `PROJECT_REPORT.md`.

---

## Mục lục

1. [Mục tiêu dự án](#1-mục-tiêu-dự-án)
2. [Nguyên tắc triển khai](#2-nguyên-tắc-triển-khai)
3. [Repo / Code tham khảo](#3-repocode-tham-khảo-chính)
4. [Use Case Diagram](#4-use-case-diagram)
5. [Kiến trúc pipeline](#5-kiến-trúc-pipeline)
6. [Data Flow Diagram](#6-data-flow-diagram)
7. [State Machine: Pinch & Gripper](#7-state-machine-pinch--gripper)
8. [Cấu trúc repo đề xuất](#8-cấu-trúc-repo-đề-xuất)
9. [Vì sao cần cả `scripts/` và `src/`?](#9-vì-sao-cần-cả-scripts-và-src)
10. [Chi tiết từng module](#10-chi-tiết-từng-module)
11. [Thứ tự thực hiện thực tế](#11-thứ-tự-thực-hiện-thực-tế)
12. [Milestone](#12-milestone)
13. [Kế hoạch 4 tuần](#13-kế-hoạch-4-tuần)
14. [Checklist chạy mượt trong MuJoCo](#14-checklist-chạy-mượt-trong-mujoco)
15. [Risk & Mitigation](#15-risk--mitigation)
16. [Dependency & Compatibility Matrix](#16-dependency--compatibility-matrix)
17. [Commands dự kiến](#17-commands-dự-kiến)
18. [File README nên có gì?](#18-file-readme-nên-có-gì)
19. [MVP cuối cùng cần nộp/demo](#19-mvp-cuối-cùng-cần-nộpdemo)
20. [Next step ngay lập tức](#20-next-step-ngay-lập-tức)
21. [Nguồn tham khảo](#21-nguồn-tham-khảo)

---

## 1. Mục tiêu dự án

Xây dựng pipeline chuyển tín hiệu từ video bàn tay/người sang điều khiển robot OpenArm trong MuJoCo:

```
Video bàn tay/người
→ lấy cổ tay người + trạng thái pinch ngón tay
→ retarget cổ tay sang OpenArm end-effector
→ pinch thì đóng gripper, không pinch thì mở gripper
→ replay mượt trong MuJoCo
```

### MVP đầu tiên cần đạt

```
Người di chuyển cổ tay trước camera
→ OpenArm end-effector di chuyển tương ứng trong MuJoCo

Người chạm ngón cái vào ngón khác
→ OpenArm gripper đóng

Người mở tay
→ OpenArm gripper mở
```

---

## 2. Nguyên tắc triển khai

Dự án không nên tự viết mọi thứ từ đầu. Cần ưu tiên dùng code/repo có sẵn, ổn định, rồi viết adapter riêng cho OpenArm.

1. Dùng `openarm_mujoco` làm nguồn chính cho model OpenArm trong MuJoCo.
2. Dùng phong cách pipeline từ `Vision-Based-Hand-Shadowing`: hand landmarks → transform → IK → preview/replay.
3. Dùng `mink` hoặc MuJoCo Jacobian cho inverse kinematics thay vì tự viết IK quá phức tạp từ đầu.
4. Dùng `LeRobot` ở giai đoạn sau cho dataset và imitation learning, không dùng ngay từ MVP đầu tiên.
5. Code phải tách module rõ ràng: perception, pinch, retargeting, IK, replay, dataset.
6. Config phải để trong YAML, không hard-code quá nhiều vào script.

---

## 3. Repo/Code tham khảo chính

| Repo / Source | Vai trò trong project | Cách dùng |
|---|---|---|
| `enactic/openarm_mujoco` | Model MuJoCo chính của OpenArm | Cài package, launch sim, inspect MJCF, lấy tên joint/site/actuator |
| `enactic/openarm` | Tổng quan hệ sinh thái OpenArm | Tham khảo openarm_description, openarm_dataset, openarm_teleop |
| `Vision-Based-Hand-Shadowing` | Gần nhất với bài toán hand → robot | Tham khảo cách lấy MediaPipe hand landmarks, transform tọa độ, IK, gripper mapping |
| `kevinzakka/mink` | IK dựa trên MuJoCo | Dùng nếu muốn IK ổn định, có joint limit và velocity limit |
| `huggingface/lerobot` | Dataset + policy sau này | Dùng sau khi MVP replay chạy được |
| MediaPipe Hands | Hand pose estimation | Lấy 21 landmarks, đặc biệt wrist và các fingertip |

**Lưu ý:** Hiện chưa có repo nào giải quyết đúng hoàn toàn bài toán:

```
video cổ tay người + pinch gesture → OpenArm end-effector + OpenArm gripper
```

Vì vậy dự án sẽ dùng code có sẵn cho các phần ổn định và tự viết adapter ở các điểm sau:

```
hand landmarks     → wrist trajectory
pinch              → gripper_cmd
wrist trajectory   → OpenArm target_pos
target_pos         → OpenArm IK/replay
```

---

## 4. Use Case Diagram

Hệ thống có 3 actor chính: **Người dùng (Operator)**, **Camera**, và **MuJoCo Simulator**.

```
╔══════════════════════════════════════════════════════════════════════╗
║                    OpenArm Retargeting System                        ║
║                                                                      ║
║   ┌──────────────────────┐    ┌────────────────────────────────┐    ║
║   │  UC-01               │    │  UC-02                         │    ║
║   │  Thu video tay        │    │  Tracking cổ tay real-time     │    ║
║   │  (offline/live)       │    │  qua webcam                    │    ║
║   └──────────┬───────────┘    └───────────────┬────────────────┘    ║
║              │                                │                      ║
║   ┌──────────▼───────────┐    ┌───────────────▼────────────────┐    ║
║   │  UC-03               │    │  UC-04                         │    ║
║   │  Phát hiện pinch      │    │  Làm mượt quỹ đạo cổ tay      │    ║
║   │  gesture              │    │  (smoothing)                   │    ║
║   └──────────┬───────────┘    └───────────────┬────────────────┘    ║
║              │                                │                      ║
║   ┌──────────▼───────────────────────────────▼────────────────┐    ║
║   │  UC-05: Retarget wrist → OpenArm target_pos                │    ║
║   └──────────────────────────┬─────────────────────────────────┘    ║
║                               │                                      ║
║   ┌───────────────────────────▼────────────────────────────────┐    ║
║   │  UC-06: Giải IK → qpos trajectory                          │    ║
║   └───────────────────────────┬────────────────────────────────┘    ║
║                               │                                      ║
║   ┌──────────────┐  ┌─────────▼──────────┐  ┌──────────────────┐  ║
║   │  UC-07       │  │  UC-08             │  │  UC-09           │  ║
║   │  Replay      │  │  Replay + gripper  │  │  Lưu dataset     │  ║
║   │  kinematic   │  │  đóng/mở          │  │  (NPZ)           │  ║
║   └──────────────┘  └────────────────────┘  └──────────────────┘  ║
║                                                                      ║
║   ┌──────────────────────────────────────────────────────────────┐  ║
║   │  UC-10: Train baseline policy (MLP/LeRobot) — giai đoạn sau │  ║
║   └──────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════╝

Actors:
  👤 Operator   ──→  UC-01, UC-02, UC-07, UC-08, UC-09, UC-10
  📷 Camera     ──→  UC-01, UC-02
  🤖 MuJoCo     ──→  UC-06, UC-07, UC-08
```

### Mô tả Use Case

| ID | Tên | Actor | Pre-condition | Post-condition |
|---|---|---|---|---|
| UC-01 | Thu video tay offline | Operator, Camera | Camera hoạt động, có đủ ánh sáng | File video `.mp4` lưu vào `data/raw_videos/` |
| UC-02 | Tracking cổ tay real-time | Operator, Camera | MediaPipe installed | Landmarks cổ tay + đầu ngón được stream real-time |
| UC-03 | Phát hiện pinch | Operator | Có hand pose data | `gripper_cmd` array với hysteresis |
| UC-04 | Làm mượt quỹ đạo | System | Có wrist raw trajectory | `wrist_smooth` không bị giật đột ngột |
| UC-05 | Retarget wrist | System | `wrist_smooth` trong không gian camera | `target_pos` trong không gian robot OpenArm |
| UC-06 | Giải IK | System, MuJoCo | `target_pos` trong workspace | `qpos` hợp lệ, IK error < 5cm |
| UC-07 | Replay kinematic | Operator, MuJoCo | Có `qpos` trajectory | Video replay MuJoCo |
| UC-08 | Replay + gripper | Operator, MuJoCo | Có `qpos` + `gripper_cmd` | Video OpenArm đóng/mở gripper đúng lúc |
| UC-09 | Lưu dataset | Operator | Replay thành công | File `.npz` dataset cho imitation learning |
| UC-10 | Train baseline policy | Operator | Dataset đủ lớn | Model MLP hoặc ACT có thể predict action |

---

## 5. Kiến trúc pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                           │
│  ┌─────────────────┐        ┌──────────────────────┐   │
│  │  Video file      │   OR   │   Webcam (live)      │   │
│  │  (.mp4)          │        │   (cv2.VideoCapture) │   │
│  └────────┬─────────┘        └──────────┬───────────┘   │
└───────────┼──────────────────────────────┼───────────────┘
            │                              │
            ▼                              ▼
┌─────────────────────────────────────────────────────────┐
│               PERCEPTION LAYER                          │
│                                                         │
│   hand_tracking.py  (MediaPipe Hands)                   │
│   ┌─────────────────────────────────────────────────┐   │
│   │  wrist [T,3]  + fingertip_tips [T,5,3]          │   │
│   │  handedness   + valid_mask [T]                  │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────┘
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
┌─────────────────────┐     ┌────────────────────────┐
│   PINCH DETECTION   │     │   SMOOTHING MODULE     │
│                     │     │                        │
│  pinch.py           │     │  smoothing.py          │
│  ─────────────────  │     │  ──────────────────    │
│  hysteresis logic   │     │  moving avg / SG       │
│  gripper_cmd [T]    │     │  velocity clamping     │
│                     │     │  wrist_smooth [T,3]    │
└──────────┬──────────┘     └───────────┬────────────┘
           │                            │
           └──────────┬─────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│                RETARGETING LAYER                        │
│                                                         │
│   retargeting.py                                        │
│   ┌─────────────────────────────────────────────────┐   │
│   │  human_delta → scale → transform → clamp        │   │
│   │  target_pos [T,3]  +  gripper_cmd [T]           │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    IK LAYER                             │
│                                                         │
│   ik_solver.py  (mink | MuJoCo Jacobian fallback)       │
│   ┌─────────────────────────────────────────────────┐   │
│   │  target_pos → differential IK → qpos [T, nq]   │   │
│   │  ee_pos [T,3]  +  ik_error [T]                 │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  REPLAY LAYER                           │
│                                                         │
│   mujoco_replay.py                                      │
│   ┌─────────────────────────────────────────────────┐   │
│   │  Mode A: kinematic (debug nhanh)                │   │
│   │  Mode B: actuator/position control (mượt hơn)  │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│            DATASET & LEARNING LAYER                     │
│                                                         │
│   dataset.py  →  .npz  →  LeRobot / ACT (giai đoạn 2)  │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Data Flow Diagram

### Level 0 — Context Diagram

```
            ┌──────────┐
   Video ──▶│          │──▶ MuJoCo replay video
            │  OpenArm  │
  Webcam ──▶│ Retarget │──▶ Dataset (.npz)
            │  System   │
  Config ──▶│          │──▶ Debug plots & reports
            └──────────┘
```

### Level 1 — Data stores & processes

```
[D1] raw_videos/          [D2] hand_pose/            [D3] robot_targets/
     .mp4                      .npz                       .npz
      │                         │                          │
      ▼                         ▼                          ▼
  ┌────────┐    wrist      ┌─────────┐   target_pos  ┌────────┐
  │  P1    │──────────────▶│   P2    │──────────────▶│  P3    │
  │  Hand  │   fingertips  │  Pinch  │   gripper_cmd │  IK    │
  │  Track │               │  + Smooth               │  Solver│
  └────────┘               └─────────┘               └───┬────┘
       │                        │                        │
       ▼                        ▼                        ▼
  [D4] debug_videos/      [D5] plots/            [D6] robot_traj/
       hand_debug.mp4          pinch_dist.png         qpos.npz
                               wrist_smooth.png       ik_error.npz
                               target_traj.png

                                                         │
                                                         ▼
                                                    ┌────────┐
                                                    │  P4    │
                                                    │ Replay │
                                                    │MuJoCo  │
                                                    └───┬────┘
                                                        │
                                              ┌─────────┴──────────┐
                                              ▼                     ▼
                                        [D7] replay_videos/   [D8] datasets/
                                             .mp4                  .npz
```

### Data Schema chính

```
hand_pose.npz:
  timestamps    : [T]       float64  — giây từ frame đầu
  wrist         : [T, 3]    float32  — x, y, z normalized (0..1)
  thumb_tip     : [T, 3]    float32
  index_tip     : [T, 3]    float32
  middle_tip    : [T, 3]    float32
  ring_tip      : [T, 3]    float32
  pinky_tip     : [T, 3]    float32
  valid         : [T]       bool     — True nếu MediaPipe detect được
  handedness    : str        — "Right" hoặc "Left"

robot_targets.npz:
  target_pos    : [T, 3]    float32  — xyz trong frame robot (mét)
  gripper_cmd   : [T]       float32  — 0.0 = mở, 1.0 = đóng

robot_traj.npz:
  qpos          : [T, nq]   float32  — joint positions
  arm_qpos      : [T, 7]    float32  — 7 arm joints only
  ee_pos        : [T, 3]    float32  — end-effector xyz từ IK
  target_pos    : [T, 3]    float32  — target xyz (để so sánh IK error)
  ik_error      : [T]       float32  — Euclidean distance ee_pos vs target
  gripper_cmd   : [T]       float32
```

---

## 7. State Machine: Pinch & Gripper

```
                    ┌──────────────────────────────────────────┐
                    │          GRIPPER STATE MACHINE           │
                    └──────────────────────────────────────────┘

           min_dist < close_threshold
     ┌───────────────────────────────────────┐
     │                                       │
     ▼                                       │
 ┌───────────┐    min_dist > open_threshold  │
 │  CLOSED   │──────────────────────────────▶│
 │ (cmd = 1) │                               │
 └───────────┘                           ┌───────────┐
     ▲                                   │   OPEN    │
     │                                   │ (cmd = 0) │
     │   min_dist < close_threshold      └───────────┘
     └───────────────────────────────────────┘

     ┌──────────────────────────────────────────────────┐
     │                 HYSTERESIS ZONE                  │
     │   close_threshold ≤ min_dist ≤ open_threshold    │
     │                                                  │
     │   → Giữ nguyên trạng thái trước đó              │
     │   → Không nhấp nháy 0/1                         │
     └──────────────────────────────────────────────────┘

Thresholds (mặc định):
  close_threshold = 0.05  (normalized hand space, ~5% hand size)
  open_threshold  = 0.10

Pinch fingers được check:
  d_index  = dist(thumb_tip, index_tip)
  d_middle = dist(thumb_tip, middle_tip)
  d_ring   = dist(thumb_tip, ring_tip)
  d_pinky  = dist(thumb_tip, pinky_tip)
  min_dist = min(d_index, d_middle, d_ring, d_pinky)
```

### Trạng thái mở rộng (giai đoạn 2)

```
  OPEN → PARTIAL → CLOSED
  
  Khi cần gripper analog (0.0 → 1.0) thay vì binary:
  gripper_cmd = 1.0 - clamp((min_dist - 0.0) / 0.15, 0, 1)
```

---

## 8. Cấu trúc repo đề xuất

```
openarm_wrist_pinch_retarget/
│
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── pyproject.toml                    # optional
├── .gitignore
│
├── configs/
│   ├── openarm.yaml                  # đường dẫn model, joint/site/actuator name
│   ├── hand_tracking.yaml            # camera/video, handedness, confidence
│   ├── pinch.yaml                    # threshold mở/đóng gripper
│   ├── retarget.yaml                 # scale, origin, workspace limit
│   └── ik.yaml                       # solver, step size, damping, max iter
│
├── external/
│   ├── openarm/                      # clone repo tham khảo
│   ├── openarm_mujoco/
│   ├── Vision-Based-Hand-Shadowing/
│   ├── mink/
│   └── lerobot/
│
├── data/
│   ├── raw_videos/                   # video gốc (.mp4)
│   ├── hand_pose/                    # wrist/fingertip landmarks (.npz)
│   ├── robot_targets/                # target_pos + gripper_cmd (.npz)
│   ├── robot_traj/                   # qpos, ee_pos, ik_error (.npz)
│   └── datasets/                     # dataset cho imitation learning (.npz)
│
├── outputs/
│   ├── debug_videos/                 # video vẽ landmarks
│   ├── replay_videos/                # video MuJoCo replay
│   └── plots/                        # trajectory, IK error, pinch distance
│
├── scripts/
│   ├── 00_check_install.py
│   ├── 01_inspect_openarm_model.py
│   ├── 02_extract_hand_pose.py
│   ├── 03_detect_pinch.py
│   ├── 04_smooth_wrist.py
│   ├── 05_retarget_wrist_to_openarm.py
│   ├── 06_openarm_ik.py
│   ├── 07_replay_openarm_mujoco.py
│   ├── 08_log_dataset.py
│   └── 09_train_baseline_policy.py
│
├── src/
│   └── openarm_retarget/
│       ├── __init__.py
│       ├── hand_tracking.py
│       ├── pinch.py
│       ├── smoothing.py
│       ├── retargeting.py
│       ├── openarm_model.py
│       ├── ik_solver.py
│       ├── mujoco_replay.py
│       ├── dataset.py
│       └── utils.py
│
└── notebooks/
    ├── visualize_hand_pose.ipynb
    ├── visualize_retargeting.ipynb
    └── inspect_dataset.ipynb
```

---

## 9. Vì sao cần cả `scripts/` và `src/`?

`scripts/` dùng để chạy từng bước độc lập:

```bash
python scripts/02_extract_hand_pose.py --video data/raw_videos/demo_001.mp4
python scripts/03_detect_pinch.py --input data/hand_pose/demo_001.npz
python scripts/07_replay_openarm_mujoco.py --traj data/robot_traj/demo_001.npz
```

`src/openarm_retarget/` chứa logic tái sử dụng:

```
hand_tracking.py  → hàm lấy hand landmarks
pinch.py          → hàm phát hiện pinch + hysteresis
retargeting.py    → hàm map wrist sang OpenArm target
ik_solver.py      → class giải IK (mink hoặc Jacobian)
mujoco_replay.py  → class replay/render trong MuJoCo
```

Cách này giúp repo gọn, dễ debug và sau này dễ chuyển sang LeRobot hoặc robot thật.

---

## 10. Chi tiết từng module

### Module 0: Check install

**File:** `scripts/00_check_install.py`

Kiểm tra môi trường trước khi chạy project:

- Python version (>= 3.10)
- `mujoco` import được không
- `mediapipe` import được không
- `opencv-python` import được không
- `numpy`, `matplotlib`, `yaml` import được không
- Có load được OpenArm MJCF không
- `mink` có available không (optional)

**Output:**

```
[OK] Python 3.11.x
[OK] MuJoCo 3.x.x
[OK] MediaPipe 0.10.x
[OK] OpenCV 4.x.x
[OK] OpenArm model loaded — joints: 7, actuators: 8
[WARN] mink not found — will use Jacobian IK fallback
```

---

### Module 1: Inspect OpenArm model

**Files:** `scripts/01_inspect_openarm_model.py`, `src/openarm_retarget/openarm_model.py`

Tìm chính xác thông tin trong MuJoCo model:

- `nq`, `nv`, `nu`
- joint names, actuator names, site names, body names
- end-effector site, gripper actuator
- joint limits, control ranges

**Output:** `outputs/openarm_model_report.txt`

**Tiêu chí xong:**

```
ee_site = "hand_site"        # hoặc tên thực trong MJCF
gripper_actuator = "gripper" # hoặc tên thực
arm_joint_names = ["joint1", "joint2", ..., "joint7"]
```

Nếu chưa biết `ee_site`, phải chọn site/body gần gripper nhất hoặc thêm site vào MJCF clone riêng.

---

### Module 2: Hand tracking

**Files:** `scripts/02_extract_hand_pose.py`, `src/openarm_retarget/hand_tracking.py`

**Input:** `data/raw_videos/demo_001.mp4`

**Output:**
- `data/hand_pose/demo_001_hand_pose.npz`
- `outputs/debug_videos/demo_001_hand_debug.mp4`

**Landmark cần lấy:**

```
wrist       = landmark[0]   — cổ tay
thumb_tip   = landmark[4]   — đầu ngón cái
index_tip   = landmark[8]   — đầu ngón trỏ
middle_tip  = landmark[12]  — đầu ngón giữa
ring_tip    = landmark[16]  — đầu ngón áp út
pinky_tip   = landmark[20]  — đầu ngón út
```

**Data format:**

```python
{
    "timestamps": [T],           # float64, seconds
    "wrist":      [T, 3],        # float32, normalized [0..1]
    "thumb_tip":  [T, 3],
    "index_tip":  [T, 3],
    "middle_tip": [T, 3],
    "ring_tip":   [T, 3],
    "pinky_tip":  [T, 3],
    "valid":      [T],           # bool
    "handedness": "Right"        # str
}
```

**Lưu ý:** Tham khảo cách xử lý hand landmarks từ `Vision-Based-Hand-Shadowing`. Dùng MediaPipe Hands thay vì tự train model.

---

### Module 3: Pinch detection

**Files:** `scripts/03_detect_pinch.py`, `src/openarm_retarget/pinch.py`, `configs/pinch.yaml`

**Logic:**

```python
d_index  = distance(thumb_tip, index_tip)
d_middle = distance(thumb_tip, middle_tip)
d_ring   = distance(thumb_tip, ring_tip)
d_pinky  = distance(thumb_tip, pinky_tip)
min_dist = min(d_index, d_middle, d_ring, d_pinky)

# Hysteresis
if min_dist < close_threshold:   gripper_cmd = 1
elif min_dist > open_threshold:  gripper_cmd = 0
else:                            gripper_cmd = prev_state  # giữ trạng thái
```

**Output:**

```python
{
    "pinch_distance": [T],   # float32
    "pinch_finger":   [T],   # int (0=thumb-index, 1=thumb-middle, ...)
    "gripper_cmd":    [T]    # float32: 0.0 hoặc 1.0
}
```

**Tiêu chí xong:**
- Pinch → `gripper_cmd = 1`
- Mở tay → `gripper_cmd = 0`
- Không bị nhấp nháy liên tục 0/1

---

### Module 4: Smooth wrist trajectory

**Files:** `scripts/04_smooth_wrist.py`, `src/openarm_retarget/smoothing.py`

**Phương pháp ban đầu:**

1. Interpolate frame mất landmark (linear interpolation trên `valid` mask).
2. Moving average window 5 hoặc 7 frame.
3. Clamp vận tốc cổ tay nếu jump quá lớn.
4. Optional: Savitzky-Golay filter nếu cần.

**Output:**

```python
{
    "wrist_raw":    [T, 3],
    "wrist_smooth": [T, 3],
    "gripper_cmd":  [T]
}
```

**Tiêu chí xong:** Có plot `outputs/plots/demo_001_wrist_smoothing.png`.

---

### Module 5: Retarget wrist sang OpenArm

**Files:** `scripts/05_retarget_wrist_to_openarm.py`, `src/openarm_retarget/retargeting.py`, `configs/retarget.yaml`

**Công thức baseline:**

```python
human_delta[t] = wrist_smooth[t] - wrist_smooth[0]
openarm_target[t] = openarm_origin + scale * transform(human_delta[t])
```

**Config ví dụ:**

```yaml
openarm_origin: [0.35, 0.0, 0.35]

scale:
  x: 0.6
  y: 0.6
  z: 0.6

workspace_limit:
  x_min: 0.15
  x_max: 0.65
  y_min: -0.35
  y_max: 0.35
  z_min: 0.10
  z_max: 0.70

axis_mapping:
  human_x_to_robot: x
  human_y_to_robot: z_negative   # vì trục y camera hướng xuống
  human_z_to_robot: y
```

**Output:**

```python
{
    "target_pos":   [T, 3],
    "gripper_cmd":  [T]
}
```

**Tiêu chí xong:**
- Target nằm trong workspace.
- Target không giật.
- Khi tay người đi trái/phải/lên/xuống, target robot đi tương ứng.

---

### Module 6: OpenArm IK

**Files:** `scripts/06_openarm_ik.py`, `src/openarm_retarget/ik_solver.py`, `configs/ik.yaml`

**Hướng ưu tiên:**

| Priority | Solver | Ghi chú |
|---|---|---|
| 1 | `mink` differential IK | Ổn định, có joint limit và velocity limit |
| 2 | MuJoCo Jacobian (damped least squares) | Fallback nếu mink khó cài |

**Pseudocode IK loop:**

```python
for t in range(T):
    while ik_error > tolerance and iterations < max_iter:
        J = mj_jacSite(model, data, ee_site)
        delta_pos = target_pos[t] - ee_pos
        dq = J.T @ inv(J @ J.T + damping * I) @ delta_pos
        dq = clamp(dq, -max_dq, max_dq)
        data.qpos[arm_joints] += dq * step_size
        mj_forward(model, data)
        ee_pos = get_site_pos(model, data, ee_site)
        ik_error = norm(target_pos[t] - ee_pos)
```

**Output:**

```python
{
    "qpos":       [T, nq],
    "arm_qpos":   [T, 7],
    "ee_pos":     [T, 3],
    "target_pos": [T, 3],
    "ik_error":   [T],
    "gripper_cmd":[T]
}
```

**Tiêu chí xong:**

```
mean IK error < 5 cm  → MVP đạt
mean IK error < 2 cm  → tốt
```

**Lưu ý để OpenArm chạy mượt:**
- Không nhảy `qpos` trực tiếp quá xa giữa 2 frame.
- Dùng interpolation giữa các target.
- Dùng velocity limit và joint limit.
- Dùng low-pass filter trên `target_pos`.
- Dùng damping trong IK.
- Nếu target ngoài workspace, clamp trước khi IK.

---

### Module 7: Replay OpenArm trong MuJoCo

**Files:** `scripts/07_replay_openarm_mujoco.py`, `src/openarm_retarget/mujoco_replay.py`

**Hai chế độ replay:**

**Mode A — Kinematic replay** (debug nhanh):

```python
data.qpos[:] = qpos[t]
mj_forward(model, data)
renderer.update_scene(data)
pixels = renderer.render()
```

**Mode B — Actuator replay** (mượt hơn):

```python
data.ctrl[arm_actuators] = qpos[t]     # position control
data.ctrl[gripper_actuator] = gripper_cmd[t]
for _ in range(n_substeps):
    mj_step(model, data)
renderer.update_scene(data)
```

**Output:** `outputs/replay_videos/demo_001_openarm_replay.mp4`

**Tiêu chí xong:**
- OpenArm di chuyển theo cổ tay người.
- Pinch thì gripper đóng, không pinch thì mở.
- Video không bị giật mạnh.

---

### Module 8: Dataset logging

**Files:** `scripts/08_log_dataset.py`, `src/openarm_retarget/dataset.py`

**Dataset format MVP:**

```python
obs = {
    "wrist_smooth":  [T, 3],
    "target_pos":    [T, 3],
    "qpos":          [T, nq],
    "qvel":          [T, nv],
    "ee_pos":        [T, 3],
    "gripper_state": [T, 1]
}

action = {
    "arm_joint_target": [T, 7],
    "gripper_cmd":      [T, 1]
}
```

**Output:** `data/datasets/openarm_wrist_pinch_v1.npz`

---

### Module 9: Baseline learning

**File:** `scripts/09_train_baseline_policy.py`

**Chỉ làm sau khi replay ổn.**

**Baseline đơn giản:**

```
Input:  qpos + target_pos + gripper_state
Output: next_arm_joint_target + next_gripper_cmd
Model:  MLP (2-3 hidden layers)
Loss:   MSE arm + BCE gripper
```

Sau đó nâng cấp sang LeRobot/ACT.

---

## 11. Thứ tự thực hiện thực tế

```
Bước 1:   Cài và chạy openarm_mujoco gốc
Bước 2:   Inspect OpenArm model → joint/site/actuator
Bước 3:   Chạy hand tracking từ video/webcam
Bước 4:   Detect pinch → gripper_cmd
Bước 5:   Smooth wrist trajectory
Bước 6:   Retarget wrist → OpenArm target_pos
Bước 7:   IK target_pos → qpos
Bước 8:   Replay qpos + gripper trong MuJoCo
Bước 9:   Tối ưu cho mượt
Bước 10:  Lưu dataset
Bước 11:  Train baseline nếu cần
```

**Nguyên tắc:** Không ghép 2 module mới cùng lúc. Mỗi bước phải có output kiểm chứng được trước khi sang bước tiếp.

---

## 12. Milestone

### Milestone 1: OpenArm MuJoCo chạy được

**Output:**
```
openarm-mujoco-launch chạy được
outputs/openarm_model_report.txt có joint/site/actuator
```

**Done khi:**
- Có thể mở MuJoCo viewer.
- Có thể load model bằng Python.
- Biết tên `ee_site` và `gripper_actuator`.

---

### Milestone 2: Hand tracking chạy được

**Output:**
```
data/hand_pose/demo_001_hand_pose.npz
outputs/debug_videos/demo_001_hand_debug.mp4
```

**Done khi:** Video debug bám đúng cổ tay và đầu ngón tay trên ít nhất 90% frame.

---

### Milestone 3: Pinch detection chạy được

**Output:**
```
data/hand_pose/demo_001_hand_pose_with_pinch.npz
outputs/plots/pinch_distance_plot.png
```

**Done khi:**
- Pinch → cmd = 1, mở tay → cmd = 0.
- Tín hiệu không nhấp nháy (< 2 flip/giây trong vùng hysteresis).

---

### Milestone 4: Wrist retargeting chạy được

**Output:**
```
data/robot_targets/demo_001_target.npz
outputs/plots/target_trajectory_plot.png
```

**Done khi:**
- Target nằm trong workspace OpenArm.
- Target đi đúng hướng theo cổ tay người.

---

### Milestone 5: IK chạy được

**Output:**
```
data/robot_traj/demo_001_qpos.npz
outputs/plots/ik_error_plot.png
```

**Done khi:**
- Mean IK error < 5 cm cho MVP.
- Không có frame nào IK diverge (error > 20 cm).

---

### Milestone 6: Replay OpenArm + gripper ✅ MVP

**Output:**
```
outputs/replay_videos/openarm_wrist_pinch_demo.mp4
```

**Done khi:**
- OpenArm đi theo cổ tay người.
- Pinch thì gripper đóng, mở tay thì gripper mở.
- Chuyển động tương đối mượt (không giật đột ngột).

---

### Milestone 7: Dataset

**Output:**
```
data/datasets/openarm_wrist_pinch_dataset_v1.npz
```

**Done khi:**
- Load dataset lại được.
- Replay lại trajectory từ dataset được.

---

## 13. Kế hoạch 4 tuần

### Tuần 1: OpenArm MuJoCo + Hand Tracking

| Ngày | Việc làm |
|---|---|
| 1-2 | Cài `openarm-mujoco`, chạy `openarm-mujoco-launch` |
| 2-3 | Viết `01_inspect_openarm_model.py`, lưu report |
| 3-4 | Viết `02_extract_hand_pose.py` với video demo |
| 4-5 | Tạo video debug hand landmarks, kiểm tra accuracy |

**Deliverable:**
```
outputs/openarm_model_report.txt
outputs/debug_videos/demo_001_hand_debug.mp4
data/hand_pose/demo_001_hand_pose.npz
```

---

### Tuần 2: Pinch + Retargeting

| Ngày | Việc làm |
|---|---|
| 1-2 | Viết `03_detect_pinch.py` + tune hysteresis |
| 2-3 | Viết `04_smooth_wrist.py` + so sánh raw vs smooth |
| 3-4 | Viết `05_retarget_wrist_to_openarm.py` + tune scale |
| 4-5 | Vẽ plot wrist và target, kiểm tra workspace |

**Deliverable:**
```
outputs/plots/pinch_distance.png
outputs/plots/wrist_smoothing.png
outputs/plots/target_trajectory.png
data/robot_targets/demo_001_target.npz
```

---

### Tuần 3: IK + Replay

| Ngày | Việc làm |
|---|---|
| 1 | Thử cài `mink`, nếu fail thì chuyển sang Jacobian fallback |
| 2-3 | Viết `06_openarm_ik.py`, tune damping và step size |
| 3-4 | Viết `07_replay_openarm_mujoco.py`, test Mode A trước |
| 4-5 | Replay gripper theo `gripper_cmd`, test Mode B |

**Deliverable:**
```
data/robot_traj/demo_001_qpos.npz
outputs/plots/ik_error.png
outputs/replay_videos/demo_001_openarm_replay.mp4
```

---

### Tuần 4: Làm mượt + Dataset + Báo cáo

| Ngày | Việc làm |
|---|---|
| 1-2 | Tối ưu smoothing + retarget scale/workspace |
| 2-3 | Tối ưu replay bằng actuator/position control |
| 3 | Viết `08_log_dataset.py`, lưu dataset |
| 4-5 | Viết README + PROJECT_REPORT + quay final demo |

**Deliverable:**
```
data/datasets/openarm_wrist_pinch_v1.npz
README.md
PROJECT_REPORT.md
outputs/replay_videos/final_demo.mp4
```

---

### Tuần 5: Live webcam + Bimanual IK + MuJoCo

| Ngày | Việc làm | Trạng thái |
|---|---|---|
| 1 | Camera thread latest-frame, MSMF/MJPG | Hoàn thành |
| 2 | MediaPipe live tracking và ánh xạ hai tay | Hoàn thành |
| 3 | Stateful IK, smoothing và safety khi mất tay | Hoàn thành |
| 4 | Hiển thị camera + MuJoCo, đo latency | Hoàn thành |
| 5 | Ghi live dataset, test, benchmark, tài liệu | Hoàn thành |

**Command:**

```bash
openarm-retarget live
```

**Nghiệm thu:**

```text
tracking/IK latency p95 < 150 ms       đạt: 78 ms
combined display latency p95 < 150 ms đạt: 94 ms
hai tay + hai gripper                  đạt
mất tay giữ pose, gripper mở an toàn   đạt
ghi live session NPZ                   đạt
```

MediaPipe Python GPU delegate không khả dụng trong wheel Windows hiện tại.
Hệ thống dùng XNNPACK CPU; GPU có thể được khai thác sau bằng MediaPipe C++
tự build hoặc một backend inference khác.

---

## 14. Checklist chạy mượt trong MuJoCo

```
[ ] Không đưa target_pos ngoài workspace (clamp trước IK)
[ ] Smooth wrist trước khi retarget
[ ] Smooth target_pos sau retarget nếu còn giật
[ ] Dùng damping trong IK (lambda = 1e-4 đến 1e-2)
[ ] Clamp joint limit (dùng model.jnt_range)
[ ] Clamp velocity giữa hai timestep (max_dq = 0.05 rad)
[ ] Interpolate qpos giữa các frame (cubic spline)
[ ] Replay bằng actuator control thay vì chỉ set qpos nếu cần video mượt
[ ] Render FPS cố định (thường 30fps)
[ ] Lưu ik_error để debug
[ ] Kiểm tra không có NaN trong qpos hoặc target_pos
[ ] Nếu IK diverge, fallback về qpos trước đó
```

---

## 15. Risk & Mitigation

| # | Rủi ro | Xác suất | Mức độ | Biện pháp |
|---|---|---|---|---|
| R1 | MediaPipe mất track tay (occlusion, ánh sáng kém) | Cao | Trung bình | Interpolate valid frame, cảnh báo khi valid < 80% |
| R2 | IK không hội tụ cho target ngoài workspace | Trung bình | Cao | Clamp target_pos trước IK, hard workspace limit |
| R3 | `mink` không cài được trên môi trường hiện tại | Trung bình | Trung bình | Chuẩn bị sẵn Jacobian fallback trong `ik_solver.py` |
| R4 | OpenArm MJCF không có `ee_site` rõ ràng | Thấp | Cao | Clone MJCF, tự thêm `<site>` gần gripper |
| R5 | Quỹ đạo giật mạnh do IK error tích lũy | Trung bình | Trung bình | Low-pass filter trên `target_pos`, velocity clamp |
| R6 | Pinch signal không ổn định do tay run nhỏ | Cao | Thấp | Tăng hysteresis zone, dùng temporal filter trên `min_dist` |
| R7 | Trục tọa độ camera vs robot bị flip | Trung bình | Cao | Visualize retarget plot ngay sau Module 5, fix `axis_mapping` |

---

## 16. Dependency & Compatibility Matrix

### Core dependencies

```
python        >= 3.10
mujoco        >= 3.1.0       (Python bindings)
mediapipe     >= 0.10.0
opencv-python >= 4.8.0
numpy         >= 1.24.0
scipy         >= 1.10.0      (Savitzky-Golay filter)
matplotlib    >= 3.7.0
pyyaml        >= 6.0
```

### Optional dependencies

```
mink          >= 0.3.0       (differential IK, ưu tiên dùng)
lerobot       >= 2.0.0       (dataset + policy, giai đoạn 2)
torch         >= 2.0.0       (chỉ cần cho Module 9)
```

### Compatibility notes

| Package | Vấn đề biết trước | Giải pháp |
|---|---|---|
| `mink` | Cần MuJoCo >= 3.x | Kiểm tra trong `00_check_install.py` |
| `mediapipe` | ARM Mac có thể cần build từ source | Dùng `mediapipe-silicon` nếu cần |
| `openarm_mujoco` | Model MJCF path có thể thay đổi theo version | Inspect bằng `pkg_resources.files()` |

### requirements.txt đề xuất

```
mujoco>=3.1.0
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

## 17. Commands dự kiến

### Cài OpenArm MuJoCo

```bash
pip install openarm-mujoco
openarm-mujoco-launch
```

### Cài dependencies project

```bash
pip install -r requirements.txt
# Optional:
pip install mink
```

### Chạy từng bước pipeline

```bash
# Step 0: Kiểm tra môi trường
python scripts/00_check_install.py

# Step 1: Inspect model
python scripts/01_inspect_openarm_model.py

# Step 2: Extract hand pose
python scripts/02_extract_hand_pose.py \
  --video data/raw_videos/demo_001.mp4 \
  --output data/hand_pose/demo_001_hand_pose.npz

# Step 3: Detect pinch
python scripts/03_detect_pinch.py \
  --input data/hand_pose/demo_001_hand_pose.npz \
  --config configs/pinch.yaml

# Step 4: Smooth wrist
python scripts/04_smooth_wrist.py \
  --input data/hand_pose/demo_001_hand_pose.npz \
  --window 7

# Step 5: Retarget
python scripts/05_retarget_wrist_to_openarm.py \
  --input data/hand_pose/demo_001_smooth.npz \
  --config configs/retarget.yaml

# Step 6: IK
python scripts/06_openarm_ik.py \
  --input data/robot_targets/demo_001_target.npz \
  --config configs/ik.yaml

# Step 7: Replay
python scripts/07_replay_openarm_mujoco.py \
  --traj data/robot_traj/demo_001_qpos.npz \
  --mode B \
  --output outputs/replay_videos/demo_001_openarm_replay.mp4
```

### Chạy webcam live (giai đoạn sau)

```bash
python scripts/02_extract_hand_pose.py --webcam 0
```

---

## 18. File README nên có gì?

`README.md` nên ngắn, gồm:

```
1. Project title + badge (Python version, license)
2. Mục tiêu (1-2 câu)
3. Demo GIF / video thumbnail
4. Pipeline (hình ảnh hoặc ASCII)
5. Setup (pip install, openarm-mujoco-launch)
6. Cách chạy demo (copy-paste commands)
7. Cấu trúc repo (cây thư mục tóm tắt)
8. Kết quả hiện tại (milestone đã đạt)
9. Known issues
10. Next steps
11. License
```

Không viết README quá dài. Chi tiết để trong `PROJECT_PLAN.md` và `PROJECT_REPORT.md`.

---

## 19. MVP cuối cùng cần nộp/demo

**Input:**
```
1 video tay người di chuyển và pinch/mở tay
```

**Output:**
```
1 video OpenArm trong MuJoCo:
  - end-effector đi theo quỹ đạo cổ tay người
  - pinch thì gripper đóng
  - không pinch thì gripper mở
  - chuyển động tương đối mượt
```

**Files kèm theo:**

```
PROJECT_PLAN.md
README.md
outputs/replay_videos/final_demo.mp4
outputs/plots/ik_error.png
outputs/plots/pinch_distance.png
data/datasets/openarm_wrist_pinch_v1.npz    (nếu đã có)
```

---

## 20. Next step ngay lập tức

Việc đầu tiên không phải train model, mà là đảm bảo OpenArm MuJoCo chạy ổn:

```
1. Cài openarm-mujoco
2. Launch simulation gốc
3. Viết inspect script
4. Xác định joint/site/actuator
5. Chạy replay đơn giản: set một vài qpos hoặc ctrl để thấy OpenArm cử động
```

Sau khi OpenArm chạy được mượt trong MuJoCo mới ghép hand tracking và pinch vào.

---

## 21. Nguồn tham khảo

| Resource | URL |
|---|---|
| OpenArm MuJoCo | https://github.com/enactic/openarm_mujoco |
| OpenArm main repo | https://github.com/enactic/openarm |
| Vision-Based-Hand-Shadowing paper | https://arxiv.org/html/2603.11383v1 |
| Mink IK | https://github.com/kevinzakka/mink |
| LeRobot | https://huggingface.co/lerobot |
| MediaPipe Hands | https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker |
| MuJoCo Python docs | https://mujoco.readthedocs.io/en/stable/python.html |
| MuJoCo Jacobian API | https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jacsub |

---

*End of PROJECT_PLAN v2 — OpenArm Wrist & Pinch Retargeting*
