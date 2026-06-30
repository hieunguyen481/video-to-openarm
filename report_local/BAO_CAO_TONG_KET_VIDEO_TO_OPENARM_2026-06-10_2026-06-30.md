# BÁO CÁO TỔNG KẾT DỰ ÁN VIDEO TO OPENARM

**Giai đoạn:** 10/06/2026–30/06/2026

**Bài toán:** Chuyển chuyển động hai bàn tay trong video thành quỹ đạo điều khiển robot OpenArm hai tay

**Môi trường đánh giá:** MuJoCo, video ego-centric, 30 FPS

**Đầu ra chính:** quỹ đạo 14 khớp tay máy, trạng thái hai gripper, replay robot, dữ liệu huấn luyện và báo cáo chất lượng

---

## 1. Tóm tắt

Dự án xây dựng một pipeline end-to-end:

```text
Video / webcam
→ phát hiện và ước lượng pose hai bàn tay
→ theo dõi danh tính tay + xử lý frame thiếu
→ nhận dạng pinch và làm mượt quỹ đạo
→ retarget chuyển động người sang workspace OpenArm
→ giải inverse kinematics hai tay
→ điều khiển hai gripper
→ replay trong MuJoCo
→ lưu dataset, đồ thị, quality report và video đối chiếu
```

Trong ba tuần, bài toán được phát triển từ baseline MediaPipe 2D thành ba nhánh chính:

1. **2D/pseudo-depth:** MediaPipe, EgoForce YOLO và WiLoR YOLO hand-only.
2. **World-space:** HaWoR kết hợp MANO, camera motion/SLAM và xử lý video dài theo chunk.
3. **Full-3D camera-space:** WiLoR/MANO 3D, có depth, global orientation và IK định hướng.

Kết quả cuối giai đoạn không có một phương pháp thắng tuyệt đối trên mọi tiêu chí:

- **HaWoR overlap 2 giây** đạt IK convergence cao nhất trên `factory002_middle`: **96,71%**.
- **WiLoR YOLO tuned 2D** đạt **93,1% convergence**, mean IK error **1,22 cm**, nhẹ hơn HaWoR.
- **WiLoR full-3D cơ bản** đạt **93,33% convergence**, mean error **1,40 cm** trên đủ **2.100 frame**, đồng thời cung cấp 3D và orientation; đây là nền tảng phù hợp nhất để tiếp tục nghiên cứu.
- **MediaPipe PhaseB** vẫn là baseline đơn giản, dễ tái lập, nhưng bị giới hạn bởi depth proxy.

Kết luận quan trọng nhất là: **nút thắt của hệ thống nằm ở perception, depth, hệ tọa độ và chuẩn hóa chuyển động; IK chỉ phản ánh khả năng robot đạt target đã cho, không chứng minh target đó đúng với người.**

---

## 2. Mục tiêu và phạm vi

### 2.1 Mục tiêu

Từ video quan sát hai tay người, hệ thống phải:

- Ước lượng quỹ đạo cổ tay trái/phải theo thời gian.
- Nhận biết thao tác chụm ngón để đóng/mở từng gripper.
- Ánh xạ chuyển động người sang hai workspace hợp lệ của OpenArm.
- Tìm quỹ đạo khớp trơn, liên tục và có sai số end-effector thấp.
- Tái hiện kết quả trong MuJoCo và sinh dữ liệu cho imitation learning.

### 2.2 Phạm vi được chủ động thu hẹp

Dự án không cố khôi phục toàn bộ cơ thể hoặc toàn bộ động học ngón tay. Hai tín hiệu được ưu tiên là:

- **Wrist position:** điều khiển vị trí end-effector.
- **Pinch state:** điều khiển gripper nhị phân/liên tục.

Việc thu hẹp này làm bài toán đủ rõ để kiểm chứng từng mô-đun, nhưng vẫn giữ được hành vi thao tác cốt lõi của robot hai tay.

---

## 3. Cơ sở lý thuyết

### 3.1 Ước lượng bàn tay từ camera đơn

Với một điểm 3D trong hệ camera \((X,Y,Z)\), phép chiếu phối cảnh xấp xỉ:

\[
u=f_x\frac{X}{Z}+c_x,\qquad
v=f_y\frac{Y}{Z}+c_y
\]

Từ một ảnh đơn, nhiều điểm 3D khác nhau có thể cho cùng tọa độ ảnh \((u,v)\). Vì vậy depth không xác định duy nhất nếu không có prior hình học, mô hình học sâu, chuyển động camera hoặc cảm biến bổ sung.

Các nhánh trong dự án xử lý vấn đề này theo ba mức:

- **MediaPipe/YOLO 2D:** lấy \(x,y\) trên ảnh và dùng kích thước lòng bàn tay làm depth proxy.
- **WiLoR full-3D:** dự đoán MANO pose và camera translation để có khớp tay trong camera-space.
- **HaWoR:** ước lượng hand pose theo world-space và kết hợp thông tin chuyển động camera/SLAM.

### 3.2 Palm scale như depth proxy

Baseline tính kích thước lòng bàn tay từ khoảng cách wrist đến các MCP:

\[
s_t=\frac{1}{4}\sum_{i\in\{5,9,13,17\}}
\left\|p_{i,t}-p_{\text{wrist},t}\right\|
\]

Theo phối cảnh, bàn tay gần camera thường có kích thước ảnh lớn hơn. Tuy nhiên \(s_t\) còn thay đổi do xoay bàn tay, co ngón, che khuất và lỗi landmark; vì vậy nó chỉ là proxy, không phải depth vật lý.

### 3.3 Nhận dạng pinch

Pinch được suy ra từ khoảng cách ngón cái đến các đầu ngón:

\[
d_{k,t}=\left\|p_{\text{thumb},t}-p_{k,t}\right\|
\]

Gripper đóng khi ít nhất một \(d_{k,t}\) nhỏ hơn ngưỡng đóng và mở khi vượt ngưỡng mở. Hysteresis cùng yêu cầu trạng thái ổn định qua nhiều frame giúp tránh rung đóng/mở và pinch giả khi tay mới xuất hiện.

### 3.4 Retargeting

Retargeting biến quỹ đạo cổ tay người \(h_t\) thành target robot \(x_t\):

\[
x_t=o_r+A\,S\,(h_t-h_{\text{ref}})
\]

Trong đó:

- \(o_r\): origin của end-effector robot.
- \(h_{\text{ref}}\): mốc người, ban đầu thường lấy frame đầu; về sau có thể cấu hình cố định.
- \(S=\text{diag}(s_x,s_y,s_z)\): scale từng trục.
- \(A\): ma trận đổi trục, đổi dấu và quy ước camera–robot.

Target sau đó được clip vào workspace mỗi tay. Scale quá lớn tạo target ngoài tầm với; scale quá nhỏ làm metric IK đẹp nhưng robot gần như không chuyển động. Do đó retarget phải cân bằng **độ chính xác, biên độ và tính giống chuyển động thật**.

### 3.5 Inverse kinematics bằng Jacobian DLS

Với vị trí end-effector \(x(q)\), sai số tại một bước là:

\[
e=x^*-x(q)
\]

Jacobian liên hệ thay đổi khớp và end-effector:

\[
\Delta x\approx J(q)\Delta q
\]

Damped least squares giải:

\[
\Delta q=J^T(JJ^T+\lambda^2I)^{-1}e
\]

Hệ hai tay ghép sai số và Jacobian của tay trái/phải thành một bài toán chung cho 14 joint. Damping giúp ổn định gần singularity; giới hạn \(\Delta q\) và thay đổi mỗi frame giúp quỹ đạo bớt giật.

### 3.6 IK có orientation

WiLoR cung cấp ma trận global orientation. Sai số quay được xấp xỉ từ:

\[
R_e=R^*R^T
\]

và phần phản đối xứng của \(R_e\) tạo rotation error vector. Hệ thống ghép position error và rotation error:

\[
e_{\text{all}}=
\begin{bmatrix}
e_{p,L}\\
w_Re_{R,L}\\
e_{p,R}\\
w_Re_{R,R}
\end{bmatrix}
\]

Orientation làm tư thế cổ tay tự nhiên hơn nhưng tiêu thụ bậc tự do và có thể cạnh tranh với position target. Thực nghiệm xác nhận weight orientation chưa phù hợp sẽ làm giảm convergence.

### 3.7 Lọc theo thời gian

Các kỹ thuật đã thử:

- Nội suy tuyến tính qua frame mất tracking.
- Moving average.
- Median filter để loại spike.
- Savitzky–Golay để giảm nhiễu nhưng giữ xu hướng vận tốc.
- Giới hạn tốc độ/quãng nhảy.
- Làm mượt rotation qua quaternion, sửa dấu quaternion và chiếu lại lên nhóm quay \(SO(3)\).

Lọc quá yếu giữ nhiễu; lọc quá mạnh làm mất chuyển động thật, tăng lag hoặc loại nhầm frame hợp lệ. Vì vậy bộ lọc phải gắn với confidence và vận tốc chuyển động.

---

## 4. Kiến trúc hệ thống

### 4.1 Kiến trúc logic

```text
┌─────────────────────────────────────────────────────────────┐
│ Input: video file / webcam / EgoWorld-LeRobot              │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Perception                                                  │
│ MediaPipe | YOLO hand-only | WiLoR MANO 3D | HaWoR world   │
│ Output chung: timestamps, left_*, right_*, valid, metadata  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Temporal processing                                         │
│ identity/handedness, gap interpolation, outlier rejection,  │
│ smoothing, pinch hysteresis                                 │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Retargeting                                                 │
│ reference + axis mapping + per-axis scale + workspace clip  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Bimanual IK                                                 │
│ Jacobian DLS, 14 arm joints, optional wrist orientation     │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Output                                                      │
│ MuJoCo replay, grippers, NPZ dataset, plots, quality JSON,  │
│ human-vs-robot comparison video                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Mô-đun chính trong repo

| Nhóm | Thành phần | Vai trò |
|---|---|---|
| Perception | `hand_tracking.py`, `bimanual.py` | MediaPipe và schema hai tay |
| 3D/world | `hawor_adapter.py`, các script WiLoR 3D | Chuyển output mô hình ngoài sang schema chung |
| Gesture | `pinch.py` | Khoảng cách đầu ngón, hysteresis, gripper state |
| Signal | `smoothing.py` | Nội suy, lọc và giới hạn chuyển động |
| Mapping | `retargeting.py` | Origin, reference, axis mapping, scale, workspace |
| Robot | `openarm_model.py` | Đọc model MuJoCo, joint/site/actuator metadata |
| IK | `ik_solver.py` | IK một tay/hai tay, DLS, orientation tùy chọn |
| Orchestration | `pipeline.py`, `cli.py` | Chạy toàn pipeline offline/live |
| Replay | `mujoco_replay.py`, `viewer.py` | Mô phỏng và hiển thị |
| Data | `dataset.py`, `io.py`, `lerobot_loader.py` | NPZ, metadata, dataset, EgoWorld/LeRobot |
| Evaluation | `validation.py`, `plots.py`, `comparison_video.py` | Metric, biểu đồ và video đối chiếu |

### 4.3 Schema dữ liệu thống nhất

Các nguồn perception khác nhau đều được đổi sang schema:

```text
timestamps
left_valid / right_valid
left_wrist / right_wrist
left_thumb_tip / right_thumb_tip
left_index_tip / right_index_tip
left_middle_tip / right_middle_tip
left_ring_tip / right_ring_tip
left_pinky_tip / right_pinky_tip
left_palm_scale / right_palm_scale
[left_global_orient / right_global_orient]
metadata
```

Thiết kế này là một thành tựu kiến trúc quan trọng: perception có thể thay thế mà không phải viết lại pinch, smoothing, retarget, IK, replay và dataset.

### 4.4 Pipeline video dài của HaWoR

Do HaWoR full-length từng crash ở khoảng frame 1.444/2.100, hệ thống thêm:

```text
video dài
→ chunk 10 giây
→ HaWoR từng chunk trong Docker/GPU
→ convert MANO/world pose
→ overlap 2 giây
→ kiểm tra identity
→ cosine blending
→ ghép trajectory
→ chạy lại retarget + IK trên toàn video
```

Chunking giải quyết giới hạn tài nguyên, nhưng đưa vào bài toán mới: mỗi chunk có hệ chuẩn hóa riêng, tay có thể đổi identity và offset biên có thể tích lũy thành drift.

---

## 5. Quá trình phát triển

### Giai đoạn 1 — Baseline hai tay

- Thu hẹp bài toán về wrist + pinch.
- Xây pipeline offline hai tay, 14 joint và hai gripper.
- Thêm webcam recorder, MuJoCo viewer, replay, dataset và video so sánh.
- Nghiệm thu trên `demo_001`: 562 frame, tracking gần 100%, IK convergence 100%, mean error 0,88 cm.

Kết quả này chứng minh kiến trúc hoạt động, nhưng video dễ và chuyển động trong workspace thuận lợi nên chưa đại diện video nhà máy.

### Giai đoạn 2 — Ego-centric và PhaseB

- Chuyển quy ước camera sang góc nhìn đầu/cổ.
- Bỏ swap tay theo camera đối diện.
- Thiết kế lại axis mapping và workspace trái/phải.
- Thử nhiều scale; chốt PhaseB \(x=0,35,\ y=0,50,\ z=0,25\).
- Tích hợp loader EgoWorld/LeRobot.

Thử nghiệm ngày 18 cho thấy giảm scale có thể tăng convergence nhưng làm robot gần như đứng yên. Đây là lý do rollback và giữ PhaseB thay vì chạy theo metric.

### Giai đoạn 3 — Nâng cấp detector 2D

- Viết adapter EgoForce YOLO hand-only.
- Thử WiLoR/YOLO raw, filtered và tuned.
- Chốt preset WiLoR YOLO tuned với workspace tách hai tay và `max_frame_delta_q=0,15`.

WiLoR YOLO tuned cải thiện PhaseB rõ rệt nhưng vẫn sử dụng pseudo-depth, nên chưa giải quyết bản chất nhập nhằng 3D.

### Giai đoạn 4 — HaWoR world-space

- Chuẩn bị WSL2, Docker CUDA 11.7, PyTorch 1.13, DROID-SLAM, Metric3D, MANO và weights HaWoR.
- Xác nhận GPU RTX 3060 hoạt động trong container.
- Viết adapter đọc `joblib` world result, giải mã MANO và đổi trục HaWoR.
- Sửa axis mapping qua thử nghiệm 10 giây.
- Xây pipeline chunk, overlap, identity check, blending và stitching.

HaWoR đạt convergence rất cao trên bản ghép `factory002_middle`, nhưng kết quả trên `test_v2` cho thấy stitching có thể thất bại nghiêm trọng nếu offset bị cộng dồn.

### Giai đoạn 5 — WiLoR full-3D camera-space

- Dựng môi trường WiLoR full model và YOLO detector.
- Trích xuất 21 khớp MANO, camera translation và global orientation.
- Chuẩn hóa ảnh và log-depth theo percentile.
- Thử jump rejection, median, Savitzky–Golay và quaternion smoothing.
- Thêm `human_reference` tuyệt đối.
- Mở rộng bimanual IK với rotational Jacobian.

Nhánh chạy ổn định đủ 70 giây. Các biến thể absolute/filter/orientation chưa thắng cấu hình cơ bản, nhưng cung cấp nền tảng đúng hơn về mặt hình học.

---

## 6. So sánh thực nghiệm

### 6.1 Bảng kết quả chính trên `factory002_middle`

| Phương pháp | Biểu diễn đầu vào | Quy mô | Tracking L/R | IK convergence | Mean IK error | Nhận xét |
|---|---|---:|---:|---:|---:|---|
| MediaPipe PhaseB | 2D + palm scale | 2.100 | 84,0% / 82,4% | 71,0% | 2,25 cm | Baseline nhẹ, depth yếu |
| EgoForce YOLO | 2D + palm scale | 2.100 | ~95% / ~98% | 71,1% | 2,11 cm | Tracking tốt nhưng không tăng IK |
| WiLoR YOLO raw | 2D + pseudo-depth | 2.100 | 83,6% / 90,4% | 73,7% | 2,01 cm | Cải thiện nhỏ |
| WiLoR YOLO tuned | 2D + pseudo-depth | 2.100 | — | 93,1% | 1,22 cm | Kết quả 2D tốt nhất |
| HaWoR axis_v2 | world-space, MANO/SLAM | 300 | — | 93,33% | 1,22 cm | Clip 10 giây |
| HaWoR stitched | world-space theo chunk | 2.100 | gần 100% | 92,52% | ~1,19 cm* | Không overlap |
| HaWoR overlap 2 s | world-space + blending | 2.100 | gần 100% | **96,71%** | ~1,32 cm* | Convergence cao nhất |
| WiLoR full-3D | camera-space MANO | 2.100 | 88,6% / 92,0% | 93,33% | 1,40 cm | Ổn định, giàu thông tin |
| WiLoR 3D filtered | 3D + lọc mạnh | 2.100 | 86,6% / 89,0% | 93,19% | 1,80 cm | Lọc loại nhầm chuyển động |
| WiLoR 3D absolute | 3D + reference cố định | 2.100 | 86,6% / 89,0% | 92,14% | 1,83 cm | Giữ bố cục nhưng khó IK hơn |
| WiLoR 3D orientation | 3D + rotation target | 2.100 | 86,6% / 89,0% | 91,05% | 1,68 cm | Tư thế tiềm năng tốt hơn, position metric giảm |

\* Mean của sai số trung bình tay trái/phải trong quality report, không phải trung bình theo chunk.

### 6.2 Cách đọc bảng đúng

Các kết quả không hoàn toàn đồng nhất về perception, normalization, preset và cách ghép; vì vậy không nên xem chúng như benchmark tuyệt đối. Bốn điểm cần lưu ý:

1. **IK error đo robot so với target robot**, không đo target robot so với tay người.
2. Scale nhỏ hoặc clipping mạnh có thể làm IK error đẹp nhưng giảm motion fidelity.
3. Tracking cao không bảo đảm depth và identity đúng.
4. Kết quả 10 giây không đại diện video 70 giây.

### 6.3 MediaPipe PhaseB so với các detector 2D

MediaPipe PhaseB có ưu điểm triển khai đơn giản, tốc độ tốt và đủ làm baseline. EgoForce YOLO tăng tracking nhưng convergence gần như không đổi: điều này chứng minh detector 2D tốt hơn chưa đủ nếu depth vẫn lấy từ palm scale.

WiLoR YOLO tuned tăng convergence từ 71,0% lên 93,1% và giảm mean error từ 2,25 cm xuống 1,22 cm. Phần tăng này đến từ tổng hợp tuning detector, scale, workspace và giới hạn vận tốc khớp, chứ không chỉ do thay mô hình perception.

### 6.4 HaWoR so với WiLoR full-3D

| Tiêu chí | HaWoR world-space | WiLoR full-3D camera-space |
|---|---|---|
| Thông tin không gian | World-space, có camera motion/SLAM | Camera-space 3D theo từng frame |
| Hạ tầng | Docker nặng, nhiều weights, CUDA/DROID-SLAM | Nhẹ hơn HaWoR, vẫn cần GPU |
| Video dài | Phải chunk do crash/tài nguyên | Chạy trực tiếp đủ 70 giây |
| Kết quả tốt nhất | 96,71% IK với overlap | 93,33% IK, 1,40 cm |
| Rủi ro | Stitching, drift, chuẩn hóa từng chunk | Scale/depth calibration, tracking loss |
| Orientation | Có MANO pose nhưng chưa tích hợp đầy đủ vào IK ở nhánh này | Đã truyền global orientation vào IK |
| Khả năng vận hành | Phức tạp | Ổn định và dễ lặp lại hơn |

HaWoR dẫn đầu convergence trên một video, nhưng thất bại trên `test_v2` khi căn biên cộng dồn:

- Có căn biên: **3,39%**.
- Không căn biên: **55,17%**.

Điều này cho thấy HaWoR chưa đủ robust để chọn làm pipeline chính. WiLoR full-3D có metric thấp hơn đôi chút nhưng có đường chạy đơn giản, ổn định hơn và dễ phát triển orientation/absolute mapping.

### 6.5 Absolute retarget và orientation

Trên clip 10 giây, absolute retarget tăng convergence từ 88,0% lên 95,0%. Trên 70 giây, nó giảm từ 93,33% xuống 92,14% và tăng mean error từ 1,40 cm lên 1,83 cm. Nguyên nhân hợp lý là reference cố định giữ khoảng cách tuyệt đối nhưng cũng đưa target đến vùng khó với robot; frame đầu tương đối dễ IK hơn nhưng làm mất bố cục thật.

Orientation giới hạn đạt 94,67% trên 10 giây nhưng chỉ 91,05% trên 70 giây. Position và rotation cùng tranh bậc tự do, trong khi mapping orientation camera–robot và confidence chưa đủ ổn định. Vì vậy orientation nên là mục tiêu mềm, bật theo chất lượng dữ liệu thay vì constraint cố định mọi frame.

### 6.6 Temporal filtering

WiLoR 3D filtered làm tracking giảm từ 88,6%/92,0% xuống 86,6%/89,0%, convergence gần như không tăng và mean error tăng 1,40→1,80 cm. Jump rejection đang loại cả chuyển động nhanh hợp lệ; smoothing cố định không thích nghi với vận tốc. Kết quả này ủng hộ confidence-aware/adaptive filtering thay vì tăng cửa sổ lọc.

---

## 7. Thành quả kỹ thuật

- Hoàn thiện pipeline hai tay gồm 14 arm joint và hai gripper độc lập.
- Thiết kế schema hand pose thống nhất cho nhiều backend perception.
- Xây CLI, webcam recording, offline pipeline, live control và MuJoCo viewer.
- Có pinch hysteresis, interpolation, smoothing, retarget, workspace constraint và Jacobian DLS IK.
- Có replay, human-vs-robot comparison, plots, quality report và dataset NPZ.
- Tích hợp EgoWorld/LeRobot loader và baseline policy để chuẩn bị imitation learning.
- Tạo ba bộ preset tái lập: WiLoR YOLO tuned, HaWoR world tuned và WiLoR 3D camera.
- Dựng hạ tầng WSL2/Docker/GPU cho HaWoR, DROID-SLAM, Metric3D và MANO.
- Xử lý video dài bằng chunk, overlap, identity cost, cosine blending và stitching.
- Mở rộng retarget với human reference tuyệt đối.
- Mở rộng bimanual IK từ position-only sang position + orientation.
- Xây test cho perception, retarget, IK, pipeline, replay, dataset, I/O và adapter.

---

## 8. Hạn chế

### 8.1 Hạn chế dữ liệu và đánh giá

- Phần lớn tuning tập trung vào `factory002_middle`; nguy cơ overfit theo video.
- Chưa có ground-truth 3D của tay người hoặc robot target.
- Chưa có benchmark nhiều người, nhiều góc quay, ánh sáng và tốc độ.
- Đánh giá trực quan chưa được chuyển thành thang điểm hoặc annotation có hệ thống.

### 8.2 Hạn chế perception

- Occlusion, motion blur, tay ra khỏi khung và đổi handedness vẫn gây lỗi.
- Camera đơn không cung cấp depth tuyệt đối chắc chắn.
- WiLoR/HaWoR dựa trên MANO prior, có thể tạo pose hợp lý về hình học nhưng không đúng tuyệt đối.

### 8.3 Hạn chế retarget và IK

- Scale/workspace chủ yếu được tune thủ công.
- Clipping target có thể che lỗi perception và làm méo trajectory.
- Position IK chưa tối ưu posture, collision và joint comfort đầy đủ.
- Orientation mapping chưa confidence-aware.
- Chưa đánh giá self-collision, object contact và dynamic feasibility.

### 8.4 Hạn chế vận hành

- HaWoR nặng, nhiều dependency và dễ gặp lỗi tài nguyên.
- Chunk stitching chưa robust trên video khác.
- Chưa có pipeline tự động chọn backend hoặc fallback khi tracking kém.

---

## 9. Hướng đi chính

### 9.1 Chọn WiLoR full-3D làm nhánh phát triển chính

WiLoR full-3D nên là pipeline nghiên cứu chính vì:

- Chạy trực tiếp đủ 70 giây.
- Cung cấp 3D camera-space và orientation.
- Metric gần WiLoR tuned/HaWoR nhưng hạ tầng ít phức tạp hơn HaWoR.
- Phù hợp để phát triển calibration, adaptive filtering và orientation-aware retarget.

Giữ:

- **MediaPipe PhaseB** làm baseline nhẹ/fallback.
- **WiLoR YOLO tuned** làm baseline 2D mạnh.
- **HaWoR** làm nhánh world-space tham chiếu, không dùng mặc định cho đến khi stitching ổn định trên nhiều video.

### 9.2 Thiết kế vòng thí nghiệm tiếp theo

Mỗi thử nghiệm chỉ thay một yếu tố và báo cáo tối thiểu:

- Tracking ratio từng tay.
- Mean/P95/max IK error.
- Convergence ratio.
- Motion range của tay người, target và end-effector.
- Tỷ lệ workspace clipping.
- Vận tốc/jerk và số jump.
- Hand identity switches.
- Pinch precision/transition count.
- Runtime/FPS và GPU memory.
- Điểm đánh giá trực quan trên các đoạn khó.

### 9.3 Calibration và retarget

- Ước lượng reference từ median của một cửa sổ ổn định, không lấy frame đầu.
- Calibration riêng từng tay và từng người.
- Giữ khoảng cách tương đối hai tay nhưng không ép absolute target ngoài workspace.
- Dùng robust percentile cho scale, có giới hạn và margin workspace.
- Báo cáo clipping thay vì âm thầm clip.

### 9.4 Lọc thích nghi

- Gating theo detector confidence và reprojection/pose confidence.
- Ngưỡng jump phụ thuộc vận tốc cục bộ.
- One Euro hoặc Kalman filter với tham số theo trạng thái chuyển động.
- Không đánh dấu invalid chỉ vì chuyển động nhanh nếu track identity vẫn chắc chắn.

### 9.5 Orientation mềm

- Chỉ bật orientation khi position error đã dưới ngưỡng.
- Weight theo confidence và giảm gần joint limit/singularity.
- Giới hạn tốc độ đổi orientation theo frame.
- Đánh giá bằng góc sai lệch cổ tay và chất lượng trực quan, không chỉ position error.

### 9.6 Hoàn thiện HaWoR

- Không dùng offset cộng dồn làm mặc định.
- Dùng overlap transform robust, giới hạn offset và từ chối boundary có cost cao.
- Chuẩn hóa toàn sequence hoặc anchor theo camera/world frame chung.
- Kiểm thử ít nhất ba video trước khi so sánh lại với WiLoR 3D.

### 9.7 Chuẩn bị imitation learning

Chỉ thu dataset huấn luyện sau khi retarget ổn định. Mỗi episode cần:

- Đồng bộ timestamp video, pose, target, qpos và gripper.
- Lưu validity/confidence để mask frame lỗi.
- Cân bằng trạng thái mở/đóng gripper.
- Tách train/validation theo video hoặc người, không tách ngẫu nhiên theo frame.
- Lưu version của backend perception và preset để tái lập.

---

## 10. Kết luận

Dự án đã đi từ một ý tưởng wrist + pinch thành hệ thống retarget hai tay hoàn chỉnh, có nhiều backend perception, mô phỏng MuJoCo, đánh giá định lượng, video đối chiếu và dữ liệu cho học bắt chước.

Ba bài học cốt lõi là:

1. **Depth và hệ tọa độ là vấn đề trung tâm.** Thay detector 2D không đủ nếu depth vẫn là palm-scale proxy.
2. **IK convergence không đồng nghĩa với chuyển động đúng.** Target sai, scale nhỏ hoặc clipping mạnh vẫn có thể tạo metric đẹp.
3. **Sự ổn định toàn video quan trọng hơn kết quả clip ngắn.** Absolute retarget, filtering, orientation và stitching đều cho thấy lợi ích ngắn hạn có thể biến mất trên 70 giây.

Hướng chính nên là **WiLoR full-3D camera-space + calibration robust + lọc thích nghi + orientation mềm**, đánh giá đồng thời bằng metric hình học, động học và video trực quan. HaWoR tiếp tục là nhánh world-space tiềm năng; MediaPipe và WiLoR YOLO tuned được giữ làm baseline để đo đúng giá trị của từng cải tiến.
