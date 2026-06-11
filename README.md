# Video to OpenArm

Pipeline chuyển chuyển động cổ tay và động tác pinch từ video/webcam thành quỹ đạo
end-effector và lệnh gripper cho OpenArm trong MuJoCo.

Tài liệu thiết kế đầy đủ nằm tại [PROJECT_PLAN.md](PROJECT_PLAN.md). Hướng dẫn chạy
từng bước và kết quả kiểm chứng sẽ được cập nhật cùng quá trình triển khai.

## Cài đặt tối thiểu

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
```

Để chạy toàn bộ vision và simulation:

```bash
python -m pip install -e ".[all,dev]"
```

