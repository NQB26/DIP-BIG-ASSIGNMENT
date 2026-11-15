# DIP-BIG-ASSIGNMENT

## B.Phân hệ Tracking & Safety Logic (`src/tracking_rules/`)

### 1. Tổng quan & Vai trò trong hệ thống
Phân hệ **Tracking & Safety Logic** đóng vai trò xử lý nghiệp vụ an toàn giữa đầu ra của mô hình nhận diện đối tượng (Object Detection) và giao diện cảnh báo (Dashboard). Nhiệm vụ chính của module bao gồm:
- **Theo dõi đa đối tượng (Multi-Object Tracking):** Duy trì định danh (`track_id`) ổn định cho từng người qua các khung hình liên tiếp.
- **Phân tích hình học không gian (Spatial Analysis):** Kiểm tra vị trí tương đối giữa người, trang bị bảo hộ (`lab_coat`), vật thể (`cell_phone`) và các vùng giám sát (`zones`).
- **Lọc nhiễu thời gian (Temporal Filtering):** Sử dụng máy trạng thái hữu hạn (FSM) để loại bỏ cảnh báo giả do mất detection thoáng qua và chống lặp cảnh báo.

```
[Detection Output] ──▶ [Tracker (ByteTrack)] ──▶ [Spatial Rule Engine] ──▶ [Temporal Filter (FSM)] ──▶ [Confirmed JSON Events]
                                                         ▲
                                           [zones.yaml & rules.yaml]
```

---

### 2. Đặc tả thuật toán 3 quy tắc an toàn

Các tham số ngưỡng của từng quy tắc được tách rời và quản lý tập trung tại `configs/zones.yaml` và `configs/rules.yaml`.

#### R1 – Không mặc áo blouse (`NO_LAB_COAT`)
- **Phạm vi áp dụng:** Người (`person`) di chuyển bên trong vùng thí nghiệm chung (`type: "general"`).
- **Phương pháp ghép cặp:** Sử dụng chỉ số **IoA (Intersection over Area)** thay vì IoU thông thường để khắc phục sự chênh lệch kích thước giữa khung người và khung áo:
  $$\text{IoA} = \frac{\text{Area}(\text{bbox}_{\text{person}} \cap \text{bbox}_{\text{lab\_coat}})}{\text{Area}(\text{bbox}_{\text{lab\_coat}})}$$
- **Điều kiện xác nhận:**
  - Chiều cao khung người thỏa mãn $\text{height}(\text{bbox}_{\text{person}}) \ge \text{min\_person\_height\_px}$ (loại bỏ người ở quá xa hoặc bị cắt khung hình).
  - Không tìm thấy `lab_coat` nào có $\text{IoA} \ge \text{min\_ioa\_threshold}$ liên tục trong khoảng thời gian `min_duration_sec` ($2.5\text{s}$).

#### R2 – Đi vào vùng hạn chế (`RESTRICTED_ZONE`)
- **Phạm vi áp dụng:** Các đa giác vùng cấm/hạn chế (`type: "restricted"`) được định nghĩa bằng tọa độ chuẩn hóa $[0.0, 1.0]$ trong `configs/zones.yaml`.
- **Phương pháp kiểm tra:** Sử dụng thuật toán **Point-in-Polygon** đối với **điểm chân (Bottom-Center Anchor)** của khung người để hạn chế sai số góc nhìn camera:
  $$P_{\text{foot}} = \left(\frac{x_1 + x_2}{2},\; y_2\right)$$
- **Điều kiện xác nhận:** Điểm $P_{\text{foot}}$ nằm bên trong đa giác vùng hạn chế liên tục tối thiểu `min_duration_sec` ($1.5\text{s}$).

#### R3 – Sử dụng điện thoại trong khu vực thí nghiệm (`PHONE_USAGE`)
- **Phạm vi áp dụng:** Người (`person`) hoạt động trong khu vực thí nghiệm chung (`type: "general"`).
- **Phương pháp ghép cặp:** Xác định **vùng thân trên / tay (Upper-Body RoI)** của người giới hạn từ $y_1$ đến $y_1 + \text{upper\_body\_ratio} \times (y_2 - y_1)$ (mặc định $65\%$ chiều cao phía trên). Kiểm tra tâm của khung `cell_phone` có nằm trong vùng thân trên của `person` hay không (loại trừ trường hợp điện thoại đặt trên bàn).
- **Điều kiện xác nhận:** Điện thoại duy trì trong vùng thân trên của cùng một `track_id` liên tục tối thiểu `min_duration_sec` ($3.0\text{s}$).

---

### 3. Bộ lọc thời gian & Máy trạng thái (Temporal Filtering FSM)

Để tránh hiện tượng nhấp nháy cảnh báo (flickering) và trùng lặp sự kiện, mỗi cặp `(track_id, rule_id)` được quản lý độc lập qua 4 trạng thái:

```
  ┌──────────┐  Phát hiện vi phạm   ┌──────────┐  Đạt ngưỡng T (s)   ┌───────────┐
  │  NORMAL  │ ───────────────────▶ │ PENDING  │ ──────────────────▶ │ CONFIRMED │
  └──────────┘                      └──────────┘                     └───────────┘
       ▲                              │      ▲                             │
       │    Hết thời gian             │      │ Mất dấu < grace_period      │ Xuất Event
       │    cooldown                  ▼      │ (giữ nguyên bộ đếm)         ▼
  ┌──────────┐                  ┌────────────────┐                   ┌──────────┐
  │ COOLDOWN │ ◀────────────────│ Reset nếu mất  │                   │ COOLDOWN │
  └──────────┘                  │ dấu quá lâu    │                   └──────────┘
                                └────────────────┘
```

- **`NORMAL`:** Trạng thái bình thường, chưa ghi nhận vi phạm.
- **`PENDING`:** Bắt đầu tích lũy thời gian vi phạm. Nếu AI bị mất nhận diện thoáng qua ngắn hơn `grace_period_sec` ($0.3\text{s} - 0.4\text{s}$), bộ đếm thời gian được giữ nguyên để chống đứt quãng.
- **`CONFIRMED`:** Thời gian vi phạm tích lũy vượt ngưỡng `min_duration_sec` $\rightarrow$ Phát sinh một bản ghi sự kiện (Event) gửi sang module quản lý sự kiện và Dashboard.
- **`COOLDOWN`:** Tạm khóa cảnh báo cho chính cặp `(track_id, rule_id)` trong khoảng `cooldown_sec` ($10\text{s} - 15\text{s}$) nhằm tránh spam cảnh báo cho cùng một hành vi đang diễn ra.

---

### 4. Hợp đồng dữ liệu (Data Contract)

#### 4.1. Đầu vào từ Detection (`Detection`)
Danh sách các đối tượng nhận diện được trên từng khung hình:
```json
{
  "frame_id": 145,
  "timestamp": 4.83,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.91,
      "bbox": [320.0, 140.0, 450.0, 480.0]
    },
    {
      "class_name": "cell_phone",
      "confidence": 0.84,
      "bbox": [375.0, 240.0, 405.0, 285.0]
    }
  ]
}
```

#### 4.2. Đầu ra từ Tracking (`Track`)
Danh sách đối tượng sau khi được gán ID xuyên suốt các khung hình:
```json
{
  "track_id": 3,
  "class_name": "person",
  "confidence": 0.91,
  "bbox": [320.0, 140.0, 450.0, 480.0]
}
```

#### 4.3. Đầu ra sự kiện vi phạm (`RuleEvent`)
Cấu trúc JSON chuẩn truyền sang Event Manager & Dashboard khi trạng thái chuyển sang `CONFIRMED`:
```json
{
  "event_id": "EVT_20261115_0001",
  "rule_id": "R3",
  "rule_name": "PHONE_USAGE",
  "track_id": 3,
  "status": "CONFIRMED",
  "timestamp": "2026-11-15T10:15:32.400",
  "video_time_sec": 4.83,
  "duration_sec": 3.05,
  "zone_id": "lab_main_area",
  "bbox": [320.0, 140.0, 450.0, 480.0],
  "evidence": {
    "image_path": null,
    "clip_path": null
  }
}
```
*(Ghi chú: Trường `evidence` được khởi tạo là `null` tại Rule Engine và sẽ được Event Manager cập nhật đường dẫn sau khi lưu ảnh/clip minh chứng).*

---

### 5. Cấu trúc mã nguồn dự kiến (`src/tracking_rules/`)

```text
src/tracking_rules/
├── __init__.py
├── tracker.py          # Wrapper tích hợp ByteTrack duy trì track_id
├── geometry.py         # Các hàm hình học: bottom_center, point_in_polygon, compute_ioa
├── state_machine.py    # Bộ lọc thời gian FSM (Debounce & Cooldown)
└── rule_engine.py      # Điều phối kiểm tra R1, R2, R3 và xuất danh sách RuleEvent
```