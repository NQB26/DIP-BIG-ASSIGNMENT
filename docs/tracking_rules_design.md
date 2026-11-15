## B. Phân hệ Tracking & Safety Logic (`src/tracking_rules/`)

### 1. Tổng quan

```
[Detection] ─▶ [PersonTracker (ByteTrack)] ─▶ [Rule Engine: hình học R1/R2/R3] ─▶ [FSM lọc thời gian] ─▶ [RuleEvent JSON]
                                                       ▲
                                        configs/zones.yaml, configs/rules.yaml
```

Nguyên tắc thiết kế đã chốt:

1. **Chỉ track `person`.** `lab_coat` và `cell_phone` được ghép với person bằng hình học ở từng frame (không track) → ít ID switch, code đơn giản.
2. **Thời gian = video time** (`frame_id / fps`), không dùng đồng hồ hệ thống → chạy offline/real-time cho cùng kết quả, test lặp lại được.
3. **Ngưỡng nằm hết trong YAML**; code không có số "ma thuật".
4. **Mỗi cặp `(track_id, rule_id)` có 1 FSM riêng**; track vắng mặt vẫn được cập nhật `violated=False` để grace/cooldown chạy đúng.
5. **Detector chạy ngưỡng thấp (0.25)**; việc lọc theo conf do tracker/rule đảm nhiệm (ByteTrack cần cả detection conf thấp để nối track).

### 2. Tracking (ByteTrack)

| Tham số (`configs/rules.yaml → tracker`, có mặc định trong code) | Mặc định | Ý nghĩa / khi nào chỉnh |
|---|---|---|
| `lost_buffer_sec` | 1.5 | Giữ track khi bị che. Tăng nếu hay bị đổi ID sau khi che; giảm nếu ID "ma" tồn tại lâu |
| `activation_threshold` | 0.5 | Conf tối thiểu để khởi tạo track mới |
| `high_conf_threshold` | 0.6 | Ngưỡng tách detection conf cao/thấp (2 giai đoạn association) |
| `min_consecutive_frames` | 2 | Số frame liên tiếp mới xác nhận track (giảm ID rác) |
| `min_iou` | 0.1 | IoU tối thiểu khi ghép |

Triển khai: `tracker.py` (`PersonTracker`) dùng `trackers.ByteTrackTracker`; chỉ trả các track đã xác nhận (`track_id >= 0`).

### 3. Ba quy tắc

Điều kiện chung: `person.conf ≥ person_conf_min` và **chân** `P_foot = ((x1+x2)/2, y2)` nằm trong zone phù hợp. Nếu không khai báo zone `general` nào, toàn khung hình được coi là `general`.

**R1 – `NO_LAB_COAT`** (zone `general`)
- Áp dụng khi `height(person) ≥ min_person_height_px` (giữ `min_person_height_px: 100` như `rules.yaml` hiện có; nếu muốn độc lập độ phân giải, xoá key này và dùng `common.min_person_height_ratio`, mặc định 0.25).
- Ghép áo–người: `IoA = Area(coat ∩ person) / Area(coat) ≥ min_ioa_threshold (0.6)` và `coat.conf ≥ coat_conf_min`. **Mỗi áo chỉ gán cho 1 người** (người có IoA cao nhất) để 2 người đứng sát nhau không "dùng chung" 1 áo.
- `coat_hold_sec = 0.5`: sau lần thấy áo gần nhất, vẫn coi là có áo 0.5s (chống detector chớp tắt).
- Vi phạm khi *không* có áo; xác nhận sau `min_duration_sec = 2.5s`.

**R2 – `RESTRICTED_ZONE`**
- Vi phạm khi `P_foot` nằm trong polygon `type: restricted` (point-in-polygon, ray casting). Xác nhận sau `1.5s`.

**R3 – `PHONE_USAGE`** (zone `general`)
- ROI thân trên: `y1 … y1 + upper_body_ratio × (y2−y1)` (0.65), cùng chiều rộng bbox person.
- Tâm `cell_phone` (`conf ≥ 0.3`) nằm trong ROI → gán cho person có tâm ROI gần nhất. Xác nhận sau `3.0s`.
- Điện thoại đặt trên bàn thường thấp hơn ROI nên bị loại; xem mục "Hạn chế".

### 4. FSM lọc thời gian

```
NORMAL ──vi phạm──▶ PENDING ──đủ min_duration──▶ CONFIRMED ──(bước kế)──▶ COOLDOWN ──hết cooldown──▶ NORMAL
                      │
                      └── mất vi phạm > grace_period ──▶ NORMAL   (reset bộ đếm)
```

| Trạng thái | Hành vi |
|---|---|
| `NORMAL` | Chưa vi phạm. Có vi phạm → `PENDING`, ghi `start_t` |
| `PENDING` | Mất vi phạm ≤ `grace_period_sec` → giữ nguyên bộ đếm; > grace → về `NORMAL`. `t − start_t ≥ min_duration_sec` → `CONFIRMED` |
| `CONFIRMED` | Trạng thái thoáng qua đúng 1 bước: **phát 1 event**, rồi sang `COOLDOWN` |
| `COOLDOWN` | Bỏ qua mọi đầu vào trong `cooldown_sec`; hết thì về `NORMAL`. Nếu hành vi vẫn tiếp diễn, chu kỳ lặp lại (nhắc lại sau ≈ `cooldown + min_duration`) |

(Bản nháp cũ có mũi tên "mất dấu quá lâu → COOLDOWN"; đã sửa thành `→ NORMAL` vì chưa phát cảnh báo thì không cần cooldown.)

**Chống trùng khi ID switch:** nếu cùng `rule_id` đã có event trong `cooldown_sec` gần nhất và bbox mới có IoU ≥ `dedup_iou (0.5)` với bbox event đó thì bỏ qua (người cũ bị đổi sang `track_id` mới sẽ không bị báo lại).

### 5. Data contract (khớp `data_contract.md` hiện có)

**Đầu vào Rule Engine** giữ nguyên: mỗi frame có `tracks` (person kèm `track_id`) và `objects` (`lab_coat`, `cell_phone`, key tên lớp là `class`). `RuleEngine.update(frame_id, t, tracks, objects, (W, H))` nhận đúng hai danh sách này (đổi `class` → `class_name` khi tạo `Track`/`Detection`; tên lớp được chuẩn hoá `cell phone` → `cell_phone`).

**Đầu ra `RuleEvent`**: giữ nguyên toàn bộ trường trong contract để Dashboard của Hậu không phải sửa; mọi trường mới đều **tuỳ chọn (additive)**.

| Nhóm | Trường |
|---|---|
| Lõi (có sẵn trong contract) | `event_id`, `rule_id`, `rule_name`, `track_id`, `status`, `timestamp`, `video_time_sec`, `duration_sec`, `zone_id`, `bbox`, `evidence{image_path, clip_path}` |
| Mở rộng (mới, tuỳ chọn) | `schema_version`, `severity`, `frame_id`, `violation_start_sec`, `zone_type`, `frame_size`, `confidence`, `details`, `source` |

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
  "evidence": { "image_path": null, "clip_path": null },

  "schema_version": "1.0",
  "severity": "medium",
  "frame_id": 145,
  "violation_start_sec": 1.78,
  "zone_type": "general",
  "frame_size": { "width": 1280, "height": 720 },
  "confidence": 0.91,
  "details": { "phone_bbox": [375.0, 240.0, 405.0, 285.0], "phone_confidence": 0.84 },
  "source": { "id": "lab_cam01.mp4", "type": "video_file" }
}
```

Ý nghĩa các trường mở rộng:
- `severity`: lấy từ `rules.yaml` (R2 = `high`) để Dashboard tô màu/sắp xếp.
- `frame_id`, `violation_start_sec`: tua đúng đoạn minh chứng cho Event Manager.
- `zone_type`, `frame_size`: Dashboard tự vẽ lại bbox/zone theo tỉ lệ.
- `details` theo rule: R1 `{best_ioa, person_height_px}`; R2 `{foot_point_px, foot_point_norm}`; R3 `{phone_bbox, phone_confidence}`.
- `evidence` luôn `null` ở Rule Engine; **Event Manager** điền sau khi lưu ảnh/clip.
- `timestamp` giữ định dạng giờ địa phương không múi giờ như contract gốc (nếu sau này cần múi giờ thì đổi cùng Hậu).

Schema đầy đủ: `events/event_schema.json` (`additionalProperties` mở, nên thêm trường sau này không làm vỡ validate). Mọi event ghi ra `events.jsonl`.

### 6. Cách xác định và vẽ zone

**Nguyên tắc**
1. **Camera cố định.** Zone gắn với khung hình; nếu camera bị xoay/dời thì phải vẽ lại.
2. **Lưu toạ độ chuẩn hoá `[0,1]`** (`x/W`, `y/H`) → dùng được cho mọi độ phân giải. Vẽ trên đúng góc nhìn/crop mà detector nhìn thấy (không letterbox/crop khác).
3. **Vẽ theo mặt sàn, không theo vật thể.** R1/R3 và R2 đều dựa vào *điểm chân*, nên polygon phải phủ phần **sàn** nơi người đứng, không phủ cả thân người hay tủ/bàn.
4. **`general`**: phủ toàn bộ sàn khu thí nghiệm, bỏ cửa ra vào/hành lang (tránh báo "không áo blouse" khi người vừa bước vào).
5. **`restricted`**: vẽ phần sàn phía trước khu vực cấm, **nới ra ~0.5 m** để chân chạm vào là bắt được; có thể chồng lên `general`.
6. Polygon lồi, 4–8 đỉnh là đủ; đỉnh đi theo 1 chiều (cùng/ngược kim đồng hồ), không tự cắt nhau.
7. Chọn frame **không có người che** (`--frame-idx`) và nên có người đứng ở các mép zone để kiểm chứng.

**Công cụ**: `scripts/draw_zones.py`

```bash
python scripts/draw_zones.py --video data/lab.mp4 --frame-idx 0 --out configs/zones.yaml --camera-id cam01
```

Chuột trái thêm đỉnh · chuột phải/`u` xoá đỉnh · `g` đóng thành zone `general` · `r` đóng thành zone `restricted` · `d` xoá zone cuối · `s` lưu · `q` thoát. `zone_id` tự đặt `general_1`, `restricted_1`… → **mở YAML đổi `zone_id`/`name` thành tên có nghĩa** vì `zone_id` xuất hiện trong event và Dashboard. Chạy lại lệnh sẽ nạp zone cũ để chỉnh tiếp.

Định dạng `configs/zones.yaml` (giữ đúng file hiện có: `zone_id`, `name`, `type`, `polygon`):

```yaml
zones:
  - zone_id: "lab_main_area"
    name: "Khu vực thí nghiệm chung"
    type: "general"
    polygon: [[0.05, 0.10], [0.95, 0.10], [0.95, 0.95], [0.05, 0.95]]
  - zone_id: "restricted_chemical_zone"
    name: "Khu vực tủ hóa chất hạn chế"
    type: "restricted"
    polygon: [[0.65, 0.20], [0.92, 0.20], [0.92, 0.75], [0.65, 0.75]]
```

Hai zone mẫu hiện là hình chữ nhật đặt tạm: phải vẽ lại trên khung hình camera thật.

### 7. Hướng dẫn test tracker trên video

**Cài đặt:** `pip install -r requirements_tracking.txt`

**Bước 1 – Unit test logic (không cần video/GPU):**
```bash
PYTHONPATH=src pytest tests -q
```
Bao phủ hình học, FSM (confirm/grace/cooldown), R1/R2/R3, ID-switch dedup, schema.

**Bước 2 – Chạy pipeline trên video:**
```bash
python scripts/test_tracker.py --video data/lab.mp4 --weights models/best.pt \
       --zones configs/zones.yaml --rules configs/rules.yaml --out runs/test01 --show
```
Đầu ra trong `runs/test01/`: `annotated.mp4` (overlay), `events.jsonl`, `tracks.csv`, `summary.json`.

Cách đọc overlay: khung trắng = bình thường, **vàng** = đang `PENDING` (hiện `R1 1.4/2.5s`), **đỏ** = đã cảnh báo; chấm vàng = điểm chân (R2); khung tím mỏng = ROI thân trên (R3); khung xanh dương/tím = `lab_coat`/`cell_phone`.

**Bước 3 – Tinh chỉnh không cần chạy lại detector:** lưu detection ra `dets.jsonl` (đúng contract §4.1, mỗi dòng `{"frame_id":..,"detections":[...]}`), rồi chạy `--dets-json dets.jsonl` thay cho `--weights` để thử ngưỡng rule trong vài giây.

**Bộ video test tối thiểu (mỗi clip 20–60 s):**

| Clip | Kiểm tra |
|---|---|
| 1 người đi bộ, mặc áo | Giữ 1 ID; không có event nào (đo false positive R1) |
| 1 người không mặc áo | R1 bắn đúng 1 lần sau ~2.5s, lặp lại sau cooldown |
| 2 người đi ngang, che nhau | ID có bị đổi không (ID switch) |
| Người đi qua ranh giới zone restricted | R2 bắn khi chân vào; đi lướt <1.5s thì không bắn |
| Cầm/nghe điện thoại; điện thoại đặt trên bàn | R3 bắn ở case cầm, không bắn ở case để bàn |
| Người ra khỏi khung rồi quay lại | ID mới là bình thường; kiểm tra không báo trùng |

**Chỉ số đánh giá**
- *Tracker:* số ID duy nhất so với số người thực; `short_tracks_lt_1s` (track vụn) trong `summary.json`; đếm ID switch thủ công khi xem `annotated.mp4` (hoặc dùng `tracks.csv` + nhãn tay với `py-motmetrics` để ra IDF1/ID-switch).
- *Rule:* tự gán nhãn các đoạn vi phạm thật (bắt đầu–kết thúc) rồi so với `events.jsonl` với dung sai ±2 s → Precision/Recall cho từng R1–R3.

**Chỉnh tham số theo triệu chứng**

| Triệu chứng | Chỉnh |
|---|---|
| Người cùng 1 người bị đổi ID nhiều | tăng `lost_buffer_sec`, giảm `high_conf_threshold`, kiểm tra detector có đứt person |
| Nhiều ID rác ngắn | tăng `min_consecutive_frames`, `activation_threshold` |
| R1 báo nhầm người đang mặc áo | giảm `min_ioa_threshold`, tăng `coat_hold_sec`/`min_duration_sec`, hạ `coat_conf_min` |
| R2 bắn khi mới đi sát vùng cấm | nới zone ít hơn, tăng `min_duration_sec` |
| R3 bắn khi điện thoại trên bàn | giảm `upper_body_ratio` (0.55), tăng `phone_conf_min` |
| R3 bỏ sót khi điện thoại bị tay che | giảm `phone_conf_min`, tăng `grace_period_sec` |
| Event lặp quá dày | tăng `cooldown_sec` |

### 8. Hạn chế đã biết
- Che khuất kéo dài làm mất áo/ID có thể gây R1 giả; giảm bằng `coat_hold_sec`, `min_duration_sec` hoặc detector tốt hơn cho `lab_coat`.
- R3 chỉ dựa vị trí 2D của điện thoại; chưa phân biệt "đang cầm" và "đặt trên bàn phía trước người" khi bàn nằm trong ROI. Hướng mở rộng: thêm zone `ignore` cho mặt bàn hoặc dùng pose (cổ tay).
- Zone dựa vào camera cố định; camera di động cần homography/vẽ lại.
