# DIP-BIG-ASSIGNMENT

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

## C. Integration / Dashboard

Phần việc của TV3 theo kế hoạch: pipeline đọc video/webcam, tích hợp module, dashboard, lưu ảnh/clip sự kiện **bất đồng bộ**, log, đánh giá end-to-end.

### 1. Cài đặt vào repo

Chép các thư mục này vào thư mục gốc dự án `DIP-BIG-ASSIGNMENT-main`:

```
src/pipeline/     runner.py, event_manager.py, tracker_adapter.py, overlay.py, metrics.py, eventlog.py
src/dashboard/    server.py, stats.py, static/index.html
src/evaluation/   evaluate.py
tests/test_tv3.py
scripts/run_pipeline.ps1, scripts/run_dashboard.ps1
data/test_ground_truth/ground_truth_TEMPLATE.json
```

**Không cần cài thêm thư viện**: chỉ dùng `opencv-python`, `numpy`, `pyyaml` và các gói sẵn có của dự án. Dashboard dùng thư viện chuẩn Python.

Các script tự thêm `src/` và thư mục gốc vào `sys.path`, nên `tracking_rules` và `perception` phải nằm ở `src/` hoặc ở thư mục gốc.

### 2. Chạy

Terminal 1 (đã kích hoạt `.venv`, đứng ở thư mục gốc):

```powershell
.\scripts\run_pipeline.ps1 -Rules .\configs\rules.yaml -Out .\runs\demo_all -Show
# hoặc đầy đủ tham số:
python .\src\pipeline\runner.py --source .\data\zone_test.mp4 --weights .\yolo11n.pt `
  --lab-coat-weights $coat --zones .\configs\zones.yaml --rules .\configs\rules.yaml `
  --out .\runs\demo_all --imgsz 512 --device cpu --coco-conf 0.25 --coat-conf 0.25
# webcam:
python .\src\pipeline\runner.py --source 0 --out .\runs\webcam ...
```

Terminal 2:

```powershell
.\scripts\run_dashboard.ps1          # mở http://127.0.0.1:8000
```

Dashboard cập nhật mỗi 1,5 giây: khung hình trực tiếp, FPS/độ trễ, thẻ KPI theo rule, biểu đồ cảnh báo theo thời gian, danh sách cảnh báo (lọc/tìm kiếm), cửa sổ chi tiết có **ảnh + clip minh chứng**, gán nhãn **Đúng (TP) / Cảnh báo giả (FP)**, xuất CSV. Dashboard cũng đọc được `events.jsonl` của `test_tracker.py` (các thư mục `runs\r1_validation_v3`… sẽ hiện trong ô "Lượt chạy"; các lượt này không có ảnh/clip).

Tùy chọn pipeline hữu ích: `--max-frames`, `--frame-stride 2` (xử lý mỗi 2 khung để tăng FPS trên CPU), `--realtime` (phát video đúng tốc độ gốc để demo), `--clip-pre/--clip-post` (mặc định 3 s / 2 s), `--no-save-video`.

Thư mục một lượt chạy gồm: `events.jsonl`, `status.json`, `pipeline_summary.json`, `live.jpg`, `pipeline_annotated.mp4`, `evidence/EVT_…_R#.jpg|mp4`, `reviews.json` (khi đã gán nhãn).

### 3. Đánh giá (Mốc C)

1. Xem video, ghi các khoảng vi phạm thật vào `data/test_ground_truth/<video>.json` (theo mẫu). Thống nhất cả nhóm cách đếm trước khi tính.
2. Chạy:

```powershell
python .\src\evaluation\evaluate.py --run demo=.\runs\demo_all --gt .\data\test_ground_truth\zone_test.json --tolerance 2.0 --out .\reports
```

3. So sánh **trước/sau temporal filter**: chạy pipeline thêm một lần với bản sao file rules đặt `min_duration_sec: 0`, `grace_period_sec: 0`, `cooldown_sec: 0` (xuất ra thư mục `--out` khác), rồi:

```powershell
python .\src\evaluation\evaluate.py --run no_filter=.\runs\r2_nofilter --run with_filter=.\runs\r2_filter --gt ... --out .\reports
```

4. So sánh **baseline vs fine-tuned**: chạy pipeline với `--lab-coat-weights` khác nhau, mỗi lần một `--out`, rồi truyền nhiều `--run`.

Quy ước ghép: cảnh báo phủ khoảng `[violation_start_sec, video_time_sec]`; khớp ground truth cùng rule khi hai khoảng giao nhau sau khi nới ±`--tolerance` giây; ghép 1–1, cảnh báo đầu tiên là TP, các cảnh báo trùng cho cùng một vi phạm tính là FP. **Cảnh báo giả/giờ** = FP / số giờ video; thời lượng lấy từ `--duration-sec`, file ground truth hoặc `pipeline_summary.json`, nếu không có thì để trống (không đoán). Không có ground truth thì dùng nhãn TP/FP từ dashboard để tính Precision (Recall cần ground truth).

### 4. Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

18 test: log sự kiện, evidence bất đồng bộ (không chặn pipeline), pipeline end-to-end với detector giả, server (Range, chặn path traversal, review), thống kê, ghép sự kiện/metric.

### 5. Điểm cần kiểm tra / giới hạn (đọc trước khi demo)

- **Tracker chưa được kiểm thử với bản thật.** `tracker.py` (TV2) không có trong file được cung cấp nên `tracker_adapter.py` dò chữ ký `PersonTracker` bằng `inspect` (tên tham số như `frame_id`, `frame`, `t`, `fps`…). Test dùng tracker giả. Nếu lỗi khi chạy, chỉ sửa `PersonTrackerAdapter.update` trong file đó. Tùy chọn `--tracker simple` là tracker IoU dự phòng, **không phải ByteTrack**, chỉ để chạy thử.
- **Pipeline này không thay `scripts/test_tracker.py`**; chưa so sánh được kết quả hai đường chạy trên video thật vì không có video/weights trong môi trường phát triển. Hãy chạy một video và đối chiếu số sự kiện R1/R2/R3 với lượt chạy `test_tracker.py` hiện có.
- **Chưa chạy với YOLO thật, webcam thật hay màn hình thật**: kiểm thử chỉ dùng detector giả, và giao diện web mới kiểm tra cú pháp JS cùng các API, chưa mở bằng trình duyệt. Hãy mở dashboard một lần và xem trực quan.
- **Codec clip**: ghi thử `avc1`/`H264` rồi `mp4v`; nếu rơi xuống `mp4v` và máy có `ffmpeg` trong PATH thì tự chuyển H.264. Nếu không, trình duyệt có thể không phát clip (dashboard báo rõ, file vẫn mở được bằng trình phát video). Có thể xuất hiện log lỗi OpenCV/FFmpeg khi thử codec, đây là bình thường.
- **Độ trễ** (`details.pipeline_latency_ms`) là từ lúc giải mã khung đến lúc cảnh báo được ghi, không gồm độ trễ camera/driver. Nó được đặt trong `details` vì chưa thấy `events/event_schema.json`; nếu schema cho phép, chuyển sang trường riêng. Cảnh báo chỉ phát sau `min_duration_sec` của rule, nên độ trễ đến khi hiện trên dashboard = thời gian xác nhận + độ trễ xử lý + tối đa 1,5 s polling.
- Nhãn trên video chỉ dùng ASCII (giới hạn của `cv2.putText`).
- Dashboard **không có xác thực**, mặc định chỉ mở ở `127.0.0.1`; không đưa ra mạng khi clip có hình người thật.
- Mất frame/mất hàng đợi: nếu hàng đợi ghi clip đầy (8 clip), clip bị bỏ và ghi `DROPPED` thay vì làm chậm pipeline; ảnh snapshot và log vẫn được lưu.
