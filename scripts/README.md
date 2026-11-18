# Test Tracker với YOLO11n (Tracking & Safety Logic)

Hướng dẫn chạy và kiểm tra phần **Detection → ByteTrack → Rule Engine** bằng mô hình có sẵn `yolo11n.pt` (COCO).

> `yolo11n` nhận được `person` và `cell phone` nhưng **không có `lab_coat`**. Vì vậy khi dùng nó phải tắt R1 (xem Bước 2). R2 và R3 vẫn test được bình thường.

## 1. Cấu trúc

```
configs/rules.yaml        ngưỡng rule + tham số ByteTrack
configs/zones.yaml        zone (toạ độ chuẩn hoá 0-1)
events/event_schema.json  schema của RuleEvent
src/tracking_rules/       geometry, state_machine, tracker, rule_engine
scripts/draw_zones.py     vẽ zone trên frame video / webcam
scripts/test_tracker.py   chạy pipeline trên file video
scripts/test_webcam.py    chạy pipeline realtime bằng webcam
scripts/make_synthetic.py sinh video + detection giả (không cần camera/model)
tests/test_rules.py       unit test logic
```

## 2. Cài đặt

```powershell
pip install -r requirements_tracking.txt
```

Lần chạy đầu, `yolo11n.pt` được Ultralytics tự tải về (cần Internet).

Mở `configs/rules.yaml` và tắt R1:

```yaml
rules:
  R1:
    enabled: false
```

## 3. Vẽ zone (làm một lần cho mỗi vị trí camera)

```powershell
# từ webcam
py scripts/draw_zones.py --camera 0 --out configs/zones.yaml
# hoặc từ video
py scripts/draw_zones.py --video data/lab.mp4 --out configs/zones.yaml
```

| Phím | Tác dụng |
|---|---|
| Chuột trái | thêm đỉnh |
| Chuột phải / `u` | xoá đỉnh cuối |
| `g` | đóng đa giác thành zone `general` |
| `r` | đóng đa giác thành zone `restricted` |
| `d` | xoá zone vừa tạo |
| `s` | lưu file |
| `q` / `ESC` | thoát |

Mẹo: vẽ theo **mặt sàn** (rule xét điểm chân của người), vùng `restricted` nên nới ra khoảng 0.5 m. Zone `zone_id` tự đặt là `general_1`, `restricted_1`… nên mở YAML đổi thành tên có nghĩa. Nếu dùng `--flip` ở webcam thì vẽ zone cũng phải thêm `--flip`.

## 4. Test bằng webcam

```powershell
py scripts/test_webcam.py --weights yolo11n.pt --imgsz 480
```

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--source 1` | dùng camera khác (hoặc đường dẫn video/RTSP) |
| `--flip` | lật gương ngang |
| `--no-zones` | không dùng zone (cả khung hình là `general`) |
| `--device cpu` / `--device 0` | chọn CPU hoặc GPU |
| `--save-video runs/cam.mp4` | lưu video có overlay |

Phím khi chạy: `q`/`ESC` thoát, `z` bật/tắt zone, `p` tạm dừng, `s` chụp ảnh. Event được in ra terminal và ghi vào `runs/webcam/events.jsonl`.

## 5. Test trên file video

```powershell
py scripts/test_tracker.py --video data/lab.mp4 --weights yolo11n.pt --imgsz 480 --out runs/test01 --show
```

Kết quả trong `runs/test01/`:

| File | Nội dung |
|---|---|
| `annotated.mp4` | video có overlay |
| `events.jsonl` | mỗi dòng một RuleEvent |
| `tracks.csv` | `frame_id,t,track_id,x1,y1,x2,y2,conf` |
| `summary.json` | số frame, số ID duy nhất, độ sống track, số track ngắn, số event theo rule |

Cách đọc overlay: khung **trắng** = bình thường, **vàng** = đang đếm thời gian vi phạm (vd `R3 1.4/3.0s`), **đỏ** = đã cảnh báo. Chấm vàng = điểm chân (R2), khung tím mỏng = vùng thân trên (R3).

## 6. Chưa có video hay camera: dùng dữ liệu giả

```powershell
py scripts/make_synthetic.py --out data/synthetic
py scripts/test_tracker.py --video data/synthetic/no_coat.mp4 --dets-json data/synthetic/no_coat.jsonl --out runs/no_coat
```

Mỗi kịch bản phải dùng đúng cặp `.mp4` và `.jsonl` cùng tên. `--out` chỉ là thư mục lưu kết quả.

| Kịch bản | Kết quả kỳ vọng (`events_by_rule`) |
|---|---|
| `coat_ok` | `{}` |
| `no_coat` | `{"R1": 1}` |
| `restricted` | `{"R2": 1}` |
| `restricted_brief` | `{}` |
| `phone_hand` | `{"R3": 1}` |
| `phone_table` | `{}` |
| `cross_two_people` | `{}`, 2 ID |
| `leave_return` | `{"R1": 1}`, 2 ID |

Cách này chỉ kiểm tra logic rule, FSM, ByteTrack và schema event, không đánh giá chất lượng detector. Unit test:

```powershell
$env:PYTHONPATH="src"; pytest tests -q
```

## 7. Cách kiểm tra tracker

Với webcam hoặc video, lần lượt làm các tình huống sau và quan sát ID trên overlay:

| Tình huống | Kỳ vọng |
|---|---|
| Đi lại, quay người một mình | ID giữ nguyên |
| Hai người đi ngang qua nhau | Không đổi ID giữa hai người |
| Bị che khoảng 1 giây rồi lộ ra | ID còn nguyên |
| Ra khỏi khung rồi quay lại | ID mới là bình thường |
| Đứng im, ngồi, đứng lên | Track không bị đứt |
| R3: cầm điện thoại ngang ngực quá 3 giây | Báo `PHONE_USAGE`; để điện thoại trên bàn thấp thì không báo |
| R2: chân bước vào vùng `restricted` quá 1.5 giây | Báo `RESTRICTED_ZONE`; chạm thoáng qua thì không báo |

`summary.json` cho biết nhanh: `unique_track_ids` nên xấp xỉ số người thật, `short_tracks_lt_1s` nên gần 0.

## 8. Chỉnh tham số tracker (`configs/rules.yaml → tracker`)

| Triệu chứng | Chỉnh |
|---|---|
| Đổi ID sau khi bị che | tăng `lost_buffer_sec` (1.5 → 2.5) |
| Nhiều ID rác, track ngắn | tăng `min_consecutive_frames` (2 → 3) hoặc `activation_threshold` (0.5 → 0.6) |
| Mất track khi detector chập chờn | giảm `high_conf_threshold` (0.6 → 0.5) |
| Hai người gần nhau bị lẫn ID | tăng `min_iou` (0.1 → 0.2) |
| R3 bắn khi điện thoại ở trên bàn | giảm `rules.R3.upper_body_ratio` (0.65 → 0.55) |
| R3 bỏ sót vì điện thoại bị tay che | giảm `rules.R3.phone_conf_min`, tăng `grace_period_sec` |
| Event lặp quá dày | tăng `cooldown_sec` |

Sau khi sửa, chạy lại trên **cùng video** để so sánh.

## 9. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Không tìm thấy file weights` | dùng `--weights yolo11n.pt` hoặc trỏ đúng đường dẫn `.pt` |
| `Không mở được nguồn '0'` | thử `--source 1`, tắt Zoom/Teams đang dùng webcam, cấp quyền camera trong Windows Settings |
| FPS thấp (< 10) | giảm `--imgsz` (416), dùng `--device 0` nếu có GPU, hoặc thử video thay cho webcam |
| R2 không bắn dù người đứng trong vùng cấm | chân người nằm ngoài khung hình hoặc zone chưa phủ mặt sàn; đặt camera xa hơn / vẽ lại zone |
| Có nhiều event R1 | quên đặt `R1: enabled: false` khi dùng `yolo11n` |

## 10. Format dữ liệu

- Đầu vào Rule Engine và đầu ra `RuleEvent` theo `docs/data_contract_original.md`; các trường mở rộng (`severity`, `frame_id`, `details`…) là tuỳ chọn.
- Schema: `events/event_schema.json`. Thiết kế chi tiết: `docs/tracking_rules_design.md`.