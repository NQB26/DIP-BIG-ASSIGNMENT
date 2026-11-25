# TV3 – Integration / Dashboard

Phần việc của TV3 theo kế hoạch: pipeline đọc video/webcam, tích hợp module, dashboard, lưu ảnh/clip sự kiện **bất đồng bộ**, log, đánh giá end-to-end.

## 1. Cài đặt vào repo

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

## 2. Chạy

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

## 3. Đánh giá (Mốc C)

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

## 4. Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

18 test: log sự kiện, evidence bất đồng bộ (không chặn pipeline), pipeline end-to-end với detector giả, server (Range, chặn path traversal, review), thống kê, ghép sự kiện/metric.

## 5. Điểm cần kiểm tra / giới hạn (đọc trước khi demo)

- **Tracker chưa được kiểm thử với bản thật.** `tracker.py` (TV2) không có trong file được cung cấp nên `tracker_adapter.py` dò chữ ký `PersonTracker` bằng `inspect` (tên tham số như `frame_id`, `frame`, `t`, `fps`…). Test dùng tracker giả. Nếu lỗi khi chạy, chỉ sửa `PersonTrackerAdapter.update` trong file đó. Tùy chọn `--tracker simple` là tracker IoU dự phòng, **không phải ByteTrack**, chỉ để chạy thử.
- **Pipeline này không thay `scripts/test_tracker.py`**; chưa so sánh được kết quả hai đường chạy trên video thật vì không có video/weights trong môi trường phát triển. Hãy chạy một video và đối chiếu số sự kiện R1/R2/R3 với lượt chạy `test_tracker.py` hiện có.
- **Chưa chạy với YOLO thật, webcam thật hay màn hình thật**: kiểm thử chỉ dùng detector giả, và giao diện web mới kiểm tra cú pháp JS cùng các API, chưa mở bằng trình duyệt. Hãy mở dashboard một lần và xem trực quan.
- **Codec clip**: ghi thử `avc1`/`H264` rồi `mp4v`; nếu rơi xuống `mp4v` và máy có `ffmpeg` trong PATH thì tự chuyển H.264. Nếu không, trình duyệt có thể không phát clip (dashboard báo rõ, file vẫn mở được bằng trình phát video). Có thể xuất hiện log lỗi OpenCV/FFmpeg khi thử codec, đây là bình thường.
- **Độ trễ** (`details.pipeline_latency_ms`) là từ lúc giải mã khung đến lúc cảnh báo được ghi, không gồm độ trễ camera/driver. Nó được đặt trong `details` vì chưa thấy `events/event_schema.json`; nếu schema cho phép, chuyển sang trường riêng. Cảnh báo chỉ phát sau `min_duration_sec` của rule, nên độ trễ đến khi hiện trên dashboard = thời gian xác nhận + độ trễ xử lý + tối đa 1,5 s polling.
- Nhãn trên video chỉ dùng ASCII (giới hạn của `cv2.putText`).
- Dashboard **không có xác thực**, mặc định chỉ mở ở `127.0.0.1`; không đưa ra mạng khi clip có hình người thật.
- Mất frame/mất hàng đợi: nếu hàng đợi ghi clip đầy (8 clip), clip bị bỏ và ghi `DROPPED` thay vì làm chậm pipeline; ảnh snapshot và log vẫn được lưu.
