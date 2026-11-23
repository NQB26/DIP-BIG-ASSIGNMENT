# DIP-BIG-ASSIGNMENT --- Chạy Rule 1, Rule 2, Rule 3 và vẽ Zone

## 1. Mở dự án và kích hoạt môi trường

Mở Windows PowerShell:

``` powershell
cd C:\Users\ACER\Downloads\DIP-BIG-ASSIGNMENT-main

.\.venv\Scripts\Activate.ps1
```

Nếu gặp lỗi Execution Policy:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
```

Kiểm tra môi trường:

``` powershell
python --version
python -c "import torch, cv2, ultralytics, yaml; print('PyTorch:',torch.__version__); print('OpenCV:',cv2.__version__); print('Ultralytics:',ultralytics.__version__); print('CUDA:',torch.cuda.is_available())"
```

## 2. Khai báo video và model

Chạy trong cùng cửa sổ PowerShell:

``` powershell
$video = ".\data\zone_test.mp4"
$person = ".\yolo11n.pt"
$coat = ".\runs\detect\runs\detect\labcoat_v3_baseline\weights\best.pt"
```

Kiểm tra đường dẫn:

``` powershell
Test-Path $video
Test-Path $person
Test-Path $coat
Test-Path ".\scripts\test_tracker.py"
Test-Path ".\configs\zones.yaml"
Test-Path ".\configs\rules_r1_test.yaml"
```

Các đường dẫn cần trả về `True`. Nếu `$coat` là `False`, hãy kiểm tra
lại đường dẫn thực tế của `best.pt`.

## 3. Kiểm tra cấu hình Rule

Các file được sử dụng:

-   R1: `configs\rules_r1_test.yaml`
-   R2: `configs\rules_r2_test.yaml`
-   R3: `configs\rules_r3_test.yaml`

Mỗi file chỉ được bật một Rule:

  File                   R1    R2    R3
  ---------------------- ----- ----- -----
  `rules_r1_test.yaml`   ON    OFF   OFF
  `rules_r2_test.yaml`   OFF   ON    OFF
  `rules_r3_test.yaml`   OFF   OFF   ON

R1 đã có file `rules_r1_test.yaml` được dùng để test. Không ghi đè file
này. Trước khi chạy R2/R3, kiểm tra hai file tương ứng có tồn tại và
bật/tắt đúng. Không tự tạo cấu hình từ `rules.yaml` nếu chưa kiểm tra
schema và các tham số hiện tại.

Nếu YAML có cấu trúc `rules.R1`, `rules.R2`, `rules.R3`, có thể kiểm tra
bằng:

``` powershell
@'
from pathlib import Path
import yaml

for name in ("rules_r1_test.yaml", "rules_r2_test.yaml", "rules_r3_test.yaml"):
    path = Path("configs") / name
    if not path.exists():
        print(f"\nMISSING: {path}")
        continue
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    print(f"\n{path}:")
    rules = cfg.get("rules", {})
    for rid in ("R1", "R2", "R3"):
        if rid in rules:
            print(f"  {rid}: enabled={rules[rid].get('enabled')}")
        else:
            print(f"  {rid}: không tìm thấy trong cfg['rules']")
'@ | python -
```

Nếu cấu trúc YAML khác, hãy mở file để kiểm tra đúng schema; không dùng
đoạn kiểm tra trên làm kết luận.

## 4. Chạy Rule 1 --- No Lab Coat

``` powershell
python .\scripts\test_tracker.py `
  --video $video `
  --weights $person `
  --lab-coat-weights $coat `
  --zones ".\configs\zones.yaml" `
  --rules ".\configs\rules_r1_test.yaml" `
  --out ".\runs\r1_validation_v3" `
  --imgsz 512 `
  --device cpu `
  --coco-conf 0.25 `
  --coat-conf 0.25 `
  --max-frames 900 `
  --show
```

Output: `runs\r1_validation_v3\`

## 5. Chạy Rule 2 --- Restricted Zone

``` powershell
python .\scripts\test_tracker.py `
  --video $video `
  --weights $person `
  --lab-coat-weights $coat `
  --zones ".\configs\zones.yaml" `
  --rules ".\configs\rules_r2_test.yaml" `
  --out ".\runs\r2_validation_v3" `
  --imgsz 512 `
  --device cpu `
  --coco-conf 0.25 `
  --coat-conf 0.25 `
  --max-frames 900 `
  --show
```

Output: `runs\r2_validation_v3\`

Kiểm tra người đi vào polygon restricted, vị trí anchor của người và
thời gian duy trì vi phạm theo cấu hình.

## 6. Chạy Rule 3 --- Phone Usage

``` powershell
python .\scripts\test_tracker.py `
  --video $video `
  --weights $person `
  --lab-coat-weights $coat `
  --zones ".\configs\zones.yaml" `
  --rules ".\configs\rules_r3_test.yaml" `
  --out ".\runs\r3_validation_v3" `
  --imgsz 512 `
  --device cpu `
  --coco-conf 0.25 `
  --coat-conf 0.25 `
  --max-frames 900 `
  --show
```

Output: `runs\r3_validation_v3\`

R3 dùng phát hiện điện thoại, pose/keypoints và association với Person
track theo logic hiện tại. Cần xem video để đánh giá nhầm lẫn; điện
thoại xuất hiện trong khung hình không đồng nghĩa chắc chắn người đang
sử dụng nó.

## 7. Mở video đã vẽ bounding box

`--show` mở cửa sổ hiển thị trong lúc xử lý. Nhấn `Q` để thoát cửa sổ.

Sau khi chạy xong, mở video kết quả:

``` powershell
Start-Process (Resolve-Path ".\runs\r1_validation_v3\annotated.mp4")
Start-Process (Resolve-Path ".\runs\r2_validation_v3\annotated.mp4")
Start-Process (Resolve-Path ".\runs\r3_validation_v3\annotated.mp4")
```

Nếu không tìm thấy `annotated.mp4`, kiểm tra tên file thực tế:

``` powershell
Get-ChildItem .\runs\r1_validation_v3 -File
Get-ChildItem .\runs\r2_validation_v3 -File
Get-ChildItem .\runs\r3_validation_v3 -File
```

## 8. Đọc báo cáo

Ví dụ với R1:

``` powershell
Get-Content ".\runs\r1_validation_v3\summary.json"
Get-Content ".\runs\r1_validation_v3\events.jsonl"

Import-Csv ".\runs\r1_validation_v3\tracks.csv" |
    Select-Object -First 20 |
    Format-Table
```

Với R2/R3, thay tên thư mục output tương ứng.

-   `summary.json`: thống kê tổng hợp nếu script xuất file này.
-   `events.jsonl`: sự kiện được ghi nhận nếu script xuất file này.
-   `tracks.csv`: dữ liệu tracking nếu script xuất file này.

PENDING chưa chắc đã là event cuối cùng được ghi vào `events.jsonl`.

## 9. Vẽ vùng Zone

Trước tiên kiểm tra script và CLI thực tế:

``` powershell
Test-Path ".\scripts\draw_zones.py"
python .\scripts\draw_zones.py --help
```

Nếu phần trợ giúp xác nhận script nhận các tham số `--video`,
`--frame-idx`, `--out`, `--camera-id`, hãy sao lưu cấu hình trước:

``` powershell
Copy-Item ".\configs\zones.yaml" ".\configs\zones_backup.yaml"
```

Sau đó chạy:

``` powershell
python .\scripts\draw_zones.py `
  --video ".\data\zone_test.mp4" `
  --frame-idx 0 `
  --out ".\configs\zones.yaml" `
  --camera-id zone_test
```

**Cảnh báo:** lệnh trên ghi kết quả vào `configs\zones.yaml`, tức là
thay đổi file Zone đang được các Rule sử dụng. Chỉ chạy khi CLI thực tế
xác nhận các tham số trên và bạn muốn cập nhật vùng. Nếu `--help` khác,
hãy làm theo CLI của script, không chạy lệnh mẫu này.

Phím tắt thêm/xóa điểm, chọn General/Restricted và lưu phụ thuộc vào
code của `draw_zones.py`; hãy theo hướng dẫn hiện ra trong cửa sổ hoặc
trong `--help`, không mặc định phím tắt.

Kiểm tra sau khi lưu:

``` powershell
Get-Content ".\configs\zones.yaml"
```

## 10. Thứ tự thao tác khuyến nghị

1.  Mở PowerShell, vào thư mục dự án và kích hoạt `.venv`.
2.  Khai báo `$video`, `$person`, `$coat`.
3.  Kiểm tra đường dẫn video, model, script và YAML.
4.  Xác nhận mỗi file cấu hình chỉ bật đúng một Rule.
5.  Chạy R1, xem video và báo cáo.
6.  Chạy R2, xem video và báo cáo.
7.  Chạy R3, xem video và báo cáo.
8.  Chỉ chỉnh Zone khi cần; sao lưu `zones.yaml` trước khi ghi đè.
9.  Chạy lại các Rule sau khi thay đổi Zone.

### Giới hạn số frame

`--max-frames 900` xử lý tối đa 900 frame đầu. Muốn xử lý toàn bộ video,
kiểm tra:

``` powershell
python .\scripts\test_tracker.py --help
```

Chỉ dùng `--max-frames 0` nếu trợ giúp của phiên bản script xác nhận `0`
có nghĩa là không giới hạn.
